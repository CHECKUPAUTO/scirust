//! Held-out engineering-metric gate for bounded SciAgent specialization.
//!
//! P7.3 deliberately keeps training loss out of the retention decision. A
//! candidate update is compared with its baseline on the exact same frozen
//! held-out task IDs. Missing measurements, task-set drift, budget overflow or
//! an unmet engineering criterion fail closed.

use crate::sha256::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const ENGINEERING_SPECIALIZATION_CRITERION_VERSION: u64 = 1;
pub const ENGINEERING_SPECIALIZATION_REPORT_VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringSpecializationCriterion {
    pub version: u64,
    pub max_evaluation_budget: u64,
    pub min_valid_patch_set_rate: f64,
    pub max_compile_brier_score: f64,
    pub min_first_pass_gate_success_rate: f64,
    pub min_accepted_candidate_yield_per_budget: f64,
    pub min_valid_patch_set_rate_delta: f64,
    pub min_compile_brier_improvement: f64,
    pub min_first_pass_gate_success_rate_delta: f64,
    pub min_accepted_candidate_yield_per_budget_delta: f64,
}

impl EngineeringSpecializationCriterion {
    pub fn validate(&self) -> Result<(), EngineeringSpecializationError> {
        if self.version != ENGINEERING_SPECIALIZATION_CRITERION_VERSION
        {
            return Err(EngineeringSpecializationError::InvalidCriterion(format!(
                "unsupported criterion version {}",
                self.version
            )));
        }
        if self.max_evaluation_budget == 0
        {
            return Err(EngineeringSpecializationError::InvalidCriterion(
                "max_evaluation_budget must be non-zero".to_string(),
            ));
        }
        validate_probability("min_valid_patch_set_rate", self.min_valid_patch_set_rate)?;
        validate_probability("max_compile_brier_score", self.max_compile_brier_score)?;
        validate_probability(
            "min_first_pass_gate_success_rate",
            self.min_first_pass_gate_success_rate,
        )?;
        validate_non_negative_finite(
            "min_accepted_candidate_yield_per_budget",
            self.min_accepted_candidate_yield_per_budget,
        )?;
        validate_non_negative_finite(
            "min_valid_patch_set_rate_delta",
            self.min_valid_patch_set_rate_delta,
        )?;
        validate_non_negative_finite(
            "min_compile_brier_improvement",
            self.min_compile_brier_improvement,
        )?;
        validate_non_negative_finite(
            "min_first_pass_gate_success_rate_delta",
            self.min_first_pass_gate_success_rate_delta,
        )?;
        validate_non_negative_finite(
            "min_accepted_candidate_yield_per_budget_delta",
            self.min_accepted_candidate_yield_per_budget_delta,
        )?;
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, EngineeringSpecializationError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self).map_err(EngineeringSpecializationError::Json)?;
        Ok(sha256_hex(&encoded))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeldOutEngineeringObservation {
    pub task_spec_id: String,
    pub patch_set_valid: Option<bool>,
    pub compile_pass_probability: Option<f64>,
    pub compile_passed: Option<bool>,
    pub first_pass_gate_success: Option<bool>,
    pub accepted_candidate: Option<bool>,
    pub evaluation_cost: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeldOutEngineeringMetrics {
    pub tasks: usize,
    pub valid_patch_set_rate: f64,
    pub compile_brier_score: f64,
    pub first_pass_gate_success_rate: f64,
    pub accepted_candidates: usize,
    pub evaluation_budget: u64,
    pub accepted_candidate_yield_per_budget: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecializationRetentionDecision {
    Retain,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringSpecializationReport {
    pub report_version: u64,
    pub criterion_sha256: String,
    pub held_out_task_ids: Vec<String>,
    pub baseline: HeldOutEngineeringMetrics,
    pub candidate: HeldOutEngineeringMetrics,
    pub decision: SpecializationRetentionDecision,
    pub failed_criteria: Vec<String>,
}

impl EngineeringSpecializationReport {
    pub fn evaluate(
        criterion: &EngineeringSpecializationCriterion,
        baseline: &[HeldOutEngineeringObservation],
        candidate: &[HeldOutEngineeringObservation],
    ) -> Result<Self, EngineeringSpecializationError> {
        criterion.validate()?;
        let baseline_by_task = canonical_observations("baseline", baseline)?;
        let candidate_by_task = canonical_observations("candidate", candidate)?;
        let baseline_ids: Vec<_> = baseline_by_task.keys().cloned().collect();
        let candidate_ids: Vec<_> = candidate_by_task.keys().cloned().collect();
        if baseline_ids != candidate_ids
        {
            return Err(EngineeringSpecializationError::HeldOutTaskSetMismatch {
                baseline: baseline_ids,
                candidate: candidate_ids,
            });
        }

        let baseline_metrics = measure(&baseline_by_task)?;
        let candidate_metrics = measure(&candidate_by_task)?;
        if baseline_metrics.evaluation_budget > criterion.max_evaluation_budget
        {
            return Err(EngineeringSpecializationError::BudgetExceeded {
                side: "baseline",
                actual: baseline_metrics.evaluation_budget,
                maximum: criterion.max_evaluation_budget,
            });
        }
        if candidate_metrics.evaluation_budget > criterion.max_evaluation_budget
        {
            return Err(EngineeringSpecializationError::BudgetExceeded {
                side: "candidate",
                actual: candidate_metrics.evaluation_budget,
                maximum: criterion.max_evaluation_budget,
            });
        }

        let mut failed = Vec::new();
        require_at_least(
            &mut failed,
            "valid_patch_set_rate.absolute",
            candidate_metrics.valid_patch_set_rate,
            criterion.min_valid_patch_set_rate,
        );
        require_at_least(
            &mut failed,
            "valid_patch_set_rate.delta",
            candidate_metrics.valid_patch_set_rate - baseline_metrics.valid_patch_set_rate,
            criterion.min_valid_patch_set_rate_delta,
        );
        require_at_most(
            &mut failed,
            "compile_brier_score.absolute",
            candidate_metrics.compile_brier_score,
            criterion.max_compile_brier_score,
        );
        require_at_least(
            &mut failed,
            "compile_brier_score.improvement",
            baseline_metrics.compile_brier_score - candidate_metrics.compile_brier_score,
            criterion.min_compile_brier_improvement,
        );
        require_at_least(
            &mut failed,
            "first_pass_gate_success_rate.absolute",
            candidate_metrics.first_pass_gate_success_rate,
            criterion.min_first_pass_gate_success_rate,
        );
        require_at_least(
            &mut failed,
            "first_pass_gate_success_rate.delta",
            candidate_metrics.first_pass_gate_success_rate
                - baseline_metrics.first_pass_gate_success_rate,
            criterion.min_first_pass_gate_success_rate_delta,
        );
        require_at_least(
            &mut failed,
            "accepted_candidate_yield_per_budget.absolute",
            candidate_metrics.accepted_candidate_yield_per_budget,
            criterion.min_accepted_candidate_yield_per_budget,
        );
        require_at_least(
            &mut failed,
            "accepted_candidate_yield_per_budget.delta",
            candidate_metrics.accepted_candidate_yield_per_budget
                - baseline_metrics.accepted_candidate_yield_per_budget,
            criterion.min_accepted_candidate_yield_per_budget_delta,
        );

        let decision = if failed.is_empty()
        {
            SpecializationRetentionDecision::Retain
        }
        else
        {
            SpecializationRetentionDecision::Reject
        };

        Ok(Self {
            report_version: ENGINEERING_SPECIALIZATION_REPORT_VERSION,
            criterion_sha256: criterion.sha256()?,
            held_out_task_ids: baseline_by_task.keys().cloned().collect(),
            baseline: baseline_metrics,
            candidate: candidate_metrics,
            decision,
            failed_criteria: failed,
        })
    }
}

#[derive(Debug)]
pub enum EngineeringSpecializationError {
    Json(serde_json::Error),
    InvalidCriterion(String),
    EmptyHeldOutSet,
    InvalidObservation {
        side: &'static str,
        task_spec_id: String,
        reason: String,
    },
    DuplicateTask {
        side: &'static str,
        task_spec_id: String,
    },
    HeldOutTaskSetMismatch {
        baseline: Vec<String>,
        candidate: Vec<String>,
    },
    BudgetExceeded {
        side: &'static str,
        actual: u64,
        maximum: u64,
    },
}

impl fmt::Display for EngineeringSpecializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Json(error) => write!(f, "engineering specialization JSON: {error}"),
            Self::InvalidCriterion(reason) =>
            {
                write!(f, "engineering specialization criterion: {reason}")
            },
            Self::EmptyHeldOutSet => write!(f, "held-out engineering set is empty"),
            Self::InvalidObservation {
                side,
                task_spec_id,
                reason,
            } => write!(f, "{side} observation {task_spec_id}: {reason}"),
            Self::DuplicateTask { side, task_spec_id } =>
            {
                write!(f, "duplicate {side} held-out task: {task_spec_id}")
            },
            Self::HeldOutTaskSetMismatch {
                baseline,
                candidate,
            } => write!(
                f,
                "held-out task sets differ: baseline={baseline:?}, candidate={candidate:?}"
            ),
            Self::BudgetExceeded {
                side,
                actual,
                maximum,
            } => write!(
                f,
                "{side} evaluation budget {actual} exceeds frozen maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for EngineeringSpecializationError {}

fn canonical_observations<'a>(
    side: &'static str,
    observations: &'a [HeldOutEngineeringObservation],
) -> Result<BTreeMap<String, &'a HeldOutEngineeringObservation>, EngineeringSpecializationError> {
    if observations.is_empty()
    {
        return Err(EngineeringSpecializationError::EmptyHeldOutSet);
    }
    let mut by_task = BTreeMap::new();
    for observation in observations
    {
        let task_spec_id = observation.task_spec_id.trim();
        if task_spec_id.is_empty() || task_spec_id != observation.task_spec_id
        {
            return Err(EngineeringSpecializationError::InvalidObservation {
                side,
                task_spec_id: observation.task_spec_id.clone(),
                reason: "task_spec_id must be non-empty and canonical".to_string(),
            });
        }
        if by_task
            .insert(observation.task_spec_id.clone(), observation)
            .is_some()
        {
            return Err(EngineeringSpecializationError::DuplicateTask {
                side,
                task_spec_id: observation.task_spec_id.clone(),
            });
        }
    }
    Ok(by_task)
}

fn measure(
    observations: &BTreeMap<String, &HeldOutEngineeringObservation>,
) -> Result<HeldOutEngineeringMetrics, EngineeringSpecializationError> {
    let mut valid_patch_sets = 0usize;
    let mut compile_brier_sum = 0.0f64;
    let mut first_pass_gate_successes = 0usize;
    let mut accepted_candidates = 0usize;
    let mut evaluation_budget = 0u64;

    for (task_spec_id, observation) in observations
    {
        let patch_set_valid =
            required_measurement(observation.patch_set_valid, task_spec_id, "patch_set_valid")?;
        let compile_probability = required_measurement(
            observation.compile_pass_probability,
            task_spec_id,
            "compile_pass_probability",
        )?;
        if !compile_probability.is_finite() || !(0.0..=1.0).contains(&compile_probability)
        {
            return Err(EngineeringSpecializationError::InvalidObservation {
                side: "measurement",
                task_spec_id: task_spec_id.clone(),
                reason: "compile_pass_probability must be finite and in [0,1]".to_string(),
            });
        }
        let compile_passed =
            required_measurement(observation.compile_passed, task_spec_id, "compile_passed")?;
        let gate_success = required_measurement(
            observation.first_pass_gate_success,
            task_spec_id,
            "first_pass_gate_success",
        )?;
        let accepted = required_measurement(
            observation.accepted_candidate,
            task_spec_id,
            "accepted_candidate",
        )?;
        let cost =
            required_measurement(observation.evaluation_cost, task_spec_id, "evaluation_cost")?;
        if cost == 0
        {
            return Err(EngineeringSpecializationError::InvalidObservation {
                side: "measurement",
                task_spec_id: task_spec_id.clone(),
                reason: "evaluation_cost must be non-zero".to_string(),
            });
        }

        valid_patch_sets += usize::from(patch_set_valid);
        let compile_target = if compile_passed { 1.0 } else { 0.0 };
        compile_brier_sum += (compile_probability - compile_target).powi(2);
        first_pass_gate_successes += usize::from(gate_success);
        accepted_candidates += usize::from(accepted);
        evaluation_budget = evaluation_budget.checked_add(cost).ok_or_else(|| {
            EngineeringSpecializationError::InvalidObservation {
                side: "measurement",
                task_spec_id: task_spec_id.clone(),
                reason: "evaluation budget overflow".to_string(),
            }
        })?;
    }

    let tasks = observations.len();
    let tasks_f64 = tasks as f64;
    Ok(HeldOutEngineeringMetrics {
        tasks,
        valid_patch_set_rate: valid_patch_sets as f64 / tasks_f64,
        compile_brier_score: compile_brier_sum / tasks_f64,
        first_pass_gate_success_rate: first_pass_gate_successes as f64 / tasks_f64,
        accepted_candidates,
        evaluation_budget,
        accepted_candidate_yield_per_budget: accepted_candidates as f64 / evaluation_budget as f64,
    })
}

fn required_measurement<T: Copy>(
    value: Option<T>,
    task_spec_id: &str,
    field: &'static str,
) -> Result<T, EngineeringSpecializationError> {
    value.ok_or_else(|| EngineeringSpecializationError::InvalidObservation {
        side: "measurement",
        task_spec_id: task_spec_id.to_string(),
        reason: format!("missing required held-out measurement {field}"),
    })
}

fn validate_probability(
    name: &'static str,
    value: f64,
) -> Result<(), EngineeringSpecializationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value)
    {
        return Err(EngineeringSpecializationError::InvalidCriterion(format!(
            "{name} must be finite and in [0,1]"
        )));
    }
    Ok(())
}

fn validate_non_negative_finite(
    name: &'static str,
    value: f64,
) -> Result<(), EngineeringSpecializationError> {
    if !value.is_finite() || value < 0.0
    {
        return Err(EngineeringSpecializationError::InvalidCriterion(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn require_at_least(failed: &mut Vec<String>, name: &'static str, actual: f64, minimum: f64) {
    if actual < minimum
    {
        failed.push(format!("{name}: actual={actual:.12} minimum={minimum:.12}"));
    }
}

fn require_at_most(failed: &mut Vec<String>, name: &'static str, actual: f64, maximum: f64) {
    if actual > maximum
    {
        failed.push(format!("{name}: actual={actual:.12} maximum={maximum:.12}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn criterion() -> EngineeringSpecializationCriterion {
        EngineeringSpecializationCriterion {
            version: ENGINEERING_SPECIALIZATION_CRITERION_VERSION,
            max_evaluation_budget: 100,
            min_valid_patch_set_rate: 0.75,
            max_compile_brier_score: 0.20,
            min_first_pass_gate_success_rate: 0.50,
            min_accepted_candidate_yield_per_budget: 0.05,
            min_valid_patch_set_rate_delta: 0.25,
            min_compile_brier_improvement: 0.05,
            min_first_pass_gate_success_rate_delta: 0.25,
            min_accepted_candidate_yield_per_budget_delta: 0.025,
        }
    }

    fn observation(
        task: &str,
        valid: bool,
        compile_probability: f64,
        compiled: bool,
        gate_success: bool,
        accepted: bool,
        cost: u64,
    ) -> HeldOutEngineeringObservation {
        HeldOutEngineeringObservation {
            task_spec_id: task.to_string(),
            patch_set_valid: Some(valid),
            compile_pass_probability: Some(compile_probability),
            compile_passed: Some(compiled),
            first_pass_gate_success: Some(gate_success),
            accepted_candidate: Some(accepted),
            evaluation_cost: Some(cost),
        }
    }

    fn baseline() -> Vec<HeldOutEngineeringObservation> {
        vec![
            observation("task-a", true, 0.55, true, true, true, 10),
            observation("task-b", false, 0.60, false, false, false, 10),
            observation("task-c", true, 0.45, true, false, false, 10),
            observation("task-d", false, 0.50, false, false, false, 10),
        ]
    }

    fn improved_candidate() -> Vec<HeldOutEngineeringObservation> {
        vec![
            observation("task-d", true, 0.15, false, true, true, 10),
            observation("task-b", true, 0.10, false, false, false, 10),
            observation("task-a", true, 0.90, true, true, true, 10),
            observation("task-c", true, 0.85, true, true, true, 10),
        ]
    }

    #[test]
    fn retains_only_when_frozen_held_out_engineering_criterion_passes() {
        let report = EngineeringSpecializationReport::evaluate(
            &criterion(),
            &baseline(),
            &improved_candidate(),
        )
        .unwrap();
        assert_eq!(report.decision, SpecializationRetentionDecision::Retain);
        assert!(report.failed_criteria.is_empty());
        assert_eq!(
            report.held_out_task_ids,
            vec!["task-a", "task-b", "task-c", "task-d"]
        );
        assert!(report.candidate.valid_patch_set_rate > report.baseline.valid_patch_set_rate);
        assert!(report.candidate.compile_brier_score < report.baseline.compile_brier_score);
        assert!(
            report.candidate.first_pass_gate_success_rate
                > report.baseline.first_pass_gate_success_rate
        );
        assert!(
            report.candidate.accepted_candidate_yield_per_budget
                > report.baseline.accepted_candidate_yield_per_budget
        );
    }

    #[test]
    fn rejects_candidate_when_training_independent_engineering_metric_regresses() {
        let mut candidate = improved_candidate();
        candidate[0].first_pass_gate_success = Some(false);
        candidate[2].first_pass_gate_success = Some(false);
        let report =
            EngineeringSpecializationReport::evaluate(&criterion(), &baseline(), &candidate)
                .unwrap();
        assert_eq!(report.decision, SpecializationRetentionDecision::Reject);
        assert!(
            report
                .failed_criteria
                .iter()
                .any(|failure| failure.starts_with("first_pass_gate_success_rate.delta"))
        );
    }

    #[test]
    fn missing_measurement_fails_closed() {
        let mut candidate = improved_candidate();
        candidate[0].compile_pass_probability = None;
        let error =
            EngineeringSpecializationReport::evaluate(&criterion(), &baseline(), &candidate)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("missing required held-out measurement")
        );
    }

    #[test]
    fn task_set_drift_fails_closed() {
        let mut candidate = improved_candidate();
        candidate[0].task_spec_id = "task-other".to_string();
        let error =
            EngineeringSpecializationReport::evaluate(&criterion(), &baseline(), &candidate)
                .unwrap_err();
        assert!(matches!(
            error,
            EngineeringSpecializationError::HeldOutTaskSetMismatch { .. }
        ));
    }

    #[test]
    fn budget_overflow_fails_closed() {
        let mut candidate = improved_candidate();
        candidate[0].evaluation_cost = Some(101);
        let error =
            EngineeringSpecializationReport::evaluate(&criterion(), &baseline(), &candidate)
                .unwrap_err();
        assert!(matches!(
            error,
            EngineeringSpecializationError::BudgetExceeded {
                side: "candidate",
                ..
            }
        ));
    }

    #[test]
    fn criterion_identity_is_deterministic_and_sensitive() {
        let first = criterion();
        let mut second = first.clone();
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
        second.min_valid_patch_set_rate_delta += 0.01;
        assert_ne!(first.sha256().unwrap(), second.sha256().unwrap());
    }
}
