use crate::{ComputeClass, FeatureHypothesis, TemporalAvailability};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalBatch {
    pub schema_version: u32,
    pub experiment_id: String,
    pub proposer: String,
    pub hypotheses: Vec<FeatureHypothesis>,
    #[serde(default)]
    pub maximum_scalar_ops: Option<u64>,
    #[serde(default)]
    pub maximum_persistent_state_scalars: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedHypothesis {
    pub hypothesis: FeatureHypothesis,
    pub score: f64,
    pub novelty_score: f64,
    pub cost_score: f64,
    pub observability_score: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalRejection {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalReview {
    pub schema_version: u32,
    pub experiment_id: String,
    pub proposer: String,
    pub accepted: Vec<RankedHypothesis>,
    pub rejected: Vec<ProposalRejection>,
}

pub fn review_proposals(
    batch: &ProposalBatch,
    available_signals: &[String],
    existing_hypothesis_ids: &[String],
) -> Result<ProposalReview, String> {
    if batch.schema_version != 1 {
        return Err(format!(
            "unsupported proposal batch schema {}",
            batch.schema_version
        ));
    }
    if batch.experiment_id.trim().is_empty() {
        return Err("experiment_id must not be empty".to_string());
    }
    if batch.proposer.trim().is_empty() {
        return Err("proposer must not be empty".to_string());
    }

    let available: BTreeSet<&str> = available_signals.iter().map(String::as_str).collect();
    let existing: BTreeSet<&str> = existing_hypothesis_ids.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for hypothesis in &batch.hypotheses {
        if let Err(reason) = hypothesis.validate() {
            rejected.push(ProposalRejection {
                id: hypothesis.id.clone(),
                reason,
            });
            continue;
        }
        if !seen.insert(hypothesis.id.as_str()) {
            rejected.push(ProposalRejection {
                id: hypothesis.id.clone(),
                reason: "duplicate hypothesis id in proposal batch".to_string(),
            });
            continue;
        }
        if existing.contains(hypothesis.id.as_str()) {
            rejected.push(ProposalRejection {
                id: hypothesis.id.clone(),
                reason: "hypothesis id already exists in the deterministic catalog".to_string(),
            });
            continue;
        }
        if let Some(signal) = hypothesis
            .required_signals
            .iter()
            .find(|signal| !available.contains(signal.as_str()))
        {
            rejected.push(ProposalRejection {
                id: hypothesis.id.clone(),
                reason: format!("required runtime signal `{signal}` is not instrumented"),
            });
            continue;
        }
        if exceeds_budget(batch, hypothesis) {
            rejected.push(ProposalRejection {
                id: hypothesis.id.clone(),
                reason: "runtime cost exceeds the declared proposal budget".to_string(),
            });
            continue;
        }

        accepted.push(rank(hypothesis));
    }

    accepted.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.hypothesis.id.cmp(&right.hypothesis.id))
    });
    rejected.sort_by(|left, right| left.id.cmp(&right.id));

    Ok(ProposalReview {
        schema_version: 1,
        experiment_id: batch.experiment_id.clone(),
        proposer: batch.proposer.clone(),
        accepted,
        rejected,
    })
}

fn exceeds_budget(batch: &ProposalBatch, hypothesis: &FeatureHypothesis) -> bool {
    batch.maximum_scalar_ops.is_some_and(|limit| {
        hypothesis.runtime_cost.estimated_scalar_ops > limit
    }) || batch.maximum_persistent_state_scalars.is_some_and(|limit| {
        hypothesis.runtime_cost.persistent_state_scalars > limit
    })
}

fn rank(hypothesis: &FeatureHypothesis) -> RankedHypothesis {
    let novelty_score = match hypothesis.family {
        crate::FeatureFamily::Level => 0.25,
        crate::FeatureFamily::DistributionShape => 0.45,
        crate::FeatureFamily::TemporalDelta => 0.65,
        crate::FeatureFamily::TemporalSlope => 0.75,
        crate::FeatureFamily::RollingStatistic => 0.8,
        crate::FeatureFamily::CrossSignalInteraction => 0.85,
        crate::FeatureFamily::LayerAggregate => 0.9,
        crate::FeatureFamily::Stability => 1.0,
    };
    let cost_score = compute_cost_score(hypothesis.runtime_cost.compute_class)
        / (1.0 + hypothesis.runtime_cost.estimated_scalar_ops as f64).ln_1p();
    let observability_score = match hypothesis.temporal_availability {
        TemporalAvailability::CurrentDecision => 1.0,
        TemporalAvailability::PastOnly => 0.9,
        TemporalAvailability::FutureDependent
        | TemporalAvailability::TaskOutcomeDependent => 0.0,
    };
    let score = 0.45 * novelty_score + 0.35 * cost_score + 0.20 * observability_score;

    RankedHypothesis {
        hypothesis: hypothesis.clone(),
        score,
        novelty_score,
        cost_score,
        observability_score,
    }
}

fn compute_cost_score(class: ComputeClass) -> f64 {
    match class {
        ComputeClass::Constant => 1.0,
        ComputeClass::LinearHeads => 0.8,
        ComputeClass::LinearLayers => 0.7,
        ComputeClass::LinearTokens => 0.6,
        ComputeClass::QuadraticTokens => 0.1,
    }
}

pub fn summarize_rejections(review: &ProposalReview) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for rejection in &review.rejected {
        *counts.entry(rejection.reason.clone()).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureFamily, RuntimeCost};

    fn hypothesis(id: &str, signal: &str) -> FeatureHypothesis {
        FeatureHypothesis {
            id: id.to_string(),
            name: id.to_string(),
            family: FeatureFamily::Stability,
            expression: format!("abs({signal})"),
            required_signals: vec![signal.to_string()],
            temporal_availability: TemporalAvailability::CurrentDecision,
            runtime_cost: RuntimeCost::constant(2),
            rationale: "detects instability".to_string(),
            expected_failure_mode: "prediction divergence".to_string(),
            ablation_group: "agent_proposals".to_string(),
            deterministic: true,
        }
    }

    #[test]
    fn unavailable_signals_are_rejected() {
        let batch = ProposalBatch {
            schema_version: 1,
            experiment_id: "test".to_string(),
            proposer: "sciagent".to_string(),
            hypotheses: vec![hypothesis("missing", "not_instrumented")],
            maximum_scalar_ops: Some(8),
            maximum_persistent_state_scalars: Some(2),
        };
        let review = review_proposals(&batch, &["drift".to_string()], &[]).unwrap();
        assert!(review.accepted.is_empty());
        assert_eq!(review.rejected.len(), 1);
    }

    #[test]
    fn review_is_deterministic_and_sorted() {
        let batch = ProposalBatch {
            schema_version: 1,
            experiment_id: "test".to_string(),
            proposer: "sciagent".to_string(),
            hypotheses: vec![hypothesis("zeta", "drift"), hypothesis("alpha", "drift")],
            maximum_scalar_ops: Some(8),
            maximum_persistent_state_scalars: Some(2),
        };
        let first = review_proposals(&batch, &["drift".to_string()], &[]).unwrap();
        let second = review_proposals(&batch, &["drift".to_string()], &[]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.accepted[0].hypothesis.id, "alpha");
    }
}
