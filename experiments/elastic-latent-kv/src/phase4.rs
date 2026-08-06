//! Phase 4 online rank adaptation for the Elastic Latent KV experiment.
//!
//! Phase 4 adds a deterministic stateful controller above the strict-budget
//! planner from Phase 3. Budget-forced downgrades and quality recovery are
//! immediate, while discretionary rank changes require a configurable number
//! of consecutive identical proposals. This suppresses oscillation without
//! allowing the active representation to exceed the current strict budget.

use crate::phase3::{
    BudgetCandidate, BudgetError, BudgetScenario, BudgetScenarioReport, run_budget_scenario,
};
use core::fmt;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Errors returned by deterministic Phase 4 adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationError {
    /// A required count or schedule length was zero.
    ZeroField {
        /// Human-readable field name.
        field: &'static str,
    },
    /// A quality scale or target was non-finite or non-positive.
    InvalidThreshold {
        /// Human-readable field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// A planner proposal unexpectedly exceeds the current strict budget.
    ProposalOverBudget {
        /// Bytes used by the proposal.
        proposal_bytes: u64,
        /// Current strict budget.
        budget_bytes: u64,
    },
    /// An integer counter overflowed.
    ArithmeticOverflow,
    /// A Phase 3 planning operation failed.
    Budget(BudgetError),
}

impl fmt::Display for AdaptationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroField { field } => write!(formatter, "{field} must be non-zero"),
            Self::InvalidThreshold { field, value } =>
            {
                write!(
                    formatter,
                    "{field} must be finite and positive, received {value}"
                )
            },
            Self::ProposalOverBudget {
                proposal_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "proposal uses {proposal_bytes} bytes but the strict budget is {budget_bytes}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "adaptation counter overflow"),
            Self::Budget(error) => write!(formatter, "budget planning error: {error}"),
        }
    }
}

impl std::error::Error for AdaptationError {}

impl From<BudgetError> for AdaptationError {
    fn from(error: BudgetError) -> Self {
        Self::Budget(error)
    }
}

/// Current reconstruction and attention quality limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityTargets {
    /// Maximum key reconstruction relative RMS.
    pub key_relative_root_mean_square: f64,
    /// Maximum value reconstruction relative RMS.
    pub value_relative_root_mean_square: f64,
    /// Maximum dense-versus-latent attention absolute error.
    pub attention_max_absolute: f64,
}

impl QualityTargets {
    /// Validates all targets as finite and non-negative.
    pub fn validate(self) -> Result<(), AdaptationError> {
        for (field, value) in [
            (
                "key_relative_root_mean_square",
                self.key_relative_root_mean_square,
            ),
            (
                "value_relative_root_mean_square",
                self.value_relative_root_mean_square,
            ),
            ("attention_max_absolute", self.attention_max_absolute),
        ]
        {
            if !value.is_finite() || value < 0.0
            {
                return Err(AdaptationError::InvalidThreshold { field, value });
            }
        }
        Ok(())
    }

    fn evaluate(self, candidate: &BudgetCandidate) -> BudgetCandidate {
        let mut evaluated = candidate.clone();
        let key_ratio = target_ratio(
            candidate.key_reconstruction.relative_root_mean_square,
            self.key_relative_root_mean_square,
        );
        let value_ratio = target_ratio(
            candidate.value_reconstruction.relative_root_mean_square,
            self.value_relative_root_mean_square,
        );
        let attention_ratio = target_ratio(
            candidate.attention.max_absolute,
            self.attention_max_absolute,
        );

        evaluated.worst_target_ratio = key_ratio.max(value_ratio).max(attention_ratio);
        evaluated.quality_guard_met =
            key_ratio <= 1.0 && value_ratio <= 1.0 && attention_ratio <= 1.0;
        evaluated
    }
}

/// Reason associated with one controller decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionReason {
    /// The first valid proposal becomes active.
    Initial,
    /// The active plan is retained.
    Retained,
    /// The previous active plan no longer fits the strict budget.
    ForcedBudget,
    /// A quality-compliant proposal replaces a non-compliant active plan.
    QualityRecovery,
    /// A discretionary proposal reached the confirmation threshold.
    ConfirmedProposal,
}

impl TransitionReason {
    /// Returns the stable lowercase CSV representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self
        {
            Self::Initial => "initial",
            Self::Retained => "retained",
            Self::ForcedBudget => "forced_budget",
            Self::QualityRecovery => "quality_recovery",
            Self::ConfirmedProposal => "confirmed_proposal",
        }
    }
}

/// Stateful adaptation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptivePolicy {
    /// Consecutive identical discretionary proposals required before adoption.
    pub confirmation_steps: usize,
}

impl AdaptivePolicy {
    /// Validates the policy.
    pub fn validate(self) -> Result<(), AdaptationError> {
        if self.confirmation_steps == 0
        {
            return Err(AdaptationError::ZeroField {
                field: "confirmation_steps",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RankPair {
    key_rank: usize,
    value_rank: usize,
}

impl RankPair {
    const fn from_candidate(candidate: &BudgetCandidate) -> Self {
        Self {
            key_rank: candidate.key_rank,
            value_rank: candidate.value_rank,
        }
    }
}

/// Complete decision returned by one controller observation.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptationDecision {
    /// Raw Phase 3 proposal evaluated against the current quality targets.
    pub proposal: BudgetCandidate,
    /// Plan active after applying the adaptation policy and current targets.
    pub active: BudgetCandidate,
    /// Whether the active rank pair changed during this observation.
    pub transition_applied: bool,
    /// Deterministic transition reason.
    pub reason: TransitionReason,
    /// Current number of consecutive observations for the pending proposal.
    pub pending_confirmation_count: usize,
    /// Whether a different proposal was intentionally suppressed by hysteresis.
    pub suppressed_proposal: bool,
}

/// Deterministic stateful rank controller.
#[derive(Debug, Clone)]
pub struct AdaptiveRankController {
    policy: AdaptivePolicy,
    active: Option<BudgetCandidate>,
    pending_pair: Option<RankPair>,
    pending_count: usize,
}

impl AdaptiveRankController {
    /// Creates an empty controller.
    pub fn new(policy: AdaptivePolicy) -> Result<Self, AdaptationError> {
        policy.validate()?;
        Ok(Self {
            policy,
            active: None,
            pending_pair: None,
            pending_count: 0,
        })
    }

    /// Returns the currently active candidate, when initialized.
    #[must_use]
    pub fn active(&self) -> Option<&BudgetCandidate> {
        self.active.as_ref()
    }

    /// Observes one planner proposal under the current budget and quality targets.
    pub fn observe(
        &mut self,
        proposal: &BudgetCandidate,
        budget_bytes: u64,
        targets: QualityTargets,
    ) -> Result<AdaptationDecision, AdaptationError> {
        targets.validate()?;

        if budget_bytes == 0
        {
            return Err(AdaptationError::ZeroField {
                field: "budget_bytes",
            });
        }
        if proposal.storage.total_bytes > budget_bytes
        {
            return Err(AdaptationError::ProposalOverBudget {
                proposal_bytes: proposal.storage.total_bytes,
                budget_bytes,
            });
        }

        let proposal = targets.evaluate(proposal);
        let Some(current) = self.active.as_ref()
        else
        {
            self.active = Some(proposal.clone());
            self.clear_pending();
            return Ok(self.decision(&proposal, true, TransitionReason::Initial, false));
        };
        let current = targets.evaluate(current);
        self.active = Some(current.clone());

        let current_pair = RankPair::from_candidate(&current);
        let proposal_pair = RankPair::from_candidate(&proposal);

        if current_pair == proposal_pair
        {
            self.active = Some(proposal.clone());
            self.clear_pending();
            return Ok(self.decision(&proposal, false, TransitionReason::Retained, false));
        }

        if current.storage.total_bytes > budget_bytes
        {
            self.active = Some(proposal.clone());
            self.clear_pending();
            return Ok(self.decision(&proposal, true, TransitionReason::ForcedBudget, false));
        }

        if !current.quality_guard_met && proposal.quality_guard_met
        {
            self.active = Some(proposal.clone());
            self.clear_pending();
            return Ok(self.decision(&proposal, true, TransitionReason::QualityRecovery, false));
        }

        if self.pending_pair == Some(proposal_pair)
        {
            self.pending_count = self
                .pending_count
                .checked_add(1)
                .ok_or(AdaptationError::ArithmeticOverflow)?;
        }
        else
        {
            self.pending_pair = Some(proposal_pair);
            self.pending_count = 1;
        }

        if self.pending_count >= self.policy.confirmation_steps
        {
            self.active = Some(proposal.clone());
            self.clear_pending();
            return Ok(self.decision(&proposal, true, TransitionReason::ConfirmedProposal, false));
        }

        Ok(self.decision(&proposal, false, TransitionReason::Retained, true))
    }

    fn clear_pending(&mut self) {
        self.pending_pair = None;
        self.pending_count = 0;
    }

    fn decision(
        &self,
        proposal: &BudgetCandidate,
        transition_applied: bool,
        reason: TransitionReason,
        suppressed_proposal: bool,
    ) -> AdaptationDecision {
        AdaptationDecision {
            proposal: proposal.clone(),
            active: self
                .active
                .as_ref()
                .expect("controller decision requires an active candidate")
                .clone(),
            transition_applied,
            reason,
            pending_confirmation_count: self.pending_count,
            suppressed_proposal,
        }
    }
}

/// One online observation applied to a base Phase 3 scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveObservation {
    /// Budget numerator applied to the base budget denominator.
    pub budget_numerator: u64,
    /// Positive multiplier applied to all three base quality targets.
    pub quality_scale: f64,
}

impl AdaptiveObservation {
    fn validate(self) -> Result<(), AdaptationError> {
        if self.budget_numerator == 0
        {
            return Err(AdaptationError::ZeroField {
                field: "budget_numerator",
            });
        }
        if !self.quality_scale.is_finite() || self.quality_scale <= 0.0
        {
            return Err(AdaptationError::InvalidThreshold {
                field: "quality_scale",
                value: self.quality_scale,
            });
        }
        Ok(())
    }
}

/// One deterministic online-adaptation scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveScenario {
    /// Base Phase 3 dataset and strict quality configuration.
    pub base: BudgetScenario,
    /// Ordered online budget and quality observations.
    pub observations: Vec<AdaptiveObservation>,
    /// Stateful hysteresis policy.
    pub policy: AdaptivePolicy,
}

impl AdaptiveScenario {
    /// Validates the base scenario, observations and policy.
    pub fn validate(&self) -> Result<(), AdaptationError> {
        self.base.validate()?;
        self.policy.validate()?;

        if self.observations.is_empty()
        {
            return Err(AdaptationError::ZeroField {
                field: "observations",
            });
        }
        for observation in &self.observations
        {
            observation.validate()?;
        }
        Ok(())
    }
}

/// One traced step in an adaptive timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveStepReport {
    /// Zero-based step index.
    pub step_index: usize,
    /// Online observation used for this step.
    pub observation: AdaptiveObservation,
    /// Quality targets after applying the observation scale.
    pub targets: QualityTargets,
    /// Raw strict-budget planner report.
    pub planner: BudgetScenarioReport,
    /// Stateful controller decision.
    pub decision: AdaptationDecision,
    /// Stable FNV-1a fingerprint of the step outcome.
    pub step_fingerprint: u64,
}

impl AdaptiveStepReport {
    /// Serializes one stable CSV row.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        let scenario = &self.planner.scenario;
        let proposal = &self.decision.proposal;
        let active = &self.decision.active;

        format!(
            concat!(
                "{},{},{},{},{:.9e},{},{},{:.9e},{:.9e},{:.9e},{:.9e},",
                "{},{},{},{},{},{:.9e},{},{},{},{},{:.9e},{},{},{},{},",
                "{:.9e},{:.9e},{:.9e},{:.9e},{:016x}"
            ),
            scenario.seed,
            self.step_index,
            scenario.token_count,
            scenario.head_dimension,
            scenario.noise_amplitude,
            self.observation.budget_numerator,
            scenario.budget_denominator,
            self.observation.quality_scale,
            self.targets.key_relative_root_mean_square,
            self.targets.value_relative_root_mean_square,
            self.targets.attention_max_absolute,
            self.planner.plan.budget_bytes,
            self.decision.pending_confirmation_count,
            proposal.key_rank,
            proposal.value_rank,
            proposal.storage.total_bytes,
            proposal.worst_target_ratio,
            u8::from(proposal.quality_guard_met),
            active.key_rank,
            active.value_rank,
            active.storage.total_bytes,
            active.worst_target_ratio,
            u8::from(active.quality_guard_met),
            u8::from(self.decision.transition_applied),
            self.decision.reason.as_str(),
            u8::from(self.decision.suppressed_proposal),
            active.storage.compression_ratio,
            active.attention.max_absolute,
            active.key_reconstruction.relative_root_mean_square,
            active.value_reconstruction.relative_root_mean_square,
            self.step_fingerprint,
        )
    }
}

/// Complete report for one adaptive budget and quality timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveTimelineReport {
    /// Original adaptive scenario.
    pub scenario: AdaptiveScenario,
    /// Ordered step reports.
    pub steps: Vec<AdaptiveStepReport>,
    /// Number of rank-pair transitions, including initialization.
    pub transition_count: usize,
    /// Number of immediate budget-forced transitions.
    pub forced_budget_transitions: usize,
    /// Number of immediate quality-recovery transitions.
    pub quality_recovery_transitions: usize,
    /// Number of confirmed discretionary transitions.
    pub confirmed_transitions: usize,
    /// Number of proposals suppressed by hysteresis.
    pub suppressed_proposals: usize,
}

/// Stable CSV header for Phase 4 step reports.
pub const CSV_HEADER: &str = concat!(
    "seed,step_index,token_count,head_dimension,noise_amplitude,",
    "budget_numerator,budget_denominator,quality_scale,key_target_relative_rms,",
    "value_target_relative_rms,attention_target_max_absolute,budget_bytes,",
    "pending_confirmation_count,proposal_key_rank,proposal_value_rank,",
    "proposal_total_bytes,proposal_worst_target_ratio,proposal_quality_guard_met,",
    "active_key_rank,active_value_rank,active_total_bytes,active_worst_target_ratio,",
    "active_quality_guard_met,transition_applied,transition_reason,",
    "suppressed_proposal,compression_ratio,attention_max_absolute,",
    "key_reconstruction_relative_rms,value_reconstruction_relative_rms,",
    "step_fingerprint"
);

/// Runs one deterministic adaptive timeline.
pub fn run_adaptive_scenario(
    scenario: &AdaptiveScenario,
) -> Result<AdaptiveTimelineReport, AdaptationError> {
    scenario.validate()?;

    let mut controller = AdaptiveRankController::new(scenario.policy)?;
    let mut steps = Vec::with_capacity(scenario.observations.len());
    let mut transition_count = 0_usize;
    let mut forced_budget_transitions = 0_usize;
    let mut quality_recovery_transitions = 0_usize;
    let mut confirmed_transitions = 0_usize;
    let mut suppressed_proposals = 0_usize;

    for (step_index, observation) in scenario.observations.iter().copied().enumerate()
    {
        let targets = scaled_targets(&scenario.base, observation.quality_scale)?;
        let mut budget_scenario = scenario.base.clone();
        budget_scenario.budget_numerator = observation.budget_numerator;
        budget_scenario.key_target_relative_root_mean_square =
            targets.key_relative_root_mean_square;
        budget_scenario.value_target_relative_root_mean_square =
            targets.value_relative_root_mean_square;
        budget_scenario.attention_target_max_absolute = targets.attention_max_absolute;

        let planner = run_budget_scenario(&budget_scenario)?;
        let decision =
            controller.observe(&planner.plan.selected, planner.plan.budget_bytes, targets)?;

        if decision.transition_applied
        {
            transition_count = transition_count
                .checked_add(1)
                .ok_or(AdaptationError::ArithmeticOverflow)?;
        }
        match decision.reason
        {
            TransitionReason::ForcedBudget =>
            {
                forced_budget_transitions = forced_budget_transitions
                    .checked_add(1)
                    .ok_or(AdaptationError::ArithmeticOverflow)?;
            },
            TransitionReason::QualityRecovery =>
            {
                quality_recovery_transitions = quality_recovery_transitions
                    .checked_add(1)
                    .ok_or(AdaptationError::ArithmeticOverflow)?;
            },
            TransitionReason::ConfirmedProposal =>
            {
                confirmed_transitions = confirmed_transitions
                    .checked_add(1)
                    .ok_or(AdaptationError::ArithmeticOverflow)?;
            },
            TransitionReason::Initial | TransitionReason::Retained =>
            {},
        }
        if decision.suppressed_proposal
        {
            suppressed_proposals = suppressed_proposals
                .checked_add(1)
                .ok_or(AdaptationError::ArithmeticOverflow)?;
        }

        let step_fingerprint = fingerprint_step(
            step_index,
            observation,
            planner.plan.budget_bytes,
            targets,
            &decision,
        );
        steps.push(AdaptiveStepReport {
            step_index,
            observation,
            targets,
            planner,
            decision,
            step_fingerprint,
        });
    }

    Ok(AdaptiveTimelineReport {
        scenario: scenario.clone(),
        steps,
        transition_count,
        forced_budget_transitions,
        quality_recovery_transitions,
        confirmed_transitions,
        suppressed_proposals,
    })
}

/// Returns the deterministic four-timeline Phase 4 suite.
#[must_use]
pub fn standard_scenarios() -> Vec<AdaptiveScenario> {
    let dimensions = [8_usize, 16];
    let variants = [(0.0_f32, 2_usize, 3_usize), (0.03_f32, 3, 4)];
    let observations = vec![
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1.0,
        },
        AdaptiveObservation {
            budget_numerator: 19,
            quality_scale: 1.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1_000_000.0,
        },
        AdaptiveObservation {
            budget_numerator: 75,
            quality_scale: 1.0,
        },
    ];
    let mut scenarios = Vec::with_capacity(dimensions.len() * variants.len());

    for dimension in dimensions.into_iter()
    {
        for (variant_index, (noise_amplitude, intrinsic_key_rank, intrinsic_value_rank)) in
            variants.into_iter().enumerate()
        {
            let maximum_key_rank = (intrinsic_key_rank + 4).min(dimension);
            let maximum_value_rank = (intrinsic_value_rank + 4).min(dimension);
            let token_count = (maximum_key_rank.max(maximum_value_rank) + 9).max(16);
            let exact = noise_amplitude == 0.0;
            let seed = 0xE1A5_7400_0000_0000_u64
                ^ ((dimension as u64) << 32)
                ^ ((variant_index as u64) << 24)
                ^ ((intrinsic_key_rank as u64) << 16)
                ^ ((intrinsic_value_rank as u64) << 8);

            scenarios.push(AdaptiveScenario {
                base: BudgetScenario {
                    token_count,
                    head_dimension: dimension,
                    query_count: 4,
                    intrinsic_key_rank,
                    intrinsic_value_rank,
                    maximum_key_rank,
                    maximum_value_rank,
                    budget_numerator: observations[0].budget_numerator,
                    budget_denominator: 100,
                    key_target_relative_root_mean_square: if exact { 1.0e-5 } else { 0.03 },
                    value_target_relative_root_mean_square: if exact { 1.0e-5 } else { 0.03 },
                    attention_target_max_absolute: if exact { 2.0e-5 } else { 0.005 },
                    noise_amplitude,
                    signal_amplitude: 1.0,
                    seed,
                },
                observations: observations.clone(),
                policy: AdaptivePolicy {
                    confirmation_steps: 3,
                },
            });
        }
    }

    scenarios
}

/// Runs the deterministic four-timeline Phase 4 suite.
pub fn run_standard_suite() -> Result<Vec<AdaptiveTimelineReport>, AdaptationError> {
    standard_scenarios()
        .iter()
        .map(run_adaptive_scenario)
        .collect()
}

/// Serializes all Phase 4 timeline steps as stable newline-terminated CSV.
#[must_use]
pub fn suite_to_csv(reports: &[AdaptiveTimelineReport]) -> String {
    let mut csv = String::new();
    csv.push_str(CSV_HEADER);
    csv.push('\n');

    for report in reports
    {
        for step in &report.steps
        {
            csv.push_str(&step.to_csv_row());
            csv.push('\n');
        }
    }

    csv
}

fn scaled_targets(
    base: &BudgetScenario,
    quality_scale: f64,
) -> Result<QualityTargets, AdaptationError> {
    if !quality_scale.is_finite() || quality_scale <= 0.0
    {
        return Err(AdaptationError::InvalidThreshold {
            field: "quality_scale",
            value: quality_scale,
        });
    }

    let targets = QualityTargets {
        key_relative_root_mean_square: base.key_target_relative_root_mean_square * quality_scale,
        value_relative_root_mean_square: base.value_target_relative_root_mean_square
            * quality_scale,
        attention_max_absolute: base.attention_target_max_absolute * quality_scale,
    };
    targets.validate()?;
    Ok(targets)
}

fn target_ratio(error: f64, target: f64) -> f64 {
    if target == 0.0
    {
        if error == 0.0 { 0.0 } else { f64::INFINITY }
    }
    else
    {
        error / target
    }
}

fn fingerprint_step(
    step_index: usize,
    observation: AdaptiveObservation,
    budget_bytes: u64,
    targets: QualityTargets,
    decision: &AdaptationDecision,
) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    hash_usize(&mut hash, step_index);
    hash_u64(&mut hash, observation.budget_numerator);
    hash_u64(&mut hash, observation.quality_scale.to_bits());
    hash_u64(&mut hash, budget_bytes);
    hash_u64(&mut hash, targets.key_relative_root_mean_square.to_bits());
    hash_u64(&mut hash, targets.value_relative_root_mean_square.to_bits());
    hash_u64(&mut hash, targets.attention_max_absolute.to_bits());
    hash_usize(&mut hash, decision.proposal.key_rank);
    hash_usize(&mut hash, decision.proposal.value_rank);
    hash_usize(&mut hash, decision.active.key_rank);
    hash_usize(&mut hash, decision.active.value_rank);
    hash_u64(&mut hash, decision.active.storage.total_bytes);
    hash_u64(&mut hash, decision.active.worst_target_ratio.to_bits());
    hash_u64(&mut hash, decision.active.attention.max_absolute.to_bits());
    hash_u64(&mut hash, u64::from(decision.transition_applied));
    hash_u64(&mut hash, u64::from(decision.suppressed_proposal));
    for byte in decision.reason.as_str().as_bytes()
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_usize(hash: &mut u64, value: usize) {
    hash_u64(hash, value as u64);
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes()
    {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptationError, AdaptivePolicy, AdaptiveRankController, CSV_HEADER, QualityTargets,
        TransitionReason, run_adaptive_scenario, run_standard_suite, standard_scenarios,
        suite_to_csv,
    };
    use crate::phase2::{ProjectedAttentionMetrics, ReconstructionMetrics};
    use crate::phase3::{BudgetCandidate, StorageAccounting};

    fn candidate(
        key_rank: usize,
        value_rank: usize,
        total_bytes: u64,
        error: f64,
    ) -> BudgetCandidate {
        let reconstruction = ReconstructionMetrics {
            vectors: 1,
            elements: 1,
            max_absolute: error,
            mean_absolute: error,
            root_mean_square: error,
            relative_root_mean_square: error,
            retained_energy: (1.0 - error).max(0.0),
        };

        BudgetCandidate {
            key_rank,
            value_rank,
            storage: StorageAccounting {
                dense_bytes: 1_024,
                coefficient_bytes: total_bytes / 2,
                basis_bytes: total_bytes - total_bytes / 2,
                total_bytes,
                savings_bytes: 1_024_u64.saturating_sub(total_bytes),
                compression_ratio: 1_024.0 / total_bytes as f64,
            },
            key_reconstruction: reconstruction.clone(),
            value_reconstruction: reconstruction,
            attention: ProjectedAttentionMetrics {
                elements: 1,
                max_absolute: error,
                mean_absolute: error,
                root_mean_square: error,
                output_fingerprint: ((key_rank as u64) << 32) | value_rank as u64,
            },
            worst_target_ratio: error,
            quality_guard_met: error <= 1.0,
        }
    }

    fn targets(limit: f64) -> QualityTargets {
        QualityTargets {
            key_relative_root_mean_square: limit,
            value_relative_root_mean_square: limit,
            attention_max_absolute: limit,
        }
    }

    #[test]
    fn policy_rejects_zero_confirmation_steps() {
        assert_eq!(
            AdaptivePolicy {
                confirmation_steps: 0,
            }
            .validate(),
            Err(AdaptationError::ZeroField {
                field: "confirmation_steps",
            })
        );
    }

    #[test]
    fn initial_proposal_is_adopted() {
        let mut controller = AdaptiveRankController::new(AdaptivePolicy {
            confirmation_steps: 3,
        })
        .unwrap();
        let proposal = candidate(2, 3, 400, 0.5);
        let decision = controller.observe(&proposal, 500, targets(1.0)).unwrap();

        assert!(decision.transition_applied);
        assert_eq!(decision.reason, TransitionReason::Initial);
        assert_eq!(decision.active, proposal);
    }

    #[test]
    fn budget_violation_forces_immediate_downgrade() {
        let mut controller = AdaptiveRankController::new(AdaptivePolicy {
            confirmation_steps: 3,
        })
        .unwrap();
        controller
            .observe(&candidate(4, 4, 700, 0.5), 800, targets(1.0))
            .unwrap();
        let downgrade = candidate(1, 1, 180, 1.4);
        let decision = controller.observe(&downgrade, 200, targets(1.0)).unwrap();

        assert!(decision.transition_applied);
        assert_eq!(decision.reason, TransitionReason::ForcedBudget);
        assert_eq!(decision.active.key_rank, 1);
        assert_eq!(decision.active.value_rank, 1);
    }

    #[test]
    fn discretionary_change_requires_consecutive_confirmation() {
        let mut controller = AdaptiveRankController::new(AdaptivePolicy {
            confirmation_steps: 3,
        })
        .unwrap();
        let low = candidate(1, 1, 180, 0.8);
        let high = candidate(3, 4, 600, 0.5);
        controller.observe(&low, 800, targets(1.0)).unwrap();

        let first = controller.observe(&high, 800, targets(1.0)).unwrap();
        let second = controller.observe(&high, 800, targets(1.0)).unwrap();
        let third = controller.observe(&high, 800, targets(1.0)).unwrap();

        assert!(!first.transition_applied);
        assert!(!second.transition_applied);
        assert!(third.transition_applied);
        assert_eq!(third.reason, TransitionReason::ConfirmedProposal);
        assert_eq!(third.active.key_rank, 3);
        assert_eq!(third.active.value_rank, 4);
    }

    #[test]
    fn oscillating_proposal_is_suppressed() {
        let mut controller = AdaptiveRankController::new(AdaptivePolicy {
            confirmation_steps: 3,
        })
        .unwrap();
        let low = candidate(1, 1, 180, 0.8);
        let high = candidate(3, 3, 520, 0.5);
        controller.observe(&low, 800, targets(1.0)).unwrap();

        let first_high = controller.observe(&high, 800, targets(1.0)).unwrap();
        let reset = controller.observe(&low, 800, targets(1.0)).unwrap();
        let second_high = controller.observe(&high, 800, targets(1.0)).unwrap();

        assert!(first_high.suppressed_proposal);
        assert!(!reset.transition_applied);
        assert!(second_high.suppressed_proposal);
        assert_eq!(second_high.pending_confirmation_count, 1);
        assert_eq!(second_high.active.key_rank, 1);
        assert_eq!(second_high.active.value_rank, 1);
    }

    #[test]
    fn quality_recovery_is_immediate_under_new_targets() {
        let mut controller = AdaptiveRankController::new(AdaptivePolicy {
            confirmation_steps: 4,
        })
        .unwrap();
        let low = candidate(1, 1, 180, 0.8);
        let recovered = candidate(2, 3, 400, 0.1);
        controller.observe(&low, 800, targets(1.0)).unwrap();
        let decision = controller.observe(&recovered, 800, targets(0.2)).unwrap();

        assert!(decision.transition_applied);
        assert_eq!(decision.reason, TransitionReason::QualityRecovery);
        assert_eq!(decision.active.key_rank, 2);
        assert_eq!(decision.active.value_rank, 3);
    }

    #[test]
    fn proposal_over_budget_is_rejected() {
        let mut controller = AdaptiveRankController::new(AdaptivePolicy {
            confirmation_steps: 2,
        })
        .unwrap();
        let error = controller
            .observe(&candidate(2, 2, 300, 0.5), 299, targets(1.0))
            .unwrap_err();

        assert_eq!(
            error,
            AdaptationError::ProposalOverBudget {
                proposal_bytes: 300,
                budget_bytes: 299,
            }
        );
    }

    #[test]
    fn standard_suite_is_deterministic_and_budget_safe() {
        let first = run_standard_suite().unwrap();
        let second = run_standard_suite().unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        assert!(first.iter().all(|report| report.steps.len() == 13));
        assert!(first.iter().all(|report| report.steps.iter().all(|step| {
            step.decision.active.storage.total_bytes <= step.planner.plan.budget_bytes
        })));
        assert!(
            first
                .iter()
                .map(|report| report.forced_budget_transitions)
                .sum::<usize>()
                > 0
        );
        assert!(
            first
                .iter()
                .map(|report| report.quality_recovery_transitions)
                .sum::<usize>()
                > 0
        );
        assert!(
            first
                .iter()
                .map(|report| report.confirmed_transitions)
                .sum::<usize>()
                > 0
        );
        assert!(
            first
                .iter()
                .map(|report| report.suppressed_proposals)
                .sum::<usize>()
                > 0
        );
        assert!(
            first
                .iter()
                .filter(|report| report.scenario.base.noise_amplitude == 0.0)
                .all(|report| {
                    report.forced_budget_transitions > 0
                        && report.quality_recovery_transitions > 0
                        && report.confirmed_transitions > 0
                        && report.suppressed_proposals > 0
                })
        );
    }

    #[test]
    fn exact_timelines_finish_at_intrinsic_ranks() {
        for scenario in standard_scenarios()
            .into_iter()
            .filter(|scenario| scenario.base.noise_amplitude == 0.0)
        {
            let report = run_adaptive_scenario(&scenario).unwrap();
            let final_step = report.steps.last().unwrap();

            assert!(final_step.decision.active.quality_guard_met);
            assert_eq!(
                final_step.decision.active.key_rank,
                scenario.base.intrinsic_key_rank
            );
            assert_eq!(
                final_step.decision.active.value_rank,
                scenario.base.intrinsic_value_rank
            );
        }
    }

    #[test]
    fn csv_export_has_expected_shape() {
        let reports = run_standard_suite().unwrap();
        let csv = suite_to_csv(&reports);
        let mut lines = csv.lines();

        assert_eq!(lines.next(), Some(CSV_HEADER));
        assert_eq!(lines.count(), 52);
        assert_eq!(CSV_HEADER.split(',').count(), 31);
        assert!(
            reports
                .iter()
                .flat_map(|report| &report.steps)
                .all(|step| { step.to_csv_row().split(',').count() == 31 })
        );
    }

    #[test]
    fn empty_observation_schedule_is_rejected() {
        let mut scenario = standard_scenarios().remove(0);
        scenario.observations.clear();

        assert_eq!(
            scenario.validate(),
            Err(AdaptationError::ZeroField {
                field: "observations",
            })
        );
    }
}
