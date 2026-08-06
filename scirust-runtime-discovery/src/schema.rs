use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFamily {
    Level,
    TemporalDelta,
    TemporalSlope,
    RollingStatistic,
    DistributionShape,
    CrossSignalInteraction,
    LayerAggregate,
    Stability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalAvailability {
    CurrentDecision,
    PastOnly,
    FutureDependent,
    TaskOutcomeDependent,
}

impl TemporalAvailability {
    pub fn is_runtime_safe(self) -> bool {
        matches!(self, Self::CurrentDecision | Self::PastOnly)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeClass {
    Constant,
    LinearHeads,
    LinearTokens,
    LinearLayers,
    QuadraticTokens,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCost {
    pub compute_class: ComputeClass,
    pub estimated_scalar_ops: u64,
    pub persistent_state_scalars: u32,
    pub temporary_state_scalars: u32,
}

impl RuntimeCost {
    pub fn constant(estimated_scalar_ops: u64) -> Self {
        Self {
            compute_class: ComputeClass::Constant,
            estimated_scalar_ops,
            persistent_state_scalars: 0,
            temporary_state_scalars: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceBoundary {
    pub development_split: String,
    pub validation_split: String,
    pub holdout_split: String,
    pub holdout_must_remain_unread: bool,
    pub task_outcomes_forbidden_at_runtime: bool,
}

impl Default for EvidenceBoundary {
    fn default() -> Self {
        Self {
            development_split: "development_train".to_string(),
            validation_split: "development_validation".to_string(),
            holdout_split: "untouched_confirmation".to_string(),
            holdout_must_remain_unread: true,
            task_outcomes_forbidden_at_runtime: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureHypothesis {
    pub id: String,
    pub name: String,
    pub family: FeatureFamily,
    pub expression: String,
    pub required_signals: Vec<String>,
    pub temporal_availability: TemporalAvailability,
    pub runtime_cost: RuntimeCost,
    pub rationale: String,
    pub expected_failure_mode: String,
    pub ablation_group: String,
    pub deterministic: bool,
}

impl FeatureHypothesis {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("feature hypothesis id must not be empty".to_string());
        }
        if self.name.trim().is_empty() {
            return Err(format!("feature `{}` has an empty name", self.id));
        }
        if self.expression.trim().is_empty() {
            return Err(format!("feature `{}` has an empty expression", self.id));
        }
        if self.required_signals.is_empty() {
            return Err(format!("feature `{}` has no required signal", self.id));
        }
        if !self.temporal_availability.is_runtime_safe() {
            return Err(format!(
                "feature `{}` is not available at decision time",
                self.id
            ));
        }
        if !self.deterministic {
            return Err(format!("feature `{}` is not deterministic", self.id));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryRequest {
    pub schema_version: u32,
    pub experiment_id: String,
    pub base_features: Vec<String>,
    #[serde(default)]
    pub available_signals: Vec<String>,
    #[serde(default)]
    pub observed_false_positive_ids: Vec<String>,
    #[serde(default)]
    pub evidence_boundary: EvidenceBoundary,
    #[serde(default = "default_max_hypotheses")]
    pub max_hypotheses: usize,
}

fn default_max_hypotheses() -> usize {
    256
}

impl DiscoveryRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported discovery request schema {}",
                self.schema_version
            ));
        }
        if self.experiment_id.trim().is_empty() {
            return Err("experiment_id must not be empty".to_string());
        }
        if self.base_features.is_empty() {
            return Err("at least one base feature is required".to_string());
        }
        if self.max_hypotheses == 0 {
            return Err("max_hypotheses must be positive".to_string());
        }
        let mut unique = BTreeSet::new();
        for feature in &self.base_features {
            if !unique.insert(feature) {
                return Err(format!("duplicate base feature `{feature}`"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureCatalog {
    pub schema_version: u32,
    pub experiment_id: String,
    pub evidence_boundary: EvidenceBoundary,
    pub hypotheses: Vec<FeatureHypothesis>,
    pub rejected: Vec<RejectedHypothesis>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedHypothesis {
    pub id: String,
    pub reason: String,
}

impl FeatureCatalog {
    pub fn validate(&self) -> Result<(), String> {
        let mut ids = BTreeSet::new();
        for hypothesis in &self.hypotheses {
            hypothesis.validate()?;
            if !ids.insert(&hypothesis.id) {
                return Err(format!("duplicate hypothesis id `{}`", hypothesis.id));
            }
        }
        Ok(())
    }
}
