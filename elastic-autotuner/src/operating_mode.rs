//! Operational semantics for `Cold`, `Learn`, `Locked`, and `Audit` modes.
//!
//! The mode enum is not merely telemetry: these helpers define which source may
//! choose the production plan, whether exploration is permitted, and whether a
//! validated selection may mutate the persistent plan cache.

use crate::{
    ELASTIC_SCHEMA_VERSION, ElasticAutoTuner, ElasticExecutionPlan, ElasticMode, ElasticPlanCache,
    ElasticPlanKey,
};

/// Startup behavior implied by one tuner operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticStartupStrategy {
    /// Use the statically qualified safe heuristic immediately; do not explore.
    QualifiedHeuristic,
    /// Use the heuristic immediately while controlled exploration is allowed.
    QualifiedHeuristicThenExplore,
    /// Use a validated persisted plan while controlled exploration may improve it.
    PersistedThenExplore,
    /// Production must use the validated persisted plan and nothing else.
    PersistedOnly,
    /// Replay validation/measurement only; do not alter production selection.
    AuditOnly,
}

/// Explicit capabilities attached to the current operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticModePermissions {
    pub may_use_heuristic_plan: bool,
    pub may_use_persisted_plan: bool,
    pub may_measure: bool,
    pub may_explore: bool,
    pub may_mutate_selection: bool,
    pub audit_replay: bool,
}

impl ElasticModePermissions {
    pub const fn for_mode(mode: ElasticMode) -> Self {
        match mode {
            ElasticMode::Cold => Self {
                may_use_heuristic_plan: true,
                may_use_persisted_plan: false,
                may_measure: false,
                may_explore: false,
                may_mutate_selection: false,
                audit_replay: false,
            },
            ElasticMode::Learn => Self {
                may_use_heuristic_plan: true,
                may_use_persisted_plan: true,
                may_measure: true,
                may_explore: true,
                may_mutate_selection: true,
                audit_replay: false,
            },
            ElasticMode::Locked => Self {
                may_use_heuristic_plan: false,
                may_use_persisted_plan: true,
                may_measure: false,
                may_explore: false,
                may_mutate_selection: false,
                audit_replay: false,
            },
            ElasticMode::Audit => Self {
                may_use_heuristic_plan: false,
                may_use_persisted_plan: true,
                may_measure: true,
                may_explore: false,
                may_mutate_selection: false,
                audit_replay: true,
            },
        }
    }
}

impl ElasticAutoTuner {
    pub const fn mode_permissions(&self) -> ElasticModePermissions {
        ElasticModePermissions::for_mode(self.config().mode)
    }

    /// Decide the startup behavior without silently relaxing the selected mode.
    pub fn startup_strategy(
        &self,
        has_valid_persisted_plan: bool,
    ) -> Result<ElasticStartupStrategy, ElasticModeError> {
        match self.config().mode {
            ElasticMode::Cold => Ok(ElasticStartupStrategy::QualifiedHeuristic),
            ElasticMode::Learn => {
                if has_valid_persisted_plan {
                    Ok(ElasticStartupStrategy::PersistedThenExplore)
                } else {
                    Ok(ElasticStartupStrategy::QualifiedHeuristicThenExplore)
                }
            }
            ElasticMode::Locked => {
                if has_valid_persisted_plan {
                    Ok(ElasticStartupStrategy::PersistedOnly)
                } else {
                    Err(ElasticModeError::LockedPlanMissing)
                }
            }
            ElasticMode::Audit => Ok(ElasticStartupStrategy::AuditOnly),
        }
    }

    /// Load a cached plan according to the selected operating mode.
    ///
    /// `Cold` deliberately ignores retained plans and starts from the safe
    /// heuristic. `Locked` fails closed if the exact cache key has no valid plan.
    pub fn load_plan_for_mode<C: ElasticPlanCache>(
        &self,
        cache: &C,
        key: &ElasticPlanKey,
    ) -> Result<Option<ElasticExecutionPlan>, ElasticModeError> {
        if self.config().mode == ElasticMode::Cold {
            return Ok(None);
        }

        let plan = cache.load(key);
        match plan {
            Some(plan) => {
                validate_cached_plan(key, &plan)?;
                Ok(Some(plan))
            }
            None if self.config().mode == ElasticMode::Locked => {
                Err(ElasticModeError::LockedPlanMissing)
            }
            None => Ok(None),
        }
    }

    /// Commit a validated production selection only in `Learn` mode.
    ///
    /// `Cold` has no learned state, `Locked` is immutable, and `Audit` must never
    /// change production selection while replaying evidence.
    pub fn store_plan_for_mode<C: ElasticPlanCache>(
        &self,
        cache: &mut C,
        key: ElasticPlanKey,
        plan: ElasticExecutionPlan,
    ) -> Result<(), ElasticModeError> {
        if !self.mode_permissions().may_mutate_selection {
            return Err(ElasticModeError::SelectionMutationForbidden {
                mode: self.config().mode,
            });
        }
        validate_cached_plan(&key, &plan)?;
        cache.store(key, plan);
        Ok(())
    }

    pub fn require_exploration(&self) -> Result<(), ElasticModeError> {
        if self.mode_permissions().may_explore {
            Ok(())
        } else {
            Err(ElasticModeError::ExplorationForbidden {
                mode: self.config().mode,
            })
        }
    }

    pub fn require_measurement(&self) -> Result<(), ElasticModeError> {
        if self.mode_permissions().may_measure {
            Ok(())
        } else {
            Err(ElasticModeError::MeasurementForbidden {
                mode: self.config().mode,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticModeError {
    LockedPlanMissing,
    ExplorationForbidden { mode: ElasticMode },
    MeasurementForbidden { mode: ElasticMode },
    SelectionMutationForbidden { mode: ElasticMode },
    CacheKeyMismatch,
    UnsupportedPlanSchema { expected: u32, actual: u32 },
    InvalidCachedEvidence,
}

impl core::fmt::Display for ElasticModeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::LockedPlanMissing => write!(f, "locked mode requires a validated persisted plan"),
            Self::ExplorationForbidden { mode } => {
                write!(f, "exploration is forbidden in {mode:?} mode")
            }
            Self::MeasurementForbidden { mode } => {
                write!(f, "measurement is forbidden in {mode:?} mode")
            }
            Self::SelectionMutationForbidden { mode } => {
                write!(f, "selection mutation is forbidden in {mode:?} mode")
            }
            Self::CacheKeyMismatch => write!(f, "cached plan does not match its plan-cache key"),
            Self::UnsupportedPlanSchema { expected, actual } => write!(
                f,
                "cached plan schema mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidCachedEvidence => write!(f, "cached plan contains invalid evidence"),
        }
    }
}

impl std::error::Error for ElasticModeError {}

fn validate_cached_plan(
    key: &ElasticPlanKey,
    plan: &ElasticExecutionPlan,
) -> Result<(), ElasticModeError> {
    if plan.schema_version != ELASTIC_SCHEMA_VERSION {
        return Err(ElasticModeError::UnsupportedPlanSchema {
            expected: ELASTIC_SCHEMA_VERSION,
            actual: plan.schema_version,
        });
    }
    if key.schema_version != ELASTIC_SCHEMA_VERSION
        || plan.hardware != key.hardware
        || plan.problem != key.problem
        || plan.objective != key.objective
    {
        return Err(ElasticModeError::CacheKeyMismatch);
    }
    plan.evidence
        .validate()
        .map_err(|_| ElasticModeError::InvalidCachedEvidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ElasticCandidate, ElasticConfig, ElasticEvidence, ElasticHardwareProfile, ElasticMeasurement,
        ElasticObjective, ElasticParameter, ElasticProblemClass, InMemoryElasticPlanCache,
    };
    use scirust_compute::{DeviceCapabilities, HardwareCapabilities};

    fn hardware() -> ElasticHardwareProfile {
        let capabilities =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        ElasticHardwareProfile::from_capabilities(&capabilities).unwrap()
    }

    fn plan(objective: ElasticObjective) -> (ElasticPlanKey, ElasticExecutionPlan) {
        let hardware = hardware();
        let problem = ElasticProblemClass::new("sgemm", vec![1, 2, 3]);
        let candidate = ElasticCandidate::new(
            "sgemm",
            vec![1],
            [ElasticParameter {
                name: "path".into(),
                value: 0,
            }],
            true,
            0,
        )
        .unwrap();
        let evidence = ElasticEvidence::validated(
            candidate,
            vec![9],
            ElasticMeasurement {
                sample_count: 3,
                median_ns: 10,
                p95_ns: 11,
                p99_ns: 12,
                mad_ns: 1,
            },
        )
        .unwrap();
        let key = ElasticPlanKey::new(hardware.clone(), problem.clone(), objective);
        let plan = ElasticExecutionPlan {
            schema_version: ELASTIC_SCHEMA_VERSION,
            hardware,
            problem,
            objective,
            evidence,
        };
        (key, plan)
    }

    fn tuner(mode: ElasticMode) -> ElasticAutoTuner {
        ElasticAutoTuner::new(ElasticConfig {
            mode,
            ..ElasticConfig::default()
        })
    }

    #[test]
    fn startup_strategies_match_the_documented_modes() {
        assert_eq!(
            tuner(ElasticMode::Cold).startup_strategy(true).unwrap(),
            ElasticStartupStrategy::QualifiedHeuristic
        );
        assert_eq!(
            tuner(ElasticMode::Learn).startup_strategy(false).unwrap(),
            ElasticStartupStrategy::QualifiedHeuristicThenExplore
        );
        assert_eq!(
            tuner(ElasticMode::Learn).startup_strategy(true).unwrap(),
            ElasticStartupStrategy::PersistedThenExplore
        );
        assert_eq!(
            tuner(ElasticMode::Locked).startup_strategy(true).unwrap(),
            ElasticStartupStrategy::PersistedOnly
        );
        assert_eq!(
            tuner(ElasticMode::Locked).startup_strategy(false),
            Err(ElasticModeError::LockedPlanMissing)
        );
        assert_eq!(
            tuner(ElasticMode::Audit).startup_strategy(false).unwrap(),
            ElasticStartupStrategy::AuditOnly
        );
    }

    #[test]
    fn only_learn_may_explore_and_mutate_selection() {
        for mode in [ElasticMode::Cold, ElasticMode::Locked, ElasticMode::Audit] {
            assert!(tuner(mode).require_exploration().is_err());
            let (key, plan) = plan(ElasticObjective::MinLatency);
            let mut cache = InMemoryElasticPlanCache::default();
            assert!(tuner(mode).store_plan_for_mode(&mut cache, key, plan).is_err());
        }

        let (key, plan) = plan(ElasticObjective::MinLatency);
        let mut cache = InMemoryElasticPlanCache::default();
        tuner(ElasticMode::Learn)
            .store_plan_for_mode(&mut cache, key.clone(), plan.clone())
            .unwrap();
        assert_eq!(cache.load(&key), Some(plan));
    }

    #[test]
    fn locked_mode_fails_closed_without_exact_persisted_plan() {
        let (key, plan) = plan(ElasticObjective::MinLatency);
        let mut cache = InMemoryElasticPlanCache::default();
        assert_eq!(
            tuner(ElasticMode::Locked).load_plan_for_mode(&cache, &key),
            Err(ElasticModeError::LockedPlanMissing)
        );
        cache.store(key.clone(), plan.clone());
        assert_eq!(
            tuner(ElasticMode::Locked)
                .load_plan_for_mode(&cache, &key)
                .unwrap(),
            Some(plan)
        );
    }

    #[test]
    fn cold_ignores_persisted_state_and_uses_heuristic() {
        let (key, plan) = plan(ElasticObjective::MinLatency);
        let mut cache = InMemoryElasticPlanCache::default();
        cache.store(key.clone(), plan);
        assert_eq!(
            tuner(ElasticMode::Cold)
                .load_plan_for_mode(&cache, &key)
                .unwrap(),
            None
        );
    }

    #[test]
    fn audit_may_measure_but_never_mutate_selection() {
        let audit = tuner(ElasticMode::Audit);
        assert_eq!(audit.require_measurement(), Ok(()));
        assert!(!audit.mode_permissions().may_explore);
        assert!(!audit.mode_permissions().may_mutate_selection);
        assert!(audit.mode_permissions().audit_replay);
    }

    #[test]
    fn cache_key_mismatch_is_rejected_before_use_or_store() {
        let (key, mut plan) = plan(ElasticObjective::MinLatency);
        plan.objective = ElasticObjective::MinTemporaryMemory;
        let mut cache = InMemoryElasticPlanCache::default();
        cache.store(key.clone(), plan.clone());
        assert_eq!(
            tuner(ElasticMode::Locked).load_plan_for_mode(&cache, &key),
            Err(ElasticModeError::CacheKeyMismatch)
        );
        assert_eq!(
            tuner(ElasticMode::Learn).store_plan_for_mode(&mut cache, key, plan),
            Err(ElasticModeError::CacheKeyMismatch)
        );
    }
}
