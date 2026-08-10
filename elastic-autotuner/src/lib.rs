//! # ElasticAutoTuner
//!
//! Deterministic, reusable kernel-autotuning contracts for SciRust.
//!
//! This crate owns *planning evidence*, not kernel implementations. Clients such
//! as FLAT-ATTENTION or SciRust GEMM describe a problem class and a deterministic
//! search space; a cost model ranks statically valid candidates; measurement and
//! correctness evidence can later promote a candidate into an execution plan.

use scirust_compute::{
    HardwareCapabilities, ProfileEncodingError, canonical_hardware_profile_bytes,
};
use std::collections::BTreeMap;

/// Schema version for the public core records in this crate.
pub const ELASTIC_SCHEMA_VERSION: u32 = 1;

/// Tuner operating mode. Production can run fully locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElasticMode {
    Cold,
    Learn,
    Locked,
    Audit,
}

/// Explicit optimization objective; changing objective changes plan selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElasticObjective {
    MinLatency,
    MaxThroughput,
    MinTemporaryMemory,
    BalancedLatencyMemory,
    DeterministicOnly,
}

/// Top-level deterministic policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticConfig {
    pub mode: ElasticMode,
    pub objective: ElasticObjective,
    /// Maximum candidates retained after static ranking. `0` means no truncation.
    pub max_ranked_candidates: usize,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            mode: ElasticMode::Cold,
            objective: ElasticObjective::MinLatency,
            max_ranked_candidates: 0,
        }
    }
}

/// Stable execution-relevant hardware identity.
///
/// The bytes come from SciRust's canonical compute-profile encoding rather than
/// marketing names or iteration-order-dependent maps.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElasticHardwareProfile {
    pub schema_version: u32,
    canonical_bytes: Vec<u8>,
}

impl ElasticHardwareProfile {
    pub fn from_capabilities(
        capabilities: &HardwareCapabilities,
    ) -> Result<Self, ProfileEncodingError> {
        Ok(Self {
            schema_version: ELASTIC_SCHEMA_VERSION,
            canonical_bytes: canonical_hardware_profile_bytes(capabilities)?,
        })
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

/// Client-defined workload class.
///
/// Exact sequence sizes need not be embedded: clients should encode meaningful
/// validity regions/buckets in `class_key`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElasticProblemClass {
    family: String,
    class_key: Vec<u8>,
}

impl ElasticProblemClass {
    pub fn new(family: impl Into<String>, class_key: impl Into<Vec<u8>>) -> Self {
        Self {
            family: family.into(),
            class_key: class_key.into(),
        }
    }

    pub fn family(&self) -> &str {
        &self.family
    }

    pub fn class_key(&self) -> &[u8] {
        &self.class_key
    }
}

/// One named integer tuning parameter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElasticParameter {
    pub name: String,
    pub value: i64,
}

/// A deterministic kernel/configuration candidate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElasticCandidate {
    pub kernel_family: String,
    pub kernel_revision: Vec<u8>,
    parameters: Vec<ElasticParameter>,
    pub deterministic: bool,
    pub temporary_bytes: u64,
}

impl ElasticCandidate {
    /// Build a canonical candidate. Parameter names are sorted and duplicate
    /// names are rejected so construction order cannot change candidate identity.
    pub fn new(
        kernel_family: impl Into<String>,
        kernel_revision: impl Into<Vec<u8>>,
        parameters: impl IntoIterator<Item = ElasticParameter>,
        deterministic: bool,
        temporary_bytes: u64,
    ) -> Result<Self, ElasticCandidateError> {
        let mut parameters: Vec<_> = parameters.into_iter().collect();
        parameters.sort();
        for pair in parameters.windows(2)
        {
            if pair[0].name == pair[1].name
            {
                return Err(ElasticCandidateError::DuplicateParameter(
                    pair[0].name.clone(),
                ));
            }
        }
        Ok(Self {
            kernel_family: kernel_family.into(),
            kernel_revision: kernel_revision.into(),
            parameters,
            deterministic,
            temporary_bytes,
        })
    }

    pub fn parameters(&self) -> &[ElasticParameter] {
        &self.parameters
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElasticCandidateError {
    DuplicateParameter(String),
}

impl core::fmt::Display for ElasticCandidateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::DuplicateParameter(name) => write!(f, "duplicate tuning parameter `{name}`"),
        }
    }
}

impl std::error::Error for ElasticCandidateError {}

/// Deterministic candidate source owned by a kernel family.
pub trait ElasticSearchSpace {
    fn candidates(
        &self,
        hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        output: &mut Vec<ElasticCandidate>,
    );
}

/// Static validity gate run before any benchmark.
pub trait ElasticConstraintSolver {
    fn is_valid(
        &self,
        hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        candidate: &ElasticCandidate,
    ) -> bool;
}

/// Analytical ranking model. Lower cost ranks first.
pub trait ElasticCostModel {
    fn estimated_cost_units(
        &self,
        hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        candidate: &ElasticCandidate,
        objective: ElasticObjective,
    ) -> u64;
}

/// One accepted timing summary. Times are integer nanoseconds to keep ordering
/// deterministic and serialization-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticMeasurement {
    pub sample_count: u32,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub mad_ns: u64,
}

/// Correctness + measurement evidence attached to a concrete candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticEvidence {
    pub candidate: ElasticCandidate,
    /// Client-provided hash/digest bytes for the correctness evidence.
    pub correctness_evidence: Vec<u8>,
    pub measurement: ElasticMeasurement,
}

/// Selected, validated execution plan for one profile/problem/objective tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticExecutionPlan {
    pub schema_version: u32,
    pub hardware: ElasticHardwareProfile,
    pub problem: ElasticProblemClass,
    pub objective: ElasticObjective,
    pub evidence: ElasticEvidence,
}

/// Cache key deliberately excludes exact candidate details: it identifies the
/// hardware/problem/objective region for which one validated plan is selected.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElasticPlanKey {
    pub schema_version: u32,
    pub hardware: ElasticHardwareProfile,
    pub problem: ElasticProblemClass,
    pub objective: ElasticObjective,
}

impl ElasticPlanKey {
    pub fn new(
        hardware: ElasticHardwareProfile,
        problem: ElasticProblemClass,
        objective: ElasticObjective,
    ) -> Self {
        Self {
            schema_version: ELASTIC_SCHEMA_VERSION,
            hardware,
            problem,
            objective,
        }
    }
}

/// Persistent/in-memory cache abstraction. Storage format is intentionally not
/// fixed by the foundation slice.
pub trait ElasticPlanCache {
    fn load(&self, key: &ElasticPlanKey) -> Option<ElasticExecutionPlan>;
    fn store(&mut self, key: ElasticPlanKey, plan: ElasticExecutionPlan);
}

/// Deterministic reference cache useful for tests and small deployments.
#[derive(Debug, Default)]
pub struct InMemoryElasticPlanCache {
    plans: BTreeMap<ElasticPlanKey, ElasticExecutionPlan>,
}

impl ElasticPlanCache for InMemoryElasticPlanCache {
    fn load(&self, key: &ElasticPlanKey) -> Option<ElasticExecutionPlan> {
        self.plans.get(key).cloned()
    }

    fn store(&mut self, key: ElasticPlanKey, plan: ElasticExecutionPlan) {
        self.plans.insert(key, plan);
    }
}

/// Ordered statically-qualified candidate with its analytical score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedCandidate {
    pub estimated_cost_units: u64,
    pub candidate: ElasticCandidate,
}

/// Reusable deterministic autotuner coordinator.
#[derive(Debug, Clone, Copy)]
pub struct ElasticAutoTuner {
    config: ElasticConfig,
}

impl ElasticAutoTuner {
    pub const fn new(config: ElasticConfig) -> Self {
        Self { config }
    }

    pub const fn config(&self) -> ElasticConfig {
        self.config
    }

    /// Generate, canonicalize, constrain and analytically rank candidates.
    ///
    /// For identical inputs and deterministic client implementations, output
    /// ordering is stable. Candidate identity is the final tie-breaker.
    pub fn rank_candidates<S, V, C>(
        &self,
        hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        search_space: &S,
        constraints: &V,
        cost_model: &C,
    ) -> Vec<RankedCandidate>
    where
        S: ElasticSearchSpace,
        V: ElasticConstraintSolver,
        C: ElasticCostModel,
    {
        let mut candidates = Vec::new();
        search_space.candidates(hardware, problem, &mut candidates);
        candidates.sort();
        candidates.dedup();

        let mut ranked: Vec<_> = candidates
            .into_iter()
            .filter(|candidate| {
                constraints.is_valid(hardware, problem, candidate)
                    && (self.config.objective != ElasticObjective::DeterministicOnly
                        || candidate.deterministic)
            })
            .map(|candidate| RankedCandidate {
                estimated_cost_units: cost_model.estimated_cost_units(
                    hardware,
                    problem,
                    &candidate,
                    self.config.objective,
                ),
                candidate,
            })
            .collect();

        ranked.sort_by(|left, right| {
            left.estimated_cost_units
                .cmp(&right.estimated_cost_units)
                .then_with(|| left.candidate.cmp(&right.candidate))
        });
        if self.config.max_ranked_candidates != 0
            && ranked.len() > self.config.max_ranked_candidates
        {
            ranked.truncate(self.config.max_ranked_candidates);
        }
        ranked
    }

    /// Promote already-qualified evidence into a plan. Timing collection itself
    /// belongs to the measurement layer; this function cannot accept evidence
    /// with an empty correctness proof or zero samples.
    pub fn plan_from_evidence(
        &self,
        hardware: ElasticHardwareProfile,
        problem: ElasticProblemClass,
        evidence: ElasticEvidence,
    ) -> Result<ElasticExecutionPlan, ElasticEvidenceError> {
        if evidence.correctness_evidence.is_empty()
        {
            return Err(ElasticEvidenceError::MissingCorrectnessEvidence);
        }
        if evidence.measurement.sample_count == 0
        {
            return Err(ElasticEvidenceError::NoMeasurements);
        }
        if self.config.objective == ElasticObjective::DeterministicOnly
            && !evidence.candidate.deterministic
        {
            return Err(ElasticEvidenceError::NonDeterministicCandidate);
        }
        Ok(ElasticExecutionPlan {
            schema_version: ELASTIC_SCHEMA_VERSION,
            hardware,
            problem,
            objective: self.config.objective,
            evidence,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticEvidenceError {
    MissingCorrectnessEvidence,
    NoMeasurements,
    NonDeterministicCandidate,
}

impl core::fmt::Display for ElasticEvidenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::MissingCorrectnessEvidence => write!(f, "correctness evidence is required"),
            Self::NoMeasurements => write!(f, "at least one timing sample is required"),
            Self::NonDeterministicCandidate => {
                write!(f, "deterministic-only objective rejected the candidate")
            },
        }
    }
}

impl std::error::Error for ElasticEvidenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_compute::{DeviceCapabilities, HardwareCapabilities};

    struct Space;
    impl ElasticSearchSpace for Space {
        fn candidates(
            &self,
            _hardware: &ElasticHardwareProfile,
            _problem: &ElasticProblemClass,
            output: &mut Vec<ElasticCandidate>,
        ) {
            for tile in [64_i64, 16, 32, 16]
            {
                output.push(
                    ElasticCandidate::new(
                        "gemm",
                        b"kernel-v1".to_vec(),
                        [ElasticParameter {
                            name: "tile".into(),
                            value: tile,
                        }],
                        tile != 64,
                        tile as u64 * 4,
                    )
                    .unwrap(),
                );
            }
        }
    }

    struct Constraints;
    impl ElasticConstraintSolver for Constraints {
        fn is_valid(
            &self,
            _hardware: &ElasticHardwareProfile,
            _problem: &ElasticProblemClass,
            candidate: &ElasticCandidate,
        ) -> bool {
            candidate.parameters()[0].value >= 16
        }
    }

    struct Cost;
    impl ElasticCostModel for Cost {
        fn estimated_cost_units(
            &self,
            _hardware: &ElasticHardwareProfile,
            _problem: &ElasticProblemClass,
            candidate: &ElasticCandidate,
            _objective: ElasticObjective,
        ) -> u64 {
            candidate.parameters()[0].value.abs_diff(32)
        }
    }

    fn profile() -> ElasticHardwareProfile {
        let hardware =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        ElasticHardwareProfile::from_capabilities(&hardware).unwrap()
    }

    #[test]
    fn candidate_generation_is_canonical_and_deduplicated() {
        let tuner = ElasticAutoTuner::new(ElasticConfig::default());
        let problem = ElasticProblemClass::new("gemm", b"m128-k128-n128".to_vec());
        let ranked = tuner.rank_candidates(&profile(), &problem, &Space, &Constraints, &Cost);
        let tiles: Vec<_> = ranked
            .iter()
            .map(|rank| rank.candidate.parameters()[0].value)
            .collect();
        assert_eq!(tiles, vec![32, 16, 64]);
    }

    #[test]
    fn deterministic_objective_filters_non_deterministic_candidates() {
        let tuner = ElasticAutoTuner::new(ElasticConfig {
            objective: ElasticObjective::DeterministicOnly,
            ..ElasticConfig::default()
        });
        let problem = ElasticProblemClass::new("gemm", vec![1]);
        let ranked = tuner.rank_candidates(&profile(), &problem, &Space, &Constraints, &Cost);
        assert!(ranked.iter().all(|rank| rank.candidate.deterministic));
        assert!(
            !ranked
                .iter()
                .any(|rank| rank.candidate.parameters()[0].value == 64)
        );
    }

    #[test]
    fn correctness_gate_precedes_plan_promotion() {
        let tuner = ElasticAutoTuner::new(ElasticConfig::default());
        let hardware = profile();
        let problem = ElasticProblemClass::new("attention", vec![2, 4, 8]);
        let candidate = ElasticCandidate::new(
            "flat",
            vec![7],
            std::iter::empty::<ElasticParameter>(),
            true,
            0,
        )
        .unwrap();
        let measurement = ElasticMeasurement {
            sample_count: 10,
            median_ns: 100,
            p95_ns: 110,
            p99_ns: 120,
            mad_ns: 3,
        };
        let missing = tuner.plan_from_evidence(
            hardware.clone(),
            problem.clone(),
            ElasticEvidence {
                candidate: candidate.clone(),
                correctness_evidence: Vec::new(),
                measurement,
            },
        );
        assert_eq!(
            missing.unwrap_err(),
            ElasticEvidenceError::MissingCorrectnessEvidence
        );

        let plan = tuner
            .plan_from_evidence(
                hardware,
                problem,
                ElasticEvidence {
                    candidate,
                    correctness_evidence: vec![1, 2, 3],
                    measurement,
                },
            )
            .unwrap();
        assert_eq!(plan.schema_version, ELASTIC_SCHEMA_VERSION);
    }

    #[test]
    fn in_memory_plan_cache_is_keyed_by_profile_problem_and_objective() {
        let tuner = ElasticAutoTuner::new(ElasticConfig::default());
        let hardware = profile();
        let problem = ElasticProblemClass::new("gemm", vec![9]);
        let candidate = ElasticCandidate::new(
            "gemm",
            vec![1],
            std::iter::empty::<ElasticParameter>(),
            true,
            0,
        )
        .unwrap();
        let plan = tuner
            .plan_from_evidence(
                hardware.clone(),
                problem.clone(),
                ElasticEvidence {
                    candidate,
                    correctness_evidence: vec![42],
                    measurement: ElasticMeasurement {
                        sample_count: 3,
                        median_ns: 10,
                        p95_ns: 11,
                        p99_ns: 12,
                        mad_ns: 1,
                    },
                },
            )
            .unwrap();
        let key = ElasticPlanKey::new(hardware, problem, ElasticObjective::MinLatency);
        let mut cache = InMemoryElasticPlanCache::default();
        cache.store(key.clone(), plan.clone());
        assert_eq!(cache.load(&key), Some(plan));
    }
}
