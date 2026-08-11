//! ElasticAutoTuner planning boundary for the resident FLAT attention path.
//!
//! This module is available only with the `flat-autotune` feature. It keeps
//! FLAT kernel details below the SciRust facade while binding every selected
//! plan to a real WGPU hardware profile, a bucketed attention problem class,
//! and the exact reviewed FLAT revision pinned by `scirust-gpu`.
//!
//! The planner does not benchmark inside a latency-sensitive execution path.
//! `Cold` and cache-miss `Learn` use the currently qualified safe FLAT path,
//! `Locked` requires an exact validated persisted plan, and `Audit` never
//! invents a production selection when no persisted plan exists.

pub use crate::{FlatM11ResidentConfig, GpuMatrix, WgpuContext, WgpuFlatM11Bridge};
pub use elastic_autotuner::operating_mode::{ElasticModeError, ElasticStartupStrategy};
pub use elastic_autotuner::{
    ElasticAutoTuner, ElasticCandidate, ElasticCandidateError, ElasticConfig, ElasticEvidence,
    ElasticEvidenceError, ElasticExecutionPlan, ElasticHardwareProfile, ElasticMode,
    ElasticObjective, ElasticParameter, ElasticProblemClass, ElasticSelectionError,
    InMemoryElasticPlanCache, RankedCandidate,
};

use crate::{BackendResult, WgpuComputeAdapter};
use elastic_autotuner::{
    ElasticConstraintSolver, ElasticCostModel, ElasticPlanCache, ElasticPlanKey, ElasticSearchSpace,
};
use scirust_compute::ComputeBackend;

const FLAT_ELASTIC_FAMILY: &str = "flat-attention-f32-wgpu";
const FLAT_KERNEL_FAMILY: &str = "flat-m11-external-asymmetric-projection";
const FLAT_KERNEL_REVISION: &[u8] = b"flat-attention@311f6b89e001d69f53cddcd2f9ba396a6f80c746";
const FLAT_WGSL_MAX_HEAD_DIM: usize = 128;
const FLAT_WGSL_QUERY_ROWS: i64 = 4;
const FLAT_WGSL_KV_TILE: i64 = 8;
const FLAT_WGSL_WORKGROUP_SIZE: i64 = 64;

/// Representation already owned by the resident K/V cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatKvRepresentation {
    /// K is still a raw projected tensor; FLAT rotates Q and K in the fused path.
    Raw,
    /// K has already been RoPE-rotated exactly once by the cache owner.
    PreRotated,
}

impl FlatKvRepresentation {
    const fn class_id(self) -> u8 {
        match self
        {
            Self::Raw => 0,
            Self::PreRotated => 1,
        }
    }
}

/// High-level workload family used for coarse plan validity regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatProblemKind {
    Decode,
    Prefill,
    CrossAttention,
}

impl FlatProblemKind {
    const fn class_id(self) -> u8 {
        match self
        {
            Self::Decode => 0,
            Self::Prefill => 1,
            Self::CrossAttention => 2,
        }
    }
}

/// One semantically validated FLAT request presented to the tuner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatElasticRequest {
    pub config: FlatM11ResidentConfig,
    pub kv_representation: FlatKvRepresentation,
}

impl FlatElasticRequest {
    pub fn new(
        config: FlatM11ResidentConfig,
        kv_representation: FlatKvRepresentation,
    ) -> Result<Self, FlatElasticError> {
        validate_request(config)?;
        Ok(Self {
            config,
            kv_representation,
        })
    }

    pub const fn problem_kind(self) -> FlatProblemKind {
        if !self.config.causal && self.config.query_len != self.config.kv_len
        {
            FlatProblemKind::CrossAttention
        }
        else if self.config.query_len == 1
        {
            FlatProblemKind::Decode
        }
        else
        {
            FlatProblemKind::Prefill
        }
    }
}

/// Planner adapter for one attention validity region.
#[derive(Debug, Clone)]
pub struct FlatElasticPlanner {
    request: FlatElasticRequest,
    problem_class: ElasticProblemClass,
}

impl FlatElasticPlanner {
    pub fn new(request: FlatElasticRequest) -> Result<Self, FlatElasticError> {
        Ok(Self {
            problem_class: problem_class_for(request)?,
            request,
        })
    }

    pub const fn request(&self) -> FlatElasticRequest {
        self.request
    }

    pub fn problem_class(&self) -> &ElasticProblemClass {
        &self.problem_class
    }

    /// Build the canonical Elastic hardware identity from the same WGPU context
    /// that owns the resident FLAT buffers. Cloning the context does not acquire
    /// another adapter or device.
    pub fn hardware_profile(
        &self,
        context: &WgpuContext,
    ) -> Result<ElasticHardwareProfile, FlatElasticError> {
        let adapter = WgpuComputeAdapter::from_context(context.clone());
        let capabilities = ComputeBackend::hardware_capabilities(&adapter);
        ElasticHardwareProfile::from_capabilities(&capabilities)
            .map_err(FlatElasticError::HardwareProfile)
    }

    /// Stable cache key for this hardware/problem/objective validity region.
    pub fn plan_key(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: ElasticHardwareProfile,
    ) -> ElasticPlanKey {
        ElasticPlanKey::new(
            hardware,
            self.problem_class.clone(),
            tuner.config().objective,
        )
    }

    /// Statically rank every currently executable FLAT candidate.
    pub fn rank_candidates(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: &ElasticHardwareProfile,
    ) -> Vec<RankedCandidate> {
        tuner.rank_candidates(hardware, &self.problem_class, self, self, self)
    }

    /// Resolve the production selection without silently weakening the selected
    /// operating mode. This method performs no measurement or exploration.
    pub fn resolve_for_mode<C: ElasticPlanCache>(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: ElasticHardwareProfile,
        cache: &C,
    ) -> Result<FlatPlanResolution, FlatElasticError> {
        let key = self.plan_key(tuner, hardware.clone());
        if let Some(plan) = tuner
            .load_plan_for_mode(cache, &key)
            .map_err(FlatElasticError::Mode)?
        {
            self.require_current_candidate(&plan.evidence.candidate)?;
            return Ok(FlatPlanResolution::Production(Box::new(FlatElasticPlan {
                request: self.request,
                candidate: plan.evidence.candidate.clone(),
                elastic_plan: Some(plan),
                origin: FlatPlanOrigin::Persisted,
            })));
        }

        match tuner
            .startup_strategy(false)
            .map_err(FlatElasticError::Mode)?
        {
            ElasticStartupStrategy::QualifiedHeuristic
            | ElasticStartupStrategy::QualifiedHeuristicThenExplore =>
            {
                let candidate = self
                    .rank_candidates(tuner, &hardware)
                    .into_iter()
                    .next()
                    .ok_or(FlatElasticError::NoQualifiedCandidate)?
                    .candidate;
                self.require_current_candidate(&candidate)?;
                Ok(FlatPlanResolution::Production(Box::new(FlatElasticPlan {
                    request: self.request,
                    candidate,
                    elastic_plan: None,
                    origin: FlatPlanOrigin::QualifiedHeuristic,
                })))
            },
            ElasticStartupStrategy::AuditOnly => Ok(FlatPlanResolution::AuditOnly),
            ElasticStartupStrategy::PersistedOnly
            | ElasticStartupStrategy::PersistedThenExplore =>
            {
                Err(FlatElasticError::MissingProductionPlan)
            },
        }
    }

    /// Evaluate caller-collected, already-qualified evidence outside the latency-sensitive
    /// execution path. This method never reads a clock, launches a benchmark, mutates the plan
    /// cache, or broadens the statically qualified candidate set.
    pub fn evaluate_measured_evidence(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: ElasticHardwareProfile,
        evidence: &[ElasticEvidence],
    ) -> Result<ElasticExecutionPlan, FlatElasticError> {
        tuner
            .require_measurement()
            .map_err(FlatElasticError::Mode)?;
        for record in evidence
        {
            self.require_current_candidate(&record.candidate)?;
        }
        let ranked = self.rank_candidates(tuner, &hardware);
        let selected = tuner
            .select_measured_evidence(&ranked, evidence)
            .map_err(FlatElasticError::Selection)?;
        tuner
            .plan_from_evidence(hardware, self.problem_class.clone(), selected)
            .map_err(FlatElasticError::Evidence)
    }

    /// Promote already-qualified measured evidence to the production cache. Exploration and
    /// selection mutation are both required, so the existing mode contract permits this only
    /// in `Learn`. `Audit` may call `evaluate_measured_evidence` but cannot enter this method.
    pub fn promote_measured_evidence<C: ElasticPlanCache>(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: ElasticHardwareProfile,
        evidence: &[ElasticEvidence],
        cache: &mut C,
    ) -> Result<ElasticExecutionPlan, FlatElasticError> {
        tuner
            .require_exploration()
            .map_err(FlatElasticError::Mode)?;
        let plan = self.evaluate_measured_evidence(tuner, hardware.clone(), evidence)?;
        let key = self.plan_key(tuner, hardware);
        tuner
            .store_plan_for_mode(cache, key, plan.clone())
            .map_err(FlatElasticError::Mode)?;
        Ok(plan)
    }

    /// Fail closed when persisted evidence names a stale, modified, or foreign
    /// kernel rather than silently mapping it onto the current qualified path.
    pub fn require_current_candidate(
        &self,
        candidate: &ElasticCandidate,
    ) -> Result<(), FlatElasticError> {
        if current_candidate()? == *candidate
        {
            Ok(())
        }
        else
        {
            Err(FlatElasticError::UnknownCandidate)
        }
    }
}

impl ElasticSearchSpace for FlatElasticPlanner {
    fn candidates(
        &self,
        _hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        output: &mut Vec<ElasticCandidate>,
    ) {
        if problem == &self.problem_class
        {
            if let Ok(candidate) = current_candidate()
            {
                output.push(candidate);
            }
        }
    }
}

impl ElasticConstraintSolver for FlatElasticPlanner {
    fn is_valid(
        &self,
        _hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        candidate: &ElasticCandidate,
    ) -> bool {
        problem == &self.problem_class && self.require_current_candidate(candidate).is_ok()
    }
}

impl ElasticCostModel for FlatElasticPlanner {
    fn estimated_cost_units(
        &self,
        _hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        candidate: &ElasticCandidate,
        objective: ElasticObjective,
    ) -> u64 {
        if problem != &self.problem_class || self.require_current_candidate(candidate).is_err()
        {
            return u64::MAX;
        }
        match objective
        {
            ElasticObjective::MinTemporaryMemory => candidate.temporary_bytes,
            ElasticObjective::MinLatency
            | ElasticObjective::MaxThroughput
            | ElasticObjective::BalancedLatencyMemory
            | ElasticObjective::DeterministicOnly => 0,
        }
    }
}

/// Source of the production selection returned by [`FlatElasticPlanner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatPlanOrigin {
    QualifiedHeuristic,
    Persisted,
}

/// Result of resolving one tuner mode.
#[derive(Debug)]
pub enum FlatPlanResolution {
    /// A concrete exact FLAT path may execute production work.
    Production(Box<FlatElasticPlan>),
    /// Audit mode has no persisted production selection to replay.
    AuditOnly,
}

/// Exact executable FLAT plan. The candidate has already been checked against
/// the current pinned FLAT revision before this value is constructed.
#[derive(Debug)]
pub struct FlatElasticPlan {
    request: FlatElasticRequest,
    candidate: ElasticCandidate,
    elastic_plan: Option<ElasticExecutionPlan>,
    origin: FlatPlanOrigin,
}

impl FlatElasticPlan {
    pub const fn request(&self) -> FlatElasticRequest {
        self.request
    }

    pub fn candidate(&self) -> &ElasticCandidate {
        &self.candidate
    }

    pub const fn origin(&self) -> FlatPlanOrigin {
        self.origin
    }

    pub fn elastic_plan(&self) -> Option<&ElasticExecutionPlan> {
        self.elastic_plan.as_ref()
    }

    /// Execute with the exact existing zero-copy SciRust ↔ FLAT bridge. This
    /// does not change SciAgent's default selection policy.
    pub fn execute(
        &self,
        bridge: &WgpuFlatM11Bridge,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
    ) -> BackendResult<GpuMatrix> {
        match self.request.kv_representation
        {
            FlatKvRepresentation::Raw => bridge.forward(q, k, v, self.request.config),
            FlatKvRepresentation::PreRotated =>
            {
                bridge.forward_pre_rotated_k(q, k, v, self.request.config)
            },
        }
    }
}

#[derive(Debug)]
pub enum FlatElasticError {
    InvalidRequest(&'static str),
    RequestOverflow(&'static str),
    HardwareProfile(scirust_compute::ProfileEncodingError),
    CandidateEncoding(ElasticCandidateError),
    Evidence(ElasticEvidenceError),
    Selection(ElasticSelectionError),
    Mode(ElasticModeError),
    NoQualifiedCandidate,
    UnknownCandidate,
    MissingProductionPlan,
}

impl core::fmt::Display for FlatElasticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::InvalidRequest(message) => write!(f, "invalid FLAT request: {message}"),
            Self::RequestOverflow(field) => write!(f, "FLAT request `{field}` exceeds index space"),
            Self::HardwareProfile(error) =>
            {
                write!(f, "cannot encode WGPU hardware profile: {error:?}")
            },
            Self::CandidateEncoding(error) => write!(f, "cannot encode FLAT candidate: {error}"),
            Self::Evidence(error) => write!(f, "invalid FLAT tuning evidence: {error}"),
            Self::Selection(error) => write!(f, "cannot select measured FLAT evidence: {error}"),
            Self::Mode(error) => write!(f, "Elastic mode rejected FLAT plan resolution: {error}"),
            Self::NoQualifiedCandidate => write!(f, "no qualified FLAT candidate for this request"),
            Self::UnknownCandidate =>
            {
                write!(f, "Elastic plan names a stale or foreign FLAT candidate")
            },
            Self::MissingProductionPlan =>
            {
                write!(f, "selected Elastic mode requires a persisted FLAT plan")
            },
        }
    }
}

impl std::error::Error for FlatElasticError {}

fn current_candidate() -> Result<ElasticCandidate, FlatElasticError> {
    ElasticCandidate::new(
        FLAT_KERNEL_FAMILY,
        FLAT_KERNEL_REVISION.to_vec(),
        [
            ElasticParameter {
                name: "kv_tile".into(),
                value: FLAT_WGSL_KV_TILE,
            },
            ElasticParameter {
                name: "query_rows".into(),
                value: FLAT_WGSL_QUERY_ROWS,
            },
            ElasticParameter {
                name: "workgroup_size".into(),
                value: FLAT_WGSL_WORKGROUP_SIZE,
            },
        ],
        false,
        0,
    )
    .map_err(FlatElasticError::CandidateEncoding)
}

fn problem_class_for(request: FlatElasticRequest) -> Result<ElasticProblemClass, FlatElasticError> {
    let config = request.config;
    let head_ratio =
        config
            .q_heads
            .checked_div(config.kv_heads)
            .ok_or(FlatElasticError::InvalidRequest(
                "kv_heads must be non-zero",
            ))?;
    let head_ratio = u32::try_from(head_ratio)
        .map_err(|_| FlatElasticError::RequestOverflow("q_heads/kv_heads"))?;
    let head_dim = u16::try_from(config.head_dim)
        .map_err(|_| FlatElasticError::RequestOverflow("head_dim"))?;

    let mut key = Vec::with_capacity(20);
    key.extend_from_slice(b"FAP1");
    key.push(request.problem_kind().class_id());
    key.push(0); // f32 storage/accumulation class for the current bridge.
    key.extend_from_slice(&head_dim.to_le_bytes());
    key.extend_from_slice(&head_ratio.to_le_bytes());
    key.push(length_bucket(config.batch));
    key.push(length_bucket(config.query_len));
    key.push(length_bucket(config.kv_len));
    key.push(u8::from(config.causal));
    key.push(request.kv_representation.class_id());
    Ok(ElasticProblemClass::new(FLAT_ELASTIC_FAMILY, key))
}

fn length_bucket(length: usize) -> u8 {
    match length
    {
        0 => 0,
        1 => 1,
        2..=4 => 2,
        5..=16 => 3,
        17..=64 => 4,
        65..=256 => 5,
        257..=1024 => 6,
        1025..=4096 => 7,
        _ => 8,
    }
}

fn validate_request(config: FlatM11ResidentConfig) -> Result<(), FlatElasticError> {
    if config.batch == 0
        || config.q_heads == 0
        || config.kv_heads == 0
        || config.query_len == 0
        || config.kv_len == 0
        || config.head_dim == 0
    {
        return Err(FlatElasticError::InvalidRequest(
            "batch, heads, lengths, and head_dim must be non-zero",
        ));
    }
    if !config.q_heads.is_multiple_of(config.kv_heads)
    {
        return Err(FlatElasticError::InvalidRequest(
            "q_heads must be divisible by kv_heads",
        ));
    }
    if config.head_dim > FLAT_WGSL_MAX_HEAD_DIM
    {
        return Err(FlatElasticError::InvalidRequest(
            "head_dim exceeds the pinned portable FLAT limit",
        ));
    }
    if !config.head_dim.is_multiple_of(2)
    {
        return Err(FlatElasticError::InvalidRequest(
            "RoPE head_dim must contain complete even/odd pairs",
        ));
    }
    if !config.theta.is_finite() || config.theta <= 0.0
    {
        return Err(FlatElasticError::InvalidRequest(
            "RoPE theta must be finite and positive",
        ));
    }
    if let Some(scale) = config.softmax_scale
    {
        if !scale.is_finite() || scale <= 0.0
        {
            return Err(FlatElasticError::InvalidRequest(
                "softmax scale must be finite and positive",
            ));
        }
    }

    validate_position(
        config.query_position_offset,
        config.query_len,
        "query_position_offset",
    )?;
    validate_position(
        config.query_rope_position_offset,
        config.query_len,
        "query_rope_position_offset",
    )?;
    validate_position(
        config.kv_rope_position_offset,
        config.kv_len,
        "kv_rope_position_offset",
    )?;
    let causal_exclusive = config
        .query_position_offset
        .checked_add(config.query_len)
        .ok_or(FlatElasticError::RequestOverflow("causal query range"))?;
    if causal_exclusive > u32::MAX as usize
    {
        return Err(FlatElasticError::RequestOverflow("causal query range"));
    }
    Ok(())
}

fn validate_position(
    offset: usize,
    length: usize,
    field: &'static str,
) -> Result<(), FlatElasticError> {
    let final_position = offset
        .checked_add(length.saturating_sub(1))
        .ok_or(FlatElasticError::RequestOverflow(field))?;
    if final_position > u32::MAX as usize
    {
        return Err(FlatElasticError::RequestOverflow(field));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use elastic_autotuner::{ElasticMeasurement, ElasticPlanCache};
    use scirust_compute::{DeviceCapabilities, HardwareCapabilities};

    fn config(query_len: usize, kv_len: usize) -> FlatM11ResidentConfig {
        FlatM11ResidentConfig {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            query_len,
            kv_len,
            head_dim: 64,
            causal: true,
            softmax_scale: None,
            query_position_offset: kv_len.saturating_sub(query_len),
            theta: 10_000.0,
            query_rope_position_offset: kv_len.saturating_sub(query_len),
            kv_rope_position_offset: 0,
        }
    }

    fn planner(query_len: usize, kv_len: usize) -> FlatElasticPlanner {
        FlatElasticPlanner::new(
            FlatElasticRequest::new(config(query_len, kv_len), FlatKvRepresentation::PreRotated)
                .unwrap(),
        )
        .unwrap()
    }

    fn hardware() -> ElasticHardwareProfile {
        let capabilities =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        ElasticHardwareProfile::from_capabilities(&capabilities).unwrap()
    }

    fn tuner(mode: ElasticMode, objective: ElasticObjective) -> ElasticAutoTuner {
        ElasticAutoTuner::new(ElasticConfig {
            mode,
            objective,
            max_ranked_candidates: 0,
        })
    }

    fn measured_plan(
        planner: &FlatElasticPlanner,
        tuner: &ElasticAutoTuner,
        hardware: ElasticHardwareProfile,
        candidate: ElasticCandidate,
    ) -> ElasticExecutionPlan {
        let evidence = ElasticEvidence::validated(
            candidate,
            vec![1],
            ElasticMeasurement {
                sample_count: 3,
                median_ns: 10,
                p95_ns: 11,
                p99_ns: 12,
                mad_ns: 1,
            },
        )
        .unwrap();
        tuner
            .plan_from_evidence(hardware, planner.problem_class().clone(), evidence)
            .unwrap()
    }

    #[test]
    fn request_rejects_invalid_grouped_head_mapping() {
        let mut invalid = config(1, 17);
        invalid.q_heads = 7;
        assert!(matches!(
            FlatElasticRequest::new(invalid, FlatKvRepresentation::PreRotated),
            Err(FlatElasticError::InvalidRequest(_))
        ));
    }

    #[test]
    fn problem_classes_cover_validity_regions_not_exact_lengths() {
        let left = planner(1, 17);
        let same_region = planner(1, 64);
        let next_region = planner(1, 65);
        assert_eq!(left.problem_class(), same_region.problem_class());
        assert_ne!(left.problem_class(), next_region.problem_class());
    }

    #[test]
    fn noncausal_rectangular_single_query_is_cross_attention() {
        let mut cross = config(1, 17);
        cross.causal = false;
        let request = FlatElasticRequest::new(cross, FlatKvRepresentation::PreRotated).unwrap();
        assert_eq!(request.problem_kind(), FlatProblemKind::CrossAttention);
    }

    #[test]
    fn every_length_bucket_boundary_is_an_explicit_plan_transition() {
        for (left, right) in [
            (1, 2),
            (4, 5),
            (16, 17),
            (64, 65),
            (256, 257),
            (1024, 1025),
            (4096, 4097),
        ]
        {
            assert_ne!(length_bucket(left), length_bucket(right));
        }

        for (left, right) in [
            (2, 4),
            (5, 16),
            (17, 64),
            (65, 256),
            (257, 1024),
            (1025, 4096),
            (4097, 8192),
        ]
        {
            assert_eq!(length_bucket(left), length_bucket(right));
        }
    }

    #[test]
    fn current_candidate_is_exact_and_stale_candidates_fail_closed() {
        let planner = planner(1, 17);
        let candidate = current_candidate().unwrap();
        planner.require_current_candidate(&candidate).unwrap();
        let stale = ElasticCandidate::new(
            FLAT_KERNEL_FAMILY,
            b"flat-attention@stale".to_vec(),
            candidate.parameters().iter().cloned(),
            false,
            0,
        )
        .unwrap();
        assert!(matches!(
            planner.require_current_candidate(&stale),
            Err(FlatElasticError::UnknownCandidate)
        ));
    }

    #[test]
    fn cold_uses_qualified_heuristic_and_deterministic_only_fails_closed() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let cache = InMemoryElasticPlanCache::default();
        let cold = tuner(ElasticMode::Cold, ElasticObjective::MinLatency);
        let resolved = planner
            .resolve_for_mode(&cold, hardware.clone(), &cache)
            .unwrap();
        match resolved
        {
            FlatPlanResolution::Production(plan) =>
            {
                assert_eq!(plan.origin(), FlatPlanOrigin::QualifiedHeuristic);
                assert!(plan.elastic_plan().is_none());
            },
            FlatPlanResolution::AuditOnly => panic!("cold mode must produce a safe heuristic"),
        }

        let deterministic = tuner(ElasticMode::Cold, ElasticObjective::DeterministicOnly);
        assert!(matches!(
            planner.resolve_for_mode(&deterministic, hardware, &cache),
            Err(FlatElasticError::NoQualifiedCandidate)
        ));
    }

    #[test]
    fn locked_requires_exact_persisted_candidate() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let locked = tuner(ElasticMode::Locked, ElasticObjective::MinLatency);
        let mut cache = InMemoryElasticPlanCache::default();
        assert!(matches!(
            planner.resolve_for_mode(&locked, hardware.clone(), &cache),
            Err(FlatElasticError::Mode(ElasticModeError::LockedPlanMissing))
        ));

        let plan = measured_plan(
            &planner,
            &locked,
            hardware.clone(),
            current_candidate().unwrap(),
        );
        let key = planner.plan_key(&locked, hardware.clone());
        cache.store(key, plan);
        let resolved = planner.resolve_for_mode(&locked, hardware, &cache).unwrap();
        match resolved
        {
            FlatPlanResolution::Production(plan) =>
            {
                assert_eq!(plan.origin(), FlatPlanOrigin::Persisted);
                assert!(plan.elastic_plan().is_some());
            },
            FlatPlanResolution::AuditOnly => panic!("locked mode had a persisted plan"),
        }
    }

    #[test]
    fn audit_without_persisted_plan_does_not_choose_production_candidate() {
        let planner = planner(1, 17);
        let audit = tuner(ElasticMode::Audit, ElasticObjective::MinLatency);
        let cache = InMemoryElasticPlanCache::default();
        assert!(matches!(
            planner
                .resolve_for_mode(&audit, hardware(), &cache)
                .unwrap(),
            FlatPlanResolution::AuditOnly
        ));
    }

    fn evidence(candidate: ElasticCandidate, median_ns: u64) -> ElasticEvidence {
        ElasticEvidence::validated(
            candidate,
            vec![1],
            ElasticMeasurement {
                sample_count: 5,
                median_ns,
                p95_ns: median_ns + 1,
                p99_ns: median_ns + 2,
                mad_ns: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn cold_and_locked_cannot_evaluate_new_measurements() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let records = [evidence(current_candidate().unwrap(), 10)];
        for mode in [ElasticMode::Cold, ElasticMode::Locked]
        {
            let tuner = tuner(mode, ElasticObjective::MinLatency);
            assert!(matches!(
                planner.evaluate_measured_evidence(&tuner, hardware.clone(), &records),
                Err(FlatElasticError::Mode(ElasticModeError::MeasurementForbidden { mode: rejected }))
                    if rejected == mode
            ));
        }
    }

    #[test]
    fn audit_can_replay_evidence_but_cannot_promote_selection() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let audit = tuner(ElasticMode::Audit, ElasticObjective::MinLatency);
        let records = [evidence(current_candidate().unwrap(), 10)];
        let plan = planner
            .evaluate_measured_evidence(&audit, hardware.clone(), &records)
            .unwrap();
        assert_eq!(plan.evidence.candidate, records[0].candidate);

        let mut cache = InMemoryElasticPlanCache::default();
        assert!(matches!(
            planner.promote_measured_evidence(&audit, hardware.clone(), &records, &mut cache,),
            Err(FlatElasticError::Mode(
                ElasticModeError::ExplorationForbidden {
                    mode: ElasticMode::Audit
                }
            ))
        ));
        let key = planner.plan_key(&audit, hardware);
        assert!(cache.load(&key).is_none());
    }

    #[test]
    fn learn_promotes_validated_evidence_and_resolves_persisted_plan() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let learn = tuner(ElasticMode::Learn, ElasticObjective::MinLatency);
        let records = [evidence(current_candidate().unwrap(), 10)];
        let mut cache = InMemoryElasticPlanCache::default();
        let promoted = planner
            .promote_measured_evidence(&learn, hardware.clone(), &records, &mut cache)
            .unwrap();
        assert_eq!(promoted.evidence.candidate, records[0].candidate);

        let resolved = planner.resolve_for_mode(&learn, hardware, &cache).unwrap();
        match resolved
        {
            FlatPlanResolution::Production(plan) =>
            {
                assert_eq!(plan.origin(), FlatPlanOrigin::Persisted);
                assert_eq!(plan.candidate(), &records[0].candidate);
            },
            FlatPlanResolution::AuditOnly => panic!("Learn must resolve the promoted plan"),
        }
    }

    #[test]
    fn foreign_evidence_fails_closed_before_measured_selection() {
        let planner = planner(1, 17);
        let learn = tuner(ElasticMode::Learn, ElasticObjective::MinLatency);
        let current = current_candidate().unwrap();
        let foreign = ElasticCandidate::new(
            "foreign-flat-family",
            current.kernel_revision.clone(),
            current.parameters().iter().cloned(),
            false,
            0,
        )
        .unwrap();
        let records = [evidence(foreign, 1)];
        assert!(matches!(
            planner.evaluate_measured_evidence(&learn, hardware(), &records),
            Err(FlatElasticError::UnknownCandidate)
        ));
    }

    #[test]
    fn foreign_persisted_candidate_is_rejected_after_cache_validation() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let locked = tuner(ElasticMode::Locked, ElasticObjective::MinLatency);
        let current = current_candidate().unwrap();
        let foreign = ElasticCandidate::new(
            "foreign-flat-family",
            current.kernel_revision.clone(),
            current.parameters().iter().cloned(),
            false,
            0,
        )
        .unwrap();
        let plan = measured_plan(&planner, &locked, hardware.clone(), foreign);
        let key = planner.plan_key(&locked, hardware.clone());
        let mut cache = InMemoryElasticPlanCache::default();
        cache.store(key, plan);
        assert!(matches!(
            planner.resolve_for_mode(&locked, hardware, &cache),
            Err(FlatElasticError::UnknownCandidate)
        ));
    }
}
