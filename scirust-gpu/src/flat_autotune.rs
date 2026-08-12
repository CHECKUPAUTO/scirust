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
pub use elastic_autotuner::measurement_protocol::ElasticResidenceMode;
pub use elastic_autotuner::operating_mode::{ElasticModeError, ElasticStartupStrategy};
pub use elastic_autotuner::persistence::{ElasticPersistedPlan, ElasticPersistenceError};
pub use elastic_autotuner::{
    ElasticAutoTuner, ElasticCandidate, ElasticCandidateError, ElasticConfig, ElasticEvidence,
    ElasticEvidenceError, ElasticExecutionPlan, ElasticHardwareProfile, ElasticMode,
    ElasticObjective, ElasticParameter, ElasticProblemClass, ElasticSelectionError,
    InMemoryElasticPlanCache, RankedCandidate,
};

use crate::{BackendError, WgpuComputeAdapter};
use elastic_autotuner::{
    ElasticConstraintSolver, ElasticCostModel, ElasticPlanCache, ElasticPlanKey, ElasticSearchSpace,
};
use scirust_compute::ComputeBackend;
use std::collections::BTreeSet;

const FLAT_ELASTIC_FAMILY: &str = "flat-attention-f32-wgpu";
const FLAT_KERNEL_FAMILY: &str = "flat-m11-external-asymmetric-projection";
const FLAT_M15_KERNEL_FAMILY: &str = "flat-m15-resident-decode";
const FLAT_KERNEL_REVISION: &[u8] = b"flat-attention@31a33f5e7193dda5ab777c079154ec5ee49ddf4b";

#[cfg(test)]
#[test]
fn flat_candidate_revision_matches_manifest_pin() {
    let revision = FLAT_KERNEL_REVISION
        .strip_prefix(b"flat-attention@")
        .expect("FLAT candidate revision must use the flat-attention@<sha> identity");
    let revision = core::str::from_utf8(revision).expect("FLAT revision SHA must be UTF-8");
    let dependency = include_str!("../Cargo.toml")
        .lines()
        .find(|line| line.trim_start().starts_with("flat-attention = {"))
        .expect("scirust-gpu manifest must declare the FLAT dependency");
    let expected = format!("rev = \"{revision}\"");
    assert!(
        dependency.contains(&expected),
        "Elastic candidate identity {revision} does not match Cargo FLAT pin: {dependency}"
    );
}

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
        hardware_profile_from_context(context)
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
                problem_class: self.problem_class.clone(),
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
                    problem_class: self.problem_class.clone(),
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
            || (m15_candidate_eligible(self.request) && m15_candidate()? == *candidate)
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
            if m15_candidate_eligible(self.request)
            {
                if let Ok(candidate) = m15_candidate()
                {
                    output.push(candidate);
                }
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
            | ElasticObjective::DeterministicOnly =>
            {
                if current_candidate().is_ok_and(|current| current == *candidate)
                {
                    0
                }
                else
                {
                    1
                }
            },
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
    problem_class: ElasticProblemClass,
    candidate: ElasticCandidate,
    elastic_plan: Option<ElasticExecutionPlan>,
    origin: FlatPlanOrigin,
}

impl FlatElasticPlan {
    pub fn problem_class(&self) -> &ElasticProblemClass {
        &self.problem_class
    }

    /// Whether a dynamic request lies inside this plan's qualified validity region.
    pub fn accepts(&self, request: FlatElasticRequest) -> bool {
        problem_class_for(request).is_ok_and(|problem| problem == self.problem_class)
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

    /// Execute a request that lies inside this plan's qualified validity region.
    /// The dynamic geometry/positions come from `request`; the selected candidate comes from
    /// the plan. This is what lets sequential decode reuse one plan until an H2 boundary.
    pub fn execute(
        &self,
        bridge: &WgpuFlatM11Bridge,
        request: FlatElasticRequest,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
    ) -> Result<GpuMatrix, FlatElasticError> {
        if !self.accepts(request)
        {
            return Err(FlatElasticError::PlanRegionMismatch);
        }
        if current_candidate()? == self.candidate
        {
            let result = match request.kv_representation
            {
                FlatKvRepresentation::Raw => bridge.forward(q, k, v, request.config),
                FlatKvRepresentation::PreRotated =>
                {
                    bridge.forward_pre_rotated_k(q, k, v, request.config)
                },
            };
            return result.map_err(FlatElasticError::Backend);
        }
        if m15_candidate_eligible(request) && m15_candidate()? == self.candidate
        {
            return bridge
                .forward_pre_rotated_k_m15(q, k, v, request.config)
                .map_err(FlatElasticError::Backend);
        }
        Err(FlatElasticError::UnknownCandidate)
    }
}

/// Stateful execution-policy cache for latency-sensitive FLAT callers.
///
/// Planning is repeated only when the request crosses an H2 validity-region boundary. The
/// runtime never benchmarks: caller-collected evidence must be evaluated/promoted explicitly
/// through the H3 methods. This type is intentionally below SciAgent so model code asks for an
/// attention execution policy without knowing FLAT kernel identities.
pub struct FlatElasticRuntime<C> {
    tuner: ElasticAutoTuner,
    hardware: ElasticHardwareProfile,
    cache: C,
    active_plan: Option<FlatElasticPlan>,
}

impl<C: ElasticPlanCache> FlatElasticRuntime<C> {
    pub const fn from_hardware(
        config: ElasticConfig,
        hardware: ElasticHardwareProfile,
        cache: C,
    ) -> Self {
        Self {
            tuner: ElasticAutoTuner::new(config),
            hardware,
            cache,
            active_plan: None,
        }
    }

    pub const fn tuner(&self) -> &ElasticAutoTuner {
        &self.tuner
    }

    pub const fn hardware(&self) -> &ElasticHardwareProfile {
        &self.hardware
    }

    pub const fn cache(&self) -> &C {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut C {
        &mut self.cache
    }

    pub const fn active_plan(&self) -> Option<&FlatElasticPlan> {
        self.active_plan.as_ref()
    }

    /// Resolve at most once per validity region. No measurement or exploration occurs here.
    pub fn resolve_request(
        &mut self,
        request: FlatElasticRequest,
    ) -> Result<&FlatElasticPlan, FlatElasticError> {
        let reuse = self
            .active_plan
            .as_ref()
            .is_some_and(|plan| plan.accepts(request));
        if !reuse
        {
            let planner = FlatElasticPlanner::new(request)?;
            self.active_plan =
                match planner.resolve_for_mode(&self.tuner, self.hardware.clone(), &self.cache)?
                {
                    FlatPlanResolution::Production(plan) => Some(*plan),
                    FlatPlanResolution::AuditOnly => None,
                };
        }
        self.active_plan
            .as_ref()
            .ok_or(FlatElasticError::MissingProductionPlan)
    }

    pub fn execute(
        &mut self,
        bridge: &WgpuFlatM11Bridge,
        request: FlatElasticRequest,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
    ) -> Result<GpuMatrix, FlatElasticError> {
        let plan = self.resolve_request(request)?;
        plan.execute(bridge, request, q, k, v)
    }

    /// Replay measured evidence without changing the production plan/cache.
    pub fn evaluate_measured_evidence(
        &self,
        request: FlatElasticRequest,
        evidence: &[ElasticEvidence],
    ) -> Result<ElasticExecutionPlan, FlatElasticError> {
        FlatElasticPlanner::new(request)?.evaluate_measured_evidence(
            &self.tuner,
            self.hardware.clone(),
            evidence,
        )
    }

    /// Learn-only promotion. The active selection is invalidated so the next production request
    /// reloads the newly persisted validated plan for its region.
    pub fn promote_measured_evidence(
        &mut self,
        request: FlatElasticRequest,
        evidence: &[ElasticEvidence],
    ) -> Result<ElasticExecutionPlan, FlatElasticError> {
        let plan = FlatElasticPlanner::new(request)?.promote_measured_evidence(
            &self.tuner,
            self.hardware.clone(),
            evidence,
            &mut self.cache,
        )?;
        self.active_plan = None;
        Ok(plan)
    }
}

impl FlatElasticRuntime<InMemoryElasticPlanCache> {
    /// Construct the normal in-process runtime from the exact resident WGPU context.
    pub fn in_memory(
        config: ElasticConfig,
        context: &WgpuContext,
    ) -> Result<Self, FlatElasticError> {
        Ok(Self::from_hardware(
            config,
            hardware_profile_from_context(context)?,
            InMemoryElasticPlanCache::default(),
        ))
    }

    /// Construct an in-process runtime after decoding and validating caller-supplied
    /// persisted records. This boundary performs no filesystem I/O: callers own
    /// record discovery/loading and provide the exact bytes before decode begins.
    ///
    /// Every record must be an explicitly selected resident measurement for the
    /// exact runtime hardware/objective, name a current FLAT candidate revision,
    /// carry the current FLAT revision as an invalidation dependency, and contain
    /// every caller-required dependency. Duplicate validity-region keys fail closed.
    pub fn in_memory_with_persisted_records(
        config: ElasticConfig,
        context: &WgpuContext,
        encoded_records: &[Vec<u8>],
        required_invalidation_dependencies: &[Vec<u8>],
    ) -> Result<Self, FlatElasticError> {
        let hardware = hardware_profile_from_context(context)?;
        let cache = persisted_cache_from_records(
            config,
            &hardware,
            encoded_records,
            required_invalidation_dependencies,
        )?;
        Ok(Self::from_hardware(config, hardware, cache))
    }
}

fn persisted_cache_from_records(
    config: ElasticConfig,
    hardware: &ElasticHardwareProfile,
    encoded_records: &[Vec<u8>],
    required_invalidation_dependencies: &[Vec<u8>],
) -> Result<InMemoryElasticPlanCache, FlatElasticError> {
    let mut cache = InMemoryElasticPlanCache::default();
    let mut seen_keys = BTreeSet::new();

    for encoded in encoded_records
    {
        let record =
            ElasticPersistedPlan::decode(encoded).map_err(FlatElasticError::Persistence)?;
        validate_persisted_bootstrap_record(
            &record,
            config,
            hardware,
            required_invalidation_dependencies,
        )?;
        let key = ElasticPlanKey::new(
            record.plan.hardware.clone(),
            record.plan.problem.clone(),
            record.plan.objective,
        );
        if !seen_keys.insert(key.clone())
        {
            return Err(FlatElasticError::DuplicatePersistedPlanKey);
        }
        cache.store(key, record.plan);
    }
    Ok(cache)
}

fn validate_persisted_bootstrap_record(
    record: &ElasticPersistedPlan,
    config: ElasticConfig,
    hardware: &ElasticHardwareProfile,
    required_invalidation_dependencies: &[Vec<u8>],
) -> Result<(), FlatElasticError> {
    if !record.selected
    {
        return Err(FlatElasticError::PersistedRecordNotSelected);
    }
    if record.recorded_unix_ns == 0
    {
        return Err(FlatElasticError::PersistedRecordMissingTimestamp);
    }
    if record.measurement_protocol.residence_mode != ElasticResidenceMode::Resident
    {
        return Err(FlatElasticError::PersistedRecordNotResident);
    }
    if &record.plan.hardware != hardware
    {
        return Err(FlatElasticError::PersistedHardwareMismatch);
    }
    if record.plan.objective != config.objective
    {
        return Err(FlatElasticError::PersistedObjectiveMismatch);
    }
    if config.objective == ElasticObjective::DeterministicOnly
        && !record.plan.evidence.candidate.deterministic
    {
        return Err(FlatElasticError::Selection(
            ElasticSelectionError::NonDeterministicCandidate,
        ));
    }
    if record.plan.problem.family() != FLAT_ELASTIC_FAMILY
    {
        return Err(FlatElasticError::PersistedProblemFamilyMismatch);
    }
    require_known_candidate_identity(&record.plan.evidence.candidate)?;
    require_invalidation_dependency(record, FLAT_KERNEL_REVISION)?;
    for dependency in required_invalidation_dependencies
    {
        require_invalidation_dependency(record, dependency)?;
    }
    Ok(())
}

fn require_known_candidate_identity(candidate: &ElasticCandidate) -> Result<(), FlatElasticError> {
    if *candidate == current_candidate()? || *candidate == m15_candidate()?
    {
        Ok(())
    }
    else
    {
        Err(FlatElasticError::UnknownCandidate)
    }
}

fn require_invalidation_dependency(
    record: &ElasticPersistedPlan,
    required: &[u8],
) -> Result<(), FlatElasticError> {
    if !required.is_empty()
        && record
            .invalidation_dependencies
            .iter()
            .any(|dependency| dependency.as_slice() == required)
    {
        Ok(())
    }
    else
    {
        Err(FlatElasticError::MissingInvalidationDependency(
            required.to_vec(),
        ))
    }
}

#[derive(Debug)]
pub enum FlatElasticError {
    InvalidRequest(&'static str),
    RequestOverflow(&'static str),
    HardwareProfile(scirust_compute::ProfileEncodingError),
    Backend(BackendError),
    CandidateEncoding(ElasticCandidateError),
    Evidence(ElasticEvidenceError),
    Selection(ElasticSelectionError),
    Mode(ElasticModeError),
    Persistence(ElasticPersistenceError),
    PersistedRecordNotSelected,
    PersistedRecordMissingTimestamp,
    PersistedRecordNotResident,
    PersistedHardwareMismatch,
    PersistedObjectiveMismatch,
    PersistedProblemFamilyMismatch,
    MissingInvalidationDependency(Vec<u8>),
    DuplicatePersistedPlanKey,
    NoQualifiedCandidate,
    UnknownCandidate,
    PlanRegionMismatch,
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
            Self::Backend(error) => write!(f, "FLAT backend execution failed: {error}"),
            Self::CandidateEncoding(error) => write!(f, "cannot encode FLAT candidate: {error}"),
            Self::Evidence(error) => write!(f, "invalid FLAT tuning evidence: {error}"),
            Self::Selection(error) => write!(f, "cannot select measured FLAT evidence: {error}"),
            Self::Mode(error) => write!(f, "Elastic mode rejected FLAT plan resolution: {error}"),
            Self::Persistence(error) => write!(f, "invalid persisted Elastic record: {error}"),
            Self::PersistedRecordNotSelected => write!(
                f,
                "persisted Elastic record is not a selected production plan"
            ),
            Self::PersistedRecordMissingTimestamp => write!(
                f,
                "persisted Elastic record has no caller-supplied timestamp"
            ),
            Self::PersistedRecordNotResident => write!(
                f,
                "persisted FLAT evidence must use resident measurement semantics"
            ),
            Self::PersistedHardwareMismatch =>
            {
                write!(f, "persisted FLAT plan targets different hardware")
            },
            Self::PersistedObjectiveMismatch =>
            {
                write!(f, "persisted FLAT plan targets a different objective")
            },
            Self::PersistedProblemFamilyMismatch =>
            {
                write!(f, "persisted plan is not a FLAT attention problem")
            },
            Self::MissingInvalidationDependency(dependency) => write!(
                f,
                "persisted FLAT plan is missing invalidation dependency `{}`",
                String::from_utf8_lossy(dependency)
            ),
            Self::DuplicatePersistedPlanKey => write!(
                f,
                "multiple persisted records target the same FLAT validity region"
            ),
            Self::NoQualifiedCandidate => write!(f, "no qualified FLAT candidate for this request"),
            Self::UnknownCandidate =>
            {
                write!(f, "Elastic plan names a stale or foreign FLAT candidate")
            },
            Self::PlanRegionMismatch =>
            {
                write!(
                    f,
                    "FLAT execution request is outside the selected plan validity region"
                )
            },
            Self::MissingProductionPlan =>
            {
                write!(f, "selected Elastic mode requires a persisted FLAT plan")
            },
        }
    }
}

impl std::error::Error for FlatElasticError {}

fn hardware_profile_from_context(
    context: &WgpuContext,
) -> Result<ElasticHardwareProfile, FlatElasticError> {
    let adapter = WgpuComputeAdapter::from_context(context.clone());
    let capabilities = ComputeBackend::hardware_capabilities(&adapter);
    ElasticHardwareProfile::from_capabilities(&capabilities)
        .map_err(FlatElasticError::HardwareProfile)
}

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

fn m15_candidate() -> Result<ElasticCandidate, FlatElasticError> {
    ElasticCandidate::new(
        FLAT_M15_KERNEL_FAMILY,
        FLAT_KERNEL_REVISION.to_vec(),
        [
            ElasticParameter {
                name: "query_rows".into(),
                value: 1,
            },
            ElasticParameter {
                name: "resident_kv".into(),
                value: 1,
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

fn m15_candidate_eligible(request: FlatElasticRequest) -> bool {
    request.problem_kind() == FlatProblemKind::Decode
        && request.config.causal
        && request.config.query_len == 1
        && request.kv_representation == FlatKvRepresentation::PreRotated
        && request
            .config
            .query_position_offset
            .checked_add(1)
            .is_some_and(|visible| visible >= request.config.kv_len)
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
    use elastic_autotuner::measurement_protocol::{
        ElasticMeasurementProtocol, ElasticSynchronizationBoundary, ElasticTimingSource,
    };
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

    fn encoded_persisted_plan(
        planner: &FlatElasticPlanner,
        hardware: ElasticHardwareProfile,
        candidate: ElasticCandidate,
        selected: bool,
        recorded_unix_ns: u64,
        residence_mode: ElasticResidenceMode,
        dependencies: Vec<Vec<u8>>,
    ) -> Vec<u8> {
        let locked = tuner(ElasticMode::Locked, ElasticObjective::MinLatency);
        let plan = measured_plan(planner, &locked, hardware, candidate);
        ElasticPersistedPlan::new(
            plan,
            ElasticMeasurementProtocol::new(
                1,
                3,
                ElasticTimingSource::HostWallClock,
                residence_mode,
                ElasticSynchronizationBoundary::PerIteration,
            ),
            selected,
            recorded_unix_ns,
            b"flat-bootstrap-test".to_vec(),
            dependencies,
        )
        .unwrap()
        .encode()
        .unwrap()
    }

    #[test]
    fn persisted_bootstrap_loads_locked_plan_before_request_resolution() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let required = b"scirust@test-source".to_vec();
        let encoded = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            m15_candidate().unwrap(),
            true,
            1,
            ElasticResidenceMode::Resident,
            vec![FLAT_KERNEL_REVISION.to_vec(), required.clone()],
        );
        let runtime_config = ElasticConfig {
            mode: ElasticMode::Locked,
            objective: ElasticObjective::MinLatency,
            max_ranked_candidates: 0,
        };
        let cache =
            persisted_cache_from_records(runtime_config, &hardware, &[encoded], &[required])
                .unwrap();
        let mut runtime = FlatElasticRuntime::from_hardware(runtime_config, hardware, cache);
        let request =
            FlatElasticRequest::new(config(1, 17), FlatKvRepresentation::PreRotated).unwrap();
        let resolved = runtime.resolve_request(request).unwrap();
        assert_eq!(resolved.origin(), FlatPlanOrigin::Persisted);
        assert_eq!(resolved.candidate(), &m15_candidate().unwrap());
    }

    #[test]
    fn persisted_bootstrap_rejects_unselected_missing_timestamp_and_transfer_evidence() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let config = ElasticConfig {
            mode: ElasticMode::Locked,
            objective: ElasticObjective::MinLatency,
            max_ranked_candidates: 0,
        };
        let deps = vec![FLAT_KERNEL_REVISION.to_vec()];

        let unselected = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            current_candidate().unwrap(),
            false,
            1,
            ElasticResidenceMode::Resident,
            deps.clone(),
        );
        assert!(matches!(
            persisted_cache_from_records(config, &hardware, &[unselected], &[]),
            Err(FlatElasticError::PersistedRecordNotSelected)
        ));

        let no_timestamp = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            current_candidate().unwrap(),
            true,
            0,
            ElasticResidenceMode::Resident,
            deps.clone(),
        );
        assert!(matches!(
            persisted_cache_from_records(config, &hardware, &[no_timestamp], &[]),
            Err(FlatElasticError::PersistedRecordMissingTimestamp)
        ));

        let transfer = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            current_candidate().unwrap(),
            true,
            1,
            ElasticResidenceMode::TransferInclusive,
            deps,
        );
        assert!(matches!(
            persisted_cache_from_records(config, &hardware, &[transfer], &[]),
            Err(FlatElasticError::PersistedRecordNotResident)
        ));
    }

    #[test]
    fn persisted_bootstrap_rejects_nondeterministic_deterministic_only_plan() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let locked = tuner(ElasticMode::Locked, ElasticObjective::MinLatency);
        let mut plan = measured_plan(
            &planner,
            &locked,
            hardware.clone(),
            current_candidate().unwrap(),
        );
        assert!(!plan.evidence.candidate.deterministic);
        plan.objective = ElasticObjective::DeterministicOnly;
        let encoded = ElasticPersistedPlan::new(
            plan,
            ElasticMeasurementProtocol::new(
                1,
                3,
                ElasticTimingSource::HostWallClock,
                ElasticResidenceMode::Resident,
                ElasticSynchronizationBoundary::PerIteration,
            ),
            true,
            1,
            b"forged-deterministic-only".to_vec(),
            vec![FLAT_KERNEL_REVISION.to_vec()],
        )
        .unwrap()
        .encode()
        .unwrap();
        let deterministic = ElasticConfig {
            mode: ElasticMode::Locked,
            objective: ElasticObjective::DeterministicOnly,
            max_ranked_candidates: 0,
        };
        assert!(matches!(
            persisted_cache_from_records(deterministic, &hardware, &[encoded], &[]),
            Err(FlatElasticError::Selection(
                ElasticSelectionError::NonDeterministicCandidate
            ))
        ));
    }

    #[test]
    fn persisted_bootstrap_rejects_missing_dependencies_stale_candidates_and_duplicates() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let config = ElasticConfig {
            mode: ElasticMode::Locked,
            objective: ElasticObjective::MinLatency,
            max_ranked_candidates: 0,
        };
        let required = b"scirust@test-source".to_vec();
        let current = current_candidate().unwrap();
        let encoded = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            current.clone(),
            true,
            1,
            ElasticResidenceMode::Resident,
            vec![FLAT_KERNEL_REVISION.to_vec()],
        );
        assert!(matches!(
            persisted_cache_from_records(config, &hardware, &[encoded], &[required]),
            Err(FlatElasticError::MissingInvalidationDependency(_))
        ));

        let stale = ElasticCandidate::new(
            FLAT_KERNEL_FAMILY,
            b"flat-attention@stale".to_vec(),
            current.parameters().iter().cloned(),
            false,
            0,
        )
        .unwrap();
        let stale = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            stale,
            true,
            1,
            ElasticResidenceMode::Resident,
            vec![FLAT_KERNEL_REVISION.to_vec()],
        );
        assert!(matches!(
            persisted_cache_from_records(config, &hardware, &[stale], &[]),
            Err(FlatElasticError::UnknownCandidate)
        ));

        let valid = encoded_persisted_plan(
            &planner,
            hardware.clone(),
            current_candidate().unwrap(),
            true,
            1,
            ElasticResidenceMode::Resident,
            vec![FLAT_KERNEL_REVISION.to_vec()],
        );
        assert!(matches!(
            persisted_cache_from_records(config, &hardware, &[valid.clone(), valid], &[]),
            Err(FlatElasticError::DuplicatePersistedPlanKey)
        ));
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
    fn selected_plan_reuses_dynamic_requests_inside_one_region() {
        let planner = planner(1, 17);
        let cold = tuner(ElasticMode::Cold, ElasticObjective::MinLatency);
        let cache = InMemoryElasticPlanCache::default();
        let plan = match planner.resolve_for_mode(&cold, hardware(), &cache).unwrap()
        {
            FlatPlanResolution::Production(plan) => plan,
            FlatPlanResolution::AuditOnly => panic!("cold must resolve a production plan"),
        };
        let same_region =
            FlatElasticRequest::new(config(1, 64), FlatKvRepresentation::PreRotated).unwrap();
        let next_region =
            FlatElasticRequest::new(config(1, 65), FlatKvRepresentation::PreRotated).unwrap();
        assert!(plan.accepts(same_region));
        assert!(!plan.accepts(next_region));
    }

    #[test]
    fn runtime_reuses_region_and_transitions_only_at_bucket_boundary() {
        let mut runtime = FlatElasticRuntime::from_hardware(
            ElasticConfig {
                mode: ElasticMode::Cold,
                objective: ElasticObjective::MinLatency,
                max_ranked_candidates: 0,
            },
            hardware(),
            InMemoryElasticPlanCache::default(),
        );
        let first =
            FlatElasticRequest::new(config(1, 17), FlatKvRepresentation::PreRotated).unwrap();
        let same =
            FlatElasticRequest::new(config(1, 64), FlatKvRepresentation::PreRotated).unwrap();
        let next =
            FlatElasticRequest::new(config(1, 65), FlatKvRepresentation::PreRotated).unwrap();

        let first_class = runtime
            .resolve_request(first)
            .unwrap()
            .problem_class()
            .clone();
        let same_class = runtime
            .resolve_request(same)
            .unwrap()
            .problem_class()
            .clone();
        assert_eq!(first_class, same_class);
        let next_class = runtime
            .resolve_request(next)
            .unwrap()
            .problem_class()
            .clone();
        assert_ne!(same_class, next_class);
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
    fn decode_pre_rotated_exposes_m11_and_m15_while_cold_keeps_m11() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let cold = tuner(ElasticMode::Cold, ElasticObjective::MinLatency);
        let ranked = planner.rank_candidates(&cold, &hardware);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].candidate, current_candidate().unwrap());
        assert!(
            ranked
                .iter()
                .any(|ranked| ranked.candidate == m15_candidate().unwrap())
        );

        let cache = InMemoryElasticPlanCache::default();
        let plan = match planner.resolve_for_mode(&cold, hardware, &cache).unwrap()
        {
            FlatPlanResolution::Production(plan) => plan,
            FlatPlanResolution::AuditOnly => panic!("cold must resolve a production plan"),
        };
        assert_eq!(plan.candidate(), &current_candidate().unwrap());
    }

    #[test]
    fn m15_is_not_exposed_outside_fully_visible_pre_rotated_causal_decode() {
        let cold = tuner(ElasticMode::Cold, ElasticObjective::MinLatency);
        let hardware = hardware();

        let raw = FlatElasticPlanner::new(
            FlatElasticRequest::new(config(1, 17), FlatKvRepresentation::Raw).unwrap(),
        )
        .unwrap();
        assert_eq!(raw.rank_candidates(&cold, &hardware).len(), 1);

        let prefill = planner(4, 17);
        assert_eq!(prefill.rank_candidates(&cold, &hardware).len(), 1);

        let mut partially_visible = config(1, 17);
        partially_visible.query_position_offset = 0;
        let partially_visible = FlatElasticPlanner::new(
            FlatElasticRequest::new(partially_visible, FlatKvRepresentation::PreRotated).unwrap(),
        )
        .unwrap();
        assert_eq!(partially_visible.rank_candidates(&cold, &hardware).len(), 1);
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
    fn learn_can_promote_measured_m15_over_the_safe_m11_fallback() {
        let planner = planner(1, 17);
        let hardware = hardware();
        let learn = tuner(ElasticMode::Learn, ElasticObjective::MinLatency);
        let records = [
            evidence(current_candidate().unwrap(), 20),
            evidence(m15_candidate().unwrap(), 10),
        ];
        let mut cache = InMemoryElasticPlanCache::default();
        let promoted = planner
            .promote_measured_evidence(&learn, hardware.clone(), &records, &mut cache)
            .unwrap();
        assert_eq!(promoted.evidence.candidate, m15_candidate().unwrap());

        let resolved = planner.resolve_for_mode(&learn, hardware, &cache).unwrap();
        match resolved
        {
            FlatPlanResolution::Production(plan) =>
            {
                assert_eq!(plan.origin(), FlatPlanOrigin::Persisted);
                assert_eq!(plan.candidate(), &m15_candidate().unwrap());
            },
            FlatPlanResolution::AuditOnly => panic!("Learn must resolve the promoted M15 plan"),
        }
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
