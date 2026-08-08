extern crate alloc;

use alloc::vec::Vec;

use crate::{
    CapabilitySet, DType, HardwareCapabilities, IsaCapabilities, IsaFeature, MemorySpace,
    ReproducibilityLevel, SupportLevel, VectorModel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SupportRequirement {
    #[default]
    Any,
    Required,
    Forbidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum VectorRequirement {
    #[default]
    Any,
    Scalar,
    Vectorized {
        min_bits: Option<u32>,
    },
    FixedWidth {
        min_bits: Option<u32>,
    },
    Scalable,
}

/// Semantic requirements for one kernel implementation.
///
/// Portable implementations should express semantic needs here. `required_isa`
/// is reserved for genuinely ISA-specific implementations rather than as a
/// substitute for numeric, vector, matrix, memory or reproducibility contracts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelRequirements {
    pub storage_dtypes: Vec<DType>,
    pub arithmetic_dtypes: Vec<DType>,
    pub accumulation_dtypes: Vec<DType>,
    pub required_isa: Vec<IsaFeature>,
    pub vector: VectorRequirement,
    pub matrix_acceleration: SupportRequirement,
    pub memory_spaces: Vec<MemorySpace>,
    pub async_execution: SupportRequirement,
    pub subgroup_operations: SupportRequirement,
    pub atomic_i64: SupportRequirement,
    pub reproducibility: Option<ReproducibilityLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatchDisposition {
    Compatible,
    Indeterminate,
    Incompatible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RequirementIssue {
    StorageDTypeUnknown(DType),
    StorageDTypeUnsupported(DType),
    ArithmeticDTypeUnknown(DType),
    ArithmeticDTypeUnsupported(DType),
    AccumulationDTypeUnknown(DType),
    AccumulationDTypeUnsupported(DType),
    IsaUnknown(IsaFeature),
    IsaUnsupported(IsaFeature),
    VectorModelUnknown,
    VectorModelIncompatible {
        required: VectorRequirement,
        observed: VectorModel,
    },
    VectorWidthUnknown {
        min_bits: u32,
    },
    VectorWidthInsufficient {
        min_bits: u32,
        observed_max_bits: u32,
    },
    MatrixAccelerationUnknown,
    MatrixAccelerationMismatch {
        required: SupportRequirement,
        observed: SupportLevel,
    },
    MemorySpaceUnknown(MemorySpace),
    MemorySpaceUnsupported(MemorySpace),
    AsyncExecutionUnknown,
    AsyncExecutionMismatch {
        required: SupportRequirement,
        observed: SupportLevel,
    },
    SubgroupOperationsUnknown,
    SubgroupOperationsMismatch {
        required: SupportRequirement,
        observed: SupportLevel,
    },
    AtomicI64Unknown,
    AtomicI64Mismatch {
        required: SupportRequirement,
        observed: SupportLevel,
    },
    ReproducibilityUnknown(ReproducibilityLevel),
    ReproducibilityUnsupported(ReproducibilityLevel),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchReport {
    pub disposition: MatchDisposition,
    pub issues: Vec<RequirementIssue>,
}

impl MatchReport {
    pub fn is_compatible(&self) -> bool {
        self.disposition == MatchDisposition::Compatible
    }

    pub fn is_incompatible(&self) -> bool {
        self.disposition == MatchDisposition::Incompatible
    }
}

#[derive(Debug, Default)]
struct MatchState {
    issues: Vec<RequirementIssue>,
    has_unknown: bool,
    has_incompatible: bool,
}

impl MatchState {
    fn unknown(&mut self, issue: RequirementIssue) {
        self.has_unknown = true;
        self.issues.push(issue);
    }

    fn incompatible(&mut self, issue: RequirementIssue) {
        self.has_incompatible = true;
        self.issues.push(issue);
    }

    fn finish(self) -> MatchReport {
        let disposition = if self.has_incompatible
        {
            MatchDisposition::Incompatible
        }
        else if self.has_unknown
        {
            MatchDisposition::Indeterminate
        }
        else
        {
            MatchDisposition::Compatible
        };

        MatchReport {
            disposition,
            issues: self.issues,
        }
    }
}

/// Match semantic kernel requirements against explicit hardware facts.
///
/// Explicit negative facts yield `Incompatible`. Missing knowledge yields
/// `Indeterminate`; it is never silently converted to `Unsupported`.
pub fn match_requirements(
    requirements: &KernelRequirements,
    capabilities: &HardwareCapabilities,
) -> MatchReport {
    let mut state = MatchState::default();

    match_value_set(
        &requirements.storage_dtypes,
        &capabilities.numeric.storage_dtypes,
        RequirementIssue::StorageDTypeUnknown,
        RequirementIssue::StorageDTypeUnsupported,
        &mut state,
    );
    match_value_set(
        &requirements.arithmetic_dtypes,
        &capabilities.numeric.arithmetic_dtypes,
        RequirementIssue::ArithmeticDTypeUnknown,
        RequirementIssue::ArithmeticDTypeUnsupported,
        &mut state,
    );
    match_value_set(
        &requirements.accumulation_dtypes,
        &capabilities.numeric.accumulation_dtypes,
        RequirementIssue::AccumulationDTypeUnknown,
        RequirementIssue::AccumulationDTypeUnsupported,
        &mut state,
    );
    match_value_set(
        &requirements.required_isa,
        &capabilities.isa.features,
        RequirementIssue::IsaUnknown,
        RequirementIssue::IsaUnsupported,
        &mut state,
    );
    match_vector(requirements.vector, &capabilities.isa, &mut state);

    match_support(
        requirements.matrix_acceleration,
        capabilities.matrix.accelerated,
        RequirementIssue::MatrixAccelerationUnknown,
        |required, observed| RequirementIssue::MatrixAccelerationMismatch { required, observed },
        &mut state,
    );

    match_value_set(
        &requirements.memory_spaces,
        &capabilities.memory.spaces,
        RequirementIssue::MemorySpaceUnknown,
        RequirementIssue::MemorySpaceUnsupported,
        &mut state,
    );

    match_support(
        requirements.async_execution,
        capabilities.execution.async_execution,
        RequirementIssue::AsyncExecutionUnknown,
        |required, observed| RequirementIssue::AsyncExecutionMismatch { required, observed },
        &mut state,
    );
    match_support(
        requirements.subgroup_operations,
        capabilities.execution.subgroup_operations,
        RequirementIssue::SubgroupOperationsUnknown,
        |required, observed| RequirementIssue::SubgroupOperationsMismatch { required, observed },
        &mut state,
    );
    match_support(
        requirements.atomic_i64,
        capabilities.execution.atomic_i64,
        RequirementIssue::AtomicI64Unknown,
        |required, observed| RequirementIssue::AtomicI64Mismatch { required, observed },
        &mut state,
    );

    if let Some(level) = requirements.reproducibility
    {
        match capabilities.reproducibility.support_level(level)
        {
            SupportLevel::Supported =>
            {},
            SupportLevel::Unsupported =>
            {
                state.incompatible(RequirementIssue::ReproducibilityUnsupported(level));
            },
            SupportLevel::Unknown =>
            {
                state.unknown(RequirementIssue::ReproducibilityUnknown(level));
            },
        }
    }

    state.finish()
}

fn match_value_set<T, U, N>(
    required: &[T],
    available: &CapabilitySet<T>,
    unknown_issue: U,
    unsupported_issue: N,
    state: &mut MatchState,
) where
    T: Clone + PartialEq,
    U: Fn(T) -> RequirementIssue,
    N: Fn(T) -> RequirementIssue,
{
    for value in required
    {
        match available.support_level(value)
        {
            SupportLevel::Supported =>
            {},
            SupportLevel::Unsupported =>
            {
                state.incompatible(unsupported_issue(value.clone()));
            },
            SupportLevel::Unknown =>
            {
                state.unknown(unknown_issue(value.clone()));
            },
        }
    }
}

fn match_vector(requirement: VectorRequirement, isa: &IsaCapabilities, state: &mut MatchState) {
    if requirement == VectorRequirement::Any
    {
        return;
    }

    let observed = isa.vector_model;
    if observed == VectorModel::Unknown
    {
        state.unknown(RequirementIssue::VectorModelUnknown);
        return;
    }

    let model_matches = match requirement
    {
        VectorRequirement::Any => true,
        VectorRequirement::Scalar => observed == VectorModel::Scalar,
        VectorRequirement::Vectorized { .. } =>
        {
            matches!(observed, VectorModel::FixedWidth | VectorModel::Scalable)
        },
        VectorRequirement::FixedWidth { .. } => observed == VectorModel::FixedWidth,
        VectorRequirement::Scalable => observed == VectorModel::Scalable,
    };

    if !model_matches
    {
        state.incompatible(RequirementIssue::VectorModelIncompatible {
            required: requirement,
            observed,
        });
        return;
    }

    let min_bits =
        match requirement
        {
            VectorRequirement::Vectorized { min_bits }
            | VectorRequirement::FixedWidth { min_bits } => min_bits,
            _ => None,
        };

    if let Some(min_bits) = min_bits
    {
        match isa.max_vector_bits
        {
            Some(observed_max_bits) if observed_max_bits < min_bits =>
            {
                state.incompatible(RequirementIssue::VectorWidthInsufficient {
                    min_bits,
                    observed_max_bits,
                });
            },
            Some(_) =>
            {},
            None => state.unknown(RequirementIssue::VectorWidthUnknown { min_bits }),
        }
    }
}

fn match_support<F>(
    requirement: SupportRequirement,
    observed: SupportLevel,
    unknown_issue: RequirementIssue,
    mismatch_issue: F,
    state: &mut MatchState,
) where
    F: FnOnce(SupportRequirement, SupportLevel) -> RequirementIssue,
{
    if requirement == SupportRequirement::Any
    {
        return;
    }

    match observed
    {
        SupportLevel::Unknown => state.unknown(unknown_issue),
        SupportLevel::Supported if requirement == SupportRequirement::Forbidden =>
        {
            state.incompatible(mismatch_issue(requirement, observed));
        },
        SupportLevel::Unsupported if requirement == SupportRequirement::Required =>
        {
            state.incompatible(mismatch_issue(requirement, observed));
        },
        SupportLevel::Supported | SupportLevel::Unsupported =>
        {},
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutionCandidate<'a> {
    pub name: &'a str,
    /// Static policy priority. Lower values are preferred.
    pub priority: u32,
    pub capabilities: &'a HardwareCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PlannerPolicy {
    pub allow_indeterminate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSelection<'a> {
    pub name: &'a str,
    pub priority: u32,
    pub report: MatchReport,
}

/// Deterministically select a candidate without timing-dependent autotuning.
pub fn select_candidate<'a>(
    requirements: &KernelRequirements,
    candidates: &[ExecutionCandidate<'a>],
    policy: PlannerPolicy,
) -> Option<CandidateSelection<'a>> {
    let mut best: Option<CandidateSelection<'a>> = None;

    for candidate in candidates
    {
        let report = match_requirements(requirements, candidate.capabilities);
        let eligible = match report.disposition
        {
            MatchDisposition::Compatible => true,
            MatchDisposition::Indeterminate => policy.allow_indeterminate,
            MatchDisposition::Incompatible => false,
        };

        if !eligible
        {
            continue;
        }

        let selection = CandidateSelection {
            name: candidate.name,
            priority: candidate.priority,
            report,
        };

        if best
            .as_ref()
            .map(|current| candidate_is_better(&selection, current))
            .unwrap_or(true)
        {
            best = Some(selection);
        }
    }

    best
}

fn candidate_is_better(
    candidate: &CandidateSelection<'_>,
    current: &CandidateSelection<'_>,
) -> bool {
    let candidate_rank = disposition_rank(candidate.report.disposition);
    let current_rank = disposition_rank(current.report.disposition);

    candidate_rank < current_rank
        || (candidate_rank == current_rank
            && (candidate.priority < current.priority
                || (candidate.priority == current.priority && candidate.name < current.name)))
}

const fn disposition_rank(disposition: MatchDisposition) -> u8 {
    match disposition
    {
        MatchDisposition::Compatible => 0,
        MatchDisposition::Indeterminate => 1,
        MatchDisposition::Incompatible => 2,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        Architecture, DeviceCapabilities, DeviceId, DeviceKind, ExecutionCapabilities,
        MatrixCapabilities, MemoryCapabilities, NumericCapabilities, ReproducibilityCapabilities,
    };

    fn known_set<T: PartialEq>(supported: Vec<T>, unsupported: Vec<T>) -> CapabilitySet<T> {
        let mut set = CapabilitySet::default();
        for value in supported
        {
            set.set_support(value, SupportLevel::Supported);
        }
        for value in unsupported
        {
            set.set_support(value, SupportLevel::Unsupported);
        }
        set
    }

    fn hardware() -> HardwareCapabilities {
        HardwareCapabilities {
            device: DeviceId::new(DeviceKind::Cpu, 0),
            architecture: Architecture::current_host(),
            isa: IsaCapabilities {
                features: known_set(vec![IsaFeature::Avx2], vec![IsaFeature::Avx512F]),
                vector_model: VectorModel::FixedWidth,
                min_vector_bits: Some(128),
                max_vector_bits: Some(256),
            },
            numeric: NumericCapabilities {
                storage_dtypes: known_set(vec![DType::F32], vec![DType::F64]),
                arithmetic_dtypes: known_set(vec![DType::F32], vec![DType::F64]),
                accumulation_dtypes: known_set(vec![DType::F32], vec![DType::F64]),
            },
            matrix: MatrixCapabilities {
                accelerated: SupportLevel::Unsupported,
                ..MatrixCapabilities::default()
            },
            memory: MemoryCapabilities {
                spaces: known_set(vec![MemorySpace::Host], vec![MemorySpace::Device]),
                ..MemoryCapabilities::default()
            },
            execution: ExecutionCapabilities {
                async_execution: SupportLevel::Unsupported,
                subgroup_operations: SupportLevel::Unknown,
                atomic_i64: SupportLevel::Supported,
                ..ExecutionCapabilities::default()
            },
            reproducibility: ReproducibilityCapabilities {
                modes: known_set(
                    vec![ReproducibilityLevel::Deterministic],
                    vec![ReproducibilityLevel::FastApproximate],
                ),
            },
        }
    }

    #[test]
    fn empty_requirements_are_compatible() {
        let report = match_requirements(&KernelRequirements::default(), &hardware());
        assert_eq!(report.disposition, MatchDisposition::Compatible);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn explicit_negative_facts_are_incompatible() {
        let requirements = KernelRequirements {
            storage_dtypes: vec![DType::F64],
            required_isa: vec![IsaFeature::Avx512F],
            memory_spaces: vec![MemorySpace::Device],
            ..KernelRequirements::default()
        };
        let report = match_requirements(&requirements, &hardware());
        assert_eq!(report.disposition, MatchDisposition::Incompatible);
        assert!(
            report
                .issues
                .contains(&RequirementIssue::StorageDTypeUnsupported(DType::F64))
        );
        assert!(
            report
                .issues
                .contains(&RequirementIssue::IsaUnsupported(IsaFeature::Avx512F))
        );
        assert!(report
            .issues
            .contains(&RequirementIssue::MemorySpaceUnsupported(
                MemorySpace::Device
            )));
    }

    #[test]
    fn unprobed_fact_stays_indeterminate() {
        let requirements = KernelRequirements {
            arithmetic_dtypes: vec![DType::Bf16],
            ..KernelRequirements::default()
        };
        let report = match_requirements(&requirements, &hardware());
        assert_eq!(report.disposition, MatchDisposition::Indeterminate);
        assert_eq!(
            report.issues,
            vec![RequirementIssue::ArithmeticDTypeUnknown(DType::Bf16)]
        );
    }

    #[test]
    fn vector_width_is_checked_without_performance_assumptions() {
        let compatible = KernelRequirements {
            vector: VectorRequirement::FixedWidth {
                min_bits: Some(256),
            },
            ..KernelRequirements::default()
        };
        let incompatible = KernelRequirements {
            vector: VectorRequirement::FixedWidth {
                min_bits: Some(512),
            },
            ..KernelRequirements::default()
        };
        assert_eq!(
            match_requirements(&compatible, &hardware()).disposition,
            MatchDisposition::Compatible
        );
        assert_eq!(
            match_requirements(&incompatible, &hardware()).disposition,
            MatchDisposition::Incompatible
        );
    }

    #[test]
    fn planner_prefers_proven_compatibility_over_lower_priority_unknown() {
        let proven = hardware();
        let mut unknown = hardware();
        unknown.isa.vector_model = VectorModel::Unknown;
        let requirements = KernelRequirements {
            vector: VectorRequirement::Vectorized { min_bits: None },
            ..KernelRequirements::default()
        };
        let candidates = [
            ExecutionCandidate {
                name: "unknown-fast",
                priority: 0,
                capabilities: &unknown,
            },
            ExecutionCandidate {
                name: "proven",
                priority: 10,
                capabilities: &proven,
            },
        ];
        let selected = select_candidate(
            &requirements,
            &candidates,
            PlannerPolicy {
                allow_indeterminate: true,
            },
        )
        .unwrap();
        assert_eq!(selected.name, "proven");
    }

    #[test]
    fn planner_tie_breaking_is_stable() {
        let first = hardware();
        let second = hardware();
        let candidates = [
            ExecutionCandidate {
                name: "zeta",
                priority: 4,
                capabilities: &first,
            },
            ExecutionCandidate {
                name: "alpha",
                priority: 4,
                capabilities: &second,
            },
        ];
        let selected = select_candidate(
            &KernelRequirements::default(),
            &candidates,
            PlannerPolicy::default(),
        )
        .unwrap();
        assert_eq!(selected.name, "alpha");
    }

    #[test]
    fn default_policy_rejects_indeterminate_candidates() {
        let capabilities = DeviceCapabilities::reference_cpu().hardware_baseline();
        let requirements = KernelRequirements {
            arithmetic_dtypes: vec![DType::F32],
            ..KernelRequirements::default()
        };
        let candidates = [ExecutionCandidate {
            name: "reference",
            priority: 0,
            capabilities: &capabilities,
        }];
        assert!(select_candidate(&requirements, &candidates, PlannerPolicy::default()).is_none());
    }
}
