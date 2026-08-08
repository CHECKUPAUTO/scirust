extern crate alloc;

use alloc::vec::Vec;

use crate::{
    DeviceCapabilities, HardwareCapabilities, KernelRequirements, MatchDisposition, MatchReport,
    PlannerPolicy, match_requirements,
};

/// Portable launch limits that are known independently of an architecture or vendor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionLimits {
    /// Maximum workgroup size accepted in each launch dimension.
    ///
    /// `None` means the limit is unknown. A missing observation is never treated
    /// as a zero-sized or unsupported execution dimension.
    pub max_workgroup_size: [Option<u32>; 3],
}

impl ExecutionLimits {
    /// Preserve the concrete workgroup limits already exposed by the legacy
    /// backend capability contract.
    #[must_use]
    pub fn from_device_capabilities(capabilities: &DeviceCapabilities) -> Self {
        Self {
            max_workgroup_size: capabilities.max_workgroup_size.map(Some),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct WorkgroupRequirement {
    /// Minimum supported workgroup size in each dimension.
    ///
    /// `None` leaves a dimension unconstrained.
    pub min_size: [Option<u32>; 3],
}

impl WorkgroupRequirement {
    #[must_use]
    pub const fn x(minimum: u32) -> Self {
        Self {
            min_size: [Some(minimum), None, None],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WorkgroupDimension {
    X,
    Y,
    Z,
}

impl WorkgroupDimension {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImplementationIssue {
    WorkgroupLimitUnknown {
        dimension: WorkgroupDimension,
        required_minimum: u32,
    },
    WorkgroupLimitInsufficient {
        dimension: WorkgroupDimension,
        required_minimum: u32,
        observed_maximum: u32,
    },
}

/// Requirements for one implementation of a logical operation.
///
/// `kernel` describes semantic hardware requirements already understood by the
/// generic matcher. `workgroup` adds portable launch-width requirements needed
/// by implementations whose shader has a fixed or minimum workgroup geometry.
#[derive(Debug, Clone, Copy)]
pub struct ImplementationRequirements<'a> {
    pub kernel: &'a KernelRequirements,
    pub workgroup: WorkgroupRequirement,
}

impl<'a> ImplementationRequirements<'a> {
    #[must_use]
    pub const fn new(kernel: &'a KernelRequirements) -> Self {
        Self {
            kernel,
            workgroup: WorkgroupRequirement {
                min_size: [None, None, None],
            },
        }
    }

    #[must_use]
    pub const fn with_workgroup(mut self, workgroup: WorkgroupRequirement) -> Self {
        self.workgroup = workgroup;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImplementationCandidate<'a> {
    pub name: &'a str,
    /// Static policy priority. Lower values are preferred.
    pub priority: u32,
    pub requirements: ImplementationRequirements<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplementationMatchReport {
    pub disposition: MatchDisposition,
    pub kernel: MatchReport,
    pub issues: Vec<ImplementationIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplementationSelection<'a> {
    pub name: &'a str,
    pub priority: u32,
    pub report: ImplementationMatchReport,
}

/// Match one implementation against semantic hardware capabilities and portable
/// execution limits.
#[must_use]
pub fn match_implementation(
    requirements: ImplementationRequirements<'_>,
    hardware: &HardwareCapabilities,
    limits: &ExecutionLimits,
) -> ImplementationMatchReport {
    let kernel = match_requirements(requirements.kernel, hardware);
    let mut issues = Vec::new();
    let mut workgroup_unknown = false;
    let mut workgroup_incompatible = false;

    for (index, dimension) in WorkgroupDimension::ALL.into_iter().enumerate()
    {
        let Some(required_minimum) = requirements.workgroup.min_size[index]
        else
        {
            continue;
        };
        if required_minimum == 0
        {
            continue;
        }

        match limits.max_workgroup_size[index]
        {
            Some(observed_maximum) if observed_maximum < required_minimum =>
            {
                workgroup_incompatible = true;
                issues.push(ImplementationIssue::WorkgroupLimitInsufficient {
                    dimension,
                    required_minimum,
                    observed_maximum,
                });
            },
            Some(_) =>
            {},
            None =>
            {
                workgroup_unknown = true;
                issues.push(ImplementationIssue::WorkgroupLimitUnknown {
                    dimension,
                    required_minimum,
                });
            },
        }
    }

    let disposition = if kernel.disposition == MatchDisposition::Incompatible
        || workgroup_incompatible
    {
        MatchDisposition::Incompatible
    }
    else if kernel.disposition == MatchDisposition::Indeterminate || workgroup_unknown
    {
        MatchDisposition::Indeterminate
    }
    else
    {
        MatchDisposition::Compatible
    };

    ImplementationMatchReport {
        disposition,
        kernel,
        issues,
    }
}

/// Deterministically select among implementations of one operation on the same
/// device.
///
/// This complements [`crate::select_candidate`], which selects among hardware
/// candidates for one common requirement set. Here each implementation carries
/// its own requirements while all candidates are evaluated against the same
/// hardware and launch limits.
#[must_use]
pub fn select_implementation<'a>(
    hardware: &HardwareCapabilities,
    limits: &ExecutionLimits,
    candidates: &[ImplementationCandidate<'a>],
    policy: PlannerPolicy,
) -> Option<ImplementationSelection<'a>> {
    let mut best: Option<ImplementationSelection<'a>> = None;

    for candidate in candidates
    {
        let report = match_implementation(candidate.requirements, hardware, limits);
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

        let selection = ImplementationSelection {
            name: candidate.name,
            priority: candidate.priority,
            report,
        };
        if best
            .as_ref()
            .map(|current| implementation_is_better(&selection, current))
            .unwrap_or(true)
        {
            best = Some(selection);
        }
    }

    best
}

fn implementation_is_better(
    candidate: &ImplementationSelection<'_>,
    current: &ImplementationSelection<'_>,
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
    use alloc::{string::ToString, vec};

    use super::*;
    use crate::{DeviceId, DeviceKind, ExecutionCapabilities, SupportLevel, SupportRequirement};

    fn hardware(async_execution: SupportLevel) -> HardwareCapabilities {
        let mut profile =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        profile.execution = ExecutionCapabilities {
            async_execution,
            ..ExecutionCapabilities::default()
        };
        profile
    }

    #[test]
    fn legacy_workgroup_limits_are_preserved_as_known_facts() {
        let capabilities = DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Wgpu, 0),
            name: "synthetic-wgpu".to_string(),
            supported_dtypes: vec![],
            max_buffer_bytes: None,
            max_workgroup_size: [256, 8, 4],
            supports_async_execution: true,
        };
        let limits = ExecutionLimits::from_device_capabilities(&capabilities);
        assert_eq!(limits.max_workgroup_size, [Some(256), Some(8), Some(4)]);
    }

    #[test]
    fn workgroup_matching_preserves_sufficient_insufficient_and_unknown() {
        let kernel = KernelRequirements::default();
        let requirements =
            ImplementationRequirements::new(&kernel).with_workgroup(WorkgroupRequirement::x(64));
        let profile = hardware(SupportLevel::Unsupported);

        let sufficient = match_implementation(
            requirements,
            &profile,
            &ExecutionLimits {
                max_workgroup_size: [Some(64), Some(1), Some(1)],
            },
        );
        assert_eq!(sufficient.disposition, MatchDisposition::Compatible);
        assert!(sufficient.issues.is_empty());

        let insufficient = match_implementation(
            requirements,
            &profile,
            &ExecutionLimits {
                max_workgroup_size: [Some(32), Some(1), Some(1)],
            },
        );
        assert_eq!(insufficient.disposition, MatchDisposition::Incompatible);
        assert!(matches!(
            insufficient.issues.as_slice(),
            [ImplementationIssue::WorkgroupLimitInsufficient {
                dimension: WorkgroupDimension::X,
                required_minimum: 64,
                observed_maximum: 32,
            }]
        ));

        let unknown = match_implementation(requirements, &profile, &ExecutionLimits::default());
        assert_eq!(unknown.disposition, MatchDisposition::Indeterminate);
        assert!(matches!(
            unknown.issues.as_slice(),
            [ImplementationIssue::WorkgroupLimitUnknown {
                dimension: WorkgroupDimension::X,
                required_minimum: 64,
            }]
        ));
    }

    #[test]
    fn compatible_fallback_beats_indeterminate_preferred_implementation() {
        let parallel_kernel = KernelRequirements::default();
        let sequential_kernel = KernelRequirements::default();
        let candidates = [
            ImplementationCandidate {
                name: "parallel-64",
                priority: 0,
                requirements: ImplementationRequirements::new(&parallel_kernel)
                    .with_workgroup(WorkgroupRequirement::x(64)),
            },
            ImplementationCandidate {
                name: "sequential",
                priority: 1,
                requirements: ImplementationRequirements::new(&sequential_kernel),
            },
        ];
        let selected = select_implementation(
            &hardware(SupportLevel::Unsupported),
            &ExecutionLimits::default(),
            &candidates,
            PlannerPolicy {
                allow_indeterminate: true,
            },
        )
        .expect("the compatible sequential implementation must remain eligible");
        assert_eq!(selected.name, "sequential");
        assert_eq!(selected.report.disposition, MatchDisposition::Compatible);
    }

    #[test]
    fn known_workgroup_width_promotes_or_rejects_parallel_candidate() {
        let parallel_kernel = KernelRequirements::default();
        let sequential_kernel = KernelRequirements::default();
        let candidates = [
            ImplementationCandidate {
                name: "parallel-64",
                priority: 0,
                requirements: ImplementationRequirements::new(&parallel_kernel)
                    .with_workgroup(WorkgroupRequirement::x(64)),
            },
            ImplementationCandidate {
                name: "sequential",
                priority: 1,
                requirements: ImplementationRequirements::new(&sequential_kernel),
            },
        ];
        let profile = hardware(SupportLevel::Unsupported);

        let wide = select_implementation(
            &profile,
            &ExecutionLimits {
                max_workgroup_size: [Some(256), Some(1), Some(1)],
            },
            &candidates,
            PlannerPolicy::default(),
        )
        .unwrap();
        assert_eq!(wide.name, "parallel-64");

        let narrow = select_implementation(
            &profile,
            &ExecutionLimits {
                max_workgroup_size: [Some(32), Some(1), Some(1)],
            },
            &candidates,
            PlannerPolicy::default(),
        )
        .unwrap();
        assert_eq!(narrow.name, "sequential");
    }

    #[test]
    fn kernel_requirements_and_static_tie_breaks_are_preserved() {
        let async_kernel = KernelRequirements {
            async_execution: SupportRequirement::Required,
            ..KernelRequirements::default()
        };
        let generic_kernel = KernelRequirements::default();
        let candidates = [
            ImplementationCandidate {
                name: "z-async",
                priority: 0,
                requirements: ImplementationRequirements::new(&async_kernel),
            },
            ImplementationCandidate {
                name: "b-generic",
                priority: 1,
                requirements: ImplementationRequirements::new(&generic_kernel),
            },
        ];
        let selected = select_implementation(
            &hardware(SupportLevel::Unsupported),
            &ExecutionLimits::default(),
            &candidates,
            PlannerPolicy::default(),
        )
        .unwrap();
        assert_eq!(selected.name, "b-generic");

        let first_kernel = KernelRequirements::default();
        let second_kernel = KernelRequirements::default();
        let tie = [
            ImplementationCandidate {
                name: "zeta",
                priority: 0,
                requirements: ImplementationRequirements::new(&first_kernel),
            },
            ImplementationCandidate {
                name: "alpha",
                priority: 0,
                requirements: ImplementationRequirements::new(&second_kernel),
            },
        ];
        let selected = select_implementation(
            &hardware(SupportLevel::Unsupported),
            &ExecutionLimits::default(),
            &tie,
            PlannerPolicy::default(),
        )
        .unwrap();
        assert_eq!(selected.name, "alpha");
    }
}
