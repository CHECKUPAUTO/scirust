//! Host-only SciRust -> FLAT-ATTENTION -> ElasticXxx contextual planner.
//!
//! This module is deliberately **advisory only**. It does not own a WGPU
//! device, compile a pipeline, or execute the selected realization. SciRust's
//! current resident FLAT runtime is still pinned to the older WGPU-compatible
//! FLAT revision, while the planner below is pinned to the newer host-only FLAT
//! Kernel IR / ElasticXxx bridge. Keeping those rails separate prevents a
//! planner result from being executed through an ABI it was not qualified for.
//!
//! Ownership is explicit:
//! - SciRust owns the dense-attention semantic request and its freshness epoch;
//! - FLAT owns Kernel IR, bounded candidate generation, M24 dispatch filtering,
//!   and optional M26 evidence;
//! - ElasticXxx owns generic capability filtering, deterministic selection, and
//!   recommendation freshness validation.

#![forbid(unsafe_code)]

use core::fmt;

use elastic_core::LogicalResourceId;
pub use elastic_core::{
    FreshnessSnapshot, ObservationEpoch, PlannerEpoch, RecommendationContext,
    RecommendationFreshnessError, ResourceGeneration,
};
pub use flat_attention_planner::RuntimeDeviceCapabilities;
pub use flat_attention_planner::kernel_autotune::SelectionRecord as FlatTuningRecord;
use flat_attention_planner::kernel_candidates::SelectionPolicy as FlatSelectionPolicy;
use flat_attention_planner::kernel_ir::{AttentionProblem, KERNEL_MAX_HEAD_DIM};
pub use flat_elastic_kernel::contextual::ContextualAdapterPlan;
use flat_elastic_kernel::contextual::generate_and_plan_with_context;
use flat_elastic_kernel::{AdapterError, latency_policy};

/// Exact merged FLAT revision used by the contextual planner rail.
pub const CONTEXTUAL_FLAT_REVISION: &str = "75d3bd684643aedb98f55a892f93d727a8187cea";
/// Exact merged ElasticXxx revision used transitively and directly here.
pub const CONTEXTUAL_ELASTICXXX_REVISION: &str = "9130a412857335cc5120b013b91552dd0808f9f1";

const SCIRUST_RESOURCE_PREFIX: &str = "scirust/dense-attention/";

/// SciRust-owned semantic description of dense multi-head self-attention.
///
/// This intentionally does not represent GQA/MQA, cross-attention, or unequal
/// query/KV lengths. Those families need their own FLAT Kernel IR contracts;
/// silently folding them into the dense Q4 family would be incorrect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseAttentionSpec {
    /// Batch size.
    pub batch: usize,
    /// Query/key/value head count.
    pub heads: usize,
    /// Shared query/key/value sequence length.
    pub seq_len: usize,
    /// Width of one attention head.
    pub head_dim: usize,
    /// Whether autoregressive causal masking is applied.
    pub causal: bool,
}

impl DenseAttentionSpec {
    /// Validate and lower the SciRust semantic request to FLAT's dense Kernel IR.
    pub fn to_flat_problem(self) -> Result<AttentionProblem, ContextualPlanningError> {
        if self.batch == 0
        {
            return Err(ContextualPlanningError::ZeroDimension("batch"));
        }
        if self.heads == 0
        {
            return Err(ContextualPlanningError::ZeroDimension("heads"));
        }
        if self.seq_len == 0
        {
            return Err(ContextualPlanningError::ZeroDimension("seq_len"));
        }
        if self.head_dim == 0
        {
            return Err(ContextualPlanningError::ZeroDimension("head_dim"));
        }
        if self.head_dim > KERNEL_MAX_HEAD_DIM as usize
        {
            return Err(ContextualPlanningError::HeadDimensionTooLarge {
                head_dim: self.head_dim,
                max: KERNEL_MAX_HEAD_DIM,
            });
        }

        let batch_heads = self
            .batch
            .checked_mul(self.heads)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(ContextualPlanningError::IndexSpaceOverflow("batch_heads"))?;
        let seq_len = u32::try_from(self.seq_len)
            .map_err(|_| ContextualPlanningError::IndexSpaceOverflow("seq_len"))?;
        let head_dim = u32::try_from(self.head_dim)
            .map_err(|_| ContextualPlanningError::IndexSpaceOverflow("head_dim"))?;

        Ok(AttentionProblem {
            batch_heads,
            seq_len,
            head_dim,
            causal: self.causal,
        })
    }
}

/// Errors at the SciRust semantic/planner boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContextualPlanningError {
    /// A dense-attention dimension was zero.
    ZeroDimension(&'static str),
    /// The head width is outside the currently qualified FLAT dense family.
    HeadDimensionTooLarge {
        /// Requested head width.
        head_dim: usize,
        /// Largest FLAT dense head width.
        max: u32,
    },
    /// A semantic dimension cannot be represented in FLAT's u32 index space.
    IndexSpaceOverflow(&'static str),
    /// SciRust's stable semantic resource identifier could not be constructed.
    InvalidResourceIdentity,
    /// The FLAT -> ElasticXxx adapter rejected the request.
    Adapter(AdapterError),
}

impl fmt::Display for ContextualPlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroDimension(field) => write!(f, "dense attention dimension `{field}` is zero"),
            Self::HeadDimensionTooLarge { head_dim, max } => write!(
                f,
                "dense attention head_dim {head_dim} exceeds FLAT qualified maximum {max}"
            ),
            Self::IndexSpaceOverflow(field) =>
            {
                write!(
                    f,
                    "dense attention dimension `{field}` exceeds FLAT u32 index space"
                )
            },
            Self::InvalidResourceIdentity =>
            {
                write!(f, "SciRust contextual planner resource identity is invalid")
            },
            Self::Adapter(error) => write!(f, "FLAT contextual planner failed: {error}"),
        }
    }
}

impl std::error::Error for ContextualPlanningError {}

impl From<AdapterError> for ContextualPlanningError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(error)
    }
}

/// Stable SciRust semantic resource tracked by recommendation freshness.
pub fn semantic_resource_id(
    spec: DenseAttentionSpec,
) -> Result<LogicalResourceId, ContextualPlanningError> {
    let problem = spec.to_flat_problem()?;
    LogicalResourceId::new(format!(
        "{SCIRUST_RESOURCE_PREFIX}{}",
        problem.canonical_record()
    ))
    .map_err(|_| ContextualPlanningError::InvalidResourceIdentity)
}

/// Build the exact dependency context carried by one advisory recommendation.
pub fn recommendation_context(
    spec: DenseAttentionSpec,
    planner_epoch: PlannerEpoch,
    observation_epoch: ObservationEpoch,
    resource_generation: ResourceGeneration,
) -> Result<RecommendationContext, ContextualPlanningError> {
    Ok(RecommendationContext::new(planner_epoch, observation_epoch)
        .with_resource_generation(semantic_resource_id(spec)?, resource_generation))
}

/// Build a trusted current snapshot for revalidating a recommendation before
/// any downstream consumer acts on it.
pub fn freshness_snapshot(
    spec: DenseAttentionSpec,
    planner_epoch: PlannerEpoch,
    observation_epoch: ObservationEpoch,
    resource_generation: ResourceGeneration,
) -> Result<FreshnessSnapshot, ContextualPlanningError> {
    Ok(FreshnessSnapshot::new(planner_epoch, observation_epoch)
        .with_resource_generation(semantic_resource_id(spec)?, resource_generation))
}

/// Produce an advisory contextual plan for one SciRust dense-attention request.
///
/// `capabilities` must come from an authoritative runtime/device discovery
/// boundary. This function never guesses missing limits. `tuning` may carry
/// FLAT M26 evidence; without comparative evidence ElasticXxx may honestly
/// return an insufficient-evidence outcome. `accept_uncontested_fallback` only
/// permits the explicitly documented ElasticXxx single-survivor fallback.
///
/// The returned plan is **not executable by this module**. A future runtime
/// integration may consume it only after WGPU/runtime ABI unification and after
/// [`ContextualAdapterPlan::selected_flat_candidate_if_fresh`] succeeds against
/// a trusted [`FreshnessSnapshot`].
pub fn plan_contextual(
    spec: DenseAttentionSpec,
    capabilities: &RuntimeDeviceCapabilities,
    tuning: Option<&FlatTuningRecord>,
    planner_epoch: PlannerEpoch,
    observation_epoch: ObservationEpoch,
    resource_generation: ResourceGeneration,
    accept_uncontested_fallback: bool,
) -> Result<ContextualAdapterPlan, ContextualPlanningError> {
    let problem = spec.to_flat_problem()?;
    let context =
        recommendation_context(spec, planner_epoch, observation_epoch, resource_generation)?;
    let elastic_policy = latency_policy(accept_uncontested_fallback)?;
    Ok(generate_and_plan_with_context(
        &problem,
        capabilities,
        &FlatSelectionPolicy::default(),
        tuning,
        &elastic_policy,
        context,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> DenseAttentionSpec {
        DenseAttentionSpec {
            batch: 2,
            heads: 4,
            seq_len: 128,
            head_dim: 64,
            causal: true,
        }
    }

    fn portable_caps() -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65_535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1_024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32_768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: false,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            f16_supported: false,
        }
    }

    #[test]
    fn dense_spec_lowers_to_exact_flat_geometry() {
        let problem = spec().to_flat_problem().expect("valid problem");
        assert_eq!(problem.batch_heads, 8);
        assert_eq!(problem.seq_len, 128);
        assert_eq!(problem.head_dim, 64);
        assert!(problem.causal);
    }

    #[test]
    fn unsupported_head_width_fails_before_planning() {
        let mut request = spec();
        request.head_dim = KERNEL_MAX_HEAD_DIM as usize + 1;
        assert_eq!(
            request.to_flat_problem(),
            Err(ContextualPlanningError::HeadDimensionTooLarge {
                head_dim: KERNEL_MAX_HEAD_DIM as usize + 1,
                max: KERNEL_MAX_HEAD_DIM,
            })
        );
    }

    #[test]
    fn recommendation_tracks_scirust_semantic_generation() {
        let context = recommendation_context(
            spec(),
            PlannerEpoch::new(3),
            ObservationEpoch::new(5),
            ResourceGeneration::new(7),
        )
        .expect("context");
        let resource = semantic_resource_id(spec()).expect("resource id");
        assert_eq!(
            context.resource_generation(&resource),
            Some(ResourceGeneration::new(7))
        );
    }

    #[test]
    fn stale_semantic_generation_is_rejected() {
        let context = recommendation_context(
            spec(),
            PlannerEpoch::new(3),
            ObservationEpoch::new(5),
            ResourceGeneration::new(7),
        )
        .expect("context");
        let stale = freshness_snapshot(
            spec(),
            PlannerEpoch::new(3),
            ObservationEpoch::new(5),
            ResourceGeneration::new(8),
        )
        .expect("snapshot");
        assert!(matches!(
            context.validate_freshness(&stale),
            Err(RecommendationFreshnessError::ResourceGenerationMismatch { .. })
        ));
    }

    #[test]
    fn portable_profile_reaches_flat_and_elastic_without_execution() {
        let plan = plan_contextual(
            spec(),
            &portable_caps(),
            None,
            PlannerEpoch::new(1),
            ObservationEpoch::new(1),
            ResourceGeneration::new(1),
            false,
        )
        .expect("host-only contextual planning must succeed");
        assert!(!plan.flat_candidates.is_empty());
        assert!(plan.flat_candidates.iter().all(|candidate| {
            candidate.static_requirements().iter().all(|requirement| {
                !matches!(
                    requirement,
                    flat_attention_planner::kernel_ir::CapabilityRequirement::SubgroupOperations
                )
            })
        }));
    }

    #[test]
    fn manifest_pins_match_reviewed_revisions() {
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains(&format!("rev = \"{CONTEXTUAL_FLAT_REVISION}\"")));
        assert!(manifest.contains(&format!("rev = \"{CONTEXTUAL_ELASTICXXX_REVISION}\"")));
    }
}
