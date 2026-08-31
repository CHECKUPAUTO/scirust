//! Canonical backend-neutral tensor graph IR for SciRust.
//!
//! This crate contains graph structure, tensor metadata and pure graph
//! transformations. It does not embed tensor storage, executable kernels,
//! devices, streams, or backend state.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod autodiff;
mod error;
mod graph;
mod ids;
mod operation;
mod optimize;
mod physical;
mod representation;
mod shard;
mod verify;
mod vmap;

/// Build a reusable first-order linearization graph. The returned [`JvpGraph`]
/// exposes explicit tangent inputs, so it can be executed repeatedly for
/// different tangent vectors without rebuilding the primal transform.
pub use autodiff::jvp as linearize;
pub use autodiff::{AutodiffError, GradGraph, JvpGraph, VjpGraph, grad, jvp, value_and_grad, vjp};
pub use error::GraphError;
pub use graph::{Graph, Node, TensorType};
pub use ids::{ConstantId, NodeId};
pub use operation::{Operation, Scalar};
pub use optimize::{
    OptimizationConfig, OptimizationError, OptimizationStats, OptimizedGraph, optimize_graph,
};
pub use physical::{
    ContentIdentity, EffectiveBitsRate, LayoutIdentity, MaterializationClass,
    PhysicalAccountingError, PhysicalAccountingScope, PhysicalSegment, PhysicalSegmentId,
    PhysicalSegmentReference, PhysicalSegmentRole, ReconstructionRole, ResidentMaterialization,
    SegmentLifetime, SegmentUse,
};
pub use representation::{
    PrimitiveRepresentation, Rebinding, RepresentationComponent, RepresentationError,
    RepresentationId, RepresentationPlan, StorageBits,
};
pub use shard::{
    AxisShard, DeviceMesh, MeshAxis, PartitionSpec, RankShard, ShardError, ShardMapGraph,
    ShardPlan, ShardPolicy, plan_sharding, shard_map,
};
pub use verify::{SemanticError, validate_semantics};
pub use vmap::{VmapError, VmapGraph, vmap};

pub use scirust_compute::{DType, Shape};
