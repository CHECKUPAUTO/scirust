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
mod shard;
mod verify;
mod vmap;

pub use autodiff::{
    AutodiffError, GradGraph, JvpGraph, VjpGraph, grad, jvp, value_and_grad, vjp,
};
pub use error::GraphError;
pub use graph::{Graph, Node, TensorType};
pub use ids::{ConstantId, NodeId};
pub use operation::{Operation, Scalar};
pub use optimize::{
    OptimizationConfig, OptimizationError, OptimizationStats, OptimizedGraph, optimize_graph,
};
pub use shard::{
    AxisShard, DeviceMesh, MeshAxis, PartitionSpec, RankShard, ShardError, ShardMapGraph,
    ShardPlan, ShardPolicy, plan_sharding, shard_map,
};
pub use verify::{SemanticError, validate_semantics};
pub use vmap::{VmapError, VmapGraph, vmap};

pub use scirust_compute::{DType, Shape};
