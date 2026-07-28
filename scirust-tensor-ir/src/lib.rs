//! Canonical backend-neutral tensor graph IR for SciRust.
//!
//! This crate contains graph structure and tensor metadata only. It does not
//! embed tensor storage, executable kernels, devices, streams, or backend state.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod error;
mod graph;
mod ids;
mod operation;

pub use error::GraphError;
pub use graph::{Graph, Node, TensorType};
pub use ids::{ConstantId, NodeId};
pub use operation::{Operation, Scalar};

pub use scirust_compute::{DType, Shape};
