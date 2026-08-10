//! Incremental high-performance AutoDiff extensions.
//!
//! Kept behind one small module boundary so new execution/storage strategies can
//! land without repeatedly touching the legacy monolithic `lib.rs` implementation.

#[path = "differential_tensor.rs"]
mod differential_tensor;
#[path = "sparse_jacobian.rs"]
mod sparse_jacobian;

pub use differential_tensor::DifferentialTensor;
pub use sparse_jacobian::{ColumnColoring, JacobianSparsity, SparsityError};
