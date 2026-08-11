//! Incremental high-performance AutoDiff extensions.
//!
//! Kept behind one small module boundary so new execution/storage strategies can
//! land without repeatedly touching the legacy monolithic `lib.rs` implementation.

#[path = "differential_tensor.rs"]
mod differential_tensor;

pub use differential_tensor::DifferentialTensor;
