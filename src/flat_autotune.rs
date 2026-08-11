//! Public SciRust facade for the reusable FLAT + ElasticAutoTuner planner.
//!
//! The implementation lives in `scirust-gpu` so SciAgent and other model runtimes
//! can consume the same contract without depending back on the root `scirust` crate.

pub use scirust_gpu::flat_autotune::*;
