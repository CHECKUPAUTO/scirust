//! Public SciRust facade for FLAT attention planning.
//!
//! The existing exports remain the production `scirust-gpu` +
//! `ElasticAutoTuner` path. [`contextual`] is a separate host-only advisory rail
//! through the reviewed FLAT Kernel IR and ElasticXxx freshness contracts; it
//! deliberately exposes no execution method while SciRust and current FLAT use
//! different WGPU major versions.

pub use scirust_gpu::flat_autotune::*;

pub mod contextual;
