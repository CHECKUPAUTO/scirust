//! Deterministic Phase 6 quantization for the Elastic Latent KV experiment.
//!
//! This isolated reference module compares FP32 latent coefficients and sparse
//! residual values with row-wise symmetric INT8 and packed signed INT4 storage.
//! It enumerates independent key/value formats, enforces a strict byte budget,
//! and validates reconstruction plus attention quality against an FP32 oracle.

include!("phase6/part1.rs");
include!("phase6/part2.rs");
include!("phase6/part3.rs");
