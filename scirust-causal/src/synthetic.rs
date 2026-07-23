//! Deterministic seeded generation of synthetic causal samples.
//!
//! Given a [`TriangularCubicFlow`] and a `SplitMix64` seed, `generate_causal_samples`
//! produces `X = flow⁻¹(noise)` so that `flow(X) = noise`. Same seed ⇒ identical
//! output. The seed lives on the (public) [`SyntheticDataConfig`]; the returned
//! `Matrix` does not embed it, so pair the samples with their config for
//! provenance (a self-describing causal dataset with recorded seed/assumptions is
//! the job of the typed causal data model, not this generator).

use crate::TriangularCubicFlow;
use crate::error::CausalError;
use scirust_solvers::Matrix;
use scirust_solvers::linalg::SplitMix64;

/// Configuration for synthetic-sample generation.
pub struct SyntheticDataConfig {
    /// `SplitMix64` seed — the sole source of randomness; recorded here.
    pub seed: u64,
    /// Variable count.
    pub dim: usize,
    /// Number of samples (rows) to generate.
    pub sample_count: usize,
}

impl SyntheticDataConfig {
    /// Validates and constructs a configuration.
    ///
    /// # Errors
    ///
    /// [`CausalError::ZeroDimension`] / [`CausalError::ZeroSamples`] /
    /// [`CausalError::AllocationOverflow`] for degenerate sizes.
    pub fn new(seed: u64, dim: usize, sample_count: usize) -> Result<Self, CausalError> {
        if dim == 0
        {
            return Err(CausalError::ZeroDimension);
        }
        if sample_count == 0
        {
            return Err(CausalError::ZeroSamples);
        }
        if sample_count.checked_mul(dim).is_none()
        {
            return Err(CausalError::AllocationOverflow);
        }
        Ok(Self {
            seed,
            dim,
            sample_count,
        })
    }
}

/// Generates an `n×d` matrix of `SplitMix64(seed)` standard-Gaussian draws in
/// fixed row-major order (fully determined by `config.seed`).
pub fn generate_noise_matrix(config: &SyntheticDataConfig) -> Matrix {
    let mut rng = SplitMix64::new(config.seed);
    let mut data = Vec::with_capacity(config.sample_count * config.dim);
    for _ in 0..config.sample_count * config.dim
    {
        data.push(rng.next_gaussian());
    }
    Matrix::from_row_major(config.sample_count, config.dim, data)
}

/// Generates `X = flow⁻¹(noise)` for seeded Gaussian `noise`, so `flow(X)` is
/// standard Gaussian by construction. Deterministic in `config.seed`.
///
/// # Errors
///
/// [`CausalError`] if the flow inverse fails or a sample is non-finite.
pub fn generate_causal_samples(
    flow: &TriangularCubicFlow,
    config: &SyntheticDataConfig,
) -> Result<Matrix, CausalError> {
    let noise = generate_noise_matrix(config);
    let mut samples_data = Vec::with_capacity(config.sample_count * config.dim);

    for row in 0..config.sample_count
    {
        let noise_row: Vec<f64> = (0..config.dim).map(|col| noise[(row, col)]).collect();

        let sample = flow.inverse(&noise_row)?;
        samples_data.extend(sample);
    }

    for (index, &v) in samples_data.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "generate_causal_samples",
                index,
                value: v,
            });
        }
    }

    Ok(Matrix::from_row_major(
        config.sample_count,
        config.dim,
        samples_data,
    ))
}
