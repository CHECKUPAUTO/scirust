use crate::TriangularCubicFlow;
use crate::error::CausalError;
use scirust_solvers::Matrix;
use scirust_solvers::linalg::SplitMix64;

pub struct SyntheticDataConfig {
    pub seed: u64,
    pub dim: usize,
    pub sample_count: usize,
}

impl SyntheticDataConfig {
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

pub fn generate_noise_matrix(config: &SyntheticDataConfig) -> Matrix {
    let mut rng = SplitMix64::new(config.seed);
    let mut data = Vec::with_capacity(config.sample_count * config.dim);
    for _ in 0..config.sample_count * config.dim
    {
        data.push(rng.next_gaussian());
    }
    Matrix::from_row_major(config.sample_count, config.dim, data)
}

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

    for &v in samples_data.iter()
    {
        if !v.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "generate_causal_samples",
                index: 0,
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
