//! Deterministic online basis learning and versioning for Elastic Latent KV.
//!
//! Explicit index loops are intentional: they define the scalar update and
//! Gram-Schmidt order used by the reproducibility contract.

#![allow(clippy::needless_range_loop)]

use core::fmt;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Immutable learner configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisLearningConfig {
    pub dimension: usize,
    pub rank: usize,
    pub learning_rate: f32,
    pub reorthogonalize_interval: usize,
    pub minimum_samples_between_versions: usize,
    pub minimum_quality_gain_bps: u16,
    pub maximum_versions: usize,
}

/// Immutable metadata for one committed basis epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisVersion {
    pub version: u32,
    pub samples_seen: usize,
    pub quality_bps: u16,
    pub fingerprint: u64,
}

/// Result of one online observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisObservation {
    pub quality_bps: u16,
    pub committed: Option<BasisVersion>,
}

/// Errors returned by online basis construction or observation.
#[derive(Debug, Clone, PartialEq)]
pub enum BasisLearningError {
    ZeroField(&'static str),
    RankTooLarge,
    InvalidLearningRate,
    BasisLength { expected: usize, actual: usize },
    SampleLength { expected: usize, actual: usize },
    NonFinite,
}

impl fmt::Display for BasisLearningError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroField(field) => write!(output, "{field} must be non-zero"),
            Self::RankTooLarge => write!(output, "basis rank exceeds dense dimension"),
            Self::InvalidLearningRate =>
            {
                write!(output, "learning rate must be finite and in (0, 1]")
            },
            Self::BasisLength { expected, actual } =>
            {
                write!(
                    output,
                    "basis length mismatch: expected {expected}, got {actual}"
                )
            },
            Self::SampleLength { expected, actual } =>
            {
                write!(
                    output,
                    "sample length mismatch: expected {expected}, got {actual}"
                )
            },
            Self::NonFinite => write!(output, "basis learning input contains a non-finite value"),
        }
    }
}

impl std::error::Error for BasisLearningError {}

/// Online Oja-style learner with deterministic scalar update order.
#[derive(Debug, Clone)]
pub struct DeterministicBasisLearner {
    config: BasisLearningConfig,
    basis: Vec<f32>,
    coefficients: Vec<f32>,
    residual: Vec<f32>,
    versions: Vec<BasisVersion>,
    samples_seen: usize,
    last_version_sample: usize,
    last_committed_quality_bps: u16,
    current_version: u32,
}

impl DeterministicBasisLearner {
    pub fn new(
        config: BasisLearningConfig,
        initial_basis: Vec<f32>,
    ) -> Result<Self, BasisLearningError> {
        validate_config(config)?;
        let expected = config.dimension.saturating_mul(config.rank);
        if initial_basis.len() != expected
        {
            return Err(BasisLearningError::BasisLength {
                expected,
                actual: initial_basis.len(),
            });
        }
        if initial_basis.iter().any(|value| !value.is_finite())
        {
            return Err(BasisLearningError::NonFinite);
        }

        let mut learner = Self {
            config,
            basis: initial_basis,
            coefficients: vec![0.0; config.rank],
            residual: vec![0.0; config.dimension],
            versions: Vec::with_capacity(config.maximum_versions),
            samples_seen: 0,
            last_version_sample: 0,
            last_committed_quality_bps: 0,
            current_version: 0,
        };
        learner.orthonormalize();
        learner.versions.push(BasisVersion {
            version: 0,
            samples_seen: 0,
            quality_bps: 0,
            fingerprint: basis_fingerprint(&learner.basis),
        });
        Ok(learner)
    }

    #[must_use]
    pub fn basis(&self) -> &[f32] {
        &self.basis
    }

    #[must_use]
    pub fn versions(&self) -> &[BasisVersion] {
        &self.versions
    }

    #[must_use]
    pub const fn current_version(&self) -> u32 {
        self.current_version
    }

    #[must_use]
    pub const fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    pub fn observe(&mut self, sample: &[f32]) -> Result<BasisObservation, BasisLearningError> {
        validate_sample(sample, self.config.dimension)?;
        self.project(sample);
        self.reconstruct_residual(sample);

        for column in 0..self.config.rank
        {
            let coefficient = self.coefficients[column];
            for row in 0..self.config.dimension
            {
                let index = row * self.config.rank + column;
                self.basis[index] += self.config.learning_rate * coefficient * self.residual[row];
            }
        }

        self.samples_seen = self.samples_seen.saturating_add(1);
        if self
            .samples_seen
            .is_multiple_of(self.config.reorthogonalize_interval)
        {
            self.orthonormalize();
        }

        let quality_bps = self.quality_bps(sample);
        let enough_samples = self.samples_seen.saturating_sub(self.last_version_sample)
            >= self.config.minimum_samples_between_versions;
        let enough_gain = quality_bps.saturating_sub(self.last_committed_quality_bps)
            >= self.config.minimum_quality_gain_bps;
        let has_capacity = self.versions.len() < self.config.maximum_versions;

        let committed = if enough_samples && enough_gain && has_capacity
        {
            self.current_version = self.current_version.saturating_add(1);
            self.last_version_sample = self.samples_seen;
            self.last_committed_quality_bps = quality_bps;
            let version = BasisVersion {
                version: self.current_version,
                samples_seen: self.samples_seen,
                quality_bps,
                fingerprint: basis_fingerprint(&self.basis),
            };
            self.versions.push(version);
            Some(version)
        }
        else
        {
            None
        };

        Ok(BasisObservation {
            quality_bps,
            committed,
        })
    }

    fn project(&mut self, sample: &[f32]) {
        self.coefficients.fill(0.0);
        for row in 0..self.config.dimension
        {
            let offset = row * self.config.rank;
            for column in 0..self.config.rank
            {
                self.coefficients[column] += self.basis[offset + column] * sample[row];
            }
        }
    }

    fn reconstruct_residual(&mut self, sample: &[f32]) {
        for row in 0..self.config.dimension
        {
            let offset = row * self.config.rank;
            let mut reconstructed = 0.0_f32;
            for column in 0..self.config.rank
            {
                reconstructed += self.basis[offset + column] * self.coefficients[column];
            }
            self.residual[row] = sample[row] - reconstructed;
        }
    }

    fn quality_bps(&mut self, sample: &[f32]) -> u16 {
        self.project(sample);
        let captured: f32 = self.coefficients.iter().map(|value| value * value).sum();
        let total: f32 = sample.iter().map(|value| value * value).sum();
        if total == 0.0
        {
            return 10_000;
        }
        ((captured / total).clamp(0.0, 1.0) * 10_000.0).round() as u16
    }

    fn orthonormalize(&mut self) {
        for column in 0..self.config.rank
        {
            for previous in 0..column
            {
                let mut dot = 0.0_f32;
                for row in 0..self.config.dimension
                {
                    let current = self.basis[row * self.config.rank + column];
                    let previous_value = self.basis[row * self.config.rank + previous];
                    dot += current * previous_value;
                }
                for row in 0..self.config.dimension
                {
                    let current_index = row * self.config.rank + column;
                    let previous_value = self.basis[row * self.config.rank + previous];
                    self.basis[current_index] -= dot * previous_value;
                }
            }

            let mut norm_squared = 0.0_f32;
            for row in 0..self.config.dimension
            {
                let value = self.basis[row * self.config.rank + column];
                norm_squared += value * value;
            }
            if norm_squared <= f32::EPSILON
            {
                for row in 0..self.config.dimension
                {
                    self.basis[row * self.config.rank + column] = 0.0;
                }
                self.basis[(column % self.config.dimension) * self.config.rank + column] = 1.0;
                norm_squared = 1.0;
            }
            let inverse_norm = norm_squared.sqrt().recip();
            for row in 0..self.config.dimension
            {
                self.basis[row * self.config.rank + column] *= inverse_norm;
            }
        }
    }
}

fn validate_config(config: BasisLearningConfig) -> Result<(), BasisLearningError> {
    if config.dimension == 0
    {
        return Err(BasisLearningError::ZeroField("dimension"));
    }
    if config.rank == 0
    {
        return Err(BasisLearningError::ZeroField("rank"));
    }
    if config.rank > config.dimension
    {
        return Err(BasisLearningError::RankTooLarge);
    }
    if !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
    {
        return Err(BasisLearningError::InvalidLearningRate);
    }
    if config.reorthogonalize_interval == 0
    {
        return Err(BasisLearningError::ZeroField("reorthogonalize_interval"));
    }
    if config.minimum_samples_between_versions == 0
    {
        return Err(BasisLearningError::ZeroField(
            "minimum_samples_between_versions",
        ));
    }
    if config.maximum_versions == 0
    {
        return Err(BasisLearningError::ZeroField("maximum_versions"));
    }
    Ok(())
}

fn validate_sample(sample: &[f32], dimension: usize) -> Result<(), BasisLearningError> {
    if sample.len() != dimension
    {
        return Err(BasisLearningError::SampleLength {
            expected: dimension,
            actual: sample.len(),
        });
    }
    if sample.iter().any(|value| !value.is_finite())
    {
        return Err(BasisLearningError::NonFinite);
    }
    Ok(())
}

/// Computes a stable fingerprint over exact `f32` bit patterns.
#[must_use]
pub fn basis_fingerprint(basis: &[f32]) -> u64 {
    let mut fingerprint = FNV_OFFSET;
    for value in basis
    {
        fingerprint ^= u64::from(value.to_bits());
        fingerprint = fingerprint.wrapping_mul(FNV_PRIME);
    }
    fingerprint
}

#[cfg(test)]
mod tests {
    use super::{BasisLearningConfig, DeterministicBasisLearner, basis_fingerprint};

    fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * rank];
        for diagonal in 0..rank
        {
            basis[diagonal * rank + diagonal] = 1.0;
        }
        basis
    }

    fn config() -> BasisLearningConfig {
        BasisLearningConfig {
            dimension: 6,
            rank: 3,
            learning_rate: 0.05,
            reorthogonalize_interval: 2,
            minimum_samples_between_versions: 2,
            minimum_quality_gain_bps: 0,
            maximum_versions: 8,
        }
    }

    #[test]
    fn repeated_training_is_bit_identical() {
        let samples = [
            [1.0, 0.2, -0.1, 0.7, 0.0, 0.3],
            [0.8, -0.4, 0.5, 0.2, 0.1, -0.2],
            [0.3, 0.9, 0.1, -0.5, 0.4, 0.6],
            [-0.2, 0.1, 0.7, 0.8, -0.3, 0.2],
        ];
        let mut first = DeterministicBasisLearner::new(config(), identity_prefix(6, 3)).unwrap();
        let mut second = DeterministicBasisLearner::new(config(), identity_prefix(6, 3)).unwrap();
        for sample in samples
        {
            first.observe(&sample).unwrap();
            second.observe(&sample).unwrap();
        }
        assert_eq!(
            basis_fingerprint(first.basis()),
            basis_fingerprint(second.basis())
        );
        assert_eq!(first.versions(), second.versions());
    }

    #[test]
    fn scheduled_reorthogonalization_normalizes_columns() {
        let mut learner = DeterministicBasisLearner::new(config(), identity_prefix(6, 3)).unwrap();
        learner.observe(&[0.4, 0.2, -0.7, 0.8, 0.5, -0.1]).unwrap();
        learner.observe(&[0.1, -0.5, 0.9, 0.3, 0.2, 0.6]).unwrap();
        for column in 0..3
        {
            let norm = (0..6)
                .map(|row| learner.basis()[row * 3 + column].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn version_records_respect_fixed_capacity() {
        let mut learner = DeterministicBasisLearner::new(config(), identity_prefix(6, 3)).unwrap();
        for index in 0..40
        {
            let sample = [index as f32 * 0.01 + 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
            learner.observe(&sample).unwrap();
        }
        assert!(learner.versions().len() <= config().maximum_versions);
    }
}
