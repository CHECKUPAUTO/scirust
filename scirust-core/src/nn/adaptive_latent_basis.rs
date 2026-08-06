//! Deterministic online basis learning and versioning for Elastic Latent KV.

use core::fmt;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Immutable learner configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisLearningConfig {
    /// Dense vector dimension.
    pub dimension: usize,
    /// Learned basis rank.
    pub rank: usize,
    /// Oja-style update rate.
    pub learning_rate: f32,
    /// Number of samples between deterministic Gram-Schmidt passes.
    pub reorthogonalize_interval: usize,
    /// Minimum samples between committed basis versions.
    pub minimum_samples_between_versions: usize,
    /// Minimum quality improvement required to commit a version.
    pub minimum_quality_gain_bps: u16,
    /// Maximum committed version records retained by the learner.
    pub maximum_versions: usize,
}

/// Immutable metadata for one committed basis epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisVersion {
    /// Monotonic version number, starting at zero for the initial basis.
    pub version: u32,
    /// Number of samples observed when the version was committed.
    pub samples_seen: usize,
    /// Captured-energy estimate in basis points.
    pub quality_bps: u16,
    /// Stable FNV-1a fingerprint of all basis coefficients.
    pub fingerprint: u64,
}

/// Result of one online observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisObservation {
    /// Quality after applying the sample and any scheduled re-orthogonalization.
    pub quality_bps: u16,
    /// Newly committed version, if the commit conditions were met.
    pub committed: Option<BasisVersion>,
}

/// Errors returned by online basis construction or observation.
#[derive(Debug, Clone, PartialEq)]
pub enum BasisLearningError {
    /// A required integer field was zero.
    ZeroField(&'static str),
    /// Rank exceeded the dense dimension.
    RankTooLarge,
    /// Learning rate was outside `(0, 1]` or non-finite.
    InvalidLearningRate,
    /// Initial basis length mismatch.
    BasisLength {
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Sample length mismatch.
    SampleLength {
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A basis or sample value was non-finite.
    NonFinite,
}

impl fmt::Display for BasisLearningError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroField(field) => write!(output, "{field} must be non-zero"),
            Self::RankTooLarge => write!(output, "basis rank exceeds dense dimension"),
            Self::InvalidLearningRate => write!(output, "learning rate must be finite and in (0, 1]"),
            Self::BasisLength { expected, actual } => {
                write!(output, "basis length mismatch: expected {expected}, got {actual}")
            }
            Self::SampleLength { expected, actual } => {
                write!(output, "sample length mismatch: expected {expected}, got {actual}")
            }
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
    /// Creates a learner from a row-major `[dimension, rank]` basis.
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
        let initial = BasisVersion {
            version: 0,
            samples_seen: 0,
            quality_bps: 0,
            fingerprint: basis_fingerprint(&learner.basis),
        };
        learner.versions.push(initial);
        Ok(learner)
    }

    /// Returns the current row-major basis.
    #[must_use]
    pub fn basis(&self) -> &[f32] {
        &self.basis
    }

    /// Returns committed version metadata.
    #[must_use]
    pub fn versions(&self) -> &[BasisVersion] {
        &self.versions
    }

    /// Returns the current basis version number.
    #[must_use]
    pub const fn current_version(&self) -> u32 {
        self.current_version
    }

    /// Returns the number of observed samples.
    #[must_use]
    pub const fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    /// Applies one deterministic online update without growing scratch buffers.
    pub fn observe(&mut self, sample: &[f32]) -> Result<BasisObservation, BasisLearningError> {
        if sample.len() != self.config.dimension
        {
            return Err(BasisLearningError::SampleLength {
                expected: self.config.dimension,
                actual: sample.len(),
            });
        }
        if sample.iter().any(|value| !value.is_finite())
        {
            return Err(BasisLearningError::NonFinite);
        }

        self.project(sample);
        self.reconstruct_residual(sample);
        for column in 0..self.config.rank
        {
            let coefficient = self.coefficients[column];
            for row in 0..self.config.dimension
            {
                let index = row * self.config.rank + column;
                self.basis[index] +=
                    self.config.learning_rate * coefficient * self.residual[row];
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
        let enough_samples = self
            .samples_seen
            .saturating_sub(self.last_version_sample)
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
                    dot += self.basis[row * self.config.rank + column]
                        * self.basis[row * self.config.rank + previous];
                }
                for row in 0..self.config.dimension
                {
                    let index = row * self.config.rank + column;
                    self.basis[index] -= dot * self.basis[row * self.config.rank + previous];
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
        assert_eq!(basis_fingerprint(first.basis()), basis_fingerprint(second.basis()));
        assert_eq!(first.versions(), second.versions());
    }

    #[test]
    fn reorthogonalized_columns_have_unit_norm() {
        let mut learner = DeterministicBasisLearner::new(config(), identity_prefix(6, 3)).unwrap();
        learner.observe(&[0.4, 0.2, -0.7, 0.8, 0.5, -0.1]).unwrap();
        learner.observe(&[0.1, -0.5, 0.9, 0.3, 0.2, 0.6]).unwrap();
        for column in 0..3
        {
            let norm: f32 = (0..6)
                .map(|row| learner.basis()[row * 3 + column].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn version_capacity_never_grows() {
        let mut learner = DeterministicBasisLearner::new(config(), identity_prefix(6, 3)).unwrap();
        for index in 0..40
        {
            let sample = [index as f32 * 0.01 + 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
            learner.observe(&sample).unwrap();
        }
        assert!(learner.versions().len() <= config().maximum_versions);
    }
}
