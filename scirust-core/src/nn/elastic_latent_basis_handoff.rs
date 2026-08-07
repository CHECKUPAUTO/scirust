//! Phase 10 committed-basis archive and Phase 13 runtime handoff.
//!
//! `DeterministicBasisLearner` keeps the live online basis and version metadata.
//! This bridge wraps it with a fixed-capacity archive containing the exact basis
//! snapshot associated with every committed version. All archive storage is
//! allocated at construction; `observe` only copies into already-reserved slots.
//!
//! New Phase 13 runtimes select the highest committed version shared by the key
//! and value learners of each head. Existing runtimes keep their construction
//! bases, so a later learner commit only affects future cache epochs.

use crate::nn::adaptive_latent_basis::{
    BasisLearningConfig, BasisLearningError, BasisObservation, BasisVersion,
    DeterministicBasisLearner,
};
use crate::nn::adaptive_latent_kv::AdaptiveQualityProfile;
use crate::nn::elastic_latent_runtime::{
    ElasticLatentDecodeRuntime, ElasticLatentRuntimeConfig, ElasticLatentRuntimeError,
    HeadCalibration,
};
use crate::nn::transformer::attention::MultiHeadAttention;
use core::fmt;

/// Phase 10 learner plus fixed-capacity immutable snapshots of committed bases.
#[derive(Debug, Clone)]
pub struct CommittedBasisLearner {
    learner: DeterministicBasisLearner,
    committed_bases: Vec<f32>,
    committed_slots: usize,
}

impl CommittedBasisLearner {
    /// Constructs the learner and preallocates storage for every possible commit.
    pub fn new(
        config: BasisLearningConfig,
        initial_basis: Vec<f32>,
    ) -> Result<Self, BasisHandoffError> {
        let maximum_versions = config.maximum_versions;
        let learner = DeterministicBasisLearner::new(config, initial_basis)?;
        let basis_len = learner.basis().len();
        let archive_len = basis_len
            .checked_mul(maximum_versions)
            .ok_or(BasisHandoffError::ArchiveCapacityOverflow)?;
        let mut committed_bases = vec![0.0; archive_len];
        committed_bases[..basis_len].copy_from_slice(learner.basis());
        Ok(Self {
            learner,
            committed_bases,
            committed_slots: 1,
        })
    }

    /// Observes one sample and snapshots the basis when Phase 10 commits a version.
    ///
    /// No archive allocation occurs here: the destination slot was reserved by
    /// [`Self::new`].
    pub fn observe(&mut self, sample: &[f32]) -> Result<BasisObservation, BasisHandoffError> {
        let observation = self.learner.observe(sample)?;
        if let Some(version) = observation.committed
        {
            let slot = self
                .learner
                .versions()
                .iter()
                .position(|candidate| *candidate == version)
                .expect("a committed version must be present in learner metadata");
            let basis_len = self.learner.basis().len();
            let start = slot * basis_len;
            let end = start + basis_len;
            self.committed_bases[start..end].copy_from_slice(self.learner.basis());
            self.committed_slots = self.committed_slots.max(slot + 1);
        }
        Ok(observation)
    }

    /// Current mutable online basis. It may be newer than the latest commit.
    #[must_use]
    pub fn basis(&self) -> &[f32] {
        self.learner.basis()
    }

    /// Metadata for every committed Phase 10 version.
    #[must_use]
    pub fn versions(&self) -> &[BasisVersion] {
        self.learner.versions()
    }

    /// Latest committed version number.
    #[must_use]
    pub const fn current_version(&self) -> u32 {
        self.learner.current_version()
    }

    /// Number of online samples observed by the wrapped learner.
    #[must_use]
    pub const fn samples_seen(&self) -> usize {
        self.learner.samples_seen()
    }

    /// Returns the immutable basis snapshot associated with `version`.
    #[must_use]
    pub fn committed_basis(&self, version: u32) -> Option<&[f32]> {
        let slot = self
            .learner
            .versions()
            .iter()
            .position(|candidate| candidate.version == version)?;
        if slot >= self.committed_slots
        {
            return None;
        }
        let basis_len = self.learner.basis().len();
        let start = slot * basis_len;
        Some(&self.committed_bases[start..start + basis_len])
    }

    /// Returns the latest immutable committed basis snapshot.
    #[must_use]
    pub fn latest_committed_basis(&self) -> &[f32] {
        self.committed_basis(self.current_version())
            .expect("the current committed version always has an archive slot")
    }
}

/// Phase 10 learned key/value sources for one attention head.
#[derive(Clone, Copy)]
pub struct LearnedHeadBasis<'a> {
    pub key: &'a CommittedBasisLearner,
    pub value: &'a CommittedBasisLearner,
    pub quality: AdaptiveQualityProfile<'a>,
}

/// Errors surfaced while archiving or handing learned bases to Phase 13.
#[derive(Debug)]
pub enum BasisHandoffError {
    Learning(BasisLearningError),
    ArchiveCapacityOverflow,
    HeadCount {
        expected: usize,
        actual: usize,
    },
    MissingCommittedVersion {
        head: usize,
        channel: &'static str,
        version: u32,
    },
    BasisShape {
        head: usize,
        channel: &'static str,
        expected: usize,
        actual: usize,
    },
    Runtime(ElasticLatentRuntimeError),
}

impl fmt::Display for BasisHandoffError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Learning(error) => write!(output, "{error}"),
            Self::ArchiveCapacityOverflow => write!(output, "committed basis archive size overflow"),
            Self::HeadCount { expected, actual } => write!(
                output,
                "learned head count mismatch: expected {expected}, got {actual}"
            ),
            Self::MissingCommittedVersion {
                head,
                channel,
                version,
            } => write!(
                output,
                "head {head} {channel} learner has no committed basis for version {version}"
            ),
            Self::BasisShape {
                head,
                channel,
                expected,
                actual,
            } => write!(
                output,
                "head {head} {channel} committed basis length mismatch: expected {expected}, got {actual}"
            ),
            Self::Runtime(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for BasisHandoffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Learning(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BasisLearningError> for BasisHandoffError {
    fn from(error: BasisLearningError) -> Self {
        Self::Learning(error)
    }
}

impl From<ElasticLatentRuntimeError> for BasisHandoffError {
    fn from(error: ElasticLatentRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

/// Resolves one head to the newest key/value basis version committed by both learners.
///
/// Phase 13 currently consumes a full row-major `[dimension, dimension]` basis
/// because its policy may select any prefix rank up to the dense head dimension.
pub fn resolve_committed_head_calibration<'a>(
    head: usize,
    dimension: usize,
    learned: LearnedHeadBasis<'a>,
) -> Result<HeadCalibration<'a>, BasisHandoffError> {
    let basis_version = learned
        .key
        .current_version()
        .min(learned.value.current_version());
    let full_key_basis = learned.key.committed_basis(basis_version).ok_or(
        BasisHandoffError::MissingCommittedVersion {
            head,
            channel: "key",
            version: basis_version,
        },
    )?;
    let full_value_basis = learned.value.committed_basis(basis_version).ok_or(
        BasisHandoffError::MissingCommittedVersion {
            head,
            channel: "value",
            version: basis_version,
        },
    )?;
    let expected = dimension.saturating_mul(dimension);
    validate_basis_shape(head, "key", full_key_basis, expected)?;
    validate_basis_shape(head, "value", full_value_basis, expected)?;
    Ok(HeadCalibration {
        full_key_basis,
        full_value_basis,
        quality: learned.quality,
        basis_version,
    })
}

/// Builds a fresh Phase 13 decode epoch from the latest common committed bases.
///
/// The returned runtime owns all encoded cache state and therefore does not
/// borrow the learners. Subsequent online observations can safely advance the
/// learners; those commits become visible only when another runtime is created.
pub fn runtime_from_committed_bases(
    attention: &MultiHeadAttention,
    config: ElasticLatentRuntimeConfig,
    learned_heads: &[LearnedHeadBasis<'_>],
) -> Result<ElasticLatentDecodeRuntime, BasisHandoffError> {
    if learned_heads.len() != attention.n_heads
    {
        return Err(BasisHandoffError::HeadCount {
            expected: attention.n_heads,
            actual: learned_heads.len(),
        });
    }
    let mut calibrations = Vec::with_capacity(learned_heads.len());
    for (head, learned) in learned_heads.iter().copied().enumerate()
    {
        calibrations.push(resolve_committed_head_calibration(
            head,
            attention.d_head,
            learned,
        )?);
    }
    Ok(ElasticLatentDecodeRuntime::new(
        attention,
        config,
        &calibrations,
    )?)
}

fn validate_basis_shape(
    head: usize,
    channel: &'static str,
    basis: &[f32],
    expected: usize,
) -> Result<(), BasisHandoffError> {
    if basis.len() != expected
    {
        return Err(BasisHandoffError::BasisShape {
            head,
            channel,
            expected,
            actual: basis.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CommittedBasisLearner, LearnedHeadBasis, resolve_committed_head_calibration,
        runtime_from_committed_bases,
    };
    use crate::nn::adaptive_latent_basis::{BasisLearningConfig, basis_fingerprint};
    use crate::nn::adaptive_latent_kv::AdaptiveQualityProfile;
    use crate::nn::elastic_latent_runtime::ElasticLatentRuntimeConfig;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::latent_kv_kernels::LatentKernelKind;
    use crate::nn::latent_kv_lifecycle::{CompressionTier, LifecycleConfig};
    use crate::nn::rng::PcgEngine;
    use crate::nn::transformer::attention::MultiHeadAttention;

    fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * rank];
        for diagonal in 0..rank
        {
            basis[diagonal * rank + diagonal] = 1.0;
        }
        basis
    }

    fn learner_config(dimension: usize, rank: usize) -> BasisLearningConfig {
        BasisLearningConfig {
            dimension,
            rank,
            learning_rate: 0.05,
            reorthogonalize_interval: 1,
            minimum_samples_between_versions: 1,
            minimum_quality_gain_bps: 0,
            maximum_versions: 8,
        }
    }

    fn f32_tier() -> CompressionTier {
        CompressionTier {
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            maximum_residual_slots: 0,
            rank_divisor: 1,
        }
    }

    #[test]
    fn committed_snapshot_survives_future_training() {
        let mut learner = CommittedBasisLearner::new(
            learner_config(3, 2),
            identity_prefix(3, 2),
        )
        .unwrap();
        let first = learner.observe(&[1.0, 0.0, 1.0]).unwrap();
        let version = first.committed.expect("first sample commits");
        let fingerprint = basis_fingerprint(learner.committed_basis(version.version).unwrap());
        assert_eq!(fingerprint, version.fingerprint);

        learner.observe(&[0.0, 1.0, 1.0]).unwrap();
        let preserved = learner.committed_basis(version.version).unwrap();
        assert_eq!(basis_fingerprint(preserved), version.fingerprint);
    }

    #[test]
    fn resolver_uses_highest_common_key_value_version() {
        const RANK_QUALITY: [u16; 2] = [5_000, 10_000];
        const RESIDUAL_GAIN: [u16; 1] = [0];
        let mut key =
            CommittedBasisLearner::new(learner_config(2, 2), identity_prefix(2, 2)).unwrap();
        let mut value =
            CommittedBasisLearner::new(learner_config(2, 2), identity_prefix(2, 2)).unwrap();
        key.observe(&[1.0, 0.0]).unwrap();
        key.observe(&[0.0, 1.0]).unwrap();
        value.observe(&[1.0, 0.0]).unwrap();
        assert_eq!(key.current_version(), 2);
        assert_eq!(value.current_version(), 1);

        let quality = AdaptiveQualityProfile {
            key_rank_quality_bps: &RANK_QUALITY,
            value_rank_quality_bps: &RANK_QUALITY,
            key_residual_gain_bps: &RESIDUAL_GAIN,
            value_residual_gain_bps: &RESIDUAL_GAIN,
        };
        let calibration = resolve_committed_head_calibration(
            0,
            2,
            LearnedHeadBasis {
                key: &key,
                value: &value,
                quality,
            },
        )
        .unwrap();
        assert_eq!(calibration.basis_version, 1);
        assert_eq!(
            basis_fingerprint(calibration.full_key_basis),
            key.versions()[1].fingerprint
        );
        assert_eq!(
            basis_fingerprint(calibration.full_value_basis),
            value.versions()[1].fingerprint
        );
    }

    #[test]
    fn fresh_runtime_consumes_committed_full_rank_epoch() {
        const RANK_QUALITY: [u16; 2] = [5_000, 10_000];
        const RESIDUAL_GAIN: [u16; 1] = [0];
        let mut rng = PcgEngine::new(101);
        let attention = MultiHeadAttention::new(2, 1, false, &KaimingNormal, &Zeros, &mut rng);
        let mut key =
            CommittedBasisLearner::new(learner_config(2, 2), identity_prefix(2, 2)).unwrap();
        let mut value =
            CommittedBasisLearner::new(learner_config(2, 2), identity_prefix(2, 2)).unwrap();
        key.observe(&[1.0, 0.0]).unwrap();
        value.observe(&[1.0, 0.0]).unwrap();
        let quality = AdaptiveQualityProfile {
            key_rank_quality_bps: &RANK_QUALITY,
            value_rank_quality_bps: &RANK_QUALITY,
            key_residual_gain_bps: &RESIDUAL_GAIN,
            value_residual_gain_bps: &RESIDUAL_GAIN,
        };
        let tier = f32_tier();
        let capacity = 4;
        let mut runtime = runtime_from_committed_bases(
            &attention,
            ElasticLatentRuntimeConfig {
                capacity_tokens: capacity,
                minimum_rank: 1,
                maximum_rank: 2,
                maximum_residual_slots: 0,
                persistent_budget_bytes: 4_096,
                allocated_ceiling_bytes: 65_536,
                lifecycle: LifecycleConfig {
                    capacity_tokens: capacity,
                    hot_tokens: 1,
                    warm_tokens: 1,
                    hot: tier,
                    warm: tier,
                    cold: tier,
                },
                kernel: LatentKernelKind::Scalar,
            },
            &[LearnedHeadBasis {
                key: &key,
                value: &value,
                quality,
            }],
        )
        .unwrap();
        let output = runtime.decode_step(&attention, &[0.25, -0.5]).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(runtime.telemetry().steps, 1);
    }
}
