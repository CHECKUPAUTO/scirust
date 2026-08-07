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

/// Owned construction-time calibration resolved from immutable Phase 10 commits.
///
/// Phase 13 currently accepts a square basis carrier. A lower-rank learned basis
/// is therefore copied into the leading columns of a square matrix and the
/// unused columns are zero-filled. `maximum_rank` is validated not to exceed the
/// learned prefix, so the zero-filled carrier columns can never be selected.
pub struct ResolvedHeadCalibration<'a> {
    full_key_basis: Vec<f32>,
    full_value_basis: Vec<f32>,
    quality: AdaptiveQualityProfile<'a>,
    basis_version: u32,
    key_rank: usize,
    value_rank: usize,
}

impl ResolvedHeadCalibration<'_> {
    /// Common K/V committed version represented by this calibration.
    #[must_use]
    pub const fn basis_version(&self) -> u32 {
        self.basis_version
    }

    /// Learned key basis rank before square-carrier materialization.
    #[must_use]
    pub const fn key_rank(&self) -> usize {
        self.key_rank
    }

    /// Learned value basis rank before square-carrier materialization.
    #[must_use]
    pub const fn value_rank(&self) -> usize {
        self.value_rank
    }

    /// Borrows the resolved data in the form consumed by Phase 13.
    #[must_use]
    pub fn as_head_calibration(&self) -> HeadCalibration<'_> {
        HeadCalibration {
            full_key_basis: &self.full_key_basis,
            full_value_basis: &self.full_value_basis,
            quality: self.quality,
            basis_version: self.basis_version,
        }
    }
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
    InvalidBasisShape {
        head: usize,
        channel: &'static str,
        dimension: usize,
        elements: usize,
    },
    LearnedRankTooLarge {
        head: usize,
        channel: &'static str,
        dimension: usize,
        rank: usize,
    },
    MaximumRankExceedsLearned {
        head: usize,
        maximum_rank: usize,
        key_rank: usize,
        value_rank: usize,
    },
    Runtime(ElasticLatentRuntimeError),
}

impl fmt::Display for BasisHandoffError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Learning(error) => write!(output, "{error}"),
            Self::ArchiveCapacityOverflow =>
            {
                write!(output, "committed basis archive size overflow")
            },
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
            Self::InvalidBasisShape {
                head,
                channel,
                dimension,
                elements,
            } => write!(
                output,
                "head {head} {channel} committed basis with {elements} elements is not row-major [dimension={dimension}, rank]"
            ),
            Self::LearnedRankTooLarge {
                head,
                channel,
                dimension,
                rank,
            } => write!(
                output,
                "head {head} {channel} learned rank {rank} exceeds dimension {dimension}"
            ),
            Self::MaximumRankExceedsLearned {
                head,
                maximum_rank,
                key_rank,
                value_rank,
            } => write!(
                output,
                "head {head} runtime maximum rank {maximum_rank} exceeds learned key/value ranks {key_rank}/{value_rank}"
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
/// Lower-rank learned bases remain lower-rank semantically. They are copied into
/// square construction-time carriers only because the current Phase 13 backend
/// API accepts a full row-major matrix; the planner is forbidden from selecting
/// any of the zero-filled columns beyond the learned ranks.
pub fn resolve_committed_head_calibration<'a>(
    head: usize,
    dimension: usize,
    maximum_rank: usize,
    learned: LearnedHeadBasis<'a>,
) -> Result<ResolvedHeadCalibration<'a>, BasisHandoffError> {
    let basis_version = learned
        .key
        .current_version()
        .min(learned.value.current_version());
    let key_basis = learned.key.committed_basis(basis_version).ok_or(
        BasisHandoffError::MissingCommittedVersion {
            head,
            channel: "key",
            version: basis_version,
        },
    )?;
    let value_basis = learned.value.committed_basis(basis_version).ok_or(
        BasisHandoffError::MissingCommittedVersion {
            head,
            channel: "value",
            version: basis_version,
        },
    )?;
    let (full_key_basis, key_rank) = materialize_square_basis(head, "key", dimension, key_basis)?;
    let (full_value_basis, value_rank) =
        materialize_square_basis(head, "value", dimension, value_basis)?;
    if maximum_rank > key_rank || maximum_rank > value_rank
    {
        return Err(BasisHandoffError::MaximumRankExceedsLearned {
            head,
            maximum_rank,
            key_rank,
            value_rank,
        });
    }
    Ok(ResolvedHeadCalibration {
        full_key_basis,
        full_value_basis,
        quality: learned.quality,
        basis_version,
        key_rank,
        value_rank,
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
    let mut resolved = Vec::with_capacity(learned_heads.len());
    for (head, learned) in learned_heads.iter().copied().enumerate()
    {
        resolved.push(resolve_committed_head_calibration(
            head,
            attention.d_head,
            config.maximum_rank,
            learned,
        )?);
    }
    let calibrations: Vec<_> = resolved
        .iter()
        .map(ResolvedHeadCalibration::as_head_calibration)
        .collect();
    Ok(ElasticLatentDecodeRuntime::new(
        attention,
        config,
        &calibrations,
    )?)
}

fn materialize_square_basis(
    head: usize,
    channel: &'static str,
    dimension: usize,
    basis: &[f32],
) -> Result<(Vec<f32>, usize), BasisHandoffError> {
    if dimension == 0 || basis.is_empty() || basis.len() % dimension != 0
    {
        return Err(BasisHandoffError::InvalidBasisShape {
            head,
            channel,
            dimension,
            elements: basis.len(),
        });
    }
    let rank = basis.len() / dimension;
    if rank > dimension
    {
        return Err(BasisHandoffError::LearnedRankTooLarge {
            head,
            channel,
            dimension,
            rank,
        });
    }
    let mut full = vec![0.0; dimension.saturating_mul(dimension)];
    for row in 0..dimension
    {
        let source = row * rank;
        let destination = row * dimension;
        full[destination..destination + rank].copy_from_slice(&basis[source..source + rank]);
    }
    Ok((full, rank))
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
        let mut learner =
            CommittedBasisLearner::new(learner_config(3, 2), identity_prefix(3, 2)).unwrap();
        let first = learner.observe(&[1.0, 0.0, 1.0]).unwrap();
        let version = first.committed.expect("first sample commits");
        let fingerprint = basis_fingerprint(learner.committed_basis(version.version).unwrap());
        assert_eq!(fingerprint, version.fingerprint);

        learner.observe(&[0.0, 1.0, 1.0]).unwrap();
        let preserved = learner.committed_basis(version.version).unwrap();
        assert_eq!(basis_fingerprint(preserved), version.fingerprint);
    }

    #[test]
    fn resolver_uses_highest_common_lower_rank_version() {
        const RANK_QUALITY: [u16; 3] = [4_000, 8_000, 10_000];
        const RESIDUAL_GAIN: [u16; 1] = [0];
        let mut key =
            CommittedBasisLearner::new(learner_config(3, 2), identity_prefix(3, 2)).unwrap();
        let mut value =
            CommittedBasisLearner::new(learner_config(3, 2), identity_prefix(3, 2)).unwrap();
        key.observe(&[1.0, 0.0, 1.0]).unwrap();
        key.observe(&[0.0, 1.0, 1.0]).unwrap();
        value.observe(&[1.0, 0.0, 1.0]).unwrap();
        assert_eq!(key.current_version(), 2);
        assert_eq!(value.current_version(), 1);

        let quality = AdaptiveQualityProfile {
            key_rank_quality_bps: &RANK_QUALITY,
            value_rank_quality_bps: &RANK_QUALITY,
            key_residual_gain_bps: &RESIDUAL_GAIN,
            value_residual_gain_bps: &RESIDUAL_GAIN,
        };
        let resolved = resolve_committed_head_calibration(
            0,
            3,
            2,
            LearnedHeadBasis {
                key: &key,
                value: &value,
                quality,
            },
        )
        .unwrap();
        assert_eq!(resolved.basis_version(), 1);
        assert_eq!(resolved.key_rank(), 2);
        assert_eq!(resolved.value_rank(), 2);
        let calibration = resolved.as_head_calibration();
        let archived_key = key.committed_basis(1).unwrap();
        let archived_value = value.committed_basis(1).unwrap();
        for row in 0..3
        {
            assert_eq!(
                &calibration.full_key_basis[row * 3..row * 3 + 2],
                &archived_key[row * 2..row * 2 + 2]
            );
            assert_eq!(
                &calibration.full_value_basis[row * 3..row * 3 + 2],
                &archived_value[row * 2..row * 2 + 2]
            );
            assert_eq!(calibration.full_key_basis[row * 3 + 2], 0.0);
            assert_eq!(calibration.full_value_basis[row * 3 + 2], 0.0);
        }
    }

    #[test]
    fn fresh_runtime_consumes_committed_lower_rank_epoch() {
        const RANK_QUALITY: [u16; 3] = [4_000, 8_000, 10_000];
        const RESIDUAL_GAIN: [u16; 1] = [0];
        let mut rng = PcgEngine::new(101);
        let attention = MultiHeadAttention::new(3, 1, false, &KaimingNormal, &Zeros, &mut rng);
        let mut key =
            CommittedBasisLearner::new(learner_config(3, 2), identity_prefix(3, 2)).unwrap();
        let mut value =
            CommittedBasisLearner::new(learner_config(3, 2), identity_prefix(3, 2)).unwrap();
        key.observe(&[1.0, 0.0, 1.0]).unwrap();
        value.observe(&[1.0, 0.0, 1.0]).unwrap();
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
        let output = runtime
            .decode_step(&attention, &[0.25, -0.5, 0.75])
            .unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(runtime.telemetry().steps, 1);
    }
}
