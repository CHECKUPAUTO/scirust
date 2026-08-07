//! Transformer-level Phase 10 -> Phase 13 committed-basis handoff.
//!
//! This module resolves immutable Phase 10 key/value snapshots per attention
//! head, then constructs the existing Phase 13 `ElasticLatentEncoderSession`.
//! The resulting session owns its decode runtimes, so later learner commits are
//! visible only to future sessions/cache epochs.

use crate::nn::elastic_latent_basis_handoff::{
    BasisHandoffError, LearnedHeadBasis, ResolvedHeadCalibration,
    resolve_committed_head_calibration,
};
use crate::nn::elastic_latent_runtime::ElasticLatentRuntimeConfig;
use crate::nn::elastic_latent_transformer::{
    ElasticLatentEncoderSession, ElasticLatentLayerConfig, ElasticLatentTransformerError,
};
use crate::nn::transformer::encoder::TransformerEncoder;
use core::fmt;

/// Learned Phase 10 basis sources plus the Phase 13 runtime policy for one layer.
#[derive(Clone, Copy)]
pub struct LearnedLayerBasis<'a> {
    pub runtime: ElasticLatentRuntimeConfig,
    pub heads: &'a [LearnedHeadBasis<'a>],
}

/// Errors surfaced while opening a Transformer decode epoch from learned bases.
#[derive(Debug)]
pub enum EncoderBasisHandoffError {
    LayerCount {
        expected: usize,
        actual: usize,
    },
    Head {
        layer: usize,
        source: BasisHandoffError,
    },
    Session(ElasticLatentTransformerError),
}

impl fmt::Display for EncoderBasisHandoffError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::LayerCount { expected, actual } => write!(
                output,
                "learned layer count mismatch: expected {expected}, got {actual}"
            ),
            Self::Head { layer, source } =>
            {
                write!(
                    output,
                    "elastic basis handoff failed for layer {layer}: {source}"
                )
            },
            Self::Session(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for EncoderBasisHandoffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Head { source, .. } => Some(source),
            Self::Session(error) => Some(error),
            Self::LayerCount { .. } => None,
        }
    }
}

/// Opens a fresh Phase 13 Transformer session from the latest common committed
/// key/value version available for every attention head.
pub fn encoder_session_from_committed_bases(
    encoder: &TransformerEncoder,
    learned_layers: &[LearnedLayerBasis<'_>],
) -> Result<ElasticLatentEncoderSession, EncoderBasisHandoffError> {
    if learned_layers.len() != encoder.blocks.len()
    {
        return Err(EncoderBasisHandoffError::LayerCount {
            expected: encoder.blocks.len(),
            actual: learned_layers.len(),
        });
    }

    let mut resolved_layers: Vec<Vec<ResolvedHeadCalibration<'_>>> =
        Vec::with_capacity(learned_layers.len());
    for (layer, (block, learned_layer)) in encoder.blocks.iter().zip(learned_layers).enumerate()
    {
        if learned_layer.heads.len() != block.mha.n_heads
        {
            return Err(EncoderBasisHandoffError::Head {
                layer,
                source: BasisHandoffError::HeadCount {
                    expected: block.mha.n_heads,
                    actual: learned_layer.heads.len(),
                },
            });
        }
        let mut resolved_heads = Vec::with_capacity(learned_layer.heads.len());
        for (head, learned) in learned_layer.heads.iter().copied().enumerate()
        {
            resolved_heads.push(
                resolve_committed_head_calibration(
                    head,
                    block.mha.d_head,
                    learned_layer.runtime.maximum_rank,
                    learned,
                )
                .map_err(|source| EncoderBasisHandoffError::Head { layer, source })?,
            );
        }
        resolved_layers.push(resolved_heads);
    }

    let calibration_layers: Vec<Vec<_>> = resolved_layers
        .iter()
        .map(|heads| {
            heads
                .iter()
                .map(ResolvedHeadCalibration::as_head_calibration)
                .collect()
        })
        .collect();
    let configs: Vec<ElasticLatentLayerConfig<'_>> = learned_layers
        .iter()
        .zip(&calibration_layers)
        .map(|(learned, heads)| ElasticLatentLayerConfig {
            runtime: learned.runtime,
            heads,
        })
        .collect();

    ElasticLatentEncoderSession::new(encoder, &configs).map_err(EncoderBasisHandoffError::Session)
}

#[cfg(test)]
mod tests {
    use super::{LearnedLayerBasis, encoder_session_from_committed_bases};
    use crate::nn::adaptive_latent_basis::BasisLearningConfig;
    use crate::nn::adaptive_latent_kv::AdaptiveQualityProfile;
    use crate::nn::elastic_latent_basis_handoff::{CommittedBasisLearner, LearnedHeadBasis};
    use crate::nn::elastic_latent_runtime::ElasticLatentRuntimeConfig;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::latent_kv_kernels::LatentKernelKind;
    use crate::nn::latent_kv_lifecycle::{CompressionTier, LifecycleConfig};
    use crate::nn::rng::PcgEngine;
    use crate::nn::transformer::encoder::TransformerEncoder;

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
            maximum_versions: 4,
        }
    }

    fn runtime_config() -> ElasticLatentRuntimeConfig {
        let tier = CompressionTier {
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            maximum_residual_slots: 0,
            rank_divisor: 1,
        };
        ElasticLatentRuntimeConfig {
            capacity_tokens: 4,
            minimum_rank: 1,
            maximum_rank: 2,
            maximum_residual_slots: 0,
            persistent_budget_bytes: 4_096,
            allocated_ceiling_bytes: 65_536,
            lifecycle: LifecycleConfig {
                capacity_tokens: 4,
                hot_tokens: 1,
                warm_tokens: 1,
                hot: tier,
                warm: tier,
                cold: tier,
            },
            kernel: LatentKernelKind::Scalar,
        }
    }

    #[test]
    fn transformer_session_opens_from_committed_lower_rank_bases() {
        const RANK_QUALITY: [u16; 3] = [4_000, 8_000, 10_000];
        const RESIDUAL_GAIN: [u16; 1] = [0];
        let mut rng = PcgEngine::new(211);
        let encoder = TransformerEncoder::new(1, 3, 1, 6, true, &KaimingNormal, &Zeros, &mut rng);
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
        let heads = [LearnedHeadBasis {
            key: &key,
            value: &value,
            quality,
        }];
        let layers = [LearnedLayerBasis {
            runtime: runtime_config(),
            heads: &heads,
        }];

        let session = encoder_session_from_committed_bases(&encoder, &layers).unwrap();
        assert_eq!(session.layers(), 1);
    }
}
