//! Opt-in Elastic Latent KV bridge for the legacy incremental Transformer path.
//!
//! The historical `TransformerEncoder::infer_step` keeps its dense internal KV
//! cache for backwards compatibility. This module provides a session-scoped
//! counterpart that preserves the surrounding LayerNorm/residual/FFN pipeline
//! while routing every attention layer through `ElasticLatentDecodeRuntime`.
//!
//! This API is inference-only. The numeric attention result is reintroduced as
//! a tape input, so gradients do not propagate through the Elastic Latent KV
//! attention step. Training continues to use the existing differentiable paths.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::elastic_latent_runtime::{
    ElasticLatentDecodeRuntime, ElasticLatentRuntimeConfig, ElasticLatentRuntimeError,
    ElasticLatentTelemetry, HeadCalibration,
};
use crate::nn::module::Module;
use crate::nn::transformer::encoder::TransformerEncoder;
use core::fmt;

/// Per-layer construction inputs for one encoder decode session.
#[derive(Clone, Copy)]
pub struct ElasticLatentLayerConfig<'a> {
    pub runtime: ElasticLatentRuntimeConfig,
    pub heads: &'a [HeadCalibration<'a>],
}

/// Errors surfaced by the Transformer-level Elastic Latent KV bridge.
#[derive(Debug)]
pub enum ElasticLatentTransformerError {
    LayerCount {
        expected: usize,
        actual: usize,
    },
    ModelWidth {
        expected: usize,
        actual: usize,
    },
    TokenShape {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    PositionMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    Runtime {
        layer: usize,
        source: ElasticLatentRuntimeError,
    },
}

impl fmt::Display for ElasticLatentTransformerError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::LayerCount { expected, actual } =>
            {
                write!(
                    output,
                    "elastic runtime layer mismatch: expected {expected}, got {actual}"
                )
            },
            Self::ModelWidth { expected, actual } =>
            {
                write!(
                    output,
                    "elastic runtime model width mismatch: expected {expected}, got {actual}"
                )
            },
            Self::TokenShape { expected, actual } =>
            {
                write!(
                    output,
                    "elastic incremental token shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            },
            Self::PositionMismatch {
                layer,
                expected,
                actual,
            } =>
            {
                write!(
                    output,
                    "elastic layer {layer} position mismatch: expected {expected}, got {actual}"
                )
            },
            Self::Runtime { layer, source } =>
            {
                write!(output, "elastic layer {layer} decode failed: {source}")
            },
        }
    }
}

impl std::error::Error for ElasticLatentTransformerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Runtime { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// One bounded Elastic Latent KV runtime per Transformer encoder block.
pub struct ElasticLatentEncoderSession {
    runtimes: Vec<ElasticLatentDecodeRuntime>,
    d_model: usize,
}

impl ElasticLatentEncoderSession {
    /// Constructs a session whose layer topology is frozen to `encoder`.
    pub fn new(
        encoder: &TransformerEncoder,
        layers: &[ElasticLatentLayerConfig<'_>],
    ) -> Result<Self, ElasticLatentTransformerError> {
        if layers.len() != encoder.blocks.len()
        {
            return Err(ElasticLatentTransformerError::LayerCount {
                expected: encoder.blocks.len(),
                actual: layers.len(),
            });
        }

        let mut runtimes = Vec::with_capacity(layers.len());
        for (layer, (block, config)) in encoder.blocks.iter().zip(layers).enumerate()
        {
            let runtime = ElasticLatentDecodeRuntime::new(&block.mha, config.runtime, config.heads)
                .map_err(|source| ElasticLatentTransformerError::Runtime { layer, source })?;
            runtimes.push(runtime);
        }
        Ok(Self {
            runtimes,
            d_model: encoder.d_model,
        })
    }

    /// Number of attention layers owned by this decode session.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.runtimes.len()
    }

    /// Returns bounded telemetry for one layer without allocating a summary.
    #[must_use]
    pub fn layer_telemetry(&self, layer: usize) -> Option<ElasticLatentTelemetry> {
        self.runtimes.get(layer).map(|runtime| runtime.telemetry())
    }

    /// Executes one incremental encoder token through the Elastic Latent KV path.
    ///
    /// `pos` must equal the number of tokens already admitted by every layer.
    /// This rejects skipped, repeated, or out-of-order positions before any KV
    /// state is mutated.
    pub fn infer_step<'t>(
        &mut self,
        encoder: &mut TransformerEncoder,
        tape: &'t Tape,
        x_token: Var<'t>,
        pos: usize,
    ) -> Result<Var<'t>, ElasticLatentTransformerError> {
        if encoder.blocks.len() != self.runtimes.len()
        {
            return Err(ElasticLatentTransformerError::LayerCount {
                expected: self.runtimes.len(),
                actual: encoder.blocks.len(),
            });
        }
        if encoder.d_model != self.d_model
        {
            return Err(ElasticLatentTransformerError::ModelWidth {
                expected: self.d_model,
                actual: encoder.d_model,
            });
        }

        let token = tape.value(x_token.idx());
        if token.rows != 1 || token.cols != self.d_model
        {
            return Err(ElasticLatentTransformerError::TokenShape {
                expected: (1, self.d_model),
                actual: (token.rows, token.cols),
            });
        }

        for (layer, runtime) in self.runtimes.iter().enumerate()
        {
            let expected = runtime.telemetry().steps;
            if expected != pos
            {
                return Err(ElasticLatentTransformerError::PositionMismatch {
                    layer,
                    expected,
                    actual: pos,
                });
            }
        }

        let mut hidden = x_token;
        for (layer, (block, runtime)) in encoder
            .blocks
            .iter_mut()
            .zip(self.runtimes.iter_mut())
            .enumerate()
        {
            let ln1_out = block.ln1.forward(tape, hidden);
            let normalized = tape.value(ln1_out.idx());
            debug_assert_eq!(normalized.rows, 1);
            debug_assert_eq!(normalized.cols, block.d_model);
            let attention = runtime
                .decode_step(&block.mha, &normalized.data)
                .map_err(|source| ElasticLatentTransformerError::Runtime { layer, source })?;
            let attention = tape.input(Tensor::from_vec(attention, 1, block.d_model));
            let residual = hidden
                .try_add(attention)
                .expect("elastic attention output preserves model width");

            let ln2_out = block.ln2.forward(tape, residual);
            let feed_forward = block.ffn1.forward(tape, ln2_out).relu();
            let feed_forward = block.ffn2.forward(tape, feed_forward);
            hidden = residual
                .try_add(feed_forward)
                .expect("feed-forward output preserves model width");
        }

        Ok(encoder.final_ln.forward(tape, hidden))
    }
}

/// Method-style opt-in for callers migrating from `TransformerEncoder::infer_step`.
pub trait ElasticLatentInferStep {
    fn infer_step_elastic<'t>(
        &mut self,
        session: &mut ElasticLatentEncoderSession,
        tape: &'t Tape,
        x_token: Var<'t>,
        pos: usize,
    ) -> Result<Var<'t>, ElasticLatentTransformerError>;
}

impl ElasticLatentInferStep for TransformerEncoder {
    fn infer_step_elastic<'t>(
        &mut self,
        session: &mut ElasticLatentEncoderSession,
        tape: &'t Tape,
        x_token: Var<'t>,
        pos: usize,
    ) -> Result<Var<'t>, ElasticLatentTransformerError> {
        session.infer_step(self, tape, x_token, pos)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ElasticLatentEncoderSession, ElasticLatentInferStep, ElasticLatentLayerConfig,
        ElasticLatentTransformerError,
    };
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::nn::adaptive_latent_kv::AdaptiveQualityProfile;
    use crate::nn::elastic_latent_runtime::{ElasticLatentRuntimeConfig, HeadCalibration};
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::latent_kv_kernels::LatentKernelKind;
    use crate::nn::latent_kv_lifecycle::{CompressionTier, LifecycleConfig};
    use crate::nn::rng::PcgEngine;
    use crate::nn::transformer::encoder::TransformerEncoder;

    static QUALITY: [u16; 4] = [2_500, 5_000, 7_500, 10_000];
    static RESIDUAL: [u16; 1] = [0];

    fn identity(dimension: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * dimension];
        for index in 0..dimension
        {
            basis[index * dimension + index] = 1.0;
        }
        basis
    }

    fn runtime_config(capacity: usize) -> ElasticLatentRuntimeConfig {
        let tier = CompressionTier {
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            maximum_residual_slots: 0,
            rank_divisor: 1,
        };
        ElasticLatentRuntimeConfig {
            capacity_tokens: capacity,
            minimum_rank: 4,
            maximum_rank: 4,
            maximum_residual_slots: 0,
            persistent_budget_bytes: 4_096,
            allocated_ceiling_bytes: 16_384,
            lifecycle: LifecycleConfig {
                capacity_tokens: capacity,
                hot_tokens: capacity.min(2),
                warm_tokens: capacity.saturating_sub(capacity.min(2)).min(2),
                hot: tier,
                warm: tier,
                cold: tier,
            },
            kernel: LatentKernelKind::Scalar,
        }
    }

    #[test]
    fn elastic_encoder_matches_dense_legacy_at_full_rank() {
        let mut rng = PcgEngine::new(17);
        let encoder = TransformerEncoder::new(2, 8, 2, 16, true, &KaimingNormal, &Zeros, &mut rng);
        let mut dense = encoder.clone();
        let mut elastic = encoder;
        let basis = identity(4);
        let profile = AdaptiveQualityProfile {
            key_rank_quality_bps: &QUALITY,
            value_rank_quality_bps: &QUALITY,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        };
        let head = HeadCalibration {
            full_key_basis: &basis,
            full_value_basis: &basis,
            quality: profile,
            basis_version: 5,
        };
        let heads0 = [head; 2];
        let heads1 = [head; 2];
        let layers = [
            ElasticLatentLayerConfig {
                runtime: runtime_config(4),
                heads: &heads0,
            },
            ElasticLatentLayerConfig {
                runtime: runtime_config(4),
                heads: &heads1,
            },
        ];
        let mut session = ElasticLatentEncoderSession::new(&elastic, &layers).unwrap();
        let dense_tape = Tape::new();
        let elastic_tape = Tape::new();

        for pos in 0..3
        {
            let data: Vec<f32> = (0..8)
                .map(|index| ((pos * 8 + index) as f32 * 0.17).sin() * 0.4)
                .collect();
            let dense_token = dense_tape.input(Tensor::from_vec(data.clone(), 1, 8));
            let elastic_token = elastic_tape.input(Tensor::from_vec(data, 1, 8));
            let dense_out = dense.infer_step(&dense_tape, dense_token, pos);
            let elastic_out = elastic
                .infer_step_elastic(&mut session, &elastic_tape, elastic_token, pos)
                .unwrap();
            let expected = dense_tape.value(dense_out.idx());
            let actual = elastic_tape.value(elastic_out.idx());
            for (left, right) in expected.data.iter().zip(&actual.data)
            {
                assert!(
                    (left - right).abs() <= 3.0e-4,
                    "dense={left} elastic={right} pos={pos}"
                );
            }
        }

        assert_eq!(session.layers(), 2);
        assert_eq!(session.layer_telemetry(0).unwrap().steps, 3);
        assert_eq!(session.layer_telemetry(1).unwrap().steps, 3);
        assert!(
            elastic
                .blocks
                .iter()
                .all(|block| block.mha.kv_cache.borrow().is_none())
        );
    }

    #[test]
    fn elastic_encoder_rejects_out_of_order_position_before_mutation() {
        let mut rng = PcgEngine::new(23);
        let mut encoder =
            TransformerEncoder::new(1, 8, 2, 16, true, &KaimingNormal, &Zeros, &mut rng);
        let basis = identity(4);
        let profile = AdaptiveQualityProfile {
            key_rank_quality_bps: &QUALITY,
            value_rank_quality_bps: &QUALITY,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        };
        let heads = [HeadCalibration {
            full_key_basis: &basis,
            full_value_basis: &basis,
            quality: profile,
            basis_version: 0,
        }; 2];
        let layers = [ElasticLatentLayerConfig {
            runtime: runtime_config(2),
            heads: &heads,
        }];
        let mut session = ElasticLatentEncoderSession::new(&encoder, &layers).unwrap();
        let tape = Tape::new();
        let token = tape.input(Tensor::from_vec(vec![0.1; 8], 1, 8));
        let error = encoder
            .infer_step_elastic(&mut session, &tape, token, 1)
            .unwrap_err();
        assert!(matches!(
            error,
            ElasticLatentTransformerError::PositionMismatch {
                layer: 0,
                expected: 0,
                actual: 1
            }
        ));
        assert_eq!(session.layer_telemetry(0).unwrap().steps, 0);
        assert!(encoder.blocks[0].mha.kv_cache.borrow().is_none());
    }

    #[test]
    fn elastic_encoder_rejects_layer_topology_mismatch() {
        let mut rng = PcgEngine::new(29);
        let encoder = TransformerEncoder::new(1, 8, 2, 16, true, &KaimingNormal, &Zeros, &mut rng);
        let error = match ElasticLatentEncoderSession::new(&encoder, &[])
        {
            Ok(_) => panic!("mismatched topology must fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ElasticLatentTransformerError::LayerCount {
                expected: 1,
                actual: 0
            }
        ));
    }
}
