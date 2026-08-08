//! Absorb SLHA residual-sign hyperplanes into ElasticMLA Q/K projections.
//!
//! The base [`crate::elastic_mla_plan::ElasticMlaLayerWeights`] already separates
//! explicit native RoPE coordinates from a continuous NoPE latent lane. This module
//! adds a third Q/K lane: real-valued residual hyperplane logits whose sign bits feed
//! the SLHA correction term. Hyperplanes are orthogonal to the continuous coarse
//! basis, so their projection of the original NoPE vector equals their projection of
//! the omitted coarse residual without reconstructing that residual.

use core::fmt;

use scirust_core::nn::{
    SlhaResidualSketch, SlhaResidualSketchError, SlhaScoreError, SlhaScoreProfile,
};

use crate::attention::GQAAttention;
use crate::elastic_mla_plan::{ElasticMlaBases, ElasticMlaLayerWeights, ElasticMlaPlanError};
use crate::model::SciAgentModel;

#[derive(Debug, Clone, PartialEq)]
pub enum ElasticMlaSlhaProjectionError {
    BasePlan(ElasticMlaPlanError),
    Sketch(SlhaResidualSketchError),
    ScoreProfile(SlhaScoreError),
    LayerBasisCount {
        layers: usize,
        bases: usize,
    },
    Overflow,
}

impl fmt::Display for ElasticMlaSlhaProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::BasePlan(error) => write!(formatter, "{error}"),
            Self::Sketch(error) => write!(formatter, "{error}"),
            Self::ScoreProfile(error) => write!(formatter, "{error}"),
            Self::LayerBasisCount { layers, bases } => write!(
                formatter,
                "ElasticMLA-SLHA basis count mismatch: model has {layers} layers, got {bases}"
            ),
            Self::Overflow => write!(formatter, "ElasticMLA-SLHA projection size overflow"),
        }
    }
}

impl std::error::Error for ElasticMlaSlhaProjectionError {}

impl From<ElasticMlaPlanError> for ElasticMlaSlhaProjectionError {
    fn from(error: ElasticMlaPlanError) -> Self {
        Self::BasePlan(error)
    }
}

impl From<SlhaResidualSketchError> for ElasticMlaSlhaProjectionError {
    fn from(error: SlhaResidualSketchError) -> Self {
        Self::Sketch(error)
    }
}

impl From<SlhaScoreError> for ElasticMlaSlhaProjectionError {
    fn from(error: SlhaScoreError) -> Self {
        Self::ScoreProfile(error)
    }
}

/// One ElasticMLA layer plus residual-sign projection weights.
#[derive(Debug, Clone)]
pub struct ElasticMlaSlhaLayerWeights {
    base: ElasticMlaLayerWeights,
    residual_bits: usize,
    score_profile: SlhaScoreProfile,
    sketch_fingerprints: Vec<u64>,
    q_residual_logits: Vec<f32>,
    k_residual_logits: Vec<f32>,
}

impl ElasticMlaSlhaLayerWeights {
    pub fn from_attention(
        attention: &GQAAttention,
        bases: &ElasticMlaBases,
        residual_bits: usize,
        group_dim: usize,
        seed: u64,
    ) -> Result<Self, ElasticMlaSlhaProjectionError> {
        let base = ElasticMlaLayerWeights::from_attention(attention, bases)?;
        let score_profile = SlhaScoreProfile::new(base.key_rank(), residual_bits, group_dim)?;
        let nope_dimensions = base.nope_dimensions();
        let q_cols = attention
            .n_heads
            .checked_mul(residual_bits)
            .ok_or(ElasticMlaSlhaProjectionError::Overflow)?;
        let k_cols = attention
            .n_kv_heads
            .checked_mul(residual_bits)
            .ok_or(ElasticMlaSlhaProjectionError::Overflow)?;
        let mut q_residual_logits = vec![0.0f32; attention.d_model * q_cols];
        let mut k_residual_logits = vec![0.0f32; attention.d_model * k_cols];
        let mut sketches = Vec::with_capacity(attention.n_kv_heads);
        let mut sketch_fingerprints = Vec::with_capacity(attention.n_kv_heads);

        for kv_head in 0..attention.n_kv_heads
        {
            let sketch_seed = derive_seed(seed, kv_head as u64);
            let sketch = SlhaResidualSketch::from_orthonormal_coarse_basis(
                nope_dimensions,
                base.key_rank(),
                residual_bits,
                sketch_seed,
                bases.key_head(kv_head),
            )?;
            sketch_fingerprints.push(sketch.fingerprint());
            sketches.push(sketch);
        }

        let repeat = attention.n_heads / attention.n_kv_heads;
        let nope_channels = bases.selection().nope_channels();
        let kv_dim = attention.n_kv_heads * attention.d_head;

        for input in 0..attention.d_model
        {
            for q_head in 0..attention.n_heads
            {
                let kv_head = q_head / repeat;
                let projection = sketches[kv_head].projection();
                for bit in 0..residual_bits
                {
                    let mut sum = 0.0f32;
                    for (local, &channel) in nope_channels.iter().enumerate()
                    {
                        sum += attention.w_q.weight.data
                            [input * attention.d_model + q_head * attention.d_head + channel]
                            * projection[local * residual_bits + bit];
                    }
                    q_residual_logits[input * q_cols + q_head * residual_bits + bit] = sum;
                }
            }

            for kv_head in 0..attention.n_kv_heads
            {
                let projection = sketches[kv_head].projection();
                for bit in 0..residual_bits
                {
                    let mut sum = 0.0f32;
                    for (local, &channel) in nope_channels.iter().enumerate()
                    {
                        sum += attention.w_k.weight.data
                            [input * kv_dim + kv_head * attention.d_head + channel]
                            * projection[local * residual_bits + bit];
                    }
                    k_residual_logits[input * k_cols + kv_head * residual_bits + bit] = sum;
                }
            }
        }

        Ok(Self {
            base,
            residual_bits,
            score_profile,
            sketch_fingerprints,
            q_residual_logits,
            k_residual_logits,
        })
    }

    #[must_use]
    pub const fn base(&self) -> &ElasticMlaLayerWeights {
        &self.base
    }

    #[must_use]
    pub const fn residual_bits(&self) -> usize {
        self.residual_bits
    }

    #[must_use]
    pub const fn score_profile(&self) -> SlhaScoreProfile {
        self.score_profile
    }

    #[must_use]
    pub fn sketch_fingerprints(&self) -> &[u64] {
        &self.sketch_fingerprints
    }

    #[must_use]
    pub fn q_residual_logits(&self) -> &[f32] {
        &self.q_residual_logits
    }

    #[must_use]
    pub fn k_residual_logits(&self) -> &[f32] {
        &self.k_residual_logits
    }

    /// Q outputs per head before head assembly: explicit RoPE, continuous coarse
    /// coefficients, and real residual hyperplane logits whose signs are packed.
    #[must_use]
    pub fn q_projection_width_per_head(&self) -> usize {
        self.base
            .rope_dimensions()
            .saturating_add(self.base.key_rank())
            .saturating_add(self.residual_bits)
    }

    /// K output width per KV head has the same three lanes as Q.
    #[must_use]
    pub fn k_projection_width_per_kv_head(&self) -> usize {
        self.q_projection_width_per_head()
    }

    /// Whether the new Q/K representation is no wider than the original dense head.
    /// This is a structural bandwidth condition, not a latency claim.
    #[must_use]
    pub fn qk_does_not_widen_dense_head(&self) -> bool {
        self.q_projection_width_per_head() <= self.base.d_head()
    }
}

#[derive(Debug, Clone)]
pub struct ElasticMlaSlhaProjectionPlan {
    layers: Vec<ElasticMlaSlhaLayerWeights>,
}

impl ElasticMlaSlhaProjectionPlan {
    pub fn from_model(
        model: &SciAgentModel,
        bases: &[ElasticMlaBases],
        residual_bits: usize,
        group_dim: usize,
        seed: u64,
    ) -> Result<Self, ElasticMlaSlhaProjectionError> {
        if model.layers.len() != bases.len()
        {
            return Err(ElasticMlaSlhaProjectionError::LayerBasisCount {
                layers: model.layers.len(),
                bases: bases.len(),
            });
        }
        let mut layers = Vec::with_capacity(model.layers.len());
        for (index, (layer, basis)) in model.layers.iter().zip(bases).enumerate()
        {
            layers.push(ElasticMlaSlhaLayerWeights::from_attention(
                &layer.attn,
                basis,
                residual_bits,
                group_dim,
                derive_seed(seed, index as u64),
            )?);
        }
        Ok(Self { layers })
    }

    #[must_use]
    pub fn layers(&self) -> &[ElasticMlaSlhaLayerWeights] {
        &self.layers
    }
}

const fn derive_seed(seed: u64, index: u64) -> u64 {
    splitmix64(seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;
    use crate::elastic_mla_plan::{ElasticMlaBases, RopePairSelection};

    fn matmul_row(input: &[f32], weight: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; cols];
        for col in 0..cols
        {
            for row in 0..rows
            {
                output[col] += input[row] * weight[row * cols + col];
            }
        }
        output
    }

    #[test]
    fn absorbed_residual_logits_match_direct_sketch_projection() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let selection = RopePairSelection::high_frequency_prefix(attention.d_head, 4).unwrap();
        let nope_dimensions = selection.nope_dimensions();
        let key_rank = 2;
        let bases = ElasticMlaBases::coordinate_prefix(
            attention.n_kv_heads,
            selection.clone(),
            key_rank,
            4,
        )
        .unwrap();
        let residual_bits = 3;
        let seed = 0x534c_4841_454c_4153;
        let plan = ElasticMlaSlhaLayerWeights::from_attention(
            attention,
            &bases,
            residual_bits,
            2,
            seed,
        )
        .unwrap();
        assert!(plan.qk_does_not_widen_dense_head());

        let input: Vec<f32> = (0..attention.d_model)
            .map(|index| (index as f32 - 9.0) * 0.0234375)
            .collect();
        let q_dense = matmul_row(
            &input,
            &attention.w_q.weight.data,
            attention.d_model,
            attention.d_model,
        );
        let q_residual = matmul_row(
            &input,
            plan.q_residual_logits(),
            attention.d_model,
            attention.n_heads * residual_bits,
        );

        let kv_head = 0usize;
        let sketch = SlhaResidualSketch::from_orthonormal_coarse_basis(
            nope_dimensions,
            key_rank,
            residual_bits,
            derive_seed(seed, kv_head as u64),
            bases.key_head(kv_head),
        )
        .unwrap();
        let mut q_nope = vec![0.0f32; nope_dimensions];
        for (local, &channel) in selection.nope_channels().iter().enumerate()
        {
            q_nope[local] = q_dense[channel];
        }
        let mut direct = vec![0.0f32; residual_bits];
        sketch.project_into(&q_nope, &mut direct).unwrap();
        for bit in 0..residual_bits
        {
            let absorbed = q_residual[bit];
            assert!(
                (absorbed - direct[bit]).abs() <= 2.0e-6,
                "bit {bit}: absorbed={absorbed}, direct={} ",
                direct[bit]
            );
        }
    }

    #[test]
    fn same_inputs_produce_same_sketch_fingerprints() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let selection = RopePairSelection::high_frequency_prefix(attention.d_head, 4).unwrap();
        let bases = ElasticMlaBases::coordinate_prefix(
            attention.n_kv_heads,
            selection,
            2,
            4,
        )
        .unwrap();
        let first =
            ElasticMlaSlhaLayerWeights::from_attention(attention, &bases, 3, 2, 99).unwrap();
        let second =
            ElasticMlaSlhaLayerWeights::from_attention(attention, &bases, 3, 2, 99).unwrap();
        assert_eq!(first.sketch_fingerprints(), second.sketch_fingerprints());
        assert_eq!(first.q_residual_logits(), second.q_residual_logits());
        assert_eq!(first.k_residual_logits(), second.k_residual_logits());
    }
}