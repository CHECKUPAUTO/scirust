//! Storage planning for ElasticMLA × SLHA-style hybrid key scoring.
//!
//! This module does not serialize a new tile and does not duplicate SLHAv2's CUDA
//! engine. It answers the lower-level question required before choosing a physical
//! layout: how many score-bearing bytes does one ElasticMLA KV head need when the
//! NoPE key lane uses the parametric SLHA score contract and the value lane remains
//! reconstruction-free latent data?
//!
//! Explicit RoPE keys stay bf16 initially because they carry positional geometry.
//! NoPE keys use grouped signed INT4 plus an optional sign residual; values use a
//! packed 4-bit latent payload. Codec metadata and lifecycle indexes are deliberately
//! accounted separately by the future runtime layout rather than hidden inside a
//! padded per-head struct.

use core::fmt;

use scirust_core::nn::{SlhaScoreError, SlhaScoreProfile};

use crate::elastic_mla_plan::ElasticMlaLayerWeights;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElasticMlaSlhaError {
    ScoreProfile(SlhaScoreError),
    CoarseRankMismatch {
        profile_dims: usize,
        key_rank: usize,
    },
    ValueRankZero,
    Overflow,
}

impl fmt::Display for ElasticMlaSlhaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ScoreProfile(error) => write!(formatter, "{error}"),
            Self::CoarseRankMismatch {
                profile_dims,
                key_rank,
            } => write!(
                formatter,
                "SLHA coarse dimension {profile_dims} does not match ElasticMLA key rank {key_rank}"
            ),
            Self::ValueRankZero => write!(formatter, "ElasticMLA value rank must be non-zero"),
            Self::Overflow => write!(formatter, "ElasticMLA SLHA storage accounting overflow"),
        }
    }
}

impl std::error::Error for ElasticMlaSlhaError {}

impl From<SlhaScoreError> for ElasticMlaSlhaError {
    fn from(error: SlhaScoreError) -> Self {
        Self::ScoreProfile(error)
    }
}

/// Logical score/value payload for one ElasticMLA layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticMlaSlhaLayout {
    n_kv_heads: usize,
    d_head: usize,
    rope_dimensions: usize,
    key_rank: usize,
    value_rank: usize,
    score_profile: SlhaScoreProfile,
}

impl ElasticMlaSlhaLayout {
    /// Construct a layer layout with a caller-selected residual width.
    ///
    /// `residual_bits` is the width of the sign projection, not necessarily the
    /// number of discarded NoPE coordinates; a learned/random residual projector may
    /// map into a wider binary space just like SLHAv2's canonical 256-bit residual.
    pub fn from_layer(
        layer: &ElasticMlaLayerWeights,
        residual_bits: usize,
        group_dim: usize,
    ) -> Result<Self, ElasticMlaSlhaError> {
        if layer.value_rank() == 0
        {
            return Err(ElasticMlaSlhaError::ValueRankZero);
        }
        let score_profile = SlhaScoreProfile::new(layer.key_rank(), residual_bits, group_dim)?;
        Self::with_profile(layer, score_profile)
    }

    pub fn with_profile(
        layer: &ElasticMlaLayerWeights,
        score_profile: SlhaScoreProfile,
    ) -> Result<Self, ElasticMlaSlhaError> {
        if score_profile.coarse_dims() != layer.key_rank()
        {
            return Err(ElasticMlaSlhaError::CoarseRankMismatch {
                profile_dims: score_profile.coarse_dims(),
                key_rank: layer.key_rank(),
            });
        }
        Ok(Self {
            n_kv_heads: layer.n_kv_heads(),
            d_head: layer.d_head(),
            rope_dimensions: layer.rope_dimensions(),
            key_rank: layer.key_rank(),
            value_rank: layer.value_rank(),
            score_profile,
        })
    }

    #[must_use]
    pub const fn score_profile(self) -> SlhaScoreProfile {
        self.score_profile
    }

    /// Explicit positional K coordinates stay bf16 in the first implementation.
    #[must_use]
    pub fn rope_key_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        checked_mul3(
            self.n_kv_heads,
            self.rope_dimensions,
            core::mem::size_of::<u16>(),
        )
    }

    /// NoPE key bytes that participate in HOT scoring across all KV heads.
    #[must_use]
    pub fn hot_key_payload_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        checked_mul(self.n_kv_heads, self.score_profile.hot_payload_bytes())
    }

    /// NoPE key bytes after the sign-residual plane has been paged out.
    #[must_use]
    pub fn warm_key_payload_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        checked_mul(self.n_kv_heads, self.score_profile.warm_payload_bytes())
    }

    /// Packed INT4 value coefficients. Quantizer scales/format metadata are kept in
    /// side arrays and intentionally excluded from this score-bearing payload count.
    #[must_use]
    pub fn value_payload_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        checked_mul(self.n_kv_heads, self.value_rank.div_ceil(2))
    }

    #[must_use]
    pub fn hot_payload_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        self.rope_key_bytes_per_token()?
            .checked_add(self.hot_key_payload_bytes_per_token()?)
            .and_then(|bytes| bytes.checked_add(self.value_payload_bytes_per_token().ok()?))
            .ok_or(ElasticMlaSlhaError::Overflow)
    }

    #[must_use]
    pub fn warm_payload_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        self.rope_key_bytes_per_token()?
            .checked_add(self.warm_key_payload_bytes_per_token()?)
            .and_then(|bytes| bytes.checked_add(self.value_payload_bytes_per_token().ok()?))
            .ok_or(ElasticMlaSlhaError::Overflow)
    }

    /// Conventional bf16 GQA stores dense K and dense V for every KV head.
    #[must_use]
    pub fn dense_bf16_payload_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        checked_mul3(
            self.n_kv_heads,
            self.d_head
                .checked_mul(2)
                .ok_or(ElasticMlaSlhaError::Overflow)?,
            core::mem::size_of::<u16>(),
        )
    }

    /// Integer ratio numerator/denominator helper avoids presenting a rounded float
    /// as an exact storage claim. `compressed / dense` is returned.
    #[must_use]
    pub fn hot_to_dense_ratio(self) -> Result<(usize, usize), ElasticMlaSlhaError> {
        Ok((
            self.hot_payload_bytes_per_token()?,
            self.dense_bf16_payload_bytes_per_token()?,
        ))
    }

    #[must_use]
    pub fn warm_to_dense_ratio(self) -> Result<(usize, usize), ElasticMlaSlhaError> {
        Ok((
            self.warm_payload_bytes_per_token()?,
            self.dense_bf16_payload_bytes_per_token()?,
        ))
    }

    /// Side-array scale bytes if both K and V use one f32 global scale plus one
    /// u8 micro-scale per group and KV head. Lifecycle flags/positions remain runtime
    /// policy metadata and are not included here.
    #[must_use]
    pub fn int4_scale_metadata_bytes_per_token(self) -> Result<usize, ElasticMlaSlhaError> {
        let key_groups = self.score_profile.group_count();
        let value_groups = self.value_rank.div_ceil(self.score_profile.group_dim());
        let per_head = core::mem::size_of::<f32>()
            .checked_add(key_groups)
            .and_then(|bytes| bytes.checked_add(core::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_add(value_groups))
            .ok_or(ElasticMlaSlhaError::Overflow)?;
        checked_mul(self.n_kv_heads, per_head)
    }

    #[must_use]
    pub const fn key_rank(self) -> usize {
        self.key_rank
    }

    #[must_use]
    pub const fn value_rank(self) -> usize {
        self.value_rank
    }
}

fn checked_mul(left: usize, right: usize) -> Result<usize, ElasticMlaSlhaError> {
    left.checked_mul(right)
        .ok_or(ElasticMlaSlhaError::Overflow)
}

fn checked_mul3(a: usize, b: usize, c: usize) -> Result<usize, ElasticMlaSlhaError> {
    a.checked_mul(b)
        .and_then(|value| value.checked_mul(c))
        .ok_or(ElasticMlaSlhaError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;
    use crate::elastic_mla_plan::{ElasticMlaBases, ElasticMlaLayerWeights, RopePairSelection};
    use crate::model::SciAgentModel;

    fn reduced_layer() -> ElasticMlaLayerWeights {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let selection = RopePairSelection::high_frequency_prefix(attention.d_head, 4).unwrap();
        let bases = ElasticMlaBases::coordinate_prefix(
            attention.n_kv_heads,
            selection,
            4,
            4,
        )
        .unwrap();
        ElasticMlaLayerWeights::from_attention(attention, &bases).unwrap()
    }

    #[test]
    fn profile_must_match_elastic_key_rank() {
        let layer = reduced_layer();
        let wrong = SlhaScoreProfile::new(8, 64, 4).unwrap();
        assert!(matches!(
            ElasticMlaSlhaLayout::with_profile(&layer, wrong),
            Err(ElasticMlaSlhaError::CoarseRankMismatch { .. })
        ));
    }

    #[test]
    fn hot_and_warm_payloads_are_smaller_than_dense_for_reduced_layer() {
        let layer = reduced_layer();
        let layout = ElasticMlaSlhaLayout::from_layer(&layer, 64, 4).unwrap();
        let dense = layout.dense_bf16_payload_bytes_per_token().unwrap();
        let hot = layout.hot_payload_bytes_per_token().unwrap();
        let warm = layout.warm_payload_bytes_per_token().unwrap();
        assert!(hot < dense);
        assert!(warm < hot);
    }

    #[test]
    fn canonical_slhav2_profile_only_fits_matching_key_rank() {
        let layer = reduced_layer();
        assert!(matches!(
            ElasticMlaSlhaLayout::with_profile(&layer, SlhaScoreProfile::slhav2_128()),
            Err(ElasticMlaSlhaError::CoarseRankMismatch {
                profile_dims: 128,
                ..
            })
        ));
    }
}