//! Parametric SLHA-style hybrid latent score contract.
//!
//! The original SLHAv2 kernel uses a fixed 128-dimensional INT4 coarse vector and
//! a 256-bit sign residual. SciRust models do not all share that geometry: SCIAGENT
//! currently has 64-wide attention heads and ElasticMLA may reduce the NoPE key lane
//! further. Padding a smaller head into the SLHAv2 layout would waste bandwidth and
//! coupling two heads would change attention semantics.
//!
//! This module therefore captures the *score semantics* independently from one
//! serialized tile layout. [`SlhaScoreProfile::slhav2_128`] pins the original
//! geometry; [`SlhaScoreProfile::new`] permits model-specific geometries while
//! preserving the same grouped signed-INT4 + optional sign-residual equation.
//!
//! Residual widths need not be multiples of 64. Storage still uses `u64` words, but
//! unused high bits in the final word are masked out of the Hamming distance and out
//! of the declared `d_s` term. This lets a model store exactly the useful number of
//! binary residual directions instead of padding its mathematical score to a word.
//!
//! Scoring is allocation-free and fixed-order. Device implementations can use this
//! scalar path as their differential oracle.

use core::fmt;

/// Geometry validation or score-buffer mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlhaScoreError {
    ZeroDimension {
        field: &'static str,
    },
    Length {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        name: &'static str,
        index: usize,
    },
}

impl fmt::Display for SlhaScoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroDimension { field } => write!(formatter, "{field} must be non-zero"),
            Self::Length {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFinite { name, index } =>
            {
                write!(formatter, "{name} contains a non-finite scalar at {index}")
            },
        }
    }
}

impl std::error::Error for SlhaScoreError {}

/// Model-independent geometry for grouped signed-INT4 coarse scoring plus a
/// sign-Hamming residual correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlhaScoreProfile {
    coarse_dims: usize,
    residual_bits: usize,
    group_dim: usize,
}

impl SlhaScoreProfile {
    /// Build a validated score geometry.
    pub const fn new(
        coarse_dims: usize,
        residual_bits: usize,
        group_dim: usize,
    ) -> Result<Self, SlhaScoreError> {
        if coarse_dims == 0
        {
            return Err(SlhaScoreError::ZeroDimension {
                field: "coarse_dims",
            });
        }
        if residual_bits == 0
        {
            return Err(SlhaScoreError::ZeroDimension {
                field: "residual_bits",
            });
        }
        if group_dim == 0
        {
            return Err(SlhaScoreError::ZeroDimension { field: "group_dim" });
        }
        Ok(Self {
            coarse_dims,
            residual_bits,
            group_dim,
        })
    }

    /// Exact geometry of the current SLHAv2 score kernel.
    #[must_use]
    pub const fn slhav2_128() -> Self {
        Self {
            coarse_dims: 128,
            residual_bits: 256,
            group_dim: 16,
        }
    }

    #[must_use]
    pub const fn coarse_dims(self) -> usize {
        self.coarse_dims
    }

    #[must_use]
    pub const fn residual_bits(self) -> usize {
        self.residual_bits
    }

    #[must_use]
    pub const fn group_dim(self) -> usize {
        self.group_dim
    }

    /// Packed signed-INT4 bytes, two coarse coefficients per byte.
    #[must_use]
    pub const fn latent_bytes(self) -> usize {
        self.coarse_dims.div_ceil(2)
    }

    /// Number of storage words required for the declared residual bits.
    #[must_use]
    pub const fn residual_words(self) -> usize {
        self.residual_bits.div_ceil(64)
    }

    #[must_use]
    pub const fn residual_bytes(self) -> usize {
        self.residual_words() * core::mem::size_of::<u64>()
    }

    #[must_use]
    pub const fn group_count(self) -> usize {
        self.coarse_dims.div_ceil(self.group_dim)
    }

    /// Score-bearing bytes only: INT4 latent + sign residual. Serialization
    /// metadata is deliberately not included because different runtimes can keep
    /// scale/state metadata in structure-of-arrays form.
    #[must_use]
    pub const fn hot_payload_bytes(self) -> usize {
        self.latent_bytes() + self.residual_bytes()
    }

    /// WARM keeps the coarse latent and drops/bypasses the residual plane.
    #[must_use]
    pub const fn warm_payload_bytes(self) -> usize {
        self.latent_bytes()
    }

    /// Original SLHAv2 serialized tile size when and only when this profile has
    /// its canonical geometry. The remaining 32 bytes are SLHAv2 metadata.
    #[must_use]
    pub const fn slhav2_serialized_tile_bytes(self) -> Option<usize> {
        if self.coarse_dims == 128 && self.residual_bits == 256 && self.group_dim == 16
        {
            Some(128)
        }
        else
        {
            None
        }
    }

    /// Grouped signed-INT4 coarse score, matching SLHAv2's zero point and effective
    /// scale convention: `(nibble - 8) * scale * group_scale / 255`.
    pub fn score_warm_int4(
        self,
        query_coarse: &[f32],
        packed_latent: &[u8],
        scale: f32,
        group_scales: &[u8],
    ) -> Result<f32, SlhaScoreError> {
        self.validate_coarse_inputs(query_coarse, packed_latent, scale, group_scales)?;
        Ok(self.coarse_dot_int4(query_coarse, packed_latent, scale, group_scales))
    }

    /// HOT score = coarse score + sign-Hamming residual correction.
    #[allow(clippy::too_many_arguments)] // Mirrors the SLHA HOT score contract.
    pub fn score_hot_int4(
        self,
        query_coarse: &[f32],
        packed_latent: &[u8],
        scale: f32,
        group_scales: &[u8],
        query_sign: &[u64],
        residual_bitmap: &[u64],
        dynamic_lambda: f32,
    ) -> Result<f32, SlhaScoreError> {
        self.validate_coarse_inputs(query_coarse, packed_latent, scale, group_scales)?;
        require_length("query sign", query_sign.len(), self.residual_words())?;
        require_length(
            "residual bitmap",
            residual_bitmap.len(),
            self.residual_words(),
        )?;
        if !dynamic_lambda.is_finite()
        {
            return Err(SlhaScoreError::NonFinite {
                name: "dynamic_lambda",
                index: 0,
            });
        }

        let coarse = self.coarse_dot_int4(query_coarse, packed_latent, scale, group_scales);
        let hamming = self.hamming_valid_bits(query_sign, residual_bitmap);
        Ok(coarse + dynamic_lambda * (self.residual_bits as f32 - 2.0 * hamming as f32))
    }

    fn validate_coarse_inputs(
        self,
        query_coarse: &[f32],
        packed_latent: &[u8],
        scale: f32,
        group_scales: &[u8],
    ) -> Result<(), SlhaScoreError> {
        require_length("coarse query", query_coarse.len(), self.coarse_dims)?;
        require_length("packed latent", packed_latent.len(), self.latent_bytes())?;
        require_length("group scales", group_scales.len(), self.group_count())?;
        require_finite("coarse query", query_coarse)?;
        if !scale.is_finite()
        {
            return Err(SlhaScoreError::NonFinite {
                name: "scale",
                index: 0,
            });
        }
        Ok(())
    }

    fn coarse_dot_int4(
        self,
        query_coarse: &[f32],
        packed_latent: &[u8],
        scale: f32,
        group_scales: &[u8],
    ) -> f32 {
        let mut sum = 0.0f32;
        for dimension in 0..self.coarse_dims
        {
            let byte = packed_latent[dimension / 2];
            let nibble = if dimension.is_multiple_of(2)
            {
                byte & 0x0f
            }
            else
            {
                byte >> 4
            };
            let level = nibble as i32 - 8;
            let group = dimension / self.group_dim;
            let effective_scale = scale * group_scales[group] as f32 * (1.0 / 255.0);
            sum += query_coarse[dimension] * level as f32 * effective_scale;
        }
        sum
    }

    fn hamming_valid_bits(self, left: &[u64], right: &[u64]) -> u32 {
        let words = self.residual_words();
        let tail_bits = self.residual_bits % 64;
        let mut hamming = 0u32;
        for word in 0..words
        {
            let mut different = left[word] ^ right[word];
            if word + 1 == words && tail_bits != 0
            {
                different &= (1u64 << tail_bits) - 1;
            }
            hamming = hamming.saturating_add(different.count_ones());
        }
        hamming
    }
}

fn require_length(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SlhaScoreError> {
    if actual != expected
    {
        return Err(SlhaScoreError::Length {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), SlhaScoreError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite())
    {
        return Err(SlhaScoreError::NonFinite { name, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_slhav2_geometry_is_pinned() {
        let profile = SlhaScoreProfile::slhav2_128();
        assert_eq!(profile.coarse_dims(), 128);
        assert_eq!(profile.residual_bits(), 256);
        assert_eq!(profile.group_dim(), 16);
        assert_eq!(profile.latent_bytes(), 64);
        assert_eq!(profile.residual_words(), 4);
        assert_eq!(profile.residual_bytes(), 32);
        assert_eq!(profile.group_count(), 8);
        assert_eq!(profile.hot_payload_bytes(), 96);
        assert_eq!(profile.warm_payload_bytes(), 64);
        assert_eq!(profile.slhav2_serialized_tile_bytes(), Some(128));
    }

    #[test]
    fn model_specific_profile_does_not_inherit_slhav2_padding() {
        let profile = SlhaScoreProfile::new(24, 24, 8).unwrap();
        assert_eq!(profile.latent_bytes(), 12);
        assert_eq!(profile.residual_words(), 1);
        assert_eq!(profile.residual_bytes(), 8);
        assert_eq!(profile.group_count(), 3);
        assert_eq!(profile.hot_payload_bytes(), 20);
        assert_eq!(profile.slhav2_serialized_tile_bytes(), None);
    }

    #[test]
    fn warm_int4_uses_signed_zero_point_and_group_scale() {
        let profile = SlhaScoreProfile::new(4, 4, 2).unwrap();
        // nibbles: 8 -> 0, 9 -> +1, 7 -> -1, 10 -> +2.
        let packed = [0x98u8, 0xa7u8];
        let query = [1.0f32, 2.0, 3.0, 4.0];
        let scales = [255u8, 128u8];
        let actual = profile
            .score_warm_int4(&query, &packed, 2.0, &scales)
            .unwrap();
        let expected = 2.0 * 2.0 + (-3.0 + 8.0) * (2.0 * 128.0 / 255.0);
        assert!((actual - expected).abs() <= 1.0e-6);
    }

    #[test]
    fn hot_residual_matches_hamming_equation() {
        let profile = SlhaScoreProfile::new(2, 64, 2).unwrap();
        let packed = [0x88u8]; // zero coarse score.
        let query = [1.0f32, -2.0];
        let query_sign = [0xffff_0000_ffff_0000u64];
        let residual = [0xffff_ffff_0000_0000u64];
        let hamming = (query_sign[0] ^ residual[0]).count_ones();
        let actual = profile
            .score_hot_int4(&query, &packed, 1.0, &[255], &query_sign, &residual, 0.25)
            .unwrap();
        let expected = 0.25 * (64.0 - 2.0 * hamming as f32);
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn unused_tail_bits_do_not_change_compact_residual_score() {
        let profile = SlhaScoreProfile::new(2, 5, 2).unwrap();
        let packed = [0x88u8];
        let query = [0.0f32; 2];
        let left = [0b1_0101u64];
        let right_a = [0b0_0011u64];
        let right_b = [right_a[0] | (!0u64 << 5)];
        let a = profile
            .score_hot_int4(&query, &packed, 1.0, &[255], &left, &right_a, 0.5)
            .unwrap();
        let b = profile
            .score_hot_int4(&query, &packed, 1.0, &[255], &left, &right_b, 0.5)
            .unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn malformed_buffers_fail_closed() {
        let profile = SlhaScoreProfile::new(8, 17, 4).unwrap();
        assert!(matches!(
            profile.score_warm_int4(&[0.0; 7], &[0; 4], 1.0, &[255; 2]),
            Err(SlhaScoreError::Length {
                name: "coarse query",
                ..
            })
        ));
        assert_eq!(profile.residual_words(), 1);
    }
}
