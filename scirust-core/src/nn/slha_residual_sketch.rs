//! Deterministic sign-LSH residual sketch orthogonal to a learned coarse basis.
//!
//! SLHA-style scoring augments a continuous coarse projection with sign bits from the
//! key residual outside that coarse subspace. Reconstructing the dense residual at
//! decode time would defeat ElasticMLA's reconstruction-free design. Instead, this
//! module constructs deterministic random hyperplanes that are themselves orthogonal
//! to the coarse basis.
//!
//! If `U` is the orthonormal coarse basis and `r` is one generated hyperplane with
//! `U^T r = 0`, then for any vector `x`:
//!
//! ```text
//! r^T x = r^T (x - U U^T x)
//! ```
//!
//! The residual sign can therefore be produced directly by a projection absorbed
//! into Q/K weights. No dense reconstruction or subtraction is required per token.

use core::fmt;

const ORTHONORMAL_TOLERANCE: f32 = 2.0e-4;
const MIN_NORM_SQUARED: f32 = 1.0e-10;
const MAX_CANDIDATE_ATTEMPTS: usize = 32;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, PartialEq)]
pub enum SlhaResidualSketchError {
    ZeroDimension(&'static str),
    RankTooLarge {
        rank: usize,
        dimension: usize,
    },
    NoResidualSubspace,
    BasisLength {
        expected: usize,
        actual: usize,
    },
    NonFiniteBasis {
        index: usize,
    },
    CoarseBasisNotOrthonormal {
        left: usize,
        right: usize,
        value: f32,
    },
    ProjectionLength {
        expected: usize,
        actual: usize,
    },
    OutputLength {
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        index: usize,
    },
    DegenerateHyperplane {
        bit: usize,
    },
    Overflow,
}

impl fmt::Display for SlhaResidualSketchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroDimension(field) => write!(formatter, "{field} must be non-zero"),
            Self::RankTooLarge { rank, dimension } => write!(
                formatter,
                "coarse rank {rank} exceeds residual-space dimension {dimension}"
            ),
            Self::NoResidualSubspace => write!(formatter, "coarse basis spans the full space"),
            Self::BasisLength { expected, actual } => write!(
                formatter,
                "coarse basis length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteBasis { index } => write!(
                formatter,
                "coarse basis contains a non-finite scalar at {index}"
            ),
            Self::CoarseBasisNotOrthonormal { left, right, value } => write!(
                formatter,
                "coarse basis columns {left}/{right} violate orthonormality: {value}"
            ),
            Self::ProjectionLength { expected, actual } => write!(
                formatter,
                "residual projection length mismatch: expected {expected}, got {actual}"
            ),
            Self::OutputLength { expected, actual } => write!(
                formatter,
                "residual output length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteInput { index } => write!(
                formatter,
                "residual sketch input contains a non-finite scalar at {index}"
            ),
            Self::DegenerateHyperplane { bit } => write!(
                formatter,
                "could not construct residual hyperplane for bit {bit}"
            ),
            Self::Overflow => write!(formatter, "residual sketch size overflow"),
        }
    }
}

impl std::error::Error for SlhaResidualSketchError {}

/// Immutable deterministic residual-sign projection.
///
/// `coarse_basis` and `projection` are row-major `[dimension, columns]` matrices.
#[derive(Debug, Clone)]
pub struct SlhaResidualSketch {
    dimension: usize,
    coarse_rank: usize,
    residual_bits: usize,
    seed: u64,
    projection: Vec<f32>,
    fingerprint: u64,
}

impl SlhaResidualSketch {
    pub fn from_orthonormal_coarse_basis(
        dimension: usize,
        coarse_rank: usize,
        residual_bits: usize,
        seed: u64,
        coarse_basis: &[f32],
    ) -> Result<Self, SlhaResidualSketchError> {
        if dimension == 0
        {
            return Err(SlhaResidualSketchError::ZeroDimension("dimension"));
        }
        if coarse_rank == 0
        {
            return Err(SlhaResidualSketchError::ZeroDimension("coarse_rank"));
        }
        if residual_bits == 0
        {
            return Err(SlhaResidualSketchError::ZeroDimension("residual_bits"));
        }
        if coarse_rank > dimension
        {
            return Err(SlhaResidualSketchError::RankTooLarge {
                rank: coarse_rank,
                dimension,
            });
        }
        if coarse_rank == dimension
        {
            return Err(SlhaResidualSketchError::NoResidualSubspace);
        }
        let expected = dimension
            .checked_mul(coarse_rank)
            .ok_or(SlhaResidualSketchError::Overflow)?;
        if coarse_basis.len() != expected
        {
            return Err(SlhaResidualSketchError::BasisLength {
                expected,
                actual: coarse_basis.len(),
            });
        }
        if let Some(index) = coarse_basis.iter().position(|value| !value.is_finite())
        {
            return Err(SlhaResidualSketchError::NonFiniteBasis { index });
        }
        validate_orthonormal(dimension, coarse_rank, coarse_basis)?;

        let projection_len = dimension
            .checked_mul(residual_bits)
            .ok_or(SlhaResidualSketchError::Overflow)?;
        let mut projection = vec![0.0f32; projection_len];
        let mut candidate = vec![0.0f32; dimension];

        for bit in 0..residual_bits
        {
            let mut accepted = false;
            for attempt in 0..MAX_CANDIDATE_ATTEMPTS
            {
                fill_rademacher(&mut candidate, seed, bit, attempt);
                remove_coarse_component(&mut candidate, dimension, coarse_rank, coarse_basis);
                // A second fixed-order pass limits loss of orthogonality in f32.
                remove_coarse_component(&mut candidate, dimension, coarse_rank, coarse_basis);
                let norm_squared = candidate.iter().map(|value| value * value).sum::<f32>();
                if norm_squared <= MIN_NORM_SQUARED || !norm_squared.is_finite()
                {
                    continue;
                }
                let inverse_norm = norm_squared.sqrt().recip();
                for row in 0..dimension
                {
                    projection[row * residual_bits + bit] = candidate[row] * inverse_norm;
                }
                accepted = true;
                break;
            }
            if !accepted
            {
                return Err(SlhaResidualSketchError::DegenerateHyperplane { bit });
            }
        }

        let fingerprint = projection_fingerprint(&projection);
        Ok(Self {
            dimension,
            coarse_rank,
            residual_bits,
            seed,
            projection,
            fingerprint,
        })
    }

    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub const fn coarse_rank(&self) -> usize {
        self.coarse_rank
    }

    #[must_use]
    pub const fn residual_bits(&self) -> usize {
        self.residual_bits
    }

    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Row-major `[dimension, residual_bits]` projection matrix. This matrix can be
    /// multiplied into a model's existing Q/K projection weights offline.
    #[must_use]
    pub fn projection(&self) -> &[f32] {
        &self.projection
    }

    /// Compute real-valued residual-hyperplane projections without allocation.
    pub fn project_into(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), SlhaResidualSketchError> {
        self.validate_input(input)?;
        if output.len() != self.residual_bits
        {
            return Err(SlhaResidualSketchError::OutputLength {
                expected: self.residual_bits,
                actual: output.len(),
            });
        }
        output.fill(0.0);
        for row in 0..self.dimension
        {
            let input_value = input[row];
            let offset = row * self.residual_bits;
            for bit in 0..self.residual_bits
            {
                output[bit] += input_value * self.projection[offset + bit];
            }
        }
        Ok(())
    }

    /// Pack the signs of the residual projections. Bit `1` means a negative
    /// projection, matching the sign convention used by SLHAv2.
    pub fn sign_bits_into(
        &self,
        input: &[f32],
        output_words: &mut [u64],
    ) -> Result<(), SlhaResidualSketchError> {
        self.validate_input(input)?;
        let expected_words = self.residual_bits.div_ceil(64);
        if output_words.len() != expected_words
        {
            return Err(SlhaResidualSketchError::OutputLength {
                expected: expected_words,
                actual: output_words.len(),
            });
        }
        output_words.fill(0);
        for bit in 0..self.residual_bits
        {
            let mut projection = 0.0f32;
            for row in 0..self.dimension
            {
                projection += input[row] * self.projection[row * self.residual_bits + bit];
            }
            if projection < 0.0
            {
                output_words[bit / 64] |= 1u64 << (bit % 64);
            }
        }
        Ok(())
    }

    fn validate_input(&self, input: &[f32]) -> Result<(), SlhaResidualSketchError> {
        if input.len() != self.dimension
        {
            return Err(SlhaResidualSketchError::ProjectionLength {
                expected: self.dimension,
                actual: input.len(),
            });
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite())
        {
            return Err(SlhaResidualSketchError::NonFiniteInput { index });
        }
        Ok(())
    }
}

fn validate_orthonormal(
    dimension: usize,
    rank: usize,
    basis: &[f32],
) -> Result<(), SlhaResidualSketchError> {
    for left in 0..rank
    {
        for right in 0..=left
        {
            let mut dot = 0.0f32;
            for row in 0..dimension
            {
                dot += basis[row * rank + left] * basis[row * rank + right];
            }
            let expected = if left == right { 1.0 } else { 0.0 };
            if (dot - expected).abs() > ORTHONORMAL_TOLERANCE
            {
                return Err(SlhaResidualSketchError::CoarseBasisNotOrthonormal {
                    left,
                    right,
                    value: dot,
                });
            }
        }
    }
    Ok(())
}

fn remove_coarse_component(
    vector: &mut [f32],
    dimension: usize,
    rank: usize,
    coarse_basis: &[f32],
) {
    for column in 0..rank
    {
        let mut dot = 0.0f32;
        for row in 0..dimension
        {
            dot += vector[row] * coarse_basis[row * rank + column];
        }
        for row in 0..dimension
        {
            vector[row] -= dot * coarse_basis[row * rank + column];
        }
    }
}

fn fill_rademacher(output: &mut [f32], seed: u64, bit: usize, attempt: usize) {
    let bit_key = (bit as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let attempt_key = (attempt as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    for (row, value) in output.iter_mut().enumerate()
    {
        let row_key = (row as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        let mixed = splitmix64(seed ^ bit_key ^ attempt_key ^ row_key);
        *value = if mixed & 1 == 0 { -1.0 } else { 1.0 };
    }
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn projection_fingerprint(values: &[f32]) -> u64 {
    let mut fingerprint = FNV_OFFSET;
    for value in values
    {
        fingerprint ^= u64::from(value.to_bits());
        fingerprint = fingerprint.wrapping_mul(FNV_PRIME);
    }
    fingerprint
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
        let mut basis = vec![0.0f32; dimension * rank];
        for column in 0..rank
        {
            basis[column * rank + column] = 1.0;
        }
        basis
    }

    #[test]
    fn sketch_is_exactly_replayable_from_seed_and_basis() {
        let basis = identity_prefix(8, 3);
        let first = SlhaResidualSketch::from_orthonormal_coarse_basis(8, 3, 7, 42, &basis).unwrap();
        let second = SlhaResidualSketch::from_orthonormal_coarse_basis(8, 3, 7, 42, &basis).unwrap();
        assert_eq!(first.projection(), second.projection());
        assert_eq!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn every_hyperplane_is_orthogonal_to_coarse_basis() {
        let basis = identity_prefix(12, 4);
        let sketch =
            SlhaResidualSketch::from_orthonormal_coarse_basis(12, 4, 13, 7, &basis).unwrap();
        for bit in 0..sketch.residual_bits()
        {
            for coarse in 0..4
            {
                let mut dot = 0.0f32;
                for row in 0..12
                {
                    dot += basis[row * 4 + coarse]
                        * sketch.projection()[row * sketch.residual_bits() + bit];
                }
                assert!(dot.abs() <= 1.0e-5, "bit={bit} coarse={coarse} dot={dot}");
            }
        }
    }

    #[test]
    fn sketch_of_vector_equals_sketch_of_its_coarse_residual() {
        let dimension = 8;
        let rank = 3;
        let basis = identity_prefix(dimension, rank);
        let sketch = SlhaResidualSketch::from_orthonormal_coarse_basis(
            dimension,
            rank,
            11,
            0x534c_4841,
            &basis,
        )
        .unwrap();
        let input = [1.0f32, -2.0, 3.0, 0.5, -0.25, 4.0, -5.0, 2.5];
        let mut residual = input;
        for index in 0..rank
        {
            residual[index] = 0.0;
        }
        let mut input_bits = [0u64; 1];
        let mut residual_bits = [0u64; 1];
        sketch.sign_bits_into(&input, &mut input_bits).unwrap();
        sketch
            .sign_bits_into(&residual, &mut residual_bits)
            .unwrap();
        assert_eq!(input_bits, residual_bits);
    }

    #[test]
    fn non_orthonormal_basis_fails_closed() {
        let basis = vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            SlhaResidualSketch::from_orthonormal_coarse_basis(3, 2, 4, 1, &basis),
            Err(SlhaResidualSketchError::CoarseBasisNotOrthonormal { .. })
        ));
    }

    #[test]
    fn sign_output_uses_exact_declared_bit_count() {
        let basis = identity_prefix(6, 2);
        let sketch = SlhaResidualSketch::from_orthonormal_coarse_basis(6, 2, 65, 9, &basis).unwrap();
        let input = [1.0f32, 2.0, -1.0, 0.5, -0.25, 3.0];
        let mut words = [0u64; 2];
        sketch.sign_bits_into(&input, &mut words).unwrap();
        assert_eq!(words[1] & !1u64, 0);
    }
}