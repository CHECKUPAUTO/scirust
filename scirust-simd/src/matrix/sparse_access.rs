//! Sparse indexed-access primitives for SIMD planning.
//!
//! Gather and scatter are modeled separately because they have different
//! correctness constraints. Reads can be vectorized whenever every index is in
//! bounds. Writes/accumulations need an explicit uniqueness/conflict contract
//! before a hardware scatter path can preserve deterministic semantics.

/// Read strategy actually used by [`gather_f32_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseReadStrategy {
    ScalarIndexed,
    Avx2Gather,
}

/// Write strategies exposed to planners.
///
/// Only `ScalarDeterministic` is executable in this slice. The AVX-512 variant
/// is an explicit candidate contract for a later path that must first prove
/// target-index uniqueness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseWriteStrategy {
    ScalarDeterministic,
    Avx512ScatterUnique,
}

/// Indexed-access validation error. Validation completes before `out` is touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseAccessError {
    LengthMismatch {
        indices: usize,
        output: usize,
    },
    IndexOutOfBounds {
        position: usize,
        index: usize,
        source_len: usize,
    },
}

impl core::fmt::Display for SparseAccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::LengthMismatch { indices, output } => write!(
                f,
                "sparse gather length mismatch: {indices} indices for {output} output elements"
            ),
            Self::IndexOutOfBounds {
                position,
                index,
                source_len,
            } => write!(
                f,
                "sparse gather index {index} at position {position} is outside source length {source_len}"
            ),
        }
    }
}

impl std::error::Error for SparseAccessError {}

/// Return the preferred currently-executable sparse read strategy.
pub fn preferred_sparse_read_strategy() -> SparseReadStrategy {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
    {
        return SparseReadStrategy::Avx2Gather;
    }

    SparseReadStrategy::ScalarIndexed
}

/// Gather arbitrary `f32` elements from `source` into caller-owned `out`.
///
/// The function performs a full bounds-validation pass before writing output.
/// On x86_64/AVX2 it uses `_mm256_i32gather_ps` for chunks whose indices fit
/// the hardware's signed 32-bit index operand. Other chunks and architectures
/// use scalar indexed loads. No heap allocation occurs.
pub fn gather_f32_into(
    source: &[f32],
    indices: &[usize],
    out: &mut [f32],
) -> Result<SparseReadStrategy, SparseAccessError> {
    if indices.len() != out.len()
    {
        return Err(SparseAccessError::LengthMismatch {
            indices: indices.len(),
            output: out.len(),
        });
    }
    for (position, &index) in indices.iter().enumerate()
    {
        if index >= source.len()
        {
            return Err(SparseAccessError::IndexOutOfBounds {
                position,
                index,
                source_len: source.len(),
            });
        }
    }

    let strategy = preferred_sparse_read_strategy();
    let mut position = 0usize;

    #[cfg(target_arch = "x86_64")]
    if strategy == SparseReadStrategy::Avx2Gather
    {
        // SAFETY: AVX2 was detected above. All indices were bounds-checked before
        // this block. `_mm256_i32gather_ps` addresses base + index*4; a chunk is
        // vectorized only when every usize index is representable as i32.
        unsafe {
            use core::arch::x86_64::*;
            while position + 8 <= indices.len()
            {
                let chunk = &indices[position..position + 8];
                if chunk.iter().all(|&index| i32::try_from(index).is_ok())
                {
                    let offsets = _mm256_set_epi32(
                        chunk[7] as i32,
                        chunk[6] as i32,
                        chunk[5] as i32,
                        chunk[4] as i32,
                        chunk[3] as i32,
                        chunk[2] as i32,
                        chunk[1] as i32,
                        chunk[0] as i32,
                    );
                    let gathered = _mm256_i32gather_ps(source.as_ptr(), offsets, 4);
                    _mm256_storeu_ps(out.as_mut_ptr().add(position), gathered);
                    position += 8;
                }
                else
                {
                    break;
                }
            }
        }
    }

    while position < indices.len()
    {
        out[position] = source[indices[position]];
        position += 1;
    }

    Ok(strategy)
}

/// Deterministic sparse write/accumulate baseline.
///
/// Duplicate destination indices are legal and are applied in the exact caller
/// order. This semantics is the reference contract that any future hardware
/// scatter implementation must preserve or explicitly require uniqueness to
/// avoid ambiguity.
pub fn scatter_add_f32_deterministic(
    indices: &[usize],
    values: &[f32],
    out: &mut [f32],
) -> Result<(), SparseAccessError> {
    if indices.len() != values.len()
    {
        return Err(SparseAccessError::LengthMismatch {
            indices: indices.len(),
            output: values.len(),
        });
    }
    for (position, &index) in indices.iter().enumerate()
    {
        if index >= out.len()
        {
            return Err(SparseAccessError::IndexOutOfBounds {
                position,
                index,
                source_len: out.len(),
            });
        }
    }
    for (&index, &value) in indices.iter().zip(values)
    {
        out[index] += value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_matches_scalar_reference_for_irregular_indices() {
        let source: Vec<f32> = (0..128).map(|index| index as f32 * 0.25 - 3.0).collect();
        let indices = [17, 2, 91, 4, 63, 8, 127, 0, 33, 7, 12];
        let expected: Vec<f32> = indices.iter().map(|&index| source[index]).collect();
        let mut out = [0.0f32; 11];
        let _strategy = gather_f32_into(&source, &indices, &mut out).unwrap();
        assert_eq!(out.as_slice(), expected.as_slice());
    }

    #[test]
    fn invalid_gather_leaves_output_untouched() {
        let source = [1.0f32, 2.0, 3.0];
        let indices = [0usize, 3, 1];
        let mut out = [9.0f32; 3];
        let error = gather_f32_into(&source, &indices, &mut out).unwrap_err();
        assert_eq!(
            error,
            SparseAccessError::IndexOutOfBounds {
                position: 1,
                index: 3,
                source_len: 3,
            }
        );
        assert_eq!(out, [9.0; 3]);
    }

    #[test]
    fn deterministic_scatter_preserves_duplicate_order() {
        let indices = [1usize, 1, 0, 1];
        let values = [1.0f32, 2.0, 4.0, -0.5];
        let mut out = [0.0f32; 3];
        scatter_add_f32_deterministic(&indices, &values, &mut out).unwrap();
        assert_eq!(out, [4.0, 2.5, 0.0]);
    }

    #[test]
    fn write_strategy_keeps_unique_scatter_separate_from_reference_semantics() {
        assert_ne!(
            SparseWriteStrategy::ScalarDeterministic,
            SparseWriteStrategy::Avx512ScatterUnique
        );
    }
}
