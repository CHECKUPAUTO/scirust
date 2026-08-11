//! Stable explicit-SIMD kernels for `scirust_autodiff::DualPack<f32, W>`.
//!
//! The primal stays scalar while the fixed-width tangent array is processed in
//! AVX-512/AVX2/SSE2 or NEON chunks with a scalar tail. Multiplication uses
//! explicit multiply + add rather than FMA so this deterministic path preserves
//! the scalar operation structure.

use scirust_autodiff::DualPack;

/// Add two dual packs with explicit SIMD tangent processing and no allocation.
#[inline]
pub fn add_f32<const W: usize>(left: DualPack<f32, W>, right: DualPack<f32, W>) -> DualPack<f32, W> {
    let mut tangent = [0.0f32; W];
    let mut index = 0usize;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        if std::arch::is_x86_feature_detected!("avx512f")
        {
            while index + 16 <= W
            {
                let a = _mm512_loadu_ps(left.tangent.as_ptr().add(index));
                let b = _mm512_loadu_ps(right.tangent.as_ptr().add(index));
                _mm512_storeu_ps(tangent.as_mut_ptr().add(index), _mm512_add_ps(a, b));
                index += 16;
            }
        }
        else if std::arch::is_x86_feature_detected!("avx2")
        {
            while index + 8 <= W
            {
                let a = _mm256_loadu_ps(left.tangent.as_ptr().add(index));
                let b = _mm256_loadu_ps(right.tangent.as_ptr().add(index));
                _mm256_storeu_ps(tangent.as_mut_ptr().add(index), _mm256_add_ps(a, b));
                index += 8;
            }
        }
        else if std::arch::is_x86_feature_detected!("sse2")
        {
            while index + 4 <= W
            {
                let a = _mm_loadu_ps(left.tangent.as_ptr().add(index));
                let b = _mm_loadu_ps(right.tangent.as_ptr().add(index));
                _mm_storeu_ps(tangent.as_mut_ptr().add(index), _mm_add_ps(a, b));
                index += 4;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while index + 4 <= W
        {
            let a = vld1q_f32(left.tangent.as_ptr().add(index));
            let b = vld1q_f32(right.tangent.as_ptr().add(index));
            vst1q_f32(tangent.as_mut_ptr().add(index), vaddq_f32(a, b));
            index += 4;
        }
    }

    while index < W
    {
        tangent[index] = left.tangent[index] + right.tangent[index];
        index += 1;
    }

    DualPack::seeded(left.value + right.value, tangent)
}

/// Product rule with explicit SIMD tangent processing and no allocation:
/// `d(left*right) = right.value*dleft + left.value*dright`.
#[inline]
pub fn mul_f32<const W: usize>(left: DualPack<f32, W>, right: DualPack<f32, W>) -> DualPack<f32, W> {
    let mut tangent = [0.0f32; W];
    let mut index = 0usize;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        if std::arch::is_x86_feature_detected!("avx512f")
        {
            let left_value = _mm512_set1_ps(left.value);
            let right_value = _mm512_set1_ps(right.value);
            while index + 16 <= W
            {
                let dl = _mm512_loadu_ps(left.tangent.as_ptr().add(index));
                let dr = _mm512_loadu_ps(right.tangent.as_ptr().add(index));
                let result = _mm512_add_ps(
                    _mm512_mul_ps(right_value, dl),
                    _mm512_mul_ps(left_value, dr),
                );
                _mm512_storeu_ps(tangent.as_mut_ptr().add(index), result);
                index += 16;
            }
        }
        else if std::arch::is_x86_feature_detected!("avx2")
        {
            let left_value = _mm256_set1_ps(left.value);
            let right_value = _mm256_set1_ps(right.value);
            while index + 8 <= W
            {
                let dl = _mm256_loadu_ps(left.tangent.as_ptr().add(index));
                let dr = _mm256_loadu_ps(right.tangent.as_ptr().add(index));
                let result = _mm256_add_ps(
                    _mm256_mul_ps(right_value, dl),
                    _mm256_mul_ps(left_value, dr),
                );
                _mm256_storeu_ps(tangent.as_mut_ptr().add(index), result);
                index += 8;
            }
        }
        else if std::arch::is_x86_feature_detected!("sse2")
        {
            let left_value = _mm_set1_ps(left.value);
            let right_value = _mm_set1_ps(right.value);
            while index + 4 <= W
            {
                let dl = _mm_loadu_ps(left.tangent.as_ptr().add(index));
                let dr = _mm_loadu_ps(right.tangent.as_ptr().add(index));
                let result = _mm_add_ps(_mm_mul_ps(right_value, dl), _mm_mul_ps(left_value, dr));
                _mm_storeu_ps(tangent.as_mut_ptr().add(index), result);
                index += 4;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let left_value = vdupq_n_f32(left.value);
        let right_value = vdupq_n_f32(right.value);
        while index + 4 <= W
        {
            let dl = vld1q_f32(left.tangent.as_ptr().add(index));
            let dr = vld1q_f32(right.tangent.as_ptr().add(index));
            let result = vaddq_f32(vmulq_f32(right_value, dl), vmulq_f32(left_value, dr));
            vst1q_f32(tangent.as_mut_ptr().add(index), result);
            index += 4;
        }
    }

    while index < W
    {
        tangent[index] =
            right.value * left.tangent[index] + left.value * right.tangent[index];
        index += 1;
    }

    DualPack::seeded(left.value * right.value, tangent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded<const W: usize>(base: f32, slope: f32) -> DualPack<f32, W> {
        let mut tangent = [0.0f32; W];
        for (index, value) in tangent.iter_mut().enumerate()
        {
            *value = base + index as f32 * slope;
        }
        DualPack::seeded(base, tangent)
    }

    #[test]
    fn simd_add_matches_dual_pack_for_tail_width() {
        let left = seeded::<19>(0.25, 0.125);
        let right = seeded::<19>(-0.5, 0.0625);
        assert_eq!(add_f32(left, right), left + right);
    }

    #[test]
    fn simd_product_rule_matches_dual_pack_for_common_widths() {
        let left8 = seeded::<8>(0.5, 0.125);
        let right8 = seeded::<8>(-0.25, 0.25);
        assert_eq!(mul_f32(left8, right8), left8 * right8);

        let left16 = seeded::<16>(0.75, 0.0625);
        let right16 = seeded::<16>(1.25, -0.03125);
        assert_eq!(mul_f32(left16, right16), left16 * right16);
    }

    #[test]
    fn zero_width_is_valid_and_allocation_free() {
        let left = DualPack::<f32, 0>::constant(2.0);
        let right = DualPack::<f32, 0>::constant(3.0);
        assert_eq!(mul_f32(left, right), DualPack::constant(6.0));
    }
}
