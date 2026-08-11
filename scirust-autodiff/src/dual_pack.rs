//! Static multi-direction forward-mode automatic differentiation.
//!
//! [`DualPack`] carries one primal scalar and `W` tangent directions inline in
//! the value. `W` is a const generic, so evaluation needs no heap allocation and
//! the compiler sees a fixed-width contiguous tangent array that later SIMD
//! kernels can specialize.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// Minimal floating-point surface required by [`DualPack`].
///
/// Kept local and dependency-free so `scirust-autodiff` does not pull a numeric
/// trait crate into its production graph.
pub trait DualPackScalar:
    Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_i32(value: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powi(self, exponent: i32) -> Self;
}

impl DualPackScalar for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn powi(self, exponent: i32) -> Self {
        self.powi(exponent)
    }
}

impl DualPackScalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn powi(self, exponent: i32) -> Self {
        self.powi(exponent)
    }
}

/// A primal scalar with `W` simultaneously propagated tangent directions.
///
/// This is the static forward-mode building block for small dense gradients and
/// compressed sparse Jacobians. Unlike a tape, all derivative storage is inline
/// and no graph or heap node is created during arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DualPack<T: DualPackScalar, const W: usize> {
    pub value: T,
    pub tangent: [T; W],
}

impl<T: DualPackScalar, const W: usize> DualPack<T, W> {
    /// Constant primal with all tangent directions zeroed.
    #[inline]
    pub fn constant(value: T) -> Self {
        Self {
            value,
            tangent: [T::zero(); W],
        }
    }

    /// Independent variable seeded into one tangent lane.
    ///
    /// # Panics
    /// Panics when `lane >= W`.
    #[inline]
    pub fn variable(value: T, lane: usize) -> Self {
        assert!(
            lane < W,
            "DualPack variable lane {lane} is outside width {W}"
        );
        let mut tangent = [T::zero(); W];
        tangent[lane] = T::one();
        Self { value, tangent }
    }

    /// Construct directly from a primal value and caller-supplied seed vector.
    #[inline]
    pub const fn seeded(value: T, tangent: [T; W]) -> Self {
        Self { value, tangent }
    }

    /// Number of tangent lanes carried by this type.
    #[inline]
    pub const fn width() -> usize {
        W
    }

    /// Apply `local_derivative * tangent` lane-wise while preserving the existing
    /// scalar `Dual` invariant that a zero seed contributes exactly zero even when
    /// the local derivative is non-finite at a domain edge.
    #[inline]
    fn chain(self, output_value: T, local_derivative: T) -> Self {
        let mut tangent = [T::zero(); W];
        for (out, seed) in tangent.iter_mut().zip(self.tangent)
        {
            *out = if seed == T::zero()
            {
                T::zero()
            }
            else
            {
                local_derivative * seed
            };
        }
        Self {
            value: output_value,
            tangent,
        }
    }

    #[inline]
    pub fn sin(self) -> Self {
        self.chain(self.value.sin(), self.value.cos())
    }

    #[inline]
    pub fn cos(self) -> Self {
        self.chain(self.value.cos(), -self.value.sin())
    }

    #[inline]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        self.chain(value, value)
    }

    #[inline]
    pub fn ln(self) -> Self {
        self.chain(self.value.ln(), T::one() / self.value)
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let two = T::one() + T::one();
        self.chain(value, T::one() / (two * value))
    }

    #[inline]
    pub fn powi(self, exponent: i32) -> Self {
        let value = self.value.powi(exponent);
        if exponent == 0
        {
            return Self::constant(value);
        }
        let factor = T::from_i32(exponent) * self.value.powi(exponent - 1);
        self.chain(value, factor)
    }
}

impl<const W: usize> DualPack<f32, W> {
    /// Add two packs while processing tangent lanes with explicit architecture
    /// SIMD where available. The scalar tail preserves support for arbitrary `W`.
    #[inline]
    pub fn simd_add(self, rhs: Self) -> Self {
        let mut tangent = [0.0f32; W];
        let mut index = 0usize;

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        unsafe {
            use core::arch::x86_64::*;
            if std::arch::is_x86_feature_detected!("avx512f")
            {
                while index + 16 <= W
                {
                    let a = _mm512_loadu_ps(self.tangent.as_ptr().add(index));
                    let b = _mm512_loadu_ps(rhs.tangent.as_ptr().add(index));
                    _mm512_storeu_ps(tangent.as_mut_ptr().add(index), _mm512_add_ps(a, b));
                    index += 16;
                }
            }
            else if std::arch::is_x86_feature_detected!("avx2")
            {
                while index + 8 <= W
                {
                    let a = _mm256_loadu_ps(self.tangent.as_ptr().add(index));
                    let b = _mm256_loadu_ps(rhs.tangent.as_ptr().add(index));
                    _mm256_storeu_ps(tangent.as_mut_ptr().add(index), _mm256_add_ps(a, b));
                    index += 8;
                }
            }
            else if std::arch::is_x86_feature_detected!("sse2")
            {
                while index + 4 <= W
                {
                    let a = _mm_loadu_ps(self.tangent.as_ptr().add(index));
                    let b = _mm_loadu_ps(rhs.tangent.as_ptr().add(index));
                    _mm_storeu_ps(tangent.as_mut_ptr().add(index), _mm_add_ps(a, b));
                    index += 4;
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", not(miri)))]
        unsafe {
            use core::arch::aarch64::*;
            while index + 4 <= W
            {
                let a = vld1q_f32(self.tangent.as_ptr().add(index));
                let b = vld1q_f32(rhs.tangent.as_ptr().add(index));
                vst1q_f32(tangent.as_mut_ptr().add(index), vaddq_f32(a, b));
                index += 4;
            }
        }

        while index < W
        {
            tangent[index] = self.tangent[index] + rhs.tangent[index];
            index += 1;
        }

        Self::seeded(self.value + rhs.value, tangent)
    }

    /// Apply the product rule with explicit SIMD tangent processing. Multiply
    /// and add remain separate so this path does not silently alter rounding via
    /// FMA relative to the scalar oracle.
    #[inline]
    pub fn simd_mul(self, rhs: Self) -> Self {
        let mut tangent = [0.0f32; W];
        let mut index = 0usize;

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        unsafe {
            use core::arch::x86_64::*;
            if std::arch::is_x86_feature_detected!("avx512f")
            {
                let left_value = _mm512_set1_ps(self.value);
                let right_value = _mm512_set1_ps(rhs.value);
                while index + 16 <= W
                {
                    let dl = _mm512_loadu_ps(self.tangent.as_ptr().add(index));
                    let dr = _mm512_loadu_ps(rhs.tangent.as_ptr().add(index));
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
                let left_value = _mm256_set1_ps(self.value);
                let right_value = _mm256_set1_ps(rhs.value);
                while index + 8 <= W
                {
                    let dl = _mm256_loadu_ps(self.tangent.as_ptr().add(index));
                    let dr = _mm256_loadu_ps(rhs.tangent.as_ptr().add(index));
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
                let left_value = _mm_set1_ps(self.value);
                let right_value = _mm_set1_ps(rhs.value);
                while index + 4 <= W
                {
                    let dl = _mm_loadu_ps(self.tangent.as_ptr().add(index));
                    let dr = _mm_loadu_ps(rhs.tangent.as_ptr().add(index));
                    let result = _mm_add_ps(_mm_mul_ps(right_value, dl), _mm_mul_ps(left_value, dr));
                    _mm_storeu_ps(tangent.as_mut_ptr().add(index), result);
                    index += 4;
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", not(miri)))]
        unsafe {
            use core::arch::aarch64::*;
            let left_value = vdupq_n_f32(self.value);
            let right_value = vdupq_n_f32(rhs.value);
            while index + 4 <= W
            {
                let dl = vld1q_f32(self.tangent.as_ptr().add(index));
                let dr = vld1q_f32(rhs.tangent.as_ptr().add(index));
                let result = vaddq_f32(vmulq_f32(right_value, dl), vmulq_f32(left_value, dr));
                vst1q_f32(tangent.as_mut_ptr().add(index), result);
                index += 4;
            }
        }

        while index < W
        {
            tangent[index] = rhs.value * self.tangent[index] + self.value * rhs.tangent[index];
            index += 1;
        }

        Self::seeded(self.value * rhs.value, tangent)
    }
}

impl<T: DualPackScalar, const W: usize> Add for DualPack<T, W> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut tangent = [T::zero(); W];
        for (index, out) in tangent.iter_mut().enumerate()
        {
            *out = self.tangent[index] + rhs.tangent[index];
        }
        Self {
            value: self.value + rhs.value,
            tangent,
        }
    }
}

impl<T: DualPackScalar, const W: usize> Sub for DualPack<T, W> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut tangent = [T::zero(); W];
        for (index, out) in tangent.iter_mut().enumerate()
        {
            *out = self.tangent[index] - rhs.tangent[index];
        }
        Self {
            value: self.value - rhs.value,
            tangent,
        }
    }
}

impl<T: DualPackScalar, const W: usize> Mul for DualPack<T, W> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut tangent = [T::zero(); W];
        for (index, out) in tangent.iter_mut().enumerate()
        {
            *out = rhs.value * self.tangent[index] + self.value * rhs.tangent[index];
        }
        Self {
            value: self.value * rhs.value,
            tangent,
        }
    }
}

impl<T: DualPackScalar, const W: usize> Div for DualPack<T, W> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.value * rhs.value;
        let mut tangent = [T::zero(); W];
        for (index, out) in tangent.iter_mut().enumerate()
        {
            let left_seed = self.tangent[index];
            let right_seed = rhs.tangent[index];
            let left = if left_seed == T::zero()
            {
                T::zero()
            }
            else
            {
                (T::one() / rhs.value) * left_seed
            };
            let right = if right_seed == T::zero()
            {
                T::zero()
            }
            else
            {
                (self.value / denominator) * right_seed
            };
            *out = left - right;
        }
        Self {
            value: self.value / rhs.value,
            tangent,
        }
    }
}

impl<T: DualPackScalar, const W: usize> Neg for DualPack<T, W> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let mut tangent = [T::zero(); W];
        for (out, seed) in tangent.iter_mut().zip(self.tangent)
        {
            *out = -seed;
        }
        Self {
            value: -self.value,
            tangent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_variables_share_one_forward_evaluation() {
        let x = DualPack::<f64, 2>::variable(2.0, 0);
        let y = DualPack::<f64, 2>::variable(3.0, 1);
        let result = x.powi(2) * y + x.sin();

        assert!((result.value - (12.0 + 2.0f64.sin())).abs() < 1e-12);
        assert!((result.tangent[0] - (12.0 + 2.0f64.cos())).abs() < 1e-12);
        assert!((result.tangent[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn supports_f32_and_fixed_compile_time_width() {
        let x = DualPack::<f32, 8>::variable(1.5, 6);
        let result = x * x;
        assert_eq!(DualPack::<f32, 8>::width(), 8);
        assert_eq!(result.value, 2.25);
        assert_eq!(result.tangent[6], 3.0);
        for lane in 0..8
        {
            if lane != 6
            {
                assert_eq!(result.tangent[lane], 0.0);
            }
        }
    }

    #[test]
    fn constant_domain_edge_does_not_poison_unrelated_lane() {
        let x = DualPack::<f64, 2>::variable(5.0, 0);
        let y = DualPack::<f64, 2>::constant(0.0).sqrt();
        let result = x + y;
        assert_eq!(result.tangent[0], 1.0);
        assert_eq!(result.tangent[1], 0.0);
        assert!(result.tangent[0].is_finite());
        assert!(result.tangent[1].is_finite());
    }

    #[test]
    fn division_keeps_zero_seed_neutral_at_singularity() {
        let x = DualPack::<f64, 2>::variable(4.0, 0);
        let zero = DualPack::<f64, 2>::constant(0.0);
        let singular = DualPack::<f64, 2>::constant(2.0) / zero;
        let result = x + singular;
        assert_eq!(result.tangent[0], 1.0);
        assert_eq!(result.tangent[1], 0.0);
    }

    fn seeded_f32<const W: usize>(base: f32, slope: f32) -> DualPack<f32, W> {
        let mut tangent = [0.0f32; W];
        for (index, value) in tangent.iter_mut().enumerate()
        {
            *value = base + index as f32 * slope;
        }
        DualPack::seeded(base, tangent)
    }

    #[test]
    fn simd_add_matches_scalar_for_tail_width() {
        let left = seeded_f32::<19>(0.25, 0.125);
        let right = seeded_f32::<19>(-0.5, 0.0625);
        assert_eq!(left.simd_add(right), left + right);
    }

    #[test]
    fn simd_product_rule_matches_scalar_for_common_widths() {
        let left8 = seeded_f32::<8>(0.5, 0.125);
        let right8 = seeded_f32::<8>(-0.25, 0.25);
        assert_eq!(left8.simd_mul(right8), left8 * right8);

        let left16 = seeded_f32::<16>(0.75, 0.0625);
        let right16 = seeded_f32::<16>(1.25, -0.03125);
        assert_eq!(left16.simd_mul(right16), left16 * right16);
    }

    #[test]
    fn simd_zero_width_is_valid_and_allocation_free() {
        let left = DualPack::<f32, 0>::constant(2.0);
        let right = DualPack::<f32, 0>::constant(3.0);
        assert_eq!(left.simd_mul(right), DualPack::constant(6.0));
    }
}
