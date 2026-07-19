//! Auditable single-precision complex arithmetic for quantum amplitudes.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex scalar with a stable C-compatible layout: `re` followed by `im`.
///
/// Both components use IEEE-754 binary32 (`f32`). Quantum state storage is
/// therefore exactly eight bytes per amplitude, excluding `Vec` metadata.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex32 {
    /// Real component.
    pub re: f32,
    /// Imaginary component.
    pub im: f32,
}

impl Complex32 {
    /// Constructs `re + i·im`.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Additive identity.
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Multiplicative identity.
    #[inline]
    pub const fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Complex conjugate.
    #[inline]
    pub const fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Squared Euclidean norm, `re² + im²`.
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Euclidean norm, `sqrt(re² + im²)`.
    #[inline]
    pub fn norm(self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Unit phase `cos(theta) + i·sin(theta)`, with `theta` in radians.
    #[inline]
    pub fn from_phase(theta: f32) -> Self {
        Self::new(theta.cos(), theta.sin())
    }

    /// Returns true when both components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Component-wise absolute-tolerance comparison, intended for tests and
    /// explicit numerical validation rather than `PartialEq` replacement.
    #[inline]
    pub fn approx_eq(self, other: Self, tolerance: f32) -> bool {
        tolerance.is_finite()
            && tolerance >= 0.0
            && (self.re - other.re).abs() <= tolerance
            && (self.im - other.im).abs() <= tolerance
    }
}

impl Add for Complex32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Neg for Complex32 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl Mul for Complex32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl MulAssign for Complex32 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<f32> for Complex32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<Complex32> for f32 {
    type Output = Complex32;

    #[inline]
    fn mul(self, rhs: Complex32) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for Complex32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl Div for Complex32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.norm_sqr();
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denominator,
            (self.im * rhs.re - self.re * rhs.im) / denominator,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1.0e-6;

    #[test]
    fn i_squared_is_minus_one() {
        let i = Complex32::new(0.0, 1.0);
        assert!((i * i).approx_eq(Complex32::new(-1.0, 0.0), TOLERANCE));
    }

    #[test]
    fn conjugate_product_is_squared_norm() {
        let value = Complex32::new(3.0, -4.0);
        assert!((value * value.conj()).approx_eq(Complex32::new(25.0, 0.0), TOLERANCE));
        assert!((value.norm() - 5.0).abs() <= TOLERANCE);
    }

    #[test]
    fn multiplication_and_division_match_manual_values() {
        let a = Complex32::new(3.0, 2.0);
        let b = Complex32::new(1.0, -5.0);
        assert!((a * b).approx_eq(Complex32::new(13.0, -13.0), TOLERANCE));
        assert!(((a * b) / b).approx_eq(a, TOLERANCE));
    }

    #[test]
    fn phase_values_match_quadrants() {
        let half_pi = core::f32::consts::FRAC_PI_2;
        assert!(Complex32::from_phase(0.0).approx_eq(Complex32::one(), TOLERANCE));
        assert!(Complex32::from_phase(half_pi).approx_eq(Complex32::new(0.0, 1.0), TOLERANCE));
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        let a = Complex32::new(0.5, -0.25);
        let b = Complex32::new(2.0, 3.0);
        let c = Complex32::new(-1.0, 4.0);
        assert!((a * (b + c)).approx_eq(a * b + a * c, TOLERANCE));
    }

    #[test]
    fn layout_is_two_adjacent_f32_values() {
        assert_eq!(
            core::mem::size_of::<Complex32>(),
            2 * core::mem::size_of::<f32>()
        );
        assert_eq!(
            core::mem::align_of::<Complex32>(),
            core::mem::align_of::<f32>()
        );
    }
}
