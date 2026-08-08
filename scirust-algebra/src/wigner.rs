//! Complex Wigner `D` matrix elements for integer-spin SO(3) representations.

use crate::harmonics::wigner_small_d;

/// Minimal complex scalar used by representation-theory reference kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex64
{
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

impl Complex64
{
    /// Construct a complex scalar.
    pub const fn new(re: f64, im: f64) -> Self
    {
        Self { re, im }
    }

    /// Unit complex phase `exp(i * angle)`.
    pub fn cis(angle: f64) -> Self
    {
        Self {
            re: angle.cos(),
            im: angle.sin(),
        }
    }

    /// Complex multiplication.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self
    {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    /// Multiply by a real scalar.
    #[inline]
    pub fn scale(self, value: f64) -> Self
    {
        Self {
            re: self.re * value,
            im: self.im * value,
        }
    }

    /// Complex conjugate.
    #[inline]
    pub const fn conjugate(self) -> Self
    {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude.
    #[inline]
    pub fn norm_squared(self) -> f64
    {
        self.re * self.re + self.im * self.im
    }
}

/// Integer-spin Wigner matrix element
/// `D^l_{m',m}(alpha, beta, gamma)` in the z-y-z Euler convention.
///
/// The convention is
/// `D = exp(-i m' alpha) d^l_{m',m}(beta) exp(-i m gamma)`.
pub fn wigner_d(
    l: usize,
    m_prime: isize,
    m: isize,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> Option<Complex64>
{
    let small = wigner_small_d(l, m_prime, m, beta)?;
    let left = Complex64::cis(-(m_prime as f64) * alpha);
    let right = Complex64::cis(-(m as f64) * gamma);
    Some(left.mul(right).scale(small))
}

/// Normalized complex spherical harmonic `Y_l^m(theta, phi)` for signed `m`.
///
/// Positive orders use the Condon-Shortley convention from
/// [`crate::representation::associated_legendre`]. Negative orders are obtained from
/// `Y_l^{-m} = (-1)^m conjugate(Y_l^m)`.
pub fn complex_spherical_harmonic(
    l: usize,
    m: isize,
    theta: f64,
    phi: f64,
) -> Option<Complex64>
{
    let l_signed = isize::try_from(l).ok()?;
    if m.abs() > l_signed
    {
        return None;
    }
    let order = m.unsigned_abs();
    let normalization = crate::harmonics::spherical_harmonic_normalization(l, order)?;
    let legendre = crate::representation::associated_legendre(l, order, theta.cos())?;
    let positive = Complex64::cis(order as f64 * phi).scale(normalization * legendre);
    if m >= 0
    {
        Some(positive)
    }
    else
    {
        let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
        Some(positive.conjugate().scale(sign))
    }
}

#[cfg(test)]
mod tests
{
    use core::f64::consts::PI;

    use super::*;

    #[test]
    fn wigner_d_reduces_to_small_d_without_z_rotations()
    {
        let beta = 0.71;
        let full = wigner_d(2, 1, -1, 0.0, beta, 0.0).unwrap();
        let small = wigner_small_d(2, 1, -1, beta).unwrap();
        assert!((full.re - small).abs() < 1e-14);
        assert!(full.im.abs() < 1e-14);
    }

    #[test]
    fn scalar_representation_is_identity()
    {
        let value = wigner_d(0, 0, 0, 0.4, 0.8, -1.2).unwrap();
        assert!((value.re - 1.0).abs() < 1e-14);
        assert!(value.im.abs() < 1e-14);
    }

    #[test]
    fn y00_is_real_and_normalized()
    {
        let value = complex_spherical_harmonic(0, 0, 1.1, 2.2).unwrap();
        assert!((value.re - 1.0 / (4.0 * PI).sqrt()).abs() < 1e-14);
        assert!(value.im.abs() < 1e-14);
    }

    #[test]
    fn negative_m_obeys_conjugation_identity()
    {
        let positive = complex_spherical_harmonic(3, 2, 0.9, 1.4).unwrap();
        let negative = complex_spherical_harmonic(3, -2, 0.9, 1.4).unwrap();
        assert!((negative.re - positive.re).abs() < 1e-14);
        assert!((negative.im + positive.im).abs() < 1e-14);
    }
}
