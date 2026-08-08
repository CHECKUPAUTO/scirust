//! Normalized spherical harmonics and integer-spin Wigner rotation primitives.
//!
//! The routines are deterministic scalar reference kernels intended to anchor later
//! SIMD/tensor implementations. They allocate no heap storage.

use core::f64::consts::PI;

use crate::representation::associated_legendre;

/// Normalization factor for the complex spherical harmonic `Y_l^m` with `m >= 0`.
///
/// Returns `None` when `m > l` or when the factorial products exceed finite `f64`
/// range. The associated Legendre convention in [`associated_legendre`] already
/// includes the Condon-Shortley phase.
pub fn spherical_harmonic_normalization(l: usize, m: usize) -> Option<f64>
{
    if m > l
    {
        return None;
    }
    let numerator = factorial(l - m)?;
    let denominator = factorial(l + m)?;
    let value = ((2 * l + 1) as f64 / (4.0 * PI) * numerator / denominator).sqrt();
    value.is_finite().then_some(value)
}

/// Normalized real spherical harmonic cosine branch for `m >= 0`.
///
/// For `m == 0` this is the zonal harmonic. For `m > 0`, the usual real basis
/// normalization factor `sqrt(2)` is applied.
pub fn real_spherical_harmonic_cosine(
    l: usize,
    m: usize,
    theta: f64,
    phi: f64,
) -> Option<f64>
{
    let legendre = associated_legendre(l, m, theta.cos())?;
    let normalization = spherical_harmonic_normalization(l, m)?;
    let real_factor = if m == 0 { 1.0 } else { 2.0_f64.sqrt() };
    Some(real_factor * normalization * legendre * (m as f64 * phi).cos())
}

/// Normalized real spherical harmonic sine branch for `m > 0`.
///
/// `m == 0` is identically zero and is accepted for convenience.
pub fn real_spherical_harmonic_sine(
    l: usize,
    m: usize,
    theta: f64,
    phi: f64,
) -> Option<f64>
{
    if m == 0
    {
        return (l == 0 || l > 0).then_some(0.0);
    }
    let legendre = associated_legendre(l, m, theta.cos())?;
    let normalization = spherical_harmonic_normalization(l, m)?;
    Some(2.0_f64.sqrt() * normalization * legendre * (m as f64 * phi).sin())
}

/// Integer-spin Wigner small-`d` matrix element `d^l_{m',m}(beta)`.
///
/// The implementation uses the exact finite factorial sum and therefore serves as a
/// deterministic reference for later recurrence/SIMD kernels. `m` and `m_prime` must
/// lie in `[-l, l]`. Large `l` values for which factorials are not representable in
/// finite `f64` return `None`.
pub fn wigner_small_d(
    l: usize,
    m_prime: isize,
    m: isize,
    beta: f64,
) -> Option<f64>
{
    let l_signed = isize::try_from(l).ok()?;
    if m.abs() > l_signed || m_prime.abs() > l_signed
    {
        return None;
    }

    let lp_m = usize::try_from(l_signed + m).ok()?;
    let lm_m = usize::try_from(l_signed - m).ok()?;
    let lp_mp = usize::try_from(l_signed + m_prime).ok()?;
    let lm_mp = usize::try_from(l_signed - m_prime).ok()?;
    let prefactor = (factorial(lp_m)?
        * factorial(lm_m)?
        * factorial(lp_mp)?
        * factorial(lm_mp)?)
    .sqrt();
    if !prefactor.is_finite()
    {
        return None;
    }

    let k_min = 0_isize.max(m - m_prime);
    let k_max = (l_signed + m).min(l_signed - m_prime);
    if k_min > k_max
    {
        return Some(0.0);
    }

    let cos_half = (0.5 * beta).cos();
    let sin_half = (0.5 * beta).sin();
    let mut sum = 0.0;
    let mut k = k_min;
    while k <= k_max
    {
        let a = usize::try_from(l_signed + m - k).ok()?;
        let b = usize::try_from(k).ok()?;
        let c = usize::try_from(m_prime - m + k).ok()?;
        let d = usize::try_from(l_signed - m_prime - k).ok()?;
        let denominator = factorial(a)? * factorial(b)? * factorial(c)? * factorial(d)?;
        if !denominator.is_finite() || denominator == 0.0
        {
            return None;
        }

        let cos_power = i32::try_from(2 * l_signed + m - m_prime - 2 * k).ok()?;
        let sin_power = i32::try_from(m_prime - m + 2 * k).ok()?;
        if cos_power < 0 || sin_power < 0
        {
            return None;
        }
        let sign_exponent = k - m_prime + m;
        let sign = if sign_exponent.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        sum += sign
            * prefactor
            / denominator
            * cos_half.powi(cos_power)
            * sin_half.powi(sin_power);
        k += 1;
    }
    sum.is_finite().then_some(sum)
}

fn factorial(value: usize) -> Option<f64>
{
    let mut out = 1.0_f64;
    let mut i = 2usize;
    while i <= value
    {
        out *= i as f64;
        if !out.is_finite()
        {
            return None;
        }
        i += 1;
    }
    Some(out)
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn y00_has_standard_normalization()
    {
        let expected = 1.0 / (4.0 * PI).sqrt();
        let actual = real_spherical_harmonic_cosine(0, 0, 0.7, 1.3).unwrap();
        assert!((actual - expected).abs() < 1e-14);
    }

    #[test]
    fn wigner_l1_m0_m0_is_cos_beta()
    {
        let beta = 0.73;
        let actual = wigner_small_d(1, 0, 0, beta).unwrap();
        assert!((actual - beta.cos()).abs() < 1e-14);
    }

    #[test]
    fn wigner_l1_highest_weight_matches_half_angle()
    {
        let beta = 1.17;
        let actual = wigner_small_d(1, 1, 1, beta).unwrap();
        let expected = (0.5 * beta).cos().powi(2);
        assert!((actual - expected).abs() < 1e-14);
    }

    #[test]
    fn wigner_rejects_out_of_range_m()
    {
        assert_eq!(wigner_small_d(2, 3, 0, 0.5), None);
        assert_eq!(wigner_small_d(2, 0, -3, 0.5), None);
    }
}
