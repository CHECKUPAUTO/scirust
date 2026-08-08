//! Integer-angular-momentum coupling coefficients for SO(3).
//!
//! These deterministic scalar reference routines implement the Racah factorial sum
//! for Wigner 3j symbols and derive Clebsch-Gordan coefficients from it. They are
//! allocation-free and intended to generate/validate equivariant coupling tables.

/// Compute the Wigner 3j symbol
/// `(j1 j2 j3; m1 m2 m3)` for non-negative integer angular momenta.
///
/// Returns `None` when magnetic quantum numbers are out of range or when an internal
/// factorial is no longer representable as finite `f64`. Selection-rule violations
/// such as a failed triangle condition or `m1 + m2 + m3 != 0` return `Some(0.0)`.
pub fn wigner_3j(
    j1: usize,
    j2: usize,
    j3: usize,
    m1: isize,
    m2: isize,
    m3: isize,
) -> Option<f64>
{
    let j1i = isize::try_from(j1).ok()?;
    let j2i = isize::try_from(j2).ok()?;
    let j3i = isize::try_from(j3).ok()?;
    if m1.abs() > j1i || m2.abs() > j2i || m3.abs() > j3i
    {
        return None;
    }
    if m1 + m2 + m3 != 0 || !triangle_allowed(j1, j2, j3)
    {
        return Some(0.0);
    }

    let delta_numerator = factorial(j1 + j2 - j3)?
        * factorial(j1 + j3 - j2)?
        * factorial(j2 + j3 - j1)?;
    let delta_denominator = factorial(j1 + j2 + j3 + 1)?;

    let magnetic_factor = factorial(usize::try_from(j1i + m1).ok()?)?
        * factorial(usize::try_from(j1i - m1).ok()?)?
        * factorial(usize::try_from(j2i + m2).ok()?)?
        * factorial(usize::try_from(j2i - m2).ok()?)?
        * factorial(usize::try_from(j3i + m3).ok()?)?
        * factorial(usize::try_from(j3i - m3).ok()?)?;

    let prefactor = (delta_numerator / delta_denominator * magnetic_factor).sqrt();
    if !prefactor.is_finite()
    {
        return None;
    }

    let a = isize::try_from(j1 + j2 - j3).ok()?;
    let b = j1i - m1;
    let c = j2i + m2;
    let d = j3i - j2i + m1;
    let e = j3i - j1i - m2;
    let z_min = 0_isize.max(-d).max(-e);
    let z_max = a.min(b).min(c);
    if z_min > z_max
    {
        return Some(0.0);
    }

    let mut sum = 0.0;
    let mut z = z_min;
    while z <= z_max
    {
        let denominator = factorial(usize::try_from(z).ok()?)?
            * factorial(usize::try_from(a - z).ok()?)?
            * factorial(usize::try_from(b - z).ok()?)?
            * factorial(usize::try_from(c - z).ok()?)?
            * factorial(usize::try_from(d + z).ok()?)?
            * factorial(usize::try_from(e + z).ok()?)?;
        if !denominator.is_finite() || denominator == 0.0
        {
            return None;
        }
        let sign = parity_sign(z);
        sum += sign / denominator;
        z += 1;
    }

    let phase = parity_sign(j1i - j2i - m3);
    let result = phase * prefactor * sum;
    result.is_finite().then_some(result)
}

/// Compute the Clebsch-Gordan coefficient
/// `<j1 m1, j2 m2 | j m>` for non-negative integer angular momenta.
///
/// The convention is related to [`wigner_3j`] by
/// `(-1)^(j1-j2+m) sqrt(2j+1) (j1 j2 j; m1 m2 -m)`.
pub fn clebsch_gordan(
    j1: usize,
    m1: isize,
    j2: usize,
    m2: isize,
    j: usize,
    m: isize,
) -> Option<f64>
{
    let ji = isize::try_from(j).ok()?;
    if m.abs() > ji
    {
        return None;
    }
    if m1 + m2 != m
    {
        return Some(0.0);
    }
    let j1i = isize::try_from(j1).ok()?;
    let j2i = isize::try_from(j2).ok()?;
    let symbol = wigner_3j(j1, j2, j, m1, m2, -m)?;
    let phase = parity_sign(j1i - j2i + m);
    Some(phase * ((2 * j + 1) as f64).sqrt() * symbol)
}

/// Write all coefficients coupling fixed `j1`, `j2` into a selected `(j, m)` channel.
///
/// Coefficients are emitted in lexicographic `(m1, m2)` order into caller-owned
/// storage. Only pairs satisfying `m1 + m2 = m` are written. Returns the number of
/// coefficients produced or `None` when the output buffer is too small or the quantum
/// numbers are invalid.
pub fn clebsch_gordan_channel_into(
    j1: usize,
    j2: usize,
    j: usize,
    m: isize,
    output: &mut [f64],
) -> Option<usize>
{
    if !triangle_allowed(j1, j2, j) || m.abs() > isize::try_from(j).ok()?
    {
        return None;
    }
    let j1i = isize::try_from(j1).ok()?;
    let j2i = isize::try_from(j2).ok()?;
    let mut count = 0usize;
    let mut m1 = -j1i;
    while m1 <= j1i
    {
        let m2 = m - m1;
        if m2 >= -j2i && m2 <= j2i
        {
            if count >= output.len()
            {
                return None;
            }
            output[count] = clebsch_gordan(j1, m1, j2, m2, j, m)?;
            count += 1;
        }
        m1 += 1;
    }
    Some(count)
}

#[inline]
fn triangle_allowed(j1: usize, j2: usize, j3: usize) -> bool
{
    j1.abs_diff(j2) <= j3 && j3 <= j1.saturating_add(j2)
}

#[inline]
fn parity_sign(exponent: isize) -> f64
{
    if exponent.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
}

fn factorial(value: usize) -> Option<f64>
{
    let mut result = 1.0;
    let mut n = 2usize;
    while n <= value
    {
        result *= n as f64;
        if !result.is_finite()
        {
            return None;
        }
        n += 1;
    }
    Some(result)
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn scalar_coupling_is_identity()
    {
        assert!((clebsch_gordan(2, 1, 0, 0, 2, 1).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn two_vectors_couple_to_scalar_with_standard_signs()
    {
        let normalization = 3.0_f64.sqrt();
        assert!((clebsch_gordan(1, 1, 1, -1, 0, 0).unwrap() - 1.0 / normalization).abs() < 1e-14);
        assert!((clebsch_gordan(1, 0, 1, 0, 0, 0).unwrap() + 1.0 / normalization).abs() < 1e-14);
        assert!((clebsch_gordan(1, -1, 1, 1, 0, 0).unwrap() - 1.0 / normalization).abs() < 1e-14);
    }

    #[test]
    fn selected_channel_is_normalized()
    {
        let mut coefficients = [0.0; 5];
        let len = clebsch_gordan_channel_into(2, 2, 3, 1, &mut coefficients).unwrap();
        let norm = coefficients[..len].iter().map(|value| value * value).sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn selection_rules_return_zero()
    {
        assert_eq!(wigner_3j(1, 1, 3, 0, 0, 0), Some(0.0));
        assert_eq!(wigner_3j(1, 1, 1, 1, 1, -1), Some(0.0));
        assert_eq!(clebsch_gordan(1, 1, 1, 1, 1, 1), Some(0.0));
    }
}
