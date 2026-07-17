//! Representation validation and the effective-representation derivation.
//!
//! The sedenion is never the *sole* source of truth (the exact payload is kept
//! separately, see [`crate::ConceptRecord`]), but the sedenion still has to be a
//! valid unit vector for indexing. This module centralizes the finite / zero /
//! normalization checks so every path validates identically and no function can
//! silently emit `NaN` or `∞`.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::error::{HypermemoryError, Result};

/// True iff every one of the 16 lanes is finite (no `NaN`, no `±∞`).
#[inline]
#[must_use]
pub fn is_finite(s: &SedenionSimd) -> bool {
    // Fixed index order 0..16; no reduction, no reassociation.
    s.to_array().iter().all(|x| x.is_finite())
}

/// Validate that a sedenion is lane-wise finite, else
/// [`HypermemoryError::NonFiniteRepresentation`].
#[inline]
pub fn validate_finite(s: &SedenionSimd) -> Result<()> {
    if is_finite(s)
    {
        Ok(())
    }
    else
    {
        Err(HypermemoryError::NonFiniteRepresentation)
    }
}

/// The squared norm computed in fixed index order (auditable, reproducible).
///
/// [`SedenionSimd::norm_sqr`] uses the SIMD tree reduction; here we use the
/// scalar left-to-right order so the *validation* threshold decisions are
/// bit-reproducible and beyond dispute, matching the reduction order the index
/// uses for scoring.
#[inline]
#[must_use]
pub fn norm_sqr_ordered(s: &SedenionSimd) -> f32 {
    let a = s.to_array();
    let mut acc = 0.0f32;
    for x in a
    {
        acc += x * x;
    }
    acc
}

/// Compute the **effective representation** `normalize(anchor + residual)` with
/// full, explicit degenerate-input handling.
///
/// Failure modes, in the order they are checked:
///
/// * a non-finite lane in the sum → [`HypermemoryError::NonFiniteRepresentation`];
/// * a non-finite squared norm (overflow) → `NonFiniteRepresentation`;
/// * a zero squared norm → [`HypermemoryError::ZeroNormRepresentation`];
/// * a non-finite normalized result (defensive re-check) → `NonFiniteRepresentation`.
///
/// On success the result is a unit-norm sedenion. Normalization is a single
/// scalar multiply (`scale(1/√‖·‖²)`); no library call that could reassociate.
pub fn effective_representation(
    anchor: &SedenionSimd,
    residual: &SedenionSimd,
) -> Result<SedenionSimd> {
    let sum = *anchor + *residual;
    validate_finite(&sum)?;

    let n2 = norm_sqr_ordered(&sum);
    if !n2.is_finite()
    {
        return Err(HypermemoryError::NonFiniteRepresentation);
    }
    if n2 <= 0.0
    {
        return Err(HypermemoryError::ZeroNormRepresentation);
    }

    let effective = sum.scale(1.0 / n2.sqrt());
    // Defensive: a finite sum with a finite positive norm cannot normally
    // produce a non-finite result, but re-check so the contract ("never emit
    // NaN/inf") holds unconditionally.
    validate_finite(&effective)?;
    Ok(effective)
}

/// Normalize a raw 16-lane array to unit norm, with the same degenerate-input
/// handling and the same fixed index-order arithmetic as
/// [`effective_representation`].
///
/// This is the shared query-normalization path for *both* the sedenion index
/// and the real-vector baseline, so their results are bit-identical by
/// construction (the crux of the Phase 1 falsification test F1). The per-lane
/// multiply `q[i] * (1/√‖q‖²)` is identical whether performed lane-wise on a
/// SIMD register or scalar-by-scalar, so a normalized `SedenionSimd` and this
/// normalized array agree bit-for-bit.
pub fn normalize_array(q: &[f32; 16]) -> Result<[f32; 16]> {
    if !q.iter().all(|x| x.is_finite())
    {
        return Err(HypermemoryError::NonFiniteRepresentation);
    }
    let mut n2 = 0.0f32;
    for &x in q
    {
        n2 += x * x;
    }
    if !n2.is_finite()
    {
        return Err(HypermemoryError::NonFiniteRepresentation);
    }
    if n2 <= 0.0
    {
        return Err(HypermemoryError::ZeroNormRepresentation);
    }
    let inv = 1.0 / n2.sqrt();
    let mut out = [0.0f32; 16];
    for (o, &x) in out.iter_mut().zip(q.iter())
    {
        *o = x * inv;
    }
    if !out.iter().all(|x| x.is_finite())
    {
        return Err(HypermemoryError::NonFiniteRepresentation);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sed(a: [f32; 16]) -> SedenionSimd {
        SedenionSimd::from_array(a)
    }

    #[test]
    fn normalize_array_matches_effective_representation_bitwise() {
        // The shared array path and the sedenion path must agree bit-for-bit,
        // which is what makes the sedenion index and real baseline identical.
        let anchor = SedenionSimd::unit(1) + SedenionSimd::unit(10).scale(0.7);
        let eff = effective_representation(&anchor, &SedenionSimd::ZERO).unwrap();
        let arr = normalize_array(&anchor.to_array()).unwrap();
        assert_eq!(eff.to_array(), arr);
    }

    #[test]
    fn normalize_array_rejects_degenerate_inputs() {
        assert_eq!(
            normalize_array(&[0.0; 16]),
            Err(HypermemoryError::ZeroNormRepresentation)
        );
        let mut nan = [0.0f32; 16];
        nan[0] = f32::NAN;
        assert_eq!(
            normalize_array(&nan),
            Err(HypermemoryError::NonFiniteRepresentation)
        );
    }

    #[test]
    fn finite_representation_is_accepted() {
        let s = SedenionSimd::unit(0);
        assert!(is_finite(&s));
        assert!(validate_finite(&s).is_ok());
    }

    #[test]
    fn nan_is_rejected() {
        let mut a = [0.0f32; 16];
        a[3] = f32::NAN;
        assert!(!is_finite(&sed(a)));
        assert_eq!(
            validate_finite(&sed(a)),
            Err(HypermemoryError::NonFiniteRepresentation)
        );
    }

    #[test]
    fn positive_and_negative_infinity_are_rejected() {
        for v in [f32::INFINITY, f32::NEG_INFINITY]
        {
            let mut a = [0.0f32; 16];
            a[0] = v;
            assert_eq!(
                effective_representation(&sed(a), &SedenionSimd::ZERO),
                Err(HypermemoryError::NonFiniteRepresentation)
            );
        }
    }

    #[test]
    fn zero_representation_is_rejected_when_normalization_required() {
        assert_eq!(
            effective_representation(&SedenionSimd::ZERO, &SedenionSimd::ZERO),
            Err(HypermemoryError::ZeroNormRepresentation)
        );
    }

    #[test]
    fn effective_is_unit_norm_and_deterministic() {
        let anchor = SedenionSimd::unit(1) + SedenionSimd::unit(10);
        let e1 = effective_representation(&anchor, &SedenionSimd::ZERO).unwrap();
        let e2 = effective_representation(&anchor, &SedenionSimd::ZERO).unwrap();
        // Bit-identical across repeated evaluation.
        assert_eq!(e1.to_array(), e2.to_array());
        // Unit norm (‖e₁+e₁₀‖² = 2 → each nonzero lane = 1/√2).
        let n2 = norm_sqr_ordered(&e1);
        assert!((n2 - 1.0).abs() < 1e-6, "‖effective‖² = {n2}");
    }

    #[test]
    fn residual_shifts_the_effective_vector() {
        let anchor = SedenionSimd::unit(0);
        let residual = SedenionSimd::unit(1).scale(0.25);
        let e = effective_representation(&anchor, &residual).unwrap();
        let arr = e.to_array();
        // Both lanes populated; direction is (1, 0.25, 0..)/‖·‖.
        assert!(arr[0] > 0.0 && arr[1] > 0.0);
        assert!((norm_sqr_ordered(&e) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overflowing_norm_is_reported_not_silently_zeroed() {
        // A huge anchor whose squared norm overflows f32 must be detected, not
        // silently normalized to the zero vector.
        let big = f32::MAX;
        let s = sed([big; 16]);
        assert_eq!(
            effective_representation(&s, &SedenionSimd::ZERO),
            Err(HypermemoryError::NonFiniteRepresentation)
        );
    }
}
