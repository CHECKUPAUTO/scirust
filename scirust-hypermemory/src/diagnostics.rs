//! Zero-divisor risk instrumentation for sedenion products.
//!
//! Sedenions contain zero divisors — non-zero `x, y` with `x·y = 0` — so a
//! product collapsing toward zero is a first-class engineering hazard, not a
//! bug. [`ProductDiagnostics`] measures every relation product without ever
//! rejecting it by default: the caller decides what to do with a near-zero
//! result. No path here emits a silent `NaN`; non-finite results are flagged.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::representation::{is_finite, norm_sqr_ordered};

/// The default near-zero-divisor threshold, applied to the **squared** norm of
/// a product. A product whose `result_norm_sqr ≤ threshold` (with both operands
/// strictly non-zero) is flagged as a near-zero divisor. Chosen small enough to
/// only trip on genuine (near-)annihilation of `f32` unit-scale operands.
pub const DEFAULT_NEAR_ZERO_THRESHOLD: f32 = 1e-12;

/// Norm and zero-divisor measurements of a single sedenion product `lhs · rhs`.
///
/// All norms use the fixed index-order scalar reduction
/// ([`norm_sqr_ordered`]), matching the index's scoring order, so a measurement
/// is bit-reproducible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProductDiagnostics {
    lhs_norm_sqr: f32,
    rhs_norm_sqr: f32,
    result_norm_sqr: f32,
    near_zero_divisor: bool,
    finite: bool,
}

impl ProductDiagnostics {
    /// Measure `lhs · rhs` against `threshold` (compared to the result's
    /// **squared** norm, inclusive `≤`).
    ///
    /// `near_zero_divisor` is true iff **both** operands are strictly non-zero
    /// (`lhs_norm_sqr > 0 ∧ rhs_norm_sqr > 0`) **and**
    /// `result_norm_sqr ≤ threshold`. A non-finite result yields `finite =
    /// false` and (because `NaN`/`∞ ≤ threshold` is false) is never flagged as a
    /// zero divisor.
    #[must_use]
    pub fn measure(lhs: &SedenionSimd, rhs: &SedenionSimd, threshold: f32) -> Self {
        let result = *lhs * *rhs;
        let lhs_norm_sqr = norm_sqr_ordered(lhs);
        let rhs_norm_sqr = norm_sqr_ordered(rhs);
        let result_norm_sqr = norm_sqr_ordered(&result);
        let finite = is_finite(&result);
        let near_zero_divisor =
            lhs_norm_sqr > 0.0 && rhs_norm_sqr > 0.0 && result_norm_sqr <= threshold;
        Self {
            lhs_norm_sqr,
            rhs_norm_sqr,
            result_norm_sqr,
            near_zero_divisor,
            finite,
        }
    }

    /// Measure using [`DEFAULT_NEAR_ZERO_THRESHOLD`].
    #[must_use]
    pub fn measure_default(lhs: &SedenionSimd, rhs: &SedenionSimd) -> Self {
        Self::measure(lhs, rhs, DEFAULT_NEAR_ZERO_THRESHOLD)
    }

    /// Squared norm of the left operand.
    #[inline]
    #[must_use]
    pub const fn lhs_norm_sqr(&self) -> f32 {
        self.lhs_norm_sqr
    }

    /// Squared norm of the right operand.
    #[inline]
    #[must_use]
    pub const fn rhs_norm_sqr(&self) -> f32 {
        self.rhs_norm_sqr
    }

    /// Squared norm of the product.
    #[inline]
    #[must_use]
    pub const fn result_norm_sqr(&self) -> f32 {
        self.result_norm_sqr
    }

    /// Whether this product is flagged as a near-zero divisor.
    #[inline]
    #[must_use]
    pub const fn near_zero_divisor(&self) -> bool {
        self.near_zero_divisor
    }

    /// Whether the product is lane-wise finite.
    #[inline]
    #[must_use]
    pub const fn finite(&self) -> bool {
        self.finite
    }

    /// Whether normalizing the product is safe (non-zero norm and finite).
    #[inline]
    #[must_use]
    pub const fn normalization_safe(&self) -> bool {
        self.result_norm_sqr > 0.0 && self.finite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_zero_divisor_is_flagged() {
        // The canonical SciRust identity: (e1 + e10)·(e4 − e15) = 0 exactly.
        let x = SedenionSimd::unit(1) + SedenionSimd::unit(10);
        let y = SedenionSimd::unit(4) - SedenionSimd::unit(15);
        let d = ProductDiagnostics::measure_default(&x, &y);
        assert_eq!(d.lhs_norm_sqr(), 2.0);
        assert_eq!(d.rhs_norm_sqr(), 2.0);
        assert_eq!(d.result_norm_sqr(), 0.0);
        assert!(
            d.near_zero_divisor(),
            "both operands non-zero, product zero"
        );
        assert!(d.finite());
        assert!(!d.normalization_safe(), "cannot normalize the zero result");
    }

    #[test]
    fn ordinary_product_is_not_flagged() {
        let x = SedenionSimd::unit(0); // real unit
        let y = SedenionSimd::unit(1);
        let d = ProductDiagnostics::measure_default(&x, &y);
        assert!(!d.near_zero_divisor());
        assert!(d.finite());
        assert!(d.normalization_safe());
        assert_eq!(d.result_norm_sqr(), 1.0);
    }

    #[test]
    fn zero_operand_is_not_a_zero_divisor() {
        // A product that is zero *because an operand is zero* is not a zero
        // divisor — the "both operands non-zero" guard rules it out.
        let d = ProductDiagnostics::measure_default(&SedenionSimd::ZERO, &SedenionSimd::unit(1));
        assert_eq!(d.result_norm_sqr(), 0.0);
        assert!(!d.near_zero_divisor());
    }

    #[test]
    fn threshold_is_respected() {
        // A tiny but non-zero product is flagged only under a large threshold.
        let x = SedenionSimd::unit(0).scale(1e-4);
        let y = SedenionSimd::unit(0).scale(1e-4); // product norm² ≈ 1e-16
        let strict = ProductDiagnostics::measure(&x, &y, 1e-20);
        assert!(!strict.near_zero_divisor());
        let loose = ProductDiagnostics::measure(&x, &y, 1e-12);
        assert!(loose.near_zero_divisor());
    }
}
