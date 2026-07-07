//! Six-Sigma yield accounting — DPMO, throughput yield, rolled throughput yield,
//! and the sigma-level conversions that tie them together.
//!
//! Capability indices ([`crate::capability`]) describe *one* characteristic; a
//! product is a **chain** of process steps, each with its own defect rate, and
//! the business question is the yield of the whole chain. Six-Sigma accounting
//! answers it with a small, exact algebra:
//!
//! ```text
//! DPU   = defects / units                    (defects per unit)
//! DPMO  = DPU / opportunities × 10⁶           (defects per million opportunities)
//! Y_tp  = e^(−DPU)                            (throughput yield, Poisson model)
//! RTY   = ∏ Y_tp,i                            (rolled throughput yield of a chain)
//! Y_nrm = RTY^(1/steps)                       (normalised, per-step yield)
//! ```
//!
//! The **rolled throughput yield** — the probability a unit clears *every* step
//! with no rework — is the number a single step's capability cannot show: five
//! 99 %-yield steps still roll up to only 95 %.
//!
//! ## Sigma level
//!
//! A yield maps to a process sigma via the normal quantile, and Motorola's
//! convention adds the long-term `1.5σ` shift so that the familiar "6σ ⇒ 3.4
//! DPMO" holds:
//!
//! ```text
//! Z(Y) = Φ⁻¹(Y) + shift ,     Y(Z) = Φ(Z − shift) ,     DPMO = 10⁶·(1 − Y) .
//! ```
//!
//! Pass `shift = 1.5` for the customary short-term sigma level, `shift = 0` for
//! the honest long-term `Z_bench` (matching [`crate::capability::sigma_level`]).

use crate::special::{inv_normal_cdf, normal_cdf};
use serde::{Deserialize, Serialize};

/// Defects per unit, `defects / units` (0 for a non-positive unit count).
pub fn dpu(defects: f64, units: f64) -> f64 {
    if units <= 0.0
    {
        return 0.0;
    }
    (defects / units).max(0.0)
}

/// Defects per million opportunities, `defects / (units · opportunities) · 10⁶`.
/// `opportunities` is the count of independent ways to create a defect per unit.
pub fn dpmo(defects: f64, units: f64, opportunities: f64) -> f64 {
    if units <= 0.0 || opportunities <= 0.0
    {
        return 0.0;
    }
    (defects / (units * opportunities)).max(0.0) * 1e6
}

/// Throughput yield of a step from its `dpu`, `Y = e^(−DPU)` — the Poisson
/// probability a unit passes with zero defects. Clamped to `[0, 1]`.
pub fn throughput_yield(dpu: f64) -> f64 {
    (-dpu.max(0.0)).exp()
}

/// Rolled throughput yield of a chain, `∏ Yᵢ` — the probability a unit clears
/// every step defect-free. An empty chain yields 1; any step outside `[0, 1]` is
/// clamped.
pub fn rolled_throughput_yield(step_yields: &[f64]) -> f64 {
    step_yields.iter().map(|y| y.clamp(0.0, 1.0)).product()
}

/// Normalised (per-step geometric-mean) yield, `RTY^(1/steps)` — the uniform
/// step yield that would produce the same roll-up. Returns `rty` for
/// `steps == 0`.
pub fn normalized_yield(rty: f64, steps: usize) -> f64 {
    if steps == 0
    {
        return rty;
    }
    rty.clamp(0.0, 1.0).powf(1.0 / steps as f64)
}

/// Process sigma level from a yield, `Φ⁻¹(Y) + shift`. Use `shift = 1.5` for the
/// short-term Six-Sigma convention, `0` for the long-term `Z_bench`. A perfect
/// yield returns `+∞`.
pub fn sigma_from_yield(yield_fraction: f64, shift: f64) -> f64 {
    if yield_fraction >= 1.0
    {
        return f64::INFINITY;
    }
    if yield_fraction <= 0.0
    {
        return f64::NEG_INFINITY;
    }
    inv_normal_cdf(yield_fraction) + shift
}

/// Yield from a process sigma level, `Φ(Z − shift)` — the inverse of
/// [`sigma_from_yield`].
pub fn yield_from_sigma(sigma: f64, shift: f64) -> f64 {
    normal_cdf(sigma - shift)
}

/// Process sigma level from a DPMO figure, `Φ⁻¹(1 − DPMO/10⁶) + shift`.
pub fn sigma_from_dpmo(dpmo: f64, shift: f64) -> f64 {
    sigma_from_yield(1.0 - dpmo.clamp(0.0, 1e6) / 1e6, shift)
}

/// DPMO from a process sigma level, `10⁶·(1 − Φ(Z − shift))` — the inverse of
/// [`sigma_from_dpmo`].
pub fn dpmo_from_sigma(sigma: f64, shift: f64) -> f64 {
    1e6 * (1.0 - yield_from_sigma(sigma, shift))
}

/// A rolled-up yield report for a multi-step process.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProcessReport {
    /// Number of process steps.
    pub steps: usize,
    /// Rolled throughput yield `∏ Yᵢ`.
    pub rolled_throughput_yield: f64,
    /// Normalised per-step yield `RTY^(1/steps)`.
    pub normalized_yield: f64,
    /// Equivalent total defects per unit, `−ln(RTY)`.
    pub total_dpu: f64,
    /// Short-term process sigma level (with the supplied `shift`).
    pub sigma_level: f64,
    /// Rolled-up DPMO treating the whole unit as one opportunity,
    /// `10⁶·(1 − RTY)`.
    pub dpmo: f64,
}

/// Roll a chain of per-step throughput yields into a [`ProcessReport`]. `shift`
/// is the sigma-level shift (`1.5` short-term, `0` long-term). Returns `None`
/// for an empty chain.
pub fn process_report(step_yields: &[f64], shift: f64) -> Option<ProcessReport> {
    if step_yields.is_empty()
    {
        return None;
    }
    let rty = rolled_throughput_yield(step_yields);
    let steps = step_yields.len();
    Some(ProcessReport {
        steps,
        rolled_throughput_yield: rty,
        normalized_yield: normalized_yield(rty, steps),
        total_dpu: -rty.max(f64::MIN_POSITIVE).ln(),
        sigma_level: sigma_from_yield(rty, shift),
        dpmo: 1e6 * (1.0 - rty),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn six_sigma_is_three_point_four_dpmo() {
        // The defining Six-Sigma figure: 6σ short-term ⇒ ≈3.4 DPMO.
        let d = dpmo_from_sigma(6.0, 1.5);
        assert!((d - 3.4).abs() < 0.1, "6σ DPMO = {d}");
        // And the round trip recovers 6.
        assert_relative_eq!(sigma_from_dpmo(d, 1.5), 6.0, epsilon = 1e-9);
    }

    #[test]
    fn yield_sigma_round_trip() {
        for &y in &[0.5, 0.9, 0.99, 0.9973, 0.999_997]
        {
            let s = sigma_from_yield(y, 1.5);
            assert_relative_eq!(yield_from_sigma(s, 1.5), y, epsilon = 1e-9);
        }
        // Long-term (no shift) matches the capability sigma_level convention.
        assert_relative_eq!(sigma_from_yield(normal_cdf(4.0), 0.0), 4.0, epsilon = 1e-9);
    }

    #[test]
    fn rolled_yield_multiplies_and_normalises() {
        // Five identical 99 % steps.
        let steps = vec![0.99; 5];
        let rty = rolled_throughput_yield(&steps);
        assert_relative_eq!(rty, 0.99_f64.powi(5), epsilon = 1e-12);
        // Rolls up below any single step.
        assert!(rty < 0.99);
        // Normalised yield recovers the per-step figure.
        assert_relative_eq!(normalized_yield(rty, 5), 0.99, epsilon = 1e-12);
    }

    #[test]
    fn throughput_yield_matches_dpu() {
        let d = dpu(7.0, 100.0);
        assert_relative_eq!(d, 0.07, epsilon = 1e-12);
        let y = throughput_yield(d);
        // −ln(Y) recovers the DPU.
        assert_relative_eq!(-y.ln(), d, epsilon = 1e-12);
    }

    #[test]
    fn process_report_is_consistent() {
        let r = process_report(&[0.98, 0.995, 0.97, 0.99], 1.5).unwrap();
        assert_eq!(r.steps, 4);
        assert_relative_eq!(
            r.rolled_throughput_yield,
            0.98 * 0.995 * 0.97 * 0.99,
            epsilon = 1e-12
        );
        // total_dpu = −ln(RTY); sigma/DPMO consistent with the yield.
        assert_relative_eq!(
            (-r.total_dpu).exp(),
            r.rolled_throughput_yield,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            r.dpmo,
            1e6 * (1.0 - r.rolled_throughput_yield),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            yield_from_sigma(r.sigma_level, 1.5),
            r.rolled_throughput_yield,
            epsilon = 1e-9
        );
    }
}
