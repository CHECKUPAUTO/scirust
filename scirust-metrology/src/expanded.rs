//! Expanded uncertainty — the reporting step of the GUM.
//!
//! [`crate::gum`] gives the combined standard uncertainty `u_c`. To state a
//! result as `y ± U` with a stated **coverage probability** `p` (usually 95 %),
//! the GUM multiplies `u_c` by a coverage factor `k`:
//!
//! ```text
//! U = k·u_c ,     k = t_{(1+p)/2}(ν_eff) ,
//! ```
//!
//! the Student-`t` quantile at the **effective degrees of freedom** given by the
//! Welch–Satterthwaite formula from each component's contribution `uᵢ` and its
//! own degrees of freedom `νᵢ`:
//!
//! ```text
//! ν_eff = u_c⁴ / Σ (uᵢ⁴ / νᵢ) ,     u_c² = Σ uᵢ² .
//! ```
//!
//! A Type B component believed exact takes `νᵢ = ∞` (no contribution to the
//! denominator), and as `ν_eff → ∞` the factor tends to the normal quantile
//! (`k → 1.96` at 95 %). The `t`-quantile uses a Cornish–Fisher expansion, very
//! accurate for the `ν_eff ≳ 3` typical of an uncertainty budget.

use serde::{Deserialize, Serialize};

/// Standard-normal quantile `Φ⁻¹(p)` (Acklam's rational approximation,
/// `|error| < 1.15e-9`).
fn inv_normal(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow
    {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
    else if p <= phigh
    {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
    else
    {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Student-`t` quantile `t_p(ν)` by the Cornish–Fisher expansion about the normal
/// quantile — accurate for `ν ≳ 3`, exact in the `ν → ∞` limit.
pub fn t_quantile(p: f64, nu: f64) -> f64 {
    let z = inv_normal(p);
    if !nu.is_finite() || nu > 1e8
    {
        return z;
    }
    let z3 = z.powi(3);
    let z5 = z.powi(5);
    let z7 = z.powi(7);
    let z9 = z.powi(9);
    let g1 = (z3 + z) / 4.0;
    let g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0;
    let g3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0;
    let g4 = (79.0 * z9 + 776.0 * z7 + 1482.0 * z5 - 1920.0 * z3 - 945.0 * z) / 92160.0;
    z + g1 / nu + g2 / nu.powi(2) + g3 / nu.powi(3) + g4 / nu.powi(4)
}

/// Welch–Satterthwaite effective degrees of freedom from the uncertainty
/// components `(uᵢ, νᵢ)` (each contribution and its degrees of freedom;
/// `νᵢ = f64::INFINITY` for an assumed-exact Type B term). Returns `+∞` when no
/// component carries finite degrees of freedom.
pub fn effective_dof(components: &[(f64, f64)]) -> f64 {
    let uc2: f64 = components.iter().map(|&(u, _)| u * u).sum();
    let denom: f64 = components
        .iter()
        .filter(|&&(_, nu)| nu.is_finite() && nu > 0.0)
        .map(|&(u, nu)| u.powi(4) / nu)
        .sum();
    if denom <= 0.0
    {
        return f64::INFINITY;
    }
    (uc2 * uc2) / denom
}

/// Coverage factor `k = t_{(1+p)/2}(ν_eff)` for a two-sided coverage probability
/// `coverage_prob` (e.g. 0.95).
pub fn coverage_factor(nu_eff: f64, coverage_prob: f64) -> f64 {
    t_quantile(0.5 + 0.5 * coverage_prob, nu_eff)
}

/// A GUM expanded-uncertainty statement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpandedUncertainty {
    /// Combined standard uncertainty `u_c = √(Σ uᵢ²)`.
    pub combined: f64,
    /// Effective degrees of freedom `ν_eff`.
    pub effective_dof: f64,
    /// Coverage factor `k`.
    pub coverage_factor: f64,
    /// Expanded uncertainty `U = k·u_c`.
    pub expanded: f64,
    /// Lower coverage limit `value − U`.
    pub lower: f64,
    /// Upper coverage limit `value + U`.
    pub upper: f64,
}

/// Full expanded-uncertainty statement for a measured `value` from its
/// uncertainty budget `components` (`(uᵢ, νᵢ)`) at coverage probability
/// `coverage_prob`.
pub fn expanded_uncertainty(
    value: f64,
    components: &[(f64, f64)],
    coverage_prob: f64,
) -> ExpandedUncertainty {
    let combined = components.iter().map(|&(u, _)| u * u).sum::<f64>().sqrt();
    let nu_eff = effective_dof(components);
    let k = coverage_factor(nu_eff, coverage_prob);
    let u = k * combined;
    ExpandedUncertainty {
        combined,
        effective_dof: nu_eff,
        coverage_factor: k,
        expanded: u,
        lower: value - u,
        upper: value + u,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_quantile_matches_tables() {
        // Student-t 0.975 quantiles (two-sided 95 %).
        assert!((t_quantile(0.975, 10.0) - 2.2281).abs() < 5e-3);
        assert!((t_quantile(0.975, 20.0) - 2.0860).abs() < 5e-3);
        assert!((t_quantile(0.975, 30.0) - 2.0423).abs() < 5e-3);
        assert!((t_quantile(0.975, 60.0) - 2.0003).abs() < 5e-3);
        // ν → ∞ tends to the normal quantile.
        assert!((t_quantile(0.975, f64::INFINITY) - 1.95996).abs() < 1e-4);
        assert!((t_quantile(0.975, 1e9) - 1.95996).abs() < 1e-3);
    }

    #[test]
    fn welch_satterthwaite_effective_dof() {
        // u=[0.3 (ν=5), 0.4 (exact)] ⇒ u_c=0.5, ν_eff = 0.5⁴/(0.3⁴/5) = 38.58.
        let nu = effective_dof(&[(0.3, 5.0), (0.4, f64::INFINITY)]);
        assert!((nu - 38.58).abs() < 0.1, "nu_eff {nu}");
        // All-exact budget ⇒ infinite dof.
        assert!(effective_dof(&[(0.1, f64::INFINITY), (0.2, f64::INFINITY)]).is_infinite());
    }

    #[test]
    fn expanded_uncertainty_is_k_times_uc() {
        let e = expanded_uncertainty(100.0, &[(0.3, 5.0), (0.4, f64::INFINITY)], 0.95);
        assert!((e.combined - 0.5).abs() < 1e-12);
        // k ≈ t_0.975(38.6) ≈ 2.02.
        assert!(
            (e.coverage_factor - 2.02).abs() < 0.03,
            "k {}",
            e.coverage_factor
        );
        assert!((e.expanded - e.coverage_factor * e.combined).abs() < 1e-12);
        assert!((e.upper - e.lower - 2.0 * e.expanded).abs() < 1e-12);
        assert!(e.lower < 100.0 && e.upper > 100.0);
    }

    #[test]
    fn large_dof_gives_k_near_two() {
        // Many-degrees-of-freedom budget ⇒ k ≈ 1.96 at 95 %.
        let e = expanded_uncertainty(0.0, &[(1.0, 1000.0)], 0.95);
        assert!(
            (e.coverage_factor - 1.96).abs() < 0.01,
            "k {}",
            e.coverage_factor
        );
    }
}
