//! Rational-subgroup capability study — within vs overall (AIAG / ISO 22514-2).
//!
//! [`crate::capability`] takes one flat sample and computes `Cp`/`Cpk` from its
//! overall spread. A proper capability study instead collects **rational
//! subgroups** (e.g. five parts each hour) and separates two variances:
//!
//! - **within-subgroup** (short-term, common-cause only) — estimated from the
//!   mean subgroup range `R̄` or mean subgroup deviation `s̄` via the control-chart
//!   constants `σ̂_within = R̄/d₂ = s̄/c₄`. It drives the *capability* indices
//!   `Cp`/`Cpk`: what the process **could** do if perfectly centred and stable.
//! - **overall** (long-term, common + special cause) — the ordinary sample
//!   deviation of every reading about the grand mean. It drives the *performance*
//!   indices `Pp`/`Ppk`: what the process **actually** delivered.
//!
//! The gap between them is the footprint of drift and shift between subgroups —
//! the same short-vs-long-term story as [`crate::drift`], but measured from a
//! designed study rather than assumed as a `1.5σ` rule. A large `Cp` with a small
//! `Pp` says the instantaneous process is fine but it wanders; close values say a
//! stable process whose only lever is its inherent spread.

use crate::capability::{cp, cpk};
use serde::{Deserialize, Serialize};

/// Hartley's constant `d₂` (mean of the relative range `W = R/σ`) for subgroup
/// sizes `n = 2..=25`. Returns `None` outside that range.
fn d2(n: usize) -> Option<f64> {
    const D2: [f64; 24] = [
        1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078, 3.173, 3.258, 3.336, 3.407,
        3.472, 3.532, 3.588, 3.640, 3.689, 3.735, 3.778, 3.819, 3.858, 3.895, 3.931,
    ];
    if (2..=25).contains(&n)
    {
        Some(D2[n - 2])
    }
    else
    {
        None
    }
}

/// Unbiasing constant `c₄` (so `E[s] = c₄σ`) for subgroup sizes `n = 2..=25`.
fn c4(n: usize) -> Option<f64> {
    const C4: [f64; 24] = [
        0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594, 0.9650, 0.9693, 0.9727, 0.9754, 0.9776,
        0.9794, 0.9810, 0.9823, 0.9835, 0.9845, 0.9854, 0.9862, 0.9869, 0.9876, 0.9882, 0.9887,
        0.9892, 0.9896,
    ];
    if (2..=25).contains(&n)
    {
        Some(C4[n - 2])
    }
    else
    {
        None
    }
}

/// A rational-subgroup capability study: within- and overall-spread estimates and
/// the capability (`Cp`/`Cpk`, short-term) vs performance (`Pp`/`Ppk`, long-term)
/// index pairs they produce.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SubgroupCapability {
    /// Grand mean of every reading.
    pub grand_mean: f64,
    /// Within-subgroup deviation `σ̂_within = R̄/d₂` (range method).
    pub sigma_within: f64,
    /// Overall deviation `σ̂_overall` (all readings about the grand mean).
    pub sigma_overall: f64,
    /// Mean subgroup range `R̄`.
    pub mean_range: f64,
    /// `Cp` from the within-subgroup spread (potential capability).
    pub cp: f64,
    /// `Cpk` from the within-subgroup spread (actual capability).
    pub cpk: f64,
    /// `Pp` from the overall spread (potential performance).
    pub pp: f64,
    /// `Ppk` from the overall spread (actual performance).
    pub ppk: f64,
}

/// Run a capability study over balanced rational subgroups `subgroups[i]` against
/// the bilateral specification `[lsl, usl]`. Every subgroup must have the same
/// size `m ∈ [2, 25]`, and there must be at least two subgroups. The
/// within-subgroup spread uses the range method `σ̂_within = R̄/d₂`; the overall
/// spread is the sample deviation of all `N` readings about the grand mean.
///
/// Returns `None` on an unbalanced design, a subgroup size outside `[2, 25]`,
/// fewer than two subgroups, or `usl ≤ lsl`.
pub fn subgroup_capability(
    subgroups: &[Vec<f64>],
    lsl: f64,
    usl: f64,
) -> Option<SubgroupCapability> {
    if subgroups.len() < 2 || usl <= lsl
    {
        return None;
    }
    let m = subgroups[0].len();
    let d2m = d2(m)?;
    if subgroups.iter().any(|g| g.len() != m)
    {
        return None;
    }

    let k = subgroups.len();
    let n_total = (k * m) as f64;
    let grand_mean = subgroups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total;

    // Mean subgroup range ⇒ within-subgroup sigma.
    let mean_range = subgroups
        .iter()
        .map(|g| {
            let (mut lo, mut hi) = (g[0], g[0]);
            for &x in g
            {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            hi - lo
        })
        .sum::<f64>()
        / k as f64;
    let sigma_within = mean_range / d2m;

    // Overall sigma: sample deviation of all readings about the grand mean.
    let sse: f64 = subgroups
        .iter()
        .flat_map(|g| g.iter())
        .map(|&x| (x - grand_mean).powi(2))
        .sum();
    let sigma_overall = (sse / (n_total - 1.0)).sqrt();

    Some(SubgroupCapability {
        grand_mean,
        sigma_within,
        sigma_overall,
        mean_range,
        cp: cp(sigma_within, lsl, usl),
        cpk: cpk(grand_mean, sigma_within, lsl, usl),
        pp: cp(sigma_overall, lsl, usl),
        ppk: cpk(grand_mean, sigma_overall, lsl, usl),
    })
}

/// Estimate the within-subgroup deviation from the mean subgroup deviation via
/// the `s` method, `σ̂_within = s̄/c₄`, where `s̄` is the mean of the per-subgroup
/// sample deviations (divisor `m − 1`). An independent alternative to the range
/// method used by [`subgroup_capability`]. Returns `None` for a size outside
/// `[2, 25]` or an unbalanced design.
pub fn sigma_within_s_method(subgroups: &[Vec<f64>]) -> Option<f64> {
    let m = subgroups.first()?.len();
    let c4m = c4(m)?;
    if subgroups.len() < 2 || subgroups.iter().any(|g| g.len() != m)
    {
        return None;
    }
    let s_bar = subgroups
        .iter()
        .map(|g| {
            let mean = g.iter().sum::<f64>() / m as f64;
            let var = g.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
            var.sqrt()
        })
        .sum::<f64>()
        / subgroups.len() as f64;
    Some(s_bar / c4m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cp_uses_within_and_pp_uses_overall() {
        let subgroups = vec![
            vec![10.0, 10.1, 9.9, 10.05],
            vec![10.2, 10.15, 10.25, 10.1], // shifted up ⇒ inflates overall spread
            vec![9.8, 9.85, 9.75, 9.9],
        ];
        let s = subgroup_capability(&subgroups, 9.0, 11.0).unwrap();
        // Between-subgroup shift ⇒ overall spread exceeds within ⇒ Pp < Cp.
        assert!(s.sigma_overall > s.sigma_within);
        assert!(s.pp < s.cp);
        // Cp identity from the within spread.
        assert_relative_eq!(s.cp, (11.0 - 9.0) / (6.0 * s.sigma_within), epsilon = 1e-12);
    }

    #[test]
    fn range_and_s_methods_agree_on_within_sigma() {
        // Two estimators of the same within-subgroup sigma should be close.
        let subgroups = vec![
            vec![5.01, 4.98, 5.02, 5.00, 4.99],
            vec![5.03, 5.00, 4.97, 5.01, 5.02],
            vec![4.98, 5.02, 5.00, 4.99, 5.01],
            vec![5.00, 5.01, 4.99, 5.02, 4.98],
        ];
        let s = subgroup_capability(&subgroups, 4.9, 5.1).unwrap();
        let s_method = sigma_within_s_method(&subgroups).unwrap();
        assert_relative_eq!(s.sigma_within, s_method, epsilon = 0.15 * s.sigma_within);
    }

    #[test]
    fn rejects_unbalanced_or_out_of_range() {
        // One subgroup ⇒ None.
        assert!(subgroup_capability(&[vec![1.0, 2.0]], 0.0, 3.0).is_none());
        // Ragged sizes ⇒ None.
        assert!(subgroup_capability(&[vec![1.0, 2.0], vec![1.0]], 0.0, 3.0).is_none());
        // Subgroup size 1 (out of [2,25]) ⇒ None.
        assert!(subgroup_capability(&[vec![1.0], vec![2.0]], 0.0, 3.0).is_none());
        // usl ≤ lsl ⇒ None.
        assert!(subgroup_capability(&[vec![1.0, 2.0], vec![1.5, 2.5]], 3.0, 3.0).is_none());
    }

    #[test]
    fn stable_process_has_close_cp_and_pp() {
        // No between-subgroup shift ⇒ within ≈ overall ⇒ Cp ≈ Pp.
        let subgroups = vec![
            vec![0.0, 1.0, -1.0, 0.5, -0.5],
            vec![0.5, -0.5, 1.0, -1.0, 0.0],
            vec![-1.0, 0.5, 0.0, 1.0, -0.5],
            vec![1.0, 0.0, -0.5, -1.0, 0.5],
        ];
        let s = subgroup_capability(&subgroups, -6.0, 6.0).unwrap();
        assert_relative_eq!(s.cp, s.pp, epsilon = 0.25 * s.cp);
    }
}
