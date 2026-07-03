//! Acceptance sampling by inertia (Pillet & Maire).
//!
//! A batch is accepted when a sample of `n` parts has an estimated inertia
//! `Î ≤ k · I_max`. The rigorous basis: for a sample from `N(μ, σ²)` with
//! `δ = μ − T`, the sum of squared deviations from target satisfies
//!
//! ```text
//! Σ(xᵢ − T)² = n · Î² ,   n·Î²/σ²  ~  χ'²(n, λ),   λ = n·δ²/σ²
//! ```
//!
//! — a **non-central** chi-square with `n` degrees of freedom (`Î²` is the
//! population second moment about target). The probability of accepting a batch
//! whose true state is `(δ, σ)` is therefore
//!
//! ```text
//! P_accept = F_{χ'²}( n·(k·I_max)² / σ² ; n, λ ) .
//! ```
//!
//! Both risks are governed by the **fully-dispersed** split (`δ = 0`,
//! `σ = I`): among all `(δ, σ)` at a fixed inertia `I`, that split maximises
//! the spread of `Î`, so it is simultaneously the worst case for the producer
//! (rejecting a good batch at `I_max`) and the consumer (accepting a bad batch
//! at `I_bad`). Designing at `δ = 0` collapses the non-central law to a central
//! `χ²(n)` and yields a plan that is conservative for every other split.
//!
//! Reference: M. Pillet & E. Maire, *Inertial tolerancing and acceptance
//! sampling* (hal-00834744).

use crate::special::{chi2_cdf, chi2_quantile, ncchi2_cdf};
use serde::{Deserialize, Serialize};

/// A single-sampling plan by inertia: draw `n` parts, accept the batch if the
/// estimated inertia `Î ≤ factor · I_max`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplingPlan {
    /// Sample size `n`.
    pub n: usize,
    /// Acceptance factor `k`: accept when `Î ≤ k · I_max`.
    pub factor: f64,
}

impl SamplingPlan {
    /// A plan with the given sample size and acceptance factor.
    pub fn new(n: usize, factor: f64) -> Self {
        Self { n, factor }
    }

    /// Probability of accepting a batch whose true state is `(off_centering,
    /// sigma)`, given the inertia budget `i_max`.
    ///
    /// Uses the non-central chi-square acceptance law. For `σ = 0` the estimate
    /// is deterministic (`Î = |δ|`), so acceptance is a hard `|δ| ≤ k·I_max`.
    pub fn probability_of_acceptance(&self, i_max: f64, off_centering: f64, sigma: f64) -> f64 {
        let limit = self.factor * i_max; // acceptance limit on Î
        if sigma <= 0.0
        {
            return if off_centering.abs() <= limit
            {
                1.0
            }
            else
            {
                0.0
            };
        }
        let n = self.n.max(1) as f64;
        let lambda = n * off_centering * off_centering / (sigma * sigma);
        let x = n * limit * limit / (sigma * sigma);
        ncchi2_cdf(n, lambda, x)
    }

    /// Probability of accepting a batch of a given true `inertia`, split so that
    /// a fraction `off_centering_ratio ∈ [0, 1]` of `I²` is off-centering and
    /// the rest is dispersion (`δ² = ratio·I²`, `σ² = (1−ratio)·I²`).
    /// `ratio = 0` is the fully-dispersed worst case.
    pub fn probability_of_acceptance_at(
        &self,
        i_max: f64,
        inertia: f64,
        off_centering_ratio: f64,
    ) -> f64 {
        let r = off_centering_ratio.clamp(0.0, 1.0);
        let delta = (r * inertia * inertia).sqrt();
        let sigma = ((1.0 - r) * inertia * inertia).sqrt();
        self.probability_of_acceptance(i_max, delta, sigma)
    }

    /// Operating-characteristic curve: probability of acceptance as the true
    /// inertia ranges over `points` equally-spaced values in
    /// `[0, max_ratio · i_max]`, evaluated at the fully-dispersed worst case
    /// (`off_centering_ratio = 0`). Returns `(inertia, P_accept)` pairs.
    pub fn oc_curve(&self, i_max: f64, max_ratio: f64, points: usize) -> Vec<(f64, f64)> {
        if points == 0
        {
            return Vec::new();
        }
        (0..points)
            .map(|j| {
                let i = max_ratio * i_max * j as f64 / (points - 1).max(1) as f64;
                (i, self.probability_of_acceptance_at(i_max, i, 0.0))
            })
            .collect()
    }
}

/// Design the acceptance factor `k` for a fixed sample size `n` so a batch
/// exactly on the inertia budget (`I = I_max`, fully dispersed) is accepted
/// with probability `1 − alpha` (producer's risk `alpha`):
///
/// ```text
/// k = √( χ²_{n; 1−alpha} / n ) .
/// ```
///
/// This coincides with the piloting chart's upper limit divided by `I_max`.
pub fn plan_for_producer_risk(n: usize, alpha: f64) -> SamplingPlan {
    let nf = n.max(1) as f64;
    let k = (chi2_quantile(nf, 1.0 - alpha) / nf).sqrt();
    SamplingPlan::new(n, k)
}

/// Design a single-sampling plan `(n, k)` meeting a **double** specification:
/// accept a good batch at `I_max` with probability `≥ 1 − alpha` (producer's
/// risk), and accept a bad batch at `ratio_bad · I_max` with probability
/// `≤ beta` (consumer's risk). Both evaluated at the fully-dispersed worst
/// case, so `k = √(χ²_{n;1−alpha}/n)` and the consumer condition is
///
/// ```text
/// F_{χ²}( χ²_{n;1−alpha} / ratio_bad² ; n )  ≤  beta .
/// ```
///
/// Returns the smallest `n ∈ [2, max_n]` (with its `k`) that satisfies the
/// consumer condition, or `None` if none does within `max_n`. Requires
/// `ratio_bad > 1`.
pub fn design_plan(alpha: f64, beta: f64, ratio_bad: f64, max_n: usize) -> Option<SamplingPlan> {
    if ratio_bad <= 1.0
    {
        return None;
    }
    for n in 2..=max_n
    {
        let nf = n as f64;
        let crit = chi2_quantile(nf, 1.0 - alpha); // = n·k²
        let consumer = chi2_cdf(nf, crit / (ratio_bad * ratio_bad));
        if consumer <= beta
        {
            return Some(SamplingPlan::new(n, (crit / nf).sqrt()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn producer_plan_hits_target_acceptance_at_budget() {
        // A batch exactly at I_max (fully dispersed) accepts with prob 1−alpha.
        let plan = plan_for_producer_risk(5, 0.05);
        let p = plan.probability_of_acceptance_at(0.1, 0.1, 0.0);
        assert_relative_eq!(p, 0.95, epsilon = 5e-3);
        // Matches k = √(χ²_{5;0.95}/5).
        let want_k = (crate::special::chi2_quantile(5.0, 0.95) / 5.0).sqrt();
        assert_relative_eq!(plan.factor, want_k, epsilon = 1e-12);
    }

    #[test]
    fn oc_curve_is_monotone_decreasing() {
        let plan = plan_for_producer_risk(10, 0.05);
        let oc = plan.oc_curve(0.1, 3.0, 25);
        assert_eq!(oc.len(), 25);
        // P(accept) starts near 1 (tiny inertia) and decays toward 0.
        assert!(oc[0].1 > 0.99);
        assert!(oc.last().unwrap().1 < 0.02);
        for w in oc.windows(2)
        {
            assert!(w[1].1 <= w[0].1 + 1e-9);
        }
    }

    #[test]
    fn off_centering_split_is_never_worse_than_dispersed() {
        // At fixed inertia, the fully-dispersed split maximises acceptance-prob
        // spread, so it accepts a BAD batch (I > limit) with the highest prob.
        let plan = plan_for_producer_risk(6, 0.05);
        let i_max = 0.1;
        let bad = 0.15;
        let dispersed = plan.probability_of_acceptance_at(i_max, bad, 0.0);
        let centered = plan.probability_of_acceptance_at(i_max, bad, 0.9);
        assert!(dispersed >= centered);
    }

    #[test]
    fn zero_sigma_is_a_hard_threshold() {
        let plan = SamplingPlan::new(5, 1.2);
        // |δ| just under / over the limit k·I_max = 0.12.
        assert_eq!(plan.probability_of_acceptance(0.1, 0.11, 0.0), 1.0);
        assert_eq!(plan.probability_of_acceptance(0.1, 0.13, 0.0), 0.0);
    }

    #[test]
    fn tiny_sigma_is_continuous_with_the_hard_threshold_and_fast() {
        // Regression: σ = 1e-8 drives λ = n·δ²/σ² ≈ 7e14 into ncchi2_cdf; the
        // call must terminate quickly and match the σ→0 limit by continuity.
        let plan = SamplingPlan::new(5, 1.2); // limit = 0.12·? no: k·I_max
        let i_max = 0.1; // limit = 1.2·0.1 = 0.12
        // δ exactly on the limit ⇒ ≈ 0.5; below ⇒ ≈ 1; above ⇒ ≈ 0.
        assert!((plan.probability_of_acceptance(i_max, 0.12, 1e-8) - 0.5).abs() < 0.02);
        assert!(plan.probability_of_acceptance(i_max, 0.11, 1e-8) > 0.999);
        assert!(plan.probability_of_acceptance(i_max, 0.13, 1e-8) < 0.001);
        // A fully-off-centre split (ratio→1) is the same near-degenerate regime.
        assert!(plan.probability_of_acceptance_at(i_max, 0.2, 1.0) < 0.001);
    }

    #[test]
    fn design_plan_meets_both_risks() {
        let alpha = 0.05;
        let beta = 0.10;
        let ratio_bad = 2.0;
        let plan = design_plan(alpha, beta, ratio_bad, 200).expect("a plan should exist");
        // Producer: good batch at I_max accepted with prob ≥ 1−alpha.
        let good = plan.probability_of_acceptance_at(0.1, 0.1, 0.0);
        assert!(good >= 1.0 - alpha - 1e-3);
        // Consumer: bad batch at 2·I_max accepted with prob ≤ beta.
        let bad = plan.probability_of_acceptance_at(0.1, 0.2, 0.0);
        assert!(bad <= beta + 1e-3);
    }

    #[test]
    fn design_plan_rejects_bad_ratio() {
        assert!(design_plan(0.05, 0.1, 0.9, 100).is_none());
    }
}
