//! Acceptance sampling **by variables** — ISO 3951 / MIL-STD-414 (Form *k*).
//!
//! Where [`crate::sampling`] accepts a lot on its *inertia*, the classical
//! variables plan accepts on the standardised distance from the sample mean to
//! the specification limit. Measure `n` items, form the quality statistic
//!
//! ```text
//! Q_U = (USL − x̄) / σ        (upper spec)
//! Q_L = (x̄ − LSL) / σ        (lower spec)
//! ```
//!
//! and accept the lot when `Q ≥ k`, the acceptance constant. Because a single
//! measured mean carries far more information than a pass/fail count, a variables
//! plan reaches the same producer/consumer protection as an attributes plan with
//! a **much smaller sample** — the reason it is the workhorse of dimensional
//! inspection.
//!
//! ## Operating characteristic
//!
//! For a normal process with fraction nonconforming `p` beyond the (upper) limit,
//! the quality index is `z_p = −Φ⁻¹(p)` (so `p = Φ(−z_p)`; larger `z_p` ⇒ better
//! quality). With known `σ`, `Q_U ∼ N(z_p, 1/n)`, hence the probability of
//! acceptance is the closed form
//!
//! ```text
//! Pa(p) = Φ( √n_eff · (z_p − k) ) .
//! ```
//!
//! With `σ` **unknown** the sample deviation `s` replaces it; its extra noise
//! inflates the required sample size by `1 + k²/2` (MIL-STD-414), equivalently
//! the plan behaves like a known-`σ` plan at the effective size
//! `n_eff = n / (1 + k²/2)`. [`design_variables_plan`] inverts the OC through the
//! two points `(AQL, 1−α)` and `(RQL, β)` to size `(n, k)`.
//!
//! ## Double specification
//!
//! With both limits present, [`VariablesPlan::max_process_sigma`] returns the
//! largest process spread a *centred* lot can carry and still be accepted,
//! `MSD = (USL − LSL) / (2k)` — the ISO 3951 maximum-sample-standard-deviation
//! idea, and a direct sibling of the inertial budget `I_max = IT/6`.

use crate::special::{inv_normal_cdf, normal_cdf};
use serde::{Deserialize, Serialize};

/// A single-sampling variables plan: measure `sample_size`, accept when the
/// quality statistic `Q = (limit − x̄)/spread` (upper) or `(x̄ − limit)/spread`
/// (lower) is at least `acceptance_constant`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VariablesPlan {
    /// Number of items to measure, `n`.
    pub sample_size: usize,
    /// Acceptance constant `k`.
    pub acceptance_constant: f64,
    /// `true` for the known-`σ` (`σ`) method, `false` for the unknown-`σ` (`s`)
    /// method — the latter uses the sample deviation and a larger `n`.
    pub known_sigma: bool,
}

impl VariablesPlan {
    /// Build a plan from an explicit `(n, k)` and method.
    pub fn new(sample_size: usize, acceptance_constant: f64, known_sigma: bool) -> Self {
        VariablesPlan {
            sample_size,
            acceptance_constant,
            known_sigma,
        }
    }

    /// Effective sample size for the OC: `n` for the known-`σ` method, discounted
    /// by `1 + k²/2` for the unknown-`σ` method.
    fn effective_n(&self) -> f64 {
        let n = self.sample_size as f64;
        if self.known_sigma
        {
            n
        }
        else
        {
            n / (1.0 + self.acceptance_constant * self.acceptance_constant / 2.0)
        }
    }

    /// Probability of accepting a lot whose true fraction nonconforming (beyond a
    /// single limit) is `p`, `Pa(p) = Φ(√n_eff·(z_p − k))` with `z_p = −Φ⁻¹(p)`.
    /// Returns 1 for `p ≤ 0` and 0 for `p ≥ 1`.
    pub fn probability_of_acceptance(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 1.0;
        }
        if p >= 1.0
        {
            return 0.0;
        }
        let z_p = -inv_normal_cdf(p);
        normal_cdf(self.effective_n().sqrt() * (z_p - self.acceptance_constant))
    }

    /// Accept against an **upper** limit: `(usl − mean)/spread ≥ k`. Pass `σ`
    /// (known-`σ` method) or the sample deviation `s` (unknown-`σ` method) as
    /// `spread`. A non-positive `spread` accepts iff the mean is within limit.
    pub fn accepts_upper(&self, mean: f64, spread: f64, usl: f64) -> bool {
        if spread <= 0.0
        {
            return mean <= usl;
        }
        (usl - mean) / spread >= self.acceptance_constant
    }

    /// Accept against a **lower** limit: `(mean − lsl)/spread ≥ k`.
    pub fn accepts_lower(&self, mean: f64, spread: f64, lsl: f64) -> bool {
        if spread <= 0.0
        {
            return mean >= lsl;
        }
        (mean - lsl) / spread >= self.acceptance_constant
    }

    /// Accept against a **double** specification: both one-sided tests must pass.
    pub fn accepts_double(&self, mean: f64, spread: f64, lsl: f64, usl: f64) -> bool {
        self.accepts_lower(mean, spread, lsl) && self.accepts_upper(mean, spread, usl)
    }

    /// Maximum process standard deviation (`MSD`) a **centred** lot may carry and
    /// still be accepted under a double spec: `(usl − lsl)/(2k)`. A process with
    /// `σ` above this is rejected even when perfectly centred — the variables
    /// analogue of the inertial budget `I_max`.
    pub fn max_process_sigma(&self, lsl: f64, usl: f64) -> f64 {
        if self.acceptance_constant <= 0.0 || usl <= lsl
        {
            return 0.0;
        }
        (usl - lsl) / (2.0 * self.acceptance_constant)
    }

    /// The operating-characteristic curve as `points` pairs `(p, Pa(p))` for the
    /// fraction nonconforming `p` swept over `[0, max_p]`.
    pub fn oc_curve(&self, max_p: f64, points: usize) -> Vec<(f64, f64)> {
        if points == 0
        {
            return Vec::new();
        }
        (0..points)
            .map(|i| {
                let p = max_p * i as f64 / (points - 1).max(1) as f64;
                (p, self.probability_of_acceptance(p))
            })
            .collect()
    }
}

/// Design a variables plan whose OC passes (closely) through the producer point
/// `(aql, 1−alpha)` and the consumer point `(rql, beta)`: fractions nonconforming
/// `aql < rql`, risks `alpha`, `beta` in `(0, 1)`. The classical two-point
/// solution is
///
/// ```text
/// √n = (z_{1−α} + z_{1−β}) / (z_aql − z_rql) ,
/// k  = (z_aql·z_{1−β} + z_rql·z_{1−α}) / (z_{1−α} + z_{1−β}) ,
/// ```
///
/// with `z_p = −Φ⁻¹(p)`. For the unknown-`σ` method (`known_sigma = false`) the
/// sample size is inflated by `1 + k²/2`. Returns `None` on out-of-range inputs
/// or if `aql ≥ rql`.
pub fn design_variables_plan(
    aql: f64,
    rql: f64,
    alpha: f64,
    beta: f64,
    known_sigma: bool,
) -> Option<VariablesPlan> {
    if !(0.0..1.0).contains(&aql)
        || !(0.0..1.0).contains(&rql)
        || aql <= 0.0
        || aql >= rql
        || alpha <= 0.0
        || alpha >= 1.0
        || beta <= 0.0
        || beta >= 1.0
    {
        return None;
    }
    let z_aql = -inv_normal_cdf(aql);
    let z_rql = -inv_normal_cdf(rql);
    let z_a = inv_normal_cdf(1.0 - alpha);
    let z_b = inv_normal_cdf(1.0 - beta);
    let denom = z_aql - z_rql;
    if denom <= 0.0
    {
        return None;
    }
    let sqrt_n = (z_a + z_b) / denom;
    let n_sigma = sqrt_n * sqrt_n;
    let k = (z_aql * z_b + z_rql * z_a) / (z_a + z_b);
    let n = if known_sigma
    {
        n_sigma.ceil()
    }
    else
    {
        (n_sigma * (1.0 + k * k / 2.0)).ceil()
    };
    Some(VariablesPlan {
        sample_size: (n as usize).max(2),
        acceptance_constant: k,
        known_sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn oc_passes_through_design_points() {
        let aql = 0.01;
        let rql = 0.08;
        let (alpha, beta) = (0.05, 0.10);
        let plan = design_variables_plan(aql, rql, alpha, beta, true).unwrap();
        // Because n is rounded up, protection is at least as good as requested.
        let pa_aql = plan.probability_of_acceptance(aql);
        let pa_rql = plan.probability_of_acceptance(rql);
        assert!((pa_aql - (1.0 - alpha)).abs() < 0.03, "Pa(AQL) = {pa_aql}");
        assert!((pa_rql - beta).abs() < 0.03, "Pa(RQL) = {pa_rql}");
    }

    #[test]
    fn unknown_sigma_needs_a_bigger_sample() {
        let known = design_variables_plan(0.01, 0.08, 0.05, 0.10, true).unwrap();
        let unknown = design_variables_plan(0.01, 0.08, 0.05, 0.10, false).unwrap();
        // Same acceptance constant, larger sample for the s-method.
        assert_relative_eq!(
            known.acceptance_constant,
            unknown.acceptance_constant,
            epsilon = 1e-12
        );
        assert!(unknown.sample_size > known.sample_size);
    }

    #[test]
    fn oc_is_monotone_decreasing_in_p() {
        let plan = VariablesPlan::new(20, 2.0, true);
        let curve = plan.oc_curve(0.2, 40);
        for w in curve.windows(2)
        {
            assert!(w[1].1 <= w[0].1 + 1e-12);
        }
        // Perfect quality accepted, all-bad rejected.
        assert_relative_eq!(plan.probability_of_acceptance(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(plan.probability_of_acceptance(1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn centred_lot_at_msd_sits_exactly_on_k() {
        let plan = VariablesPlan::new(15, 1.8, true);
        let (lsl, usl) = (10.0, 20.0);
        let msd = plan.max_process_sigma(lsl, usl);
        // A centred lot with σ = MSD lands both statistics exactly on k.
        let mean = 0.5 * (lsl + usl);
        assert_relative_eq!(
            (usl - mean) / msd,
            plan.acceptance_constant,
            epsilon = 1e-12
        );
        // Just above MSD ⇒ rejected even though perfectly centred.
        assert!(!plan.accepts_double(mean, msd * 1.001, lsl, usl));
        assert!(plan.accepts_double(mean, msd * 0.999, lsl, usl));
    }

    #[test]
    fn rejects_bad_design_inputs() {
        assert!(design_variables_plan(0.08, 0.01, 0.05, 0.10, true).is_none()); // aql≥rql
        assert!(design_variables_plan(0.0, 0.08, 0.05, 0.10, true).is_none());
        assert!(design_variables_plan(0.01, 0.08, 0.0, 0.10, true).is_none());
    }
}
