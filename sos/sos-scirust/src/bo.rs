//! Bayesian-optimization design search over a continuous design box — gap #1
//! tier 2 of the `sos-scirust` scoping plan, the search-based counterpart to
//! tier 1's closed-form ranking of a pre-enumerated candidate set.
//!
//! `sos-planner`'s [`Planner::recommend`](sos_planner::Planner::recommend)
//! ranks a *discrete* set of `Candidate`s; this module answers a different
//! question — "what's the best next design in this whole continuous box?" —
//! by reusing `scirust-automl`'s seeded `bayesian_optimize`/
//! `expected_improvement` loop to maximize `sos-planner`'s own
//! [`UtilityPolicy::utility`], treating tier 1's closed-form
//! [`GpEigEstimator::estimate`] (at a caller-supplied cost) as the black-box
//! objective. `sos-planner` gains no new code for this either — `UtilityPolicy`
//! is reused exactly as ranking already uses it.
//!
//! ## Determinism, honestly
//!
//! Unlike tier 1's per-point `estimate`, this *search* is seeded and
//! stochastic (`scirust-automl`'s internal `XorShift64`, per its own
//! `bayesian_optimize` signature). SDE §08 §6 classifies `automl` as
//! [`DeterminismLevel::L1`], not `L3`, and this module honors that
//! classification even though the EIG *value* at the returned point is itself
//! exact: `meet(L3, L1) = L1`, because *which* point the search returns is a
//! function of the seed. The seed is always caller-supplied (never generated
//! internally) and echoed back in [`BoResult`] for `ReproMeta`.

use scirust_automl::{XorShift64, bayesian_optimize};
use scirust_gp::Kernel;
use sos_core::{DeterminismLevel, ObjectId};
use sos_planner::{Candidate, Cost, Estimate, UtilityPolicy};

use crate::eig::GpEigEstimator;

/// The outcome of a continuous-design-box Bayesian-optimization search: the
/// best design point found, its EIG, its cost, the resulting utility, and the
/// seed that makes the search itself reproducible.
#[derive(Debug, Clone, PartialEq)]
pub struct BoResult {
    /// The best design point found in the box.
    pub x: Vec<f64>,
    /// Its expected information gain at `x` — exact given `x`, but
    /// [`DeterminismLevel::L1`] overall since *which* `x` was returned is a
    /// function of the seeded search (`se_milli` is still `0`: conditional on
    /// `x`, the closed-form value has no sampling error of its own).
    pub eig: Estimate,
    /// Its cost, from the caller's cost function.
    pub cost: Cost,
    /// Its fixed-point utility under the policy the search maximized.
    pub utility: i64,
    /// The seed that reproduces this search.
    pub seed: u64,
}

impl BoResult {
    /// Package this result as a real [`Candidate`] for `experiment`.
    #[must_use]
    pub fn candidate(&self, experiment: ObjectId) -> Candidate {
        Candidate::new(experiment, self.eig, self.cost)
    }
}

impl<K: Kernel> GpEigEstimator<K> {
    /// Search a continuous design box `bounds` (one `(lo, hi)` pair per
    /// dimension — must match the fitted GP's own training dimensionality,
    /// the same precondition [`scirust_gp::GaussianProcess::predict`] itself
    /// carries) for the design maximizing `policy.utility(eig, cost)`, via
    /// `scirust-automl`'s seeded Bayesian-optimization loop.
    ///
    /// `cost_fn` gives the (caller-defined) cost of running the experiment at
    /// a candidate design point. `max_iter` is the total evaluation budget
    /// (including `initial_samples`); `initial_samples` is how many of those
    /// are uniform-random warm-start evaluations before the
    /// expected-improvement-guided loop takes over.
    ///
    /// [`UtilityPolicy::EigPerCost`] is everywhere finite and is the natural
    /// fit for a continuous, gradient-guided search;
    /// [`UtilityPolicy::EigBudgeted`]'s hard exclusion of over-budget designs
    /// makes the utility surface discontinuous, which the numerical-gradient
    /// ascent inside `bayesian_optimize` is not designed to handle
    /// gracefully — prefer it for tier 1's discrete ranking instead.
    #[must_use]
    pub fn search_best_design(
        &self,
        bounds: &[(f64, f64)],
        cost_fn: &dyn Fn(&[f64]) -> Cost,
        policy: UtilityPolicy,
        max_iter: usize,
        initial_samples: usize,
        seed: u64,
    ) -> BoResult {
        let mut objective = |x: &[f64]| -> f64 {
            let eig = self.estimate(x);
            let cost = cost_fn(x);
            policy.utility(&eig, &cost) as f64
        };
        let mut rng = XorShift64::new(seed);
        let (x, _) = bayesian_optimize(&mut objective, bounds, max_iter, initial_samples, &mut rng);

        // Recompute the exact, canonical eig/cost/utility at the returned
        // point rather than trust the search's own float bookkeeping.
        let at_x = self.estimate(&x);
        let cost = cost_fn(&x);
        let utility = policy.utility(&at_x, &cost);
        let eig = Estimate {
            level: DeterminismLevel::L1,
            ..at_x
        };

        BoResult {
            x,
            eig,
            cost,
            utility,
            seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use scirust_gp::{GaussianProcess, Rbf};
    use sos_core::{HashAlgo, ObjectId};

    use super::*;

    fn sine_gp() -> GaussianProcess<Rbf> {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![f64::from(i) * 0.5]).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi[0].sin()).collect();
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        GaussianProcess::fit(&x, &y, kernel, 1e-4).unwrap()
    }

    fn unit_cost(_x: &[f64]) -> Cost {
        Cost::new(1, 0, 0, 0)
    }

    #[test]
    fn search_is_reproducible_given_the_same_seed() {
        let est = GpEigEstimator::new(sine_gp(), 0.05).unwrap();
        let bounds = [(-10.0, 20.0)];

        let a = est.search_best_design(&bounds, &unit_cost, UtilityPolicy::EigPerCost, 30, 8, 7);
        let b = est.search_best_design(&bounds, &unit_cost, UtilityPolicy::EigPerCost, 30, 8, 7);
        assert_eq!(a, b, "identical seed must give a bit-identical result");
    }

    #[test]
    fn search_favors_the_unexplored_side_of_the_box() {
        // Training data covers [0, 3.5]; a box biased entirely to the
        // unexplored side should settle on a point with real, substantial
        // EIG rather than one on top of the training data.
        let est = GpEigEstimator::new(sine_gp(), 0.05).unwrap();
        let bounds = [(10.0, 20.0)];

        let result =
            est.search_best_design(&bounds, &unit_cost, UtilityPolicy::EigPerCost, 40, 10, 42);
        assert!(
            result.x[0] >= 10.0 && result.x[0] <= 20.0,
            "must stay in bounds"
        );
        assert!(
            result.eig.bits_milli > 500,
            "expected a substantial EIG far from training data, got {}",
            result.eig.bits_milli
        );
    }

    #[test]
    fn result_is_l1_even_though_the_value_is_exact() {
        let est = GpEigEstimator::new(sine_gp(), 0.05).unwrap();
        let bounds = [(-5.0, 15.0)];
        let result =
            est.search_best_design(&bounds, &unit_cost, UtilityPolicy::EigPerCost, 20, 5, 1);
        assert_eq!(result.eig.level, sos_core::DeterminismLevel::L1);
        assert_eq!(result.eig.se_milli, 0);
        assert_eq!(result.seed, 1);
    }

    #[test]
    fn candidate_packages_cleanly() {
        let est = GpEigEstimator::new(sine_gp(), 0.05).unwrap();
        let bounds = [(-5.0, 15.0)];
        let result =
            est.search_best_design(&bounds, &unit_cost, UtilityPolicy::EigPerCost, 20, 5, 1);
        let id = ObjectId::compute(HashAlgo::default(), b"design", b"bo-best");
        let candidate = result.candidate(id);
        assert_eq!(candidate.experiment, id);
        assert_eq!(candidate.eig, result.eig);
        assert_eq!(candidate.cost, result.cost);
    }
}
