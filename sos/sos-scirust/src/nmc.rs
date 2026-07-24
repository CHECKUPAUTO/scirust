//! Seeded nested-Monte-Carlo expected-information-gain estimation for
//! discrete hypothesis discrimination with non-Gaussian likelihoods — gap #1
//! tier 3 of the `sos-scirust` scoping plan, the last of the three EIG tiers
//! SDE §08 §3 names.
//!
//! Tiers 1/2 assume a Gaussian-process surrogate over a continuous latent
//! quantity. This tier is for a genuinely different scenario: a *small,
//! finite* set of competing hypotheses (e.g. candidate count-data models —
//! `scirust-stats`' `Poisson`, `Binomial`, `NegativeBinomial`, ...) that a
//! proposed experiment could discriminate between, each predicting a
//! *different* non-Gaussian distribution over the observation.
//!
//! ## The estimator
//!
//! Expected information gain is the expected reduction in the entropy of the
//! posterior over hypotheses:
//!
//! `EIG = H[prior] − E_y[ H[posterior(· | y)] ]`
//!
//! Because the hypothesis set is small and finite, the *inner* Bayes update
//! is computed **exactly** for every sampled `y` — a finite, `K`-term
//! log-sum-exp, not a second round of resampling. Only the *outer*
//! expectation over `y` is Monte Carlo: `y` is drawn from the true mixture
//! (pick a hypothesis from the prior, then sample its distribution) using
//! `scirust-stats`' seeded [`SplitMix64`], and the estimate's own standard
//! error is the real standard error of that Monte-Carlo mean
//! ([`scirust_stats::describe::std_error`]) — never a fabricated number, per
//! SDE §05 §4.
//!
//! ## Determinism, honestly
//!
//! Like tier 2, this is seeded and stochastic: [`DeterminismLevel::L1`],
//! never `L3`. Unlike tiers 1/2, the standard error here is genuinely
//! non-zero — this is the one tier whose `Estimate.se_milli` is not just
//! formally present but actually informative.

use scirust_stats::describe::{mean, std_error};
use scirust_stats::{DiscreteDistribution, SplitMix64};
use sos_core::{DeterminismLevel, ObjectId};
use sos_planner::{Candidate, Cost, Estimate, MILLIBITS_PER_BIT};

use crate::error::{Result, ScirustError};

/// A nested-Monte-Carlo EIG estimator over a fixed, finite set of hypotheses:
/// a prior over them, and the outer Monte-Carlo sample count.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedMcEigEstimator {
    prior: Vec<f64>,
    n_samples: usize,
}

impl NestedMcEigEstimator {
    /// Build an estimator from an explicit prior over the hypotheses
    /// (normalized internally to sum to `1`) and the outer Monte-Carlo
    /// sample count.
    ///
    /// # Errors
    /// [`ScirustError::InvalidPrior`] if `prior` is empty, has a non-finite
    /// or negative entry, or sums to `0`. [`ScirustError::TooFewSamples`] if
    /// `n_samples < 2` — a standard error needs at least two samples to mean
    /// anything ([`scirust_stats::describe::variance`]'s own documented
    /// floor).
    pub fn new(prior: Vec<f64>, n_samples: usize) -> Result<Self> {
        if prior.is_empty() || prior.iter().any(|p| !p.is_finite() || *p < 0.0)
        {
            return Err(ScirustError::InvalidPrior);
        }
        let sum: f64 = prior.iter().sum();
        if sum <= 0.0
        {
            return Err(ScirustError::InvalidPrior);
        }
        if n_samples < 2
        {
            return Err(ScirustError::TooFewSamples(n_samples));
        }
        let prior = prior.into_iter().map(|p| p / sum).collect();
        Ok(Self { prior, n_samples })
    }

    /// An estimator with a uniform prior over `k` hypotheses.
    ///
    /// # Errors
    /// Same as [`NestedMcEigEstimator::new`] (`k == 0` is
    /// [`ScirustError::InvalidPrior`]).
    pub fn uniform(k: usize, n_samples: usize) -> Result<Self> {
        Self::new(vec![1.0; k], n_samples)
    }

    /// The number of hypotheses this estimator's prior spans.
    #[must_use]
    pub fn hypotheses(&self) -> usize {
        self.prior.len()
    }

    /// Estimate the expected information gain a proposed experiment yields,
    /// given the `K` distributions it would induce — one per hypothesis, in
    /// the same order as the prior — over a shared, discrete observation
    /// space.
    ///
    /// # Errors
    /// [`ScirustError::HypothesisCountMismatch`] if `models.len()` does not
    /// equal the number of hypotheses the prior spans.
    pub fn estimate(&self, models: &[&dyn DiscreteDistribution], seed: u64) -> Result<Estimate> {
        if models.len() != self.prior.len()
        {
            return Err(ScirustError::HypothesisCountMismatch {
                models: models.len(),
                hypotheses: self.prior.len(),
            });
        }

        let ln_prior: Vec<f64> = self.prior.iter().map(|p| p.ln()).collect();
        let prior_entropy = entropy_bits(&self.prior);

        let mut rng = SplitMix64::new(seed);
        let mut posterior_entropies = Vec::with_capacity(self.n_samples);
        for _ in 0..self.n_samples
        {
            let h = sample_categorical(&self.prior, &mut rng);
            let y = models[h].sample(&mut rng);

            let log_joint: Vec<f64> = ln_prior
                .iter()
                .zip(models.iter())
                .map(|(lp, m)| lp + m.ln_pmf(y))
                .collect();
            let log_marginal = log_sum_exp(&log_joint);
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|lj| (lj - log_marginal).exp())
                .collect();
            posterior_entropies.push(entropy_bits(&posterior));
        }

        let eig_bits = prior_entropy - mean(&posterior_entropies);
        let se_bits = std_error(&posterior_entropies);

        let bits_milli = (eig_bits * MILLIBITS_PER_BIT as f64).round() as i64;
        let se_milli = (se_bits * MILLIBITS_PER_BIT as f64).round() as i64;
        Ok(Estimate::new(bits_milli, se_milli, DeterminismLevel::L1))
    }

    /// Estimate and package the result as a real [`Candidate`] for
    /// `experiment`, costed at `cost`.
    ///
    /// # Errors
    /// Same as [`NestedMcEigEstimator::estimate`].
    pub fn candidate(
        &self,
        experiment: ObjectId,
        models: &[&dyn DiscreteDistribution],
        cost: Cost,
        seed: u64,
    ) -> Result<Candidate> {
        Ok(Candidate::new(
            experiment,
            self.estimate(models, seed)?,
            cost,
        ))
    }
}

/// Shannon entropy in bits, `0 log 0 := 0`.
fn entropy_bits(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.log2())
        .sum::<f64>()
}

/// Numerically stable `ln(Σ exp(xs))`.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY
    {
        return f64::NEG_INFINITY;
    }
    m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// Draw a category index from a normalized categorical distribution via
/// inverse-CDF.
fn sample_categorical(probs: &[f64], rng: &mut SplitMix64) -> usize {
    let u = rng.next_f64();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate()
    {
        acc += p;
        if u < acc
        {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use scirust_stats::Poisson;
    use sos_core::{HashAlgo, ObjectId};

    use super::*;

    #[test]
    fn rejects_invalid_prior() {
        assert_eq!(
            NestedMcEigEstimator::new(vec![], 10).unwrap_err(),
            ScirustError::InvalidPrior
        );
        assert_eq!(
            NestedMcEigEstimator::new(vec![0.5, -0.5], 10).unwrap_err(),
            ScirustError::InvalidPrior
        );
        assert_eq!(
            NestedMcEigEstimator::new(vec![0.5, f64::NAN], 10).unwrap_err(),
            ScirustError::InvalidPrior
        );
        assert_eq!(
            NestedMcEigEstimator::new(vec![0.0, 0.0], 10).unwrap_err(),
            ScirustError::InvalidPrior
        );
    }

    #[test]
    fn rejects_too_few_samples() {
        assert_eq!(
            NestedMcEigEstimator::new(vec![0.5, 0.5], 0).unwrap_err(),
            ScirustError::TooFewSamples(0)
        );
        assert_eq!(
            NestedMcEigEstimator::new(vec![0.5, 0.5], 1).unwrap_err(),
            ScirustError::TooFewSamples(1)
        );
    }

    #[test]
    fn uniform_prior_sums_to_one_after_normalization() {
        let est = NestedMcEigEstimator::new(vec![2.0, 2.0, 2.0], 10).unwrap();
        assert_eq!(est.hypotheses(), 3);
        assert!((est.prior.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(est.prior.iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-12));
    }

    #[test]
    fn rejects_hypothesis_count_mismatch() {
        let est = NestedMcEigEstimator::uniform(2, 10).unwrap();
        let a = Poisson::new(1.0);
        let b = Poisson::new(2.0);
        let c = Poisson::new(3.0);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b, &c];
        assert_eq!(
            est.estimate(&models, 1).unwrap_err(),
            ScirustError::HypothesisCountMismatch {
                models: 3,
                hypotheses: 2
            }
        );
    }

    #[test]
    fn identical_hypotheses_give_near_zero_eig() {
        // Every hypothesis predicts the same distribution: observing y can
        // never shift belief about which is true, so EIG should be ~0.
        let est = NestedMcEigEstimator::uniform(3, 2000).unwrap();
        let a = Poisson::new(5.0);
        let b = Poisson::new(5.0);
        let c = Poisson::new(5.0);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b, &c];
        let e = est.estimate(&models, 7).unwrap();
        assert!(
            e.bits_milli.abs() < 50,
            "expected ~0 millibits, got {}",
            e.bits_milli
        );
    }

    #[test]
    fn perfectly_distinguishable_hypotheses_recover_full_prior_entropy() {
        // Practically disjoint supports (mean 0.001 vs mean 500): observing y
        // essentially always reveals which hypothesis is true, so EIG should
        // recover ~all of the 2-way uniform prior's 1 bit of entropy.
        let est = NestedMcEigEstimator::uniform(2, 2000).unwrap();
        let a = Poisson::new(0.001);
        let b = Poisson::new(500.0);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b];
        let e = est.estimate(&models, 11).unwrap();
        assert!(
            e.bits_milli > 950,
            "expected ~1000 millibits (1 bit), got {}",
            e.bits_milli
        );
    }

    #[test]
    fn standard_error_is_real_and_shrinks_with_more_samples() {
        let a = Poisson::new(3.0);
        let b = Poisson::new(6.0);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b];

        let few = NestedMcEigEstimator::uniform(2, 10)
            .unwrap()
            .estimate(&models, 42)
            .unwrap();
        assert!(few.se_milli > 0, "SE should be genuinely non-zero");

        let many = NestedMcEigEstimator::uniform(2, 20_000)
            .unwrap()
            .estimate(&models, 42)
            .unwrap();
        assert!(
            many.se_milli < few.se_milli,
            "more samples should tighten the SE: few={} many={}",
            few.se_milli,
            many.se_milli
        );
    }

    #[test]
    fn is_reproducible_given_the_same_seed() {
        let a = Poisson::new(2.0);
        let b = Poisson::new(9.0);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b];
        let est = NestedMcEigEstimator::uniform(2, 500).unwrap();
        let x = est.estimate(&models, 99).unwrap();
        let y = est.estimate(&models, 99).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn every_estimate_is_l1_never_l3() {
        let a = Poisson::new(4.0);
        let b = Poisson::new(4.5);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b];
        let e = NestedMcEigEstimator::uniform(2, 50)
            .unwrap()
            .estimate(&models, 3)
            .unwrap();
        assert_eq!(e.level, DeterminismLevel::L1);
    }

    #[test]
    fn candidate_packages_cleanly() {
        let a = Poisson::new(1.0);
        let b = Poisson::new(8.0);
        let models: Vec<&dyn DiscreteDistribution> = vec![&a, &b];
        let est = NestedMcEigEstimator::uniform(2, 50).unwrap();
        let id = ObjectId::compute(HashAlgo::default(), b"design", b"nmc-design");
        let candidate = est
            .candidate(id, &models, Cost::new(1, 0, 0, 0), 5)
            .unwrap();
        assert_eq!(candidate.experiment, id);
        assert_eq!(candidate.cost, Cost::new(1, 0, 0, 0));
    }
}
