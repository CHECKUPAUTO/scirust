//! Closed-form expected-information-gain estimates for a Gaussian-process
//! surrogate — the cheap, exact fast path of the `sos-scirust` EIG bridge
//! (SDE §08 §3; the `sos-scirust` scoping plan's gap #1, tier 1).
//!
//! `sos-planner` deliberately consumes [`Estimate`]s without computing them
//! ([`sos_planner`] crate docs); this module is the real computation those
//! estimates can now come from, wrapping [`scirust_gp::GaussianProcess`]
//! rather than duplicating its linear algebra (Invariant VIII).
//!
//! ## What is deliberately not here yet
//!
//! The Bayesian-optimization search loop over a continuous design box
//! (`scirust-automl::bayesian_optimize`) and the nested-Monte-Carlo fallback
//! for discrete, non-Gaussian hypothesis discrimination
//! (`scirust-stats::SplitMix64`) are separate tiers of the same gap,
//! addressing different situations — a follow-on increment, not a stub.
//! Likewise, resolving *which* estimator to use through `sos-registry` is
//! deferred: `sos-scirust` is the "Static Rust... the default" transport
//! (RFC-0002 §10 §1), so direct, in-process construction is the expected
//! shape for now.

use scirust_gp::{GaussianProcess, Kernel};
use sos_core::ObjectId;
use sos_planner::{Candidate, Cost, Estimate, MILLIBITS_PER_BIT};

use crate::error::{Result, ScirustError};

/// Wraps a fitted [`GaussianProcess`] to produce real [`Estimate`]s: the
/// closed-form mutual information between a candidate design's latent
/// function value and a hypothetical noisy observation of it,
///
/// `I = 0.5 · log2(1 + var / noise)` bits,
///
/// the standard Gaussian-channel mutual-information formula. Because it is
/// analytic in the GP's own posterior variance — no sampling — every
/// [`Estimate`] this produces is exact: [`DeterminismLevel::L3`] with zero
/// standard error ([`Estimate::exact`]).
///
/// [`DeterminismLevel::L3`]: sos_core::DeterminismLevel::L3
#[derive(Debug, Clone)]
pub struct GpEigEstimator<K: Kernel> {
    gp: GaussianProcess<K>,
    observation_noise: f64,
}

impl<K: Kernel> GpEigEstimator<K> {
    /// Wrap a fitted GP, declaring the observation noise variance a
    /// hypothetical new sample at a candidate design would carry. This need
    /// not equal the GP's own fit noise — a proposed experiment can use a
    /// differently-instrumented measurement.
    ///
    /// # Errors
    /// [`ScirustError::NonPositiveNoise`] if `observation_noise` is not
    /// finite and strictly positive (this rejects NaN, infinity, zero, and
    /// negative values alike).
    pub fn new(gp: GaussianProcess<K>, observation_noise: f64) -> Result<Self> {
        if !observation_noise.is_finite() || observation_noise <= 0.0
        {
            return Err(ScirustError::NonPositiveNoise(observation_noise));
        }
        Ok(Self {
            gp,
            observation_noise,
        })
    }

    /// The expected information gain at candidate design `x`.
    #[must_use]
    pub fn estimate(&self, x: &[f64]) -> Estimate {
        let (_mean, var) = self.gp.predict(x);
        let bits = 0.5 * (1.0 + var / self.observation_noise).ln() / std::f64::consts::LN_2;
        let bits_milli = (bits * MILLIBITS_PER_BIT as f64).round() as i64;
        Estimate::exact(bits_milli)
    }

    /// The expected information gain at each of `xs`, in order. Equivalent to
    /// calling [`GpEigEstimator::estimate`] on each row, batched as
    /// [`GaussianProcess::predict_many`] batches predictions.
    #[must_use]
    pub fn estimate_many(&self, xs: &[Vec<f64>]) -> Vec<Estimate> {
        xs.iter().map(|x| self.estimate(x)).collect()
    }

    /// Build a real [`Candidate`] for `experiment` at design point `x`,
    /// costed at `cost` — the bridge's actual deliverable: a `Candidate`
    /// [`sos_planner::Planner::recommend`] can rank, backed by a real GP
    /// posterior rather than a consumer-supplied number.
    #[must_use]
    pub fn candidate(&self, experiment: ObjectId, x: &[f64], cost: Cost) -> Candidate {
        Candidate::new(experiment, self.estimate(x), cost)
    }
}

#[cfg(test)]
mod tests {
    use scirust_gp::Rbf;
    use sos_core::{HashAlgo, ObjectId};
    use sos_planner::{GreedyPlanner, Planner, StopVerdict, UtilityPolicy};

    use super::*;

    fn fit(x: &[Vec<f64>], y: &[f64]) -> GaussianProcess<Rbf> {
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        GaussianProcess::fit(x, y, kernel, 1e-6).unwrap()
    }

    #[test]
    fn rejects_non_positive_noise() {
        let gp = fit(&[vec![0.0], vec![1.0]], &[0.0, 1.0]);
        assert_eq!(
            GpEigEstimator::new(gp.clone(), 0.0).unwrap_err(),
            ScirustError::NonPositiveNoise(0.0)
        );
        assert_eq!(
            GpEigEstimator::new(gp.clone(), -1.0).unwrap_err(),
            ScirustError::NonPositiveNoise(-1.0)
        );
        // NaN/infinity fail the strict `> 0.0` intent too, and never compare
        // equal to themselves, so match the variant rather than assert_eq!.
        assert!(matches!(
            GpEigEstimator::new(gp.clone(), f64::NAN).unwrap_err(),
            ScirustError::NonPositiveNoise(n) if n.is_nan()
        ));
        assert_eq!(
            GpEigEstimator::new(gp, f64::INFINITY).unwrap_err(),
            ScirustError::NonPositiveNoise(f64::INFINITY)
        );
    }

    #[test]
    fn estimate_is_exact_and_zero_error() {
        let gp = fit(&[vec![0.0], vec![1.0], vec![2.0]], &[0.0, 1.0, 0.0]);
        let est = GpEigEstimator::new(gp, 0.1).unwrap();
        let e = est.estimate(&[0.5]);
        assert_eq!(e.se_milli, 0);
        assert_eq!(e.level, sos_core::DeterminismLevel::L3);
    }

    #[test]
    fn higher_variance_yields_higher_eig() {
        // Training data only informs x in [0, 2]; a point far outside has
        // higher posterior variance, hence more to learn from observing it.
        let gp = fit(&[vec![0.0], vec![1.0], vec![2.0]], &[0.0, 1.0, 0.0]);
        let est = GpEigEstimator::new(gp, 0.1).unwrap();

        let near = est.estimate(&[1.0]); // at a training point: low variance
        let far = est.estimate(&[50.0]); // far away: variance ~ prior
        assert!(
            far.bits_milli > near.bits_milli,
            "far={} near={}",
            far.bits_milli,
            near.bits_milli
        );
    }

    #[test]
    fn zero_variance_at_training_point_gives_zero_eig() {
        // Noiseless interpolation: at a training point posterior variance is
        // ~0, so there is (almost) nothing left to learn from observing it.
        let gp = fit(&[vec![0.0], vec![1.0]], &[0.0, 1.0]);
        let est = GpEigEstimator::new(gp, 1.0).unwrap();
        let e = est.estimate(&[0.0]);
        assert!(
            e.bits_milli.abs() < 5,
            "expected ~0 milibits, got {}",
            e.bits_milli
        );
    }

    #[test]
    fn estimate_many_matches_estimate() {
        let gp = fit(&[vec![0.0], vec![1.0]], &[0.0, 1.0]);
        let est = GpEigEstimator::new(gp, 0.2).unwrap();
        let xs = vec![vec![0.5], vec![5.0]];
        let batched = est.estimate_many(&xs);
        let individual: Vec<_> = xs.iter().map(|x| est.estimate(x)).collect();
        assert_eq!(batched, individual);
    }

    #[test]
    fn candidate_flows_into_a_real_plan() {
        // The actual deliverable: real GP-backed candidates ranked by the
        // unmodified sos-planner machinery — no change to Planner/Candidate.
        let gp = fit(&[vec![0.0], vec![1.0], vec![2.0]], &[0.0, 1.5, 0.0]);
        let est = GpEigEstimator::new(gp, 0.05).unwrap();

        let design = |tag: &[u8]| ObjectId::compute(HashAlgo::default(), b"design", tag);
        let near = est.candidate(design(b"near"), &[1.0], Cost::new(1, 0, 0, 0));
        let far = est.candidate(design(b"far"), &[100.0], Cost::new(1, 0, 0, 0));

        let plan = GreedyPlanner::new()
            .recommend(&[near, far], UtilityPolicy::EigPerCost, 1)
            .unwrap();
        // The far design has far higher posterior variance (hence EIG) at
        // equal cost, so it should win.
        assert_eq!(plan.verdict, StopVerdict::Recommend(far.experiment));
    }
}
