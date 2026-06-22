//! # Expert Iteration (ExIt)
//!
//! Expert Iteration (Anthony et al., 2017 — the algorithm behind AlphaZero-style
//! self-play) alternates two roles:
//!
//! - an **apprentice** — a fast policy that acts directly, and
//! - an **expert** — the same policy *augmented with search* (look-ahead,
//!   sampling, optimisation), which is slower but stronger.
//!
//! Each round, the expert produces improved targets, the apprentice is *distilled*
//! to imitate them, and the now-stronger apprentice makes the *next* expert
//! stronger too. That feedback is the recursion. Adoption is elitist, so the
//! policy never regresses.
//!
//! Implement [`ExpertIterationTask`] and call [`ExpertIteration::run`].

use crate::{Fitness, Guard, Report, StopReason, rng_from_seed};
use rand::rngs::StdRng;

/// A task improvable by expert iteration.
pub trait ExpertIterationTask {
    /// A situation the policy must act in.
    type Sample: Clone;
    /// The fast policy (the apprentice).
    type Policy: Clone;
    /// A training target produced by the expert (e.g. an improved action).
    type Target: Clone;

    /// Sample situations to train on this round.
    fn samples(&self, rng: &mut StdRng) -> Vec<Self::Sample>;

    /// The starting apprentice policy.
    fn base_policy(&self) -> Self::Policy;

    /// The **expert**: improve on the apprentice at `sample` using search /
    /// look-ahead, returning a better target than the bare policy would.
    fn expert(
        &self,
        policy: &Self::Policy,
        sample: &Self::Sample,
        rng: &mut StdRng,
    ) -> Self::Target;

    /// Distil a new apprentice that imitates the expert targets.
    fn distil(&self, base: &Self::Policy, data: &[(Self::Sample, Self::Target)]) -> Self::Policy;

    /// Evaluate a policy. Higher is better.
    fn evaluate(&self, policy: &Self::Policy) -> Fitness;
}

/// Driver for the expert-iteration loop.
#[derive(Debug, Clone)]
pub struct ExpertIteration {
    seed: u64,
}

impl ExpertIteration {
    /// New driver with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Run expert iteration until the guard stops it.
    pub fn run<T: ExpertIterationTask>(&self, task: &T, guard: &Guard) -> (T::Policy, Report) {
        let mut rng = rng_from_seed(self.seed);

        let mut best = task.base_policy();
        let mut best_fit = task.evaluate(&best);

        let mut history = Vec::with_capacity(guard.max_iters);
        let mut accepted = 0usize;
        let mut since_improve = 0usize;
        let mut iterations = 0usize;
        let mut stop_reason = StopReason::MaxIterations;

        for iter in 0..guard.max_iters
        {
            iterations = iter + 1;

            // Expert produces improved targets for a fresh batch of situations.
            let samples = task.samples(&mut rng);
            let mut data = Vec::with_capacity(samples.len());
            for s in samples
            {
                let target = task.expert(&best, &s, &mut rng);
                data.push((s, target));
            }

            // Distil a candidate apprentice and adopt it only if it is better.
            let candidate = task.distil(&best, &data);
            let cand_fit = task.evaluate(&candidate);
            if cand_fit > best_fit + guard.min_delta
            {
                best = candidate;
                best_fit = cand_fit;
                accepted += 1;
                since_improve = 0;
            }
            else
            {
                since_improve += 1;
            }
            history.push(best_fit);

            if let Some(t) = guard.target
            {
                if best_fit >= t
                {
                    stop_reason = StopReason::TargetReached;
                    break;
                }
            }
            if guard.patience > 0 && since_improve >= guard.patience
            {
                stop_reason = StopReason::Converged;
                break;
            }
        }

        (
            best,
            Report {
                iterations,
                accepted,
                best_fitness: best_fit,
                history,
                stop_reason,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Toy control task: the policy is a scalar action `a`; reward is `-(a-π)²`.
    /// The apprentice carries its current best action; the *expert* does a tiny
    /// random search around it (look-ahead), which beats the bare policy and
    /// pulls the distilled apprentice toward the optimum a = π.
    struct AimAtPi;
    impl ExpertIterationTask for AimAtPi {
        type Sample = (); // single-state problem
        type Policy = f64; // current action
        type Target = f64; // improved action

        fn samples(&self, _rng: &mut StdRng) -> Vec<()> {
            vec![(); 4]
        }
        fn base_policy(&self) -> f64 {
            0.0
        }
        fn expert(&self, policy: &f64, _s: &(), rng: &mut StdRng) -> f64 {
            // Search: try a handful of perturbations, return the best.
            let reward = |a: f64| -(a - std::f64::consts::PI).powi(2);
            let mut best_a = *policy;
            let mut best_r = reward(*policy);
            for _ in 0..16
            {
                let a = policy + rng.gen_range(-0.5..0.5);
                let r = reward(a);
                if r > best_r
                {
                    best_r = r;
                    best_a = a;
                }
            }
            best_a
        }
        fn distil(&self, _base: &f64, data: &[((), f64)]) -> f64 {
            // Imitate the expert: average of its targets.
            data.iter().map(|(_, t)| *t).sum::<f64>() / data.len() as f64
        }
        fn evaluate(&self, policy: &f64) -> Fitness {
            -(policy - std::f64::consts::PI).powi(2)
        }
    }

    #[test]
    fn expert_iteration_climbs_to_pi() {
        let (policy, report) =
            ExpertIteration::new(99).run(&AimAtPi, &Guard::new().max_iters(200).target(-1e-4));
        assert!(report.is_monotone());
        assert!(
            (policy - std::f64::consts::PI).abs() < 0.05,
            "policy {policy} should approach π"
        );
        assert!(report.accepted > 0);
    }
}
