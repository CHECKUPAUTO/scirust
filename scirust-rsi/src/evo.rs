//! # (1+λ)-ES with Rechenberg's 1/5 success rule
//!
//! A self-adapting evolution strategy. The optimiser keeps one parent, samples
//! `λ` Gaussian offspring around it with step size `σ`, and keeps the best (an
//! elitist `(1+λ)` selection, so the parent never gets worse).
//!
//! The *self-improvement* here is at the meta level: the optimiser **tunes its
//! own σ** with Rechenberg's 1/5 rule — if more than 1/5 of recent generations
//! improved the parent, σ grows (be bolder); otherwise σ shrinks (refine). It
//! is the simplest classical algorithm that adapts its own search parameters,
//! and it has provable convergence on the sphere.

use crate::{Fitness, Guard, Report, StopReason, rng_from_seed};
use rand_distr::{Distribution, Normal};

/// Self-adapting `(1+λ)` evolution strategy.
#[derive(Debug, Clone)]
pub struct OnePlusLambda {
    seed: u64,
    lambda: usize,
    sigma0: f64,
    /// Multiplicative factor for the 1/5 rule (0 < c < 1). σ grows by `1/c` on
    /// success and shrinks by `c` otherwise.
    c: f64,
    /// Generations between σ updates (the success-rate measurement window).
    window: usize,
}

impl OnePlusLambda {
    /// New strategy with the given seed and defaults (λ=10, σ₀=1.0, c=0.85).
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            lambda: 10,
            sigma0: 1.0,
            c: 0.85,
            window: 10,
        }
    }

    /// Number of offspring per generation.
    pub fn lambda(mut self, l: usize) -> Self {
        self.lambda = l.max(1);
        self
    }

    /// Initial mutation step size σ₀.
    pub fn sigma0(mut self, s: f64) -> Self {
        self.sigma0 = s;
        self
    }

    /// 1/5-rule contraction factor `c` (0 < c < 1).
    pub fn c(mut self, c: f64) -> Self {
        self.c = c.clamp(1e-3, 0.999);
        self
    }

    /// Generations per σ-adaptation window.
    pub fn window(mut self, w: usize) -> Self {
        self.window = w.max(1);
        self
    }

    /// Maximise `f` starting from `x0`. Returns the best point, its fitness, and
    /// an auditable [`Report`]. The reported best-so-far is non-decreasing.
    pub fn optimize<F>(&self, x0: Vec<f64>, f: F, guard: &Guard) -> (Vec<f64>, Fitness, Report)
    where
        F: Fn(&[f64]) -> Fitness,
    {
        let mut rng = rng_from_seed(self.seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dim = x0.len();

        let mut parent = x0;
        let mut parent_fit = f(&parent);
        let mut sigma = self.sigma0;

        let mut history = Vec::with_capacity(guard.max_iters);
        let mut accepted = 0usize;
        let mut since_improve = 0usize;
        let mut successes_in_window = 0usize;
        let mut iterations = 0usize;
        let mut stop_reason = StopReason::MaxIterations;

        for gen in 0..guard.max_iters
        {
            iterations = gen + 1;

            // Sample λ offspring and find the best one.
            let mut best_child = parent.clone();
            let mut best_child_fit = f64::NEG_INFINITY;
            for _ in 0..self.lambda
            {
                let child: Vec<f64> = parent
                    .iter()
                    .map(|&p| p + sigma * normal.sample(&mut rng))
                    .collect();
                let fit = f(&child);
                if fit > best_child_fit
                {
                    best_child_fit = fit;
                    best_child = child;
                }
            }

            // Elitist (1+λ) selection -> monotone parent fitness.
            let improved = best_child_fit > parent_fit + guard.min_delta;
            if improved
            {
                parent = best_child;
                parent_fit = best_child_fit;
                accepted += 1;
                successes_in_window += 1;
                since_improve = 0;
            }
            else
            {
                since_improve += 1;
            }
            history.push(parent_fit);

            // Rechenberg's 1/5 rule: adapt σ once per window.
            if (gen + 1) % self.window == 0
            {
                let ps = successes_in_window as f64 / self.window as f64;
                if ps > 0.2
                {
                    sigma /= self.c; // doing well -> be bolder
                }
                else
                {
                    sigma *= self.c; // struggling -> refine
                }
                debug_assert!(dim == parent.len());
                successes_in_window = 0;
            }

            if let Some(t) = guard.target
            {
                if parent_fit >= t
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
            parent,
            parent_fit,
            Report {
                iterations,
                accepted,
                best_fitness: parent_fit,
                history,
                stop_reason,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench;

    #[test]
    fn solves_sphere() {
        let (x, fit, report) = OnePlusLambda::new(0xABCD).lambda(12).sigma0(1.0).optimize(
            vec![4.0; 6],
            bench::sphere,
            &Guard::new().max_iters(2_000).target(-1e-8),
        );
        assert!(report.is_monotone());
        assert!(fit > -1e-6, "fitness {fit} not near optimum");
        for v in &x
        {
            assert!(v.abs() < 1e-2);
        }
    }

    #[test]
    fn solves_rosenbrock() {
        let (_x, fit, report) = OnePlusLambda::new(7).lambda(20).sigma0(0.5).optimize(
            vec![0.0; 4],
            bench::rosenbrock,
            &Guard::new().max_iters(20_000),
        );
        assert!(report.is_monotone());
        assert!(fit > -1e-2, "rosenbrock fitness {fit} too far from 0");
    }

    #[test]
    fn handles_multimodal_rastrigin() {
        // Not guaranteed to find the global optimum, but must stay monotone and
        // improve substantially from the start.
        let (_x, _fit, report) = OnePlusLambda::new(2024).lambda(30).sigma0(2.0).optimize(
            vec![3.0; 3],
            bench::rastrigin,
            &Guard::new().max_iters(3_000),
        );
        assert!(report.is_monotone());
        assert!(report.total_gain() > 0.0);
    }

    #[test]
    fn step_size_adaptation_makes_progress() {
        // With a tiny initial σ the 1/5 rule must grow it enough to move.
        let (_x, fit, _r) = OnePlusLambda::new(11)
            .lambda(8)
            .sigma0(1e-3)
            .window(5)
            .optimize(vec![5.0; 3], bench::sphere, &Guard::new().max_iters(5_000));
        assert!(
            fit > -1.0,
            "σ adaptation failed to make progress (fit {fit})"
        );
    }
}
