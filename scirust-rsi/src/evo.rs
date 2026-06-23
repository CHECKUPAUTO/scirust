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

use crate::{Fitness, Guard, LoopState, Report, rng_from_seed};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// A resumable checkpoint of an `(1+λ)`-ES search: the best point reached and the
/// adapted step size `σ`. Serialize it (any serde format) to persist a run, then
/// hand it to [`OnePlusLambda::resume`] to warm-start from where you left off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvoState {
    /// Best parameter vector found so far.
    pub point: Vec<f64>,
    /// The mutation step size reached by the 1/5 rule.
    pub sigma: f64,
}

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
        let (point, _sigma, report) = self.run_seq(x0, self.sigma0, &f, guard);
        let best = report.best_fitness;
        (point, best, report)
    }

    /// Run and return a resumable [`EvoState`] (best point + adapted σ) alongside
    /// the report, so a later call can warm-start from where this one stopped.
    pub fn optimize_resumable<F>(&self, x0: Vec<f64>, f: F, guard: &Guard) -> (EvoState, Report)
    where
        F: Fn(&[f64]) -> Fitness,
    {
        let (point, sigma, report) = self.run_seq(x0, self.sigma0, &f, guard);
        (EvoState { point, sigma }, report)
    }

    /// Continue a search from a saved [`EvoState`] — a **warm start** from the
    /// stored point and adapted σ. The RNG restarts from the strategy's seed, so
    /// this is not a bit-identical continuation of the original stream; it resumes
    /// the *search state*, which is what persistence/resume needs.
    pub fn resume<F>(&self, state: &EvoState, f: F, guard: &Guard) -> (EvoState, Report)
    where
        F: Fn(&[f64]) -> Fitness,
    {
        let (point, sigma, report) = self.run_seq(state.point.clone(), state.sigma, &f, guard);
        (EvoState { point, sigma }, report)
    }

    /// Shared sequential core: run from `(parent, sigma)` and return the final
    /// point, the adapted σ, and the report.
    fn run_seq<F>(&self, x0: Vec<f64>, sigma0: f64, f: &F, guard: &Guard) -> (Vec<f64>, f64, Report)
    where
        F: Fn(&[f64]) -> Fitness,
    {
        let mut rng = rng_from_seed(self.seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut parent = x0;
        let mut sigma = sigma0;
        let mut successes_in_window = 0usize;
        let mut ctrl = LoopState::new(guard, f(&parent));

        while ctrl.next_iter()
        {
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
            if ctrl.offer(best_child_fit)
            {
                parent = best_child;
                successes_in_window += 1;
            }

            // Rechenberg's 1/5 rule: adapt σ once per window.
            if ctrl.iterations().is_multiple_of(self.window)
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
                successes_in_window = 0;
            }

            if ctrl.done()
            {
                break;
            }
        }

        (parent, sigma, ctrl.into_report())
    }

    /// Like [`optimize`](Self::optimize) but evaluates each generation's λ
    /// offspring in **parallel** (rayon). Offspring are still *sampled*
    /// sequentially, so the RNG draw order — and therefore the whole run — is
    /// **bit-identical** to `optimize`; only fitness evaluation is parallelised.
    /// Worth it when `f` is expensive. Requires the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn optimize_parallel<F>(
        &self,
        x0: Vec<f64>,
        f: F,
        guard: &Guard,
    ) -> (Vec<f64>, Fitness, Report)
    where
        F: Fn(&[f64]) -> Fitness + Sync,
    {
        use rayon::prelude::*;

        let mut rng = rng_from_seed(self.seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut parent = x0;
        let mut sigma = self.sigma0;
        let mut successes_in_window = 0usize;
        let mut ctrl = LoopState::new(guard, f(&parent));

        while ctrl.next_iter()
        {
            // Sample λ offspring sequentially (preserves the RNG draw order)...
            let children: Vec<Vec<f64>> = (0..self.lambda)
                .map(|_| {
                    parent
                        .iter()
                        .map(|&p| p + sigma * normal.sample(&mut rng))
                        .collect()
                })
                .collect();
            // ...then evaluate their fitness in parallel.
            let fits: Vec<Fitness> = children.par_iter().map(|c| f(c)).collect();

            // Best child = first strictly-greater (identical to the sequential rule).
            let mut best_child = parent.clone();
            let mut best_child_fit = f64::NEG_INFINITY;
            for (child, fit) in children.into_iter().zip(fits)
            {
                if fit > best_child_fit
                {
                    best_child_fit = fit;
                    best_child = child;
                }
            }

            if ctrl.offer(best_child_fit)
            {
                parent = best_child;
                successes_in_window += 1;
            }

            if ctrl.iterations().is_multiple_of(self.window)
            {
                let ps = successes_in_window as f64 / self.window as f64;
                if ps > 0.2
                {
                    sigma /= self.c;
                }
                else
                {
                    sigma *= self.c;
                }
                successes_in_window = 0;
            }

            if ctrl.done()
            {
                break;
            }
        }

        let best_fit = ctrl.best_fit();
        (parent, best_fit, ctrl.into_report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench;

    #[test]
    fn resume_warm_starts_from_a_checkpoint() {
        let opt = OnePlusLambda::new(7).lambda(10).sigma0(1.0);
        // Short first leg -> checkpoint.
        let (state, r1) =
            opt.optimize_resumable(vec![4.0; 5], bench::sphere, &Guard::new().max_iters(100));
        // Persist + reload the checkpoint (any serde format works).
        let json = serde_json::to_string(&state).unwrap();
        let reloaded: EvoState = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded.point, state.point);
        // Resume -> must not regress past the checkpoint's best.
        let (state2, r2) = opt.resume(
            &reloaded,
            bench::sphere,
            &Guard::new().max_iters(2000).target(-1e-8),
        );
        assert!(r1.is_monotone() && r2.is_monotone());
        assert!(
            r2.best_fitness >= r1.best_fitness,
            "resume should not lose ground"
        );
        assert!(
            r2.best_fitness > -1e-3,
            "resumed run should keep converging"
        );
        assert!(state2.sigma > 0.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_is_bit_identical_to_sequential() {
        let guard = Guard::new().max_iters(500);
        let mk = || OnePlusLambda::new(0xABCD).lambda(12).sigma0(1.0);
        let (xa, fa, ra) = mk().optimize(vec![4.0; 6], bench::sphere, &guard);
        let (xb, fb, rb) = mk().optimize_parallel(vec![4.0; 6], bench::sphere, &guard);
        assert_eq!(xa, xb, "best point must match exactly");
        assert_eq!(fa, fb, "best fitness must match exactly");
        assert_eq!(
            ra.history, rb.history,
            "convergence curve must match exactly"
        );
    }

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
