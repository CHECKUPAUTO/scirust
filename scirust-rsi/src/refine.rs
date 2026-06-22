//! # Self-Refine — iterative critique-and-revise
//!
//! The simplest recursive self-improvement loop: produce a solution, *critique
//! it*, produce a revised solution from the critique, and repeat. Adoption is
//! elitist (via [`crate::ascend`]), so the kept solution only ever gets better.
//!
//! Implement [`RefineTask`] for your problem and call [`SelfRefiner::run`].
//!
//! ```
//! use rand::rngs::StdRng;
//! use scirust_rsi::{Guard, Fitness};
//! use scirust_rsi::refine::{RefineTask, SelfRefiner};
//!
//! // Toy task: find the integer in [0, 100] closest to 42 by nudging.
//! struct Nudge;
//! impl RefineTask for Nudge {
//!     type Solution = i64;
//!     fn initial(&self, _rng: &mut StdRng) -> i64 { 0 }
//!     fn score(&self, s: &i64) -> Fitness { -((*s - 42).abs() as f64) }
//!     fn refine(&self, s: &i64, rng: &mut StdRng) -> i64 {
//!         use rand::Rng;
//!         // "critique": we're below/above the target -> step toward it (noisily).
//!         let dir = if *s < 42 { 1 } else { -1 };
//!         (s + dir * rng.gen_range(1..=5)).clamp(0, 100)
//!     }
//! }
//!
//! let (best, report) = SelfRefiner::new(7).run(&Nudge, &Guard::new().max_iters(200).target(0.0));
//! assert_eq!(best, 42);
//! assert!(report.is_monotone());
//! ```

use crate::{Fitness, Guard, Report, ascend, rng_from_seed};
use rand::rngs::StdRng;

/// A task amenable to self-refinement.
pub trait RefineTask {
    /// The artefact being improved (an answer, a plan, a parameter vector, ...).
    type Solution: Clone;

    /// Produce the starting solution.
    fn initial(&self, rng: &mut StdRng) -> Self::Solution;

    /// Score a solution. Higher is better.
    fn score(&self, sol: &Self::Solution) -> Fitness;

    /// Produce a revised solution. This is where a "critique" is applied — for
    /// an LLM this would be a self-critique prompt; here it is any (possibly
    /// stochastic) transformation that tends to improve `sol`.
    fn refine(&self, sol: &Self::Solution, rng: &mut StdRng) -> Self::Solution;
}

/// Driver for the [`RefineTask`] loop.
#[derive(Debug, Clone)]
pub struct SelfRefiner {
    seed: u64,
}

impl SelfRefiner {
    /// New refiner with the given RNG seed (for reproducibility).
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Run critique-and-revise until the guard stops it. Returns the best
    /// solution found and an auditable [`Report`].
    pub fn run<T: RefineTask>(&self, task: &T, guard: &Guard) -> (T::Solution, Report) {
        let mut rng = rng_from_seed(self.seed);
        let initial = task.initial(&mut rng);
        let init_fit = task.score(&initial);
        ascend(
            initial,
            init_fit,
            |best, _iter, rng| {
                let cand = task.refine(best, rng);
                let fit = task.score(&cand);
                (cand, fit)
            },
            guard,
            &mut rng,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StopReason;

    /// Refine a real vector toward the origin by shrinking the largest coord.
    struct ShrinkToOrigin {
        dim: usize,
    }
    impl RefineTask for ShrinkToOrigin {
        type Solution = Vec<f64>;
        fn initial(&self, _rng: &mut StdRng) -> Vec<f64> {
            vec![5.0; self.dim]
        }
        fn score(&self, s: &Vec<f64>) -> Fitness {
            -s.iter().map(|v| v * v).sum::<f64>()
        }
        fn refine(&self, s: &Vec<f64>, rng: &mut StdRng) -> Vec<f64> {
            use rand::Rng;
            let mut out = s.clone();
            // Critique: the worst coordinate is the one with largest magnitude.
            let i = (0..out.len())
                .max_by(|&a, &b| out[a].abs().partial_cmp(&out[b].abs()).unwrap())
                .unwrap();
            out[i] *= rng.gen_range(0.3..0.9);
            out
        }
    }

    #[test]
    fn self_refine_converges_to_origin() {
        let (best, report) =
            SelfRefiner::new(42).run(&ShrinkToOrigin { dim: 4 }, &Guard::new().max_iters(300));
        assert!(report.is_monotone());
        for v in &best
        {
            assert!(v.abs() < 1e-2, "coord {v} not shrunk");
        }
        assert!(report.best_fitness > -1e-3);
    }

    #[test]
    fn self_refine_respects_patience() {
        // Once at the origin, refine can't improve -> patience triggers.
        let (_b, report) = SelfRefiner::new(1).run(
            &ShrinkToOrigin { dim: 2 },
            &Guard::new().max_iters(10_000).patience(30),
        );
        assert_eq!(report.stop_reason, StopReason::Converged);
    }
}
