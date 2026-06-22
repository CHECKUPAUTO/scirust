//! Ready-made trait implementations so an agent can wire a loop from plain
//! closures, without declaring a new type.
//!
//! ```
//! use scirust_rsi::adapters::FnRefine;
//! use scirust_rsi::refine::SelfRefiner;
//! use scirust_rsi::Guard;
//! use rand::Rng;
//!
//! // Climb a scalar toward 100 by nudging it upward.
//! let task = FnRefine::new(
//!     |_rng| 0.0_f64,                          // initial
//!     |x: &f64| *x,                            // score (higher is better)
//!     |x: &f64, rng: &mut rand::rngs::StdRng| x + rng.gen_range(0.0..2.0), // refine
//! );
//! let (best, report) = SelfRefiner::new(1).run(&task, &Guard::new().max_iters(500).target(100.0));
//! assert!(best >= 100.0);
//! assert!(report.is_monotone());
//! ```

use crate::Fitness;
use crate::refine::RefineTask;
use rand::rngs::StdRng;
use std::marker::PhantomData;

/// A [`RefineTask`] assembled from three closures: `initial`, `score`, `refine`.
pub struct FnRefine<S, I, Sc, R> {
    initial: I,
    score: Sc,
    refine: R,
    _marker: PhantomData<fn() -> S>,
}

impl<S, I, Sc, R> FnRefine<S, I, Sc, R>
where
    S: Clone,
    I: Fn(&mut StdRng) -> S,
    Sc: Fn(&S) -> Fitness,
    R: Fn(&S, &mut StdRng) -> S,
{
    /// Build the task from its `initial`, `score`, and `refine` closures.
    pub fn new(initial: I, score: Sc, refine: R) -> Self {
        Self {
            initial,
            score,
            refine,
            _marker: PhantomData,
        }
    }
}

impl<S, I, Sc, R> RefineTask for FnRefine<S, I, Sc, R>
where
    S: Clone,
    I: Fn(&mut StdRng) -> S,
    Sc: Fn(&S) -> Fitness,
    R: Fn(&S, &mut StdRng) -> S,
{
    type Solution = S;

    fn initial(&self, rng: &mut StdRng) -> S {
        (self.initial)(rng)
    }
    fn score(&self, sol: &S) -> Fitness {
        (self.score)(sol)
    }
    fn refine(&self, sol: &S, rng: &mut StdRng) -> S {
        (self.refine)(sol, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Guard;
    use crate::refine::SelfRefiner;

    #[test]
    fn fn_refine_solves_a_closure_task() {
        // Shrink a vector toward the origin via closures only.
        let task = FnRefine::new(
            |_rng: &mut StdRng| vec![6.0, -5.0, 4.0],
            |v: &Vec<f64>| -v.iter().map(|x| x * x).sum::<f64>(),
            |v: &Vec<f64>, rng: &mut StdRng| {
                use rand::Rng;
                let mut out = v.clone();
                let i = (0..out.len())
                    .max_by(|&a, &b| out[a].abs().partial_cmp(&out[b].abs()).unwrap())
                    .unwrap();
                out[i] *= rng.gen_range(0.3..0.9);
                out
            },
        );
        let (best, report) = SelfRefiner::new(7).run(&task, &Guard::new().max_iters(300));
        assert!(report.is_monotone());
        for v in &best
        {
            assert!(v.abs() < 1e-2);
        }
    }
}
