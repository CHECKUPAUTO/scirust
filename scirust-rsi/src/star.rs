//! # STaR — Self-Taught Reasoner (bootstrapped self-training)
//!
//! STaR (Zelikman et al., 2022) is the canonical *self-improvement* loop for a
//! model that can both *attempt* problems and *learn* from examples:
//!
//! 1. The current model attempts every problem (optionally several times).
//! 2. Attempts that turn out **correct** are kept as new training data.
//! 3. The model is retrained on the accumulated correct attempts.
//! 4. If the retrained model evaluates *better*, it is adopted; repeat.
//!
//! The model literally teaches itself from its own successful reasoning. The
//! loop is elitist (a worse retrain is rejected), so held-out performance is
//! non-decreasing.
//!
//! Implement [`BootstrapTask`] and call [`Star::run`].

use crate::{Fitness, Guard, Report, StopReason, rng_from_seed};
use rand::rngs::StdRng;

/// A task that can be improved by self-training on its own correct attempts.
pub trait BootstrapTask {
    /// A problem instance.
    type Problem: Clone;
    /// A candidate solution / reasoning trace.
    type Solution: Clone;
    /// The learnable model.
    type Model: Clone;

    /// The pool of problems to attempt each round.
    fn problems(&self) -> Vec<Self::Problem>;

    /// The starting model (before any self-training).
    fn base_model(&self) -> Self::Model;

    /// Attempt `problem` with `model`. May be stochastic (hence the RNG and the
    /// `samples` knob in [`Star`]).
    fn attempt(
        &self,
        model: &Self::Model,
        problem: &Self::Problem,
        rng: &mut StdRng,
    ) -> Self::Solution;

    /// Was this attempt correct? Only correct attempts become training data —
    /// this is the filter that makes self-training sound.
    fn is_correct(&self, problem: &Self::Problem, sol: &Self::Solution) -> bool;

    /// Retrain a fresh model on the accumulated `(problem, correct-solution)`
    /// dataset.
    fn learn(&self, base: &Self::Model, data: &[(Self::Problem, Self::Solution)]) -> Self::Model;

    /// Evaluate a model (e.g. accuracy on a held-out set). Higher is better.
    fn evaluate(&self, model: &Self::Model) -> Fitness;
}

/// Driver for the STaR bootstrapping loop.
#[derive(Debug, Clone)]
pub struct Star {
    seed: u64,
    /// How many attempts per problem each round (sampling helps find a correct
    /// trace for harder problems). Defaults to 1.
    samples: usize,
    /// Carry correct attempts across rounds (the canonical STaR accumulates).
    accumulate: bool,
}

impl Star {
    /// New STaR driver with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            samples: 1,
            accumulate: true,
        }
    }

    /// Number of attempts per problem per round.
    pub fn samples(mut self, k: usize) -> Self {
        self.samples = k.max(1);
        self
    }

    /// Whether correct attempts accumulate across rounds (default true).
    pub fn accumulate(mut self, yes: bool) -> Self {
        self.accumulate = yes;
        self
    }

    /// Run the loop. Each iteration is one full attempt/filter/retrain round.
    /// Returns the best model and an auditable [`Report`].
    pub fn run<T: BootstrapTask>(&self, task: &T, guard: &Guard) -> (T::Model, Report) {
        let mut rng = rng_from_seed(self.seed);
        let problems = task.problems();

        let mut best = task.base_model();
        let mut best_fit = task.evaluate(&best);
        let mut dataset: Vec<(T::Problem, T::Solution)> = Vec::new();

        let mut history = Vec::with_capacity(guard.max_iters);
        let mut accepted = 0usize;
        let mut since_improve = 0usize;
        let mut iterations = 0usize;
        let mut stop_reason = StopReason::MaxIterations;

        for iter in 0..guard.max_iters
        {
            iterations = iter + 1;

            // 1+2. Attempt every problem; keep the correct traces.
            if !self.accumulate
            {
                dataset.clear();
            }
            for p in &problems
            {
                for _ in 0..self.samples
                {
                    let sol = task.attempt(&best, p, &mut rng);
                    if task.is_correct(p, &sol)
                    {
                        dataset.push((p.clone(), sol));
                        break; // one correct trace per problem per round is enough
                    }
                }
            }

            // 3. Retrain on the harvested correct attempts.
            let candidate = task.learn(&best, &dataset);
            let cand_fit = task.evaluate(&candidate);

            // 4. Elitist adoption.
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
    use std::collections::HashSet;

    /// Toy "reasoner": learns a 1-D threshold classifier on `y = (x >= 3)`.
    ///
    /// The model is a threshold `t`. It attempts a problem by guessing a label;
    /// correct guesses near the true boundary teach it where the threshold is.
    /// `learn` moves the threshold to the midpoint of the labelled data — so the
    /// more correct attempts it harvests, the sharper the boundary becomes.
    struct ThresholdReasoner {
        xs: Vec<i32>,
    }
    impl ThresholdReasoner {
        fn label(x: i32) -> bool {
            x >= 3
        }
    }
    impl BootstrapTask for ThresholdReasoner {
        type Problem = i32; // an input x
        type Solution = bool; // predicted label
        type Model = f64; // threshold

        fn problems(&self) -> Vec<i32> {
            self.xs.clone()
        }
        fn base_model(&self) -> f64 {
            0.5 // starts mediocre (~0.67 accuracy), to be sharpened
        }
        fn attempt(&self, model: &f64, problem: &i32, rng: &mut StdRng) -> bool {
            // Noisy prediction around the current threshold. The jitter is wide
            // enough that boundary inputs are sometimes labelled either way, so
            // correct attempts for *both* classes can be harvested.
            let jitter: f64 = rng.gen_range(-1.5..1.5);
            (*problem as f64) >= (*model + jitter)
        }
        fn is_correct(&self, problem: &i32, sol: &bool) -> bool {
            *sol == Self::label(*problem)
        }
        fn learn(&self, base: &f64, data: &[(i32, bool)]) -> f64 {
            if data.is_empty()
            {
                return *base;
            }
            // Midpoint between the largest "false" x and the smallest "true" x.
            let max_false = data.iter().filter(|(_, y)| !*y).map(|(x, _)| *x).max();
            let min_true = data.iter().filter(|(_, y)| *y).map(|(x, _)| *x).min();
            match (max_false, min_true)
            {
                (Some(f), Some(t)) => (f as f64 + t as f64) / 2.0,
                _ => *base,
            }
        }
        fn evaluate(&self, model: &f64) -> Fitness {
            // Accuracy on the full set.
            let correct = self
                .xs
                .iter()
                .filter(|&&x| ((x as f64) >= *model) == Self::label(x))
                .count();
            correct as f64 / self.xs.len() as f64
        }
    }

    #[test]
    fn star_bootstraps_accuracy_upward() {
        let task = ThresholdReasoner {
            xs: (0..6).collect(),
        };
        let base_acc = task.evaluate(&task.base_model());
        let (model, report) = Star::new(123)
            .samples(8)
            .run(&task, &Guard::new().max_iters(40).target(1.0));

        assert!(report.is_monotone(), "accuracy must not regress");
        assert!(
            report.best_fitness >= base_acc,
            "self-training should not hurt"
        );
        // The learned threshold should land in the correct (2, 3) gap.
        assert!(
            model > 2.0 && model <= 3.0,
            "threshold {model} should separate the classes"
        );
        assert!((report.best_fitness - 1.0).abs() < 1e-9);
    }

    #[test]
    fn star_history_has_no_duplicates_dropped() {
        // Sanity: distinct fitness values are actually recorded over the run.
        let task = ThresholdReasoner {
            xs: (0..6).collect(),
        };
        let (_m, report) = Star::new(7)
            .samples(4)
            .run(&task, &Guard::new().max_iters(30));
        let uniq: HashSet<u64> = report.history.iter().map(|f| f.to_bits()).collect();
        assert!(
            uniq.len() >= 2,
            "expected measurable improvement over the run"
        );
    }
}
