//! # LLM-driven self-improvement (generator + critic)
//!
//! The generic loops elsewhere in this crate need a way to *propose* a better
//! candidate. When that proposer is a language model, this module is the bridge:
//! you implement [`Generator`] (call your model) and [`Critic`] (score / critique
//! a candidate), and [`LlmRefine`] runs the bounded, elitist best-of-`n`
//! self-refine loop:
//!
//! 1. Build a prompt from the current best solution and its critique.
//! 2. Ask the generator for `n` candidate solutions.
//! 3. Score them; keep the best **only if** it beats the incumbent.
//! 4. Repeat until the [`Guard`] stops it.
//!
//! Because adoption is elitist, the loop inherits the same non-regression
//! guarantee as the rest of the crate: the agent can never ship a worse answer
//! than it already had. The generator and critic are *yours* — the engine runs
//! no model and executes no code itself.
//!
//! ## Wiring a real model
//!
//! ```no_run
//! # use scirust_rsi::llm::Generator;
//! # use rand::rngs::StdRng;
//! struct Claude { /* http client, api key, model id... */ }
//! impl Generator for Claude {
//!     fn propose(&mut self, prompt: &str, n: usize, _rng: &mut StdRng) -> Vec<String> {
//!         // POST `prompt` to the Messages API `n` times (or one call asking for
//!         // n variants) and return the completions. Keep it deterministic by
//!         // pinning temperature/seed if your provider supports it.
//!         let _ = (prompt, n);
//!         Vec::new()
//!     }
//! }
//! ```

use crate::{Fitness, Guard, LoopState, Report, rng_from_seed};
use rand::rngs::StdRng;

/// A source of candidate solutions — typically an LLM, but anything that turns a
/// prompt into textual candidates qualifies.
pub trait Generator {
    /// Produce up to `n` candidate solutions for `prompt`. Returning fewer (even
    /// zero) is fine; the loop simply records no improvement that round.
    fn propose(&mut self, prompt: &str, n: usize, rng: &mut StdRng) -> Vec<String>;
}

/// Scores and optionally critiques candidate solutions.
pub trait Critic {
    /// Score a candidate. **Higher is better** (e.g. fraction of tests passed
    /// minus a length penalty). This is the agent's evaluator.
    fn score(&mut self, candidate: &str) -> Fitness;

    /// Optional natural-language critique fed back into the next prompt. The
    /// default returns nothing, turning the loop into plain best-of-`n` sampling.
    fn critique(&mut self, _candidate: &str, _score: Fitness) -> String {
        String::new()
    }
}

/// Marker delimiting the current solution inside the prompt, so a generator can
/// locate it reliably. Public so custom generators can parse against it.
pub const CURRENT_MARKER: &str = "[CURRENT SOLUTION]";
/// Marker delimiting the critique inside the prompt.
pub const CRITIQUE_MARKER: &str = "[CRITIQUE]";

/// Assemble the prompt handed to the [`Generator`] each round.
pub fn build_prompt(task: &str, best: &str, critique: &str) -> String {
    format!(
        "{task}\n\n{CURRENT_MARKER}\n{best}\n\n{CRITIQUE_MARKER}\n{critique}\n\n\
         Return a single improved solution that scores higher.",
    )
}

/// Driver for the LLM-/generator-backed self-refine loop.
#[derive(Debug, Clone)]
pub struct LlmRefine {
    seed: u64,
    samples: usize,
    task: String,
}

impl LlmRefine {
    /// New driver with the given RNG seed (passed to the generator for any
    /// stochastic sampling it does).
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            samples: 4,
            task: String::new(),
        }
    }

    /// Candidates requested per round (the `n` in best-of-`n`). Default 4.
    pub fn samples(mut self, n: usize) -> Self {
        self.samples = n.max(1);
        self
    }

    /// A task description woven into every prompt.
    pub fn task(mut self, description: &str) -> Self {
        self.task = description.to_string();
        self
    }

    /// Run the loop from `seed_solution`. Returns the best solution found, its
    /// score, and an auditable [`Report`].
    pub fn run<G: Generator, C: Critic>(
        &self,
        seed_solution: &str,
        generator: &mut G,
        critic: &mut C,
        guard: &Guard,
    ) -> (String, Fitness, Report) {
        let mut rng = rng_from_seed(self.seed);
        let mut best = seed_solution.to_string();
        let mut ctrl = LoopState::new(guard, critic.score(&best));

        while ctrl.next_iter()
        {
            // 1. Prompt = current best + its critique.
            let critique = critic.critique(&best, ctrl.best_fit());
            let prompt = build_prompt(&self.task, &best, &critique);

            // 2+3. Best-of-n: take the highest-scoring proposal of the round.
            let mut round_best: Option<(String, Fitness)> = None;
            for cand in generator.propose(&prompt, self.samples, &mut rng)
            {
                let s = critic.score(&cand);
                if round_best.as_ref().is_none_or(|(_, bs)| s > *bs)
                {
                    round_best = Some((cand, s));
                }
            }

            // 4. Elitist adoption.
            match round_best
            {
                Some((cand, s)) =>
                {
                    if ctrl.offer(s)
                    {
                        best = cand;
                    }
                },
                None =>
                {
                    // No candidate this round: record a non-improvement so
                    // patience/convergence still progress.
                    ctrl.offer(f64::NEG_INFINITY);
                },
            }

            if ctrl.done()
            {
                break;
            }
        }

        let best_fit = ctrl.best_fit();
        (best, best_fit, ctrl.into_report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Mock "LLM": reads the current numeric solution out of the prompt and
    /// returns perturbations of it. Stands in for a real model so the loop is
    /// fully testable and deterministic offline.
    struct MockModel;
    impl MockModel {
        fn current_value(prompt: &str) -> f64 {
            // Parse the line right after the CURRENT marker.
            let after = prompt
                .split_once(CURRENT_MARKER)
                .map(|(_, rest)| rest.trim_start())
                .unwrap_or("");
            let line = after.lines().next().unwrap_or("0");
            line.trim().parse::<f64>().unwrap_or(0.0)
        }
    }
    impl Generator for MockModel {
        fn propose(&mut self, prompt: &str, n: usize, rng: &mut StdRng) -> Vec<String> {
            let v = Self::current_value(prompt);
            (0..n)
                .map(|_| format!("{}", v + rng.gen_range(-2.0..2.0)))
                .collect()
        }
    }

    /// Critic: prefers solutions whose value is close to 42.
    struct CloseTo42;
    impl Critic for CloseTo42 {
        fn score(&mut self, candidate: &str) -> Fitness {
            let x = candidate.trim().parse::<f64>().unwrap_or(0.0);
            -(x - 42.0).powi(2)
        }
        fn critique(&mut self, candidate: &str, _score: Fitness) -> String {
            let x = candidate.trim().parse::<f64>().unwrap_or(0.0);
            if x < 42.0
            {
                "too low, increase it".into()
            }
            else
            {
                "too high, decrease it".into()
            }
        }
    }

    #[test]
    fn llm_refine_converges_toward_target() {
        let (best, fit, report) = LlmRefine::new(2024)
            .samples(16)
            .task("Find the number closest to the secret target.")
            .run(
                "0",
                &mut MockModel,
                &mut CloseTo42,
                &Guard::new().max_iters(300),
            );

        assert!(report.is_monotone(), "best-of-n must not regress");
        let x = best.trim().parse::<f64>().unwrap();
        assert!((x - 42.0).abs() < 2.0, "should approach 42, got {x}");
        assert!(fit > -4.0, "fitness {fit} too far from optimum");
        assert!(report.accepted > 0);
    }

    #[test]
    fn llm_refine_handles_empty_generator() {
        // A generator that returns nothing must not panic and must converge via
        // patience without ever improving.
        struct Silent;
        impl Generator for Silent {
            fn propose(&mut self, _p: &str, _n: usize, _r: &mut StdRng) -> Vec<String> {
                Vec::new()
            }
        }
        let (_b, _f, report) = LlmRefine::new(1).run(
            "0",
            &mut Silent,
            &mut CloseTo42,
            &Guard::new().max_iters(100).patience(5),
        );
        assert_eq!(report.stop_reason, crate::StopReason::Converged);
        assert_eq!(report.accepted, 0);
    }
}
