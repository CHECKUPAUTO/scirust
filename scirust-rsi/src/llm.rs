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

// ===========================================================================
// Real Claude API generator (optional `anthropic` feature)
// ===========================================================================

/// A [`Generator`] backed by Anthropic's Claude Messages API.
///
/// Enabled by the `anthropic` cargo feature; off by default so the core engine
/// stays dependency-light and fully offline. Uses blocking HTTP (`ureq`); there
/// is no official Rust SDK, so this calls the REST endpoint directly.
///
/// ```no_run
/// # #[cfg(feature = "anthropic")] {
/// use scirust_rsi::llm::{LlmRefine, Critic, anthropic::ClaudeGenerator};
/// use scirust_rsi::Guard;
///
/// struct Tests;
/// impl Critic for Tests {
///     fn score(&mut self, candidate: &str) -> f64 { run_tests(candidate) }
/// }
///
/// // Reads ANTHROPIC_API_KEY from the environment; default model claude-opus-4-8.
/// let mut gen = ClaudeGenerator::from_env().unwrap().max_tokens(2048);
/// let (best, fit, report) = LlmRefine::new(1)
///     .samples(4)
///     .task("Improve this Rust function so all tests pass.")
///     .run("// seed\n", &mut gen, &mut Tests, &Guard::new().max_iters(8).target(1.0));
/// # let _ = (best, fit, report);
/// # }
/// # fn run_tests(_: &str) -> f64 { 1.0 }
/// ```
#[cfg(feature = "anthropic")]
pub mod anthropic {
    use super::Generator;
    use rand::rngs::StdRng;

    /// Anthropic Messages API endpoint.
    const API_URL: &str = "https://api.anthropic.com/v1/messages";
    /// Pinned API version header value.
    const API_VERSION: &str = "2023-06-01";
    /// Default model — the latest, most capable Claude model.
    pub const DEFAULT_MODEL: &str = "claude-opus-4-8";

    /// A [`Generator`] that calls the Claude Messages API.
    #[derive(Debug, Clone)]
    pub struct ClaudeGenerator {
        api_key: String,
        model: String,
        max_tokens: u32,
        system: Option<String>,
        timeout_secs: u64,
    }

    impl ClaudeGenerator {
        /// Build a generator with an explicit API key. Defaults: model
        /// [`DEFAULT_MODEL`], 1024 max tokens, 120s request timeout.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: DEFAULT_MODEL.to_string(),
                max_tokens: 1024,
                system: None,
                timeout_secs: 120,
            }
        }

        /// Build a generator reading the key from `ANTHROPIC_API_KEY`. Returns
        /// `Err` with the variable name if it is unset.
        pub fn from_env() -> Result<Self, String> {
            let key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| "ANTHROPIC_API_KEY is not set".to_string())?;
            Ok(Self::new(key))
        }

        /// Select the model (e.g. `"claude-sonnet-4-6"`). Defaults to [`DEFAULT_MODEL`].
        pub fn model(mut self, model: impl Into<String>) -> Self {
            self.model = model.into();
            self
        }

        /// Cap output tokens per completion.
        pub fn max_tokens(mut self, n: u32) -> Self {
            self.max_tokens = n.max(1);
            self
        }

        /// Set a system prompt applied to every request.
        pub fn system(mut self, system: impl Into<String>) -> Self {
            self.system = Some(system.into());
            self
        }

        /// Per-request timeout in seconds.
        pub fn timeout_secs(mut self, secs: u64) -> Self {
            self.timeout_secs = secs.max(1);
            self
        }

        /// One Messages API call. Returns the first text block, or `None` on a
        /// transport error or a refusal (empty content).
        fn call(&self, prompt: &str) -> Option<String> {
            let mut body = serde_json::json!({
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{ "role": "user", "content": prompt }],
            });
            // Note: temperature / top_p / thinking are intentionally omitted —
            // the latest Claude models reject sampling params (400).
            if let Some(sys) = &self.system
            {
                body["system"] = serde_json::Value::String(sys.clone());
            }

            let agent = ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .build();

            let resp = agent
                .post(API_URL)
                .set("x-api-key", &self.api_key)
                .set("anthropic-version", API_VERSION)
                .set("content-type", "application/json")
                .send_json(body)
                .ok()?;

            let json: serde_json::Value = resp.into_json().ok()?;
            // content is an array of blocks; return the first text block.
            json.get("content")?
                .as_array()?
                .iter()
                .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .and_then(|b| b.get("text"))
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
        }
    }

    impl Generator for ClaudeGenerator {
        fn propose(&mut self, prompt: &str, n: usize, _rng: &mut StdRng) -> Vec<String> {
            // One call per requested sample. Failed/refused calls are skipped, so
            // the loop simply records no improvement that round (never panics).
            (0..n).filter_map(|_| self.call(prompt)).collect()
        }
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
