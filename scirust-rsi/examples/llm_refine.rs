//! LLM-driven self-improvement loop, demonstrated offline with a *mock* model.
//!
//! In production you replace `MockModel` with a real `Generator` that calls your
//! language model (see `src/llm.rs` docs). The loop, bounds, and non-regression
//! guarantee are identical either way.
//!
//! Run with: `cargo run -p scirust-rsi --example llm_refine`

use rand::Rng;
use rand::rngs::StdRng;
use scirust_rsi::llm::{CURRENT_MARKER, Critic, Generator, LlmRefine, build_prompt};
use scirust_rsi::{Fitness, Guard};

/// Stand-in for an LLM: reads the current candidate from the prompt and returns
/// edited variants. A real model would instead complete `prompt`.
struct MockModel;
impl Generator for MockModel {
    fn propose(&mut self, prompt: &str, n: usize, rng: &mut StdRng) -> Vec<String> {
        let current = prompt
            .split_once(CURRENT_MARKER)
            .and_then(|(_, rest)| rest.trim_start().lines().next())
            .and_then(|l| l.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        (0..n)
            .map(|_| format!("{:.4}", current + rng.gen_range(-3.0..3.0)))
            .collect()
    }
}

/// Evaluator: rewards getting close to the (hidden) target 42, with a critique
/// the next prompt can use.
struct FitToTarget;
impl Critic for FitToTarget {
    fn score(&mut self, candidate: &str) -> Fitness {
        let x = candidate.trim().parse::<f64>().unwrap_or(0.0);
        -(x - 42.0).powi(2)
    }
    fn critique(&mut self, candidate: &str, _score: Fitness) -> String {
        let x = candidate.trim().parse::<f64>().unwrap_or(0.0);
        if x < 42.0
        {
            "The value is too low — increase it.".into()
        }
        else
        {
            "The value is too high — decrease it.".into()
        }
    }
}

fn main() {
    println!("=== Example prompt handed to the generator ===");
    println!(
        "{}\n",
        build_prompt(
            "Find the number closest to the secret target.",
            "0",
            "(no critique yet)"
        )
    );

    println!("=== LLM-driven self-refine (mock model, best-of-16) ===");
    let (best, fit, report) = LlmRefine::new(2024)
        .samples(16)
        .task("Find the number closest to the secret target.")
        .run(
            "0",
            &mut MockModel,
            &mut FitToTarget,
            &Guard::new().max_iters(200).patience(30),
        );

    println!("  best solution : {best}");
    println!("  score         : {fit:.6} (optimum 0 at value 42)");
    println!(
        "  rounds        : {} ({:?}), accepted {} improvements, monotone = {}",
        report.iterations,
        report.stop_reason,
        report.accepted,
        report.is_monotone()
    );
    println!("\nSwap MockModel for a real Generator (one HTTP call) and the same");
    println!("bounded, non-regressing loop drives a real model.");
}
