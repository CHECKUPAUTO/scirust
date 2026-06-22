//! Live demo of the Claude-backed generator driving the self-refine loop.
//!
//! Requires the `anthropic` feature **and** an `ANTHROPIC_API_KEY`. It makes
//! real Claude Messages API calls (a handful of small completions — it costs a
//! few tokens). Without the key it prints instructions and exits cleanly.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run -p scirust-rsi --example claude_refine --features anthropic

#[cfg(not(feature = "anthropic"))]
fn main() {
    eprintln!(
        "This example needs the `anthropic` feature:\n  \
         cargo run -p scirust-rsi --example claude_refine --features anthropic"
    );
}

#[cfg(feature = "anthropic")]
fn main() {
    use scirust_rsi::Guard;
    use scirust_rsi::llm::anthropic::ClaudeGenerator;
    use scirust_rsi::llm::{Critic, LlmRefine};

    // Evaluator: reward a one-line slogan that is ~10–14 words and mentions
    // "determinism". Purely illustrative — swap in compile/test scoring for
    // real code tasks.
    struct SloganCritic;
    impl Critic for SloganCritic {
        fn score(&mut self, candidate: &str) -> f64 {
            let line = candidate.trim().lines().next().unwrap_or("").trim();
            let words = line.split_whitespace().count() as f64;
            let length_score = -((words - 12.0).abs()); // peak at 12 words
            let keyword = if line.to_lowercase().contains("determinism")
            {
                5.0
            }
            else
            {
                0.0
            };
            length_score + keyword
        }
        fn critique(&mut self, candidate: &str, _score: f64) -> String {
            let line = candidate.trim().lines().next().unwrap_or("").trim();
            let words = line.split_whitespace().count();
            let mut notes = Vec::new();
            if !line.to_lowercase().contains("determinism")
            {
                notes.push("must contain the word 'determinism'".to_string());
            }
            if words < 10
            {
                notes.push(format!("too short ({words} words) — aim for ~12"));
            }
            else if words > 14
            {
                notes.push(format!("too long ({words} words) — aim for ~12"));
            }
            if notes.is_empty()
            {
                "good — minor polish only".into()
            }
            else
            {
                notes.join("; ")
            }
        }
    }

    let mut generator = match ClaudeGenerator::from_env()
    {
        Ok(g) => g.max_tokens(256),
        Err(e) =>
        {
            eprintln!("{e}. Set ANTHROPIC_API_KEY to run this example.");
            return;
        },
    };

    println!("=== Claude-driven self-refine (real API calls) ===");
    println!("Model: claude-opus-4-8 (default); refining a one-line slogan.\n");

    let (best, fit, report) = LlmRefine::new(1)
        .samples(2)
        .task("Write ONE punchy one-line slogan (~12 words) for a deterministic ML framework. Output only the slogan.")
        .run(
            "Fast machine learning.",
            &mut generator,
            &mut SloganCritic,
            &Guard::new().max_iters(6).patience(3),
        );

    println!("best slogan : {best:?}");
    println!(
        "score {fit:.2} in {} rounds ({:?}), accepted {} improvements, monotone = {}",
        report.iterations,
        report.stop_reason,
        report.accepted,
        report.is_monotone()
    );
    println!("\nThe loop is bounded and elitist — it can only keep a strictly-better slogan.");
}
