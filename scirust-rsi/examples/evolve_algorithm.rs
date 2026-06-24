//! # Mode B, end to end: evolving a real algorithm with a generator + scirust
//!
//! A runnable demonstration of **LLM-driven program evolution** kept fully
//! offline and deterministic. The "algorithm" being evolved is a small program
//! in a tiny stack language (reverse-Polish arithmetic over the input `x`); the
//! task is **symbolic regression** — discover a program that reproduces a hidden
//! target function from sample points.
//!
//! The division of labour is exactly the one described for Mode B:
//!
//! - **Generator** (`MutatingGenerator`) stands in for Claude: it reads the
//!   current best program out of the prompt and proposes mutated variants.
//!   Swapping in the real model is a one-liner (see the note in `main`).
//! - **Critic** (`Interpreter`) is *RSI's evaluator*: it **runs** each proposed
//!   program through a sandboxed interpreter and scores it by accuracy. scirust
//!   never executes anything itself — the harness here does, safely.
//! - [`LlmRefine`] is scirust's bounded, elitist loop: best-of-`n` each round,
//!   adopt only a strict improvement, so the evolved program never regresses.
//!
//! Run with: `cargo run -p scirust-rsi --example evolve_algorithm`

use rand::Rng;
use rand::rngs::StdRng;
use scirust_rsi::llm::{CURRENT_MARKER, Critic, Generator, LlmRefine};
use scirust_rsi::{Fitness, Guard};

/// The hidden function the evolved program must rediscover: f(x) = x² − x + 2.
fn target(x: f64) -> f64 {
    x * x - x + 2.0
}

/// Evaluate a reverse-Polish program (whitespace-separated tokens) at `x`.
/// Tokens: `x`, a numeric literal, or one of `+ - *`. Returns `None` if the
/// program is malformed (bad stack), which the critic turns into a low score.
/// This is the *sandbox*: a fixed interpreter over arithmetic, no host access.
fn eval_rpn(tokens: &[&str], x: f64) -> Option<f64> {
    let mut stack: Vec<f64> = Vec::new();
    for &t in tokens
    {
        match t
        {
            "+" =>
            {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(a + b);
            },
            "-" =>
            {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(a - b);
            },
            "*" =>
            {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(a * b);
            },
            "x" => stack.push(x),
            lit => stack.push(lit.parse::<f64>().ok()?),
        }
    }
    if stack.len() == 1
    {
        Some(stack[0])
    }
    else
    {
        None
    }
}

/// RSI's evaluator: runs a candidate program over the sample points and scores
/// it by negative mean-squared error, with a tiny length penalty (Occam's razor
/// — prefer the simpler program among equally accurate ones).
struct Interpreter {
    xs: Vec<f64>,
}
impl Critic for Interpreter {
    fn score(&mut self, candidate: &str) -> Fitness {
        let tokens: Vec<&str> = candidate.split_whitespace().collect();
        if tokens.is_empty()
        {
            return -1e9;
        }
        let mut se = 0.0;
        for &x in &self.xs
        {
            match eval_rpn(&tokens, x)
            {
                Some(y) if y.is_finite() => se += (y - target(x)).powi(2),
                _ => return -1e9, // malformed or non-finite => unusable
            }
        }
        let mse = se / self.xs.len() as f64;
        -mse - 0.001 * tokens.len() as f64
    }
    fn critique(&mut self, candidate: &str, score: Fitness) -> String {
        // A real critic could point at the worst sample; here a short hint is
        // enough to show the feedback channel is wired.
        format!("current program scores {score:.4}; reduce the error on x². ({candidate})")
    }
}

/// Stands in for Claude: proposes mutated variants of the current best program.
/// A real generator would *write* programs; this one edits the incumbent with a
/// few structured operators — the same shape a capable model would produce.
struct MutatingGenerator;
impl MutatingGenerator {
    /// Pull the current program out of the prompt (the line after the marker).
    fn current(prompt: &str) -> Vec<String> {
        let after = prompt
            .split_once(CURRENT_MARKER)
            .map(|(_, rest)| rest.trim_start())
            .unwrap_or("");
        after
            .lines()
            .next()
            .unwrap_or("x")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    fn random_operand(rng: &mut StdRng) -> String {
        match rng.gen_range(0..4)
        {
            0 => "x".to_string(),
            n => (n as f64).to_string(), // small constants 1, 2, 3
        }
    }

    /// One structured edit, mirroring how a model would revise a program.
    fn single_edit(mut toks: Vec<String>, rng: &mut StdRng) -> Vec<String> {
        match rng.gen_range(0..6)
        {
            // Grow: append "<operand> <op>" so the stack stays balanced.
            0 =>
            {
                toks.push(Self::random_operand(rng));
                toks.push(["+", "-", "*"][rng.gen_range(0..3)].to_string());
            },
            // Replace one token with another of a compatible kind.
            1 if !toks.is_empty() =>
            {
                let i = rng.gen_range(0..toks.len());
                toks[i] = if matches!(toks[i].as_str(), "+" | "-" | "*")
                {
                    ["+", "-", "*"][rng.gen_range(0..3)].to_string()
                }
                else
                {
                    Self::random_operand(rng)
                };
            },
            // Perturb a numeric literal.
            2 =>
            {
                if let Some(i) = toks.iter().position(|t| t.parse::<f64>().is_ok())
                {
                    let v: f64 = toks[i].parse().unwrap();
                    toks[i] = (v + rng.gen_range(-1.0..1.0)).round().to_string();
                }
            },
            // Prepend "x <op>" to fold the input in earlier.
            3 =>
            {
                let op = ["+", "-", "*"][rng.gen_range(0..3)].to_string();
                toks.insert(0, op);
                toks.insert(0, "x".to_string());
                toks.insert(0, "x".to_string());
            },
            // Drop a trailing token (shrink).
            4 if toks.len() > 1 =>
            {
                toks.pop();
            },
            // Insert an operand somewhere.
            _ =>
            {
                let i = rng.gen_range(0..=toks.len());
                toks.insert(i, Self::random_operand(rng));
            },
        }
        toks
    }

    /// A proposal = several structured edits in a row. Bigger, more diverse
    /// jumps let the search escape plateaus a single edit can't — much as a
    /// capable model rewrites a chunk of the program at once rather than nudging
    /// one token.
    fn mutate(mut toks: Vec<String>, rng: &mut StdRng) -> Vec<String> {
        for _ in 0..rng.gen_range(1..=4)
        {
            toks = Self::single_edit(toks, rng);
        }
        toks
    }
}
impl Generator for MutatingGenerator {
    fn propose(&mut self, prompt: &str, n: usize, rng: &mut StdRng) -> Vec<String> {
        let cur = Self::current(prompt);
        (0..n)
            .map(|_| Self::mutate(cur.clone(), rng).join(" "))
            .collect()
    }
}

fn main() {
    // Sample points the program is scored on.
    let xs: Vec<f64> = (-3..=3).map(|i| i as f64).collect();
    let mut critic = Interpreter { xs: xs.clone() };

    // Bounded, elitist loop. best-of-32 per round; stop early once the program
    // is essentially perfect (the small residual is just the length penalty).
    let guard = Guard::new().max_iters(1200).target(-0.01);
    let mut generator = MutatingGenerator;

    // --- To use the REAL Claude instead of the mock, swap one line ----------
    //   cargo run -p scirust-rsi --example evolve_algorithm --features anthropic
    //   (and export ANTHROPIC_API_KEY):
    //
    //   let mut generator = scirust_rsi::llm::anthropic::ClaudeGenerator::from_env()
    //       .unwrap()
    //       .system("You evolve reverse-Polish arithmetic programs over x. \
    //                Reply with ONLY the program, tokens space-separated.");
    // ------------------------------------------------------------------------

    let seed_program = "x"; // start from the identity program f(x) = x
    let (best, fit, report) = LlmRefine::new(20240624)
        .samples(32)
        .task(
            "Evolve a reverse-Polish program over `x` (tokens: x, integers, + - *) \
             that matches the hidden target function on the sample points.",
        )
        .run(seed_program, &mut generator, &mut critic, &guard);

    println!("=== Mode B: evolving an algorithm (symbolic regression) ===\n");
    println!("hidden target : f(x) = x² − x + 2");
    println!("seed program  : \"{seed_program}\"   (score {:.4})", {
        critic.score(seed_program)
    });
    println!("evolved program: \"{best}\"");
    println!("final score    : {fit:.6}   (≈ −MSE; 0 is perfect)");
    println!(
        "iterations     : {}   accepted: {} ({:.1}% of rounds)   stop: {:?}",
        report.iterations,
        report.accepted,
        report.acceptance_rate() * 100.0,
        report.stop_reason
    );
    println!(
        "monotone       : {}  (the program never got worse)",
        report.is_monotone()
    );

    // Show the evolved program reproducing the target on each sample.
    let toks: Vec<&str> = best.split_whitespace().collect();
    println!("\n   x  | target | evolved");
    println!("  ----+--------+--------");
    for &x in &xs
    {
        let y = eval_rpn(&toks, x).unwrap_or(f64::NAN);
        println!("  {x:>3} | {:>6.1} | {y:>6.1}", target(x));
    }

    println!(
        "\nThe generator proposed the programs; scirust's elitist Guard kept only\n\
         strictly-better ones — bounded, reproducible, and non-regressing."
    );
}
