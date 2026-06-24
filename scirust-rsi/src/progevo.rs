//! # Program evolution from input→output examples
//!
//! Evolve a small program — a reverse-Polish arithmetic expression over one
//! input `x` (tokens: `x`, numeric literals, and `+ - * /`) — so that it
//! reproduces a set of `(input, output)` examples the caller supplies. The
//! search is the crate's bounded, elitist [`LlmRefine`] loop driven by a
//! built-in mutation generator, so it needs **no model and no API key** and is
//! fully reproducible from a seed.
//!
//! This is the engine behind the `scirust-rsi-mcp` server: an LLM (or any
//! caller) passes the examples and an optional starting program, and gets back
//! an evolved program plus an auditable [`Report`]. The same safety properties
//! as the rest of the crate hold — the result never regresses below the seed,
//! the run always terminates, and nothing is executed beyond this fixed
//! arithmetic interpreter (no host access, no codegen).
//!
//! ```
//! use scirust_rsi::{Guard, progevo};
//! // Discover "double the input": x -> 2x.
//! let examples = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
//! let out = progevo::evolve(&examples, "x", 32, 1, &Guard::new().max_iters(800).target(-1e-9));
//! assert!(out.report.is_monotone());
//! assert!(out.mse < 1e-6, "should fit exactly, got mse {}", out.mse);
//! ```

use crate::llm::{CURRENT_MARKER, Critic, Generator, LlmRefine};
use crate::{Fitness, Guard, Report};
use rand::Rng;
use rand::rngs::StdRng;

/// Evaluate a reverse-Polish program at `x`. Tokens are whitespace-separated:
/// `x`, a numeric literal, or one of `+ - * /`. Returns `None` if the program is
/// malformed (unbalanced stack) or divides by zero — the *sandbox* is this fixed
/// arithmetic interpreter; nothing else is ever executed.
pub fn eval_rpn(program: &str, x: f64) -> Option<f64> {
    let mut stack: Vec<f64> = Vec::new();
    for t in program.split_whitespace()
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
            "/" =>
            {
                let b = stack.pop()?;
                let a = stack.pop()?;
                if b == 0.0
                {
                    return None;
                }
                stack.push(a / b);
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

/// Mean-squared error of `program` over the examples. `f64::INFINITY` if the
/// program is unusable (malformed or non-finite) on any input.
pub fn mse(program: &str, examples: &[(f64, f64)]) -> f64 {
    if examples.is_empty()
    {
        return 0.0;
    }
    let mut se = 0.0;
    for &(x, y) in examples
    {
        match eval_rpn(program, x)
        {
            Some(p) if p.is_finite() => se += (p - y).powi(2),
            _ => return f64::INFINITY,
        }
    }
    se / examples.len() as f64
}

/// The outcome of [`evolve`].
#[derive(Debug, Clone)]
pub struct Evolved {
    /// The best program found (whitespace-separated RPN tokens).
    pub program: String,
    /// Its mean-squared error on the supplied examples (0 = perfect fit).
    pub mse: f64,
    /// Its raw loop fitness (`-mse` minus a small length penalty).
    pub fitness: Fitness,
    /// The auditable run report (monotone, iterations, stop reason, …).
    pub report: Report,
}

/// Scores a candidate program by `-mse` with a tiny length penalty (Occam's
/// razor — prefer the simpler program among equally accurate ones).
struct FitCritic {
    examples: Vec<(f64, f64)>,
}
impl Critic for FitCritic {
    fn score(&mut self, candidate: &str) -> Fitness {
        if candidate.split_whitespace().next().is_none()
        {
            return -1e9;
        }
        let m = mse(candidate, &self.examples);
        if !m.is_finite()
        {
            return -1e9;
        }
        -m - 0.001 * candidate.split_whitespace().count() as f64
    }
}

/// Built-in generator: proposes mutated variants of the current best program
/// with a few structured edits. Stands in for an LLM so evolution runs offline.
struct MutatingGenerator;
impl MutatingGenerator {
    fn current(prompt: &str) -> Vec<String> {
        let after = prompt
            .split_once(CURRENT_MARKER)
            .map(|(_, rest)| rest.trim_start())
            .unwrap_or("");
        let toks: Vec<String> = after
            .lines()
            .next()
            .unwrap_or("x")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        if toks.is_empty()
        {
            vec!["x".to_string()]
        }
        else
        {
            toks
        }
    }

    fn random_operand(rng: &mut StdRng) -> String {
        match rng.gen_range(0..4)
        {
            0 => "x".to_string(),
            n => (n as f64).to_string(),
        }
    }

    const OPS: [&'static str; 4] = ["+", "-", "*", "/"];

    fn single_edit(mut toks: Vec<String>, rng: &mut StdRng) -> Vec<String> {
        match rng.gen_range(0..6)
        {
            0 =>
            {
                toks.push(Self::random_operand(rng));
                toks.push(Self::OPS[rng.gen_range(0..Self::OPS.len())].to_string());
            },
            1 if !toks.is_empty() =>
            {
                let i = rng.gen_range(0..toks.len());
                toks[i] = if Self::OPS.contains(&toks[i].as_str())
                {
                    Self::OPS[rng.gen_range(0..Self::OPS.len())].to_string()
                }
                else
                {
                    Self::random_operand(rng)
                };
            },
            2 =>
            {
                if let Some(i) = toks.iter().position(|t| t.parse::<f64>().is_ok())
                {
                    let v: f64 = toks[i].parse().unwrap();
                    toks[i] = (v + rng.gen_range(-1.0..1.0)).round().to_string();
                }
            },
            3 =>
            {
                let op = Self::OPS[rng.gen_range(0..Self::OPS.len())].to_string();
                toks.insert(0, op);
                toks.insert(0, "x".to_string());
                toks.insert(0, "x".to_string());
            },
            4 if toks.len() > 1 =>
            {
                toks.pop();
            },
            _ =>
            {
                let i = rng.gen_range(0..=toks.len());
                toks.insert(i, Self::random_operand(rng));
            },
        }
        toks
    }

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

/// Evolve a program fitting the `(input, output)` `examples`, starting from
/// `seed_program` (use `"x"` for "no prior algorithm"). `samples` is the
/// best-of-`n` width per round and `rng_seed` makes the run reproducible. The
/// `guard` bounds the search (iteration cap, optional target/patience).
///
/// The returned program never scores worse than `seed_program` (elitist), and
/// the run always terminates (bounded).
pub fn evolve(
    examples: &[(f64, f64)],
    seed_program: &str,
    samples: usize,
    rng_seed: u64,
    guard: &Guard,
) -> Evolved {
    let seed = if seed_program.split_whitespace().next().is_none()
    {
        "x"
    }
    else
    {
        seed_program
    };
    let mut critic = FitCritic {
        examples: examples.to_vec(),
    };
    let mut generator = MutatingGenerator;
    let (program, fitness, report) = LlmRefine::new(rng_seed)
        .samples(samples.max(1))
        .task(
            "Evolve a reverse-Polish program over `x` (tokens: x, numbers, + - * /) \
             that reproduces the target input→output examples.",
        )
        .run(seed, &mut generator, &mut critic, guard);
    let m = mse(&program, examples);
    Evolved {
        program,
        mse: m,
        fitness,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_handles_ops_and_malformed() {
        assert_eq!(eval_rpn("x x *", 3.0), Some(9.0));
        assert_eq!(eval_rpn("x 1 +", 4.0), Some(5.0));
        assert_eq!(eval_rpn("6 x /", 2.0), Some(3.0));
        assert_eq!(eval_rpn("x 0 /", 1.0), None); // divide by zero
        assert_eq!(eval_rpn("x x", 1.0), None); // unbalanced
        assert_eq!(eval_rpn("x +", 1.0), None); // operator starved
    }

    #[test]
    fn evolves_a_doubling_program() {
        let examples = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)];
        let out = evolve(
            &examples,
            "x",
            32,
            7,
            &Guard::new().max_iters(1500).target(-1e-9),
        );
        assert!(out.report.is_monotone(), "evolution must not regress");
        assert!(
            out.mse < 1e-6,
            "should fit x->2x exactly, got mse {}",
            out.mse
        );
    }

    #[test]
    fn evolves_a_quadratic_from_examples() {
        // y = x^2 + 1
        let examples: Vec<(f64, f64)> =
            (-3..=3).map(|i| (i as f64, (i * i) as f64 + 1.0)).collect();
        let out = evolve(
            &examples,
            "x",
            48,
            2024,
            &Guard::new().max_iters(2500).target(-1e-9),
        );
        assert!(out.report.is_monotone());
        assert!(out.mse < 1e-6, "should fit x^2+1, got mse {}", out.mse);
    }

    #[test]
    fn never_regresses_below_seed() {
        // A perfect seed must be preserved (elitist), never replaced by worse.
        let examples = [(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
        let out = evolve(&examples, "x x *", 16, 1, &Guard::new().max_iters(300));
        assert!(
            out.mse < 1e-9,
            "perfect seed must be kept, got mse {}",
            out.mse
        );
    }
}
