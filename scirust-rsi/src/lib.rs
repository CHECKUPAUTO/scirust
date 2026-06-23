//! # scirust-rsi — Recursive Self-Improvement (bounded & sandboxed)
//!
//! This crate implements the family of algorithms that let a learning system
//! *improve itself*, in the precise, well-understood sense used in the machine
//! learning literature — **not** the science-fiction sense of an unbounded,
//! self-rewriting agent.
//!
//! Every loop here is:
//!
//! - **Bounded** — a [`Guard`] caps iterations, wall-clock-equivalent budget,
//!   and patience, so the process always terminates.
//! - **Monotone** — improvement is *elitist*: a new candidate is adopted only
//!   when it is measurably better than the incumbent, so the best-so-far score
//!   is non-decreasing. The system can never make itself worse.
//! - **Sandboxed** — the algorithms operate on data structures and scalar
//!   objectives. They never execute generated code, touch the host, or modify
//!   their own binary. "Self-improvement" means *the model the system carries
//!   gets better at a measured task*, nothing more.
//! - **Reproducible** — every loop is seeded; the same seed yields the same run.
//!
//! ## The algorithms
//!
//! | Module | Algorithm | The self-improvement signal |
//! |---|---|---|
//! | [`refine`] | **Self-Refine** | critique-and-revise loop on one solution |
//! | [`star`] | **STaR** (Self-Taught Reasoner) | retrain on the system's own correct attempts |
//! | [`expert_iteration`] | **Expert Iteration** | distil a search-augmented "expert" back into the policy |
//! | [`evo`] | **(1+λ)-ES + Rechenberg's 1/5 rule** | the optimiser self-tunes its own mutation strength |
//! | [`pbt`] | **Population-Based Training** | members copy winners and perturb their own hyper-parameters |
//! | [`llm`] | **LLM-driven self-refine** | a language model proposes candidates; best-of-`n`, elitist |
//!
//! All of them share one elitist controller (termination, non-regression, and a
//! wall-clock [`Guard::time_budget`]); [`ascend`] exposes it directly. The
//! [`adapters`] module lets you build a loop from plain closures, no new type
//! needed.
//!
//! ## Quick start
//!
//! ```
//! use scirust_rsi::{Guard, evo::OnePlusLambda};
//!
//! // Maximise -sphere(x)  (i.e. minimise the sphere function) in 5 dims.
//! let opt = OnePlusLambda::new(0xC0FFEE).lambda(8).sigma0(0.5);
//! let guard = Guard::new().max_iters(500).target(-1e-6);
//! let (x, fit, report) = opt.optimize(vec![3.0; 5], |x| -x.iter().map(|v| v * v).sum::<f64>(), &guard);
//!
//! assert!(fit > -1e-3, "should converge near the optimum, got {fit}");
//! assert!(report.is_monotone(), "best-so-far must never decrease");
//! for v in &x { assert!(v.abs() < 1e-1); }
//! ```

#![forbid(unsafe_code)]

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub mod adapters;
mod control;
pub mod evo;
pub mod expert_iteration;
pub mod llm;
pub mod pbt;
pub mod refine;
pub mod star;

pub(crate) use control::LoopState;

/// A scalar quality score. **Higher is always better.** Loops *maximise* it; to
/// minimise a cost, return its negation (see the crate-level example).
pub type Fitness = f64;

// ===========================================================================
// 1. SAFETY GUARD — every loop is bounded and reproducible by construction
// ===========================================================================

/// Why an improvement loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Reached the iteration cap.
    MaxIterations,
    /// Best fitness reached the requested target.
    TargetReached,
    /// No strict improvement for `patience` consecutive iterations.
    Converged,
    /// The wall-clock time budget was exhausted before another iteration began.
    TimeBudget,
}

/// Termination policy shared by every loop in this crate.
///
/// The guard is what turns "recursive self-improvement" into a *terminating,
/// non-regressing* procedure. Construct it with the builder methods:
///
/// ```
/// use scirust_rsi::Guard;
/// let g = Guard::new().max_iters(1_000).patience(50).target(0.99).min_delta(1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct Guard {
    /// Hard cap on iterations. Guarantees termination.
    pub max_iters: usize,
    /// Stop after this many iterations with no strict improvement (0 = never).
    pub patience: usize,
    /// Stop as soon as the best fitness reaches this value (if set).
    pub target: Option<Fitness>,
    /// An improvement must exceed the incumbent by more than this to count.
    pub min_delta: Fitness,
    /// Optional wall-clock budget. Checked before each iteration begins, so the
    /// loop stops once this elapses (it never interrupts an iteration mid-flight).
    pub time_budget: Option<Duration>,
}

impl Default for Guard {
    fn default() -> Self {
        Self {
            max_iters: 1_000,
            patience: 0,
            target: None,
            min_delta: 0.0,
            time_budget: None,
        }
    }
}

impl Guard {
    /// A guard with sensible defaults (1000 iterations, no patience/target).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hard iteration cap.
    pub fn max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }

    /// Stop after `n` iterations without a strict improvement (0 disables it).
    pub fn patience(mut self, n: usize) -> Self {
        self.patience = n;
        self
    }

    /// Stop once the best fitness reaches `t`.
    pub fn target(mut self, t: Fitness) -> Self {
        self.target = Some(t);
        self
    }

    /// Minimum margin a candidate must beat the incumbent by to be adopted.
    pub fn min_delta(mut self, d: Fitness) -> Self {
        self.min_delta = d;
        self
    }

    /// Stop once this much wall-clock time has elapsed (checked between iterations).
    pub fn time_budget(mut self, d: Duration) -> Self {
        self.time_budget = Some(d);
        self
    }
}

// ===========================================================================
// 2. REPORT — an auditable trace of the run
// ===========================================================================

/// An auditable summary of an improvement run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Iterations actually executed.
    pub iterations: usize,
    /// Number of times a strictly-better candidate was adopted.
    pub accepted: usize,
    /// Best fitness found.
    pub best_fitness: Fitness,
    /// Best-so-far fitness at the end of each iteration (length == `iterations`).
    pub history: Vec<Fitness>,
    /// Why the loop stopped.
    pub stop_reason: StopReason,
}

impl Report {
    /// True if the best-so-far trace never decreased — the central safety
    /// property of every loop in this crate.
    pub fn is_monotone(&self) -> bool {
        self.history.windows(2).all(|w| w[1] >= w[0])
    }

    /// Total improvement from first to last recorded iteration.
    pub fn total_gain(&self) -> Fitness {
        match (self.history.first(), self.history.last())
        {
            (Some(a), Some(b)) => b - a,
            _ => 0.0,
        }
    }

    /// Fraction of iterations that adopted a strictly-better candidate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.iterations == 0
        {
            0.0
        }
        else
        {
            self.accepted as f64 / self.iterations as f64
        }
    }

    /// The best-so-far convergence curve as CSV (`iteration,best_fitness`, with a
    /// header and 1-based iteration index). Plot it or diff runs — no deps.
    pub fn history_csv(&self) -> String {
        let mut out = String::from("iteration,best_fitness\n");
        for (i, v) in self.history.iter().enumerate()
        {
            out.push_str(&format!("{},{}\n", i + 1, v));
        }
        out
    }
}

// ===========================================================================
// 3. ASCEND — the elitist primitive every algorithm is built on
// ===========================================================================

/// Elitist, monotone improvement driver.
///
/// Starting from `(initial, init_fit)`, it repeatedly asks `propose` for a
/// candidate derived from the current best, and adopts it **only** if its
/// fitness strictly exceeds the incumbent by more than `guard.min_delta`. The
/// best-so-far fitness is therefore non-decreasing for the whole run.
///
/// `propose(&best, iter, &mut rng) -> (candidate, candidate_fitness)`.
///
/// This is the engine under [`refine`], [`star`], [`evo`] and friends; use it
/// directly when you have an ad-hoc proposal distribution.
pub fn ascend<S, P>(
    initial: S,
    init_fit: Fitness,
    mut propose: P,
    guard: &Guard,
    rng: &mut StdRng,
) -> (S, Report)
where
    P: FnMut(&S, usize, &mut StdRng) -> (S, Fitness),
{
    let mut best = initial;
    let mut ctrl = LoopState::new(guard, init_fit);

    while ctrl.next_iter()
    {
        let iter = ctrl.iterations() - 1; // 0-based index for the caller
        let (cand, cand_fit) = propose(&best, iter, rng);
        if ctrl.offer(cand_fit)
        {
            best = cand;
        }
        if ctrl.done()
        {
            break;
        }
    }

    (best, ctrl.into_report())
}

/// Build a seeded, reproducible RNG. All loops route through this so a given
/// seed always reproduces the same run.
pub(crate) fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ===========================================================================
// 4. BUILT-IN BENCHMARK OBJECTIVES (used by tests and examples)
// ===========================================================================

/// Standard continuous-optimisation test functions, expressed as *fitness*
/// (higher is better, optimum at 0) so they plug straight into the maximisers.
pub mod bench {
    /// `-Σ xᵢ²` — smooth, convex, optimum 0 at the origin.
    pub fn sphere(x: &[f64]) -> f64 {
        -x.iter().map(|v| v * v).sum::<f64>()
    }

    /// Negated Rastrigin — highly multi-modal, optimum 0 at the origin.
    pub fn rastrigin(x: &[f64]) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        let s: f64 = x
            .iter()
            .map(|&v| v * v - a * (2.0 * std::f64::consts::PI * v).cos())
            .sum();
        -(a * n + s)
    }

    /// Negated Rosenbrock — narrow curved valley, optimum 0 at all-ones.
    pub fn rosenbrock(x: &[f64]) -> f64 {
        let s: f64 = x
            .windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum();
        -s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascend_is_monotone_and_elitist() {
        // A noisy proposer that sometimes regresses; ascend must never adopt
        // a worse candidate, so best-so-far is non-decreasing.
        let mut rng = rng_from_seed(1);
        let guard = Guard::new().max_iters(200);
        let (_best, report) = ascend(
            0.0_f64,
            0.0,
            |best, _i, rng| {
                use rand::Rng;
                let cand = best + rng.gen_range(-1.0..1.5); // biased upward, but noisy
                (cand, cand)
            },
            &guard,
            &mut rng,
        );
        assert!(report.is_monotone());
        assert!(report.best_fitness >= 0.0);
        assert!(report.total_gain() > 0.0);
    }

    #[test]
    fn guard_target_stops_early() {
        let mut rng = rng_from_seed(2);
        let guard = Guard::new().max_iters(10_000).target(5.0);
        let (_b, report) = ascend(
            0.0_f64,
            0.0,
            |best, _i, _r| (best + 0.1, best + 0.1),
            &guard,
            &mut rng,
        );
        assert_eq!(report.stop_reason, StopReason::TargetReached);
        assert!(report.best_fitness >= 5.0);
        assert!(report.iterations < 10_000);
    }

    #[test]
    fn guard_patience_detects_convergence() {
        let mut rng = rng_from_seed(3);
        let guard = Guard::new().max_iters(1_000).patience(20);
        // Proposer can never improve past the start -> converges via patience.
        let (_b, report) = ascend(10.0_f64, 10.0, |_best, _i, _r| (0.0, 0.0), &guard, &mut rng);
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert_eq!(report.iterations, 20);
    }

    #[test]
    fn report_serializes_to_json() {
        let r = Report {
            iterations: 3,
            accepted: 2,
            best_fitness: 1.5,
            history: vec![0.0, 1.0, 1.5],
            stop_reason: StopReason::MaxIterations,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Report = serde_json::from_str(&s).unwrap();
        assert_eq!(back.iterations, 3);
        assert!(back.is_monotone());
    }

    #[test]
    fn report_csv_and_acceptance_rate() {
        let r = Report {
            iterations: 4,
            accepted: 2,
            best_fitness: 1.5,
            history: vec![0.0, 1.0, 1.0, 1.5],
            stop_reason: StopReason::MaxIterations,
        };
        assert!((r.acceptance_rate() - 0.5).abs() < 1e-12);
        let csv = r.history_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "iteration,best_fitness");
        assert_eq!(lines[1], "1,0");
        assert_eq!(lines[4], "4,1.5");
        assert_eq!(lines.len(), 1 + r.history.len());
    }
}
