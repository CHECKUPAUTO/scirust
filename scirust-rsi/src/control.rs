//! Shared loop controller used by every driver in the crate.
//!
//! Centralising the bookkeeping here means all the algorithms ([`crate::ascend`],
//! STaR, Expert Iteration, the evolution strategy, PBT and the LLM loop) inherit
//! *identical* termination and non-regression semantics, including the
//! wall-clock [`Guard::time_budget`].

use crate::{Fitness, Guard, Report, StopReason};
use std::time::Instant;

/// Tracks the best-so-far fitness, the history trace, and all stop conditions
/// for one improvement run. Drivers own their payload (the best solution/model)
/// and delegate every numeric/stop decision to this controller.
pub(crate) struct LoopState<'g> {
    guard: &'g Guard,
    best_fit: Fitness,
    history: Vec<Fitness>,
    accepted: usize,
    since_improve: usize,
    iterations: usize,
    start: Instant,
    stop: Option<StopReason>,
}

impl<'g> LoopState<'g> {
    /// Start a run with the incumbent's initial fitness.
    pub fn new(guard: &'g Guard, init_fit: Fitness) -> Self {
        Self {
            guard,
            best_fit: init_fit,
            history: Vec::with_capacity(guard.max_iters.min(4096)),
            accepted: 0,
            since_improve: 0,
            iterations: 0,
            start: Instant::now(),
            stop: None,
        }
    }

    /// 1-based count of iterations begun so far.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Current best-so-far fitness.
    pub fn best_fit(&self) -> Fitness {
        self.best_fit
    }

    /// Begin one iteration. Returns `false` (and records the stop reason) when a
    /// *pre-iteration* budget is exhausted: the iteration cap or the wall-clock
    /// time budget. Always terminates because `max_iters` is finite.
    pub fn next_iter(&mut self) -> bool {
        if self.stop.is_some()
        {
            return false;
        }
        if self.iterations >= self.guard.max_iters
        {
            self.stop = Some(StopReason::MaxIterations);
            return false;
        }
        if let Some(budget) = self.guard.time_budget
        {
            if self.start.elapsed() >= budget
            {
                self.stop = Some(StopReason::TimeBudget);
                return false;
            }
        }
        self.iterations += 1;
        true
    }

    /// Offer a candidate's fitness. Returns `true` iff it strictly improves the
    /// incumbent by more than `min_delta` — the caller should then adopt its
    /// payload. Records the (non-decreasing) best-so-far into the history.
    pub fn offer(&mut self, cand_fit: Fitness) -> bool {
        let improved = cand_fit > self.best_fit + self.guard.min_delta;
        if improved
        {
            self.best_fit = cand_fit;
            self.accepted += 1;
            self.since_improve = 0;
        }
        else
        {
            self.since_improve += 1;
        }
        self.history.push(self.best_fit);
        improved
    }

    /// Check *post-offer* stop conditions (target reached or patience exhausted).
    /// Returns `true` to stop the loop.
    pub fn done(&mut self) -> bool {
        if let Some(t) = self.guard.target
        {
            if self.best_fit >= t
            {
                self.stop = Some(StopReason::TargetReached);
                return true;
            }
        }
        if self.guard.patience > 0 && self.since_improve >= self.guard.patience
        {
            self.stop = Some(StopReason::Converged);
            return true;
        }
        false
    }

    /// Consume the controller into an auditable [`Report`].
    pub fn into_report(self) -> Report {
        Report {
            iterations: self.iterations,
            accepted: self.accepted,
            best_fitness: self.best_fit,
            history: self.history,
            stop_reason: self.stop.unwrap_or(StopReason::MaxIterations),
        }
    }
}
