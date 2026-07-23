//! The memoization table, the [`StageExecutor`] extension point, and
//! [`run_plan`] — the deterministic, memoized scheduler.

use std::collections::BTreeMap;

use sos_core::{Digest, ObjectId};

use crate::cache::CacheKey;
use crate::error::Result;
use crate::ledger::{LedgerStep, RunLedger, StepOutcome};
use crate::plan::{Plan, Stage};

/// A memo of stage results keyed by [`CacheKey`] — the "solved sub-DAG is never
/// re-solved" store. Backends persist this in the object store's named refs; the
/// in-memory [`MemoTable`] is a complete, deterministic implementation.
pub trait Memo {
    /// The recorded outputs for `key`, if this invocation was already solved.
    fn get(&self, key: &CacheKey) -> Option<Vec<ObjectId>>;
    /// Record `outputs` as the result of `key`.
    fn put(&mut self, key: CacheKey, outputs: Vec<ObjectId>);
}

/// A complete in-memory [`Memo`] — a real, deterministic `BTreeMap`-backed table
/// (not a mock). Persistent memoization implements the same trait.
#[derive(Debug, Clone, Default)]
pub struct MemoTable {
    map: BTreeMap<CacheKey, Vec<ObjectId>>,
}

impl MemoTable {
    /// An empty memo table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of solved invocations recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl Memo for MemoTable {
    fn get(&self, key: &CacheKey) -> Option<Vec<ObjectId>> {
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: CacheKey, outputs: Vec<ObjectId>) {
        self.map.insert(key, outputs);
    }
}

/// The pluggable execution of a **cache-missed** stage: read the stage's inputs,
/// append objects, return their ids. This is the one place engine/backend logic
/// enters the scheduler — a curiosity sweep, a reasoning derivation, a
/// simulation. Implementations are provided by the engine crates and backend
/// adapters (Invariant VIII); the scheduler itself never runs a stage's logic.
///
/// The contract is **purity**: given the same [`Stage`], an executor must append
/// the same objects and return the same ids — that is what makes the cache sound.
pub trait StageExecutor {
    /// Execute `stage`, returning the ids of the objects it produced.
    ///
    /// # Errors
    /// Return [`crate::WorkflowError::StageFailed`] if the stage cannot complete;
    /// the scheduler propagates it without corrupting the ledger.
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>>;
}

/// Run `plan` deterministically with content-addressed memoization, returning the
/// [`RunLedger`] of the schedule taken (RFC-0002 §08.2–3).
///
/// For each stage, in the plan's deterministic [schedule](Plan::schedule) order,
/// its [`CacheKey`] is computed under `env_digest`. On a **hit** the recorded
/// outputs are reused and nothing runs; on a **miss** `executor` runs the stage,
/// the result is memoized, and both cases are recorded in the ledger. Re-running
/// an unchanged plan against a warm `memo` is therefore all cache hits — provably
/// identical and nearly free (this is also what makes a crashed run resumable).
///
/// # Errors
/// Propagates any [`StageExecutor::run`] failure.
pub fn run_plan<M: Memo, E: StageExecutor>(
    plan: &Plan,
    env_digest: &Digest,
    memo: &mut M,
    executor: &mut E,
) -> Result<RunLedger> {
    let mut steps = Vec::with_capacity(plan.stages().len());
    for id in plan.schedule()
    {
        let stage = plan.get(&id).expect("scheduled id belongs to the plan");
        let cache_key = stage.cache_key(env_digest);
        let (outcome, outputs) = match memo.get(&cache_key)
        {
            Some(cached) => (StepOutcome::CacheHit, cached),
            None =>
            {
                let produced = executor.run(stage)?;
                memo.put(cache_key, produced.clone());
                (StepOutcome::Ran, produced)
            },
        };
        steps.push(LedgerStep {
            stage: id,
            cache_key,
            outcome,
            outputs,
        });
    }
    Ok(RunLedger {
        plan_digest: plan.digest(),
        env_digest: *env_digest,
        steps,
    })
}
