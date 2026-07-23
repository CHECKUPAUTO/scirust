//! [`rerun`] — re-realize a workflow hermetically under a pinned [`EnvLock`].

use sos_workflow::{Memo, Plan, RunLedger, StageExecutor, run_plan};

use crate::error::Result;
use crate::lock::EnvLock;

/// Re-execute `plan` under the environment pinned by `lock`, returning the
/// [`RunLedger`] of the run (RFC-0002 §09.7).
///
/// The lock's [`env_digest`](EnvLock::env_digest) is the environment component of
/// every stage's cache key, so this is the mechanism that makes reproduction
/// checkable: re-running under a lock that **binds** the original reproduces every
/// stage from cache (identical outputs, nearly free), while re-running under a
/// **drifted** lock changes the environment digest and re-executes — the drift is
/// realized as recomputation, never a silent mismatch.
///
/// # Errors
/// Propagates any [`sos_workflow::StageExecutor::run`] failure as
/// [`crate::ReproError::Workflow`].
pub fn rerun<M: Memo, E: StageExecutor>(
    plan: &Plan,
    lock: &EnvLock,
    memo: &mut M,
    executor: &mut E,
) -> Result<RunLedger> {
    let env = lock.env_digest();
    Ok(run_plan(plan, &env, memo, executor)?)
}
