//! Reproducibility-engine error type.

use thiserror::Error;

/// Errors produced by the reproducibility engine.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReproError {
    /// A verification was given a different number of claims and reproduced
    /// outcomes — every declared node must have exactly one reproduction.
    #[error("verify inputs disagree: {claims} claims vs {reproduced} reproduced outcomes")]
    LengthMismatch {
        /// Number of declared node claims.
        claims: usize,
        /// Number of reproduced outcomes supplied.
        reproduced: usize,
    },
    /// A workflow error occurred while re-realizing (`rerun`) a plan.
    #[error("workflow error during rerun: {0}")]
    Workflow(#[from] sos_workflow::WorkflowError),
    /// The two ledgers ran different plans, so there is nothing to compare —
    /// see [`crate::driver::verify_rerun`] on why this is refused rather than
    /// reported on whatever the runs have in common.
    #[error("ledgers ran different plans: {original} vs {rerun}")]
    PlanMismatch {
        /// The original run's plan digest.
        original: sos_core::Digest,
        /// The re-run's plan digest.
        rerun: sos_core::Digest,
    },
    /// The two ledgers' schedules do not correspond, despite agreeing on the
    /// plan — a hand-edited or corrupted ledger, since a plan's schedule is
    /// deterministic.
    #[error("schedules do not correspond: {detail}")]
    ScheduleMismatch {
        /// What differed.
        detail: String,
    },
    /// A stage emitted a different number of objects on re-run. A stage's
    /// arity is a property of its plugin, which `plan_digest` already pins,
    /// so this is structural rather than a numeric divergence.
    #[error("stage {stage} produced {original} object(s) originally but {rerun} on re-run")]
    OutputArityMismatch {
        /// The stage whose arity changed.
        stage: sos_workflow::StageId,
        /// How many objects it produced originally.
        original: usize,
        /// How many it produced on re-run.
        rerun: usize,
    },
    /// An original output could not be read from the store, so its declared
    /// level is unknown. Never assumed — an unverifiable claim must not
    /// become a reproduced one.
    #[error("cannot read the declared level of {id}{}", .detail.as_ref().map(|d| format!(": {d}")).unwrap_or_default())]
    UnreadableClaim {
        /// The object whose claim could not be read.
        id: sos_core::ObjectId,
        /// Why, when the object was present but unparsable.
        detail: Option<String>,
    },
    /// No recorded run in the store produced this object, so there is nothing
    /// to re-execute.
    #[error("no stored RunLedger records producing {0}")]
    NoProducingLedger(sos_core::ObjectId),
    /// More than one recorded run produced this object. Which run to verify is
    /// the caller's question, not one to guess at — pass the intended ledger to
    /// [`crate::verify_rerun`] directly.
    #[error("{count} stored runs produced {id}; verify a specific one with verify_rerun")]
    AmbiguousProducer {
        /// The object with more than one recorded producer.
        id: sos_core::ObjectId,
        /// How many runs claim it.
        count: usize,
    },
    /// The producing run's plan is not in the store, so it cannot be
    /// re-executed. A ledger records a plan's *digest*, not the plan.
    #[error("the plan {0} that produced this object is not stored")]
    PlanNotStored(sos_core::Digest),
    /// The numeric backend failed to certify an `L2`/`L1` node.
    #[error("certifier failed for stage {stage}: {detail}")]
    Certifier {
        /// The stage whose node needed certifying.
        stage: sos_workflow::StageId,
        /// The backend's reason.
        detail: String,
    },
}

/// Convenience alias for reproducibility results.
pub type Result<T> = core::result::Result<T, ReproError>;
