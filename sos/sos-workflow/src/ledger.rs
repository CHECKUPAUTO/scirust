//! The [`RunLedger`] and its parts: the immutable record of *how* a plan ran.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, Digest, ObjectId};

use crate::cache::CacheKey;
use crate::plan::StageId;

/// Whether a stage's outputs came from cache or from a fresh execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepOutcome {
    /// The [`CacheKey`] was already known — outputs were reused, nothing ran.
    CacheHit,
    /// The stage was executed and its outputs recorded.
    Ran,
}

impl StepOutcome {
    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::CacheHit => "cache-hit",
            Self::Ran => "ran",
        }
    }
}

impl Canonical for StepOutcome {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
    }
}

/// One entry in a [`RunLedger`]: what happened for a single scheduled stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerStep {
    /// The stage that ran (or was reused).
    pub stage: StageId,
    /// Its content-addressed cache key.
    pub cache_key: CacheKey,
    /// Whether it hit the cache or executed.
    pub outcome: StepOutcome,
    /// The object ids it produced (from cache or fresh) — identical either way.
    pub outputs: Vec<ObjectId>,
}

impl Canonical for LedgerStep {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.stage);
        enc.value(&self.cache_key);
        enc.value(&self.outcome);
        enc.seq(&self.outputs);
    }
}

/// The immutable record of a workflow run (RFC-0002 §08.1, SDE 04 §6): the plan
/// that ran, the environment it ran in, and the **schedule actually taken** with
/// every cache hit / miss. Control flow is data too — the ledger makes the
/// engine's *behavior*, not just its outputs, reproducible and auditable.
///
/// A `RunLedger` is itself a content-addressed [`Body`], so "why did the engine
/// run the stages in this order, and what did it reuse?" is a citable object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunLedger {
    /// The digest of the plan that was run.
    pub plan_digest: Digest,
    /// The environment digest the run was keyed against.
    pub env_digest: Digest,
    /// The steps, in the deterministic schedule order.
    pub steps: Vec<LedgerStep>,
}

impl RunLedger {
    /// How many stages actually executed (cache misses).
    #[must_use]
    pub fn ran_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.outcome == StepOutcome::Ran)
            .count()
    }

    /// How many stages were served from cache.
    #[must_use]
    pub fn cache_hit_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.outcome == StepOutcome::CacheHit)
            .count()
    }
}

impl Canonical for RunLedger {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.bytes(self.plan_digest.as_bytes());
        enc.bytes(self.env_digest.as_bytes());
        enc.seq(&self.steps);
    }
}

impl Body for RunLedger {
    const KIND: &'static str = "RunLedger";
    const SCHEMA_VERSION: u32 = 1;
}
