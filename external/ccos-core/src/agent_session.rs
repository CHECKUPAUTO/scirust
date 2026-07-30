//! # Agent session — an event-sourced cognitive timeline (time-travel debugging)
//!
//! This is the capability that a RAG stack (LangGraph, LlamaIndex, …) structurally
//! lacks, and the one our measurements point CCOS toward: not *better retrieval*
//! (a lexical baseline matches that), but a **deterministic, replayable, auditable**
//! record of how an agent's working memory evolved.
//!
//! An [`AgentSession`] wraps a [`CcosMemory`] and records every cognitive operation
//! (ingest, failure signal, recall) as an ordered timeline. Because every operation
//! is deterministic, the state after the first `n` operations can be reconstructed
//! exactly with [`replay_to`](AgentSession::replay_to) — you can *rewind the agent's
//! mind*. And with [`recall_what_if`](AgentSession::recall_what_if) you can replay to
//! a point and re-run a recall under **different parameters** (e.g. a larger token
//! budget) to see whether the agent would have made a better decision — the
//! "time-travel debugger" for an LLM agent's context.
//!
//! ## Example
//!
//! ```
//! use ccos_core::agent_session::AgentSession;
//! use ccos_core::external_memory::{ExternalMemory, Recall};
//!
//! let mut s = AgentSession::new();
//! s.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
//! s.ingest("src/api.rs", "use crate::db;\npub fn handle() -> i64 { db::timeout() }\n");
//! s.signal_failure("file:src/api.rs", 3).ok();
//!
//! // A tight budget recalls a thin window…
//! let tight = s.recall(Recall::around("file:src/api.rs"), 30);
//! // …rewind to that step and replay with a larger budget — the what-if.
//! let roomy = s.recall_what_if(s.len() - 1, &Recall::around("file:src/api.rs"), 4000);
//! assert!(roomy.items.len() >= tight.items.len());
//!
//! // Replay reconstructs the exact state (determinism / auditability).
//! assert_eq!(s.replay_to(s.len()).stats().nodes, s.memory().stats().nodes);
//! ```

use crate::compressor::{CausalCompressor, CcrRef};
use crate::event_log::LogIntegrity;
use crate::external_memory::{
    CcosMemory, ExternalMemory, IngestReport, MemoryError, Recall, RecallWindow,
};
use crate::license::{Feature, LicenseError, Licensing};
use crate::memory::ScoringWeights;
use crate::trace::parse_cargo_test_output;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One recorded cognitive operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Op {
    Ingest {
        uri: String,
        source: String,
    },
    Failure {
        node: String,
        depth: u32,
    },
    Recall {
        recall: Recall,
        budget: usize,
    },
    /// A compiler/test failure fed back in: pressure injected on the faulting
    /// files (the "page fault"), then a refreshed recall around them. The
    /// propagation `depth` is recorded so replay reproduces the exact pressure
    /// (old logs predate it and default to the historical depth of 2).
    PageFault {
        files: Vec<String>,
        #[serde(default = "legacy_page_fault_depth")]
        depth: u32,
    },
    /// The causal scoring weights were **retuned** from the replayable log (slice
    /// C). Logged so the change is auditable *and* reproduced on replay — the
    /// learned policy is part of the timeline, not a side channel, so
    /// `replay == live` still holds.
    Retune {
        weights: ScoringWeights,
    },
    /// An explicit **dual-evidence assertion** (Q-Pages): `evidence` supports (`supports = true`)
    /// or contradicts (`false`) `claim`, with `weight`. Logged so the agent-asserted belief surface
    /// is part of the replayable timeline — `replay == live` holds for contradictions too, not just
    /// for ingested structure. Appended last so old logs (which never contain it) stay readable.
    Assert {
        evidence: String,
        claim: String,
        supports: bool,
        weight: f64,
    },
}

/// Failure-propagation depth a `page_fault` injects. Configurable via
/// `CCOS_PAGE_FAULT_DEPTH` (default 3): deeper reaches a cause further down the
/// dependency chain from the symptom, at the cost of less focus. A field run
/// showed depth 2 left a 3-hop cause un-pressurised, hence the bump.
fn page_fault_depth() -> u32 {
    std::env::var("CCOS_PAGE_FAULT_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// The depth `PageFault` ops used before it was recorded — for replaying old logs.
fn legacy_page_fault_depth() -> u32 {
    2
}

/// Whether a recall window holds the engaged node `uri` at **file granularity**:
/// a window item for the same source file — its file node *or* any of its symbol
/// nodes — counts as the relevant context having been surfaced. Used by the
/// log-tuned retrieval reward (slice C).
fn window_holds(win: &RecallWindow, uri: &str) -> bool {
    let want = engaged_path(uri);
    win.items.iter().any(|it| engaged_path(&it.uri) == want)
}

/// The source path a node uri refers to, stripped of any `kind:` prefix and
/// `:symbol` suffix — so `sym:src/x.rs:foo`, `file:src/x.rs`, and a bare
/// `src/x.rs` all reduce to `src/x.rs`.
fn engaged_path(uri: &str) -> &str {
    let rest = ["file:", "sym:", "mod:", "use:", "dep:"]
        .iter()
        .find_map(|p| uri.strip_prefix(p))
        .unwrap_or(uri);
    rest.split(':').next().unwrap_or(rest)
}

/// The on-disk timeline sidecar (`<workspace>.oplog`): the replay **baseline**
/// the session started from, the ordered op-log, and the number of older ops
/// already folded into the baseline by compaction. Persisting these lets the
/// whole cognitive timeline — not just the memory snapshot — survive a restart,
/// so [`replay_to`](AgentSession::replay_to) / time-travel work across runs.
#[derive(Serialize, Deserialize)]
struct PersistedTimeline {
    baseline: Option<String>,
    ops: Vec<Op>,
    /// Ops folded into `baseline` by compaction (absent in pre-compaction logs).
    #[serde(default)]
    folded: usize,
    /// Tamper-evident hash chain over the op tail: `chain[i]` is the SHA-256 link
    /// for `ops[i]` (see [`op_link_hash`]). Absent in pre-chain logs (backfilled
    /// on open — the chain protects the timeline from then on).
    #[serde(default)]
    chain: Vec<String>,
    /// The chain predecessor of `ops[0]`: [`TIMELINE_GENESIS`] for an uncompacted
    /// timeline, or the head of the folded prefix after compaction — so the
    /// surviving chain provably extends the folded history and the head stays a
    /// commitment to *every* op since genesis.
    #[serde(default)]
    anchor: String,
    /// SHA-256 commitment to `baseline` (empty when the timeline replays from
    /// empty). Ops alone don't pin the state the tail replays *on top of*; this
    /// does, so a baseline edit is as detectable as an op edit.
    #[serde(default)]
    baseline_hash: String,
    /// This session's agent identity for multi-agent sync (empty = unset).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    agent: String,
    /// Imported foreign timelines, keyed by agent id (empty for a solo session).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    foreign: std::collections::BTreeMap<String, ForeignLog>,
    /// TOFU-pinned signing keys (agent id → base64url ed25519 public key) —
    /// see [`AgentSession::import_bundle`]. Absent for never-signed federations.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    known_keys: std::collections::BTreeMap<String, String>,
}

/// One foreign agent's imported timeline segment, as persisted in the sidecar:
/// its ops from genesis, their chain links, and nothing else — the chain *is*
/// the provenance. Grow-only: imports may only extend it (see
/// [`AgentSession::import_bundle`]).
#[derive(Clone, Serialize, Deserialize)]
struct ForeignLog {
    ops: Vec<Op>,
    chain: Vec<String>,
}

/// A portable, chain-verified segment of one agent's timeline — the transport
/// unit of the **distributed multi-agent store** (paper §9 item 5). Produced by
/// [`AgentSession::export_bundle`], consumed by [`AgentSession::import_bundle`];
/// a plain JSON file, so the exchange works over any medium — including none at
/// all (sneakernet), preserving the air-gappable posture. The receiver trusts
/// nothing: every link is re-verified against the ops, and a segment must
/// *extend* the already-known chain of its agent, so a mutated bundle **or an
/// equivocating agent** (two divergent histories under one identity) is
/// detected, not merged.
#[derive(Clone, Serialize, Deserialize)]
pub struct SyncBundle {
    /// Bundle format version (additive evolution).
    pub version: u32,
    /// The exporting agent's identity (must be non-empty and unique per agent).
    pub agent: String,
    /// Absolute step of `ops[0]` in the exporter's timeline (0 = from genesis).
    pub first_seq: usize,
    /// Chain predecessor of `ops[0]` — the genesis marker when `first_seq == 0`.
    pub anchor: String,
    ops: Vec<Op>,
    chain: Vec<String>,
    /// The exporter's base64url ed25519 public key, present iff the bundle is
    /// signed (`signed-sync` builds sign automatically when the workspace has a
    /// key — see [`generate_workspace_key`]). Additive: unsigned bundles omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pubkey: Option<String>,
    /// base64url ed25519 signature over the bundle's canonical payload (its own
    /// JSON with the signature fields blanked), present iff signed. The hash
    /// chain proves *integrity*; this proves
    /// *authenticity* — only the key holder can speak as this agent id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,
}

impl SyncBundle {
    /// Ops carried by this bundle.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the bundle carries no ops.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Serialize for transport (plain JSON — inspectable, diffable, air-gap friendly).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// The canonical byte string a bundle signature covers: the bundle's own
    /// compact JSON with the signature fields blanked. Field order is the struct's
    /// declaration order (serde), so signer and verifier agree byte-for-byte.
    #[cfg_attr(not(feature = "signed-sync"), allow(dead_code))]
    fn signing_payload(&self) -> String {
        let mut unsigned = self.clone();
        unsigned.pubkey = None;
        unsigned.sig = None;
        unsigned.to_json().unwrap_or_default()
    }

    /// Parse a transported bundle. Verification happens at import, not here.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Why a sync export/import was refused. Every refusal is a *detected* condition —
/// the sync layer never silently drops or merges questionable history.
#[derive(Debug, PartialEq, Eq)]
pub enum SyncError {
    /// The session has no agent identity (see [`AgentSession::set_agent`]).
    NoAgentId,
    /// A bundle authored by this session's own agent id cannot be imported.
    SelfImport,
    /// The requested range reaches into compacted history — no longer separable
    /// into ops. Export earlier, or run federated agents with compaction off
    /// (`CCOS_OPLOG_MAX=0`).
    CompactedHistory,
    /// The bundle starts past the locally-known end of that agent's log.
    Gap {
        /// Ops of this agent known locally.
        known: usize,
        /// Where the bundle starts instead.
        bundle_start: usize,
    },
    /// A link in the bundle fails verification: the bundle was mutated in transit.
    Tampered(String),
    /// The bundle's overlap disagrees with the locally-known chain: the agent
    /// published two different histories under one identity (equivocation) —
    /// exactly what the per-agent hash chain exists to catch.
    Diverged {
        /// First absolute step whose link disagrees.
        at: usize,
    },
    /// The bundle's ed25519 signature does not verify over its canonical payload
    /// (or its key/signature fields are malformed).
    BadSignature(String),
    /// The bundle is signed with a different key than the one TOFU-pinned for
    /// this agent id — an identity-spoof or an unannounced key swap; both refuse.
    KeyMismatch {
        /// The claimed agent id.
        agent: String,
    },
    /// An UNSIGNED bundle arrived from an agent whose key is pinned — a
    /// downgrade attempt (stripping the signature must not bypass pinning).
    UnsignedFromPinned {
        /// The claimed agent id.
        agent: String,
    },
    /// The bundle is signed but this build has no signature support — refused
    /// rather than silently accepted unverified. Rebuild with
    /// `--features signed-sync` to verify it.
    SignedUnsupported,
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyncError::NoAgentId => write!(f, "session has no agent id (set one first)"),
            SyncError::SelfImport => write!(f, "bundle was authored by this agent"),
            SyncError::CompactedHistory => write!(
                f,
                "range reaches into compacted history (export earlier or set CCOS_OPLOG_MAX=0)"
            ),
            SyncError::Gap {
                known,
                bundle_start,
            } => write!(
                f,
                "bundle starts at step {bundle_start} but only {known} op(s) are known — missing intermediate bundle"
            ),
            SyncError::Tampered(detail) => write!(f, "bundle failed verification: {detail}"),
            SyncError::Diverged { at } => write!(
                f,
                "agent equivocation detected: bundle disagrees with the known chain at step {at}"
            ),
            SyncError::BadSignature(detail) => {
                write!(f, "bundle signature does not verify: {detail}")
            }
            SyncError::KeyMismatch { agent } => write!(
                f,
                "bundle from '{agent}' is signed with a different key than the one pinned for that agent — identity spoof or unannounced key swap"
            ),
            SyncError::UnsignedFromPinned { agent } => write!(
                f,
                "unsigned bundle from '{agent}', whose key is pinned — a stripped signature does not bypass pinning"
            ),
            SyncError::SignedUnsupported => write!(
                f,
                "bundle is signed but this build cannot verify signatures — rebuild with --features signed-sync"
            ),
        }
    }
}

impl std::error::Error for SyncError {}

/// Genesis predecessor for the first link of an uncompacted timeline chain.
const TIMELINE_GENESIS: &str = "GENESIS";

/// SHA-256 link over `(prev, absolute step, op)`. Ops carry no wall-clock ids or
/// timestamps (the timeline is indexed by logical step), so the *whole* record is
/// hashed and the chain is bit-reproducible across runs — the same
/// exclude-nothing-nondeterministic rule as [`EventLog`](crate::event_log::EventLog).
fn op_link_hash(prev: &str, seq: usize, op: &Op) -> String {
    let op_json = serde_json::to_string(op).unwrap_or_default();
    crate::util::sha256_hex(&format!("{prev}|{seq}|{op_json}"))
}

/// The chain commitment to a replay baseline (empty string for "replays from empty").
fn baseline_commitment(baseline: Option<&str>) -> String {
    baseline.map(crate::util::sha256_hex).unwrap_or_default()
}

/// Recompute the whole chain for `ops` from `anchor` at compaction floor `folded` —
/// used to backfill pre-chain sidecars and by tests.
fn chain_over(anchor: &str, folded: usize, ops: &[Op]) -> Vec<String> {
    let mut chain = Vec::with_capacity(ops.len());
    let mut prev = anchor.to_string();
    for (i, op) in ops.iter().enumerate() {
        let h = op_link_hash(&prev, folded + i, op);
        chain.push(h.clone());
        prev = h;
    }
    chain
}

/// Verify a timeline's canonical hash chain: the baseline commitment, the
/// link count, and every link from the anchor forward. Pure — shared by
/// [`AgentSession::verify_timeline`] and [`audit_workspace`] so the CLI can audit
/// a sidecar without opening (and potentially self-healing) the session.
fn verify_timeline_parts(
    anchor: &str,
    folded: usize,
    ops: &[Op],
    chain: &[String],
    baseline: Option<&str>,
    baseline_hash: &str,
) -> LogIntegrity {
    let mut errors = Vec::new();
    let mut verified = 0usize;

    // A pre-chain sidecar has nothing to verify against — reported as valid with
    // zero verified links (the caller can surface "legacy"); the chain is
    // established the next time the session checkpoints.
    if chain.is_empty() && anchor.is_empty() && baseline_hash.is_empty() {
        return LogIntegrity {
            valid: true,
            verified_events: 0,
            errors,
        };
    }

    if baseline_commitment(baseline) != baseline_hash {
        errors.push("baseline does not match its recorded commitment".to_string());
    }
    if chain.len() != ops.len() {
        errors.push(format!(
            "chain has {} link(s) for {} op(s): an op was inserted or removed",
            chain.len(),
            ops.len()
        ));
    }
    let mut prev = anchor.to_string();
    for (i, (op, link)) in ops.iter().zip(chain.iter()).enumerate() {
        let seq = folded + i;
        if op_link_hash(&prev, seq, op) == *link {
            verified += 1;
        } else {
            errors.push(format!("link t={} broken: op or hash mutated", seq + 1));
        }
        prev = link.clone();
    }
    LogIntegrity {
        valid: errors.is_empty(),
        verified_events: verified,
        errors,
    }
}

/// A CLI-facing audit of a workspace's timeline sidecar — see [`audit_workspace`].
#[derive(Debug)]
pub struct TimelineAudit {
    /// Chain verification result (baseline commitment + every link).
    pub integrity: LogIntegrity,
    /// Ops in the live tail (on top of `folded` compacted ones).
    pub ops: usize,
    /// Ops already folded into the baseline by compaction.
    pub folded: usize,
    /// The chain head — a single hash committing to the whole recorded history.
    pub head: String,
    /// True for a pre-chain sidecar: nothing to verify yet; the chain is
    /// established on the session's next checkpoint.
    pub legacy: bool,
}

/// Audit the tamper-evident chain of a workspace's `.oplog` sidecar **without**
/// opening the session (so a tampered or stale timeline is *reported*, never
/// healed or rejected as a side effect). `None` when the workspace has no
/// sidecar; a sidecar that no longer parses is reported as invalid.
pub fn audit_workspace(path: impl AsRef<Path>) -> Option<TimelineAudit> {
    let file = crate::external_memory::workspace_file(path.as_ref());
    let sidecar = oplog_sidecar(&file);
    let raw = std::fs::read_to_string(&sidecar).ok()?;
    let Ok(t) = serde_json::from_str::<PersistedTimeline>(&raw) else {
        return Some(TimelineAudit {
            integrity: LogIntegrity {
                valid: false,
                verified_events: 0,
                errors: vec!["sidecar does not parse as a timeline".to_string()],
            },
            ops: 0,
            folded: 0,
            head: String::new(),
            legacy: false,
        });
    };
    let legacy = t.chain.is_empty() && t.anchor.is_empty() && t.baseline_hash.is_empty();
    let integrity = verify_timeline_parts(
        &t.anchor,
        t.folded,
        &t.ops,
        &t.chain,
        t.baseline.as_deref(),
        &t.baseline_hash,
    );
    let head = t.chain.last().cloned().unwrap_or(t.anchor);
    Some(TimelineAudit {
        integrity,
        ops: t.ops.len(),
        folded: t.folded,
        head,
        legacy,
    })
}

/// Per-source **authority overrides** for assertions — the Pro `CustomAuthorityWeights` knob. Maps an
/// evidence/source node id to the authority weight its assertions should carry, overriding the
/// per-call weight. Empty is the default — and the only state a community session can reach, since
/// [`AgentSession::set_custom_authorities`] is license-gated — so without Pro every source keeps its
/// uniform per-call authority (today's behaviour, unchanged).
#[derive(Debug, Clone, Default)]
pub struct CustomAuthorityMap {
    weights: std::collections::HashMap<String, f64>,
}

impl CustomAuthorityMap {
    /// An empty map (no overrides).
    pub fn new() -> Self {
        Self::default()
    }

    /// Override `source`'s assertion authority with `weight` (clamped to `[0, 1]`). Chainable.
    pub fn set(&mut self, source: impl Into<String>, weight: f64) -> &mut Self {
        self.weights.insert(source.into(), weight.clamp(0.0, 1.0));
        self
    }

    /// The override for `source`, if one was set.
    pub fn get(&self, source: &str) -> Option<f64> {
        self.weights.get(source).copied()
    }

    /// Whether no overrides are set (the community default).
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// The recorded operation a node's causal-score drift is attributed to, from
/// [`AgentSession::attribute_drift`].
#[derive(Debug, Clone, PartialEq)]
pub struct DriftCause {
    /// The node whose trajectory was analysed.
    pub node: String,
    /// Absolute timeline step of the culprit operation (the `t=` index in [`AgentSession::timeline`]).
    pub step: usize,
    /// Signed score change across the break: positive = the node's score rose (it was pulled *into*
    /// the working set), negative = it fell (drifted *out*).
    pub delta: f64,
    /// The CUSUM statistic — how pronounced the level shift is.
    pub cusum: f64,
    /// The human-readable journal line for the culprit operation.
    pub op: String,
}

/// An event-sourced agent memory session: every operation is recorded so the
/// state is replayable and auditable. See the [module docs](crate::agent_session).
pub struct AgentSession {
    live: CcosMemory,
    ops: Vec<Op>,
    /// JSON snapshot of the memory the session's timeline replays *on top of*. A
    /// freshly [`new`](Self::new) session has none (the baseline is empty); a
    /// session [`open`](Self::open)ed from a checkpoint carries either the state
    /// it was seeded from (so [`replay_to`](Self::replay_to) layers the timeline
    /// on top of it) or `None` when the op-log reproduces the memory from empty.
    /// Compaction also folds older ops into this baseline.
    baseline: Option<String>,
    /// Number of older ops folded into `baseline` by [`compact`](Self::compact).
    /// Timeline indices stay absolute: logical length is `folded + ops.len()`, and
    /// a `replay_to(step)` for `step <= folded` collapses to the baseline (the
    /// compaction floor). Keeps the op-log bounded for a long-running daemon.
    folded: usize,
    /// Tamper-evident SHA-256 chain over `ops` (`chain[i]` links `ops[i]` to its
    /// predecessor — see [`op_link_hash`]). Passive metadata: replay never reads
    /// it, so `replay == live` is untouched; [`verify_timeline`](Self::verify_timeline)
    /// and `ccos verify` read it to prove the recorded history unmutated.
    chain: Vec<String>,
    /// Chain predecessor of `ops[0]` — [`TIMELINE_GENESIS`], or the folded
    /// prefix's head after compaction (head continuity across folds).
    anchor: String,
    /// Commitment to `baseline`, so the state the tail replays on top of is as
    /// tamper-evident as the ops themselves.
    baseline_hash: String,
    /// Agent identity for multi-agent sync (empty until [`set_agent`](Self::set_agent)).
    agent: String,
    /// Chain-verified foreign timelines imported from other agents, keyed by
    /// agent id. Grow-only; never mixed into the own timeline — the merged
    /// knowledge is materialized on demand by [`merged_view`](Self::merged_view).
    foreign: std::collections::BTreeMap<String, ForeignLog>,
    /// TOFU-pinned signing keys: agent id → base64url ed25519 public key, set by
    /// the first *signed* bundle imported from that agent and immutable after —
    /// a later bundle under the same id with a different key, or an unsigned one,
    /// is refused. Persisted in the sidecar (additive).
    known_keys: std::collections::BTreeMap<String, String>,
    /// This workspace's ed25519 signing seed, loaded from the `<workspace>.ccos.key`
    /// sidecar when the `signed-sync` build finds one (`None` otherwise — including
    /// always on a crypto-free build, which never reads the key file). Never
    /// persisted in the timeline sidecar.
    signing_seed: Option<[u8; 32]>,
    /// Where the timeline sidecar lives (`<workspace>.oplog`); `None` for an
    /// in-memory [`new`](Self::new) session.
    oplog_path: Option<PathBuf>,
    /// The reversible context-compression pipeline (CCR store). Owns the
    /// originals cache so the host LLM can call `ccos_retrieve` across calls.
    /// See [`crate::compressor`].
    compressor: CausalCompressor,
    /// Runtime licensing state — loaded fresh from the host on `open`, never
    /// persisted (it is not part of the serialized timeline, so `replay == live`
    /// is unaffected). Gates the Pro feature surface; the core is never touched.
    licensing: Licensing,
    /// Per-source authority overrides (the Pro `CustomAuthorityWeights` knob).
    /// Empty unless installed via the license-gated
    /// [`set_custom_authorities`](Self::set_custom_authorities). Its *effect* flows
    /// into the logged `Op::Assert` weight, so it needs no persistence of its own.
    custom_authorities: CustomAuthorityMap,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    /// A fresh in-memory session with no checkpoint path (nothing is persisted).
    pub fn new() -> Self {
        let licensing = Licensing::community();
        let mut live = CcosMemory::new();
        // An in-memory session is always community, so it uses the uniform INT4 embedding
        // fallback (the grouped SLHAv2 scheme is a Pro [`Feature::SlhAv2Embeddings`]).
        // File-backed [`open`](Self::open) derives this same flag from the host license tier.
        live.set_slhav2_embeddings(
            licensing.allows(Feature::SlhAv2Embeddings, crate::license::now_unix()),
        );
        AgentSession {
            live,
            ops: Vec::new(),
            baseline: None,
            folded: 0,
            chain: Vec::new(),
            anchor: TIMELINE_GENESIS.to_string(),
            baseline_hash: baseline_commitment(None),
            agent: String::new(),
            foreign: std::collections::BTreeMap::new(),
            known_keys: std::collections::BTreeMap::new(),
            signing_seed: None,
            oplog_path: None,
            compressor: CausalCompressor::new(),
            licensing,
            custom_authorities: CustomAuthorityMap::new(),
        }
    }

    /// Open a **persistent** session backed by `path`: load the causal memory
    /// from the checkpoint if it exists (otherwise start empty), bind the path
    /// for [`checkpoint`](Self::checkpoint), and restore the cognitive timeline
    /// from the sidecar (`<path>.oplog`) when present. The memory snapshot is the
    /// same form `ccos memory` reads/writes, so both transports can share one
    /// `workspace.ccos`; the sidecar adds the op-log on top.
    ///
    /// With a sidecar, [`replay_to`](Self::replay_to) / time-travel span the whole
    /// recorded history across restarts. The op-log is trusted only if it
    /// reproduces the loaded memory exactly; on a mismatch (e.g. the snapshot was
    /// mutated out-of-band by `ccos memory`) the snapshot wins and the timeline
    /// restarts from it — the memory is never corrupted by a stale log.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        // Resolve a directory workspace to its state file so the snapshot and the
        // op-log sidecar land together (see CcosMemory::open).
        let file = crate::external_memory::workspace_file(path.as_ref());
        let mut live = CcosMemory::open(&file)?;
        let oplog_path = oplog_sidecar(&file);

        // Restore the timeline from the sidecar if one is there and parses; a
        // missing/corrupt sidecar just means "no recorded history yet".
        let restored = std::fs::read_to_string(&oplog_path)
            .ok()
            .and_then(|s| serde_json::from_str::<PersistedTimeline>(&s).ok());

        // Licensing is loaded fresh from the host (env / file), never from the timeline — so it is
        // independent of replay, and a shared workspace runs correctly under any tier.
        let now = crate::license::now_unix();
        let licensing = Licensing::detect(now);

        // The SLHAv2 grouped-INT4 embedding scheme is a Pro feature
        // ([`Feature::SlhAv2Embeddings`]): a Pro session keeps the adaptive grouped quantization,
        // a community session falls back to uniform INT4. Decided silently via `allows` (not
        // `require`) so opening a community session does not log a refusal on every open; an
        // explicit request goes through [`enable_slhav2_embeddings`](Self::enable_slhav2_embeddings).
        // Like licensing itself, the scheme reflects the host tier, not the timeline — the core
        // recall path is unchanged, only the embedding precision varies with the tier.
        live.set_slhav2_embeddings(licensing.allows(Feature::SlhAv2Embeddings, now));

        let mut session = match restored {
            Some(mut t) => {
                // A pre-chain sidecar has no chain to check — establish one now
                // (deterministic backfill), so the history is protected from here
                // on. A *chained* sidecar must verify before it is trusted:
                // a broken link means the recorded history was mutated on disk,
                // and silently self-healing would destroy the evidence — so open
                // refuses, leaving the sidecar intact for forensics
                // (`ccos verify <workspace>` reports the broken links).
                if t.chain.is_empty() && t.anchor.is_empty() && t.baseline_hash.is_empty() {
                    t.anchor = TIMELINE_GENESIS.to_string();
                    t.baseline_hash = baseline_commitment(t.baseline.as_deref());
                    t.chain = chain_over(&t.anchor, t.folded, &t.ops);
                } else {
                    let integrity = verify_timeline_parts(
                        &t.anchor,
                        t.folded,
                        &t.ops,
                        &t.chain,
                        t.baseline.as_deref(),
                        &t.baseline_hash,
                    );
                    if !integrity.valid {
                        return Err(MemoryError::TimelineTampered(integrity.errors.join("; ")));
                    }
                }
                AgentSession {
                    live,
                    ops: t.ops,
                    baseline: t.baseline,
                    folded: t.folded,
                    chain: t.chain,
                    anchor: t.anchor,
                    baseline_hash: t.baseline_hash,
                    agent: t.agent,
                    foreign: t.foreign,
                    known_keys: t.known_keys,
                    signing_seed: load_signing_seed(&file),
                    oplog_path: Some(oplog_path),
                    compressor: CausalCompressor::new(),
                    licensing,
                    custom_authorities: CustomAuthorityMap::new(),
                }
            }
            None => {
                let baseline = Some(live.to_json()?);
                let baseline_hash = baseline_commitment(baseline.as_deref());
                AgentSession {
                    live,
                    ops: Vec::new(),
                    baseline,
                    folded: 0,
                    chain: Vec::new(),
                    anchor: TIMELINE_GENESIS.to_string(),
                    baseline_hash,
                    agent: String::new(),
                    foreign: std::collections::BTreeMap::new(),
                    known_keys: std::collections::BTreeMap::new(),
                    signing_seed: load_signing_seed(&file),
                    oplog_path: Some(oplog_path),
                    compressor: CausalCompressor::new(),
                    licensing,
                    custom_authorities: CustomAuthorityMap::new(),
                }
            }
        };

        // Consistency guard: a restored op-log must reproduce the loaded memory.
        // If not, trust the snapshot (authoritative state) and reset the timeline.
        // Reaching here the chain verified, so this is legitimate staleness (an
        // out-of-band `ccos memory` write), not tampering.
        if !session.ops.is_empty() && !session.timeline_reproduces_memory() {
            session.baseline = Some(session.live.to_json()?);
            session.ops.clear();
            session.folded = 0;
            session.chain.clear();
            session.anchor = TIMELINE_GENESIS.to_string();
            session.baseline_hash = baseline_commitment(session.baseline.as_deref());
        }
        Ok(session)
    }

    /// Persist the live memory **and** the cognitive timeline. First **compacts**
    /// the op-log if it has grown past the threshold (so a long-running daemon
    /// stays bounded), then writes the memory snapshot to the bound checkpoint path
    /// (shared with `ccos memory`) and the baseline + op-log to the `<path>.oplog`
    /// sidecar. Returns [`MemoryError::NoPath`] for an in-memory `new` session (the
    /// caller can treat that as a no-op).
    pub fn checkpoint(&mut self) -> Result<(), MemoryError> {
        self.compact_if_needed();
        self.live.checkpoint()?;
        if let Some(p) = &self.oplog_path {
            let timeline = PersistedTimeline {
                baseline: self.baseline.clone(),
                ops: self.ops.clone(),
                folded: self.folded,
                chain: self.chain.clone(),
                anchor: self.anchor.clone(),
                baseline_hash: self.baseline_hash.clone(),
                agent: self.agent.clone(),
                foreign: self.foreign.clone(),
                known_keys: self.known_keys.clone(),
            };
            crate::util::write_durable(p, serde_json::to_string(&timeline)?.as_bytes())?;
        }
        Ok(())
    }

    /// Fold all but the most recent `keep` operations into the baseline snapshot,
    /// bounding the op-log. The live memory is unchanged — only the *representation*
    /// of the timeline (a newer baseline plus a shorter tail). Recent rewind depth
    /// (the last `keep` ops) is preserved; [`replay_to`](Self::replay_to) below the
    /// new floor collapses to the baseline. A no-op when `ops.len() <= keep`.
    pub fn compact(&mut self, keep: usize) {
        if self.ops.len() <= keep {
            return;
        }
        let fold = self.ops.len() - keep;
        let floor = self.replay_to(self.folded + fold);
        if let Ok(snapshot) = floor.to_json() {
            // The folded prefix's last link becomes the anchor, so the surviving
            // chain still extends the folded history: the head is unchanged by a
            // compaction and keeps committing to every op since genesis.
            self.anchor = self.chain[fold - 1].clone();
            self.chain.drain(0..fold);
            self.baseline_hash = baseline_commitment(Some(&snapshot));
            self.baseline = Some(snapshot);
            self.ops.drain(0..fold);
            self.folded += fold;
        }
    }

    /// Compact when the op-log exceeds `CCOS_OPLOG_MAX` (default 512), folding down
    /// to `CCOS_OPLOG_KEEP` (default 128) retained ops. `CCOS_OPLOG_MAX=0` disables.
    fn compact_if_needed(&mut self) {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let (max, keep) = (env("CCOS_OPLOG_MAX", 512), env("CCOS_OPLOG_KEEP", 128));
        if max > 0 && self.ops.len() > max {
            self.compact(keep);
        }
    }

    /// Whether replaying the recorded op-log on top of the baseline reproduces the
    /// live memory's structure (nodes/edges/files) — the invariant `open` checks
    /// before trusting a restored timeline.
    fn timeline_reproduces_memory(&self) -> bool {
        let replayed = self.replay_to(self.len());
        let (a, b) = (replayed.stats(), self.live.stats());
        a.nodes == b.nodes && a.edges == b.edges && a.files == b.files
    }

    /// Append `op` to the timeline **and** extend the tamper-evident chain — the
    /// single write path for every recorded operation.
    fn record(&mut self, op: Op) {
        let prev = self
            .chain
            .last()
            .cloned()
            .unwrap_or_else(|| self.anchor.clone());
        let seq = self.folded + self.ops.len();
        self.chain.push(op_link_hash(&prev, seq, &op));
        self.ops.push(op);
    }

    /// Verify the timeline's canonical hash chain — the baseline commitment and
    /// every op link from the anchor forward. Any post-hoc mutation of a recorded
    /// op, a reorder, an insertion/deletion, or a baseline edit breaks it. Purely
    /// a read: replay never consults the chain, so `replay == live` is untouched.
    pub fn verify_timeline(&self) -> LogIntegrity {
        verify_timeline_parts(
            &self.anchor,
            self.folded,
            &self.ops,
            &self.chain,
            self.baseline.as_deref(),
            &self.baseline_hash,
        )
    }

    /// The chain head — one hash committing to the entire recorded history since
    /// genesis (compaction moves the anchor, never the head). Two sessions with
    /// the same head provably recorded the same timeline.
    pub fn timeline_head(&self) -> String {
        self.chain
            .last()
            .cloned()
            .unwrap_or_else(|| self.anchor.clone())
    }

    // ── Distributed multi-agent store (paper §9 item 5) ──────────────────────────
    //
    // Design: every agent keeps exactly one append-only, hash-chained timeline of
    // its OWN ops. Sharing is the exchange of chain-verified segments (SyncBundle,
    // a plain JSON file — any transport, including sneakernet, so the air-gappable
    // posture survives federation). Imports are stored per agent and NEVER mixed
    // into the own timeline: the shared brain is a pure function, `merged_view`,
    // that replays every known timeline in canonical agent order from empty. Two
    // agents holding the same set of timelines therefore materialize bit-identical
    // views — convergence by construction, no consensus protocol, no network, no
    // new dependency, and `replay == live` untouched.

    /// Set this session's **agent identity** for multi-agent sync (persisted in
    /// the sidecar). Must be non-empty and unique among the exchanging agents.
    pub fn set_agent(&mut self, id: impl Into<String>) {
        self.agent = id.into();
    }

    /// This session's agent identity (empty until [`set_agent`](Self::set_agent)).
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// Imported foreign agents, each with how many of their ops are known and
    /// their chain head, in canonical (sorted) order.
    pub fn foreign_agents(&self) -> Vec<(String, usize, String)> {
        self.foreign
            .iter()
            .map(|(id, log)| {
                let head = log
                    .chain
                    .last()
                    .cloned()
                    .unwrap_or_else(|| TIMELINE_GENESIS.to_string());
                (id.clone(), log.ops.len(), head)
            })
            .collect()
    }

    /// Export this agent's ops from absolute step `since` onward as a
    /// chain-verified [`SyncBundle`]. Requires an agent id, and refuses a range
    /// reaching into compacted history — folded ops are no longer separable, so
    /// a receiver could not re-verify them (federated agents should run with
    /// compaction off: `CCOS_OPLOG_MAX=0`).
    pub fn export_bundle(&self, since: usize) -> Result<SyncBundle, SyncError> {
        if self.agent.is_empty() {
            return Err(SyncError::NoAgentId);
        }
        if since < self.folded {
            return Err(SyncError::CompactedHistory);
        }
        let start = (since - self.folded).min(self.ops.len());
        let anchor = if start == 0 {
            self.anchor.clone()
        } else {
            self.chain[start - 1].clone()
        };
        #[cfg_attr(not(feature = "signed-sync"), allow(unused_mut))]
        let mut bundle = SyncBundle {
            version: 1,
            agent: self.agent.clone(),
            first_seq: since,
            anchor,
            ops: self.ops[start..].to_vec(),
            chain: self.chain[start..].to_vec(),
            pubkey: None,
            sig: None,
        };
        // A workspace holding an identity key signs every export (signed-sync
        // builds only — a crypto-free build never loads the key, and the
        // receiving side's pinning is what protects against the resulting
        // unsigned export). Signing happens over the canonical payload with the
        // signature fields blanked, so verify recomputes the same bytes.
        #[cfg(feature = "signed-sync")]
        if let Some(seed) = &self.signing_seed {
            use ed25519_dalek::Signer;
            let sk = ed25519_dalek::SigningKey::from_bytes(seed);
            let payload = bundle.signing_payload();
            bundle.pubkey = Some(crate::license::b64url_encode(
                &sk.verifying_key().to_bytes(),
            ));
            bundle.sig = Some(crate::license::b64url_encode(
                &sk.sign(payload.as_bytes()).to_bytes(),
            ));
        }
        Ok(bundle)
    }

    /// Import another agent's [`SyncBundle`], verifying before trusting:
    ///
    /// 1. every link recomputes from the bundle's anchor — a mutated bundle is
    ///    [`SyncError::Tampered`];
    /// 2. the segment must **extend** this agent's locally-known chain — a first
    ///    import starts at genesis, a later one at or before the known end, and
    ///    every overlapping link must be identical, so one agent publishing two
    ///    histories under one identity is caught as [`SyncError::Diverged`]
    ///    (equivocation) and a skipped segment as [`SyncError::Gap`].
    ///
    /// Returns how many **new** ops were appended (0 for a pure overlap). The own
    /// timeline is never touched; [`merged_view`](Self::merged_view) materializes
    /// the combined knowledge, [`checkpoint`](Self::checkpoint) persists the import.
    pub fn import_bundle(&mut self, bundle: &SyncBundle) -> Result<usize, SyncError> {
        if bundle.agent.is_empty() {
            return Err(SyncError::Tampered("bundle has no agent id".into()));
        }
        if bundle.agent == self.agent {
            return Err(SyncError::SelfImport);
        }
        // 0. Authenticity, before anything else. The chain below proves the
        //    segment is internally consistent; only the signature proves WHO
        //    recorded it. TOFU pinning: the first signed bundle pins the key for
        //    that agent id; afterwards a different key (spoof / silent swap) and
        //    an unsigned bundle (stripped-signature downgrade) both refuse.
        //    Pinning itself happens only after the whole import succeeds — a
        //    refused bundle changes no state, keys included.
        let pin_after: Option<(String, String)> = match (&bundle.pubkey, &bundle.sig) {
            (Some(pubkey), Some(sig)) => {
                verify_bundle_signature(bundle, pubkey, sig)?;
                match self.known_keys.get(&bundle.agent) {
                    Some(pinned) if pinned != pubkey => {
                        return Err(SyncError::KeyMismatch {
                            agent: bundle.agent.clone(),
                        });
                    }
                    Some(_) => None,
                    None => Some((bundle.agent.clone(), pubkey.clone())),
                }
            }
            (None, None) => {
                if self.known_keys.contains_key(&bundle.agent) {
                    return Err(SyncError::UnsignedFromPinned {
                        agent: bundle.agent.clone(),
                    });
                }
                None
            }
            _ => {
                return Err(SyncError::BadSignature(
                    "pubkey and sig must both be present or both absent".into(),
                ));
            }
        };
        if bundle.ops.len() != bundle.chain.len() {
            return Err(SyncError::Tampered(format!(
                "{} op(s) but {} chain link(s)",
                bundle.ops.len(),
                bundle.chain.len()
            )));
        }
        if bundle.first_seq == 0 && bundle.anchor != TIMELINE_GENESIS {
            return Err(SyncError::Tampered(
                "a genesis bundle must anchor at GENESIS".into(),
            ));
        }
        // 1. Internal verification: every link recomputes from the anchor.
        let mut prev = bundle.anchor.clone();
        for (i, (op, link)) in bundle.ops.iter().zip(bundle.chain.iter()).enumerate() {
            let seq = bundle.first_seq + i;
            if op_link_hash(&prev, seq, op) != *link {
                return Err(SyncError::Tampered(format!(
                    "link t={} does not verify",
                    seq + 1
                )));
            }
            prev = link.clone();
        }
        // 2. Continuity against what is already known of this agent. Checked
        //    before mutating anything, so a refused bundle changes no state.
        let known_len = self.foreign.get(&bundle.agent).map_or(0, |l| l.ops.len());
        if bundle.first_seq > known_len {
            return Err(SyncError::Gap {
                known: known_len,
                bundle_start: bundle.first_seq,
            });
        }
        if let Some(known) = self.foreign.get(&bundle.agent) {
            // The anchor must equal the known link just before the bundle's start…
            if bundle.first_seq > 0 && bundle.anchor != known.chain[bundle.first_seq - 1] {
                return Err(SyncError::Diverged {
                    at: bundle.first_seq,
                });
            }
            // …and every overlapping link must agree.
            for (i, link) in bundle.chain.iter().enumerate() {
                let seq = bundle.first_seq + i;
                match known.chain.get(seq) {
                    Some(existing) if existing != link => {
                        return Err(SyncError::Diverged { at: seq + 1 });
                    }
                    Some(_) => {}
                    None => break,
                }
            }
        }
        // Append the genuinely-new suffix.
        let known = self
            .foreign
            .entry(bundle.agent.clone())
            .or_insert(ForeignLog {
                ops: Vec::new(),
                chain: Vec::new(),
            });
        let overlap = (known_len - bundle.first_seq).min(bundle.ops.len());
        let added = bundle.ops.len() - overlap;
        known.ops.extend_from_slice(&bundle.ops[overlap..]);
        known.chain.extend_from_slice(&bundle.chain[overlap..]);
        if let Some((agent, pubkey)) = pin_after {
            self.known_keys.insert(agent, pubkey);
        }
        Ok(added)
    }

    /// The base64url public key of this workspace's signing identity, when a
    /// `signed-sync` build loaded one from `<workspace>.ccos.key` (`None`
    /// otherwise — always on a crypto-free build).
    pub fn signing_pubkey(&self) -> Option<String> {
        #[cfg(feature = "signed-sync")]
        {
            self.signing_seed.as_ref().map(|seed| {
                crate::license::b64url_encode(
                    &ed25519_dalek::SigningKey::from_bytes(seed)
                        .verifying_key()
                        .to_bytes(),
                )
            })
        }
        #[cfg(not(feature = "signed-sync"))]
        None
    }

    /// The TOFU-pinned signing keys (agent id → base64url public key) this
    /// session trusts. Readable on every build — pins recorded by a
    /// `signed-sync` build keep protecting the workspace when it is later
    /// opened by a crypto-free build (unsigned bundles from pinned agents
    /// still refuse there).
    pub fn pinned_keys(&self) -> &std::collections::BTreeMap<String, String> {
        &self.known_keys
    }

    /// Materialize the **shared brain**: replay every known timeline — this
    /// agent's own recorded tail and every imported foreign log — from empty, in
    /// canonical (sorted-agent-id) order, with the exact same deferred-resolve
    /// semantics as [`replay_to`](Self::replay_to) (one shared op-applier). A pure
    /// function of the known timelines: any two agents holding the same set
    /// materialize **bit-identical** views — the store's convergence guarantee
    /// (tested). A seeded or compacted local baseline is deliberately not part
    /// of the exchanged history: the log is the shared truth.
    pub fn merged_view(&self) -> CcosMemory {
        let own_id = if self.agent.is_empty() {
            "local"
        } else {
            self.agent.as_str()
        };
        let mut streams: std::collections::BTreeMap<&str, &[Op]> =
            std::collections::BTreeMap::new();
        streams.insert(own_id, &self.ops);
        for (id, log) in &self.foreign {
            streams.insert(id.as_str(), &log.ops);
        }
        let mut m = CcosMemory::new();
        let mut dirty = false;
        for ops in streams.values() {
            for op in ops.iter() {
                apply_op(&mut m, op, &mut dirty);
            }
        }
        if dirty {
            m.resolve();
        }
        m
    }

    /// Record and apply an ingest.
    pub fn ingest(&mut self, uri: &str, source: &str) -> IngestReport {
        self.record(Op::Ingest {
            uri: uri.to_string(),
            source: source.to_string(),
        });
        self.live.ingest_source(uri, source)
    }

    /// Ingest many files as one **deferred batch** — the bulk sibling of [`ingest`](Self::ingest).
    /// Every file is recorded as an `Op::Ingest` (so the timeline, the hash chain, and the
    /// `replay == live` invariant are exactly as if each had been [`ingest`](Self::ingest)ed one by
    /// one) but applied with [`ingest_deferred`](CcosMemory::ingest_deferred); the three whole-graph
    /// resolve passes (imports, calls, data-flow) then run **once** at the end instead of after
    /// every file — O(N) over the batch instead of the O(N²) a per-file loop pays. Because
    /// resolution is order-independent (B2-full: prune + rebuild from the final state), the
    /// resulting graph is byte-identical to calling [`ingest`](Self::ingest) in a loop, and the live
    /// memory is left fully resolved, so any following recall / serialise reads cross-file edges as
    /// usual.
    ///
    /// Returns one [`IngestReport`] per file, in order. Each report's `edges_added` counts that
    /// file's **direct** edges only; the cross-file (import / call / data-flow) edges are added by
    /// the single final resolve, not attributed back to individual files.
    pub fn ingest_batch<U, S>(
        &mut self,
        files: impl IntoIterator<Item = (U, S)>,
    ) -> Vec<IngestReport>
    where
        U: AsRef<str>,
        S: AsRef<str>,
    {
        let mut reports = Vec::new();
        for (uri, source) in files {
            let (uri, source) = (uri.as_ref(), source.as_ref());
            self.record(Op::Ingest {
                uri: uri.to_string(),
                source: source.to_string(),
            });
            reports.push(self.live.ingest_deferred(uri, source));
        }
        // One resolve for the whole batch — the live analogue of what `replay_to` now does.
        self.live.resolve();
        reports
    }

    /// Read-only access to the session's runtime licensing state (tier, licensee).
    pub fn licensing(&self) -> &Licensing {
        &self.licensing
    }

    /// Install an already-determined [`Licensing`] state — for an embedding host that ran its own
    /// verifier, or for tests. This does not bypass verification: a
    /// [`License`](crate::license::License) is only ever produced by a verifier or the explicit
    /// `community` / `licensed` constructors.
    pub fn set_licensing(&mut self, licensing: Licensing) {
        self.licensing = licensing;
    }

    /// **Install per-source custom authority weights** — the Pro `CustomAuthorityWeights` feature.
    /// Gated: returns [`LicenseError::FeatureLocked`] in the community tier and changes nothing, so
    /// the core assertion path is never degraded — unlicensed assertions simply keep their uniform
    /// per-call authority. With Pro, subsequent [`assert_support`](Self::assert_support) /
    /// [`assert_contradiction`](Self::assert_contradiction) calls whose `evidence` is in `map` use the
    /// mapped weight; the **final** weight is what gets logged as `Op::Assert`, so `replay == live`
    /// holds with no need to persist the map itself.
    pub fn set_custom_authorities(&mut self, map: CustomAuthorityMap) -> Result<(), LicenseError> {
        self.licensing
            .require(Feature::CustomAuthorityWeights, crate::license::now_unix())?;
        self.custom_authorities = map;
        Ok(())
    }

    /// **Enable the SLHAv2 grouped-INT4 embedding scheme** — the Pro [`Feature::SlhAv2Embeddings`]
    /// feature. Gated like [`set_custom_authorities`](Self::set_custom_authorities): returns
    /// [`LicenseError::FeatureLocked`] in the community tier and changes nothing (the session keeps
    /// the uniform INT4 fallback it was opened with). With Pro, subsequent
    /// [`build_embeddings`](crate::external_memory::CcosMemory::build_embeddings) calls quantize with
    /// the adaptive grouped scheme. The scheme is runtime-only (never persisted), so `replay == live`
    /// holds with no need to record it — it is re-derived from the host tier at open, exactly like
    /// the licensing state.
    pub fn enable_slhav2_embeddings(&mut self) -> Result<(), LicenseError> {
        self.licensing
            .require(Feature::SlhAv2Embeddings, crate::license::now_unix())?;
        self.live.set_slhav2_embeddings(true);
        Ok(())
    }

    /// Record and apply a **support** assertion — `evidence` is evidence *for* `claim` (the
    /// affirmative surface of the claim's [Q-Page](crate::memory::MemoryGraph::qbelief)). Logged as
    /// an `Op::Assert` so it replays identically. Returns whether a new edge was added.
    pub fn assert_support(&mut self, evidence: &str, claim: &str, weight: f64) -> bool {
        // Pro custom-authority override; a community session's map is always empty, so the input
        // weight is used unchanged (today's behaviour).
        let weight = self.custom_authorities.get(evidence).unwrap_or(weight);
        self.record(Op::Assert {
            evidence: evidence.to_string(),
            claim: claim.to_string(),
            supports: true,
            weight,
        });
        self.live.assert_support(evidence, claim, weight)
    }

    /// Record and apply a **contradiction** assertion — `evidence` is evidence *against* `claim`
    /// (the negative surface). The dual of [`assert_support`](Self::assert_support); logged as an
    /// `Op::Assert` so `replay == live` holds for the contested-knowledge channel too.
    pub fn assert_contradiction(&mut self, evidence: &str, claim: &str, weight: f64) -> bool {
        // Pro custom-authority override (see `assert_support`); community map is empty → unchanged.
        let weight = self.custom_authorities.get(evidence).unwrap_or(weight);
        self.record(Op::Assert {
            evidence: evidence.to_string(),
            claim: claim.to_string(),
            supports: false,
            weight,
        });
        self.live.assert_contradiction(evidence, claim, weight)
    }

    /// Bring `uri` up to date against a persisted workspace **without** logging a
    /// redundant op when it is unchanged — for read-side tools (`ccos focus`) that
    /// re-scan a tree each run. Records (and applies) an `Ingest` op only when the
    /// content actually changed; returns whether it re-ingested.
    pub fn sync(&mut self, uri: &str, source: &str) -> bool {
        if self.live.file_unchanged(uri, source) {
            return false;
        }
        self.ingest(uri, source);
        true
    }

    /// Record and apply a failure signal.
    pub fn signal_failure(&mut self, node: &str, depth: u32) -> Result<usize, MemoryError> {
        self.record(Op::Failure {
            node: node.to_string(),
            depth,
        });
        self.live.signal_failure(node, depth)
    }

    /// Record and run a recall. Read-only for *selection*, with one deterministic
    /// side effect: a recall *around* a node that was demoted to the COLD tier
    /// pages it (and its cold neighbours) back first — a page fault on the read
    /// path. The op is logged and the page-in is reproduced on replay, so the
    /// timeline stays complete and `replay == live`.
    pub fn recall(&mut self, recall: Recall, budget: usize) -> RecallWindow {
        // Page fault on the read path: a recall *around* a demoted node pages it
        // (and its cold neighbours) back from the COLD tier first, so the cold
        // tier is transparent to a recalling agent.
        if let Recall::Around(uri) = &recall {
            self.live.ensure_resident(uri);
        }
        let window = self.live.recall(&recall, budget);
        self.record(Op::Recall { recall, budget });
        window
    }

    /// Record and run a **compressed** recall: same selection as [`Self::recall`],
    /// then each item's content is passed through the [`CausalCompressor`] and
    /// the original is cached in the session's CCR store. The host LLM can call
    /// [`retrieve_original`](Self::retrieve_original) (exposed as the
    /// `ccos_retrieve` MCP tool) to fetch any original back — the CCOS
    /// equivalent of headroom's reversible `headroom_retrieve`. This is the
    /// real *compression* pass CCOS historically lacked; it is a pure
    /// post-selection transform, so the causal graph, the scoring, the paging
    /// and the hash-chain replay invariants are untouched.
    pub fn recall_compressed(&mut self, recall: Recall, budget: usize) -> RecallWindow {
        if let Recall::Around(uri) = &recall {
            self.live.ensure_resident(uri);
        }
        let window = self
            .live
            .recall_compressed(&recall, budget, &mut self.compressor);
        self.record(Op::Recall { recall, budget });
        window
    }

    /// Retrieve an original content blob from the CCR store (backend for the
    /// `ccos_retrieve` MCP tool). `None` when the ref is unknown or has been
    /// evicted by the store's capacity cap.
    pub fn retrieve_original(&self, ccr: &CcrRef) -> Option<&str> {
        self.compressor.retrieve(ccr)
    }

    /// **Budget feedback loop** — compression-aware recall that re-spends the
    /// tokens freed by compression on more causal nodes (up to `max_rounds`
    /// passes). See [`CcosMemory::recall_compressed_with_feedback`]. This is
    /// the CCOS differentiator vs headroom: headroom compresses a fixed
    /// selection; CCOS grows the selection with the freed space, so the host
    /// gets strictly more causal signal at the same emitted-token cost.
    pub fn recall_compressed_with_feedback(
        &mut self,
        recall: Recall,
        budget: usize,
        max_rounds: usize,
    ) -> RecallWindow {
        if let Recall::Around(uri) = &recall {
            self.live.ensure_resident(uri);
        }
        let window = self.live.recall_compressed_with_feedback(
            &recall,
            budget,
            &mut self.compressor,
            max_rounds,
        );
        self.record(Op::Recall { recall, budget });
        window
    }

    /// Read-only access to the compression pipeline's last-run per-item stats
    /// (algorithm, tokens before/after, ratio). Useful for a `ccos stats`-style
    /// compression dashboard.
    pub fn last_compression_stats(&self) -> &[crate::compressor::CompressionStat] {
        &self.compressor.last_stats
    }

    /// **Context page fault**: feed `cargo test` / compiler output back into the
    /// session. The faulting source locations are parsed out (a direct symptom→
    /// cause signal), failure pressure is injected on those in-memory files, and a
    /// refreshed window is recalled around the fault — the compiler-in-the-loop
    /// step. It is logged like any other op, so the whole correction loop replays.
    /// Returns the refreshed context window for the agent's next attempt.
    pub fn page_fault(&mut self, compiler_output: &str, budget: usize) -> RecallWindow {
        let files = parse_cargo_test_output(compiler_output).files();
        let depth = page_fault_depth();
        for f in &files {
            let _ = self.live.signal_failure(&format!("file:{f}"), depth);
        }
        let recall = files
            .first()
            .map(|f| Recall::around(format!("file:{f}")))
            .unwrap_or(Recall::WorkingSet);
        let window = self.live.recall(&recall, budget);
        self.record(Op::PageFault { files, depth });
        window
    }

    /// Logical timeline length: ops recorded **plus** those already folded into the
    /// baseline by compaction. Stays stable across a compaction (the floor rises,
    /// the tail shrinks), so absolute `step` indices keep their meaning.
    pub fn len(&self) -> usize {
        self.folded + self.ops.len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The compaction floor: the logical step at/below which history has been folded
    /// into the baseline and is no longer separable (0 when nothing is compacted).
    pub fn floor(&self) -> usize {
        self.folded
    }

    /// Read-only access to the live memory.
    pub fn memory(&self) -> &CcosMemory {
        &self.live
    }

    /// Deterministically reconstruct the memory state after the first `step`
    /// (logical) operations, replaying the mutating ones on top of the session's
    /// baseline (empty for a [`new`](Self::new) session, the loaded checkpoint or a
    /// compaction floor for an [`open`](Self::open)ed one). `step` is clamped to
    /// [`len`](Self::len); a `step` at or below the compaction floor returns the
    /// baseline (older history has been folded in and is no longer separable).
    pub fn replay_to(&self, step: usize) -> CcosMemory {
        let mut m = match &self.baseline {
            Some(snapshot) => CcosMemory::from_json(snapshot).unwrap_or_default(),
            None => CcosMemory::new(),
        };
        // Map the logical step onto the retained tail (everything <= folded is the
        // baseline already).
        let tail = step.min(self.len()).saturating_sub(self.folded);
        // Batch the whole-graph resolution exactly as a bulk live ingest would: defer each
        // `Ingest` (mark dirty) and run the three resolve passes **once** before the next op
        // that reads cross-file edges — a recall page-in, a failure / page-fault propagation —
        // and once at the end. B2-full made resolution order-independent (prune + rebuild from
        // the final state), so this O(N)-over-the-run reconstruction yields the byte-identical
        // graph the old resolve-after-every-ingest (O(N²)) path produced — `replay == live`
        // still holds, now in linear time. Ingestion itself never demotes to COLD (only an
        // explicit resident-cap change does, and none is logged), so deferring the resolve
        // cannot reorder paging.
        let mut dirty = false;
        for op in self.ops.iter().take(tail) {
            apply_op(&mut m, op, &mut dirty);
        }
        // Resolve the trailing deferred ingests so the returned memory is fully resolved:
        // every `&self` reader of it — `recall_what_if`, `stats`, `graph()`, and `compact`'s
        // `to_json` — needs the cross-file edges, and a debug build asserts on an unresolved
        // read.
        if dirty {
            m.resolve();
        }
        m
    }

    /// **Time-travel what-if**: rewind to just before (logical) operation `step`,
    /// then run a recall with (possibly) different parameters — does the agent get
    /// a better window? `step` is clamped to the timeline length.
    pub fn recall_what_if(&self, step: usize, recall: &Recall, budget: usize) -> RecallWindow {
        self.replay_to(step).recall(recall, budget)
    }

    /// **Belief/tension timeline** — the dynamic, conflict-resolution view of this session's history:
    /// replay to each step (every `stride` ops) and record each tracked claim's belief and tension,
    /// yielding the temporal-profile tensor `Θ[claim, {Belief, Tension}, t]` over the *real* recorded
    /// timeline (the productionized form of the `temporal_tensor_crux` measurement). `half_life > 0`
    /// applies the knowledge-half-life decay. Like [`retrieval_reward`](Self::retrieval_reward) it
    /// replays once per sampled step, so it is an **offline** analysis (≈ O(N²/stride)), not a hot path.
    pub fn belief_tension_timeline(
        &self,
        claims: &[crate::memory::NodeId],
        stride: usize,
        half_life: f64,
    ) -> crate::spectral::TemporalProfile {
        let stride = stride.max(1);
        let graphs: Vec<crate::memory::MemoryGraph> = (0..=self.len())
            .step_by(stride)
            .map(|t| self.replay_to(t).graph().clone())
            .collect();
        crate::spectral::temporal_profile(graphs.iter(), claims, half_life)
    }

    // ── slice C: self-improving retrieval from the replayable log ─────────────

    /// Replay the timeline under candidate scoring `weights` and score how often
    /// recall **helped**: for each recorded recall, was the node the agent engaged
    /// *next* (a failure signal or page-fault) present — at file granularity — in
    /// the window that recall would have produced under `weights`? Returns the hit
    /// rate over all such recall→engagement pairs, or `None` when the log has no
    /// judged pair (nothing to learn from yet).
    ///
    /// This is **counterfactual** evaluation on the hash-chained log: the same log
    /// yields the same number, deterministically. The reward is an honest *proxy* —
    /// it assumes the agent's next failing/faulting node is the context recall
    /// should have surfaced. Cost is one full replay per call (each recall op is
    /// re-run), so it is an offline/maintenance operation, not a hot path.
    pub fn retrieval_reward(&self, weights: &ScoringWeights) -> Option<f64> {
        let mut m = match &self.baseline {
            Some(snapshot) => CcosMemory::from_json(snapshot).unwrap_or_default(),
            None => CcosMemory::new(),
        };
        m.set_scoring_weights(*weights);
        let mut pending: Option<RecallWindow> = None;
        let (mut hits, mut total) = (0usize, 0usize);
        // Same deferred-resolution batching as `replay_to`: defer each ingest and resolve
        // once before the next op that reads cross-file edges (a recall's selection, a
        // failure / page-fault's propagation). The reward is a pure function of those same
        // replayed states, so the number is byte-identical — only the per-ingest O(N²)
        // resolution collapses to O(N) over the log.
        let mut dirty = false;
        for op in &self.ops {
            match op {
                Op::Ingest { uri, source } => {
                    m.ingest_deferred(uri, source);
                    dirty = true;
                }
                Op::Recall { recall, budget } => {
                    if dirty {
                        m.resolve();
                        dirty = false;
                    }
                    if let Recall::Around(uri) = recall {
                        m.ensure_resident(uri);
                    }
                    pending = Some(m.recall(recall, *budget));
                }
                Op::Failure { node, depth } => {
                    if dirty {
                        m.resolve();
                        dirty = false;
                    }
                    if let Some(win) = pending.take() {
                        total += 1;
                        if window_holds(&win, node) {
                            hits += 1;
                        }
                    }
                    let _ = m.signal_failure(node, *depth);
                }
                Op::PageFault { files, depth } => {
                    if dirty {
                        m.resolve();
                        dirty = false;
                    }
                    if let Some(win) = pending.take() {
                        total += 1;
                        if files.iter().any(|f| window_holds(&win, f)) {
                            hits += 1;
                        }
                    }
                    for f in files {
                        let _ = m.signal_failure(&format!("file:{f}"), *depth);
                    }
                }
                Op::Retune { .. } => {
                    // Hold `weights` fixed across the whole evaluation; a recorded
                    // retune is what we are *measuring against*, not applying here.
                }
                Op::Assert {
                    evidence,
                    claim,
                    supports,
                    weight,
                } => {
                    // Re-apply so the replayed graph matches the timeline; belief edges do not
                    // affect the recall-hit metric, but the state must stay consistent.
                    if *supports {
                        m.assert_support(evidence, claim, *weight);
                    } else {
                        m.assert_contradiction(evidence, claim, *weight);
                    }
                }
            }
        }
        (total > 0).then_some(hits as f64 / total as f64)
    }

    /// The retrieval hit rate under the **current** scoring weights — the baseline
    /// the tuner improves on. `None` when the log has no judged recall yet.
    pub fn retrieval_hit_rate(&self) -> Option<f64> {
        self.retrieval_reward(&self.live.scoring_weights())
    }

    /// Learn scoring weights that maximise the retrieval reward over the
    /// replayable log, by **deterministic coordinate ascent** from the current
    /// weights (a fixed multiplicative grid, fixed dimension order, strict
    /// improvement only — so the same log always yields the same weights). Pure: it
    /// does not mutate the session. Returns the current weights unchanged when
    /// there is nothing to learn from.
    pub fn tune_recall_weights(&self) -> ScoringWeights {
        let mut best = self.live.scoring_weights();
        let Some(mut best_reward) = self.retrieval_reward(&best) else {
            return best;
        };
        const FACTORS: [f64; 6] = [0.25, 0.5, 0.75, 1.5, 2.0, 4.0];
        // Absolute candidates for the centrality weight: it starts at 0, which a
        // multiplicative move can never escape, so it is tried as a set of fixed
        // values (relative to the base weight scale).
        const CENTRALITY: [f64; 4] = [0.05, 0.15, 0.3, 0.5];
        for _pass in 0..2 {
            for dim in 0..4 {
                for &f in &FACTORS {
                    let mut cand = best;
                    match dim {
                        0 => cand.w_base *= f,
                        1 => cand.w_failure *= f,
                        2 => cand.w_recency *= f,
                        _ => cand.w_access *= f,
                    }
                    if let Some(r) = self.retrieval_reward(&cand) {
                        if r > best_reward {
                            best = cand;
                            best_reward = r;
                        }
                    }
                }
            }
            // Structural centrality (absolute candidates, plus 0 = off).
            for &c in CENTRALITY.iter().chain(std::iter::once(&0.0)) {
                let mut cand = best;
                cand.w_centrality = c;
                if let Some(r) = self.retrieval_reward(&cand) {
                    if r > best_reward {
                        best = cand;
                        best_reward = r;
                    }
                }
            }
        }
        best
    }

    /// Learn weights from the log ([`tune_recall_weights`](Self::tune_recall_weights))
    /// and **adopt** them when they strictly improve the measured hit rate: apply
    /// them to the live memory *and* record a retune op so the change is
    /// auditable and reproduced on replay (`replay == live` holds). Returns `true`
    /// when weights were adopted. The CCOS-native loop — the replayable history is
    /// the training data.
    pub fn adopt_tuned_recall_weights(&mut self) -> bool {
        let before = self.retrieval_hit_rate();
        let tuned = self.tune_recall_weights();
        let after = self.retrieval_reward(&tuned);
        if after > before {
            self.live.set_scoring_weights(tuned);
            self.record(Op::Retune { weights: tuned });
            true
        } else {
            false
        }
    }

    /// **Causal-of-drift attribution.** Reconstruct a node's causal-score trajectory across the
    /// replayable history — [`replay_to`](Self::replay_to) at every step — then locate the single
    /// most pronounced level shift with a deterministic CUSUM change-point
    /// ([`crate::drift::changepoint`]) and charge it to the recorded operation applied at that step.
    /// Turns the post-mortem's descriptive "score drifted" into "op *X* moved it, by Δ, at step *k*".
    ///
    /// Returns `None` when the node never drifts (a flat trajectory) or the break falls **below the
    /// compaction floor** (those ops are folded into the baseline and are no longer individually
    /// attributable — reported honestly rather than mis-charged). Read-only and deterministic (built
    /// purely from `replay_to`, which is byte-reproducible), so it adds no snapshot state and
    /// `replay == live` is untouched. Offline by nature: it does one replay per step, so keep it to
    /// the post-mortem path, not a hot loop. A capability a stateless retriever cannot have — it has
    /// no per-item trajectory and no operation log to attribute a change to.
    pub fn attribute_drift(&self, node: &str) -> Option<DriftCause> {
        let series = self.score_trajectory(node);
        let cp = crate::drift::changepoint(&series)?;
        // `series[k]` is the state after `k` ops, so the shift into `series[cp.index]` was caused by
        // the op at absolute step `cp.index` (the timeline lists it as `t=<index>`).
        let step = cp.index;
        if step <= self.folded {
            return None; // below the compaction floor: the culprit op is no longer retained
        }
        let op = self.timeline().into_iter().find(|l| {
            l.strip_prefix("t=")
                .and_then(|r| r.split_whitespace().next())
                .and_then(|x| x.parse::<usize>().ok())
                == Some(step)
        })?;
        Some(DriftCause {
            node: node.to_string(),
            step,
            delta: cp.delta,
            cusum: cp.cusum,
            op,
        })
    }

    /// A node's causal-score trajectory across the replayable history: its
    /// [`compute_node_score`](crate::memory::MemoryGraph::compute_node_score) at every step
    /// `0..=len` (`0.0` where the node is absent). A pure, deterministic function of the op-log —
    /// the basis for [`attribute_drift`](Self::attribute_drift) and
    /// [`align_node_trajectory`](Self::align_node_trajectory).
    pub fn score_trajectory(&self, node: &str) -> Vec<f64> {
        let id = crate::memory::NodeId(node.to_string());
        (0..=self.len())
            .map(|step| {
                let mem = self.replay_to(step);
                let g = mem.graph();
                g.node(&id)
                    .map(|gn| g.compute_node_score(gn))
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// **Timeline alignment** for regression hunting: DTW-align a node's score trajectory in this
    /// session against `other`'s ([`crate::dtw::align`]), so two runs of the same task that took
    /// slightly different step counts can still be compared, and the first step at which they stopped
    /// tracking (differing by more than `divergence_threshold`) is pinpointed. Read-only and
    /// deterministic (both trajectories are pure replays). A stateless retriever has no per-item
    /// trajectory to align across two runs.
    pub fn align_node_trajectory(
        &self,
        other: &AgentSession,
        node: &str,
        divergence_threshold: f64,
    ) -> crate::dtw::Alignment {
        let mine = self.score_trajectory(node);
        let theirs = other.score_trajectory(node);
        crate::dtw::align(&mine, &theirs, divergence_threshold)
    }

    /// **Fork the timeline** at `step`: a new, independent session whose history is this one's
    /// baseline plus `ops[..step]`, and whose live memory is exactly [`replay_to(step)`](Self::replay_to).
    /// Append *counterfactual* operations to the branch through the ordinary public API
    /// (`ingest` / `signal_failure` / `assert_support` / …) — they are recorded as real ops on the
    /// branch's own op-log, so `replay == live` holds for the alternate history **by construction**
    /// (no op parser, no new replay machinery). Compare the two worlds with
    /// [`align_node_trajectory`](Self::align_node_trajectory) — together they are the
    /// *branch-and-align* primitive: "what if the failure had landed earlier / the assertion had
    /// never been made?", answered replayably.
    ///
    /// The trunk is untouched (history stays append-only and tamper-evident); the branch is
    /// **in-memory** (`oplog_path = None`), so a checkpoint of the branch can never clobber the
    /// trunk's sidecar — persist it explicitly to a new path if wanted. `step` is clamped to
    /// `[compaction floor, len]`: ops folded into the baseline cannot be forked *within* (their
    /// individual effects are no longer retained — the same honesty as `replay_to`). A stateless
    /// retriever has one corpus and no history: it cannot fork, inject, and replay a divergent past.
    pub fn fork_at(&self, step: usize) -> AgentSession {
        let step = step.min(self.len()).max(self.folded);
        let tail = step - self.folded;
        AgentSession {
            live: self.replay_to(step),
            ops: self.ops[..tail].to_vec(),
            baseline: self.baseline.clone(),
            folded: self.folded,
            chain: self.chain[..tail].to_vec(),
            anchor: self.anchor.clone(),
            baseline_hash: self.baseline_hash.clone(),
            agent: self.agent.clone(),
            foreign: self.foreign.clone(),
            known_keys: self.known_keys.clone(),
            signing_seed: self.signing_seed,
            oplog_path: None,
            compressor: CausalCompressor::new(),
            licensing: self.licensing.clone(),
            custom_authorities: self.custom_authorities.clone(),
        }
    }

    /// A human-readable journal of the cognitive timeline. If compaction has folded
    /// older ops into the baseline, a leading marker stands in for them (their
    /// details are no longer retained); the live tail follows at its absolute step.
    pub fn timeline(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.folded > 0 {
            out.push(format!(
                "t≤{}  «{} earlier operation(s) compacted into the baseline»",
                self.folded, self.folded
            ));
        }
        out.extend(self.ops.iter().enumerate().map(|(i, op)| {
            let t = self.folded + i + 1;
            match op {
                Op::Ingest { uri, .. } => format!("t={t}  Ingest({uri})"),
                Op::Failure { node, depth } => {
                    format!("t={t}  SignalFailure({node}, depth={depth})")
                }
                Op::Recall { recall, budget } => {
                    format!("t={t}  Recall({recall:?}, budget={budget})")
                }
                Op::PageFault { files, depth } => {
                    format!("t={t}  PageFault(files={files:?}, depth={depth})")
                }
                Op::Retune { .. } => format!("t={t}  Retune(scoring weights from log)"),
                Op::Assert {
                    evidence,
                    claim,
                    supports,
                    weight,
                } => format!(
                    "t={t}  Assert({evidence} {} {claim}, w={weight})",
                    if *supports { "supports" } else { "contradicts" }
                ),
            }
        }));
        out
    }
}

/// Apply one recorded op to a memory with replay's deferred-resolve discipline
/// (`dirty` tracks pending deferred ingests; ops that read cross-file edges resolve
/// first). Shared by [`AgentSession::replay_to`] and [`AgentSession::merged_view`],
/// so both reconstruct with byte-identical semantics.
fn apply_op(m: &mut CcosMemory, op: &Op, dirty: &mut bool) {
    match op {
        Op::Ingest { uri, source } => {
            m.ingest_deferred(uri, source);
            *dirty = true;
        }
        Op::Failure { node, depth } => {
            if *dirty {
                m.resolve();
                *dirty = false;
            }
            let _ = m.signal_failure(node, *depth);
        }
        Op::PageFault { files, depth } => {
            if *dirty {
                m.resolve();
                *dirty = false;
            }
            for f in files {
                let _ = m.signal_failure(&format!("file:{f}"), *depth);
            }
        }
        Op::Recall { recall, .. } => {
            // A recall is read-only for *selection*, but a recall *around*
            // a demoted node pages it (and its cold neighbours) back from
            // the COLD tier — a side effect on the resident/cold partition
            // that reads edges. Resolve first so the page-in sees the same
            // graph live did (deterministic page-in, replay == live).
            if *dirty {
                m.resolve();
                *dirty = false;
            }
            if let Recall::Around(uri) = recall {
                m.ensure_resident(uri);
            }
        }
        Op::Retune { weights } => {
            // Reproduce the learned-weights change so replay == live. No edge
            // read, so it does not force a resolve of the deferred ingests.
            m.set_scoring_weights(*weights);
        }
        Op::Assert {
            evidence,
            claim,
            supports,
            weight,
        } => {
            // Re-apply the dual-evidence assertion so the belief surfaces (and the
            // QBelief derived from them) reconstruct identically — replay == live.
            // Belief edges are not resolution-owned (a later `resolve` never prunes
            // them) and the assert reads no cross-file edge, so it forces no resolve.
            if *supports {
                m.assert_support(evidence, claim, *weight);
            } else {
                m.assert_contradiction(evidence, claim, *weight);
            }
        }
    }
}

/// The signing-key sidecar path for a workspace: `<path>.key` (64 hex chars —
/// the raw 32-byte ed25519 seed). Lives NEXT to the state, never inside the
/// timeline sidecar, so sharing an `.oplog` or a bundle can never leak it.
#[cfg_attr(not(feature = "signed-sync"), allow(dead_code))]
fn key_sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".key");
    PathBuf::from(s)
}

/// Load the workspace signing seed on a `signed-sync` build: `<path>.key`
/// holding 64 hex chars. Unreadable / malformed / absent → `None` (an export
/// is then unsigned; the importer's pinning is the guard that notices).
#[cfg(feature = "signed-sync")]
fn load_signing_seed(workspace_file: &Path) -> Option<[u8; 32]> {
    let raw = std::fs::read_to_string(key_sidecar(workspace_file)).ok()?;
    crate::util::from_hex32(raw.trim())
}

/// A crypto-free build never reads the key file (it could not use the seed).
#[cfg(not(feature = "signed-sync"))]
fn load_signing_seed(_workspace_file: &Path) -> Option<[u8; 32]> {
    None
}

/// Generate this workspace's ed25519 signing identity: writes a fresh 32-byte
/// seed (OS entropy) as 64 hex chars to `<workspace>.ccos.key` and returns the
/// base64url **public** key. Refuses to overwrite an existing key — key rotation
/// is an explicit human act (delete the file first), because every peer that
/// pinned the old key will refuse the new one until they re-pin.
#[cfg(feature = "signed-sync")]
pub fn generate_workspace_key(path: impl AsRef<Path>) -> Result<String, MemoryError> {
    let file = crate::external_memory::workspace_file(path.as_ref());
    let key_path = key_sidecar(&file);
    if key_path.exists() {
        return Err(MemoryError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "{} already exists — key rotation must be explicit (delete it first; peers that pinned the old key will refuse the new one)",
                key_path.display()
            ),
        )));
    }
    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed)
        .map_err(|e| MemoryError::Io(std::io::Error::other(format!("entropy: {e}"))))?;
    crate::util::write_durable(&key_path, crate::util::hex32(&seed).as_bytes())?;
    let pk = ed25519_dalek::SigningKey::from_bytes(&seed)
        .verifying_key()
        .to_bytes();
    Ok(crate::license::b64url_encode(&pk))
}

/// Verify a bundle's ed25519 signature over its canonical payload.
#[cfg(feature = "signed-sync")]
fn verify_bundle_signature(bundle: &SyncBundle, pubkey: &str, sig: &str) -> Result<(), SyncError> {
    let pk_bytes: [u8; 32] = crate::license::b64url_decode(pubkey)
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| SyncError::BadSignature("malformed public key".into()))?;
    let sig_bytes: [u8; 64] = crate::license::b64url_decode(sig)
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| SyncError::BadSignature("malformed signature".into()))?;
    let vk = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes)
        .map_err(|e| SyncError::BadSignature(format!("invalid public key: {e}")))?;
    let payload = bundle.signing_payload();
    vk.verify_strict(
        payload.as_bytes(),
        &ed25519_dalek::Signature::from_bytes(&sig_bytes),
    )
    .map_err(|_| SyncError::BadSignature("signature does not match payload".into()))
}

/// Without signature support, a signed bundle is REFUSED (never silently
/// accepted unverified) — the crypto-free default build keeps working with
/// unsigned federations and tells the operator how to verify signed ones.
#[cfg(not(feature = "signed-sync"))]
fn verify_bundle_signature(
    _bundle: &SyncBundle,
    _pubkey: &str,
    _sig: &str,
) -> Result<(), SyncError> {
    Err(SyncError::SignedUnsupported)
}

/// The timeline sidecar path for a workspace: `<path>.oplog`.
fn oplog_sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".oplog");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_session() -> AgentSession {
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
        s.ingest(
            "src/repo.rs",
            "use crate::db;\npub fn fetch() -> i64 { db::timeout() }\n",
        );
        s.ingest(
            "src/api.rs",
            "use crate::repo;\npub fn handle() -> i64 { repo::fetch() }\n",
        );
        s.signal_failure("file:src/api.rs", 3).unwrap();
        s
    }

    #[test]
    fn attribute_drift_charges_a_score_move_to_a_real_op_deterministically() {
        let s = chain_session();
        // api.rs's causal score moves over the history (it is created, then the failure lands on it).
        let cause = s
            .attribute_drift("file:src/api.rs")
            .expect("api.rs has a non-flat score trajectory");
        assert!(
            (1..=s.len()).contains(&cause.step),
            "attributed to a real timeline step: {}",
            cause.step
        );
        // The change-point → op wiring is exact: the culprit line is the timeline op at that step.
        assert!(
            cause.op.starts_with(&format!("t={}", cause.step)),
            "op line matches the attributed step: {}",
            cause.op
        );
        assert!(cause.delta.abs() > 0.0, "a real level shift was found");
        // Read-only and reconstructed from replay ⇒ byte-identical every call (replay == live).
        assert_eq!(
            s.attribute_drift("file:src/api.rs"),
            Some(cause),
            "attribution is deterministic"
        );
        // A node that never existed has no trajectory to attribute.
        assert!(s.attribute_drift("file:src/nope.rs").is_none());
    }

    #[test]
    fn attribute_drift_pins_a_late_failure_on_the_failure_op() {
        // Keep a node flat for a long plateau (read-only recalls do not move its score), then land a
        // direct failure. With the plateau dominating the mean, the CUSUM break falls on the failure.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn a() -> i64 { 0 }\n"); // t1: a appears
        for _ in 0..12 {
            s.recall(Recall::working_set(), 512); // t2..t13: recorded, read-only ⇒ a stays flat
        }
        let fail_step = s.len() + 1;
        s.signal_failure("file:src/a.rs", 0).unwrap(); // final op: a's score jumps
        let cause = s
            .attribute_drift("file:src/a.rs")
            .expect("a.rs drifts on the failure");
        assert_eq!(
            cause.step, fail_step,
            "the score shift is charged to the failure op, not the ingest (op = {})",
            cause.op
        );
        assert!(
            cause.op.contains("SignalFailure"),
            "culprit is the failure: {}",
            cause.op
        );
        assert!(cause.delta > 0.0, "failure pressure raises the score");
    }

    #[test]
    fn align_node_trajectory_finds_where_two_runs_diverge() {
        // Two runs ingest the same file and recall it; run B additionally fails it, so B's score
        // trajectory splits from A's — DTW aligns the differing lengths and finds the divergence.
        let base = || {
            let mut s = AgentSession::new();
            s.ingest("src/a.rs", "pub fn a() -> i64 { 0 }\n");
            for _ in 0..5 {
                s.recall(Recall::working_set(), 512);
            }
            s
        };
        let a = base();
        let mut b = base();
        b.signal_failure("file:src/a.rs", 0).unwrap(); // B diverges here (extra op + score jump)
        let al = a.align_node_trajectory(&b, "file:src/a.rs", 0.1);
        assert!(al.distance > 0.0, "the runs differ after B's failure");
        assert!(al.divergence.is_some(), "a divergence onset is found");
        // A run aligned with itself never diverges (and is deterministic).
        let self_al = a.align_node_trajectory(&a, "file:src/a.rs", 0.1);
        assert_eq!(self_al.distance, 0.0);
        assert!(self_al.divergence.is_none());
    }

    #[test]
    fn fork_reconstructs_the_past_and_leaves_the_trunk_untouched() {
        let trunk = chain_session(); // 3 ingests + a failure (len 4)
        let trunk_len = trunk.len();
        // Fork before the failure: the branch's history is exactly the first 3 ops.
        let branch = trunk.fork_at(3);
        assert_eq!(branch.len(), 3);
        // Its live state matches the trunk's replay at that step (same working-set trajectory).
        assert_eq!(
            branch.score_trajectory("file:src/api.rs"),
            trunk.score_trajectory("file:src/api.rs")[..=3].to_vec(),
            "the branch's history is the trunk's prefix"
        );
        // The trunk is untouched by forking.
        assert_eq!(trunk.len(), trunk_len);
        // Clamping: beyond len collapses to len; a fresh session's floor is 0.
        assert_eq!(trunk.fork_at(999).len(), trunk_len);
    }

    #[test]
    fn branch_and_align_a_counterfactual_history() {
        // The full primitive: fork the real history before its failure, inject a DIFFERENT
        // counterfactual op, and DTW-align the two worlds to find where they diverge.
        let trunk = chain_session(); // ...ends with signal_failure(api.rs)
        let mut branch = trunk.fork_at(3); // just before the failure
        branch.signal_failure("file:src/db.rs", 3).unwrap(); // counterfactual: the failure lands on db instead
                                                             // The counterfactual op is a REAL op on the branch's log ⇒ replay == live for the branch.
        assert_eq!(branch.len(), 4);
        assert_eq!(
            branch.score_trajectory("file:src/db.rs"),
            branch.score_trajectory("file:src/db.rs"),
            "branch trajectories are deterministic"
        );
        // The two worlds diverge on db.rs (pressured only in the branch).
        let al = trunk.align_node_trajectory(&branch, "file:src/db.rs", 0.05);
        assert!(
            al.divergence.is_some(),
            "the counterfactual failure moves db.rs differently: dist={}",
            al.distance
        );
        // And the trunk's own history still ends with the REAL failure (append-only, untouched).
        assert!(trunk
            .timeline()
            .last()
            .unwrap()
            .contains("SignalFailure(file:src/api.rs"));
    }

    // ── slice C: self-improving retrieval from the replayable log ─────────────

    /// A session whose log has a recall that *misses* the node engaged next under
    /// the default weights, but is fixable by re-weighting: the engaged file `d`
    /// has a high access count (re-ingested), while a failing competitor `f` wins
    /// the tight default window.
    fn learnable_session() -> AgentSession {
        let mut s = AgentSession::new();
        for _ in 0..40 {
            s.ingest("src/d.rs", "pub fn delta() -> u32 { 0 }\n"); // raise d's access
        }
        s.ingest("src/f.rs", "pub fn frank() -> u32 { 1 }\n");
        s.signal_failure("file:src/f.rs", 0).unwrap(); // f is the hot default-window winner
        s.recall(Recall::working_set(), 8); // tight budget → ~1 node
        s.signal_failure("file:src/d.rs", 0).unwrap(); // the agent actually engages d
        s
    }

    #[test]
    fn retrieval_reward_scores_recall_hits_and_misses() {
        // HIT: recall around A, then fail on A → A is in its own region window.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn alpha() -> u32 { 1 }\n");
        s.ingest("src/z.rs", "pub fn zeta() -> u32 { 2 }\n");
        s.recall(Recall::around("file:src/a.rs"), 4096);
        s.signal_failure("file:src/a.rs", 1).unwrap();
        assert_eq!(
            s.retrieval_hit_rate(),
            Some(1.0),
            "recall around A then fail on A is a hit"
        );

        // MISS: recall around A, then fail on the disconnected Z.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn alpha() -> u32 { 1 }\n");
        s.ingest("src/z.rs", "pub fn zeta() -> u32 { 2 }\n");
        s.recall(Recall::around("file:src/a.rs"), 4096);
        s.signal_failure("file:src/z.rs", 1).unwrap();
        assert_eq!(
            s.retrieval_hit_rate(),
            Some(0.0),
            "recall around A then fail on disconnected Z is a miss"
        );

        // No judged pair yet → nothing to learn from.
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn alpha() -> u32 { 1 }\n");
        assert_eq!(s.retrieval_hit_rate(), None);
    }

    #[test]
    fn tune_recall_weights_is_deterministic() {
        let a = learnable_session().tune_recall_weights();
        let b = learnable_session().tune_recall_weights();
        assert_eq!(a.w_base, b.w_base);
        assert_eq!(a.w_failure, b.w_failure);
        assert_eq!(a.w_recency, b.w_recency);
        assert_eq!(a.w_access, b.w_access);
    }

    #[test]
    fn tuning_improves_retrieval_on_a_learnable_log() {
        let s = learnable_session();
        let before = s.retrieval_hit_rate();
        let tuned = s.tune_recall_weights();
        let after = s.retrieval_reward(&tuned);
        assert!(
            after > before,
            "log-tuned weights improve the measured hit rate: {before:?} -> {after:?}"
        );
    }

    #[test]
    fn adopting_tuned_weights_is_logged_and_replay_matches_live() {
        let mut s = learnable_session();
        let adopted = s.adopt_tuned_recall_weights();
        assert!(
            adopted,
            "the learnable log should yield an improving retune"
        );
        // The retune was recorded as an op, so replaying the whole timeline
        // reproduces the live scoring weights — replay == live still holds.
        let replayed = s.replay_to(s.len());
        let live = s.live.scoring_weights();
        let rep = replayed.scoring_weights();
        assert_eq!(live.w_base, rep.w_base);
        assert_eq!(live.w_failure, rep.w_failure);
        assert_eq!(live.w_recency, rep.w_recency);
        assert_eq!(live.w_access, rep.w_access);
    }

    #[test]
    fn replay_is_deterministic() {
        let s = chain_session();
        let replayed = s.replay_to(s.len());
        // Same structure as the live memory.
        assert_eq!(replayed.stats().nodes, s.memory().stats().nodes);
        assert_eq!(replayed.stats().edges, s.memory().stats().edges);
        // And replaying twice gives identical node counts (determinism).
        assert_eq!(replayed.stats().nodes, s.replay_to(s.len()).stats().nodes);
    }

    /// A structural fingerprint of a memory's resolved graph — sorted node ids + sorted
    /// edges `(source, target, weight-bits)` — independent of map iteration order. Two
    /// memories with the same fingerprint have the same resolved causal structure.
    fn graph_fingerprint(mem: &CcosMemory) -> (Vec<String>, Vec<(String, String, u64)>) {
        let g = mem.graph();
        let mut nodes: Vec<String> = g.node_ids().map(|id| id.0.clone()).collect();
        nodes.sort();
        let mut edges: Vec<(String, String, u64)> = g
            .edges()
            .iter()
            .map(|e| (e.source.0.clone(), e.target.0.clone(), e.weight.to_bits()))
            .collect();
        edges.sort();
        (nodes, edges)
    }

    #[test]
    fn ingest_batch_equals_ingest_loop_and_replays() {
        // The batched deferred path must produce the byte-identical graph an eager per-file
        // loop does (B2-full order-independence, lifted to the session API), and the batch's
        // recorded timeline must still replay == live — through the new deferred `replay_to`.
        let files = [
            ("src/a.rs", "pub fn target() -> i32 { 1 }\n"),
            (
                "src/caller.rs",
                "use crate::a::target;\npub fn run() -> i32 { target() + 1 }\n",
            ),
            (
                "src/b.rs",
                "pub const K: i32 = 7;\npub fn k() -> i32 { K + 1 }\n",
            ),
        ];

        // Eager: one `ingest` per file (resolve after each).
        let mut eager = AgentSession::new();
        for (p, s) in &files {
            eager.ingest(p, s);
        }

        // Batch: every file deferred, resolved exactly once at the end.
        let mut batch = AgentSession::new();
        let reports = batch.ingest_batch(files.iter().map(|(p, s)| (*p, *s)));
        assert_eq!(reports.len(), files.len(), "one report per ingested file");

        // 1) Batch ingest == eager loop, byte-for-byte over nodes + resolved edges.
        assert_eq!(
            graph_fingerprint(eager.memory()),
            graph_fingerprint(batch.memory()),
            "batch ingest yields the same resolved graph as an eager per-file loop"
        );
        // The cross-file call edge IS present (proves the single final resolve ran).
        assert!(
            batch.memory().graph().edges().len() > files.len(),
            "the final resolve added cross-file edges, not just direct ones"
        );

        // 2) The batch session's timeline replays to the same graph (deferred replay path).
        assert_eq!(
            graph_fingerprint(batch.memory()),
            graph_fingerprint(&batch.replay_to(batch.len())),
            "batch timeline replays == live"
        );
    }

    #[test]
    fn assertions_replay_identically_belief_and_edges() {
        // Explicit dual-evidence assertions on a claim must replay byte-for-byte: the same edges
        // AND the same *derived* QBelief — replay == live for the contested-knowledge surface, not
        // just for ingested structure.
        let mut s = AgentSession::new();
        s.ingest("src/claim.rs", "pub fn claim() {}");
        s.assert_support("ev:a", "file:src/claim.rs", 1.0);
        s.assert_support("ev:b", "file:src/claim.rs", 1.0);
        s.assert_contradiction("ev:x", "file:src/claim.rs", 1.0);

        let claim = crate::memory::NodeId("file:src/claim.rs".to_string());
        let live_q = s.memory().graph().qbelief(&claim);
        let replayed = s.replay_to(s.len());

        assert_eq!(
            replayed.graph().edges().len(),
            s.memory().graph().edges().len(),
            "replay reconstructs the same edge set"
        );
        assert_eq!(
            replayed.graph().qbelief(&claim),
            live_q,
            "derived belief reconstructs identically on replay (replay == live)"
        );
        assert_eq!(
            (live_q.support, live_q.contradiction),
            (2.0, 1.0),
            "two support + one contradiction asserted"
        );
    }

    #[test]
    fn custom_authorities_gate_in_community_and_apply_under_pro_with_replay() {
        use crate::license::License;
        let support_of = |s: &AgentSession, claim: &str| {
            s.memory()
                .graph()
                .qbelief(&crate::memory::NodeId(claim.to_string()))
                .support
        };
        let mut map = CustomAuthorityMap::new();
        map.set("ev:trusted", 0.9);

        // Community: the Pro setter is refused and changes nothing — the assertion keeps its input
        // weight (0.2), so the core belief path is never degraded.
        let mut community = AgentSession::new();
        assert!(
            community.set_custom_authorities(map.clone()).is_err(),
            "community tier refuses the Pro custom-authority setter"
        );
        community.assert_support("ev:trusted", "file:c.rs", 0.2);
        assert!(
            (support_of(&community, "file:c.rs") - 0.2).abs() < 1e-9,
            "community used the input weight 0.2, not the custom 0.9"
        );

        // Pro: install the map → the override applies, and replay reproduces it (replay == live),
        // because the FINAL weight is what gets logged as Op::Assert.
        let mut pro = AgentSession::new();
        pro.set_licensing(Licensing::licensed(License {
            licensee: "acme".into(),
            expires_at: None,
            machine: None,
        }));
        pro.set_custom_authorities(map)
            .expect("Pro tier installs custom authorities");
        pro.assert_support("ev:trusted", "file:p.rs", 0.2); // input 0.2 → overridden to 0.9
        let live = pro
            .memory()
            .graph()
            .qbelief(&crate::memory::NodeId("file:p.rs".to_string()));
        assert!(
            (live.support - 0.9).abs() < 1e-9,
            "Pro applied the custom authority 0.9, not the input 0.2"
        );
        let replay = pro
            .replay_to(pro.len())
            .graph()
            .qbelief(&crate::memory::NodeId("file:p.rs".to_string()));
        assert_eq!(
            live, replay,
            "replay reproduces the custom-weighted belief (replay == live)"
        );
    }

    #[test]
    fn slhav2_embeddings_are_gated_by_tier() {
        use crate::license::{License, Tier};

        // An in-memory session is community, so it opens with the uniform INT4 embedding
        // fallback (the grouped SLHAv2 scheme is a Pro feature).
        let community = AgentSession::new();
        assert_eq!(
            community.licensing().tier(crate::license::now_unix()),
            Tier::Community
        );
        assert!(
            !community.memory().uses_slhav2_embeddings(),
            "community session uses uniform INT4, not grouped SLHAv2"
        );

        // The explicit Pro setter is refused in community and changes nothing.
        let mut c2 = AgentSession::new();
        assert!(
            c2.enable_slhav2_embeddings().is_err(),
            "community tier refuses the Pro slhav2 setter"
        );
        assert!(!c2.memory().uses_slhav2_embeddings());

        // Under Pro, the setter enables the grouped SLHAv2 scheme.
        let mut pro = AgentSession::new();
        pro.set_licensing(Licensing::licensed(License {
            licensee: "acme".into(),
            expires_at: None,
            machine: None,
        }));
        pro.enable_slhav2_embeddings()
            .expect("Pro tier enables slhav2 embeddings");
        assert!(
            pro.memory().uses_slhav2_embeddings(),
            "Pro session uses grouped SLHAv2"
        );
    }

    #[test]
    fn belief_tension_timeline_tracks_a_claim_over_the_log() {
        // A claim believed at step 1, then contested at step 2 — the timeline must show tension rise.
        let mut s = AgentSession::new();
        s.assert_support("ev_s", "claim:x", 1.0);
        s.assert_contradiction("ev_c", "claim:x", 1.0);
        let claim = crate::memory::NodeId("claim:x".to_string());

        let prof = s.belief_tension_timeline(std::slice::from_ref(&claim), 1, 0.0);
        let tension = prof.tension_series(&claim);
        assert!(tension.len() >= 3, "sampled replay_to(0..=len)");
        assert_eq!(tension[0], 0.0, "t0 (empty replay) ⇒ no tension");
        assert!(
            *tension.last().unwrap() > 0.0,
            "the contradiction makes the claim contested by the end"
        );
        // belief goes from positive (only support) toward 0 (balanced) as tension rises.
        let belief = prof.belief_series(&claim);
        assert!(belief[1] > *belief.last().unwrap());
    }

    #[test]
    fn replay_to_earlier_step_has_fewer_files() {
        let s = chain_session();
        // After 1 op only db.rs is ingested; after 3, all three files.
        assert_eq!(s.replay_to(1).stats().files, 1);
        assert!(s.replay_to(3).stats().files >= 3);
    }

    #[test]
    fn what_if_larger_budget_widens_the_window() {
        let mut s = chain_session();
        let tight = s.recall(Recall::around("file:src/api.rs"), 20);
        let step = s.len() - 1; // the recall we just made
        let roomy = s.recall_what_if(step, &Recall::around("file:src/api.rs"), 8000);
        assert!(
            roomy.tokens >= tight.tokens && roomy.items.len() >= tight.items.len(),
            "a larger budget at the same step yields a window at least as large"
        );
    }

    #[test]
    fn timeline_records_every_operation() {
        let mut s = chain_session();
        s.recall(Recall::working_set(), 1000);
        let tl = s.timeline();
        assert_eq!(tl.len(), 5); // 3 ingest + 1 failure + 1 recall
        assert!(tl[0].contains("Ingest(src/db.rs)"));
        assert!(tl[3].contains("SignalFailure"));
        assert!(tl[4].contains("Recall"));
    }

    #[test]
    fn open_persists_and_reloads_across_sessions() {
        let path = tmp("ccos-sess-reload");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/db.rs", "pub fn q() -> i64 { 1 }\n");
            s.ingest(
                "src/api.rs",
                "use crate::db;\npub fn h() -> i64 { db::q() }\n",
            );
            s.checkpoint().unwrap();
        }
        // A brand-new session reloads the causal memory straight from disk.
        let s2 = AgentSession::open(&path).unwrap();
        assert!(s2.memory().stats().files >= 2, "files survived the reload");
        assert!(
            s2.memory().verify().valid,
            "hash chain still verifies after reload"
        );
        cleanup(&path);
    }

    /// A session built entirely through `AgentSession` keeps its **whole** timeline
    /// across a restart: the op-log makes `replay_to` span the full history.
    #[test]
    fn timeline_spans_full_history_across_restart() {
        let path = tmp("ccos-sess-history");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/db.rs", "pub fn q() {}\n");
            s.ingest("src/api.rs", "use crate::db;\npub fn h() { db::q() }\n");
            s.checkpoint().unwrap();
        }
        let s2 = AgentSession::open(&path).unwrap();
        // The recorded timeline came back, not just an empty post-reload tail.
        assert_eq!(s2.len(), 2, "both ingests were restored");
        assert!(s2.timeline()[0].contains("Ingest(src/db.rs)"));
        // Full cross-restart replay: floor is empty, top reproduces the memory.
        assert_eq!(s2.replay_to(0).stats().files, 0, "replay floor is empty");
        assert_eq!(s2.replay_to(s2.len()).stats().files, 2);
        assert_eq!(
            s2.replay_to(s2.len()).stats().nodes,
            s2.memory().stats().nodes,
            "replayed history matches the loaded memory"
        );
        cleanup(&path);
    }

    /// Seeding from a `ccos memory` snapshot (a checkpoint with no op-log): the
    /// timeline starts empty and layers on the seed as its replay baseline, then
    /// becomes durable once checkpointed.
    #[test]
    fn timeline_layers_on_a_ccos_memory_seed() {
        let path = tmp("ccos-sess-seed");
        cleanup(&path);
        // Simulate `ccos memory`: write a snapshot only (no sidecar).
        {
            let mut mem = CcosMemory::open(&path).unwrap();
            mem.ingest_source("src/seed.rs", "pub fn s() {}\n");
            mem.checkpoint().unwrap();
        }
        let mut s = AgentSession::open(&path).unwrap();
        let seed_files = s.memory().stats().files;
        assert!(seed_files >= 1);
        assert!(
            s.is_empty(),
            "no recorded timeline yet (seeded from snapshot)"
        );
        s.ingest("src/new.rs", "pub fn n() {}\n");
        // Replay floor is the seed, not empty.
        assert_eq!(s.replay_to(0).stats().files, seed_files);
        assert_eq!(s.replay_to(s.len()).stats().files, seed_files + 1);
        s.checkpoint().unwrap();
        // The seeded baseline + op survive a reopen.
        let s2 = AgentSession::open(&path).unwrap();
        assert_eq!(s2.len(), 1);
        assert_eq!(s2.replay_to(0).stats().files, seed_files);
        assert_eq!(s2.replay_to(s2.len()).stats().files, seed_files + 1);
        cleanup(&path);
    }

    /// If the snapshot is mutated out-of-band (e.g. by `ccos memory`) so the op-log
    /// no longer reproduces it, the snapshot wins and the timeline resets — the
    /// memory is never corrupted by a stale log.
    #[test]
    fn stale_oplog_self_heals_to_the_snapshot() {
        let path = tmp("ccos-sess-heal");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            s.ingest("src/a.rs", "pub fn a() {}\n");
            s.checkpoint().unwrap();
        }
        // Out-of-band mutation of the snapshot only (the sidecar is left stale).
        {
            let mut mem = CcosMemory::open(&path).unwrap();
            mem.ingest_source("src/b.rs", "pub fn b() {}\n");
            mem.checkpoint().unwrap();
        }
        let s = AgentSession::open(&path).unwrap();
        assert_eq!(
            s.memory().stats().files,
            2,
            "authoritative snapshot (a + b) wins"
        );
        assert!(s.is_empty(), "stale timeline was reset");
        assert!(s.memory().verify().valid);
        cleanup(&path);
    }

    #[test]
    fn new_session_has_no_checkpoint_path() {
        let mut s = AgentSession::new();
        assert!(matches!(s.checkpoint(), Err(MemoryError::NoPath)));
    }

    /// Field regression: a launcher created `workspace.ccos` as a *directory*, so
    /// opening it used to fail with "Is a directory". A directory workspace must
    /// place its state file inside it and round-trip.
    #[test]
    fn open_accepts_a_directory_workspace() {
        let dir = tmp("ccos-ws-dir");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        {
            let mut s = AgentSession::open(&dir).unwrap(); // a DIRECTORY, not a file
            s.ingest("src/db.rs", "pub fn q() {}\n");
            s.ingest("src/api.rs", "use crate::db;\npub fn h() { db::q() }\n");
            s.checkpoint().unwrap();
        }
        assert!(
            dir.join("workspace.ccos").is_file(),
            "state file lives inside the dir"
        );
        assert!(
            dir.join("workspace.ccos.oplog").is_file(),
            "oplog lives inside the dir"
        );
        let s2 = AgentSession::open(&dir).unwrap();
        assert!(
            s2.memory().stats().files >= 2,
            "reloads from the directory workspace"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Compaction bounds the op-log by folding older ops into the baseline, while
    /// keeping logical indices, recent rewind depth, and the live memory intact.
    #[test]
    fn compaction_bounds_the_log_and_preserves_recent_replay() {
        let mut s = AgentSession::new();
        for i in 0..10 {
            s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
        }
        let (full_files, len_before, at7) = (
            s.memory().stats().files,
            s.len(),
            s.replay_to(7).stats().files,
        );

        s.compact(4); // fold the oldest 6, keep the last 4

        assert_eq!(s.len(), len_before, "logical length is stable");
        assert_eq!(s.ops.len(), 4, "op-log bounded to the retained tail");
        assert_eq!(s.folded, 6);
        assert_eq!(
            s.memory().stats().files,
            full_files,
            "live memory untouched"
        );
        // Replays in the retained tail are exact; the full replay matches live.
        assert_eq!(s.replay_to(7).stats().files, at7);
        assert_eq!(s.replay_to(s.len()).stats().nodes, s.memory().stats().nodes);
        // A step below the floor collapses to the floor (the folded baseline).
        assert_eq!(s.replay_to(2).stats().files, s.replay_to(6).stats().files);
        assert!(
            s.timeline()[0].contains("compacted"),
            "journal notes the fold"
        );
    }

    /// A compacted timeline (folded count + tail) survives a restart.
    #[test]
    fn compaction_survives_restart() {
        let path = tmp("ccos-sess-compact");
        cleanup(&path);
        {
            let mut s = AgentSession::open(&path).unwrap();
            for i in 0..8 {
                s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
            }
            s.compact(3); // folded = 5, tail = 3
            assert_eq!(s.len(), 8);
            s.checkpoint().unwrap();
        }
        let s2 = AgentSession::open(&path).unwrap();
        assert_eq!(s2.len(), 8, "logical length survives compaction + restart");
        assert_eq!(s2.memory().stats().files, 8);
        assert!(s2.memory().verify().valid);
        assert_eq!(
            s2.replay_to(s2.len()).stats().nodes,
            s2.memory().stats().nodes,
            "restored timeline still reproduces the memory"
        );
        assert!(s2.timeline()[0].contains("compacted"));
        cleanup(&path);
    }

    /// A unique temp path for a persistence test.
    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{tag}-{}.json", std::process::id()))
    }

    /// Remove a workspace and its `.oplog` sidecar.
    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(super::oplog_sidecar(path));
    }

    #[test]
    fn sync_reingests_only_changed_files() {
        let mut s = AgentSession::new();
        // First sight of a file → re-ingest (records an op).
        assert!(s.sync("src/a.rs", "pub fn a() {}\n"));
        let after_first = s.len();
        // Same content again → no-op, no new op recorded.
        assert!(!s.sync("src/a.rs", "pub fn a() {}\n"));
        assert_eq!(s.len(), after_first, "unchanged file logs no op");
        // Changed content → re-ingest.
        assert!(s.sync("src/a.rs", "pub fn a() -> i32 { 1 }\n"));
        assert_eq!(s.len(), after_first + 1);
    }

    #[test]
    fn page_fault_pulls_the_faulting_file_in_and_replays() {
        let mut s = chain_session();
        // A test failure pointing at the deep cause db.rs.
        let err = "thread 'main' panicked at src/db.rs:1:14:\nattempt to add with overflow\n";
        let win = s.page_fault(err, 8000);
        assert!(
            win.items.iter().any(|i| i.uri == "file:src/db.rs"),
            "page fault recalls the faulting file"
        );
        // The whole correction loop replays: reconstructed state matches live.
        let replayed = s.replay_to(s.len());
        assert_eq!(replayed.stats().nodes, s.memory().stats().nodes);
        assert!(s.timeline().last().unwrap().contains("PageFault"));
    }

    #[test]
    fn recall_around_pages_a_demoted_anchor_back_from_cold() {
        use crate::memory::NodeId;
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn query() -> i64 { 0 }\n");
        s.ingest(
            "src/api.rs",
            "use crate::db;\npub fn h() -> i64 { db::query() }\n",
        );
        // Shrink the frugal resident window so some nodes demote to COLD.
        s.live.set_max_resident(1);
        assert!(
            s.live.graph().cold_count() > 0,
            "some nodes demoted to COLD"
        );

        let cold_id = s.live.graph().cold_ids().next().unwrap().0.clone();
        assert!(s.live.graph().is_cold(&NodeId(cold_id.clone())));

        // A recall *around* the demoted node pages it back from COLD first
        // (the page fault on the read path) — transparent to the agent.
        let _ = s.recall(Recall::around(&cold_id), 4096);
        assert!(
            s.live.graph().contains_node(&NodeId(cold_id.clone())),
            "the cold anchor is paged back in by recall"
        );
        assert!(!s.live.graph().is_cold(&NodeId(cold_id)), "no longer cold");
    }

    #[test]
    fn page_fault_resurrects_a_demoted_faulting_file() {
        use crate::memory::NodeId;
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn query() -> i64 { 0 }\n");
        s.ingest(
            "src/api.rs",
            "use crate::db;\npub fn h() -> i64 { db::query() }\n",
        );
        s.live.set_max_resident(1);
        assert!(s.live.graph().cold_count() > 0);

        // A crash trace blaming db.rs pages the (demoted) file back in before recall.
        let win = s.page_fault("thread 'main' panicked at src/db.rs:1:1:\nboom\n", 4096);
        assert!(
            s.live
                .graph()
                .contains_node(&NodeId("file:src/db.rs".into())),
            "the faulting file is resident after the page fault"
        );
        assert!(win.items.iter().any(|i| i.uri == "file:src/db.rs"));
    }

    /// A field finding: depth-2 propagation left a 3-hop cause un-pressurised. With
    /// the default depth (3) a page-fault on the symptom reaches a 3-hop-deep cause,
    /// and the recorded depth replays the exact same pressure (determinism).
    #[test]
    fn page_fault_reaches_a_three_hop_cause_and_replays_the_depth() {
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "use crate::b;\npub fn f() -> i64 { b::g() }\n");
        s.ingest("src/b.rs", "use crate::c;\npub fn g() -> i64 { c::h() }\n");
        s.ingest("src/c.rs", "use crate::d;\npub fn h() -> i64 { d::i() }\n");
        s.ingest("src/d.rs", "pub fn i() -> i64 { 0 }\n");
        // Panic at the entry a.rs (the symptom); the cause d.rs is 3 hops down.
        s.page_fault("thread 'main' panicked at src/a.rs:1:1:\nboom\n", 8000);

        let fr = |m: &CcosMemory| {
            m.graph()
                .nodes
                .iter()
                .find(|(id, _)| id.0 == "file:src/d.rs")
                .map(|(_, n)| n.failure_relevance)
                .unwrap_or(0.0)
        };
        assert!(
            fr(s.memory()) > 0.0,
            "depth-3 page-fault pressurises the 3-hop cause d.rs"
        );
        // Replay uses the recorded depth → reproduces the same pressure on d.rs.
        assert!((fr(&s.replay_to(s.len())) - fr(s.memory())).abs() < 1e-9);
    }

    // ── Tamper-evident timeline chain (P1 — fold tamper-evidence into the primary log) ──

    /// Build a persisted two-op workspace and return its sidecar path.
    fn chained_workspace(tag: &str) -> std::path::PathBuf {
        let path = tmp(tag);
        cleanup(&path);
        let mut s = AgentSession::open(&path).unwrap();
        s.ingest("src/a.rs", "pub fn a() {}\n");
        s.ingest("src/b.rs", "use crate::a;\npub fn b() { a::a() }\n");
        assert!(s.verify_timeline().valid);
        s.checkpoint().unwrap();
        path
    }

    /// Load, mutate, and rewrite a sidecar's JSON.
    fn mutate_sidecar(path: &std::path::Path, f: impl FnOnce(&mut serde_json::Value)) {
        let sidecar = super::oplog_sidecar(path);
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&sidecar, serde_json::to_string(&v).unwrap()).unwrap();
    }

    #[test]
    fn oplog_chain_detects_payload_tampering() {
        let path = chained_workspace("ccos-chain-tamper");
        // Rewrite recorded history: the first ingest now claims another file.
        mutate_sidecar(&path, |v| {
            v["ops"][0]["Ingest"]["uri"] = serde_json::Value::from("src/evil.rs");
        });
        let audit = audit_workspace(&path).unwrap();
        assert!(!audit.integrity.valid, "mutated op must break its link");
        assert!(matches!(
            AgentSession::open(&path),
            Err(MemoryError::TimelineTampered(_))
        ));
        // The evidence is preserved: the sidecar still fails an audit after the
        // refused open (nothing healed it away).
        assert!(!audit_workspace(&path).unwrap().integrity.valid);
        cleanup(&path);
    }

    #[test]
    fn oplog_chain_detects_reorder_and_mid_deletion() {
        let reordered = chained_workspace("ccos-chain-reorder");
        mutate_sidecar(&reordered, |v| {
            let ops = v["ops"].as_array_mut().unwrap();
            ops.swap(0, 1);
        });
        assert!(!audit_workspace(&reordered).unwrap().integrity.valid);
        cleanup(&reordered);

        // Deleting an op *and* its link keeps lengths equal — the successor's
        // link still breaks (its prev changed).
        let deleted = chained_workspace("ccos-chain-delete");
        mutate_sidecar(&deleted, |v| {
            v["ops"].as_array_mut().unwrap().remove(0);
            v["chain"].as_array_mut().unwrap().remove(0);
        });
        assert!(!audit_workspace(&deleted).unwrap().integrity.valid);
        cleanup(&deleted);
    }

    #[test]
    fn oplog_chain_detects_baseline_tampering() {
        let path = tmp("ccos-chain-baseline");
        cleanup(&path);
        let mut s = AgentSession::open(&path).unwrap();
        for i in 0..6 {
            s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
        }
        s.compact(2); // ops fold into the baseline — the chain must now pin it
        assert!(s.verify_timeline().valid);
        s.checkpoint().unwrap();
        mutate_sidecar(&path, |v| {
            let doctored = v["baseline"].as_str().unwrap().replace("f0", "fX");
            v["baseline"] = serde_json::Value::from(doctored);
        });
        let audit = audit_workspace(&path).unwrap();
        assert!(
            !audit.integrity.valid,
            "a doctored baseline must fail its commitment"
        );
        cleanup(&path);
    }

    #[test]
    fn legacy_oplog_backfills_chain_and_verifies() {
        let path = chained_workspace("ccos-chain-legacy");
        // Strip the chain fields — the sidecar an older CCOS wrote.
        mutate_sidecar(&path, |v| {
            let o = v.as_object_mut().unwrap();
            o.remove("chain");
            o.remove("anchor");
            o.remove("baseline_hash");
        });
        let audit = audit_workspace(&path).unwrap();
        assert!(
            audit.legacy && audit.integrity.valid,
            "legacy = nothing to verify yet"
        );
        // Open backfills deterministically; the next checkpoint persists the chain.
        let mut s = AgentSession::open(&path).unwrap();
        assert!(s.verify_timeline().valid);
        assert!(s.verify_timeline().verified_events > 0, "chain established");
        s.checkpoint().unwrap();
        let audit = audit_workspace(&path).unwrap();
        assert!(!audit.legacy && audit.integrity.valid);
        cleanup(&path);
    }

    #[test]
    fn chain_head_survives_checkpoint_reopen_and_compaction() {
        let path = tmp("ccos-chain-head");
        cleanup(&path);
        let mut s = AgentSession::open(&path).unwrap();
        for i in 0..8 {
            s.ingest(&format!("src/m{i}.rs"), &format!("pub fn m{i}() {{}}\n"));
        }
        let head = s.timeline_head();
        // Compaction folds the prefix but must not move the head: the anchor
        // takes over the folded history's commitment.
        s.compact(3);
        assert_eq!(s.timeline_head(), head, "compaction keeps the head");
        assert!(s.verify_timeline().valid);
        s.checkpoint().unwrap();
        let s2 = AgentSession::open(&path).unwrap();
        assert_eq!(s2.timeline_head(), head, "head survives a restart");
        assert!(s2.verify_timeline().valid);
        cleanup(&path);
    }

    #[test]
    fn fork_inherits_a_valid_chain_prefix() {
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn a() {}\n");
        s.ingest("src/b.rs", "pub fn b() {}\n");
        s.ingest("src/c.rs", "pub fn c() {}\n");
        let fork = s.fork_at(2);
        assert!(fork.verify_timeline().valid);
        assert_ne!(fork.timeline_head(), s.timeline_head());
        // The fork's chain is a strict prefix of the trunk's: same links up to
        // the fork point.
        assert_eq!(fork.chain[..], s.chain[..2]);
    }

    #[test]
    fn identical_timelines_have_identical_heads() {
        let build = || {
            let mut s = AgentSession::new();
            s.ingest("src/a.rs", "pub fn a() {}\n");
            s.signal_failure("file:src/a.rs", 2).unwrap();
            s.timeline_head()
        };
        assert_eq!(build(), build(), "the chain is bit-reproducible");
    }

    // ── Distributed multi-agent store (paper §9 item 5) ─────────────────────────

    /// Bit-exact convergence fingerprint — the store's official one
    /// ([`CcosMemory::state_fingerprint`]: graph + sources + both chain heads,
    /// excluding only the non-deterministic audit ids).
    fn view_hash(m: &CcosMemory) -> String {
        m.state_fingerprint().unwrap()
    }

    /// Two agents with disjoint knowledge; B's api.rs calls into A's db.rs.
    fn two_agents() -> (AgentSession, AgentSession) {
        let mut a = AgentSession::new();
        a.set_agent("agent-a");
        a.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
        a.assert_support("src/db.rs", "claim:db-tested", 0.9);
        let mut b = AgentSession::new();
        b.set_agent("agent-b");
        b.ingest(
            "src/api.rs",
            "use crate::db;\npub fn handle() -> i64 { db::timeout() }\n",
        );
        (a, b)
    }

    #[test]
    fn two_agents_exchange_and_converge_bit_identically() {
        let (mut a, mut b) = two_agents();
        let (ba, bb) = (a.export_bundle(0).unwrap(), b.export_bundle(0).unwrap());
        assert_eq!(a.import_bundle(&bb).unwrap(), 1, "B's ingest arrives");
        assert_eq!(
            b.import_bundle(&ba).unwrap(),
            2,
            "A's ingest + assert arrive"
        );

        let (va, vb) = (a.merged_view(), b.merged_view());
        assert_eq!(view_hash(&va), view_hash(&vb), "views are bit-identical");
        // The merged view holds knowledge neither agent had alone: the cross-file
        // Calls edge from B's api.rs into A's db.rs. Call edges need the syn
        // parser — the line-heuristic fallback build (`--no-default-features`)
        // converges identically (the hash assertion above) but mints no fn→fn
        // edges, so the edge claim is syn-only.
        if cfg!(feature = "syn-parser") {
            assert!(va.graph().edges.iter().any(|e| {
                e.edge_type == crate::memory::EdgeType::Calls
                    && e.source.0.contains("api.rs")
                    && e.target.0.contains("db.rs")
            }));
        }
        // And a third agent importing both bundles materializes the same view.
        let mut c = AgentSession::new();
        c.set_agent("agent-c");
        c.import_bundle(&a.export_bundle(0).unwrap()).unwrap();
        c.import_bundle(&b.export_bundle(0).unwrap()).unwrap();
        assert_eq!(view_hash(&c.merged_view()), view_hash(&va));
    }

    #[test]
    fn import_verifies_the_bundle_chain() {
        let (a, mut b) = two_agents();
        let mut tampered = a.export_bundle(0).unwrap();
        tampered.ops[0] = Op::Ingest {
            uri: "src/evil.rs".into(),
            source: "pub fn pwn() {}\n".into(),
        };
        assert!(matches!(
            b.import_bundle(&tampered),
            Err(SyncError::Tampered(_))
        ));
        assert!(
            b.foreign_agents().is_empty(),
            "refused import changes nothing"
        );
    }

    #[test]
    fn import_detects_equivocation_and_gaps() {
        let (mut a, mut b) = two_agents();
        b.import_bundle(&a.export_bundle(0).unwrap()).unwrap();
        // A "rewrites history": an alternative timeline of the same length under
        // the same identity — the overlap disagrees ⇒ equivocation.
        let mut a2 = AgentSession::new();
        a2.set_agent("agent-a");
        a2.ingest("src/db.rs", "pub fn timeout() -> i64 { 9999 }\n");
        a2.assert_support("src/db.rs", "claim:db-tested", 0.9);
        assert!(matches!(
            b.import_bundle(&a2.export_bundle(0).unwrap()),
            Err(SyncError::Diverged { at: 1 })
        ));
        // A gap: A advances by two ops but exports only from step 3.
        a.ingest("src/x.rs", "pub fn x() {}\n");
        a.ingest("src/y.rs", "pub fn y() {}\n");
        assert!(matches!(
            b.import_bundle(&a.export_bundle(3).unwrap()),
            Err(SyncError::Gap {
                known: 2,
                bundle_start: 3
            })
        ));
    }

    #[test]
    fn incremental_import_equals_full_import() {
        let (mut a, mut b) = two_agents();
        b.import_bundle(&a.export_bundle(0).unwrap()).unwrap();
        a.ingest("src/extra.rs", "pub fn extra() -> i64 { 1 }\n");
        // Incremental: only the new suffix travels; overlap re-import is a no-op.
        assert_eq!(b.import_bundle(&a.export_bundle(2).unwrap()).unwrap(), 1);
        assert_eq!(b.import_bundle(&a.export_bundle(0).unwrap()).unwrap(), 0);
        // A fresh receiver of one full bundle sees the identical view.
        let mut c = AgentSession::new();
        c.set_agent("agent-c");
        c.import_bundle(&a.export_bundle(0).unwrap()).unwrap();
        c.import_bundle(&b.export_bundle(0).unwrap()).unwrap();
        assert_eq!(view_hash(&b.merged_view()), view_hash(&c.merged_view()));
    }

    #[test]
    fn sync_state_survives_checkpoint_and_reopen() {
        let path = tmp("ccos-sync-persist");
        cleanup(&path);
        let (a, _) = two_agents();
        let before = {
            let mut s = AgentSession::open(&path).unwrap();
            s.set_agent("agent-b");
            s.ingest(
                "src/api.rs",
                "use crate::db;\npub fn handle() -> i64 { db::timeout() }\n",
            );
            s.import_bundle(&a.export_bundle(0).unwrap()).unwrap();
            s.checkpoint().unwrap();
            view_hash(&s.merged_view())
        };
        let s2 = AgentSession::open(&path).unwrap();
        assert_eq!(s2.agent(), "agent-b", "identity persisted");
        assert_eq!(s2.foreign_agents().len(), 1, "foreign log persisted");
        assert_eq!(view_hash(&s2.merged_view()), before, "view reconstructs");
        assert!(s2.verify_timeline().valid, "own chain still valid");
        cleanup(&path);
    }

    #[test]
    fn export_requires_identity_and_uncompacted_history_and_rejects_self_import() {
        let mut s = AgentSession::new();
        s.ingest("src/a.rs", "pub fn a() {}\n");
        assert!(matches!(s.export_bundle(0), Err(SyncError::NoAgentId)));
        s.set_agent("agent-a");
        let own = s.export_bundle(0).unwrap();
        assert!(matches!(s.import_bundle(&own), Err(SyncError::SelfImport)));
        for i in 0..8 {
            s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
        }
        s.compact(2);
        assert!(matches!(
            s.export_bundle(0),
            Err(SyncError::CompactedHistory)
        ));
        // The still-separable tail exports fine.
        assert!(s.export_bundle(s.floor()).is_ok());
    }

    #[test]
    fn bundles_roundtrip_as_json() {
        let (a, mut b) = two_agents();
        let json = a.export_bundle(0).unwrap().to_json().unwrap();
        let parsed = SyncBundle::from_json(&json).unwrap();
        assert_eq!(parsed.agent, "agent-a");
        assert_eq!(b.import_bundle(&parsed).unwrap(), 2);
    }

    // ── Signed sync: cryptographic agent identity ────────────────────────────────

    /// Half a signature pair is malformed on EVERY build (no crypto needed to
    /// see that), and on a crypto-free build a fully-signed bundle is refused
    /// outright rather than silently accepted unverified.
    #[test]
    fn signature_field_pairing_is_enforced_on_every_build() {
        let (a, mut b) = two_agents();
        let mut bundle = a.export_bundle(0).unwrap();
        bundle.pubkey = Some("AAAA".into());
        assert!(matches!(
            b.import_bundle(&bundle),
            Err(SyncError::BadSignature(_))
        ));
        bundle.sig = Some("BBBB".into());
        // Now a "signed" bundle: crypto builds verify it (and fail — garbage),
        // crypto-free builds refuse it as unsupported. Either way: refused.
        assert!(b.import_bundle(&bundle).is_err());
        // A refused bundle changed nothing: no pin, no foreign ops.
        assert!(b.pinned_keys().is_empty());
        assert!(b.foreign.is_empty());
    }

    #[cfg(feature = "signed-sync")]
    mod signed {
        use super::*;

        /// A session with a generated workspace key, ingesting one file.
        fn keyed_session(tag: &str, agent: &str, uri: &str) -> (AgentSession, std::path::PathBuf) {
            let path = tmp(tag);
            cleanup(&path);
            let _ = std::fs::remove_file(super::super::key_sidecar(&path));
            let pk = generate_workspace_key(&path).unwrap();
            assert!(!pk.is_empty());
            let mut s = AgentSession::open(&path).unwrap();
            s.set_agent(agent);
            s.ingest(uri, "pub fn f() {}\n");
            (s, path)
        }

        fn cleanup_keyed(path: &std::path::Path) {
            cleanup(path);
            let _ = std::fs::remove_file(super::super::key_sidecar(path));
        }

        #[test]
        fn signed_roundtrip_pins_the_key_and_survives_restart() {
            let (a, pa) = keyed_session("ccos-sig-a", "agent-a", "src/a.rs");
            let bundle = a.export_bundle(0).unwrap();
            assert!(
                bundle.pubkey.is_some() && bundle.sig.is_some(),
                "keyed export signs"
            );
            assert_eq!(bundle.pubkey, a.signing_pubkey());

            let pb = tmp("ccos-sig-b");
            cleanup(&pb);
            let mut b = AgentSession::open(&pb).unwrap();
            b.set_agent("agent-b");
            assert_eq!(b.import_bundle(&bundle).unwrap(), 1);
            assert_eq!(
                b.pinned_keys().get("agent-a"),
                bundle.pubkey.as_ref(),
                "first signed import pins the key"
            );
            b.checkpoint().unwrap();
            let b2 = AgentSession::open(&pb).unwrap();
            assert_eq!(
                b2.pinned_keys().get("agent-a"),
                bundle.pubkey.as_ref(),
                "the pin survives a restart"
            );
            cleanup_keyed(&pa);
            cleanup(&pb);
        }

        #[test]
        fn tampered_signed_bundle_fails_signature_before_chain() {
            let (a, pa) = keyed_session("ccos-sig-tamper", "agent-a", "src/a.rs");
            let mut bundle = a.export_bundle(0).unwrap();
            // Mutate an op AND recompute its chain link, so the chain itself is
            // internally consistent — only the signature still tells the truth.
            bundle.ops[0] = Op::Ingest {
                uri: "src/evil.rs".into(),
                source: "pub fn evil() {}\n".into(),
            };
            bundle.chain[0] = op_link_hash(&bundle.anchor, 0, &bundle.ops[0]);
            let mut b = AgentSession::new();
            b.set_agent("agent-b");
            assert!(matches!(
                b.import_bundle(&bundle),
                Err(SyncError::BadSignature(_))
            ));
            cleanup_keyed(&pa);
        }

        #[test]
        fn spoofed_key_and_stripped_signature_both_refuse() {
            let (a, pa) = keyed_session("ccos-sig-spoof-a", "agent-a", "src/a.rs");
            let genuine = a.export_bundle(0).unwrap();
            let mut b = AgentSession::new();
            b.set_agent("agent-b");
            b.import_bundle(&genuine).unwrap(); // pins agent-a's real key

            // An imposter with a DIFFERENT key claims to be agent-a.
            let (imposter, pi) = keyed_session("ccos-sig-spoof-i", "agent-a", "src/a.rs");
            let forged = imposter.export_bundle(0).unwrap();
            assert_ne!(forged.pubkey, genuine.pubkey);
            assert!(matches!(
                b.import_bundle(&forged),
                Err(SyncError::KeyMismatch { .. })
            ));

            // Stripping the signature entirely must not bypass the pin.
            let mut stripped = genuine;
            stripped.pubkey = None;
            stripped.sig = None;
            assert!(matches!(
                b.import_bundle(&stripped),
                Err(SyncError::UnsignedFromPinned { .. })
            ));
            cleanup_keyed(&pa);
            cleanup_keyed(&pi);
        }

        #[test]
        fn keygen_refuses_to_overwrite() {
            let path = tmp("ccos-sig-keygen");
            cleanup_keyed(&path);
            let pk1 = generate_workspace_key(&path).unwrap();
            let again = generate_workspace_key(&path);
            assert!(again.is_err(), "rotation must be explicit");
            // The original key is intact and still loads.
            let s = AgentSession::open(&path).unwrap();
            assert_eq!(s.signing_pubkey(), Some(pk1));
            cleanup_keyed(&path);
        }
    }
}
