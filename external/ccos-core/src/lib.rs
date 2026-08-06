//! # CCOS — Causal Context Operating System
//!
//! CCOS is an experimental "kernel" that treats an LLM's working context like a
//! virtual-memory system: source code is parsed into a **causal memory graph**,
//! nodes are scored by importance / failure-relevance / recency, and a bounded
//! "context window" is paged in and out much like RAM ↔ VRAM. Every state
//! transition is recorded in an append-only **event log** so a session can be
//! replayed deterministically.
//!
//! ## Quick start
//!
//! The core entry types are re-exported at the crate root:
//!
//! ```
//! use ccos_core::{CcosMemory, ExternalMemory, Recall};
//!
//! let mut mem = CcosMemory::new();
//! mem.ingest_source("src/db.rs", "pub fn query() -> i64 { 0 }\n");
//! let window = mem.recall(&Recall::working_set(), 1024);
//! assert!(!window.items.is_empty());
//! ```
//!
//! ## Modules
//!
//! - [`parser`] — dependency-light line-based AST extraction (modules, `use`
//!   statements, symbols) from Rust source.
//! - [`memory`] — the causal [`memory::MemoryGraph`]: scoring, failure
//!   propagation, deterministic eviction/paging and context-window selection.
//! - [`incremental`] — `O(Δ)` graph updates: only the changed file's subgraph is
//!   re-evaluated on each edit ([`incremental::IncrementalGraphEngine`]).
//! - [`event_log`] — append-only [`event_log::EventLog`] with deterministic
//!   replay over typed [`event_log::EventPayload`] records.
//! - [`distributed_event_log`] — hash-chained, tamper-evident event log with an
//!   integrity verifier.
//! - [`llm`] — async client for an Ollama-style `/api/generate` endpoint with
//!   retries and a deterministic offline fallback.
//! - [`guard`] — validation/sanitization layer that rejects malformed model
//!   output and substitutes a safe, valid-JSON fallback.
//! - [`sanitizer`] — deterministic Unicode de-obfuscation of ingested text:
//!   surfaces hidden-character injection vectors (Trojan-Source bidi overrides,
//!   zero-width formatting, Unicode-Tags ASCII smuggling) as explicit, auditable
//!   literals rather than silently stripping them.
//! - [`hashing_tokenizer`] — vocabulary-free, fixed-size, deterministic feature
//!   hashing (the "hashing trick") turning text into the vector `X`.
//! - [`injection_classifier`] — a linear log-space (multinomial-Naive-Bayes)
//!   *signal* over `X` with an immutable SHA-256-verified weight blob and a
//!   forensic, per-feature explanation of every score.
//! - [`consensus`] — majority and confidence-weighted multi-model voting.
//! - [`adversarial`] — fault injector (JSON corruption, hallucination, prompt
//!   injection, timeouts) used to harden the guard and the graph.
//! - [`persist`] — save/load a full [`persist::KernelSnapshot`] (graph + both
//!   logs) to JSON for cross-session replay and verification.
//! - [`query`] — read-only causal queries (impact/cause walks, hot set, GraphML
//!   export) behind the `top`, `blame` and `export` subcommands.
//! - [`trace`] — the dynamic layer: parse `cargo test` / panic / backtrace output
//!   into the source locations a crash touched (a direct symptom→cause path), to
//!   seed a *context page fault* instead of a diffuse structural walk.
//! - [`agent_session`] — an event-sourced cognitive timeline: record an agent's
//!   memory operations, replay the exact state at any step, and run *what-if*
//!   recalls (time-travel debugging) — the deterministic/auditable angle RAG lacks.
//! - [`external_memory`] — a documented façade ([`external_memory::ExternalMemory`]
//!   / [`external_memory::CcosMemory`]) an agent uses to treat CCOS as external
//!   working memory: ingest source, signal failures, recall a bounded causal
//!   window, verify, and checkpoint.
//! - [`mcp`] — a dependency-free [Model Context Protocol](https://modelcontextprotocol.io)
//!   server (stdio JSON-RPC 2.0) that exposes the [`external_memory`] façade as MCP
//!   tools, so any MCP-compatible agent can use CCOS as native working memory.
//! - [`postmortem`] — an interactive **time-travel debugger** over an
//!   [`agent_session::AgentSession`]: walk a recorded (or persisted) cognitive
//!   timeline by hand, inspect how the recalled context window drifts, and diff two
//!   points in the agent's history.
//! - [`region_engine`] — the **Context Region Engine** (v0.3): clusters the
//!   graph into spatial [`region_engine::ContextRegionEngine`] regions that are
//!   hydrated as context windows, with a dynamic [`context_policy`] admission
//!   policy and deterministic replay. See [`context_region`], [`region_metrics`].
//!
//! ## Wiring of the recent modules
//!
//! All of these are now on the **live path**: [`compressor`] (reversible CCR
//! compression of the recalled window), [`sanitizer`] (inline Unicode
//! de-obfuscation at ingest), [`injection_classifier`] (an injection-signal score
//! on every [`external_memory::IngestReport`], via a shared detector), and
//! [`embeddings`] (semantic recall through [`external_memory::Recall::Semantic`]).
//! [`eviction_policy`] is wired into [`memory::MemoryGraph::enforce_paging`] but
//! is **untrained by default** — in which case paging is *exactly* the
//! deterministic greedy (lowest score first), so it is never worse; train it
//! offline via [`memory::MemoryGraph::train_eviction_policy`] to give it effect.
//!
//! ## Invariants
//!
//! The memory graph maintains `edges ⊆ nodes × nodes` at all times (see
//! [`memory::MemoryGraph::add_edge`] and
//! [`memory::MemoryGraph::prune_dangling_edges`]). The `nodes`/`edges` stores are
//! `pub(crate)`, reachable from outside only through read accessors
//! ([`memory::MemoryGraph::node`], [`node_mut`](memory::MemoryGraph::node_mut),
//! [`edges`](memory::MemoryGraph::edges), …) and the structural mutators above —
//! so an external caller cannot push a dangling edge or orphan a node and break
//! the invariant. Eviction order is deterministic, so replays and snapshot hashes
//! are reproducible regardless of `HashMap` iteration order.

pub mod adversarial;
pub mod agent_session;
pub mod cold_index;
pub mod compressor;
pub mod conformal;
pub mod consensus;
pub mod distributed_event_log;
pub mod drift;
pub mod dtw;
pub mod egress;
pub mod embeddings;
pub mod event_log;
pub mod eviction_policy;
pub mod external_memory;
pub mod extractor;
pub mod guard;
pub mod hashing_tokenizer;
pub mod incremental;
pub mod injection_classifier;
pub mod license;
pub mod lingam;
#[cfg(feature = "llm")]
pub mod llm;
pub mod lsa;
pub mod lzss;
pub mod mcp;
pub mod memory;
pub mod migrate;
// Quarantined neural embedder (off-by-default `neural-embed` feature): an
// `retrieval::Encoder` over a LOCAL Ollama-style /api/embeddings endpoint. The
// default build compiles none of it and stays deterministic + replay-exact —
// that is the quarantine. See the module docs and docs/NEURAL_EMBED.md.
#[cfg(feature = "neural-embed")]
pub mod neural_embed;
// Pro OctaSoma semantic memory (off-by-default `octasoma` feature): region-sharded,
// embedding-based semantic anchors expanded through the causal graph — the
// validated scope→rerank cascade. Compiling it is the cargo feature; *using* it is
// gated by the offline license (`Feature::OctaSomaMemory`). The default build
// compiles none of it and stays deterministic + replay-exact. See the module docs.
#[cfg(feature = "octasoma")]
pub mod octa_index;
pub mod parser;
pub mod persist;
pub mod postmortem;
pub mod query;
pub mod retrieval;
pub mod retrodict;
pub mod sanitizer;
pub mod setup;
pub mod spectral;
pub mod trace;
pub mod util;

// ── CCOS v0.3 — Autonomous Context Runtime ──────────────────────────
pub mod agents;
pub mod benchmark;
pub mod causal_flash;
pub mod persistence;
pub mod scheduler;
#[cfg(feature = "llm")]
pub mod workspace;

// ── CCOS v0.3 — Context Region Engine (spatial memory) ──────────────
pub mod context_policy;
pub mod context_region;
#[cfg(feature = "llm")]
pub mod eval;
pub mod experiment;
pub mod region_engine;
pub mod region_metrics;

// ── Core re-exports ─────────────────────────────────────────────────
//
// The handful of entry types a library consumer needs, lifted to the crate root
// so they can be reached as `ccos_core::CcosMemory` / `ccos_core::Recall` instead of the
// full module path. The modules above remain public for everything else.
pub use agent_session::AgentSession;
pub use event_log::EventLog;
pub use external_memory::{
    CcosMemory, ExternalMemory, IngestReport, Integrity, MemoryError, Recall, RecallItem,
    RecallWindow,
};
pub use memory::{EdgeType, GraphEdge, GraphNode, MemoryGraph, NodeId, NodeType, ScoringWeights};
pub use persist::KernelSnapshot;
