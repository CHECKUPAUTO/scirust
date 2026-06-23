//! `scirust-trader` — auditable crypto-trading pipeline.
//!
//! Architecture (5 layers, each replaceable independently):
//!
//! ```text
//!  [market]  →  [indicators]  →  [model]  →  [certify]  →  [agent+LLM]  →  [proof]
//! ```
//!
//! Design rules
//! ------------
//! 1. **Determinism first** — every numeric step uses pinned reduction order.
//! 2. **LLM never decides alone** — the agent trait forces SciRust to emit the
//!    certified prediction first; the LLM only narrates and sanity-checks.
//! 3. **Every decision is sealed** — a `DecisionProof` with SHA-256 manifest is
//!    written to disk for third-party replay.
//! 4. **No real order execution in the MVP** — a `MockExchange` simulates fills
//!    so the pipeline is safe to test end-to-end.

pub mod agent;
pub mod certify;
pub mod cli;
pub mod indicators;
pub mod market;
pub mod model;
pub mod proof;
pub mod risk;
