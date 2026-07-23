//! # `sos-reasoning` — the SOS Reasoning Engine (deterministic core)
//!
//! The Reasoning Engine is the **compiler of the scientific OS**: it transforms
//! knowledge into conclusions deterministically, and — like a compiler emitting
//! debug symbols — every conclusion ships with a [`Derivation`] that explains it
//! (RFC-0002 §05). It is **LLM-free**: reasoning is symbolic and graph-theoretic,
//! reproducible bit-for-bit.
//!
//! This crate is the pure, backend-agnostic core of the engine. It reasons over
//! the [`sos_knowledge::KnowledgeGraph`] and provides:
//!
//! * [`Reasoner`] / the [`Reason`] trait — the query surface:
//!   * [`Reason::entails`] — decide whether `from --relation--> to` is derivable
//!     (a direct edge, or a chain of a **transitive** relation), returning a
//!     [`Conclusion`] with a [`Verdict`] and a [`Derivation`] that cites the exact
//!     edges used.
//!   * [`Reason::contradictions`] — find recorded incompatibilities: asserted
//!     `contradicts` edges and mutual-`supersedes` cycles.
//! * [`Derivation`] / [`DerivationStep`] — the explanation, itself a
//!   content-addressed `Object<Derivation>` (so an explanation can be stored,
//!   cited, and re-verified).
//! * [`Soundness`] (`Proof` | `Check`) — the honest strength label: a `Proof` is
//!   sound; a `Check` is deterministic evidence, not a theorem.
//! * [`Contradiction`] — a recorded incompatibility, a content-addressed object.
//!
//! ## What is deliberately *not* here yet
//!
//! The heavier reasoning — Datalog inference, SAT/SMT, e-graph equality
//! saturation, theorem proving, and analogy detection by subgraph isomorphism —
//! is provided by wrapping `scirust-neuro-symbolic` / `scirust-graph`. Because
//! `scirust-*` may appear only in the backend adapter crates (Invariant VIII,
//! RFC-0002 §01), that logic lives in `sos-scirust`, not here. This crate ships
//! only the sound, deterministic reasoning it can fully implement and test —
//! no stub.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Object, Body, Author, canonical::{Canonical, CanonicalEncoder}};
//! use sos_store::{MemoryStore, TypedStore};
//! use sos_knowledge::{seal_edge, Relation, KnowledgeGraph};
//! use sos_reasoning::{Reasoner, Reason, Verdict, Soundness};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct N { s: String }
//! impl Canonical for N { fn encode(&self, e: &mut CanonicalEncoder) { e.str(&self.s); } }
//! impl Body for N { const KIND: &'static str = "N"; const SCHEMA_VERSION: u32 = 1; }
//!
//! let mut store = MemoryStore::new();
//! let a = Author::engine("test");
//! let node = |s: &str, st: &mut MemoryStore| {
//!     let o = Object::builder(N { s: s.into() }).author(Author::human("x")).seal();
//!     st.put_object(&o).unwrap(); o.id
//! };
//! let (x, y, z) = (node("x", &mut store), node("y", &mut store), node("z", &mut store));
//! // x specializes y, y specializes z  ⇒  x specializes z (by transitivity).
//! store.put_object(&seal_edge(x, y, Relation::Specializes, a.clone())).unwrap();
//! store.put_object(&seal_edge(y, z, Relation::Specializes, a)).unwrap();
//!
//! let kg = KnowledgeGraph::build(&store).unwrap();
//! let r = Reasoner::new(&kg);
//! let c = r.entails(x, &Relation::Specializes, z);
//! assert_eq!(c.verdict, Verdict::Proven);
//! assert_eq!(c.derivation.soundness, Soundness::Proof); // transitivity is sound
//! assert_eq!(c.derivation.steps.len(), 2);              // two edges chained
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod contradiction;
pub mod derivation;
pub mod error;
pub mod reason;
pub mod soundness;

pub use contradiction::{Contradiction, seal_contradiction};
pub use derivation::{Derivation, DerivationStep};
pub use error::{ReasoningError, Result};
pub use reason::{Conclusion, Reason, Reasoner, is_transitive};
pub use soundness::{Soundness, Verdict};
