//! # `sos-curiosity` — the SOS Curiosity Engine (deterministic core)
//!
//! The Curiosity Engine is the **init / idle daemon** of the scientific OS: the
//! always-available process that *generates* scientific questions by finding
//! unknowns in the knowledge graph (RFC-0002 §06). Where every other engine
//! answers questions, this one **asks** them — and it does so **deterministically
//! and LLM-free**, so the research agenda itself is reproducible, defensible, and
//! auditable (a "curiosity sweep" is a citable, re-derivable computation).
//!
//! This crate is the pure, backend-agnostic core. It scans an
//! [`sos_knowledge::KnowledgeGraph`] and provides:
//!
//! * [`Curiosity`] / the [`BeCurious`] trait — [`BeCurious::sweep`] runs the
//!   deterministic scanners under a [`CuriosityPolicy`] and a [`Budget`] and
//!   returns [`ScoredQuestion`]s ranked highest-priority first. Every one carries
//!   a [`sos_reasoning::Derivation`] — *why* it is worth asking, grounded in real
//!   graph structure, never a hunch.
//! * The scanners ([`Strategy`]): **contradiction-hunt** (reusing the Reasoning
//!   Engine's [`Reason::contradictions`](sos_reasoning::Reason::contradictions)),
//!   **under-connected** (weakly-linked / isolated nodes), and **weakly-supported**
//!   (claims refuted yet unsupported).
//! * [`ScientificQuestion`] — a content-addressed `Object`, grounded in the real
//!   nodes it concerns (a question that grounds in nothing is never emitted).
//! * [`CuriosityPolicy`] / [`Priority`] — explicit, versioned, **integer
//!   fixed-point** scoring: no opaque priorities (Invariant VI), bit-exact
//!   ranking (`L3`), overflow-proof (saturating arithmetic).
//!
//! ## What is deliberately *not* here yet
//!
//! The scanners that need a backend are deferred per Invariant VIII (RFC-0002
//! §01): **maximal-information-gain** (EIG/BOED, `sos-planner`), **cross-domain
//! analogy** by subgraph isomorphism (`scirust-graph`), **unexplored-parameters**
//! (`scirust-symbolic`), centrality/modularity connectivity metrics
//! (`scirust-graph`), and **cognitive proposals** (`sos-ccos`, an untrusted
//! proposer whose suggestions would be scored by this same policy). This crate
//! ships only the lenses it can fully implement over the graph alone — no stub.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Object, Body, Author, canonical::{Canonical, CanonicalEncoder}};
//! use sos_store::{MemoryStore, TypedStore};
//! use sos_knowledge::{seal_edge, Relation, KnowledgeGraph};
//! use sos_curiosity::{Curiosity, BeCurious, CuriosityPolicy, Budget, Strategy};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct N { s: String }
//! impl Canonical for N { fn encode(&self, e: &mut CanonicalEncoder) { e.str(&self.s); } }
//! impl Body for N { const KIND: &'static str = "N"; const SCHEMA_VERSION: u32 = 1; }
//!
//! let mut store = MemoryStore::new();
//! let node = |s: &str, st: &mut MemoryStore| {
//!     let o = Object::builder(N { s: s.into() }).author(Author::human("x")).seal();
//!     st.put_object(&o).unwrap(); o.id
//! };
//! let (a, b) = (node("phlogiston", &mut store), node("oxygen", &mut store));
//! // Assert that the two theories contradict — the curiosity engine should ask
//! // how to resolve it.
//! store.put_object(&seal_edge(a, b, Relation::Contradicts, Author::engine("t"))).unwrap();
//!
//! let kg = KnowledgeGraph::build(&store).unwrap();
//! let questions = Curiosity::new(&kg).sweep(&CuriosityPolicy::default(), &Budget::new(10));
//!
//! assert!(!questions.is_empty());
//! let top = &questions[0];
//! assert_eq!(top.question.strategy, Strategy::ContradictionHunt);
//! assert!(top.question.is_grounded());          // cites the two real nodes
//! assert!(top.priority.total > 0);              // an explicit, non-opaque score
//! assert_eq!(top.derivation.steps.len(), 1);    // and an explanation
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod error;
pub mod policy;
pub mod question;
pub mod strategy;
pub mod sweep;

pub use error::{CuriosityError, Result};
pub use policy::{CuriosityPolicy, Features, Priority, SCALE};
pub use question::{ScientificQuestion, seal_question};
pub use strategy::Strategy;
pub use sweep::{BeCurious, Budget, Curiosity, ScoredQuestion};

// Re-exported so callers can inspect the explanation a question carries without
// depending on `sos-reasoning` directly.
pub use sos_reasoning::{Derivation, DerivationStep, Soundness};
