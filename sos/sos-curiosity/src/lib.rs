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
//!   **under-connected** (weakly-linked / isolated nodes), **weakly-supported**
//!   (claims refuted yet unsupported), and **maximal-information-gain** over
//!   planner-supplied designs ([`Curiosity::with_designs`]).
//! * [`ScientificQuestion`] — a content-addressed `Object`, grounded in the real
//!   nodes it concerns (a question that grounds in nothing is never emitted).
//! * [`CuriosityPolicy`] / [`Priority`] — explicit, versioned, **integer
//!   fixed-point** scoring: no opaque priorities (Invariant VI), bit-exact
//!   ranking (`L3`), overflow-proof (saturating arithmetic).
//!
//! ## The information lens, and what it does *not* do
//!
//! [`Strategy::MaxInfoGain`] is the sanctioned `sos-curiosity` → `sos-planner`
//! composition edge (RFC-0002 §11.5 rule 3). Curiosity never *computes* EIG —
//! that is the Planning Engine's job, backed by real numerics in `sos-scirust`
//! per Invariant VIII. It **consumes** a planner
//! [`Candidate`](sos_planner::Candidate) and asks the question a real estimate
//! justifies, scored by the very same [`CuriosityPolicy`] as every structural
//! question so the two are directly comparable on one agenda.
//!
//! Two honesty properties are deliberate: the lens scores the **conservative
//! lower bound** (point − standard error), never the point estimate, and it
//! **skips designs whose EIG is not significant** rather than emitting a
//! question whose premise is "this might teach us nothing" — mirroring the
//! planner's own admission rule. See [`Features::from_design`].
//!
//! ## What is deliberately *not* here yet
//!
//! The remaining scanners need backends that have not attached, and are
//! deferred per Invariant VIII (RFC-0002 §01) — no stub: **cross-domain
//! analogy** by subgraph isomorphism (`scirust-graph`),
//! **unexplored-parameters** (`scirust-symbolic`), centrality/modularity
//! connectivity metrics (`scirust-graph`), and **cognitive proposals**
//! (`sos-ccos`, an untrusted proposer whose suggestions would be scored by this
//! same policy).
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
//!
//! ## Example — the information lens
//!
//! ```
//! use sos_core::{DeterminismLevel, HashAlgo, ObjectId};
//! use sos_knowledge::KnowledgeGraph;
//! use sos_planner::{Candidate, Cost, Estimate};
//! use sos_curiosity::{
//!     Curiosity, BeCurious, CuriosityPolicy, Budget, Strategy, DEFAULT_EIG_SATURATION_MILLI,
//! };
//! use sos_store::MemoryStore;
//!
//! let kg = KnowledgeGraph::build(&MemoryStore::new()).unwrap();
//! let design = |t: &[u8]| ObjectId::compute(HashAlgo::default(), b"design", t);
//!
//! // Two designs the Planning Engine has estimated. Curiosity computes no EIG
//! // itself — it asks the question a real estimate justifies.
//! let designs = [
//!     // 1.5 bits, measured exactly.
//!     Candidate::new(design(b"decisive"), Estimate::exact(1_500), Cost::new(1, 0, 0, 0)),
//!     // 0.02 ± 0.03 bits — not significant, so no question is raised at all.
//!     Candidate::new(design(b"noise"), Estimate::new(20, 30, DeterminismLevel::L1), Cost::new(1, 0, 0, 0)),
//! ];
//!
//! let questions = Curiosity::new(&kg)
//!     .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
//!     .sweep(&CuriosityPolicy::default(), &Budget::new(10));
//!
//! // Only the design that can defend a real claim is on the agenda.
//! assert_eq!(questions.len(), 1);
//! assert_eq!(questions[0].question.strategy, Strategy::MaxInfoGain);
//! assert_eq!(questions[0].question.subject, vec![design(b"decisive")]);
//! assert!(questions[0].priority.info_gain > 0);
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
pub use sweep::{BeCurious, Budget, Curiosity, DEFAULT_EIG_SATURATION_MILLI, ScoredQuestion};

// Re-exported so callers can inspect the explanation a question carries without
// depending on `sos-reasoning` directly.
pub use sos_reasoning::{Derivation, DerivationStep, Soundness};
