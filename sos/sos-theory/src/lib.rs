//! # `sos-theory` — the SOS Theory Engine (deterministic core)
//!
//! Theories are **first-class, immutable, evolving objects** — not a status flag
//! on a hypothesis (RFC-0002 §07.3). A [`Theory`] records every field the mandate
//! lists — axioms, assumptions, equations, domain of validity, supporting *and*
//! contradicting evidence, confidence, citations, revision parent, and competing
//! theories — as [`ObjectId`](sos_core::ObjectId)s into the graph, so a theory is
//! a **view over provenance**, not a document.
//!
//! Two principles the mandate insists on are enforced here:
//!
//! * **Anomalies are retained, never hidden.** `contradicting` evidence is a
//!   first-class field; [`Theory::revise`] carries it forward, so "what does this
//!   theory fail to explain?" is always answerable.
//! * **Theories evolve; provenance does not.** A revision is a *new* node citing
//!   its parent — the parent is never deleted and stays queryable. The
//!   [`Theories`] engine can walk the whole [revision lineage](TheoryEngine::revision_chain).
//!
//! Competing theories **coexist**: [`TheoryEngine::compare`] ranks rivals over
//! their shared [`Scope`] (domain of validity) rather than forcing a single
//! winner.
//!
//! ## What is deliberately *not* here yet
//!
//! Per Invariant VIII (RFC-0002 §01), the parts that need a backend are deferred
//! — **no stub**:
//! * **Bayes-factor ranking.** [`compare`](TheoryEngine::compare) ranks by
//!   retained evidential balance ([`RankBasis::EvidentialBalance`]); posterior-odds
//!   `Confidence` ranking awaits the statistics backend.
//! * **Discriminating-experiment planning.** "Which experiment best separates two
//!   rivals" is an expected-information-gain query to `sos-planner`.
//!
//! ## Example
//!
//! ```
//! use sos_core::Author;
//! use sos_store::{MemoryStore, TypedStore};
//! use sos_theory::{Theory, Scope, seal_theory, Theories, TheoryEngine};
//!
//! let mut store = MemoryStore::new();
//! let a = Author::engine("physics");
//!
//! // Newtonian gravity: valid at low velocity.
//! let newton = seal_theory(
//!     Theory::builder(Scope::from_predicates(["low-velocity"])).build(),
//!     a.clone(),
//! );
//! store.put_object(&newton).unwrap();
//!
//! // A revision that must address an anomaly Newton cannot explain. (`anomaly`
//! // is any evidence id; we reuse a known id here for brevity.)
//! let anomaly = newton.id;
//! let gr = seal_theory(newton.body.revise(newton.id, &[anomaly]), a);
//! store.put_object(&gr).unwrap();
//!
//! let engine = Theories::new(&store);
//! // The lineage is queryable: the revision supersedes Newton.
//! assert_eq!(engine.revision_chain(gr.id).unwrap(), vec![gr.id, newton.id]);
//! // The successor retains the forcing anomaly (nothing hidden).
//! assert!(engine.get(gr.id).unwrap().contradicting.contains(&anomaly));
//! ```
//!
//! [`ObjectId`]: sos_core::ObjectId

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod engine;
pub mod error;
pub mod scope;
pub mod theory;

pub use engine::{RankBasis, RankedTheory, Ranking, Theories, TheoryEngine};
pub use error::{Result, TheoryError};
pub use scope::Scope;
pub use theory::{Theory, TheoryBuilder, seal_theory};
