//! # `sos-knowledge` — the SOS Knowledge Engine (typed knowledge graph)
//!
//! The Knowledge Engine is SOS's store of *what is believed to hold* — a **real
//! semantic knowledge graph**, not documentation (RFC-0002 §04). Its nodes are
//! ordinary content-addressed [`sos_core::Object`]s (laws, equations,
//! invariants, papers, …) and its **edges are first-class objects too**: an
//! [`Edge`] carries a typed [`Relation`] between two node ids, so every asserted
//! relationship is itself hashed and provenance-bound.
//!
//! This crate is the pure, backend-agnostic core of the engine:
//!
//! * [`Relation`] — the core edge vocabulary (`is-a`, `generalizes`,
//!   `contradicts`, `analogous-to`, …) plus [`Relation::Custom`].
//! * [`Edge`] — a relation between two objects, sealed as an `Object<Edge>` and
//!   stored like any other object ([`seal_edge`]).
//! * [`KnowledgeGraph`] — a deterministic, in-memory view built from any
//!   [`sos_store::ObjectStore`]: it reads the stored [`Edge`]s and answers
//!   **structural** queries — `neighbors`, `in_neighbors`, `related`, and
//!   shortest `path` — via the [`Knowledge`] trait.
//!
//! ## What is deliberately *not* here yet
//!
//! The heavier, *semantic* reasoning over the graph — Datalog inference, e-graph
//! equality, and cross-domain analogy detection by subgraph isomorphism — is the
//! job of the Reasoning Engine, which wraps `scirust-neuro-symbolic` and
//! `scirust-graph`. Because `scirust-*` may appear only in the backend adapter
//! crates (Invariant VIII, RFC-0002 §01), that logic lives in `sos-reasoning` /
//! `sos-scirust`, not here. This crate ships only the deterministic graph
//! substrate it can fully implement and test — no stub.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Object, Body, Author, canonical::{Canonical, CanonicalEncoder}};
//! use sos_store::{MemoryStore, TypedStore};
//! use sos_knowledge::{seal_edge, Relation, KnowledgeGraph, Knowledge};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct Law { name: String }
//! impl Canonical for Law { fn encode(&self, e: &mut CanonicalEncoder) { e.str(&self.name); } }
//! impl Body for Law { const KIND: &'static str = "Law"; const SCHEMA_VERSION: u32 = 1; }
//!
//! let mut store = MemoryStore::new();
//! let a = Author::human("ada");
//! let kepler = Object::builder(Law { name: "kepler-3".into() }).author(a.clone()).seal();
//! let newton = Object::builder(Law { name: "newton-gravity".into() }).author(a.clone()).seal();
//! store.put_object(&kepler).unwrap();
//! store.put_object(&newton).unwrap();
//!
//! // Assert "kepler-3 specializes newton-gravity" — an edge is just an object.
//! let edge = seal_edge(kepler.id, newton.id, Relation::Specializes, a);
//! store.put_object(&edge).unwrap();
//!
//! let kg = KnowledgeGraph::build(&store).unwrap();
//! assert_eq!(kg.neighbors(kepler.id, &Relation::Specializes), vec![newton.id]);
//! assert_eq!(kg.in_neighbors(newton.id, &Relation::Specializes), vec![kepler.id]);
//! assert_eq!(kg.related(kepler.id, newton.id), vec![Relation::Specializes]);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod edge;
pub mod error;
pub mod graph;
pub mod relation;

pub use edge::{Edge, seal_edge};
pub use error::{KnowledgeError, Result};
pub use graph::{EdgeRef, Knowledge, KnowledgeGraph};
pub use relation::Relation;
