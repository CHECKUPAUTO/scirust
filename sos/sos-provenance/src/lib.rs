//! # `sos-provenance` — the SOS Provenance Engine
//!
//! Provenance in SOS is not a log bolted onto the side; it **is** the edge set
//! of the object graph (RFC-0002 §06.1). This crate makes that edge set
//! *queryable*: given any [`sos_store::ObjectStore`], it answers the two
//! questions the reproducibility crisis cannot —
//!
//! * **"Why do we believe X?"** → [`ProvenanceGraph::ancestors`] (the transitive
//!   closure of X's parents: everything X was derived from).
//! * **"What breaks if X is retracted?"** → [`ProvenanceGraph::descendants`]
//!   (everything derived, transitively, from X).
//!
//! Both are graph traversals with complete answers by construction, because the
//! object graph is an append-only Merkle DAG.
//!
//! It also provides deterministic [`env`] capture: filling an
//! [`sos_core::EnvRecord`] with the build's hardware class and OS so the
//! reproducibility key ([`sos_core::EnvRecord::digest`]) reflects where an
//! object was actually produced.
//!
//! ## What is deliberately *not* here yet
//!
//! **Signing.** The [`sos_core::Signature`] slot is filled by the Merkle/Lamport
//! attestation of `scirust-provenance`, which is a `scirust-*` crate. Per the
//! backend-independence invariant (RFC-0002 §01.VIII) `scirust-*` may appear only
//! in the two adapter crates, so the concrete signer lives in `sos-scirust`
//! (a later increment) implementing a `Signer`/`Verifier` interface — not here.
//! This crate stays backend-agnostic and pure.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Object, Body, Author, canonical::{Canonical, CanonicalEncoder}};
//! use sos_store::{MemoryStore, TypedStore};
//! use sos_provenance::ProvenanceGraph;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct N { s: String }
//! impl Canonical for N { fn encode(&self, e: &mut CanonicalEncoder) { e.str(&self.s); } }
//! impl Body for N { const KIND: &'static str = "N"; const SCHEMA_VERSION: u32 = 1; }
//!
//! let mut store = MemoryStore::new();
//! let root = Object::builder(N { s: "question".into() }).author(Author::human("a")).seal();
//! store.put_object(&root).unwrap();
//! let derived = Object::builder(N { s: "hypothesis".into() })
//!     .author(Author::human("a")).parents(vec![root.id]).seal();
//! store.put_object(&derived).unwrap();
//!
//! let g = ProvenanceGraph::build(&store).unwrap();
//! assert_eq!(g.ancestors(derived.id), vec![root.id]);   // why we believe it
//! assert_eq!(g.descendants(root.id), vec![derived.id]);  // what depends on it
//! assert_eq!(g.roots(), vec![root.id]);                  // the study's questions/axioms
//! assert_eq!(g.tips(), vec![derived.id]);                // the open frontier
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod env;
pub mod error;
pub mod graph;

pub use env::{EnvCapture, detect_hardware, detect_os};
pub use error::{ProvError, Result};
pub use graph::{ProvenanceGraph, ancestors, descendants};
