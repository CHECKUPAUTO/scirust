//! # `sos-core` — the Scientific Operating System kernel
//!
//! This crate is the trusted core of the [Scientific Operating System
//! (SOS)](https://github.com/Memorithm/scirust) described in `docs/sos/`
//! (RFC-0002). It defines the **immutable, content-addressed scientific
//! object** — the single data structure every SOS engine reads and writes —
//! together with the primitives that give it its guarantees:
//!
//! * [`canonical`] — deterministic byte encoding: equal values encode to
//!   byte-identical output on any machine, and distinct values never collide.
//! * [`hash`] / [`id`] — content addressing: an object's identity is the
//!   [`ObjectId`] hash of its canonical form, so identical reasoning yields
//!   identical ids everywhere, and tampering is detectable.
//! * [`determinism`] — the honest four-level taxonomy ([`DeterminismLevel`],
//!   `L0..L3`) with `meet`-based propagation, so a reproducibility claim is
//!   never a bare boolean.
//! * [`repro`] / [`provenance`] — the full reproducibility and provenance
//!   metadata every object carries (seed, environment digest, producer,
//!   author, optional signature).
//! * [`object`] — the [`Object`] envelope that binds all of the above to a
//!   kind-specific [`Body`].
//!
//! ## Design invariants (RFC-0002 §01)
//!
//! * **Immutable & content-addressed.** An [`Object`] is never mutated; its
//!   [`ObjectId`] is a function of its content (including its parents), making
//!   the object graph a Merkle DAG.
//! * **Deterministic.** No wall-clock or randomness enters an object's
//!   identity. The advisory wall-clock field is excluded from the hash; the
//!   authoritative order is the logical [`LamportClock`].
//! * **Honest about reproducibility.** Every object declares the
//!   [`DeterminismLevel`] it realized; the kernel provides the `meet` used to
//!   propagate the weakest level along any dependency path.
//!
//! ## Purity
//!
//! `#![forbid(unsafe_code)]` and `#![deny(missing_docs)]` are enforced
//! crate-wide: SOS is pure, safe, fully-documented Rust with no FFI. The
//! content hash is SHA-256 (via the pure-Rust `sha2` crate, consistent with
//! `scirust-provenance` and `scirust-sciagent::CcosLog`); the hash algorithm
//! is versioned in [`HashAlgo`] so a future addition (e.g. BLAKE3) is a
//! non-breaking, explicitly-tagged extension rather than a silent change.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Object, Body, Author, DeterminismLevel, canonical::{Canonical, CanonicalEncoder}};
//! use serde::{Serialize, Deserialize};
//!
//! // A minimal domain body: a scientific question.
//! #[derive(Clone, Serialize, Deserialize)]
//! struct Question { text: String }
//!
//! impl Canonical for Question {
//!     fn encode(&self, enc: &mut CanonicalEncoder) { enc.str(&self.text); }
//! }
//! impl Body for Question {
//!     const KIND: &'static str = "Question";
//!     const SCHEMA_VERSION: u32 = 1;
//! }
//!
//! let q = Object::builder(Question { text: "Does A cause B?".into() })
//!     .author(Author::human("ada"))
//!     .level(DeterminismLevel::L3)
//!     .seal();
//!
//! // The id is reproducible: sealing the identical content again yields it.
//! let q2 = Object::builder(Question { text: "Does A cause B?".into() })
//!     .author(Author::human("ada"))
//!     .level(DeterminismLevel::L3)
//!     .seal();
//! assert_eq!(q.id, q2.id);
//! assert!(q.verify_id());
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod canonical;
pub mod clock;
pub mod determinism;
pub mod error;
pub mod hash;
pub mod id;
pub mod kind;
pub mod object;
pub mod provenance;
pub mod repro;
pub mod version;

pub use clock::LamportClock;
pub use determinism::DeterminismLevel;
pub use error::{Result, SosError};
pub use hash::{Digest, HashAlgo};
pub use id::ObjectId;
pub use kind::Kind;
pub use object::{Body, Object};
pub use provenance::{Author, ProducerRef, Signature};
pub use repro::{BackendVersion, EnvRecord, ReproMeta, RngId};
pub use version::SemVer;
