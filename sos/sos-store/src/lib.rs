//! # `sos-store` — the SOS Storage Layer (the kernel's filesystem)
//!
//! A **content-addressed object store**: the append-only, deduplicating,
//! tamper-evident substrate on which the Scientific Operating System keeps its
//! objects (RFC-0002 §09.1). It is Git-shaped — "everything is a hashed object"
//! — and small enough that a laptop-local backend and a shared object-storage
//! backend implement the same trait.
//!
//! ## What it provides
//!
//! * [`ObjectStore`] — the backend trait: raw put/get keyed by
//!   [`sos_core::ObjectId`], content-addressed [`BlobRef`] blobs, and mutable
//!   named [`NamedRef`]s (the GC roots).
//! * [`TypedStore`] — a blanket extension over any [`ObjectStore`] giving
//!   [`TypedStore::put_object`] / [`TypedStore::get_object`], which **verify the
//!   content-address on the way in and out** so corruption or tampering is
//!   caught, never silently returned.
//! * [`gc`] — explicit, reachability-based garbage collection from the named
//!   refs, so a discarded experimental branch is pruned only on purpose.
//! * [`MemoryStore`] — a complete in-memory backend (not a mock: a real,
//!   deterministic `HashMap`-backed store, useful for tests, ephemeral
//!   sessions, and embedding).
//! * [`FileStore`] — a complete, persistent, filesystem-backed store: objects
//!   and blobs as content-addressed files under a git-style sharded directory
//!   layout, refs as a small JSON index. State survives closing and reopening
//!   the store. A remote/object-storage backend implementing the same
//!   [`ObjectStore`] trait is a follow-on increment.
//!
//! ## Guarantees
//!
//! * **Idempotent, content-addressed puts.** Storing an object (or blob) you
//!   already hold is a no-op returning the same id — the root of dedup.
//! * **Integrity on read and write.** [`TypedStore`] refuses to store an object
//!   whose id does not match its content, and refuses to return one that was
//!   corrupted in storage — surfacing [`sos_core::SosError::IdMismatch`].
//! * **Deterministic iteration.** [`ObjectStore::object_ids`] and
//!   [`ObjectStore::refs`] return sorted results, so nothing downstream depends
//!   on hash-map order.
//!
//! ## Example
//!
//! ```
//! use sos_core::{Object, Body, Author, canonical::{Canonical, CanonicalEncoder}};
//! use sos_store::{MemoryStore, ObjectStore, TypedStore};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct Note { text: String }
//! impl Canonical for Note {
//!     fn encode(&self, e: &mut CanonicalEncoder) { e.str(&self.text); }
//! }
//! impl Body for Note { const KIND: &'static str = "Note"; const SCHEMA_VERSION: u32 = 1; }
//!
//! let mut store = MemoryStore::new();
//! let note = Object::builder(Note { text: "hello".into() })
//!     .author(Author::human("ada"))
//!     .seal();
//! let id = store.put_object(&note).unwrap();
//!
//! // Round-trips to an identical, integrity-checked object.
//! let back: Object<Note> = store.get_object(id).unwrap().unwrap();
//! assert_eq!(back.id, id);
//! assert_eq!(back.body.text, "hello");
//!
//! // Storing it again is a no-op (content-addressed dedup).
//! store.put_object(&note).unwrap();
//! assert_eq!(store.object_ids().len(), 1);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod blob;
pub mod error;
pub mod file;
pub mod mem;
pub mod record;
pub mod store;

pub use blob::BlobRef;
pub use error::{Result, StoreError};
pub use file::FileStore;
pub use mem::MemoryStore;
pub use record::{NamedRef, ObjectHeader, StoredRecord};
pub use store::{GcReport, ObjectStore, TypedStore, gc, reachable};
