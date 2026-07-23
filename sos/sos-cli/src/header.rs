//! [`GenericHeader`] — every identity field of a stored object *except its
//! body*, read without knowing the body's concrete Rust type.
//!
//! [`sos_store::TypedStore::get_object`] is generic over a compile-time body
//! type `B: Body`, which is exactly right for code that knows what kind of
//! object it's asking for. A CLI walking an arbitrary store does not: `sos
//! log` must print *every* object regardless of kind. This mirrors the same
//! trick [`sos_store::ObjectHeader`] and `sos-store`'s own `FileStore` already
//! use for `id`/`parents` — deserialize only the fields that don't depend on
//! `B`, ignoring the rest. It cannot recompute or verify the object's content
//! hash (that needs the body's [`Canonical`](sos_core::canonical::Canonical)
//! encoding, which requires knowing `B`); commands that need a real integrity
//! check dispatch on [`GenericHeader::kind`] to a concrete type instead (see
//! `verify.rs`).

use serde::Deserialize;
use sos_core::{
    Author, DeterminismLevel, Kind, LamportClock, ObjectId, ProducerRef, ReproMeta, SemVer,
};

/// Every field of [`sos_core::Object`] except `body`, `wall`, and `signature`
/// (the latter two are display-only extras this reader does not need).
#[derive(Debug, Clone, Deserialize)]
pub struct GenericHeader {
    /// The object's content address.
    pub id: ObjectId,
    /// The object's kind (type name + schema version).
    pub kind: Kind,
    /// The object's content-lineage version.
    pub version: SemVer,
    /// The authoritative logical clock.
    pub logical: LamportClock,
    /// Direct provenance parents.
    pub parents: Vec<ObjectId>,
    /// The producing engine/plugin.
    pub producer: ProducerRef,
    /// The initiating principal.
    pub author: Author,
    /// Reproducibility metadata.
    pub repro: ReproMeta,
    /// The realized determinism level.
    pub level: DeterminismLevel,
}

impl GenericHeader {
    /// Parse a header out of a stored object's raw interchange bytes.
    ///
    /// # Errors
    /// [`serde_json::Error`] if `bytes` is not valid JSON or is missing one of
    /// these fields (every object envelope has all of them, so this only fails
    /// on a corrupt or foreign record).
    pub fn parse(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}
