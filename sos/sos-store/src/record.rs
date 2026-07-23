//! The stored representations: [`StoredRecord`], [`NamedRef`], [`ObjectHeader`].

use serde::{Deserialize, Serialize};
use sos_core::ObjectId;
use sos_core::kind::Kind;

/// A type-erased stored object: its [`Kind`] plus its serialized interchange
/// bytes. The store keeps objects in this erased form so a single backend can
/// hold every kind of object; [`crate::TypedStore`] re-hydrates them to a
/// concrete `Object<B>` on read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredRecord {
    /// The kind (name + schema version) of the stored object.
    pub kind: Kind,
    /// The object's serialized interchange bytes (JSON).
    pub bytes: Vec<u8>,
}

impl StoredRecord {
    /// Construct a stored record.
    #[must_use]
    pub fn new(kind: Kind, bytes: Vec<u8>) -> Self {
        Self { kind, bytes }
    }
}

/// A mutable named pointer into the store — the equivalent of a Git ref /
/// branch, and a **root for garbage collection** ([`crate::gc`]). Studies,
/// publications, and tags are named refs; anything reachable from one is
/// retained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedRef {
    /// The ref name, e.g. `"study/kepler"` or `"publication/1619"`.
    pub name: String,
    /// The object the ref currently points at.
    pub target: ObjectId,
}

impl NamedRef {
    /// Construct a named ref.
    #[must_use]
    pub fn new(name: impl Into<String>, target: ObjectId) -> Self {
        Self {
            name: name.into(),
            target,
        }
    }
}

/// The provenance header of a stored object — just the fields GC needs. It is
/// deserialized from an object's interchange bytes **without knowing the body
/// type**: serde ignores the other fields, so the store can walk the
/// provenance DAG generically (RFC-0002 §04.5 append-only reachability).
#[derive(Debug, Clone, Deserialize)]
pub struct ObjectHeader {
    /// The object's own id.
    pub id: ObjectId,
    /// The object's direct provenance parents.
    pub parents: Vec<ObjectId>,
}
