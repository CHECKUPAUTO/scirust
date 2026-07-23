//! The [`ObjectStore`] backend trait, the [`TypedStore`] extension, and
//! reachability [`gc`].

use std::collections::HashSet;

use sos_core::{Body, Object, ObjectId, SosError};

use crate::blob::BlobRef;
use crate::error::{Result, StoreError};
use crate::record::{NamedRef, ObjectHeader, StoredRecord};

/// A content-addressed object store backend.
///
/// Implementations key objects and blobs by their content address and hold
/// mutable named refs. Two behavioural contracts every backend must honour:
///
/// * **Idempotent, first-wins puts.** [`ObjectStore::put_raw`] and
///   [`ObjectStore::put_blob`] on an id/ref that already exists are no-ops that
///   return the existing content — the basis of content-addressed dedup. (The
///   advisory wall-clock and detached signature are *not* part of an id, so a
///   re-put of an object differing only in those retains the first-stored
///   form; attach a signature before the first put.)
/// * **Deterministic iteration.** [`ObjectStore::object_ids`] and
///   [`ObjectStore::refs`] return **sorted** results so nothing depends on
///   hash-map order.
pub trait ObjectStore {
    /// Store a type-erased record under `id`. Idempotent (first-wins).
    fn put_raw(&mut self, id: ObjectId, record: StoredRecord);
    /// Fetch the record at `id`, if present.
    fn get_raw(&self, id: ObjectId) -> Option<StoredRecord>;
    /// Whether an object with `id` is stored.
    fn has(&self, id: ObjectId) -> bool;
    /// All stored object ids, **sorted**.
    fn object_ids(&self) -> Vec<ObjectId>;
    /// Remove the object at `id`; returns whether it was present.
    fn remove_raw(&mut self, id: ObjectId) -> bool;

    /// Store `bytes` as a content-addressed blob, returning its [`BlobRef`].
    /// Idempotent.
    fn put_blob(&mut self, bytes: &[u8]) -> BlobRef;
    /// Fetch a blob's bytes, if present.
    fn get_blob(&self, r: BlobRef) -> Option<Vec<u8>>;
    /// Whether a blob is stored.
    fn has_blob(&self, r: BlobRef) -> bool;

    /// Point the named ref `name` at `target` (creating or moving it).
    fn set_ref(&mut self, name: &str, target: ObjectId);
    /// The object `name` points at, if the ref exists.
    fn get_ref(&self, name: &str) -> Option<ObjectId>;
    /// Remove a named ref; returns whether it existed.
    fn remove_ref(&mut self, name: &str) -> bool;
    /// All named refs, **sorted by name**.
    fn refs(&self) -> Vec<NamedRef>;
}

/// Typed, integrity-checked access layered over any [`ObjectStore`].
///
/// This is where content addressing earns its keep: [`TypedStore::put_object`]
/// refuses to store an object whose id does not match its content, and
/// [`TypedStore::get_object`] refuses to return one that was corrupted or is of
/// the wrong kind. Blanket-implemented for every backend.
pub trait TypedStore: ObjectStore {
    /// Verify and store an object, returning its content address.
    ///
    /// # Errors
    /// [`StoreError::Core`] if the object's id does not match its content (a
    /// caller bug or in-memory corruption); [`StoreError::Serde`] if
    /// serialization fails.
    fn put_object<B: Body>(&mut self, obj: &Object<B>) -> Result<ObjectId> {
        obj.check_id()?; // never store a mis-addressed object
        let bytes = serde_json::to_vec(obj)?;
        let id = obj.id;
        self.put_raw(id, StoredRecord::new(obj.kind.clone(), bytes));
        Ok(id)
    }

    /// Fetch and verify an object of body type `B` at `id`.
    ///
    /// Returns `Ok(None)` if nothing is stored at `id`.
    ///
    /// # Errors
    /// [`StoreError::KindMismatch`] if a different kind is stored there;
    /// [`StoreError::Serde`] if the stored bytes do not deserialize as `B`;
    /// [`StoreError::Core`] if the stored object fails its content-address check
    /// (tamper/corruption detected on read).
    fn get_object<B: Body>(&self, id: ObjectId) -> Result<Option<Object<B>>> {
        let Some(rec) = self.get_raw(id)
        else
        {
            return Ok(None);
        };
        let requested = B::kind();
        if rec.kind != requested
        {
            return Err(StoreError::KindMismatch {
                requested,
                stored: rec.kind,
            });
        }
        let obj: Object<B> = serde_json::from_slice(&rec.bytes)?;
        obj.check_id()?; // internal consistency: obj.id == hash(content)
        if obj.id != id
        {
            // The bytes stored at `id` address a *different* object — a backend
            // integrity failure (a misfiled record), distinct from internal
            // tampering. Surface it rather than returning the wrong object.
            return Err(StoreError::Core(SosError::IdMismatch {
                stored: id,
                recomputed: obj.id,
            }));
        }
        Ok(Some(obj))
    }
}

impl<S: ObjectStore + ?Sized> TypedStore for S {}

/// The result of a [`gc`] pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcReport {
    /// How many objects were pruned.
    pub removed: usize,
    /// How many objects remain after the pass.
    pub retained: usize,
}

/// The set of object ids reachable from the store's named refs by walking
/// provenance parents (ancestors).
///
/// Reachability follows `parents`, so a ref pointing at a "tip" object retains
/// everything that object was derived from. The traversal reads only the
/// generic [`ObjectHeader`], so it works without knowing any body type.
///
/// # Errors
/// [`StoreError::Serde`] if a stored object's bytes are not a valid object
/// header (i.e. were not written via [`TypedStore::put_object`]).
pub fn reachable<S: ObjectStore + ?Sized>(store: &S) -> Result<HashSet<ObjectId>> {
    let mut seen = HashSet::new();
    let mut stack: Vec<ObjectId> = store.refs().into_iter().map(|r| r.target).collect();
    while let Some(id) = stack.pop()
    {
        if !seen.insert(id)
        {
            continue;
        }
        if let Some(rec) = store.get_raw(id)
        {
            let header: ObjectHeader = serde_json::from_slice(&rec.bytes)?;
            stack.extend(header.parents);
        }
        // A missing id (dangling ref or parent) is simply a leaf of the walk.
    }
    Ok(seen)
}

/// Prune every object **not** reachable from the store's named refs.
///
/// GC is explicit and opt-in by design (RFC-0002 §03.7): a recorded dead-end is
/// scientifically meaningful, so nothing is collected until you call this.
/// Blobs are not swept (their reachability is body-level, a later increment);
/// only objects are.
///
/// # Errors
/// Propagates [`reachable`]'s errors.
pub fn gc<S: ObjectStore + ?Sized>(store: &mut S) -> Result<GcReport> {
    let keep = reachable(store)?;
    let all = store.object_ids();
    let total = all.len();
    let mut removed = 0usize;
    for id in all
    {
        if !keep.contains(&id) && store.remove_raw(id)
        {
            removed += 1;
        }
    }
    Ok(GcReport {
        removed,
        retained: total - removed,
    })
}
