//! [`MemoryStore`] — a complete in-memory [`ObjectStore`] backend.
//!
//! This is a real, deterministic store (not a mock): a `HashMap`-backed object
//! and blob store with `BTreeMap` refs for sorted iteration. It is ideal for
//! tests, ephemeral sessions, and embedding. Persistent local and
//! object-storage backends implement the same [`ObjectStore`] trait.

use std::collections::{BTreeMap, HashMap};

use sos_core::ObjectId;

use crate::blob::BlobRef;
use crate::record::{NamedRef, StoredRecord};
use crate::store::ObjectStore;

/// An in-memory content-addressed object store.
#[derive(Debug, Default, Clone)]
pub struct MemoryStore {
    objects: HashMap<ObjectId, StoredRecord>,
    blobs: HashMap<BlobRef, Vec<u8>>,
    refs: BTreeMap<String, ObjectId>,
}

impl MemoryStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of stored objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the store holds no objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// The number of stored blobs.
    #[must_use]
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }
}

impl ObjectStore for MemoryStore {
    fn put_raw(&mut self, id: ObjectId, record: StoredRecord) {
        // First-wins: content addressing guarantees identical identity content,
        // so an existing entry is authoritative.
        self.objects.entry(id).or_insert(record);
    }

    fn get_raw(&self, id: ObjectId) -> Option<StoredRecord> {
        self.objects.get(&id).cloned()
    }

    fn has(&self, id: ObjectId) -> bool {
        self.objects.contains_key(&id)
    }

    fn object_ids(&self) -> Vec<ObjectId> {
        let mut ids: Vec<ObjectId> = self.objects.keys().copied().collect();
        ids.sort();
        ids
    }

    fn remove_raw(&mut self, id: ObjectId) -> bool {
        self.objects.remove(&id).is_some()
    }

    fn put_blob(&mut self, bytes: &[u8]) -> BlobRef {
        let r = BlobRef::of(bytes);
        self.blobs.entry(r).or_insert_with(|| bytes.to_vec());
        r
    }

    fn get_blob(&self, r: BlobRef) -> Option<Vec<u8>> {
        self.blobs.get(&r).cloned()
    }

    fn has_blob(&self, r: BlobRef) -> bool {
        self.blobs.contains_key(&r)
    }

    fn set_ref(&mut self, name: &str, target: ObjectId) {
        self.refs.insert(name.to_string(), target);
    }

    fn get_ref(&self, name: &str) -> Option<ObjectId> {
        self.refs.get(name).copied()
    }

    fn remove_ref(&mut self, name: &str) -> bool {
        self.refs.remove(name).is_some()
    }

    fn refs(&self) -> Vec<NamedRef> {
        // BTreeMap iterates in sorted key order — deterministic by construction.
        self.refs
            .iter()
            .map(|(name, target)| NamedRef::new(name.clone(), *target))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:T:v1", tag)
    }

    #[test]
    fn blobs_are_idempotent_and_content_addressed() {
        let mut s = MemoryStore::new();
        let a = s.put_blob(b"hello");
        let b = s.put_blob(b"hello");
        assert_eq!(a, b);
        assert_eq!(s.blob_count(), 1);
        assert_eq!(s.get_blob(a).unwrap(), b"hello");
        assert!(s.has_blob(a));
        assert!(s.get_blob(BlobRef::of(b"absent")).is_none());
    }

    #[test]
    fn refs_are_sorted_and_mutable() {
        let mut s = MemoryStore::new();
        s.set_ref("b/two", oid(b"2"));
        s.set_ref("a/one", oid(b"1"));
        let names: Vec<String> = s.refs().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["a/one", "b/two"]); // sorted
        assert_eq!(s.get_ref("a/one"), Some(oid(b"1")));
        assert!(s.remove_ref("a/one"));
        assert!(!s.remove_ref("a/one"));
        assert_eq!(s.get_ref("a/one"), None);
    }

    #[test]
    fn object_ids_are_sorted() {
        let mut s = MemoryStore::new();
        for tag in [b"3".as_slice(), b"1", b"2"]
        {
            s.put_raw(
                oid(tag),
                StoredRecord::new(sos_core::Kind::new("T", 1), vec![]),
            );
        }
        let ids = s.object_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
        assert_eq!(ids.len(), 3);
    }
}
