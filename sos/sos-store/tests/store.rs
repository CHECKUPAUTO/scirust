//! Integration tests for the storage layer: typed round-trips, kind checks,
//! tamper detection on read, reachability GC, and a seeded property test.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};
use sos_store::{MemoryStore, ObjectStore, StoreError, StoredRecord, TypedStore, gc};

// --- two distinct body types, to exercise kind checking ---

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Note {
    text: String,
}
impl Canonical for Note {
    fn encode(&self, e: &mut CanonicalEncoder) {
        e.str(&self.text);
    }
}
impl Body for Note {
    const KIND: &'static str = "Note";
    const SCHEMA_VERSION: u32 = 1;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Tag {
    n: u64,
}
impl Canonical for Tag {
    fn encode(&self, e: &mut CanonicalEncoder) {
        e.u64(self.n);
    }
}
impl Body for Tag {
    const KIND: &'static str = "Tag";
    const SCHEMA_VERSION: u32 = 1;
}

fn note(text: &str) -> Object<Note> {
    Object::builder(Note { text: text.into() })
        .author(Author::human("t"))
        .seal()
}

#[test]
fn typed_roundtrip_and_dedup() {
    let mut s = MemoryStore::new();
    let n = note("hello");
    let id = s.put_object(&n).unwrap();
    assert!(s.has(id));

    let back: Object<Note> = s.get_object(id).unwrap().unwrap();
    assert_eq!(back.id, id);
    assert_eq!(back.body.text, "hello");
    assert!(back.verify_id());

    // Idempotent: re-put is a no-op.
    s.put_object(&n).unwrap();
    assert_eq!(s.len(), 1);

    // Absent id yields Ok(None).
    let missing = ObjectId::compute(sos_core::HashAlgo::default(), b"x", b"y");
    assert!(s.get_object::<Note>(missing).unwrap().is_none());
}

#[test]
fn wrong_kind_is_rejected() {
    let mut s = MemoryStore::new();
    let id = s.put_object(&note("x")).unwrap();
    // Ask for the wrong body type at that id.
    let err = s.get_object::<Tag>(id).unwrap_err();
    assert!(matches!(err, StoreError::KindMismatch { .. }));
}

#[test]
fn corruption_in_storage_is_detected_on_read() {
    let mut s = MemoryStore::new();
    let n = note("honest");
    // Tamper: serialize a DIFFERENT object's bytes under n's id.
    let evil = note("tampered");
    let evil_bytes = serde_json::to_vec(&evil).unwrap();
    s.put_raw(n.id, StoredRecord::new(Note::kind(), evil_bytes));

    // The stored bytes hash to evil.id, not n.id → id mismatch on read.
    let err = s.get_object::<Note>(n.id).unwrap_err();
    assert!(matches!(
        err,
        StoreError::Core(sos_core::SosError::IdMismatch { .. })
    ));
}

#[test]
fn gc_retains_ancestors_of_refs_and_prunes_the_rest() {
    let mut s = MemoryStore::new();
    // root <- child <- grandchild  (parents point at ancestors)
    let root = note("root");
    s.put_object(&root).unwrap();
    let child = Object::builder(Note {
        text: "child".into(),
    })
    .author(Author::human("t"))
    .parents(vec![root.id])
    .seal();
    s.put_object(&child).unwrap();
    let grandchild = Object::builder(Note {
        text: "grandchild".into(),
    })
    .author(Author::human("t"))
    .parents(vec![child.id])
    .seal();
    s.put_object(&grandchild).unwrap();
    assert_eq!(s.len(), 3);

    // A ref at the tip keeps the whole chain.
    s.set_ref("tip", grandchild.id);
    let report = gc(&mut s).unwrap();
    assert_eq!(report.removed, 0);
    assert_eq!(report.retained, 3);

    // Move the ref to the root: child and grandchild are DESCENDANTS, not
    // ancestors, so they become unreachable and are pruned.
    s.remove_ref("tip");
    s.set_ref("root", root.id);
    let report = gc(&mut s).unwrap();
    assert_eq!(report.removed, 2);
    assert_eq!(report.retained, 1);
    assert!(s.has(root.id));
    assert!(!s.has(child.id));
    assert!(!s.has(grandchild.id));
}

#[test]
fn gc_with_no_refs_prunes_everything() {
    let mut s = MemoryStore::new();
    for i in 0..5
    {
        s.put_object(&note(&format!("n{i}"))).unwrap();
    }
    let report = gc(&mut s).unwrap();
    assert_eq!(report.removed, 5);
    assert_eq!(report.retained, 0);
    assert!(s.is_empty());
}

// --- seeded property test: many objects, all retrievable & verified ---

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[test]
fn many_objects_all_retrievable_and_a_blob_each() {
    let mut s = MemoryStore::new();
    let mut rng = SplitMix64(0x5EED_1234_5678_9ABC);
    let mut ids = Vec::new();
    for _ in 0..400
    {
        let n = note(&format!("obj-{}", rng.next_u64()));
        let id = s.put_object(&n).unwrap();
        ids.push(id);
        let r = s.put_blob(format!("blob-{}", rng.next_u64()).as_bytes());
        assert!(s.has_blob(r));
    }
    // Every stored id round-trips and verifies.
    for id in ids
    {
        let obj: Object<Note> = s.get_object(id).unwrap().unwrap();
        assert!(obj.verify_id());
    }
}
