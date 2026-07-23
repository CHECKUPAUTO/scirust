//! Integration tests for [`FileStore`]: the same behavioral contract
//! `tests/store.rs` proves for [`MemoryStore`] (typed round-trip, kind
//! checking, tamper detection, reachability GC) plus the properties unique to
//! a persistent backend — state surviving a close/reopen cycle, and a fresh
//! handle at the same root seeing what another handle wrote.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, HashAlgo, Object, ObjectId};
use sos_store::{FileStore, ObjectStore, StoreError, StoredRecord, TypedStore, gc};

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

/// A fresh, empty temp directory unique to `test_name`, cleaned up before use
/// (in case a prior run left it behind) but **not** after — left for a
/// developer to inspect on failure, and cleaned by the OS's temp-dir policy.
fn temp_root(test_name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "sos-store-filestore-integration-{test_name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    dir
}

#[test]
fn typed_roundtrip_and_dedup() {
    let root = temp_root("roundtrip");
    let mut s = FileStore::open(&root).unwrap();
    let n = note("hello");
    let id = s.put_object(&n).unwrap();
    assert!(s.has(id));

    let back: Object<Note> = s.get_object(id).unwrap().unwrap();
    assert_eq!(back.id, id);
    assert_eq!(back.body.text, "hello");
    assert!(back.verify_id());

    s.put_object(&n).unwrap(); // idempotent
    assert_eq!(s.len(), 1);

    let missing = ObjectId::compute(HashAlgo::default(), b"x", b"y");
    assert!(s.get_object::<Note>(missing).unwrap().is_none());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn wrong_kind_is_rejected() {
    let root = temp_root("wrong-kind");
    let mut s = FileStore::open(&root).unwrap();
    let id = s.put_object(&note("x")).unwrap();
    let err = s.get_object::<Tag>(id).unwrap_err();
    assert!(matches!(err, StoreError::KindMismatch { .. }));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn corruption_in_storage_is_detected_on_read() {
    let root = temp_root("corruption");
    let mut s = FileStore::open(&root).unwrap();
    let n = note("honest");
    let evil = note("tampered");
    let evil_bytes = serde_json::to_vec(&evil).unwrap();
    s.put_raw(n.id, StoredRecord::new(Note::kind(), evil_bytes));

    let err = s.get_object::<Note>(n.id).unwrap_err();
    assert!(matches!(
        err,
        StoreError::Core(sos_core::SosError::IdMismatch { .. })
    ));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn gc_retains_ancestors_of_refs_and_prunes_the_rest() {
    let root = temp_root("gc");
    let mut s = FileStore::open(&root).unwrap();
    let root_obj = note("root");
    s.put_object(&root_obj).unwrap();
    let child = Object::builder(Note {
        text: "child".into(),
    })
    .author(Author::human("t"))
    .parents(vec![root_obj.id])
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

    s.set_ref("tip", grandchild.id);
    let report = gc(&mut s).unwrap();
    assert_eq!(report.removed, 0);
    assert_eq!(report.retained, 3);

    s.remove_ref("tip");
    s.set_ref("root", root_obj.id);
    let report = gc(&mut s).unwrap();
    assert_eq!(report.removed, 2);
    assert_eq!(report.retained, 1);
    assert!(s.has(root_obj.id));
    assert!(!s.has(child.id));
    assert!(!s.has(grandchild.id));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn state_survives_a_close_and_reopen_cycle() {
    let root = temp_root("close-reopen");
    let (id, blob, obj_count);
    {
        let mut s = FileStore::open(&root).unwrap();
        id = s.put_object(&note("persisted")).unwrap();
        blob = s.put_blob(b"payload");
        s.set_ref("head", id);
        obj_count = s.len();
    } // `s` dropped — simulates the process exiting.

    let reopened = FileStore::open(&root).unwrap();
    assert_eq!(reopened.len(), obj_count);
    let back: Object<Note> = reopened.get_object(id).unwrap().unwrap();
    assert_eq!(back.body.text, "persisted");
    assert_eq!(reopened.get_blob(blob).unwrap(), b"payload");
    assert_eq!(reopened.get_ref("head"), Some(id));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn a_second_handle_at_the_same_root_sees_what_the_first_wrote() {
    let root = temp_root("second-handle");
    let mut writer = FileStore::open(&root).unwrap();
    let id = writer.put_object(&note("shared")).unwrap();
    writer.set_ref("main", id);

    let reader = FileStore::open(&root).unwrap();
    assert!(reader.has(id));
    assert_eq!(reader.get_ref("main"), Some(id));

    // And a write through the second handle is visible if the first re-reads.
    let mut second = FileStore::open(&root).unwrap();
    let id2 = second.put_object(&note("also-shared")).unwrap();
    drop(second);
    let refreshed = FileStore::open(&root).unwrap();
    assert!(refreshed.has(id2));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn many_objects_all_retrievable_across_shards() {
    let root = temp_root("many-shards");
    let mut s = FileStore::open(&root).unwrap();
    let mut ids = Vec::new();
    for i in 0..64
    {
        let id = s.put_object(&note(&format!("obj-{i}"))).unwrap();
        ids.push(id);
    }
    assert_eq!(s.len(), 64);
    for id in ids
    {
        let obj: Object<Note> = s.get_object(id).unwrap().unwrap();
        assert!(obj.verify_id());
    }
    fs::remove_dir_all(&root).ok();
}
