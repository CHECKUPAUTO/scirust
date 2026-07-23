//! [`FileStore`] — a persistent, filesystem-backed [`ObjectStore`].
//!
//! A real on-disk backend, laid out git-style: objects and blobs are files
//! named by their content-address hex digest under a two-level shard directory
//! (`objects/ab/cdef01…`), so a store with many entries never puts too many
//! files in one directory. Refs are a single small JSON index
//! (`refs.json`) — deliberately **not** one file per ref name, because
//! [`ObjectStore::set_ref`] takes an arbitrary caller-supplied name and the
//! trait gives it no way to reject one; turning that string directly into a
//! filesystem path would open a path-traversal hole (a name like
//! `"../../etc/x"`). Keeping refs in one content-addressed-free index avoids
//! the problem entirely rather than trying to sanitize it away.
//!
//! ## Durability
//!
//! Every write goes to a `.tmp` sibling file first, then is renamed into place,
//! so a crash mid-write never leaves a half-written object or a corrupted
//! `refs.json` — the reader either sees the old file or the new one, never a
//! mix. Content-addressed files (objects, blobs) are also **first-wins**: if
//! the target path already exists, the write is skipped entirely, matching
//! [`ObjectStore`]'s idempotent-put contract and sidestepping the one case
//! where a cross-platform atomic rename is awkward (Windows refuses to rename
//! over an existing file; this backend only needs that fallback for
//! `refs.json`, which is handled explicitly).
//!
//! ## Error handling
//!
//! [`ObjectStore`]'s methods are **infallible by design** (no `Result`), a
//! shape that predates this backend and is depended on by every crate that
//! already takes `impl ObjectStore` or `dyn ObjectStore`. Changing it would
//! ripple a breaking change through all of them. So this backend draws the
//! same line [`crate::TypedStore`] draws for content: **absence is `None`;
//! corruption or a genuine I/O fault after the store was successfully opened
//! is an abort**, with a clear panic message naming the path and the
//! underlying error — never silently swallowed, never returned as if the data
//! were simply missing. The one place a real, recoverable setup error
//! (permissions, a path that cannot be created) is expected is
//! [`FileStore::open`], which is fallible.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sos_core::ObjectId;
use sos_core::kind::Kind;

use crate::blob::BlobRef;
use crate::error::Result;
use crate::record::{NamedRef, StoredRecord};
use crate::store::ObjectStore;

const OBJECTS_DIR: &str = "objects";
const BLOBS_DIR: &str = "blobs";
const REFS_FILE: &str = "refs.json";
/// Hex characters used as the shard (sub-directory) prefix of a content file's
/// path — 2 hex chars = 256 buckets, the same fan-out Git uses for loose objects.
const SHARD_LEN: usize = 2;

/// Only the field this backend needs to recover a stored object's [`Kind`]
/// without knowing its body type — the same trick [`crate::record::ObjectHeader`]
/// uses for `id`/`parents`.
#[derive(Deserialize)]
struct KindHeader {
    kind: Kind,
}

/// Panic with a clear, actionable message naming the path and the I/O fault.
/// Reserved for conditions that mean the storage medium itself is broken
/// (permission revoked after opening, disk failure, a byte-for-byte corrupted
/// file at a content-addressed path) — never for ordinary absence, which is
/// handled by callers before this is reached.
fn expect_io<T>(result: std::io::Result<T>, path: &Path, action: &str) -> T {
    result.unwrap_or_else(|e| {
        panic!(
            "sos-store: FileStore failed to {action} `{}`: {e}",
            path.display()
        )
    })
}

/// Write `bytes` to `path` atomically: write a `.tmp` sibling, then rename it
/// into place. Creates parent directories as needed. If a plain rename fails
/// because `path` already exists (Windows does not allow renaming over an
/// existing file, unlike POSIX), the destination is removed and the rename is
/// retried — still race-free within a single process, the only concurrency
/// this backend claims to support.
fn write_atomic(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent()
    {
        expect_io(fs::create_dir_all(parent), parent, "create directory");
    }
    let tmp = tmp_sibling(path);
    expect_io(fs::write(&tmp, bytes), &tmp, "write");
    if fs::rename(&tmp, path).is_err()
    {
        let _ = fs::remove_file(path);
        expect_io(fs::rename(&tmp, path), path, "rename into place");
    }
}

/// `path` with `.tmp` appended to its filename (robust even when `path` already
/// has a dot in it, e.g. `refs.json` → `refs.json.tmp`, unlike
/// [`Path::with_extension`] which would replace the existing extension).
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

/// The two-level shard path for a 64-hex-character digest:
/// `<base>/<first 2 hex chars>/<remaining 62 hex chars>`.
fn shard_path(base: &Path, hex: &str) -> PathBuf {
    let (shard, rest) = hex.split_at(SHARD_LEN.min(hex.len()));
    base.join(shard).join(rest)
}

/// Parse a shard directory + filename back into the original hex digest.
fn unshard(shard: &str, file_name: &str) -> Option<String> {
    if shard.len() == SHARD_LEN
    {
        Some(format!("{shard}{file_name}"))
    }
    else
    {
        None
    }
}

/// List every content file under a two-level sharded directory tree, returning
/// their reconstructed hex digests. An absent `base` directory (an empty store)
/// yields an empty list rather than an error.
fn list_sharded(base: &Path) -> Vec<String> {
    let Ok(shards) = fs::read_dir(base)
    else
    {
        return Vec::new();
    };
    let mut hexes = Vec::new();
    for shard_entry in shards.filter_map(std::result::Result::ok)
    {
        let shard_path = shard_entry.path();
        if !shard_path.is_dir()
        {
            continue;
        }
        let Some(shard_name) = shard_entry.file_name().to_str().map(str::to_owned)
        else
        {
            continue;
        };
        let Ok(files) = fs::read_dir(&shard_path)
        else
        {
            continue;
        };
        for file_entry in files.filter_map(std::result::Result::ok)
        {
            let Some(file_name) = file_entry.file_name().to_str().map(str::to_owned)
            else
            {
                continue;
            };
            if file_name.ends_with(".tmp")
            {
                continue; // a write in progress or left over from a crash
            }
            if let Some(hex) = unshard(&shard_name, &file_name)
            {
                hexes.push(hex);
            }
        }
    }
    hexes
}

/// A persistent, filesystem-backed content-addressed object store.
///
/// See the [module docs](self) for the on-disk layout, durability, and error
/// handling this backend commits to.
#[derive(Debug, Clone)]
pub struct FileStore {
    root: PathBuf,
    refs: BTreeMap<String, ObjectId>,
}

impl FileStore {
    /// Open (or create) a store rooted at `root`.
    ///
    /// Creates `root`, `root/objects`, and `root/blobs` if they do not already
    /// exist, and loads `root/refs.json` if present (an absent refs file is
    /// treated as an empty ref set, not an error — a freshly created store has
    /// no refs yet).
    ///
    /// # Errors
    /// [`StoreError::Io`] if `root` cannot be created or read, or if an
    /// existing `refs.json` cannot be read or does not parse.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join(OBJECTS_DIR))?;
        fs::create_dir_all(root.join(BLOBS_DIR))?;
        let refs_path = root.join(REFS_FILE);
        let refs = if refs_path.exists()
        {
            let bytes = fs::read(&refs_path)?;
            serde_json::from_slice(&bytes)?
        }
        else
        {
            BTreeMap::new()
        };
        Ok(Self { root, refs })
    }

    /// The store's root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The number of stored objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.object_ids().len()
    }

    /// Whether the store holds no objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of stored blobs.
    #[must_use]
    pub fn blob_count(&self) -> usize {
        list_sharded(&self.root.join(BLOBS_DIR)).len()
    }

    fn object_path(&self, id: ObjectId) -> PathBuf {
        shard_path(&self.root.join(OBJECTS_DIR), &id.digest().to_hex())
    }

    fn blob_path(&self, r: BlobRef) -> PathBuf {
        shard_path(&self.root.join(BLOBS_DIR), &r.digest().to_hex())
    }

    fn write_refs(&self) {
        let path = self.root.join(REFS_FILE);
        let bytes = serde_json::to_vec(&self.refs)
            .unwrap_or_else(|e| panic!("sos-store: FileStore failed to serialize refs: {e}"));
        write_atomic(&path, &bytes);
    }
}

impl ObjectStore for FileStore {
    fn put_raw(&mut self, id: ObjectId, record: StoredRecord) {
        let path = self.object_path(id);
        if !path.exists()
        {
            // First-wins: content addressing guarantees identical content for
            // the same id, so an existing file is already authoritative.
            write_atomic(&path, &record.bytes);
        }
    }

    fn get_raw(&self, id: ObjectId) -> Option<StoredRecord> {
        let path = self.object_path(id);
        if !path.exists()
        {
            return None;
        }
        let bytes = expect_io(fs::read(&path), &path, "read");
        let header: KindHeader = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            panic!(
                "sos-store: FileStore found a corrupted object at `{}`: {e}",
                path.display()
            )
        });
        Some(StoredRecord::new(header.kind, bytes))
    }

    fn has(&self, id: ObjectId) -> bool {
        self.object_path(id).exists()
    }

    fn object_ids(&self) -> Vec<ObjectId> {
        let mut ids: Vec<ObjectId> = list_sharded(&self.root.join(OBJECTS_DIR))
            .iter()
            .filter_map(|hex| sos_core::Digest::from_hex(hex).ok())
            .map(ObjectId::from_digest)
            .collect();
        ids.sort();
        ids
    }

    fn remove_raw(&mut self, id: ObjectId) -> bool {
        let path = self.object_path(id);
        if path.exists()
        {
            expect_io(fs::remove_file(&path), &path, "remove");
            true
        }
        else
        {
            false
        }
    }

    fn put_blob(&mut self, bytes: &[u8]) -> BlobRef {
        let r = BlobRef::of(bytes);
        let path = self.blob_path(r);
        if !path.exists()
        {
            write_atomic(&path, bytes);
        }
        r
    }

    fn get_blob(&self, r: BlobRef) -> Option<Vec<u8>> {
        let path = self.blob_path(r);
        if path.exists()
        {
            Some(expect_io(fs::read(&path), &path, "read"))
        }
        else
        {
            None
        }
    }

    fn has_blob(&self, r: BlobRef) -> bool {
        self.blob_path(r).exists()
    }

    fn set_ref(&mut self, name: &str, target: ObjectId) {
        self.refs.insert(name.to_owned(), target);
        self.write_refs();
    }

    fn get_ref(&self, name: &str) -> Option<ObjectId> {
        self.refs.get(name).copied()
    }

    fn remove_ref(&mut self, name: &str) -> bool {
        let removed = self.refs.remove(name).is_some();
        if removed
        {
            self.write_refs();
        }
        removed
    }

    fn refs(&self) -> Vec<NamedRef> {
        // `self.refs` is a `BTreeMap`, so iteration is already sorted by name.
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

    fn temp_root(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "sos-store-filestore-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn opening_creates_the_directory_layout() {
        let root = temp_root("layout");
        let store = FileStore::open(&root).unwrap();
        assert!(root.join(OBJECTS_DIR).is_dir());
        assert!(root.join(BLOBS_DIR).is_dir());
        assert!(store.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn puts_are_idempotent_and_first_wins() {
        let root = temp_root("idempotent");
        let mut store = FileStore::open(&root).unwrap();
        let id = oid(b"x");
        let rec = StoredRecord::new(
            Kind::new("T", 1),
            br#"{"kind":{"name":"T","schema_version":1}}"#.to_vec(),
        );
        store.put_raw(id, rec.clone());
        // A second put at the same id is a no-op even with different bytes —
        // first-wins, matching MemoryStore's contract.
        store.put_raw(
            id,
            StoredRecord::new(Kind::new("T", 1), b"garbage".to_vec()),
        );
        assert_eq!(store.get_raw(id).unwrap().bytes, rec.bytes);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn blobs_round_trip_and_are_content_addressed() {
        let root = temp_root("blobs");
        let mut store = FileStore::open(&root).unwrap();
        let a = store.put_blob(b"hello");
        let b = store.put_blob(b"hello");
        assert_eq!(a, b);
        assert_eq!(store.blob_count(), 1);
        assert_eq!(store.get_blob(a).unwrap(), b"hello");
        assert!(store.has_blob(a));
        assert!(store.get_blob(BlobRef::of(b"absent")).is_none());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn refs_persist_and_stay_sorted() {
        let root = temp_root("refs");
        let mut store = FileStore::open(&root).unwrap();
        store.set_ref("b/two", oid(b"2"));
        store.set_ref("a/one", oid(b"1"));
        let names: Vec<String> = store.refs().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["a/one", "b/two"]);
        assert!(store.remove_ref("a/one"));
        assert!(!store.remove_ref("a/one"));
        assert_eq!(store.get_ref("a/one"), None);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn state_survives_reopening_the_same_root() {
        let root = temp_root("reopen");
        {
            let mut store = FileStore::open(&root).unwrap();
            store.put_raw(
                oid(b"x"),
                StoredRecord::new(
                    Kind::new("T", 1),
                    br#"{"kind":{"name":"T","schema_version":1}}"#.to_vec(),
                ),
            );
            store.put_blob(b"payload");
            store.set_ref("head", oid(b"x"));
        }
        // Fresh handle, same directory: everything is still there.
        let reopened = FileStore::open(&root).unwrap();
        assert!(reopened.has(oid(b"x")));
        assert!(reopened.has_blob(BlobRef::of(b"payload")));
        assert_eq!(reopened.get_ref("head"), Some(oid(b"x")));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn object_ids_and_removal_work_like_memory_store() {
        let root = temp_root("ids");
        let mut store = FileStore::open(&root).unwrap();
        for tag in [b"3".as_slice(), b"1", b"2"]
        {
            store.put_raw(
                oid(tag),
                StoredRecord::new(
                    Kind::new("T", 1),
                    br#"{"kind":{"name":"T","schema_version":1}}"#.to_vec(),
                ),
            );
        }
        let ids = store.object_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
        assert_eq!(ids.len(), 3);
        assert!(store.remove_raw(oid(b"1")));
        assert!(!store.remove_raw(oid(b"1")));
        assert_eq!(store.object_ids().len(), 2);
        fs::remove_dir_all(&root).ok();
    }
}
