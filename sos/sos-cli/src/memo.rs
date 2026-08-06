//! [`FileMemo`] — the memo table `sos run` keeps between invocations.
//!
//! [`MemoTable`](sos_workflow::MemoTable) is complete and correct, but it lives
//! for exactly one call: run the same study twice and every stage recomputes,
//! landing on the same object ids because the backends are `L3`. Not wrong —
//! just the whole point of memoization thrown away between invocations. For a
//! study of any size that is the difference between a usable command and one
//! nobody runs twice.
//!
//! `FileMemo` stores the same `CacheKey → outputs` map as JSON beside the
//! object store, at [`MEMO_FILE`].
//!
//! ## Persisting a cache is not free of consequences
//!
//! An in-memory memo cannot serve a stale answer: it starts empty, so
//! everything it returns was computed by the same process, from the same code.
//! A persisted one *can*, and that hazard is worth stating rather than
//! discovering.
//!
//! A [`CacheKey`] covers the stage descriptor (including the plugin's pinned
//! content hash), the inputs, the config address, the seed and the environment
//! digest. What it does **not** cover is the version of the binary that
//! resolved that plugin name to code. `sos run`'s built-in plugins are pinned
//! to a hash derived from their *name*, which is deliberately stable across
//! releases so a study written last year still names the same plugin — so
//! upgrading `sos` and re-running a study would otherwise return the old
//! object ids from cache, even if the new build computes something different.
//!
//! [`env_digest`] closes that: the binary's own version is folded into the
//! environment digest, so a new release invalidates every cached stage. The
//! same version on two machines still memoizes identically — which is the
//! host-independence the stage handlers record — while two *different* builds
//! deliberately do not share a cache. That is the honest reading of "identical
//! inputs ⇒ cache hit": the code is one of the inputs.
//!
//! ## A corrupt memo is an error, not a silent reset
//!
//! Discarding an unreadable memo and recomputing would be *safe* — a cache
//! loss costs time, never correctness. It is still refused: a memo file that
//! does not parse means something wrote something unexpected next to a
//! scientific record, and quietly deleting the evidence is the wrong default.
//! Deleting it is one command; noticing that it was corrupt is not recoverable
//! once the tool has papered over it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sos_core::{Digest, HashAlgo, ObjectId};
use sos_workflow::{CacheKey, Memo};

use crate::error::Result;

/// The memo file's name inside a store root.
pub const MEMO_FILE: &str = "memo.json";

/// The default environment label when `--env` is not given.
///
/// Deliberately not a hostname or a toolchain string: a study should memoize
/// identically on every machine unless the caller asks otherwise. See
/// [`env_digest`] for what *is* folded in regardless.
pub const DEFAULT_ENV_LABEL: &str = "unspecified";

/// The environment digest a run is keyed by: the caller's label **and this
/// binary's version**.
///
/// See the module docs — the version is what stops an upgraded `sos` from
/// serving cached results computed by the previous one.
#[must_use]
pub fn env_digest(label: &str) -> Digest {
    let mut enc = sos_core::canonical::CanonicalEncoder::new();
    enc.str(label);
    enc.str(env!("CARGO_PKG_VERSION"));
    HashAlgo::Sha256.hash(b"sos-cli:env", &enc.finish())
}

/// A [`Memo`] backed by a JSON file beside the object store.
#[derive(Debug)]
pub struct FileMemo {
    path: PathBuf,
    map: BTreeMap<CacheKey, Vec<ObjectId>>,
    dirty: bool,
}

impl FileMemo {
    /// Open the memo for the store rooted at `root`, reading any existing
    /// entries.
    ///
    /// # Errors
    /// [`crate::error::CliError::Io`] if the file exists but cannot be read;
    /// [`crate::error::CliError::Serde`] if it exists and does not parse — see
    /// the module docs on why that is refused rather than reset.
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join(MEMO_FILE);
        let map = if path.is_file()
        {
            serde_json::from_slice(&std::fs::read(&path)?)?
        }
        else
        {
            BTreeMap::new()
        };
        Ok(Self {
            path,
            map,
            dirty: false,
        })
    }

    /// Write the memo back, if anything changed.
    ///
    /// # Errors
    /// [`crate::error::CliError::Io`] if the file cannot be written.
    pub fn flush(&mut self) -> Result<()> {
        if !self.dirty
        {
            return Ok(());
        }
        if let Some(parent) = self.path.parent()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, serde_json::to_vec_pretty(&self.map)?)?;
        self.dirty = false;
        Ok(())
    }

    /// How many stage invocations are remembered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether nothing is remembered yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl Memo for FileMemo {
    fn get(&self, key: &CacheKey) -> Option<Vec<ObjectId>> {
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: CacheKey, outputs: Vec<ObjectId>) {
        self.map.insert(key, outputs);
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("sos-memo-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn key(tag: &[u8]) -> CacheKey {
        // A CacheKey is a Digest newtype; round-tripping one through JSON is
        // the only way to build an arbitrary one from outside the crate.
        let digest = HashAlgo::Sha256.hash(b"memo-test", tag);
        serde_json::from_value(serde_json::json!(digest.to_hex())).unwrap()
    }

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::Sha256, b"memo-test", tag)
    }

    #[test]
    fn an_absent_memo_opens_empty() {
        let root = temp_root("absent");
        let memo = FileMemo::open(&root).unwrap();
        assert!(memo.is_empty());
        assert!(memo.get(&key(b"anything")).is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn entries_survive_a_flush_and_reopen() {
        let root = temp_root("roundtrip");
        {
            let mut memo = FileMemo::open(&root).unwrap();
            memo.put(key(b"a"), vec![oid(b"out-a")]);
            memo.put(key(b"b"), vec![oid(b"out-b1"), oid(b"out-b2")]);
            memo.flush().unwrap();
        }
        let memo = FileMemo::open(&root).unwrap();
        assert_eq!(memo.len(), 2);
        assert_eq!(memo.get(&key(b"a")), Some(vec![oid(b"out-a")]));
        assert_eq!(
            memo.get(&key(b"b")),
            Some(vec![oid(b"out-b1"), oid(b"out-b2")])
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_flush_with_nothing_new_writes_no_file() {
        // Opening a store read-only must not create a memo out of nowhere.
        let root = temp_root("clean");
        FileMemo::open(&root).unwrap().flush().unwrap();
        assert!(!root.join(MEMO_FILE).exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_corrupt_memo_is_an_error_not_a_silent_reset() {
        let root = temp_root("corrupt");
        std::fs::write(root.join(MEMO_FILE), b"{not json").unwrap();
        assert!(FileMemo::open(&root).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_environment_digest_binds_the_binary_version() {
        // The hazard persisting a cache introduces: without the version in the
        // key, an upgraded `sos` would serve results the previous build
        // computed. Checked by construction — the digest is not the bare hash
        // of the label.
        let bare = HashAlgo::Sha256.hash(b"sos-cli:env", DEFAULT_ENV_LABEL.as_bytes());
        assert_ne!(env_digest(DEFAULT_ENV_LABEL), bare);

        // And it still separates labels, so `--env` keeps working.
        assert_ne!(env_digest(DEFAULT_ENV_LABEL), env_digest("lab-3"));
        assert_eq!(env_digest("lab-3"), env_digest("lab-3"));
    }
}
