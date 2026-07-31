//! Small shared utilities used across the kernel.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Hex-encoded SHA-256 of a string — the canonical content hash used
/// throughout CCOS (file hashes, prompt/response hashes, chain links).
pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Raw 32-byte SHA-256 of a string. The compact form of [`sha256_hex`] — half the
/// bytes, no heap allocation — used as the in-RAM key of a spilled COLD blob (the
/// on-disk filename is still its [`hex32`]).
pub fn sha256_bytes(input: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hasher.finalize().into()
}

/// Lowercase-hex of a 32-byte hash — the on-disk key / wire form of a content hash.
pub fn hex32(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Parse a 64-char lowercase-hex string back to a 32-byte hash; `None` unless it is
/// exactly 64 valid hex digits.
pub fn from_hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[2 * i] as char).to_digit(16)?;
        let lo = (bytes[2 * i + 1] as char).to_digit(16)?;
        *slot = (hi * 16 + lo) as u8;
    }
    Some(out)
}

/// Write `bytes` to `path` **durably and atomically**: write to a temporary
/// sibling, `fsync` it, rename it over `path`, then best-effort `fsync` the
/// parent directory. After this returns the data has reached stable storage and
/// `path` is never left half-written — the basis of CCOS's "replayable after a
/// crash" guarantee. A plain [`std::fs::write`] only reaches the kernel page
/// cache, so a power loss or daemon crash can corrupt or truncate the file. The
/// extra cost is one `fsync`, negligible at an agent's inference cadence.
pub fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    // Ensure the target directory exists — a workspace path like `.ccos/ws.ccos`
    // (an editor's default) must not fail to persist just because `.ccos/` was
    // never created. Without this the checkpoint silently fails and every run is
    // cold, defeating the whole `--workspace` O(Δ) freshness.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    // Arm the cleanup guard only once `create_new` has *succeeded*, which proves
    // this call created the file: the guard can then never unlink a temp file
    // belonging to anyone else.
    let (mut file, mut tmp) = create_temp_sibling(path)?;
    #[cfg(unix)]
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(bytes)?;
    file.sync_all()?; // flush contents + metadata to disk before we rename
    drop(file);

    std::fs::rename(&tmp.path, path)?; // atomic replace on a POSIX filesystem
    tmp.keep(); // renamed away: there is nothing left to unlink

    // Make the rename itself durable by fsync-ing the directory entry. Opening a
    // directory for fsync is not portable everywhere, so this is best-effort.
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// A temp file that unlinks itself unless [`keep`](TempSibling::keep) is called.
///
/// Every step between creating the temp file and renaming it can fail, and each
/// `?` returns early. Without this guard the half-written file stayed on disk,
/// and because the name was only `<path>.tmp.<pid>` the *next* call in the same
/// process hit `create_new` on an existing file and failed with `AlreadyExists`
/// — for the life of the process. One transient ENOSPC or EIO thus latched into
/// a permanent outage that only a restart cleared. `Drop` also covers unwinding,
/// which an explicit cleanup call on the error paths would not.
struct TempSibling {
    path: std::path::PathBuf,
    /// Cleared once the file has been renamed into place and must not be removed.
    armed: bool,
}

impl TempSibling {
    /// Give up ownership: the file is now the real one, not debris.
    fn keep(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempSibling {
    fn drop(&mut self) {
        if self.armed {
            // Best-effort: a failure here leaves exactly the debris we had
            // before, and a unique name per attempt keeps it from latching.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Create `<path>.tmp.<pid>.<n>` exclusively, where `n` counts attempts within
/// this process.
///
/// The name is unique **per attempt** rather than per process, so debris from any
/// cause — a `SIGKILL` between create and rename, a full disk, a build of CCOS
/// that predates the guard above (whose temps were `<path>.tmp.<pid>`, a name
/// this scheme never reuses) — cannot make the next attempt fail. `AlreadyExists`
/// is retried with a fresh number for the same reason: a recycled pid meeting its
/// own leftovers must cost at most a retry, never a save.
///
/// The counter never reaches persisted state — the temp file is renamed or
/// unlinked, never read — so replay determinism is untouched.
fn create_temp_sibling(path: &Path) -> io::Result<(File, TempSibling)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ATTEMPT: AtomicU64 = AtomicU64::new(0);

    let mut last = None;
    for _ in 0..16 {
        let n = ATTEMPT.fetch_add(1, Ordering::Relaxed);
        let mut name = path.as_os_str().to_os_string();
        name.push(format!(".tmp.{}.{n}", std::process::id()));
        let candidate = std::path::PathBuf::from(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(f) => {
                return Ok((
                    f,
                    TempSibling {
                        path: candidate,
                        armed: true,
                    },
                ))
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "no free temporary name next to the target",
        )
    }))
}

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable_and_distinct() {
        assert_eq!(sha256_hex("hello"), sha256_hex("hello"));
        assert_ne!(sha256_hex("hello"), sha256_hex("world"));
        // Known vector for "abc".
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_bytes_matches_hex_and_round_trips() {
        // The raw form is exactly the hex form, just un-encoded.
        assert_eq!(hex32(&sha256_bytes("abc")), sha256_hex("abc"));
        for s in ["", "hello", "abc", "the quick brown fox"] {
            let raw = sha256_bytes(s);
            assert_eq!(
                from_hex32(&hex32(&raw)),
                Some(raw),
                "hex round-trip for {s:?}"
            );
        }
        // Malformed hex is rejected, not silently truncated.
        assert_eq!(from_hex32("nothex"), None);
        assert_eq!(from_hex32(&"a".repeat(63)), None);
    }

    #[test]
    fn write_durable_writes_and_replaces_atomically() {
        let path = std::env::temp_dir().join(format!("ccos-durable-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        write_durable(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        // Overwriting replaces the whole file (no leftover temp sibling).
        write_durable(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        let mut tmp = path.clone().into_os_string();
        tmp.push(format!(".tmp.{}", std::process::id()));
        assert!(
            !std::path::Path::new(&tmp).exists(),
            "temp sibling is renamed away, not left behind"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Names of every `.tmp` sibling sitting in `dir`.
    #[cfg(test)]
    fn temp_debris(dir: &std::path::Path) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut found: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp."))
            .collect();
        found.sort();
        found
    }

    /// A failed save must not poison the next one.
    ///
    /// Every step after the temp file is created can fail, and each `?` used to
    /// return early leaving `<path>.tmp.<pid>` behind. Since that name depended
    /// only on the pid, the next call hit `create_new` on an existing file and
    /// failed with `AlreadyExists` — and so did every call after it, for the life
    /// of the process. A single transient I/O error became a permanent outage
    /// that only a restart cleared: `ccos-license-server` answering 500 to every
    /// sale while `/healthz` stayed green.
    #[test]
    fn a_failed_write_leaves_no_debris_and_does_not_latch() {
        let dir = std::env::temp_dir().join(format!("ccos-durable-latch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ccos");

        // Fail *after* the temp file exists: renaming onto a directory cannot work.
        std::fs::create_dir(&path).unwrap();
        let failed = write_durable(&path, b"first attempt");
        assert!(failed.is_err(), "renaming onto a directory must fail");
        assert_eq!(
            temp_debris(&dir),
            Vec::<String>::new(),
            "the temp sibling must not outlive the failed attempt"
        );

        // With the obstruction gone a healthy save must succeed. Before the fix
        // this returned AlreadyExists, having tripped over its own debris.
        std::fs::remove_dir(&path).unwrap();
        write_durable(&path, b"second attempt").expect("the failure must not latch");
        assert_eq!(std::fs::read(&path).unwrap(), b"second attempt");
        assert_eq!(temp_debris(&dir), Vec::<String>::new());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Debris from a cause `Drop` cannot cover — a `SIGKILL` between create and
    /// rename, or a build predating the guard, whose temps were `<path>.tmp.<pid>`
    /// — must not block a later save either.
    #[test]
    fn pre_existing_debris_does_not_block_a_save() {
        let dir = std::env::temp_dir().join(format!("ccos-durable-debris-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ccos");

        // Exactly what the old code left behind, plus what the new naming would use.
        for suffix in [
            format!(".tmp.{}", std::process::id()),
            format!(".tmp.{}.0", std::process::id()),
        ] {
            let mut name = path.clone().into_os_string();
            name.push(suffix);
            std::fs::write(std::path::PathBuf::from(name), b"orphan").unwrap();
        }

        write_durable(&path, b"payload").expect("stale debris must not block a save");
        assert_eq!(std::fs::read(&path).unwrap(), b"payload");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_durable_creates_a_missing_parent_dir() {
        // An editor's default workspace path (`.ccos/ws.ccos`) must persist even
        // when its directory does not exist yet.
        let dir = std::env::temp_dir().join(format!("ccos-mkdir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("ws.ccos");
        write_durable(&path, b"ok").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"ok");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
