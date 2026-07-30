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
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        #[cfg(unix)]
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        f.write_all(bytes)?;
        f.sync_all()?; // flush contents + metadata to disk before we rename
    }
    std::fs::rename(&tmp, path)?; // atomic replace on a POSIX filesystem

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
