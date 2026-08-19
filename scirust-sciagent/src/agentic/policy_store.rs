//! Durable session approval policy — append-only, replayable, fail-closed.
//!
//! Mirrors the DeepSeek harness `approval/policy` session event (last valid
//! event wins on replay) with SciRust-native machinery: each policy switch is
//! validated BEFORE it is committed, appended as one JSON line, and chained to
//! the previous line with SHA-256 so truncation or corruption is detectable.
//! A corrupt or unverifiable log fails closed (effective policy = `Never`),
//! never the other way.

use super::permission::ApprovalPolicy;
use crate::sha256::sha256_hex;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// SHA-256 chain hash of the empty log.
pub const POLICY_GENESIS_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// One durable policy event (one JSON line in the log).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalPolicyEvent {
    pub sequence: u64,
    pub policy: ApprovalPolicy,
    pub source: String,
    pub prev_hash: String,
    pub chain_hash: String,
}

impl ApprovalPolicyEvent {
    fn new(sequence: u64, policy: ApprovalPolicy, source: &str, prev_hash: &str) -> Self {
        let mut event = Self {
            sequence,
            policy,
            source: source.to_string(),
            prev_hash: prev_hash.to_string(),
            chain_hash: String::new(),
        };
        event.chain_hash = event.compute_chain_hash();
        event
    }

    /// SHA-256 over a length-prefixed serialization of every field.
    fn compute_chain_hash(&self) -> String {
        let mut bytes = Vec::with_capacity(8 + 8 + 4 + self.source.len() + 64 + 64);
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&(self.source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.source.as_bytes());
        bytes.extend_from_slice(self.policy.label().as_bytes());
        bytes.extend_from_slice(self.prev_hash.as_bytes());
        sha256_hex(&bytes)
    }

    fn verify_chain_hash(&self) -> bool {
        self.chain_hash == self.compute_chain_hash()
    }

    fn to_json_line(&self) -> String {
        format!(
            "{{\"seq\":{},\"policy\":\"{}\",\"source\":\"{}\",\"prev\":\"{}\",\"hash\":\"{}\"}}\n",
            self.sequence,
            self.policy.label(),
            json_escape(&self.source),
            self.prev_hash,
            self.chain_hash,
        )
    }

    fn from_json_line(line: &str) -> Result<Self, String> {
        let line = line.trim();
        if line.is_empty()
        {
            return Err("empty policy log line".to_string());
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid policy log JSON: {error}"))?;
        let sequence = value
            .get("seq")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "policy event missing seq".to_string())?;
        let policy = value
            .get("policy")
            .and_then(|v| v.as_str())
            .map(ApprovalPolicy::parse)
            .ok_or_else(|| "policy event missing policy".to_string())?;
        let source = value
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let prev_hash = value
            .get("prev")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let chain_hash = value
            .get("hash")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        Ok(Self {
            sequence,
            policy,
            source,
            prev_hash,
            chain_hash,
        })
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Durable policy store abstraction. SciAgent never guesses a storage
/// location: front-ends install an explicit store (or use the on-disk
/// implementation with an explicit path).
pub trait ApprovalPolicyStore: Send + Sync {
    /// Append one validated policy event. Returns the committed sequence.
    fn append(&self, policy: ApprovalPolicy, source: &str) -> Result<u64, String>;

    /// Effective policy after replay, or `None` when the log holds no event.
    /// Corruption/verification failure yields `Err` (callers fail closed).
    fn effective(&self) -> Result<Option<ApprovalPolicy>, String>;

    /// All committed events in order (bounded RAM projection is allowed; the
    /// durable log is authoritative).
    fn events(&self) -> Result<Vec<ApprovalPolicyEvent>, String>;
}

/// Append-only on-disk policy log: one JSON line per event, SHA-256 chained.
///
/// A failed append leaves the log byte-identical (no partial line is visible
/// to a later replay: loading stops at the first invalid line). Loading
/// enforces strict sequencing `0..n` and chain verification; any violation
/// fails closed with an error the caller must treat as `Never`.
#[derive(Debug)]
pub struct FileApprovalPolicyStore {
    path: PathBuf,
}

impl FileApprovalPolicyStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn load(&self) -> Result<Vec<ApprovalPolicyEvent>, String> {
        let file = match std::fs::File::open(&self.path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) =>
            {
                return Err(format!(
                    "approval policy log {} cannot be opened: {error}",
                    self.path.display()
                ));
            },
        };
        let mut events = Vec::new();
        let mut expected_prev = POLICY_GENESIS_HASH.to_string();
        for (index, line) in BufReader::new(file).lines().enumerate()
        {
            let line = line.map_err(|error| {
                format!(
                    "approval policy log {} is unreadable at line {}: {error}",
                    self.path.display(),
                    index + 1
                )
            })?;
            if line.trim().is_empty()
            {
                continue;
            }
            let event = ApprovalPolicyEvent::from_json_line(&line).map_err(|error| {
                format!(
                    "approval policy log {} is corrupt at line {}: {error}",
                    self.path.display(),
                    index + 1
                )
            })?;
            if event.sequence != events.len() as u64
            {
                return Err(format!(
                    "approval policy log {} has a sequence gap at line {}",
                    self.path.display(),
                    index + 1
                ));
            }
            if event.prev_hash != expected_prev || !event.verify_chain_hash()
            {
                return Err(format!(
                    "approval policy log {} failed chain verification at line {}",
                    self.path.display(),
                    index + 1
                ));
            }
            expected_prev = event.chain_hash.clone();
            events.push(event);
        }
        Ok(events)
    }
}

impl ApprovalPolicyStore for FileApprovalPolicyStore {
    fn append(&self, policy: ApprovalPolicy, source: &str) -> Result<u64, String> {
        let events = self.load()?;
        let prev_hash = events
            .last()
            .map(|event| event.chain_hash.clone())
            .unwrap_or_else(|| POLICY_GENESIS_HASH.to_string());
        let event = ApprovalPolicyEvent::new(events.len() as u64, policy, source, &prev_hash);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| {
                format!(
                    "approval policy log {} cannot be opened for append: {error}",
                    self.path.display()
                )
            })?;
        file.write_all(event.to_json_line().as_bytes())
            .map_err(|error| {
                format!(
                    "approval policy log {} append failed: {error}",
                    self.path.display()
                )
            })?;
        file.flush().map_err(|error| {
            format!(
                "approval policy log {} flush failed: {error}",
                self.path.display()
            )
        })?;
        Ok(event.sequence)
    }

    fn effective(&self) -> Result<Option<ApprovalPolicy>, String> {
        Ok(self.load()?.last().map(|event| event.policy))
    }

    fn events(&self) -> Result<Vec<ApprovalPolicyEvent>, String> {
        self.load()
    }
}

/// In-memory store used when no durable path is configured. It still
/// validates and sequences events exactly like the durable store.
#[derive(Debug, Default)]
pub struct MemoryApprovalPolicyStore {
    events: std::sync::Mutex<Vec<ApprovalPolicyEvent>>,
}

impl ApprovalPolicyStore for MemoryApprovalPolicyStore {
    fn append(&self, policy: ApprovalPolicy, source: &str) -> Result<u64, String> {
        let mut events = self
            .events
            .lock()
            .map_err(|_| "approval policy memory is unavailable".to_string())?;
        let prev_hash = events
            .last()
            .map(|event| event.chain_hash.clone())
            .unwrap_or_else(|| POLICY_GENESIS_HASH.to_string());
        let event = ApprovalPolicyEvent::new(events.len() as u64, policy, source, &prev_hash);
        let sequence = event.sequence;
        events.push(event);
        Ok(sequence)
    }

    fn effective(&self) -> Result<Option<ApprovalPolicy>, String> {
        Ok(self
            .events
            .lock()
            .map_err(|_| "approval policy memory is unavailable".to_string())?
            .last()
            .map(|event| event.policy))
    }

    fn events(&self) -> Result<Vec<ApprovalPolicyEvent>, String> {
        Ok(self
            .events
            .lock()
            .map_err(|_| "approval policy memory is unavailable".to_string())?
            .clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_log() -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "scirust-policy-{}-{unique}.jsonl",
            std::process::id()
        ));
        path
    }

    #[test]
    fn no_events_defaults_to_none() {
        let store = MemoryApprovalPolicyStore::default();
        assert_eq!(store.effective().unwrap(), None);
        let path = temp_log();
        let file_store = FileApprovalPolicyStore::new(&path);
        assert_eq!(file_store.effective().unwrap(), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ask_then_never_last_event_wins() {
        let store = MemoryApprovalPolicyStore::default();
        store.append(ApprovalPolicy::Ask, "runtime").unwrap();
        assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Ask));
        store.append(ApprovalPolicy::Never, "runtime").unwrap();
        assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Never));
        store.append(ApprovalPolicy::Ask, "user").unwrap();
        assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Ask));
        assert_eq!(store.events().unwrap().len(), 3);
    }

    #[test]
    fn file_store_round_trip_and_restart_replay() {
        let path = temp_log();
        {
            let store = FileApprovalPolicyStore::new(&path);
            store.append(ApprovalPolicy::Ask, "runtime").unwrap();
            store.append(ApprovalPolicy::Never, "delegation").unwrap();
        }
        {
            // Simulated restart: a fresh store over the same file.
            let store = FileApprovalPolicyStore::new(&path);
            assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Never));
            let events = store.events().unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[0].policy, ApprovalPolicy::Ask);
            assert_eq!(events[1].policy, ApprovalPolicy::Never);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncated_log_fails_closed() {
        let path = temp_log();
        {
            let store = FileApprovalPolicyStore::new(&path);
            store.append(ApprovalPolicy::Ask, "runtime").unwrap();
            store.append(ApprovalPolicy::Never, "runtime").unwrap();
        }
        // Simulate a torn tail write: cut the LAST line in half. A partial
        // JSON line must be rejected instead of being silently skipped.
        let content = std::fs::read_to_string(&path).unwrap();
        let cut = content.len() - 10;
        std::fs::write(&path, &content[..cut]).unwrap();
        {
            let store = FileApprovalPolicyStore::new(&path);
            let error = store
                .effective()
                .expect_err("a torn tail line must not load");
            assert!(error.contains("corrupt"), "{error}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn shorter_valid_chain_replays_without_error() {
        // Truncating a whole committed line yields a shorter but still valid
        // chain; replay must succeed (the durable log is authoritative).
        let path = temp_log();
        {
            let store = FileApprovalPolicyStore::new(&path);
            store.append(ApprovalPolicy::Ask, "runtime").unwrap();
            store.append(ApprovalPolicy::Never, "runtime").unwrap();
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let first_line = content.lines().next().unwrap();
        std::fs::write(&path, format!("{first_line}\n")).unwrap();
        let store = FileApprovalPolicyStore::new(&path);
        assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Ask));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_line_fails_closed() {
        let path = temp_log();
        {
            let store = FileApprovalPolicyStore::new(&path);
            store.append(ApprovalPolicy::Ask, "runtime").unwrap();
        }
        std::fs::write(&path, "this is not json\n").unwrap();
        let store = FileApprovalPolicyStore::new(&path);
        let error = store.effective().expect_err("corrupt json must not load");
        assert!(error.contains("corrupt"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sequence_gap_fails_closed() {
        let path = temp_log();
        {
            let store = FileApprovalPolicyStore::new(&path);
            store.append(ApprovalPolicy::Ask, "runtime").unwrap();
            store.append(ApprovalPolicy::Never, "runtime").unwrap();
        }
        // Remove the first line: the remaining line starts at seq 1, which
        // must be rejected as a gap.
        let content = std::fs::read_to_string(&path).unwrap();
        let last_line = content.lines().last().unwrap();
        std::fs::write(&path, format!("{last_line}\n")).unwrap();
        let store = FileApprovalPolicyStore::new(&path);
        let error = store.effective().expect_err("a sequence gap must not load");
        assert!(error.contains("sequence"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failed_append_does_not_change_authority() {
        let path = temp_log();
        let store = FileApprovalPolicyStore::new(&path);
        store.append(ApprovalPolicy::Ask, "runtime").unwrap();
        // Point the store at a path that cannot be appended (a directory).
        let dir = temp_log();
        std::fs::create_dir_all(&dir).unwrap();
        let broken = FileApprovalPolicyStore::new(&dir);
        assert!(broken.append(ApprovalPolicy::Never, "runtime").is_err());
        assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Ask));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn never_survives_restart() {
        let path = temp_log();
        {
            let store = FileApprovalPolicyStore::new(&path);
            store.append(ApprovalPolicy::Never, "deployment").unwrap();
        }
        {
            let store = FileApprovalPolicyStore::new(&path);
            assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Never));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sessions_are_isolated_by_path() {
        let path_a = temp_log();
        let path_b = temp_log();
        let store_a = FileApprovalPolicyStore::new(&path_a);
        let store_b = FileApprovalPolicyStore::new(&path_b);
        store_a.append(ApprovalPolicy::Never, "runtime").unwrap();
        assert_eq!(store_b.effective().unwrap(), None);
        store_b.append(ApprovalPolicy::Ask, "runtime").unwrap();
        assert_eq!(store_a.effective().unwrap(), Some(ApprovalPolicy::Never));
        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(&path_b);
    }
}
