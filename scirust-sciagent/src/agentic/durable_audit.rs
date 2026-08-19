//! Durable approval audit — append-only session event log.
//!
//! [`super::approval_audit::InMemoryApprovalAudit`] remains the bounded,
//! process-local journal for supervision UIs. [`FileApprovalAudit`] is the
//! durable counterpart: every
//! event is appended as one JSON line and chained to the previous line with
//! SHA-256, so ordering is deterministic, corruption or truncation is
//! detectable, and a restart replays the exact same audit trail. A corrupt
//! log fails closed: recording into it is refused until the operator replaces
//! it, and reads surface the corruption instead of guessing.
//!
//! No durable authority is created by an event that never durably committed:
//! [`FileApprovalAudit::record`] returns an error before the caller can act
//! on a grant if the append failed.

use super::approval_audit::ApprovalAuditEvent;
use super::approval_audit::ApprovalAuditSink;
use crate::sha256::sha256_hex;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// SHA-256 chain hash of the empty log.
pub const AUDIT_GENESIS_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// One durable audit line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableAuditEntry {
    pub sequence: u64,
    pub event: ApprovalAuditEvent,
    pub prev_hash: String,
    pub chain_hash: String,
}

impl DurableAuditEntry {
    fn new(sequence: u64, event: ApprovalAuditEvent, prev_hash: &str) -> Self {
        let mut entry = Self {
            sequence,
            event,
            prev_hash: prev_hash.to_string(),
            chain_hash: String::new(),
        };
        entry.chain_hash = entry.compute_chain_hash();
        entry
    }

    fn compute_chain_hash(&self) -> String {
        let canonical = serde_json::to_vec(&self.event).unwrap_or_default();
        let mut bytes = Vec::with_capacity(8 + canonical.len() + 64);
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&(canonical.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&canonical);
        bytes.extend_from_slice(self.prev_hash.as_bytes());
        sha256_hex(&bytes)
    }

    fn verify_chain_hash(&self) -> bool {
        self.chain_hash == self.compute_chain_hash()
    }

    fn to_json_line(&self) -> String {
        let event_json = serde_json::to_string(&self.event).unwrap_or_default();
        format!(
            "{{\"seq\":{},\"event\":{},\"prev\":\"{}\",\"hash\":\"{}\"}}\n",
            self.sequence, event_json, self.prev_hash, self.chain_hash,
        )
    }

    fn from_json_line(line: &str) -> Result<Self, String> {
        let line = line.trim();
        if line.is_empty()
        {
            return Err("empty audit log line".to_string());
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid audit log JSON: {error}"))?;
        let sequence = value
            .get("seq")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "audit entry missing seq".to_string())?;
        let event: ApprovalAuditEvent = serde_json::from_value(
            value
                .get("event")
                .cloned()
                .ok_or_else(|| "audit entry missing event".to_string())?,
        )
        .map_err(|error| format!("invalid audit event JSON: {error}"))?;
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
            event,
            prev_hash,
            chain_hash,
        })
    }
}

/// Append-only durable audit log implementing the existing
/// [`ApprovalAuditSink`] seam.
#[derive(Debug)]
pub struct FileApprovalAudit {
    path: PathBuf,
}

impl FileApprovalAudit {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replay the full durable log in order. Corruption, chain breaks or
    /// sequence gaps fail closed with an error.
    pub fn replay(&self) -> Result<Vec<ApprovalAuditEvent>, String> {
        Ok(self.load()?.into_iter().map(|entry| entry.event).collect())
    }

    fn load(&self) -> Result<Vec<DurableAuditEntry>, String> {
        let file = match std::fs::File::open(&self.path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) =>
            {
                return Err(format!(
                    "approval audit log {} cannot be opened: {error}",
                    self.path.display()
                ));
            },
        };
        let mut entries = Vec::new();
        let mut expected_prev = AUDIT_GENESIS_HASH.to_string();
        for (index, line) in BufReader::new(file).lines().enumerate()
        {
            let line = line.map_err(|error| {
                format!(
                    "approval audit log {} is unreadable at line {}: {error}",
                    self.path.display(),
                    index + 1
                )
            })?;
            if line.trim().is_empty()
            {
                continue;
            }
            let entry = DurableAuditEntry::from_json_line(&line).map_err(|error| {
                format!(
                    "approval audit log {} is corrupt at line {}: {error}",
                    self.path.display(),
                    index + 1
                )
            })?;
            if entry.sequence != entries.len() as u64
            {
                return Err(format!(
                    "approval audit log {} has a sequence gap at line {}",
                    self.path.display(),
                    index + 1
                ));
            }
            if entry.prev_hash != expected_prev || !entry.verify_chain_hash()
            {
                return Err(format!(
                    "approval audit log {} failed chain verification at line {}",
                    self.path.display(),
                    index + 1
                ));
            }
            expected_prev = entry.chain_hash.clone();
            entries.push(entry);
        }
        Ok(entries)
    }
}

impl ApprovalAuditSink for FileApprovalAudit {
    fn record(&self, event: ApprovalAuditEvent) -> Result<(), String> {
        let entries = self.load()?;
        let prev_hash = entries
            .last()
            .map(|entry| entry.chain_hash.clone())
            .unwrap_or_else(|| AUDIT_GENESIS_HASH.to_string());
        let entry = DurableAuditEntry::new(entries.len() as u64, event, &prev_hash);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| {
                format!(
                    "approval audit log {} cannot be opened for append: {error}",
                    self.path.display()
                )
            })?;
        file.write_all(entry.to_json_line().as_bytes())
            .map_err(|error| {
                format!(
                    "approval audit log {} append failed: {error}",
                    self.path.display()
                )
            })?;
        file.flush().map_err(|error| {
            format!(
                "approval audit log {} flush failed: {error}",
                self.path.display()
            )
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::approval_audit::{ApprovalChannel, ApprovalLifecycle, ApprovalResolution};

    fn temp_log() -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "scirust-audit-{}-{unique}.jsonl",
            std::process::id()
        ));
        path
    }

    #[test]
    fn durable_append_replay_round_trip() {
        let path = temp_log();
        let audit = FileApprovalAudit::new(&path);
        audit
            .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
            .unwrap();
        audit
            .record(ApprovalAuditEvent::tool_resolved(
                "c1",
                "build",
                "core",
                ApprovalResolution::AllowedOnce,
            ))
            .unwrap();
        let events = audit.replay().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].lifecycle, ApprovalLifecycle::Requested);
        assert_eq!(events[1].lifecycle, ApprovalLifecycle::Resolved);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn restart_replays_exact_trail() {
        let path = temp_log();
        {
            let audit = FileApprovalAudit::new(&path);
            audit
                .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
                .unwrap();
        }
        {
            let audit = FileApprovalAudit::new(&path);
            let events = audit.replay().unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].call_id, "c1");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn torn_tail_fails_closed() {
        let path = temp_log();
        {
            let audit = FileApprovalAudit::new(&path);
            audit
                .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
                .unwrap();
            audit
                .record(ApprovalAuditEvent::tool_resolved(
                    "c1",
                    "build",
                    "core",
                    ApprovalResolution::AllowedOnce,
                ))
                .unwrap();
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let cut = content.len() - 10;
        std::fs::write(&path, &content[..cut]).unwrap();
        let audit = FileApprovalAudit::new(&path);
        let error = audit.replay().expect_err("torn tail must fail closed");
        assert!(error.contains("corrupt"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tampered_event_breaks_chain() {
        let path = temp_log();
        {
            let audit = FileApprovalAudit::new(&path);
            audit
                .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
                .unwrap();
        }
        // Rewrite the line with a different call_id but the same hash fields.
        let content = std::fs::read_to_string(&path).unwrap();
        let tampered = content.replace("c1", "evil");
        std::fs::write(&path, tampered).unwrap();
        let audit = FileApprovalAudit::new(&path);
        let error = audit.replay().expect_err("tampering must fail closed");
        assert!(error.contains("chain verification"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failed_append_does_not_change_authority() {
        let path = temp_log();
        let audit = FileApprovalAudit::new(&path);
        audit
            .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
            .unwrap();
        let dir = temp_log();
        std::fs::create_dir_all(&dir).unwrap();
        let broken = FileApprovalAudit::new(&dir);
        assert!(
            broken
                .record(ApprovalAuditEvent::tool_requested("c2", "test", ""))
                .is_err()
        );
        assert_eq!(audit.replay().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_log_replays_empty() {
        let path = temp_log();
        let audit = FileApprovalAudit::new(&path);
        assert!(audit.replay().unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn durable_audit_records_request_id_pairs() {
        let path = temp_log();
        let audit = FileApprovalAudit::new(&path);
        let id = crate::agentic::approval_request::ApprovalRequestId::generate();
        audit
            .record(ApprovalAuditEvent::tool_requested_with_id(
                id.clone(),
                "c1",
                "build",
                "core",
            ))
            .unwrap();
        audit
            .record(ApprovalAuditEvent::tool_resolved_with_id(
                id,
                "c1",
                "build",
                "core",
                ApprovalResolution::AllowedOnce,
            ))
            .unwrap();
        let events = audit.replay().unwrap();
        assert_eq!(events[0].request_id, events[1].request_id);
        assert_eq!(events[0].channel, ApprovalChannel::Tool);
        let _ = std::fs::remove_file(&path);
    }
}
