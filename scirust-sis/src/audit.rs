//! Hash-chained audit log (SHA-256) of safety-relevant events: trip
//! decisions, cause-and-effect matrix changes, proof-test interval resizing.
//! Same principle as `scirust-mcp::audit`/`scirust-discovery::audit` (each
//! entry embeds the hash of the previous one, making tampering after the
//! fact detectable) — the direct analogue, for process safety, of what
//! Triton/Trisis (2017) showed is missing when SIS logic isn't tamper-
//! evident: nobody could prove the safety logic hadn't been altered until
//! it visibly misbehaved.

use scirust_sciagent::sha256::sha256_hex;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEntry {
    pub seq: u64,
    pub timestamp_unix_ms: u128,
    pub event_type: String,
    pub description: String,
    pub prev_hash: String,
    pub hash: String,
}

fn payload(e: &AuditEntry) -> String {
    format!(
        "{}|{}|{}|{}|{}",
        e.seq, e.timestamp_unix_ms, e.event_type, e.description, e.prev_hash
    )
}

#[derive(Debug, Default)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    head: String,
}

impl AuditLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            head: GENESIS_HASH.to_string(),
        }
    }

    pub fn record(&mut self, event_type: &str, description: &str) -> &AuditEntry {
        let seq = self.entries.len() as u64;
        let timestamp_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut entry = AuditEntry {
            seq,
            timestamp_unix_ms,
            event_type: event_type.to_string(),
            description: description.to_string(),
            prev_hash: self.head.clone(),
            hash: String::new(),
        };
        entry.hash = sha256_hex(payload(&entry).as_bytes());
        self.head = entry.hash.clone();
        self.entries.push(entry);
        self.entries.last().expect("just pushed")
    }

    pub fn verify_chain(&self) -> bool {
        let mut prev = GENESIS_HASH.to_string();
        for e in &self.entries
        {
            if e.prev_hash != prev
            {
                return false;
            }
            if sha256_hex(payload(e).as_bytes()) != e.hash
            {
                return false;
            }
            prev = e.hash.clone();
        }
        true
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_starts_valid() {
        assert!(AuditLog::new().verify_chain());
    }

    #[test]
    fn recording_keeps_chain_valid_and_links_entries() {
        let mut log = AuditLog::new();
        log.record("trip_decision", "SIF-101 tripped on high pressure demand");
        log.record(
            "ce_matrix_change",
            "linked High Pressure PT-101 -> Close XV-201",
        );
        assert_eq!(log.len(), 2);
        assert!(log.verify_chain());
        assert_eq!(log.entries()[1].prev_hash, log.entries()[0].hash);
    }

    #[test]
    fn tampering_is_detected() {
        let mut log = AuditLog::new();
        log.record("trip_decision", "SIF-101 did not trip on demand");
        log.entries[0].description = "SIF-101 tripped correctly".to_string();
        assert!(!log.verify_chain());
    }
}
