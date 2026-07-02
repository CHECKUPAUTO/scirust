//! Journal d'audit hash-chaîné (SHA-256) de chaque tentative de découverte
//! — dans la portée autorisée ou refusée. Contrairement à
//! `scirust-mcp::audit` (qui hache les arguments d'un appel d'outil
//! générique pour ne rien exposer en clair), l'IP cible et le protocole
//! sont ici stockés **en clair** : c'est précisément le fait que doit
//! prouver ce journal pour un audit de conformité (« quel appareil a été
//! contacté, quand, sous quelle autorisation, avec quel résultat »).

use scirust_sciagent::sha256::sha256_hex;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEntry {
    pub seq: u64,
    pub timestamp_unix_ms: u128,
    pub operator: String,
    pub zone: String,
    pub target: String,
    pub protocol: String,
    pub outcome: String,
    pub prev_hash: String,
    pub hash: String,
}

fn payload(e: &AuditEntry) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        e.seq,
        e.timestamp_unix_ms,
        e.operator,
        e.zone,
        e.target,
        e.protocol,
        e.outcome,
        e.prev_hash
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

    pub fn record(
        &mut self,
        operator: &str,
        zone: &str,
        target: &str,
        protocol: &str,
        outcome: &str,
    ) -> &AuditEntry {
        let seq = self.entries.len() as u64;
        let timestamp_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut entry = AuditEntry {
            seq,
            timestamp_unix_ms,
            operator: operator.to_string(),
            zone: zone.to_string(),
            target: target.to_string(),
            protocol: protocol.to_string(),
            outcome: outcome.to_string(),
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
        log.record("alice", "zone-a", "192.168.1.10", "opcua", "found");
        log.record("alice", "zone-a", "192.168.1.11", "modbus", "refused");
        assert_eq!(log.len(), 2);
        assert!(log.verify_chain());
        assert_eq!(log.entries()[1].prev_hash, log.entries()[0].hash);
    }

    #[test]
    fn tampering_is_detected() {
        let mut log = AuditLog::new();
        log.record("alice", "zone-a", "192.168.1.10", "opcua", "refused");
        log.entries[0].outcome = "found".to_string();
        assert!(!log.verify_chain());
    }

    #[test]
    fn target_and_protocol_are_stored_in_clear_for_compliance() {
        let mut log = AuditLog::new();
        log.record("alice", "zone-a", "192.168.1.10", "opcua", "found");
        let exported = log.export_json().unwrap();
        assert!(exported.contains("192.168.1.10"));
        assert!(exported.contains("opcua"));
    }
}
