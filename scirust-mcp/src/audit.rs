//! Journal d'audit hash-chaîné pour chaque appel d'outil MCP.
//!
//! Même principe que `scirust-func-safety::audit` (une chaîne à la
//! blockchain : chaque entrée inclut le hash de la précédente), mais avec un
//! vrai SHA-256 (`scirust_sciagent::sha256`) plutôt qu'un hash maison — pour
//! un journal destiné à servir de preuve d'audit, la résistance aux
//! collisions n'est pas négociable. Chaque appel — succès ou échec — est
//! enregistré, avec le hash des arguments et du résultat plutôt que leur
//! contenu en clair (le journal peut être exporté sans exposer de données
//! potentiellement sensibles issues d'une infrastructure cliente).

use scirust_sciagent::sha256::sha256_hex;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEntry {
    pub seq: u64,
    pub timestamp_unix_ms: u128,
    pub tool: String,
    pub arguments_hash: String,
    pub outcome: String,
    pub result_hash: String,
    pub prev_hash: String,
    pub hash: String,
}

fn payload(e: &AuditEntry) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        e.seq, e.timestamp_unix_ms, e.tool, e.arguments_hash, e.outcome, e.result_hash, e.prev_hash
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

    /// Ajoute une entrée pour un appel d'outil et renvoie une référence vers
    /// elle. `outcome` est `"ok"` ou `"error"`.
    pub fn record(
        &mut self,
        tool: &str,
        arguments: &serde_json::Value,
        outcome: &str,
        result: &serde_json::Value,
    ) -> &AuditEntry {
        let seq = self.entries.len() as u64;
        let timestamp_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut entry = AuditEntry {
            seq,
            timestamp_unix_ms,
            tool: tool.to_string(),
            arguments_hash: sha256_hex(arguments.to_string().as_bytes()),
            outcome: outcome.to_string(),
            result_hash: sha256_hex(result.to_string().as_bytes()),
            prev_hash: self.head.clone(),
            hash: String::new(),
        };
        entry.hash = sha256_hex(payload(&entry).as_bytes());
        self.head = entry.hash.clone();
        self.entries.push(entry);
        self.entries.last().expect("just pushed")
    }

    /// Revérifie toute la chaîne depuis la genèse — détecte toute
    /// modification, insertion ou suppression d'entrée après coup.
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
    use serde_json::json;

    #[test]
    fn chain_starts_valid() {
        let log = AuditLog::new();
        assert!(log.verify_chain());
    }

    #[test]
    fn recording_extends_valid_chain() {
        let mut log = AuditLog::new();
        log.record(
            "linalg_svd",
            &json!({"a": [[1.0]]}),
            "ok",
            &json!({"s": [1.0]}),
        );
        log.record(
            "dev_search",
            &json!({"pattern": "fn main"}),
            "ok",
            &json!("match"),
        );
        assert_eq!(log.len(), 2);
        assert!(log.verify_chain());
        assert_eq!(log.entries()[1].prev_hash, log.entries()[0].hash);
    }

    #[test]
    fn tampering_breaks_verification() {
        let mut log = AuditLog::new();
        log.record("tool_a", &json!({}), "ok", &json!({}));
        log.record("tool_b", &json!({}), "ok", &json!({}));
        // Falsifie une entrée après coup — le résultat prétendu "ok" masque
        // en réalité une erreur.
        let tampered = &mut log.entries[0];
        tampered.outcome = "error".to_string();
        assert!(!log.verify_chain());
    }

    #[test]
    fn arguments_and_results_are_hashed_not_stored_in_clear() {
        let mut log = AuditLog::new();
        log.record("tool_a", &json!({"secret": "topsecret"}), "ok", &json!({}));
        let exported = log.export_json().unwrap();
        assert!(!exported.contains("topsecret"));
    }
}
