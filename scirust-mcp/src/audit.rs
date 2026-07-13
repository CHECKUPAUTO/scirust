//! Journal d'audit hash-chaîné pour chaque appel d'outil MCP.
//!
//! Même principe que `scirust-func-safety::audit` (une chaîne à la
//! blockchain : chaque entrée inclut le hash de la précédente), mais avec un
//! vrai SHA-256 (`scirust_sciagent::sha256`) plutôt qu'un hash maison — pour
//! une trace d'intégrité d'audit, la résistance aux
//! collisions n'est pas négociable. Chaque appel — succès ou échec — est
//! enregistré, avec le hash des arguments et du résultat plutôt que leur
//! contenu en clair (le journal peut être exporté sans exposer de données
//! potentiellement sensibles issues d'une infrastructure cliente).
//!
//! # Limite de confiance
//!
//! La chaîne SHA-256 n'est ni signée ni authentifiée par MAC. Sa validation
//! établit la cohérence interne d'un export, pas l'identité de son producteur :
//! un acteur capable de remplacer l'export entier peut recalculer une chaîne
//! cohérente. Une preuve d'altération exige donc un checkpoint (`head`) conservé
//! par un canal de confiance indépendant. Une `anchor` de confiance prouve la
//! continuité avec la fenêtre antérieure, mais n'authentifie pas à elle seule les
//! nouvelles entrées.

use scirust_sciagent::sha256::sha256_hex;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const DEFAULT_MAX_ENTRIES: usize = 10_000;
pub const AUDIT_EXPORT_VERSION: u32 = 1;

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

/// Self-contained, versioned snapshot of the retained audit window.
///
/// `anchor` is the hash immediately preceding `entries[0]` after rotation;
/// `head` and `next_seq` bind the other end of the retained chain. Consumers
/// should call [`AuditExport::validate`] before processing the snapshot and
/// compare `head` with an independently trusted checkpoint when tamper evidence
/// is required. Validation does not authenticate the snapshot's producer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditExport {
    pub version: u32,
    pub anchor: String,
    pub head: String,
    pub next_seq: u64,
    pub entries: Vec<AuditEntry>,
}

fn payload(e: &AuditEntry) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        e.seq, e.timestamp_unix_ms, e.tool, e.arguments_hash, e.outcome, e.result_hash, e.prev_hash
    )
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl AuditExport {
    /// Validate hashes, links, sequence continuity, and snapshot metadata.
    ///
    /// This checks internal consistency only. It does not establish origin or
    /// authenticity because the export contains no signature or keyed MAC.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != AUDIT_EXPORT_VERSION
        {
            return Err(format!(
                "unsupported audit export version {}; expected {}",
                self.version, AUDIT_EXPORT_VERSION
            ));
        }
        if !is_sha256_hex(&self.anchor)
        {
            return Err("audit anchor is not a lowercase SHA-256 digest".to_string());
        }
        if !is_sha256_hex(&self.head)
        {
            return Err("audit head is not a lowercase SHA-256 digest".to_string());
        }

        if self.entries.is_empty()
        {
            if self.anchor != self.head
            {
                return Err("empty audit export must have identical anchor and head".to_string());
            }
            if self.next_seq != 0
            {
                return Err("empty audit export must have next_seq 0".to_string());
            }
            return Ok(());
        }

        let first_seq = self.entries[0].seq;
        if first_seq == 0 && self.anchor != GENESIS_HASH
        {
            return Err("unrotated audit export must start at the genesis anchor".to_string());
        }

        let mut previous_hash = self.anchor.as_str();
        for (offset, entry) in self.entries.iter().enumerate()
        {
            let offset = u64::try_from(offset)
                .map_err(|_| "audit export contains too many entries".to_string())?;
            let expected_seq = first_seq
                .checked_add(offset)
                .ok_or_else(|| "audit sequence overflow".to_string())?;
            if entry.seq != expected_seq
            {
                return Err(format!(
                    "non-contiguous audit sequence: expected {expected_seq}, got {}",
                    entry.seq
                ));
            }
            if entry.prev_hash != previous_hash
            {
                return Err(format!("broken audit link at sequence {}", entry.seq));
            }
            if !is_sha256_hex(&entry.arguments_hash)
                || !is_sha256_hex(&entry.result_hash)
                || !is_sha256_hex(&entry.prev_hash)
                || !is_sha256_hex(&entry.hash)
            {
                return Err(format!(
                    "invalid SHA-256 digest at audit sequence {}",
                    entry.seq
                ));
            }
            if sha256_hex(payload(entry).as_bytes()) != entry.hash
            {
                return Err(format!("audit hash mismatch at sequence {}", entry.seq));
            }
            previous_hash = &entry.hash;
        }

        if self.head != previous_hash
        {
            return Err("audit head does not match the final retained entry".to_string());
        }
        let expected_next_seq = self
            .entries
            .last()
            .expect("non-empty checked")
            .seq
            .checked_add(1)
            .ok_or_else(|| "audit sequence overflow".to_string())?;
        if self.next_seq != expected_next_seq
        {
            return Err(format!(
                "invalid next_seq: expected {expected_next_seq}, got {}",
                self.next_seq
            ));
        }
        Ok(())
    }

    /// Validate the snapshot and compare its head with an independently
    /// trusted checkpoint.
    ///
    /// The checkpoint, not this unkeyed hash chain, supplies the trust root.
    pub fn validate_against_head(&self, trusted_head: &str) -> Result<(), String> {
        self.validate()?;
        if !is_sha256_hex(trusted_head)
        {
            return Err("trusted audit head is not a lowercase SHA-256 digest".to_string());
        }
        if self.head != trusted_head
        {
            return Err("audit head does not match the trusted checkpoint".to_string());
        }
        Ok(())
    }

    /// Deserialize and internally validate an exported audit snapshot.
    ///
    /// This does not authenticate the snapshot; compare its head with a
    /// checkpoint obtained through an independent trusted channel.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let export: Self = serde_json::from_str(json).map_err(|error| error.to_string())?;
        export.validate()?;
        Ok(export)
    }
}

#[derive(Debug)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    head: String,
    anchor: String,
    next_seq: u64,
    max_entries: usize,
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditLog {
    pub fn new() -> Self {
        Self::with_max_entries(DEFAULT_MAX_ENTRIES)
    }

    /// Create a bounded in-memory audit window. When the window is full, its
    /// oldest entry is discarded and its hash becomes the verification anchor
    /// for the remaining chain.
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            head: GENESIS_HASH.to_string(),
            anchor: GENESIS_HASH.to_string(),
            next_seq: 0,
            max_entries: max_entries.max(1),
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
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
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
        if self.entries.len() == self.max_entries
        {
            let removed = self.entries.remove(0);
            self.anchor = removed.hash;
        }
        self.entries.push(entry);
        self.entries.last().expect("just pushed")
    }

    /// Revérifie la cohérence de la fenêtre depuis son ancre. Détecte les
    /// modifications qui ne recalculent pas la chaîne ; l'authenticité exige un
    /// checkpoint externe ou une signature/MAC.
    pub fn verify_chain(&self) -> bool {
        let mut prev = self.anchor.clone();
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
        if self.head != prev
        {
            return false;
        }
        match self.entries.last()
        {
            Some(last) => last.seq.checked_add(1) == Some(self.next_seq),
            None => self.next_seq == 0 && self.anchor == self.head,
        }
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

    /// Capture a self-contained snapshot whose internal consistency remains
    /// verifiable after the bounded in-memory window has rotated.
    pub fn export(&self) -> AuditExport {
        AuditExport {
            version: AUDIT_EXPORT_VERSION,
            anchor: self.anchor.clone(),
            head: self.head.clone(),
            next_seq: self.next_seq,
            entries: self.entries.clone(),
        }
    }

    /// Export the historical entries-only JSON shape.
    ///
    /// This method preserves the original public wire format. New integrations
    /// that need rotation metadata should use [`AuditLog::export_snapshot_json`].
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.entries)
    }

    /// Export a versioned envelope with the retained entries and the metadata
    /// required for internal validation after rotation.
    ///
    /// The envelope is not authenticated. Persist its `head` through an
    /// independent trusted channel if later tamper evidence is required.
    pub fn export_snapshot_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.export())
    }

    /// Explicit alias for the historical entries-only JSON shape.
    pub fn export_entries_json(&self) -> Result<String, serde_json::Error> {
        self.export_json()
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

    #[test]
    fn bounded_window_keeps_chain_valid_and_sequence_monotonic() {
        let mut log = AuditLog::with_max_entries(2);
        for i in 0..3
        {
            log.record("tool", &json!({"i": i}), "ok", &json!({}));
        }
        assert_eq!(log.len(), 2);
        assert_eq!(log.entries()[0].seq, 1);
        assert_eq!(log.entries()[1].seq, 2);
        assert!(log.verify_chain());
    }

    #[test]
    fn rotated_export_contains_the_anchor_and_validates_internal_consistency() {
        let mut log = AuditLog::with_max_entries(2);
        for i in 0..4
        {
            log.record("tool", &json!({"i": i}), "ok", &json!({"i": i}));
        }

        let trusted_head = log.export().head;
        let json = log.export_snapshot_json().unwrap();
        let export = AuditExport::from_json(&json).unwrap();

        assert_eq!(export.version, AUDIT_EXPORT_VERSION);
        assert_ne!(export.anchor, GENESIS_HASH);
        assert_eq!(export.entries.len(), 2);
        assert_eq!(export.entries[0].seq, 2);
        assert_eq!(export.entries[0].prev_hash, export.anchor);
        assert_eq!(export.head, export.entries[1].hash);
        assert_eq!(export.next_seq, 4);
        assert!(export.validate_against_head(&trusted_head).is_ok());
        assert!(export.validate_against_head(GENESIS_HASH).is_err());
    }

    #[test]
    fn whole_export_rewrite_requires_a_trusted_head_to_detect() {
        let mut log = AuditLog::new();
        log.record("tool_a", &json!({"value": 1}), "ok", &json!({}));
        log.record("tool_b", &json!({"value": 2}), "ok", &json!({}));
        let mut forged = log.export();
        let trusted_head = forged.head.clone();

        // An attacker who can replace the complete unkeyed export can rewrite
        // an entry and recompute every subsequent link while retaining the same
        // genesis anchor. Internal validation alone cannot establish origin.
        forged.entries[0].outcome = "error".to_string();
        let mut previous_hash = forged.anchor.clone();
        for entry in &mut forged.entries
        {
            entry.prev_hash = previous_hash;
            entry.hash = sha256_hex(payload(entry).as_bytes());
            previous_hash = entry.hash.clone();
        }
        forged.head = previous_hash;

        assert!(forged.validate().is_ok());
        assert!(forged.validate_against_head(&trusted_head).is_err());
    }

    #[test]
    fn exported_metadata_and_entries_are_tamper_evident() {
        let mut log = AuditLog::with_max_entries(2);
        for i in 0..3
        {
            log.record("tool", &json!({"i": i}), "ok", &json!({}));
        }
        let export = log.export();

        let mut bad_version = export.clone();
        bad_version.version += 1;
        assert!(bad_version.validate().is_err());

        let mut bad_anchor = export.clone();
        bad_anchor.anchor = GENESIS_HASH.to_string();
        assert!(bad_anchor.validate().is_err());

        let mut bad_head = export.clone();
        bad_head.head = GENESIS_HASH.to_string();
        assert!(bad_head.validate().is_err());

        let mut bad_next_seq = export.clone();
        bad_next_seq.next_seq += 1;
        assert!(bad_next_seq.validate().is_err());

        let mut bad_entry = export;
        bad_entry.entries[0].outcome = "error".to_string();
        assert!(bad_entry.validate().is_err());
    }

    #[test]
    fn legacy_export_shape_is_preserved_and_snapshot_is_explicit() {
        let mut log = AuditLog::new();
        log.record("tool", &json!({}), "ok", &json!({}));

        let legacy: serde_json::Value = serde_json::from_str(&log.export_json().unwrap()).unwrap();
        let explicit_legacy: serde_json::Value =
            serde_json::from_str(&log.export_entries_json().unwrap()).unwrap();
        let envelope: serde_json::Value =
            serde_json::from_str(&log.export_snapshot_json().unwrap()).unwrap();

        assert!(legacy.is_array());
        assert_eq!(legacy, explicit_legacy);
        assert_eq!(envelope["version"], AUDIT_EXPORT_VERSION);
        assert!(envelope["entries"].is_array());
    }
}
