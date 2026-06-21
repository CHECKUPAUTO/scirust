//! Decision proof — the audit trail.
//!
//! Every decision is sealed into a `DecisionProof` containing:
//! - all individual `DecisionRecord`s
//! - a SHA-256 manifest hash chaining them together
//! - the SciRust version and timestamp
//!
//! A third party can replay the proof by re-running the model on the sealed
//! snapshots and checking that the fingerprints match.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent::CertifiedPrediction;

/// A single decision record — one trading step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub prediction: CertifiedPrediction,
    pub narration: String,
    pub llm_consistent: bool,
    pub timestamp_ms: i64,
}

/// The sealed proof file — contains all records and a manifest hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionProof {
    pub scirust_version: String,
    pub timestamp_ms: i64,
    pub num_decisions: usize,
    pub records: Vec<DecisionRecord>,
    pub manifest_hash: String,
}

impl DecisionProof {
    /// Build a proof from a list of decision records.
    pub fn from_records(records: &[DecisionRecord]) -> Self {
        let manifest_hash = compute_manifest_hash(records);
        DecisionProof {
            scirust_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            num_decisions: records.len(),
            records: records.to_vec(),
            manifest_hash,
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Verify a proof: recompute the manifest hash and check it matches.
    pub fn verify(&self) -> bool {
        let recomputed = compute_manifest_hash(&self.records);
        recomputed == self.manifest_hash
    }

    /// Write the proof to a file (returns the path written).
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    /// Load a proof from a file.
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Summary statistics for the dashboard.
    pub fn summary(&self) -> ProofSummary {
        let longs = self.records.iter().filter(|r| r.prediction.action == crate::agent::Action::Long).count();
        let shorts = self.records.iter().filter(|r| r.prediction.action == crate::agent::Action::Short).count();
        let flats = self.records.iter().filter(|r| r.prediction.action == crate::agent::Action::Flat).count();
        let avg_uncertainty: f32 = if self.records.is_empty() {
            0.0
        } else {
            self.records.iter().map(|r| r.prediction.bounds.uncertainty).sum::<f32>()
                / self.records.len() as f32
        };
        let consistent = self.records.iter().filter(|r| r.llm_consistent).count();
        ProofSummary {
            num_decisions: self.num_decisions,
            longs,
            shorts,
            flats,
            avg_uncertainty,
            llm_consistent: consistent,
            llm_total: self.records.len(),
            manifest_hash: self.manifest_hash.clone(),
        }
    }
}

/// Summary of a proof for quick display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSummary {
    pub num_decisions: usize,
    pub longs: usize,
    pub shorts: usize,
    pub flats: usize,
    pub avg_uncertainty: f32,
    pub llm_consistent: usize,
    pub llm_total: usize,
    pub manifest_hash: String,
}

/// Compute the SHA-256 manifest hash of all records.
///
/// The hash is computed over the canonical JSON encoding of each record's
/// prediction (not the narration — that's LLM-generated and non-deterministic).
fn compute_manifest_hash(records: &[DecisionRecord]) -> String {
    let mut hasher = Sha256::new();
    for record in records {
        let json = serde_json::to_string(&record.prediction).unwrap_or_default();
        hasher.update(json.as_bytes());
        hasher.update(b"|");
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Action;
    use crate::certify::{CertifiedBounds, Interval};

    fn make_record(symbol: &str, action: Action) -> DecisionRecord {
        let pred = CertifiedPrediction {
            symbol: symbol.to_string(),
            action,
            raw_prediction: 0.01,
            bounds: CertifiedBounds {
                eps: 0.01,
                output: Interval::new(0.0, 0.02),
                midpoint: 0.01,
                uncertainty: 0.01,
                weights_fingerprint: "abc".to_string(),
            },
            feature_attribution: std::collections::BTreeMap::new(),
            snapshot_fingerprint: "def".to_string(),
            weights_fingerprint: "abc".to_string(),
            last_close: 50_000.0,
        };
        DecisionRecord {
            prediction: pred,
            narration: "test".to_string(),
            llm_consistent: true,
            timestamp_ms: 1700000000000,
        }
    }

    #[test]
    fn proof_roundtrips_json() {
        let records = vec![make_record("BTC/USDT", Action::Long), make_record("ETH/USDT", Action::Short)];
        let proof = DecisionProof::from_records(&records);
        let json = proof.to_json();
        let parsed: DecisionProof = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.num_decisions, 2);
        assert_eq!(parsed.manifest_hash, proof.manifest_hash);
    }

    #[test]
    fn proof_verifies() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let proof = DecisionProof::from_records(&records);
        assert!(proof.verify());
    }

    #[test]
    fn tampered_proof_fails_verification() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let mut proof = DecisionProof::from_records(&records);
        proof.records[0].prediction.raw_prediction = 999.0;
        assert!(!proof.verify());
    }

    #[test]
    fn summary_counts_correctly() {
        let records = vec![
            make_record("BTC", Action::Long),
            make_record("BTC", Action::Long),
            make_record("BTC", Action::Short),
            make_record("BTC", Action::Flat),
        ];
        let proof = DecisionProof::from_records(&records);
        let s = proof.summary();
        assert_eq!(s.num_decisions, 4);
        assert_eq!(s.longs, 2);
        assert_eq!(s.shorts, 1);
        assert_eq!(s.flats, 1);
        assert_eq!(s.llm_consistent, 4);
    }

    #[test]
    fn save_and_load_file() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let proof = DecisionProof::from_records(&records);
        let path = "/tmp/scirust_test_proof.json";
        proof.save_to_file(path).unwrap();
        let loaded = DecisionProof::load_from_file(path).unwrap();
        assert_eq!(loaded.manifest_hash, proof.manifest_hash);
        assert!(loaded.verify());
    }
}