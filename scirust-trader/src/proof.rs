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

/// A proof cannot be produced from non-finite numerical evidence or from an
/// object that cannot be serialized canonically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofError {
    pub field: &'static str,
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot seal decision proof: invalid {}", self.field)
    }
}

impl std::error::Error for ProofError {}

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
    pub fn from_records(records: &[DecisionRecord]) -> Result<Self, ProofError> {
        let scirust_version = env!("CARGO_PKG_VERSION").to_string();
        let timestamp_ms = chrono::Utc::now().timestamp_millis();
        let num_decisions = records.len();
        let manifest_hash =
            compute_manifest_hash(&scirust_version, timestamp_ms, num_decisions, records)?;
        Ok(DecisionProof {
            scirust_version,
            timestamp_ms,
            num_decisions,
            records: records.to_vec(),
            manifest_hash,
        })
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Verify a proof: recompute the manifest hash and check it matches.
    pub fn verify(&self) -> bool {
        self.num_decisions == self.records.len()
            && compute_manifest_hash(
                &self.scirust_version,
                self.timestamp_ms,
                self.num_decisions,
                &self.records,
            )
            .map(|recomputed| recomputed == self.manifest_hash)
            .unwrap_or(false)
    }

    /// Write the proof to a file (returns the path written).
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = self
            .to_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a proof from a file.
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Summary statistics for the dashboard.
    pub fn summary(&self) -> ProofSummary {
        let longs = self
            .records
            .iter()
            .filter(|r| r.prediction.action == crate::agent::Action::Long)
            .count();
        let shorts = self
            .records
            .iter()
            .filter(|r| r.prediction.action == crate::agent::Action::Short)
            .count();
        let flats = self
            .records
            .iter()
            .filter(|r| r.prediction.action == crate::agent::Action::Flat)
            .count();
        let avg_uncertainty: f32 = if self.records.is_empty()
        {
            0.0
        }
        else
        {
            self.records
                .iter()
                .map(|r| r.prediction.bounds.uncertainty)
                .sum::<f32>()
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

/// Versioned payload committed by `manifest_hash`. The domain tag prevents a
/// proof manifest from being confused with another SHA-256-based format.
#[derive(Serialize)]
struct DecisionProofManifest<'a> {
    domain: &'static str,
    scirust_version: &'a str,
    timestamp_ms: i64,
    num_decisions: usize,
    records: &'a [DecisionRecord],
}

/// Compute the SHA-256 manifest hash of all proof fields except the hash itself.
fn compute_manifest_hash(
    scirust_version: &str,
    timestamp_ms: i64,
    num_decisions: usize,
    records: &[DecisionRecord],
) -> Result<String, ProofError> {
    for record in records
    {
        validate_prediction(&record.prediction)?;
    }
    let manifest = DecisionProofManifest {
        domain: "scirust.decision-proof.manifest.v1",
        scirust_version,
        timestamp_ms,
        num_decisions,
        records,
    };
    let canonical = serde_json::to_vec(&manifest).map_err(|_| ProofError {
        field: "proof serialization",
    })?;
    let mut hasher = Sha256::new();
    hasher.update((canonical.len() as u64).to_le_bytes());
    hasher.update(&canonical);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_prediction(prediction: &CertifiedPrediction) -> Result<(), ProofError> {
    let numeric_fields = [
        ("raw_prediction", prediction.raw_prediction),
        ("bounds.eps", prediction.bounds.eps),
        ("bounds.output.lo", prediction.bounds.output.lo),
        ("bounds.output.hi", prediction.bounds.output.hi),
        ("bounds.midpoint", prediction.bounds.midpoint),
        ("bounds.uncertainty", prediction.bounds.uncertainty),
        ("last_close", prediction.last_close),
    ];
    if let Some((field, _)) = numeric_fields.iter().find(|(_, value)| !value.is_finite())
    {
        return Err(ProofError { field });
    }
    if prediction
        .feature_attribution
        .values()
        .any(|value| !value.is_finite())
    {
        return Err(ProofError {
            field: "feature_attribution",
        });
    }
    Ok(())
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
        let records = vec![
            make_record("BTC/USDT", Action::Long),
            make_record("ETH/USDT", Action::Short),
        ];
        let proof = DecisionProof::from_records(&records).unwrap();
        let json = proof.to_json().unwrap();
        let parsed: DecisionProof = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.num_decisions, 2);
        assert_eq!(parsed.manifest_hash, proof.manifest_hash);
    }

    #[test]
    fn proof_verifies() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let proof = DecisionProof::from_records(&records).unwrap();
        assert!(proof.verify());
    }

    #[test]
    fn tampered_proof_fails_verification() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let mut proof = DecisionProof::from_records(&records).unwrap();
        proof.records[0].prediction.raw_prediction = 999.0;
        assert!(!proof.verify());
    }

    #[test]
    fn tampering_with_record_audit_fields_fails_verification() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let proof = DecisionProof::from_records(&records).unwrap();

        let mut changed = proof.clone();
        changed.records[0].narration.push_str(" (edited)");
        assert!(!changed.verify());

        let mut changed = proof.clone();
        changed.records[0].llm_consistent = false;
        assert!(!changed.verify());

        let mut changed = proof.clone();
        changed.records[0].timestamp_ms += 1;
        assert!(!changed.verify());
    }

    #[test]
    fn tampering_with_proof_metadata_fails_verification() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let proof = DecisionProof::from_records(&records).unwrap();

        let mut changed = proof.clone();
        changed.scirust_version.push_str("-forged");
        assert!(!changed.verify());

        let mut changed = proof.clone();
        changed.timestamp_ms += 1;
        assert!(!changed.verify());
    }

    #[test]
    fn summary_counts_correctly() {
        let records = vec![
            make_record("BTC", Action::Long),
            make_record("BTC", Action::Long),
            make_record("BTC", Action::Short),
            make_record("BTC", Action::Flat),
        ];
        let proof = DecisionProof::from_records(&records).unwrap();
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
        let proof = DecisionProof::from_records(&records).unwrap();
        let path =
            std::env::temp_dir().join(format!("scirust-test-proof-{}.json", std::process::id()));
        let path = path.to_string_lossy();
        proof.save_to_file(&path).unwrap();
        let loaded = DecisionProof::load_from_file(&path).unwrap();
        assert_eq!(loaded.manifest_hash, proof.manifest_hash);
        assert!(loaded.verify());
    }

    #[test]
    fn non_finite_prediction_cannot_collapse_to_empty_hash_input() {
        let mut nan_record = make_record("BTC/USDT", Action::Long);
        nan_record.prediction.raw_prediction = f32::NAN;
        let mut inf_record = make_record("BTC/USDT", Action::Long);
        inf_record.prediction.raw_prediction = f32::INFINITY;
        assert!(DecisionProof::from_records(&[nan_record]).is_err());
        assert!(DecisionProof::from_records(&[inf_record]).is_err());
    }

    #[test]
    fn tampered_non_finite_proof_fails_verification() {
        let records = vec![make_record("BTC/USDT", Action::Long)];
        let mut proof = DecisionProof::from_records(&records).unwrap();
        proof.records[0].prediction.bounds.uncertainty = f32::NAN;
        assert!(!proof.verify());
    }
}
