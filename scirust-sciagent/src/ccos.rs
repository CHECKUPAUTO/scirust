use std::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::sha256::sha256_hex;

/// Genesis value for the hash chain (no previous entry).
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// A single entry in the CCOS attestation log.
#[derive(Clone, Debug)]
pub struct CcosEntry {
    pub sequence: u64,
    pub model_version: String,
    pub input_hash: String,
    pub output_hash: String,
    pub timestamp_ns: u128,
    pub prev_hash: String,
    pub chain_hash: String,
}

/// Hash-chained attestation log for verifiable inference.
///
/// Each inference produces an entry cryptographically chained to the
/// previous one. The chain can be verified by recomputing all hashes.
pub struct CcosLog {
    entries: Vec<CcosEntry>,
}

impl CcosEntry {
    pub fn new(
        sequence: u64,
        model_version: &str,
        input_tokens: &[usize],
        output_tokens: &[usize],
        prev_hash: &str,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self::new_with_timestamp(
            sequence,
            model_version,
            input_tokens,
            output_tokens,
            prev_hash,
            ts,
        )
    }

    pub fn new_with_timestamp(
        sequence: u64,
        model_version: &str,
        input_tokens: &[usize],
        output_tokens: &[usize],
        prev_hash: &str,
        timestamp_ns: u128,
    ) -> Self {
        let input_hash = hash_slice(input_tokens);
        let output_hash = hash_slice(output_tokens);

        let mut entry = Self {
            sequence,
            model_version: model_version.to_string(),
            input_hash,
            output_hash,
            timestamp_ns,
            prev_hash: prev_hash.to_string(),
            chain_hash: String::new(),
        };
        entry.chain_hash = entry.compute_chain_hash();
        entry
    }

    /// SHA-256 over an unambiguous, length-prefixed serialization of every
    /// field — including `timestamp_ns`, so a backdated entry breaks the
    /// chain. (The previous implementation used `DefaultHasher`: SipHash is
    /// not collision-resistant, its output is not guaranteed stable across
    /// Rust releases, and the timestamp was left out of the hash entirely.)
    fn compute_chain_hash(&self) -> String {
        let mut bytes = Vec::with_capacity(
            8 + 8
                + self.model_version.len()
                + self.input_hash.len()
                + self.output_hash.len()
                + self.prev_hash.len()
                + 16,
        );
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&(self.model_version.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.model_version.as_bytes());
        bytes.extend_from_slice(self.input_hash.as_bytes());
        bytes.extend_from_slice(self.output_hash.as_bytes());
        bytes.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        bytes.extend_from_slice(self.prev_hash.as_bytes());
        sha256_hex(&bytes)
    }

    pub fn verify_chain_hash(&self) -> bool {
        self.chain_hash == self.compute_chain_hash()
    }
}

impl Default for CcosLog {
    fn default() -> Self {
        Self::new()
    }
}

impl CcosLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn append(
        &mut self,
        model_version: &str,
        input_tokens: &[usize],
        output_tokens: &[usize],
    ) -> &CcosEntry {
        let sequence = self.entries.len() as u64;
        let prev_hash = self
            .entries
            .last()
            .map(|e| e.chain_hash.clone())
            .unwrap_or_else(|| String::from(GENESIS_HASH));
        let entry = CcosEntry::new(
            sequence,
            model_version,
            input_tokens,
            output_tokens,
            &prev_hash,
        );
        self.entries.push(entry);
        self.entries.last().unwrap()
    }

    pub fn entries(&self) -> &[CcosEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify the entire chain from genesis to the last entry.
    pub fn verify(&self) -> bool {
        if self.entries.is_empty()
        {
            return true;
        }
        let mut expected_prev = String::from(GENESIS_HASH);
        for entry in &self.entries
        {
            if entry.prev_hash != expected_prev
            {
                return false;
            }
            if !entry.verify_chain_hash()
            {
                return false;
            }
            expected_prev = entry.chain_hash.clone();
        }
        true
    }

    /// Export as JSON lines.
    pub fn to_json_lines(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries
        {
            let _ = writeln!(
                out,
                r#"{{"seq":{},"model":"{}","input":"{}","output":"{}","ts":{},"prev":"{}","hash":"{}"}}"#,
                entry.sequence,
                entry.model_version,
                entry.input_hash,
                entry.output_hash,
                entry.timestamp_ns,
                entry.prev_hash,
                entry.chain_hash,
            );
        }
        out
    }

    /// The chain hash of the last entry (or genesis if empty).
    pub fn current_chain_hash(&self) -> String {
        self.entries
            .last()
            .map(|e| e.chain_hash.clone())
            .unwrap_or_else(|| String::from(GENESIS_HASH))
    }
}

fn hash_slice(data: &[usize]) -> String {
    let mut bytes = Vec::with_capacity(8 + data.len() * 8);
    bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for &tok in data
    {
        bytes.extend_from_slice(&(tok as u64).to_le_bytes());
    }
    sha256_hex(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccos_append_and_verify() {
        let mut log = CcosLog::new();
        assert!(log.verify());
        assert_eq!(log.current_chain_hash(), GENESIS_HASH);

        log.append("debug-v1", &[1, 2, 3], &[4, 5]);
        assert!(log.verify());
        assert_eq!(log.len(), 1);
        assert_ne!(log.current_chain_hash(), GENESIS_HASH);

        log.append("debug-v1", &[6, 7], &[8, 9, 10]);
        assert!(log.verify());
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_ccos_hash_chain_integrity() {
        let mut log = CcosLog::new();
        log.append("v1", &[1], &[2]);
        log.append("v1", &[3], &[4]);

        let json = log.to_json_lines();
        assert!(json.lines().count() == 2);
    }

    #[test]
    fn timestamp_tampering_breaks_the_chain() {
        let mut log = CcosLog::new();
        log.append("v1", &[1], &[2]);
        assert!(log.verify());
        if let Some(entry) = log.entries.last_mut()
        {
            entry.timestamp_ns += 1; // backdating/forward-dating
        }
        assert!(!log.verify(), "timestamp must be covered by the chain hash");
    }

    #[test]
    fn chain_hash_is_pinned_across_rust_versions() {
        // Known-answer test: this value must NEVER change, on any
        // architecture or Rust release — that is the whole point of moving
        // off DefaultHasher. If this fails, the chain format was broken.
        let entry = CcosEntry::new_with_timestamp(
            7,
            "small-v1",
            &[1, 2, 3],
            &[9],
            GENESIS_HASH,
            123_456_789,
        );
        assert_eq!(
            entry.chain_hash,
            "8bea4cb2026bc2bc8f4ae582f399e26ce2e9ef54dd7839209cffe72644e52fdd"
        );
    }

    #[test]
    fn test_ccos_tamper_detection() {
        let mut log = CcosLog::new();
        log.append("v1", &[1], &[2]);
        log.append("v1", &[3], &[4]);

        assert!(log.verify());

        if let Some(entry) = log.entries.last_mut()
        {
            entry.input_hash = String::from("tampered");
        }
        assert!(!log.verify());
    }

    #[test]
    fn test_ccos_deterministic_hash() {
        let entry1 = CcosEntry::new_with_timestamp(0, "v1", &[1, 2], &[3, 4], GENESIS_HASH, 0);
        let entry2 = CcosEntry::new_with_timestamp(0, "v1", &[1, 2], &[3, 4], GENESIS_HASH, 0);
        assert_eq!(entry1.chain_hash, entry2.chain_hash);
    }
}
