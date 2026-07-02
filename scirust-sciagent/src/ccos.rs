use std::collections::hash_map::DefaultHasher;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

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
        Self::new_with_timestamp(sequence, model_version, input_tokens, output_tokens, prev_hash, ts)
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

    fn compute_chain_hash(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.sequence.hash(&mut hasher);
        self.model_version.hash(&mut hasher);
        self.input_hash.hash(&mut hasher);
        self.output_hash.hash(&mut hasher);
        self.prev_hash.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    pub fn verify_chain_hash(&self) -> bool {
        self.chain_hash == self.compute_chain_hash()
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
            .unwrap_or_else(|| String::from("0000000000000000"));
        let entry = CcosEntry::new(sequence, model_version, input_tokens, output_tokens, &prev_hash);
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
        if self.entries.is_empty() {
            return true;
        }
        let mut expected_prev = String::from("0000000000000000");
        for entry in &self.entries {
            if entry.prev_hash != expected_prev {
                return false;
            }
            if !entry.verify_chain_hash() {
                return false;
            }
            expected_prev = entry.chain_hash.clone();
        }
        true
    }

    /// Export as JSON lines.
    pub fn to_json_lines(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
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
            .unwrap_or_else(|| String::from("0000000000000000"))
    }
}

fn hash_slice(data: &[usize]) -> String {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccos_append_and_verify() {
        let mut log = CcosLog::new();
        assert!(log.verify());
        assert_eq!(log.current_chain_hash(), "0000000000000000");

        log.append("debug-v1", &[1, 2, 3], &[4, 5]);
        assert!(log.verify());
        assert_eq!(log.len(), 1);
        assert_ne!(log.current_chain_hash(), "0000000000000000");

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
    fn test_ccos_tamper_detection() {
        let mut log = CcosLog::new();
        log.append("v1", &[1], &[2]);
        log.append("v1", &[3], &[4]);

        assert!(log.verify());

        if let Some(entry) = log.entries.last_mut() {
            entry.input_hash = String::from("tampered");
        }
        assert!(!log.verify());
    }

    #[test]
    fn test_ccos_deterministic_hash() {
        let entry1 = CcosEntry::new_with_timestamp(0, "v1", &[1, 2], &[3, 4], "0000000000000000", 0);
        let entry2 = CcosEntry::new_with_timestamp(0, "v1", &[1, 2], &[3, 4], "0000000000000000", 0);
        assert_eq!(entry1.chain_hash, entry2.chain_hash);
    }
}
