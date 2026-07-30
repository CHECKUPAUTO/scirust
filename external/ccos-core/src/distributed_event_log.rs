use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: u64,
    pub payload: String,
    pub timestamp: u64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashChainLink {
    pub event_id: u64,
    pub hash: String,
    pub previous_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedEventLog {
    pub events: VecDeque<Event>,
    pub hash_chain: Vec<HashChainLink>,
    pub next_id: u64,
}

impl DistributedEventLog {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            hash_chain: Vec::new(),
            next_id: 0,
        }
    }

    pub fn append(&mut self, payload: String, source: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let timestamp = Self::now_millis();
        let event = Event {
            id,
            payload: payload.clone(),
            timestamp,
            source,
        };

        let previous_hash = self
            .hash_chain
            .last()
            .map(|link| link.hash.clone())
            .unwrap_or_else(|| String::from("GENESIS"));

        let hash = Self::compute_link_hash(id, &payload, &previous_hash);

        let link = HashChainLink {
            event_id: id,
            hash,
            previous_hash,
        };

        self.hash_chain.push(link);
        self.events.push_back(event);
        id
    }

    pub fn replay(&self) -> Vec<Event> {
        self.events.iter().cloned().collect()
    }

    pub fn verify_integrity(&self) -> IntegrityReport {
        if self.hash_chain.is_empty() {
            return IntegrityReport {
                valid: true,
                total_events: 0,
                verified_events: 0,
                errors: vec![],
            };
        }

        let mut errors = Vec::new();
        let mut verified = 0;

        for (i, link) in self.hash_chain.iter().enumerate() {
            let event = match self.events.get(i) {
                Some(e) => e,
                None => {
                    errors.push(format!(
                        "Event index {} not found for hash chain link {}",
                        i, link.event_id
                    ));
                    continue;
                }
            };

            let expected_prev = if i == 0 {
                String::from("GENESIS")
            } else {
                self.hash_chain[i - 1].hash.clone()
            };

            if link.previous_hash != expected_prev {
                errors.push(format!(
                    "Chain broken at event {}: expected prev_hash {}, got {}",
                    link.event_id, expected_prev, link.previous_hash
                ));
            }

            let recomputed = Self::compute_link_hash(event.id, &event.payload, &link.previous_hash);

            if recomputed != link.hash {
                errors.push(format!(
                    "Hash mismatch at event {}: stored {}, computed {}",
                    link.event_id, link.hash, recomputed
                ));
            }

            verified += 1;
        }

        IntegrityReport {
            valid: errors.is_empty(),
            total_events: self.events.len(),
            verified_events: verified,
            errors,
        }
    }

    pub fn export_chain(&self) -> Vec<HashChainLink> {
        self.hash_chain.clone()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Link hash over the *replayable* content only — `id | payload |
    /// previous_hash`. The wall-clock `timestamp` is deliberately **excluded**
    /// (it is metrics-only), so the chain is reproducible across runs that ingest
    /// identical inputs — mirroring [`crate::event_log::EventLog`]'s link hash.
    fn compute_link_hash(id: u64, payload: &str, previous_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(id.to_le_bytes());
        hasher.update(payload.as_bytes());
        hasher.update(previous_hash.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn now_millis() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub valid: bool,
    pub total_events: usize,
    pub verified_events: usize,
    pub errors: Vec<String>,
}

impl Default for DistributedEventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_replay() {
        let mut log = DistributedEventLog::new();
        log.append("event_1".into(), "test".into());
        log.append("event_2".into(), "test".into());

        let replay = log.replay();
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].payload, "event_1");
        assert_eq!(replay[1].payload, "event_2");
    }

    #[test]
    fn test_hash_chain_integrity() {
        let mut log = DistributedEventLog::new();
        log.append("payload_a".into(), "module_a".into());
        log.append("payload_b".into(), "module_b".into());
        log.append("payload_c".into(), "module_c".into());

        let report = log.verify_integrity();
        assert!(report.valid);
        assert_eq!(report.verified_events, 3);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_chain_detects_tampering() {
        let mut log = DistributedEventLog::new();
        log.append("original".into(), "test".into());

        // Tamper with the stored hash
        log.hash_chain[0].hash =
            "0000000000000000000000000000000000000000000000000000000000000000".into();

        let report = log.verify_integrity();
        assert!(!report.valid);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_genesis_link_first_event() {
        let mut log = DistributedEventLog::new();
        log.append("first".into(), "test".into());

        assert_eq!(log.hash_chain[0].previous_hash, "GENESIS");
    }

    #[test]
    fn test_replay_is_deterministic() {
        let mut log = DistributedEventLog::new();
        for i in 0..10 {
            log.append(format!("event_{}", i), "test".into());
        }

        let replay1 = log.replay();
        let replay2 = log.replay();

        assert_eq!(replay1.len(), replay2.len());
        for (a, b) in replay1.iter().zip(replay2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.payload, b.payload);
            assert_eq!(a.timestamp, b.timestamp);
        }
    }

    #[test]
    fn chain_is_reproducible_across_runs_ignoring_wall_clock() {
        // Two independent logs ingesting identical (payload, source) sequences
        // must produce the SAME hash chain: the link hash excludes the wall-clock
        // timestamp, so the chain is replay-reproducible across runs.
        let build = || {
            let mut log = DistributedEventLog::new();
            log.append("file_hash_a".into(), "external-memory".into());
            log.append("file_hash_b".into(), "external-memory".into());
            log
        };
        let a = build();
        let b = build();
        let ha: Vec<&String> = a.hash_chain.iter().map(|l| &l.hash).collect();
        let hb: Vec<&String> = b.hash_chain.iter().map(|l| &l.hash).collect();
        assert_eq!(ha, hb, "chain must not depend on wall-clock time");
        assert!(a.verify_integrity().valid && b.verify_integrity().valid);
    }
}
