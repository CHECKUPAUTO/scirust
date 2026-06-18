use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: f64,
    pub event_type: String,
    pub description: String,
    pub component_id: String,
    pub decision: String,
    pub confidence: f32,
    /// Hash of the previous entry (chain)
    pub prev_hash: Vec<u8>,
    /// Hash of this entry
    pub hash: Vec<u8>,
}

/// Immutable audit log with hash chaining.
///
/// Each entry's hash includes the hash of the previous entry,
/// creating a tamper-evident chain (similar to a blockchain).
/// Required for ISO 26262 safety case documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub entries: VecDeque<AuditEntry>,
    /// Maximum entries to keep (rolling window)
    pub max_entries: usize,
    /// Current chain head hash
    pub head_hash: Vec<u8>,
}

impl AuditLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(10000)),
            max_entries,
            head_hash: vec![0u8; 32], // genesis hash
        }
    }

    /// Add a new audit entry.
    pub fn add(
        &mut self,
        event_type: &str,
        description: &str,
        component_id: &str,
        decision: &str,
        confidence: f32,
        timestamp: f64,
    ) -> AuditEntry {
        let prev_hash = self.head_hash.clone();
        let entry_data = format!(
            "{}|{}|{}|{}|{}|{}",
            timestamp, event_type, description, component_id, decision, confidence
        );
        let mut hash_input = prev_hash.clone();
        hash_input.extend(entry_data.bytes());
        let hash = simple_hash(&hash_input);

        let entry = AuditEntry {
            timestamp,
            event_type: event_type.to_string(),
            description: description.to_string(),
            component_id: component_id.to_string(),
            decision: decision.to_string(),
            confidence,
            prev_hash,
            hash: hash.clone(),
        };

        self.head_hash = hash;
        self.entries.push_back(entry.clone());

        // Enforce max entries (rolling window)
        while self.entries.len() > self.max_entries
        {
            self.entries.pop_front();
        }

        entry
    }

    /// Verify the integrity of the entire chain.
    ///
    /// In a rolling-window log, the first entry's prev_hash may point to an
    /// evicted entry. We verify internal consistency from the 2nd entry onward,
    /// and only check the hash of the first entry in isolation.
    pub fn verify_chain(&self) -> bool {
        let entries: Vec<&AuditEntry> = self.entries.iter().collect();
        if entries.is_empty()
        {
            return true;
        }
        // Verify first entry's hash is self-consistent
        let first = entries[0];
        let first_data = format!(
            "{}|{}|{}|{}|{}|{}",
            first.timestamp,
            first.event_type,
            first.description,
            first.component_id,
            first.decision,
            first.confidence
        );
        let mut hash_input = first.prev_hash.clone();
        hash_input.extend(first_data.bytes());
        if simple_hash(&hash_input) != first.hash
        {
            return false;
        }
        // Verify chain consistency from 2nd entry onward
        for window in entries.windows(2)
        {
            let prev = window[0];
            let current = window[1];
            if current.prev_hash != prev.hash
            {
                return false;
            }
            let entry_data = format!(
                "{}|{}|{}|{}|{}|{}",
                current.timestamp,
                current.event_type,
                current.description,
                current.component_id,
                current.decision,
                current.confidence
            );
            let mut hash_input = current.prev_hash.clone();
            hash_input.extend(entry_data.bytes());
            if simple_hash(&hash_input) != current.hash
            {
                return false;
            }
        }
        true
    }

    /// Export the log as JSON for compliance documentation.
    pub fn export_json(&self) -> Result<String, String> {
        let entries: Vec<&AuditEntry> = self.entries.iter().collect();
        serde_json::to_string_pretty(&entries).map_err(|e| e.to_string())
    }

    /// Query entries by component.
    pub fn filter_by_component(&self, component_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.component_id == component_id)
            .collect()
    }

    /// Query entries by event type.
    pub fn filter_by_type(&self, event_type: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Count entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A wrapper that provides chain verification utilities.
pub struct AuditChain;

impl AuditChain {
    /// Verify that one entry is the valid successor of another.
    pub fn verify_pair(prev: &AuditEntry, current: &AuditEntry) -> bool {
        if current.prev_hash != prev.hash
        {
            return false;
        }
        let entry_data = format!(
            "{}|{}|{}|{}|{}|{}",
            current.timestamp,
            current.event_type,
            current.description,
            current.component_id,
            current.decision,
            current.confidence
        );
        let mut hash_input = current.prev_hash.clone();
        hash_input.extend(entry_data.bytes());
        simple_hash(&hash_input) == current.hash
    }
}

fn simple_hash(data: &[u8]) -> Vec<u8> {
    let mut state = [0u8; 32];
    for (i, &b) in data.iter().enumerate()
    {
        let idx = i % 32;
        state[idx] = state[idx].wrapping_add(b);
        state[(idx + 1) % 32] = state[(idx + 1) % 32].wrapping_mul(b.wrapping_add(1));
        state[(idx + 2) % 32] ^= b.rotate_left(3);
    }
    for i in 0..32
    {
        state[i] = state[i].wrapping_mul(0x5B).wrapping_add(0x9E);
        state[(i + 7) % 32] ^= state[i];
    }
    state.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_add_and_verify() {
        let mut log = AuditLog::new(100);
        log.add(
            "detection",
            "Spike detected",
            "sensor-1",
            "alert",
            0.95,
            1.0,
        );
        log.add("detection", "Normal", "sensor-1", "pass", 0.99, 2.0);
        assert_eq!(log.len(), 2);
        assert!(log.verify_chain());
    }

    #[test]
    fn test_chain_tamper_detection() {
        let mut log = AuditLog::new(100);
        log.add("detection", "Event A", "c1", "alert", 0.9, 1.0);
        log.add("detection", "Event B", "c1", "pass", 0.5, 2.0);
        // Tamper with the second entry
        let entry = log.entries.back_mut().unwrap();
        entry.confidence = 0.99; // tampered
        assert!(
            !log.verify_chain(),
            "tampered chain should fail verification"
        );
    }

    #[test]
    fn test_rolling_window() {
        let mut log = AuditLog::new(5);
        for i in 0..10
        {
            log.add("test", &format!("Event {}", i), "c1", "pass", 0.9, i as f64);
        }
        assert_eq!(log.len(), 5); // only keeps last 5
        assert!(log.verify_chain());
    }

    #[test]
    fn test_filter_by_component() {
        let mut log = AuditLog::new(100);
        log.add("test", "Event A", "comp-1", "pass", 0.9, 1.0);
        log.add("test", "Event B", "comp-2", "pass", 0.9, 2.0);
        log.add("test", "Event C", "comp-1", "pass", 0.9, 3.0);
        let filtered = log.filter_by_component("comp-1");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_export_json() {
        let mut log = AuditLog::new(100);
        log.add("test", "Event", "c1", "pass", 0.9, 1.0);
        let json = log.export_json().unwrap();
        assert!(json.contains("Event"));
        assert!(json.contains("c1"));
    }

    #[test]
    fn test_empty_log() {
        let log = AuditLog::new(100);
        assert!(log.is_empty());
        assert!(log.verify_chain()); // empty chain is valid
    }

    #[test]
    fn test_audit_chain_verify_pair() {
        let mut log = AuditLog::new(100);
        let e1 = log.add("test", "A", "c1", "pass", 0.9, 1.0);
        let e2 = log.add("test", "B", "c1", "pass", 0.9, 2.0);
        assert!(AuditChain::verify_pair(&e1, &e2));
    }
}
