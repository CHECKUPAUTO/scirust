//! Kernel state persistence.
//!
//! A [`KernelSnapshot`] bundles the causal [`MemoryGraph`], the append-only
//! [`EventLog`] and the hash-chained [`DistributedEventLog`] into a single
//! self-describing JSON document so a session can be saved to disk and later
//! reloaded, verified or replayed (`ccos save` / `verify` / `replay`).

use crate::distributed_event_log::DistributedEventLog;
use crate::event_log::EventLog;
use crate::memory::MemoryGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSnapshot {
    /// Crate version that wrote the snapshot (for forward-compat checks).
    pub version: String,
    pub graph: MemoryGraph,
    pub event_log: EventLog,
    pub dist_log: DistributedEventLog,
}

impl KernelSnapshot {
    pub fn new(graph: MemoryGraph, event_log: EventLog, dist_log: DistributedEventLog) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            graph,
            event_log,
            dist_log,
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Persist the snapshot to `path` as pretty JSON.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = self
            .to_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a snapshot previously written by [`KernelSnapshot::save`].
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Verify the snapshot's integrity without mutating it: both hash chains
    /// (the primary [`EventLog`] and the [`DistributedEventLog`]) must validate
    /// and the graph must hold no dangling edges. Returns the first failure as a
    /// message. This is the single integrity check shared by the load-and-verify
    /// path here and by the runtime restore in [`crate::persistence`].
    pub fn verify_integrity(&self) -> Result<(), String> {
        let dist = self.dist_log.verify_integrity();
        if !dist.valid {
            return Err(format!(
                "distributed-log chain invalid: {} error(s)",
                dist.errors.len()
            ));
        }
        let primary = self.event_log.verify_integrity();
        if !primary.valid {
            return Err(format!(
                "event-log chain invalid: {} error(s)",
                primary.errors.len()
            ));
        }
        let dangling = self.graph.dangling_edge_count();
        if dangling != 0 {
            return Err(format!("{dangling} dangling edge(s) in graph"));
        }
        Ok(())
    }

    /// Load a snapshot **and** verify its integrity ([`verify_integrity`](Self::verify_integrity)),
    /// failing with `InvalidData` if either hash chain or the graph is corrupt.
    pub fn load_verified(path: &str) -> std::io::Result<Self> {
        let snap = Self::load(path)?;
        snap.verify_integrity()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_log::{EventPayload, EventType};
    use crate::memory::NodeType;

    #[test]
    fn snapshot_json_roundtrip_preserves_state() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("a".into(), "A".into(), "x".into(), NodeType::Module);
        graph.upsert_node("b".into(), "B".into(), "y".into(), NodeType::Symbol);
        graph.add_edge(
            "a".into(),
            "b".into(),
            0.9,
            crate::memory::EdgeType::DependsOn,
        );

        let mut event_log = EventLog::new("persist-test".into());
        event_log.append(
            EventType::CycleStart,
            EventPayload::CycleEvent {
                cycle_number: 0,
                action: "init".into(),
            },
        );

        let mut dist_log = DistributedEventLog::new();
        dist_log.append("e0".into(), "kernel".into());

        let snap = KernelSnapshot::new(graph, event_log, dist_log);
        let json = snap.to_json().unwrap();
        let restored = KernelSnapshot::from_json(&json).unwrap();

        assert_eq!(restored.graph.node_count(), 2);
        assert_eq!(restored.graph.edge_count(), 1);
        assert_eq!(restored.event_log.event_count(), 1);
        assert!(restored.dist_log.verify_integrity().valid);
        assert_eq!(restored.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn verify_integrity_passes_clean_and_rejects_a_tampered_chain() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("a".into(), "A".into(), "x".into(), NodeType::Module);
        let mut dist_log = DistributedEventLog::new();
        dist_log.append("e0".into(), "kernel".into());
        let mut snap = KernelSnapshot::new(graph, EventLog::new("t".into()), dist_log);
        assert!(snap.verify_integrity().is_ok(), "a clean snapshot verifies");
        snap.dist_log.hash_chain[0].hash = "deadbeef".repeat(8);
        assert!(
            snap.verify_integrity().is_err(),
            "a tampered chain is rejected"
        );
    }
}
