use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLog {
    pub session_id: String,
    pub events: Vec<TraceEvent>,
}

/// Result of verifying an [`EventLog`]'s canonical hash chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogIntegrity {
    /// True when every link verified and the chain is unbroken.
    pub valid: bool,
    /// Number of events whose content hash matched.
    pub verified_events: usize,
    /// Human-readable descriptions of any broken links or tampered events.
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub id: String,
    pub timestamp: u64,
    pub event_type: EventType,
    pub payload: EventPayload,
    pub sequence_number: u64,
    /// Hash of the previous event in the canonical chain (or [`EventLog::GENESIS_HASH`]
    /// for the first event). Together with `hash` this makes the *primary* event
    /// log tamper-evident, covering every run — not just persisted snapshots.
    #[serde(default)]
    pub prev_hash: String,
    /// SHA-256 link over `(prev_hash, sequence_number, event_type, payload)`. The
    /// non-deterministic `id`/`timestamp` are deliberately excluded so the chain
    /// is reproducible across runs (preserving CCOS's determinism invariant).
    #[serde(default)]
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    LlmCall,
    LlmResponse,
    Parsing,
    GraphMutation,
    FailureDetection,
    FailurePropagation,
    GuardCheck,
    CycleStart,
    CycleEnd,
    ReplayStart,
    ReplayEnd,
    Snapshot,
    AgentAction,
    // ── Context Region Engine (v0.3) ────────────────────────────────
    RegionCreated,
    RegionActivated,
    RegionMerged,
    RegionEvicted,
    ContextWindowGenerated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPayload {
    LlmCallRequest {
        model: String,
        prompt_hash: String,
        input_tokens: usize,
    },
    LlmCallResponse {
        model: String,
        response_hash: String,
        output_tokens: usize,
        latency_ms: u64,
        guard_passed: bool,
        reliability_score: f64,
    },
    Parsing {
        file_path: String,
        file_hash: String,
        modules_found: usize,
        uses_found: usize,
        symbols_found: usize,
    },
    GraphMutation {
        node_id: String,
        operation: String,
        nodes_before: usize,
        nodes_after: usize,
        edges_before: usize,
        edges_after: usize,
    },
    FailureDetection {
        node_id: String,
        failure_type: String,
        severity: f64,
    },
    FailurePropagation {
        origin_node_id: String,
        affected_nodes: Vec<String>,
        depth: u32,
    },
    GuardCheck {
        input_hash: String,
        passed: bool,
        score: f64,
        warnings: Vec<String>,
    },
    CycleEvent {
        cycle_number: u64,
        action: String,
    },
    ReplayEvent {
        original_event_id: String,
        replayed_at: u64,
    },
    Snapshot {
        nodes_count: usize,
        edges_count: usize,
        total_events: usize,
    },
    /// A graph node creation/update, recorded with enough detail to rebuild the
    /// graph by replay (see [`EventLog::record_graph`] / [`GraphReconstructor`]).
    NodeUpserted {
        id: String,
        label: String,
        content: String,
        node_type: NodeType,
    },
    /// A graph edge addition, recorded for replay-based reconstruction.
    EdgeAdded {
        source: String,
        target: String,
        weight: f64,
        edge_type: EdgeType,
    },
    // ── Context Region Engine (v0.3) ────────────────────────────────
    /// A spatial region was formed during clustering.
    RegionCreated {
        region_id: String,
        center: String,
        member_count: usize,
        temperature: f32,
        causal_density: f32,
    },
    /// A region was woken for a task (carries the logical tick for replay).
    RegionActivated {
        region_id: String,
        tick: u64,
        temperature: f32,
        activation_count: u64,
        reason: String,
    },
    /// Two or more file-buckets were merged into one region.
    RegionMerged {
        into_region: String,
        from_region: String,
        member_count: usize,
    },
    /// A cold region was evicted from the context map.
    RegionEvicted {
        region_id: String,
        temperature: f32,
        reason: String,
    },
    /// A context window was hydrated from a region.
    ContextWindowGenerated {
        region_id: String,
        file_count: usize,
        tokens_estimated: usize,
        region_score: f32,
        reason: String,
    },
    Custom {
        key: String,
        value: String,
    },
}

impl EventLog {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            events: Vec::new(),
        }
    }

    /// Genesis predecessor hash for the first event in the chain.
    pub const GENESIS_HASH: &'static str = "GENESIS";

    pub fn append(&mut self, event_type: EventType, payload: EventPayload) -> String {
        let id = Uuid::new_v4().to_string();
        let seq = self.events.len() as u64;
        let prev_hash = self.chain_head();
        let hash = Self::link_hash(&prev_hash, seq, &event_type, &payload);
        let event = TraceEvent {
            id: id.clone(),
            timestamp: Self::now_millis(),
            event_type,
            payload,
            sequence_number: seq,
            prev_hash,
            hash,
        };
        self.events.push(event);
        id
    }

    /// Deterministic hash linking one event's replayable content to its
    /// predecessor. The random `id` and wall-clock `timestamp` are excluded so
    /// the chain is reproducible across runs.
    fn link_hash(
        prev_hash: &str,
        seq: u64,
        event_type: &EventType,
        payload: &EventPayload,
    ) -> String {
        let payload_json = serde_json::to_string(payload).unwrap_or_default();
        sha256_hex(&format!("{prev_hash}|{seq}|{event_type:?}|{payload_json}"))
    }

    /// The chain head — the last event's hash, a commitment to the whole log so
    /// far, or [`EventLog::GENESIS_HASH`] when the log is empty.
    pub fn chain_head(&self) -> String {
        self.events
            .last()
            .map(|e| e.hash.clone())
            .unwrap_or_else(|| Self::GENESIS_HASH.to_string())
    }

    /// Recompute the canonical hash chain from genesis and confirm every link is
    /// intact: a modified payload, a reordering, or an inserted/dropped event
    /// breaks the chain. Reports how many events verified and any errors found.
    pub fn verify_integrity(&self) -> LogIntegrity {
        let mut errors = Vec::new();
        let mut prev = Self::GENESIS_HASH.to_string();
        let mut verified = 0usize;
        for (i, event) in self.events.iter().enumerate() {
            if event.prev_hash != prev {
                errors.push(format!(
                    "event {i} (seq {}): broken link — prev_hash does not match the chain",
                    event.sequence_number
                ));
            }
            let expected = Self::link_hash(
                &prev,
                event.sequence_number,
                &event.event_type,
                &event.payload,
            );
            if event.hash == expected {
                verified += 1;
            } else {
                errors.push(format!(
                    "event {i} (seq {}): content tampered — hash mismatch",
                    event.sequence_number
                ));
            }
            prev = event.hash.clone();
        }
        LogIntegrity {
            valid: errors.is_empty(),
            verified_events: verified,
            errors,
        }
    }

    pub fn replay_events(&self, from_sequence: u64, to_sequence: Option<u64>) -> Vec<&TraceEvent> {
        let end = to_sequence.unwrap_or(self.events.len() as u64);
        self.events
            .iter()
            .filter(|e| e.sequence_number >= from_sequence && e.sequence_number < end)
            .collect()
    }

    pub fn get_event_by_id(&self, id: &str) -> Option<&TraceEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn events_by_type(&self, event_type: EventType) -> Vec<&TraceEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn now_millis() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub fn replay_deterministic(
        &self,
        replay_handler: &mut dyn ReplayHandler,
    ) -> Result<usize, String> {
        let mut processed = 0;
        for event in &self.events {
            replay_handler.handle_event(event)?;
            processed += 1;
        }
        Ok(processed)
    }

    /// Append `NodeUpserted` + `EdgeAdded` events that fully describe `graph`,
    /// so its structure can be rebuilt purely by replaying the log (see
    /// [`GraphReconstructor`]). Emission order is deterministic.
    pub fn record_graph(&mut self, graph: &MemoryGraph) {
        let mut node_ids: Vec<&NodeId> = graph.nodes.keys().collect();
        node_ids.sort();
        for id in node_ids {
            let node = &graph.nodes[id];
            self.append(
                EventType::GraphMutation,
                EventPayload::NodeUpserted {
                    id: id.0.clone(),
                    label: node.label.clone(),
                    content: node.content.clone(),
                    node_type: node.node_type.clone(),
                },
            );
        }
        let mut edges: Vec<&crate::memory::GraphEdge> = graph.edges.iter().collect();
        edges.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then_with(|| a.target.cmp(&b.target))
        });
        for e in edges {
            self.append(
                EventType::GraphMutation,
                EventPayload::EdgeAdded {
                    source: e.source.0.clone(),
                    target: e.target.0.clone(),
                    weight: e.weight,
                    edge_type: e.edge_type.clone(),
                },
            );
        }
    }
}

pub trait ReplayHandler {
    fn handle_event(&mut self, event: &TraceEvent) -> Result<(), String>;
}

#[derive(Debug, Clone)]
pub struct EventReplayer {
    pub statistics: ReplayStatistics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayStatistics {
    pub total_events: usize,
    pub llm_calls: usize,
    pub parsing_events: usize,
    pub graph_mutations: usize,
    pub failures: usize,
    pub guard_checks: usize,
    pub cycles: usize,
    pub regions_created: usize,
    pub regions_activated: usize,
    pub context_windows: usize,
}

impl ReplayHandler for EventReplayer {
    fn handle_event(&mut self, event: &TraceEvent) -> Result<(), String> {
        self.statistics.total_events += 1;
        match &event.payload {
            EventPayload::LlmCallRequest { .. } => self.statistics.llm_calls += 1,
            EventPayload::LlmCallResponse { .. } => self.statistics.llm_calls += 1,
            EventPayload::Parsing { .. } => self.statistics.parsing_events += 1,
            EventPayload::GraphMutation { .. } => self.statistics.graph_mutations += 1,
            EventPayload::NodeUpserted { .. } | EventPayload::EdgeAdded { .. } => {
                self.statistics.graph_mutations += 1
            }
            EventPayload::FailureDetection { .. } => self.statistics.failures += 1,
            EventPayload::FailurePropagation { .. } => self.statistics.failures += 1,
            EventPayload::GuardCheck { .. } => self.statistics.guard_checks += 1,
            EventPayload::CycleEvent { .. } => self.statistics.cycles += 1,
            EventPayload::RegionCreated { .. } => self.statistics.regions_created += 1,
            EventPayload::RegionActivated { .. } => self.statistics.regions_activated += 1,
            EventPayload::ContextWindowGenerated { .. } => self.statistics.context_windows += 1,
            _ => {}
        }
        Ok(())
    }
}

impl EventReplayer {
    pub fn new() -> Self {
        Self {
            statistics: ReplayStatistics::default(),
        }
    }
}

impl Default for EventReplayer {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`ReplayHandler`] that rebuilds a [`MemoryGraph`] from the `NodeUpserted` /
/// `EdgeAdded` events emitted by [`EventLog::record_graph`]. This closes the
/// event-sourcing loop: state is fully derivable from the log, not just the
/// summary statistics produced by [`EventReplayer`].
#[derive(Debug)]
pub struct GraphReconstructor {
    pub graph: MemoryGraph,
    pub nodes_built: usize,
    pub edges_built: usize,
}

impl GraphReconstructor {
    pub fn new() -> Self {
        // Disable paging during reconstruction so a faithful copy is rebuilt
        // regardless of the original graph's size.
        Self {
            graph: MemoryGraph::new(0.0, usize::MAX),
            nodes_built: 0,
            edges_built: 0,
        }
    }
}

impl Default for GraphReconstructor {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayHandler for GraphReconstructor {
    fn handle_event(&mut self, event: &TraceEvent) -> Result<(), String> {
        match &event.payload {
            EventPayload::NodeUpserted {
                id,
                label,
                content,
                node_type,
            } => {
                self.graph.upsert_node(
                    NodeId(id.clone()),
                    label.clone(),
                    content.clone(),
                    node_type.clone(),
                );
                self.nodes_built += 1;
            }
            EventPayload::EdgeAdded {
                source,
                target,
                weight,
                edge_type,
            } => {
                let added = self.graph.add_edge(
                    NodeId(source.clone()),
                    NodeId(target.clone()),
                    *weight,
                    edge_type.clone(),
                );
                if added {
                    self.edges_built += 1;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_retrieve() {
        let mut log = EventLog::new("test-session".into());
        let id = log.append(
            EventType::CycleStart,
            EventPayload::CycleEvent {
                cycle_number: 1,
                action: "start".into(),
            },
        );
        assert_eq!(log.event_count(), 1);
        assert!(log.get_event_by_id(&id).is_some());
    }

    #[test]
    fn test_replay_events_range() {
        let mut log = EventLog::new("test".into());
        for i in 0..5 {
            log.append(
                EventType::CycleStart,
                EventPayload::CycleEvent {
                    cycle_number: i,
                    action: format!("cycle {}", i),
                },
            );
        }
        let replayed = log.replay_events(2, Some(4));
        assert_eq!(replayed.len(), 2);
    }

    #[test]
    fn test_events_by_type() {
        let mut log = EventLog::new("test".into());
        log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: "test.rs".into(),
                file_hash: "abc".into(),
                modules_found: 1,
                uses_found: 2,
                symbols_found: 3,
            },
        );
        log.append(
            EventType::LlmCall,
            EventPayload::LlmCallRequest {
                model: "test".into(),
                prompt_hash: "def".into(),
                input_tokens: 100,
            },
        );
        let parsing_events = log.events_by_type(EventType::Parsing);
        assert_eq!(parsing_events.len(), 1);
    }

    #[test]
    fn test_deterministic_replay() {
        let mut log = EventLog::new("test".into());
        log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: "a.rs".into(),
                file_hash: "h1".into(),
                modules_found: 1,
                uses_found: 0,
                symbols_found: 0,
            },
        );
        let mut replayer = EventReplayer::new();
        let result = log.replay_deterministic(&mut replayer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        assert_eq!(replayer.statistics.parsing_events, 1);
    }

    #[test]
    fn test_record_and_reconstruct_graph() {
        use crate::memory::{EdgeType, MemoryGraph, NodeType};
        let mut graph = MemoryGraph::new(0.0, usize::MAX);
        for i in 0..50 {
            graph.upsert_node(
                NodeId(format!("n{i}")),
                format!("label{i}"),
                format!("content{i}"),
                if i % 2 == 0 {
                    NodeType::Module
                } else {
                    NodeType::Symbol
                },
            );
        }
        for i in 0..49 {
            graph.add_edge(
                NodeId(format!("n{i}")),
                NodeId(format!("n{}", i + 1)),
                0.5,
                EdgeType::DependsOn,
            );
        }

        let mut log = EventLog::new("recon".into());
        log.record_graph(&graph);

        let mut recon = GraphReconstructor::new();
        log.replay_deterministic(&mut recon).unwrap();

        assert_eq!(recon.graph.node_count(), graph.node_count());
        assert_eq!(recon.graph.edge_count(), graph.edge_count());
        // Structural fidelity: same ids, labels, contents, types.
        for (id, node) in &graph.nodes {
            let r = recon
                .graph
                .nodes
                .get(id)
                .expect("reconstructed node present");
            assert_eq!(r.label, node.label);
            assert_eq!(r.content, node.content);
            assert_eq!(r.node_type, node.node_type);
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut log = EventLog::new("test".into());
        log.append(
            EventType::GuardCheck,
            EventPayload::GuardCheck {
                input_hash: "h".into(),
                passed: true,
                score: 0.9,
                warnings: vec![],
            },
        );
        let json = log.to_json();
        let restored: EventLog = EventLog::from_json(&json).unwrap();
        assert_eq!(restored.session_id, "test");
        assert_eq!(restored.event_count(), 1);
    }

    // ── Canonical hash chain (P1.2) ────────────────────────────────

    fn sample(n: u64) -> (EventType, EventPayload) {
        (
            EventType::CycleStart,
            EventPayload::CycleEvent {
                cycle_number: n,
                action: format!("c{n}"),
            },
        )
    }

    #[test]
    fn chain_valid_after_appends() {
        let mut log = EventLog::new("s".into());
        for i in 0..10 {
            let (t, p) = sample(i);
            log.append(t, p);
        }
        let integ = log.verify_integrity();
        assert!(integ.valid, "fresh chain must verify: {:?}", integ.errors);
        assert_eq!(integ.verified_events, 10);
    }

    #[test]
    fn empty_chain_is_valid() {
        let log = EventLog::new("s".into());
        let integ = log.verify_integrity();
        assert!(integ.valid);
        assert_eq!(integ.verified_events, 0);
        assert_eq!(log.chain_head(), EventLog::GENESIS_HASH);
    }

    #[test]
    fn chain_detects_payload_tampering() {
        let mut log = EventLog::new("s".into());
        for i in 0..5 {
            let (t, p) = sample(i);
            log.append(t, p);
        }
        // Tamper with one event's payload without recomputing its hash.
        log.events[2].payload = EventPayload::CycleEvent {
            cycle_number: 999,
            action: "evil".into(),
        };
        let integ = log.verify_integrity();
        assert!(!integ.valid, "tampered payload must be detected");
        assert!(integ.errors.iter().any(|e| e.contains("seq 2")));
    }

    #[test]
    fn chain_detects_reorder() {
        let mut log = EventLog::new("s".into());
        for i in 0..5 {
            let (t, p) = sample(i);
            log.append(t, p);
        }
        log.events.swap(1, 3);
        assert!(
            !log.verify_integrity().valid,
            "reordering must break the chain"
        );
    }

    #[test]
    fn chain_detects_deletion() {
        let mut log = EventLog::new("s".into());
        for i in 0..5 {
            let (t, p) = sample(i);
            log.append(t, p);
        }
        log.events.remove(2);
        assert!(
            !log.verify_integrity().valid,
            "dropping an event must break the chain"
        );
    }

    #[test]
    fn chain_survives_serialization_roundtrip() {
        let mut log = EventLog::new("s".into());
        for i in 0..6 {
            let (t, p) = sample(i);
            log.append(t, p);
        }
        let restored = EventLog::from_json(&log.to_json()).unwrap();
        assert!(restored.verify_integrity().valid);
        assert_eq!(restored.chain_head(), log.chain_head());
    }

    #[test]
    fn chain_head_is_deterministic_across_identical_logs() {
        // The canonical chain excludes the random id/timestamp, so identical
        // event sequences commit to the same head regardless of run.
        let build = || {
            let mut log = EventLog::new("ignored-session".into());
            for i in 0..8 {
                let (t, p) = sample(i);
                log.append(t, p);
            }
            log.chain_head()
        };
        assert_eq!(build(), build(), "chain head must be reproducible");
    }

    #[test]
    fn recorded_graph_chain_is_valid() {
        use crate::memory::{MemoryGraph, NodeType};
        let mut graph = MemoryGraph::new(0.0, usize::MAX);
        for i in 0..20 {
            graph.upsert_node(
                NodeId(format!("n{i}")),
                format!("l{i}"),
                String::new(),
                NodeType::Module,
            );
        }
        let mut log = EventLog::new("rec".into());
        log.record_graph(&graph);
        assert!(
            log.verify_integrity().valid,
            "record_graph appends must chain correctly"
        );
    }
}
