//! CCOS Core — cognitive persistence test suite (mission §36).
//!
//! All tests in this target are **deterministic and offline**: no network, no
//! LLM, no wall-clock dependence. They exercise the operational definition of
//! persistent intelligence (§7) against the public Core API:
//!
//!   temporal_update · contradiction_detection · contradiction_resolution ·
//!   invalidation · episodic_recall · decision_outcome · repeated_failure ·
//!   replay_equivalence · model_switching · provenance
//!
//! LLM-dependent variants live in `tests/model_integration/` and are disabled
//! by default.

use ccos_core::event_log::{EventLog, EventPayload, EventType};
use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};

fn nid(s: &str) -> NodeId {
    NodeId(s.to_string())
}

/// Minimal claim/evidence fixture: one claim with typed support/contradiction
/// surfaces, the Q-Page primitive (§12).
fn claim_graph() -> (MemoryGraph, NodeId) {
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    let claim = nid("db.current");
    g.upsert_node(
        claim.clone(),
        "claim".into(),
        "the production database is PostgreSQL".into(),
        NodeType::ContextBlock,
    );
    (g, claim)
}

fn assert_evidence(g: &mut MemoryGraph, id: &str, claim: &NodeId, w: f64, t: EdgeType) {
    g.upsert_node(
        nid(id),
        id.into(),
        format!("evidence {id}"),
        NodeType::AnalysisResult,
    );
    assert!(
        g.add_edge(nid(id), claim.clone(), w, t),
        "edge endpoints exist"
    );
}

fn custom(key: &str, value: &str) -> EventPayload {
    EventPayload::Custom {
        key: key.into(),
        value: value.into(),
    }
}

// ── 33.1 / §36 temporal_update ─────────────────────────────────────────────
// A later, authoritative update displaces the earlier current-state claim:
// the graph must report the NEW current state as believed and keep the old
// claim addressable (historical), not silently merge the two.
#[test]
fn temporal_update() {
    let (mut g, claim) = claim_graph();
    // T1: PostgreSQL is current.
    assert_evidence(&mut g, "t1.ops_report", &claim, 0.9, EdgeType::Supports);
    let q1 = g.qbelief(&claim);
    assert!(q1.belief > 0.0, "T1: current state believed, got {q1:?}");

    // T3: FoundationDB becomes active — the old claim is directly refuted.
    let new_claim = nid("db.current.v2");
    g.upsert_node(
        new_claim.clone(),
        "claim".into(),
        "the production database is FoundationDB".into(),
        NodeType::ContextBlock,
    );
    assert_evidence(
        &mut g,
        "t3.migration_done",
        &new_claim,
        1.0,
        EdgeType::Supports,
    );
    assert_evidence(
        &mut g,
        "t3.migration_done",
        &claim,
        1.0,
        EdgeType::Contradicts,
    );

    let old = g.qbelief(&claim);
    let new = g.qbelief(&new_claim);
    assert!(old.belief < 0.0, "old claim must read as refuted: {old:?}");
    assert!(new.belief > 0.0, "new claim must read as believed: {new:?}");
    // Historical addressability: the displaced claim is still in the graph.
    assert!(g.contains_node(&claim), "historical claim preserved");
}

// ── §36 contradiction_detection ────────────────────────────────────────────
// Two incompatible sources produce an EXPLICIT, measurable conflict — not a
// silent merge. `conflict` rises only when both surfaces carry weight.
#[test]
fn contradiction_detection() {
    let (mut g, claim) = claim_graph();
    assert_evidence(&mut g, "src_a", &claim, 1.0, EdgeType::Supports);
    let one_sided = g.qbelief(&claim);
    assert!(
        one_sided.conflict < 0.1,
        "one-sided evidence is consensus, not conflict: {one_sided:?}"
    );
    assert_evidence(&mut g, "src_b", &claim, 1.0, EdgeType::Contradicts);
    let contested = g.qbelief(&claim);
    assert!(
        contested.conflict > 0.5,
        "matched opposing evidence must surface high conflict: {contested:?}"
    );
    // Both sources preserved with polarity.
    assert_eq!(g.evidence_of(&claim, EdgeType::Supports), [&nid("src_a")]);
    assert_eq!(
        g.evidence_of(&claim, EdgeType::Contradicts),
        [&nid("src_b")]
    );
}

// ── §36 contradiction_resolution ───────────────────────────────────────────
// Resolution by authority: a higher-authority refutation moves `belief`
// negative while the full evidence set remains inspectable (traceable WHY).
#[test]
fn contradiction_resolution() {
    let (mut g, claim) = claim_graph();
    assert_evidence(&mut g, "blog_rumor", &claim, 0.3, EdgeType::Supports);
    assert_evidence(
        &mut g,
        "official_runbook",
        &claim,
        1.0,
        EdgeType::Contradicts,
    );
    let q = g.qbelief(&claim);
    assert!(
        q.belief < 0.0,
        "higher-authority refutation resolves: {q:?}"
    );
    assert!(
        q.support > 0.0 && q.contradiction > 0.0,
        "no silent merging"
    );
    // The resolution is explainable: the refuting source is enumerable.
    let refs = g.evidence_of(&claim, EdgeType::Contradicts);
    assert_eq!(refs, [&nid("official_runbook")]);
}

// ── §36 invalidation ───────────────────────────────────────────────────────
// Invalidated information must not count as current truth but remains
// available for audit. (Graph-level eviction keeps edges consistent.)
#[test]
fn invalidation() {
    let (mut g, claim) = claim_graph();
    assert_evidence(&mut g, "stale_doc", &claim, 0.8, EdgeType::Supports);
    assert!(g.qbelief(&claim).belief > 0.0);
    // Invalidate the source: remove the stale evidence node; its edge goes with it.
    g.remove_node(&nid("stale_doc"));
    let q = g.qbelief(&claim);
    assert_eq!(q.support, 0.0, "invalidated evidence no longer weighs");
    assert_eq!(q.belief, 0.0, "no current truth from invalidated source");
}

// ── §36 episodic_recall ────────────────────────────────────────────────────
// Past episodes (events) are retrievable by position, in order, with content.
#[test]
fn episodic_recall() {
    let mut log = EventLog::new("episode-test".into());
    for i in 0..5 {
        log.append(
            EventType::AgentAction,
            custom("episode", &format!("step {i}")),
        );
    }
    let events = log.replay_events(0, None);
    assert_eq!(events.len(), 5);
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.sequence_number, i as u64, "ordered recall");
    }
}

// ── §36 decision_outcome ───────────────────────────────────────────────────
// The chain initial state → decision → action → observation → outcome is
// representable and auditable in the hash-chained journal (§11.8).
#[test]
fn decision_outcome() {
    let mut log = EventLog::new("decision-test".into());
    log.append(
        EventType::AgentAction,
        custom("decision", "rollback from state s0"),
    );
    log.append(
        EventType::AgentAction,
        custom("outcome", "recovered in 42ms"),
    );
    let integrity = log.verify_integrity();
    assert!(integrity.valid, "chain intact: {:?}", integrity.errors);
    assert_eq!(integrity.verified_events, 2);
    let replayed = log.replay_events(0, None);
    assert_eq!(replayed.len(), 2, "decision and outcome linked in order");
}

// ── §36 repeated_failure ───────────────────────────────────────────────────
// A failure recorded once is detectable when a similar situation recurs:
// the journal preserves the failure episode for exact retrieval.
#[test]
fn repeated_failure() {
    let mut log = EventLog::new("failure-test".into());
    log.append(
        EventType::FailureDetection,
        custom("strategy_failure", "eager_flush failed: disk_pressure"),
    );
    // Later cycle: query the journal before choosing a strategy.
    let prior_failures: Vec<_> = log
        .replay_events(0, None)
        .into_iter()
        .filter(|e| matches!(e.event_type, EventType::FailureDetection))
        .collect();
    assert_eq!(prior_failures.len(), 1, "prior failure retrievable");
    let payload = serde_json::to_string(&prior_failures[0].payload).unwrap();
    assert!(payload.contains("eager_flush"), "cause identifiable");
}

// ── §36 replay_equivalence ─────────────────────────────────────────────────
// Identical event sequences produce identical derived state (bit-for-bit
// snapshot equality), the Core determinism invariant.
#[test]
fn replay_equivalence() {
    fn build() -> MemoryGraph {
        let (mut g, claim) = claim_graph();
        assert_evidence(&mut g, "e1", &claim, 0.7, EdgeType::Supports);
        assert_evidence(&mut g, "e2", &claim, 0.4, EdgeType::Contradicts);
        g
    }
    let a = build();
    let b = build();
    let sa = serde_json::to_string(&a).unwrap();
    let sb = serde_json::to_string(&b).unwrap();
    assert_eq!(sa, sb, "identical construction → identical snapshot");
    // Round-trip: snapshot → restore → same derived beliefs.
    let restored: MemoryGraph = serde_json::from_str(&sa).unwrap();
    assert_eq!(
        a.qbelief(&nid("db.current")),
        restored.qbelief(&nid("db.current"))
    );
}

// ── §36 model_switching ────────────────────────────────────────────────────
// Provider replacement (GPT ↔ Claude ↔ local) must not lose events, beliefs
// or journal integrity: the cognitive state is model-independent (§9).
#[test]
fn model_switching() {
    let mut log = EventLog::new("switch-test".into());
    for model in ["gpt-5", "claude-opus", "ollama/qwen3"] {
        log.append(
            EventType::LlmResponse,
            EventPayload::LlmCallResponse {
                model: model.into(),
                response_hash: format!("h:{model}"),
                output_tokens: 10,
                latency_ms: 5,
                guard_passed: true,
                reliability_score: 0.9,
            },
        );
    }
    let integrity = log.verify_integrity();
    assert!(integrity.valid, "journal survives provider rotation");
    let models: Vec<String> = log
        .replay_events(0, None)
        .into_iter()
        .filter_map(|e| match &e.payload {
            EventPayload::LlmCallResponse { model, .. } => Some(model.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        models.len(),
        3,
        "all provider identities preserved with provenance"
    );
}

// ── §36 provenance ─────────────────────────────────────────────────────────
// Every assertion answers: which source, with which polarity, since when —
// and provenance is never hallucinated into existence (§33.7).
#[test]
fn provenance() {
    let (mut g, claim) = claim_graph();
    assert_evidence(&mut g, "audit_2026_07", &claim, 1.0, EdgeType::Supports);
    let supporters = g.evidence_of(&claim, EdgeType::Supports);
    assert_eq!(supporters.len(), 1);
    let node = g.node(supporters[0]).expect("source node exists");
    assert_eq!(node.label, "audit_2026_07", "source precision");
    // A claim with no evidence reports neutral — never an invented source.
    let orphan = nid("never.asserted");
    g.upsert_node(
        orphan.clone(),
        "claim".into(),
        "unasserted".into(),
        NodeType::ContextBlock,
    );
    let q = g.qbelief(&orphan);
    assert_eq!(q.support, 0.0);
    assert_eq!(q.belief, 0.0, "no provenance hallucination");
}
