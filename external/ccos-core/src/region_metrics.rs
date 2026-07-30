//! # Region locality metrics
//!
//! Deterministic, LLM-free measurements that quantify *why* region-based
//! hydration should help an agent: a context window built from a causal
//! **region** is more **causally precise** than one built from the globally
//! highest-scoring nodes (CCOS v0.2's flat strategy), and it covers a task's
//! causal neighbourhood with fewer tokens.
//!
//! Ground truth is the **k-hop causal neighbourhood** of a target node
//! `N_k(t) = { v : d_causal(t, v) ≤ k }` (see [`causal_neighborhood`]). We then
//! compare two selection strategies at the region's own size budget:
//!
//! - **flat** — the top-|R| nodes by causal score (what v0.2 pages in);
//! - **region** — the members of the target's region `R`.
//!
//! and report precision `|S ∩ N_k| / |S|`, recall `|S ∩ N_k| / |N_k|`, and the
//! token cost to cover `N_k`. Everything is a pure function of the graph, so the
//! numbers are reproducible.

use crate::event_log::EventLog;
use crate::memory::{MemoryGraph, NodeId};
use crate::query::{impact_set, source_set};
use crate::region_engine::ContextRegionEngine;
use serde::Serialize;
use std::collections::BTreeSet;

/// Token cost charged per selected node (matches the scheduler/policy heuristic).
pub const TOKENS_PER_NODE: usize = 128;

/// The undirected k-hop causal neighbourhood of `target`: every node reachable
/// within `k` edges either downstream (impact) or upstream (causes), plus the
/// target itself. This is the "relevant set" `N_k(t)`.
pub fn causal_neighborhood(graph: &MemoryGraph, target: &str, k: u32) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    let origin = NodeId(target.to_string());
    set.insert(target.to_string());
    for r in impact_set(graph, &origin, k) {
        set.insert(r.id.0);
    }
    for r in source_set(graph, &origin, k) {
        set.insert(r.id.0);
    }
    set
}

/// Metrics for one selection strategy at a fixed node budget.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StrategyMetrics {
    /// Strategy name (`"flat"` or `"region"`).
    pub strategy: String,
    /// Number of nodes selected.
    pub nodes_selected: usize,
    /// Estimated token cost of the selection.
    pub tokens_estimated: usize,
    /// Selected nodes that fall in the causal neighbourhood.
    pub relevant_nodes: usize,
    /// `relevant / selected` — how on-topic the window is.
    pub causal_precision: f32,
    /// `relevant / |N_k|` — how much of the neighbourhood the window covers.
    pub causal_recall: f32,
}

/// Side-by-side locality comparison for one target node.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LocalityReport {
    /// The target node.
    pub target: String,
    /// Radius used for the neighbourhood ground truth.
    pub radius: u32,
    /// Size of the causal neighbourhood `|N_k(t)|`.
    pub neighborhood_size: usize,
    /// Flat (top-score) strategy at the region's size budget.
    pub flat: StrategyMetrics,
    /// Region (causal cluster) strategy.
    pub region: StrategyMetrics,
    /// Tokens the flat strategy must page in (scanning by score) to cover the
    /// whole neighbourhood `N_k`.
    pub flat_tokens_to_cover: usize,
    /// Tokens the region strategy pages in.
    pub region_tokens_to_cover: usize,
    /// `1 − region/flat` token saving to reach equal causal coverage (negative
    /// if the region is the more expensive option — reported honestly).
    pub token_saving_ratio: f32,
    /// `precision_region − precision_flat` at equal budget.
    pub precision_gain: f32,
}

fn metrics(
    strategy: &str,
    selected: &BTreeSet<String>,
    relevant: &BTreeSet<String>,
) -> StrategyMetrics {
    let hit = selected.intersection(relevant).count();
    let sel = selected.len();
    StrategyMetrics {
        strategy: strategy.to_string(),
        nodes_selected: sel,
        tokens_estimated: sel * TOKENS_PER_NODE,
        relevant_nodes: hit,
        causal_precision: if sel == 0 {
            0.0
        } else {
            hit as f32 / sel as f32
        },
        causal_recall: if relevant.is_empty() {
            0.0
        } else {
            hit as f32 / relevant.len() as f32
        },
    }
}

/// Build a [`LocalityReport`] comparing flat vs region selection for `target`,
/// using a `radius`-hop causal neighbourhood as ground truth. Returns `None` if
/// the target is not a node in the graph.
pub fn locality_report(graph: &MemoryGraph, target: &str, radius: u32) -> Option<LocalityReport> {
    if !graph.nodes.contains_key(&NodeId(target.to_string())) {
        return None;
    }
    let relevant = causal_neighborhood(graph, target, radius);

    // Region selection: the target's region members.
    let mut engine = ContextRegionEngine::new();
    let mut sink = EventLog::new("metrics".into());
    engine.initialize_regions(graph, &mut sink);
    let region_id = engine.region_of(target)?;
    let region: BTreeSet<String> = engine.regions[&region_id].members.iter().cloned().collect();
    let budget = region.len();

    // Flat selection: the top-|region| nodes by causal score (ties by id).
    let ranking: Vec<String> = graph
        .get_node_scores()
        .into_iter()
        .map(|(id, _)| id.0)
        .collect();
    let flat: BTreeSet<String> = ranking.iter().take(budget).cloned().collect();

    // Tokens the flat strategy must scan to cover the whole neighbourhood: the
    // deepest rank among relevant nodes (which are all eventually in the list).
    let flat_cover_nodes = relevant
        .iter()
        .filter_map(|r| ranking.iter().position(|x| x == r))
        .max()
        .map(|p| p + 1)
        .unwrap_or(0);
    let flat_tokens_to_cover = flat_cover_nodes * TOKENS_PER_NODE;
    let region_tokens_to_cover = region.len() * TOKENS_PER_NODE;

    let flat_m = metrics("flat", &flat, &relevant);
    let region_m = metrics("region", &region, &relevant);

    let token_saving_ratio = if flat_tokens_to_cover == 0 {
        0.0
    } else {
        1.0 - region_tokens_to_cover as f32 / flat_tokens_to_cover as f32
    };

    Some(LocalityReport {
        target: target.to_string(),
        radius,
        neighborhood_size: relevant.len(),
        precision_gain: region_m.causal_precision - flat_m.causal_precision,
        flat: flat_m,
        region: region_m,
        flat_tokens_to_cover,
        region_tokens_to_cover,
        token_saving_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{EdgeType, NodeType};

    /// A "decoy" graph: the target's causal cluster has modest scores, while a
    /// cluster of unrelated, high-access (high-score) nodes exists elsewhere.
    /// Flat selection is lured to the decoys; the region stays on-topic.
    fn decoy_graph() -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, 10_000);
        // Target cluster (file t.rs): t.rs + 3 symbols, internally linked.
        for id in ["file:t.rs", "sym:t.rs:a", "sym:t.rs:b", "sym:t.rs:c"] {
            g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
        }
        g.add_edge(
            "file:t.rs".into(),
            "sym:t.rs:a".into(),
            0.9,
            EdgeType::Contains,
        );
        g.add_edge(
            "sym:t.rs:a".into(),
            "sym:t.rs:b".into(),
            0.9,
            EdgeType::DependsOn,
        );
        g.add_edge(
            "sym:t.rs:b".into(),
            "sym:t.rs:c".into(),
            0.9,
            EdgeType::DependsOn,
        );
        // Decoy cluster (file d.rs): high access_count → high score, unrelated.
        for id in ["file:d.rs", "sym:d.rs:x", "sym:d.rs:y", "sym:d.rs:z"] {
            g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
            for _ in 0..50 {
                // Inflate access_count (and recency) to pump the causal score.
                g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
            }
        }
        g
    }

    #[test]
    fn region_beats_flat_on_causal_precision() {
        let g = decoy_graph();
        let report = locality_report(&g, "sym:t.rs:a", 3).expect("target exists");
        assert!(
            report.region.causal_precision >= report.flat.causal_precision,
            "region precision {:.2} must be >= flat precision {:.2}",
            report.region.causal_precision,
            report.flat.causal_precision
        );
        // The region is fully on-topic (its members are the causal cluster).
        assert!(report.region.causal_precision > 0.5);
    }

    #[test]
    fn neighborhood_is_deterministic() {
        let g = decoy_graph();
        let a = causal_neighborhood(&g, "sym:t.rs:a", 2);
        let b = causal_neighborhood(&g, "sym:t.rs:a", 2);
        assert_eq!(a, b);
        assert!(a.contains("sym:t.rs:a"));
    }

    #[test]
    fn report_is_reproducible() {
        let g = decoy_graph();
        let r1 = locality_report(&g, "sym:t.rs:b", 2);
        let r2 = locality_report(&g, "sym:t.rs:b", 2);
        assert_eq!(r1, r2, "locality report must be deterministic");
    }

    #[test]
    fn missing_target_yields_none() {
        let g = decoy_graph();
        assert!(locality_report(&g, "sym:ghost.rs:none", 2).is_none());
    }
}
