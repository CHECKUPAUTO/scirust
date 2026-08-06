//! # Context Scheduler (CCOS v0.3)
//!
//! Turns the [`MemoryGraph`] into a *paged* context
//! memory. Every node is placed in one of three zones by priority and a
//! configurable **token budget**:
//!
//! - **HOT** — loaded automatically into the working context (fits the budget).
//! - **WARM** — loaded on demand (within an extended budget).
//! - **COLD** — persisted only; not in the active context.
//!
//! The scheduler never *drops* a node — eviction only moves it to a colder
//! zone — so the working set is bounded while no information is lost.

use crate::memory::{MemoryGraph, NodeId};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Which tier of the paged context a node currently lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryZone {
    Hot,
    Warm,
    Cold,
}

/// Per-node scheduling metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledNode {
    pub node_id: NodeId,
    pub token_cost: usize,
    pub access_frequency: u64,
    pub last_access: u64,
    pub priority_score: f64,
    pub memory_zone: MemoryZone,
}

/// Outcome of an allocation/eviction pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AllocationStats {
    pub hot: usize,
    pub warm: usize,
    pub cold: usize,
    pub hot_tokens: usize,
    pub budget: usize,
}

impl AllocationStats {
    pub fn total(&self) -> usize {
        self.hot + self.warm + self.cold
    }
}

/// Paged context scheduler over a set of nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextScheduler {
    /// HOT token budget; the WARM tier extends this by `warm_ratio`.
    pub token_budget: usize,
    pub warm_ratio: f64,
    pub nodes: HashMap<NodeId, ScheduledNode>,
    pub clock: u64,
}

impl ContextScheduler {
    pub fn new(token_budget: usize) -> Self {
        Self {
            token_budget: token_budget.max(1),
            warm_ratio: 1.0,
            nodes: HashMap::new(),
            clock: 0,
        }
    }

    /// Estimate a token cost from text length (~4 chars per token, min 1).
    pub fn estimate_tokens(text: &str) -> usize {
        (text.chars().count() / 4).max(1)
    }

    /// Build a scheduler from a graph snapshot, deriving token cost from node
    /// content and priority from the graph's causal score, then allocate.
    pub fn from_graph(graph: &MemoryGraph, token_budget: usize) -> Self {
        let mut scheduler = Self::new(token_budget);
        for (id, node) in &graph.nodes {
            let token_cost =
                Self::estimate_tokens(&node.content) + Self::estimate_tokens(&node.label);
            scheduler.nodes.insert(
                id.clone(),
                ScheduledNode {
                    node_id: id.clone(),
                    token_cost,
                    access_frequency: node.access_count,
                    last_access: node.last_accessed,
                    priority_score: graph.compute_node_score(node),
                    memory_zone: MemoryZone::Cold,
                },
            );
        }
        scheduler.allocate_context();
        scheduler
    }

    /// Insert or update a node's scheduling metadata (does not re-allocate).
    pub fn upsert(&mut self, node_id: NodeId, token_cost: usize, priority_score: f64) {
        let now = self.clock;
        let entry = self.nodes.entry(node_id.clone()).or_insert(ScheduledNode {
            node_id,
            token_cost,
            access_frequency: 0,
            last_access: now,
            priority_score,
            memory_zone: MemoryZone::Cold,
        });
        entry.token_cost = token_cost;
        entry.priority_score = priority_score;
        entry.last_access = now;
    }

    /// Record an access to a node: advances the clock and bumps its frequency.
    pub fn touch(&mut self, node_id: &NodeId) {
        self.clock += 1;
        if let Some(n) = self.nodes.get_mut(node_id) {
            n.access_frequency += 1;
            n.last_access = self.clock;
        }
    }

    fn ordered_by_priority(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        ids.sort_by(|a, b| {
            let pa = self.nodes[a].priority_score;
            let pb = self.nodes[b].priority_score;
            pb.partial_cmp(&pa)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        ids
    }

    /// Partition every node into HOT / WARM / COLD by descending priority,
    /// filling the HOT budget first, then the WARM (extended) budget. Returns
    /// the resulting tier statistics. Deterministic (ties broken by node id).
    pub fn allocate_context(&mut self) -> AllocationStats {
        let warm_budget =
            self.token_budget + (self.token_budget as f64 * self.warm_ratio.max(0.0)) as usize;
        let mut used = 0usize;
        let mut stats = AllocationStats {
            budget: self.token_budget,
            ..Default::default()
        };

        for id in self.ordered_by_priority() {
            let cost = self.nodes[&id].token_cost;
            let zone = if used + cost <= self.token_budget {
                used += cost;
                stats.hot += 1;
                stats.hot_tokens = used;
                MemoryZone::Hot
            } else if used + cost <= warm_budget {
                used += cost;
                stats.warm += 1;
                MemoryZone::Warm
            } else {
                stats.cold += 1;
                MemoryZone::Cold
            };
            self.nodes.get_mut(&id).unwrap().memory_zone = zone;
        }
        stats
    }

    /// Demote the lowest-priority HOT nodes until the HOT tier fits the budget.
    /// Demoted nodes move to WARM (kept) — never discarded.
    pub fn evict_context(&mut self) -> usize {
        // Lowest priority first among HOT nodes.
        let mut hot: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.memory_zone == MemoryZone::Hot)
            .map(|(id, _)| id.clone())
            .collect();
        hot.sort_by(|a, b| {
            let pa = self.nodes[a].priority_score;
            let pb = self.nodes[b].priority_score;
            pa.partial_cmp(&pb)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(b))
        });

        let mut hot_tokens: usize = hot.iter().map(|id| self.nodes[id].token_cost).sum();
        let mut evicted = 0;
        for id in hot {
            if hot_tokens <= self.token_budget {
                break;
            }
            let cost = self.nodes[&id].token_cost;
            self.nodes.get_mut(&id).unwrap().memory_zone = MemoryZone::Warm;
            hot_tokens = hot_tokens.saturating_sub(cost);
            evicted += 1;
        }
        evicted
    }

    /// Re-pack the HOT tier to maximize retained priority per token (a density
    /// heuristic), then spill the rest to WARM/COLD. Returns the new stats.
    pub fn optimize_budget(&mut self) -> AllocationStats {
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        // Highest priority-per-token first.
        ids.sort_by(|a, b| {
            let da = self.nodes[a].priority_score / self.nodes[a].token_cost.max(1) as f64;
            let db = self.nodes[b].priority_score / self.nodes[b].token_cost.max(1) as f64;
            db.partial_cmp(&da)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(b))
        });

        let warm_budget =
            self.token_budget + (self.token_budget as f64 * self.warm_ratio.max(0.0)) as usize;
        let mut used = 0usize;
        let mut stats = AllocationStats {
            budget: self.token_budget,
            ..Default::default()
        };
        for id in ids {
            let cost = self.nodes[&id].token_cost;
            let zone = if used + cost <= self.token_budget {
                used += cost;
                stats.hot += 1;
                stats.hot_tokens = used;
                MemoryZone::Hot
            } else if used + cost <= warm_budget {
                stats.warm += 1;
                MemoryZone::Warm
            } else {
                stats.cold += 1;
                MemoryZone::Cold
            };
            self.nodes.get_mut(&id).unwrap().memory_zone = zone;
        }
        stats
    }

    /// Node ids currently in the HOT tier (the active context window).
    pub fn hot_context(&self) -> Vec<NodeId> {
        self.zone(MemoryZone::Hot)
    }
    pub fn warm_context(&self) -> Vec<NodeId> {
        self.zone(MemoryZone::Warm)
    }
    pub fn cold_context(&self) -> Vec<NodeId> {
        self.zone(MemoryZone::Cold)
    }

    fn zone(&self, zone: MemoryZone) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.memory_zone == zone)
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        ids
    }

    pub fn hot_token_usage(&self) -> usize {
        self.nodes
            .values()
            .filter(|n| n.memory_zone == MemoryZone::Hot)
            .map(|n| n.token_cost)
            .sum()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NodeType;

    fn scheduler_with(costs_priorities: &[(usize, f64)], budget: usize) -> ContextScheduler {
        let mut s = ContextScheduler::new(budget);
        s.warm_ratio = 0.5;
        for (i, (cost, prio)) in costs_priorities.iter().enumerate() {
            s.upsert(NodeId(format!("n{i}")), *cost, *prio);
        }
        s.allocate_context();
        s
    }

    #[test]
    fn test_priority_respected_in_hot() {
        // Budget fits ~2 nodes of cost 10. Highest priorities must be HOT.
        let s = scheduler_with(&[(10, 0.1), (10, 0.9), (10, 0.8), (10, 0.2)], 20);
        let hot = s.hot_context();
        assert_eq!(hot.len(), 2);
        // n1 (0.9) and n2 (0.8) are the two highest priorities.
        assert!(hot.contains(&NodeId("n1".into())));
        assert!(hot.contains(&NodeId("n2".into())));
    }

    #[test]
    fn test_hot_within_budget() {
        let s = scheduler_with(&[(10, 0.9), (10, 0.8), (10, 0.7), (10, 0.6)], 25);
        assert!(s.hot_token_usage() <= 25, "hot tier must fit the budget");
    }

    #[test]
    fn test_no_node_lost_across_zones() {
        let s = scheduler_with(&[(8, 0.9), (8, 0.5), (8, 0.4), (8, 0.3), (8, 0.1)], 16);
        let total = s.hot_context().len() + s.warm_context().len() + s.cold_context().len();
        assert_eq!(total, 5, "every node must remain in exactly one zone");
    }

    #[test]
    fn test_eviction_after_budget_shrink() {
        let mut s = scheduler_with(&[(10, 0.9), (10, 0.8), (10, 0.7)], 30);
        assert_eq!(s.hot_context().len(), 3);
        // Shrink the budget and evict: the HOT tier must contract, lost nodes
        // are demoted to WARM, not dropped.
        s.token_budget = 10;
        let evicted = s.evict_context();
        assert_eq!(evicted, 2);
        assert!(s.hot_token_usage() <= 10);
        assert_eq!(s.len(), 3, "no node lost during eviction");
    }

    #[test]
    fn test_from_graph_builds_and_allocates() {
        let mut graph = MemoryGraph::default();
        for i in 0..10 {
            graph.upsert_node(
                NodeId(format!("n{i}")),
                format!("label{i}"),
                "some content here for tokens".into(),
                NodeType::Symbol,
            );
        }
        let s = ContextScheduler::from_graph(&graph, 30);
        assert_eq!(s.len(), 10);
        assert_eq!(
            s.hot_context().len() + s.warm_context().len() + s.cold_context().len(),
            10
        );
    }

    #[test]
    fn test_optimize_budget_is_deterministic() {
        let mut a = scheduler_with(&[(5, 0.5), (10, 0.9), (3, 0.4), (7, 0.7)], 12);
        let mut b = scheduler_with(&[(5, 0.5), (10, 0.9), (3, 0.4), (7, 0.7)], 12);
        let sa = a.optimize_budget();
        let sb = b.optimize_budget();
        assert_eq!(sa, sb);
        assert!(a.hot_token_usage() <= 12);
    }
}
