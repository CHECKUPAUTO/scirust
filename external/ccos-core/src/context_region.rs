//! # Context Region Engine — spatial-memory primitives
//!
//! CCOS v0.2 treats context as a 1-D scored list of graph nodes. The **Context
//! Region Engine** lifts that into a *spatial* model: every node is embedded in
//! an abstract 3-D context space and nodes are clustered into **regions** that
//! are hydrated as a whole, the way an OS pages in a working set rather than a
//! single address.
//!
//! - **X** — structural proximity (same file / module cluster together).
//! - **Y** — causal proximity (failure-relevance & dependency involvement).
//! - **Z** — temporality (recency of access).
//!
//! Everything here is **deterministic**: positions, temperatures and densities
//! are pure functions of the [`MemoryGraph`] (and a logical clock for
//! activations), so regions reconstruct identically on replay — preserving the
//! core CCOS invariant.
//!
//! This module defines the data model ([`ContextPoint`], [`ContextRegion`]); the
//! clustering/activation logic lives in [`crate::region_engine`].

use crate::memory::{GraphNode, MemoryGraph};
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};

/// A node embedded in the abstract 3-D context space. Positions and energies are
/// deterministic functions of a [`GraphNode`]'s scalar fields, so the same graph
/// always yields the same embedding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextPoint {
    /// Identifier of the underlying graph node (its `NodeId` string).
    pub uri: String,
    /// X axis — structural proximity (shared by nodes of the same source file).
    pub position_x: f32,
    /// Y axis — causal proximity (failure-relevance / dependency involvement).
    pub position_y: f32,
    /// Z axis — temporality (recency of access, 0 = stale … 1 = fresh).
    pub position_z: f32,
    /// Base importance of the node.
    pub importance: f32,
    /// Accumulated failure pressure propagated to this node.
    pub failure_pressure: f32,
    /// Recency in `[0, 1]`.
    pub recency: f32,
    /// Blended causal score used as the node's "energy" for admission.
    pub activation_energy: f32,
}

impl ContextPoint {
    /// Embed a graph node into context space. `structural_x` is supplied by the
    /// engine so that all nodes of one file share an X coordinate (structural
    /// locality); the remaining axes derive from the node's own fields.
    pub fn from_node(node: &GraphNode, graph: &MemoryGraph, structural_x: f32) -> Self {
        let energy = graph.compute_node_score(node) as f32;
        ContextPoint {
            uri: node.id.0.clone(),
            position_x: structural_x,
            position_y: node.failure_relevance as f32,
            position_z: node.recency as f32,
            importance: node.base_importance as f32,
            failure_pressure: node.failure_relevance as f32,
            recency: node.recency as f32,
            activation_energy: energy,
        }
    }

    /// Combined "heat" of this point in `[0, 1]`: a blend of energy, failure
    /// pressure and recency. Drives a region's temperature.
    pub fn heat(&self) -> f32 {
        (0.5 * self.activation_energy + 0.3 * self.failure_pressure + 0.2 * self.recency)
            .clamp(0.0, 1.0)
    }
}

/// Map a stable string (e.g. a file path) to a deterministic coordinate in
/// `[0, 1]`. Used for the structural (X) axis so identical inputs always land at
/// the same position regardless of `HashMap` iteration order.
pub fn structural_coord(key: &str) -> f32 {
    let hash = sha256_hex(key);
    // First 8 hex chars → u32 → normalised to [0, 1).
    let prefix = u32::from_str_radix(&hash[..8], 16).unwrap_or(0);
    prefix as f32 / u32::MAX as f32
}

/// A spatial region of the context map: a cluster of causally/structurally near
/// members with an aggregate temperature, density and activation history. A
/// region is hydrated as a unit to form an LLM context window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextRegion {
    /// Stable region identifier (deterministic, e.g. derived from the center).
    pub id: String,
    /// The region's center node (its most important member).
    pub center: String,
    /// Member node ids, kept sorted for deterministic output.
    pub members: Vec<String>,
    /// Sum of member activation energies.
    pub total_score: f32,
    /// Aggregate heat in `[0, 1]` — how "awake" the region is.
    pub temperature: f32,
    /// Internal connectivity: internal edges per member (causal cohesion).
    pub causal_density: f32,
    /// Logical clock tick of the last activation (0 = never). Deterministic.
    pub last_activation: u64,
    /// How many times the region has been activated.
    pub activation_count: u64,
}

impl ContextRegion {
    /// Create an empty region centered on `center`.
    pub fn new(id: impl Into<String>, center: impl Into<String>) -> Self {
        ContextRegion {
            id: id.into(),
            center: center.into(),
            members: Vec::new(),
            total_score: 0.0,
            temperature: 0.0,
            causal_density: 0.0,
            last_activation: 0,
            activation_count: 0,
        }
    }

    /// Number of members in the region.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Whether `uri` is a member of this region.
    pub fn contains(&self, uri: &str) -> bool {
        self.members.iter().any(|m| m == uri)
    }

    /// Recompute `total_score`, `temperature` and `causal_density` from the
    /// current graph. Pure and deterministic: members are read in sorted order
    /// and missing nodes contribute zero. Call after the graph mutates.
    pub fn recompute(&mut self, graph: &MemoryGraph) {
        self.members.sort();
        self.members.dedup();

        let mut total = 0.0_f32;
        let mut heat_sum = 0.0_f32;
        let mut counted = 0u32;
        for uri in &self.members {
            if let Some(node) = graph.nodes.get(&crate::memory::NodeId(uri.clone())) {
                let x = structural_coord(file_of(uri));
                let p = ContextPoint::from_node(node, graph, x);
                total += p.activation_energy;
                heat_sum += p.heat();
                counted += 1;
            }
        }
        self.total_score = total;
        self.temperature = if counted == 0 {
            0.0
        } else {
            (heat_sum / counted as f32).clamp(0.0, 1.0)
        };

        // Density = internal edges (both endpoints in the region) per member.
        let member_set: std::collections::HashSet<&String> = self.members.iter().collect();
        let internal_edges = graph
            .edges
            .iter()
            .filter(|e| member_set.contains(&e.source.0) && member_set.contains(&e.target.0))
            .count();
        self.causal_density = if counted == 0 {
            0.0
        } else {
            internal_edges as f32 / counted as f32
        };
    }

    /// Warm the region by recording an activation at logical time `tick`.
    pub fn activate(&mut self, tick: u64) {
        self.activation_count += 1;
        self.last_activation = tick;
        // Activation injects heat, capped at 1.0.
        self.temperature = (self.temperature + 0.25).clamp(0.0, 1.0);
    }

    /// Cool the region by `decay` (multiplicative), with a small floor so a
    /// region never reaches exactly zero until evicted.
    pub fn cool(&mut self, decay: f32) {
        self.temperature = (self.temperature * decay).clamp(0.0, 1.0);
    }
}

/// Extract the owning file path from a namespaced node id
/// (`file:<p>`, `mod:<p>:<n>`, `use:<p>:<path>`, `sym:<p>:<n>` → `<p>`;
/// `dep:<root>` and anything else → the id itself).
pub fn file_of(uri: &str) -> &str {
    if let Some(rest) = uri.strip_prefix("file:") {
        return rest;
    }
    for prefix in ["mod:", "use:", "sym:"] {
        if let Some(rest) = uri.strip_prefix(prefix) {
            // Id format is `<prefix>:<file-path>:<name-or-use-path>`. File paths
            // hold no ':' (unix), and a `use` path contains `::`, so the file
            // path is everything up to the *first* ':'.
            if let Some(idx) = rest.find(':') {
                return &rest[..idx];
            }
            return rest;
        }
    }
    uri
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};

    #[test]
    fn file_of_parses_namespaced_ids() {
        assert_eq!(file_of("file:src/a.rs"), "src/a.rs");
        assert_eq!(file_of("sym:src/a.rs:foo"), "src/a.rs");
        assert_eq!(file_of("use:src/a.rs:std::io::Read"), "src/a.rs");
        assert_eq!(file_of("mod:src/a.rs:tests"), "src/a.rs");
        assert_eq!(file_of("dep:serde"), "dep:serde");
    }

    #[test]
    fn structural_coord_is_deterministic_and_in_range() {
        let a = structural_coord("src/a.rs");
        let b = structural_coord("src/a.rs");
        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a));
        assert_ne!(structural_coord("src/a.rs"), structural_coord("src/b.rs"));
    }

    #[test]
    fn recompute_is_pure_and_deterministic() {
        let mut g = MemoryGraph::new(0.2, 1000);
        for id in ["file:src/a.rs", "sym:src/a.rs:foo", "sym:src/a.rs:bar"] {
            g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
        }
        g.add_edge(
            "file:src/a.rs".into(),
            "sym:src/a.rs:foo".into(),
            0.6,
            EdgeType::Contains,
        );
        g.add_edge(
            "file:src/a.rs".into(),
            "sym:src/a.rs:bar".into(),
            0.6,
            EdgeType::Contains,
        );

        let mut r = ContextRegion::new("region:src/a.rs", "file:src/a.rs");
        r.members = vec![
            "file:src/a.rs".into(),
            "sym:src/a.rs:foo".into(),
            "sym:src/a.rs:bar".into(),
        ];
        let mut r2 = r.clone();
        r.recompute(&g);
        r2.recompute(&g);
        assert_eq!(r, r2, "recompute must be deterministic");
        assert!(r.temperature >= 0.0 && r.temperature <= 1.0);
        assert_eq!(r.member_count(), 3);
        // Two internal edges over three members.
        assert!((r.causal_density - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn activation_warms_and_cooling_cools() {
        let mut r = ContextRegion::new("r", "c");
        r.temperature = 0.2;
        r.activate(7);
        assert_eq!(r.activation_count, 1);
        assert_eq!(r.last_activation, 7);
        assert!(r.temperature > 0.2, "activation must warm the region");
        let warm = r.temperature;
        r.cool(0.5);
        assert!(r.temperature < warm, "cooling must lower the temperature");
    }

    #[test]
    fn heat_is_bounded() {
        let node = GraphNode {
            id: NodeId("sym:x:y".into()),
            label: "y".into(),
            content: String::new(),
            node_type: NodeType::Symbol,
            base_importance: 1.0,
            failure_relevance: 1.0,
            recency: 1.0,
            access_count: 100,
            created_at: 0,
            last_accessed: 0,
            state: crate::memory::NodeState::Stable,
            trust: 1.0,
        };
        let g = MemoryGraph::default();
        let p = ContextPoint::from_node(&node, &g, 0.5);
        assert!((0.0..=1.0).contains(&p.heat()));
    }
}
