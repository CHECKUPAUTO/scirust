//! # Context Region Engine
//!
//! The engine sits *above* the causal [`MemoryGraph`] and *below* the LLM:
//!
//! ```text
//! Raw code → AST parser → MemoryGraph → ContextRegionEngine → LLM window
//! ```
//!
//! It clusters graph nodes into spatial [`ContextRegion`]s and, on a task,
//! **activates** the most relevant region — answering not "which file do I load?"
//! but "which zone of knowledge must be woken?". Activation yields a
//! [`ContextWindow`].
//!
//! ## Determinism
//!
//! Clustering is a pure function of the graph (file-keyed buckets merged by
//! connected components over cross-file edges, all in sorted order). Activation
//! advances a **logical clock**, never wall-clock time. Therefore a session
//! replays bit-for-bit: rebuild the graph from the event log, re-cluster, and
//! re-apply the recorded activations — see [`ContextRegionEngine::replay_from`].

use crate::context_policy::ContextPolicy;
use crate::context_region::{file_of, ContextRegion};
use crate::event_log::{EventLog, EventPayload, EventType};
use crate::memory::{MemoryGraph, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// The hydrated context handed to an agent/LLM: a region's files plus the
/// rationale and an estimated token cost.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextWindow {
    /// Id of the activated region.
    pub region: String,
    /// Distinct source files in the region, sorted.
    pub files: Vec<String>,
    /// Estimated token cost of the window.
    pub tokens_estimated: usize,
    /// The region's admission score under the active policy, in `[0, 1]`.
    pub region_score: f32,
    /// Human-readable reason the region was woken.
    pub reason: String,
}

/// What woke the engine.
#[derive(Debug, Clone)]
pub enum RegionQuery {
    /// Focus on the region owning a specific node (an edited or failing id).
    Node(String),
    /// No explicit target — wake the hottest region.
    Hottest,
}

/// Clusters the causal graph into spatial regions and activates them.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ContextRegionEngine {
    /// Regions keyed by id (sorted for deterministic iteration/output).
    pub regions: BTreeMap<String, ContextRegion>,
    /// Logical activation clock — deterministic, advanced on each activation.
    pub clock: u64,
}

/// Region key for a node: external dependencies collapse into one `external`
/// bucket; every other node is keyed by its owning file.
fn region_key(uri: &str) -> String {
    if uri.starts_with("dep:") {
        "external".to_string()
    } else {
        file_of(uri).to_string()
    }
}

impl ContextRegionEngine {
    /// Create an empty engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of regions currently held.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Group every graph node into a region id → members map. Nodes are bucketed
    /// by owning file, then file-buckets that are joined by a **cross-file edge**
    /// (a genuine structural/causal link, not a shared external dependency) are
    /// merged via connected components. Pure and deterministic.
    pub fn cluster_nodes(graph: &MemoryGraph) -> BTreeMap<String, Vec<String>> {
        // 1. Bucket nodes by file key.
        let mut buckets: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for id in graph.nodes.keys() {
            buckets
                .entry(region_key(&id.0))
                .or_default()
                .push(id.0.clone());
        }

        // 2. Build adjacency between (non-external) file keys via cross-file edges.
        let mut adj: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for key in buckets.keys() {
            adj.entry(key.clone()).or_default();
        }
        for e in &graph.edges {
            let (ks, kt) = (region_key(&e.source.0), region_key(&e.target.0));
            if ks != kt && ks != "external" && kt != "external" {
                adj.get_mut(&ks).unwrap().insert(kt.clone());
                adj.get_mut(&kt).unwrap().insert(ks.clone());
            }
        }

        // 3. Connected components over file keys (sorted BFS → deterministic).
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut clusters: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for start in buckets.keys() {
            if seen.contains(start) || start == "external" {
                continue;
            }
            // Collect the component containing `start`.
            let mut component: Vec<String> = Vec::new();
            let mut queue: VecDeque<String> = VecDeque::new();
            queue.push_back(start.clone());
            seen.insert(start.clone());
            while let Some(k) = queue.pop_front() {
                component.push(k.clone());
                for nb in &adj[&k] {
                    if seen.insert(nb.clone()) {
                        queue.push_back(nb.clone());
                    }
                }
            }
            component.sort();
            // Representative = smallest file key in the component.
            let rep = format!("region:{}", component[0]);
            let mut members: Vec<String> = Vec::new();
            for k in &component {
                if let Some(b) = buckets.get(k) {
                    members.extend(b.iter().cloned());
                }
            }
            members.sort();
            clusters.insert(rep, members);
        }

        // 4. External dependencies form their own cold region (if any).
        if let Some(ext) = buckets.get("external") {
            let mut members = ext.clone();
            members.sort();
            clusters.insert("region:external".to_string(), members);
        }

        clusters
    }

    /// Build (or rebuild) the region map purely from the graph — no events
    /// emitted. Used by both [`Self::initialize_regions`] and replay so they
    /// share an identical base state.
    fn cluster_and_build(&mut self, graph: &MemoryGraph) {
        self.regions.clear();
        for (id, members) in Self::cluster_nodes(graph) {
            let center = pick_center(graph, &members);
            let mut region = ContextRegion::new(id.clone(), center);
            region.members = members;
            region.recompute(graph);
            self.regions.insert(id, region);
        }
    }

    /// Initialise regions from the graph, emitting a `RegionCreated` event per
    /// region (and a `RegionMerged` event for every multi-file region) into
    /// `log`, so the session is fully reconstructable.
    pub fn initialize_regions(&mut self, graph: &MemoryGraph, log: &mut EventLog) {
        self.cluster_and_build(graph);
        for region in self.regions.values() {
            // A multi-file region is the result of merging file-buckets.
            let files = distinct_files(&region.members);
            if files.len() > 1 {
                log.append(
                    EventType::RegionMerged,
                    EventPayload::RegionMerged {
                        into_region: region.id.clone(),
                        from_region: files.join(","),
                        member_count: region.member_count(),
                    },
                );
            }
            log.append(
                EventType::RegionCreated,
                EventPayload::RegionCreated {
                    region_id: region.id.clone(),
                    center: region.center.clone(),
                    member_count: region.member_count(),
                    temperature: region.temperature,
                    causal_density: region.causal_density,
                },
            );
        }
    }

    /// Activate the best region for `query` under `policy`, returning the
    /// hydrated [`ContextWindow`] and emitting `RegionActivated` +
    /// `ContextWindowGenerated` events. The region is warmed and the logical
    /// clock advances.
    pub fn activate_region(
        &mut self,
        graph: &MemoryGraph,
        query: &RegionQuery,
        policy: &ContextPolicy,
        log: &mut EventLog,
    ) -> Option<ContextWindow> {
        let target_id = match query {
            RegionQuery::Node(uri) => self.region_of(uri)?,
            RegionQuery::Hottest => self.hottest_region()?,
        };

        let tick = self.clock;
        let reason = self.activation_reason(graph, &target_id);
        let region = self.regions.get_mut(&target_id)?;
        region.activate(tick);

        let files = distinct_files(&region.members);
        let region_score = policy.calculate_admission_score(region);
        let tokens = ContextPolicy::estimate_tokens(region);
        let activation_count = region.activation_count;
        let temperature = region.temperature;
        let region_id = region.id.clone();

        self.clock += 1;

        log.append(
            EventType::RegionActivated,
            EventPayload::RegionActivated {
                region_id: region_id.clone(),
                tick,
                temperature,
                activation_count,
                reason: reason.clone(),
            },
        );
        log.append(
            EventType::ContextWindowGenerated,
            EventPayload::ContextWindowGenerated {
                region_id: region_id.clone(),
                file_count: files.len(),
                tokens_estimated: tokens,
                region_score,
                reason: reason.clone(),
            },
        );

        Some(ContextWindow {
            region: region_id,
            files,
            tokens_estimated: tokens,
            region_score,
            reason,
        })
    }

    /// Cool every region by `decay` and evict any region whose temperature falls
    /// below `evict_below` (emitting `RegionEvicted`). Returns the evicted ids.
    pub fn tick_cooldown(
        &mut self,
        decay: f32,
        evict_below: f32,
        log: &mut EventLog,
    ) -> Vec<String> {
        for region in self.regions.values_mut() {
            region.cool(decay);
        }
        let evicted: Vec<String> = self
            .regions
            .iter()
            .filter(|(_, r)| r.temperature < evict_below)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &evicted {
            let temp = self.regions.get(id).map(|r| r.temperature).unwrap_or(0.0);
            self.regions.remove(id);
            log.append(
                EventType::RegionEvicted,
                EventPayload::RegionEvicted {
                    region_id: id.clone(),
                    temperature: temp,
                    reason: "temperature below eviction floor".to_string(),
                },
            );
        }
        evicted
    }

    /// The id of the region owning `uri`, if any.
    pub fn region_of(&self, uri: &str) -> Option<String> {
        self.regions
            .iter()
            .find(|(_, r)| r.contains(uri))
            .map(|(id, _)| id.clone())
    }

    /// The hottest region's id (max temperature, ties broken by id).
    pub fn hottest_region(&self) -> Option<String> {
        self.regions
            .values()
            .max_by(|a, b| {
                a.temperature
                    .partial_cmp(&b.temperature)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.id.cmp(&a.id)) // smaller id wins on a tie
            })
            .map(|r| r.id.clone())
    }

    /// Why a region is being woken — names failure propagation when any member
    /// carries failure pressure, else structural/causal recency.
    fn activation_reason(&self, graph: &MemoryGraph, region_id: &str) -> String {
        let Some(region) = self.regions.get(region_id) else {
            return "causal proximity".to_string();
        };
        let has_failure = region.members.iter().any(|m| {
            graph
                .nodes
                .get(&NodeId(m.clone()))
                .map(|n| n.failure_relevance > 0.0)
                .unwrap_or(false)
        });
        if has_failure {
            "failure propagation + causal proximity".to_string()
        } else {
            "causal proximity + recency".to_string()
        }
    }

    /// Reconstruct an engine from a rebuilt `graph` plus the `log` of region
    /// events: re-cluster (identical base state), then replay activations and
    /// evictions in order. Equivalent to the live engine that produced the log.
    pub fn replay_from(graph: &MemoryGraph, log: &EventLog) -> Self {
        let mut engine = Self::new();
        engine.cluster_and_build(graph);
        for ev in &log.events {
            match &ev.payload {
                EventPayload::RegionActivated {
                    region_id, tick, ..
                } => {
                    if let Some(r) = engine.regions.get_mut(region_id) {
                        r.activate(*tick);
                    }
                    engine.clock = engine.clock.max(tick + 1);
                }
                EventPayload::RegionEvicted { region_id, .. } => {
                    engine.regions.remove(region_id);
                }
                _ => {}
            }
        }
        engine
    }
}

/// Highest-score node among `members` (ties broken by id) — the region center.
fn pick_center(graph: &MemoryGraph, members: &[String]) -> String {
    members
        .iter()
        .max_by(|a, b| {
            let sa = graph
                .nodes
                .get(&NodeId((*a).clone()))
                .map(|n| graph.compute_node_score(n))
                .unwrap_or(0.0);
            let sb = graph
                .nodes
                .get(&NodeId((*b).clone()))
                .map(|n| graph.compute_node_score(n))
                .unwrap_or(0.0);
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.cmp(a))
        })
        .cloned()
        .unwrap_or_default()
}

/// Distinct source files referenced by a region's members, sorted.
fn distinct_files(members: &[String]) -> Vec<String> {
    let mut files: BTreeSet<String> = BTreeSet::new();
    for m in members {
        files.insert(file_of(m).to_string());
    }
    files.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{EdgeType, NodeType};

    /// A graph with two files (a.rs: x,y) and (b.rs: z), and an internal edge.
    fn two_file_graph() -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, 10_000);
        for id in [
            "file:a.rs",
            "sym:a.rs:x",
            "sym:a.rs:y",
            "file:b.rs",
            "sym:b.rs:z",
        ] {
            g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
        }
        g.add_edge(
            "file:a.rs".into(),
            "sym:a.rs:x".into(),
            0.6,
            EdgeType::Contains,
        );
        g.add_edge(
            "file:a.rs".into(),
            "sym:a.rs:y".into(),
            0.6,
            EdgeType::Contains,
        );
        g.add_edge(
            "file:b.rs".into(),
            "sym:b.rs:z".into(),
            0.6,
            EdgeType::Contains,
        );
        g
    }

    #[test]
    fn clusters_one_region_per_file() {
        let g = two_file_graph();
        let mut engine = ContextRegionEngine::new();
        let mut log = EventLog::new("t".into());
        engine.initialize_regions(&g, &mut log);
        // a.rs and b.rs are independent → two regions.
        assert!(engine.regions.contains_key("region:a.rs"));
        assert!(engine.regions.contains_key("region:b.rs"));
        assert_eq!(engine.region_count(), 2);
    }

    #[test]
    fn cross_file_edge_merges_regions() {
        let mut g = two_file_graph();
        // A genuine cross-file dependency a.rs::x → b.rs::z merges the regions.
        g.add_edge(
            "sym:a.rs:x".into(),
            "sym:b.rs:z".into(),
            0.7,
            EdgeType::DependsOn,
        );
        let mut engine = ContextRegionEngine::new();
        let mut log = EventLog::new("t".into());
        engine.initialize_regions(&g, &mut log);
        assert_eq!(engine.region_count(), 1, "linked files form one region");
        let region = engine.regions.values().next().unwrap();
        assert_eq!(distinct_files(&region.members), vec!["a.rs", "b.rs"]);
    }

    #[test]
    fn activation_yields_window_and_warms_region() {
        let g = two_file_graph();
        let mut engine = ContextRegionEngine::new();
        let mut log = EventLog::new("t".into());
        engine.initialize_regions(&g, &mut log);
        let policy = ContextPolicy::default();
        let win = engine
            .activate_region(
                &g,
                &RegionQuery::Node("sym:a.rs:x".into()),
                &policy,
                &mut log,
            )
            .expect("region exists");
        assert_eq!(win.region, "region:a.rs");
        assert!(win.files.contains(&"a.rs".to_string()));
        assert!(win.tokens_estimated > 0);
        assert_eq!(engine.regions["region:a.rs"].activation_count, 1);
    }
}
