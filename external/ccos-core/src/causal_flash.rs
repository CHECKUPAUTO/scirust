//! Causal Flash — a bounded, deterministic causal-cone context filter.
//!
//! At 10k+ nodes, recomputing a *global* structural signal (eigenvector
//! centrality, a full PageRank sweep) every tick is wasteful when the agent is
//! only working on a handful of nodes. Causal Flash instead computes relevance
//! on a **local cone** rooted at the active frontier — the graph analogue of
//! restricting attention to a window, but with an exact, non-probabilistic
//! definition rather than an approximation of the global measure.
//!
//! ## What it is (and the honest boundary)
//!
//! This is **not** flash-attention's exact tiling of a global op — a fixed
//! depth-`n` neighbourhood is a *truncation* of global centrality, so we do not
//! pretend to reproduce it. Instead the relevance we compute is **exactly** the
//! decayed dependency mass reachable *within the cone* — a well-defined
//! projection of the graph onto the cone, with no randomness and no reliance on
//! the rest of the graph. Completeness is reported honestly: the returned window
//! carries a [`CausalWindow::complete`] flag that is `true` **iff** the
//! dependency closure closed before the horizon (nothing was cut).
//!
//! ## Edge convention
//!
//! CCOS edges read `A → B` = *A depends on B* (see [`crate::memory`]). So a
//! node's **out-edges are its dependencies** (what it needs — required for the
//! LLM to *understand* it) and its **in-edges are its callers / dependents**
//! (what breaks if it changes — required for *impact*). Causal Flash follows
//! out-edges to build the dependency cone and adds a one-hop in-edge ring for
//! impact.
//!
//! ## Determinism (`replay == live`)
//!
//! Every step uses a total order on [`NodeId`]: seeds, frontier expansion, and
//! the emitted summary are all sorted, so the output never depends on
//! `HashMap` iteration order. The window is a pure read-only fold over the
//! resident graph. The adjacency index it uses is cached on the graph for speed
//! but is **runtime-only** (`#[serde(skip)]`, keyed on `edges.len()`, rebuilt
//! deterministically) — so snapshots stay byte-identical and a replay
//! reproduces the same window byte-for-byte.

use std::collections::{BTreeMap, BTreeSet};

use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeState};

/// Knobs for [`MemoryGraph::causal_flash_window`]. All read-only; nothing here
/// is persisted.
#[derive(Debug, Clone)]
pub struct CausalFlashConfig {
    /// Maximum dependency depth `n` from the seed frontier.
    pub horizon: usize,
    /// Also seed from low-trust nodes (a poisoned/suspect node and its cone are
    /// often exactly what the agent must review), not just `Working`.
    pub include_low_trust_seeds: bool,
    /// Seed a node when `include_low_trust_seeds` and `trust < trust_threshold`.
    pub trust_threshold: f64,
    /// Per-hop distance decay in `(0, 1]`; a dependency `k` hops away
    /// contributes `decay^k · (edge weights)` to relevance.
    pub decay: f64,
    /// Add the one-hop in-edge ring (callers/dependents) for impact context.
    pub include_callers: bool,
    /// Token budget: cap the summary at this many nodes. Truncation removes
    /// **callers only** (impact context), lowest-relevance first — the
    /// dependency closure is never dropped, so structural correctness is
    /// preserved. `None` = unbounded.
    pub max_nodes: Option<usize>,
}

impl Default for CausalFlashConfig {
    fn default() -> Self {
        Self {
            horizon: 3,
            include_low_trust_seeds: false,
            trust_threshold: 0.5,
            decay: 0.5,
            include_callers: true,
            max_nodes: None,
        }
    }
}

/// Why a node is in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalRole {
    /// A member of the active frontier (the query root).
    Seed,
    /// Reached by following dependency (out-) edges from the seeds.
    Dependency,
    /// A one-hop caller/dependent (in-edge) of a cone node — impact context.
    Caller,
}

/// One entry of the summary.
#[derive(Debug, Clone)]
pub struct CausalWindowNode {
    pub id: NodeId,
    pub state: NodeState,
    pub trust: f64,
    pub role: CausalRole,
    /// Dependency distance from the nearest seed (`0` for seeds and for callers
    /// attached to a seed).
    pub depth: usize,
    /// Decayed dependency mass reachable within the cone — the locally-exact
    /// relevance.
    pub relevance: f64,
}

/// The high-density causal summary: a dependency-distance-layered node list
/// plus honest completeness metadata.
#[derive(Debug, Clone)]
pub struct CausalWindow {
    /// Nodes ordered for reading: seeds first, then dependencies by increasing
    /// distance and decreasing relevance, then callers — a topological layering
    /// rooted at the working set. `NodeId` breaks every tie.
    pub nodes: Vec<CausalWindowNode>,
    /// `true` iff the dependency closure closed before the horizon — i.e. no
    /// dependency was cut. Callers dropped for the token budget do NOT clear
    /// this (they are impact context, not correctness).
    pub complete: bool,
    /// How many nodes were cut: dependencies beyond the horizon plus any callers
    /// dropped for `max_nodes`.
    pub omitted: usize,
    /// Number of seed nodes the window was rooted at.
    pub seed_count: usize,
}

/// Edge types that constitute a *directional dependency* for the cone. Matches
/// the "downstream dependency" set walked by failure/`do()` propagation; the
/// symmetric `RelatedTo` and the belief edges (`Supports`/`Contradicts`) carry
/// different semantics and are excluded.
fn is_dependency_edge(t: &EdgeType) -> bool {
    matches!(
        t,
        EdgeType::DependsOn
            | EdgeType::Contains
            | EdgeType::References
            | EdgeType::Causes
            | EdgeType::Calls
            | EdgeType::DataFlow
    )
}

/// A reusable dependency-adjacency index over the resident graph, built once
/// (`O(E)`) and shared across many [`MemoryGraph::causal_flash_window_with`]
/// calls so the hot path is `O(cone)` rather than `O(E)` per tick. Owns its
/// ids, so it outlives the borrow that built it and can be cached by the caller
/// and rebuilt only when edges change.
#[derive(Debug, Clone, Default)]
pub struct CausalAdjacency {
    /// `source → [(dependency, weight)]`, each list sorted by id.
    fwd: BTreeMap<NodeId, Vec<(NodeId, f64)>>,
    /// `target → [caller]`, each list sorted.
    rev: BTreeMap<NodeId, Vec<NodeId>>,
}

impl MemoryGraph {
    /// Build a reusable [`CausalAdjacency`] over the resident dependency edges.
    /// `O(E)`. Cache it and pass it to [`causal_flash_window_with`] on the hot
    /// path; rebuild only when the edge set changes.
    ///
    /// [`causal_flash_window_with`]: MemoryGraph::causal_flash_window_with
    pub fn causal_adjacency(&self) -> CausalAdjacency {
        let mut fwd: BTreeMap<NodeId, Vec<(NodeId, f64)>> = BTreeMap::new();
        let mut rev: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for e in self.edges() {
            if !is_dependency_edge(&e.edge_type) {
                continue;
            }
            fwd.entry(e.source.clone())
                .or_default()
                .push((e.target.clone(), e.weight));
            rev.entry(e.target.clone())
                .or_default()
                .push(e.source.clone());
        }
        for v in fwd.values_mut() {
            v.sort_by(|a, b| a.0.cmp(&b.0));
        }
        for v in rev.values_mut() {
            v.sort();
        }
        CausalAdjacency { fwd, rev }
    }

    /// Compute the [`CausalWindow`] for the current active frontier under `cfg`.
    ///
    /// The adjacency index is **cached on the graph** and rebuilt only when the
    /// edge set changes, so across the ticks between edits this is
    /// `O(N + cone)` — the `O(N)` `Working`-seed scan plus an `O(cone)` traversal
    /// over the cached index, no per-call `O(E)` re-index. That already beats the
    /// `O(iterations · (N + E))` power iteration of the global eigenvector
    /// centrality it replaces, and returns a bounded, token-budgeted window whose
    /// size is invariant to `N`. When the caller also tracks its own seed
    /// frontier, [`causal_flash_window_with`] with an explicit seed list is a
    /// full `O(cone)`.
    ///
    /// [`causal_flash_window_with`]: MemoryGraph::causal_flash_window_with
    pub fn causal_flash_window(&self, cfg: &CausalFlashConfig) -> CausalWindow {
        // Seeds: Working nodes (∪ low-trust when enabled), never Orphan.
        let mut seeds: BTreeSet<NodeId> = BTreeSet::new();
        for (id, node) in self.node_entries() {
            if node.state == NodeState::Orphan {
                continue;
            }
            let is_working = node.state == NodeState::Working;
            let is_low_trust = cfg.include_low_trust_seeds && node.trust < cfg.trust_threshold;
            if is_working || is_low_trust {
                seeds.insert(id.clone());
            }
        }
        self.with_causal_adjacency(|adj| self.cone_core(cfg, adj, seeds))
    }

    /// The `O(cone)` hot path: reuse a cached [`CausalAdjacency`] and pass the
    /// seed frontier explicitly (the agent already knows what it is working on),
    /// so neither the `O(E)` edge scan nor the `O(N)` `Working`-node scan is
    /// repeated. Seeds are taken as given (the `include_low_trust_seeds` /
    /// `trust_threshold` config is ignored here); non-resident or `Orphan` seed
    /// ids are dropped.
    pub fn causal_flash_window_with(
        &self,
        cfg: &CausalFlashConfig,
        adj: &CausalAdjacency,
        seed_ids: &[NodeId],
    ) -> CausalWindow {
        let mut seeds: BTreeSet<NodeId> = BTreeSet::new();
        for id in seed_ids {
            if self.node(id).is_some_and(|n| n.state != NodeState::Orphan) {
                seeds.insert(id.clone());
            }
        }
        self.cone_core(cfg, adj, seeds)
    }

    /// Shared cone traversal + assembly over a prebuilt adjacency and an explicit
    /// seed set. Deterministic: every structure is keyed/ordered by `NodeId`.
    fn cone_core(
        &self,
        cfg: &CausalFlashConfig,
        adj: &CausalAdjacency,
        seeds: BTreeSet<NodeId>,
    ) -> CausalWindow {
        let decay = cfg.decay.clamp(f64::MIN_POSITIVE, 1.0);
        let seed_count = seeds.len();

        let mut depth: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut relevance: BTreeMap<NodeId, f64> = BTreeMap::new();
        for s in &seeds {
            depth.insert(s.clone(), 0);
            relevance.insert(s.clone(), 1.0);
        }

        // Layered BFS along dependency edges, bounded by the horizon.
        let mut frontier: Vec<NodeId> = seeds.iter().cloned().collect();
        let mut cut_deps: BTreeSet<NodeId> = BTreeSet::new();
        for d in 1..=cfg.horizon {
            let mut next: BTreeSet<NodeId> = BTreeSet::new();
            for parent in &frontier {
                let prel = relevance[parent];
                if let Some(children) = adj.fwd.get(parent) {
                    for (child, w) in children {
                        // Orphan (dead/unreachable) nodes are excluded from the
                        // structural cone, exactly as they are from centrality.
                        if self.node(child).map(|x| x.state) == Some(NodeState::Orphan) {
                            continue;
                        }
                        let contrib = prel * decay * w;
                        *relevance.entry(child.clone()).or_insert(0.0) += contrib;
                        if !depth.contains_key(child) {
                            depth.insert(child.clone(), d);
                            next.insert(child.clone());
                        }
                    }
                }
            }
            frontier = next.into_iter().collect();
            if frontier.is_empty() {
                break; // closure reached a fixpoint before the horizon
            }
        }
        // Anything still on the frontier after the last layer is a cut dependency.
        if cfg.horizon > 0 {
            for parent in &frontier {
                if let Some(children) = adj.fwd.get(parent) {
                    for (child, _) in children {
                        // An excluded Orphan is not an omitted dependency.
                        if self.node(child).map(|x| x.state) == Some(NodeState::Orphan) {
                            continue;
                        }
                        if !depth.contains_key(child) {
                            cut_deps.insert(child.clone());
                        }
                    }
                }
            }
        }
        let complete = cut_deps.is_empty();

        // Assemble cone nodes (seed / dependency), skipping any id not resident.
        let mut out: Vec<CausalWindowNode> = Vec::new();
        for (id, &d) in &depth {
            let Some(node) = self.node(id) else {
                continue;
            };
            let role = if seeds.contains(id) {
                CausalRole::Seed
            } else {
                CausalRole::Dependency
            };
            out.push(CausalWindowNode {
                id: id.clone(),
                state: node.state,
                trust: node.trust,
                role,
                depth: d,
                relevance: relevance.get(id).copied().unwrap_or(0.0),
            });
        }

        // One-hop caller ring (impact), for cone nodes with in-edges. Deterministic
        // via the sorted reverse adjacency and a BTreeSet of already-included ids.
        let cone_ids: BTreeSet<NodeId> = depth.keys().cloned().collect();
        let mut caller_added: BTreeSet<NodeId> = BTreeSet::new();
        if cfg.include_callers {
            for cone in &cone_ids {
                if let Some(callers) = adj.rev.get(cone) {
                    let base = relevance.get(cone).copied().unwrap_or(0.0);
                    for caller in callers {
                        if cone_ids.contains(caller) || caller_added.contains(caller) {
                            continue;
                        }
                        let Some(node) = self.node(caller) else {
                            continue;
                        };
                        if node.state == NodeState::Orphan {
                            continue;
                        }
                        caller_added.insert(caller.clone());
                        out.push(CausalWindowNode {
                            id: caller.clone(),
                            state: node.state,
                            trust: node.trust,
                            role: CausalRole::Caller,
                            depth: depth[cone],
                            relevance: base * decay,
                        });
                    }
                }
            }
        }

        // Reading order: (role rank, depth asc, relevance desc, id asc) — seeds
        // first, then nearest/most-relevant dependencies, then callers.
        fn role_rank(r: CausalRole) -> u8 {
            match r {
                CausalRole::Seed => 0,
                CausalRole::Dependency => 1,
                CausalRole::Caller => 2,
            }
        }
        out.sort_by(|a, b| {
            role_rank(a.role)
                .cmp(&role_rank(b.role))
                .then(a.depth.cmp(&b.depth))
                .then(b.relevance.total_cmp(&a.relevance))
                .then(a.id.cmp(&b.id))
        });

        let mut omitted = cut_deps.len();

        // Token budget: drop callers (lowest relevance first) until within
        // `max_nodes`. Dependencies are never dropped.
        if let Some(cap) = cfg.max_nodes {
            if out.len() > cap {
                let mut callers: Vec<usize> = out
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.role == CausalRole::Caller)
                    .map(|(i, _)| i)
                    .collect();
                callers.sort_by(|&i, &j| {
                    out[i]
                        .relevance
                        .total_cmp(&out[j].relevance)
                        .then(out[i].id.cmp(&out[j].id))
                });
                let need_to_drop = out.len() - cap;
                let drop: BTreeSet<usize> = callers.into_iter().take(need_to_drop).collect();
                if !drop.is_empty() {
                    omitted += drop.len();
                    let mut kept = Vec::with_capacity(out.len() - drop.len());
                    for (i, n) in out.into_iter().enumerate() {
                        if !drop.contains(&i) {
                            kept.push(n);
                        }
                    }
                    out = kept;
                }
            }
        }

        CausalWindow {
            nodes: out,
            complete,
            omitted,
            seed_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NodeType;

    fn n(g: &mut MemoryGraph, id: &str) {
        g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Module);
    }
    fn dep(g: &mut MemoryGraph, a: &str, b: &str) {
        // a depends on b (a → b).
        g.add_edge(a.into(), b.into(), 1.0, EdgeType::DependsOn);
    }

    // Seed = Working; the dependency cone follows out-edges; Orphan excluded.
    #[test]
    fn cone_follows_dependencies_and_excludes_orphans() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "dep1", "dep2", "orphan", "unrelated"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "dep1");
        dep(&mut g, "dep1", "dep2");
        dep(&mut g, "w", "orphan");
        g.set_node_state(&"w".into(), NodeState::Working);
        g.set_node_state(&"orphan".into(), NodeState::Orphan);

        let win = g.causal_flash_window(&CausalFlashConfig::default());
        let ids: Vec<&str> = win.nodes.iter().map(|x| x.id.0.as_str()).collect();
        assert!(ids.contains(&"w"), "seed present");
        assert!(
            ids.contains(&"dep1") && ids.contains(&"dep2"),
            "deps reached"
        );
        assert!(
            !ids.contains(&"orphan"),
            "orphan excluded even as a dependency"
        );
        assert!(!ids.contains(&"unrelated"), "disconnected node excluded");
        assert_eq!(win.seed_count, 1);
        assert!(win.complete, "closure fits inside default horizon 3");
    }

    // Horizon truncation is reported: a dependency one hop past `n` is cut and
    // `complete` goes false with a matching `omitted`.
    #[test]
    fn horizon_bounds_the_cone_and_reports_incompleteness() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "a", "b", "c"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "a");
        dep(&mut g, "a", "b");
        dep(&mut g, "b", "c");
        g.set_node_state(&"w".into(), NodeState::Working);

        let cfg = CausalFlashConfig {
            horizon: 2,
            include_callers: false,
            ..Default::default()
        };
        let win = g.causal_flash_window(&cfg);
        let ids: Vec<&str> = win.nodes.iter().map(|x| x.id.0.as_str()).collect();
        assert!(ids.contains(&"a") && ids.contains(&"b"), "within horizon");
        assert!(!ids.contains(&"c"), "c is at depth 3 > horizon 2");
        assert!(!win.complete, "closure was cut at the horizon");
        assert_eq!(win.omitted, 1, "exactly c was omitted");
    }

    // Relevance decays with dependency distance.
    #[test]
    fn relevance_decays_with_depth() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "a", "b"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "a");
        dep(&mut g, "a", "b");
        g.set_node_state(&"w".into(), NodeState::Working);
        let win = g.causal_flash_window(&CausalFlashConfig {
            include_callers: false,
            ..Default::default()
        });
        let rel = |id: &str| win.nodes.iter().find(|x| x.id.0 == id).unwrap().relevance;
        assert!(rel("w") > rel("a") && rel("a") > rel("b"), "monotone decay");
    }

    // In-edges surface as Caller-role impact context.
    #[test]
    fn callers_appear_as_impact_context() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["caller", "w", "dep"] {
            n(&mut g, id);
        }
        dep(&mut g, "caller", "w"); // caller depends on w  ⇒ w's in-edge
        dep(&mut g, "w", "dep");
        g.set_node_state(&"w".into(), NodeState::Working);

        let win = g.causal_flash_window(&CausalFlashConfig::default());
        let caller = win.nodes.iter().find(|x| x.id.0 == "caller");
        assert!(caller.is_some(), "caller included");
        assert_eq!(caller.unwrap().role, CausalRole::Caller);
    }

    // Deterministic and insertion-order independent: two graphs built in
    // different orders yield byte-identical windows.
    #[test]
    fn window_is_deterministic_regardless_of_insertion_order() {
        let build = |order: &[&str]| {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            for id in order {
                n(&mut g, id);
            }
            dep(&mut g, "w", "a");
            dep(&mut g, "w", "b");
            dep(&mut g, "a", "c");
            g.set_node_state(&"w".into(), NodeState::Working);
            g.causal_flash_window(&CausalFlashConfig::default())
        };
        let w1 = build(&["w", "a", "b", "c"]);
        let w2 = build(&["c", "b", "a", "w"]);
        let ids1: Vec<(&str, usize)> = w1
            .nodes
            .iter()
            .map(|x| (x.id.0.as_str(), x.depth))
            .collect();
        let ids2: Vec<(&str, usize)> = w2
            .nodes
            .iter()
            .map(|x| (x.id.0.as_str(), x.depth))
            .collect();
        assert_eq!(ids1, ids2, "order-independent, deterministic emission");
    }

    // The token budget drops callers only — the dependency closure is preserved.
    #[test]
    fn budget_truncates_callers_not_dependencies() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "d1", "d2", "c1", "c2", "c3"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "d1");
        dep(&mut g, "d1", "d2");
        // three callers of w
        dep(&mut g, "c1", "w");
        dep(&mut g, "c2", "w");
        dep(&mut g, "c3", "w");
        g.set_node_state(&"w".into(), NodeState::Working);

        let cfg = CausalFlashConfig {
            max_nodes: Some(3),
            ..Default::default()
        };
        let win = g.causal_flash_window(&cfg);
        assert!(win.nodes.len() <= 3, "within budget");
        // The full dependency closure (w, d1, d2) must survive.
        for keep in ["w", "d1", "d2"] {
            assert!(
                win.nodes.iter().any(|x| x.id.0 == keep),
                "dependency {keep} must not be dropped"
            );
        }
        assert!(win.omitted >= 1, "some callers were dropped for the budget");
        assert!(
            win.nodes.iter().all(|x| x.role != CausalRole::Caller) || win.nodes.len() == 3,
            "callers are the only drop candidates"
        );
    }

    // The scale claim, made non-flaky: the cone's OUTPUT (and thus its cost) is
    // invariant to total graph size. The same local structure surrounded by 10
    // vs 3000 unrelated bulk nodes yields the identical window — a global
    // centrality pass would instead scale with the bulk.
    #[test]
    fn cone_is_invariant_to_total_graph_size() {
        let window_ids = |bulk: usize| -> Vec<String> {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            // Fixed local cone: w → d1 → d2, and a caller c → w.
            for id in ["w", "d1", "d2", "c"] {
                n(&mut g, id);
            }
            dep(&mut g, "w", "d1");
            dep(&mut g, "d1", "d2");
            dep(&mut g, "c", "w");
            g.set_node_state(&"w".into(), NodeState::Working);
            // Unrelated bulk: a disconnected dependency chain of Stable nodes.
            for i in 0..bulk {
                n(&mut g, &format!("bulk_{i}"));
                if i > 0 {
                    dep(&mut g, &format!("bulk_{i}"), &format!("bulk_{}", i - 1));
                }
            }
            let mut ids: Vec<String> = g
                .causal_flash_window(&CausalFlashConfig::default())
                .nodes
                .iter()
                .map(|x| x.id.0.clone())
                .collect();
            ids.sort();
            ids
        };
        assert_eq!(
            window_ids(10),
            window_ids(3000),
            "the cone must not grow with unrelated graph bulk"
        );
        // And it is exactly the local structure (w, d1, d2 + caller c).
        assert_eq!(window_ids(10), vec!["c", "d1", "d2", "w"]);
    }

    // The hot path (prebuilt adjacency + explicit seeds) yields the identical
    // window to the convenience path, so caching the index is a pure speedup.
    #[test]
    fn prebuilt_index_hot_path_matches_the_convenience_path() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "a", "b", "c"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "a");
        dep(&mut g, "a", "b");
        dep(&mut g, "c", "w");
        g.set_node_state(&"w".into(), NodeState::Working);

        let cfg = CausalFlashConfig::default();
        let convenience = g.causal_flash_window(&cfg);

        let adj = g.causal_adjacency();
        let hot = g.causal_flash_window_with(&cfg, &adj, &["w".into()]);

        let ids = |w: &CausalWindow| -> Vec<(String, usize)> {
            w.nodes.iter().map(|x| (x.id.0.clone(), x.depth)).collect()
        };
        assert_eq!(ids(&convenience), ids(&hot), "hot path == convenience path");
        assert_eq!(convenience.complete, hot.complete);
        assert_eq!(convenience.seed_count, hot.seed_count);
    }

    // The cached adjacency stays correct across edits: adding an edge changes
    // edges.len(), so the next window sees the new dependency.
    #[test]
    fn cached_adjacency_invalidates_when_edges_change() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "a", "b"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "a");
        g.set_node_state(&"w".into(), NodeState::Working);

        let cfg = CausalFlashConfig {
            include_callers: false,
            ..Default::default()
        };
        let ids = |g: &MemoryGraph| -> Vec<String> {
            let mut v: Vec<String> = g
                .causal_flash_window(&cfg)
                .nodes
                .iter()
                .map(|x| x.id.0.clone())
                .collect();
            v.sort();
            v
        };
        // First call builds and caches the index.
        assert_eq!(ids(&g), vec!["a", "w"]);
        // A new dependency edge must be reflected (cache keyed on edges.len()).
        dep(&mut g, "a", "b");
        assert_eq!(ids(&g), vec!["a", "b", "w"], "cache picked up the new edge");
    }

    // The runtime cache is serde(skip): computing a window never changes the
    // serialized snapshot, so `replay == live` and snapshot-hash invariants hold.
    #[test]
    fn window_computation_leaves_the_snapshot_byte_identical() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["w", "a"] {
            n(&mut g, id);
        }
        dep(&mut g, "w", "a");
        g.set_node_state(&"w".into(), NodeState::Working);

        let before = serde_json::to_string(&g).unwrap();
        let _ = g.causal_flash_window(&CausalFlashConfig::default());
        let after = serde_json::to_string(&g).unwrap();
        assert_eq!(
            before, after,
            "the adjacency cache must not enter the snapshot"
        );
    }

    // Low-trust seeding brings a suspect node (and its cone) into view even when
    // it is not Working.
    #[test]
    fn low_trust_seeding_is_opt_in() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for id in ["poison", "used"] {
            n(&mut g, id);
        }
        dep(&mut g, "poison", "used");
        if let Some(node) = g.node_mut(&"poison".into()) {
            node.trust = 0.1;
        }

        let off = g.causal_flash_window(&CausalFlashConfig::default());
        assert_eq!(
            off.seed_count, 0,
            "no Working, no low-trust seeding ⇒ empty"
        );

        let on = g.causal_flash_window(&CausalFlashConfig {
            include_low_trust_seeds: true,
            trust_threshold: 0.5,
            ..Default::default()
        });
        assert!(on.nodes.iter().any(|x| x.id.0 == "poison"), "poison seeded");
        assert!(
            on.nodes.iter().any(|x| x.id.0 == "used"),
            "its cone included"
        );
    }
}
