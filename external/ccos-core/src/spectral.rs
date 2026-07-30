//! Deterministic **eigenvector centrality** over the causal-memory graph.
//!
//! This is the foundational, fully-verifiable first slice of the "spectral"
//! direction (ROADMAP #13: *eigenvector centrality → spectral regions → temporal
//! tensor*). It computes the classic eigenvector-centrality ranking — the dominant
//! eigenvector of the adjacency, `A x = λ x` — by **power iteration**, as a clean,
//! self-contained, dependency-free primitive. Nothing here is wired into the CLI,
//! the scoring path, or the snapshot; it is a pure read-only ranking signal you can
//! call on any [`MemoryGraph`]. The richer pipeline
//! (spectral regions, the temporal tensor, any `scirust` fusion) is **deliberately
//! deferred** to a later design pass and adds no new dependency here.
//!
//! ## Relationship to [`MemoryGraph::eigencentrality`]
//!
//! [`MemoryGraph::eigencentrality`] is the
//! *damped, directed* Katz / PageRank realization (`x ← (1−d)/N + d·Aᵀx`), which stays
//! well-defined on a directed, largely-acyclic code graph. [`eigenvector_centrality`]
//! is the **undirected, undamped** textbook form: it symmetrizes the adjacency and runs
//! pure power iteration to the Perron eigenvector. The two are complementary signals; this
//! module keeps the textbook spectral form isolated so the deferred spectral work can build
//! on a clean `A x = λ x` brick rather than the damped operator.
//!
//! ## Method, and why these choices
//!
//! * **Symmetrization.** Each edge `s → t` is treated as an *undirected* connection:
//!   it contributes to both `A[s][t]` and `A[t][s]` (a parallel edge in either
//!   direction just adds mass). A code dependency graph is largely a **DAG**, so the
//!   *directed* adjacency has spectral radius `0` and a degenerate eigenvector — power
//!   iteration on it collapses to zero. The symmetric adjacency is a non-negative
//!   **symmetric** matrix, so by Perron–Frobenius each connected component has a unique
//!   positive dominant eigenvector, and power iteration converges to it. This is the
//!   standard realization of eigenvector centrality on an otherwise-acyclic graph.
//! * **Spectral shift (`A + I`).** Power iteration is run on `A + I` — a unit self-loop on
//!   every node — not on `A` directly. A symmetrized path / tree is **bipartite**, so its
//!   eigenvalues come in `±` pairs about `0` and plain power iteration on `A` *oscillates*
//!   between the dominant pair without converging. The shift makes every eigenvalue `λ + 1`,
//!   so the largest is strictly positive and unique and the iteration converges monotonically;
//!   crucially `A` and `A + I` share the **same eigenvectors**, so the returned ordering is
//!   exactly that of the adjacency's dominant eigenvector. (This is the same shift the
//!   deferred spectral-region work would use on the Laplacian.)
//! * **Normalization.** The vector is normalized to unit **L2** (Euclidean) norm —
//!   the conventional normalization for an eigenvector (`‖x‖₂ = 1`). It is also the
//!   quantity power iteration naturally preserves, and unlike L1 it never degenerates
//!   when a component carries little mass.
//! * **Determinism.** Node ids are processed in **sorted** order and edges are
//!   accumulated in a **canonical sorted order**, so the floating-point summation order
//!   is a pure function of the graph's *structure*, independent of `HashMap` iteration
//!   or edge-insertion order. Iteration stops at a fixed L2 **tolerance**
//!   ([`CONVERGENCE_TOL`]) or a hard **iteration cap** ([`MAX_ITERS`]), whichever comes
//!   first, so it **always terminates** with byte-identical output run-to-run. The
//!   initial vector is the **uniform** vector `1/√n` (no randomness). The Perron vector
//!   is oriented **non-negative** by a deterministic sign convention, so the sign never
//!   flips between runs.
//!
//! ## Edge cases (all well-defined, never `NaN`/panic)
//!
//! * **Empty graph** → empty map.
//! * **Edgeless graph** (nodes but no edges) → the adjacency is the zero matrix and the
//!   dominant eigenvector is undefined; we return the natural limit, the **uniform**
//!   value `1/√n` for every node (all equal), matching the convention that with no
//!   structure no node is privileged.
//! * **Isolated / dangling node** (degree 0 in a graph that *does* have edges) → score
//!   **`0.0`**: nothing feeds it, so its mass decays and renormalization on the connected
//!   structure leaves it at zero.
//! * **Disconnected components** → handled transparently: power iteration runs over the
//!   whole symmetric adjacency at once, and the single global L2 normalization ranks the
//!   component with the largest dominant eigenvalue highest (each node still gets a
//!   finite, deterministic score).
//!
//! `Orphan` nodes (dead code, [`NodeState::Orphan`]) are
//! excluded from the structural graph — consistent with how the rest of CCOS treats
//! topology — so they receive no entry in the result and their edges drop out.
//!
//! ```
//! use ccos_core::memory::{EdgeType, MemoryGraph, NodeType};
//! use ccos_core::spectral::eigenvector_centrality;
//!
//! let mut g = MemoryGraph::new(0.2, usize::MAX);
//! for n in ["hub", "a", "b", "c"] {
//!     g.upsert_node(n.into(), n.into(), String::new(), NodeType::Module);
//! }
//! for leaf in ["a", "b", "c"] {
//!     g.add_edge(leaf.into(), "hub".into(), 1.0, EdgeType::DependsOn);
//! }
//! let c = eigenvector_centrality(&g);
//! // The hub the leaves connect to is the most central node.
//! assert!(c[&"hub".into()] > c[&"a".into()]);
//! ```

use crate::memory::{MemoryGraph, NodeId, NodeState};
use std::collections::BTreeMap;

/// L2 convergence tolerance: power iteration stops once the L2 distance between two
/// successive (normalized) iterates falls below this.
pub const CONVERGENCE_TOL: f64 = 1e-12;

/// Hard cap on power-iteration steps, so the routine **always** terminates
/// deterministically even when the tolerance is never reached.
pub const MAX_ITERS: usize = 1000;

/// Eigenvector centrality of every (non-[`Orphan`](NodeState::Orphan)) node, computed
/// by deterministic **power iteration** on the **symmetrized** adjacency
/// (`A x = λ x`), L2-normalized.
///
/// Returns a [`BTreeMap`] keyed by [`NodeId`] so iteration / equality is stable and
/// results are byte-identical run-to-run. See the [module docs](crate::spectral) for
/// the symmetrization, normalization, and edge-case (empty / edgeless / isolated /
/// disconnected) choices. Pure and side-effect-free: it borrows the graph read-only and
/// never mutates it, the snapshot, or any cache.
///
/// # Examples
///
/// ```
/// use ccos_core::memory::MemoryGraph;
/// use ccos_core::spectral::eigenvector_centrality;
///
/// // An empty graph yields an empty ranking.
/// let g = MemoryGraph::new(0.2, usize::MAX);
/// assert!(eigenvector_centrality(&g).is_empty());
/// ```
pub fn eigenvector_centrality(graph: &MemoryGraph) -> BTreeMap<NodeId, f64> {
    // Stable node ordering ⇒ deterministic float accumulation. Orphan (dead) nodes are
    // excluded from the structural graph, exactly as the rest of CCOS treats topology.
    let mut ids: Vec<&NodeId> = graph
        .node_entries()
        .filter(|(_, node)| node.state != NodeState::Orphan)
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    let n = ids.len();
    if n == 0 {
        return BTreeMap::new();
    }

    // id → dense index, over the sorted id list.
    let index: std::collections::HashMap<&NodeId, usize> =
        ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    // Build the **symmetrized** adjacency as a sorted edge list of (i, j) index pairs,
    // each undirected edge contributing both orientations. Self-loops are kept (they add
    // to the node's own diagonal, the standard treatment). Edges whose endpoints are not
    // both in `index` (e.g. an endpoint is Orphan) are skipped — they are not part of the
    // structural graph.
    let mut adj: Vec<(usize, usize)> = Vec::with_capacity(graph.edges().len() * 2);
    let mut degree = vec![0u64; n];
    for e in graph.edges() {
        if let (Some(&s), Some(&t)) = (index.get(&e.source), index.get(&e.target)) {
            adj.push((s, t));
            adj.push((t, s));
            degree[s] += 1;
            degree[t] += 1;
        }
    }
    // Canonical order ⇒ the accumulation below is invariant to edge-insertion order, so
    // the result is a pure function of the graph's structure, identical across processes.
    adj.sort_unstable();

    // Edgeless graph: the adjacency is the zero matrix, whose dominant eigenvector is
    // undefined. Return the natural limit — the uniform unit-L2 vector (all nodes equal),
    // i.e. with no structure no node is privileged.
    if adj.is_empty() {
        let uniform = 1.0 / (n as f64).sqrt();
        return ids.into_iter().map(|id| (id.clone(), uniform)).collect();
    }

    // Power iteration from the uniform unit vector on **`A + I`** (a unit self-loop on
    // every node), `x` kept L2-normalized each step; stop when the iterate barely moves
    // (TOL) or at the hard cap (MAX_ITERS). The `+ I` spectral shift is essential, not
    // cosmetic: a symmetrized path/tree adjacency is **bipartite**, whose eigenvalues come
    // in ± pairs about 0, so plain power iteration on `A` *oscillates* between the dominant
    // pair and never settles. Shifting to `A + I` makes every eigenvalue `λ + 1`, so the
    // largest is strictly positive and unique (Perron) — convergence is monotone — while
    // the **eigenvectors are unchanged** (`A` and `A + I` share them), so the centrality
    // *ordering* this returns is exactly that of the adjacency's dominant eigenvector. (A
    // degree-0 node's `A + I` eigenvalue is `1`, strictly below any edged component's, so it
    // already decays toward `0`; we additionally pin it to *exactly* `0` after the loop so
    // the "isolated ⇒ 0" contract is bit-exact, not merely "vanishingly small".)
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..MAX_ITERS {
        // y = (A + I)·x: seed with x (the +I diagonal), then add the symmetric adjacency
        // contributions in sorted order (⇒ deterministic summation order).
        let mut y = x.clone();
        for &(i, j) in &adj {
            y[i] += x[j];
        }
        // Re-normalize to unit L2.
        let norm = l2_norm(&y);
        if norm == 0.0 {
            // Unreachable while `x` is non-zero (the `+ I` term copies `x` into `y`), but
            // guard against a zero vector rather than ever dividing by zero.
            break;
        }
        for v in &mut y {
            *v /= norm;
        }
        // Converged once the update barely changes the vector.
        let delta = l2_distance(&x, &y);
        x = y;
        if delta < CONVERGENCE_TOL {
            break;
        }
    }

    // Pin every degree-0 (isolated / dangling) node to **exactly** `0.0`: it carries no
    // edge, so its eigenvector-centrality support is empty. Iteration already drives it to a
    // vanishing residual; zeroing makes the "isolated ⇒ 0" contract bit-exact and removes the
    // tolerance-stop residual from the result.
    for (i, &deg) in degree.iter().enumerate() {
        if deg == 0 {
            x[i] = 0.0;
        }
    }
    // Re-normalize to unit L2 over the surviving (connected) structure, so zeroing the
    // isolates does not leave the vector slightly under unit norm. `adj` is non-empty here,
    // so at least one connected node is non-zero and the norm is positive.
    let norm = l2_norm(&x);
    if norm > 0.0 {
        for v in &mut x {
            *v /= norm;
        }
    }

    // Orient the Perron eigenvector non-negative by a deterministic sign convention: the
    // sum of components is non-negative. (Power iteration on a non-negative matrix already
    // converges to the non-negative Perron vector, but pinning the sign keeps output stable
    // against any floating-point sign drift.)
    if x.iter().sum::<f64>() < 0.0 {
        for v in &mut x {
            *v = -*v;
        }
    }

    ids.into_iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), x[i]))
        .collect()
}

/// L2 (Euclidean) norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// L2 distance between two equal-length vectors.
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal profile — Θ[claim, {Belief, Tension}, t] (the belief "fever curve")
// ─────────────────────────────────────────────────────────────────────────────

/// One claim's belief state at one time step — the two thermodynamic components CCOS tracks over
/// time: the signed **belief** ∈ [−1, 1] (direction + strength) and the **tension**
/// (`QBelief.conflict`) ∈ [0, 1] (how strongly and evenly the claim is contested).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeliefTension {
    /// Signed support fraction in `[−1, 1]`.
    pub belief: f64,
    /// Geometric evidence balance in `[0, 1]` — the contested-ness / "temperature" of the claim.
    pub tension: f64,
}

/// The **temporal-profile tensor** `Θ[claim, {Belief, Tension}, t]`: for a set of claims, their
/// belief/tension trajectory across an ordered sequence of graph states (time steps). This is the
/// *dynamic*, conflict-resolution reading of the graph — how belief and tension **evolve** as
/// evidence is injected, propagated, and decayed — rather than a static structural ranking. It is the
/// productionized form of the `temporal_tensor_crux` measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalProfile {
    /// The tracked claims, in the order their columns appear within each frame.
    pub claims: Vec<NodeId>,
    /// `frames[t][i]` is claim `claims[i]`'s belief/tension at time step `t`.
    pub frames: Vec<Vec<BeliefTension>>,
}

impl TemporalProfile {
    /// Number of time steps recorded.
    pub fn steps(&self) -> usize {
        self.frames.len()
    }

    /// `claim`'s **tension** trajectory over time (empty when the claim is not tracked).
    pub fn tension_series(&self, claim: &NodeId) -> Vec<f64> {
        match self.claims.iter().position(|c| c == claim) {
            Some(i) => self.frames.iter().map(|f| f[i].tension).collect(),
            None => Vec::new(),
        }
    }

    /// `claim`'s **belief** trajectory over time (empty when the claim is not tracked).
    pub fn belief_series(&self, claim: &NodeId) -> Vec<f64> {
        match self.claims.iter().position(|c| c == claim) {
            Some(i) => self.frames.iter().map(|f| f[i].belief).collect(),
            None => Vec::new(),
        }
    }

    /// **System temperature** — the mean tension across all tracked claims at each time step, the
    /// aggregate "fever curve" of the knowledge base. Empty when no claims are tracked.
    pub fn temperature(&self) -> Vec<f64> {
        if self.claims.is_empty() {
            return Vec::new();
        }
        let n = self.claims.len() as f64;
        self.frames
            .iter()
            .map(|f| f.iter().map(|bt| bt.tension).sum::<f64>() / n)
            .collect()
    }

    /// The peak system temperature (max mean-tension over time); `0.0` when empty.
    pub fn peak_temperature(&self) -> f64 {
        self.temperature().into_iter().fold(0.0_f64, f64::max)
    }

    /// **Retrodicted belief** for `claim`: the raw [`belief_series`](Self::belief_series) run through
    /// a deterministic RTS Kalman smoother ([`crate::retrodict::rts_smooth`]), so each past step's
    /// belief is reconstructed with the minimum-variance estimate given *all* steps — including
    /// future ones. Where the raw fever-curve shows what the engine believed moment-to-moment, this
    /// shows what it *should* have believed at each moment given hindsight — the retrodiction a
    /// stateless retriever cannot phrase. `q`/`r` are the process/measurement variances (drift speed
    /// vs reading noise). Empty when the claim is not tracked.
    pub fn retrodicted_belief(&self, claim: &NodeId, q: f64, r: f64) -> Vec<f64> {
        crate::retrodict::rts_smooth(&self.belief_series(claim), q, r)
    }

    /// **Retrodicted tension** for `claim` — the RTS-smoothed [`tension_series`](Self::tension_series),
    /// the dual of [`retrodicted_belief`](Self::retrodicted_belief). Empty when the claim is not
    /// tracked.
    pub fn retrodicted_tension(&self, claim: &NodeId, q: f64, r: f64) -> Vec<f64> {
        crate::retrodict::rts_smooth(&self.tension_series(claim), q, r)
    }
}

/// Build the [`TemporalProfile`] `Θ[claim, {Belief, Tension}, t]` for `claims` across an ordered
/// sequence of graph states — one graph per time step (e.g. successive
/// [`AgentSession::replay_to`](crate::agent_session::AgentSession::replay_to) states, or snapshots of
/// a scripted scenario). Each cell is the claim's [`QBelief`](crate::memory::QBelief) at that step:
/// `belief` and `conflict` (tension). `half_life > 0` applies the knowledge-half-life decay
/// ([`qbelief_decayed`](MemoryGraph::qbelief_decayed)), so relaxation over time is captured;
/// `half_life <= 0` uses the plain, undecayed belief. Pure and deterministic.
pub fn temporal_profile<'a>(
    graphs: impl IntoIterator<Item = &'a MemoryGraph>,
    claims: &[NodeId],
    half_life: f64,
) -> TemporalProfile {
    let frames = graphs
        .into_iter()
        .map(|g| {
            claims
                .iter()
                .map(|c| {
                    let q = g.qbelief_decayed(c, half_life);
                    BeliefTension {
                        belief: q.belief,
                        tension: q.conflict,
                    }
                })
                .collect()
        })
        .collect();
    TemporalProfile {
        claims: claims.to_vec(),
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{EdgeType, MemoryGraph, NodeType};

    /// Build a graph directly (no parser) from node ids and directed edges.
    fn graph(nodes: &[&str], edges: &[(&str, &str)]) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in nodes {
            g.upsert_node((*n).into(), (*n).into(), String::new(), NodeType::Module);
        }
        for &(s, t) in edges {
            g.add_edge(s.into(), t.into(), 1.0, EdgeType::DependsOn);
        }
        g
    }

    fn score(c: &BTreeMap<NodeId, f64>, id: &str) -> f64 {
        *c.get(&id.into()).expect("node present in centrality map")
    }

    /// Hand-computable graph: a path `a — b — c`. By symmetry the middle node `b`
    /// (degree 2) must dominate the two endpoints (degree 1), and the endpoints must
    /// tie. This pins the dominant-eigenvector ordering on a graph whose answer is
    /// obvious by inspection.
    #[test]
    fn path_graph_orders_middle_above_endpoints() {
        let c = eigenvector_centrality(&graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]));
        assert_eq!(c.len(), 3);
        assert!(
            score(&c, "b") > score(&c, "a"),
            "the degree-2 middle node dominates an endpoint"
        );
        assert!(
            score(&c, "b") > score(&c, "c"),
            "the degree-2 middle node dominates the other endpoint"
        );
        assert!(
            (score(&c, "a") - score(&c, "c")).abs() < 1e-9,
            "the two symmetric endpoints have equal centrality"
        );
        // The vector is L2-normalized.
        let norm: f64 = c.values().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "result is unit-L2 normalized");
        // All scores are finite and non-negative (Perron vector).
        assert!(c.values().all(|v| v.is_finite() && *v >= 0.0));
    }

    /// Star graph: a center connected to many leaves. The center must have strictly the
    /// highest centrality, and (by symmetry) all leaves tie below it.
    #[test]
    fn star_graph_center_is_most_central() {
        let leaves = ["l0", "l1", "l2", "l3", "l4"];
        let mut nodes = vec!["center"];
        nodes.extend_from_slice(&leaves);
        let edges: Vec<(&str, &str)> = leaves.iter().map(|l| ("center", *l)).collect();
        let c = eigenvector_centrality(&graph(&nodes, &edges));

        let center = score(&c, "center");
        for l in leaves {
            assert!(center > score(&c, l), "the star center outranks every leaf");
        }
        // Leaves are symmetric ⇒ all equal.
        let first = score(&c, "l0");
        for l in &leaves[1..] {
            assert!(
                (score(&c, l) - first).abs() < 1e-9,
                "symmetric leaves share one centrality value"
            );
        }
    }

    /// Determinism: two independent runs on the same graph produce **byte-identical**
    /// scores (same key set, same `f64` bit patterns), and the result is invariant to
    /// the *order* edges were inserted.
    #[test]
    fn two_runs_are_byte_identical() {
        let build = || {
            eigenvector_centrality(&graph(
                &["a", "b", "c", "hub"],
                &[("a", "hub"), ("b", "hub"), ("c", "hub"), ("a", "b")],
            ))
        };
        let r1 = build();
        let r2 = build();
        assert_eq!(r1.len(), r2.len());
        for (k, v) in &r1 {
            // Compare the raw bits so this is a true byte-for-byte determinism check.
            assert_eq!(v.to_bits(), r2[k].to_bits(), "score for {k} is identical");
        }

        // Same graph, edges inserted in a different order ⇒ identical result.
        let reordered = eigenvector_centrality(&graph(
            &["a", "b", "c", "hub"],
            &[("a", "b"), ("c", "hub"), ("a", "hub"), ("b", "hub")],
        ));
        for (k, v) in &r1 {
            assert_eq!(
                v.to_bits(),
                reordered[k].to_bits(),
                "result is invariant to edge-insertion order for {k}"
            );
        }
    }

    /// Empty graph → empty result.
    #[test]
    fn empty_graph_is_empty() {
        let g = MemoryGraph::new(0.2, usize::MAX);
        assert!(eigenvector_centrality(&g).is_empty());
    }

    /// Edge-case stress: an edgeless graph, a single isolated node, and a graph mixing a
    /// connected pair with two disconnected isolates. None may produce `NaN` or panic.
    #[test]
    fn isolated_and_disconnected_nodes_are_well_defined() {
        // Edgeless graph: all nodes equal (uniform unit-L2), nothing NaN.
        let edgeless = eigenvector_centrality(&graph(&["a", "b", "c"], &[]));
        assert_eq!(edgeless.len(), 3);
        let uniform = 1.0 / 3.0_f64.sqrt();
        assert!(edgeless
            .values()
            .all(|v| v.is_finite() && (v - uniform).abs() < 1e-9));

        // A single isolated node (no edges anywhere): defined as the uniform value 1/√1 = 1.
        let solo = eigenvector_centrality(&graph(&["only"], &[]));
        assert_eq!(solo.len(), 1);
        assert!((score(&solo, "only") - 1.0).abs() < 1e-9);

        // Connected pair `p — q` plus two fully-isolated nodes `x`, `y`. The isolates get
        // exactly 0 (nothing feeds them); the connected pair is symmetric and positive.
        let mixed = eigenvector_centrality(&graph(&["p", "q", "x", "y"], &[("p", "q")]));
        assert!(mixed.values().all(|v| v.is_finite()), "no NaN anywhere");
        assert_eq!(score(&mixed, "x"), 0.0, "an isolated node scores exactly 0");
        assert_eq!(score(&mixed, "y"), 0.0, "an isolated node scores exactly 0");
        assert!(score(&mixed, "p") > 0.0 && score(&mixed, "q") > 0.0);
        assert!(
            (score(&mixed, "p") - score(&mixed, "q")).abs() < 1e-9,
            "the symmetric connected pair ties"
        );
    }

    /// A directed cycle is, after symmetrization, the regular (every node degree 2)
    /// undirected ring — so every node must receive **exactly equal** centrality. This
    /// confirms the symmetrization is genuine (a purely directed reading would not be
    /// vertex-transitive on a DAG-like input) and that the routine converges cleanly.
    #[test]
    fn symmetrized_cycle_is_uniform() {
        let c = eigenvector_centrality(&graph(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
        ));
        let first = score(&c, "a");
        for n in ["b", "c", "d"] {
            assert!(
                (score(&c, n) - first).abs() < 1e-9,
                "every node of a symmetric ring has equal centrality"
            );
        }
        assert!(c.values().all(|v| *v > 0.0));
    }

    #[test]
    fn temporal_profile_tracks_belief_and_tension_over_time() {
        // Two graph states: t0 one-sided support (believed, no tension); t1 a contradiction arrives
        // (the claim becomes contested) — the profile must show tension rise and belief drop.
        let claim: NodeId = "claim".into();
        let mut g0 = MemoryGraph::new(0.0, usize::MAX);
        for id in ["claim", "s0", "c0"] {
            g0.upsert_node(id.into(), id.into(), String::new(), NodeType::ContextBlock);
        }
        g0.add_edge("s0".into(), "claim".into(), 1.0, EdgeType::Supports);
        let mut g1 = g0.clone();
        g1.add_edge("c0".into(), "claim".into(), 1.0, EdgeType::Contradicts);

        let prof = temporal_profile([&g0, &g1], std::slice::from_ref(&claim), 0.0);
        assert_eq!(prof.steps(), 2);
        let tension = prof.tension_series(&claim);
        let belief = prof.belief_series(&claim);
        assert_eq!(tension[0], 0.0, "one-sided evidence ⇒ no tension at t0");
        assert!(
            tension[1] > tension[0],
            "the contradiction raises tension at t1"
        );
        assert!(
            belief[0] > belief[1],
            "belief drops as the claim becomes contested"
        );
        // A single tracked claim ⇒ system temperature is its own tension series.
        assert_eq!(prof.temperature(), tension);
        assert!(prof.peak_temperature() > 0.0);
        // An untracked claim yields an empty series (no panic).
        assert!(prof.tension_series(&"absent".into()).is_empty());
    }

    #[test]
    fn retrodicts_belief_over_a_scripted_trajectory() {
        // A claim that starts unknown, gains support over three steps, then holds. Retrodiction folds
        // the later, firmer belief back so the early steps read higher than the raw fever-curve did.
        let claim: NodeId = "claim".into();
        let mut graphs = Vec::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node(
            "claim".into(),
            "claim".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        graphs.push(g.clone()); // t0: no evidence, belief 0
        for i in 0..3 {
            let s = format!("s{i}");
            g.upsert_node(
                s.clone().into(),
                s.clone(),
                String::new(),
                NodeType::ContextBlock,
            );
            g.add_edge(s.into(), "claim".into(), 1.0, EdgeType::Supports);
            graphs.push(g.clone()); // t1..t3: belief climbs toward +1
        }
        let refs: Vec<&MemoryGraph> = graphs.iter().collect();
        let prof = temporal_profile(refs, std::slice::from_ref(&claim), 0.0);

        let raw = prof.belief_series(&claim);
        let smoothed = prof.retrodicted_belief(&claim, 0.02, 0.1);
        assert_eq!(smoothed.len(), raw.len());
        assert_eq!(raw[0], 0.0, "starts unknown");
        assert!(
            smoothed[0] > raw[0],
            "retrodiction lifts the initial belief given the firmer future: {smoothed:?}"
        );
        // Deterministic (replay == live).
        let again = prof.retrodicted_belief(&claim, 0.02, 0.1);
        assert_eq!(
            smoothed.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            again.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
        assert!(prof
            .retrodicted_belief(&"absent".into(), 0.02, 0.1)
            .is_empty());
    }
}
