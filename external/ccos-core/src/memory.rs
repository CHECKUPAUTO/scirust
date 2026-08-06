use crate::cold_index::HuskStore;
use crate::compressor::{CausalAst, CausalCrusher, CausalSumm, ContentRouter, Route};
use crate::eviction_policy::{
    bucket_pressure, bucket_recency, bucket_score, bucket_size, EvictionPolicy, PagingState, EVICT,
    KEEP,
};
use crate::util::{hex32, sha256_bytes};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::path::PathBuf;

/// Memtable size (husks buffered before flushing to a segment) and read-cache
/// capacity for the Lever 2 [`HuskStore`]. Runtime tuning constants; generous enough
/// that the on-disk index stays cheap on small graphs.
const HUSK_BUFFER_LIMIT: usize = 4096;
const HUSK_CACHE_CAP: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub String);

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        NodeId(s.to_string())
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        NodeId(s)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub id: NodeId,
    pub label: String,
    pub content: String,
    pub node_type: NodeType,
    pub base_importance: f64,
    pub failure_relevance: f64,
    pub recency: f64,
    pub access_count: u64,
    pub created_at: u64,
    pub last_accessed: u64,
    /// Lifecycle state (see [`NodeState`]). Kept **separate from topology** so a node's
    /// health/attention cannot pollute the structural centrality signal. `serde(default)` +
    /// skip-if-`Stable` keeps existing snapshots byte-identical on the default path.
    #[serde(default, skip_serializing_if = "NodeState::is_stable")]
    pub state: NodeState,
    /// **Provenance trust** in `[0, 1]`, `1.0 = fully trusted` (the default). Stamped at ingest from
    /// the injection signal: a source that trips the injection flag, or carries a Trojan-Source /
    /// ASCII-smuggling anomaly, gets a discounted trust so everything it produces is structurally
    /// down-weighted in recall (via the gated `w_trust` term) — a poisoned file cannot silently
    /// become high-authority context. Only a *flagged* source deviates from `1.0`, so clean code
    /// keeps `1.0`; `serde(default = 1.0)` + skip-if-full-trust then keeps clean snapshots
    /// byte-identical. Recomputed on replay from the same deterministic ingest, so `replay == live`.
    #[serde(default = "full_trust", skip_serializing_if = "is_full_trust")]
    pub trust: f64,
}

/// Default [`GraphNode::trust`] (fully trusted); also fills the field when an older snapshot
/// (written before it existed) is deserialised.
fn full_trust() -> f64 {
    1.0
}

/// `serde` skip predicate: elide `trust` when the node is fully trusted, keeping clean snapshots
/// byte-identical.
fn is_full_trust(v: &f64) -> bool {
    *v >= 1.0
}

/// Lifecycle state of a node, kept orthogonal to graph topology so a node's *health* /
/// *attention* does not distort the structural signal centrality reads. Default `Stable`,
/// so it is off-by-default and snapshot-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NodeState {
    /// Verified code, part of the load-bearing structure. The default.
    #[default]
    Stable,
    /// Code under active modification — "hot" (the current focus) but structurally fragile.
    /// **Pinned** in working memory (a retention boost) even as its recency decays.
    Working,
    /// Dead / unreachable code — **excluded** from the structural centrality calc, and
    /// **evicted first** even when recently touched.
    Orphan,
}

impl NodeState {
    /// `serde` skip predicate: elide the default so snapshots stay byte-identical.
    fn is_stable(&self) -> bool {
        matches!(self, NodeState::Stable)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Module,
    Symbol,
    ContextBlock,
    AnalysisResult,
    CodeRegion,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f64,
    pub edge_type: EdgeType,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeType {
    DependsOn,
    Contains,
    References,
    Causes,
    RelatedTo,
    /// A `caller → callee` function-call edge, resolved from in-body call-sites by
    /// [`MemoryGraph::resolve_symbol_calls`]. Captures fn→fn structure that import edges miss
    /// (a call crosses files even when the two functions share no vocabulary). Appended last so
    /// the enum's serialized form is strictly additive — old snapshots never contain `Calls`.
    Calls,
    /// A `function → static/const` data-flow edge: the function reads (references) a module-level
    /// `static`/`const` item, resolved from in-body data-references by
    /// [`MemoryGraph::resolve_data_flow`]. Captures the shared-mutable-state channel that connects
    /// functions through globals — invisible to call and import edges. Appended last so the
    /// serialized form stays strictly additive.
    DataFlow,
    /// An `evidence → claim` **support** edge: the source node is evidence *for* the target claim —
    /// the affirmative surface (`S_A`) of a [Q-Page](MemoryGraph::qbelief). Asserted explicitly
    /// (by an agent/tool, not derived from source), so it is the dual partner of [`Contradicts`].
    /// Appended last so the serialized form stays strictly additive.
    ///
    /// [`Contradicts`]: EdgeType::Contradicts
    Supports,
    /// An `evidence → claim` **contradiction** edge: the source node is evidence *against* the
    /// target claim — the negative surface (`S_¬A`) of a [Q-Page](MemoryGraph::qbelief). The signal
    /// a contradiction-aware retriever surfaces that similarity-only retrieval is blind to (a
    /// retriever ranks by *relatedness*, which cannot tell support from refutation). Asserted
    /// explicitly. Appended last so the serialized form stays strictly additive.
    Contradicts,
}

/// Prior strength `ε` in the [`QBelief`] formulas — a pseudo-count that keeps a claim with little
/// evidence near the neutral `belief` of `0`. Deliberately a **unit prior, not a tiny div-by-zero
/// guard**: as `ε → 0` the ratios become scale-invariant (a single faint assertion would read as
/// fully resolved, `belief ≈ ±1`), which would defeat decay; at `ε = 1`, sparse evidence stays near
/// neutral and only *accumulated* evidence approaches the `±1` / `1` extremes.
const BELIEF_EPS: f64 = 1.0;

/// The derived dual-evidence belief state of a claim node — the **Q-Page** primitive.
///
/// A claim accumulates two opposing evidence surfaces: incoming [`Supports`](EdgeType::Supports)
/// edges (the affirmative surface `S_A`) and incoming [`Contradicts`](EdgeType::Contradicts) edges
/// (the negative surface `S_¬A`). Each edge's weight is the **authority** of that assertion (the
/// reliability of its source, in `[0, 1]`), so `support` / `contradiction` are authority-weighted
/// sums. [`MemoryGraph::qbelief`] folds those edges into this summary. It is **derived, never
/// stored** — recomputed from the edge set, so it adds no snapshot state and is reconstructed
/// bit-for-bit on replay.
///
/// `belief` and `conflict` are deliberately **orthogonal axes** (a claim can be near-certain *and*
/// contested, or uncertain *and* uncontested), with prior strength `ε = BELIEF_EPS` (a unit prior):
/// * `belief` — the **signed** support fraction `(s − c) / (s + c + ε)` ∈ `[−1, 1]`: `0` at no (or
///   perfectly balanced) evidence, `→ +1` as one-sided support accumulates, `→ −1` for
///   contradiction. The **sign** is the direction (believed vs refuted), the **magnitude** the
///   strength.
/// * `conflict` — the **geometric** evidence balance `2·√(s·c) / (s + c + ε)` ∈ `[0, 1]`: `0` when
///   the evidence is one-sided (consensus, nothing to resolve), `→ 1` when support and contradiction
///   are both strong and matched (genuine tension). High *only* when both surfaces carry weight —
///   the resolution signal similarity-only retrieval can never report, because relatedness has no
///   notion of polarity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QBelief {
    /// Authority-weighted sum of incoming `Supports` edges (the `S_A` surface).
    pub support: f64,
    /// Authority-weighted sum of incoming `Contradicts` edges (the `S_¬A` surface).
    pub contradiction: f64,
    /// Signed support fraction in `[−1, 1]` (`0` ⇔ no / balanced evidence; sign = direction).
    pub belief: f64,
    /// Geometric evidence balance in `[0, 1]`: `0` one-sided, `→ 1` strong and evenly contested.
    pub conflict: f64,
}

impl QBelief {
    /// Whether this claim is **validated** as a confident, actionable fact: believed strongly enough
    /// (`belief ≥ min_belief`) **and** not too contested (`conflict ≤ max_conflict`). The strategic
    /// filter separating "facts I can act on" from claims still in dispute — the two gates are
    /// orthogonal, so a strongly-believed *but contested* claim is (correctly) **not** validated.
    pub fn is_validated(&self, min_belief: f64, max_conflict: f64) -> bool {
        self.belief >= min_belief && self.conflict <= max_conflict
    }
}

/// A compact unicode **tension bar** for a [`QBelief`]: the signed belief followed by a 10-cell bar of
/// the `conflict` magnitude — `+0.80 ███░░░░░░░` reads "believed, lightly contested". Pure and
/// deterministic; the renderer behind the Pro *tension visualization*. The underlying `conflict` is
/// always computed in the core — only this rendering is gated.
pub fn render_tension_bar(q: &QBelief) -> String {
    let cells = 10usize;
    let filled = ((q.conflict * cells as f64).round() as usize).min(cells);
    let bar: String = "█".repeat(filled) + &"░".repeat(cells - filled);
    let sign = if q.belief >= 0.0 { '+' } else { '-' };
    format!("{sign}{:.2} {bar}", q.belief.abs())
}

/// Build a [`QBelief`] from already-summed (authority-weighted) support/contradiction evidence — the
/// shared core of [`MemoryGraph::qbelief`] and [`MemoryGraph::qbelief_decayed`]. `belief` is the
/// signed support fraction, `conflict` the geometric balance; see [`QBelief`] for the formulas. The
/// denominator is `s + c + ε ≥ ε > 0`, so it never divides by zero, and `s, c ≥ 0` keeps the `√`
/// real.
fn belief_from_sums(support: f64, contradiction: f64) -> QBelief {
    let denom = support + contradiction + BELIEF_EPS;
    let belief = (support - contradiction) / denom;
    let conflict = 2.0 * (support * contradiction).sqrt() / denom;
    QBelief {
        support,
        contradiction,
        belief,
        conflict,
    }
}

/// Tunable coefficients of the causal score and failure-propagation decay.
///
/// A node's score is
/// `clamp(w_base·imp + w_failure·fail + w_recency·rec + w_access·ln(1+acc), 0, 1)`,
/// and injected failure pressure attenuates by `failure_decay^depth` per hop.
/// [`Default`] reproduces the constants CCOS shipped with; [`ScoringWeights::from_env`]
/// lets an external optimiser (the causal-validation harness) override them per
/// run without recompiling — the knobs Phase 3 of that harness searches over.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ScoringWeights {
    /// Weight on a node's intrinsic importance.
    pub w_base: f64,
    /// Weight on propagated failure relevance (the dominant term by default).
    pub w_failure: f64,
    /// Weight on recency.
    pub w_recency: f64,
    /// Weight on `ln(access_count)`.
    pub w_access: f64,
    /// Weight on **structural centrality** — `ln(1 + in_degree)`, the number of
    /// incoming causal edges. A hub (a shared module / interface that many nodes
    /// depend on) is structurally more important than a leaf, independent of how
    /// recently it was touched. **Default `0.0`** (off): the score is then
    /// byte-identical to before this term existed, so snapshots/replay are
    /// unchanged. Set it (or let [`crate::agent_session::AgentSession::tune_recall_weights`]
    /// learn it) to retain hubs more strongly. `skip_serializing_if` elides it when
    /// `0.0` so the default serialized form is unchanged.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub w_centrality: f64,
    /// How the centrality term measures structural importance (only consulted when
    /// `w_centrality != 0`). [`CentralityMode::InDegree`] (default) is the raw incoming-
    /// edge count `ln(1 + in_degree)` — cheap and local. [`CentralityMode::Eigenvector`]
    /// is a *global*, recursive importance (damped power iteration, see
    /// [`eigencentrality`](MemoryGraph::eigencentrality)): a node is important if
    /// *important* nodes depend on it. Elided from the serialized form when default, so
    /// existing snapshots are byte-identical.
    #[serde(default, skip_serializing_if = "CentralityMode::is_default")]
    pub centrality_mode: CentralityMode,
    /// Weight on a node's **provenance distrust** `(1 − trust)`: a source that tripped the injection
    /// signal (or a Trojan-Source / ASCII-smuggling anomaly) is structurally down-weighted in recall,
    /// so poisoned context sinks below the budget. **Default `0.0`** (off) — the score is then
    /// byte-identical to before this term existed; a deployment opts into quarantine by raising it.
    /// `skip_serializing_if` elides it at `0.0` so existing snapshots are unchanged.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub w_trust: f64,
    /// Geometric attenuation of failure pressure per propagation hop.
    pub failure_decay: f64,
    /// Out-degree at which a node starts **distributing** (rather than
    /// replicating) failure pressure across its edges. At or below this fan-out,
    /// propagation is unchanged — so sparse causal chains still reach depth; above
    /// it, a hub's emission is damped by `failure_fanout / out_degree`, so one
    /// over-connected node (e.g. a file with dozens of contained symbols) cannot
    /// flood the graph. See `docs/FIELD_CAMPAIGN_H.md` (root cause #2).
    #[serde(default = "default_failure_fanout")]
    pub failure_fanout: f64,
}

/// How [`ScoringWeights::centrality_mode`] turns graph structure into a node's
/// centrality score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CentralityMode {
    /// `ln(1 + in_degree)` — importance = the raw count of incoming causal edges.
    /// Local and O(1) per node (one cached in-degree pass); the shipped behaviour.
    #[default]
    InDegree,
    /// **Eigenvector centrality** — importance propagates recursively: a node is
    /// important if *important* nodes depend on it, so a hub that many *other hubs*
    /// rely on outranks a hub that only leaves rely on (which in-degree cannot tell
    /// apart). Computed by deterministic **damped power iteration** — see
    /// [`MemoryGraph::eigencentrality`] for why the damping (Katz/PageRank form) is
    /// the correct realization of `A x = λ x` on a code graph, which is largely a DAG.
    Eigenvector,
}

impl CentralityMode {
    /// `serde` skip predicate: elide the default so existing snapshots are unchanged.
    fn is_default(&self) -> bool {
        matches!(self, CentralityMode::InDegree)
    }
}

/// Default [`ScoringWeights::failure_fanout`]; also fills the field when an older
/// snapshot (written before it existed) is deserialised.
fn default_failure_fanout() -> f64 {
    6.0
}

/// `skip_serializing_if` predicate: elide an `f64` weight when it is exactly `0.0`
/// (an off-by-default term), keeping the serialized form unchanged.
fn is_zero_f64(x: &f64) -> bool {
    *x == 0.0
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            w_base: 0.15,
            w_failure: 0.50,
            w_recency: 0.30,
            w_access: 0.05,
            w_centrality: 0.0,
            w_trust: 0.0,
            centrality_mode: CentralityMode::InDegree,
            failure_decay: 0.8,
            failure_fanout: default_failure_fanout(),
        }
    }
}

impl ScoringWeights {
    /// Read overrides from the environment, falling back to [`Default`] for any
    /// variable that is unset or unparsable. Recognised variables: `CCOS_W_BASE`,
    /// `CCOS_W_FAILURE`, `CCOS_W_RECENCY`, `CCOS_W_ACCESS`, `CCOS_FAILURE_DECAY`,
    /// `CCOS_FAILURE_FANOUT`.
    pub fn from_env() -> Self {
        let d = Self::default();
        let get = |key: &str, fallback: f64| -> f64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|x| x.is_finite())
                .unwrap_or(fallback)
        };
        Self {
            w_base: get("CCOS_W_BASE", d.w_base),
            w_failure: get("CCOS_W_FAILURE", d.w_failure),
            w_recency: get("CCOS_W_RECENCY", d.w_recency),
            w_access: get("CCOS_W_ACCESS", d.w_access),
            w_centrality: get("CCOS_W_CENTRALITY", d.w_centrality),
            w_trust: get("CCOS_W_TRUST", d.w_trust),
            // `CCOS_CENTRALITY_MODE=eigenvector` switches to the global recursive
            // centrality; anything else (or unset) keeps the in-degree default.
            centrality_mode: match std::env::var("CCOS_CENTRALITY_MODE")
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("eigenvector") | Some("eigen") | Some("pagerank") => {
                    CentralityMode::Eigenvector
                }
                _ => d.centrality_mode,
            },
            failure_decay: get("CCOS_FAILURE_DECAY", d.failure_decay),
            failure_fanout: get("CCOS_FAILURE_FANOUT", d.failure_fanout),
        }
    }
}

/// Serialize a `NodeId → GraphNode` map in **sorted key order** so a snapshot is
/// byte-canonical. The resident node map is a `HashMap` (O(1) lookups on the hot
/// path) whose iteration order is nondeterministic, so a plain derive lets two
/// memories with identical state serialize to different bytes (same length, shuffled
/// order). Sorting on the way out makes the stronger invariant hold — *identical
/// state ⇒ byte-identical snapshot*, not merely identical *sorted* hash.
/// Deserialization is unaffected: a `HashMap` reads from a JSON map in any order.
fn serialize_sorted_nodes<S>(
    nodes: &HashMap<NodeId, GraphNode>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut keys: Vec<&NodeId> = nodes.keys().collect();
    keys.sort();
    let mut map = serializer.serialize_map(Some(keys.len()))?;
    for k in keys {
        map.serialize_entry(k, &nodes[k])?;
    }
    map.end()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraph {
    #[serde(serialize_with = "serialize_sorted_nodes")]
    pub(crate) nodes: HashMap<NodeId, GraphNode>,
    pub(crate) edges: Vec<GraphEdge>,
    pub paging_threshold: f64,
    pub max_in_memory_nodes: usize,
    pub clock: u64,
    /// Scoring/decay coefficients (see [`ScoringWeights`]). Serialised so a
    /// snapshot records the weights it was scored under; `serde(default)` keeps
    /// older snapshots (written before this field existed) loadable.
    #[serde(default)]
    pub scoring_weights: ScoringWeights,
    /// Learned eviction policy consulted by [`enforce_paging`](Self::enforce_paging).
    /// Default is **untrained**, in which case eviction is exactly the
    /// deterministic greedy (lowest score first) — so enabling it is never worse.
    /// Train it offline ([`EvictionPolicy::fit`]) and inject via
    /// [`set_eviction_policy`](Self::set_eviction_policy). Serialised (a `BTreeMap`
    /// Q-table) so a snapshot records the policy it paged under; `serde(default)`
    /// keeps pre-existing snapshots loadable.
    #[serde(default = "EvictionPolicy::new")]
    pub eviction_policy: EvictionPolicy,
    /// The **COLD tier** — the "swap". Nodes evicted from the resident graph by
    /// [`enforce_paging`](Self::enforce_paging) are *demoted* here instead of
    /// dropped, with the edges incident to them at demotion time, so the working
    /// memory is **non-destructive**: nothing is lost, anything can be
    /// [`page_in`](Self::page_in)ed back on demand. The resident set
    /// ([`node_count`](Self::node_count)) stays bounded by `max_in_memory_nodes`;
    /// the backing store (this map) is the unbounded "virtual memory" behind it.
    /// `BTreeMap` ⇒ deterministic iteration/serialization; `serde(default)` keeps
    /// pre-existing snapshots loadable.
    #[serde(default)]
    cold: BTreeMap<NodeId, ColdNode>,
    /// Optional on-disk **spill store** for COLD content — the "swap file". A
    /// runtime handle only (`#[serde(skip)]`): a snapshot records the spilled
    /// *stubs* (see [`ColdNode::spill`]), and the directory travels alongside,
    /// re-attached on restore via [`attach_cold_spill`](Self::attach_cold_spill).
    /// `None` (the default) ⇒ COLD content stays fully resident in RAM,
    /// byte-identical to a graph that never knew about spill, so every existing
    /// invariant (replay == live, snapshot hash) is untouched on the default path.
    #[serde(skip)]
    spill: Option<SpillConfig>,
    /// Lever 2 (slice 5c): the **authoritative** on-disk index of deep-spilled husks.
    /// The COLD tier's *entry count* is no longer `O(N)` resident — husks live in
    /// segments + a bounded cache, not one `BTreeMap` node each. Runtime handle only
    /// (`#[serde(skip)]`), opened in a sibling `<dir>.husks` directory and re-attached
    /// on restore like the spill store; deep-spilled state thus travels in that
    /// directory, not the JSON snapshot. `None` (the default, no spill attached) ⇒ no
    /// deep tier, byte-identical to a graph that never knew about it.
    #[serde(skip)]
    husk_store: Option<HuskStore>,
    /// Lever 2 brick 8: an on-disk **reverse-adjacency index** of the COLD tier —
    /// for every archived cold edge `a─b`, the keys `"a\x1fb"` and `"b\x1fa"`. Lets
    /// [`cold_neighbours`](Self::cold_neighbours) answer with one prefix scan
    /// (`O(degree)`) instead of deserializing the whole deep tier (`O(N)`). Maintained
    /// at demote (add) and page-in / removal (drop), opened in a sibling `<dir>.radj`
    /// directory, runtime-only (`#[serde(skip)]`) and backfilled from the current cold
    /// tier on attach. `None` ⇒ no spill attached, and `cold_neighbours` falls back to
    /// the resident edge scan (there is no deep tier then).
    #[serde(skip)]
    radj: Option<HuskStore>,
    /// Optional **compaction budget** for the COLD tier (slice 4). When `Some(b)`,
    /// total COLD *content* (inline + spilled) is kept toward `b` bytes by
    /// **lossily compacting** the coldest entries — code is skeletonised, prose
    /// summarised (CausalSumm/CausalAst), the full original discarded — so the
    /// *backing store itself* stays frugal. A runtime knob (`#[serde(skip)]`);
    /// `None` (default) ⇒ no compaction, COLD stays lossless. This is the deepest
    /// tier: "infinite" working memory is a *direction*, and at the bottom
    /// frugality wins — CCOS compacts to a summary (observable via
    /// [`is_compacted`](Self::is_compacted)), never silently drops.
    #[serde(skip)]
    cold_content_budget: Option<usize>,
    /// Optional **resident-metadata budget** for the COLD tier (slice 5). When
    /// `Some(b)`, the bytes the COLD tier keeps in RAM ([`cold_resident_bytes`](Self::cold_resident_bytes))
    /// are driven toward `b` by **deep-spilling** the coldest entries — moving each
    /// one's `label` and full `edges` to the on-disk store and keeping only the
    /// neighbour ids (`adj`) resident. Still **lossless** (the body
    /// faults back on [`page_in`](Self::page_in)); the measured-dominant resident
    /// cost (edges, `docs/MEASUREMENT_cold_ram.md`) is shrunk to ids, not dropped or
    /// contracted. A runtime knob (`#[serde(skip)]`); `None` (default) ⇒ no
    /// deep-spill, COLD metadata stays fully resident — byte-identical on the
    /// default path. Needs an attached spill store (the same "swap file").
    #[serde(skip)]
    cold_resident_budget: Option<usize>,
    /// Cached in-degree map for the structural-centrality score term, keyed on
    /// `edges.len()` (edges are only ever appended or `retain`-pruned, so the
    /// length changes whenever the edge set does). Runtime-only; rebuilt lazily,
    /// and only ever consulted when `scoring_weights.w_centrality != 0`.
    #[serde(skip)]
    indegree_cache: RefCell<Option<(usize, HashMap<NodeId, u32>)>>,
    /// Cached eigenvector-centrality vector for the structural-centrality score term,
    /// keyed on `(nodes.len(), edges.len())` (a cheap proxy for "the graph changed",
    /// matching [`indegree_cache`](Self::indegree_cache)). Runtime-only; rebuilt lazily
    /// by deterministic power iteration, and only ever consulted when
    /// `w_centrality != 0` *and* `centrality_mode == Eigenvector`.
    #[serde(skip)]
    eigencentrality_cache: RefCell<EigenCentralityCache>,
    /// Cached dependency adjacency for [`causal_flash_window`](Self::causal_flash_window),
    /// keyed on `edges.len()` (edges are only appended or `retain`-pruned, so the
    /// length changes whenever the edge set does — the same staleness proxy as
    /// [`indegree_cache`](Self::indegree_cache)). Runtime-only, rebuilt lazily, so
    /// a `Working`-rooted cone is `O(cone)`-amortized across the ticks between
    /// edits instead of re-indexing every edge on every call. Not serialized ⇒
    /// snapshots stay byte-identical and `replay == live` holds (the index is a
    /// pure function of the resident edge set, rebuilt deterministically).
    #[serde(skip)]
    causal_adj_cache: RefCell<Option<(usize, crate::causal_flash::CausalAdjacency)>>,
    /// In-body call-sites awaiting resolution into `Calls` edges, keyed by source file →
    /// `(caller, callee, line)` in source order. Populated by the parser at ingest, consumed by
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). **Runtime-only** (`serde(skip)`): the
    /// resolved `Calls` *edges* are the durable state; this raw input is rebuilt on the replay
    /// re-ingest, so the snapshot is unchanged and `replay == live` holds. `BTreeMap` ⇒ sorted,
    /// deterministic iteration.
    #[serde(skip)]
    pending_calls: BTreeMap<String, Vec<(String, String, usize)>>,
    /// In-body `static`/`const` references awaiting resolution into `DataFlow` edges, keyed by
    /// source file → `(reader, name, line)`. Same **runtime-only** contract as `pending_calls`.
    #[serde(skip)]
    pending_data_refs: BTreeMap<String, Vec<(String, String, usize)>>,
    /// Renamed-import bindings for call resolution, keyed by source file → `(local_name,
    /// target_path)` — one entry per `use a::b as c` (`("c", "a::b")`). Consulted by
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls) to rewrite a call to an alias onto its
    /// real target. Same **runtime-only** (`serde(skip)`, rebuilt on the replay re-ingest) and
    /// deterministic-`BTreeMap` contract as `pending_calls`.
    #[serde(skip)]
    pending_aliases: BTreeMap<String, Vec<(String, String)>>,
    /// Per-impl-block method-ownership facts, keyed source file → `(type_name, method)` — one per
    /// `impl Foo { fn bar }` whose Self type is concrete (Slice 3). The input to the `(type, method) →
    /// symbol` index [`resolve_symbol_calls`](Self::resolve_symbol_calls) builds to resolve a `Foo::bar`
    /// callee (minted by `x.bar()` receiver-type inference, or written explicitly as `Foo::bar()`).
    /// Same **runtime-only** (`serde(skip)`, rebuilt on the replay re-ingest) and deterministic-
    /// `BTreeMap` contract as `pending_calls`; the resolved `Calls` edges are the durable state.
    #[serde(skip)]
    pending_methods: BTreeMap<String, Vec<(String, String)>>,
    /// Node ids of `static`/`const` symbols (the only valid `DataFlow` targets). The graph node
    /// stores `NodeType`, not the finer `SymbolKind`, so the parser marks these at ingest. Runtime
    /// -only; rebuilt on the replay re-ingest, and filtered to still-resident nodes at resolve time.
    #[serde(skip)]
    data_symbols: std::collections::BTreeSet<NodeId>,
    /// O(1) duplicate-edge index: the `(source, target, edge_type)` triples already present in
    /// `edges`. Consulted by [`add_edge`](Self::add_edge) instead of a linear scan — the whole-graph
    /// resolve passes add an edge per ref, so the old O(E) scan made ingestion ~O(N³)
    /// (`examples/ingest_profile.rs`). Runtime-only (`#[serde(skip)]`): rebuilt lazily from `edges`
    /// whenever the two fall out of length-sync (just deserialized, or after a bulk edge removal).
    #[serde(skip)]
    edge_set: HashSet<(NodeId, NodeId, EdgeType)>,
}

/// Cache value behind [`MemoryGraph::node_eigencentrality`]: the centrality vector keyed
/// by `(nodes.len(), edges.len())` — a cheap "did the graph change" stamp. Aliased to keep
/// the field type readable (and clippy's `type_complexity` happy).
type EigenCentralityCache = Option<((usize, usize), HashMap<NodeId, f64>)>;

/// A bound spill store plus the resident-content budget that triggers a flush.
/// Not serialised — held only while a [`MemoryGraph`] is live.
#[derive(Debug, Clone)]
struct SpillConfig {
    store: ColdSpill,
    /// Compresseur maintenu en vie tant que le store existe. Le reconstruire à
    /// chaque blob recopierait les tables du dictionnaire et annulerait le gain de
    /// débit (2,4 Mo/s contre 19 mesurés) ; il est donc créé une fois, quand le
    /// dictionnaire est constitué.
    compressor: Option<crate::lzss::Compressor>,
    /// Max bytes of COLD *content* kept inline (resident) before the coldest is
    /// flushed to disk. `usize::MAX` ⇒ never flush.
    inline_budget: usize,
}

/// An on-disk, content-addressed store for vacated COLD content — the unbounded
/// "swap" backing the bounded resident window. A blob is keyed by the **SHA-256**
/// of its content (the same addressing scheme as the CCR store,
/// [`crate::compressor::CcrRef`]), so identical content is **deduplicated** to a
/// single file, and a read **re-verifies** the hash: a truncated or tampered
/// blob is a detectable miss, never a silent empty restore. Spill thus *extends*
/// the integrity story rather than weakening it. Lossless and verbatim (no codec
/// yet — content-dedup is the only space win at this layer).
#[derive(Debug, Clone)]
pub struct ColdSpill {
    dir: PathBuf,
    /// Dictionnaire partagé du codec (format 2), s'il a déjà été constitué pour ce
    /// store. `None` ⇒ les blobs sont écrits au format 1, ce qui reste lisible.
    dict: Option<std::sync::Arc<crate::lzss::Dict>>,
}

/// Nom du dictionnaire et de son empreinte dans le répertoire de spill.
const DICT_FILE: &str = "dict.v2.bin";
const DICT_HASH_FILE: &str = "dict.v2.sha256";

/// Volume de contenu froid à partir duquel constituer un dictionnaire vaut le coup.
/// Mesuré : un dictionnaire de 32 Ko n'est amorti qu'au-delà d'environ 250 Ko de
/// contenus spillés — en deçà il coûte plus d'octets qu'il n'en fait gagner. Le
/// seuil est donc une condition de rentabilité, pas un réglage de confort.
const DICT_MIN_CONTENT: usize = 256 * 1024;

impl ColdSpill {
    /// Open (creating if needed) a spill directory.
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let dict = Self::load_dict(&dir);
        Ok(Self { dir, dict })
    }

    /// Charge le dictionnaire du répertoire s'il est présent **et intact**. Son
    /// empreinte est vérifiée : un dictionnaire remplacé ou tronqué rendrait
    /// illisible tout blob encodé avec lui, donc l'incompatibilité doit être
    /// détectée et non subie. On l'ignore alors, ce qui transforme les blobs
    /// format 2 en cold-miss — le comportement déjà retenu pour un blob altéré,
    /// jamais une restauration silencieusement fausse.
    fn load_dict(dir: &std::path::Path) -> Option<std::sync::Arc<crate::lzss::Dict>> {
        let bytes = std::fs::read(dir.join(DICT_FILE)).ok()?;
        let want = std::fs::read_to_string(dir.join(DICT_HASH_FILE)).ok()?;
        if hex32(&sha256_bytes(&String::from_utf8_lossy(&bytes))) != want.trim() {
            return None;
        }
        Some(std::sync::Arc::new(crate::lzss::Dict::new(bytes)))
    }

    /// Constitue le dictionnaire à partir d'un échantillon de contenus, puis le
    /// **fige** : il est écrit une fois avec son empreinte et n'est plus jamais
    /// modifié, puisque le remplacer invaliderait les blobs déjà encodés.
    /// Sans effet si un dictionnaire existe déjà.
    fn build_dict(&mut self, sample: &[String]) {
        if self.dict.is_some() {
            return;
        }
        let mut bytes = Vec::with_capacity(crate::lzss::V2_WINDOW);
        for c in sample {
            if bytes.len() >= crate::lzss::V2_WINDOW {
                break;
            }
            bytes.extend_from_slice(c.as_bytes());
        }
        bytes.truncate(crate::lzss::V2_WINDOW);
        if bytes.len() < 4096 {
            return; // trop peu de matière pour que le dictionnaire serve
        }
        let hash = hex32(&sha256_bytes(&String::from_utf8_lossy(&bytes)));
        if std::fs::write(self.dir.join(DICT_FILE), &bytes).is_err()
            || std::fs::write(self.dir.join(DICT_HASH_FILE), &hash).is_err()
        {
            return;
        }
        self.dict = Some(std::sync::Arc::new(crate::lzss::Dict::new(bytes)));
    }

    /// Le dictionnaire actif, s'il y en a un.
    fn dict(&self) -> Option<&std::sync::Arc<crate::lzss::Dict>> {
        self.dict.as_ref()
    }

    /// Write `content` addressed by its SHA-256, returning the raw 32-byte key. The
    /// on-disk filename is its [`hex32`] and the file holds the **LZSS-compressed**
    /// bytes (lossless; the key/integrity hash is of the *original* content, so
    /// dedup is unchanged). Idempotent — content-addressed *and* the codec is
    /// deterministic, so re-spilling the same blob is a no-op write.
    fn put(
        &self,
        content: &str,
        comp: Option<&mut crate::lzss::Compressor>,
    ) -> std::io::Result<[u8; 32]> {
        let hash = sha256_bytes(content);
        let path = self.dir.join(hex32(&hash));
        if !path.exists() {
            let encoded = match comp {
                // `compress_with` retient le plus petit des candidats, donc passer
                // par le dictionnaire ne peut jamais produire un blob plus gros que
                // le format 1 — même sur un contenu qu'il n'aide pas.
                Some(c) => c.compress_with(content.as_bytes()),
                None => crate::lzss::compress(content.as_bytes()),
            };
            std::fs::write(&path, encoded)?;
        }
        Ok(hash)
    }

    /// Read a blob by hash: decompress, then **verify** integrity. `None` if the
    /// file is missing, the codec stream is malformed, the bytes aren't UTF-8, or
    /// the decompressed content no longer hashes to `hash` (tampered/corrupt /
    /// codec bug) — all surfaced as a cold-miss by the caller, never a silent wrong
    /// restore.
    fn get(&self, hash: &[u8; 32]) -> Option<String> {
        let blob = std::fs::read(self.dir.join(hex32(hash))).ok()?;
        let dict = self.dict.as_ref().map(|d| d.bytes());
        let text = String::from_utf8(crate::lzss::decompress_with(&blob, dict)?).ok()?;
        (sha256_bytes(&text) == *hash).then_some(text)
    }

    /// Delete a blob by hash (best-effort; a missing file is fine). Used to
    /// reclaim a spilled original once no COLD entry references it any more — the
    /// content-addressed store's garbage collection.
    fn remove(&self, hash: &[u8; 32]) {
        let _ = std::fs::remove_file(self.dir.join(hex32(hash)));
    }
}

/// A node demoted out of the resident graph into the [`MemoryGraph`] COLD tier,
/// kept with the edges incident to it at demotion time so a later `page_in` can
/// restore its causal structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdNode {
    node: GraphNode,
    edges: Vec<GraphEdge>,
    /// When `Some`, this node's `content` blob has been vacated to the on-disk
    /// spill store and `node.content` is empty; it must be faulted back in
    /// (verified by hash) before the node can be paged resident. `None` ⇒
    /// content is inline (resident), the default and the only state when no
    /// spill store is attached. `skip_serializing_if` keeps the serialized form
    /// byte-identical to the pre-spill layout whenever nothing is spilled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spill: Option<SpillRef>,
    /// When `true`, `node.content` is a **lossy** compaction (a CausalSumm summary
    /// or CausalAst skeleton) of the original, produced once the COLD compaction
    /// budget was exceeded; the full original has been discarded. Elided from the
    /// serialized form when `false` (the default) so the layout is unchanged
    /// whenever nothing is compacted.
    #[serde(default, skip_serializing_if = "is_false")]
    compacted: bool,
    /// When `true`, compaction tried this entry and could not make it any smaller
    /// (its content is already at the summary/skeleton floor), so it is excluded
    /// from future compaction candidates — otherwise every ingest would re-attempt
    /// it (and re-read its blob from disk) for no gain, and a tier of only
    /// un-shrinkable entries would busy-loop the enforcer. A fresh ingest of the
    /// node drops the whole COLD shadow, so the flag never goes stale. Elided when
    /// `false` (the default) — byte-identical serialization.
    #[serde(default, skip_serializing_if = "is_false")]
    at_floor: bool,
}

/// `skip_serializing_if` predicate: elide a `bool` field when it is `false`.
fn is_false(b: &bool) -> bool {
    !*b
}

/// A **deep-spilled** COLD entry (slice 5b): the most aggressive, still-lossless
/// tier. The *entire* `ColdNode` (node, content folded inline, edges, flags) is
/// serialized into one content-addressed blob, and all that stays resident is this
/// compact husk — the body-blob stub plus the neighbour **ids** (`adj`), so
/// [`cold_neighbours`](MemoryGraph::cold_neighbours) and region paging keep working
/// without touching disk. Replacing the full ~`size_of::<ColdNode>()` struct with
/// this husk is what bounds the per-entry resident *floor* the measurement flagged
/// (`docs/MEASUREMENT_cold_ram.md`); the node faults back, hash-verified, on
/// [`page_in`](MemoryGraph::page_in). Deep husks are **terminal** — already at the
/// floor, they are not re-scored for further spill/compaction.
/// Pack neighbour ids into one length-prefixed byte buffer (each id: a `u32` LE
/// length + its UTF-8 bytes). One allocation for the whole adjacency instead of a
/// `Vec` plus a `String` per id — slice 5c Lever 1, attacking the ~9× allocation
/// overhead the measurement found (`docs/DESIGN_cold_entry_count.md`).
fn pack_adj(ids: &[NodeId]) -> Box<[u8]> {
    let mut buf = Vec::new();
    for id in ids {
        let bytes = id.0.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }
    buf.into_boxed_slice()
}

/// Iterate the ids packed by [`pack_adj`] without allocating (yields borrowed
/// `&str`). Stops cleanly on a truncated or non-UTF-8 buffer.
fn unpack_adj(packed: &[u8]) -> impl Iterator<Item = &str> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        let lb = packed.get(pos..pos + 4)?;
        let len = u32::from_le_bytes(lb.try_into().ok()?) as usize;
        pos += 4;
        let s = std::str::from_utf8(packed.get(pos..pos + len)?).ok()?;
        pos += len;
        Some(s)
    })
}

/// (De)serialize the packed adjacency as a plain JSON array of id strings, so a
/// deep-husk snapshot is byte-identical to the previous `Vec<NodeId>` layout while
/// RAM holds the compact packed form.
mod packed_adj {
    use super::{pack_adj, unpack_adj, NodeId};
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(adj: &[u8], s: S) -> Result<S::Ok, S::Error> {
        let ids: Vec<&str> = unpack_adj(adj).collect();
        let mut seq = s.serialize_seq(Some(ids.len()))?;
        for id in ids {
            seq.serialize_element(id)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Box<[u8]>, D::Error> {
        let ids: Vec<String> = Vec::deserialize(d)?;
        Ok(pack_adj(&ids.into_iter().map(NodeId).collect::<Vec<_>>()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepHusk {
    /// Stub of the on-disk blob holding the whole serialized `ColdNode`.
    body: SpillRef,
    /// Neighbour ids — the other endpoint of each edge the entry carried at
    /// deep-spill time — **packed** into one length-prefixed buffer (sorted + deduped
    /// at deep-spill, so deterministic). The only structure kept resident, so the
    /// cold↔cold adjacency survives without faulting the blob; see [`pack_adj`].
    #[serde(with = "packed_adj")]
    adj: Box<[u8]>,
}

/// (De)serialize a 32-byte content hash as its lowercase-hex string, so the stub
/// keeps its compact in-RAM form (a raw `[u8; 32]` — no heap allocation) while the
/// *serialized* snapshot stays the same readable, canonical hex it always was.
mod hex_hash {
    use crate::util::{from_hex32, hex32};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(h: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex32(h))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        from_hex32(&s).ok_or_else(|| serde::de::Error::custom("invalid 32-byte hex hash"))
    }
}

/// A stub left in RAM after a COLD node's content is flushed to the on-disk spill
/// store: enough to fault it back (the hash key + integrity check) and to account
/// for its disk footprint (the original length) without touching the disk. The hash
/// is held raw (`[u8; 32]`, no heap) — its on-disk filename and serialized form are
/// the hex of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpillRef {
    /// Raw SHA-256 of the vacated content — the on-disk key (as [`hex32`]) *and* the
    /// read-time integrity check.
    #[serde(with = "hex_hash")]
    hash: [u8; 32],
    /// Byte length of the original content (so budget/stat math need not fault
    /// the blob back in).
    len: usize,
}

impl MemoryGraph {
    pub fn new(paging_threshold: f64, max_in_memory_nodes: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            paging_threshold,
            max_in_memory_nodes,
            clock: 0,
            scoring_weights: ScoringWeights::default(),
            eviction_policy: EvictionPolicy::new(),
            cold: BTreeMap::new(),
            spill: None,
            husk_store: None,
            radj: None,
            cold_content_budget: None,
            cold_resident_budget: None,
            indegree_cache: RefCell::new(None),
            eigencentrality_cache: RefCell::new(None),
            causal_adj_cache: RefCell::new(None),
            pending_calls: BTreeMap::new(),
            pending_data_refs: BTreeMap::new(),
            pending_aliases: BTreeMap::new(),
            pending_methods: BTreeMap::new(),
            data_symbols: std::collections::BTreeSet::new(),
            edge_set: HashSet::new(),
        }
    }

    /// Run `f` with the graph's dependency-adjacency index, rebuilt only when the
    /// edge set changed (keyed on `edges.len()`). This keeps
    /// [`causal_flash_window`](Self::causal_flash_window) `O(cone)`-amortized —
    /// no per-call `O(E)` re-index — while keeping the `RefCell` cache
    /// encapsulated here. `pub(crate)` so the `causal_flash` module can drive it.
    pub(crate) fn with_causal_adjacency<R>(
        &self,
        f: impl FnOnce(&crate::causal_flash::CausalAdjacency) -> R,
    ) -> R {
        let mut cache = self.causal_adj_cache.borrow_mut();
        if cache.as_ref().map(|(k, _)| *k) != Some(self.edges.len()) {
            *cache = Some((self.edges.len(), self.causal_adjacency()));
        }
        f(&cache.as_ref().unwrap().1)
    }

    /// The paging eviction floor, overridable by `CCOS_PAGING_THRESHOLD` (else `default`). The same
    /// env-override convention as [`ScoringWeights::from_env`] — a non-finite or unparsable value
    /// falls back to the default, so a misconfigured env never destabilises paging.
    pub fn paging_threshold_from_env(default: f64) -> f64 {
        std::env::var("CCOS_PAGING_THRESHOLD")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|x| x.is_finite() && *x >= 0.0)
            .unwrap_or(default)
    }

    /// Construct with the paging knobs read from the environment: `CCOS_PAGING_THRESHOLD` (the
    /// eviction floor) and `CCOS_MAX_RESIDENT` (the resident-node cap), each falling back to the
    /// given default. Mirrors [`ScoringWeights::from_env`] so the whole frugal-window behaviour is
    /// env-tunable (the validation harness sweeps these without recompiling), default-identical.
    pub fn new_from_env(default_threshold: f64, default_max: usize) -> Self {
        let max = std::env::var("CCOS_MAX_RESIDENT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(default_max);
        Self::new(Self::paging_threshold_from_env(default_threshold), max)
    }

    /// Replace the eviction policy consulted by [`enforce_paging`](Self::enforce_paging).
    pub fn set_eviction_policy(&mut self, policy: EvictionPolicy) {
        self.eviction_policy = policy;
    }

    /// Train the eviction policy in place from a replay of
    /// `(state, action, reward, next_state)` paging transitions.
    pub fn train_eviction_policy<I>(&mut self, transitions: I)
    where
        I: IntoIterator<Item = (PagingState, u8, f64, PagingState)>,
    {
        self.eviction_policy.fit(transitions);
    }

    /// Replace the scoring/decay coefficients (see [`ScoringWeights`]). Set these
    /// before ingesting or re-paging so the new weights drive eviction.
    pub fn set_scoring_weights(&mut self, weights: ScoringWeights) {
        self.scoring_weights = weights;
    }

    pub fn tick(&mut self) {
        self.clock += 1;
        // Apply recency decay to all nodes
        let decay = 0.95_f64;
        for node in self.nodes.values_mut() {
            node.recency *= decay;
            if node.recency < 0.01 {
                node.recency = 0.01;
            }
        }
    }

    pub fn upsert_node(&mut self, id: NodeId, label: String, content: String, node_type: NodeType) {
        let now = self.clock;
        // A fresh ingest supersedes any demoted (COLD) shadow of this node — full
        // entry or deep husk — reclaiming its on-disk blob(s) if it was the last
        // referent.
        self.forget_cold_shadow(&id);
        match self.nodes.get_mut(&id) {
            Some(existing) => {
                existing.label = label;
                existing.content = content;
                existing.node_type = node_type;
                existing.recency = 1.0;
                existing.last_accessed = now;
                existing.access_count += 1;
            }
            None => {
                let node = GraphNode {
                    id: id.clone(),
                    label,
                    content,
                    node_type,
                    base_importance: 0.5,
                    failure_relevance: 0.0,
                    recency: 1.0,
                    access_count: 1,
                    created_at: now,
                    last_accessed: now,
                    state: NodeState::Stable,
                    trust: full_trust(),
                };
                self.nodes.insert(id, node);
            }
        }
        self.enforce_paging();
    }

    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        weight: f64,
        edge_type: EdgeType,
    ) -> bool {
        // Refuse to create dangling edges: both endpoints must already exist as
        // nodes. This preserves the invariant `edges ⊆ nodes × nodes`, which
        // bounds the edge set and keeps the graph consistent when paging evicts
        // a node mid-construction (otherwise edges to evicted nodes accumulate
        // forever and are never reclaimed by `retain`).
        if !self.nodes.contains_key(&source) || !self.nodes.contains_key(&target) {
            return false;
        }
        // O(1) duplicate check via `edge_set` (the `(source, target, type)` membership index) — the
        // hot path: the whole-graph resolve passes call `add_edge` per ref, so the old linear scan of
        // `self.edges` made ingestion ~O(N³) (see `examples/ingest_profile.rs`). The index is
        // `serde(skip)`, so rebuild it lazily whenever it has fallen out of length-sync with `edges`
        // (just deserialized, or after a bulk edge removal).
        if self.edge_set.len() != self.edges.len() {
            self.edge_set = self
                .edges
                .iter()
                .map(|e| (e.source.clone(), e.target.clone(), e.edge_type.clone()))
                .collect();
        }
        if !self
            .edge_set
            .insert((source.clone(), target.clone(), edge_type.clone()))
        {
            return false; // an identical edge already exists
        }
        let now = self.clock;
        self.edges.push(GraphEdge {
            source,
            target,
            weight,
            edge_type,
            created_at: now,
        });
        true
    }

    pub fn remove_node(&mut self, id: &NodeId) {
        self.nodes.remove(id);
        // Explicit removal forgets the COLD shadow too — full entry or deep husk —
        // and reclaims its on-disk blob(s) if no other entry still references them.
        self.forget_cold_shadow(id);
        self.edges.retain(|e| &e.source != id && &e.target != id);
    }

    pub fn set_failure_relevance(&mut self, id: &NodeId, relevance: f64) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.failure_relevance = relevance.clamp(0.0, 1.0);
            node.recency = 1.0;
            node.last_accessed = self.clock;
        }
    }

    /// A node's [provenance trust](GraphNode::trust) in `[0, 1]` (`1.0` = fully trusted / absent).
    pub fn node_trust(&self, id: &NodeId) -> f64 {
        self.nodes.get(id).map(|n| n.trust).unwrap_or(1.0)
    }

    /// Stamp provenance [`trust`](GraphNode::trust) (clamped to `[0, 1]`) on a node — lower = less
    /// trusted. With the gated `w_trust` scoring term on, a low-trust node sinks in recall.
    pub fn set_trust(&mut self, id: &NodeId, trust: f64) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.trust = trust.clamp(0.0, 1.0);
        }
    }

    /// Stamp `trust` on **every node produced by `file_path`** — its file node, symbols, uses and
    /// modules (the same id family [`evict_file_nodes`](crate::incremental::IncrementalGraphEngine::evict_file_nodes)
    /// targets). Used at ingest to taint a suspicious source's whole footprint in one call.
    pub fn stamp_file_trust(&mut self, file_path: &str, trust: f64) {
        let trust = trust.clamp(0.0, 1.0);
        let prefix = format!("file:{file_path}");
        let mod_prefix = format!("mod:{file_path}:");
        let use_prefix = format!("use:{file_path}:");
        let sym_prefix = format!("sym:{file_path}:");
        for (id, node) in self.nodes.iter_mut() {
            let s = id.0.as_str();
            if s == prefix
                || s.starts_with(&mod_prefix)
                || s.starts_with(&use_prefix)
                || s.starts_with(&sym_prefix)
            {
                node.trust = trust;
            }
        }
    }

    /// **One-hop trust propagation.** A node that depends on a low-trust node inherits an attenuated
    /// share of the distrust, so a poisoned dependency drags down its consumers. For each directional
    /// dependency edge `A → B` (A depends on B) with `trust[B] < 1`, lower `A`'s trust toward
    /// `1 − (1 − trust[B]) · edge.weight · damping`. A single **bounded** hop (no cascade),
    /// deterministic (edges in canonical order, read-then-apply so ordering cannot leak). Returns the
    /// number of nodes whose trust dropped.
    pub fn propagate_trust(&mut self, damping: f64) -> usize {
        let damping = damping.clamp(0.0, 1.0);
        let mut edges: Vec<(NodeId, NodeId, f64)> = self
            .edges
            .iter()
            .filter(|e| {
                matches!(
                    e.edge_type,
                    EdgeType::Calls | EdgeType::DependsOn | EdgeType::DataFlow
                )
            })
            .map(|e| (e.source.clone(), e.target.clone(), e.weight))
            .collect();
        edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        // Read-only: compute the minimum inherited trust per dependent, then apply.
        let mut updates: BTreeMap<NodeId, f64> = BTreeMap::new();
        for (dependent, dependency, weight) in &edges {
            let dep_trust = self.node_trust(dependency);
            if dep_trust < 1.0 {
                let inherited = 1.0 - (1.0 - dep_trust) * weight * damping;
                let entry = updates
                    .entry(dependent.clone())
                    .or_insert_with(|| self.node_trust(dependent));
                if inherited < *entry {
                    *entry = inherited;
                }
            }
        }
        let mut changed = 0;
        for (id, t) in updates {
            if let Some(node) = self.nodes.get_mut(&id) {
                if t < node.trust {
                    node.trust = t.clamp(0.0, 1.0);
                    changed += 1;
                }
            }
        }
        changed
    }

    /// Record an access to a node that is **already resident** — refresh its
    /// recency to full and bump its access count, exactly as
    /// [`page_in`](Self::page_in) does when it faults a *cold* node back. This
    /// models a **resident hit**: the agent re-reads a node already in working
    /// memory, so a frequently-used resident node ages like a paged-in one
    /// rather than only ever decaying via [`tick`](Self::tick). Returns `true`
    /// if the node was resident (and thus touched); `false` if it is cold or
    /// absent — `touch` never resurrects a demoted node (that is
    /// [`page_in`](Self::page_in)'s job), so the COLD tier and the
    /// `replay == live` path are untouched. Deterministic: `&mut self` only
    /// updates the node's recency/access bookkeeping.
    pub fn touch(&mut self, id: &NodeId) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.recency = 1.0;
            node.last_accessed = self.clock;
            node.access_count = node.access_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Set a node's lifecycle [`NodeState`]. State affects the structural centrality (an
    /// `Orphan` is excluded) and the eviction score (`Orphan` first, `Working` pinned), so
    /// this invalidates the centrality caches — which key only on edge/node counts — when the
    /// state actually changes. No-op if the node is absent. Deterministic.
    pub fn set_node_state(&mut self, id: &NodeId, state: NodeState) {
        let changed = match self.nodes.get_mut(id) {
            Some(node) => {
                let was = node.state;
                node.state = state;
                was != state
            }
            None => false,
        };
        if changed {
            self.indegree_cache.borrow_mut().take();
            self.eigencentrality_cache.borrow_mut().take();
        }
    }

    /// Whether `id` is a resident `Orphan` (excluded from the structural centrality signal).
    fn is_orphan(&self, id: &NodeId) -> bool {
        self.nodes
            .get(id)
            .is_some_and(|n| n.state == NodeState::Orphan)
    }

    pub fn compute_node_score(&self, node: &GraphNode) -> f64 {
        let w = &self.scoring_weights;
        let base = node.base_importance * w.w_base;
        let failure = node.failure_relevance * w.w_failure;
        let recency = node.recency * w.w_recency;
        let access = (node.access_count.max(1) as f64).ln() * w.w_access;
        // Structural-centrality term. Off by default (`w_centrality == 0`), in which case
        // this is byte-identical to the previous score and neither the in-degree map nor
        // the eigenvector vector is ever built. When on, the mode picks the signal:
        // local `ln(1 + in_degree)` (default) or global recursive eigenvector centrality.
        let centrality = if w.w_centrality != 0.0 {
            let signal = match w.centrality_mode {
                CentralityMode::InDegree => (1.0 + self.node_in_degree(&node.id) as f64).ln(),
                CentralityMode::Eigenvector => self.node_eigencentrality(&node.id),
            };
            signal * w.w_centrality
        } else {
            0.0
        };
        // Lifecycle bias (orthogonal to topology): `Working` is **pinned** (kept resident as
        // the current focus even as recency decays); `Orphan` is driven to the bottom so it is
        // **evicted first** regardless of recency. `Stable` (the default) is neutral ⇒ the
        // score is byte-identical to before this field existed.
        let state_bias = match node.state {
            NodeState::Stable => 0.0,
            NodeState::Working => 0.3,
            NodeState::Orphan => -1.0,
        };
        // Provenance-distrust penalty. Off by default (`w_trust == 0`), so this is byte-identical to
        // before the term existed; a node from a fully-trusted source (`trust == 1`) is never
        // penalised. When on, a suspicious source's nodes sink below the recall budget.
        let distrust = if w.w_trust != 0.0 {
            (1.0 - node.trust.clamp(0.0, 1.0)) * w.w_trust
        } else {
            0.0
        };
        (base + failure + recency + access + centrality + state_bias - distrust).clamp(0.0, 1.0)
    }

    /// In-degree of `id` among the **resident** graph — the count of incoming
    /// causal edges whose target is `id`. Only edges between two resident nodes
    /// are in `self.edges` (paging archives incident edges on demote), so this is
    /// the *resident* structural signal the centrality term ([`compute_node_score`](Self::compute_node_score))
    /// scores on, not the global in-degree. Cached on `edges.len()`; deterministic.
    pub fn node_in_degree(&self, id: &NodeId) -> u32 {
        let mut cache = self.indegree_cache.borrow_mut();
        if cache.as_ref().map(|(v, _)| *v) != Some(self.edges.len()) {
            let mut m: HashMap<NodeId, u32> = HashMap::new();
            for e in &self.edges {
                // Edges incident to an Orphan don't count toward the load-bearing structure:
                // dead code depending on a node should not inflate its centrality.
                if self.is_orphan(&e.source) || self.is_orphan(&e.target) {
                    continue;
                }
                *m.entry(e.target.clone()).or_default() += 1;
            }
            *cache = Some((self.edges.len(), m));
        }
        cache.as_ref().unwrap().1.get(id).copied().unwrap_or(0)
    }

    /// **Eigenvector centrality** of the resident graph, by deterministic damped power
    /// iteration. Importance flows along causal edges into their target, so a node is
    /// central when *central* nodes depend on it — distinguishing a hub that other hubs
    /// rely on from a hub that only leaves rely on, which [`node_in_degree`](Self::node_in_degree)
    /// cannot. Returns a `NodeId → score` map normalized to `[0, 1]` (most-central = 1.0).
    ///
    /// **Why damped.** Pure eigenvector centrality solves `A x = λ x` for the principal
    /// eigenvalue, but a code dependency graph is largely a **DAG** (imports / contains
    /// flow one way); its adjacency is nilpotent (`λ_max = 0`), so the pure vector
    /// collapses onto the sinks and is ill-defined. The damped iteration
    /// `x ← (1−d)/N + d·(Aᵀ x with out-degree split)` (the Katz / PageRank form) is the
    /// eigenvector of the *damped* operator and stays well-defined and strictly positive
    /// on any directed graph — the standard, correct realization on real code.
    ///
    /// **Deterministic:** a fixed iteration count accumulating over **sorted** node ids,
    /// so it is a pure function of the graph (no `HashMap`-order or float-summation-order
    /// dependence). Like all scoring it is a read-only ranking signal and never enters the
    /// snapshot/replay hash, so its `f64`s are safe.
    pub fn eigencentrality(&self) -> HashMap<NodeId, f64> {
        // Stable ordering ⇒ deterministic accumulation order. `Orphan` nodes are excluded from
        // the structural graph (and so get centrality 0), and their edges drop out below because
        // their endpoints are absent from `index`. `n` is therefore the *non-Orphan* count.
        let mut ids: Vec<&NodeId> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.state != NodeState::Orphan)
            .map(|(id, _)| id)
            .collect();
        ids.sort();
        let n = ids.len();
        if n == 0 {
            return HashMap::new();
        }
        let index: HashMap<&NodeId, usize> =
            ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        // Resident edges as (source_idx, target_idx); out-degree splits each node's mass
        // over its out-edges (PageRank) so one over-connected node cannot flood the rest.
        let mut outdeg = vec![0u32; n];
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            if let (Some(&s), Some(&t)) = (index.get(&e.source), index.get(&e.target)) {
                edges.push((s, t));
                outdeg[s] += 1;
            }
        }
        // Canonical edge order so the floating-point accumulation below is invariant to how
        // the edges were *inserted* (import resolution adds them in HashMap order). With the
        // sorted ids above, this makes eigencentrality a pure function of the graph's
        // *structure*, not its construction order — deterministic across processes.
        edges.sort_unstable();
        const ITERS: usize = 64;
        const DAMP: f64 = 0.85;
        let base = (1.0 - DAMP) / n as f64;
        let mut x = vec![1.0 / n as f64; n];
        for _ in 0..ITERS {
            let mut next = vec![base; n];
            for &(s, t) in &edges {
                // `s` is a source ⇒ outdeg[s] ≥ 1, so this never divides by zero.
                next[t] += DAMP * x[s] / outdeg[s] as f64;
            }
            x = next;
        }
        // Normalize to [0,1] by the max so the term sits on the same scale as the rest of
        // the (clamped [0,1]) score. No edges anywhere ⇒ all equal ⇒ all 1.0.
        let max = x.iter().copied().fold(0.0_f64, f64::max);
        let inv = if max > 0.0 { 1.0 / max } else { 1.0 };
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), x[i] * inv))
            .collect()
    }

    /// Cached per-node eigenvector centrality for the scoring hot path (mirrors
    /// [`node_in_degree`](Self::node_in_degree)): the whole vector is recomputed only when
    /// `(nodes.len(), edges.len())` changes, then looked up per node.
    fn node_eigencentrality(&self, id: &NodeId) -> f64 {
        let key = (self.nodes.len(), self.edges.len());
        let mut cache = self.eigencentrality_cache.borrow_mut();
        if cache.as_ref().map(|(k, _)| *k) != Some(key) {
            *cache = Some((key, self.eigencentrality()));
        }
        cache.as_ref().unwrap().1.get(id).copied().unwrap_or(0.0)
    }

    pub fn select_context_window(&self, max_tokens: usize) -> Vec<&GraphNode> {
        let estimated_tokens_per_node = 128;
        let max_nodes = (max_tokens / estimated_tokens_per_node).max(1);

        struct ScoredRef<'a> {
            node: &'a GraphNode,
            score: f64,
        }
        impl<'a> PartialEq for ScoredRef<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.node == other.node
            }
        }
        impl<'a> Eq for ScoredRef<'a> {}
        impl<'a> PartialOrd for ScoredRef<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl<'a> Ord for ScoredRef<'a> {
            fn cmp(&self, other: &Self) -> Ordering {
                // Order by score, breaking ties on node id so the heap pops a
                // deterministic node when scores are equal.
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| self.node.id.cmp(&other.node.id))
            }
        }

        let mut heap: BinaryHeap<ScoredRef> = BinaryHeap::new();
        for node in self.nodes.values() {
            let score = self.compute_node_score(node);
            heap.push(ScoredRef { node, score });
        }

        let mut selected: Vec<&GraphNode> = Vec::new();
        while let Some(scored) = heap.pop() {
            if selected.len() >= max_nodes {
                break;
            }
            selected.push(scored.node);
        }
        selected
    }

    pub fn enforce_paging(&mut self) {
        if self.nodes.len() <= self.max_in_memory_nodes {
            return;
        }
        let total = self.nodes.len();
        // Recency rank (0 = most recently accessed), for the eviction-policy state.
        let mut by_recency: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.clone(), n.recency))
            .collect();
        by_recency.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let recency_rank: HashMap<NodeId, usize> = by_recency
            .into_iter()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();

        let trained = self.eviction_policy.is_trained();
        let to_remove: Vec<NodeId> = {
            // Eviction priority = base causal score, nudged by the learned
            // policy's keep−evict preference. When the policy is untrained the
            // nudge is exactly 0, so this is byte-identical to the deterministic
            // greedy (lowest score first, ties broken by node id) and replay /
            // snapshot hashes are unchanged.
            let mut entries: Vec<(&NodeId, f64)> = self
                .nodes
                .iter()
                .map(|(id, node)| {
                    let base = self.compute_node_score(node);
                    let bias = if trained {
                        let state = PagingState {
                            score: bucket_score(base),
                            recency: bucket_recency(recency_rank[id], total),
                            pressure: bucket_pressure(node.failure_relevance),
                            size: bucket_size((node.content.len() + node.label.len()) / 4),
                        };
                        self.eviction_policy.q_value(state, KEEP)
                            - self.eviction_policy.q_value(state, EVICT)
                    } else {
                        0.0
                    };
                    (id, base + bias)
                })
                .collect();
            entries.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(b.0))
            });
            let remove_count = self.nodes.len() - self.max_in_memory_nodes;
            entries
                .iter()
                .take(remove_count)
                .map(|(id, _)| (*id).clone())
                .collect()
        };
        for id in &to_remove {
            self.demote(id);
        }
        // Defensive: guarantee no edge survives pointing at an evicted node.
        self.prune_dangling_edges();
        // Newly-demoted content may push the COLD tier over its budgets: compact
        // the coldest tail first (shrink), then spill what remains (move to disk).
        self.enforce_cold_content_budget();
        self.enforce_cold_budget();
        self.enforce_cold_resident_budget();
    }

    /// Demote a node out of the resident graph into the COLD tier, archiving the
    /// edges incident to it so [`page_in`](Self::page_in) can later restore its
    /// causal structure. Non-destructive: the node and its links are kept, just
    /// no longer resident.
    fn demote(&mut self, id: &NodeId) {
        if let Some(node) = self.nodes.remove(id) {
            let incident: Vec<GraphEdge> = self
                .edges
                .iter()
                .filter(|e| &e.source == id || &e.target == id)
                .cloned()
                .collect();
            self.edges.retain(|e| &e.source != id && &e.target != id);
            // The other endpoint of each archived edge — recorded in the reverse index
            // (brick 8) so cold_neighbours is O(degree), not an O(N) tier scan.
            let others: Vec<NodeId> = incident
                .iter()
                .map(|e| {
                    if &e.source == id {
                        e.target.clone()
                    } else {
                        e.source.clone()
                    }
                })
                .collect();
            self.cold.insert(
                id.clone(),
                ColdNode {
                    node,
                    edges: incident,
                    spill: None,
                    compacted: false,
                    at_floor: false,
                },
            );
            for other in others {
                self.radj_add_edge(id, &other);
            }
        }
    }

    /// Restore a node from the **COLD tier** into the resident graph — a page-in
    /// (a swap). The node is marked freshly accessed; if the resident set is at
    /// capacity, the lowest-scored **other** node is demoted to make room — the
    /// just-requested node is never the one bounced back out. Any archived edge
    /// whose other endpoint is resident is re-linked. Returns `true` if the node
    /// was cold and is now resident. Deterministic (tie-break on id).
    /// Remonte une entrée **deep-spillée** d'un cran : son corps sérialisé est
    /// refaulté depuis le blob (vérifié par hachage) et réinséré dans la carte
    /// `cold`. `false` si le blob manque, a été altéré, ou n'est pas
    /// désérialisable — un cold-miss, jamais une demi-restauration silencieuse.
    ///
    /// Extrait de [`page_in`] pour être réutilisable : ré-héberger une arête chez
    /// un voisin deep-spillé exige de le ramener d'abord dans `cold`, puisque le
    /// husk ne porte que l'adjacence et que les arêtes de l'entrée vivent dans le
    /// blob de son corps.
    fn fault_in_deep(&mut self, id: &NodeId) -> bool {
        let Some(husk) = self.deep_get(id) else {
            return false;
        };
        let body_hash = husk.body.hash;
        match self
            .spill
            .as_ref()
            .and_then(|cfg| cfg.store.get(&body_hash))
            .and_then(|s| serde_json::from_str::<ColdNode>(&s).ok())
        {
            Some(node) => {
                if let Some(hs) = self.husk_store.as_mut() {
                    let _ = hs.delete(&id.0); // the husk leaves the on-disk tier
                }
                self.cold.insert(id.clone(), node);
                // The husk's body blob is now unreferenced — the node is back with
                // its content folded inline — so reclaim it unless another husk
                // still shares it. (Without this, page-in orphans the blob: a slow
                // disk leak no later `remove` can find.)
                self.release_blob_if_orphan(&body_hash);
                true
            }
            None => false,
        }
    }

    pub fn page_in(&mut self, id: &NodeId) -> bool {
        // A deep-spilled entry lives as a compact husk in `cold_deep`: fault its
        // whole serialized node back (hash-verified) into the full COLD map first,
        // then the normal page-in path below runs unchanged. A missing / tampered /
        // undeserializable body is a cold-miss — never a silent half-restore.
        if self.deep_contains(id) && !self.fault_in_deep(id) {
            return false;
        }
        // Fault spilled content back from disk (verified by hash). A spilled entry
        // whose blob is missing/tampered, or whose store has been detached, is a
        // cold-miss — never a silent empty restore. (A just-restored deep husk has
        // its content folded inline, so this is a no-op for it.)
        let spilled_hash = self
            .cold
            .get(id)
            .and_then(|c| c.spill.as_ref().map(|s| s.hash));
        if let Some(hash) = spilled_hash {
            match self.spill.as_ref().and_then(|cfg| cfg.store.get(&hash)) {
                Some(text) => {
                    if let Some(entry) = self.cold.get_mut(id) {
                        entry.node.content = text;
                        entry.spill = None;
                    }
                    // The content blob is now unreferenced by this entry; reclaim it
                    // unless another cold entry shares it (content blob ⇒ cheap variant).
                    self.release_content_blob_if_orphan(&hash);
                }
                None => return false,
            }
        }
        let Some(mut cold) = self.cold.remove(id) else {
            return false;
        };
        cold.node.recency = 1.0;
        cold.node.last_accessed = self.clock;
        cold.node.access_count = cold.node.access_count.saturating_add(1);
        self.nodes.insert(id.clone(), cold.node);
        // Re-home every archived edge. `demote` removes an edge from `self.edges`
        // and stores it on the node being demoted, so it is archived on exactly
        // ONE side — and this `ColdNode` was just removed from `self.cold`, which
        // makes this the last place the edge exists. Dropping it here because the
        // far endpoint happens to still be cold destroyed it permanently and broke
        // the tier's documented "nothing is lost" contract: a demote/page-in cycle
        // lost 5 of 16 edges, including cross-file `DependsOn` and `Calls` links.
        for e in cold.edges {
            let other = if &e.source == id {
                e.target.clone()
            } else {
                e.source.clone()
            };
            if self.nodes.contains_key(&e.source) && self.nodes.contains_key(&e.target) {
                // Both endpoints resident again: re-link it, and the pair no longer
                // has anything archived, so its reverse-adjacency entry can go.
                let known = self.edges.iter().any(|x| {
                    x.source == e.source && x.target == e.target && x.edge_type == e.edge_type
                });
                if !known {
                    self.edges.push(e);
                }
                self.radj_del_edge(id, &other);
            } else if let Some(neighbour) = self.cold.get_mut(&other) {
                // Far end still cold: hand the edge to it, so paging *that* node
                // in later restores the link. The reverse-adjacency entry stays on
                // purpose — `cold_neighbours` filters on "still cold", and that
                // entry is what lets `ensure_resident` pull this neighbour back.
                let known = neighbour.edges.iter().any(|x| {
                    x.source == e.source && x.target == e.target && x.edge_type == e.edge_type
                });
                if !known {
                    neighbour.edges.push(e);
                }
            } else if self.deep_contains(&other) && self.fault_in_deep(&other) {
                // Far end deep-spilled: it is neither resident nor in `cold`, so
                // without this it had nowhere to go and was destroyed — measured,
                // 2 of 4 edges lost on a demote/deep-spill/page-in round trip.
                // Fault it back one tier so it can take custody of the edge; the
                // reverse-adjacency entry stays, as for a plain cold neighbour.
                if let Some(neighbour) = self.cold.get_mut(&other) {
                    let known = neighbour.edges.iter().any(|x| {
                        x.source == e.source && x.target == e.target && x.edge_type == e.edge_type
                    });
                    if !known {
                        neighbour.edges.push(e);
                    }
                }
            } else {
                // Neither resident, cold, nor deep-spilled: the far endpoint has
                // left the graph entirely, so the edge is genuinely stale.
                self.radj_del_edge(id, &other);
            }
        }
        // Swap: while over capacity, demote the lowest-scored node *other* than
        // the one just paged in (deterministic tie-break on id).
        while self.nodes.len() > self.max_in_memory_nodes {
            let victim = self
                .nodes
                .iter()
                .filter(|(nid, _)| *nid != id)
                .min_by(|(aid, an), (bid, bn)| {
                    self.compute_node_score(an)
                        .partial_cmp(&self.compute_node_score(bn))
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| aid.cmp(bid))
                })
                .map(|(nid, _)| nid.clone());
            match victim {
                Some(v) => self.demote(&v),
                None => break, // only `id` is resident and still over cap — keep it
            }
        }
        self.prune_dangling_edges();
        // Swap-demoted victims add content to the COLD tier; re-check both budgets
        // (compact the coldest tail, then spill what remains).
        self.enforce_cold_content_budget();
        self.enforce_cold_budget();
        self.enforce_cold_resident_budget();
        true
    }

    /// Number of nodes in the COLD tier (demoted but retrievable via [`page_in`](Self::page_in)) —
    /// full entries plus deep-spilled husks.
    pub fn cold_count(&self) -> usize {
        self.cold.len() + self.deep_count()
    }

    /// Whether `id` is currently demoted to the COLD tier (full entry or deep husk).
    pub fn is_cold(&self, id: &NodeId) -> bool {
        self.cold.contains_key(id) || self.deep_contains(id)
    }

    /// Live deep-spilled ids (keys only, no husk deserialization).
    fn deep_live_keys(&self) -> Vec<NodeId> {
        self.husk_store.as_ref().map_or(Vec::new(), |hs| {
            hs.live_entries()
                .unwrap_or_default()
                .into_iter()
                .map(|(k, _)| NodeId(k))
                .collect()
        })
    }

    /// The ids currently in the COLD tier, in sorted (deterministic) order — both
    /// full entries and deep-spilled husks (the latter from the on-disk index).
    /// Yields owned ids (the deep ones are reconstructed from the index, not borrowed).
    pub fn cold_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        let mut ids: Vec<NodeId> = self
            .cold
            .keys()
            .cloned()
            .chain(self.deep_live_keys())
            .collect();
        ids.sort();
        ids.dedup();
        ids.into_iter()
    }

    /// The **cold** neighbours of `id` — the other endpoints of any COLD-archived
    /// edge incident to `id` that are themselves cold, i.e. the rest of `id`'s causal
    /// region that would page in alongside it. Sorted (deterministic).
    ///
    /// With the reverse-adjacency index attached (Lever 2 brick 8), a single keyed
    /// prefix scan gives every cold edge incident to `id` in both directions —
    /// `O(degree)`, no deep-tier scan. Without a spill store there is no deep tier, so
    /// it falls back to reading the resident cold-full entries' edges directly.
    pub fn cold_neighbours(&self, id: &NodeId) -> Vec<NodeId> {
        let mut out: BTreeSet<NodeId> = BTreeSet::new();
        if let Some(radj) = self.radj.as_ref() {
            let prefix = format!("{}\u{1f}", id.0);
            for (k, _) in radj.scan_prefix(&prefix).unwrap_or_default() {
                let other = NodeId(k[prefix.len()..].to_owned());
                // Keep only neighbours that are themselves still cold.
                if other != *id && (self.cold.contains_key(&other) || self.deep_contains(&other)) {
                    out.insert(other);
                }
            }
            return out.into_iter().collect();
        }
        // No spill store ⇒ no deep tier; read the resident cold-full edges straight.
        for c in self.cold.values() {
            for e in &c.edges {
                let other = if &e.source == id {
                    &e.target
                } else if &e.target == id {
                    &e.source
                } else {
                    continue;
                };
                if other != id && self.cold.contains_key(other) {
                    out.insert(other.clone());
                }
            }
        }
        out.into_iter().collect()
    }

    /// Attach an on-disk **spill store** (the "swap file") for COLD content,
    /// flushing the coldest content to `dir` whenever resident COLD content
    /// exceeds `inline_budget` bytes. Content is addressed by SHA-256 (so a
    /// blob is deduplicated and integrity-checked on read) and faulted back by
    /// [`page_in`](Self::page_in) on demand. Re-attaching the *same* directory
    /// after a snapshot restore lets previously-spilled stubs fault back in.
    /// Applies the budget immediately. Errors only if `dir` can't be created.
    pub fn attach_cold_spill(
        &mut self,
        dir: impl Into<PathBuf>,
        inline_budget: usize,
    ) -> std::io::Result<()> {
        let dir = dir.into();
        // Lever 2 husk index, in a *sibling* `<dir>.husks` directory — kept out of the
        // spill (blob) directory so that stays purely the content-addressed blob store
        // (the "no orphaned blobs" invariant counts only blobs there).
        let mut husk_dir = dir.clone().into_os_string();
        husk_dir.push(".husks");
        let husk_store =
            HuskStore::open(PathBuf::from(husk_dir), HUSK_BUFFER_LIMIT, HUSK_CACHE_CAP)?;
        // Reverse-adjacency index (brick 8), a second sibling directory.
        let mut radj_dir = dir.clone().into_os_string();
        radj_dir.push(".radj");
        let radj = HuskStore::open(PathBuf::from(radj_dir), HUSK_BUFFER_LIMIT, HUSK_CACHE_CAP)?;
        let store = ColdSpill::new(dir)?;
        let compressor = store.dict().cloned().map(crate::lzss::Compressor::new);
        self.spill = Some(SpillConfig {
            store,
            compressor,
            inline_budget,
        });
        self.husk_store = Some(husk_store);
        self.radj = Some(radj);
        self.backfill_radj();
        self.enforce_cold_budget();
        self.enforce_cold_resident_budget();
        Ok(())
    }

    /// Seed the reverse-adjacency index from the COLD tier already present at attach
    /// time (entries demoted before the index existed, or restored from a snapshot),
    /// so [`cold_neighbours`](Self::cold_neighbours) is correct immediately.
    fn backfill_radj(&mut self) {
        let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
        for (id, c) in &self.cold {
            for e in &c.edges {
                let other = if &e.source == id {
                    &e.target
                } else {
                    &e.source
                };
                edges.push((id.clone(), other.clone()));
            }
        }
        for (id, h) in self.deep_entries() {
            for o in unpack_adj(&h.adj) {
                edges.push((id.clone(), NodeId(o.to_owned())));
            }
        }
        for (a, b) in edges {
            self.radj_add_edge(&a, &b);
        }
    }

    /// Detach the spill store. Already-spilled entries stay stubbed and become
    /// unreachable (a cold-miss on [`page_in`](Self::page_in)) until the same
    /// directory is re-attached. Mainly for tests and controlled teardown.
    pub fn detach_cold_spill(&mut self) {
        self.spill = None;
        self.husk_store = None;
        self.radj = None;
    }

    /// Flush the COLD tier's on-disk indices (husk store + reverse adjacency) so their
    /// in-RAM write buffers reach durable segments. Call before a checkpoint so the
    /// `<dir>.husks` / `<dir>.radj` directories are consistent with the snapshot and a
    /// crash loses nothing committed since the last checkpoint. Spill *blobs* are
    /// already written durably on each put (and segments via `write_durable`), so the
    /// only volatile cold-tier state is these two memtables. A no-op without a store.
    ///
    /// Crash model: the **event log** is the source of truth — `replay == live` is
    /// reconstructed from it regardless of the cold tier, which is a rebuildable cache.
    /// This flush makes the cache itself recover to its last checkpoint instead of its
    /// last segment flush.
    pub fn flush_cold_tier(&mut self) -> std::io::Result<()> {
        if let Some(hs) = self.husk_store.as_mut() {
            hs.flush()?;
        }
        if let Some(radj) = self.radj.as_mut() {
            radj.flush()?;
        }
        Ok(())
    }

    /// Record an archived cold edge `a─b` in the reverse-adjacency index (both
    /// directions). A no-op without an attached index. Best-effort: a write miss only
    /// degrades `cold_neighbours` to (still-correct) incompleteness, never corrupts.
    fn radj_add_edge(&mut self, a: &NodeId, b: &NodeId) {
        if let Some(radj) = self.radj.as_mut() {
            let _ = radj.put(&format!("{}\u{1f}{}", a.0, b.0), Vec::new());
            let _ = radj.put(&format!("{}\u{1f}{}", b.0, a.0), Vec::new());
        }
    }

    /// Drop the reverse-adjacency entries for edge `a─b` (both directions).
    fn radj_del_edge(&mut self, a: &NodeId, b: &NodeId) {
        if let Some(radj) = self.radj.as_mut() {
            let _ = radj.delete(&format!("{}\u{1f}{}", a.0, b.0));
            let _ = radj.delete(&format!("{}\u{1f}{}", b.0, a.0));
        }
    }

    /// Whether an on-disk COLD spill store is currently attached.
    pub fn has_cold_spill(&self) -> bool {
        self.spill.is_some()
    }

    /// Bytes of COLD content currently held **inline** (resident in RAM, not yet
    /// spilled). This is the quantity the spill budget bounds.
    pub fn cold_inline_bytes(&self) -> usize {
        self.cold
            .values()
            .filter(|c| c.spill.is_none())
            .map(|c| c.node.content.len())
            .sum()
    }

    /// Number of COLD entries whose content has been spilled to disk.
    pub fn cold_spilled_count(&self) -> usize {
        self.cold.values().filter(|c| c.spill.is_some()).count()
    }

    /// Whether `id` is a COLD entry whose content currently lives on disk (spilled).
    pub fn is_spilled(&self, id: &NodeId) -> bool {
        self.cold.get(id).is_some_and(|c| c.spill.is_some())
    }

    /// Bytes of COLD content spilled to disk (sum of original content lengths;
    /// the on-disk store deduplicates identical blobs, so the actual disk
    /// footprint is ≤ this).
    pub fn cold_spilled_bytes(&self) -> usize {
        self.cold
            .values()
            .filter_map(|c| c.spill.as_ref().map(|s| s.len))
            .sum()
    }

    /// Estimated **resident** bytes the COLD tier still holds in RAM — the part
    /// that does *not* go to disk even when content is spilled: the `BTreeMap` key,
    /// the node's id/label (and any inline content), the archived edges, and the
    /// spill-hash stub. This is the O(N) footprint slice 5 bounds; a logical
    /// estimate (string lengths + struct sizes, ignoring allocator slack), honest
    /// for "how much RAM is stuck per cold entry". A **deep-spilled** entry is just
    /// a compact `DeepHusk` — body-blob hash + neighbour ids — so it contributes a
    /// small fraction of a full entry (the whole `ColdNode` struct is gone).
    pub fn cold_resident_bytes(&self) -> usize {
        let full: usize = self
            .cold
            .iter()
            .map(|(k, c)| Self::entry_resident_bytes(k, c))
            .sum();
        // Deep-spilled husks live in the on-disk index now; their resident cost is the
        // store's bounded footprint (memtable + sparse indices + cache), not one entry
        // per husk — the whole point of Lever 2.
        let deep = self
            .husk_store
            .as_ref()
            .map_or(0, HuskStore::resident_bytes);
        full + deep
    }

    /// Per-entry resident-byte estimate for a full `ColdNode`, shared by
    /// [`cold_resident_bytes`](Self::cold_resident_bytes) and the deep-spill enforcer
    /// (so the budget loop accounts exactly as the stat reports).
    fn entry_resident_bytes(key: &NodeId, c: &ColdNode) -> usize {
        // The spill stub's hash is a raw `[u8; 32]` inline in `size_of::<ColdNode>()`
        // (no heap), so it needs no separate term — only the variable-length strings
        // and archived edges do.
        let mut b = std::mem::size_of::<ColdNode>() + std::mem::size_of::<NodeId>();
        b += key.0.len() + c.node.id.0.len() + c.node.label.len() + c.node.content.len();
        for e in &c.edges {
            b += std::mem::size_of::<GraphEdge>() + e.source.0.len() + e.target.0.len();
        }
        b
    }

    /// Flush the coldest resident COLD content to the spill store until resident
    /// COLD content is within `inline_budget`. Deterministic: candidates are
    /// ordered coldest-first by causal score, ties broken on node id. A no-op
    /// without an attached store, or when already within budget. A blob that
    /// fails to write is left **inline** (kept in RAM) — spill never drops data.
    /// Constitue le dictionnaire partagé du store de spill si le volume de contenu
    /// froid justifie son coût, et arme le compresseur correspondant.
    ///
    /// Le seuil n'est pas arbitraire : mesuré sur un tier COLD réel, un
    /// dictionnaire de 32 Ko n'est amorti qu'au-delà d'environ 250 Ko de contenus
    /// spillés. L'activer plus tôt ferait *perdre* de l'espace, ce qui serait le
    /// contraire du but recherché.
    fn maybe_build_spill_dictionary(&mut self, candidates: &[(NodeId, f64)]) {
        if self
            .spill
            .as_ref()
            .is_none_or(|cfg| cfg.store.dict().is_some())
        {
            return;
        }
        let mut sample: Vec<String> = Vec::new();
        let mut volume = 0usize;
        for (id, _) in candidates {
            if let Some(c) = self.cold.get(id) {
                volume += c.node.content.len();
                if sample.len() < 512 {
                    sample.push(c.node.content.clone());
                }
            }
        }
        if volume < DICT_MIN_CONTENT {
            return;
        }
        if let Some(cfg) = self.spill.as_mut() {
            cfg.store.build_dict(&sample);
            cfg.compressor = cfg.store.dict().cloned().map(crate::lzss::Compressor::new);
        }
    }

    fn enforce_cold_budget(&mut self) {
        let budget = match self.spill.as_ref() {
            Some(cfg) => cfg.inline_budget,
            None => return,
        };
        let mut resident = self.cold_inline_bytes();
        if resident <= budget {
            return;
        }
        // Coldest-first candidate order (deterministic); scores computed up front
        // so the mutation loop borrows neither `self.scoring_weights` nor the map.
        let mut candidates: Vec<(NodeId, f64)> = self
            .cold
            .iter()
            .filter(|(_, c)| c.spill.is_none() && !c.node.content.is_empty())
            .map(|(id, c)| (id.clone(), self.compute_node_score(&c.node)))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        // Constituer le dictionnaire au premier spill qui en vaut la peine, à
        // partir des contenus sur le point d'être écrits — ils sont représentatifs
        // de ce que ce store contiendra. En deçà du seuil on s'en passe : le
        // dictionnaire coûterait plus d'octets qu'il n'en ferait gagner.
        self.maybe_build_spill_dictionary(&candidates);
        for (id, _) in candidates {
            if resident <= budget {
                break;
            }
            let content = match self.cold.get(&id) {
                Some(c) if c.spill.is_none() => c.node.content.clone(),
                _ => continue,
            };
            let written = self.spill.as_mut().and_then(|cfg| {
                let SpillConfig {
                    store, compressor, ..
                } = cfg;
                store.put(&content, compressor.as_mut()).ok()
            });
            if let Some(hash) = written {
                if let Some(entry) = self.cold.get_mut(&id) {
                    let len = entry.node.content.len();
                    entry.node.content = String::new();
                    entry.spill = Some(SpillRef { hash, len });
                    resident = resident.saturating_sub(len);
                }
            }
        }
    }

    /// Set the COLD **resident-metadata budget** (slice 5) — the most aggressive
    /// tier. With `Some(b)`, the bytes the COLD tier keeps in RAM
    /// ([`cold_resident_bytes`](Self::cold_resident_bytes)) are driven toward `b` by
    /// **deep-spilling** the coldest entries (label + full edges → one on-disk blob,
    /// only neighbour ids kept resident). Still **lossless** — the body faults back
    /// on [`page_in`](Self::page_in). Applies immediately. `None` (default) disables
    /// it. Needs an attached spill store; a no-op without one. Like spill/compaction
    /// it is a runtime mode layered on the deterministic default path, not a change
    /// to it (replay and the default snapshot stay byte-identical).
    pub fn set_cold_resident_budget(&mut self, budget: Option<usize>) {
        self.cold_resident_budget = budget;
        self.enforce_cold_resident_budget();
    }

    /// Read a deep-spilled husk from the on-disk [`HuskStore`] (Lever 2). `None` if no
    /// store is attached, the key is absent, or the bytes don't deserialize.
    fn deep_get(&self, id: &NodeId) -> Option<DeepHusk> {
        let bytes = self.husk_store.as_ref()?.get(&id.0).ok().flatten()?;
        serde_json::from_slice::<DeepHusk>(&bytes).ok()
    }

    /// Whether `id` has a deep-spilled husk in the on-disk store.
    fn deep_contains(&self, id: &NodeId) -> bool {
        self.husk_store
            .as_ref()
            .and_then(|hs| hs.get(&id.0).ok().flatten())
            .is_some()
    }

    /// Every live deep-spilled `(id, husk)` — a full store scan. Lever 2 brick 6:
    /// until a keyed adjacency index lands, cold-neighbour and GC sweeps enumerate
    /// the whole deep tier this way.
    fn deep_entries(&self) -> Vec<(NodeId, DeepHusk)> {
        let Some(hs) = self.husk_store.as_ref() else {
            return Vec::new();
        };
        hs.live_entries()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(k, b)| {
                serde_json::from_slice::<DeepHusk>(&b)
                    .ok()
                    .map(|h| (NodeId(k), h))
            })
            .collect()
    }

    /// Number of live deep-spilled husks.
    fn deep_count(&self) -> usize {
        self.husk_store
            .as_ref()
            .map_or(0, |hs| hs.live_entries().map_or(0, |v| v.len()))
    }

    /// Number of COLD entries deep-spilled to disk (archived whole, represented in
    /// RAM only by a compact `DeepHusk` in the on-disk index).
    pub fn cold_deep_spilled_count(&self) -> usize {
        self.deep_count()
    }

    /// Whether `id` is a deep-spilled COLD entry (its whole node on disk, only a
    /// compact husk — body-blob stub + neighbour ids — in the index).
    pub fn is_deep_spilled(&self, id: &NodeId) -> bool {
        self.deep_contains(id)
    }

    /// Deep-spill the coldest COLD entries until resident COLD metadata
    /// ([`cold_resident_bytes`](Self::cold_resident_bytes)) is within
    /// [`cold_resident_budget`](Self::cold_resident_budget). Each chosen entry is
    /// serialized **whole** (node + content folded inline + edges) into one
    /// content-addressed blob and replaced in RAM by a compact `DeepHusk` (the
    /// body-blob stub + the neighbour ids) moved into [`cold_deep`](Self::cold_deep) —
    /// shedding the full `ColdNode` struct, which is the per-entry resident floor.
    /// Deterministic: coldest-first by causal score, ties on id. **Lossless**: the
    /// whole node faults back in [`page_in`](Self::page_in). A no-op without an
    /// attached store or a budget, or when already within it. An entry whose blob
    /// fails to write (or whose spilled content can't be faulted to fold in) is left
    /// intact — deep-spill never drops data; the budget is approached best-effort,
    /// never by dropping a node and never below the compact-husk floor.
    fn enforce_cold_resident_budget(&mut self) {
        let Some(budget) = self.cold_resident_budget else {
            return;
        };
        if self.spill.is_none() {
            return;
        }
        let mut resident = self.cold_resident_bytes();
        if resident <= budget {
            return;
        }
        // Coldest-first candidates from the full COLD map (deep husks already live in
        // `cold_deep` and are terminal). Scored up front so the mutation loop borrows
        // neither the weights nor the map.
        let mut candidates: Vec<(NodeId, f64)> = self
            .cold
            .iter()
            .map(|(id, c)| (id.clone(), self.compute_node_score(&c.node)))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        // Constituer le dictionnaire au premier spill qui en vaut la peine, à
        // partir des contenus sur le point d'être écrits — ils sont représentatifs
        // de ce que ce store contiendra. En deçà du seuil on s'en passe : le
        // dictionnaire coûterait plus d'octets qu'il n'en ferait gagner.
        self.maybe_build_spill_dictionary(&candidates);
        for (id, _) in candidates {
            if resident <= budget {
                break;
            }
            let Some(c) = self.cold.get(&id) else {
                continue;
            };
            let before = Self::entry_resident_bytes(&id, c);
            // Neighbour ids = the other endpoint of each archived edge (each is
            // incident to this node), sorted + deduped, self-loops dropped.
            let mut adj: Vec<NodeId> = c
                .edges
                .iter()
                .map(|e| {
                    if e.source == c.node.id {
                        e.target.clone()
                    } else {
                        e.source.clone()
                    }
                })
                .filter(|n| *n != c.node.id)
                .collect();
            adj.sort();
            adj.dedup();
            let adj_bytes: usize = adj
                .iter()
                .map(|n| std::mem::size_of::<NodeId>() + n.0.len())
                .sum();
            // The full `ColdNode` struct is replaced by a compact husk (body stub +
            // ids), so this almost always shrinks; the guard still parks the rare
            // entry that wouldn't — the floor, never a drop (mirrors compaction). The
            // body hash is inline in `size_of::<DeepHusk>()` (a raw `[u8; 32]`).
            let projected =
                std::mem::size_of::<DeepHusk>() + std::mem::size_of::<NodeId>() + adj_bytes;
            if projected >= before {
                continue;
            }
            // Fault any spilled content back inline so the body blob is
            // self-contained (one blob per husk → simple GC); its old content blob is
            // then orphaned and reclaimed below.
            let mut entry = c.clone();
            let old_content_hash = entry.spill.as_ref().map(|s| s.hash);
            if let Some(hash) = &old_content_hash {
                match self.spill.as_ref().and_then(|cfg| cfg.store.get(hash)) {
                    Some(text) => {
                        entry.node.content = text;
                        entry.spill = None;
                    }
                    None => continue, // content blob missing — leave the entry intact
                }
            }
            // Serialize the whole content-inline node as the body blob.
            let Ok(serialized) = serde_json::to_string(&entry) else {
                continue;
            };
            let Some(cfg) = self.spill.as_mut() else {
                return;
            };
            let SpillConfig {
                store, compressor, ..
            } = cfg;
            let Some(body_hash) = store.put(&serialized, compressor.as_mut()).ok() else {
                continue;
            };
            // Commit: archive the whole entry as a husk in the on-disk index, then
            // drop the resident full entry. The index is authoritative now (Lever 2),
            // so if the store write fails we leave the entry full rather than lose it.
            let husk = DeepHusk {
                body: SpillRef {
                    hash: body_hash,
                    len: serialized.len(),
                },
                adj: pack_adj(&adj),
            };
            let Ok(bytes) = serde_json::to_vec(&husk) else {
                continue;
            };
            let stored = self
                .husk_store
                .as_mut()
                .is_some_and(|hs| hs.put(&id.0, bytes).is_ok());
            if !stored {
                continue;
            }
            self.cold.remove(&id);
            if let Some(hash) = old_content_hash {
                self.release_content_blob_if_orphan(&hash);
            }
            // The full entry's resident cost is freed; the husk's is the store's
            // bounded footprint (counted in `cold_resident_bytes`), not per-entry.
            resident = resident.saturating_sub(before);
        }
    }

    /// Set the COLD **compaction budget** (slice 4) — the deepest tier. With
    /// `Some(bytes)`, total COLD *content* (inline + spilled) is driven toward
    /// `bytes` by **lossily compacting** the coldest entries: code is
    /// skeletonised, prose summarised, the full original discarded — so the
    /// backing store itself stays frugal. Applies immediately. `None` (default)
    /// disables compaction (COLD stays lossless). Compaction is **lossy** and
    /// observable ([`is_compacted`](Self::is_compacted)); it is *not* part of
    /// replay, so — like the spill store — it is an operational mode layered on
    /// the deterministic default path, not a change to it.
    pub fn set_cold_content_budget(&mut self, budget: Option<usize>) {
        self.cold_content_budget = budget;
        self.enforce_cold_content_budget();
    }

    /// Number of COLD entries whose content has been lossily compacted.
    pub fn cold_compacted_count(&self) -> usize {
        self.cold.values().filter(|c| c.compacted).count()
    }

    /// Whether `id` is a COLD entry whose content is a lossy compaction (a
    /// summary/skeleton), i.e. its full original has been discarded.
    pub fn is_compacted(&self, id: &NodeId) -> bool {
        self.cold.get(id).is_some_and(|c| c.compacted)
    }

    /// Whether `id` is a COLD entry parked at the compaction floor — compaction
    /// tried it and could not shrink it, so it is excluded from further attempts.
    pub fn is_at_floor(&self, id: &NodeId) -> bool {
        self.cold.get(id).is_some_and(|c| c.at_floor)
    }

    /// Lossy compaction of one content blob, routed by node kind: code →
    /// skeleton, JSON → crushed, prose → extractive summary. Pure/deterministic.
    /// Returns the compact form (callers adopt it only when it is actually
    /// shorter than the original).
    fn compact_content(kind: &str, content: &str) -> String {
        match ContentRouter::classify(kind, content) {
            Route::Code => CausalAst::skeletonize(content),
            Route::Json => CausalCrusher::crush(content),
            Route::Prose => CausalSumm::summarize(content, 0, 0.0),
        }
    }

    /// Compact the coldest COLD content until total COLD content (inline +
    /// spilled) is within [`cold_content_budget`](Self::cold_content_budget).
    /// Deterministic: coldest-first by causal score, ties on id. A no-op without
    /// a budget or when already within it. **Lossy**: each compacted entry's full
    /// original is replaced by its summary/skeleton and discarded (a spilled
    /// original's on-disk blob is orphaned — reclaimed by a future GC pass). An
    /// entry that cannot be made smaller is skipped (the summary is the floor),
    /// so the budget is approached best-effort, never by dropping a node.
    /// Reclaim the on-disk spill blob for `hash` **iff** no COLD entry still
    /// references it. Safe with content-dedup (two cold nodes can share a blob):
    /// the file is deleted only once its last referent is gone. A no-op without an
    /// attached store. This is the GC that keeps the spill store from leaking
    /// orphaned blobs when a spilled node is re-ingested, removed, or compacted.
    /// Reclaim a spilled **content** blob iff no resident COLD full entry references
    /// it. A content hash is never a deep-husk *body* hash (different byte strings ⇒
    /// different SHA-256), so this deliberately skips the husk scan that
    /// [`release_blob_if_orphan`](Self::release_blob_if_orphan) does — keeping the
    /// deep-spill and compaction loops, which release content blobs, off the `O(N)`
    /// deep-tier sweep (otherwise deep-spilling N entries is `O(N²)` disk).
    fn release_content_blob_if_orphan(&self, hash: &[u8; 32]) {
        let Some(cfg) = self.spill.as_ref() else {
            return;
        };
        let referenced = self
            .cold
            .values()
            .any(|c| c.spill.as_ref().is_some_and(|s| s.hash == *hash));
        if !referenced {
            cfg.store.remove(hash);
        }
    }

    fn release_blob_if_orphan(&self, hash: &[u8; 32]) {
        let Some(cfg) = self.spill.as_ref() else {
            return;
        };
        let referenced_by_full = self
            .cold
            .values()
            .any(|c| c.spill.as_ref().is_some_and(|s| s.hash == *hash));
        // Lever 2: husks are on disk, so this scans the index — correct but O(N) per
        // release. Only body-blob releases (page-in, removal) hit it; content-blob
        // releases use the cheaper variant above.
        let referenced_by_husk = self
            .deep_entries()
            .iter()
            .any(|(_, h)| h.body.hash == *hash);
        if !referenced_by_full && !referenced_by_husk {
            cfg.store.remove(hash);
        }
    }

    /// Drop any COLD shadow of `id` — a full `ColdNode` or a deep `DeepHusk` —
    /// and reclaim its on-disk blob(s) if no other entry still references them. Used
    /// when a fresh ingest supersedes the shadow, or on explicit removal.
    fn forget_cold_shadow(&mut self, id: &NodeId) {
        // Gather the reverse-adjacency entries to drop (full entry: archived edges;
        // deep husk: neighbour ids) alongside reclaiming the blobs.
        let mut others: Vec<NodeId> = Vec::new();
        if let Some(old) = self.cold.remove(id) {
            for e in &old.edges {
                others.push(if &e.source == id {
                    e.target.clone()
                } else {
                    e.source.clone()
                });
            }
            if let Some(s) = old.spill {
                self.release_content_blob_if_orphan(&s.hash);
            }
        }
        if let Some(husk) = self.deep_get(id) {
            for o in unpack_adj(&husk.adj) {
                others.push(NodeId(o.to_owned()));
            }
            if let Some(hs) = self.husk_store.as_mut() {
                let _ = hs.delete(&id.0); // drop the husk from the on-disk index
            }
            self.release_blob_if_orphan(&husk.body.hash);
        }
        for other in others {
            self.radj_del_edge(id, &other);
        }
    }

    fn enforce_cold_content_budget(&mut self) {
        let Some(budget) = self.cold_content_budget else {
            return;
        };
        let mut total = self.cold_inline_bytes() + self.cold_spilled_bytes();
        if total <= budget {
            return;
        }
        // Coldest-first candidates: not already compacted, and not parked at the
        // compaction floor (un-shrinkable — see `ColdNode::at_floor`).
        let mut candidates: Vec<(NodeId, f64)> = self
            .cold
            .iter()
            .filter(|(_, c)| !c.compacted && !c.at_floor)
            .map(|(id, c)| (id.clone(), self.compute_node_score(&c.node)))
            .collect();
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        // Constituer le dictionnaire au premier spill qui en vaut la peine, à
        // partir des contenus sur le point d'être écrits — ils sont représentatifs
        // de ce que ce store contiendra. En deçà du seuil on s'en passe : le
        // dictionnaire coûterait plus d'octets qu'il n'en ferait gagner.
        self.maybe_build_spill_dictionary(&candidates);
        for (id, _) in candidates {
            if total <= budget {
                break;
            }
            // Materialise the current content (faulting a spilled blob back in),
            // keeping the old spill hash so its blob can be reclaimed after.
            let (content, kind, old_len, old_hash) = match self.cold.get(&id) {
                Some(c) if !c.compacted => {
                    let kind = format!("{:?}", c.node.node_type);
                    match &c.spill {
                        Some(s) => match self.spill.as_ref().and_then(|cfg| cfg.store.get(&s.hash))
                        {
                            Some(text) => (text, kind, s.len, Some(s.hash)),
                            None => continue, // store detached / blob gone — leave it
                        },
                        None => (c.node.content.clone(), kind, c.node.content.len(), None),
                    }
                }
                _ => continue,
            };
            let compact = Self::compact_content(&kind, &content);
            if compact.len() >= content.len() {
                // Already at the floor: park it so later passes skip it (no repeat
                // work, no repeat disk read) rather than re-attempting every ingest.
                if let Some(entry) = self.cold.get_mut(&id) {
                    entry.at_floor = true;
                }
                continue;
            }
            let new_len = compact.len();
            if let Some(entry) = self.cold.get_mut(&id) {
                entry.node.content = compact;
                entry.spill = None; // discard the full original
                entry.compacted = true;
            }
            // The original (content) blob may now be orphaned — reclaim it if so.
            if let Some(h) = old_hash {
                self.release_content_blob_if_orphan(&h);
            }
            total = total.saturating_sub(old_len).saturating_add(new_len);
        }
    }

    /// Remove any edge whose endpoints are no longer present in the graph and
    /// return how many were pruned. With `add_edge` rejecting dangling edges
    /// this is normally a no-op, but it enforces the `edges ⊆ nodes × nodes`
    /// invariant after bulk or out-of-band mutations.
    pub fn prune_dangling_edges(&mut self) -> usize {
        let before = self.edges.len();
        let nodes = &self.nodes;
        self.edges
            .retain(|e| nodes.contains_key(&e.source) && nodes.contains_key(&e.target));
        before - self.edges.len()
    }

    pub fn propagate_failure(&mut self, origin_id: &NodeId, depth: u32, max_depth: u32) {
        if depth > max_depth {
            return;
        }
        // Get the origin's current failure relevance as the propagation base
        let base_value = self
            .nodes
            .get(origin_id)
            .map(|n| n.failure_relevance)
            .unwrap_or(0.0);

        // Find all edges where origin is the source
        let targets: Vec<(NodeId, f64)> = self
            .edges
            .iter()
            .filter(|e| &e.source == origin_id)
            .map(|e| (e.target.clone(), e.weight))
            .collect();

        let decay = self.scoring_weights.failure_decay;
        // Degree-aware damping: a node *distributes* its pressure across its
        // out-edges rather than replicating it to each. At or below `failure_fanout`
        // this is a no-op (`damp == 1`), so sparse causal chains still reach depth;
        // a hub (a file with dozens of contained symbols) is damped by
        // `fanout / out_degree`, which stops one over-connected node from flooding
        // the graph (FIELD_CAMPAIGN_H.md, root cause #2).
        let fanout = self.scoring_weights.failure_fanout.max(1.0);
        let damp = (fanout / (targets.len() as f64).max(fanout)).min(1.0);
        let floor = self.paging_threshold;
        for (target, weight) in targets {
            let propagation = base_value * weight * decay.powi(depth as i32) * damp;
            if let Some(node) = self.nodes.get_mut(&target) {
                node.failure_relevance = (node.failure_relevance + propagation).clamp(0.0, 1.0);
                node.recency = 1.0;
                node.last_accessed = self.clock;
            }
            // Stop relaying once the pressure added this hop is below the paging
            // floor: it cannot page anything in, and continuing only floods.
            if propagation > floor {
                self.propagate_failure(&target, depth + 1, max_depth);
            }
        }
    }

    /// Like [`propagate_failure`](Self::propagate_failure) but **bidirectional**:
    /// pressure flows to neighbours in *both* edge directions, so an injected
    /// fault reaches its upstream causes (callers/importers) as well as its
    /// downstream dependencies. Recursion only continues into a neighbour whose
    /// failure relevance actually grew, which (with the `depth ≤ max_depth` bound)
    /// keeps it terminating on cyclic graphs. Use when the failing node's *cause*
    /// may lie either side of it — e.g. bug-fix localisation.
    pub fn propagate_failure_bidirectional(
        &mut self,
        origin_id: &NodeId,
        depth: u32,
        max_depth: u32,
    ) {
        if depth > max_depth {
            return;
        }
        let base_value = self
            .nodes
            .get(origin_id)
            .map(|n| n.failure_relevance)
            .unwrap_or(0.0);

        // Neighbours via edges in either direction.
        let neighbours: Vec<(NodeId, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                if &e.source == origin_id {
                    Some((e.target.clone(), e.weight))
                } else if &e.target == origin_id {
                    Some((e.source.clone(), e.weight))
                } else {
                    None
                }
            })
            .collect();

        let decay = self.scoring_weights.failure_decay;
        // Degree-aware damping (see [`propagate_failure`](Self::propagate_failure)):
        // a hub distributes pressure across its neighbours instead of replicating it.
        let fanout = self.scoring_weights.failure_fanout.max(1.0);
        let damp = (fanout / (neighbours.len() as f64).max(fanout)).min(1.0);
        let floor = self.paging_threshold;
        for (nb, weight) in neighbours {
            let propagation = base_value * weight * decay.powi(depth as i32) * damp;
            let mut grew = false;
            if let Some(node) = self.nodes.get_mut(&nb) {
                let before = node.failure_relevance;
                node.failure_relevance = (before + propagation).clamp(0.0, 1.0);
                node.recency = 1.0;
                node.last_accessed = self.clock;
                grew = node.failure_relevance > before + 1e-9;
            }
            // Continue only into neighbours that actually grew (cycle-safe) and only
            // while the added pressure can still page something in (anti-flood).
            if grew && propagation > floor {
                self.propagate_failure_bidirectional(&nb, depth + 1, max_depth);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// A node by id, if present. (Read accessor — `nodes` is `pub(crate)` so
    /// external callers cannot break the `edges ⊆ nodes²` invariant by removing
    /// a node out from under its edges.)
    pub fn node(&self, id: &NodeId) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// A mutable node by id. Editing a node's *fields* (score, recency, failure
    /// pressure…) cannot break the structural invariant — only adding/removing
    /// nodes or edges can — so this is safe to expose where the structural
    /// mutators ([`add_edge`](Self::add_edge), [`upsert_node`](Self::upsert_node))
    /// are not.
    pub fn node_mut(&mut self, id: &NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    /// Whether a node with this id is present.
    pub fn contains_node(&self, id: &NodeId) -> bool {
        self.nodes.contains_key(id)
    }

    /// All node ids (unordered — sort if you need determinism).
    pub fn node_ids(&self) -> impl Iterator<Item = &NodeId> + '_ {
        self.nodes.keys()
    }

    /// All `(id, node)` pairs (unordered).
    pub fn node_entries(&self) -> impl Iterator<Item = (&NodeId, &GraphNode)> + '_ {
        self.nodes.iter()
    }

    /// All nodes (unordered).
    pub fn node_values(&self) -> impl Iterator<Item = &GraphNode> + '_ {
        self.nodes.values()
    }

    /// The edges as a **read-only** slice — callers can inspect but not push a
    /// dangling edge or remove an endpoint.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// The [`QBelief`] of a claim node — its dual-evidence belief state derived from the incoming
    /// [`Supports`](EdgeType::Supports) / [`Contradicts`](EdgeType::Contradicts) edges. A pure,
    /// deterministic fold over the resident edge set (no stored state, so `replay == live` holds and
    /// it costs nothing in the snapshot). A node with no such edges returns the neutral
    /// (`belief 0`, `conflict 0`). See [`QBelief`] for the `belief`/`conflict` formulas.
    pub fn qbelief(&self, claim: &NodeId) -> QBelief {
        let mut support = 0.0_f64;
        let mut contradiction = 0.0_f64;
        for e in &self.edges {
            if &e.target != claim {
                continue;
            }
            match e.edge_type {
                EdgeType::Supports => support += e.weight,
                EdgeType::Contradicts => contradiction += e.weight,
                _ => {}
            }
        }
        belief_from_sums(support, contradiction)
    }

    /// Every claim's [`QBelief`] in a **single** `O(edges + claims)` pass — the batch companion to the
    /// per-claim [`qbelief`](Self::qbelief) (which scans all edges each call, so `N` calls are
    /// `O(N·edges)`). Only nodes that are the target of at least one `Supports`/`Contradicts` edge
    /// appear; any other node has the neutral belief (`belief == 0`) by definition, so a caller treats a
    /// missing entry as neutral. Uses the same `belief_from_sums` formula as `qbelief`, so the two agree
    /// node-for-node, and accumulates in the graph's stable edge order so it is deterministic
    /// (`replay == live`). The causal-topology LSA weighting (`external_memory`) reads it to scale each
    /// document by its authority without paying the `O(N·edges)` of per-node `qbelief`.
    pub fn qbeliefs(&self) -> BTreeMap<NodeId, QBelief> {
        let mut sums: BTreeMap<NodeId, (f64, f64)> = BTreeMap::new();
        for e in &self.edges {
            match e.edge_type {
                EdgeType::Supports => sums.entry(e.target.clone()).or_default().0 += e.weight,
                EdgeType::Contradicts => sums.entry(e.target.clone()).or_default().1 += e.weight,
                _ => {}
            }
        }
        sums.into_iter()
            .map(|(id, (s, c))| (id, belief_from_sums(s, c)))
            .collect()
    }

    /// A **time-decayed** [`QBelief`]: each evidence edge's weight is scaled by
    /// `0.5^(age / half_life)`, where `age` is the clock ticks elapsed since the edge was asserted
    /// (its `created_at` vs the current [`clock`](Self::clock)) and `half_life` is the ticks an
    /// unreinforced piece of evidence takes to count for half. Stale evidence fades, so a claim
    /// whose evidence has all aged sees its support/contradiction **magnitudes** shrink toward 0, so
    /// both `belief` and `conflict` relax toward the neutral `0` (with the `ε` prior, faint evidence
    /// is both neutral *and* uncontested) — the "knowledge half-life". A freshly (re-)asserted edge
    /// keeps full weight, so re-assertion **reinforces**: recent one-sided evidence tips `belief` back
    /// up and drives `conflict` down, so it resolves a stale, never-reaffirmed dissent that plain
    /// [`qbelief`](Self::qbelief) would keep treating as fully contested forever.
    /// Lazy and pure: `O(edges)`, computed on demand from `created_at`/`clock` with no stored decay
    /// state, so it is deterministic and `replay == live` holds (replay reproduces the same clock and
    /// edges). `half_life <= 0` ⇒ no decay (identical to [`qbelief`](Self::qbelief)).
    pub fn qbelief_decayed(&self, claim: &NodeId, half_life: f64) -> QBelief {
        if half_life <= 0.0 {
            return self.qbelief(claim);
        }
        let now = self.clock;
        let mut support = 0.0_f64;
        let mut contradiction = 0.0_f64;
        for e in &self.edges {
            if &e.target != claim {
                continue;
            }
            let decayed = match e.edge_type {
                EdgeType::Supports | EdgeType::Contradicts => {
                    let age = now.saturating_sub(e.created_at) as f64;
                    e.weight * 0.5_f64.powf(age / half_life)
                }
                _ => continue,
            };
            if e.edge_type == EdgeType::Supports {
                support += decayed;
            } else {
                contradiction += decayed;
            }
        }
        belief_from_sums(support, contradiction)
    }

    /// **Belief propagation** along causal edges — a single deterministic hop. For every
    /// [`EdgeType::Causes`] edge `A → B` whose source claim `A` is **resolved** — its
    /// [`qbelief`](Self::qbelief) `belief` (signed, in `[−1, 1]`) has magnitude at least
    /// `resolve_threshold`: strongly believed (`belief ≥ resolve_threshold`) or strongly refuted
    /// (`belief ≤ −resolve_threshold`) — emit a *derived* evidence edge on the effect `B`: a
    /// [`Supports`](EdgeType::Supports) edge when the cause is believed, a
    /// [`Contradicts`](EdgeType::Contradicts) edge when it is refuted. So a claim with no direct
    /// evidence of its own still acquires a belief from the causes it depends on — belief revision
    /// across the causal graph, which a static evidence store cannot do. The derived weight is
    /// **attenuated** — `edge.weight · damping · |belief|` — so a barely-resolved or weakly-causal
    /// link contributes little and `B`'s induced belief is always weaker than `A`'s. Self-loops and
    /// unresolved causes (`belief` near `0`) are skipped.
    ///
    /// **One hop only** in this call: the derived edges are not themselves propagated onward — a
    /// chain `A → B → C` needs a second call.
    /// [`propagate_beliefs_converge`](Self::propagate_beliefs_converge) runs this sweep to a bounded
    /// fixpoint so the whole chain revises in a single call; a priority scheduler (topological order,
    /// fewer rounds) and recording it as a replayable `Op::Propagate` when wired into the agent loop
    /// remain later slices. Deterministic: candidates are collected read-only, **sorted**, then
    /// added (so iteration order cannot leak in) and [`add_edge`](Self::add_edge) dedups, so the pass
    /// is **idempotent**. Returns the number of derived edges added. `resolve_threshold` is clamped to
    /// `[0.0, 1.0]` (a distance from the neutral `0`), `damping` to `[0.0, 1.0]`.
    pub fn propagate_beliefs(&mut self, resolve_threshold: f64, damping: f64) -> usize {
        let resolve_threshold = resolve_threshold.clamp(0.0, 1.0);
        let damping = damping.clamp(0.0, 1.0);
        // Phase 1 (read-only): collect a derived edge for every resolved cause. `(source, target,
        // supports, weight)`; the weight is not part of the dedup key.
        let mut to_add: Vec<(NodeId, NodeId, bool, f64)> = Vec::new();
        for e in &self.edges {
            if e.edge_type != EdgeType::Causes || e.source == e.target {
                continue;
            }
            let belief = self.qbelief(&e.source).belief; // signed, in [-1, 1]
            if belief.abs() < resolve_threshold {
                continue; // cause not resolved (near neutral 0) ⇒ no propagation
            }
            let supports = belief > 0.0; // sign of the resolved belief is the polarity
            let certainty = belief.abs(); // in [0, 1]
            let weight = e.weight * damping * certainty;
            to_add.push((e.source.clone(), e.target.clone(), supports, weight));
        }
        // Deterministic add order; dedup on (source, target, polarity) — weight is not a key.
        to_add.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        to_add.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2);
        // Phase 2 (mutate): add the derived edges. `add_edge` dedups against existing edges too.
        let mut added = 0;
        for (src, tgt, supports, weight) in to_add {
            let et = if supports {
                EdgeType::Supports
            } else {
                EdgeType::Contradicts
            };
            if self.add_edge(src, tgt, weight, et) {
                added += 1;
            }
        }
        added
    }

    /// **Multi-hop belief propagation to a bounded fixpoint.** Repeatedly applies the single
    /// [`propagate_beliefs`](Self::propagate_beliefs) sweep until it **converges** — a round derives
    /// no new edge — or `max_rounds` is reached. Each sweep advances the belief wavefront exactly one
    /// causal hop (round *k* resolves a claim *k* `Causes`-hops downstream of a resolved cause), so a
    /// chain `A → B → C` fully revises in one call instead of a manual sweep per hop — the belief
    /// revision a static evidence store cannot do, now across the *whole* causal chain.
    ///
    /// The attenuation is what makes it safe: every hop multiplies the induced weight by
    /// `damping · |belief| < 1`, so a claim's inherited belief shrinks each hop and the wavefront
    /// **stops on its own** once it falls below `resolve_threshold` — a single resolved source cannot
    /// cascade into a storm.
    ///
    /// Deterministic and guaranteed to terminate. Every round reuses `propagate_beliefs`'
    /// collect → sort → dedup → [`add_edge`](Self::add_edge) discipline, and `add_edge` dedups on
    /// `(source, target, type)` (weight is not a key), so once a derived edge exists re-deriving it
    /// is a no-op — which makes "a round added zero new edges" a well-defined **fixpoint**. Since
    /// each directed `(cause, effect, polarity)` triple derives at most one edge, the total work is
    /// bounded no matter how large `max_rounds` is; the cap only backstops a `Causes` **cycle**,
    /// where mutual causation could otherwise keep re-resolving. Returns the total number of derived
    /// edges added across all rounds. `resolve_threshold` and `damping` are clamped per sweep by
    /// `propagate_beliefs`.
    pub fn propagate_beliefs_converge(
        &mut self,
        resolve_threshold: f64,
        damping: f64,
        max_rounds: usize,
    ) -> usize {
        let mut total = 0;
        for _ in 0..max_rounds {
            let added = self.propagate_beliefs(resolve_threshold, damping);
            if added == 0 {
                break; // fixpoint: this sweep derived nothing new
            }
            total += added;
        }
        total
    }

    /// **`do(X)` — the interventional impact of changing `origin`.** Returns the nodes that
    /// (transitively) **depend on** `origin` — the ones a change to it structurally forces — each
    /// weighted by the attenuated strength of the dependency path. This is a Pearl-style intervention,
    /// not a similarity query: `origin`'s own inputs are *severed* (it is pinned to `magnitude` and
    /// never re-driven), and impact flows strictly to its dependents.
    ///
    /// In CCOS's edge convention an edge `A → B` means **A depends on B** (`propagate_failure` walks
    /// out-edges as "downstream dependencies"; the reverse are the "upstream causes / callers"). So
    /// `do(X)` walks **in-edges** of the directional dependency subset ([`Calls`](EdgeType::Calls) /
    /// [`DependsOn`](EdgeType::DependsOn) / [`DataFlow`](EdgeType::DataFlow)): the callers/importers of
    /// `origin`, then *their* callers, up to `max_depth` hops, attenuating by `damping < 1` per hop.
    ///
    /// Bounded and always-terminating (a fixed `max_depth` wavefront, `origin` severed each round so a
    /// dependency cycle cannot re-drive it) and deterministic (edges iterated in canonical
    /// `(source, target)` order, impact accumulated in a [`BTreeMap`] — no `HashMap` order leaks in).
    /// Read-only: it mutates nothing, so it is `derived-not-stored` and `replay == live` is untouched.
    /// Returns `(node, impact)` pairs (excluding `origin`), sorted by impact descending then id. A
    /// probabilistic retriever has no such operator — it cannot sever a variable's inputs and
    /// enumerate the set its change forces.
    pub fn intervene(
        &self,
        origin: &NodeId,
        magnitude: f64,
        damping: f64,
        max_depth: usize,
    ) -> Vec<(NodeId, f64)> {
        if !self.nodes.contains_key(origin) {
            return Vec::new();
        }
        let damping = damping.clamp(0.0, 1.0);
        let orphan = |id: &NodeId| {
            self.nodes
                .get(id)
                .map(|n| n.state == NodeState::Orphan)
                .unwrap_or(true)
        };
        // The directional dependency edges, in canonical order, both endpoints live.
        let mut dep_edges: Vec<&GraphEdge> = self
            .edges
            .iter()
            .filter(|e| {
                matches!(
                    e.edge_type,
                    EdgeType::Calls | EdgeType::DependsOn | EdgeType::DataFlow
                ) && !orphan(&e.source)
                    && !orphan(&e.target)
            })
            .collect();
        dep_edges.sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.target)));

        let mut impact: BTreeMap<NodeId, f64> = BTreeMap::new();
        let mut current: BTreeMap<NodeId, f64> = BTreeMap::new();
        current.insert(origin.clone(), magnitude.max(0.0));
        for _ in 0..max_depth {
            let mut next: BTreeMap<NodeId, f64> = BTreeMap::new();
            for e in &dep_edges {
                // `e.source` depends on `e.target`; if the target carries impact, its dependent
                // (the source, a caller/importer) is forced by it.
                if let Some(&mass) = current.get(&e.target) {
                    let contrib = mass * e.weight * damping;
                    if contrib > 0.0 {
                        *next.entry(e.source.clone()).or_insert(0.0) += contrib;
                    }
                }
            }
            next.remove(origin); // severed: the intervened node is never re-driven (cycle-safe)
            if next.is_empty() {
                break;
            }
            for (n, add) in &next {
                *impact.entry(n.clone()).or_insert(0.0) += add;
            }
            current = next;
        }
        let mut out: Vec<(NodeId, f64)> = impact.into_iter().collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// **`blame(X)` — the candidate root causes of a failure at `origin`.** The dual of
    /// [`intervene`](Self::intervene): where `do(X)` walks *in-edges* to find what **depends on** `X`,
    /// `blame` walks *out-edges* of the same directional dependency subset ([`Calls`](EdgeType::Calls)
    /// / [`DependsOn`](EdgeType::DependsOn) / [`DataFlow`](EdgeType::DataFlow)) to find what `X`
    /// **depends on** — its transitive dependencies, the nodes a bug in `X` could actually live in.
    /// This is the differentiated bug-resolution direction: the culprit of a symptom is usually an
    /// upstream dependency of low similarity, which a cosine retriever ranks poorly and a directed
    /// dependency walk ranks first.
    ///
    /// Same guarantees as [`intervene`](Self::intervene): bounded (`max_depth` wavefront, `origin`
    /// severed so a cycle cannot re-drive it), deterministic (canonical edge order, [`BTreeMap`]
    /// accumulation), and read-only. Returns `(node, weight)` pairs (excluding `origin`), sorted by
    /// weight descending then id.
    pub fn blame(&self, origin: &NodeId, damping: f64, max_depth: usize) -> Vec<(NodeId, f64)> {
        if !self.nodes.contains_key(origin) {
            return Vec::new();
        }
        let damping = damping.clamp(0.0, 1.0);
        let orphan = |id: &NodeId| {
            self.nodes
                .get(id)
                .map(|n| n.state == NodeState::Orphan)
                .unwrap_or(true)
        };
        let mut dep_edges: Vec<&GraphEdge> = self
            .edges
            .iter()
            .filter(|e| {
                matches!(
                    e.edge_type,
                    EdgeType::Calls | EdgeType::DependsOn | EdgeType::DataFlow
                ) && !orphan(&e.source)
                    && !orphan(&e.target)
            })
            .collect();
        dep_edges.sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.target)));

        let mut blame: BTreeMap<NodeId, f64> = BTreeMap::new();
        let mut current: BTreeMap<NodeId, f64> = BTreeMap::new();
        current.insert(origin.clone(), 1.0);
        for _ in 0..max_depth {
            let mut next: BTreeMap<NodeId, f64> = BTreeMap::new();
            for e in &dep_edges {
                // Forward along a dependency: if `e.source` carries blame, its dependency `e.target`
                // (what it relies on) is a candidate cause.
                if let Some(&mass) = current.get(&e.source) {
                    let contrib = mass * e.weight * damping;
                    if contrib > 0.0 {
                        *next.entry(e.target.clone()).or_insert(0.0) += contrib;
                    }
                }
            }
            next.remove(origin);
            if next.is_empty() {
                break;
            }
            for (n, add) in &next {
                *blame.entry(n.clone()).or_insert(0.0) += add;
            }
            current = next;
        }
        let mut out: Vec<(NodeId, f64)> = blame.into_iter().collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// The evidence nodes pointing at `claim` with a given **polarity** —
    /// [`EdgeType::Supports`] for the affirmative surface, [`EdgeType::Contradicts`] for the
    /// negative one. Returned **sorted and deduped**, so the surface is deterministic. A
    /// contradiction-aware recall returns *both* surfaces of a claim; similarity-only retrieval
    /// returns neither *as such*, because it cannot label an item as support vs refutation.
    pub fn evidence_of(&self, claim: &NodeId, polarity: EdgeType) -> Vec<&NodeId> {
        let mut v: Vec<&NodeId> = self
            .edges
            .iter()
            .filter(|e| &e.target == claim && e.edge_type == polarity)
            .map(|e| &e.source)
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// Every claim node carrying at least one `Supports`/`Contradicts` assertion, paired with its
    /// derived [`QBelief`], sorted by **descending conflict** then id (deterministic). The data behind
    /// the Pro tension / audit surfaces — the belief computation itself is always available in the
    /// core; only the *reporting* of it is gated.
    pub fn claim_beliefs(&self) -> Vec<(NodeId, QBelief)> {
        let mut claims: Vec<NodeId> = self
            .edges
            .iter()
            .filter(|e| matches!(e.edge_type, EdgeType::Supports | EdgeType::Contradicts))
            .map(|e| e.target.clone())
            .collect();
        claims.sort();
        claims.dedup();
        let mut out: Vec<(NodeId, QBelief)> = claims
            .into_iter()
            .map(|c| {
                let q = self.qbelief(&c);
                (c, q)
            })
            .collect();
        out.sort_by(|a, b| {
            b.1.conflict
                .partial_cmp(&a.1.conflict)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// Count edges that violate `edges ⊆ nodes²` (an endpoint is missing),
    /// **without** modifying the graph — the read-only counterpart of
    /// [`prune_dangling_edges`](Self::prune_dangling_edges), used by integrity
    /// checks that must not mutate the snapshot they verify.
    pub fn dangling_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| !self.nodes.contains_key(&e.source) || !self.nodes.contains_key(&e.target))
            .count()
    }

    /// **Order-independent resolution** — the driver behind the deferred
    /// [`CcosMemory::resolve`](crate::external_memory::CcosMemory::resolve). It
    /// *prunes* the edges the three resolve passes own
    /// ([`prune_resolution_edges`](Self::prune_resolution_edges)), then rebuilds them
    /// from the **current** graph, so a resolved edge is a pure function of the final
    /// node + pending-ref state — never of ingestion order.
    ///
    /// Why prune-then-rebuild: the passes are add-only and *resolve-uniquely-or-skip*,
    /// so a bare `foo()` linked while `foo` was globally-unique would keep that `Calls`
    /// edge even after a later file makes `foo` ambiguous (an order-dependent stale
    /// edge). Pruning first means a name that became ambiguous is simply not re-linked,
    /// so per-file (eager) and batch (deferred) ingestion — and a replay that re-ingests
    /// the same files — all converge on the **same** graph (`replay == live` stays
    /// exact, and the replayable path may batch). Returns the number of cross-file
    /// **import** edges (re)added, matching the historical `link_module_imports` count.
    pub fn resolve_all(&mut self) -> usize {
        self.prune_resolution_edges();
        let cross = self.link_module_imports();
        let _ = self.resolve_symbol_calls();
        let _ = self.resolve_data_flow();
        cross
    }

    /// Remove the resolution-owned edges [`resolve_all`](Self::resolve_all) is about to
    /// rebuild, **without losing anything that can't be rebuilt**. Two ownership tiers:
    ///
    /// - **Import / module-hierarchy** edges — `file: → file:` [`DependsOn`](EdgeType::DependsOn)
    ///   / [`Contains`](EdgeType::Contains). Always pruned: `link_module_imports` rebuilds
    ///   them from the **durable** `file:`/`use:` node set, so this is loss-free even for a
    ///   checkpoint-loaded graph. (Parser `DependsOn`/`Contains` always target a
    ///   `use:`/`dep:`/`mod:`/`sym:` node, never `file: → file:`, so they're untouched.)
    /// - **`Calls` / `DataFlow`** — rebuilt from the **runtime-only** (`serde(skip)`)
    ///   `pending_calls`/`pending_data_refs`, which are empty after a load. Pruned **only**
    ///   when the edge's source file still has pending refs (this session, or a replay
    ///   re-ingest) so it *will* be rebuilt; a loaded file with no pending refs keeps its
    ///   edges (they were resolved correctly at ingest — pruning them would lose them).
    ///
    /// `edge_set` is rebuilt lazily by the next [`add_edge`](Self::add_edge) on the length
    /// mismatch, the same contract [`prune_dangling_edges`](Self::prune_dangling_edges)
    /// relies on. Returns the number of edges removed.
    pub fn prune_resolution_edges(&mut self) -> usize {
        let before = self.edges.len();
        // Disjoint field borrows: `retain` over `edges` while reading the pending maps.
        let pending_calls = &self.pending_calls;
        let pending_data_refs = &self.pending_data_refs;
        self.edges.retain(|e| match e.edge_type {
            EdgeType::DependsOn | EdgeType::Contains => {
                !(e.source.0.starts_with("file:") && e.target.0.starts_with("file:"))
            }
            EdgeType::Calls => !sym_file(&e.source).is_some_and(|f| pending_calls.contains_key(f)),
            EdgeType::DataFlow => {
                !sym_file(&e.source).is_some_and(|f| pending_data_refs.contains_key(f))
            }
            _ => true,
        });
        before - self.edges.len()
    }

    /// Resolve intra-crate imports into `file → file` dependency edges.
    ///
    /// The parser records imports as `use:<file>:<path>` nodes but does not link
    /// the importing file to the file that *defines* the imported module, so
    /// causally-related files end up connected only through shared `dep:` hubs.
    /// This pass maps each file to its module path (`src/a/b.rs` → `a::b`,
    /// `…/mod.rs|lib.rs|main.rs` to the parent) and, for every `use:` node,
    /// links the importer to the file whose module is the longest matching
    /// prefix of the import path. Idempotent (duplicate edges are rejected);
    /// returns the number of edges added. Opt-in — callers invoke it after
    /// ingesting a set of files (see [`crate::external_memory`]).
    pub fn link_module_imports(&mut self) -> usize {
        // Deterministic iteration: visit node ids in sorted order so the HashMap seed can't
        // change import resolution across processes (e.g. which file wins a `(crate, module)`
        // key collision via the last `insert`) — fresh ingestion must be reproducible, the
        // same property replay relies on.
        let mut sorted_ids: Vec<&NodeId> = self.nodes.keys().collect();
        sorted_ids.sort();
        // (crate, intra-module path) → file node.
        let mut index: HashMap<(String, String), NodeId> = HashMap::new();
        for id in &sorted_ids {
            if let Some(path) = id.0.strip_prefix("file:") {
                if let Some(km) = crate_and_module(path) {
                    index.insert(km, (*id).clone());
                }
            }
        }
        let mut to_add: Vec<(NodeId, NodeId, EdgeType)> = Vec::new();

        // (a) imports: importer → defining file.
        for id in &sorted_ids {
            let Some(rest) = id.0.strip_prefix("use:") else {
                continue;
            };
            let Some((file, usepath)) = rest.split_once(':') else {
                continue;
            };
            let importer = NodeId(format!("file:{file}"));
            if !self.nodes.contains_key(&importer) {
                continue;
            }
            let importer_crate = crate_and_module(file).map(|(c, _)| c).unwrap_or_default();
            if let Some(target) = resolve_use(&importer_crate, usepath, &index) {
                if target != importer {
                    to_add.push((importer, target, EdgeType::DependsOn));
                }
            }
        }

        // (b) module hierarchy: a parent module's file → its sub-module's file
        // (e.g. `filter/mod.rs → filter/owner.rs`). The parser records `pub mod x;`
        // only as a node, so without this a sub-module reached through a re-export
        // is orphaned and failure pressure can never flow into it.
        let mut entries: Vec<((String, String), NodeId)> =
            index.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort();
        for ((krate, module), child) in &entries {
            if module.is_empty() {
                continue;
            }
            let parent_module = match module.rsplit_once("::") {
                Some((p, _)) => p.to_string(),
                None => String::new(), // top-level module → crate root (lib/main)
            };
            if let Some(parent) = index.get(&(krate.clone(), parent_module)) {
                if parent != child {
                    to_add.push((parent.clone(), child.clone(), EdgeType::Contains));
                }
            }
        }

        let mut added = 0;
        for (s, t, ty) in to_add {
            if self.add_edge(s, t, 0.85, ty) {
                added += 1;
            }
        }
        added
    }

    /// Record (replacing) this file's in-body call-sites, the input to
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). Empty ⇒ drop the entry, so a
    /// re-ingest that removed every call leaves nothing stale. Each tuple is `(caller, callee,
    /// line)` in source order.
    pub fn set_pending_calls(&mut self, file: &str, calls: Vec<(String, String, usize)>) {
        if calls.is_empty() {
            self.pending_calls.remove(file);
        } else {
            self.pending_calls.insert(file.to_string(), calls);
        }
    }

    /// Record a file's in-body `static`/`const` references, awaiting
    /// [`resolve_data_flow`](Self::resolve_data_flow). Mirrors
    /// [`set_pending_calls`](Self::set_pending_calls).
    pub fn set_pending_data_refs(&mut self, file: &str, refs: Vec<(String, String, usize)>) {
        if refs.is_empty() {
            self.pending_data_refs.remove(file);
        } else {
            self.pending_data_refs.insert(file.to_string(), refs);
        }
    }

    /// Record (replacing) this file's renamed-import bindings, the alias input to
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). Each tuple is `(local_name,
    /// target_path)` — `use a::b as c` ⇒ `("c", "a::b")`. Empty ⇒ drop the entry (a re-ingest that
    /// removed every rename leaves nothing stale). Mirrors [`set_pending_calls`](Self::set_pending_calls).
    pub fn set_pending_aliases(&mut self, file: &str, aliases: Vec<(String, String)>) {
        if aliases.is_empty() {
            self.pending_aliases.remove(file);
        } else {
            self.pending_aliases.insert(file.to_string(), aliases);
        }
    }

    /// Record (replacing) this file's impl-block method-ownership facts — each `(type_name, method)`
    /// from an `impl Foo { fn bar }` — the input to the `(type, method) → symbol` index
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls) uses to resolve a `Foo::bar` callee
    /// (Slice 3). Empty ⇒ drop the entry. Mirrors [`set_pending_calls`](Self::set_pending_calls).
    pub fn set_pending_methods(&mut self, file: &str, methods: Vec<(String, String)>) {
        if methods.is_empty() {
            self.pending_methods.remove(file);
        } else {
            self.pending_methods.insert(file.to_string(), methods);
        }
    }

    /// Mark a symbol node as a `static`/`const` — the only valid `DataFlow` target. The parser
    /// calls this at ingest (the graph node itself does not carry `SymbolKind`).
    pub fn mark_data_symbol(&mut self, id: NodeId) {
        self.data_symbols.insert(id);
    }

    /// Resolve recorded `static`/`const` references into `reader → item` [`EdgeType::DataFlow`]
    /// edges — the shared-global-state channel call and import edges miss (a function reads a
    /// global defined in a file it never imports by name). A **whole-graph** pass, run after
    /// [`resolve_symbol_calls`](Self::resolve_symbol_calls). Always **resolve-uniquely-or-skip**, so
    /// a wrong edge is never invented. A **bare** ref `X` runs the same Tier A→B→C ladder as a bare
    /// call (import-scoped `use m::X` → the reader's own module → exactly one graph-wide `X`), against
    /// the data-symbol-only index — so a shared global reached through an explicit import, or defined
    /// in the reader's module, links even when `X` is not globally unique. A **qualified** ref `m::…::X` (Slice 2)
    /// pins its module prefix to a defining file via the SAME machinery as qualified calls
    /// (`resolve_qualified`) — crate-rooted, or an `alias::X` expanded through the file's imports —
    /// then links to that file's `sym:<file>:X` *only if it is a marked data symbol*;
    /// unresolvable/ambiguous qualified refs skip (and never fall back to the bare global index).
    /// Self-edges dropped. Deterministic: indices over the **sorted** `data_symbols`/node ids and
    /// `pending_data_refs`, candidate edges sorted+deduped before insertion. Returns the number of
    /// `DataFlow` edges added.
    pub fn resolve_data_flow(&mut self) -> usize {
        // Build the resolution indices, then collect candidate edges — in an inner scope so its
        // borrows of `self` end before the mutating `add_edge` below.
        let mut to_add: Vec<(NodeId, NodeId)> = {
            // Bare-name index: global-unique count over resident data symbols.
            let mut name_count: HashMap<&str, u32> = HashMap::new();
            let mut name_first: HashMap<&str, &NodeId> = HashMap::new();
            // Qualified-path index: (crate, module, name) → data-symbol node, restricted to data
            // symbols so a qualified ref can only ever resolve to a `static`/`const` (never a fn).
            let mut defs: HashMap<(String, String, String), NodeId> = HashMap::new();
            for id in &self.data_symbols {
                if !self.nodes.contains_key(id) {
                    continue; // evicted since it was marked — skip the stale entry
                }
                if let Some((file, name)) =
                    id.0.strip_prefix("sym:").and_then(|r| r.rsplit_once(':'))
                {
                    if let Some((c, m)) = crate_and_module(file) {
                        defs.insert((c, m, name.to_string()), id.clone());
                    }
                }
                if let Some((_, name)) = id.0.rsplit_once(':') {
                    *name_count.entry(name).or_insert(0) += 1;
                    name_first.entry(name).or_insert(id);
                }
            }
            // `file:`/`use:` indices for qualified-prefix resolution (mirrors `resolve_symbol_calls`).
            let mut file_index: HashMap<(String, String), NodeId> = HashMap::new();
            let mut scope: HashMap<String, Vec<String>> = HashMap::new();
            let mut sorted: Vec<&NodeId> = self.nodes.keys().collect();
            sorted.sort();
            for id in &sorted {
                let s = id.0.as_str();
                if let Some(path) = s.strip_prefix("file:") {
                    if let Some(km) = crate_and_module(path) {
                        file_index.insert(km, (*id).clone());
                    }
                } else if let Some(rest) = s.strip_prefix("use:") {
                    if let Some((f, usepath)) = rest.split_once(':') {
                        scope
                            .entry(f.to_string())
                            .or_default()
                            .push(usepath.to_string());
                    }
                }
            }

            // Renamed-import index (`use m::X as Y` ⇒ file → { Y: "m::X" }), mirroring
            // `resolve_symbol_calls`. First binding wins (deterministic: sorted `pending_aliases`).
            let mut alias_index: HashMap<&str, HashMap<&str, &str>> = HashMap::new();
            for (f, aliases) in &self.pending_aliases {
                let m = alias_index.entry(f.as_str()).or_default();
                for (local, target) in aliases {
                    m.entry(local.as_str()).or_insert(target.as_str());
                }
            }
            let empty_aliases: HashMap<&str, &str> = HashMap::new();

            let mut acc: Vec<(NodeId, NodeId)> = Vec::new();
            for (file, refs) in &self.pending_data_refs {
                let (fcrate, fmodule) = crate_and_module(file).unwrap_or_default();
                let file_alias = alias_index.get(file.as_str()).unwrap_or(&empty_aliases);
                for (reader, name, _line) in refs {
                    let reader_sym = NodeId(format!("sym:{file}:{reader}"));
                    if !self.nodes.contains_key(&reader_sym) {
                        continue;
                    }
                    // Renamed-import rewrite: `use m::X as Y` binds local `Y` to target `m::X`. If the
                    // ref's leading segment is such a local, rewrite it onto the target and resolve the
                    // rewritten path (mirrors resolve_call's alias handling), so an aliased data ref
                    // links to its real `static`/`const`. An alias is applied at most once.
                    let first = name.split("::").next().unwrap_or(name);
                    let rewritten: String = match file_alias.get(first) {
                        Some(target) => match name.split_once("::") {
                            Some((_, rest)) => format!("{target}::{rest}"),
                            None => (*target).to_string(),
                        },
                        None => name.clone(),
                    };
                    let target: Option<NodeId> = if rewritten.contains("::") {
                        // Qualified `m::…::X` — resolve exactly like a qualified call, against the
                        // data-symbol-only `defs` so the result is guaranteed a `static`/`const`.
                        resolve_qualified(file, &rewritten, &fcrate, &scope, &file_index, &defs)
                    } else {
                        // Bare `X` — the data-flow analogue of the call resolver's Tier A→B→C ladder
                        // (import-scoped → reader's own module → global-unique), against the
                        // data-symbol-only indices, so a shared global reached through an explicit
                        // `use m::X` — or defined in the reader's module — links even when `X` is not
                        // globally unique. resolve-uniquely-or-skip at every tier.
                        resolve_bare_data(
                            &rewritten,
                            file,
                            &fcrate,
                            &fmodule,
                            &scope,
                            &file_index,
                            &defs,
                            &name_count,
                            &name_first,
                        )
                    };
                    if let Some(t) = target {
                        if t != reader_sym {
                            acc.push((reader_sym.clone(), t));
                        }
                    }
                }
            }
            acc
        };
        to_add.sort();
        to_add.dedup();
        let mut added = 0usize;
        for (s, t) in to_add {
            if self.add_edge(s, t, 0.6, EdgeType::DataFlow) {
                added += 1;
            }
        }
        added
    }

    /// Resolve recorded call-sites into `caller → callee` [`EdgeType::Calls`] edges — the fn→fn
    /// structure import edges miss (a call crosses files even when the two functions share no
    /// vocabulary). A **whole-graph** pass (a call may target a symbol in a file ingested later);
    /// run right after [`link_module_imports`](Self::link_module_imports). Resolution is a strict
    /// precision ladder, **resolve-uniquely-or-skip** at every tier, so a wrong edge is never
    /// invented. A **`Self::method`** callee (a captured `self.m()` or `Self::assoc()`, Slice 3)
    /// resolves the method name in the caller's OWN module only — never via imports or global-unique,
    /// so a method is never mislinked to a same-named free function elsewhere. A **qualified** callee
    /// (`mod::…::name`, Slice 2) resolves its module prefix to a
    /// file — crate-rooted (`crate::m::name`) directly, or an `alias::name` by expanding the leading
    /// segment through the file's imports — then takes `sym:<file>:name`; unresolvable/ambiguous
    /// qualified paths skip without falling back. A **bare** callee (`foo()`, Slice 1) uses the
    /// ladder: (A) import-scoped — the file does `use …::foo`, its module resolving (as in
    /// `link_module_imports`) to the defining file with a real `sym:<file>:foo`; (B) same-module —
    /// `foo` defined in the caller's own file/module; (C) global-unique — exactly one `sym:*:foo`
    /// exists graph-wide. Self-edges are dropped. Deterministic: indices built over
    /// **sorted** node ids, calls iterated in sorted (file, source) order, candidate edges
    /// sorted+deduped before insertion. Returns the number of Calls edges added.
    pub fn resolve_symbol_calls(&mut self) -> usize {
        let mut sorted: Vec<&NodeId> = self.nodes.keys().collect();
        sorted.sort();
        let mut file_index: HashMap<(String, String), NodeId> = HashMap::new(); // (crate,module)->file
        let mut defs: HashMap<(String, String, String), NodeId> = HashMap::new(); // (crate,module,name)->sym
        let mut name_count: HashMap<String, u32> = HashMap::new();
        let mut name_first: HashMap<String, NodeId> = HashMap::new();
        let mut scope: HashMap<String, Vec<String>> = HashMap::new(); // file -> use paths
        for id in &sorted {
            let s = id.0.as_str();
            if let Some(path) = s.strip_prefix("file:") {
                if let Some(km) = crate_and_module(path) {
                    file_index.insert(km, (*id).clone());
                }
            } else if let Some(rest) = s.strip_prefix("sym:") {
                if let Some((file, name)) = rest.rsplit_once(':') {
                    if let Some((c, m)) = crate_and_module(file) {
                        defs.insert((c, m, name.to_string()), (*id).clone());
                    }
                    *name_count.entry(name.to_string()).or_default() += 1;
                    name_first
                        .entry(name.to_string())
                        .or_insert_with(|| (*id).clone());
                }
            } else if let Some(rest) = s.strip_prefix("use:") {
                if let Some((file, usepath)) = rest.split_once(':') {
                    scope
                        .entry(file.to_string())
                        .or_default()
                        .push(usepath.to_string());
                }
            }
        }

        // Per-file alias lookup: local name → target path (`use a::b as c` ⇒ `c` → `a::b`). Built
        // from the renamed-import bindings the parser handed over. A later call to a local name in
        // this map is rewritten onto its target before resolution (see `resolve_call`). An empty
        // (default) map for a file ⇒ no aliases, the unchanged path.
        let mut alias_index: HashMap<&str, HashMap<&str, &str>> = HashMap::new();
        for (file, aliases) in &self.pending_aliases {
            let m = alias_index.entry(file.as_str()).or_default();
            for (local, target) in aliases {
                // First binding wins (deterministic: pending lists are source-order). A duplicate
                // local name only arises in non-compiling Rust; keeping the first is harmless.
                m.entry(local.as_str()).or_insert(target.as_str());
            }
        }
        let empty_aliases: HashMap<&str, &str> = HashMap::new();

        // Slice 3 — the `(type_name, method) → method symbol` index for `x.bar()` / `Type::assoc()`
        // resolution, built from the impl-block facts the parser handed over (`pending_methods`).
        // `tm_count` carries each bucket's cardinality so a type name shared by two impls (a cross-file
        // / cross-crate homonym) is **ambiguous and skipped**, never resolved last-writer-wins.
        // Cardinality is counted over **all** ingested method defs (NOT just resident ones), so a
        // homonym whose one definition is paged COLD is still seen as ambiguous → skipped, rather than
        // resolved to the surviving twin (a false edge a resident-only count could mint). Only a
        // **resident** symbol can be the link *target* (you cannot point a fresh edge at a paged node);
        // a unique method whose symbol is COLD therefore drops the edge — precision-safe. Built over the
        // sorted `pending_methods` ⇒ deterministic and a pure function of the ingested op-stream.
        let mut type_method: HashMap<(String, String), NodeId> = HashMap::new();
        let mut tm_count: HashMap<(String, String), u32> = HashMap::new();
        for (file, methods) in &self.pending_methods {
            for (ty, method) in methods {
                let key = (ty.clone(), method.clone());
                *tm_count.entry(key.clone()).or_default() += 1;
                let sym = NodeId(format!("sym:{file}:{method}"));
                if self.nodes.contains_key(&sym) {
                    type_method.entry(key).or_insert(sym);
                }
            }
        }

        let mut to_add: Vec<(NodeId, NodeId)> = Vec::new();
        for (file, calls) in &self.pending_calls {
            let (fcrate, fmodule) = crate_and_module(file).unwrap_or_default();
            let file_aliases = alias_index.get(file.as_str()).unwrap_or(&empty_aliases);
            for (caller, callee, _line) in calls {
                let caller_sym = NodeId(format!("sym:{file}:{caller}"));
                if !self.nodes.contains_key(&caller_sym) {
                    continue;
                }
                if let Some(t) = resolve_call(
                    file,
                    callee,
                    &fcrate,
                    &fmodule,
                    &scope,
                    &file_index,
                    &defs,
                    &name_count,
                    &name_first,
                    file_aliases,
                    &type_method,
                    &tm_count,
                ) {
                    if t != caller_sym {
                        to_add.push((caller_sym, t));
                    }
                }
            }
        }
        to_add.sort();
        to_add.dedup();
        let mut added = 0;
        for (s, t) in to_add {
            if self.add_edge(s, t, 0.75, EdgeType::Calls) {
                added += 1;
            }
        }
        added
    }

    /// Detect directed dependency cycles via an iterative (stack-safe) DFS.
    /// Each returned vector lists the nodes forming one cycle, in order.
    pub fn find_cycles(&self) -> Vec<Vec<NodeId>> {
        use std::collections::BTreeMap;
        let mut adj: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for id in self.nodes.keys() {
            adj.entry(id.clone()).or_default();
        }
        for e in &self.edges {
            if self.nodes.contains_key(&e.source) && self.nodes.contains_key(&e.target) {
                adj.get_mut(&e.source).unwrap().push(e.target.clone());
            }
        }
        for v in adj.values_mut() {
            v.sort();
            v.dedup();
        }

        // color: 0 = unvisited, 1 = on the current DFS path, 2 = done
        let mut color: HashMap<NodeId, u8> = HashMap::new();
        let mut cycles: Vec<Vec<NodeId>> = Vec::new();

        for start in adj.keys() {
            if *color.get(start).unwrap_or(&0) != 0 {
                continue;
            }
            let mut stack: Vec<(NodeId, usize)> = vec![(start.clone(), 0)];
            let mut path: Vec<NodeId> = vec![start.clone()];
            color.insert(start.clone(), 1);
            while let Some((node, idx)) = stack.last().cloned() {
                let children = &adj[&node];
                if idx < children.len() {
                    stack.last_mut().unwrap().1 += 1;
                    let child = children[idx].clone();
                    match *color.get(&child).unwrap_or(&0) {
                        0 => {
                            color.insert(child.clone(), 1);
                            stack.push((child.clone(), 0));
                            path.push(child);
                        }
                        1 => {
                            // Back edge: a cycle spans path[pos..].
                            if let Some(pos) = path.iter().position(|n| n == &child) {
                                cycles.push(path[pos..].to_vec());
                            }
                        }
                        _ => {}
                    }
                } else {
                    color.insert(node.clone(), 2);
                    stack.pop();
                    path.pop();
                }
            }
        }
        cycles
    }

    /// Structural difference against another graph (added/removed nodes & edges).
    pub fn diff(&self, other: &MemoryGraph) -> GraphDiff {
        use std::collections::HashSet;
        let a: HashSet<&NodeId> = self.nodes.keys().collect();
        let b: HashSet<&NodeId> = other.nodes.keys().collect();

        let mut nodes_added: Vec<NodeId> = b.difference(&a).map(|n| (*n).clone()).collect();
        let mut nodes_removed: Vec<NodeId> = a.difference(&b).map(|n| (*n).clone()).collect();
        nodes_added.sort();
        nodes_removed.sort();

        let edge_key = |e: &GraphEdge| {
            (
                e.source.0.clone(),
                e.target.0.clone(),
                format!("{:?}", e.edge_type),
            )
        };
        let ea: HashSet<_> = self.edges.iter().map(edge_key).collect();
        let eb: HashSet<_> = other.edges.iter().map(edge_key).collect();

        GraphDiff {
            nodes_added,
            nodes_removed,
            edges_added: eb.difference(&ea).count(),
            edges_removed: ea.difference(&eb).count(),
            common_nodes: a.intersection(&b).count(),
        }
    }

    /// Count nodes by type, sorted by descending frequency (then name).
    pub fn node_type_counts(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for node in self.nodes.values() {
            *counts.entry(format!("{:?}", node.node_type)).or_insert(0) += 1;
        }
        let mut out: Vec<(String, usize)> = counts.into_iter().collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Nodes with no incident edges (neither incoming nor outgoing) — isolated
    /// fragments that no other part of the graph references.
    pub fn orphan_nodes(&self) -> Vec<&NodeId> {
        use std::collections::HashSet;
        let mut connected: HashSet<&NodeId> = HashSet::new();
        for e in &self.edges {
            connected.insert(&e.source);
            connected.insert(&e.target);
        }
        let mut orphans: Vec<&NodeId> = self
            .nodes
            .keys()
            .filter(|id| !connected.contains(id))
            .collect();
        orphans.sort();
        orphans
    }

    /// Render the graph as Graphviz DOT (deterministic node/edge order, colored
    /// by type) for visualization: `ccos analyze <path> --dot graph.dot`.
    pub fn to_dot(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let color = |t: &NodeType| match t {
            NodeType::Module => "#cfe2ff",
            NodeType::Symbol => "#d1e7dd",
            NodeType::ContextBlock => "#fff3cd",
            NodeType::AnalysisResult => "#f8d7da",
            NodeType::CodeRegion => "#e2e3e5",
            NodeType::Unknown => "#ffffff",
        };

        let mut out = String::from(
            "digraph ccos {\n  rankdir=LR;\n  node [shape=box, style=\"rounded,filled\", fontname=monospace];\n",
        );

        let mut ids: Vec<&NodeId> = self.nodes.keys().collect();
        ids.sort();
        for id in &ids {
            let n = &self.nodes[*id];
            out.push_str(&format!(
                "  \"{}\" [label=\"{}\", fillcolor=\"{}\"];\n",
                esc(&id.0),
                esc(&n.label),
                color(&n.node_type),
            ));
        }

        let mut edges: Vec<&GraphEdge> = self.edges.iter().collect();
        edges.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then_with(|| a.target.cmp(&b.target))
                .then_with(|| format!("{:?}", a.edge_type).cmp(&format!("{:?}", b.edge_type)))
        });
        for e in &edges {
            out.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{:?}\"];\n",
                esc(&e.source.0),
                esc(&e.target.0),
                e.edge_type,
            ));
        }
        out.push_str("}\n");
        out
    }

    /// Symbol nodes that **nothing references** — no incoming edge other than the structural
    /// `Contains` from their own file. A deliberate **heuristic**: a `pub` API, an entry point
    /// (`main`), or a trait-impl method reached only from outside the analyzed graph are false
    /// positives, so these are dead-code *candidates*, not a proof. Deterministic (sorted ids).
    pub fn dead_symbols(&self) -> Vec<NodeId> {
        let mut referenced: std::collections::HashSet<&NodeId> = std::collections::HashSet::new();
        for e in &self.edges {
            if e.edge_type != EdgeType::Contains {
                referenced.insert(&e.target);
            }
        }
        let mut dead: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(id, n)| {
                n.node_type == NodeType::Symbol
                    && id.0.starts_with("sym:")
                    && !referenced.contains(*id)
            })
            .map(|(id, _)| id.clone())
            .collect();
        dead.sort();
        dead
    }

    pub fn get_node_scores(&self) -> Vec<(NodeId, f64)> {
        let mut scores: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|(id, node)| (id.clone(), self.compute_node_score(node)))
            .collect();
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scores
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new(0.2, 100)
    }
}

/// Structural difference between two graphs, produced by [`MemoryGraph::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub common_nodes: usize,
}

/// The `::`-joined intra-crate module path of a source file below a `src/` root.
/// `a/b.rs` → `a::b`; `mod.rs`/`lib.rs`/`main.rs` → the parent (empty at the
/// crate root). Used by [`crate_and_module`].
fn module_of(after: &str) -> Option<String> {
    let p = after.strip_suffix(".rs")?;
    let segs: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
    let last = *segs.last()?;
    let module = if matches!(last, "mod" | "lib" | "main") {
        segs[..segs.len() - 1].join("::")
    } else {
        segs.join("::")
    };
    Some(module)
}

/// The source-file segment of a `sym:<file>:<name>` node id, or `None` for any other
/// id shape. Used by [`MemoryGraph::prune_resolution_edges`] to look a `Calls`/`DataFlow`
/// edge's origin file up in the pending-ref maps.
fn sym_file(id: &NodeId) -> Option<&str> {
    id.0.strip_prefix("sym:")
        .and_then(|r| r.rsplit_once(':'))
        .map(|(file, _name)| file)
}

/// Split a file path into `(crate, intra-module)` for import resolution, robust to
/// relative, absolute, and multi-crate-workspace layouts:
/// `src/a/b.rs` → `("", "a::b")`; `/repo/src/a.rs` → `("repo", "a")`;
/// `crates/grep-matcher/src/lib.rs` → `("grep_matcher", "")`. The crate is the
/// directory name immediately above `src/` (`-` normalised to `_`), or `""` for a
/// path that simply starts with `src/`.
fn crate_and_module(file_path: &str) -> Option<(String, String)> {
    let (krate, after) = if let Some(idx) = file_path.find("/src/") {
        let krate = file_path[..idx]
            .rsplit('/')
            .next()
            .unwrap_or("")
            .replace('-', "_");
        (krate, &file_path[idx + 5..])
    } else if let Some(after) = file_path.strip_prefix("src/") {
        (String::new(), after)
    } else {
        (String::new(), file_path)
    };
    Some((krate, module_of(after)?))
}

/// Parse a `use`/call path into `(target_crate, module_segments_after_the_crate_root)`.
/// `crate::`/`self::`/`super::` (and bare `crate`/`self`/`super`) stay in the importer's crate; a
/// leading crate name (e.g. `grep_matcher::…`) targets that crate.
fn parse_modpath(importer_crate: &str, usepath: &str) -> (String, Vec<String>) {
    let (target_crate, rest): (String, &str) = if let Some(r) = usepath
        .strip_prefix("crate::")
        .or_else(|| usepath.strip_prefix("self::"))
        .or_else(|| usepath.strip_prefix("super::"))
    {
        (importer_crate.to_string(), r)
    } else if matches!(usepath, "crate" | "self" | "super") {
        (importer_crate.to_string(), "")
    } else {
        match usepath.split_once("::") {
            Some((c, r)) => (c.to_string(), r),
            None => (usepath.to_string(), ""),
        }
    };
    let segs = rest
        .split("::")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    (target_crate, segs)
}

/// Resolve an import to the defining file's node id. `crate::`/`self::`/`super::` stay in the
/// importer's crate (requiring a real sub-module match); a leading crate name targets that crate
/// and may resolve to its root (`lib.rs`). Tries the full module path then **shortens** to ancestor
/// modules — correct for a `use`-target re-exported at an ancestor (`use a::Thing`, `Thing` living
/// in `a`'s file). External paths like `std::io` match nothing.
fn resolve_use(
    importer_crate: &str,
    usepath: &str,
    index: &HashMap<(String, String), NodeId>,
) -> Option<NodeId> {
    let (target_crate, segs) = parse_modpath(importer_crate, usepath);
    // Same-crate imports must hit a real sub-module (len ≥ 1); cross-crate imports
    // may resolve to the crate root (len 0) — depending on the crate as a whole.
    let min_len = usize::from(target_crate == importer_crate);
    (min_len..=segs.len()).rev().find_map(|len| {
        let module = segs[..len].join("::");
        index.get(&(target_crate.clone(), module)).cloned()
    })
}

/// Resolve a path to the file whose module is **exactly** that path — no ancestor shortening.
/// Used to pin a qualified call's *definition* module: unlike [`resolve_use`], a `crate::a::b` whose
/// `a::b` module has no file must NOT fall back to an existing `a` (that would attribute the call to
/// a wrong-existing symbol — the inline-submodule / ancestor-fallback / `Enum::Variant` false edge).
fn resolve_module_exact(
    importer_crate: &str,
    path: &str,
    index: &HashMap<(String, String), NodeId>,
) -> Option<NodeId> {
    let (target_crate, segs) = parse_modpath(importer_crate, path);
    index.get(&(target_crate, segs.join("::"))).cloned()
}

/// Bare (non-`crate`-rooted, non-imported) module-path fallback for `m::…::name` whose leading path is
/// a LOCAL submodule of the **caller's own crate** — the common `submod::fn()` / `outer::inner::fn()`
/// idiom with no `use`. The prefix must be a REAL submodule file of the caller's crate (matched
/// EXACTLY — no ancestor shortening); its existence is authoritative (a local `mod prefix;` **shadows**
/// a same-named extern crate in Rust path resolution), so a present-but-symbol-less module yields
/// `None` (skip). Only the caller's OWN crate is consulted: a bare path is deliberately NOT resolved
/// into an unrelated ingested EXTERNAL crate — an external reading would mint an edge the compiler
/// never picks. (Adversarial review confirmed the two failures an external interpretation caused: a
/// symbol-less local module + a same-named extern crate produced a false edge, and a crate named like
/// a type stole a valid type-method edge — so the external reading is excluded.) Present-module-then-
/// symbol gating ⇒ resolve-uniquely-or-skip. Reads only the sorted-built `file_index`/`defs` ⇒
/// order-independent (replay == live); never consults `name_count`/`name_first` ⇒ no bare-name guess.
fn resolve_bare_modpath(
    prefix: &str,
    name: &str,
    fcrate: &str,
    file_index: &HashMap<(String, String), NodeId>,
    defs: &HashMap<(String, String, String), NodeId>,
) -> Option<NodeId> {
    // The prefix must be a real submodule file of the CALLER'S crate (exact — no ancestor shortening).
    // Its mere existence is authoritative (shadow rule), so the symbol is then looked up in exactly
    // that module and a miss yields None — never a fall-through to a same-named extern crate.
    file_index.get(&(fcrate.to_string(), prefix.to_string()))?;
    defs.get(&(fcrate.to_string(), prefix.to_string(), name.to_string()))
        .cloned()
}

/// Resolve a **qualified** path `mod::…::name` to its definition symbol — the shared machinery
/// behind both qualified CALLS (`m::func()`) and qualified DATA references (`m::CONST`). Pins the
/// module prefix to its defining file (crate-rooted directly, else by expanding the leading segment
/// through `file`'s imports), then takes that file's `sym:<file>:name`. **resolve-uniquely-or-skip**:
/// an alias matching no import — or more than one — skips, the module is resolved EXACTLY (no
/// ancestor shortening, so `crate::a::b::name` with no `a::b` file never falls back to `a`'s
/// symbols), and a qualified path never falls through to a bare same-module/global guess. `self`/
/// `super`/`Self`/a type prefix and external roots (`std::…`) match no import here → skipped.
fn resolve_qualified(
    file: &str,
    path: &str,
    fcrate: &str,
    scope: &HashMap<String, Vec<String>>,
    file_index: &HashMap<(String, String), NodeId>,
    defs: &HashMap<(String, String, String), NodeId>,
) -> Option<NodeId> {
    let (prefix, name) = path.rsplit_once("::")?;
    let abs: Option<String> = if prefix == "crate" || prefix.starts_with("crate::") {
        Some(prefix.to_string())
    } else {
        let first = prefix.split("::").next().unwrap_or(prefix);
        let rest = prefix.strip_prefix(first).unwrap_or(""); // "" or "::b::c"
        let mut hits = scope
            .get(file)
            .into_iter()
            .flatten()
            .filter(|u| u.rsplit("::").next() == Some(first));
        match (hits.next(), hits.next()) {
            (Some(use_path), None) => Some(format!("{use_path}{rest}")),
            // NO matching import ⇒ try the bare same-crate submodule-path fallback. It performs the
            // full file_index→defs lookup itself (returning a SYMBOL, not a module path), so
            // short-circuit here rather than flow into the resolve_module_exact tail below.
            (None, None) => return resolve_bare_modpath(prefix, name, fcrate, file_index, defs),
            // 2+ matching imports (ambiguous) still skips — never fall to the bare guess, which would
            // contradict a scope the imports already made ambiguous (Tier-A discipline).
            _ => None,
        }
    };
    abs.and_then(|a| resolve_module_exact(fcrate, &a, file_index))
        .and_then(|df| df.0.strip_prefix("file:").map(str::to_string))
        .and_then(|f| crate_and_module(&f))
        .and_then(|(c, m)| defs.get(&(c, m, name.to_string())).cloned())
}

/// The Slice-1 call→def resolution ladder used by
/// [`MemoryGraph::resolve_symbol_calls`](MemoryGraph::resolve_symbol_calls): Tier A (import-scoped),
/// then B (same-module), then C (global-unique), each **resolve-uniquely-or-skip**. Returns the
/// definition symbol node for a bare `callee()` call in `file`, or `None` to skip. `aliases` maps a
/// renamed-import local name to its original target path (`use a::b as c` ⇒ `c` → `a::b`); a call
/// whose leading segment is such a local name is rewritten onto the target first, so it resolves
/// exactly as the original path would — never onto a same-named sibling module/symbol.
#[allow(clippy::too_many_arguments)]
fn resolve_call(
    file: &str,
    callee: &str,
    fcrate: &str,
    fmodule: &str,
    scope: &HashMap<String, Vec<String>>,
    file_index: &HashMap<(String, String), NodeId>,
    defs: &HashMap<(String, String, String), NodeId>,
    name_count: &HashMap<String, u32>,
    name_first: &HashMap<String, NodeId>,
    aliases: &HashMap<&str, &str>,
    type_method: &HashMap<(String, String), NodeId>,
    tm_count: &HashMap<(String, String), u32>,
) -> Option<NodeId> {
    // Renamed import — `use a::b as c` binds local `c` to target `a::b`. If the call's LEADING
    // segment is such a local name, rewrite it onto the target (`c` ⇒ `a::b`, `c::X` ⇒ `a::b::X`)
    // and re-resolve the rewritten path as if it had been written literally. Re-resolution passes
    // an EMPTY alias map, so an alias is applied at most once (no re-aliasing, no recursion loop).
    // This keeps resolution byte-for-byte identical to the original path: a bare alias falls through
    // the same Tier A/B/C ladder, a multi-segment alias through the qualified-path logic — and the
    // resolve-uniquely-or-skip guards (exact module match, global-unique) still gate every edge, so
    // an alias whose target has no resident symbol simply yields no edge.
    if !aliases.is_empty() {
        let first = callee.split("::").next().unwrap_or(callee);
        if let Some(target) = aliases.get(first) {
            let rewritten = match callee.split_once("::") {
                Some((_, rest)) => format!("{target}::{rest}"),
                None => (*target).to_string(),
            };
            let empty: HashMap<&str, &str> = HashMap::new();
            return resolve_call(
                file,
                &rewritten,
                fcrate,
                fmodule,
                scope,
                file_index,
                defs,
                name_count,
                name_first,
                &empty,
                type_method,
                tm_count,
            );
        }
    }
    // Slice 3 — `Self::method` (a captured `self.m()` or `Self::assoc()`): the receiver is the
    // enclosing impl's type, whose methods live in the caller's own file/module. Resolve the method
    // name there ONLY — never via imports (Tier A) or global-unique (Tier C), so a method is never
    // mislinked to a free function of the same name elsewhere. Unresolvable → skip.
    if let Some(method) = callee.strip_prefix("Self::") {
        return defs
            .get(&(fcrate.to_string(), fmodule.to_string(), method.to_string()))
            .cloned();
    }
    // Slice 2 — qualified path `mod::…::name`: pin the module prefix to its defining file, then take
    // the unique `sym:<file>:name`. resolve-uniquely-or-skip, so anything unresolvable or ambiguous →
    // no edge (and a qualified path never falls through to the bare same-module/global ladder below).
    if callee.contains("::") {
        // Slice 3 — a 2-segment `A::b` is ambiguous between a **module path** (`module::fn`) and a
        // **type method** (`Type::method`, minted by `x.bar()` inference or written as `Foo::bar()`).
        // Resolve BOTH interpretations and link only when they AGREE or exactly one resolves —
        // resolve-uniquely-or-skip: a real collision where both resolve to *different* symbols is
        // dropped, never guessed. A 3+-segment path (`a::b::C::d`) has a multi-segment prefix, so the
        // type side abstains and behaviour is exactly the prior module-only resolution.
        let module_hit = resolve_qualified(file, callee, fcrate, scope, file_index, defs);
        let type_hit = match callee.rsplit_once("::") {
            Some((ty, method)) if !ty.contains("::") => {
                let key = (ty.to_string(), method.to_string());
                // Only a graph-wide-unique `(type, method)` resolves; a homonym bucket abstains.
                (tm_count.get(&key).copied() == Some(1))
                    .then(|| type_method.get(&key).cloned())
                    .flatten()
            }
            _ => None,
        };
        return match (module_hit, type_hit) {
            (Some(m), Some(t)) if m == t => Some(m),
            (Some(_), Some(_)) => None, // both resolve, to DIFFERENT symbols ⇒ ambiguous ⇒ skip
            (Some(m), None) => Some(m),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        };
    }
    // Tier A — import-scoped: an import `use <module>::callee` pins the defining module; resolve it
    // to a file and require exactly one unique `sym:<file>:callee`. (Rust-name-resolution-correct:
    // links the cross-module call without linking same-named fns in unrelated modules.)
    if let Some(paths) = scope.get(file) {
        let mut cands: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        for usepath in paths {
            if usepath.rsplit("::").next() != Some(callee) {
                continue; // this import does not bring in `callee`
            }
            let Some((module_path, _)) = usepath.rsplit_once("::") else {
                continue; // bare `use callee;` — no module path to resolve in Slice 1
            };
            if let Some(deffile_node) = resolve_use(fcrate, module_path, file_index) {
                if let Some(deffile) = deffile_node.0.strip_prefix("file:") {
                    if let Some((c, m)) = crate_and_module(deffile) {
                        if let Some(sym) = defs.get(&(c, m, callee.to_string())) {
                            cands.insert(sym.clone());
                        }
                    }
                }
            }
        }
        if cands.len() == 1 {
            return cands.into_iter().next();
        }
        if cands.len() > 1 {
            // The import scope itself is ambiguous for `callee` (only possible in non-compiling
            // Rust — duplicate imports). Honour resolve-uniquely-or-skip: do NOT fall through to a
            // same-module local guess, which would link a call the import scope already contradicts.
            return None;
        }
    }
    // Tier B — same module as the caller.
    if let Some(sym) = defs.get(&(fcrate.to_string(), fmodule.to_string(), callee.to_string())) {
        return Some(sym.clone());
    }
    // Tier C — exactly one symbol of this name graph-wide (prelude / sibling re-export).
    if name_count.get(callee).copied() == Some(1) {
        return name_first.get(callee).cloned();
    }
    None
}

/// Bare data-ref resolution — the data-flow analogue of [`resolve_call`]'s Tier A→B→C ladder for a
/// bare `name`, against the **data-symbol-only** `defs`/name indices. Tier A (import-scoped): a
/// `use <module>::name` pins the defining module (the same [`resolve_use`] resolver bare calls use)
/// and requires a unique data symbol there. Tier B: a `static`/`const` of that name in the reader's
/// own module. Tier C: exactly one such symbol graph-wide. resolve-uniquely-or-skip at every tier, so
/// a shared global reached through an explicit import (or same-module) links even when its name is not
/// globally unique — and a wrong edge is never invented.
#[allow(clippy::too_many_arguments)]
fn resolve_bare_data(
    name: &str,
    file: &str,
    fcrate: &str,
    fmodule: &str,
    scope: &HashMap<String, Vec<String>>,
    file_index: &HashMap<(String, String), NodeId>,
    defs: &HashMap<(String, String, String), NodeId>,
    name_count: &HashMap<&str, u32>,
    name_first: &HashMap<&str, &NodeId>,
) -> Option<NodeId> {
    // Tier A — import-scoped: `use <module>::name` pins the defining module; require a unique data
    // symbol there. Mirrors resolve_call Tier A exactly (same `resolve_use`), against data-only `defs`.
    if let Some(paths) = scope.get(file) {
        let mut cands: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        for usepath in paths {
            if usepath.rsplit("::").next() != Some(name) {
                continue; // this import does not bring in `name`
            }
            let Some((module_path, _)) = usepath.rsplit_once("::") else {
                continue; // bare `use name;` — no module path to resolve
            };
            if let Some(deffile_node) = resolve_use(fcrate, module_path, file_index) {
                if let Some(deffile) = deffile_node.0.strip_prefix("file:") {
                    if let Some((c, m)) = crate_and_module(deffile) {
                        if let Some(sym) = defs.get(&(c, m, name.to_string())) {
                            cands.insert(sym.clone());
                        }
                    }
                }
            }
        }
        if cands.len() == 1 {
            return cands.into_iter().next();
        }
        if cands.len() > 1 {
            return None; // ambiguous imports ⇒ skip, never fall through to a contradicted guess
        }
    }
    // Tier B — a `static`/`const` of this name in the reader's own module.
    if let Some(sym) = defs.get(&(fcrate.to_string(), fmodule.to_string(), name.to_string())) {
        return Some(sym.clone());
    }
    // Tier C — exactly one data symbol of this name graph-wide.
    if name_count.get(name).copied() == Some(1) {
        return name_first.get(name).map(|t| (*t).clone());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_module_imports_connects_cross_file_uses() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for f in ["src/api.rs", "src/repo.rs", "src/db.rs"] {
            g.upsert_node(
                format!("file:{f}").into(),
                f.into(),
                "".into(),
                NodeType::Module,
            );
        }
        g.upsert_node(
            "use:src/api.rs:crate::repo".into(),
            "".into(),
            "".into(),
            NodeType::Unknown,
        );
        g.upsert_node(
            "use:src/repo.rs:crate::db::connect".into(),
            "".into(),
            "".into(),
            NodeType::Unknown,
        );
        let added = g.link_module_imports();
        assert_eq!(added, 2, "two cross-file imports resolved");
        assert!(g
            .edges
            .iter()
            .any(|e| e.source.0 == "file:src/api.rs" && e.target.0 == "file:src/repo.rs"));
        assert!(g
            .edges
            .iter()
            .any(|e| e.source.0 == "file:src/repo.rs" && e.target.0 == "file:src/db.rs"));
        assert_eq!(g.link_module_imports(), 0, "idempotent");
    }

    #[test]
    fn link_module_imports_handles_multi_crate_and_absolute_paths() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let node = |g: &mut MemoryGraph, id: &str| {
            g.upsert_node(id.into(), "".into(), "".into(), NodeType::Module);
        };
        // Multi-crate workspace: core imports a sibling crate `util` (dir uses `-`).
        node(&mut g, "file:crates/core/src/api.rs");
        node(&mut g, "file:crates/grep-util/src/lib.rs");
        node(&mut g, "use:crates/core/src/api.rs:grep_util::helper");
        // Absolute-path mono-crate with an intra-crate import.
        node(&mut g, "file:/repo/src/a.rs");
        node(&mut g, "file:/repo/src/b.rs");
        node(&mut g, "use:/repo/src/a.rs:crate::b");
        g.link_module_imports();
        assert!(
            g.edges
                .iter()
                .any(|e| e.source.0 == "file:crates/core/src/api.rs"
                    && e.target.0 == "file:crates/grep-util/src/lib.rs"),
            "cross-crate import (grep_util ← grep-util dir) resolves to the crate root"
        );
        assert!(
            g.edges
                .iter()
                .any(|e| e.source.0 == "file:/repo/src/a.rs" && e.target.0 == "file:/repo/src/b.rs"),
            "absolute-path intra-crate import resolves"
        );
    }

    #[test]
    fn bidirectional_failure_reaches_upstream_causes() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("a".into(), "a".into(), "".into(), NodeType::Module);
        g.upsert_node("b".into(), "b".into(), "".into(), NodeType::Module);
        // a depends on b (a → b); b is upstream of nothing, a is upstream of b.
        g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        // Inject at b: downstream-only propagation cannot reach a.
        g.set_failure_relevance(&"b".into(), 1.0);
        g.propagate_failure(&"b".into(), 0, 3);
        assert_eq!(g.nodes.get(&"a".into()).unwrap().failure_relevance, 0.0);
        // Bidirectional propagation reaches the upstream cause a.
        g.propagate_failure_bidirectional(&"b".into(), 0, 3);
        assert!(g.nodes.get(&"a".into()).unwrap().failure_relevance > 0.0);
    }

    #[test]
    fn test_upsert_node() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node(
            "n1".into(),
            "Test".into(),
            "content".into(),
            NodeType::Module,
        );
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);
        graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Module);
        graph.add_edge("a".into(), "b".into(), 0.8, EdgeType::DependsOn);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn scoring_weights_default_matches_shipped_constants() {
        // Regression guard: the tunable defaults must reproduce the original
        // hard-coded score, so existing snapshots/hashes/behaviour are unchanged.
        let w = ScoringWeights::default();
        assert_eq!(w.w_base, 0.15);
        assert_eq!(w.w_failure, 0.50);
        assert_eq!(w.w_recency, 0.30);
        assert_eq!(w.w_access, 0.05);
        assert_eq!(w.failure_decay, 0.8);
        assert_eq!(w.failure_fanout, 6.0);
    }

    #[test]
    fn scoring_weights_alter_score() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("n".into(), "n".into(), "".into(), NodeType::Module);
        g.set_failure_relevance(&"n".into(), 1.0);
        let node = g.nodes.get(&"n".into()).unwrap().clone();
        g.set_scoring_weights(ScoringWeights {
            w_failure: 0.0,
            ..Default::default()
        });
        let low = g.compute_node_score(&node);
        g.set_scoring_weights(ScoringWeights {
            w_failure: 1.0,
            ..Default::default()
        });
        let high = g.compute_node_score(&node);
        assert!(
            high > low,
            "raising w_failure must raise a failing node's score"
        );
    }

    #[test]
    fn touch_refreshes_resident_score_and_never_resurrects_cold() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        g.upsert_node("a".into(), "a".into(), "x".into(), NodeType::Module);
        // Let recency decay, then touch must restore it (recency → 1.0, access++).
        for _ in 0..5 {
            g.tick();
        }
        let before = g.compute_node_score(g.node(&"a".into()).unwrap());
        assert!(g.touch(&"a".into()), "a resident node is touchable");
        let after = g.compute_node_score(g.node(&"a".into()).unwrap());
        assert!(
            after > before,
            "touch must raise the score via recency/access refresh: {before} -> {after}"
        );
        // access_count: 1 at ingest, +1 from the single touch (tick does not bump it).
        assert_eq!(g.node(&"a".into()).unwrap().access_count, 2);
        // A cold (demoted) node is never paged back in by touch.
        g.max_in_memory_nodes = 0;
        g.enforce_paging();
        assert!(g.is_cold(&"a".into()));
        assert!(
            !g.touch(&"a".into()),
            "touch must not resurrect a cold node"
        );
        assert!(
            g.is_cold(&"a".into()),
            "touch must leave the COLD tier untouched"
        );
        // An absent id is simply a miss.
        assert!(!g.touch(&"missing".into()));
    }

    /// Eigenvector centrality is a pure function of the graph and handles the trivial
    /// shapes (empty, edgeless, single hub).
    #[test]
    fn eigencentrality_is_deterministic_and_handles_edge_cases() {
        // Empty graph → empty vector.
        assert!(MemoryGraph::new(0.2, usize::MAX)
            .eigencentrality()
            .is_empty());

        // Edgeless graph → every node equal (normalized to 1.0).
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["a", "b", "c"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        let iso = g.eigencentrality();
        assert_eq!(iso.len(), 3);
        assert!(iso.values().all(|v| (v - 1.0).abs() < 1e-9));

        // A hub depended-upon by three leaves: deterministic, and the hub leads.
        let build = || {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            for n in ["hub", "x", "y", "z"] {
                g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
            }
            for s in ["x", "y", "z"] {
                g.add_edge(s.into(), "hub".into(), 1.0, EdgeType::DependsOn);
            }
            g.eigencentrality()
        };
        assert_eq!(
            build(),
            build(),
            "eigencentrality is a pure function of the graph"
        );
        let ec = build();
        let hub = ec.get(&"hub".into()).copied().unwrap();
        for leaf in ["x", "y", "z"] {
            assert!(
                hub > ec.get(&leaf.into()).copied().unwrap(),
                "hub outranks its leaves"
            );
        }
    }

    /// Eigencentrality must be invariant to the *order* edges were inserted (import
    /// resolution adds them in HashMap order), not merely reproducible for one fixed order —
    /// otherwise its `f64`s drift across processes. Same graph, edges added in two different
    /// orders ⇒ identical vector.
    #[test]
    fn eigencentrality_is_invariant_to_edge_insertion_order() {
        let mk = |edges: &[(&str, &str)]| {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            for n in ["a", "b", "c", "d", "hub"] {
                g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
            }
            for &(s, t) in edges {
                g.add_edge(s.into(), t.into(), 1.0, EdgeType::DependsOn);
            }
            g.eigencentrality()
        };
        let forward = mk(&[
            ("a", "hub"),
            ("b", "hub"),
            ("c", "hub"),
            ("d", "c"),
            ("c", "b"),
        ]);
        let shuffled = mk(&[
            ("c", "b"),
            ("d", "c"),
            ("c", "hub"),
            ("a", "hub"),
            ("b", "hub"),
        ]);
        assert_eq!(
            forward, shuffled,
            "eigencentrality must not depend on edge insertion order"
        );
    }

    /// The point of eigenvector centrality: it ranks a *recursively* central node above
    /// one with higher *raw* in-degree — exactly the case in-degree gets wrong.
    #[test]
    fn eigenvector_centrality_captures_recursive_importance_indegree_misses() {
        // 3 mid-hubs, each depended-upon by 5 leaves; all 3 mids depend on `top`.
        //   in-degree:  each mid = 5, top = 3  → ranks a mid ABOVE top
        //   eigenvector: top inherits the mids' mass → ranks top ABOVE a mid
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        g.upsert_node("top".into(), "top".into(), "".into(), NodeType::Module);
        for m in 0..3 {
            let mid = format!("mid{m}");
            g.upsert_node(mid.clone().into(), mid.clone(), "".into(), NodeType::Module);
            g.add_edge(mid.clone().into(), "top".into(), 1.0, EdgeType::DependsOn);
            for l in 0..5 {
                let leaf = format!("leaf{m}_{l}");
                g.upsert_node(
                    leaf.clone().into(),
                    leaf.clone(),
                    "".into(),
                    NodeType::Module,
                );
                g.add_edge(leaf.into(), mid.clone().into(), 1.0, EdgeType::DependsOn);
            }
        }
        assert!(
            g.node_in_degree(&"mid0".into()) > g.node_in_degree(&"top".into()),
            "in-degree ranks a mid (5) above top (3)"
        );
        let ec = g.eigencentrality();
        let top = ec.get(&"top".into()).copied().unwrap();
        let mid = ec.get(&"mid0".into()).copied().unwrap();
        assert!(
            top > mid,
            "eigenvector ranks top above a mid: top={top} mid={mid}"
        );

        // And the mode actually drives the score: with the eigenvector mode the
        // recursively-central node scores highest; with in-degree the raw-count mid does.
        let score = |g: &MemoryGraph, id: &str| g.compute_node_score(g.node(&id.into()).unwrap());
        g.set_scoring_weights(ScoringWeights {
            w_centrality: 0.5,
            centrality_mode: CentralityMode::Eigenvector,
            ..Default::default()
        });
        assert!(
            score(&g, "top") > score(&g, "mid0"),
            "eigenvector mode lifts top"
        );
        g.set_scoring_weights(ScoringWeights {
            w_centrality: 0.5,
            centrality_mode: CentralityMode::InDegree,
            ..Default::default()
        });
        assert!(
            score(&g, "mid0") >= score(&g, "top"),
            "in-degree mode favors the raw count"
        );
    }

    /// The default mode is elided from the snapshot (byte-identical) and round-trips when set.
    #[test]
    fn centrality_mode_serde_elides_default_and_round_trips() {
        let w = ScoringWeights::default();
        assert_eq!(w.centrality_mode, CentralityMode::InDegree);
        let json = serde_json::to_string(&w).unwrap();
        assert!(
            !json.contains("centrality_mode"),
            "default mode is elided: {json}"
        );
        let eigen = ScoringWeights {
            centrality_mode: CentralityMode::Eigenvector,
            ..Default::default()
        };
        let back: ScoringWeights =
            serde_json::from_str(&serde_json::to_string(&eigen).unwrap()).unwrap();
        assert_eq!(back.centrality_mode, CentralityMode::Eigenvector);
    }

    #[test]
    fn node_state_default_is_elided_and_round_trips() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        g.upsert_node("a".into(), "a".into(), "x".into(), NodeType::Module);
        assert_eq!(g.node(&"a".into()).unwrap().state, NodeState::Stable);
        // Default Stable ⇒ no `state` key ⇒ snapshots stay byte-identical.
        assert!(!serde_json::to_string(&g).unwrap().contains("\"state\""));
        g.set_node_state(&"a".into(), NodeState::Working);
        let json = serde_json::to_string(&g).unwrap();
        assert!(json.contains("Working"));
        let back: MemoryGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.node(&"a".into()).unwrap().state, NodeState::Working);
    }

    #[test]
    fn orphan_is_excluded_from_centrality_and_set_state_invalidates_cache() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["hub", "x", "y", "dead"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        for s in ["x", "y", "dead"] {
            g.add_edge(s.into(), "hub".into(), 1.0, EdgeType::DependsOn);
        }
        assert_eq!(g.node_in_degree(&"hub".into()), 3); // primes the cache
                                                        // Marking `dead` Orphan must drop it from the structural signal — and the cache,
                                                        // keyed only on edge count, must be invalidated by set_node_state.
        g.set_node_state(&"dead".into(), NodeState::Orphan);
        assert_eq!(
            g.node_in_degree(&"hub".into()),
            2,
            "an orphan dependent no longer inflates the hub's in-degree"
        );
        assert!(
            !g.eigencentrality().contains_key(&"dead".into()),
            "an orphan has no eigenvector centrality"
        );
    }

    #[test]
    fn orphan_evicted_first_and_working_pinned_regardless_of_recency() {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["dead", "work", "stable"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        let score = |g: &MemoryGraph, id: &str| g.compute_node_score(g.node(&id.into()).unwrap());
        // `dead` is freshly upserted (recency 1.0) yet, once Orphan, scores below a Stable peer.
        g.set_node_state(&"dead".into(), NodeState::Orphan);
        assert!(
            score(&g, "dead") < score(&g, "stable"),
            "orphan scores below a stable peer even at full recency → evicted first"
        );
        // `work` is Working; after decay it still outscores a decayed Stable peer (pinned).
        g.set_node_state(&"work".into(), NodeState::Working);
        for _ in 0..30 {
            g.tick();
        }
        assert!(
            score(&g, "work") > score(&g, "stable"),
            "Working is pinned above a decayed Stable peer"
        );
    }

    /// Build a graph of `(file, symbol)` defs (with `file:` + `sym:` nodes), for the call tests.
    fn graph_with_defs(defs: &[(&str, &str)]) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for (file, name) in defs {
            let fid = NodeId(format!("file:{file}"));
            g.upsert_node(fid.clone(), (*file).into(), "".into(), NodeType::Module);
            let sid = NodeId(format!("sym:{file}:{name}"));
            g.upsert_node(sid.clone(), (*name).into(), "".into(), NodeType::Symbol);
            g.add_edge(fid, sid, 0.6, EdgeType::Contains);
        }
        g
    }
    fn calls_of(g: &MemoryGraph) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::Calls)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        v.sort();
        v
    }

    fn data_flow_of(g: &MemoryGraph) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::DataFlow)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        v.sort();
        v
    }

    // ── Slice 3: (type, method) resolution of an inferred `Foo::bar` callee ───────
    #[test]
    fn resolve_type_method_links_cross_file() {
        // `Foo::bar` (minted by `x.bar()` inference) resolves to the method symbol via the
        // (type, method) index — `bar` lives in a different file than the caller and `Foo` is no module.
        let mut g = graph_with_defs(&[("src/a.rs", "bar"), ("src/b.rs", "run")]);
        g.set_pending_methods("src/a.rs", vec![("Foo".into(), "bar".into())]);
        g.set_pending_calls("src/b.rs", vec![("run".into(), "Foo::bar".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/b.rs:run".to_string(),
                "sym:src/a.rs:bar".to_string()
            )],
            "Foo::bar resolves to the method symbol via the type index"
        );
    }

    #[test]
    fn resolve_type_method_homonym_is_ambiguous_skip() {
        // Two types both named `Foo`, each with a `bar` method: the (Foo,bar) bucket is non-unique, so
        // the type interpretation abstains and the call is SKIPPED — never last-writer-wins (the prime
        // false-edge hazard the cardinality guard closes).
        let mut g = graph_with_defs(&[
            ("src/a.rs", "bar"),
            ("src/b.rs", "bar"),
            ("src/c.rs", "run"),
        ]);
        g.set_pending_methods("src/a.rs", vec![("Foo".into(), "bar".into())]);
        g.set_pending_methods("src/b.rs", vec![("Foo".into(), "bar".into())]);
        g.set_pending_calls("src/c.rs", vec![("run".into(), "Foo::bar".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "a type-name homonym makes Foo::bar ambiguous — skip, never guess: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_type_method_paged_out_homonym_still_skips() {
        // A genuine (Foo, bar) homonym where ONE method symbol is absent from the resident set (paged
        // COLD): cardinality is counted over ALL ingested method defs, not just resident ones, so the
        // homonym is still detected → skip. A resident-only count would under-count to 1 and mint a
        // false edge to the surviving resident twin. (Adversarial-review regression.)
        let mut g = graph_with_defs(&[("src/a.rs", "bar"), ("src/c.rs", "run")]);
        // src/b.rs:bar is a real second definition, but its symbol node is NOT resident — we register
        // the method-ownership fact without creating the sym node (simulating a paged-COLD twin).
        g.set_pending_methods("src/a.rs", vec![("Foo".into(), "bar".into())]);
        g.set_pending_methods("src/b.rs", vec![("Foo".into(), "bar".into())]);
        g.set_pending_calls("src/c.rs", vec![("run".into(), "Foo::bar".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "a paged-out homonym still makes Foo::bar ambiguous — no false edge to the resident twin: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_type_method_unknown_type_skips_no_name_fallback() {
        // `Foo::bar` where (Foo, bar) is NOT in the index resolves as NEITHER module nor type — it must
        // skip, never fall back to a same-named free `bar` (the type branch is not a name fallback).
        let mut g = graph_with_defs(&[("src/a.rs", "bar"), ("src/b.rs", "run")]);
        // No set_pending_methods ⇒ (Foo, bar) is absent from the index.
        g.set_pending_calls("src/b.rs", vec![("run".into(), "Foo::bar".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "unknown type Foo::bar skips, no fallback to the free bar: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_data_flow_links_reader_to_global_unique_static() {
        // `reader()` references `CONFIG`, a uniquely-named const in config.rs → a `DataFlow` edge,
        // the shared-state channel call/import edges miss.
        let mut g = graph_with_defs(&[("src/config.rs", "CONFIG"), ("src/api.rs", "reader")]);
        g.mark_data_symbol(NodeId("sym:src/config.rs:CONFIG".into()));
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/config.rs:CONFIG".to_string()
            )],
            "reader → CONFIG data-flow edge"
        );
    }

    #[test]
    fn resolve_data_flow_skips_ambiguous_global_name() {
        // `CONFIG` defined in two files (ambiguous) → the reference skips (resolve-uniquely-or-skip).
        let mut g = graph_with_defs(&[
            ("src/a.rs", "CONFIG"),
            ("src/b.rs", "CONFIG"),
            ("src/api.rs", "reader"),
        ]);
        g.mark_data_symbol(NodeId("sym:src/a.rs:CONFIG".into()));
        g.mark_data_symbol(NodeId("sym:src/b.rs:CONFIG".into()));
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "an ambiguous global static name skips — no guessed edge"
        );
    }

    #[test]
    fn resolve_data_flow_skips_reference_to_non_data_symbol() {
        // `HELPER` exists as a symbol but is NOT a static/const (it was never marked) — e.g. a
        // SCREAMING-named function. A reference to it must not become a data-flow edge.
        let mut g = graph_with_defs(&[("src/x.rs", "HELPER"), ("src/api.rs", "reader")]);
        // deliberately do NOT mark HELPER as a data symbol
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "HELPER".into(), 1)]);
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "a reference resolving to a non-static/const symbol is not a data-flow edge"
        );
    }

    /// Add an `m::name`-style import to a file (a `use:<file>:<usepath>` node + its DependsOn edge),
    /// so a qualified data-ref `alias::X` can be expanded through it — mirrors the call tests' setup.
    fn add_import(g: &mut MemoryGraph, file: &str, usepath: &str) {
        let uid = NodeId(format!("use:{file}:{usepath}"));
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId(format!("file:{file}")),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
    }

    #[test]
    fn resolve_data_flow_qualified_crate_rooted_links_to_module_const() {
        // `crate::cfg::MAX_RETRIES` resolves absolutely via its module prefix (no import needed) to
        // cfg.rs's `MAX_RETRIES`, marked as a data symbol — exactly one DataFlow edge (Slice 2).
        let mut g = graph_with_defs(&[("src/cfg.rs", "MAX_RETRIES"), ("src/api.rs", "reader")]);
        g.mark_data_symbol(NodeId("sym:src/cfg.rs:MAX_RETRIES".into()));
        g.set_pending_data_refs(
            "src/api.rs",
            vec![("reader".into(), "crate::cfg::MAX_RETRIES".into(), 1)],
        );
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/cfg.rs:MAX_RETRIES".to_string()
            )],
            "crate-rooted qualified data-ref resolves to the module's const (Slice 2)"
        );
    }

    #[test]
    fn resolve_data_flow_qualified_via_import_alias_links() {
        // `use crate::cfg;` then `cfg::MAX` — the leading segment is expanded via the import, the
        // same machinery qualified CALLS use. The const must be marked to link.
        let mut g = graph_with_defs(&[("src/cfg.rs", "MAX"), ("src/api.rs", "reader")]);
        g.mark_data_symbol(NodeId("sym:src/cfg.rs:MAX".into()));
        add_import(&mut g, "src/api.rs", "crate::cfg");
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "cfg::MAX".into(), 1)]);
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/cfg.rs:MAX".to_string()
            )],
            "module-alias qualified data-ref resolves by expanding the import (Slice 2)"
        );
    }

    #[test]
    fn resolve_data_flow_qualified_to_non_data_symbol_skips() {
        // `crate::cfg::MAX` resolves to a real symbol in cfg.rs, but it was NEVER marked as a
        // static/const (e.g. a SCREAMING-named fn). A qualified ref must resolve only to a data
        // symbol — so no edge, no false DataFlow to a function.
        let mut g = graph_with_defs(&[("src/cfg.rs", "MAX"), ("src/api.rs", "reader")]);
        // deliberately do NOT mark MAX
        g.set_pending_data_refs(
            "src/api.rs",
            vec![("reader".into(), "crate::cfg::MAX".into(), 1)],
        );
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "a qualified ref to a non-static/const symbol is not a data-flow edge"
        );
    }

    #[test]
    fn resolve_data_flow_qualified_unresolvable_skips_and_no_fallback() {
        // Three unresolvable qualified refs, each a distinct skip case — NONE may emit, and in
        // particular none may fall back to the bare-name global index (which WOULD match a unique
        // `MAX`/`LIMIT`):
        //   * `other::MAX`  — alias `other` matches no import → skip (no fallback to the global MAX);
        //   * `crate::ghost::MAX` — module `ghost` has no file → skip (no ancestor/global fallback);
        //   * `std::cfg::LIMIT` — external root, no import expands it → skip.
        let mut g = graph_with_defs(&[
            ("src/cfg.rs", "MAX"),
            ("src/cfg.rs", "LIMIT"),
            ("src/api.rs", "reader"),
        ]);
        g.mark_data_symbol(NodeId("sym:src/cfg.rs:MAX".into()));
        g.mark_data_symbol(NodeId("sym:src/cfg.rs:LIMIT".into()));
        g.set_pending_data_refs(
            "src/api.rs",
            vec![
                ("reader".into(), "other::MAX".into(), 1),
                ("reader".into(), "crate::ghost::MAX".into(), 2),
                ("reader".into(), "std::cfg::LIMIT".into(), 3),
            ],
        );
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "unresolvable qualified data-refs skip — never falling back to the bare global index"
        );
    }

    #[test]
    fn resolve_data_flow_bare_ref_resolves_via_import_scope_despite_global_ambiguity() {
        // `CONFIG` is defined in TWO files (globally ambiguous ⇒ the old bare/Tier-C rule skips), but
        // `src/api.rs` imports `use crate::other::CONFIG;`. The import pins WHICH `CONFIG` the bare
        // reference means → a DataFlow edge to `other`'s const, never the decoy (data-flow Tier A).
        let mut g = graph_with_defs(&[
            ("src/other.rs", "CONFIG"),
            ("src/decoy.rs", "CONFIG"),
            ("src/api.rs", "reader"),
        ]);
        g.mark_data_symbol(NodeId("sym:src/other.rs:CONFIG".into()));
        g.mark_data_symbol(NodeId("sym:src/decoy.rs:CONFIG".into()));
        add_import(&mut g, "src/api.rs", "crate::other::CONFIG");
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/other.rs:CONFIG".to_string()
            )],
            "an imported bare global resolves via the import scope, not the ambiguous global index"
        );
    }

    #[test]
    fn resolve_data_flow_bare_ref_resolves_in_readers_own_module_despite_global_ambiguity() {
        // `CONFIG` in the reader's OWN file plus a decoy `CONFIG` elsewhere (globally ambiguous). A
        // bare reference resolves to the same-module const (data-flow Tier B), as Rust name
        // resolution would — never guessing the decoy.
        let mut g = graph_with_defs(&[
            ("src/api.rs", "CONFIG"),
            ("src/api.rs", "reader"),
            ("src/decoy.rs", "CONFIG"),
        ]);
        g.mark_data_symbol(NodeId("sym:src/api.rs:CONFIG".into()));
        g.mark_data_symbol(NodeId("sym:src/decoy.rs:CONFIG".into()));
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/api.rs:CONFIG".to_string()
            )],
            "a bare global defined in the reader's own module resolves same-module (Tier B)"
        );
    }

    #[test]
    fn resolve_data_flow_bare_ref_ambiguous_imports_skip() {
        // Two imports bring in `CONFIG` from two different modules (only possible in non-compiling
        // Rust — duplicate imports). The import scope itself is ambiguous → skip, never fall through
        // to a same-module/global guess the imports already contradict.
        let mut g = graph_with_defs(&[
            ("src/m1.rs", "CONFIG"),
            ("src/m2.rs", "CONFIG"),
            ("src/api.rs", "reader"),
        ]);
        g.mark_data_symbol(NodeId("sym:src/m1.rs:CONFIG".into()));
        g.mark_data_symbol(NodeId("sym:src/m2.rs:CONFIG".into()));
        add_import(&mut g, "src/api.rs", "crate::m1::CONFIG");
        add_import(&mut g, "src/api.rs", "crate::m2::CONFIG");
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "CONFIG".into(), 1)]);
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "ambiguous imports for a bare global skip — never a guessed edge: {:?}",
            data_flow_of(&g)
        );
    }

    #[test]
    fn resolve_data_flow_bare_ref_resolves_through_renamed_import() {
        // `use crate::cfg::MAX as LIMIT; … LIMIT` — the renamed-import alias binds local `LIMIT` to
        // `crate::cfg::MAX`; the bare data ref is rewritten onto the target and resolves to cfg's
        // const (mirrors the call resolver's alias handling).
        let mut g = graph_with_defs(&[("src/cfg.rs", "MAX"), ("src/api.rs", "reader")]);
        g.mark_data_symbol(NodeId("sym:src/cfg.rs:MAX".into()));
        g.set_pending_aliases(
            "src/api.rs",
            vec![("LIMIT".into(), "crate::cfg::MAX".into())],
        );
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "LIMIT".into(), 1)]);
        g.resolve_data_flow();
        assert_eq!(
            data_flow_of(&g),
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/cfg.rs:MAX".to_string()
            )],
            "a renamed const import (use MAX as LIMIT) resolves the aliased ref to the real const"
        );
    }

    #[test]
    fn resolve_data_flow_renamed_import_to_non_data_symbol_skips() {
        // The alias target resolves to a real symbol, but it was never marked a static/const (e.g. a
        // SCREAMING-named fn). A data ref must resolve only to a data symbol ⇒ no edge, no false link.
        let mut g = graph_with_defs(&[("src/cfg.rs", "MAX"), ("src/api.rs", "reader")]);
        // deliberately do NOT mark MAX as a data symbol
        g.set_pending_aliases(
            "src/api.rs",
            vec![("LIMIT".into(), "crate::cfg::MAX".into())],
        );
        g.set_pending_data_refs("src/api.rs", vec![("reader".into(), "LIMIT".into(), 1)]);
        g.resolve_data_flow();
        assert!(
            data_flow_of(&g).is_empty(),
            "an aliased ref to a non-static/const target is not a data-flow edge: {:?}",
            data_flow_of(&g)
        );
    }

    #[test]
    fn resolve_data_flow_qualified_is_deterministic() {
        // Same source → identical edge set across two independent builds (qualified + bare mixed).
        let build = || {
            let mut g = graph_with_defs(&[
                ("src/cfg.rs", "MAX_RETRIES"),
                ("src/lim.rs", "CEILING"),
                ("src/api.rs", "reader"),
            ]);
            g.mark_data_symbol(NodeId("sym:src/cfg.rs:MAX_RETRIES".into()));
            g.mark_data_symbol(NodeId("sym:src/lim.rs:CEILING".into()));
            add_import(&mut g, "src/api.rs", "crate::cfg");
            g.set_pending_data_refs(
                "src/api.rs",
                vec![
                    ("reader".into(), "cfg::MAX_RETRIES".into(), 1),
                    ("reader".into(), "crate::lim::CEILING".into(), 2),
                ],
            );
            g.resolve_data_flow();
            data_flow_of(&g)
        };
        let a = build();
        let b = build();
        assert_eq!(a, b, "same source yields an identical DataFlow edge set");
        assert_eq!(
            a,
            vec![
                (
                    "sym:src/api.rs:reader".to_string(),
                    "sym:src/cfg.rs:MAX_RETRIES".to_string()
                ),
                (
                    "sym:src/api.rs:reader".to_string(),
                    "sym:src/lim.rs:CEILING".to_string()
                ),
            ],
            "both qualified refs resolve to their module consts"
        );
    }

    #[test]
    fn dead_symbols_flags_only_unreferenced() {
        // api.rs: `run` is called by nothing; `helper` is called by `run`. Only `run` is a
        // candidate — `helper` has an incoming Calls edge, and the file's Contains edges (which
        // every symbol has) do not count as references.
        let mut g = graph_with_defs(&[("src/api.rs", "run"), ("src/api.rs", "helper")]);
        g.add_edge(
            NodeId("sym:src/api.rs:run".into()),
            NodeId("sym:src/api.rs:helper".into()),
            0.75,
            EdgeType::Calls,
        );
        assert_eq!(
            g.dead_symbols(),
            vec![NodeId("sym:src/api.rs:run".into())],
            "only the unreferenced symbol is flagged"
        );
    }

    #[test]
    fn add_edge_dedup_index_stays_correct_across_remove_and_serde() {
        // The O(1) `edge_set` dedup index must mirror `edges` through every path: a duplicate is
        // rejected, a different edge type between the same pair is *not* a duplicate, a removed edge
        // becomes re-addable (the index rebuilds after the bulk remove), and a serde reload (which
        // drops the runtime index) still dedups (it rebuilds lazily).
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for id in ["a", "b"] {
            g.upsert_node(id.into(), id.into(), String::new(), NodeType::ContextBlock);
        }
        assert!(g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn));
        assert!(
            !g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn),
            "exact duplicate rejected"
        );
        assert!(
            g.add_edge("a".into(), "b".into(), 1.0, EdgeType::References),
            "a different edge type between the same pair is not a duplicate"
        );
        assert_eq!(g.edges().len(), 2);

        // Remove an endpoint → its edges go; the (now length-stale) index must let the edge re-add.
        g.remove_node(&"b".into());
        g.upsert_node(
            "b".into(),
            "b".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        assert!(
            g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn),
            "edge is re-addable after its endpoint was removed"
        );

        // Serde reload drops the `#[serde(skip)]` index; dedup must still hold (lazy rebuild).
        let json = serde_json::to_string(&g).unwrap();
        let mut back: MemoryGraph = serde_json::from_str(&json).unwrap();
        let before = back.edges().len();
        assert!(
            !back.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn),
            "dedup survives a reload"
        );
        assert_eq!(
            back.edges().len(),
            before,
            "no duplicate added after reload"
        );
    }

    // ── Q-Page dual-evidence belief layer ───────────────────────────────────────────────────────

    /// A `claim` node with `n_support` + `n_contra` evidence nodes, each linked to the claim at
    /// weight 1.0 with the matching polarity. Pure graph construction — no parser.
    fn graph_with_evidence(n_support: usize, n_contra: usize) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let claim: NodeId = "claim".into();
        g.upsert_node(
            claim.clone(),
            "claim".into(),
            "claim".into(),
            NodeType::ContextBlock,
        );
        for i in 0..n_support {
            let id = NodeId(format!("sup{i}"));
            g.upsert_node(
                id.clone(),
                "s".into(),
                String::new(),
                NodeType::ContextBlock,
            );
            g.add_edge(id, claim.clone(), 1.0, EdgeType::Supports);
        }
        for i in 0..n_contra {
            let id = NodeId(format!("con{i}"));
            g.upsert_node(
                id.clone(),
                "c".into(),
                String::new(),
                NodeType::ContextBlock,
            );
            g.add_edge(id, claim.clone(), 1.0, EdgeType::Contradicts);
        }
        g
    }

    #[test]
    fn qbelief_neutral_prior_without_evidence() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let claim: NodeId = "claim".into();
        g.upsert_node(
            claim.clone(),
            "c".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        let q = g.qbelief(&claim);
        assert_eq!(q.support, 0.0);
        assert_eq!(q.contradiction, 0.0);
        assert_eq!(q.belief, 0.0, "no evidence ⇒ neutral 0");
        assert_eq!(q.conflict, 0.0, "no evidence ⇒ no conflict");
    }

    #[test]
    fn qbelief_support_and_contradiction_math() {
        // 3 support, 1 contradiction at weight 1.0: belief = (3-1)/(3+1+1) = 0.4; conflict =
        // 2·sqrt(3)/5 ≈ 0.6928.
        let q = graph_with_evidence(3, 1).qbelief(&"claim".into());
        assert_eq!(q.support, 3.0);
        assert_eq!(q.contradiction, 1.0);
        assert!((q.belief - 0.4).abs() < 1e-12);
        assert!((q.conflict - 2.0 * 3.0_f64.sqrt() / 5.0).abs() < 1e-12);
    }

    #[test]
    fn claim_beliefs_ranks_contested_claims_first() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for id in ["evA", "evB1", "evB2", "consensus", "contested"] {
            g.upsert_node(id.into(), id.into(), String::new(), NodeType::ContextBlock);
        }
        // "consensus": one-sided support (conflict 0). "contested": balanced (high conflict).
        g.add_edge("evA".into(), "consensus".into(), 1.0, EdgeType::Supports);
        g.add_edge("evB1".into(), "contested".into(), 1.0, EdgeType::Supports);
        g.add_edge(
            "evB2".into(),
            "contested".into(),
            1.0,
            EdgeType::Contradicts,
        );

        let beliefs = g.claim_beliefs();
        assert_eq!(beliefs.len(), 2, "only the two claims carry assertions");
        assert_eq!(
            beliefs[0].0,
            NodeId::from("contested"),
            "contested claim ranks first (higher conflict)"
        );
        assert!(beliefs[0].1.conflict > beliefs[1].1.conflict);
        assert_eq!(beliefs[1].0, NodeId::from("consensus"));
        assert_eq!(
            beliefs[1].1.conflict, 0.0,
            "one-sided evidence ⇒ zero conflict"
        );
    }

    #[test]
    fn render_tension_bar_grows_with_conflict() {
        let calm = render_tension_bar(&belief_from_sums(1.0, 0.0)); // conflict 0
        let tense = render_tension_bar(&belief_from_sums(3.0, 3.0)); // high conflict
        assert!(
            tense.matches('█').count() > calm.matches('█').count(),
            "more conflict ⇒ more filled cells: {calm:?} vs {tense:?}"
        );
        assert!(calm.starts_with('+'), "signed-belief prefix: {calm:?}");
    }

    #[test]
    fn qbelief_conflict_zero_when_one_sided_and_high_when_balanced() {
        // All-support: no conflict, strong positive belief (4/(4+1) = 0.8).
        let one_sided = graph_with_evidence(4, 0).qbelief(&"claim".into());
        assert_eq!(one_sided.conflict, 0.0);
        assert!(
            (one_sided.belief - 0.8).abs() < 1e-12,
            "4 support ⇒ belief 0.8"
        );
        // Evenly matched: belief nets to 0; conflict is high — it → 1 only as evidence accumulates,
        // so at 3-vs-3 with the unit prior it is 2·3/(6+1) = 6/7.
        let balanced = graph_with_evidence(3, 3).qbelief(&"claim".into());
        assert!(balanced.belief.abs() < 1e-12, "balanced ⇒ belief 0");
        assert!(
            (balanced.conflict - 6.0 / 7.0).abs() < 1e-12,
            "balanced ⇒ high conflict 6/7"
        );
    }

    #[test]
    fn evidence_of_returns_sorted_deduped_polarity_surfaces() {
        let g = graph_with_evidence(2, 1);
        let claim: NodeId = "claim".into();
        assert_eq!(
            g.evidence_of(&claim, EdgeType::Supports),
            vec![&NodeId("sup0".into()), &NodeId("sup1".into())]
        );
        assert_eq!(
            g.evidence_of(&claim, EdgeType::Contradicts),
            vec![&NodeId("con0".into())]
        );
    }

    #[test]
    fn qbelief_is_deterministic_across_rebuilds() {
        let a = graph_with_evidence(5, 2).qbelief(&"claim".into());
        let b = graph_with_evidence(5, 2).qbelief(&"claim".into());
        assert_eq!(a, b, "same evidence ⇒ identical QBelief");
    }

    #[test]
    fn supports_contradicts_edges_survive_serde_round_trip() {
        // The two appended EdgeType variants serialize by name and deserialize losslessly — a graph
        // carrying them round-trips, so the belief derived after a snapshot reload is unchanged
        // (additive enum: old snapshots simply never contain these variants).
        let g = graph_with_evidence(2, 2);
        let json = serde_json::to_string(&g).unwrap();
        assert!(json.contains("Supports") && json.contains("Contradicts"));
        let back: MemoryGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.edges().len(), g.edges().len());
        assert_eq!(
            back.qbelief(&"claim".into()),
            g.qbelief(&"claim".into()),
            "belief is identical after a serde round-trip"
        );
    }

    // ── Q-Page decay (knowledge half-life) ──────────────────────────────────────────────────────

    #[test]
    fn qbelief_decayed_equals_qbelief_with_no_elapsed_time() {
        // All evidence asserted at the current clock (age 0, factor 1.0) ⇒ decay is a no-op.
        let g = graph_with_evidence(2, 1);
        let claim: NodeId = "claim".into();
        assert_eq!(g.qbelief_decayed(&claim, 10.0), g.qbelief(&claim));
    }

    #[test]
    fn qbelief_decayed_nonpositive_half_life_is_no_decay() {
        let mut g = graph_with_evidence(2, 1);
        for _ in 0..50 {
            g.tick();
        }
        let claim: NodeId = "claim".into();
        assert_eq!(g.qbelief_decayed(&claim, 0.0), g.qbelief(&claim));
    }

    #[test]
    fn qbelief_decayed_relaxes_belief_toward_prior_as_evidence_ages() {
        // A strongly-supported claim, then a long silence: the decayed belief relaxes from 0.75 back
        // toward the neutral 0 (the support mass faded), while plain qbelief stays frozen.
        let mut g = graph_with_evidence(3, 0);
        for _ in 0..100 {
            g.tick();
        }
        let claim: NodeId = "claim".into();
        let plain = g.qbelief(&claim);
        let decayed = g.qbelief_decayed(&claim, 10.0);
        assert!(
            (plain.belief - 0.75).abs() < 1e-9,
            "plain belief frozen at 0.75"
        );
        assert!(
            decayed.belief < 0.05 && decayed.belief >= 0.0,
            "decayed belief relaxes toward the neutral 0, got {}",
            decayed.belief
        );
        assert_eq!(decayed.conflict, 0.0, "one-sided ⇒ no conflict either way");
    }

    #[test]
    fn qbelief_decayed_fresh_evidence_resolves_stale_dissent() {
        // A claim contested long ago (one contradiction at t=0), then a FRESH support arrives. Plain
        // qbelief still sees it as fully contested forever; the decayed view lets the fresh evidence
        // win — belief rises and conflict collapses, because the stale dissent has decayed.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let claim: NodeId = "claim".into();
        for id in ["claim", "old_con", "fresh_sup"] {
            g.upsert_node(
                NodeId(id.into()),
                id.into(),
                String::new(),
                NodeType::ContextBlock,
            );
        }
        g.add_edge("old_con".into(), claim.clone(), 1.0, EdgeType::Contradicts); // t = 0
        for _ in 0..40 {
            g.tick();
        }
        g.add_edge("fresh_sup".into(), claim.clone(), 1.0, EdgeType::Supports); // t = 40

        let plain = g.qbelief(&claim);
        let decayed = g.qbelief_decayed(&claim, 10.0); // dissent aged 4 half-lives ⇒ ×0.0625
        assert!(
            (plain.conflict - 2.0 / 3.0).abs() < 1e-9,
            "plain view: still contested (1 vs 1 ⇒ conflict 2/3)"
        );
        assert!(plain.belief.abs() < 1e-9, "plain belief nets to 0");
        assert!(
            decayed.conflict < 0.3,
            "decay lets fresh support resolve the stale dissent, conflict {} (was 2/3)",
            decayed.conflict
        );
        assert!(
            decayed.belief > 0.3,
            "decayed belief rises well above the contested 0, got {}",
            decayed.belief
        );
    }

    #[test]
    fn qbelief_decayed_is_deterministic() {
        let build = || {
            let mut g = MemoryGraph::new(0.0, usize::MAX);
            let claim: NodeId = "claim".into();
            for id in ["claim", "c", "s"] {
                g.upsert_node(
                    NodeId(id.into()),
                    id.into(),
                    String::new(),
                    NodeType::ContextBlock,
                );
            }
            g.add_edge("c".into(), claim.clone(), 1.0, EdgeType::Contradicts);
            for _ in 0..30 {
                g.tick();
            }
            g.add_edge("s".into(), claim.clone(), 1.0, EdgeType::Supports);
            g.qbelief_decayed(&claim, 7.0)
        };
        assert_eq!(
            build(),
            build(),
            "same timed construction ⇒ identical decayed belief"
        );
    }

    // ── Q-Page belief propagation (single hop) ──────────────────────────────────────────────────

    /// Give `claim` `n_sup` support + `n_con` contradiction evidence edges (fresh nodes, weight 1.0).
    fn add_evidence(g: &mut MemoryGraph, claim: &str, n_sup: usize, n_con: usize) {
        let c = NodeId(claim.into());
        g.upsert_node(
            c.clone(),
            claim.into(),
            String::new(),
            NodeType::ContextBlock,
        );
        for i in 0..n_sup {
            let id = NodeId(format!("{claim}_s{i}"));
            g.upsert_node(
                id.clone(),
                "s".into(),
                String::new(),
                NodeType::ContextBlock,
            );
            g.add_edge(id, c.clone(), 1.0, EdgeType::Supports);
        }
        for i in 0..n_con {
            let id = NodeId(format!("{claim}_c{i}"));
            g.upsert_node(
                id.clone(),
                "c".into(),
                String::new(),
                NodeType::ContextBlock,
            );
            g.add_edge(id, c.clone(), 1.0, EdgeType::Contradicts);
        }
    }

    #[test]
    fn propagate_resolved_cause_supports_effect_with_no_evidence() {
        // A is resolved-true (3 supports ⇒ belief 0.75); A causes B; B has no direct evidence (0).
        // Propagation gives B a derived, attenuated support, so its belief rises — but stays weaker
        // than its cause's.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        add_evidence(&mut g, "A", 3, 0);
        g.upsert_node(
            "B".into(),
            "B".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
        let (a, b): (NodeId, NodeId) = ("A".into(), "B".into());
        assert_eq!(g.qbelief(&b).belief, 0.0, "B starts unknown");
        assert_eq!(g.propagate_beliefs(0.7, 0.6), 1);
        let a_belief = g.qbelief(&a).belief;
        let b_belief = g.qbelief(&b).belief;
        assert!((a_belief - 0.75).abs() < 1e-9, "cause unchanged at 0.75");
        assert!(
            b_belief > 0.0 && b_belief < a_belief,
            "effect inherits a weaker belief from its resolved cause: {b_belief}"
        );
    }

    #[test]
    fn propagate_disbelieved_cause_contradicts_effect() {
        // A resolved-false (3 contradictions ⇒ belief −0.75) ⇒ a derived Contradicts on its effect.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        add_evidence(&mut g, "A", 0, 3);
        g.upsert_node(
            "B".into(),
            "B".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
        assert_eq!(g.propagate_beliefs(0.7, 0.6), 1);
        assert!(
            g.qbelief(&"B".into()).belief < 0.0,
            "a disbelieved cause pushes its effect below the neutral 0"
        );
    }

    #[test]
    fn propagate_unresolved_cause_does_not_propagate() {
        // A balanced (2 support + 2 contradiction ⇒ belief 0) is not resolved ⇒ no propagation.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        add_evidence(&mut g, "A", 2, 2);
        g.upsert_node(
            "B".into(),
            "B".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
        assert_eq!(g.propagate_beliefs(0.7, 0.6), 0);
        assert_eq!(g.qbelief(&"B".into()).belief, 0.0, "B stays unknown");
    }

    #[test]
    fn propagate_skips_self_causes() {
        // A self-cause (A causes A) must never derive self-evidence.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        add_evidence(&mut g, "A", 3, 0);
        g.add_edge("A".into(), "A".into(), 1.0, EdgeType::Causes);
        let before = g.qbelief(&"A".into());
        assert_eq!(g.propagate_beliefs(0.7, 0.6), 0);
        assert_eq!(
            g.qbelief(&"A".into()),
            before,
            "self-cause adds no evidence"
        );
    }

    #[test]
    fn propagate_is_idempotent_and_deterministic() {
        let build = || {
            let mut g = MemoryGraph::new(0.0, usize::MAX);
            add_evidence(&mut g, "A", 3, 0);
            g.upsert_node(
                "B".into(),
                "B".into(),
                String::new(),
                NodeType::ContextBlock,
            );
            g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
            let first = g.propagate_beliefs(0.7, 0.6);
            let second = g.propagate_beliefs(0.7, 0.6); // already-derived edge is deduped
            (first, second, g.qbelief(&"B".into()))
        };
        let (first, second, belief) = build();
        assert_eq!(
            (first, second),
            (1, 0),
            "idempotent: re-propagation adds nothing"
        );
        let (_, _, belief2) = build();
        assert_eq!(belief, belief2, "deterministic across rebuilds");
    }

    /// The multi-hop fixture: `A → B → C` (`Causes`), `A` resolved-true (belief +0.75), `B` and `C`
    /// carrying no direct evidence of their own.
    fn belief_chain() -> MemoryGraph {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        add_evidence(&mut g, "A", 3, 0); // belief +0.75
        for c in ["B", "C"] {
            g.upsert_node(c.into(), c.into(), String::new(), NodeType::ContextBlock);
        }
        g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
        g.add_edge("B".into(), "C".into(), 1.0, EdgeType::Causes);
        g
    }

    #[test]
    fn converge_resolves_a_chain_where_one_hop_stops() {
        // Low threshold + high damping keep the induced belief above the resolve bar for two hops,
        // so convergence resolves BOTH B and C — one derived edge per hop.
        let mut g = belief_chain();
        let total = g.propagate_beliefs_converge(0.2, 0.9, 8);
        assert_eq!(total, 2, "one derived edge per hop: A→B then B→C");
        assert!(g.qbelief(&"B".into()).belief > 0.0, "B inherits from A");
        assert!(
            g.qbelief(&"C".into()).belief > 0.0,
            "C inherits belief two hops from A under convergence"
        );

        // Contrast: a single sweep resolves only the direct effect B; C is never reached.
        let mut one = belief_chain();
        assert_eq!(one.propagate_beliefs(0.2, 0.9), 1);
        assert_eq!(
            one.qbelief(&"C".into()).belief,
            0.0,
            "single hop stops at B; C stays neutral"
        );
    }

    #[test]
    fn converge_reaches_a_fixpoint_and_is_idempotent() {
        let mut g = belief_chain();
        let first = g.propagate_beliefs_converge(0.2, 0.9, 8);
        assert!(first > 0, "the first convergence derives edges");
        let second = g.propagate_beliefs_converge(0.2, 0.9, 8);
        assert_eq!(
            second, 0,
            "at the fixpoint, re-converging derives nothing new"
        );
    }

    #[test]
    fn converge_halts_when_belief_attenuates_below_threshold() {
        // A high threshold: B's inherited belief (~0.31) never clears 0.7, so the wavefront stops at
        // B exactly as a single hop would — attenuation bounds the cascade without the round cap.
        let mut g = belief_chain();
        let total = g.propagate_beliefs_converge(0.7, 0.6, 8);
        assert_eq!(
            total, 1,
            "only A→B; B never resolves, so C is never reached"
        );
        assert!(g.qbelief(&"B".into()).belief > 0.0);
        assert_eq!(
            g.qbelief(&"C".into()).belief,
            0.0,
            "the wavefront halts below threshold"
        );
    }

    #[test]
    fn converge_terminates_on_a_causes_cycle() {
        // Mutual causation A → B → A with A resolved. Convergence must TERMINATE: each directed
        // (cause, effect, polarity) triple derives at most one edge, so the cycle cannot spin — the
        // max_rounds cap is only a backstop.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        add_evidence(&mut g, "A", 3, 0);
        g.upsert_node(
            "B".into(),
            "B".into(),
            String::new(),
            NodeType::ContextBlock,
        );
        g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
        g.add_edge("B".into(), "A".into(), 1.0, EdgeType::Causes);
        let total = g.propagate_beliefs_converge(0.2, 0.9, 8);
        assert!(
            total <= 2,
            "each directed causal pair derives at most one edge; the cycle cannot spin (got {total})"
        );
        // Idempotent even through the cycle.
        assert_eq!(g.propagate_beliefs_converge(0.2, 0.9, 8), 0);
    }

    #[test]
    fn converge_is_bit_deterministic_across_rebuilds() {
        let mut a = belief_chain();
        let ta = a.propagate_beliefs_converge(0.2, 0.9, 8);
        let mut b = belief_chain();
        let tb = b.propagate_beliefs_converge(0.2, 0.9, 8);
        assert_eq!(ta, tb, "same edge count across rebuilds");
        assert_eq!(
            a.qbelief(&"C".into()).belief.to_bits(),
            b.qbelief(&"C".into()).belief.to_bits(),
            "the converged belief is bit-identical across runs (replay == live holds)"
        );
    }

    /// A call chain `api → repo → db` (each node depends on the next): `api` calls `repo` calls `db`.
    fn dependency_chain() -> MemoryGraph {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for n in ["api", "repo", "db"] {
            g.upsert_node(n.into(), n.into(), String::new(), NodeType::Module);
        }
        g.add_edge("api".into(), "repo".into(), 1.0, EdgeType::Calls); // api depends on repo
        g.add_edge("repo".into(), "db".into(), 1.0, EdgeType::Calls); // repo depends on db
        g
    }

    #[test]
    fn intervene_reaches_dependents_not_dependencies() {
        let g = dependency_chain();
        // do(db): repo (1 hop) and api (2 hops) depend on db, so both are impacted…
        let impact = g.intervene(&"db".into(), 1.0, 0.8, 4);
        let m = |name: &str| {
            impact
                .iter()
                .find(|(n, _)| n.0 == name)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };
        assert!(
            m("repo") > 0.0 && m("api") > 0.0,
            "dependents of db are impacted"
        );
        assert!(
            m("repo") > m("api"),
            "the closer dependent feels a larger impact ({} vs {})",
            m("repo"),
            m("api")
        );
        assert!(
            !impact.iter().any(|(n, _)| n.0 == "db"),
            "the intervened node is excluded from its own impact set"
        );
        // …but do(api) forces nothing: nothing depends on the top of the chain.
        assert!(
            g.intervene(&"api".into(), 1.0, 0.8, 4).is_empty(),
            "a change to the top-level caller forces no dependents"
        );
    }

    #[test]
    fn intervene_is_depth_bounded() {
        let g = dependency_chain();
        // One hop reaches only the direct dependent (repo), not api two hops away.
        let one = g.intervene(&"db".into(), 1.0, 0.8, 1);
        assert!(one.iter().any(|(n, _)| n.0 == "repo"));
        assert!(
            !one.iter().any(|(n, _)| n.0 == "api"),
            "depth 1 stops at repo"
        );
    }

    #[test]
    fn intervene_terminates_on_a_cycle_and_is_deterministic() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for n in ["a", "b"] {
            g.upsert_node(n.into(), n.into(), String::new(), NodeType::Module);
        }
        g.add_edge("a".into(), "b".into(), 1.0, EdgeType::Calls);
        g.add_edge("b".into(), "a".into(), 1.0, EdgeType::Calls); // a mutual-dependency cycle
                                                                  // Terminates (the test completing proves it) and is byte-identical run to run.
        let i1 = g.intervene(&"a".into(), 1.0, 0.5, 8);
        let i2 = g.intervene(&"a".into(), 1.0, 0.5, 8);
        assert_eq!(i1, i2, "deterministic under a cycle");
        assert!(
            i1.iter().any(|(n, _)| n.0 == "b"),
            "b depends on a ⇒ impacted"
        );
        // An absent origin has no impact set.
        assert!(g.intervene(&"ghost".into(), 1.0, 0.5, 4).is_empty());
    }

    #[test]
    fn blame_reaches_dependencies_the_dual_of_intervene() {
        let g = dependency_chain(); // api → repo → db
        let b = g.blame(&"api".into(), 0.8, 4);
        let w = |name: &str| {
            b.iter()
                .find(|(n, _)| n.0 == name)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };
        // api depends on repo (1 hop) and db (2 hops) — its candidate root causes.
        assert!(
            w("repo") > 0.0 && w("db") > 0.0,
            "dependencies of api are blamed"
        );
        assert!(w("repo") > w("db"), "the nearer dependency ranks first");
        // db is a leaf dependency ⇒ nothing to blame beneath it.
        assert!(g.blame(&"db".into(), 0.8, 4).is_empty());
        // Exact duality: do(db) reaches api (dependent); blame(api) reaches db (dependency).
        assert!(g
            .intervene(&"db".into(), 1.0, 0.8, 4)
            .iter()
            .any(|(n, _)| n.0 == "api"));
        assert!(g
            .blame(&"api".into(), 0.8, 4)
            .iter()
            .any(|(n, _)| n.0 == "db"));
    }

    #[test]
    fn trust_defaults_to_full_and_is_elided_from_snapshots() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("a".into(), "a".into(), String::new(), NodeType::Module);
        assert_eq!(
            g.node_trust(&"a".into()),
            1.0,
            "nodes default to fully trusted"
        );
        // Full trust is elided ⇒ a clean node's snapshot is byte-identical to before the field.
        let clean = serde_json::to_string(g.node(&"a".into()).unwrap()).unwrap();
        assert!(
            !clean.contains("trust"),
            "full trust is not serialized: {clean}"
        );
        // A discounted trust IS recorded (and an absent node reads as fully trusted).
        g.set_trust(&"a".into(), 0.3);
        assert!(serde_json::to_string(g.node(&"a".into()).unwrap())
            .unwrap()
            .contains("trust"));
        assert_eq!(g.node_trust(&"missing".into()), 1.0);
    }

    #[test]
    fn distrust_penalty_is_off_by_default_and_sinks_a_tainted_node() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("a".into(), "a".into(), String::new(), NodeType::Module);
        let full = g.compute_node_score(g.node(&"a".into()).unwrap());
        g.set_trust(&"a".into(), 0.0); // fully distrusted
                                       // w_trust == 0 (default) ⇒ scoring is byte-identical to before the term existed.
        assert_eq!(
            g.compute_node_score(g.node(&"a".into()).unwrap()),
            full,
            "distrust is off by default"
        );
        // Turn the gate on: a distrusted node scores strictly lower.
        g.set_scoring_weights(ScoringWeights {
            w_trust: 0.5,
            ..Default::default()
        });
        assert!(
            g.compute_node_score(g.node(&"a".into()).unwrap()) < full,
            "with w_trust on, a tainted node sinks in recall"
        );
    }

    #[test]
    fn stamp_file_trust_and_one_hop_propagation() {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for id in ["file:src/evil.rs", "file:src/user.rs"] {
            g.upsert_node(id.into(), id.into(), String::new(), NodeType::Module);
        }
        // user.rs depends on evil.rs (importer → defining file).
        g.add_edge(
            "file:src/user.rs".into(),
            "file:src/evil.rs".into(),
            1.0,
            EdgeType::DependsOn,
        );
        g.stamp_file_trust("src/evil.rs", 0.2);
        assert!((g.node_trust(&"file:src/evil.rs".into()) - 0.2).abs() < 1e-9);
        assert_eq!(
            g.node_trust(&"file:src/user.rs".into()),
            1.0,
            "not yet propagated"
        );
        // One hop: the dependent inherits an attenuated share of the distrust.
        assert_eq!(g.propagate_trust(0.9), 1);
        let inherited = g.node_trust(&"file:src/user.rs".into());
        assert!(
            inherited < 1.0 && inherited > 0.2,
            "attenuated, not full: {inherited}"
        );
        // Idempotent: a second pass changes nothing new.
        assert_eq!(g.propagate_trust(0.9), 0);
    }

    #[test]
    fn paging_from_env_falls_back_to_defaults_when_unset() {
        // With no CCOS_PAGING_THRESHOLD / CCOS_MAX_RESIDENT set, the env constructor is identical
        // to `new` with the given defaults (the env-override convention, default-identical).
        assert_eq!(MemoryGraph::paging_threshold_from_env(0.2), 0.2);
        let g = MemoryGraph::new_from_env(0.2, 123);
        assert_eq!(g.paging_threshold, 0.2);
        assert_eq!(g.max_in_memory_nodes, 123);
    }

    #[test]
    fn resolve_symbol_calls_ladder_global_unique_ambiguous_and_self() {
        // `connect` unique; `shared` defined twice (ambiguous); `run` is the caller.
        let mk = || {
            graph_with_defs(&[
                ("src/db.rs", "connect"),
                ("src/db.rs", "shared"),
                ("src/net.rs", "shared"),
                ("src/api.rs", "run"),
            ])
        };
        let mut g = mk();
        g.set_pending_calls(
            "src/api.rs",
            vec![
                ("run".into(), "connect".into(), 1), // unique → Tier C links
                ("run".into(), "shared".into(), 2),  // ambiguous → skipped
                ("run".into(), "run".into(), 3),     // self → dropped
                ("run".into(), "nope".into(), 4),    // unknown → skipped
            ],
        );
        let added = g.resolve_symbol_calls();
        assert_eq!(added, 1);
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "only the globally-unique callee links; ambiguous/self/unknown are skipped"
        );
        // Determinism: a fresh build + resolve yields the identical edge set.
        let mut g2 = mk();
        g2.set_pending_calls(
            "src/api.rs",
            vec![
                ("run".into(), "connect".into(), 1),
                ("run".into(), "shared".into(), 2),
                ("run".into(), "run".into(), 3),
            ],
        );
        g2.resolve_symbol_calls();
        assert_eq!(calls_of(&g), calls_of(&g2));
    }

    #[test]
    fn resolve_symbol_calls_tier_a_import_disambiguates() {
        // `connect` is ambiguous globally (db + net), so Tier C would skip — but an import in
        // api.rs scopes the call to db::connect, so Tier A resolves it precisely.
        let mut g = graph_with_defs(&[
            ("src/db.rs", "connect"),
            ("src/net.rs", "connect"),
            ("src/api.rs", "run"),
        ]);
        // the `use crate::db::connect;` node the parser would have produced.
        let uid = NodeId("use:src/api.rs:crate::db::connect".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls("src/api.rs", vec![("run".into(), "connect".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "the import scopes the otherwise-ambiguous call to db::connect (Tier A)"
        );
    }

    // ── Non-`use` module-path fallback (submod::fn(), extern::fn(), a::b::fn()) ──────────────────
    #[test]
    fn resolve_symbol_calls_bare_local_module_path_resolves() {
        // Gap A: `mod helpers; helpers::work()` with NO use. Leading `helpers` is a local module
        // (crate "", module "helpers") — resolve exactly via file_index then defs.
        let mut g = graph_with_defs(&[("src/helpers.rs", "work"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "helpers::work".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/helpers.rs:work".to_string()
            )],
            "bare local module path helpers::work resolves via file_index"
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_extern_crate_path_is_not_resolved() {
        // A bare `othercrate::helper()` (no use, no local module `othercrate`) is deliberately NOT
        // resolved into an unrelated ingested external crate: the fallback consults only the caller's
        // OWN crate, so an external reading can never mint an edge Rust's shadow rule would forbid.
        // (Cross-crate calls resolve via an explicit `use` — see the import tests.)
        let mut g = graph_with_defs(&[("othercrate/src/lib.rs", "helper"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "othercrate::helper".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "a bare extern-crate path is conservatively not resolved: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_nested_module_path_resolves() {
        // Gap F: `outer::inner::work()` with NO use. Full prefix `outer::inner` is an exact module.
        let mut g = graph_with_defs(&[("src/outer/inner.rs", "work"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "outer::inner::work".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/outer/inner.rs:work".to_string()
            )],
            "bare nested module path resolves via exact full-prefix module match"
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_symbolless_local_module_does_not_fall_to_extern() {
        // Adversarial-review defect 1: a local module `sib` exists but lacks `helper`, while an
        // ingested external crate `sib` HAS `helper`. In Rust `mod sib;` shadows extern crate `sib`,
        // so `sib::helper()` binds to the (symbol-less) local module ⇒ SKIP. The fallback must NOT
        // fall through to the external crate (that was a false edge before the fix).
        let mut g = graph_with_defs(&[
            ("src/sib.rs", "other"), // local module sib: ("", "sib", "other") — NO `helper`
            ("sib/src/lib.rs", "helper"), // extern crate sib: ("sib", "", "helper")
            ("src/api.rs", "run"),
        ]);
        g.set_pending_calls("src/api.rs", vec![("run".into(), "sib::helper".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "a symbol-less local module shadows the extern crate ⇒ skip, no false extern edge: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_local_symbol_wins_over_extern_crate() {
        // Both a local module `sib` and a same-named extern crate define `helper`. Rust binds
        // `sib::helper` to the LOCAL module ⇒ resolve to the local symbol, never the extern crate's.
        let mut g = graph_with_defs(&[
            ("src/sib.rs", "helper"),     // local ("", "sib", "helper")
            ("sib/src/lib.rs", "helper"), // extern ("sib", "", "helper")
            ("src/api.rs", "run"),
        ]);
        g.set_pending_calls("src/api.rs", vec![("run".into(), "sib::helper".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/sib.rs:helper".to_string()
            )],
            "a local module shadows a same-named extern crate — resolve to the local symbol"
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_missing_submodule_skips() {
        // `outer::inner::work` where `outer` exists but `outer::inner` has no file ⇒ SKIP; never
        // shorten to an ancestor `outer::work`.
        let mut g = graph_with_defs(&[
            ("src/outer.rs", "work"), // only ("", "outer") exists, NOT ("", "outer::inner")
            ("src/api.rs", "run"),
        ]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "outer::inner::work".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "missing nested submodule ⇒ no ancestor fallback: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_unknown_prefix_no_name_fallback_skips() {
        // `nope::work` where neither a local module `nope` nor an external crate `nope` has a file,
        // yet a globally-unique free `work` exists. The fallback must NOT degrade into a bare
        // final-name (Tier C) lookup.
        let mut g = graph_with_defs(&[("src/other.rs", "work"), ("src/api.rs", "run")]);
        g.set_pending_calls("src/api.rs", vec![("run".into(), "nope::work".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "unknown prefix must not fall back to a free `work`: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_missing_symbol_skips() {
        // Module resolves but the target symbol is absent ⇒ SKIP (no fall-through to global).
        let mut g = graph_with_defs(&[("src/helpers.rs", "work"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "helpers::missing".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "real module but missing symbol ⇒ skip: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_type_method_still_wins() {
        // `Foo::bar` where (Foo,bar) is a unique type method and there is NO module `Foo`. The
        // fallback contributes no module_hit; the type side resolves — the new branch must not
        // perturb it (caller in a different module, no use).
        let mut g = graph_with_defs(&[("src/a.rs", "bar"), ("src/api.rs", "run")]);
        g.set_pending_methods("src/a.rs", vec![("Foo".into(), "bar".into())]);
        g.set_pending_calls("src/api.rs", vec![("run".into(), "Foo::bar".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/a.rs:bar".to_string()
            )],
            "type method still resolves; the module-path fallback adds no spurious module_hit"
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_type_method_wins_over_same_named_extern_crate() {
        // Adversarial-review defect 2: `Registry::get` is a receiver-inferred TYPE method, and an
        // ingested external crate is (coincidentally) named `Registry` with a `get` fn. Because the
        // fallback consults only the caller's own crate (no module `Registry` there), it contributes
        // no module_hit — so the type method still resolves; the extern crate does not steal the edge.
        let mut g = graph_with_defs(&[
            ("src/a.rs", "get"),            // the real type method Registry::get
            ("Registry/src/lib.rs", "get"), // unrelated extern crate literally named `Registry`
            ("src/api.rs", "run"),
        ]);
        g.set_pending_methods("src/a.rs", vec![("Registry".into(), "get".into())]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "Registry::get".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/a.rs:get".to_string()
            )],
            "type method resolves; a same-named extern crate does not steal the edge"
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_type_tail_not_stripped_skips() {
        // `db::Conn::open` where module `db` exists but `db::Conn` does not (Conn is a type). The
        // exact module `db::Conn` is absent ⇒ SKIP; must NOT strip the type tail to link `db::open`.
        let mut g = graph_with_defs(&[
            ("src/db.rs", "open"), // ("", "db", "open") exists; ("", "db::Conn") does NOT
            ("src/api.rs", "run"),
        ]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "db::Conn::open".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "type tail must not be stripped to a free db::open: {:?}",
            calls_of(&g)
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_import_wins_over_local_module() {
        // A file has `use crate::other::helpers;` AND a local `mod helpers`, both with `work`. The
        // unique import must pin `crate::other::helpers`; the local-module fallback must NOT fire.
        let mut g = graph_with_defs(&[
            ("src/other/helpers.rs", "work"), // import target ("", "other::helpers", "work")
            ("src/helpers.rs", "work"),       // local ("", "helpers", "work")
            ("src/api.rs", "run"),
        ]);
        let uid = NodeId("use:src/api.rs:crate::other::helpers".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "helpers::work".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/other/helpers.rs:work".to_string()
            )],
            "a unique import wins; the bare fallback does not fire when an import matches"
        );
    }

    #[test]
    fn resolve_symbol_calls_bare_module_path_is_order_independent() {
        // replay == live: local submodule + nested-submodule calls resolved with defs registered in
        // two different orders ⇒ identical (indices are built over sorted node ids, not order).
        let build = |rev: bool| {
            let mut defs = vec![
                ("src/helpers.rs", "work"),
                ("src/outer/inner.rs", "work"),
                ("src/api.rs", "run"),
            ];
            if rev {
                defs.reverse();
            }
            let mut g = graph_with_defs(&defs);
            g.set_pending_calls(
                "src/api.rs",
                vec![
                    ("run".into(), "helpers::work".into(), 1),
                    ("run".into(), "outer::inner::work".into(), 2),
                ],
            );
            g.resolve_symbol_calls();
            calls_of(&g)
        };
        let a = build(false);
        assert_eq!(a.len(), 2, "both bare submodule-path calls resolve: {a:?}");
        assert_eq!(
            a,
            build(true),
            "resolution is independent of def-registration order (replay == live)"
        );
    }

    #[test]
    fn resolve_symbol_calls_ambiguous_import_scope_skips_never_falls_to_local() {
        // api.rs imports `connect` from BOTH db and net (ambiguous Tier A) AND defines a local
        // `connect`. Tier A is populated-but-ambiguous, so we skip — never linking the same-module
        // local (resolve-uniquely-or-skip). (Only reachable on non-compiling Rust; no guessing.)
        let mut g = graph_with_defs(&[
            ("src/db.rs", "connect"),
            ("src/net.rs", "connect"),
            ("src/api.rs", "connect"), // the same-module local Tier B would otherwise grab
            ("src/api.rs", "run"),
        ]);
        for p in ["crate::db::connect", "crate::net::connect"] {
            let uid = NodeId(format!("use:src/api.rs:{p}"));
            g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
            g.add_edge(
                NodeId("file:src/api.rs".into()),
                uid,
                0.5,
                EdgeType::DependsOn,
            );
        }
        g.set_pending_calls("src/api.rs", vec![("run".into(), "connect".into(), 1)]);
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "ambiguous import scope skips — it does not fall back to the same-module local"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_crate_rooted() {
        // `crate::db::connect()` resolves absolutely via its module prefix — no import needed.
        let mut g = graph_with_defs(&[("src/db.rs", "connect"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "crate::db::connect".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "crate-rooted qualified call resolves to the module's symbol (Slice 2)"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_via_import_alias() {
        // `use crate::db;` then `db::connect()` — the leading segment is expanded via the import.
        let mut g = graph_with_defs(&[("src/db.rs", "connect"), ("src/api.rs", "run")]);
        let uid = NodeId("use:src/api.rs:crate::db".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls("src/api.rs", vec![("run".into(), "db::connect".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/db.rs:connect".to_string()
            )],
            "module-alias call resolves by expanding the import to an absolute path (Slice 2)"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_external_or_unknown_type_skips() {
        // `std::mem::swap()` (external root) and `Foo::bar()` (a type with no import bringing in
        // `Foo`) expand to no local module — qualified-but-unresolvable skips, never falling back to
        // a same-file local of the same final name (`swap`/`bar` exist locally yet must NOT be
        // linked). (`Self::…` is a different case — it resolves same-module; see the Slice-3 tests.)
        let mut g = graph_with_defs(&[
            ("src/api.rs", "run"),
            ("src/api.rs", "bar"),
            ("src/api.rs", "swap"),
        ]);
        g.set_pending_calls(
            "src/api.rs",
            vec![
                ("run".into(), "std::mem::swap".into(), 1),
                ("run".into(), "Foo::bar".into(), 2),
            ],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "qualified external/unknown-type calls skip — no import expands them, no fallback to a local"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_no_ancestor_fallback() {
        // `crate::a::b::connect()` but only module `a` (src/a.rs) exists — no file for `a::b` (an
        // inline submodule, or not-yet-ingested). Must SKIP, never falling back to `a`'s `connect`
        // (the inline-submodule / ancestor-fallback false edge the exact-module guard closes).
        let mut g = graph_with_defs(&[("src/a.rs", "connect"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "crate::a::b::connect".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "qualified call to a missing submodule skips — no ancestor fallback to a::connect"
        );
    }

    #[test]
    fn resolve_symbol_calls_qualified_variant_or_assoc_collision_skips() {
        // `crate::db::Conn::open()` — `Conn` is a type, not a module, so there is no `db::Conn`
        // file. Must SKIP, not drop the `Conn` tail and link to the free fn `open` in module `db`.
        let mut g = graph_with_defs(&[("src/db.rs", "open"), ("src/api.rs", "run")]);
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "crate::db::Conn::open".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert!(
            calls_of(&g).is_empty(),
            "associated/variant call skips — the type tail is not collapsed to the module's free fn"
        );
    }

    #[test]
    fn resolve_symbol_calls_self_method_resolves_same_module() {
        // `self.helper()` (captured as `Self::helper`) inside api.rs resolves to api.rs's own
        // `helper` — the enclosing impl's method lives in the caller's module.
        let mut g = graph_with_defs(&[("src/api.rs", "run"), ("src/api.rs", "helper")]);
        g.set_pending_calls("src/api.rs", vec![("run".into(), "Self::helper".into(), 1)]);
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/api.rs:helper".to_string()
            )],
            "self.helper() resolves to the same-module method (Slice 3)"
        );
    }

    #[test]
    fn resolve_symbol_calls_self_method_never_uses_imports_or_global() {
        // api.rs has its OWN `connect` AND imports `crate::db::connect` (also a global in db.rs).
        // `self.connect()` must resolve to api.rs's own connect (same module) — NEVER the imported
        // or global db::connect. This isolates the Slice-3 same-module-only guard: if it leaked to
        // Tier A it would pick db's connect; if it fell to the qualified branch it would skip.
        let mut g = graph_with_defs(&[
            ("src/db.rs", "connect"),
            ("src/api.rs", "connect"),
            ("src/api.rs", "run"),
        ]);
        let uid = NodeId("use:src/api.rs:crate::db::connect".into());
        g.upsert_node(uid.clone(), "use".into(), "".into(), NodeType::Module);
        g.add_edge(
            NodeId("file:src/api.rs".into()),
            uid,
            0.5,
            EdgeType::DependsOn,
        );
        g.set_pending_calls(
            "src/api.rs",
            vec![("run".into(), "Self::connect".into(), 1)],
        );
        g.resolve_symbol_calls();
        assert_eq!(
            calls_of(&g),
            vec![(
                "sym:src/api.rs:run".to_string(),
                "sym:src/api.rs:connect".to_string()
            )],
            "self.connect() resolves to api's OWN connect, not the imported or global db::connect"
        );
    }

    #[test]
    fn smaller_failure_decay_reduces_distant_pressure() {
        let pressure_at_c = |decay: f64| {
            let mut g = MemoryGraph::new(0.0, usize::MAX);
            g.set_scoring_weights(ScoringWeights {
                failure_decay: decay,
                ..Default::default()
            });
            for n in ["a", "b", "c"] {
                g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
            }
            g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
            g.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
            g.set_failure_relevance(&"a".into(), 1.0);
            g.propagate_failure(&"a".into(), 0, 5);
            g.nodes.get(&"c".into()).unwrap().failure_relevance
        };
        assert!(
            pressure_at_c(0.9) > pressure_at_c(0.5),
            "a smaller decay must attenuate failure pressure two hops out"
        );
    }

    #[test]
    fn test_failure_propagation() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("root".into(), "R".into(), "".into(), NodeType::Module);
        graph.upsert_node("child".into(), "C".into(), "".into(), NodeType::Module);
        graph.add_edge("root".into(), "child".into(), 1.0, EdgeType::DependsOn);
        graph.set_failure_relevance(&"root".into(), 1.0);
        graph.propagate_failure(&"root".into(), 0, 3);
        let child = graph.nodes.get(&"child".into()).unwrap();
        assert!(child.failure_relevance > 0.0);
    }

    #[test]
    fn high_fanout_node_damps_pressure_but_low_fanout_does_not() {
        // Degree-aware damping (FIELD_CAMPAIGN_H.md #2): a focused node (out-degree
        // 1) passes pressure undamped; a hub (out-degree ≫ failure_fanout)
        // distributes it, so each neighbour receives strictly less.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        g.upsert_node("f".into(), "f".into(), "".into(), NodeType::Module);
        g.upsert_node("t".into(), "t".into(), "".into(), NodeType::Module);
        g.add_edge("f".into(), "t".into(), 1.0, EdgeType::DependsOn);
        g.upsert_node("h".into(), "h".into(), "".into(), NodeType::Module);
        for i in 0..20 {
            let leaf = format!("l{i}");
            g.upsert_node(
                leaf.clone().into(),
                leaf.clone(),
                "".into(),
                NodeType::Module,
            );
            g.add_edge("h".into(), leaf.into(), 1.0, EdgeType::Contains);
        }
        g.set_failure_relevance(&"f".into(), 1.0);
        g.propagate_failure(&"f".into(), 0, 1);
        g.set_failure_relevance(&"h".into(), 1.0);
        g.propagate_failure(&"h".into(), 0, 1);

        let focused = g.nodes.get(&"t".into()).unwrap().failure_relevance;
        let from_hub = g.nodes.get(&"l0".into()).unwrap().failure_relevance;
        assert!(
            from_hub < focused,
            "a high-fanout node must spread less pressure per neighbour: hub={from_hub} focused={focused}"
        );
        // 20 leaves with fanout 6 ⇒ ~6/20 of the undamped amount.
        assert!((from_hub - focused * 6.0 / 20.0).abs() < 1e-9);
    }

    #[test]
    fn degree_damping_preserves_sparse_chain_reach() {
        // a→b→c→d, every out-degree 1: damping is a no-op, so depth-3 propagation
        // still reaches the 3-hop cause d — the property that lets us keep a deep
        // default depth without flooding dense graphs.
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for n in ["a", "b", "c", "d"] {
            g.upsert_node(n.into(), n.into(), "".into(), NodeType::Module);
        }
        g.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("c".into(), "d".into(), 1.0, EdgeType::DependsOn);
        g.set_failure_relevance(&"a".into(), 1.0);
        g.propagate_failure(&"a".into(), 0, 3);
        assert!(
            g.nodes.get(&"d".into()).unwrap().failure_relevance > 0.0,
            "the 3-hop cause must still be pressured on a sparse chain"
        );
    }

    #[test]
    fn test_paging_enforcement() {
        let mut graph = MemoryGraph::new(0.0, 5);
        for i in 0..10 {
            graph.upsert_node(
                NodeId(format!("n{}", i)),
                format!("Node{}", i),
                "x".into(),
                NodeType::Unknown,
            );
        }
        assert!(graph.node_count() <= 5);
    }

    #[test]
    fn eviction_policy_is_wired_into_paging_and_can_override_the_greedy() {
        use crate::eviction_policy::{EvictionPolicy, PagingState, KEEP};
        // Two nodes, cap 1 → exactly one is evicted. `x` scores ~0 (bucket 0),
        // `y` scores ~0.95 (bucket 2). Build at a high cap so setup doesn't
        // auto-page, then tighten the cap and page manually.
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node("x".into(), "x".into(), "x".into(), NodeType::Symbol);
        g.upsert_node("y".into(), "y".into(), "y".into(), NodeType::Symbol);
        {
            let x = g.nodes.get_mut(&NodeId("x".into())).unwrap();
            x.base_importance = 0.0;
            x.failure_relevance = 0.0;
            x.recency = 0.0;
            x.access_count = 1;
        }
        {
            let y = g.nodes.get_mut(&NodeId("y".into())).unwrap();
            y.base_importance = 1.0;
            y.failure_relevance = 1.0;
            y.recency = 1.0;
            y.access_count = 1;
        }
        g.max_in_memory_nodes = 1;

        // Untrained → deterministic greedy: the low-score `x` is evicted.
        let mut greedy = g.clone();
        greedy.enforce_paging();
        assert!(
            greedy.nodes.contains_key(&NodeId("y".into()))
                && !greedy.nodes.contains_key(&NodeId("x".into())),
            "greedy evicts the low-score node x"
        );

        // Train the policy to strongly KEEP x's exact state → x now outranks y,
        // so y is evicted instead. Proves the policy is consulted by enforce_paging.
        let mut trained = g.clone();
        let mut policy = EvictionPolicy::new();
        let x_state = PagingState {
            score: 0,
            recency: 1,
            pressure: 0,
            size: 0,
        };
        policy.q.insert((x_state, KEEP), 100.0);
        trained.set_eviction_policy(policy);
        trained.enforce_paging();
        assert!(
            trained.nodes.contains_key(&NodeId("x".into())),
            "the trained policy protects x"
        );
        assert!(
            !trained.nodes.contains_key(&NodeId("y".into())),
            "the trained policy evicts y instead"
        );
    }

    #[test]
    fn eviction_demotes_to_cold_and_page_in_swaps_it_back() {
        // Build at a high cap so setup does not auto-page, then tighten to 1.
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node("hot".into(), "hot".into(), "x".into(), NodeType::Symbol);
        g.upsert_node("cold".into(), "cold".into(), "y".into(), NodeType::Symbol);
        g.add_edge("hot".into(), "cold".into(), 0.6, EdgeType::Contains);
        {
            let h = g.nodes.get_mut(&NodeId("hot".into())).unwrap();
            h.base_importance = 1.0;
            h.recency = 1.0;
        }
        {
            let c = g.nodes.get_mut(&NodeId("cold".into())).unwrap();
            c.base_importance = 0.0;
            c.recency = 0.0;
        }
        g.max_in_memory_nodes = 1;
        g.enforce_paging();

        // Non-destructive eviction: the victim is DEMOTED to COLD, not dropped.
        assert_eq!(g.node_count(), 1, "resident set capped");
        assert!(g.node(&NodeId("hot".into())).is_some());
        assert!(g.node(&NodeId("cold".into())).is_none(), "out of resident");
        assert!(
            g.is_cold(&NodeId("cold".into())),
            "kept in COLD — nothing lost"
        );
        assert_eq!(g.cold_count(), 1);
        assert_eq!(g.edge_count(), 0, "incident edge archived on demotion");

        // Page it back in: a swap — `cold` returns resident, `hot` demotes.
        assert!(g.page_in(&NodeId("cold".into())));
        assert!(g.node(&NodeId("cold".into())).is_some(), "paged back in");
        assert!(!g.is_cold(&NodeId("cold".into())));
        assert!(
            g.is_cold(&NodeId("hot".into())),
            "the lowest-scored other node swapped out, not the requested one"
        );
        assert_eq!(g.node_count(), 1, "resident set still capped (a swap)");

        // page_in on an id that is not cold is a no-op.
        assert!(!g.page_in(&NodeId("nope".into())));
    }

    #[test]
    fn cold_neighbours_is_symmetric_across_demotion_order() {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["a", "b", "c"] {
            g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Symbol);
        }
        g.add_edge("a".into(), "b".into(), 0.6, EdgeType::DependsOn);
        // Demote everything (a target resident set of 0).
        g.max_in_memory_nodes = 0;
        g.enforce_paging();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.cold_count(), 3);
        // a–b are neighbours regardless of which was archived with the edge; c is isolated.
        assert_eq!(
            g.cold_neighbours(&NodeId("a".into())),
            vec![NodeId("b".into())]
        );
        assert_eq!(
            g.cold_neighbours(&NodeId("b".into())),
            vec![NodeId("a".into())]
        );
        assert!(g.cold_neighbours(&NodeId("c".into())).is_empty());
    }

    // ── slice 3: COLD spill to disk (RAM-bounded content, disk-unbounded) ──────

    fn spill_temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ccos_spill_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn spill_round_trips_cold_content_losslessly() {
        // Resident cap 1 ⇒ all but one node is demoted to the COLD tier.
        let mut g = MemoryGraph::new(0.2, 1);
        let bodies: Vec<(String, String)> = (0..6)
            .map(|i| {
                (
                    format!("n{i}"),
                    format!("// node {i}\n{}\n", "payload ".repeat(50 + i)),
                )
            })
            .collect();
        for (id, body) in &bodies {
            g.upsert_node(
                id.clone().into(),
                id.clone(),
                body.clone(),
                NodeType::Symbol,
            );
        }
        assert!(
            g.cold_count() >= 4,
            "expected a populated COLD tier, got {}",
            g.cold_count()
        );
        assert!(g.cold_inline_bytes() > 0);

        // Attach a spill store with a tiny budget → coldest content flushes to disk.
        let dir = spill_temp_dir("roundtrip");
        g.attach_cold_spill(&dir, 64).unwrap();
        assert!(
            g.cold_spilled_count() > 0,
            "a 64-byte budget must force at least one spill"
        );
        assert!(
            g.cold_inline_bytes() <= 64,
            "resident COLD content must be within budget, got {}",
            g.cold_inline_bytes()
        );

        // Every body must reconstruct exactly when its node is made resident.
        for (id, body) in &bodies {
            let nid = NodeId(id.clone());
            g.page_in(&nid); // faults the blob back (hash-verified); no-op if resident
            let n = g.node(&nid).expect("node must be resident after page_in");
            assert_eq!(
                &n.content, body,
                "content for {id} must round-trip losslessly"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spill_decisions_are_deterministic() {
        fn spilled_set(dir: &std::path::Path) -> Vec<NodeId> {
            let mut g = MemoryGraph::new(0.2, 1);
            for i in 0..6 {
                let id = format!("n{i}");
                g.upsert_node(
                    id.clone().into(),
                    id,
                    format!("body {}", "z".repeat(40 + i)),
                    NodeType::Symbol,
                );
            }
            g.attach_cold_spill(dir, 50).unwrap();
            let mut out: Vec<NodeId> = g.cold_ids().filter(|id| g.is_spilled(id)).collect();
            out.sort();
            out
        }
        let d1 = spill_temp_dir("det1");
        let d2 = spill_temp_dir("det2");
        let a = spilled_set(&d1);
        let b = spilled_set(&d2);
        assert_eq!(
            a, b,
            "identical histories must spill the identical node set"
        );
        assert!(!a.is_empty(), "the budget should have forced some spills");
        std::fs::remove_dir_all(&d1).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn spill_off_by_default_leaves_serialization_unchanged() {
        let mut g = MemoryGraph::new(0.2, 1);
        for i in 0..5 {
            let id = format!("n{i}");
            g.upsert_node(
                id.clone().into(),
                id,
                format!("content {}", "q".repeat(30)),
                NodeType::Symbol,
            );
        }
        assert!(g.cold_count() > 0);
        assert_eq!(
            g.cold_spilled_count(),
            0,
            "nothing spills without an attached store"
        );
        // With no store attached, every `spill` stub is `None` and elided, so the
        // JSON carries no new field — byte-identical to the pre-spill layout.
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("\"spill\""),
            "the default path must not emit any spill stub"
        );
    }

    #[test]
    fn a_tampered_or_detached_spill_blob_is_a_cold_miss() {
        let mut g = MemoryGraph::new(0.2, 1);
        for i in 0..4 {
            let id = format!("n{i}");
            g.upsert_node(
                id.clone().into(),
                id,
                format!("secret-{i} {}", "p".repeat(60)),
                NodeType::Symbol,
            );
        }
        let dir = spill_temp_dir("tamper");
        g.attach_cold_spill(&dir, 0).unwrap(); // budget 0 ⇒ spill all cold content
        let spilled: Vec<NodeId> = g.cold_ids().filter(|id| g.is_spilled(id)).collect();
        assert!(spilled.len() >= 2, "expected several spilled nodes");

        // Tamper: rewrite every blob so its bytes no longer hash to its name.
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::write(entry.unwrap().path(), b"tampered").unwrap();
        }
        let victim = spilled[0].clone();
        assert!(
            !g.page_in(&victim),
            "a tampered blob must fail the hash check → cold-miss"
        );
        assert!(
            g.is_cold(&victim),
            "the victim stays cold, not silently restored"
        );
        assert!(
            g.node(&victim).is_none(),
            "no empty or partial node may become resident on a failed fault"
        );

        // Detaching the store makes the remaining spilled nodes unreachable until
        // the same directory is re-attached.
        g.detach_cold_spill();
        let other = spilled.iter().find(|id| **id != victim).unwrap().clone();
        assert!(!g.page_in(&other), "a detached store is a cold-miss");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── slice 4: COLD compaction (lossy, observable; the deepest tier) ─────────

    /// A deterministic multi-sentence paragraph so CausalSumm has something to
    /// extract (≥3 sentences ⇒ the summary is strictly shorter).
    fn para(tag: &str, sentences: usize) -> String {
        (0..sentences)
            .map(|i| format!("{tag} clause {i} has a handful of filler words here. "))
            .collect()
    }

    #[test]
    fn compaction_shrinks_the_coldest_tail_and_spares_warmer_cold() {
        // Resident cap 1: at equal scores `enforce_paging` demotes the *smallest*
        // id and keeps the *largest* resident, so after ingesting a,b,c,d the
        // cold tier is {a,b,c} and d is resident. Compaction is coldest-first by
        // id, so a,b compact while c (the largest cold id) is spared once the
        // budget is met.
        let mut g = MemoryGraph::new(0.2, 1);
        let a = para("a", 14);
        let b = para("b", 14);
        let c = para("c", 10);
        g.upsert_node("a".into(), "a".into(), a.clone(), NodeType::ContextBlock);
        g.upsert_node("b".into(), "b".into(), b.clone(), NodeType::ContextBlock);
        g.upsert_node("c".into(), "c".into(), c.clone(), NodeType::ContextBlock);
        g.upsert_node("d".into(), "d".into(), para("d", 6), NodeType::ContextBlock);
        assert_eq!(g.cold_count(), 3);
        assert!(
            g.is_cold(&NodeId("c".into())),
            "c is the warmest (largest-id) cold node"
        );

        let before = g.cold_inline_bytes() + g.cold_spilled_bytes();
        g.set_cold_content_budget(Some(1000));
        let after = g.cold_inline_bytes() + g.cold_spilled_bytes();

        assert!(
            g.cold_compacted_count() >= 1,
            "the budget must force compaction"
        );
        assert!(after < before, "compaction must shrink total cold content");
        assert!(after <= 1000, "compaction must reach the budget here");
        assert!(
            g.is_compacted(&NodeId("a".into())),
            "the coldest entry is compacted"
        );
        assert!(
            !g.is_compacted(&NodeId("c".into())),
            "the warmest cold node is compacted last and spared by the budget"
        );

        // Lossy contract on compacted entries; lossless on the spared one.
        let originals = [("a", &a), ("b", &b), ("c", &c)];
        let cold_ids: Vec<NodeId> = g.cold_ids().collect();
        for cid in cold_ids {
            let orig = originals.iter().find(|(id, _)| *id == cid.0).unwrap().1;
            let was_compacted = g.is_compacted(&cid);
            g.page_in(&cid);
            let content = &g.node(&cid).expect("resident after page_in").content;
            if was_compacted {
                assert!(
                    content.len() < orig.len(),
                    "compacted content must be shorter (lossy)"
                );
                assert_ne!(
                    content, orig,
                    "compacted content is a summary, not the original"
                );
            } else {
                assert_eq!(content, orig, "a spared (warm) cold node stays lossless");
            }
        }
    }

    #[test]
    fn compaction_is_off_by_default_and_lossless() {
        let mut g = MemoryGraph::new(0.2, 1);
        let bodies: Vec<(String, String)> = (0..4)
            .map(|i| (format!("n{i}"), para(&format!("n{i}"), 12)))
            .collect();
        for (id, body) in &bodies {
            g.upsert_node(
                id.clone().into(),
                id.clone(),
                body.clone(),
                NodeType::ContextBlock,
            );
        }
        assert!(g.cold_count() > 0);
        assert_eq!(g.cold_compacted_count(), 0, "no budget ⇒ no compaction");
        // No `compacted` stub on the default path ⇒ serialization unchanged.
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("\"compacted\""),
            "default path must emit no compacted flag"
        );
        // Every cold body is still the verbatim original.
        for (id, body) in &bodies {
            let nid = NodeId(id.clone());
            if g.is_cold(&nid) {
                g.page_in(&nid);
                assert_eq!(
                    &g.node(&nid).unwrap().content,
                    body,
                    "lossless without a budget"
                );
            }
        }
    }

    #[test]
    fn compaction_decisions_are_deterministic() {
        fn compacted_set() -> Vec<NodeId> {
            let mut g = MemoryGraph::new(0.2, 1);
            g.upsert_node(
                "z_warm".into(),
                "z_warm".into(),
                para("zwarm", 10),
                NodeType::ContextBlock,
            );
            for id in ["a", "b", "c"] {
                g.upsert_node(
                    id.into(),
                    id.to_string(),
                    para(id, 14),
                    NodeType::ContextBlock,
                );
            }
            g.set_cold_content_budget(Some(900));
            let mut out: Vec<NodeId> = g.cold_ids().filter(|id| g.is_compacted(id)).collect();
            out.sort();
            out
        }
        let a = compacted_set();
        let b = compacted_set();
        assert_eq!(
            a, b,
            "identical histories must compact the identical node set"
        );
        assert!(
            !a.is_empty(),
            "the budget should have forced some compaction"
        );
    }

    // ── audit pass 4: spill-blob GC (F1) and compaction floor (F5) ────────────

    #[test]
    fn cold_resident_bytes_counts_metadata_and_edges() {
        let build = |with_edge: bool| {
            let mut g = MemoryGraph::new(0.2, usize::MAX);
            g.upsert_node(
                "file:a".into(),
                "file:a".into(),
                "A".repeat(200),
                NodeType::Module,
            );
            g.upsert_node(
                "sym:a:f".into(),
                "sym:a:f".into(),
                "b".repeat(200),
                NodeType::Symbol,
            );
            if with_edge {
                g.add_edge("file:a".into(), "sym:a:f".into(), 0.6, EdgeType::Contains);
            }
            g.max_in_memory_nodes = 0;
            g.enforce_paging();
            g
        };
        let g = build(true);
        assert!(
            g.cold_resident_bytes() >= "file:a".len() + "sym:a:f".len(),
            "counts at least the node ids"
        );
        // The archived edge contributes resident bytes (it stays in RAM).
        assert!(
            build(true).cold_resident_bytes() > build(false).cold_resident_bytes(),
            "an archived edge adds to the resident COLD footprint"
        );
    }

    #[test]
    fn removing_a_spilled_node_reclaims_its_blob() {
        let mut g = MemoryGraph::new(0.2, 1);
        for i in 0..3 {
            let id = format!("n{i}");
            g.upsert_node(
                id.clone().into(),
                id,
                format!("c{i} {}", "p".repeat(80)),
                NodeType::Symbol,
            );
        }
        let dir = spill_temp_dir("gc_remove");
        g.attach_cold_spill(&dir, 0).unwrap(); // budget 0 ⇒ spill all cold content
        let spilled: Vec<NodeId> = g.cold_ids().filter(|id| g.is_spilled(id)).collect();
        assert!(spilled.len() >= 2, "expected several spilled nodes");
        let before = std::fs::read_dir(&dir).unwrap().count();
        let victim = spilled[0].clone();
        g.remove_node(&victim);
        let after = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(
            after,
            before - 1,
            "removing a spilled node reclaims exactly its (unshared) blob: {before} -> {after}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gc_keeps_a_blob_shared_by_another_cold_node() {
        // a and b have IDENTICAL content ⇒ one deduplicated blob. Removing a must
        // NOT delete it, because b still references it.
        let mut g = MemoryGraph::new(0.2, 1);
        let shared = format!("shared body {}", "x".repeat(80));
        g.upsert_node("a".into(), "a".into(), shared.clone(), NodeType::Symbol);
        g.upsert_node("b".into(), "b".into(), shared.clone(), NodeType::Symbol);
        g.upsert_node(
            "c".into(),
            "c".into(),
            format!("other {}", "y".repeat(80)),
            NodeType::Symbol,
        );
        let dir = spill_temp_dir("gc_dedup");
        g.attach_cold_spill(&dir, 0).unwrap();
        assert!(
            g.is_spilled(&NodeId("a".into())) && g.is_spilled(&NodeId("b".into())),
            "a and b should both be spilled (and dedup to one blob)"
        );
        g.remove_node(&NodeId("a".into()));
        // b's shared blob must have survived: it still faults back losslessly.
        assert!(
            g.page_in(&NodeId("b".into())),
            "b's shared blob must survive a's removal"
        );
        assert_eq!(g.node(&NodeId("b".into())).unwrap().content, shared);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_parks_unshrinkable_entries_at_the_floor() {
        let mut g = MemoryGraph::new(0.2, 1);
        // Tiny content that cannot summarise/skeletonise smaller.
        for i in 0..4 {
            let id = format!("n{i}");
            g.upsert_node(id.clone().into(), id, format!("x{i}"), NodeType::Symbol);
        }
        assert!(g.cold_count() >= 3);
        g.set_cold_content_budget(Some(1)); // impossibly tight ⇒ nothing shrinks
        let parked = g.cold_ids().filter(|id| g.is_at_floor(id)).count();
        assert!(
            parked > 0,
            "un-shrinkable cold entries are parked at the floor, not re-tried forever"
        );
        // Nothing was actually compacted (they couldn't shrink), and re-enforcing
        // is idempotent (the parked entries are skipped).
        assert_eq!(g.cold_compacted_count(), 0);
        g.set_cold_content_budget(Some(1));
        assert_eq!(g.cold_ids().filter(|id| g.is_at_floor(id)).count(), parked);
    }

    // ── structural-centrality scoring term (the Gemini-reflection idea) ───────

    #[test]
    fn centrality_is_off_by_default_and_elided() {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["hub", "a", "b"] {
            g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Symbol);
        }
        g.add_edge("a".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        // With centrality off (the default), the hub (in-degree 2) scores exactly
        // like a leaf (in-degree 0) — the term contributes nothing.
        let hub = g.compute_node_score(g.node(&NodeId("hub".into())).unwrap());
        let a = g.compute_node_score(g.node(&NodeId("a".into())).unwrap());
        assert_eq!(hub, a, "centrality off ⇒ in-degree does not move the score");
        // And the default weights serialize without the new field.
        let json = serde_json::to_string(&g.scoring_weights).unwrap();
        assert!(
            !json.contains("w_centrality"),
            "an off (0.0) centrality weight is elided: {json}"
        );
    }

    #[test]
    fn centrality_boosts_a_hub_over_a_leaf_when_enabled() {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["hub", "a", "b", "leaf"] {
            g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Symbol);
        }
        g.add_edge("a".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        g.add_edge("b".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        g.set_scoring_weights(ScoringWeights {
            w_centrality: 0.3,
            ..ScoringWeights::default()
        });
        let hub = g.compute_node_score(g.node(&NodeId("hub".into())).unwrap());
        let leaf = g.compute_node_score(g.node(&NodeId("leaf".into())).unwrap());
        assert!(
            hub > leaf,
            "a hub (in-degree 2) outscores a leaf (in-degree 0): {hub} vs {leaf}"
        );
        // The in-degree cache must track edge changes (keyed on edges.len()): a new
        // incoming edge raises the hub's score further.
        let before = hub;
        g.upsert_node("c".into(), "c".into(), "x".into(), NodeType::Symbol);
        g.add_edge("c".into(), "hub".into(), 1.0, EdgeType::DependsOn);
        let after = g.compute_node_score(g.node(&NodeId("hub".into())).unwrap());
        assert!(
            after > before,
            "a new dependant raises centrality: {after} > {before}"
        );
    }

    #[test]
    fn test_find_cycles_detects_loop() {
        let mut graph = MemoryGraph::default();
        for id in ["a", "b", "c", "d"] {
            graph.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        // a -> b -> c -> a is a cycle; d is acyclic.
        graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("c".into(), "a".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("c".into(), "d".into(), 1.0, EdgeType::DependsOn);

        let cycles = graph.find_cycles();
        assert!(!cycles.is_empty(), "must detect the a->b->c->a cycle");
        assert!(cycles[0].len() >= 3);
    }

    #[test]
    fn test_graph_diff() {
        let mut a = MemoryGraph::default();
        for id in ["x", "y", "z"] {
            a.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        a.add_edge("x".into(), "y".into(), 1.0, EdgeType::DependsOn);

        let mut b = MemoryGraph::default();
        for id in ["y", "z", "w"] {
            b.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        b.add_edge("y".into(), "z".into(), 1.0, EdgeType::DependsOn);

        let d = a.diff(&b);
        assert_eq!(d.nodes_added, vec![NodeId("w".into())]);
        assert_eq!(d.nodes_removed, vec![NodeId("x".into())]);
        assert_eq!(d.common_nodes, 2); // y, z
        assert_eq!(d.edges_added, 1); // y->z
        assert_eq!(d.edges_removed, 1); // x->y
    }

    #[test]
    fn test_to_dot_and_orphans() {
        let mut graph = MemoryGraph::default();
        graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);
        graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Symbol);
        graph.upsert_node("lonely".into(), "L".into(), "".into(), NodeType::Unknown);
        graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::Contains);

        let dot = graph.to_dot();
        assert!(dot.starts_with("digraph ccos {"));
        assert!(dot.contains("\"a\" -> \"b\""));

        let orphans = graph.orphan_nodes();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].0, "lonely");
    }

    #[test]
    fn test_find_cycles_none_when_acyclic() {
        let mut graph = MemoryGraph::default();
        for id in ["a", "b", "c"] {
            graph.upsert_node(id.into(), id.into(), "".into(), NodeType::Module);
        }
        graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn);
        graph.add_edge("b".into(), "c".into(), 1.0, EdgeType::DependsOn);
        assert!(graph.find_cycles().is_empty(), "DAG must have no cycles");
    }

    #[test]
    fn test_context_selection_returns_sorted() {
        let mut graph = MemoryGraph::default();
        for i in 0..5 {
            let mut node = GraphNode {
                id: NodeId(format!("n{}", i)),
                label: format!("N{}", i),
                content: "test".into(),
                node_type: NodeType::Unknown,
                base_importance: (i as f64) * 0.1,
                failure_relevance: 0.0,
                recency: 0.5,
                access_count: 1,
                created_at: 0,
                last_accessed: 0,
                state: NodeState::Stable,
                trust: full_trust(),
            };
            node.recency = (5 - i) as f64 * 0.1;
            graph.nodes.insert(node.id.clone(), node);
        }
        let selected = graph.select_context_window(1024);
        assert!(!selected.is_empty());
    }

    // ── slice 5: COLD deep-spill (lossless full-entry archive; resident-bounded) ──

    #[test]
    fn pack_adj_round_trips_ids() {
        // Empty, single, multi, and multibyte ids all survive the length-prefixed
        // packing (slice 5c Lever 1).
        for ids in [
            vec![],
            vec!["a"],
            vec!["file:src/x.rs", "sym:src/x.rs:fn_é"],
            vec!["", "x", "longer-id-with-dashes-0123456789"],
        ] {
            let nodes: Vec<NodeId> = ids.iter().map(|s| NodeId(s.to_string())).collect();
            let packed = pack_adj(&nodes);
            assert_eq!(
                unpack_adj(&packed).collect::<Vec<_>>(),
                ids,
                "round-trip {ids:?}"
            );
        }
        // A truncated buffer stops cleanly rather than panicking.
        assert_eq!(
            unpack_adj(&[5, 0, 0, 0, b'a']).collect::<Vec<_>>(),
            Vec::<&str>::new()
        );
    }

    /// Five nodes — a chain a→b→c→d plus a hub `h` linked to all four — demoted to
    /// COLD with a spill store attached but no content/deep pressure yet. The caller
    /// sets a resident budget to drive deep-spill.
    fn cold_region(dir: &std::path::Path) -> MemoryGraph {
        let mut g = MemoryGraph::new(0.2, 100);
        for id in ["a", "b", "c", "d", "h"] {
            g.upsert_node(
                id.into(),
                format!("label::{id}"),
                format!("content of {id} {}", "x".repeat(40)),
                NodeType::Symbol,
            );
        }
        g.add_edge("a".into(), "b".into(), 0.6, EdgeType::DependsOn);
        g.add_edge("b".into(), "c".into(), 0.6, EdgeType::DependsOn);
        g.add_edge("c".into(), "d".into(), 0.6, EdgeType::DependsOn);
        for t in ["a", "b", "c", "d"] {
            g.add_edge("h".into(), t.into(), 0.5, EdgeType::Contains);
        }
        g.attach_cold_spill(dir, usize::MAX).unwrap(); // store attached; no inline-budget spill
        g.max_in_memory_nodes = 0;
        g.enforce_paging(); // demote everything to COLD
        g
    }

    /// Ground-truth cold neighbours: scan the cold tier's actual adjacency (resident
    /// full-entry edges + deep husk ids) directly — what `cold_neighbours` returned
    /// before the reverse index. The `radj`-backed implementation must match this.
    fn reference_cold_neighbours(g: &MemoryGraph, id: &NodeId) -> Vec<NodeId> {
        let mut out: BTreeSet<NodeId> = BTreeSet::new();
        let mut consider = |a: &NodeId, b: &NodeId| {
            let other = if a == id {
                Some(b.clone())
            } else if b == id {
                Some(a.clone())
            } else {
                None
            };
            if let Some(o) = other {
                if o != *id && (g.cold.contains_key(&o) || g.deep_contains(&o)) {
                    out.insert(o);
                }
            }
        };
        for c in g.cold.values() {
            for e in &c.edges {
                consider(&e.source, &e.target);
            }
        }
        for (hid, h) in g.deep_entries() {
            for o in unpack_adj(&h.adj) {
                consider(&hid, &NodeId(o.to_owned()));
            }
        }
        out.into_iter().collect()
    }

    #[test]
    fn radj_cold_neighbours_match_ground_truth_under_random_ops() {
        // Lever 2 brick 8: the reverse-adjacency index must agree with a direct scan
        // of the cold tier after any sequence of demote / page-in / remove.
        let dir = spill_temp_dir("radj_equiv");
        let mut g = MemoryGraph::new(0.2, 6);
        g.attach_cold_spill(&dir, usize::MAX).unwrap();
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for step in 0..500 {
            match rng() % 6 {
                0 | 1 => {
                    let a = format!("n{}", rng() % 12);
                    g.upsert_node(
                        a.clone().into(),
                        format!("l{a}"),
                        format!("c{a} {}", "x".repeat((rng() % 30) as usize)),
                        NodeType::Symbol,
                    );
                    let b = format!("n{}", rng() % 12);
                    if a != b {
                        g.page_in(&a.clone().into());
                        g.page_in(&b.clone().into());
                        g.add_edge(a.into(), b.into(), 0.5, EdgeType::DependsOn);
                    }
                }
                2 => {
                    g.max_in_memory_nodes = (rng() % 5) as usize;
                    g.enforce_paging();
                }
                3 => g.set_cold_resident_budget(Some((rng() % 256) as usize)),
                4 => {
                    g.page_in(&format!("n{}", rng() % 12).into());
                }
                _ => g.remove_node(&format!("n{}", rng() % 12).into()),
            }
            if step % 5 == 0 {
                for id in g.cold_ids().collect::<Vec<_>>() {
                    assert_eq!(
                        g.cold_neighbours(&id),
                        reference_cold_neighbours(&g, &id),
                        "radj mismatch for {} at step {step}",
                        id.0,
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_tier_survives_a_simulated_crash() {
        // Lever 2 brick 9: deep-spill + flush_cold_tier makes the husk and reverse-
        // adjacency indices durable, so a *fresh* graph re-attaching the same directory
        // recovers the whole deep tier — content, neighbours and all.
        let dir = spill_temp_dir("crash");
        let expected: Vec<(NodeId, Vec<NodeId>)>;
        {
            // First "process": build a cold region, deep-spill it, flush, drop the graph.
            let mut g = cold_region(&dir);
            g.set_cold_resident_budget(Some(0)); // deep-spill every entry
            expected = g
                .cold_ids()
                .collect::<Vec<_>>()
                .into_iter()
                .map(|id| {
                    let n = g.cold_neighbours(&id);
                    (id, n)
                })
                .collect();
            g.flush_cold_tier().unwrap();
            // `g` drops here: the in-RAM write buffers vanish, the directory persists.
        }

        // Second "process": a fresh graph re-attaches the same directory.
        let mut g2 = MemoryGraph::new(0.2, 100);
        g2.attach_cold_spill(&dir, usize::MAX).unwrap();
        assert!(!expected.is_empty(), "expected deep entries");

        // Adjacency and membership recover — checked before any page-in changes the
        // cold set.
        for (id, neighbours) in &expected {
            assert!(g2.is_deep_spilled(id), "{} recovered as deep-spilled", id.0);
            assert_eq!(
                &g2.cold_neighbours(id),
                neighbours,
                "reverse adjacency recovered for {}",
                id.0
            );
        }
        // Content recovers losslessly through the disk trip.
        for (id, _) in &expected {
            assert!(g2.page_in(id), "{} pages back in after the crash", id.0);
            let node = g2.node(id).expect("resident after page-in");
            assert_eq!(
                node.content,
                format!("content of {} {}", id.0, "x".repeat(40)),
                "content recovered losslessly for {}",
                id.0
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_tier_lives_in_the_husk_store() {
        // Lever 2 brick 6: the deep tier IS the on-disk index — every deep-spilled
        // entry is reachable via `deep_get`, counted by `deep_count`, and page-in
        // removes it from the store.
        let dir = spill_temp_dir("husk_auth");
        let mut g = cold_region(&dir);
        let total = g.cold_count();
        g.set_cold_resident_budget(Some(0)); // deep-spill every cold entry

        assert_eq!(
            g.cold_deep_spilled_count(),
            total,
            "all entries deep-spilled"
        );
        for id in g.cold_ids().collect::<Vec<_>>() {
            assert!(g.is_deep_spilled(&id), "{} is in the deep tier", id.0);
            assert!(g.deep_get(&id).is_some(), "husk readable for {}", id.0);
        }

        // Page one back in: it leaves the on-disk tier entirely.
        let some_id = g.cold_ids().next().unwrap();
        g.max_in_memory_nodes = 100;
        assert!(g.page_in(&some_id), "paged back in");
        assert!(!g.is_deep_spilled(&some_id), "left the deep tier");
        assert!(
            g.deep_get(&some_id).is_none(),
            "husk removed from the store"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_round_trips_through_page_in() {
        // `a` stays resident, `b` is demoted then deep-spilled. The a–b edge is
        // archived under `b`, so paging `b` back (with `a` resident) re-links it —
        // proving label, content and the full edge record all survive the disk trip.
        let dir = spill_temp_dir("deep_rt");
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node(
            "a".into(),
            "label::a".into(),
            "alpha body".into(),
            NodeType::Symbol,
        );
        g.upsert_node(
            "b".into(),
            "label::b".into(),
            "beta body padded ".repeat(8),
            NodeType::Symbol,
        );
        // Make `a` the warmer node so `b` is the demotion victim.
        {
            let a = g.nodes.get_mut(&NodeId("a".into())).unwrap();
            a.recency = 1.0;
            a.base_importance = 1.0;
        }
        g.add_edge("a".into(), "b".into(), 0.7, EdgeType::DependsOn);
        g.max_in_memory_nodes = 1;
        g.enforce_paging();
        assert!(g.is_cold(&NodeId("b".into())), "b demoted to COLD");

        g.attach_cold_spill(&dir, usize::MAX).unwrap();
        g.set_cold_resident_budget(Some(0)); // force deep-spill of every cold entry
        assert!(g.is_deep_spilled(&NodeId("b".into())), "b deep-spilled");
        {
            // The full ColdNode is gone from RAM — only a compact husk remains.
            assert!(
                !g.cold.contains_key(&NodeId("b".into())),
                "no full ColdNode kept for a deep entry"
            );
            let h = g.deep_get(&NodeId("b".into())).unwrap();
            assert_eq!(
                unpack_adj(&h.adj).collect::<Vec<_>>(),
                vec!["a"],
                "only the neighbour id kept (packed)"
            );
            assert!(
                h.body.len > 0,
                "whole node archived to a non-empty body blob"
            );
        }

        g.max_in_memory_nodes = 2;
        assert!(g.page_in(&NodeId("b".into())), "faults the deep body back");
        let b = g.node(&NodeId("b".into())).unwrap();
        assert_eq!(b.label, "label::b", "label restored");
        assert_eq!(b.content, "beta body padded ".repeat(8), "content restored");
        assert!(
            g.edges.iter().any(|e| e.source == NodeId("a".into())
                && e.target == NodeId("b".into())
                && e.edge_type == EdgeType::DependsOn
                && (e.weight - 0.7).abs() < 1e-9),
            "the full edge record (weight + type) round-tripped"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_preserves_cold_neighbours() {
        let dir = spill_temp_dir("deep_nbr");
        let mut g = cold_region(&dir);
        let ids: Vec<NodeId> = ["a", "b", "c", "d", "h"]
            .iter()
            .map(|s| NodeId(s.to_string()))
            .collect();
        let before: Vec<Vec<NodeId>> = ids.iter().map(|id| g.cold_neighbours(id)).collect();
        g.set_cold_resident_budget(Some(0)); // deep-spill the whole tier
                                             // Every entry now lives only as a husk (the compact form beats the full
                                             // ColdNode for any entry), so cold_neighbours answers purely from resident
                                             // `adj` ids — and must reproduce the original adjacency exactly.
        assert_eq!(
            g.cold_deep_spilled_count(),
            g.cold_count(),
            "all entries deep-spilled"
        );
        let after: Vec<Vec<NodeId>> = ids.iter().map(|id| g.cold_neighbours(id)).collect();
        assert_eq!(
            before, after,
            "the resident neighbour ids reproduce the cold adjacency exactly"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_off_by_default_leaves_serialization_unchanged() {
        let dir = spill_temp_dir("deep_off");
        let g = cold_region(&dir);
        assert_eq!(g.cold_deep_spilled_count(), 0, "no budget ⇒ no deep-spill");
        // The deep-husk map is empty and elided, so the JSON is byte-identical to a
        // graph that never knew about deep-spill — and still round-trips.
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("cold_deep"),
            "no deep-husk map on the default path"
        );
        let _back: MemoryGraph = serde_json::from_str(&json).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_decisions_are_deterministic() {
        fn deep_set(dir: &std::path::Path) -> Vec<NodeId> {
            let mut g = cold_region(dir);
            g.set_cold_resident_budget(Some(g.cold_resident_bytes() / 3));
            let mut out: Vec<NodeId> = g.cold_ids().filter(|id| g.is_deep_spilled(id)).collect();
            out.sort();
            out
        }
        let d1 = spill_temp_dir("deep_det1");
        let d2 = spill_temp_dir("deep_det2");
        let a = deep_set(&d1);
        let b = deep_set(&d2);
        assert_eq!(a, b, "identical histories deep-spill the identical set");
        assert!(
            !a.is_empty(),
            "the budget should have forced some deep-spills"
        );
        std::fs::remove_dir_all(&d1).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn deep_spill_reduces_resident_bytes_without_dropping_nodes() {
        let dir = spill_temp_dir("deep_res");
        let mut g = cold_region(&dir);
        let before = g.cold_resident_bytes();
        let cold_before = g.cold_count();
        g.set_cold_resident_budget(Some(before / 3));
        assert!(g.cold_deep_spilled_count() > 0, "budget forced deep-spills");
        assert!(
            g.cold_resident_bytes() < before,
            "resident metadata shrank ({} → {})",
            before,
            g.cold_resident_bytes()
        );
        assert_eq!(
            g.cold_count(),
            cold_before,
            "non-destructive — no node dropped"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_compacts_even_a_tiny_isolated_entry() {
        // The whole point of the compact husk (slice 5b): the husk is smaller than a
        // full ColdNode struct, so *every* entry shrinks — even a 1-byte, edge-less,
        // label-less one that slice 5 had to leave at the floor. The floor is gone.
        let dir = spill_temp_dir("deep_floor");
        let mut g = MemoryGraph::new(0.2, 100);
        g.upsert_node(
            "big".into(),
            "label-big".into(),
            "B".repeat(200),
            NodeType::Symbol,
        );
        g.upsert_node("tiny".into(), String::new(), "x".into(), NodeType::Symbol);
        g.attach_cold_spill(&dir, usize::MAX).unwrap();
        g.max_in_memory_nodes = 0;
        g.enforce_paging();
        let before = g.cold_resident_bytes();
        g.set_cold_resident_budget(Some(0));
        assert!(
            g.is_deep_spilled(&NodeId("big".into())),
            "the large entry shrinks"
        );
        assert!(
            g.is_deep_spilled(&NodeId("tiny".into())),
            "and so does the tiny isolated one — no per-entry struct floor any more"
        );
        assert!(
            g.cold_resident_bytes() < before,
            "resident dropped ({} → {})",
            before,
            g.cold_resident_bytes()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deep_spill_gc_reclaims_body_blob_on_remove() {
        let dir = spill_temp_dir("deep_gc");
        let mut g = cold_region(&dir);
        g.set_cold_resident_budget(Some(0));
        let victim = g
            .cold_ids()
            .find(|id| g.is_deep_spilled(id))
            .expect("at least one entry deep-spilled");
        let files_before = std::fs::read_dir(&dir).unwrap().count();
        // Removing a deep-spilled node reclaims its body (and content) blobs when no
        // other entry shares them.
        g.remove_node(&victim);
        let files_after = std::fs::read_dir(&dir).unwrap().count();
        assert!(
            files_after < files_before,
            "removing a deep node reclaimed its blob(s) ({files_before} → {files_after})"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn page_in_deep_spilled_with_missing_body_is_a_cold_miss() {
        let dir = spill_temp_dir("deep_miss");
        let mut g = cold_region(&dir);
        g.set_cold_resident_budget(Some(0));
        let bid = NodeId("b".into());
        let body_hash = g.deep_get(&bid).unwrap().body.hash;
        // Delete the on-disk body blob (its filename is the hex of the hash): page_in
        // must refuse — no silent half-restore.
        std::fs::remove_file(dir.join(hex32(&body_hash))).unwrap();
        g.max_in_memory_nodes = 100;
        assert!(!g.page_in(&bid), "missing deep body ⇒ cold-miss");
        assert!(g.is_cold(&bid), "node stays cold, not half-restored");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn paging_a_node_in_never_destroys_an_edge_to_a_still_cold_node() {
        // The COLD tier is documented as non-destructive ("the node and its links
        // are kept, just no longer resident"). `demote` archives an edge on ONE
        // side only, so when that side is paged back in while the far end is still
        // cold, the edge used to be dropped from the only place it existed — an
        // unrecoverable loss of causal structure that `verify` cannot see, because
        // it attests the hash chain and never that the graph derives from it.
        //
        // Round-trip invariant: demote everything, page everything back, and the
        // edge set must be exactly what it was.
        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["a", "b", "c", "d"] {
            g.upsert_node(
                NodeId(n.into()),
                n.into(),
                format!("fn {n}() {{}}"),
                NodeType::Symbol,
            );
        }
        // A path plus a chord, so several edges have both endpoints demoted.
        g.add_edge(
            NodeId("a".into()),
            NodeId("b".into()),
            1.0,
            EdgeType::DependsOn,
        );
        g.add_edge(NodeId("b".into()), NodeId("c".into()), 1.0, EdgeType::Calls);
        g.add_edge(
            NodeId("c".into()),
            NodeId("d".into()),
            1.0,
            EdgeType::DependsOn,
        );
        g.add_edge(NodeId("a".into()), NodeId("d".into()), 1.0, EdgeType::Calls);

        let fingerprint = |g: &MemoryGraph| {
            let mut v: Vec<String> = g
                .edges
                .iter()
                .map(|e| format!("{}->{}:{:?}", e.source.0, e.target.0, e.edge_type))
                .collect();
            v.sort();
            v
        };
        let before = fingerprint(&g);
        assert_eq!(before.len(), 4, "fixture starts with four edges");

        // Squeeze every node into the COLD tier, then lift the cap and pull them
        // all back one at a time — the order that used to lose the cold<->cold
        // edges, since each page-in saw its neighbour still cold.
        g.max_in_memory_nodes = 1;
        g.enforce_paging();
        assert!(g.cold_count() > 0, "fixture must actually demote");

        g.max_in_memory_nodes = usize::MAX;
        let cold: Vec<NodeId> = g.cold_ids().collect();
        for id in cold {
            g.page_in(&id);
        }

        assert_eq!(g.cold_count(), 0, "everything is resident again");
        assert_eq!(
            fingerprint(&g),
            before,
            "a demote/page-in round trip must preserve every edge"
        );
    }

    #[test]
    fn paging_in_never_destroys_an_edge_to_a_deep_spilled_node() {
        // Variante deep-spill de `paging_a_node_in_never_destroys_an_edge_to_a_
        // still_cold_node`. Une arête archivée l'est d'UN seul côté ; si son autre
        // extrémité a été deep-spillée vers l'index husk, elle n'est plus ni
        // résidente ni dans `cold`, et la rendre à personne revient à la détruire.
        // Le husk ne porte que l'adjacence : les arêtes de l'entrée vivent dans le
        // blob de son corps.
        let dir = std::env::temp_dir().join(format!("ccos-deepedge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut g = MemoryGraph::new(0.2, usize::MAX);
        for n in ["a", "b", "c", "d"] {
            g.upsert_node(
                NodeId(n.into()),
                n.into(),
                format!("pub fn {n}() {{ /* corps assez long pour peser */ }}"),
                NodeType::Symbol,
            );
        }
        g.add_edge(
            NodeId("a".into()),
            NodeId("b".into()),
            1.0,
            EdgeType::DependsOn,
        );
        g.add_edge(NodeId("b".into()), NodeId("c".into()), 1.0, EdgeType::Calls);
        g.add_edge(
            NodeId("c".into()),
            NodeId("d".into()),
            1.0,
            EdgeType::DependsOn,
        );
        g.add_edge(NodeId("a".into()), NodeId("d".into()), 1.0, EdgeType::Calls);

        let fingerprint = |g: &MemoryGraph| {
            let mut v: Vec<String> = g
                .edges
                .iter()
                .map(|e| format!("{}->{}:{:?}", e.source.0, e.target.0, e.edge_type))
                .collect();
            v.sort();
            v
        };
        let before = fingerprint(&g);
        assert_eq!(before.len(), 4);

        g.attach_cold_spill(&dir, 0).expect("attach");
        // Tout démoter, puis pousser le tier COLD dans l'index husk : c'est la
        // seule différence avec le test COLD résident, et elle suffisait à perdre
        // des arêtes.
        g.max_in_memory_nodes = 1;
        g.enforce_paging();
        g.set_cold_resident_budget(Some(0));
        g.flush_cold_tier().expect("flush");

        g.max_in_memory_nodes = usize::MAX;
        for n in ["a", "b", "c", "d"] {
            g.page_in(&NodeId(n.into()));
        }

        assert_eq!(
            fingerprint(&g),
            before,
            "un aller-retour démotion/deep-spill/page-in doit préserver chaque arête"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_spill_dictionary_appears_only_once_the_volume_justifies_it() {
        // Le seuil est une condition de rentabilité mesurée : sous ~250 Ko de
        // contenu froid, un dictionnaire de 32 Ko coûte plus qu'il ne rapporte.
        // Ce test épingle les deux côtés du seuil.
        let dir = std::env::temp_dir().join(format!("ccos-dict-vol-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Petit volume : pas de dictionnaire, et les blobs restent lisibles.
        let mut g = MemoryGraph::new(0.2, 2);
        for i in 0..6 {
            g.upsert_node(
                NodeId(format!("n{i}")),
                format!("n{i}"),
                format!("pub fn small_{i}() {{}}\n"),
                NodeType::Symbol,
            );
        }
        g.attach_cold_spill(&dir, 0).expect("attach");
        g.enforce_paging();
        assert!(
            g.spill.as_ref().unwrap().store.dict().is_none(),
            "sous le seuil, aucun dictionnaire ne doit être constitué"
        );
        assert!(!dir.join(DICT_FILE).exists());

        // Gros volume : le dictionnaire est constitué et persisté.
        let dir2 = std::env::temp_dir().join(format!("ccos-dict-vol2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir2);
        let mut g2 = MemoryGraph::new(0.2, 2);
        let body = "pub fn helper(input: &str) -> Result<usize> { Ok(input.len()) }\n".repeat(40);
        for i in 0..120 {
            g2.upsert_node(
                NodeId(format!("m{i}")),
                format!("m{i}"),
                format!("// module {i}\n{body}"),
                NodeType::Symbol,
            );
        }
        g2.attach_cold_spill(&dir2, 0).expect("attach");
        g2.enforce_paging();
        assert!(
            g2.spill.as_ref().unwrap().store.dict().is_some(),
            "au-delà du seuil, le dictionnaire doit être constitué"
        );
        assert!(dir2.join(DICT_FILE).exists() && dir2.join(DICT_HASH_FILE).exists());

        // Et le contenu revient EXACTEMENT : c'est tout l'enjeu du codec.
        let ids: Vec<NodeId> = g2.cold_ids().collect();
        assert!(!ids.is_empty());
        for id in ids {
            let before = format!("// module {}\n{}", id.0.trim_start_matches('m'), body);
            assert!(g2.page_in(&id), "page-in de {id:?}");
            assert_eq!(
                g2.node(&id).expect("résident").content,
                before,
                "le contenu doit revenir à l'octet près"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn a_replaced_spill_dictionary_is_refused_rather_than_trusted() {
        // Remplacer le dictionnaire rendrait illisibles les blobs déjà encodés.
        // L'incompatibilité doit être détectée : on préfère perdre le tier COLD
        // (cold-miss) que rendre un contenu faux.
        let dir = std::env::temp_dir().join(format!("ccos-dict-tamper-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(DICT_FILE),
            b"un dictionnaire quelconque assez long pour compter",
        )
        .unwrap();
        std::fs::write(dir.join(DICT_HASH_FILE), "empreinte_qui_ne_correspond_pas").unwrap();

        let store = ColdSpill::new(&dir).expect("open");
        assert!(
            store.dict().is_none(),
            "un dictionnaire dont l'empreinte ne correspond pas doit être ignoré"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
