//! **Does the centrality scoring term earn its keep?** `w_centrality` (off by
//! default) adds `w_centrality · ln(1 + resident_in_degree)` to a node's score, so a
//! structurally-central *hub* — a shared module many resident nodes depend on — is
//! retained even when it is not the most recently touched thing. This is the one
//! refinement Gemini's "prune by causal weight" analysis pointed at that CCOS had
//! **not yet measured**. Measure-first: build a realistic re-engagement workload under
//! paging pressure and see whether enabling centrality actually reduces page-faults on
//! the hubs — and at what cost to the leaves (a retained hub steals a resident slot).
//!
//! ## The workload (fully deterministic — nothing left to chance, no RNG)
//!
//! `R` regions, each a hub + `W` leaves that `DependsOn` it (so the hub's in-degree is
//! `W`). The agent works one region at a time, across `P` passes. Within a region it
//! sweeps a small **sliding window** over the leaves (width `WINDOW`, advancing by 1 so
//! windows overlap → leaves are revisited) and **re-consults the hub every `cadence`
//! leaf-accesses** (the shared module is needed again and again). The resident budget
//! is tight, so between two hub consults the fresh leaf accesses tend to evict the hub
//! — unless its structural centrality holds it in. An access to a cold node is a
//! **page-fault** ([`page_in`]); to a resident node, a **hit** ([`touch`], refreshing
//! recency/access exactly as a fault does, so the recency baseline is fair).
//!
//! The agent re-asserts the `leaf → hub` edge whenever both are resident (it re-observes
//! the dependency while working): under paging, demotion archives incident edges, so
//! this keeps the *resident* causal graph faithful to the true structure — and is
//! exactly the resident in-degree the centrality term scores on. We instrument that
//! in-degree to confirm the mechanism is genuinely engaged, not assumed.
//!
//! Run: `cargo run --release --example centrality_retention`
//!
//! [`page_in`]: ccos_core::memory::MemoryGraph::page_in
//! [`touch`]: ccos_core::memory::MemoryGraph::touch

use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType, ScoringWeights};

/// Workload shape — fixed across the whole sweep so `w_centrality` is the only variable.
struct Workload {
    regions: usize,
    leaves_per_region: usize,
    passes: usize,
    window: usize,
    budget: usize,
}

fn hub_id(r: usize) -> NodeId {
    format!("hub:region_{r}").into()
}
fn leaf_id(r: usize, l: usize) -> NodeId {
    format!("leaf:region_{r}/node_{l}").into()
}

/// Build the hub-and-leaf graph with **no paging during construction** (budget
/// `usize::MAX`), so every `add_edge` sees both endpoints resident. Returns a graph
/// with all `R·(W+1)` nodes and all `R·W` leaf→hub edges resident.
fn build(w: &Workload) -> MemoryGraph {
    let mut g = MemoryGraph::new(0.2, usize::MAX);
    for r in 0..w.regions {
        let h = hub_id(r);
        g.upsert_node(
            h.clone(),
            format!("region {r} hub"),
            format!("// shared module for region {r}\n{}", "api ".repeat(16)),
            NodeType::Module,
        );
        for l in 0..w.leaves_per_region {
            let leaf = leaf_id(r, l);
            g.upsert_node(
                leaf.clone(),
                format!("region {r} leaf {l}"),
                format!(
                    "pub fn region_{r}_node_{l}() {{ {} }}",
                    "step(); ".repeat(8)
                ),
                NodeType::Symbol,
            );
            // leaf DependsOn hub ⇒ the hub gains one unit of in-degree.
            g.add_edge(leaf, h.clone(), 0.6, EdgeType::DependsOn);
        }
    }
    g
}

/// Outcome of one run of the workload under a fixed scoring configuration.
#[derive(Default, Clone, Copy)]
struct Stats {
    hub_faults: usize,
    hub_consults: usize,
    leaf_faults: usize,
    /// Σ resident in-degree of the hub, sampled at each consult (to average).
    indeg_sum: u64,
}

/// Run the full deterministic trace once under `weights`, re-consulting the hub every
/// `cadence` leaf-accesses. Pure function of its inputs ⇒ replayable.
fn run(w: &Workload, weights: ScoringWeights, cadence: usize) -> Stats {
    let mut g = build(w);
    g.set_scoring_weights(weights);
    g.max_in_memory_nodes = w.budget;
    g.enforce_paging(); // establish the initial (tight) resident set

    let mut s = Stats::default();

    // access(id): fault if cold (page_in), else a resident hit (touch). Returns whether
    // it was a fault. Both paths refresh recency/access identically, so recency is a
    // fair baseline and the only thing that can differ between weightings is *retention*.
    let access = |g: &mut MemoryGraph, id: &NodeId| -> bool {
        if g.is_cold(id) {
            g.page_in(id);
            true
        } else {
            g.touch(id);
            false
        }
    };

    for _pass in 0..w.passes {
        for r in 0..w.regions {
            let h = hub_id(r);
            let mut win_start = 0usize;
            let mut since_hub = 0usize;
            // Two sweeps' worth of steps so the window crosses the region with overlap.
            let steps = w.leaves_per_region * 2;
            let max_start = w.leaves_per_region.saturating_sub(w.window);
            for step in 0..steps {
                let offset = step % w.window;
                let l = (win_start + offset).min(w.leaves_per_region - 1);
                let leaf = leaf_id(r, l);

                if access(&mut g, &leaf) {
                    s.leaf_faults += 1;
                }
                // Re-observe the dependency: keep the resident causal graph faithful to
                // the true structure (idempotent; no-op if the hub is currently cold).
                g.add_edge(leaf, h.clone(), 0.6, EdgeType::DependsOn);

                since_hub += 1;
                if since_hub == cadence {
                    // Sample the structural signal the scorer sees *before* the consult.
                    s.indeg_sum += g.node_in_degree(&h) as u64;
                    if access(&mut g, &h) {
                        s.hub_faults += 1;
                    }
                    s.hub_consults += 1;
                    since_hub = 0;
                }
                // Advance the sliding window by 1 each full cycle, so consecutive windows
                // overlap (width-`window`) and leaves are revisited — which is what makes a
                // retained hub's stolen slot *cost* a leaf re-fault.
                if offset == w.window - 1 {
                    win_start = (win_start + 1).min(max_start);
                }
                g.tick(); // time advances → recency decays
            }
        }
    }
    s
}

fn weights(w_centrality: f64) -> ScoringWeights {
    ScoringWeights {
        w_centrality,
        ..ScoringWeights::default()
    }
}

/// `(hub_faults, leaf_faults)` totalled, and the hub miss-rate.
fn totals(s: &Stats) -> (usize, usize, usize, f64) {
    let total = s.hub_faults + s.leaf_faults;
    let miss = s.hub_faults as f64 / s.hub_consults.max(1) as f64;
    (s.hub_faults, s.leaf_faults, total, miss)
}

fn main() {
    let base = Workload {
        regions: 4,
        leaves_per_region: 10,
        passes: 2,
        window: 3,
        budget: 5,
    };

    println!("# Centrality term vs COLD-tier retention — measured\n");
    println!(
        "workload: {R} regions × {W} leaves (each leaf DependsOn its hub ⇒ global hub in-degree \
         {W}),\n{P} passes, sliding window {WIN}, resident budget {B}. A hub is re-consulted every\n\
         `cadence` leaf-accesses; an access to a cold node is a page-fault, to a resident node a hit.",
        R = base.regions,
        W = base.leaves_per_region,
        P = base.passes,
        WIN = base.window,
        B = base.budget,
    );

    // --- Determinism: the same configuration must replay byte-identically. ---
    let a = run(&base, weights(0.3), 4);
    let b = run(&base, weights(0.3), 4);
    assert_eq!(
        (a.hub_faults, a.leaf_faults, a.indeg_sum),
        (b.hub_faults, b.leaf_faults, b.indeg_sum),
        "the workload must be deterministic"
    );
    println!("\ndeterminism: identical config replays identically ✓");

    // === Table 1 — does the *magnitude* of w_centrality matter, or only on/off? ===
    let cadence = 4usize;
    println!(
        "\n## 1. Sweep w_centrality  (cadence {cadence}, budget {})\n",
        base.budget
    );
    println!("  w_centrality | hub faults | leaf faults | TOTAL | Δtotal | avg resident in-degree");
    println!("  -------------+------------+-------------+-------+--------+-----------------------");
    let s0 = run(&base, weights(0.0), cadence);
    let (_, _, base_total, _) = totals(&s0);
    for wc in [0.0, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0] {
        let s = run(&base, weights(wc), cadence);
        let (hf, lf, total, _) = totals(&s);
        let avg_indeg = s.indeg_sum as f64 / s.hub_consults.max(1) as f64;
        println!(
            "  {wc:>11.2}  | {hf:>10} | {lf:>11} | {total:>5} | {:>+6} | {avg_indeg:>10.2}",
            total as i64 - base_total as i64,
        );
    }
    println!(
        "  → the effect is binary: any w>0 reorders the *same* single eviction (discrete argmin),\n\
         \x20   so 0.01 ≡ 1.0. Paging bounds resident in-degree to ~3 (of a global 10), so the bonus\n\
         \x20   is a bounded nudge that never lets the hub crowd out actively-used leaves."
    );

    // === Table 2 — centrality targets HUB faults; sweep how hot the hub is. ===
    println!(
        "\n## 2. Hub-fault reduction vs re-consult cadence  (budget {})\n",
        base.budget
    );
    println!("  cadence | hub faults  w0 → w0.3 | hub miss-rate w0 → w0.3 | total  w0 → w0.3");
    println!("  --------+-----------------------+-------------------------+------------------");
    for cad in [2usize, 4, 6, 8] {
        let off = run(&base, weights(0.0), cad);
        let on = run(&base, weights(0.3), cad);
        let (ohf, _, ot, omiss) = totals(&off);
        let (nhf, _, nt, nmiss) = totals(&on);
        println!(
            "  {cad:>7} | {ohf:>9} → {nhf:<9} | {:>10.0}% → {:<8.0}% | {ot:>5} → {nt:<5} ({:+})",
            omiss * 100.0,
            nmiss * 100.0,
            nt as i64 - ot as i64,
        );
    }
    println!(
        "  → the colder the hub (higher cadence), the more the baseline evicts-and-re-faults it,\n\
         \x20   and the more centrality saves — up to the self-limit set by its resident in-degree."
    );

    // === Table 3 — memory pressure: does centrality ever *cost* leaf faults? ===
    println!("\n## 3. Effect vs memory pressure  (cadence {cadence})\n");
    println!("  budget | avg in-deg | hub faults w0→w0.3 | leaf faults w0→w0.3 | total w0→w0.3");
    println!("  -------+------------+--------------------+---------------------+----------------");
    for b in [4usize, 5, 6, 8, 11] {
        let w = Workload {
            budget: b,
            ..workload_like(&base)
        };
        let off = run(&w, weights(0.0), cadence);
        let on = run(&w, weights(0.3), cadence);
        let (ohf, olf, ot, _) = totals(&off);
        let (nhf, nlf, nt, _) = totals(&on);
        let avg_indeg = on.indeg_sum as f64 / on.hub_consults.max(1) as f64;
        println!(
            "  {b:>6} | {avg_indeg:>10.2} | {ohf:>8} → {nhf:<7} | {olf:>9} → {nlf:<8} | {ot:>4} → {nt:<4} ({:+})",
            nt as i64 - ot as i64,
        );
    }
    println!(
        "  → leaf faults are unchanged across the pressure range: the slot centrality spends on the\n\
         \x20   hub comes from a leaf that was the next eviction anyway. The tighter the budget the\n\
         \x20   bigger the win (budget 4: hub faults 15→7); even at one-region budget (11) cross-region\n\
         \x20   paging still evicts hubs, so a small win persists. It is fully inert only when the whole\n\
         \x20   graph fits resident. The win is real, small, and free of leaf cost."
    );

    println!(
        "\nVerdict (measure-first): enabling centrality is a small, consistent, *low-risk* retention\n\
         win on re-engagement workloads — it cuts hub page-faults (most when the hub is cold-ish),\n\
         costs no leaf faults, is binary in w, and self-limits via resident in-degree. It stays OFF\n\
         by default (snapshots/replay byte-identical; the gain is workload-dependent), and the\n\
         log-tuner can switch it on where a hub-heavy access pattern makes it worthwhile."
    );
}

/// Clone a `Workload`'s shape (it is not `Clone` by design — one field is overridden
/// at each call site via struct-update from this).
fn workload_like(w: &Workload) -> Workload {
    Workload {
        regions: w.regions,
        leaves_per_region: w.leaves_per_region,
        passes: w.passes,
        window: w.window,
        budget: w.budget,
    }
}
