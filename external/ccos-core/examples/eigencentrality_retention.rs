//! **Does *global* (eigenvector) centrality retain the true structural pillars better than
//! *local* (in-degree) centrality?** The flat hub-and-leaf workload in
//! `centrality_retention` cannot tell them apart — every dependent is an equal leaf. Real
//! code is *hierarchical*: a core module (`top`) is depended on by mid-modules, each
//! depended on by leaves. On the *full* graph the two signals **disagree**: in-degree ranks
//! a mid (6 leaves) above `top` (only 3 mids), but eigenvector ranks `top` above the mids,
//! because the mids' importance flows into the core they depend on (this is unit-tested).
//!
//! The *hypothesis* was that eigenvector therefore retains `top` better under paging
//! pressure. This measures it across resident budgets — and **refutes it**: the two modes
//! tie on `top` retention, because eviction scores only the **thin resident slice** of the
//! hierarchy, where `top`'s resident in-degree is ~1 and the recursive advantage collapses.
//! An honest negative result; the eigenvector payoff is on the full graph (region formation),
//! not per-tick eviction — see `docs/MEASUREMENT_eigencentrality.md`.
//!
//! Run: `cargo run --release --example eigencentrality_retention`

use ccos_core::memory::{CentralityMode, EdgeType, MemoryGraph, NodeId, NodeType, ScoringWeights};

const MIDS: usize = 3;
const LEAVES_PER_MID: usize = 6;
const PASSES: usize = 3;
const WINDOW: usize = 3;
const MID_CADENCE: usize = 4; // consult the region's mid-hub every N leaf-accesses
const TOP_CADENCE: usize = 7; // consult the shared core every N leaf-accesses (misaligned w/ regions)

fn top_id() -> NodeId {
    "core:top".into()
}
fn mid_id(m: usize) -> NodeId {
    format!("mid:module_{m}").into()
}
fn leaf_id(m: usize, l: usize) -> NodeId {
    format!("leaf:module_{m}/node_{l}").into()
}

/// Build the hierarchy (leaf → mid → top) with no paging during construction.
fn build() -> MemoryGraph {
    let mut g = MemoryGraph::new(0.2, usize::MAX);
    g.upsert_node(
        top_id(),
        "core".into(),
        "// shared core".into(),
        NodeType::Module,
    );
    for m in 0..MIDS {
        let mid = mid_id(m);
        g.upsert_node(
            mid.clone(),
            format!("module {m}"),
            "// module".into(),
            NodeType::Module,
        );
        g.add_edge(mid.clone(), top_id(), 0.7, EdgeType::DependsOn); // mid depends on top
        for l in 0..LEAVES_PER_MID {
            let leaf = leaf_id(m, l);
            g.upsert_node(
                leaf.clone(),
                format!("fn {m}_{l}"),
                "// leaf".into(),
                NodeType::Symbol,
            );
            g.add_edge(leaf, mid.clone(), 0.6, EdgeType::DependsOn); // leaf depends on mid
        }
    }
    g
}

#[derive(Default, Clone, Copy)]
struct Stats {
    top_faults: usize,
    top_consults: usize,
    mid_faults: usize,
    mid_consults: usize,
    leaf_faults: usize,
}

fn weights(mode: CentralityMode) -> ScoringWeights {
    ScoringWeights {
        w_centrality: 0.5,
        centrality_mode: mode,
        ..ScoringWeights::default()
    }
}

/// Run the deterministic hierarchical workload once under `mode` at resident `budget`.
fn run(mode: CentralityMode, budget: usize) -> Stats {
    let mut g = build();
    g.set_scoring_weights(weights(mode));
    g.max_in_memory_nodes = budget;
    g.enforce_paging();

    let mut s = Stats::default();
    let access = |g: &mut MemoryGraph, id: &NodeId| -> bool {
        if g.is_cold(id) {
            g.page_in(id);
            true
        } else {
            g.touch(id);
            false
        }
    };

    for _pass in 0..PASSES {
        for m in 0..MIDS {
            let mid = mid_id(m);
            let mut win = 0usize;
            let (mut since_mid, mut since_top) = (0usize, 0usize);
            let max_start = LEAVES_PER_MID.saturating_sub(WINDOW);
            for step in 0..LEAVES_PER_MID * 2 {
                let off = step % WINDOW;
                let leaf = leaf_id(m, (win + off).min(LEAVES_PER_MID - 1));
                if access(&mut g, &leaf) {
                    s.leaf_faults += 1;
                }
                // Re-observe the dependency chain so the *resident* graph stays faithful to
                // the true structure (both signals read it): leaf→mid and mid→top.
                g.add_edge(leaf, mid.clone(), 0.6, EdgeType::DependsOn);
                g.add_edge(mid.clone(), top_id(), 0.7, EdgeType::DependsOn);

                since_mid += 1;
                if since_mid == MID_CADENCE {
                    if access(&mut g, &mid) {
                        s.mid_faults += 1;
                    }
                    s.mid_consults += 1;
                    since_mid = 0;
                }
                since_top += 1;
                if since_top == TOP_CADENCE {
                    if access(&mut g, &top_id()) {
                        s.top_faults += 1;
                    }
                    s.top_consults += 1;
                    since_top = 0;
                }
                if off == WINDOW - 1 {
                    win = (win + 1).min(max_start);
                }
                g.tick();
            }
        }
    }
    s
}

fn main() {
    println!("# Eigenvector vs in-degree centrality — retention of a recursive pillar\n");
    println!(
        "hierarchy: 1 core (`top`) ← {MIDS} mids ← {LEAVES_PER_MID} leaves each. Global in-degree(top)={MIDS} < \
         a mid's {LEAVES_PER_MID},\nso in-degree under-ranks the real pillar. top consulted every {TOP_CADENCE} \
         leaf-accesses, mid every {MID_CADENCE}; {PASSES} passes.\n"
    );

    // Determinism: same mode + budget replays identically.
    assert_eq!(
        {
            let s = run(CentralityMode::Eigenvector, 5);
            (s.top_faults, s.mid_faults, s.leaf_faults)
        },
        {
            let s = run(CentralityMode::Eigenvector, 5);
            (s.top_faults, s.mid_faults, s.leaf_faults)
        },
        "workload must be deterministic"
    );
    println!("determinism: identical mode+budget replays identically ✓\n");

    println!("  budget | in-degree: top / total | eigenvector: top / total | Δ top-faults");
    println!("  -------+------------------------+--------------------------+-------------");
    let mut any_diff = false;
    for budget in [3usize, 4, 5, 6, 7] {
        let id = run(CentralityMode::InDegree, budget);
        let ev = run(CentralityMode::Eigenvector, budget);
        let id_tot = id.top_faults + id.mid_faults + id.leaf_faults;
        let ev_tot = ev.top_faults + ev.mid_faults + ev.leaf_faults;
        let d = id.top_faults as i64 - ev.top_faults as i64;
        if d != 0 {
            any_diff = true;
        }
        println!(
            "  {budget:>6} | {:>10} / {:<9} | {:>12} / {:<11} | {:>+5}",
            id.top_faults, id_tot, ev.top_faults, ev_tot, d
        );
    }

    println!(
        "\nReading (honest, measure-first): the two modes give {} on `top` retention across the\n\
         pressure range. Why: **paging keeps only a thin resident slice of the hierarchy** — while\n\
         the agent works one region, `top`'s *resident* in-degree is ~1 (one mid resident), so the\n\
         local and global signals see nearly the same slice and the recursive advantage is masked.\n\
         The eigenvector signal genuinely differs on the FULL graph (proved in the unit test\n\
         `eigenvector_centrality_captures_recursive_importance_indegree_misses`), but eviction acts on\n\
         the resident window, where that difference collapses. So the eigenvector/scirust payoff is on\n\
         the **full persistent topology** (region formation, offline pillar ranking, the temporal\n\
         tensor) — not per-tick eviction. A useful negative result that aims the next brick.",
        if any_diff { "DIFFERENT results" } else { "essentially identical results" }
    );
}
