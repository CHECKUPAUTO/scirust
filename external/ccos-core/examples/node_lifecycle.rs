//! **`NodeState` (Stable / Working / Orphan): does separating lifecycle from topology help?**
//! Two concrete pollutions a single 2-D graph suffers, and what the label fixes:
//!
//! 1. **Centrality pollution.** Dead code that still references a core module inflates that
//!    module's in-degree, so the "pillar" signal counts dependents that do not matter.
//! 2. **Eviction pollution.** Code you edited and then abandoned is *recent* (high recency), so
//!    a recency-driven eviction keeps it resident — stealing slots from the real working set.
//!
//! Labeling the dead nodes `Orphan` excludes them from the structural centrality and drives them
//! to the bottom of the eviction order (even at full recency). Measured on a controlled graph.
//!
//! Run: `cargo run --release --example node_lifecycle`

use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeState, NodeType};

const REALS: usize = 6;
const DEAD: usize = 6;

fn pillar() -> NodeId {
    "pillar".into()
}
fn real(i: usize) -> NodeId {
    format!("real_{i}").into()
}
fn dead(i: usize) -> NodeId {
    format!("dead_{i}").into()
}

/// A pillar depended on by REAL nodes and by DEAD nodes (dead code that still references it).
fn build() -> MemoryGraph {
    let mut g = MemoryGraph::new(0.2, usize::MAX);
    g.upsert_node(pillar(), "core".into(), "// core".into(), NodeType::Module);
    for i in 0..REALS {
        g.upsert_node(real(i), format!("real {i}"), "".into(), NodeType::Symbol);
        g.add_edge(real(i), pillar(), 1.0, EdgeType::DependsOn);
    }
    for i in 0..DEAD {
        g.upsert_node(dead(i), format!("dead {i}"), "".into(), NodeType::Symbol);
        g.add_edge(dead(i), pillar(), 1.0, EdgeType::DependsOn);
    }
    g
}

fn reals_resident(g: &MemoryGraph) -> usize {
    (0..REALS).filter(|&i| g.contains_node(&real(i))).count()
}

fn main() {
    println!("# NodeState — separating lifecycle from topology\n");

    // --- 1. Centrality pollution ---
    let mut g = build();
    let indeg_mixed = g.node_in_degree(&pillar());
    for i in 0..DEAD {
        g.set_node_state(&dead(i), NodeState::Orphan);
    }
    let indeg_clean = g.node_in_degree(&pillar());
    println!("## 1. Centrality");
    println!(
        "  pillar in-degree: {indeg_mixed} (dead code counted) → {indeg_clean} (orphans excluded) \
         — the real structural weight, not inflated by dead dependents.\n"
    );

    // --- 2. Eviction pollution ---
    let budget = 1 + REALS; // room for the pillar + the whole real working set, nothing more

    // Realistic churn: the dead code was edited and then abandoned, so it is **fresh** (high
    // recency) while the real working set has aged. A recency-driven policy therefore keeps the
    // dead code and evicts real work — the exact confusion the label resolves.
    let staged = || {
        let mut g = build();
        for _ in 0..12 {
            g.tick(); // the real working set ages
        }
        for i in 0..DEAD {
            g.touch(&dead(i)); // dead code was just edited → recency back to 1.0
        }
        g
    };

    let mut baseline = staged(); // all Stable
    baseline.max_in_memory_nodes = budget;
    baseline.enforce_paging();

    let mut labeled = staged();
    for i in 0..DEAD {
        labeled.set_node_state(&dead(i), NodeState::Orphan);
    }
    labeled.max_in_memory_nodes = budget;
    labeled.enforce_paging();

    println!(
        "## 2. Eviction (budget {budget}; {REALS} aged real + {DEAD} freshly-edited dead nodes)"
    );
    println!(
        "  real nodes retained: {}/{REALS} (all Stable) → {}/{REALS} (dead labeled Orphan)",
        reals_resident(&baseline),
        reals_resident(&labeled),
    );
    println!(
        "  dead nodes resident: {} (all Stable) → {} (labeled)\n",
        (0..DEAD)
            .filter(|&i| baseline.contains_node(&dead(i)))
            .count(),
        (0..DEAD)
            .filter(|&i| labeled.contains_node(&dead(i)))
            .count(),
    );

    println!(
        "Reading: with one undifferentiated graph, abandoned-but-recent dead code is\n\
         indistinguishable from the real working set, so it pollutes the pillar signal and squats\n\
         in memory. The `Orphan` label — a per-node enum, NOT a tensor dimension — restores the\n\
         true centrality and evicts the dead code first, freeing the slots for real work.\n\
         `Working` (not shown) is the dual: pinned resident as the current focus even as recency\n\
         decays. All off by default (`Stable`); deterministic; snapshots byte-identical."
    );
}
