//! Property-based tests: the memory-graph invariants must hold under *arbitrary*
//! sequences of file edits, not just the hand-written scenarios. This is the
//! randomized counterpart to `tests/graph_invariants.rs`.

use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::MemoryGraph;
use proptest::prelude::*;
use std::collections::HashSet;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// For any sequence of edits to a small pool of files, the graph must keep
    /// `edges ⊆ nodes × nodes` (no dangling edges) and stay bounded by the
    /// paging cap — never growing with the number of operations.
    #[test]
    fn random_edits_keep_graph_consistent(
        ops in prop::collection::vec((0u8..6u8, 0u32..50u32, any::<bool>()), 1..250)
    ) {
        const CAP: usize = 50;
        let mut graph = MemoryGraph::new(0.1, CAP);
        let mut engine = IncrementalGraphEngine::new();

        for (file_idx, sym, modify) in &ops {
            let path = format!("src/f{}.rs", file_idx % 6);
            let source = format!(
                "mod m{s};\nuse dep_{s}::lib;\npub fn func_{s}() {{}}\nstruct S{s};\n",
                s = sym
            );
            // Mix FileAdded and FileModified paths.
            let old = if *modify { Some(String::from("fn old() {}")) } else { None };
            engine.process_delta(&path, old.as_deref(), &source, &mut graph);
        }

        // Invariant 1 — no dangling edges.
        let ids: HashSet<_> = graph.node_ids().cloned().collect();
        let dangling = graph
            .edges()
            .iter()
            .filter(|e| !ids.contains(&e.source) || !ids.contains(&e.target))
            .count();
        prop_assert_eq!(dangling, 0, "found {} dangling edges", dangling);

        // Invariant 2 — node count bounded by the paging cap.
        prop_assert!(graph.node_count() <= CAP, "nodes {} exceed cap", graph.node_count());

        // Invariant 3 — edge count bounded (not O(ops)).
        prop_assert!(
            graph.edge_count() <= CAP * CAP,
            "edge count {} grew unbounded",
            graph.edge_count()
        );
    }

    /// Building the same graph twice from an identical op sequence must yield
    /// the same surviving node set (deterministic eviction).
    #[test]
    fn identical_edit_sequences_are_deterministic(
        ops in prop::collection::vec((0u8..8u8, 0u32..60u32), 1..200)
    ) {
        let build = |ops: &[(u8, u32)]| -> Vec<String> {
            let mut graph = MemoryGraph::new(0.1, 30);
            let mut engine = IncrementalGraphEngine::new();
            for (file_idx, sym) in ops {
                let path = format!("src/f{}.rs", file_idx % 8);
                let source = format!("mod m{s};\npub fn func_{s}() {{}}\nstruct S{s};\n", s = sym);
                engine.process_delta(&path, None, &source, &mut graph);
            }
            let mut ids: Vec<String> = graph.node_ids().map(|k| k.0.clone()).collect();
            ids.sort();
            ids
        };
        prop_assert_eq!(build(&ops), build(&ops));
    }
}
