//! How much RAM does the COLD tier keep once every entry is a *compact husk*
//! (slices 5/5b/[u8;32])? This is the last `O(N)` — the husk **count** — and the
//! input to slice 5c (bounding that count). Per the measure-first habit, quantify
//! it and see at what scale it bites before deciding whether the database-grade
//! fix is warranted.
//!
//! Builds N nodes (files + symbols + edges), spills content, then deep-spills the
//! whole tier so every entry is a husk, and reports resident bytes per husk split
//! into its fixed stub (struct + key + body hash) vs its variable adjacency (the
//! neighbour ids `cold_neighbours` needs resident) — the split that decides whether
//! the count can be bounded without an on-disk adjacency index.
//!
//! Run: `cargo run --release --example cold_count`

use ccos_core::memory::{EdgeType, MemoryGraph, NodeType};
use std::path::Path;

fn build_all_husks(files: usize, dir: &Path) -> MemoryGraph {
    let mut g = MemoryGraph::new(0.2, usize::MAX);
    for f in 0..files {
        let file_id = format!("file:src/module_{f}.rs");
        g.upsert_node(
            file_id.clone().into(),
            file_id.clone(),
            format!("// module {f}\n{}", "context ".repeat(20)),
            NodeType::Module,
        );
        for s in 0..3 {
            let sym = format!("sym:src/module_{f}.rs:function_{s}");
            g.upsert_node(
                sym.clone().into(),
                sym.clone(),
                format!("pub fn function_{s}() {{ {} }}", "let _x = 1; ".repeat(8)),
                NodeType::Symbol,
            );
            g.add_edge(file_id.clone().into(), sym.into(), 0.6, EdgeType::Contains);
        }
        if f > 0 {
            let prev = format!("file:src/module_{}.rs", f - 1);
            g.add_edge(
                file_id.clone().into(),
                prev.into(),
                0.5,
                EdgeType::DependsOn,
            );
        }
    }
    g.max_in_memory_nodes = 0;
    g.enforce_paging();
    g.attach_cold_spill(dir, 0).unwrap(); // spill all content
    g.set_cold_resident_budget(Some(0)); // deep-spill EVERY entry → all husks
    g
}

fn main() {
    println!("# COLD husk-count RAM — the last O(N), input to slice 5c\n");

    let dir = std::env::temp_dir().join(format!("ccos_coldcount_{}", std::process::id()));
    let g = build_all_husks(30000, &dir);

    let n = g.cold_count();
    let deep = g.cold_deep_spilled_count();
    let resident = g.cold_resident_bytes();
    let per = resident / n.max(1);
    let gib = 1024usize * 1024 * 1024;

    println!("nodes (all husks): {n}  (deep-spilled: {deep})");
    println!("resident husk metadata: {resident} B  →  {per} B / husk (logical)");
    println!(
        "\nExtrapolation: the husk tier reaches 1 GiB at ~{} husks.",
        gib / per.max(1)
    );
    println!(
        "Per husk that is the body-blob stub plus the **packed** neighbour ids (`adj` — slice\n\
         5c Lever 1: one allocation, not a Vec + a String per id). The adjacency is exactly\n\
         what cold_neighbours reads without disk, so bounding the *count* below O(N) means an\n\
         on-disk adjacency index (Lever 2 — see docs/DESIGN_cold_entry_count.md)."
    );
    println!(
        "\nNote (honest methodology): a process-RSS delta over this build is NOT a clean\n\
         per-husk figure — the build materializes the whole graph (full nodes + content)\n\
         before deep-spilling, so RSS reflects that transient peak, not the husk steady\n\
         state. The logical bytes above are the honest per-husk metric."
    );
    drop(g);
    std::fs::remove_dir_all(&dir).ok();
}
