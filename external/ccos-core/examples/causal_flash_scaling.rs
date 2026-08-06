//! **What does bounding the cone actually save at scale?** The structural signal
//! Causal Flash replaces is a *global* centrality — an eigenvector /
//! [`eigencentrality`] power iteration that sweeps the whole graph many times,
//! `O(iterations · (N + E))`. Causal Flash takes a **single pass** rooted at the
//! small `Working` frontier, so its win is the iteration factor — and its OUTPUT
//! is a bounded, token-budgeted window whose size does not grow with `N`.
//!
//! This example builds the SAME small local structure surrounded by increasing
//! unrelated bulk and times four selections:
//!   - `eigencentrality()`      — the iterated global signal the cone replaces;
//!   - `causal_flash_window()`  — the convenience cone. The adjacency index is
//!     now CACHED on the graph (rebuilt only when edges change), so this is
//!     O(N + cone) amortized — the Working-seed scan plus a cone traversal, no
//!     per-call edge re-index;
//!   - `causal_flash_window_with()` — the HOT path (cached index + explicit
//!     seed): O(cone), flat regardless of N.
//!
//! Fully deterministic — no RNG. Run:
//!   cargo run --release --example causal_flash_scaling
//!
//! [`eigencentrality`]: ccos_core::memory::MemoryGraph::eigencentrality

use std::time::Instant;

use ccos_core::causal_flash::CausalFlashConfig;
use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeState, NodeType};

fn build(bulk: usize) -> MemoryGraph {
    let mut g = MemoryGraph::new(0.2, usize::MAX);
    fn node(g: &mut MemoryGraph, id: &str) {
        g.upsert_node(id.into(), id.into(), "x".into(), NodeType::Module);
    }
    fn dep(g: &mut MemoryGraph, a: &str, b: &str) {
        g.add_edge(a.into(), b.into(), 1.0, EdgeType::DependsOn);
    }
    // Fixed local cone: w → d1 → d2 → d3, and a caller c → w.
    for id in ["w", "d1", "d2", "d3", "c"] {
        node(&mut g, id);
    }
    dep(&mut g, "w", "d1");
    dep(&mut g, "d1", "d2");
    dep(&mut g, "d2", "d3");
    dep(&mut g, "c", "w");
    g.set_node_state(&"w".into(), NodeState::Working);
    // Unrelated bulk: a long disconnected dependency chain of Stable nodes.
    for i in 0..bulk {
        node(&mut g, &format!("bulk_{i}"));
        if i > 0 {
            dep(&mut g, &format!("bulk_{i}"), &format!("bulk_{}", i - 1));
        }
    }
    g
}

fn median_us(mut f: impl FnMut() -> usize, reps: usize) -> f64 {
    let mut samples: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() {
    let cfg = CausalFlashConfig::default();
    let reps = 15;

    println!("Causal Flash vs global centrality — scaling with graph size");
    println!("(median of {reps} runs; deterministic construction, no RNG)\n");
    println!(
        "{:>9}  {:>15}  {:>14}  {:>13}  {:>12}  {:>10}",
        "nodes", "eigencentral µs", "conv(cached) µs", "HOT cone µs", "cone_nodes", "hot vs eig"
    );

    let seeds = [NodeId("w".into())];
    for &bulk in &[1_000usize, 5_000, 20_000, 50_000] {
        let g = build(bulk);
        let n = g.node_count();

        let eigen = median_us(|| g.eigencentrality().len(), reps);
        let conv = median_us(|| g.causal_flash_window(&cfg).nodes.len(), reps);

        // Hot path: build the adjacency ONCE (amortized across ticks), then the
        // per-tick call is O(cone) with an explicit seed frontier.
        let adj = g.causal_adjacency();
        let mut cone_nodes = 0usize;
        let hot = median_us(
            || {
                let w = g.causal_flash_window_with(&cfg, &adj, &seeds);
                cone_nodes = w.nodes.len();
                w.nodes.len()
            },
            reps,
        );

        println!(
            "{n:>9}  {eigen:>15.1}  {conv:>14.1}  {hot:>13.2}  {cone_nodes:>12}  {:>9.0}x",
            eigen / hot.max(f64::MIN_POSITIVE)
        );
    }

    println!(
        "\nWith the adjacency index cached on the graph, the convenience path no\n\
         longer re-indexes edges — it is O(N + cone) amortized (the Working-seed\n\
         scan). The HOT path (cached index + explicit seed) is O(cone): flat, a\n\
         few µs at any N, while the iterated eigenvector centrality it replaces\n\
         grows with N. All paths return the same constant-size, budgeted window."
    );
}
