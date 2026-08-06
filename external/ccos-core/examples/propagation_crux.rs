//! **Propagation crux: belief revision across the causal graph.** A claim with *no evidence of its
//! own* should still take on a belief from the causes it depends on — if the database is (resolved
//! to be) the bottleneck, then "scaling the database cuts p99 latency" becomes more plausible even
//! before any direct evidence for it arrives. A static evidence store cannot do this: with no
//! incoming `Supports`/`Contradicts` edge a claim stays neutral forever.
//! [`MemoryGraph::propagate_beliefs`] runs one deterministic hop over the `Causes` edges: every
//! *resolved* cause (`|belief| ≥ threshold`) emits a derived, **attenuated** edge on its effect — a
//! `Supports` from a believed cause, a `Contradicts` from a refuted one — so the effect inherits a
//! weaker, correctly-signed belief. And because the signal attenuates below the threshold, the
//! wavefront naturally **stops** instead of cascading.
//!
//! Run: `cargo run --release --example propagation_crux`

use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};

fn claim(g: &mut MemoryGraph, id: &str) {
    g.upsert_node(
        NodeId(id.into()),
        id.into(),
        String::new(),
        NodeType::ContextBlock,
    );
}

/// Give claim `id` `n_sup` support + `n_con` contradiction edges (fresh nodes, authority 1.0).
fn evidence(g: &mut MemoryGraph, id: &str, n_sup: usize, n_con: usize) {
    for i in 0..n_sup {
        let e = format!("{id}_s{i}");
        claim(g, &e);
        g.add_edge(NodeId(e), NodeId(id.into()), 1.0, EdgeType::Supports);
    }
    for i in 0..n_con {
        let e = format!("{id}_c{i}");
        claim(g, &e);
        g.add_edge(NodeId(e), NodeId(id.into()), 1.0, EdgeType::Contradicts);
    }
}

fn main() {
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    for c in ["A", "B", "C", "D", "E", "F", "G"] {
        claim(&mut g, c);
    }
    evidence(&mut g, "A", 3, 0); // A resolved-true   (belief +0.75)
    evidence(&mut g, "D", 0, 3); // D resolved-false  (belief −0.75)
    evidence(&mut g, "F", 2, 2); // F unresolved      (belief  0.00)
    g.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
    g.add_edge("B".into(), "C".into(), 1.0, EdgeType::Causes); // 2-hop: A → B → C
    g.add_edge("D".into(), "E".into(), 1.0, EdgeType::Causes);
    g.add_edge("F".into(), "G".into(), 1.0, EdgeType::Causes);

    let effects = ["B", "C", "E", "G"];
    let belief = |g: &MemoryGraph, c: &str| g.qbelief(&NodeId(c.to_string())).belief;
    let before: Vec<f64> = effects.iter().map(|c| belief(&g, c)).collect();
    let added = g.propagate_beliefs(0.7, 0.6);
    let after: Vec<f64> = effects.iter().map(|c| belief(&g, c)).collect();

    println!("# Propagation crux — belief revision across the causal graph\n");
    println!(
        "causes: A resolved-true (+0.75), D resolved-false (−0.75), F unresolved (0); each `Causes`"
    );
    println!(
        "an effect that has no evidence of its own. propagate_beliefs(threshold 0.7, damping 0.6) \
         added {added} derived edges.\n"
    );
    let desc = |c: &str| match c {
        "B" => "B — effect of A (resolved-true)",
        "C" => "C — effect of B (2 hops from A)",
        "E" => "E — effect of D (resolved-false)",
        "G" => "G — effect of F (unresolved cause)",
        _ => "",
    };
    println!("  effect                                 belief before   after one hop");
    for (i, c) in effects.iter().enumerate() {
        println!(
            "  {:<38} {:>+5.2}        {:>+5.2}",
            desc(c),
            before[i],
            after[i]
        );
    }
    println!(
        "\nReading: B and E — which have NO direct evidence — take on a weaker, correctly-signed\n\
         belief from their resolved causes: a true cause lends support (B → +0.31), a refuted cause\n\
         lends contradiction (E → −0.31). A static evidence store leaves them at 0 forever. Two\n\
         properties keep it safe: the induced belief is *attenuated* (|0.31| < |0.75|), and because\n\
         0.31 is below the resolve threshold the wavefront STOPS — C (two hops from A) stays 0, so a\n\
         single resolved source cannot cascade into a storm. G stays 0 because its cause F is\n\
         unresolved (balanced) — nothing to propagate. Multi-hop accumulation to convergence is\n\
         shown next."
    );

    // ── Multi-hop: propagate to convergence ────────────────────────────────────
    // A single hop (above) revises only the *direct* effect and the wavefront halts.
    // `propagate_beliefs_converge` keeps sweeping until a fixpoint, so belief flows down the whole
    // chain — as far as the attenuation carries it above the threshold. Here the chain A → B → C
    // is walked at a lower threshold (0.2) and higher damping (0.9) so the induced belief survives
    // two hops: one sweep resolves only B and leaves C at 0; convergence lifts C as well.
    let build_chain = || {
        let mut h = MemoryGraph::new(0.0, usize::MAX);
        for c in ["A", "B", "C"] {
            claim(&mut h, c);
        }
        evidence(&mut h, "A", 3, 0); // A resolved-true (+0.75)
        h.add_edge("A".into(), "B".into(), 1.0, EdgeType::Causes);
        h.add_edge("B".into(), "C".into(), 1.0, EdgeType::Causes);
        h
    };

    let mut one_hop = build_chain();
    one_hop.propagate_beliefs(0.2, 0.9);
    let c_one_hop = belief(&one_hop, "C");

    let mut converged = build_chain();
    let added_converge = converged.propagate_beliefs_converge(0.2, 0.9, 8);
    let b_converge = belief(&converged, "B");
    let c_converge = belief(&converged, "C");

    println!(
        "\n# Multi-hop — propagate_beliefs_converge(threshold 0.2, damping 0.9, max_rounds 8)\n"
    );
    println!(
        "  chain A → B → C, only A has evidence (+0.75). A single sweep resolves the direct effect\n  \
         B but leaves C at {c_one_hop:+.2}; convergence added {added_converge} edges (one per hop) and lifts\n  \
         B to {b_converge:+.2} and C to {c_converge:+.2} — belief reached two hops down, then the attenuation\n  \
         dropped it below the threshold and the sweep reached a fixpoint (no cascade, no cycle spin)."
    );
}
