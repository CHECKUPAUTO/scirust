//! **Decay crux: why a stale, never-reaffirmed dissent should stop deadlocking a claim.** The
//! Q-Page "knowledge half-life". [`MemoryGraph::qbelief`] weighs every assertion equally and
//! forever, so a single old objection that was never revisited keeps a claim looking *eternally
//! contested* even after fresh evidence has arrived on the other side.
//! [`MemoryGraph::qbelief_decayed`] fades each edge by `0.5^(age / half_life)`, so a fresh
//! (re-)assertion outweighs an ageing one: recent evidence **resolves** the stale dissent, and the
//! agent's memory is not held hostage by a one-off objection it never reaffirmed.
//!
//! Fixture: a claim contested at `t = 0` (one support, one contradiction). The support is then
//! re-affirmed by a *fresh* assertion at time `T` while the contradiction is never revisited. We
//! sweep `T` and contrast the plain belief (frozen — a permanent deadlock) with the decayed belief
//! (the dissent fades, the claim resolves on its own).
//!
//! Run: `cargo run --release --example decay_crux`

use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType, QBelief};

/// Build the fixture with `elapsed` ticks between the one-off objection (t=0) and the fresh
/// support (t=`elapsed`), and return `(plain, decayed)` belief at the current time.
fn scenario(elapsed: u64, half_life: f64) -> (QBelief, QBelief) {
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    let claim: NodeId = "claim".into();
    for id in ["claim", "old_objection", "fresh_support"] {
        g.upsert_node(
            NodeId(id.into()),
            id.into(),
            String::new(),
            NodeType::ContextBlock,
        );
    }
    // The objection is raised once, at t=0, and never revisited.
    g.add_edge(
        "old_objection".into(),
        claim.clone(),
        1.0,
        EdgeType::Contradicts,
    );
    for _ in 0..elapsed {
        g.tick(); // …time passes…
    }
    // Fresh support arrives now (age 0).
    g.add_edge(
        "fresh_support".into(),
        claim.clone(),
        1.0,
        EdgeType::Supports,
    );
    (g.qbelief(&claim), g.qbelief_decayed(&claim, half_life))
}

fn main() {
    let half_life = 10.0;
    println!("# Decay crux — fresh evidence resolves a stale, never-reaffirmed dissent\n");
    println!(
        "claim contested at t=0 (1 support, 1 contradiction); the support is re-affirmed fresh at\n\
         time T, the contradiction is never revisited. half-life = {half_life:.0} ticks.\n"
    );
    println!("    T      PLAIN qbelief         DECAYED qbelief");
    println!("         belief  conflict      belief  conflict   (objection age = T)");
    for t in [0u64, 5, 10, 20, 40, 80] {
        let (plain, decayed) = scenario(t, half_life);
        println!(
            "  {t:>3}     {:>5.2}    {:>5.2}      {:>5.2}    {:>5.2}",
            plain.belief, plain.conflict, decayed.belief, decayed.conflict
        );
    }
    println!(
        "\nReading: PLAIN qbelief is frozen — one un-reaffirmed objection counts as much as the fresh\n\
         support forever, so the claim reads as a permanent deadlock (conflict 0.67, belief 0) no\n\
         matter how stale the objection is. With decay, the objection's weight halves every {half_life:.0}\n\
         ticks, so as T grows the fresh support wins: conflict collapses toward 0 and belief climbs.\n\
         The claim *resolves on its own* once the dissent is old enough and unrefreshed — the\n\
         knowledge half-life that keeps an agent's memory from being held hostage by stale objections.\n\
         (Decay touches `conflict` only because the two surfaces age differently; equally-fresh\n\
         evidence would leave the balance — and so `conflict` — untouched.)"
    );
}
