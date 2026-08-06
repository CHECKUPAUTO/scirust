//! **Temporal-tensor crux: the "fever curve" of a conflict-resolution engine.** CCOS is not a social
//! network where structural centrality is the interesting signal — it is a *conflict-resolution*
//! engine, so the quantity that matters over time is the **thermodynamics of belief**: how a claim's
//! **Belief** (`B`, signed) and **Tension** (`T = QBelief.conflict`, the geometric evidence balance)
//! evolve when a contradiction is injected and the system propagates and then decays it.
//!
//! This stages a **Conflict-of-Origins** crisis and records the dynamic-profile tensor
//! `Θ[node, component, t]`, `component ∈ {Belief, Tension}`, across a scripted timeline:
//!
//!   1. **consensus** — origin A (a believed source) propagates support to the decisions that rest on
//!      it; the system is calm and one-sided.
//!   2. **injection (t₀)** — a *conflicting origin* B arrives and is refuted (the "false info").
//!   3. **propagation** — B's refutation propagates one causal hop onto the shared decisions, which now
//!      carry **both** a supporting and a contradicting surface → their **tension spikes**.
//!   4. **relaxation** — no reinforcement; the clock advances and the knowledge half-life decays every
//!      surface, so belief and tension relax back toward neutral — the fever breaks.
//!
//! The origins themselves stay cool (each is internally one-sided); the heat emerges **at the decisions
//! that depend on both** — the thermodynamic signature of a conflict of origins.
//!
//! Run: `cargo run --release --example temporal_tensor_crux`

use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};

/// Upsert a bare claim node.
fn node(g: &mut MemoryGraph, id: &str) {
    g.upsert_node(
        NodeId(id.into()),
        id.into(),
        String::new(),
        NodeType::ContextBlock,
    );
}

/// Give `target` `n` fresh, full-weight evidence edges of `polarity` (each from its own source node),
/// so the origin resolves believed (Supports) or refuted (Contradicts).
fn evidence(g: &mut MemoryGraph, target: &str, n: usize, polarity: EdgeType) {
    for i in 0..n {
        let src = format!("{target}~{i}");
        node(g, &src);
        g.add_edge(NodeId(src), NodeId(target.into()), 1.0, polarity.clone());
    }
}

/// 8-level block sparkline of `vals` mapped from `[lo, hi]` onto ▁▂▃▄▅▆▇█.
fn spark(vals: &[f64], lo: f64, hi: f64) -> String {
    const CELLS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    vals.iter()
        .map(|&v| {
            let f = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            CELLS[((f * 7.0).round() as usize).min(7)]
        })
        .collect()
}

fn main() {
    // The decisions that rest on BOTH origins — where a conflict of origins becomes tension.
    let decisions = ["set_timeout", "set_retries", "set_pool_size"];
    let tracked: Vec<&str> = {
        let mut t = vec!["origin_A", "origin_B"];
        t.extend_from_slice(&decisions);
        t
    };
    let half_life = 6.0;
    let (threshold, damping) = (0.3, 0.6);

    let mut g = MemoryGraph::new(0.0, usize::MAX);
    for &n in &tracked {
        node(&mut g, n);
    }
    // Both origins are *causes* of every shared decision: whichever origin is right dictates the call.
    for d in decisions {
        g.add_edge("origin_A".into(), d.into(), 1.0, EdgeType::Causes);
        g.add_edge("origin_B".into(), d.into(), 1.0, EdgeType::Causes);
    }

    // Θ[node][t] = (belief, tension), recorded frame by frame. Decay is always modelled (the clock is
    // 0 through the crisis, so it is a no-op until the relaxation phase advances it).
    let mut frames: Vec<(&str, Vec<(f64, f64)>)> = Vec::new();
    macro_rules! record {
        ($label:expr) => {{
            let row = tracked
                .iter()
                .map(|n| {
                    let q = g.qbelief_decayed(&NodeId(n.to_string()), half_life);
                    (q.belief, q.conflict)
                })
                .collect();
            frames.push(($label, row));
        }};
    }

    record!("t0 ·baseline");
    evidence(&mut g, "origin_A", 4, EdgeType::Supports); // origin A: a believed source
    record!("t1 ·A asserted");
    g.propagate_beliefs(threshold, damping); // consensus reaches the decisions
    record!("t2 ·propagate");
    g.propagate_beliefs(threshold, damping);
    record!("t3 ·settle");
    evidence(&mut g, "origin_B", 5, EdgeType::Contradicts); // ⚡ conflicting origin arrives, refuted
    record!("t4 ·INJECT B");
    g.propagate_beliefs(threshold, damping); // the refutation reaches the shared decisions → fever
    record!("t5 ·propagate");
    g.propagate_beliefs(threshold, damping);
    record!("t6 ·settle");
    for f in 7..=10 {
        for _ in 0..3 {
            g.tick(); // …time passes, nothing is reaffirmed…
        }
        record!(Box::leak(format!("t{f} ·decay").into_boxed_str()) as &str);
    }

    let labels: Vec<&str> = frames.iter().map(|(l, _)| *l).collect();
    let series = |node_i: usize, comp: usize| -> Vec<f64> {
        frames
            .iter()
            .map(|(_, row)| {
                if comp == 0 {
                    row[node_i].0
                } else {
                    row[node_i].1
                }
            })
            .collect()
    };

    println!("# Temporal-tensor crux — the fever curve of a conflict of origins\n");
    println!(
        "Θ[node, component, t], component ∈ {{Belief ∈ [-1,1], Tension ∈ [0,1]}}, over a scripted\n\
         Conflict-of-Origins crisis. origin_A is a believed source; origin_B is the conflicting origin,\n\
         refuted at t4. Both *cause* the three decisions. half-life = {half_life:.0} ticks.\n"
    );
    println!("  frames: {}\n", labels.join("  "));

    println!("── TENSION  Θ[·, Tension, t]   (▁ calm … █ contested) ──");
    for (i, n) in tracked.iter().enumerate() {
        let s = series(i, 1);
        let peak = s.iter().cloned().fold(0.0_f64, f64::max);
        let kind = if decisions.contains(n) {
            "decision"
        } else {
            "origin  "
        };
        println!("  {kind} {n:<13} {}   peak {peak:.2}", spark(&s, 0.0, 1.0));
    }

    println!("\n── BELIEF   Θ[·, Belief, t]    (▁ refuted -1 … █ believed +1) ──");
    for (i, n) in tracked.iter().enumerate() {
        println!("           {n:<13} {}", spark(&series(i, 0), -1.0, 1.0));
    }

    // System temperature = mean tension across the decisions — the headline fever curve.
    let temp: Vec<f64> = (0..frames.len())
        .map(|t| {
            let s: f64 = decisions
                .iter()
                .map(|d| {
                    let i = tracked.iter().position(|n| n == d).unwrap();
                    frames[t].1[i].1
                })
                .sum();
            s / decisions.len() as f64
        })
        .collect();
    let (t_inject, t_peak) = (temp[4], temp.iter().cloned().fold(0.0_f64, f64::max));
    println!("\n── SYSTEM TEMPERATURE  mean Tension over the 3 decisions ──");
    println!("  {}   peak {t_peak:.2}", spark(&temp, 0.0, 1.0));
    println!(
        "  values: {}",
        temp.iter()
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let cooled = temp.last().copied().unwrap_or(0.0);
    println!(
        "\nReading: the origins stay COOL — each is internally one-sided (A believed, B refuted), tension\n\
         ~0. The heat emerges at the THREE DECISIONS that depend on both: calm and believed under the\n\
         A-only consensus (t2–t3), then at t5 origin_B's refutation propagates one causal hop onto them,\n\
         so each now carries a supporting AND a contradicting surface — tension spikes 0 → {t_peak:.2}\n\
         together (a synchronized fever across the cluster sharing the conflicted origins). With no\n\
         reaffirmation the knowledge half-life then decays every surface, and belief + tension relax\n\
         back toward neutral ({t_peak:.2} → {cooled:.2}) — the fever breaks on its own. The conflict does\n\
         NOT cascade past the decisions: once contested they are unresolved, so the wavefront stops —\n\
         the engine localizes a conflict instead of spreading a storm.\n\
         Signal: sharp and legible (flat→spike→relax), so Θ[node, {{Belief,Tension}}, t] is a real\n\
         primitive — a client can watch the system's *fever* in the face of injected false info, see\n\
         exactly which decisions a conflicting source contaminates, and watch decay resolve it.\n\
         Deterministic (logical clock, sorted propagation, no RNG), so replay == live.\n\
         (Injection at t4 registered as temperature {t_inject:.2}; the spike is one propagation hop later.)"
    );
}
