//! **Adaptive retrieval — the improvement loop learns a cross-vocabulary mapping from feedback.**
//! The premium tier of `ccos_core::retrieval` (gated behind CCOS's own #29 license, `Feature::AdaptiveRetrieval`).
//!
//! Setup: a deliberate **vocabulary gap** — `n` (query, relevant-doc) pairs whose terms are *disjoint*
//! (queries live in dimensions `[0, n)`, docs in `[n, 2n)`), so neither lexical nor dense retrieval can
//! bridge them: a query shares zero vocabulary with its answer. The `ImprovementLoop` records confirmed
//! (query, relevant-doc) pairs and trains a linear projection by deterministic contrastive (InfoNCE)
//! learning, so the projected query and its answer converge — and Recall@k **climbs cycle after cycle**.
//!
//! Everything is deterministic: seeded init, fixed-order `f32`, hand-derived gradients. Re-run → the
//! identical curve. The numbers below are the REAL output of this run.
//!
//! Run: `cargo run --release --example retrieval_improvement`

use ccos_core::license::{License, Licensing};
use ccos_core::retrieval::feedback::ContrastiveConfig;
use ccos_core::retrieval::RetrievalAccess;

fn main() {
    println!("# Adaptive retrieval — the improvement loop learns a cross-vocabulary mapping\n");

    // n (query, relevant-doc) pairs with DISJOINT vocabulary: query i is a one-hot at i, doc i a
    // one-hot at n+i. Query and answer share no term, so base retrieval is at chance.
    let n = 12usize;
    let query = |i: usize| {
        let mut v = vec![0.0f32; 2 * n];
        v[i] = 1.0;
        v
    };
    let doc = |i: usize| {
        let mut v = vec![0.0f32; 2 * n];
        v[n + i] = 1.0;
        v
    };
    let corpus: Vec<(u64, Vec<f32>)> = (0..n).map(|i| (i as u64, doc(i))).collect();
    let eval: Vec<(Vec<f32>, u64)> = (0..n).map(|i| (query(i), i as u64)).collect();

    // ── The license gate ──────────────────────────────────────────────────────
    let now = 1_700_000_000u64;
    println!("[license] adaptive retrieval is a Pro feature (the dense/BM25/hybrid core is free):");
    match RetrievalAccess::unlock(&Licensing::community(), now) {
        Ok(_) => println!("  community tier: unexpectedly unlocked"),
        Err(e) => println!("  community tier → locked ({e}); the free retrieval core still works."),
    }
    let pro = Licensing::licensed(License {
        licensee: "demo-operator".into(),
        expires_at: None,
        machine: None,
    });
    let access = match RetrievalAccess::unlock(&pro, now) {
        Ok(a) => {
            println!("  Pro license   → unlocked.\n");
            a
        }
        Err(e) => {
            println!("  Pro license unexpectedly locked: {e}");
            return;
        }
    };

    // ── The improvement loop ────────────────────────────────────────────────────
    // Small epochs/cycle so the curve climbs visibly across cycles (cumulative training).
    let cfg = ContrastiveConfig {
        epochs: 2,
        lr: 0.06,
        temperature: 0.1,
    };
    let mut loop_ = access.improvement_loop(2 * n, 16, 1234, cfg);
    for i in 0..n {
        loop_.record(&query(i), &doc(i));
    }

    let r1 = |l: &ccos_core::retrieval::feedback::ImprovementLoop| {
        100.0 * l.evaluate_recall_at_k(&eval, &corpus, 1)
    };
    let r3 = |l: &ccos_core::retrieval::feedback::ImprovementLoop| {
        100.0 * l.evaluate_recall_at_k(&eval, &corpus, 3)
    };

    println!(
        "vocabulary gap: {n} (query, relevant-doc) pairs, DISJOINT terms — base retrieval is at chance\n"
    );
    println!("  cycle   Recall@1   Recall@3   (n={n}, 2 epochs/cycle, seeded, deterministic)");
    println!("  {}", "-".repeat(54));
    println!(
        "  {:>3}     {:>5.0}%     {:>5.0}%     base (random projection)",
        0,
        r1(&loop_),
        r3(&loop_)
    );
    for cycle in 1..=8 {
        loop_.train_cycle();
        println!(
            "  {:>3}     {:>5.0}%     {:>5.0}%",
            cycle,
            r1(&loop_),
            r3(&loop_)
        );
    }

    println!(
        "\n→ The loop learns the cross-vocabulary query→answer mapping purely from confirmed feedback,\n\
         with NO shared terms to lean on — Recall climbs from chance toward 100% as cycles accumulate.\n\
         Deterministic end to end (seeded xorshift init, fixed-order f32, hand-derived InfoNCE gradient\n\
         gradient-checked against finite differences): re-run and the curve is identical, bit for bit.\n\
         Gated behind CCOS's own offline ed25519 license (#29) — no `scirust-license` link, no FFI.\n\
         See docs/MEASUREMENT_pure_retrieval.md."
    );
}
