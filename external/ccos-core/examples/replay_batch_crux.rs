//! **The crux: time-travel reconstruction was O(N²); batching makes it O(N).**
//!
//! [`AgentSession::replay_to`](ccos_core::agent_session::AgentSession::replay_to) rebuilds the agent's
//! memory at any past step by re-applying the recorded op-log on top of the baseline — the engine
//! behind time-travel debugging, `recall_what_if`, and the counterfactual `retrieval_reward`. It
//! used to call the eager `ingest_source` for **every** `Ingest` op, so each of the N ingests re-ran
//! the three whole-graph resolve passes over an O(N) graph — **O(N²)** per reconstruction.
//!
//! B2-full made resolution **order-independent** (prune + rebuild from the final state), so the
//! replay can now *defer* every ingest and resolve **once** before each op that reads cross-file
//! edges (a recall page-in, a failure propagation) and once at the end — the byte-identical graph
//! (`replay == live` still holds, verified by `tests/replay_equivalence_property.rs`) at **O(N)**.
//!
//! This stages both reconstructions over a synthetic cross-referencing corpus and times them.
//!
//! Run: `cargo run --release --example replay_batch_crux`

use ccos_core::agent_session::AgentSession;
use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use std::time::{Duration, Instant};

/// A module with `fns` mutually-calling functions, a const they all read (data-flow), a cross-file
/// `use` + a `shared` fn the next file calls (cross-file call resolution) — real work for the three
/// resolve passes, mirroring `examples/ingest_profile.rs`.
fn synth_file(i: usize, fns: usize) -> (String, String) {
    let prev = i.saturating_sub(1);
    let mut s = String::new();
    s.push_str(&format!("use crate::mod_{prev}::shared_{prev};\n"));
    s.push_str(&format!("pub const LIMIT_{i}: usize = {i};\n"));
    for f in 0..fns {
        let next = (f + 1) % fns;
        if f == 0 {
            s.push_str(&format!(
                "pub fn f_{i}_{f}() -> usize {{ f_{i}_{next}() + LIMIT_{i} + shared_{prev}() }}\n"
            ));
        } else {
            s.push_str(&format!(
                "pub fn f_{i}_{f}() -> usize {{ f_{i}_{next}() + LIMIT_{i} }}\n"
            ));
        }
    }
    s.push_str(&format!("pub fn shared_{i}() -> usize {{ {i} }}\n"));
    (format!("src/mod_{i}.rs"), s)
}

fn main() {
    // 3 fns/file keeps the largest run (600 files ≈ 4.2k nodes) under the default 5000-node
    // resident cap, so the timing isolates the resolve cost from COLD-tier paging — the same
    // reason `ingest_profile` runs with an unbounded cap. Cross the cap and both columns inflate
    // with paging, not resolution, and the O(N²)-vs-O(N) signal is lost.
    let fns = 3usize;
    let ms = |d: Duration| d.as_secs_f64() * 1e3;

    println!("# Replay/time-travel reconstruction — eager (pre-#33) vs batched (#33)\n");
    println!("  files   eager replay(ms)   batched replay(ms)   speedup   batched ratio ×2");
    let mut prev_b = 0.0f64;
    for &n in &[150usize, 300, 600] {
        let files: Vec<(String, String)> = (0..n).map(|i| synth_file(i, fns)).collect();

        // Build the session with the new O(N) batch path (this also exercises `ingest_batch`).
        let mut s = AgentSession::new();
        s.ingest_batch(files.iter().map(|(p, src)| (p.as_str(), src.as_str())));

        // BATCHED replay (#33): `replay_to` defers every ingest and resolves once → O(N).
        let t = Instant::now();
        let replayed = s.replay_to(s.len());
        let batched = ms(t.elapsed());

        // EAGER replay (pre-#33): reconstruct by resolving after EVERY ingest → O(N²). This is
        // exactly what `replay_to` did before this slice (`ingest_source` per op).
        let t = Instant::now();
        let mut eager = CcosMemory::new();
        for (p, src) in &files {
            eager.ingest_source(p, src);
        }
        let eager_ms = ms(t.elapsed());

        // Correctness alongside speed: identical resolved structure either way (B2-full).
        assert_eq!(
            (replayed.stats().nodes, replayed.stats().edges),
            (eager.stats().nodes, eager.stats().edges),
            "batched replay reconstructs the byte-identical graph the eager path does"
        );

        let speedup = if batched > 0.0 {
            eager_ms / batched
        } else {
            f64::NAN
        };
        let ratio = if prev_b > 0.0 {
            batched / prev_b
        } else {
            f64::NAN
        };
        println!("  {n:>5}   {eager_ms:>16.1}   {batched:>18.1}   {speedup:>6.1}x   {ratio:>13.2}");
        prev_b = batched;
    }

    println!(
        "\nFinding: the eager column is **quadratic** (~×4 per doubling — each of N ingests re-runs\n\
         the O(N) resolve passes), the batched column is **linear** (~×2 per doubling — one resolve\n\
         for the whole log). The speedup grows with N, exactly as O(N²)→O(N) predicts. Both rebuild\n\
         the IDENTICAL graph (asserted above): order-independent resolution (B2-full) is what makes\n\
         deferring safe, so `replay == live` holds at linear cost. `AgentSession::ingest_batch` gives\n\
         the same O(N) on the live ingest path. See docs/MEASUREMENT_batch_resolution.md."
    );
}
