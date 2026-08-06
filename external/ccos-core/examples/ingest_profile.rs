//! **Ingestion profiler** — *measure before optimizing*. Where does CCOS actually spend time turning
//! source into the causal graph? This stages the real ingestion pipeline over a synthetic corpus and
//! times each phase separately, via the public APIs (no library instrumentation), so the dominant
//! cost is identified before any data-oriented / cache work is justified.
//!
//! Phases: (1) parse (syn AST), (2) build graph (`update_memory_graph`), (3) link imports,
//! (4) resolve calls, (5) resolve data-flow. The last three are the whole-graph passes that iterate
//! the edge set — the prime suspects for a cache-bound hotspot.
//!
//! Run: `cargo run --release --example ingest_profile`

use ccos_core::memory::MemoryGraph;
use ccos_core::parser::ASTParser;
use std::time::Instant;

/// A synthetic source file: a module with `fns` mutually-calling functions, a `const` they all read
/// (stresses data-flow), and a cross-file `use` + a `shared_i` fn the next file calls (stresses
/// cross-file call resolution).
fn synth_file(i: usize, fns: usize) -> (String, String) {
    let prev = i.saturating_sub(1);
    let mut s = String::new();
    s.push_str(&format!("use crate::mod_{prev}::shared_{prev};\n"));
    s.push_str(&format!("pub const LIMIT_{i}: usize = {i};\n"));
    for f in 0..fns {
        let next = (f + 1) % fns;
        // each fn calls the next in-file fn, reads the module const, and (f==0) calls the prev file's
        // shared fn — a real cross-file edge for the resolver to chase.
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
    let n_files = 600usize;
    let fns_per_file = 20usize;
    let files: Vec<(String, String)> = (0..n_files).map(|i| synth_file(i, fns_per_file)).collect();
    let total_bytes: usize = files.iter().map(|(_, s)| s.len()).sum();

    let parser = ASTParser::new();

    // (1) parse — syn AST over every file.
    let t = Instant::now();
    let parsed: Vec<_> = files
        .iter()
        .map(|(p, s)| parser.parse_source(p, s))
        .collect();
    let parse = t.elapsed();

    // (2) build graph — fold every ParseResult into the graph (nodes + pending calls/data-refs).
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    let t = Instant::now();
    for ((_, src), r) in files.iter().zip(&parsed) {
        parser.update_memory_graph(r, src, &mut g);
    }
    let build = t.elapsed();

    // (3) link imports — file→file dependency edges.
    let t = Instant::now();
    let import_edges = g.link_module_imports();
    let link = t.elapsed();

    // (4) resolve calls — caller→callee Calls edges (whole-graph).
    let t = Instant::now();
    let call_edges = g.resolve_symbol_calls();
    let calls = t.elapsed();

    // (5) resolve data-flow — reader→const DataFlow edges (whole-graph).
    let t = Instant::now();
    let df_edges = g.resolve_data_flow();
    let dataflow = t.elapsed();

    let total = parse + build + link + calls + dataflow;
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();

    println!(
        "# Ingestion profile — {n_files} files, {fns_per_file} fns each, {total_bytes} bytes\n"
    );
    println!(
        "graph: {} nodes, {} edges  (imports {import_edges}, calls {call_edges}, data-flow {df_edges})\n",
        g.node_ids().count(),
        g.edges().len(),
    );
    println!("  phase                 time(ms)    %       ");
    for (name, d) in [
        ("1 parse (syn AST)", parse),
        ("2 build graph", build),
        ("3 link imports", link),
        ("4 resolve calls", calls),
        ("5 resolve data-flow", dataflow),
    ] {
        println!("  {name:<22} {:>7.1}  {:>5.1}", ms(d), pct(d));
    }
    println!("  {:<22} {:>7.1}  100.0", "TOTAL", ms(total));
    println!(
        "\nthroughput: {:.1} files/s, {:.2} MB/s  (total {:.1} ms)",
        n_files as f64 / total.as_secs_f64(),
        total_bytes as f64 / 1e6 / total.as_secs_f64(),
        ms(total)
    );
    // ── Scaling: the resolve passes re-run after EVERY file (exactly what `ingest_source` does), so
    // ingestion cost is quadratic in the file count. Isolated here on a no-paging graph. ──────────
    println!(
        "\n# Scaling — whole-graph resolve re-run per file (the real ingest pattern), no paging"
    );
    println!("  files   resolve total(ms)   ratio when files ×2");
    let mut prev = 0.0;
    for &n in &[150usize, 300, 600] {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let mut resolve_ms = 0.0;
        for i in 0..n {
            let (p, s) = synth_file(i, fns_per_file);
            let r = parser.parse_source(&p, &s);
            parser.update_memory_graph(&r, &s, &mut g);
            g.link_module_imports();
            let t = Instant::now();
            g.resolve_symbol_calls();
            g.resolve_data_flow();
            resolve_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        let ratio = if prev > 0.0 {
            resolve_ms / prev
        } else {
            f64::NAN
        };
        println!("  {n:>5}   {resolve_ms:>15.1}   {ratio:>5.2}");
        prev = resolve_ms;
    }

    // ── B2-batch: defer the whole-graph resolve passes to ONE call at the batch
    // boundary (exactly what `CcosMemory::ingest_deferred` + `resolve` do). For this
    // unambiguous corpus the resolved graph is identical to the per-file path; the
    // cost collapses from the O(N²) above to a single O(N) pass. ──────────────────
    println!("\n# B2-batch — defer, then resolve ONCE after all files (no paging)");
    println!("  files   batch resolve(ms)   ratio when files ×2");
    let mut prev_b = 0.0;
    for &n in &[150usize, 300, 600] {
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        // Ingest every file WITHOUT resolving (defer the three passes)…
        for i in 0..n {
            let (p, s) = synth_file(i, fns_per_file);
            let r = parser.parse_source(&p, &s);
            parser.update_memory_graph(&r, &s, &mut g);
        }
        // …then resolve the whole graph exactly once, timing only that. `resolve_all`
        // is the order-independent driver (B2-full): it prunes the resolution-owned
        // edges, then re-runs the three passes. On this fresh graph there are no prior
        // resolution edges, so the prune is a no-op scan and the cost is the passes.
        let t = Instant::now();
        g.resolve_all();
        let batch_ms = t.elapsed().as_secs_f64() * 1e3;
        let ratio = if prev_b > 0.0 {
            batch_ms / prev_b
        } else {
            f64::NAN
        };
        println!("  {n:>5}   {batch_ms:>16.1}   {ratio:>5.2}");
        prev_b = batch_ms;
    }

    println!(
        "\nFinding: parse (syn) is cheap (~5%); the cost is the whole-graph **resolve passes**\n\
         (data-flow ~49%, calls ~23%) — never the parser, never cache layout.\n\
         B1 (applied): `add_edge` originally deduped with an O(E) LINEAR SCAN of all edges, which made\n\
         ingestion ~CUBIC (600 files = ~216 s). Replacing it with an O(1) membership-set dedup dropped\n\
         the single pass ~11x (resolve-data-flow ~70x) and the scaling to a clean ~O(N²) (×~4.3 per\n\
         file-count doubling; 600 files ≈ 11 s). The remaining quadratic is the second cause —\n\
         per-file ingestion re-runs the whole-graph passes after EVERY file.\n\
         B2-batch (applied): the resolve passes are order-independent pure functions of the FINAL\n\
         node + pending-ref set, so deferring them to ONE call at the batch boundary collapses the\n\
         O(N²) above to a single O(N) pass (the `# B2-batch` table: ~×2 per doubling, not ~×4.3).\n\
         `CcosMemory::ingest_deferred` + `resolve` expose this (via `resolve_all`).\n\
         B2-full (applied): `resolve_all` PRUNES the resolution-owned edges then rebuilds from the\n\
         final state, so per-file (eager) and batch ingestion — and a replay re-ingest — converge on\n\
         the IDENTICAL graph (no order-dependent stale edges). `replay == live` stays exact, so the\n\
         replayable path may now batch too (mechanical follow-up). See docs/MEASUREMENT_batch_resolution.md.\n\
         DOD/SoA would only shave a constant factor and was never the bottleneck.\n\
         THIS is why we measure first."
    );
}
