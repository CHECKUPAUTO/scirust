//! **Data-flow crux: what do `DataFlow` edges recover that lexical retrieval can't?** The
//! shared-global-state analogue of `call_crux` (which did it for fn→fn `Calls`). A function that
//! reads a global `FOO` literally contains the token `FOO`, so a **reader → data** link is partly
//! lexically visible — the honest question is the **co-reader** link: two functions that both read
//! the same global `MAX_RETRIES` are causally related *through the shared datum*, yet if their
//! bodies share no other domain vocabulary their pairwise lexical similarity collapses. The
//! data-flow graph links them by construction (both point at the same data symbol); a vector
//! retriever, seeing only their near-disjoint token bags, cannot.
//!
//! A `DataFlow` edge is **reader → data**: source is the reader function's symbol node, target is
//! the `static`/`const` it references (resolved by `MemoryGraph::resolve_data_flow`, which links a
//! reference only when exactly one global of that name exists graph-wide). The fixture is a handful
//! of globally-unique consts/statics, several reader functions where some *pairs* read the same
//! const but share **no** domain words otherwise, plus decoy functions. Ground truth = the
//! `DataFlow` edges the resolver actually produced (read back from `g.edges()`). We measure, for a
//! per-symbol TF-IDF baseline: (1) the rank of each **reader → data** target, and (2) the rank of
//! each **co-reader ↔ co-reader** pair (two readers of the same const) — and contrast with the
//! data-flow graph, which recovers both by construction.
//!
//! Run: `cargo run --release --example data_flow_crux`

use ccos_core::embeddings::{tokenize, TfidfEmbedder};
use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::memory::EdgeType;
use std::collections::HashMap;

fn main() {
    // Globally-unique SCREAMING_SNAKE consts/statics (the only valid DataFlow targets), then reader
    // functions with realistic-length, domain-disjoint bodies. Co-reader pairs share the SAME const
    // but otherwise describe unrelated domains (billing, orbital control, baking; irrigation, audio):
    // {charge_invoice, correct_trajectory, knead_dough} all read MAX_RETRIES; {schedule_irrigation,
    // render_waveform} both read TIMEOUT_MS. A reader names the const it reads (so reader→data keeps
    // *some* lexical signal — the const's subwords land in reader and data alike), but a reader never
    // names its co-reader, and across a real-length body that one shared concept is swamped by each
    // function's own disjoint domain words. Bodies deliberately carry only generic Rust boilerplate
    // in common (`pub`/`fn`/`let`/`i64`), which — being shared by every symbol — gives the lexical
    // baseline no help in telling a true co-reader apart from a decoy. Decoys read no global.
    let files: &[(&str, &str)] = &[
        // ── globals (DataFlow targets) ──────────────────────────────────────────────────────────
        ("src/limits.rs", "pub const MAX_RETRIES: i64 = 5;"),
        ("src/timing.rs", "pub const TIMEOUT_MS: i64 = 200;"),
        ("src/sizes.rs", "pub static BUF_BYTES: i64 = 4096;"),
        // ── readers of MAX_RETRIES (co-readers; billing / orbital / baking — disjoint domains) ───
        (
            "src/payment.rs",
            "pub fn charge_invoice(cents: i64) -> i64 {\n    \
             let mut tax = cents * 7 / 100;\n    \
             let total = cents + tax;\n    \
             let mut attempt = 0;\n    \
             while attempt < MAX_RETRIES {\n        \
             tax += 1;\n        \
             attempt += 1;\n    \
             }\n    \
             total + tax\n}",
        ),
        (
            "src/orbit.rs",
            "pub fn correct_trajectory(thrust: i64) -> i64 {\n    \
             let mut velocity = thrust * 3;\n    \
             let altitude = velocity / 9;\n    \
             let mut burn = 0;\n    \
             while burn < MAX_RETRIES {\n        \
             velocity -= altitude;\n        \
             burn += 1;\n    \
             }\n    \
             velocity + altitude\n}",
        ),
        (
            "src/recipe.rs",
            "pub fn knead_dough(flour: i64) -> i64 {\n    \
             let mut hydration = flour * 65 / 100;\n    \
             let mut fold = 0;\n    \
             while fold < MAX_RETRIES {\n        \
             hydration += flour;\n        \
             fold += 1;\n    \
             }\n    \
             hydration\n}",
        ),
        // ── readers of TIMEOUT_MS (co-readers; irrigation / audio — disjoint domains) ────────────
        (
            "src/garden.rs",
            "pub fn schedule_irrigation(soil: i64) -> i64 {\n    \
             let moisture = soil * 2;\n    \
             let valve = moisture + TIMEOUT_MS;\n    \
             valve - soil\n}",
        ),
        (
            "src/audio.rs",
            "pub fn render_waveform(pitch: i64) -> i64 {\n    \
             let sample = pitch << 2;\n    \
             let envelope = sample + TIMEOUT_MS;\n    \
             envelope - pitch\n}",
        ),
        // ── a lone reader of BUF_BYTES (reader→data, but no co-reader) ───────────────────────────
        (
            "src/codec.rs",
            "pub fn decode_packet(frame: i64) -> i64 {\n    \
             let header = frame & 255;\n    \
             header + BUF_BYTES\n}",
        ),
        // ── decoys: unrelated free functions, read no global (distractors) ───────────────────────
        (
            "src/audit.rs",
            "pub fn verify_signature(blob: i64) -> bool {\n    \
             let digest = blob ^ 31;\n    \
             digest > 0\n}",
        ),
        (
            "src/cache.rs",
            "pub fn evict_entry(key: i64) -> i64 {\n    \
             let slot = key % 16;\n    \
             slot + 1\n}",
        ),
        (
            "src/lexer.rs",
            "pub fn split_tokens(line: i64) -> i64 {\n    \
             let count = line / 8;\n    \
             count - 1\n}",
        ),
    ];

    let mut mem = CcosMemory::new();
    for (p, c) in files {
        mem.ingest_source(p, c);
    }
    let g = mem.graph();

    // All symbol nodes (id + body), for the lexical baseline.
    let mut syms: Vec<(String, String)> = g
        .node_entries()
        .filter(|(id, _)| id.0.starts_with("sym:"))
        .map(|(id, n)| (id.0.clone(), n.content.clone()))
        .collect();
    syms.sort();
    let sidx: HashMap<&str, usize> = syms
        .iter()
        .enumerate()
        .map(|(i, (s, _))| (s.as_str(), i))
        .collect();

    // Ground truth: the reader→data DataFlow edges (source = reader fn, target = static/const).
    let mut flows: Vec<(usize, usize)> = Vec::new();
    for e in g.edges() {
        if e.edge_type == EdgeType::DataFlow {
            if let (Some(&a), Some(&b)) =
                (sidx.get(e.source.0.as_str()), sidx.get(e.target.0.as_str()))
            {
                flows.push((a, b)); // a = reader, b = data symbol
            }
        }
    }
    flows.sort_unstable();
    flows.dedup();

    // Co-reader pairs: two distinct readers that point at the SAME data symbol. Group readers by
    // their data target, then emit every reader↔reader pair within a group. These functions are
    // causally related through the shared global yet need share no domain vocabulary at all.
    let mut readers_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(reader, data) in &flows {
        readers_of.entry(data).or_default().push(reader);
    }
    let mut co_readers: Vec<(usize, usize)> = Vec::new();
    for readers in readers_of.values() {
        for i in 0..readers.len() {
            for j in (i + 1)..readers.len() {
                let (a, b) = (readers[i].min(readers[j]), readers[i].max(readers[j]));
                co_readers.push((a, b));
            }
        }
    }
    co_readers.sort_unstable();
    co_readers.dedup();

    // Lexical baseline: TF-IDF over per-symbol token bags.
    let corpus: Vec<Vec<String>> = syms.iter().map(|(_, body)| tokenize(body)).collect();
    let mut tf = TfidfEmbedder::new(128);
    tf.fit(&corpus);
    let vecs: Vec<Vec<f32>> = corpus.iter().map(|t| tf.embed(t)).collect();
    // Rank of `tgt` among all symbols by cosine similarity to `src` (1 = nearest). Directional, so
    // it works for both reader→data (asymmetric roles) and co-reader↔co-reader (symmetric) probes.
    let rank_of = |src: usize, tgt: usize| -> usize {
        let s = TfidfEmbedder::cosine(&vecs[src], &vecs[tgt]);
        let mut r = 1;
        for j in 0..syms.len() {
            if j != src && j != tgt && TfidfEmbedder::cosine(&vecs[src], &vecs[j]) > s {
                r += 1;
            }
        }
        r
    };
    let report = |label: &str, pairs: &[(usize, usize)]| {
        if pairs.is_empty() {
            println!("  {label:<28} (none)");
            return;
        }
        let (mut at1, mut mrr) = (0usize, 0.0f64);
        for &(s, t) in pairs {
            let r = rank_of(s, t);
            if r == 1 {
                at1 += 1;
            }
            mrr += 1.0 / r as f64;
        }
        let n = pairs.len() as f64;
        println!(
            "  {label:<28} lexical recall@1 {:>3.0}%   MRR {:.2}   (n={})",
            100.0 * at1 as f64 / n,
            mrr / n,
            pairs.len()
        );
    };

    println!("# Data-flow crux — what DataFlow edges recover that lexical retrieval misses\n");
    println!(
        "fixture: {} symbols, {} resolved DataFlow (reader→data) edges, {} co-reader pairs\n",
        syms.len(),
        flows.len(),
        co_readers.len()
    );
    println!(
        "LEXICAL TF-IDF (per-symbol) — rank of the target among all symbols, by cosine to source:"
    );
    report("READER -> DATA", &flows);
    report("CO-READER <-> CO-READER", &co_readers);
    println!(
        "\nDATA-FLOW GRAPH — recovers both by construction (reader→data = the edge; co-readers"
    );
    println!("= the two readers sharing one data target, a 2-hop reader→data←reader path).");
    println!(
        "\nReading: a reader names the const it reads, so reader→data retains some lexical signal\n\
         (the const-name subwords land in both the reader body and the data symbol). But two\n\
         co-readers of the same global need never share a domain word, so their pairwise lexical\n\
         similarity collapses — exactly the cross-vocabulary, causally-distant relationship the\n\
         data-flow graph links by construction. This is the data-level analogue of the call crux:\n\
         structure recovers the shared-state links a vector retriever cannot see."
    );
}
