//! **The sufficient condition, key-free, via isolated models.** The *necessary* condition (is
//! the causally-needed file retrievable?) is LLM-free and measured. The *sufficient* condition
//! asks: given the window a strategy assembles, does a model produce the **correct answer**?
//! That needs a model — but not an API key. On a *synthetic* arithmetic causal chain the answer
//! cannot be known in advance, only computed from the window, so a fresh subagent that sees ONLY
//! the window is a clean, isolated judge (no training-data leakage, unlike the agent that built it).
//!
//! Two strategies select whole files under the same budget:
//! - **RAG** = pure lexical TF-IDF top-k (the standard chunk retriever) — no structural expansion.
//! - **CCOS** = the causal region: a BFS from the queried file over the import edges.
//!
//! The chain's links are named so the distant *cause* shares NO vocabulary with the query, and
//! lexical *decoys* do. So RAG fills its budget with decoys + the query file and misses the cause;
//! CCOS follows the imports to the whole chain. Only the CCOS window lets the model compute 100.
//!
//! Run: `cargo run --release --example sufficient_condition`

use ccos_core::embeddings::{tokenize, TfidfEmbedder};
use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;

const FILLER: &str =
    "\n// --- module notes: padding so the budget binds; no query-relevant terms. \
Auxiliary bookkeeping for the synthetic corpus; identical across files; carries no identifiers \
relevant to any task query whatsoever. ---\n";

fn main() {
    // Chain: result_value = (seed_constant 42 + 8) * 2 = 100. The cause is lexically dissimilar
    // from the query; the decoys are lexically similar but causally irrelevant (and wrong).
    let files: &[(&str, &str)] = &[
        ("src/seed.rs", "pub fn seed_constant() -> i64 { 42 }"),
        (
            "src/step.rs",
            "use crate::seed::seed_constant;\npub fn transform_step() -> i64 { seed_constant() + 8 }",
        ),
        (
            "src/api.rs",
            "use crate::step::transform_step;\npub fn result_value() -> i64 { transform_step() * 2 }",
        ),
        ("src/legacy.rs", "pub fn result_value_legacy() -> i64 { 7 }"),
        ("src/helper.rs", "pub fn compute_result_value_old() -> i64 { 13 }"),
        ("src/cache.rs", "pub fn result_value_fallback() -> i64 { 99 }"),
    ];
    let body: Vec<(String, String)> = files
        .iter()
        .map(|(p, c)| (p.to_string(), format!("{c}\n{FILLER}")))
        .collect();

    // Ingest → the causal graph (AST default) with resolved file→file import edges.
    let mut mem = CcosMemory::new();
    for (p, c) in &body {
        mem.ingest_source(p, c);
    }
    let g = mem.graph();

    let query = "result_value";
    let budget_files = 3; // both strategies pick 3 of the 6 files

    // --- RAG: pin the queried file (the agent is editing it), then pure TF-IDF top-k of the
    // rest (no structural expansion) — the fair lexical baseline. ---
    let api = body.iter().position(|(p, _)| p == "src/api.rs").unwrap();
    let corpus: Vec<Vec<String>> = body.iter().map(|(_, c)| tokenize(c)).collect();
    let mut tf = TfidfEmbedder::new(128);
    tf.fit(&corpus);
    let qv = tf.embed(&tokenize(query));
    let mut others: Vec<usize> = (0..body.len()).filter(|&i| i != api).collect();
    others.sort_by(|&a, &b| {
        TfidfEmbedder::cosine(&tf.embed(&corpus[b]), &qv)
            .partial_cmp(&TfidfEmbedder::cosine(&tf.embed(&corpus[a]), &qv))
            .unwrap()
    });
    let mut rag = vec![api];
    rag.extend(others.into_iter().take(budget_files - 1));

    // --- CCOS: causal region = BFS from the queried file over file→file import edges ---
    let idx: HashMap<&str, usize> = body
        .iter()
        .enumerate()
        .map(|(i, (p, _))| (p.as_str(), i))
        .collect();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); body.len()];
    for e in g.edges() {
        let (s, t) = (
            e.source.0.strip_prefix("file:"),
            e.target.0.strip_prefix("file:"),
        );
        if let (Some(s), Some(t)) = (s, t) {
            if let (Some(&a), Some(&b)) = (idx.get(s), idx.get(t)) {
                adj[a].push(b);
            }
        }
    }
    let start = idx["src/api.rs"];
    let mut ccos = Vec::new();
    let mut seen = HashSet::new();
    let mut q = VecDeque::from([start]);
    seen.insert(start);
    while let Some(u) = q.pop_front() {
        if ccos.len() == budget_files {
            break;
        }
        ccos.push(u);
        for &v in &adj[u] {
            if seen.insert(v) {
                q.push_back(v);
            }
        }
    }

    let render = |sel: &[usize]| -> String {
        sel.iter()
            .map(|&i| format!("// {}\n{}", body[i].0, files[i].1))
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let names = |sel: &[usize]| -> Vec<&str> { sel.iter().map(|&i| files[i].0).collect() };
    let has_cause = |sel: &[usize]| sel.iter().any(|&i| files[i].0 == "src/seed.rs");

    println!(
        "# Sufficient condition — synthetic causal chain, key-free (subagents as the model)\n"
    );
    println!("task: what integer does `result_value()` return?   ground truth: (42+8)*2 = 100\n");
    println!("budget = {budget_files} of 6 files:");
    println!(
        "  RAG  (lexical top-k) : {:?}   cause `seed.rs` present? {}",
        names(&rag),
        has_cause(&rag)
    );
    println!(
        "  CCOS (causal region) : {:?}   cause `seed.rs` present? {}",
        names(&ccos),
        has_cause(&ccos)
    );

    fs::write("/tmp/window_rag.txt", render(&rag)).unwrap();
    fs::write("/tmp/window_ccos.txt", render(&ccos)).unwrap();
    println!(
        "\nwindows → /tmp/window_rag.txt, /tmp/window_ccos.txt. Hand each (alone) to fresh isolated\n\
         models, ask for the integer, grade exact-match on 100: only the window containing the whole\n\
         chain (CCOS) admits the right answer."
    );
}
