//! Does causal memory select better working context than search does?
//!
//! The protocol is frozen in `docs/CCOS_COCHANGE_PROTOCOL.md` of the `scirust`
//! repository, committed before this file was written and before any number was
//! seen. The short version:
//!
//! * **Ground truth is git history, not CCOS.** A commit touching several `.rs`
//!   files is a direct observation that those files had to be touched together,
//!   recorded by humans months before CCOS existed. The repository's other
//!   retrieval harnesses take CCOS's own resolved graph as their reference, so
//!   they measure internal consistency and cannot fail.
//! * **The control is not a strawman.** BM25 from `ccos_core::retrieval` — the
//!   same tokenizer, the same deterministic arithmetic, the same corpus, the same
//!   token budget. Comparing against a blind agent would prove nothing.
//! * **The success criterion was fixed in advance**: CCOS must lead on
//!   Recall@budget at *all three* budgets, by at least 5 points at 2048.
//!
//! Run:
//!
//! ```bash
//! sh scripts/ccos/cochange_cases.sh > cases.tsv          # in the scirust repo
//! cargo run --release --example cochange_eval -- <repo-root> cases.tsv
//! ```

use ccos_core::external_memory::{CcosMemory, ExternalMemory, Recall};
use ccos_core::retrieval::Bm25Index;
use std::collections::BTreeSet;
use std::path::Path;

/// Budgets from the protocol. An advantage at a single budget is a tuning
/// artefact, not a result — which is why there are three.
const BUDGETS: [usize; 3] = [1024, 2048, 8192];

/// Same token estimate the recall assembler uses, so both systems are filled to
/// the same line.
fn tokens_of(text: &str) -> usize {
    text.chars().count() / 4
}

/// The file a recall item belongs to: `file:a/b.rs` and `sym:a/b.rs:name` both
/// map to `a/b.rs`. Connector hubs (`dep:`) belong to no file.
fn file_of_uri(uri: &str) -> Option<String> {
    if let Some(rest) = uri.strip_prefix("file:") {
        return Some(rest.to_string());
    }
    for prefix in ["sym:", "mod:", "use:"] {
        if let Some(rest) = uri.strip_prefix(prefix) {
            // `path.rs:symbol` — the path is everything before the last colon.
            return Some(match rest.rsplit_once(':') {
                Some((path, _)) => path.to_string(),
                None => rest.to_string(),
            });
        }
    }
    None
}

/// Metrics for one system over one case.
#[derive(Default, Clone, Copy)]
struct Case {
    /// Fraction of the target files that made it into the window.
    recall: f64,
    /// Reciprocal rank of the first target file, 0 when none appeared.
    rr: f64,
    /// Fraction of the window's files that were targets.
    precision: f64,
}

/// Score a ranked list of files against the targets. `ranked` is in window order
/// and excludes the anchor.
fn score(ranked: &[String], targets: &BTreeSet<String>) -> Case {
    if targets.is_empty() {
        return Case::default();
    }
    let hits = ranked.iter().filter(|f| targets.contains(*f)).count();
    let rr = ranked
        .iter()
        .position(|f| targets.contains(f))
        .map(|i| 1.0 / (i as f64 + 1.0))
        .unwrap_or(0.0);
    Case {
        recall: hits as f64 / targets.len() as f64,
        rr,
        precision: if ranked.is_empty() {
            0.0
        } else {
            hits as f64 / ranked.len() as f64
        },
    }
}

/// CCOS's answer: the files its `around` window covers, in window order.
///
/// `ensure_resident` first, exactly as the `ccos memory` façade and
/// `AgentSession::recall` do. Without it an anchor that has been demoted to the
/// COLD tier yields an empty window — which is how the first run of this harness
/// scored CCOS at a flat 0.0% on every metric at every budget. That was the
/// harness, not the product.
fn ccos_files(mem: &mut CcosMemory, anchor: &str, budget: usize) -> Vec<String> {
    let uri = format!("file:{anchor}");
    mem.ensure_resident(&uri);
    let win = mem.recall(&Recall::around(uri), budget);
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for item in &win.items {
        if let Some(f) = file_of_uri(&item.uri) {
            if f != anchor && seen.insert(f.clone()) {
                out.push(f);
            }
        }
    }
    out
}

/// The control's answer: BM25 over the same corpus, the anchor's own text as the
/// query, filled to the same token budget.
fn bm25_files(
    index: &Bm25Index,
    paths: &[String],
    sources: &[String],
    anchor_idx: usize,
    budget: usize,
) -> Vec<String> {
    // Ask for the whole corpus and cut on budget, exactly as the recall assembler
    // cuts — ranking deeper than the budget can reach would flatter neither side.
    let ranked = index.search(&sources[anchor_idx], paths.len());
    let mut out = Vec::new();
    let mut spent = 0usize;
    for scored in ranked {
        let i = scored.id as usize;
        if i == anchor_idx {
            continue;
        }
        let cost = tokens_of(&sources[i]);
        if spent + cost > budget && !out.is_empty() {
            continue; // over budget — keep packing smaller ones, as CCOS does
        }
        spent += cost;
        out.push(paths[i].clone());
    }
    out
}

/// Every `.rs` file under `root`, excluding build output.
fn collect(root: &Path) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name == "target" || name == ".git" || name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                walk(&p, root, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                if let Ok(rel) = p.strip_prefix(root) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// Mean of a metric over a slice, 0 for an empty slice.
fn mean(v: &[Case], f: impl Fn(&Case) -> f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(&f).sum::<f64>() / v.len() as f64
}

fn report(label: &str, ccos: &[Case], bm25: &[Case]) {
    println!("\n  {label}  (n = {} cas)", ccos.len());
    println!(
        "    {:<12} {:>12} {:>12} {:>10}",
        "", "CCOS", "BM25", "écart"
    );
    for (name, get) in [
        ("Recall", (|c: &Case| c.recall) as fn(&Case) -> f64),
        ("MRR", |c: &Case| c.rr),
        ("Precision", |c: &Case| c.precision),
    ] {
        let (a, b) = (mean(ccos, get), mean(bm25, get));
        println!(
            "    {:<12} {:>11.1}% {:>11.1}% {:>+9.1} pt",
            name,
            a * 100.0,
            b * 100.0,
            (a - b) * 100.0
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(root), Some(cases_path)) = (args.next(), args.next()) else {
        eprintln!("usage: cochange_eval <repo-root> <cases.tsv>");
        std::process::exit(2);
    };
    let root = Path::new(&root);

    // ── Corpus: identical for both systems ────────────────────────────────────
    let paths = collect(root);
    let sources: Vec<String> = paths
        .iter()
        .map(|p| std::fs::read_to_string(root.join(p)).unwrap_or_default())
        .collect();
    let index_of: std::collections::HashMap<&str, usize> = paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();
    eprintln!("corpus: {} fichiers .rs", paths.len());

    let mut mem = CcosMemory::new();
    // Both systems must see the same corpus (protocol §4). BM25 indexes every
    // file, so CCOS must keep every file resident too — otherwise the comparison
    // measures the default paging cap rather than the quality of causal
    // selection. `CcosMemory::new` caps at 5000 nodes, well under what 2785 files
    // produce, and the first run of this harness measured exactly that mistake.
    mem.set_max_resident(usize::MAX);
    for (p, s) in paths.iter().zip(&sources) {
        mem.ingest_source(p, s);
    }
    let stats = mem.stats();
    eprintln!(
        "graphe CCOS: {} noeuds, {} aretes",
        stats.nodes, stats.edges
    );

    let mut index = Bm25Index::default();
    for (i, s) in sources.iter().enumerate() {
        index.add(i as u64, s);
    }

    // ── Cases ─────────────────────────────────────────────────────────────────
    let raw = std::fs::read_to_string(&cases_path).expect("cases file");
    let mut commits: Vec<Vec<String>> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    // `git log` is reverse-chronological, so taking the first N cases samples the
    // most recent commits — which here are the ones that vendored CCOS's own
    // source in bulk: densely interlinked in its graph, and co-changing because
    // they were copied together rather than because a task related them. Sampling
    // that way measured CCOS on the friendliest cases the repository contains.
    // Order by content hash instead: deterministic, reproducible, and unrelated to
    // anything either system optimises for.
    commits.sort_by_key(|files| ccos_core::util::sha256_hex(&files.join("\t")));
    eprintln!("commits: {}", commits.len());

    // A full sweep over this corpus does not finish in a sitting, and that is
    // itself a measurement: set CASES_LIMIT to sample. Per-query latency is
    // reported either way, because it is the number that explains the rest.
    let limit: usize = std::env::var("CASES_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    for budget in BUDGETS {
        let (mut c_ms, mut b_ms) = (0u128, 0u128);
        let (mut c_all, mut b_all) = (Vec::new(), Vec::new());
        let (mut c_own, mut b_own) = (Vec::new(), Vec::new());

        for files in &commits {
            let known: Vec<&String> = files
                .iter()
                .filter(|f| index_of.contains_key(f.as_str()))
                .collect();
            if known.len() < 2 {
                continue;
            }
            for anchor in &known {
                if c_all.len() >= limit {
                    break;
                }
                let targets: BTreeSet<String> = known
                    .iter()
                    .filter(|f| **f != *anchor)
                    .map(|f| (*f).clone())
                    .collect();
                let ai = index_of[anchor.as_str()];

                let t = std::time::Instant::now();
                let c = score(&ccos_files(&mut mem, anchor, budget), &targets);
                c_ms += t.elapsed().as_millis();
                let t = std::time::Instant::now();
                let b = score(&bm25_files(&index, &paths, &sources, ai, budget), &targets);
                b_ms += t.elapsed().as_millis();
                c_all.push(c);
                b_all.push(b);

                // Post-hoc split, declared as such in the report: the vendored
                // `external/` tree is a third-party snapshot whose files co-change
                // because they are copied wholesale, not because a task linked
                // them. It is left in the headline because the protocol was frozen
                // without it, and shown separately so the reader can see both.
                if !anchor.starts_with("external/")
                    && !targets.iter().any(|t| t.starts_with("external/"))
                {
                    c_own.push(c);
                    b_own.push(b);
                }
            }
        }

        println!("\n─── budget {budget} tokens ───");
        let n = c_all.len().max(1) as u128;
        println!(
            "  latence moyenne par requete: CCOS {} ms | BM25 {} ms",
            c_ms / n,
            b_ms / n
        );
        report("PRINCIPAL (protocole figé)", &c_all, &b_all);
        report("post-hoc: hors external/ vendorise", &c_own, &b_own);
    }
}
