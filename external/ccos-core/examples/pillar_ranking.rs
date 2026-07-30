//! **Brick 1 of the tensor kernel: does the spectrum of the causal graph find the real
//! architectural pillars?** We treat CCOS's causal graph as a sparse rank-2 adjacency
//! tensor and rank its nodes two ways — *local* in-degree vs *global* eigenvector
//! centrality (its principal eigenvector, deterministic damped power iteration) — then test
//! both against an **objective ground truth**: a file's transitive-dependent count
//! (`query::source_set` — how many files transitively depend on it). The honest question:
//! does the spectral signal predict structural importance better than the local count?
//! Measured on CCOS's *own* source — no synthetic graph.
//!
//! Run: `cargo run --release --example pillar_ranking`

use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::query;
use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Dense ranks (0 = smallest); ties broken by index, which is fine for a correlation.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    for (pos, &i) in idx.iter().enumerate() {
        r[i] = pos as f64;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        0.0
    } else {
        num / (da.sqrt() * db.sqrt())
    }
}

/// Spearman rank correlation = Pearson on ranks (monotonic association, robust to the
/// eigenvector's skewed scale).
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn short(uri: &str) -> &str {
    uri.strip_prefix("file:").unwrap_or(uri)
}

fn main() {
    println!("# Tensor brick 1 — spectral centrality vs in-degree at finding CCOS's pillars\n");

    // Build CCOS's own causal graph (AST is the default parser; ingest resolves cross-file
    // imports into file→file edges). ~1.8k nodes < the 5000 budget ⇒ fully resident.
    let mut mem = CcosMemory::new();
    for path in rust_files(Path::new("src")) {
        if let Ok(src) = fs::read_to_string(&path) {
            mem.ingest_source(&path.to_string_lossy().replace('\\', "/"), &src);
        }
    }
    let g = mem.graph();
    let ev = g.eigencentrality();
    println!(
        "graph: {} nodes, {} edges (resident: {})\n",
        g.node_count(),
        g.edge_count(),
        g.node_count() <= 5000
    );

    // One row per file node: the two signals + the ground-truth structural importance.
    struct Row {
        uri: String,
        indeg: f64,
        eigen: f64,
        dependents: f64,
    }
    let mut rows: Vec<Row> = g
        .node_entries()
        .filter(|(id, _)| id.0.starts_with("file:"))
        .map(|(id, _)| Row {
            uri: id.0.clone(),
            indeg: g.node_in_degree(id) as f64,
            eigen: ev.get(id).copied().unwrap_or(0.0),
            // Transitive dependents: how many files transitively *depend on* this one.
            // CCOS edges point importer→imported, so a pillar's dependents are reached by
            // walking edges in reverse (`source_set`). Computed by graph reachability — by
            // neither ranking under test.
            dependents: query::source_set(g, id, 64).len() as f64,
        })
        .collect();
    // `node_entries` iterates a HashMap (nondeterministic order), and the rank tie-break
    // below is by position — so sort by id first to make the whole measurement deterministic.
    rows.sort_by(|a, b| a.uri.cmp(&b.uri));

    let top = |key: &dyn Fn(&Row) -> f64, label: &str| {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| key(&rows[b]).partial_cmp(&key(&rows[a])).unwrap());
        println!("  top 8 files by {label}:");
        for &i in idx.iter().take(8) {
            println!("    {:>7.3}  {}", key(&rows[i]), short(&rows[i].uri));
        }
        println!();
    };
    top(&|r| r.dependents, "TRANSITIVE DEPENDENTS (ground truth)");
    top(&|r| r.indeg, "in-degree (local)");
    top(&|r| r.eigen, "eigenvector centrality (global / spectral)");

    let truth: Vec<f64> = rows.iter().map(|r| r.dependents).collect();
    let indeg: Vec<f64> = rows.iter().map(|r| r.indeg).collect();
    let eigen: Vec<f64> = rows.iter().map(|r| r.eigen).collect();
    let s_indeg = spearman(&indeg, &truth);
    let s_eigen = spearman(&eigen, &truth);

    println!("Spearman correlation with the ground-truth pillar signal (transitive dependents):");
    println!("  in-degree (local)              : {s_indeg:.3}");
    println!("  eigenvector centrality (global): {s_eigen:.3}");
    let verdict = if s_eigen > s_indeg {
        format!(
            "the spectral signal predicts structural importance BETTER (+{:.3})",
            s_eigen - s_indeg
        )
    } else {
        "the spectral signal does NOT beat the local count here".to_string()
    };
    println!("\n→ {verdict}.");
    println!(
        "Deterministic (fixed-iteration power iteration over sorted ids); a read-only ranking\n\
         signal that never enters the replay hash. This is the rank-2 tensor + its spectrum on\n\
         real code — the foundation the higher-rank work has to beat to earn its dimensions."
    );
}
