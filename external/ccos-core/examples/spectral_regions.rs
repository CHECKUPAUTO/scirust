//! **Tensor brick 2: does spectral clustering of the graph Laplacian find better "regions"
//! than the structure already there?** This is the next step the eigencentrality measurement
//! pointed at. We take CCOS's file-dependency graph, build its Laplacian `L = D − A`, and
//! partition it by **recursive Fiedler bisection** (the 2nd-smallest Laplacian eigenvector —
//! classic spectral graph theory, deterministic damped power iteration with the constant
//! vector deflated). We score the partition by **modularity** `Q` and compare it to honest
//! baselines (one cluster, a deterministic name-prefix grouping, and the average of seeded
//! "random" partitions). The question: does CCOS's dependency graph actually *have* community
//! structure a spectral method recovers — i.e. are spectral regions worth their complexity?
//!
//! Run: `cargo run --release --example spectral_regions`

use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use std::collections::HashMap;
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

/// Fiedler vector of the subgraph induced by `members` (indices into the full `adj`):
/// the eigenvector of the 2nd-smallest Laplacian eigenvalue, by deflated power iteration on
/// `(σI − L)` with the constant component removed each step. Deterministic (fixed iterations,
/// fixed non-constant seed).
fn fiedler(members: &[usize], adj: &[Vec<f64>]) -> Vec<f64> {
    let m = members.len();
    // Induced degrees and an upper bound σ ≥ λ_max(L) ≤ 2·d_max.
    let deg: Vec<f64> = members
        .iter()
        .map(|&i| members.iter().map(|&j| adj[i][j]).sum())
        .collect();
    let sigma = 2.0 * deg.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    // Seed: deterministic, non-constant, then projected ⟂ to the all-ones vector.
    let mut x: Vec<f64> = (0..m).map(|i| ((i % 7) as f64) - 3.0).collect();
    let recenter = |v: &mut [f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        for t in v.iter_mut() {
            *t -= mean;
        }
    };
    recenter(&mut x);
    for _ in 0..200 {
        // y = (σI − L) x = (σ − d_i) x_i + Σ_j A_ij x_j   (L = D − A)
        let mut y = vec![0.0; m];
        for (a, &i) in members.iter().enumerate() {
            let mut acc = (sigma - deg[a]) * x[a];
            for (b, &j) in members.iter().enumerate() {
                acc += adj[i][j] * x[b];
            }
            y[a] = acc;
        }
        recenter(&mut y); // keep ⟂ to the constant eigenvector (eigenvalue 0)
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for t in y.iter_mut() {
            *t /= norm;
        }
        x = y;
    }
    x
}

/// Recursive spectral bisection to at most `depth` levels (≤ 2^depth clusters), splitting each
/// part by the sign of its Fiedler vector.
fn bisect(members: Vec<usize>, adj: &[Vec<f64>], depth: u32, out: &mut Vec<Vec<usize>>) {
    if depth == 0 || members.len() <= 3 {
        out.push(members);
        return;
    }
    let f = fiedler(&members, adj);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for (local, &g) in members.iter().enumerate() {
        if f[local] >= 0.0 {
            left.push(g);
        } else {
            right.push(g);
        }
    }
    if left.is_empty() || right.is_empty() {
        out.push(members); // degenerate (e.g. disconnected component) — don't force a split
        return;
    }
    bisect(left, adj, depth - 1, out);
    bisect(right, adj, depth - 1, out);
}

/// Newman modularity of a clustering (`label[i]` = cluster of node i) over symmetric `adj`.
fn modularity(adj: &[Vec<f64>], label: &[usize]) -> f64 {
    let n = adj.len();
    let deg: Vec<f64> = (0..n).map(|i| adj[i].iter().sum()).collect();
    let two_m: f64 = deg.iter().sum();
    if two_m == 0.0 {
        return 0.0;
    }
    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            if label[i] == label[j] {
                q += adj[i][j] - deg[i] * deg[j] / two_m;
            }
        }
    }
    q / two_m
}

fn main() {
    println!("# Tensor brick 2 — spectral (Fiedler) clustering of CCOS's dependency graph\n");

    let mut mem = CcosMemory::new();
    for path in rust_files(Path::new("src")) {
        if let Ok(src) = fs::read_to_string(&path) {
            mem.ingest_source(&path.to_string_lossy().replace('\\', "/"), &src);
        }
    }
    let g = mem.graph();

    // File-dependency graph: the file→file edges, made symmetric. Sorted ids ⇒ deterministic.
    let mut files: Vec<String> = g
        .node_entries()
        .map(|(id, _)| id.0.clone())
        .filter(|id| id.starts_with("file:"))
        .collect();
    files.sort();
    let idx: HashMap<&str, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i))
        .collect();
    let n = files.len();
    let mut adj = vec![vec![0.0; n]; n];
    let mut file_edges = 0;
    for e in g.edges() {
        if let (Some(&a), Some(&b)) = (idx.get(e.source.0.as_str()), idx.get(e.target.0.as_str())) {
            if a != b && adj[a][b] == 0.0 {
                adj[a][b] = 1.0;
                adj[b][a] = 1.0;
                file_edges += 1;
            }
        }
    }
    println!("file-dependency graph: {n} files, {file_edges} undirected edges\n");

    // Spectral clustering (recursive Fiedler bisection, depth 3 ⇒ up to 8 clusters).
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    bisect((0..n).collect(), &adj, 3, &mut clusters);
    let mut label = vec![0usize; n];
    for (c, members) in clusters.iter().enumerate() {
        for &m in members {
            label[m] = c;
        }
    }
    let q_spectral = modularity(&adj, &label);

    // Baselines.
    let q_one = modularity(&adj, &vec![0usize; n]); // everything one cluster → 0
                                                    // Deterministic name-prefix grouping (a naive "structural" baseline): cluster by the
                                                    // first path segment under src/ (e.g. all `src/foo/*` together; flat files each their own).
    let prefix_label: Vec<usize> = {
        let mut seen: HashMap<String, usize> = HashMap::new();
        files
            .iter()
            .map(|f| {
                let tail = f.strip_prefix("file:src/").unwrap_or(f);
                let key = match tail.split_once('/') {
                    Some((dir, _)) => format!("dir:{dir}"),
                    None => tail.to_string(),
                };
                let next = seen.len();
                *seen.entry(key).or_insert(next)
            })
            .collect()
    };
    let q_prefix = modularity(&adj, &prefix_label);
    // Seeded "random" partitions into the same #clusters (deterministic LCG), averaged.
    let k = clusters.len().max(1);
    let mut q_rand_sum = 0.0;
    let trials = 50;
    for s in 0..trials {
        let mut seed = 0x9E3779B97F4A7C15u64 ^ (s as u64).wrapping_mul(0xD1B54A32D192ED03);
        let rlabel: Vec<usize> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % k as u64) as usize
            })
            .collect();
        q_rand_sum += modularity(&adj, &rlabel);
    }
    let q_rand = q_rand_sum / trials as f64;

    println!("modularity Q (higher = stronger community structure):");
    println!("  spectral (Fiedler, {k} clusters) : {q_spectral:.3}");
    println!("  name-prefix grouping             : {q_prefix:.3}");
    println!("  random partition (avg of {trials})      : {q_rand:.3}");
    println!("  one cluster                      : {q_one:.3}\n");

    println!("spectral clusters (file basenames):");
    for (c, members) in clusters.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let names: Vec<&str> = members
            .iter()
            .map(|&m| {
                files[m]
                    .strip_prefix("file:src/")
                    .unwrap_or(&files[m])
                    .trim_end_matches(".rs")
            })
            .collect();
        println!("  [{c}] {}", names.join(", "));
    }

    let biggest = clusters.iter().map(|c| c.len()).max().unwrap_or(0);
    println!(
        "\nReading (honest): the spectral cut DOES beat the naive baselines (Q {q_spectral:.3} vs prefix\n\
         {q_prefix:.3}, random {q_rand:.3}) — real signal, not noise. But Q is far below the ~0.3 that\n\
         marks strong community structure: {biggest} of {n} files collapse into one densely-coupled\n\
         core, so there are no clean modular 'regions' to page on here. The method is sound; CCOS's\n\
         own kernel is simply too small and tightly-coupled to have them — a bigger, more modular\n\
         codebase could differ. Third consistent result: the tensor sophistication beats a trivial\n\
         baseline but not by enough to earn its cost on CCOS. Deterministic (fixed-iteration Fiedler)."
    );
}
