//! Deterministic benchmark for the stronger certified cluster-count bound.
//!
//! The standard certificate from `certified_medoid_clustering` carries a
//! *greedy* clique lower bound on the number of diameter-`D` clusters.
//! `certified_cluster_count_bound` replaces the greedy clique with a
//! maximum-clique search (bounded Bron–Kerbosch), so the bound is the exact
//! clique number `ω` whenever the search completes — never below the greedy
//! bound, and often above it.
//!
//! Since the minimum cluster count equals the chromatic number `χ` of the
//! incompatibility graph and `ω ≤ χ`, a clique bound that *meets* the optimal
//! count is a standalone **proof** of count-optimality — one that needs no
//! branch-and-bound over medoid cost. Four fixtures show the range:
//!
//! - `separable` — three far-apart blocks: greedy already finds the size-3
//!   clique, and the exact search confirms it is maximum;
//! - `greedy_fooled` — a `K4` hidden behind a higher-degree hub: the greedy
//!   clique stalls at 3, the maximum clique is 4, and only the stronger bound
//!   proves the 4-cluster optimum;
//! - `complete` — every pair incompatible: the maximum clique is all `n`;
//! - `odd_anticycle` — a 5-cycle incompatibility graph with `ω = 2 < χ = 3`:
//!   the **honest limit** — even the exact maximum-clique bound undershoots the
//!   true optimum, so no clique argument can certify it (branch and bound must).
//!
//! Deterministic: fixed fixtures, exact search; byte-identical across runs.

use scirust_solvers::combinatorial::{
    CertifiedClusteringMode, CertifiedMedoidClusteringConfig, DistanceMatrix,
    certified_cluster_count_bound, certified_medoid_clustering,
};

const DIAMETER: f64 = 1.0;
const FAR: f64 = 2.0; // > DIAMETER -> incompatible

/// Builds an `n × n` distance matrix from a list of incompatible pairs.
fn from_incompatible(n: usize, pairs: &[(usize, usize)]) -> DistanceMatrix {
    let mut values = vec![0.0; n * n];
    for &(i, j) in pairs
    {
        values[i * n + j] = FAR;
        values[j * n + i] = FAR;
    }
    DistanceMatrix::new(n, values).unwrap()
}

fn separable() -> DistanceMatrix {
    // Three blocks {0,1}, {2,3}, {4,5}; every cross-block pair is incompatible.
    let mut pairs = Vec::new();
    let blocks = [[0usize, 1], [2, 3], [4, 5]];
    for (a, block_a) in blocks.iter().enumerate()
    {
        for block_b in blocks.iter().skip(a + 1)
        {
            for &i in block_a
            {
                for &j in block_b
                {
                    pairs.push((i, j));
                }
            }
        }
    }
    from_incompatible(6, &pairs)
}

fn greedy_fooled() -> DistanceMatrix {
    // K4 among {0,1,2,3}; hub 4 (degree 5) incompatible with 0,1 and leaves 5,6,7.
    let mut pairs = Vec::new();
    for i in 0..4
    {
        for j in (i + 1)..4
        {
            pairs.push((i, j));
        }
    }
    for &j in &[0usize, 1, 5, 6, 7]
    {
        pairs.push((4, j));
    }
    from_incompatible(8, &pairs)
}

fn complete(n: usize) -> DistanceMatrix {
    let mut pairs = Vec::new();
    for i in 0..n
    {
        for j in (i + 1)..n
        {
            pairs.push((i, j));
        }
    }
    from_incompatible(n, &pairs)
}

fn odd_anticycle() -> DistanceMatrix {
    // A 5-cycle incompatibility graph: clique number 2, chromatic number 3.
    from_incompatible(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
}

/// The greedy clique bound the standard certificate reports.
fn greedy_bound(distances: &DistanceMatrix) -> usize {
    certified_medoid_clustering(
        distances,
        CertifiedMedoidClusteringConfig {
            maximum_cluster_diameter: DIAMETER,
            mode: CertifiedClusteringMode::Hybrid {
                maximum_nodes: 1,
                maximum_iterations: 1,
            },
        },
    )
    .unwrap()
    .certificate
    .lower_bound_cluster_count
}

/// The true optimal cluster count, proven by unbounded branch and bound.
fn optimal_count(distances: &DistanceMatrix) -> usize {
    certified_medoid_clustering(
        distances,
        CertifiedMedoidClusteringConfig {
            maximum_cluster_diameter: DIAMETER,
            mode: CertifiedClusteringMode::Exact,
        },
    )
    .unwrap()
    .certificate
    .objective_cluster_count
}

fn main() {
    println!("# certified_cluster_count_bound — stronger clique lower bound");
    println!("# fixture         n   greedy   max_clique(ω)   optimum(χ)   proves_opt");
    let fixtures: [(&str, DistanceMatrix); 4] = [
        ("separable", separable()),
        ("greedy_fooled", greedy_fooled()),
        ("complete_7", complete(7)),
        ("odd_anticycle", odd_anticycle()),
    ];
    for (name, distances) in fixtures
    {
        let greedy = greedy_bound(&distances);
        let optimum = optimal_count(&distances);
        let bound = certified_cluster_count_bound(&distances, DIAMETER, usize::MAX).unwrap();
        let proves = bound.certifies_count_optimal(optimum);
        println!(
            "{name:<15} {:>3}   {greedy:>6}   {:>13}   {optimum:>10}   {}",
            distances.size,
            format!(
                "{}{}",
                bound.lower_bound,
                if bound.clique_is_maximum { "*" } else { "" }
            ),
            if proves { "yes" } else { "no (ω<χ)" },
        );
    }
    println!(
        "# '*' marks a certified-maximum clique; proves_opt = the clique bound alone certifies the optimum."
    );
}
