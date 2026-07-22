//! Deterministic benchmark for certified diameter-constrained medoid
//! clustering.
//!
//! Seven fixture families probe the solver where greedy heuristics are known
//! to be exact, known to be suboptimal, or merely unproven:
//!
//! - `separable`: three tight, well-separated blocks — every arm finds the
//!   same partition; only searching arms *prove* it;
//! - `chain`/`bridge`: overlap structures. The bridge point is compatible
//!   with two mutually incompatible blocks, so cluster count alone ties and
//!   the observed-medoid cost decides its side;
//! - `adversarial`: seeded metric-free random matrices (no triangle
//!   inequality is assumed anywhere);
//! - `cost_ties`: a unit square and a 6-cycle whose minimum-count partitions
//!   tie on cost — the canonical-assignment rule picks the winner, printed in
//!   the last column;
//! - `weak_lower_bound`: an odd anti-cycle whose chromatic number (3)
//!   strictly exceeds its clique number (2) — the count lower bound provably
//!   undershoots, the exact proof needs thousands of nodes, and the bounded
//!   hybrid exhausts its budget;
//! - `increasing_n`: seeded random matrices of growing size, showing how
//!   explored/pruned node counts scale.
//!
//! Three arms per fixture:
//!
//! - `exact`: unbounded branch and bound — always `proven_optimal`;
//! - `hybrid_n1_i1`: greedy warm start, one improvement sweep, a one-node
//!   search budget. On non-trivial instances it usually returns a good — even
//!   optimal-valued — partition with `proven_optimal = false` and a nonzero
//!   gap: the certificate honestly separates *finding* an optimum from
//!   *proving* one;
//! - `hybrid_n64_i4`: a 64-node budget that completes the search on every
//!   family here except `weak_lower_bound`, where it returns the incumbent
//!   with an explicit gap instead of an unproven claim.
//!
//! No arm ever reports `proven_optimal = true` without an exhausted search
//! space, and no medoid-cost lower bound is claimed on budget exhaustion.
//!
//! Output is deterministic CSV on stdout (`{:.17e}` for costs and gaps); run
//! twice and compare byte-for-byte (`cmp` / SHA-256). No timestamps, no
//! timings.

use scirust_solvers::combinatorial::{
    CertifiedClusteringMode, CertifiedMedoidClusteringConfig, DistanceMatrix,
    MedoidClusteringError, certified_medoid_clustering,
};

/// SplitMix64 core, inlined so the example stays free of extra dependencies.
struct DeterministicRng(u64);

impl DeterministicRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Distance matrix of points on a line (`|xᵢ − xⱼ|`), computed once per pair
/// and mirrored, so symmetry is exact by construction.
fn line_matrix(positions: &[f64]) -> Result<DistanceMatrix, MedoidClusteringError> {
    let n = positions.len();
    let mut values = vec![0.0; n * n];

    for i in 0..n
    {
        for j in (i + 1)..n
        {
            let distance = (positions[i] - positions[j]).abs();
            values[i * n + j] = distance;
            values[j * n + i] = distance;
        }
    }

    DistanceMatrix::new(n, values)
}

/// Seeded metric-free random matrix: distances in `{0.00, 0.01, …, 9.99}`,
/// with no triangle inequality guarantee.
fn random_matrix(size: usize, seed: u64) -> Result<DistanceMatrix, MedoidClusteringError> {
    let mut rng = DeterministicRng(seed);
    let mut values = vec![0.0; size * size];

    for i in 0..size
    {
        for j in (i + 1)..size
        {
            let distance = (rng.next_u64() % 1000) as f64 / 100.0;
            values[i * size + j] = distance;
            values[j * size + i] = distance;
        }
    }

    DistanceMatrix::new(size, values)
}

/// Cycle-graph matrix: `d(i, j) = 0.9 · min(|i − j|, n − |i − j|)`. With
/// diameter `1.0` only adjacent points are compatible, so minimum-count
/// partitions are perfect matchings of adjacent pairs — all tied on cost.
fn cycle_matrix(size: usize) -> Result<DistanceMatrix, MedoidClusteringError> {
    let mut values = vec![0.0; size * size];

    for i in 0..size
    {
        for j in (i + 1)..size
        {
            let hops = (j - i).min(size - (j - i));
            let distance = 0.9 * hops as f64;
            values[i * size + j] = distance;
            values[j * size + i] = distance;
        }
    }

    DistanceMatrix::new(size, values)
}

/// Anti-cycle matrix: adjacent-on-the-cycle pairs are far (2.0), all other
/// pairs near (0.5). With diameter `1.0` the incompatibility graph is exactly
/// the cycle `C_n`; for odd `n` its chromatic number 3 strictly exceeds its
/// clique number 2, so the clique lower bound is provably weak and proving
/// three clusters requires genuine search. Every three-cluster partition
/// costs exactly `0.5 · (n − 3)`, so the search is additionally a pure
/// canonical-assignment tie-break over thousands of tied optima.
fn anti_cycle_matrix(size: usize) -> Result<DistanceMatrix, MedoidClusteringError> {
    let mut values = vec![0.0; size * size];

    for i in 0..size
    {
        for j in (i + 1)..size
        {
            let adjacent = j - i == 1 || (i == 0 && j == size - 1);
            let distance = if adjacent { 2.0 } else { 0.5 };
            values[i * size + j] = distance;
            values[j * size + i] = distance;
        }
    }

    DistanceMatrix::new(size, values)
}

fn fixtures() -> Result<Vec<(&'static str, DistanceMatrix, f64)>, MedoidClusteringError> {
    let mut result: Vec<(&'static str, DistanceMatrix, f64)> = Vec::new();

    // Three tight blocks, ten apart: unambiguous three-cluster optimum.
    result.push((
        "separable",
        line_matrix(&[0.0, 0.15, 0.3, 10.0, 10.15, 10.3, 20.0, 20.15, 20.3])?,
        1.0,
    ));

    // Adjacent-only chain: optimum pairs consecutive points.
    result.push(("chain", line_matrix(&[0.0, 0.9, 1.8, 2.7, 3.6, 4.5])?, 1.0));

    // A bridge compatible with two mutually incompatible blocks: the cost
    // objective decides its side (left, at total cost 1.05 vs 1.10).
    result.push(("bridge", line_matrix(&[0.0, 0.05, 1.0, 2.0, 2.05])?, 1.1));

    for seed_offset in 0..3u64
    {
        result.push((
            "adversarial",
            random_matrix(8, 0xA11CE + 31 * seed_offset)?,
            4.0,
        ));
    }

    // Unit square with diagonal 2: {0,1},{2,3} ties {0,3},{1,2}.
    let square = {
        let mut values = vec![0.0; 16];

        for (i, j, distance) in [
            (0usize, 1usize, 1.0f64),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (0, 3, 1.0),
            (0, 2, 2.0),
            (1, 3, 2.0),
        ]
        {
            values[i * 4 + j] = distance;
            values[j * 4 + i] = distance;
        }

        DistanceMatrix::new(4, values)?
    };

    result.push(("cost_ties", square, 1.0));

    result.push(("cost_ties", cycle_matrix(6)?, 1.0));

    // Odd anti-cycle: the clique bound (2) provably undershoots the optimum
    // (3), so the exact proof needs far more than 64 nodes — this is where
    // the bounded hybrid honestly exhausts its budget.
    result.push(("weak_lower_bound", anti_cycle_matrix(13)?, 1.0));

    for size in [4usize, 6, 8, 10]
    {
        result.push((
            "increasing_n",
            random_matrix(size, 0x0726 + size as u64)?,
            4.0,
        ));
    }

    Ok(result)
}

fn arms() -> Vec<(&'static str, CertifiedClusteringMode)> {
    vec![
        ("exact", CertifiedClusteringMode::Exact),
        (
            "hybrid_n1_i1",
            CertifiedClusteringMode::Hybrid {
                maximum_nodes: 1,
                maximum_iterations: 1,
            },
        ),
        (
            "hybrid_n64_i4",
            CertifiedClusteringMode::Hybrid {
                maximum_nodes: 64,
                maximum_iterations: 4,
            },
        ),
    ]
}

fn main() -> Result<(), MedoidClusteringError> {
    println!("# certified_medoid_clustering deterministic benchmark");
    println!(
        "# arms: exact (proven), hybrid_n1_i1 (greedy incumbent, no search), \
hybrid_n64_i4 (bounded search)"
    );
    println!(
        "# columns: family,size,diameter,arm,cluster_count,medoid_cost,lower_bound_count,\
lower_bound_cost,proven_optimal,optimality_gap,explored_nodes,pruned_nodes,assignments"
    );

    for (family, distances, diameter) in fixtures()?
    {
        for (arm, mode) in arms()
        {
            let result = certified_medoid_clustering(
                &distances,
                CertifiedMedoidClusteringConfig {
                    maximum_cluster_diameter: diameter,
                    mode,
                },
            )?;

            let certificate = &result.certificate;

            let lower_bound_cost = certificate
                .lower_bound_medoid_cost
                .map_or_else(|| "none".to_string(), |cost| format!("{cost:.17e}"));

            let assignments = result
                .assignments
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("-");

            println!(
                "{family},{},{diameter:.17e},{arm},{},{:.17e},{},{lower_bound_cost},{},{:.17e},\
{},{},{assignments}",
                distances.size,
                certificate.objective_cluster_count,
                certificate.objective_medoid_cost,
                certificate.lower_bound_cluster_count,
                certificate.proven_optimal,
                certificate.optimality_gap,
                certificate.explored_nodes,
                certificate.pruned_nodes,
            );
        }
    }

    Ok(())
}
