//! Certified diameter-constrained medoid clustering (SRCC program, phase 726).
//!
//! Replaces "greedy complete-link is good enough" with an opt-in solver that
//! either **proves** a globally optimal partition or returns the best known
//! partition together with a **valid lower bound and an explicit optimality
//! gap** — never an unproven optimality claim.
//!
//! # Problem
//!
//! Given `n` points with pairwise distances and a diameter budget `D`,
//! partition the points so that every intra-cluster pair satisfies
//! `d(i, j) ≤ D`, optimizing the lexicographic objective:
//!
//! 1. minimize the number of clusters;
//! 2. subject to that, minimize the total **observed-medoid** cost
//!    `Σ_C min_{m ∈ C} Σ_{p ∈ C} d(p, m)` (medoids are observed points, never
//!    synthetic centroids; medoid ties resolve to the smallest index);
//! 3. subject to that, return the lexicographically smallest canonical
//!    assignment vector (labels numbered by first occurrence).
//!
//! # Graph view
//!
//! Connecting every pair with `d(i, j) > D` yields the *incompatibility
//! graph*; a valid cluster is exactly an independent set, so minimizing the
//! cluster count is graph coloring. The exact solver is a deterministic
//! DSATUR-style branch and bound: dynamic saturation-first vertex selection
//! (ties by descending incompatibility degree, then ascending index), colors
//! tried in ascending label order plus at most one fresh label, pruned against
//! the incumbent's lexicographic objective. The count lower bound is a
//! deterministic greedy clique on the incompatibility graph (`χ ≥ ω` — valid,
//! not necessarily tight).
//!
//! # Certificates, honestly
//!
//! - `proven_optimal = true` **only** when the search space was exhausted;
//! - on budget exhaustion the incumbent is returned with
//!   `proven_optimal = false`, the clique lower bound on the count, **no**
//!   medoid-cost lower bound (`None` — the trivial `0.0` is the only generally
//!   valid cost bound because distances are not required to satisfy the
//!   triangle inequality, and pretending otherwise would be dishonest), and a
//!   documented positive [`ClusteringCertificate::optimality_gap`];
//! - exhausting the node budget is a certificate state, not an error.
//!
//! # Determinism
//!
//! Fixed vertex orders, deterministic tie-breaks everywhere, no randomness:
//! identical inputs produce identical partitions, certificates and node
//! counts.

use thiserror::Error;

/// A dense symmetric distance matrix (row-major, `size × size`).
#[derive(Clone, Debug, PartialEq)]
pub struct DistanceMatrix {
    /// Number of points.
    pub size: usize,
    /// Row-major distances (`values[i * size + j]`).
    pub values: Vec<f64>,
}

impl DistanceMatrix {
    /// Validates and wraps a row-major distance matrix.
    ///
    /// Requirements (each violation is a typed error): `values.len() ==
    /// size * size`; every entry finite and non-negative; an exactly zero
    /// diagonal; exact symmetry (`values[i,j] == values[j,i]` bit for bit —
    /// the documented tolerance is zero; compute each distance once and mirror
    /// it).
    pub fn new(size: usize, values: Vec<f64>) -> Result<Self, MedoidClusteringError> {
        if size == 0
        {
            return Err(MedoidClusteringError::EmptyMatrix);
        }

        if values.len() != size * size
        {
            return Err(MedoidClusteringError::SizeMismatch {
                expected: size * size,
                found: values.len(),
            });
        }

        for row in 0..size
        {
            for col in 0..size
            {
                let value = values[row * size + col];

                if !value.is_finite()
                {
                    return Err(MedoidClusteringError::NonFiniteDistance { row, col });
                }

                if value < 0.0
                {
                    return Err(MedoidClusteringError::NegativeDistance { row, col });
                }
            }
        }

        for index in 0..size
        {
            if values[index * size + index] != 0.0
            {
                return Err(MedoidClusteringError::NonZeroDiagonal { index });
            }
        }

        for row in 0..size
        {
            for col in (row + 1)..size
            {
                if values[row * size + col] != values[col * size + row]
                {
                    return Err(MedoidClusteringError::AsymmetricPair { row, col });
                }
            }
        }

        Ok(Self { size, values })
    }

    /// The distance between points `i` and `j`.
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.size + j]
    }
}

/// How hard the solver may work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CertifiedClusteringMode {
    /// Unbounded branch and bound: always returns a proven optimum (worst-case
    /// exponential time — intended for tractable instances).
    Exact,
    /// Greedy warm start, a bounded deterministic local-improvement pass, then
    /// branch and bound limited to `maximum_nodes` explored nodes. On budget
    /// exhaustion the incumbent is returned with `proven_optimal = false` and
    /// an explicit gap.
    Hybrid {
        /// Maximum branch-and-bound nodes to explore.
        maximum_nodes: usize,
        /// Maximum local-improvement sweeps over all points.
        maximum_iterations: usize,
    },
}

/// Configuration for [`certified_medoid_clustering`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedMedoidClusteringConfig {
    /// Diameter budget `D`: every intra-cluster pair must satisfy
    /// `d(i, j) ≤ D`. Must be finite and non-negative.
    pub maximum_cluster_diameter: f64,
    /// Solver effort mode.
    pub mode: CertifiedClusteringMode,
}

/// Machine-readable optimality certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusteringCertificate {
    /// Cluster count of the returned partition.
    pub objective_cluster_count: usize,
    /// Total observed-medoid cost of the returned partition.
    pub objective_medoid_cost: f64,
    /// Valid lower bound on the cluster count (greedy incompatibility clique;
    /// equals the count when proven optimal).
    pub lower_bound_cluster_count: usize,
    /// Lower bound on the medoid cost among minimum-count partitions:
    /// `Some(cost)` only when the search completed (the bound is then the
    /// optimum itself); `None` on budget exhaustion — distances are not
    /// required to satisfy the triangle inequality, so no nontrivial partial
    /// cost bound is generally valid, and none is claimed.
    pub lower_bound_medoid_cost: Option<f64>,
    /// `true` only when the branch and bound exhausted the search space.
    pub proven_optimal: bool,
    /// `0.0` when proven optimal. Otherwise: the integer count gap
    /// `objective_cluster_count − lower_bound_cluster_count` as `f64` when
    /// positive; else the conservative cost-gap fraction
    /// `objective_medoid_cost / (1 + objective_medoid_cost)` against the only
    /// generally valid cost bound (`0`).
    pub optimality_gap: f64,
    /// Branch-and-bound nodes explored.
    pub explored_nodes: usize,
    /// Nodes pruned by the incumbent bound.
    pub pruned_nodes: usize,
}

/// A certified partition.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedMedoidClusteringResult {
    /// Canonical cluster label per point (labels numbered by first
    /// occurrence: `assignments[0] == 0`, each new label is the smallest
    /// unused integer).
    pub assignments: Vec<usize>,
    /// Observed medoid point index per cluster label.
    pub medoid_indices: Vec<usize>,
    /// The optimality certificate.
    pub certificate: ClusteringCertificate,
}

/// Typed errors of the certified clustering solver.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MedoidClusteringError {
    /// The matrix has zero points.
    #[error("distance matrix has zero points")]
    EmptyMatrix,
    /// `values.len()` does not equal `size * size`.
    #[error("distance matrix expects {expected} values, found {found}")]
    SizeMismatch {
        /// Expected `size * size`.
        expected: usize,
        /// Supplied length.
        found: usize,
    },
    /// A distance is `NaN` or `±∞`.
    #[error("distance ({row}, {col}) is not finite")]
    NonFiniteDistance {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },
    /// A distance is negative.
    #[error("distance ({row}, {col}) is negative")]
    NegativeDistance {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },
    /// A diagonal entry is not exactly zero.
    #[error("diagonal entry {index} is not exactly zero")]
    NonZeroDiagonal {
        /// The offending diagonal index.
        index: usize,
    },
    /// `values[i, j] != values[j, i]` (the documented symmetry tolerance is
    /// zero).
    #[error("distances ({row}, {col}) and ({col}, {row}) differ")]
    AsymmetricPair {
        /// The row of the asymmetric pair.
        row: usize,
        /// The column of the asymmetric pair.
        col: usize,
    },
    /// The diameter budget is negative or non-finite.
    #[error("maximum cluster diameter must be finite and non-negative")]
    InvalidDiameter,
    /// A hybrid budget is zero.
    #[error("hybrid budgets must be positive")]
    InvalidBudget,
}

/// Solves the certified diameter-constrained medoid clustering problem.
pub fn certified_medoid_clustering(
    distances: &DistanceMatrix,
    config: CertifiedMedoidClusteringConfig,
) -> Result<CertifiedMedoidClusteringResult, MedoidClusteringError> {
    if !config.maximum_cluster_diameter.is_finite() || config.maximum_cluster_diameter < 0.0
    {
        return Err(MedoidClusteringError::InvalidDiameter);
    }

    if let CertifiedClusteringMode::Hybrid {
        maximum_nodes,
        maximum_iterations,
    } = config.mode
    {
        if maximum_nodes == 0 || maximum_iterations == 0
        {
            return Err(MedoidClusteringError::InvalidBudget);
        }
    }

    let n = distances.size;
    let diameter = config.maximum_cluster_diameter;

    // Incompatibility adjacency: true where the pair can never share a
    // cluster.
    let incompatible: Vec<Vec<bool>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| i != j && distances.distance(i, j) > diameter)
                .collect()
        })
        .collect();

    let degrees: Vec<usize> = incompatible
        .iter()
        .map(|row| row.iter().filter(|&&edge| edge).count())
        .collect();

    let lower_bound_cluster_count = greedy_clique_bound(&incompatible, &degrees);

    // Greedy warm start (both modes): deterministic first-fit complete-link in
    // ascending index order — an upper bound, never claimed optimal.
    let mut incumbent = greedy_first_fit(distances, &incompatible);

    if let CertifiedClusteringMode::Hybrid {
        maximum_iterations, ..
    } = config.mode
    {
        local_improve(distances, &incompatible, &mut incumbent, maximum_iterations);
    }

    let node_budget = match config.mode
    {
        CertifiedClusteringMode::Exact => usize::MAX,
        CertifiedClusteringMode::Hybrid { maximum_nodes, .. } => maximum_nodes,
    };

    let mut search = Search {
        distances,
        incompatible: &incompatible,
        degrees: &degrees,
        node_budget,
        explored_nodes: 0,
        pruned_nodes: 0,
        exhausted_budget: false,
        incumbent,
    };

    let mut colors = vec![usize::MAX; n];
    search.branch(&mut colors, 0);

    let Search {
        explored_nodes,
        pruned_nodes,
        exhausted_budget,
        incumbent,
        ..
    } = search;

    let proven_optimal = !exhausted_budget;

    let optimality_gap = if proven_optimal
    {
        0.0
    }
    else if incumbent.count > lower_bound_cluster_count
    {
        (incumbent.count - lower_bound_cluster_count) as f64
    }
    else
    {
        incumbent.cost / (1.0 + incumbent.cost)
    };

    let (assignments, medoid_indices) = canonicalize(distances, &incumbent.colors);

    Ok(CertifiedMedoidClusteringResult {
        assignments,
        medoid_indices,
        certificate: ClusteringCertificate {
            objective_cluster_count: incumbent.count,
            objective_medoid_cost: incumbent.cost,
            lower_bound_cluster_count: if proven_optimal
            {
                incumbent.count
            }
            else
            {
                lower_bound_cluster_count
            },
            lower_bound_medoid_cost: proven_optimal.then_some(incumbent.cost),
            proven_optimal,
            optimality_gap,
            explored_nodes,
            pruned_nodes,
        },
    })
}

/// A complete partition candidate in raw (non-canonical) colors.
#[derive(Clone, Debug)]
struct Candidate {
    colors: Vec<usize>,
    count: usize,
    cost: f64,
}

impl Candidate {
    /// Lexicographic objective comparison: count, then cost (total order via
    /// `total_cmp`), then the canonical assignment vector.
    fn beats(&self, other: &Candidate) -> bool {
        if self.count != other.count
        {
            return self.count < other.count;
        }

        match self.cost.total_cmp(&other.cost)
        {
            core::cmp::Ordering::Less => true,
            core::cmp::Ordering::Greater => false,
            core::cmp::Ordering::Equal =>
            {
                canonical_labels(&self.colors) < canonical_labels(&other.colors)
            },
        }
    }
}

/// Deterministic greedy clique on the incompatibility graph: seed with the
/// highest-degree vertex (ties: smallest index), grow with the
/// highest-degree compatible-with-all vertex. `χ ≥ ω`, so the clique size is a
/// valid cluster-count lower bound.
fn greedy_clique_bound(incompatible: &[Vec<bool>], degrees: &[usize]) -> usize {
    let n = incompatible.len();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| degrees[b].cmp(&degrees[a]).then(a.cmp(&b)));

    let mut clique: Vec<usize> = Vec::new();

    for &vertex in &order
    {
        if clique.iter().all(|&member| incompatible[vertex][member])
        {
            clique.push(vertex);
        }
    }

    clique.len().max(1)
}

/// Deterministic first-fit greedy complete-link in ascending index order: each
/// point joins the first existing cluster it is compatible with, else opens a
/// new one. An upper bound only.
fn greedy_first_fit(distances: &DistanceMatrix, incompatible: &[Vec<bool>]) -> Candidate {
    let n = distances.size;
    let mut colors = vec![usize::MAX; n];
    let mut used = 0usize;

    for vertex in 0..n
    {
        let mut chosen = None;

        for color in 0..used
        {
            let compatible = (0..vertex)
                .filter(|&other| colors[other] == color)
                .all(|other| !incompatible[vertex][other]);

            if compatible
            {
                chosen = Some(color);
                break;
            }
        }

        colors[vertex] = chosen.unwrap_or_else(|| {
            used += 1;
            used - 1
        });
    }

    let cost = partition_cost(distances, &colors, used);

    Candidate {
        colors,
        count: used,
        cost,
    }
}

/// Bounded deterministic local improvement: sweep points in ascending index
/// order; move a point to the smallest-label feasible cluster that strictly
/// improves the lexicographic objective. Stops at a fixed point or after
/// `maximum_iterations` sweeps.
fn local_improve(
    distances: &DistanceMatrix,
    incompatible: &[Vec<bool>],
    incumbent: &mut Candidate,
    maximum_iterations: usize,
) {
    let n = distances.size;

    for _ in 0..maximum_iterations
    {
        let mut improved = false;

        for vertex in 0..n
        {
            let current = incumbent.colors[vertex];

            for target in 0..incumbent.count
            {
                if target == current
                {
                    continue;
                }

                let feasible = (0..n)
                    .filter(|&other| other != vertex && incumbent.colors[other] == target)
                    .all(|other| !incompatible[vertex][other]);

                if !feasible
                {
                    continue;
                }

                let mut trial_colors = incumbent.colors.clone();
                trial_colors[vertex] = target;

                let trial_count = count_used(&trial_colors);
                let trial = Candidate {
                    cost: partition_cost(distances, &trial_colors, incumbent.count),
                    colors: trial_colors,
                    count: trial_count,
                };

                if trial.beats(incumbent)
                {
                    *incumbent = trial;
                    improved = true;
                    break;
                }
            }
        }

        if !improved
        {
            break;
        }
    }
}

fn count_used(colors: &[usize]) -> usize {
    let mut seen: Vec<usize> = Vec::new();

    for &color in colors
    {
        if !seen.contains(&color)
        {
            seen.push(color);
        }
    }

    seen.len()
}

/// Total observed-medoid cost of a complete partition.
fn partition_cost(distances: &DistanceMatrix, colors: &[usize], label_bound: usize) -> f64 {
    let n = distances.size;
    let mut total = 0.0;

    for label in 0..=label_bound
    {
        let members: Vec<usize> = (0..n).filter(|&p| colors[p] == label).collect();

        if members.is_empty()
        {
            continue;
        }

        total += cluster_medoid_cost(distances, &members).1;
    }

    total
}

/// Observed medoid (smallest index on ties) and its total-distance cost.
fn cluster_medoid_cost(distances: &DistanceMatrix, members: &[usize]) -> (usize, f64) {
    let mut best_medoid = members[0];
    let mut best_cost = f64::INFINITY;

    for &candidate in members
    {
        let cost: f64 = members
            .iter()
            .map(|&member| distances.distance(member, candidate))
            .sum();

        if cost.total_cmp(&best_cost).is_lt()
        {
            best_cost = cost;
            best_medoid = candidate;
        }
    }

    (best_medoid, best_cost)
}

/// Canonical first-occurrence relabeling of an arbitrary color vector.
fn canonical_labels(colors: &[usize]) -> Vec<usize> {
    let mut mapping: Vec<(usize, usize)> = Vec::new();
    let mut next = 0usize;

    colors
        .iter()
        .map(|&color| {
            if let Some(&(_, label)) = mapping.iter().find(|&&(raw, _)| raw == color)
            {
                label
            }
            else
            {
                mapping.push((color, next));
                next += 1;
                next - 1
            }
        })
        .collect()
}

/// Canonical assignments plus per-label observed medoids.
fn canonicalize(distances: &DistanceMatrix, colors: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let assignments = canonical_labels(colors);
    let label_count = assignments.iter().copied().max().map_or(0, |m| m + 1);

    let medoids = (0..label_count)
        .map(|label| {
            let members: Vec<usize> = (0..assignments.len())
                .filter(|&p| assignments[p] == label)
                .collect();

            cluster_medoid_cost(distances, &members).0
        })
        .collect();

    (assignments, medoids)
}

/// DSATUR-style branch-and-bound state.
struct Search<'a> {
    distances: &'a DistanceMatrix,
    incompatible: &'a [Vec<bool>],
    degrees: &'a [usize],
    node_budget: usize,
    explored_nodes: usize,
    pruned_nodes: usize,
    exhausted_budget: bool,
    incumbent: Candidate,
}

impl Search<'_> {
    /// Explores assignments of the remaining vertices. `assigned` counts
    /// colored vertices; `colors` uses `usize::MAX` for uncolored.
    fn branch(&mut self, colors: &mut Vec<usize>, assigned: usize) {
        let n = self.incompatible.len();

        if self.exhausted_budget
        {
            return;
        }

        if assigned == n
        {
            let used = count_used(colors);
            let candidate = Candidate {
                cost: partition_cost(self.distances, colors, used),
                colors: colors.clone(),
                count: used,
            };

            if candidate.beats(&self.incumbent)
            {
                self.incumbent = candidate;
            }

            return;
        }

        if self.explored_nodes >= self.node_budget
        {
            self.exhausted_budget = true;
            return;
        }

        self.explored_nodes += 1;

        // DSATUR selection: maximum saturation (distinct neighbour colors),
        // ties by maximum incompatibility degree, then minimum index.
        let vertex = (0..n)
            .filter(|&v| colors[v] == usize::MAX)
            .max_by(|&a, &b| {
                let sat_a = self.saturation(colors, a);
                let sat_b = self.saturation(colors, b);

                sat_a
                    .cmp(&sat_b)
                    .then(self.degrees[a].cmp(&self.degrees[b]))
                    .then(b.cmp(&a))
            })
            .unwrap_or(0);

        let used = colors
            .iter()
            .filter(|&&c| c != usize::MAX)
            .fold(0usize, |max, &c| max.max(c + 1));

        // Prune: even the best completion uses at least `used` clusters; a
        // completion cannot beat an incumbent with fewer clusters, and at
        // equal count the full comparison happens at the leaves.
        if used > self.incumbent.count
        {
            self.pruned_nodes += 1;
            return;
        }

        // Existing colors in ascending order, then at most one fresh color.
        for color in 0..used.min(self.incumbent.count)
        {
            let feasible = (0..n)
                .filter(|&other| colors[other] == color)
                .all(|other| !self.incompatible[vertex][other]);

            if !feasible
            {
                continue;
            }

            colors[vertex] = color;
            self.branch(colors, assigned + 1);
            colors[vertex] = usize::MAX;

            if self.exhausted_budget
            {
                return;
            }
        }

        if used < self.incumbent.count
        {
            colors[vertex] = used;
            self.branch(colors, assigned + 1);
            colors[vertex] = usize::MAX;
        }
        else
        {
            self.pruned_nodes += 1;
        }
    }

    fn saturation(&self, colors: &[usize], vertex: usize) -> usize {
        let mut seen: Vec<usize> = Vec::new();

        for other in 0..colors.len()
        {
            if self.incompatible[vertex][other]
                && colors[other] != usize::MAX
                && !seen.contains(&colors[other])
            {
                seen.push(colors[other]);
            }
        }

        seen.len()
    }
}

/// A certified lower bound on the minimum number of diameter-`D` clusters,
/// obtained from a clique in the incompatibility graph.
///
/// Two points are *incompatible* when `d(i, j) > D`: they can never share a
/// cluster. The minimum cluster count equals the chromatic number `χ` of the
/// incompatibility graph, and `ω ≤ χ` for the clique number `ω`, so **any**
/// clique size is a valid lower bound. [`certified_cluster_count_bound`]
/// strengthens the greedy bound baked into [`ClusteringCertificate`] to a
/// maximum-clique search, so the bound is `ω` — the tightest a clique argument
/// can give — whenever the search completes within budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterCountBound {
    /// The largest clique found: a valid lower bound on the minimum cluster
    /// count.
    pub lower_bound: usize,
    /// `true` when the Bron–Kerbosch enumeration completed within the node
    /// budget, so `lower_bound` is provably the maximum clique `ω` (the tightest
    /// clique bound). `false` when the budget was exhausted first — the bound is
    /// still valid, just not certified maximal.
    pub clique_is_maximum: bool,
    /// Bron–Kerbosch recursion nodes explored.
    pub explored_nodes: usize,
}

impl ClusterCountBound {
    /// Whether this bound **proves** that a partition using `incumbent_count`
    /// clusters is count-optimal.
    ///
    /// A valid clustering has `count ≥ χ ≥ ω = lower_bound`, so
    /// `lower_bound == incumbent_count` squeezes `χ` to exactly
    /// `incumbent_count`. (`lower_bound` can never exceed a valid clustering's
    /// count, so the comparison is effectively equality.) This holds regardless
    /// of `clique_is_maximum` — a matching clique of any size is a proof.
    pub fn certifies_count_optimal(&self, incumbent_count: usize) -> bool {
        self.lower_bound >= incumbent_count
    }
}

/// A certified maximum-clique lower bound on the minimum diameter-`D` cluster
/// count, stronger than the greedy bound in [`certified_medoid_clustering`].
///
/// Builds the incompatibility graph (`d(i, j) > D`), warm-starts with the greedy
/// degree-ordered clique, then runs a deterministic pivoted Bron–Kerbosch
/// enumeration bounded by `maximum_nodes` recursion nodes. If the enumeration
/// completes, `clique_is_maximum` is `true` and the bound is the exact maximum
/// clique `ω`; otherwise the best clique seen so far (never below the greedy
/// warm start) is returned with `clique_is_maximum = false`. Pass
/// `maximum_nodes = usize::MAX` for an unbounded exact search, or `0` to keep
/// only the greedy warm start.
///
/// # Errors
///
/// [`MedoidClusteringError::InvalidDiameter`] when the diameter is negative or
/// non-finite.
pub fn certified_cluster_count_bound(
    distances: &DistanceMatrix,
    maximum_cluster_diameter: f64,
    maximum_nodes: usize,
) -> Result<ClusterCountBound, MedoidClusteringError> {
    if !maximum_cluster_diameter.is_finite() || maximum_cluster_diameter < 0.0
    {
        return Err(MedoidClusteringError::InvalidDiameter);
    }

    let n = distances.size;
    if n == 0
    {
        return Ok(ClusterCountBound {
            lower_bound: 0,
            clique_is_maximum: true,
            explored_nodes: 0,
        });
    }

    // Incompatibility neighbourhoods: j is a neighbour of i iff d(i, j) > D.
    let neighbours: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| i != j && distances.distance(i, j) > maximum_cluster_diameter)
                .collect()
        })
        .collect();
    let adjacency: Vec<Vec<bool>> = (0..n)
        .map(|i| {
            let mut row = vec![false; n];
            for &j in &neighbours[i]
            {
                row[j] = true;
            }
            row
        })
        .collect();
    let degrees: Vec<usize> = neighbours.iter().map(Vec::len).collect();

    // Greedy warm start — always a valid clique.
    let mut best = greedy_clique_bound(&adjacency, &degrees);

    // Exact (bounded) maximum-clique enumeration.
    let mut explored_nodes = 0usize;
    let candidates: Vec<usize> = (0..n).collect();
    let exhausted = bron_kerbosch(
        &adjacency,
        0,
        candidates,
        Vec::new(),
        &mut best,
        &mut explored_nodes,
        maximum_nodes,
    );

    Ok(ClusterCountBound {
        lower_bound: best.max(1),
        clique_is_maximum: !exhausted,
        explored_nodes,
    })
}

/// Pivoted Bron–Kerbosch, tracking the largest clique in `best`. `depth` is the
/// current clique size (the `R` set is represented only by its size, since we
/// only need the maximum clique *count*). Returns `true` if the node budget was
/// hit before the enumeration finished (so the result is not certified maximal).
///
/// Determinism: candidate vertices are visited in ascending index order and the
/// pivot maximises `|P ∩ N(pivot)|`, ties broken by smallest index.
fn bron_kerbosch(
    adjacency: &[Vec<bool>],
    depth: usize,
    candidates: Vec<usize>,
    excluded: Vec<usize>,
    best: &mut usize,
    explored_nodes: &mut usize,
    maximum_nodes: usize,
) -> bool {
    *explored_nodes += 1;
    if *explored_nodes > maximum_nodes
    {
        return true;
    }

    if candidates.is_empty() && excluded.is_empty()
    {
        *best = (*best).max(depth);
        return false;
    }

    // Bound: even taking every remaining candidate cannot beat the incumbent.
    if depth + candidates.len() <= *best
    {
        return false;
    }

    // Pivot maximising |P ∩ N(u)| over P ∪ X (Tomita's rule), ties by index.
    let pivot = candidates
        .iter()
        .chain(excluded.iter())
        .copied()
        .max_by(|&a, &b| {
            let na = candidates.iter().filter(|&&v| adjacency[a][v]).count();
            let nb = candidates.iter().filter(|&&v| adjacency[b][v]).count();
            na.cmp(&nb).then(b.cmp(&a))
        });

    let extend: Vec<usize> = match pivot
    {
        Some(u) => candidates
            .iter()
            .copied()
            .filter(|&v| !adjacency[u][v])
            .collect(),
        None => candidates.clone(),
    };

    let mut candidates = candidates;
    let mut excluded = excluded;
    let mut exhausted = false;
    for vertex in extend
    {
        // Only vertices still in P are branched on.
        if !candidates.contains(&vertex)
        {
            continue;
        }
        let next_candidates: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&v| adjacency[vertex][v])
            .collect();
        let next_excluded: Vec<usize> = excluded
            .iter()
            .copied()
            .filter(|&v| adjacency[vertex][v])
            .collect();
        exhausted |= bron_kerbosch(
            adjacency,
            depth + 1,
            next_candidates,
            next_excluded,
            best,
            explored_nodes,
            maximum_nodes,
        );
        // Move vertex from P to X.
        candidates.retain(|&v| v != vertex);
        excluded.push(vertex);
        if exhausted
        {
            break;
        }
    }
    exhausted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic generator for test fixtures (SplitMix64 core).
    struct TestRng(u64);

    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn matrix(size: usize, entries: &[(usize, usize, f64)]) -> DistanceMatrix {
        let mut values = vec![0.0; size * size];

        for &(i, j, d) in entries
        {
            values[i * size + j] = d;
            values[j * size + i] = d;
        }

        DistanceMatrix::new(size, values).unwrap()
    }

    /// Deterministic pseudo-random metric-free distance matrix.
    fn random_matrix(size: usize, seed: u64) -> DistanceMatrix {
        let mut rng = TestRng(seed);
        let mut values = vec![0.0; size * size];

        for i in 0..size
        {
            for j in (i + 1)..size
            {
                let d = (rng.next_u64() % 1000) as f64 / 100.0;
                values[i * size + j] = d;
                values[j * size + i] = d;
            }
        }

        DistanceMatrix::new(size, values).unwrap()
    }

    fn exact() -> CertifiedMedoidClusteringConfig {
        CertifiedMedoidClusteringConfig {
            maximum_cluster_diameter: 1.0,
            mode: CertifiedClusteringMode::Exact,
        }
    }

    /// Exhaustive oracle: enumerate every set partition via restricted growth
    /// strings, keep diameter-feasible ones, and return the lexicographic
    /// optimum (count, cost, canonical assignment). Test-only; n <= 8.
    fn oracle(distances: &DistanceMatrix, diameter: f64) -> (usize, f64, Vec<usize>) {
        let n = distances.size;
        assert!(n <= 8, "oracle is for tiny instances only");

        let mut best: Option<(usize, f64, Vec<usize>)> = None;
        let mut rgs = vec![0usize; n];

        loop
        {
            // Feasibility: every intra-cluster pair within the diameter.
            let feasible = (0..n).all(|i| {
                ((i + 1)..n).all(|j| rgs[i] != rgs[j] || distances.distance(i, j) <= diameter)
            });

            if feasible
            {
                let count = rgs.iter().copied().max().unwrap_or(0) + 1;
                let cost = partition_cost(distances, &rgs, count);
                let labels = canonical_labels(&rgs);

                let candidate = (count, cost, labels);

                let better = match &best
                {
                    None => true,
                    Some((best_count, best_cost, best_labels)) =>
                    {
                        candidate.0 < *best_count
                            || (candidate.0 == *best_count
                                && (candidate.1.total_cmp(best_cost).is_lt()
                                    || (candidate.1.total_cmp(best_cost).is_eq()
                                        && candidate.2 < *best_labels)))
                    },
                };

                if better
                {
                    best = Some(candidate);
                }
            }

            // Next restricted growth string.
            let mut position = n;
            let mut advanced = false;

            while position > 1
            {
                position -= 1;

                let cap = rgs[..position].iter().copied().max().unwrap_or(0) + 1;

                if rgs[position] < cap
                {
                    rgs[position] += 1;

                    for entry in rgs.iter_mut().skip(position + 1)
                    {
                        *entry = 0;
                    }

                    advanced = true;
                    break;
                }
            }

            if !advanced
            {
                break;
            }
        }

        best.expect("at least the all-singletons partition is feasible")
    }

    #[test]
    fn validation_rejects_malformed_matrices() {
        assert_eq!(
            DistanceMatrix::new(0, Vec::new()),
            Err(MedoidClusteringError::EmptyMatrix)
        );
        assert_eq!(
            DistanceMatrix::new(2, vec![0.0; 3]),
            Err(MedoidClusteringError::SizeMismatch {
                expected: 4,
                found: 3
            })
        );
        assert_eq!(
            DistanceMatrix::new(2, vec![0.0, f64::NAN, f64::NAN, 0.0]),
            Err(MedoidClusteringError::NonFiniteDistance { row: 0, col: 1 })
        );
        assert_eq!(
            DistanceMatrix::new(2, vec![0.0, -1.0, -1.0, 0.0]),
            Err(MedoidClusteringError::NegativeDistance { row: 0, col: 1 })
        );
        assert_eq!(
            DistanceMatrix::new(2, vec![1.0, 2.0, 2.0, 0.0]),
            Err(MedoidClusteringError::NonZeroDiagonal { index: 0 })
        );
        assert_eq!(
            DistanceMatrix::new(2, vec![0.0, 1.0, 2.0, 0.0]),
            Err(MedoidClusteringError::AsymmetricPair { row: 0, col: 1 })
        );

        let valid = matrix(2, &[(0, 1, 1.0)]);

        assert_eq!(
            certified_medoid_clustering(
                &valid,
                CertifiedMedoidClusteringConfig {
                    maximum_cluster_diameter: f64::NAN,
                    mode: CertifiedClusteringMode::Exact,
                },
            ),
            Err(MedoidClusteringError::InvalidDiameter)
        );
        assert_eq!(
            certified_medoid_clustering(
                &valid,
                CertifiedMedoidClusteringConfig {
                    maximum_cluster_diameter: 1.0,
                    mode: CertifiedClusteringMode::Hybrid {
                        maximum_nodes: 0,
                        maximum_iterations: 1,
                    },
                },
            ),
            Err(MedoidClusteringError::InvalidBudget)
        );
    }

    #[test]
    fn single_point_is_one_proven_cluster() {
        let result = certified_medoid_clustering(&matrix(1, &[]), exact()).unwrap();

        assert_eq!(result.assignments, vec![0]);
        assert_eq!(result.medoid_indices, vec![0]);
        assert!(result.certificate.proven_optimal);
        assert_eq!(result.certificate.objective_cluster_count, 1);
        assert_eq!(result.certificate.optimality_gap, 0.0);
    }

    #[test]
    fn all_compatible_points_form_one_cluster() {
        let distances = matrix(
            4,
            &[
                (0, 1, 0.5),
                (0, 2, 0.5),
                (0, 3, 0.5),
                (1, 2, 0.5),
                (1, 3, 0.5),
                (2, 3, 0.5),
            ],
        );

        let result = certified_medoid_clustering(&distances, exact()).unwrap();

        assert_eq!(result.assignments, vec![0, 0, 0, 0]);
        assert_eq!(result.certificate.objective_cluster_count, 1);
        assert!(result.certificate.proven_optimal);
    }

    #[test]
    fn all_incompatible_points_are_singletons() {
        let distances = matrix(3, &[(0, 1, 5.0), (0, 2, 5.0), (1, 2, 5.0)]);

        let result = certified_medoid_clustering(&distances, exact()).unwrap();

        assert_eq!(result.assignments, vec![0, 1, 2]);
        assert_eq!(result.certificate.objective_cluster_count, 3);
        assert_eq!(result.certificate.lower_bound_cluster_count, 3);
        assert!(result.certificate.proven_optimal);
        assert_eq!(result.certificate.objective_medoid_cost, 0.0);
    }

    #[test]
    fn greedy_suboptimal_bridge_is_repaired_by_the_exact_search() {
        // First-fit in index order puts 1 with 0, forcing {2} and {3} apart:
        // three clusters. The optimum pairs {0,1} differently: {0,2} and
        // {1,3} give two clusters.
        let distances = matrix(
            4,
            &[
                (0, 1, 0.9),
                (0, 2, 0.8),
                (0, 3, 5.0),
                (1, 2, 5.0),
                (1, 3, 0.8),
                (2, 3, 5.0),
            ],
        );

        let greedy = greedy_first_fit(
            &distances,
            &[
                vec![false, false, false, true],
                vec![false, false, true, false],
                vec![false, true, false, true],
                vec![true, false, true, false],
            ],
        );

        assert_eq!(greedy.count, 3);

        let result = certified_medoid_clustering(&distances, exact()).unwrap();

        assert_eq!(result.certificate.objective_cluster_count, 2);
        assert_eq!(result.assignments, vec![0, 1, 0, 1]);
        assert!(result.certificate.proven_optimal);
    }

    #[test]
    fn medoid_cost_refines_equal_cluster_counts() {
        // Both {0,1},{2} and {0},{1,2} are feasible two-cluster partitions;
        // the pair {1,2} is cheaper (0.2 < 1.0), so the optimum keeps it.
        let distances = matrix(3, &[(0, 1, 1.0), (1, 2, 0.2), (0, 2, 5.0)]);

        let result = certified_medoid_clustering(&distances, exact()).unwrap();

        assert_eq!(result.certificate.objective_cluster_count, 2);
        assert_eq!(result.assignments, vec![0, 1, 1]);
        assert_eq!(result.certificate.objective_medoid_cost, 0.2);
        assert!(result.certificate.proven_optimal);
    }

    #[test]
    fn tied_optima_resolve_to_the_lexicographically_smallest_assignment() {
        // Perfect square of side 1 with diagonal 2: the two-cluster optima
        // {0,1},{2,3} and {0,3},{1,2} tie on cost; the canonical winner is
        // [0,0,1,1].
        let distances = matrix(
            4,
            &[
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (0, 3, 1.0),
                (0, 2, 2.0),
                (1, 3, 2.0),
            ],
        );

        let result = certified_medoid_clustering(&distances, exact()).unwrap();

        assert_eq!(result.certificate.objective_cluster_count, 2);
        assert_eq!(result.assignments, vec![0, 0, 1, 1]);
        assert!(result.certificate.proven_optimal);
    }

    #[test]
    fn exact_solver_matches_the_exhaustive_oracle() {
        for size in 2..=7usize
        {
            for seed in 0..12u64
            {
                let distances = random_matrix(size, 0xC0DE + seed * 31 + size as u64);

                for diameter in [2.0, 5.0, 8.0]
                {
                    let (count, cost, labels) = oracle(&distances, diameter);

                    let result = certified_medoid_clustering(
                        &distances,
                        CertifiedMedoidClusteringConfig {
                            maximum_cluster_diameter: diameter,
                            mode: CertifiedClusteringMode::Exact,
                        },
                    )
                    .unwrap();

                    assert!(result.certificate.proven_optimal);
                    assert_eq!(
                        result.certificate.objective_cluster_count, count,
                        "count mismatch: size={size} seed={seed} diameter={diameter}"
                    );
                    assert_eq!(
                        result.certificate.objective_medoid_cost.to_bits(),
                        cost.to_bits(),
                        "cost mismatch: size={size} seed={seed} diameter={diameter}"
                    );
                    assert_eq!(
                        result.assignments, labels,
                        "assignment mismatch: size={size} seed={seed} diameter={diameter}"
                    );
                }
            }
        }
    }

    #[test]
    fn hybrid_budget_exhaustion_returns_a_gapped_certificate() {
        let distances = random_matrix(9, 0xBEEF);

        let result = certified_medoid_clustering(
            &distances,
            CertifiedMedoidClusteringConfig {
                maximum_cluster_diameter: 4.0,
                mode: CertifiedClusteringMode::Hybrid {
                    maximum_nodes: 3,
                    maximum_iterations: 2,
                },
            },
        )
        .unwrap();

        assert!(!result.certificate.proven_optimal);
        assert!(result.certificate.optimality_gap > 0.0);
        assert!(result.certificate.lower_bound_medoid_cost.is_none());
        assert!(result.certificate.explored_nodes <= 3);

        // The returned partition is still feasible.
        let n = distances.size;

        for i in 0..n
        {
            for j in (i + 1)..n
            {
                if result.assignments[i] == result.assignments[j]
                {
                    assert!(distances.distance(i, j) <= 4.0);
                }
            }
        }
    }

    #[test]
    fn hybrid_with_ample_budget_proves_optimality_with_zero_gap() {
        let distances = random_matrix(6, 0xFACE);

        let exact_result = certified_medoid_clustering(
            &distances,
            CertifiedMedoidClusteringConfig {
                maximum_cluster_diameter: 4.0,
                mode: CertifiedClusteringMode::Exact,
            },
        )
        .unwrap();

        let hybrid = certified_medoid_clustering(
            &distances,
            CertifiedMedoidClusteringConfig {
                maximum_cluster_diameter: 4.0,
                mode: CertifiedClusteringMode::Hybrid {
                    maximum_nodes: 1_000_000,
                    maximum_iterations: 4,
                },
            },
        )
        .unwrap();

        assert!(hybrid.certificate.proven_optimal);
        assert_eq!(hybrid.certificate.optimality_gap, 0.0);
        assert_eq!(hybrid.assignments, exact_result.assignments);
        assert_eq!(
            hybrid.certificate.objective_medoid_cost.to_bits(),
            exact_result.certificate.objective_medoid_cost.to_bits()
        );
    }

    #[test]
    fn solver_is_deterministic() {
        let distances = random_matrix(7, 0xD00D);

        let config = CertifiedMedoidClusteringConfig {
            maximum_cluster_diameter: 3.0,
            mode: CertifiedClusteringMode::Exact,
        };

        let first = certified_medoid_clustering(&distances, config).unwrap();
        let second = certified_medoid_clustering(&distances, config).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn zero_diameter_groups_only_exact_duplicates() {
        let mut values = vec![0.0; 16];
        // Points 0 and 2 coincide (distance zero); everything else is apart.
        for (i, j, d) in [
            (0usize, 1usize, 1.0f64),
            (0, 2, 0.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ]
        {
            values[i * 4 + j] = d;
            values[j * 4 + i] = d;
        }

        let distances = DistanceMatrix::new(4, values).unwrap();

        let result = certified_medoid_clustering(
            &distances,
            CertifiedMedoidClusteringConfig {
                maximum_cluster_diameter: 0.0,
                mode: CertifiedClusteringMode::Exact,
            },
        )
        .unwrap();

        assert_eq!(result.assignments, vec![0, 1, 0, 2]);
        assert_eq!(result.certificate.objective_cluster_count, 3);
        assert!(result.certificate.proven_optimal);
    }

    /// Incompatibility graph engineered so the degree-ordered greedy clique
    /// (size 3) misses the maximum clique (size 4): a `K4` among {0,1,2,3}, plus
    /// a hub 4 of higher degree whose neighbourhood contains only a triangle.
    fn greedy_fooling_matrix() -> DistanceMatrix {
        let n = 8;
        let far = 2.0; // > diameter 1 -> incompatible
        let mut values = vec![0.0; n * n];
        let mark = |values: &mut [f64], i: usize, j: usize| {
            values[i * n + j] = far;
            values[j * n + i] = far;
        };
        // K4 clique among 0,1,2,3.
        for i in 0..4
        {
            for j in (i + 1)..4
            {
                mark(&mut values, i, j);
            }
        }
        // Hub 4 incompatible with 0,1 and the three leaves 5,6,7 (degree 5).
        for &j in &[0, 1, 5, 6, 7]
        {
            mark(&mut values, 4, j);
        }
        DistanceMatrix::new(n, values).unwrap()
    }

    #[test]
    fn exact_bound_exceeds_greedy_and_certifies_optimality() {
        let distances = greedy_fooling_matrix();

        // The greedy bound baked into the standard certificate stops at 3.
        let greedy = certified_medoid_clustering(
            &distances,
            CertifiedMedoidClusteringConfig {
                maximum_cluster_diameter: 1.0,
                mode: CertifiedClusteringMode::Hybrid {
                    maximum_nodes: 1,
                    maximum_iterations: 1,
                },
            },
        )
        .unwrap();
        assert_eq!(greedy.certificate.lower_bound_cluster_count, 3);

        // The maximum-clique bound reaches 4 and certifies it.
        let strong = certified_cluster_count_bound(&distances, 1.0, usize::MAX).unwrap();
        assert_eq!(strong.lower_bound, 4);
        assert!(strong.clique_is_maximum);
        assert!(strong.lower_bound > greedy.certificate.lower_bound_cluster_count);
        assert!(strong.certifies_count_optimal(4));
        assert!(!strong.certifies_count_optimal(5));
    }

    #[test]
    fn all_compatible_needs_one_cluster() {
        // Every off-diagonal distance is within the diameter -> empty
        // incompatibility graph -> a single cluster suffices.
        let distances = DistanceMatrix::new(5, vec![0.0; 25]).unwrap();
        let bound = certified_cluster_count_bound(&distances, 1.0, usize::MAX).unwrap();
        assert_eq!(bound.lower_bound, 1);
        assert!(bound.clique_is_maximum);
    }

    #[test]
    fn all_incompatible_needs_n_clusters() {
        // Every distinct pair exceeds the diameter -> complete incompatibility
        // graph -> the maximum clique is all n points.
        let n = 6;
        let mut values = vec![2.0; n * n];
        for i in 0..n
        {
            values[i * n + i] = 0.0;
        }
        let distances = DistanceMatrix::new(n, values).unwrap();
        let bound = certified_cluster_count_bound(&distances, 1.0, usize::MAX).unwrap();
        assert_eq!(bound.lower_bound, n);
        assert!(bound.clique_is_maximum);
        assert!(bound.certifies_count_optimal(n));
    }

    #[test]
    fn zero_budget_keeps_the_greedy_warm_start() {
        let distances = greedy_fooling_matrix();
        let bound = certified_cluster_count_bound(&distances, 1.0, 0).unwrap();
        // Only the greedy warm start ran; still a valid (if not maximal) bound.
        assert_eq!(bound.lower_bound, 3);
        assert!(!bound.clique_is_maximum);
    }

    #[test]
    fn count_bound_is_deterministic_and_validates_diameter() {
        let distances = greedy_fooling_matrix();
        let first = certified_cluster_count_bound(&distances, 1.0, usize::MAX).unwrap();
        let second = certified_cluster_count_bound(&distances, 1.0, usize::MAX).unwrap();
        assert_eq!(first, second);

        assert_eq!(
            certified_cluster_count_bound(&distances, -1.0, usize::MAX),
            Err(MedoidClusteringError::InvalidDiameter)
        );
        assert_eq!(
            certified_cluster_count_bound(&distances, f64::NAN, usize::MAX),
            Err(MedoidClusteringError::InvalidDiameter)
        );
    }
}
