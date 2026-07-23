//! End-to-end exercise of the discovery pipeline: synthetic data → optimize →
//! extract a graph.
//!
//! These tests are deliberately honest about what continuous causal-structure
//! discovery can and cannot deliver. They assert that the *machinery* is sound
//! (finite outputs, a valid acyclic graph, an honest termination reason) — they
//! do **not** assert that the recovered graph equals the data-generating DAG,
//! because on observational data it generally does not. The second test makes
//! that non-identifiability explicit: two different initializations of the same
//! optimizer on the *same data* converge to two different feasible DAGs, neither
//! of which is the true chain. This is the crate's headline caveat (see the
//! `lib.rs` "Causal interpretation" section) turned into an executable check.

use scirust_causal::{
    GraphExtractionConfig, OptimizerConfig, SyntheticDataConfig, TerminationReason,
    TriangularCubicFlow, extract_causal_dag, generate_causal_samples, optimize_causal,
};
use scirust_solvers::Matrix;

const DIM: usize = 3;

/// Deterministic samples from the true chain `0 → 1 → 2`
/// (`A[1,0] = 0.5`, `A[2,1] = 0.3`), seed 42, 20 rows.
fn chain_samples() -> Matrix {
    let true_weights = vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0];
    let flow = TriangularCubicFlow::from_row_major(DIM, true_weights).unwrap();
    let config = SyntheticDataConfig::new(42, DIM, 20).unwrap();
    generate_causal_samples(&flow, &config).unwrap()
}

/// Small sub-diagonal (strictly-lower) start — off the all-zeros saddle.
fn lower_init() -> Matrix {
    Matrix::from_row_major(
        DIM,
        DIM,
        vec![0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05, 0.05, 0.0],
    )
}

/// Small dense off-diagonal start — also off the saddle, but not biased toward
/// the lower triangle.
fn full_init() -> Matrix {
    Matrix::from_row_major(
        DIM,
        DIM,
        vec![0.0, 0.05, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 0.0],
    )
}

#[test]
fn pipeline_runs_and_extracts_a_valid_dag_off_the_saddle() -> Result<(), Box<dyn std::error::Error>>
{
    let samples = chain_samples();
    let initial = lower_init();
    let opt_config = OptimizerConfig::new(100, 20, 1.0e-5, 1.0e-10)?;

    let result = optimize_causal(&samples, &initial, 0.01, 1.0e-8, 0.0, 1.0, &opt_config)?;

    // Every numeric output is finite and coherent.
    assert!(result.objective.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert!(result.acyclicity.is_finite());
    assert!(result.outer_iterations > 0);
    assert!(
        result.inner_iterations > 0,
        "off the saddle, the optimizer must take real descent steps"
    );
    for i in 0..DIM
    {
        for j in 0..DIM
        {
            assert!(
                result.interactions[(i, j)].is_finite(),
                "non-finite coefficient at ({i}, {j})"
            );
        }
    }

    // Started off the all-zeros saddle, the run must NOT report the degenerate
    // stationary-at-initial-point outcome; it does real work.
    assert_ne!(
        result.termination,
        TerminationReason::StationaryAtInitialPoint
    );

    // The extracted graph is a genuine DAG (acyclic, valid topological order).
    // NOTE: we do not assert it equals the true chain — see the module docs and
    // the `recovered_structure_depends_on_initialization` test below.
    let graph_config = GraphExtractionConfig::new(0.1)?;
    let dag = extract_causal_dag(&result.interactions, &graph_config)?;
    assert!(dag.topo_order().is_ok(), "extracted graph is not a DAG");

    Ok(())
}

#[test]
fn recovered_structure_depends_on_initialization() -> Result<(), Box<dyn std::error::Error>> {
    // The honesty test. Same data, same optimizer, same everything — only the
    // starting point differs. If discovery recovered the truth, both runs would
    // land on the same graph (the chain 0 → 1 → 2). They do not: each converges
    // to a *different* feasible DAG. Optimization success is not identification.
    let samples = chain_samples();
    let opt_config = OptimizerConfig::new(100, 20, 1.0e-5, 1.0e-10)?;
    let graph_config = GraphExtractionConfig::new(0.1)?;

    let from_lower = optimize_causal(&samples, &lower_init(), 0.01, 1.0e-8, 0.0, 1.0, &opt_config)?;
    let from_full = optimize_causal(&samples, &full_init(), 0.01, 1.0e-8, 0.0, 1.0, &opt_config)?;

    // Both runs produce valid (acyclic) graphs...
    let dag_lower = extract_causal_dag(&from_lower.interactions, &graph_config)?;
    let dag_full = extract_causal_dag(&from_full.interactions, &graph_config)?;
    assert!(dag_lower.topo_order().is_ok());
    assert!(dag_full.topo_order().is_ok());

    // ...but they disagree on the edges. The lower-triangular start keeps the
    // 2 ← 1 coupling and drops 1 ← 0; the dense start does the opposite. Neither
    // reproduces the true chain {1 ← 0, 2 ← 1}. This divergence is the whole
    // point: on observational data the fitted DAG is a hypothesis, not the
    // ground truth.
    let edge = |a: &Matrix, i: usize, j: usize| a[(i, j)].abs() > 0.1;

    assert!(
        edge(&from_lower.interactions, 2, 1),
        "lower start should keep 2 <- 1"
    );
    assert!(
        !edge(&from_lower.interactions, 1, 0),
        "lower start should drop 1 <- 0"
    );

    assert!(
        edge(&from_full.interactions, 1, 0),
        "dense start should keep 1 <- 0"
    );
    assert!(
        !edge(&from_full.interactions, 2, 1),
        "dense start should drop 2 <- 1"
    );

    // Stated as one fact: the two initializations recover different structures.
    assert_ne!(
        edge(&from_lower.interactions, 1, 0),
        edge(&from_full.interactions, 1, 0),
        "the two initializations must recover different graphs (non-identifiability)"
    );

    Ok(())
}
