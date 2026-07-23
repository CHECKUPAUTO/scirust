//! Threshold a fitted interaction matrix into a [`CausalDag`].
//!
//! **The extracted graph is a hypothesis, not the true causal graph.** It is one
//! representative of an unknown Markov-equivalence class, valid only under the
//! assumptions documented at the crate root (causal sufficiency, faithfulness,
//! correct functional/noise form, adequate sample size), and only when the
//! optimizer that produced `interactions` actually converged — check
//! [`crate::TerminationReason`] first. Extraction performs no identifiability
//! reasoning of its own.

use crate::error::CausalError;
use scirust_graph::dag::CausalDag;
use scirust_solvers::Matrix;

/// Configuration for [`extract_causal_dag`].
pub struct GraphExtractionConfig {
    /// Off-diagonal entries with `|A[i,j]| > edge_threshold` become edges; smaller
    /// entries are dropped by definition of the threshold.
    pub edge_threshold: f64,
}

impl GraphExtractionConfig {
    /// Validates and constructs a configuration.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] when `edge_threshold` is non-finite
    /// or negative.
    pub fn new(edge_threshold: f64) -> Result<Self, CausalError> {
        if !edge_threshold.is_finite()
        {
            return Err(CausalError::InvalidConfiguration {
                name: "edge_threshold",
                value: edge_threshold,
            });
        }
        if edge_threshold < 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "edge_threshold",
                value: edge_threshold,
            });
        }
        Ok(Self { edge_threshold })
    }
}

/// Thresholds `interactions` into a [`CausalDag`], orienting a super-threshold
/// off-diagonal entry `A[row, col]` as the edge `col → row` (column influences
/// row, matching the flow/score parameterization).
///
/// Diagonal entries are **self-loops, not causal edges**, and are ignored:
/// a residual nonzero diagonal is a symptom of an un-converged optimizer (visible
/// in [`crate::TerminationReason`] / the optimizer's warnings), not a graph edge.
/// A genuine directed cycle among the off-diagonal edges is rejected loudly —
/// no edge is ever silently dropped to force acyclicity.
///
/// # Errors
///
/// [`CausalError::NotSquare`] for a non-square matrix, [`CausalError::NonFiniteWeight`]
/// for a non-finite entry, and [`CausalError::CyclicGraph`] when the thresholded
/// off-diagonal pattern contains a directed cycle.
pub fn extract_causal_dag(
    interactions: &Matrix,
    config: &GraphExtractionConfig,
) -> Result<CausalDag, CausalError> {
    let (rows, cols) = interactions.shape();

    if rows != cols
    {
        return Err(CausalError::NotSquare { rows, cols });
    }

    for row in 0..rows
    {
        for col in 0..cols
        {
            let v = interactions[(row, col)];
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteWeight { row, col, value: v });
            }
        }
    }

    let mut dag = CausalDag::new(rows);

    for row in 0..rows
    {
        for col in 0..cols
        {
            // Self-loops are not causal edges; only off-diagonal entries orient.
            if col != row && interactions[(row, col)].abs() > config.edge_threshold
            {
                dag.add_directed_edge(col, row)
                    .map_err(|_| CausalError::CyclicGraph)?;
            }
        }
    }

    Ok(dag)
}
