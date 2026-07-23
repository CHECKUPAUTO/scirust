use crate::error::CausalError;
use scirust_graph::dag::CausalDag;
use scirust_solvers::Matrix;

pub struct GraphExtractionConfig {
    pub edge_threshold: f64,
}

impl GraphExtractionConfig {
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
            if interactions[(row, col)].abs() > config.edge_threshold
            {
                dag.add_directed_edge(col, row)
                    .map_err(|_| CausalError::CyclicGraph)?;
            }
        }
    }

    Ok(dag)
}
