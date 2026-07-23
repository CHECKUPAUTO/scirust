use crate::error::CausalError;
use scirust_graph::dag::CausalDag;
use scirust_solvers::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct VariablePermutation {
    pub forward: Vec<usize>,
    pub inverse: Vec<usize>,
}

impl VariablePermutation {
    pub fn from_topo_order(order: &[usize]) -> Result<Self, CausalError> {
        let n = order.len();

        for &node in order.iter()
        {
            if node >= n
            {
                return Err(CausalError::InvalidPermutation {
                    detail: "topological order contains out-of-range node index",
                });
            }
        }

        let mut seen = vec![false; n];
        for &node in order
        {
            if seen[node]
            {
                return Err(CausalError::InvalidPermutation {
                    detail: "duplicate node in topological order",
                });
            }
            seen[node] = true;
        }

        let mut inverse = vec![0; n];
        for (pos, &node) in order.iter().enumerate()
        {
            inverse[node] = pos;
        }

        Ok(Self {
            forward: order.to_vec(),
            inverse,
        })
    }

    pub fn from_dag(dag: &CausalDag) -> Result<Self, CausalError> {
        let order = dag.topo_order().map_err(|_| CausalError::CyclicGraph)?;
        Self::from_topo_order(&order)
    }

    pub fn permute_vector(&self, v: &[f64]) -> Result<Vec<f64>, CausalError> {
        if v.len() != self.forward.len()
        {
            return Err(CausalError::DimensionMismatch {
                expected: self.forward.len(),
                got: v.len(),
            });
        }
        let mut out = vec![0.0; v.len()];
        for (new_pos, &old_pos) in self.forward.iter().enumerate()
        {
            out[new_pos] = v[old_pos];
        }
        Ok(out)
    }

    pub fn restore_vector(&self, v: &[f64]) -> Result<Vec<f64>, CausalError> {
        if v.len() != self.forward.len()
        {
            return Err(CausalError::DimensionMismatch {
                expected: self.forward.len(),
                got: v.len(),
            });
        }
        let mut out = vec![0.0; v.len()];
        for (new_pos, &old_pos) in self.forward.iter().enumerate()
        {
            out[old_pos] = v[new_pos];
        }
        Ok(out)
    }

    pub fn permute_matrix(&self, m: &Matrix) -> Result<Matrix, CausalError> {
        let (rows, cols) = m.shape();
        if rows != self.forward.len() || cols != self.forward.len()
        {
            return Err(CausalError::DimensionMismatch {
                expected: self.forward.len(),
                got: rows.max(cols),
            });
        }
        let n = rows;
        let mut out = Matrix::zeros(n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let old_i = self.forward[i];
                let old_j = self.forward[j];
                out[(i, j)] = m[(old_i, old_j)];
            }
        }
        Ok(out)
    }

    pub fn restore_matrix(&self, m: &Matrix) -> Result<Matrix, CausalError> {
        let (rows, cols) = m.shape();
        if rows != self.forward.len() || cols != self.forward.len()
        {
            return Err(CausalError::DimensionMismatch {
                expected: self.forward.len(),
                got: rows.max(cols),
            });
        }
        let n = rows;
        let mut out = Matrix::zeros(n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                out[(self.forward[i], self.forward[j])] = m[(i, j)];
            }
        }
        Ok(out)
    }

    pub fn invert(&self) -> Self {
        Self {
            forward: self.inverse.clone(),
            inverse: self.forward.clone(),
        }
    }
}

/// Reorders `interactions` into `dag`'s topological order. When the matrix's
/// nonzero pattern is consistent with the DAG (every nonzero `A[i, j]` is an edge
/// `j → i`) the permuted matrix is **strictly lower triangular**.
///
/// The result is *verified* strictly lower triangular: any nonzero on or above
/// the diagonal (a self-loop, or an interaction the DAG does not contain) means
/// the `(matrix, dag)` pair is inconsistent, and an error is returned rather than
/// a silently non-triangular matrix. (The permutation only moves values, so this
/// check is exact — no round-off.)
///
/// # Errors
///
/// [`CausalError::InvalidPermutation`] if `dag` is not a valid permutation source
/// or the `(matrix, dag)` pair is inconsistent.
pub fn triangularize_from_dag(
    interactions: &Matrix,
    dag: &CausalDag,
) -> Result<(VariablePermutation, Matrix), CausalError> {
    let perm = VariablePermutation::from_dag(dag)?;
    let tri = perm.permute_matrix(interactions)?;

    let n = tri.rows();
    for i in 0..n
    {
        for j in i..n
        {
            if tri[(i, j)] != 0.0
            {
                return Err(CausalError::InvalidPermutation {
                    detail: "matrix nonzero pattern is inconsistent with the DAG \
                             (triangularized result is not strictly lower triangular)",
                });
            }
        }
    }

    Ok((perm, tri))
}
