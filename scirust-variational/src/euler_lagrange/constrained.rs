use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct HolonomicConstraint {
    pub name: String,
    pub num_coordinates: usize,
}

pub struct ConstrainedEulerLagrange;

impl ConstrainedEulerLagrange {
    pub fn solve_augmented<F, G>(
        lagrangian: &F,
        constraints: &[G],
        q: &[f32],
        dq: &[f32],
        t: Option<f32>,
        num_constraints: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>
    where
        F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
        G: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
    {
        let n = q.len();
        let m = num_constraints;

        if m == 0
        {
            return Err(VariationalError::UnsupportedOperation {
                details: "no constraints provided".into(),
            });
        }

        let tape = NdTape::new();
        let q_vars: Vec<NdVar<'_>> = (0..n)
            .map(|i| tape.input(TensorND::new(vec![q[i]], vec![1, 1])))
            .collect();
        let dq_vars: Vec<NdVar<'_>> = (0..n)
            .map(|i| tape.input(TensorND::new(vec![dq[i]], vec![1, 1])))
            .collect();
        let tv = t.map(|tv| tape.input(TensorND::new(vec![tv], vec![1, 1])));

        let L = lagrangian(&tape, &q_vars, &dq_vars, tv);

        let mut constraint_residuals = Vec::with_capacity(m);
        for constraint in constraints
        {
            let c = constraint(&tape, &q_vars, tv);
            constraint_residuals.push(c);
        }

        let L_grads = tape.backward(L);

        let dL_dq: Vec<f32> = L_grads[..n].iter().map(|g| g.data[0]).collect();

        let eps = 1e-4;
        let mut jacobian = vec![vec![0.0; n]; m];
        for j in 0..n
        {
            let mut q_plus = q.to_vec();
            q_plus[j] += eps;
            let tape_j = NdTape::new();
            let qv_plus: Vec<NdVar<'_>> = (0..n)
                .map(|i| tape_j.input(TensorND::new(vec![q_plus[i]], vec![1, 1])))
                .collect();
            for (k, constraint) in constraints.iter().enumerate()
            {
                let c_plus = constraint(&tape_j, &qv_plus, tv);
                let c_val_plus = tape_j.value(c_plus).data[0];

                let mut q_minus = q.to_vec();
                q_minus[j] -= eps;
                let tape_j2 = NdTape::new();
                let qv_minus: Vec<NdVar<'_>> = (0..n)
                    .map(|i| tape_j2.input(TensorND::new(vec![q_minus[i]], vec![1, 1])))
                    .collect();
                let c_minus = constraint(&tape_j2, &qv_minus, tv);
                let c_val_minus = tape_j2.value(c_minus).data[0];

                jacobian[k][j] = (c_val_plus - c_val_minus) / (2.0 * eps);
            }
        }

        for (k, jac_row) in jacobian.iter().enumerate()
        {
            let norm: f32 = jac_row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm < 1e-10
            {
                return Err(VariationalError::ConstraintViolation {
                    constraint: format!("constraint {k}"),
                    residual: norm,
                });
            }
        }

        let mut augmented_matrix = vec![vec![0.0; n + m]; n + m];
        for i in 0..n
        {
            for j in 0..n
            {
                augmented_matrix[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }
        for k in 0..m
        {
            for j in 0..n
            {
                augmented_matrix[n + k][j] = jacobian[k][j];
                augmented_matrix[j][n + k] = jacobian[k][j];
            }
        }

        let mut rhs = vec![0.0; n + m];
        rhs[..n].copy_from_slice(&dL_dq);
        rhs[n..n + m].fill(0.0);

        let mut solved = rhs.clone();
        let mut x = vec![0.0; n + m];

        for col in 0..n + m
        {
            let mut max_row = col;
            let mut max_val = augmented_matrix[col][col].abs();
            for row in col + 1..n + m
            {
                let val = augmented_matrix[row][col].abs();
                if val > max_val
                {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-14
            {
                continue;
            }
            if max_row != col
            {
                augmented_matrix.swap(col, max_row);
                solved.swap(col, max_row);
            }
            for row in col + 1..n + m
            {
                let factor = augmented_matrix[row][col] / augmented_matrix[col][col];
                for k in col..=n + m - 1
                {
                    if k < n + m
                    {
                        augmented_matrix[row][k] -= factor * augmented_matrix[col][k];
                    }
                }
                solved[row] -= factor * solved[col];
            }
        }

        for i in (0..n + m).rev()
        {
            let mut sum = solved[i];
            for j in i + 1..n + m
            {
                sum -= augmented_matrix[i][j] * x[j];
            }
            x[i] = if augmented_matrix[i][i].abs() > 1e-14
            {
                sum / augmented_matrix[i][i]
            }
            else
            {
                0.0
            };
        }

        let acceleration: Vec<f32> = x[..n].to_vec();
        let multipliers: Vec<f32> = if m > 0 { x[n..].to_vec() } else { Vec::new() };

        Ok((acceleration, multipliers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_particle_lagrangian<'a>(
        tape: &'a NdTape,
        _q: &'a [NdVar<'a>],
        dq: &'a [NdVar<'a>],
        _t: Option<NdVar<'a>>,
    ) -> NdVar<'a> {
        let half = tape.input(TensorND::new(vec![0.5], vec![1, 1]));
        dq[0].mul(dq[0]).mul(half)
    }

    fn circle_constraint<'a>(
        tape: &'a NdTape,
        q: &'a [NdVar<'a>],
        _t: Option<NdVar<'a>>,
    ) -> NdVar<'a> {
        let r = tape.input(TensorND::new(vec![1.0], vec![1, 1]));
        q[0].mul(q[0]).add(q[1].mul(q[1])).sub(r.mul(r))
    }

    #[test]
    fn test_particle_on_circle() {
        let result = ConstrainedEulerLagrange::solve_augmented(
            &free_particle_lagrangian,
            &[circle_constraint],
            &[1.0, 0.0],
            &[0.0, 1.0],
            None,
            1,
        );
        assert!(result.is_ok());
        let (acc, mult) = result.unwrap();
        assert_eq!(acc.len(), 2);
        assert_eq!(mult.len(), 1);
    }
}
