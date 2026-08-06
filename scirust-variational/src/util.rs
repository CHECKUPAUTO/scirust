use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};

pub fn nd_tanh<'t>(tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
    let two = tape.input(TensorND::new(vec![2.0], vec![1, 1]));
    let one = tape.input(TensorND::new(vec![1.0], vec![1, 1]));
    x.mul(two).sigmoid().mul(two).sub(one)
}

pub fn finite_difference_gradient<F>(f: &mut F, x: &[f32], eps: f32) -> Vec<f32>
where
    F: FnMut(&[f32]) -> f32,
{
    let n = x.len();
    let _f0 = f(x);
    let mut grad = vec![0.0; n];
    for i in 0..n
    {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let fp = f(&xp);
        let mut xm = x.to_vec();
        xm[i] -= eps;
        let fm = f(&xm);
        grad[i] = (fp - fm) / (2.0 * eps);
    }
    grad
}

pub fn finite_difference_hessian<F>(f: &mut F, x: &[f32], eps: f32, n: usize) -> Vec<Vec<f32>>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let mut h = vec![vec![0.0; n]; n];
    let _f0 = f(x);
    for j in 0..n
    {
        let mut xp = x.to_vec();
        xp[j] += eps;
        let fp = f(&xp);
        let mut xm = x.to_vec();
        xm[j] -= eps;
        let fm = f(&xm);
        for i in 0..n
        {
            h[i][j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    h
}

pub fn solve_linear_system(a: &[Vec<f32>], b: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut augmented = vec![vec![0.0; n + 1]; n];
    for i in 0..n
    {
        for j in 0..n
        {
            augmented[i][j] = a[i][j];
        }
        augmented[i][n] = b[i];
    }

    for col in 0..n
    {
        let mut max_row = col;
        let mut max_val = augmented[col][col].abs();
        for row in col + 1..n
        {
            let val = augmented[row][col].abs();
            if val > max_val
            {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-16
        {
            return Err(VariationalError::SingularVelocityHessian {
                condition_number: f32::INFINITY,
                tolerance: 1e-8,
            });
        }
        if max_row != col
        {
            augmented.swap(col, max_row);
        }
        for row in col + 1..n
        {
            let factor = augmented[row][col] / augmented[col][col];
            for k in col..=n
            {
                augmented[row][k] -= factor * augmented[col][k];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut sum = augmented[i][n];
        for j in i + 1..n
        {
            sum -= augmented[i][j] * x[j];
        }
        if augmented[i][i].abs() < 1e-16
        {
            return Err(VariationalError::LinearSolveFailure {
                details: format!("zero pivot at row {i}"),
            });
        }
        x[i] = sum / augmented[i][i];
    }

    for &v in &x
    {
        if !v.is_finite()
        {
            return Err(VariationalError::NonFiniteValue {
                component: "linear_solve",
                value: v,
            });
        }
    }

    Ok(x)
}
