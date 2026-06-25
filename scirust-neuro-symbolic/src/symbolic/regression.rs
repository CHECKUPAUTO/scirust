use crate::core::{Reasoner, ReasoningError, Result};
use scirust_symbolic::Expr;

/// Symbolic regression via **multivariate linear least squares**.
///
/// Fits `y ≈ b0 + b1·x0 + b2·x1 + …` by solving the normal equations and returns
/// the model as a symbolic [`Expr`] over variables `x0, x1, …` — *all* supplied
/// input columns are fit. For non-linear, structure-discovering search (genetic
/// programming with gradient-fit constants), see the dedicated `scirust-symreg`
/// crate.
pub struct NeuralSymbolicRegression {
    /// Maximum number of input features (columns) the model will fit. Fitting
    /// data with more columns than this budget is rejected with an error rather
    /// than silently dropping dimensions.
    max_complexity: usize,
}

impl NeuralSymbolicRegression {
    pub fn new(max_complexity: usize) -> Self {
        Self { max_complexity }
    }

    /// Fit a linear model to `(x, y)` and return it as a symbolic expression.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64]) -> Result<Expr> {
        if x.is_empty() || x.len() != y.len()
        {
            return Err(ReasoningError::Symbolic(
                "empty or mismatched training data".into(),
            ));
        }
        let n_features = x[0].len();
        if n_features > self.max_complexity.max(1)
        {
            return Err(ReasoningError::Symbolic(format!(
                "input has {n_features} features but max_complexity is {}",
                self.max_complexity
            )));
        }
        let p = n_features + 1; // + intercept

        // Normal equations: (AᵀA) b = Aᵀy
        let mut ata = vec![vec![0.0f64; p]; p];
        let mut aty = vec![0.0f64; p];
        for (row, &yi) in x.iter().zip(y)
        {
            let mut feats = Vec::with_capacity(p);
            feats.push(1.0);
            feats.extend(row.iter().take(n_features).copied());
            if feats.len() != p
            {
                return Err(ReasoningError::Symbolic("ragged feature rows".into()));
            }
            for i in 0..p
            {
                aty[i] += feats[i] * yi;
                for k in 0..p
                {
                    ata[i][k] += feats[i] * feats[k];
                }
            }
        }

        let coeffs = solve_linear_system(ata, aty)
            .ok_or_else(|| ReasoningError::Symbolic("singular normal equations".into()))?;

        // Build b0 + b1·x0 + b2·x1 + …
        let mut expr = Expr::Const(coeffs[0]);
        for (j, &c) in coeffs.iter().enumerate().skip(1)
        {
            let term = Expr::Mul(
                Box::new(Expr::Const(c)),
                Box::new(Expr::Var(format!("x{}", j - 1))),
            );
            expr = Expr::Add(Box::new(expr), Box::new(term));
        }
        Ok(expr)
    }
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
// Rows `r` and `col` of `a` are borrowed in the same statement; an iterator
// rewrite would need split_at_mut for no readability gain.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n
    {
        // pivot
        let mut pivot = col;
        for r in (col + 1)..n
        {
            if a[r][col].abs() > a[pivot][col].abs()
            {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12
        {
            return None; // singular
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        // eliminate
        for r in 0..n
        {
            if r == col
            {
                continue;
            }
            let factor = a[r][col] / a[col][col];
            for c in col..n
            {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

impl Reasoner for NeuralSymbolicRegression {
    fn name(&self) -> &str {
        "NeuralSymbolicRegression"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn recovers_affine_law() {
        // y = 2*x0 + 1
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let reg = NeuralSymbolicRegression::new(4);
        let expr = reg.fit(&x, &y).unwrap();

        let mut vars = HashMap::new();
        vars.insert("x0".to_string(), 5.0);
        let pred = scirust_symbolic::eval(&expr, &vars).unwrap();
        assert!((pred - 11.0).abs() < 1e-6, "expected ~11, got {pred}");
    }

    #[test]
    fn rejects_empty_data() {
        let reg = NeuralSymbolicRegression::new(2);
        assert!(reg.fit(&[], &[]).is_err());
    }

    #[test]
    fn regression_recovers_two_feature_plane() {
        // Data lie exactly on y = 1 + 2*x0 + 5*x1.
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let y = vec![1.0, 3.0, 6.0, 8.0];
        let reg = NeuralSymbolicRegression::new(4);
        let expr = reg.fit(&x, &y).unwrap();

        let mut vars = HashMap::new();
        vars.insert("x0".to_string(), 2.0);
        vars.insert("x1".to_string(), 3.0);
        let pred = scirust_symbolic::eval(&expr, &vars).unwrap();
        // 1 + 2*2 + 5*3 = 20
        assert!((pred - 20.0).abs() < 1e-6, "expected ~20, got {pred}");
    }

    #[test]
    fn rejects_more_features_than_budget() {
        // 2 input columns but a budget of 1 ⇒ explicit error, not silent drop.
        let reg = NeuralSymbolicRegression::new(1);
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let y = vec![0.0, 2.0];
        assert!(reg.fit(&x, &y).is_err());
    }
}
