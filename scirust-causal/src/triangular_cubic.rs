use crate::error::CausalError;
use scirust_solvers::Matrix;

/// Exactly invertible cubic flow with a **strictly lower-triangular** weight
/// matrix.
///
/// This implements the map
///
/// ```text
/// y_i = x_i + (Σ_{j < i} A[i,j] · x_j)³
/// ```
///
/// for a strictly lower-triangular `A`. The Jacobian is unit lower triangular
/// with determinant exactly 1 in exact arithmetic. The inverse is obtained by
/// deterministic forward substitution (an exact algebraic formula evaluated in
/// floating-point — not an iterative approximation).
///
/// This is a proper subclass of the Drużkowski maps. No claim is made about
/// general Drużkowski or Nilsson maps, and this implementation does **not**
/// solve or prove the Jacobian conjecture.
///
/// # Example
///
/// ```
/// # use scirust_causal::TriangularCubicFlow;
/// // 2×2 strictly lower-triangular weight matrix (row-major order):
/// //   A = [[0, 0],
/// //        [2, 0]]
/// let flow = TriangularCubicFlow::from_row_major(
///     2,
///     vec![0.0, 0.0, 2.0, 0.0],
/// ).unwrap();
///
/// let x = vec![3.0, 5.0];
/// let y = flow.forward(&x).unwrap();
///
/// // y = x + (A·x)^{∘3}
/// //   = [3, 5] + ([0, 6])³
/// //   = [3, 5 + 216] = [3, 221]
/// assert_eq!(y, vec![3.0, 221.0]);
///
/// // Inverse recovers x exactly (in exact arithmetic):
/// let reconstructed = flow.inverse(&y).unwrap();
/// for (got, expected) in reconstructed.iter().zip(&x) {
///     assert!((got - expected).abs() < 1e-12);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TriangularCubicFlow {
    weights: Matrix,
}

impl TriangularCubicFlow {
    /// Construct a flow after validating that `weights` is square and strictly
    /// lower triangular.
    pub fn new(weights: Matrix) -> Result<Self, CausalError> {
        let (rows, cols) = weights.shape();

        if rows != cols
        {
            return Err(CausalError::NotSquare { rows, cols });
        }

        if rows == 0
        {
            return Err(CausalError::ZeroDimension);
        }

        for row in 0..rows
        {
            for col in 0..cols
            {
                let value = weights[(row, col)];
                if !value.is_finite()
                {
                    return Err(CausalError::NonFiniteWeight { row, col, value });
                }
            }
        }

        for row in 0..rows
        {
            for col in row..cols
            {
                let value = weights[(row, col)];
                if value != 0.0
                {
                    return Err(CausalError::NonStrictLowerTriangular { row, col, value });
                }
            }
        }

        Ok(Self { weights })
    }

    /// Construct a flow from a row-major square matrix.
    pub fn from_row_major(dim: usize, data: Vec<f64>) -> Result<Self, CausalError> {
        let expected = dim.checked_mul(dim).ok_or(CausalError::DimensionMismatch {
            expected: usize::MAX,
            got: data.len(),
        })?;

        if data.len() != expected
        {
            return Err(CausalError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }

        Self::new(Matrix::from_row_major(dim, dim, data))
    }

    /// Flow dimension.
    pub fn dim(&self) -> usize {
        self.weights.rows()
    }

    /// Read-only access to the validated weight matrix.
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    /// Evaluate `y = x + (A x)^{∘3}`.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, CausalError> {
        self.validate_vector(x)?;

        let linear = self.linear_terms(x)?;
        let mut output = x.to_vec();

        for (row, &z) in linear.iter().enumerate()
        {
            output[row] += cube(z);
        }

        for (i, &v) in output.iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "forward",
                    index: i,
                    value: v,
                });
            }
        }

        Ok(output)
    }

    /// Evaluate the exact inverse by forward substitution.
    ///
    /// Because row `i` depends only on coordinates `0..i`, all required inverse
    /// coordinates are already known when coordinate `i` is reconstructed.
    pub fn inverse(&self, y: &[f64]) -> Result<Vec<f64>, CausalError> {
        self.validate_vector(y)?;

        let dim = self.dim();
        let mut x = vec![0.0; dim];

        for row in 0..dim
        {
            let linear: f64 = x[..row]
                .iter()
                .enumerate()
                .map(|(col, &x_col)| self.weights[(row, col)] * x_col)
                .sum();

            if !linear.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "inverse linear",
                    index: row,
                    value: linear,
                });
            }

            x[row] = y[row] - cube(linear);

            if !x[row].is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "inverse",
                    index: row,
                    value: x[row],
                });
            }
        }

        Ok(x)
    }

    /// Return the analytical Jacobian at `x`.
    ///
    /// `J[i,j] = δ[i,j] + 3 z_i² A[i,j]`, where `z = A x`.
    pub fn jacobian(&self, x: &[f64]) -> Result<Matrix, CausalError> {
        self.validate_vector(x)?;

        let dim = self.dim();
        let linear = self.linear_terms(x)?;

        let j = Matrix::from_fn(dim, dim, |row, col| {
            if row == col
            {
                1.0
            }
            else if col < row
            {
                3.0 * linear[row] * linear[row] * self.weights[(row, col)]
            }
            else
            {
                0.0
            }
        });

        for row in 0..dim
        {
            for col in 0..row
            {
                let v = j[(row, col)];
                if !v.is_finite()
                {
                    return Err(CausalError::NonFiniteComputation {
                        operation: "jacobian",
                        index: row * dim + col,
                        value: v,
                    });
                }
            }
        }

        Ok(j)
    }

    /// The exact mathematical value of `log(abs(det(J)))`.
    pub fn log_abs_det_jacobian(&self) -> f64 {
        0.0
    }

    /// Analytical reverse-mode derivative.
    ///
    /// Given `upstream = ∂L/∂y`, returns:
    ///
    /// - `∂L/∂x`;
    /// - `∂L/∂A`, masked to the strictly lower-triangular parameter domain.
    pub fn backward(&self, x: &[f64], upstream: &[f64]) -> Result<(Vec<f64>, Matrix), CausalError> {
        self.validate_vector(x)?;
        self.validate_vector(upstream)?;

        let dim = self.dim();
        let linear = self.linear_terms(x)?;
        let mut grad_x = upstream.to_vec();
        let mut grad_weights = Matrix::zeros(dim, dim);

        for row in 0..dim
        {
            let q = 3.0 * upstream[row] * linear[row] * linear[row];

            if !q.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "backward q",
                    index: row,
                    value: q,
                });
            }

            for col in 0..row
            {
                grad_x[col] += self.weights[(row, col)] * q;
                grad_weights[(row, col)] = q * x[col];
            }
        }

        for (i, &v) in grad_x.iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "backward grad_x",
                    index: i,
                    value: v,
                });
            }
        }

        for row in 0..dim
        {
            for col in 0..row
            {
                let v = grad_weights[(row, col)];

                if !v.is_finite()
                {
                    return Err(CausalError::NonFiniteComputation {
                        operation: "backward grad_weights",
                        index: row * dim + col,
                        value: v,
                    });
                }
            }
        }

        Ok((grad_x, grad_weights))
    }

    fn validate_vector(&self, values: &[f64]) -> Result<(), CausalError> {
        let dim = self.dim();

        if values.len() != dim
        {
            return Err(CausalError::DimensionMismatch {
                expected: dim,
                got: values.len(),
            });
        }

        for (i, &v) in values.iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteInput { index: i, value: v });
            }
        }

        Ok(())
    }

    fn linear_terms(&self, x: &[f64]) -> Result<Vec<f64>, CausalError> {
        let dim = self.dim();
        let mut linear = vec![0.0; dim];

        for (row, linear_val) in linear.iter_mut().enumerate()
        {
            let mut value = 0.0;

            for (col, &x_col) in x.iter().take(row).enumerate()
            {
                value += self.weights[(row, col)] * x_col;
            }

            if !value.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "linear_terms",
                    index: row,
                    value,
                });
            }

            *linear_val = value;
        }

        Ok(linear)
    }
}

#[inline]
fn cube(value: f64) -> f64 {
    value * value * value
}
