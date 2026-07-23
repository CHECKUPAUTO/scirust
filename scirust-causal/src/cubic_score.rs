use crate::error::CausalError;
use scirust_solvers::Matrix;

pub struct CubicCausalScore;

impl CubicCausalScore {
    fn validate(samples: &Matrix, interactions: &Matrix) -> Result<(usize, usize), CausalError> {
        let (m, d) = samples.shape();
        let (a_rows, a_cols) = interactions.shape();

        if m == 0
        {
            return Err(CausalError::ZeroSamples);
        }
        if d == 0
        {
            return Err(CausalError::ZeroDimension);
        }
        if a_rows != a_cols
        {
            return Err(CausalError::NotSquare {
                rows: a_rows,
                cols: a_cols,
            });
        }
        if d != a_rows
        {
            return Err(CausalError::DimensionMismatch {
                expected: d,
                got: a_rows,
            });
        }

        for (i, &v) in samples.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteInput { index: i, value: v });
            }
        }

        for row in 0..a_rows
        {
            for col in 0..a_cols
            {
                let v = interactions[(row, col)];
                if !v.is_finite()
                {
                    return Err(CausalError::NonFiniteWeight { row, col, value: v });
                }
            }
        }

        Ok((m, d))
    }

    fn check_alloc(m: usize, d: usize) -> Result<(), CausalError> {
        if m.checked_mul(d).is_none()
        {
            return Err(CausalError::AllocationOverflow);
        }
        if d.checked_mul(d).is_none()
        {
            return Err(CausalError::AllocationOverflow);
        }
        Ok(())
    }

    pub fn loss(samples: &Matrix, interactions: &Matrix) -> Result<f64, CausalError> {
        let (m, _d) = Self::validate(samples, interactions)?;

        let z = Self::compute_z(samples, interactions)?;
        let r = Self::compute_r(samples, &z)?;

        let squared_sum: f64 = r.data().iter().map(|v| v * v).sum();

        if !squared_sum.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "cubic_score_loss",
                index: 0,
                value: squared_sum,
            });
        }

        Ok(squared_sum / (2.0 * m as f64))
    }

    pub fn loss_and_gradient(
        samples: &Matrix,
        interactions: &Matrix,
    ) -> Result<(f64, Matrix), CausalError> {
        let (m, _d) = Self::validate(samples, interactions)?;
        Self::check_alloc(m, interactions.rows())?;

        let a_t = interactions.transpose();
        let z = samples
            .matmul(&a_t)
            .map_err(|_| CausalError::DimensionMismatch {
                expected: samples.cols(),
                got: a_t.rows(),
            })?;

        for (i, &v) in z.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "cubic_score_z",
                    index: i,
                    value: v,
                });
            }
        }

        let r = Self::compute_r_inner(samples, &z);

        for (i, &v) in r.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "cubic_score_r",
                    index: i,
                    value: v,
                });
            }
        }

        let squared_sum: f64 = r.data().iter().map(|v| v * v).sum();
        if !squared_sum.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "cubic_score_squared_sum",
                index: 0,
                value: squared_sum,
            });
        }
        let loss = squared_sum / (2.0 * m as f64);

        let (rows, cols) = z.shape();
        let mut z_sq = Matrix::zeros(rows, cols);
        for (i, v) in z_sq.data_mut().iter_mut().enumerate()
        {
            *v = z.data()[i] * z.data()[i];
        }

        let mut rz2 = Matrix::zeros(rows, cols);
        for (i, v) in rz2.data_mut().iter_mut().enumerate()
        {
            *v = r.data()[i] * z_sq.data()[i];
        }

        let rz2_t = rz2.transpose();
        let grad_raw = rz2_t
            .matmul(samples)
            .map_err(|_| CausalError::DimensionMismatch {
                expected: rz2_t.cols(),
                got: samples.rows(),
            })?;

        let scale = 3.0 / m as f64;
        let mut grad = grad_raw;
        for v in grad.data_mut().iter_mut()
        {
            *v *= scale;
        }

        for (i, &v) in grad.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                let row = i / grad.cols();
                return Err(CausalError::NonFiniteComputation {
                    operation: "cubic_score_gradient",
                    index: row,
                    value: v,
                });
            }
        }

        Ok((loss, grad))
    }

    fn compute_z(samples: &Matrix, interactions: &Matrix) -> Result<Matrix, CausalError> {
        let a_t = interactions.transpose();
        Self::check_alloc(samples.rows(), interactions.rows())?;
        let z = samples
            .matmul(&a_t)
            .map_err(|_| CausalError::DimensionMismatch {
                expected: samples.cols(),
                got: a_t.rows(),
            })?;

        for (i, &v) in z.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "cubic_score_z",
                    index: i,
                    value: v,
                });
            }
        }

        Ok(z)
    }

    fn compute_r(samples: &Matrix, z: &Matrix) -> Result<Matrix, CausalError> {
        let (rows, cols) = samples.shape();
        let mut r = Matrix::zeros(rows, cols);
        for (i, v) in r.data_mut().iter_mut().enumerate()
        {
            let cubed = z.data()[i] * z.data()[i] * z.data()[i];
            *v = samples.data()[i] + cubed;
        }

        for (i, &v) in r.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "cubic_score_r",
                    index: i,
                    value: v,
                });
            }
        }

        Ok(r)
    }

    fn compute_r_inner(samples: &Matrix, z: &Matrix) -> Matrix {
        let (rows, cols) = samples.shape();
        let mut r = Matrix::zeros(rows, cols);
        for (i, v) in r.data_mut().iter_mut().enumerate()
        {
            let cubed = z.data()[i] * z.data()[i] * z.data()[i];
            *v = samples.data()[i] + cubed;
        }
        r
    }
}
