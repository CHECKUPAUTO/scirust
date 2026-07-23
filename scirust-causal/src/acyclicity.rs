use crate::error::CausalError;
use scirust_solvers::Matrix;

pub struct PolynomialAcyclicity;

impl PolynomialAcyclicity {
    fn validate(interactions: &Matrix) -> Result<usize, CausalError> {
        let (rows, cols) = interactions.shape();

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
                let v = interactions[(row, col)];
                if !v.is_finite()
                {
                    return Err(CausalError::NonFiniteWeight { row, col, value: v });
                }
            }
        }

        Ok(rows)
    }

    fn element_square(a: &Matrix) -> Result<Matrix, CausalError> {
        let (rows, cols) = a.shape();
        let mut b = Matrix::zeros(rows, cols);
        for (i, v) in b.data_mut().iter_mut().enumerate()
        {
            *v = a.data()[i] * a.data()[i];
        }
        for (i, &v) in b.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                let row = i / cols;
                return Err(CausalError::NonFiniteComputation {
                    operation: "acyclicity_element_square",
                    index: row,
                    value: v,
                });
            }
        }
        Ok(b)
    }

    fn matmul_safe(a: &Matrix, b: &Matrix) -> Result<Matrix, CausalError> {
        let result = a.matmul(b).map_err(|_| CausalError::DimensionMismatch {
            expected: a.cols(),
            got: b.rows(),
        })?;
        for (i, &v) in result.data().iter().enumerate()
        {
            if !v.is_finite()
            {
                let row = i / result.cols();
                return Err(CausalError::NonFiniteComputation {
                    operation: "acyclicity_matmul",
                    index: row,
                    value: v,
                });
            }
        }
        Ok(result)
    }

    fn trace(a: &Matrix) -> f64 {
        let n = a.rows();
        let mut tr = 0.0;
        for i in 0..n
        {
            tr += a[(i, i)];
        }
        tr
    }

    pub fn value(interactions: &Matrix) -> Result<f64, CausalError> {
        let d = Self::validate(interactions)?;
        let b = Self::element_square(interactions)?;

        if d == 1
        {
            let tr = b[(0, 0)];
            if !tr.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "acyclicity_value",
                    index: 0,
                    value: tr,
                });
            }
            return Ok(tr);
        }

        let mut b_pow = b.clone();
        let mut inv_fact = 1.0;
        let mut h = 0.0;

        // k=1: trace(B) / 1!
        {
            let tr = Self::trace(&b);
            if !tr.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "acyclicity_trace",
                    index: 1,
                    value: tr,
                });
            }
            h += tr;
        }

        for k in 2..=d
        {
            inv_fact /= k as f64;
            b_pow = Self::matmul_safe(&b_pow, &b)?;
            let tr = Self::trace(&b_pow);
            if !tr.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "acyclicity_trace",
                    index: k,
                    value: tr,
                });
            }
            h += tr * inv_fact;
        }

        if !h.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "acyclicity_value",
                index: 0,
                value: h,
            });
        }

        Ok(h)
    }

    pub fn value_and_gradient(interactions: &Matrix) -> Result<(f64, Matrix), CausalError> {
        let d = Self::validate(interactions)?;
        let b = Self::element_square(interactions)?;

        let mut b_pow = Matrix::identity(d);
        let mut inv_fact = 1.0;
        let mut accum = Matrix::identity(d);
        let mut h = 0.0;

        // b_pow starts as I = B^0
        // For m = 0: (B^0)^T / 0! = I / 1 = I  (already in accum)
        // Compute B^1 for k=1

        for m in 0..d
        {
            // Add (B^m)^T * inv_fact to accum (for gradient)
            // For m=0, this is handled by initial accum = I
            if m > 0
            {
                let bm_t = b_pow.transpose();
                let (rows, cols) = accum.shape();
                for i in 0..rows
                {
                    for j in 0..cols
                    {
                        accum[(i, j)] += bm_t[(i, j)] * inv_fact;
                    }
                }
            }

            let b_next = if m == 0
            {
                b.clone()
            }
            else
            {
                Self::matmul_safe(&b_pow, &b)?
            };

            let tr = Self::trace(&b_next);
            if !tr.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "acyclicity_trace",
                    index: m + 1,
                    value: tr,
                });
            }

            h += tr * inv_fact / (m + 1) as f64;

            b_pow = b_next;

            inv_fact /= (m + 1) as f64;
        }

        let (rows, cols) = accum.shape();
        let mut grad = Matrix::zeros(rows, cols);
        for i in 0..rows
        {
            for j in 0..cols
            {
                // grad_{ij} = 2·W_{ij}·accum_{ij}, and is exactly zero where
                // W_{ij} = 0. Computing it as literal 0 there avoids a spurious
                // `0 · ∞ = NaN` if `accum` overflowed at a zero-weight entry, so
                // the finiteness guard below is now unconditional and catches a
                // genuine overflow at a nonzero weight.
                let v = if interactions[(i, j)] == 0.0
                {
                    0.0
                }
                else
                {
                    2.0 * interactions[(i, j)] * accum[(i, j)]
                };
                if !v.is_finite()
                {
                    return Err(CausalError::NonFiniteComputation {
                        operation: "acyclicity_gradient",
                        index: i * cols + j,
                        value: v,
                    });
                }
                grad[(i, j)] = v;
            }
        }

        if !h.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "acyclicity_value",
                index: 0,
                value: h,
            });
        }

        Ok((h, grad))
    }
}
