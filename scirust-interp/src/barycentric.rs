//! Barycentric-form Lagrange interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::validate_nodes;

/// Global polynomial interpolant in the numerically stable barycentric form.
///
/// This represents the unique degree-`n-1` polynomial through the `n` nodes,
/// evaluated with the second (true) barycentric formula. The weights are
/// computed with a capacity scaling `(x_max - x_min) / 4` per factor to keep
/// their magnitudes near unity and avoid overflow.
///
/// **Extrapolation** is polynomial: outside the node range the value is that of
/// the interpolating polynomial, which — as with any high-degree polynomial —
/// can grow rapidly. Interior evaluation is well behaved for modest `n`.
#[derive(Debug, Clone)]
pub struct BarycentricLagrange {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Barycentric weights.
    w: Vec<f64>,
}

impl BarycentricLagrange {
    /// Build a barycentric-Lagrange interpolant.
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        let n = xs.len();
        let cap = (xs[n - 1] - xs[0]) / 4.0;
        let w: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(j, &xj)| {
                let mut wj = 1.0;
                for (k, &xk) in xs.iter().enumerate()
                {
                    if k != j
                    {
                        wj *= cap / (xj - xk);
                    }
                }
                wj
            })
            .collect();
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            w,
        })
    }
}

impl Interpolator for BarycentricLagrange {
    fn eval(&self, x: f64) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for ((&xj, &yj), &wj) in self.xs.iter().zip(&self.ys).zip(&self.w)
        {
            let diff = x - xj;
            if diff == 0.0
            {
                // Exactly on a node: return that ordinate (avoids 0/0).
                return yj;
            }
            let t = wj / diff;
            num += t * yj;
            den += t;
        }
        num / den
    }
}
