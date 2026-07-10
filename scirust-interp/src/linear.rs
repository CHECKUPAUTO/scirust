//! Piecewise-linear interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, validate_nodes};

/// Piecewise-linear interpolant through the given nodes.
///
/// Between adjacent nodes the value is the straight-line segment joining them.
/// **Extrapolation is linear**: for queries outside `[xs[0], xs[n - 1]]` the
/// slope of the nearest boundary segment is continued, so an affine input is
/// reproduced exactly everywhere, including outside the node range.
#[derive(Debug, Clone)]
pub struct LinearInterp {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl LinearInterp {
    /// Build a linear interpolant.
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
        })
    }
}

impl Interpolator for LinearInterp {
    fn eval(&self, x: f64) -> f64 {
        let i = find_segment(&self.xs, x);
        let (x0, x1) = (self.xs[i], self.xs[i + 1]);
        let (y0, y1) = (self.ys[i], self.ys[i + 1]);
        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}
