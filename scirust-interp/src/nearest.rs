//! Nearest-neighbour interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, validate_nodes};

/// Piecewise-constant nearest-neighbour interpolant.
///
/// Each query returns the ordinate of the closest node. Ties (a query exactly
/// halfway between two nodes) resolve to the **lower-indexed (left)** node.
/// This is the only method that accepts a **single** node — in which case every
/// query returns that node's value.
///
/// **Extrapolation** returns the nearest endpoint's value, which is the natural
/// consequence of the nearest-node rule.
#[derive(Debug, Clone)]
pub struct NearestNeighbor {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl NearestNeighbor {
    /// Build a nearest-neighbour interpolant.
    ///
    /// Requires at least **one** node with finite `xs` (strictly increasing
    /// when more than one) and finite `ys` of matching length; otherwise
    /// returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 1)?;
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
        })
    }
}

impl Interpolator for NearestNeighbor {
    fn eval(&self, x: f64) -> f64 {
        if self.xs.len() == 1 {
            return self.ys[0];
        }
        let i = find_segment(&self.xs, x);
        let left = x - self.xs[i];
        let right = self.xs[i + 1] - x;
        // `<=` sends exact midpoints to the left node.
        if left <= right {
            self.ys[i]
        } else {
            self.ys[i + 1]
        }
    }
}
