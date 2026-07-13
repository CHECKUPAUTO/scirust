//! Flat Minkowski spacetime in Cartesian coordinates.

use crate::{Connection, Metric};

/// Four-dimensional Minkowski spacetime with signature `(-,+,+,+)`.
///
/// Coordinates use natural geometric units, with `x^0 = c t`, so the metric
/// is dimensionless and equal to `diag(-1, 1, 1, 1)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Minkowski;

impl Metric<4> for Minkowski {
    fn components(&self, _coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl Connection<4> for Minkowski {
    fn christoffel(&self, _coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        [[[0.0; 4]; 4]; 4]
    }
}
