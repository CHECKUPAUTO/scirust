//! Spatial grids and boundary conventions.

use crate::error::{ItdError, Result};
use crate::field::Field2;

/// The convention used at the domain boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Finite domain: trapezoidal quadrature and one-sided second-order edge
    /// derivatives.
    Finite,
    /// Periodic domain: circular central differences and an arithmetic-mean
    /// integral (the grid carries no duplicated end point).
    Periodic,
}

/// A 2-D Cartesian grid, either uniform (constant `dx`, `dy`) or rectilinear
/// with strictly increasing — possibly non-uniform — per-axis coordinates.
///
/// Row index `i` runs along `y` (axis 0); column index `j` runs along `x`
/// (axis 1).
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// Uniform spacing on both axes.
    Uniform {
        /// Spacing along `x` (axis 1).
        dx: f64,
        /// Spacing along `y` (axis 0).
        dy: f64,
    },
    /// Rectilinear grid defined by explicit, strictly increasing coordinates.
    Rectilinear {
        /// Column coordinates (length `nx`).
        x: Vec<f64>,
        /// Row coordinates (length `ny`).
        y: Vec<f64>,
    },
}

impl Geometry {
    /// A uniform grid with spacings `dx`, `dy`. Both must be finite and
    /// strictly positive.
    pub fn uniform(dx: f64, dy: f64) -> Result<Self> {
        check_spacing(dx, "dx")?;
        check_spacing(dy, "dy")?;
        Ok(Geometry::Uniform { dx, dy })
    }

    /// An isotropic uniform grid (`dx == dy`).
    pub fn isotropic(spacing: f64) -> Result<Self> {
        Self::uniform(spacing, spacing)
    }

    /// A rectilinear grid from explicit coordinate arrays. Each axis must have
    /// at least three finite, strictly increasing coordinates.
    pub fn rectilinear(x: Vec<f64>, y: Vec<f64>) -> Result<Self> {
        check_coords(&x, "x")?;
        check_coords(&y, "y")?;
        Ok(Geometry::Rectilinear { x, y })
    }

    /// Checks that a field's shape is compatible with this geometry. Uniform
    /// grids accept any shape; rectilinear grids must match `(ny, nx)`.
    pub fn validate_field(&self, field: &Field2) -> Result<()> {
        if let Geometry::Rectilinear { x, y } = self
        {
            let expected = (y.len(), x.len());
            if field.shape() != expected
            {
                return Err(ItdError::ShapeMismatch(format!(
                    "field {:?} does not match rectilinear geometry {:?}",
                    field.shape(),
                    expected
                )));
            }
        }
        Ok(())
    }
}

fn check_spacing(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0
    {
        return Err(ItdError::InvalidGeometry(format!(
            "{name} must be finite and strictly positive (got {value})"
        )));
    }
    Ok(())
}

fn check_coords(coords: &[f64], name: &str) -> Result<()> {
    if coords.len() < 3
    {
        return Err(ItdError::TooFewPoints(format!(
            "axis {name} needs at least three coordinates (got {})",
            coords.len()
        )));
    }
    if !coords.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite(format!("axis {name} coordinates")));
    }
    if !coords.windows(2).all(|w| w[1] > w[0])
    {
        return Err(ItdError::InvalidGeometry(format!(
            "axis {name} coordinates must be strictly increasing"
        )));
    }
    Ok(())
}
