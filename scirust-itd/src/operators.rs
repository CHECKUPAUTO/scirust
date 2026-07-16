//! Deterministic finite-difference and quadrature operators on 2-D fields.
//!
//! These reproduce the reference simulator's spatial operators:
//!
//! * [`gradient`] matches NumPy's `gradient(..., edge_order=2)` — central
//!   differences in the interior, one-sided second-order differences at the
//!   edges (finite mode), or circular central differences (periodic mode), for
//!   both uniform and non-uniform rectilinear axes.
//! * [`vorticity`] is the 2-D curl `ω = ∂v_y/∂x − ∂v_x/∂y`.
//! * [`spatial_mean`] is the domain integral over the domain area (2-D
//!   trapezoidal quadrature; an arithmetic mean in periodic mode).
//! * [`bounded`] is the saturating map `b(x) = x / (1 + x)`.

use crate::error::{ItdError, Result};
use crate::field::Field2;
use crate::geometry::{BoundaryMode, Geometry};

/// The saturating projection `b(x) = x / (1 + x)`, clamping the argument to be
/// non-negative first, so the result lies in `[0, 1)`.
#[inline]
pub fn bounded(x: f64) -> f64 {
    let v = if x > 0.0 { x } else { 0.0 };
    v / (1.0 + v)
}

// --- 1-D building blocks -------------------------------------------------

/// Second-order gradient of `v` on a uniform grid of spacing `h`.
fn grad1d_uniform(v: &[f64], h: f64, boundary: BoundaryMode) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0; n];
    match boundary
    {
        BoundaryMode::Periodic =>
        {
            for k in 0..n
            {
                let fwd = v[(k + 1) % n];
                let bwd = v[(k + n - 1) % n];
                out[k] = (fwd - bwd) / (2.0 * h);
            }
        },
        BoundaryMode::Finite =>
        {
            out[0] = (-1.5 * v[0] + 2.0 * v[1] - 0.5 * v[2]) / h;
            for k in 1..n - 1
            {
                out[k] = (v[k + 1] - v[k - 1]) / (2.0 * h);
            }
            out[n - 1] = (1.5 * v[n - 1] - 2.0 * v[n - 2] + 0.5 * v[n - 3]) / h;
        },
    }
    out
}

/// Second-order gradient of `v` on a non-uniform axis with coordinates
/// `coords` (finite boundaries only — this is NumPy's non-uniform
/// `edge_order=2` formula).
fn grad1d_coords(v: &[f64], coords: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0; n];

    // Left edge.
    {
        let dx1 = coords[1] - coords[0];
        let dx2 = coords[2] - coords[1];
        let a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2));
        let b = (dx1 + dx2) / (dx1 * dx2);
        let c = -dx1 / (dx2 * (dx1 + dx2));
        out[0] = a * v[0] + b * v[1] + c * v[2];
    }

    // Interior.
    for k in 1..n - 1
    {
        let dx1 = coords[k] - coords[k - 1];
        let dx2 = coords[k + 1] - coords[k];
        let a = -dx2 / (dx1 * (dx1 + dx2));
        let b = (dx2 - dx1) / (dx1 * dx2);
        let c = dx1 / (dx2 * (dx1 + dx2));
        out[k] = a * v[k - 1] + b * v[k] + c * v[k + 1];
    }

    // Right edge.
    {
        let dx1 = coords[n - 2] - coords[n - 3];
        let dx2 = coords[n - 1] - coords[n - 2];
        let a = dx2 / (dx1 * (dx1 + dx2));
        let b = -(dx2 + dx1) / (dx1 * dx2);
        let c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2));
        out[n - 1] = a * v[n - 3] + b * v[n - 2] + c * v[n - 1];
    }

    out
}

/// Trapezoidal integral of `a` with uniform spacing `h` (summation left to
/// right, matching NumPy's `trapezoid`).
fn trapz_uniform(a: &[f64], h: f64) -> f64 {
    let mut acc = 0.0;
    for k in 0..a.len() - 1
    {
        acc += 0.5 * (a[k] + a[k + 1]);
    }
    acc * h
}

/// Trapezoidal integral of `a` sampled at the coordinates `coords`.
fn trapz_coords(a: &[f64], coords: &[f64]) -> f64 {
    let mut acc = 0.0;
    for k in 0..a.len() - 1
    {
        acc += (coords[k + 1] - coords[k]) * 0.5 * (a[k] + a[k + 1]);
    }
    acc
}

// --- public operators ----------------------------------------------------

fn require_grid_for_derivative(field: &Field2, geometry: &Geometry) -> Result<()> {
    geometry.validate_field(field)?;
    let (ny, nx) = field.shape();
    if ny.min(nx) < 3
    {
        return Err(ItdError::TooFewPoints(
            "gradient/vorticity need at least three points per direction".into(),
        ));
    }
    if !field.all_finite()
    {
        return Err(ItdError::NonFinite(
            "field contains a non-finite value".into(),
        ));
    }
    Ok(())
}

/// Returns `(∂field/∂y, ∂field/∂x)` — the gradient along the row axis and the
/// column axis respectively, matching the reference `(d/dy, d/dx)` order.
pub fn gradient(
    field: &Field2,
    geometry: &Geometry,
    boundary: BoundaryMode,
) -> Result<(Field2, Field2)> {
    require_grid_for_derivative(field, geometry)?;
    let (ny, nx) = field.shape();

    if let (BoundaryMode::Periodic, Geometry::Rectilinear { .. }) = (boundary, geometry)
    {
        return Err(ItdError::UnsupportedBoundary(
            "periodic gradient requires a uniform grid".into(),
        ));
    }

    let mut grad_x = Field2::zeros(ny, nx);
    let mut grad_y = Field2::zeros(ny, nx);

    // Gradient along x (axis 1): one 1-D pass per row.
    let mut row = vec![0.0; nx];
    for i in 0..ny
    {
        for j in 0..nx
        {
            row[j] = field.get(i, j);
        }
        let gx = match geometry
        {
            Geometry::Uniform { dx, .. } => grad1d_uniform(&row, *dx, boundary),
            Geometry::Rectilinear { x, .. } => grad1d_coords(&row, x),
        };
        for j in 0..nx
        {
            *grad_x.get_mut(i, j) = gx[j];
        }
    }

    // Gradient along y (axis 0): one 1-D pass per column.
    let mut col = vec![0.0; ny];
    for j in 0..nx
    {
        for i in 0..ny
        {
            col[i] = field.get(i, j);
        }
        let gy = match geometry
        {
            Geometry::Uniform { dy, .. } => grad1d_uniform(&col, *dy, boundary),
            Geometry::Rectilinear { y, .. } => grad1d_coords(&col, y),
        };
        for i in 0..ny
        {
            *grad_y.get_mut(i, j) = gy[i];
        }
    }

    Ok((grad_y, grad_x))
}

/// The 2-D curl `ω = ∂v_y/∂x − ∂v_x/∂y` of a velocity field `(vx, vy)`.
pub fn vorticity(
    vx: &Field2,
    vy: &Field2,
    geometry: &Geometry,
    boundary: BoundaryMode,
) -> Result<Field2> {
    if vx.shape() != vy.shape()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "velocity components differ: {:?} vs {:?}",
            vx.shape(),
            vy.shape()
        )));
    }
    let (_, dvy_dx) = gradient(vy, geometry, boundary)?;
    let (dvx_dy, _) = gradient(vx, geometry, boundary)?;
    dvy_dx.zip_map(&dvx_dy, |a, b| a - b)
}

/// The spatial mean of `field`: the domain integral divided by the domain
/// area. In finite mode this is a 2-D trapezoidal quadrature (supporting
/// non-uniform rectilinear grids); in periodic mode it is the arithmetic mean.
pub fn spatial_mean(field: &Field2, geometry: &Geometry, boundary: BoundaryMode) -> Result<f64> {
    geometry.validate_field(field)?;
    let (ny, nx) = field.shape();
    if ny.min(nx) < 2
    {
        return Err(ItdError::TooFewPoints(
            "spatial mean needs at least two points per direction".into(),
        ));
    }
    if !field.all_finite()
    {
        return Err(ItdError::NonFinite(
            "field contains a non-finite value".into(),
        ));
    }

    if boundary == BoundaryMode::Periodic
    {
        let sum: f64 = field.as_slice().iter().sum();
        return Ok(sum / (ny * nx) as f64);
    }

    match geometry
    {
        Geometry::Uniform { dx, dy } =>
        {
            let height = (ny - 1) as f64 * dy;
            let width = (nx - 1) as f64 * dx;
            let area = height * width;
            let mut row = vec![0.0; nx];
            let mut integral_x = vec![0.0; ny];
            for i in 0..ny
            {
                for j in 0..nx
                {
                    row[j] = field.get(i, j);
                }
                integral_x[i] = trapz_uniform(&row, *dx);
            }
            Ok(trapz_uniform(&integral_x, *dy) / area)
        },
        Geometry::Rectilinear { x, y } =>
        {
            let width = x[x.len() - 1] - x[0];
            let height = y[y.len() - 1] - y[0];
            let area = width * height;
            let mut row = vec![0.0; nx];
            let mut integral_x = vec![0.0; ny];
            for i in 0..ny
            {
                for j in 0..nx
                {
                    row[j] = field.get(i, j);
                }
                integral_x[i] = trapz_coords(&row, x);
            }
            Ok(trapz_coords(&integral_x, y) / area)
        },
    }
}
