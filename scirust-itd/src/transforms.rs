//! Cartesian geometric transforms and the covariance of a sampled field.
//!
//! A real orthogonal `2 × 2` matrix `Q` (a rotation, `det = +1`, or a
//! reflection, `det = −1`) acts on a field sampled on a uniform Cartesian grid
//! by the covariance law
//!
//! ```text
//! scalar:  f_Q(x) = f(Qᵀ(x − o) + o)
//! vector:  v_Q(x) = Q · v(Qᵀ(x − o) + o)
//! ```
//!
//! where `o` is a fixed origin. [`BilinearTransformPlan`] precomputes, once, the
//! source location of every target node and how to sample it, then applies the
//! transform to any number of fields on that grid.
//!
//! Some orthogonal transforms map every grid node exactly onto another node —
//! the identity, the `±90°` and `180°` rotations of a square grid, and the
//! square's reflections. In that case bilinear interpolation would inject
//! avoidable round-off and numerical diffusion, so the plan detects it and
//! substitutes an **exact node permutation** (the analogue of the transport
//! module's exact-snap short circuit). Otherwise it falls back to periodic-free
//! bilinear interpolation, filling target nodes whose source falls outside the
//! domain with a configurable `fill_value`.

use crate::error::{ItdError, Result};
use crate::field::Field2;

/// A validated real orthogonal `2 × 2` matrix (rotation or reflection), stored
/// row-major as `[[m00, m01], [m10, m11]]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orthogonal2 {
    m: [[f64; 2]; 2],
}

impl Orthogonal2 {
    /// Validates a `2 × 2` matrix as orthogonal: finite, `QᵀQ = I` and
    /// `|det| = 1`, both to an absolute tolerance of `1e-12` (matching the
    /// reference `validate_orthogonal_matrix`).
    pub fn new(m: [[f64; 2]; 2]) -> Result<Self> {
        if !m.iter().flatten().all(|v| v.is_finite())
        {
            return Err(ItdError::NonFinite("orthogonal matrix".into()));
        }
        const TOL: f64 = 1.0e-12;
        // gram = QᵀQ, gram[a][b] = Σ_k m[k][a] * m[k][b].
        for a in 0..2
        {
            for b in 0..2
            {
                let gram = m[0][a] * m[0][b] + m[1][a] * m[1][b];
                let expected = if a == b { 1.0 } else { 0.0 };
                if (gram - expected).abs() > TOL
                {
                    return Err(ItdError::InvalidGeometry(
                        "matrix must be orthogonal: QᵀQ = I".into(),
                    ));
                }
            }
        }
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if (det.abs() - 1.0).abs() > TOL
        {
            return Err(ItdError::InvalidGeometry(
                "an orthogonal transform must have determinant ±1".into(),
            ));
        }
        Ok(Orthogonal2 { m })
    }

    /// The direct rotation matrix `[[cosθ, −sinθ], [sinθ, cosθ]]`.
    pub fn rotation(angle_radians: f64) -> Result<Self> {
        if !angle_radians.is_finite()
        {
            return Err(ItdError::NonFinite("rotation angle".into()));
        }
        let (sine, cosine) = angle_radians.sin_cos();
        Self::new([[cosine, -sine], [sine, cosine]])
    }

    /// The matrix entries, row-major `[[m00, m01], [m10, m11]]`.
    #[inline]
    pub fn as_array(&self) -> [[f64; 2]; 2] {
        self.m
    }

    /// The determinant (`+1` for a rotation, `−1` for a reflection).
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }

    /// True when this is a proper rotation (`det ≈ +1`).
    #[inline]
    pub fn is_rotation(&self) -> bool {
        (self.determinant() - 1.0).abs() <= 1.0e-12
    }
}

/// Applies `Qᵀ` to coordinate arrays: the source coordinates
/// `(source_x, source_y) = Qᵀ (x, y)` (the reference `transform_coordinates`).
///
/// `x` and `y` must have equal length; the result has the same length.
pub fn transform_coordinates(
    x: &[f64],
    y: &[f64],
    q: &Orthogonal2,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if x.len() != y.len()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "coordinate arrays differ in length: {} vs {}",
            x.len(),
            y.len()
        )));
    }
    let m = q.m;
    let sx = x
        .iter()
        .zip(y.iter())
        .map(|(&xv, &yv)| m[0][0] * xv + m[1][0] * yv)
        .collect();
    let sy = x
        .iter()
        .zip(y.iter())
        .map(|(&xv, &yv)| m[0][1] * xv + m[1][1] * yv)
        .collect();
    Ok((sx, sy))
}

/// Validates a uniform axis (finite, strictly increasing, evenly spaced) and
/// returns `(first_coordinate, spacing)`.
fn validate_uniform_axis(coords: &[f64], name: &str) -> Result<(f64, f64)> {
    if coords.len() < 2
    {
        return Err(ItdError::TooFewPoints(format!(
            "axis {name} needs at least two coordinates"
        )));
    }
    if !coords.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite(format!("axis {name} coordinates")));
    }
    let spacing = coords[1] - coords[0];
    if spacing <= 0.0
    {
        return Err(ItdError::InvalidGeometry(format!(
            "axis {name} coordinates must be strictly increasing"
        )));
    }
    let atol = 64.0 * f64::EPSILON * spacing.abs().max(1.0);
    for w in coords.windows(2)
    {
        let d = w[1] - w[0];
        if d <= 0.0
        {
            return Err(ItdError::InvalidGeometry(format!(
                "axis {name} coordinates must be strictly increasing"
            )));
        }
        if (d - spacing).abs() > atol + 1.0e-12 * spacing.abs()
        {
            return Err(ItdError::InvalidGeometry(format!(
                "axis {name} must be uniformly sampled"
            )));
        }
    }
    Ok((coords[0], spacing))
}

/// A deterministic plan to transform any field sampled on a fixed uniform
/// Cartesian grid by a fixed orthogonal transform about a fixed origin.
///
/// Built once with [`BilinearTransformPlan::new`], then applied with
/// [`transform_scalar`](Self::transform_scalar) /
/// [`transform_vector`](Self::transform_vector) to as many fields as needed.
#[derive(Debug, Clone)]
pub struct BilinearTransformPlan {
    x0: f64,
    y0: f64,
    nx: usize,
    ny: usize,
    matrix: Orthogonal2,
    fill_value: f64,
    exact_node_map: bool,
    inside: Vec<bool>,
    ix0: Vec<usize>,
    iy0: Vec<usize>,
    tx: Vec<f64>,
    ty: Vec<f64>,
    exact_ix: Vec<usize>,
    exact_iy: Vec<usize>,
}

impl BilinearTransformPlan {
    /// Builds a transform plan for a uniform grid with the given per-axis
    /// coordinates (`x_coords` length `nx`, `y_coords` length `ny`), orthogonal
    /// `matrix`, `origin` and out-of-domain `fill_value`.
    pub fn new(
        x_coords: &[f64],
        y_coords: &[f64],
        matrix: Orthogonal2,
        origin: [f64; 2],
        fill_value: f64,
    ) -> Result<Self> {
        let (x0, dx) = validate_uniform_axis(x_coords, "x")?;
        let (y0, dy) = validate_uniform_axis(y_coords, "y")?;
        if !origin.iter().all(|v| v.is_finite())
        {
            return Err(ItdError::NonFinite("transform origin".into()));
        }
        if !fill_value.is_finite()
        {
            return Err(ItdError::NonFinite("fill value".into()));
        }
        let nx = x_coords.len();
        let ny = y_coords.len();
        let m = matrix.m;

        let count = ny * nx;
        let mut inside = vec![false; count];
        let mut ix0 = vec![0usize; count];
        let mut iy0 = vec![0usize; count];
        let mut tx = vec![0.0f64; count];
        let mut ty = vec![0.0f64; count];
        let mut exact_ix = vec![0usize; count];
        let mut exact_iy = vec![0usize; count];

        let inside_tol = 64.0 * f64::EPSILON * (nx.max(ny) as f64);
        let exact_tol = 256.0 * f64::EPSILON * (nx.max(ny) as f64);
        let mut node_aligned = true;

        for i in 0..ny
        {
            for j in 0..nx
            {
                let k = i * nx + j;
                let rel_x = x_coords[j] - origin[0];
                let rel_y = y_coords[i] - origin[1];
                let source_x = origin[0] + m[0][0] * rel_x + m[1][0] * rel_y;
                let source_y = origin[1] + m[0][1] * rel_x + m[1][1] * rel_y;

                let norm_x = (source_x - x0) / dx;
                let norm_y = (source_y - y0) / dy;

                let is_inside = norm_x >= -inside_tol
                    && norm_x <= (nx - 1) as f64 + inside_tol
                    && norm_y >= -inside_tol
                    && norm_y <= (ny - 1) as f64 + inside_tol;
                inside[k] = is_inside;

                let clipped_x = norm_x.clamp(0.0, (nx - 1) as f64);
                let clipped_y = norm_y.clamp(0.0, (ny - 1) as f64);
                let cx = (clipped_x.floor() as usize).min(nx - 2);
                let cy = (clipped_y.floor() as usize).min(ny - 2);
                ix0[k] = cx;
                iy0[k] = cy;
                tx[k] = clipped_x - cx as f64;
                ty[k] = clipped_y - cy as f64;

                let rounded_x = norm_x.round();
                let rounded_y = norm_y.round();
                if !is_inside
                    || (norm_x - rounded_x).abs() > exact_tol
                    || (norm_y - rounded_y).abs() > exact_tol
                {
                    node_aligned = false;
                }
                // Clamp defensively; when node_aligned holds these are already
                // valid grid indices.
                exact_ix[k] = (rounded_x.max(0.0) as usize).min(nx - 1);
                exact_iy[k] = (rounded_y.max(0.0) as usize).min(ny - 1);
            }
        }

        Ok(BilinearTransformPlan {
            x0,
            y0,
            nx,
            ny,
            matrix,
            fill_value,
            exact_node_map: node_aligned,
            inside,
            ix0,
            iy0,
            tx,
            ty,
            exact_ix,
            exact_iy,
        })
    }

    /// True when every target node's source is exactly a grid node, so the
    /// transform is an exact permutation rather than an interpolation.
    #[inline]
    pub fn uses_exact_node_map(&self) -> bool {
        self.exact_node_map
    }

    /// The `(ny, nx)` grid shape the plan operates on.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Samples `field` at every target node's source location (an exact node
    /// permutation when [`uses_exact_node_map`](Self::uses_exact_node_map),
    /// otherwise bilinear with `fill_value` outside the domain).
    pub fn interpolate(&self, field: &Field2) -> Result<Field2> {
        if field.shape() != (self.ny, self.nx)
        {
            return Err(ItdError::ShapeMismatch(format!(
                "field {:?} does not match plan grid {:?}",
                field.shape(),
                (self.ny, self.nx)
            )));
        }
        if !field.all_finite()
        {
            return Err(ItdError::NonFinite("field to transform".into()));
        }

        let mut out = Field2::zeros(self.ny, self.nx);
        for i in 0..self.ny
        {
            for j in 0..self.nx
            {
                let k = i * self.nx + j;
                let value = if self.exact_node_map
                {
                    field.get(self.exact_iy[k], self.exact_ix[k])
                }
                else if self.inside[k]
                {
                    let i0 = self.iy0[k];
                    let j0 = self.ix0[k];
                    let i1 = i0 + 1;
                    let j1 = j0 + 1;
                    let tx = self.tx[k];
                    let ty = self.ty[k];
                    let omtx = 1.0 - tx;
                    let omty = 1.0 - ty;
                    omtx * omty * field.get(i0, j0)
                        + tx * omty * field.get(i0, j1)
                        + omtx * ty * field.get(i1, j0)
                        + tx * ty * field.get(i1, j1)
                }
                else
                {
                    self.fill_value
                };
                *out.get_mut(i, j) = value;
            }
        }
        // Silence unused-field warnings on the retained origin metadata.
        let _ = (self.x0, self.y0);
        Ok(out)
    }

    /// Transforms a scalar field: `f_Q(x) = f(Qᵀ(x − o) + o)`.
    pub fn transform_scalar(&self, field: &Field2) -> Result<Field2> {
        self.interpolate(field)
    }

    /// Transforms a vector field: samples each component at the source location,
    /// then rotates the sampled vector by `Q`:
    /// `v_Q(x) = Q · v(Qᵀ(x − o) + o)`.
    pub fn transform_vector(&self, vx: &Field2, vy: &Field2) -> Result<(Field2, Field2)> {
        let source_vx = self.interpolate(vx)?;
        let source_vy = self.interpolate(vy)?;
        let m = self.matrix.m;
        let out_vx = source_vx.zip_map(&source_vy, |sx, sy| m[0][0] * sx + m[0][1] * sy)?;
        let out_vy = source_vx.zip_map(&source_vy, |sx, sy| m[1][0] * sx + m[1][1] * sy)?;
        Ok((out_vx, out_vy))
    }
}
