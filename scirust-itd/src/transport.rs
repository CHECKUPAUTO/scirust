//! Semi-Lagrangian periodic transport (advection) of a scalar field.
//!
//! This is the machinery behind the reference simulator's
//! *transport-compensated* temporal-deformation mode: before comparing the
//! current vorticity with its predecessor, the predecessor is advected along
//! the flow to the current time, so the temporal term measures genuine
//! deformation rather than mere transport.
//!
//! For each current grid node the departure point `x_src = x − Δt·u` is traced
//! back (single midpoint step or RK4), wrapped periodically into the domain,
//! and the previous field is sampled there by one of the reference's four
//! periodic interpolations: **bilinear**, **16-point cubic** (Lagrange), the
//! convex-limited **`cubic_local_bounded`** (cubic blended toward bilinear so
//! it stays within the local stencil bounds — no overshoot), and the
//! **`cubic_local_sum_preserving`** variant (locally bounded, then a
//! deterministic redistribution so the discrete sum matches the unlimited
//! cubic). Both trajectory methods (`midpoint`, `rk4`) are supported for every
//! interpolation.

use crate::error::{ItdError, Result};
use crate::field::Field2;

/// Periodic interpolation scheme used to sample the previous field at the
/// departure points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Periodic bilinear (4-point) interpolation.
    BilinearPeriodic,
    /// Periodic 16-point cubic Lagrange interpolation (fourth-order for smooth
    /// fields; no monotone limiter).
    CubicPeriodic,
    /// Cubic, then a **convex limiter** toward the bilinear solution so the
    /// result stays within the local bilinear-stencil bounds (no overshoot).
    /// The exact discrete sum is *not* preserved.
    CubicLocalBoundedPeriodic,
    /// The locally-bounded cubic, then a deterministic redistribution over a
    /// periodic neighbourhood so the **discrete sum equals the unlimited
    /// cubic's** — bounded *and* sum-preserving (not a flux-conservative scheme
    /// for a general flow).
    CubicLocalSumPreservingPeriodic,
}

impl Interpolation {
    /// True for the two limiter variants, which build on the cubic sample.
    fn is_limited(self) -> bool {
        matches!(
            self,
            Interpolation::CubicLocalBoundedPeriodic
                | Interpolation::CubicLocalSumPreservingPeriodic
        )
    }

    /// True for every variant that reads a 16-point cubic stencil.
    fn uses_cubic_stencil(self) -> bool {
        !matches!(self, Interpolation::BilinearPeriodic)
    }
}

/// Trajectory (back-tracing) method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trajectory {
    /// Single Euler step using the transport velocity evaluated at the
    /// interval midpoint time.
    MidpointTimeVelocity,
    /// Classical RK4 back-integration of the trajectory.
    Rk4Backtrace,
}

#[inline]
fn wrap(v: f64, origin: f64, period: f64) -> f64 {
    origin + (v - origin).rem_euclid(period)
}

/// Validates a strictly increasing, uniformly spaced periodic axis and returns
/// its spacing (`coords[1] - coords[0]`).
fn uniform_spacing(coords: &[f64], name: &str) -> Result<f64> {
    if coords.len() < 2
    {
        return Err(ItdError::TooFewPoints(format!(
            "periodic axis {name} needs at least two coordinates"
        )));
    }
    if !coords.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite(format!("periodic axis {name}")));
    }
    let spacing = coords[1] - coords[0];
    if spacing <= 0.0
    {
        return Err(ItdError::InvalidGeometry(format!(
            "periodic axis {name} must be strictly increasing"
        )));
    }
    let atol = 64.0 * f64::EPSILON * spacing.abs().max(1.0);
    for w in coords.windows(2)
    {
        let d = w[1] - w[0];
        if d <= 0.0
        {
            return Err(ItdError::InvalidGeometry(format!(
                "periodic axis {name} must be strictly increasing"
            )));
        }
        if (d - spacing).abs() > atol + 1e-12 * spacing.abs()
        {
            return Err(ItdError::InvalidGeometry(format!(
                "periodic axis {name} must be uniformly sampled"
            )));
        }
    }
    Ok(spacing)
}

/// Cubic Lagrange weights for the four relative nodes `-1, 0, 1, 2` at
/// fractional offset `t ∈ [0, 1)`.
#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    [
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ]
}

/// Advects `previous_omega` from `previous_time` to `current_time` along the
/// `velocity` field, sampling by the requested interpolation and trajectory.
///
/// `velocity(x, y, t)` is a pointwise transport velocity `(vx, vy)`.
#[allow(clippy::too_many_arguments)]
pub fn transport_previous_vorticity<VF>(
    previous_omega: &Field2,
    xc: &[f64],
    yc: &[f64],
    previous_time: f64,
    current_time: f64,
    velocity: VF,
    interpolation: Interpolation,
    trajectory: Trajectory,
) -> Result<Field2>
where
    VF: Fn(f64, f64, f64) -> (f64, f64),
{
    let dx = uniform_spacing(xc, "x")?;
    let dy = uniform_spacing(yc, "y")?;
    let (ny, nx) = (yc.len(), xc.len());
    if previous_omega.shape() != (ny, nx)
    {
        return Err(ItdError::ShapeMismatch(format!(
            "transported field {:?} does not match grid {:?}",
            previous_omega.shape(),
            (ny, nx)
        )));
    }
    if !previous_omega.all_finite()
    {
        return Err(ItdError::NonFinite("transported field".into()));
    }
    let dt = current_time - previous_time;
    if !dt.is_finite() || dt <= 0.0
    {
        return Err(ItdError::InvalidGeometry(
            "transport times must be finite and strictly increasing".into(),
        ));
    }
    if interpolation.uses_cubic_stencil() && (nx < 4 || ny < 4)
    {
        return Err(ItdError::TooFewPoints(
            "periodic cubic interpolation needs at least four points per axis".into(),
        ));
    }

    match trajectory
    {
        Trajectory::MidpointTimeVelocity =>
        {
            let t_mid = 0.5 * (previous_time + current_time);
            let vx = Field2::from_fn(ny, nx, |i, j| velocity(xc[j], yc[i], t_mid).0);
            let vy = Field2::from_fn(ny, nx, |i, j| velocity(xc[j], yc[i], t_mid).1);
            backtrace(previous_omega, xc, yc, dx, dy, &vx, &vy, dt, interpolation)
        },
        Trajectory::Rk4Backtrace =>
        {
            let (dep_x, dep_y) =
                rk4_departures(xc, yc, dx, dy, previous_time, current_time, &velocity)?;
            Ok(sample_at_departures(
                previous_omega,
                xc,
                yc,
                dx,
                dy,
                &dep_x,
                &dep_y,
                interpolation,
            ))
        },
    }
}

/// Midpoint back-trace: departure `= grid − Δt·v`, then interpolate. Includes
/// the reference's exact-snap short circuit (departures that land on grid nodes
/// to within tolerance sample the node directly).
#[allow(clippy::too_many_arguments)]
fn backtrace(
    field: &Field2,
    xc: &[f64],
    yc: &[f64],
    dx: f64,
    dy: f64,
    vx: &Field2,
    vy: &Field2,
    dt: f64,
    interpolation: Interpolation,
) -> Result<Field2> {
    let (ny, nx) = field.shape();
    if vx.shape() != (ny, nx) || vy.shape() != (ny, nx)
    {
        return Err(ItdError::ShapeMismatch("transport velocity shape".into()));
    }
    if !vx.all_finite() || !vy.all_finite()
    {
        return Err(ItdError::NonFinite("transport velocity".into()));
    }

    // Limiter variants build on the plain bilinear and cubic back-traces and
    // the local bilinear-stencil bounds around each midpoint departure.
    if interpolation.is_limited()
    {
        let bilinear = backtrace(
            field,
            xc,
            yc,
            dx,
            dy,
            vx,
            vy,
            dt,
            Interpolation::BilinearPeriodic,
        )?;
        let cubic = backtrace(
            field,
            xc,
            yc,
            dx,
            dy,
            vx,
            vy,
            dt,
            Interpolation::CubicPeriodic,
        )?;
        let (lower, upper) = midpoint_departure_bounds(field, xc, yc, dx, dy, vx, vy, dt);
        return Ok(combine_limited(
            &bilinear,
            &cubic,
            &lower,
            &upper,
            interpolation,
        ));
    }

    let period_x = dx * nx as f64;
    let period_y = dy * ny as f64;
    let x0 = xc[0];
    let y0 = yc[0];
    let exact_tol = 512.0 * f64::EPSILON * nx.max(ny) as f64;

    // First pass: normalized departure coordinates and snap flags.
    let mut norm_x = vec![0.0; ny * nx];
    let mut norm_y = vec![0.0; ny * nx];
    let mut all_snap = true;
    for i in 0..ny
    {
        for j in 0..nx
        {
            let src_x = xc[j] - dt * vx.get(i, j);
            let src_y = yc[i] - dt * vy.get(i, j);
            let nxv = (wrap(src_x, x0, period_x) - x0) / dx;
            let nyv = (wrap(src_y, y0, period_y) - y0) / dy;
            norm_x[i * nx + j] = nxv;
            norm_y[i * nx + j] = nyv;
            if (nxv - nxv.round()).abs() > exact_tol || (nyv - nyv.round()).abs() > exact_tol
            {
                all_snap = false;
            }
        }
    }

    if all_snap
    {
        return Ok(Field2::from_fn(ny, nx, |i, j| {
            let ix = (norm_x[i * nx + j].round() as i64).rem_euclid(nx as i64) as usize;
            let iy = (norm_y[i * nx + j].round() as i64).rem_euclid(ny as i64) as usize;
            field.get(iy, ix)
        }));
    }

    Ok(Field2::from_fn(ny, nx, |i, j| {
        let mut nxv = norm_x[i * nx + j];
        let mut nyv = norm_y[i * nx + j];
        let rx = nxv.round();
        let ry = nyv.round();
        if (nxv - rx).abs() <= exact_tol
        {
            nxv = rx.rem_euclid(nx as f64);
        }
        if (nyv - ry).abs() <= exact_tol
        {
            nyv = ry.rem_euclid(ny as f64);
        }
        match interpolation
        {
            Interpolation::BilinearPeriodic => bilinear_at(field, nxv, nyv),
            // Limiter variants returned above; only cubic remains.
            _ => cubic_at(field, nxv, nyv),
        }
    }))
}

/// Bilinear sample of `field` at normalized (grid-unit) periodic coordinates.
fn bilinear_at(field: &Field2, nxv: f64, nyv: f64) -> f64 {
    let (ny, nx) = field.shape();
    let fx = nxv.floor();
    let fy = nyv.floor();
    let tx = nxv - fx;
    let ty = nyv - fy;
    let ix0 = (fx as i64).rem_euclid(nx as i64) as usize;
    let iy0 = (fy as i64).rem_euclid(ny as i64) as usize;
    let ix1 = (ix0 + 1) % nx;
    let iy1 = (iy0 + 1) % ny;
    let v00 = field.get(iy0, ix0);
    let v10 = field.get(iy0, ix1);
    let v01 = field.get(iy1, ix0);
    let v11 = field.get(iy1, ix1);
    (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11
}

/// 16-point cubic-Lagrange sample of `field` at normalized periodic
/// coordinates.
fn cubic_at(field: &Field2, nxv: f64, nyv: f64) -> f64 {
    let (ny, nx) = field.shape();
    let base_x = nxv.floor();
    let base_y = nyv.floor();
    let wx = cubic_weights(nxv - base_x);
    let wy = cubic_weights(nyv - base_y);
    let bx = base_x as i64;
    let by = base_y as i64;
    let offsets = [-1i64, 0, 1, 2];
    let mut acc = 0.0;
    for (yi, &oy) in offsets.iter().enumerate()
    {
        let iy = (by + oy).rem_euclid(ny as i64) as usize;
        for (xi, &ox) in offsets.iter().enumerate()
        {
            let ix = (bx + ox).rem_euclid(nx as i64) as usize;
            acc += wy[yi] * wx[xi] * field.get(iy, ix);
        }
    }
    acc
}

/// Samples `field` at explicit departure coordinates (used by the RK4 path).
#[allow(clippy::too_many_arguments)]
fn sample_at_departures(
    field: &Field2,
    xc: &[f64],
    yc: &[f64],
    dx: f64,
    dy: f64,
    departure_x: &Field2,
    departure_y: &Field2,
    interpolation: Interpolation,
) -> Field2 {
    let (ny, nx) = field.shape();

    if interpolation.is_limited()
    {
        let bilinear = sample_at_departures(
            field,
            xc,
            yc,
            dx,
            dy,
            departure_x,
            departure_y,
            Interpolation::BilinearPeriodic,
        );
        let cubic = sample_at_departures(
            field,
            xc,
            yc,
            dx,
            dy,
            departure_x,
            departure_y,
            Interpolation::CubicPeriodic,
        );
        let (lower, upper) =
            explicit_departure_bounds(field, xc, yc, dx, dy, departure_x, departure_y);
        return combine_limited(&bilinear, &cubic, &lower, &upper, interpolation);
    }

    let period_x = dx * nx as f64;
    let period_y = dy * ny as f64;
    let x0 = xc[0];
    let y0 = yc[0];
    Field2::from_fn(ny, nx, |i, j| {
        let nxv = (wrap(departure_x.get(i, j), x0, period_x) - x0) / dx;
        let nyv = (wrap(departure_y.get(i, j), y0, period_y) - y0) / dy;
        match interpolation
        {
            Interpolation::BilinearPeriodic => bilinear_at(field, nxv, nyv),
            // Limiter variants returned above; only cubic remains.
            _ => cubic_at(field, nxv, nyv),
        }
    })
}

/// RK4 back-integration of the departure points, wrapping every stage point
/// periodically before evaluating the velocity.
fn rk4_departures<VF>(
    xc: &[f64],
    yc: &[f64],
    dx: f64,
    dy: f64,
    previous_time: f64,
    current_time: f64,
    velocity: &VF,
) -> Result<(Field2, Field2)>
where
    VF: Fn(f64, f64, f64) -> (f64, f64),
{
    let (ny, nx) = (yc.len(), xc.len());
    let period_x = dx * nx as f64;
    let period_y = dy * ny as f64;
    let x0 = xc[0];
    let y0 = yc[0];
    let dt = current_time - previous_time;
    let step = -dt;
    let t_mid = 0.5 * (previous_time + current_time);

    let eval =
        |px: f64, py: f64, t: f64| velocity(wrap(px, x0, period_x), wrap(py, y0, period_y), t);

    let mut dep_x = Field2::zeros(ny, nx);
    let mut dep_y = Field2::zeros(ny, nx);
    for i in 0..ny
    {
        for j in 0..nx
        {
            let cx = xc[j];
            let cy = yc[i];
            let (k1x, k1y) = eval(cx, cy, current_time);
            let (k2x, k2y) = eval(cx + 0.5 * step * k1x, cy + 0.5 * step * k1y, t_mid);
            let (k3x, k3y) = eval(cx + 0.5 * step * k2x, cy + 0.5 * step * k2y, t_mid);
            let (k4x, k4y) = eval(cx + step * k3x, cy + step * k3y, previous_time);
            let ddx = cx + (step / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            let ddy = cy + (step / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
            if !ddx.is_finite() || !ddy.is_finite()
            {
                return Err(ItdError::NonFinite("RK4 departure point".into()));
            }
            *dep_x.get_mut(i, j) = ddx;
            *dep_y.get_mut(i, j) = ddy;
        }
    }
    Ok((dep_x, dep_y))
}

// --- local limiters ------------------------------------------------------

/// Lower/upper bound at one cell: the min/max of the four bilinear-stencil
/// nodes surrounding the normalized (grid-unit) departure coordinate.
fn stencil_bounds_at(field: &Field2, nxv: f64, nyv: f64) -> (f64, f64) {
    let (ny, nx) = field.shape();
    let ix0 = (nxv.floor() as i64).rem_euclid(nx as i64) as usize;
    let iy0 = (nyv.floor() as i64).rem_euclid(ny as i64) as usize;
    let ix1 = (ix0 + 1) % nx;
    let iy1 = (iy0 + 1) % ny;
    let v00 = field.get(iy0, ix0);
    let v10 = field.get(iy0, ix1);
    let v01 = field.get(iy1, ix0);
    let v11 = field.get(iy1, ix1);
    let lower = v00.min(v10).min(v01).min(v11);
    let upper = v00.max(v10).max(v01).max(v11);
    (lower, upper)
}

/// Per-cell stencil bounds for the midpoint back-trace (`departure = grid −
/// Δt·v`). No exact-snap: the plain floor of the wrapped normalized coordinate
/// selects the stencil, matching the reference.
#[allow(clippy::too_many_arguments)]
fn midpoint_departure_bounds(
    field: &Field2,
    xc: &[f64],
    yc: &[f64],
    dx: f64,
    dy: f64,
    vx: &Field2,
    vy: &Field2,
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (ny, nx) = field.shape();
    let period_x = dx * nx as f64;
    let period_y = dy * ny as f64;
    let x0 = xc[0];
    let y0 = yc[0];
    let mut lower = vec![0.0; ny * nx];
    let mut upper = vec![0.0; ny * nx];
    for i in 0..ny
    {
        for j in 0..nx
        {
            let src_x = xc[j] - dt * vx.get(i, j);
            let src_y = yc[i] - dt * vy.get(i, j);
            let nxv = (wrap(src_x, x0, period_x) - x0) / dx;
            let nyv = (wrap(src_y, y0, period_y) - y0) / dy;
            let (lo, hi) = stencil_bounds_at(field, nxv, nyv);
            lower[i * nx + j] = lo;
            upper[i * nx + j] = hi;
        }
    }
    (lower, upper)
}

/// Per-cell stencil bounds for explicit departure coordinates (RK4 path).
#[allow(clippy::too_many_arguments)]
fn explicit_departure_bounds(
    field: &Field2,
    xc: &[f64],
    yc: &[f64],
    dx: f64,
    dy: f64,
    departure_x: &Field2,
    departure_y: &Field2,
) -> (Vec<f64>, Vec<f64>) {
    let (ny, nx) = field.shape();
    let period_x = dx * nx as f64;
    let period_y = dy * ny as f64;
    let x0 = xc[0];
    let y0 = yc[0];
    let mut lower = vec![0.0; ny * nx];
    let mut upper = vec![0.0; ny * nx];
    for i in 0..ny
    {
        for j in 0..nx
        {
            let nxv = (wrap(departure_x.get(i, j), x0, period_x) - x0) / dx;
            let nyv = (wrap(departure_y.get(i, j), y0, period_y) - y0) / dy;
            let (lo, hi) = stencil_bounds_at(field, nxv, nyv);
            lower[i * nx + j] = lo;
            upper[i * nx + j] = hi;
        }
    }
    (lower, upper)
}

/// Convex blend `q = low + θ·(high − low)`, `θ ∈ [0, 1]`, choosing the largest
/// `θ` that keeps `q` within `[lower, upper]` at every cell.
fn convex_limiter_field(low: &[f64], high: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    let mut scale = 1.0_f64;
    for v in lower.iter().chain(upper.iter())
    {
        scale = scale.max(v.abs());
    }
    let tolerance = 512.0 * f64::EPSILON * scale;

    let n = low.len();
    let mut out = vec![0.0; n];
    for k in 0..n
    {
        let correction = high[k] - low[k];
        let mut theta = 1.0_f64;
        if high[k] > upper[k]
        {
            let theta_upper = if correction > tolerance
            {
                (upper[k] - low[k]) / correction
            }
            else
            {
                0.0
            };
            theta = theta.min(theta_upper);
        }
        if high[k] < lower[k]
        {
            let theta_lower = if correction < -tolerance
            {
                (lower[k] - low[k]) / correction
            }
            else
            {
                0.0
            };
            theta = theta.min(theta_lower);
        }
        let theta = theta.clamp(0.0, 1.0);
        out[k] = (low[k] + theta * correction).clamp(lower[k], upper[k]);
    }
    out
}

/// Deterministic Neumaier compensated summation — a fully-`f64` stand-in for
/// the reference's extended-precision (`longdouble`) sum accounting.
fn precise_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for &v in values
    {
        let t = sum + v;
        if sum.abs() >= v.abs()
        {
            compensation += (sum - t) + v;
        }
        else
        {
            compensation += (v - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

/// Periodic dilation of a boolean mask over the 3×3 Moore neighbourhood.
fn expand_mask(mask: &[bool], ny: usize, nx: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    for i in 0..ny
    {
        for j in 0..nx
        {
            if out[i * nx + j]
            {
                continue;
            }
            let mut any = false;
            for dy in [-1i64, 0, 1]
            {
                for dxi in [-1i64, 0, 1]
                {
                    let ni = (i as i64 + dy).rem_euclid(ny as i64) as usize;
                    let nj = (j as i64 + dxi).rem_euclid(nx as i64) as usize;
                    if mask[ni * nx + nj]
                    {
                        any = true;
                    }
                }
            }
            out[i * nx + j] = any;
        }
    }
    out
}

/// Redistributes a sum defect over a periodic neighbourhood of the limited
/// (`seed`) cells, respecting the local `[lower, upper]` bounds, so the total
/// matches `target`. The neighbourhood grows only as needed; a final
/// deterministic pass mops up the rounding remainder.
fn restore_sum(
    values: &[f64],
    lower: &[f64],
    upper: &[f64],
    target: f64,
    seeds: &[bool],
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let n = ny * nx;
    let mut result = values.to_vec();

    let mut scale = 1.0_f64;
    for v in lower.iter().chain(upper.iter())
    {
        scale = scale.max(v.abs());
    }
    let tolerance = 4096.0 * f64::EPSILON * scale;

    let mut residual = target - precise_sum(&result);
    if residual.abs() <= tolerance
    {
        return result;
    }
    if !seeds.iter().any(|&s| s)
    {
        return result;
    }

    let mut support = seeds.to_vec();
    let max_radius = ny.max(nx);
    let mut capacity: Option<Vec<f64>> = None;
    for _ in 0..=max_radius
    {
        let candidate: Vec<f64> = (0..n)
            .map(|k| {
                if support[k]
                {
                    if residual > 0.0
                    {
                        (upper[k] - result[k]).max(0.0)
                    }
                    else
                    {
                        (result[k] - lower[k]).max(0.0)
                    }
                }
                else
                {
                    0.0
                }
            })
            .collect();
        let capacity_sum = precise_sum(&candidate);
        if capacity_sum + tolerance >= residual.abs()
        {
            capacity = Some(candidate);
            break;
        }
        let expanded = expand_mask(&support, ny, nx);
        if expanded == support
        {
            break;
        }
        support = expanded;
    }

    let capacity = match capacity
    {
        Some(c) => c,
        None => return result,
    };
    let capacity_sum = precise_sum(&capacity);
    let fraction = (residual.abs() / capacity_sum).clamp(0.0, 1.0);
    for k in 0..n
    {
        if residual > 0.0
        {
            result[k] += fraction * capacity[k];
        }
        else
        {
            result[k] -= fraction * capacity[k];
        }
        result[k] = result[k].clamp(lower[k], upper[k]);
    }

    // Deterministic final rounding correction, largest capacity first.
    for _ in 0..8
    {
        residual = target - precise_sum(&result);
        if residual.abs() <= tolerance
        {
            break;
        }
        let remaining_capacity: Vec<f64> = (0..n)
            .map(|k| {
                if support[k]
                {
                    if residual > 0.0
                    {
                        (upper[k] - result[k]).max(0.0)
                    }
                    else
                    {
                        (result[k] - lower[k]).max(0.0)
                    }
                }
                else
                {
                    0.0
                }
            })
            .collect();
        let mut eligible: Vec<usize> = (0..n)
            .filter(|&k| remaining_capacity[k] > tolerance)
            .collect();
        if eligible.is_empty()
        {
            break;
        }
        // Descending capacity; stable, so ties keep ascending index order.
        eligible.sort_by(|&a, &b| {
            remaining_capacity[b]
                .partial_cmp(&remaining_capacity[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for &idx in &eligible
        {
            let remaining = target - precise_sum(&result);
            if remaining.abs() <= tolerance
            {
                break;
            }
            let available = remaining_capacity[idx];
            let correction = remaining.abs().min(available);
            if correction <= 0.0
            {
                continue;
            }
            if remaining > 0.0
            {
                result[idx] += correction;
            }
            else
            {
                result[idx] -= correction;
            }
            result[idx] = result[idx].clamp(lower[idx], upper[idx]);
        }
    }

    result
}

/// Combines the bilinear and cubic fields into the requested limiter result.
fn combine_limited(
    bilinear: &Field2,
    cubic: &Field2,
    lower: &[f64],
    upper: &[f64],
    interpolation: Interpolation,
) -> Field2 {
    let (ny, nx) = cubic.shape();
    let bounded = convex_limiter_field(bilinear.as_slice(), cubic.as_slice(), lower, upper);
    match interpolation
    {
        Interpolation::CubicLocalBoundedPeriodic =>
        {
            Field2::from_vec(ny, nx, bounded).expect("limiter output shape")
        },
        _ =>
        {
            // Sum-preserving: restore the discrete sum to the unlimited cubic's.
            let cubic_slice = cubic.as_slice();
            let mut activation_scale = 1.0_f64;
            for &v in cubic_slice
            {
                activation_scale = activation_scale.max(v.abs());
            }
            let activation_tolerance = 256.0 * f64::EPSILON * activation_scale;
            let seeds: Vec<bool> = (0..bounded.len())
                .map(|k| (bounded[k] - cubic_slice[k]).abs() > activation_tolerance)
                .collect();
            let target = precise_sum(cubic_slice);
            let restored = restore_sum(&bounded, lower, upper, target, &seeds, ny, nx);
            Field2::from_vec(ny, nx, restored).expect("limiter output shape")
        },
    }
}
