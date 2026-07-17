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
//! and the previous field is sampled there by periodic bilinear or 16-point
//! cubic (Lagrange) interpolation.
//!
//! Two interpolation variants of the reference — the convex-limited
//! `cubic_local_bounded_periodic` and the discrete-sum-preserving
//! `cubic_local_sum_preserving_periodic` — are intentionally **not** ported
//! here; this module covers the two exact schemes (`bilinear`, `cubic`) and
//! both trajectory methods (`midpoint`, `rk4`).

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
    if matches!(interpolation, Interpolation::CubicPeriodic) && (nx < 4 || ny < 4)
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
            Interpolation::CubicPeriodic => cubic_at(field, nxv, nyv),
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
            Interpolation::CubicPeriodic => cubic_at(field, nxv, nyv),
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
