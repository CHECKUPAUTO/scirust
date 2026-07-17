//! Material-derivative interval diagnostic for a transported vorticity field.
//!
//! Over one time interval `[t₀, t₁]` the change of a vorticity field `ω` splits
//! into an *Eulerian* (local, fixed-point) tendency and an *advective*
//! (transport) tendency, whose sum is the *material* (Lagrangian) tendency:
//!
//! ```text
//! temporal_tendency  = (ω₁ − ω₀) / Δt
//! advective_tendency = u · ∇ω        (u and ∇ω at the interval midpoint)
//! material_tendency  = temporal_tendency + advective_tendency
//! ```
//!
//! The velocity `u` and the gradient `∇ω` are both taken at the interval
//! midpoint (`u` at `t = ½(t₀+t₁)`, `∇ω` on the midpoint field
//! `½(ω₀+ω₁)`), matching the reference simulator's centred discretisation.
//!
//! Each tendency is reduced to a single scalar **rate** by dividing its
//! root-mean-square by a reference RMS (the mean of the two fields' RMS). The
//! three rates are *independent norms*; they are **not** additive — the
//! material rate is not the sum of the Eulerian and advective rates.
//!
//! This reuses the crate's [`operators::gradient`](crate::operators::gradient)
//! (second-order, boundary-aware) and
//! [`operators::spatial_mean`](crate::operators::spatial_mean) (trapezoidal /
//! arithmetic), so the diagnostic inherits their oracle-validated behaviour.
//!
//! [`simulate_material_deformation`] is the full orchestration ported from the
//! reference `simulate_material_deformation`: it runs the historical eulerian
//! simulation ([`simulate`](crate::simulate::simulate)) unchanged, then walks
//! the same time grid once more adding the per-interval material diagnostic,
//! reduces each rate series to a duration-weighted index, interpolates the
//! interval series back to the time nodes, and reports the maximum discrepancy
//! between the diagnostic's Eulerian rate and the baseline's temporal
//! deformation (they are the same formula, so this *consistency error* is a
//! numerical certification, expected at machine-epsilon scale). As in the
//! reference, the material diagnostic is **not** injected into the structural
//! signature — the baseline result is reported unchanged alongside it.

use crate::error::{ItdError, Result};
use crate::field::Field2;
use crate::geometry::{BoundaryMode, Geometry};
use crate::operators::{gradient, spatial_mean, vorticity};
use crate::scenarios::{Config, Scenario, curvature_field};
use crate::signature::ZERO_THRESHOLD;
use crate::simulate::{SimConfig, SimulationResult, simulate};

/// The material-derivative decomposition of a vorticity field over one time
/// interval, as returned by [`material_vorticity_interval`].
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialInterval {
    /// The interval length `Δt`.
    pub delta_time: f64,
    /// The reference RMS `½(rms(ω₀) + rms(ω₁))` used to normalise the rates.
    pub reference_rms: f64,
    /// `rms(ω₀)` — the RMS of the previous field.
    pub previous_rms: f64,
    /// `rms(ω₁)` — the RMS of the current field.
    pub current_rms: f64,
    /// The local tendency field `(ω₁ − ω₀) / Δt`.
    pub temporal_tendency: Field2,
    /// The advective tendency field `u · ∇ω` at the interval midpoint.
    pub advective_tendency: Field2,
    /// The material tendency field `temporal + advective`.
    pub material_tendency: Field2,
    /// `rms(temporal_tendency) / reference_rms`, or `0` when the reference RMS
    /// is negligible.
    pub eulerian_rate: f64,
    /// `rms(advective_tendency) / reference_rms`, or `0` when negligible.
    pub advective_rate: f64,
    /// `rms(material_tendency) / reference_rms`, or `0` when negligible.
    pub material_rate: f64,
}

/// `sqrt(max(⟨field²⟩, 0)) / reference_rms`, or `0` when `reference_rms` is
/// below [`ZERO_THRESHOLD`] (matching the reference `normalized_field_rate`).
fn normalized_field_rate(
    field: &Field2,
    reference_rms: f64,
    geometry: &Geometry,
    boundary: BoundaryMode,
) -> Result<f64> {
    if reference_rms < ZERO_THRESHOLD
    {
        return Ok(0.0);
    }
    let mean_square = spatial_mean(&field.map(|v| v * v), geometry, boundary)?;
    Ok(mean_square.max(0.0).sqrt() / reference_rms)
}

/// `sqrt(max(⟨field²⟩, 0))`.
fn field_rms(field: &Field2, geometry: &Geometry, boundary: BoundaryMode) -> Result<f64> {
    let mean_square = spatial_mean(&field.map(|v| v * v), geometry, boundary)?;
    Ok(mean_square.max(0.0).sqrt())
}

/// Decomposes the evolution of a vorticity field over one time interval into its
/// Eulerian, advective and material tendencies, and reduces each to a scalar
/// RMS-normalised rate.
///
/// `previous_omega` and `current_omega` are the vorticity at the two interval
/// endpoints; `midpoint_vx`, `midpoint_vy` are the advecting velocity sampled at
/// the interval midpoint time. All four fields must share a shape with at least
/// three points per direction (the gradient stencil), and `delta_time` must be
/// finite and strictly positive.
///
/// The gradient of the midpoint field and the spatial means use the given
/// `geometry` and `boundary` convention.
pub fn material_vorticity_interval(
    previous_omega: &Field2,
    current_omega: &Field2,
    midpoint_vx: &Field2,
    midpoint_vy: &Field2,
    geometry: &Geometry,
    delta_time: f64,
    boundary: BoundaryMode,
) -> Result<MaterialInterval> {
    if !delta_time.is_finite() || delta_time <= 0.0
    {
        return Err(ItdError::InvalidGeometry(format!(
            "time interval must be finite and strictly positive (got {delta_time})"
        )));
    }

    let shape = previous_omega.shape();
    for (field, name) in [
        (current_omega, "current vorticity"),
        (midpoint_vx, "midpoint velocity x"),
        (midpoint_vy, "midpoint velocity y"),
    ]
    {
        if field.shape() != shape
        {
            return Err(ItdError::ShapeMismatch(format!(
                "{name} {:?} does not match previous vorticity {:?}",
                field.shape(),
                shape
            )));
        }
    }
    if shape.0.min(shape.1) < 3
    {
        return Err(ItdError::TooFewPoints(
            "material interval needs at least three points per direction".into(),
        ));
    }
    for (field, name) in [
        (previous_omega, "previous vorticity"),
        (current_omega, "current vorticity"),
        (midpoint_vx, "midpoint velocity x"),
        (midpoint_vy, "midpoint velocity y"),
    ]
    {
        if !field.all_finite()
        {
            return Err(ItdError::NonFinite(format!(
                "{name} contains a non-finite value"
            )));
        }
    }
    geometry.validate_field(previous_omega)?;

    let temporal_tendency = current_omega.zip_map(previous_omega, |c, p| (c - p) / delta_time)?;
    let midpoint_omega = current_omega.zip_map(previous_omega, |c, p| 0.5 * (c + p))?;

    let (gradient_y, gradient_x) = gradient(&midpoint_omega, geometry, boundary)?;

    // advective = vx * ∂ω/∂x + vy * ∂ω/∂y, node by node.
    let mut advective_tendency = Field2::zeros(shape.0, shape.1);
    for i in 0..shape.0
    {
        for j in 0..shape.1
        {
            *advective_tendency.get_mut(i, j) = midpoint_vx.get(i, j) * gradient_x.get(i, j)
                + midpoint_vy.get(i, j) * gradient_y.get(i, j);
        }
    }

    let material_tendency = temporal_tendency.zip_map(&advective_tendency, |t, a| t + a)?;

    let previous_rms = field_rms(previous_omega, geometry, boundary)?;
    let current_rms = field_rms(current_omega, geometry, boundary)?;
    let reference_rms = 0.5 * (previous_rms + current_rms);

    let eulerian_rate =
        normalized_field_rate(&temporal_tendency, reference_rms, geometry, boundary)?;
    let advective_rate =
        normalized_field_rate(&advective_tendency, reference_rms, geometry, boundary)?;
    let material_rate =
        normalized_field_rate(&material_tendency, reference_rms, geometry, boundary)?;

    Ok(MaterialInterval {
        delta_time,
        reference_rms,
        previous_rms,
        current_rms,
        temporal_tendency,
        advective_tendency,
        material_tendency,
        eulerian_rate,
        advective_rate,
        material_rate,
    })
}

/// Which velocity field advects the vorticity in the material diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvectionSource {
    /// The scenario's own velocity field (the reference default).
    VelocityField,
    /// A separately supplied advection field (when transport is known from
    /// another source).
    AdvectionVelocityField,
}

/// The outcome of [`simulate_material_deformation`]: the untouched eulerian
/// baseline plus the per-interval material-derivative diagnostic, its
/// duration-weighted indices, its node-interpolated series, and the
/// eulerian-consistency certification.
///
/// Field names mirror the reference result keys (`material_*`), without the
/// redundant prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialDeformation {
    /// The unchanged eulerian baseline run (the material diagnostic is not
    /// injected into the structural signature).
    pub baseline: SimulationResult,
    /// Which velocity field advected the vorticity.
    pub advection_source: AdvectionSource,
    /// Per-interval Eulerian rate (`material_eulerian_rate_interval`), length
    /// `n − 1`.
    pub eulerian_rate_interval: Vec<f64>,
    /// Per-interval advective rate (`material_advective_rate_interval`).
    pub advective_rate_interval: Vec<f64>,
    /// Per-interval material rate (`material_deformation_interval`).
    pub material_deformation_interval: Vec<f64>,
    /// The Eulerian rate interpolated to the time nodes
    /// (`material_eulerian_rate`), length `n`.
    pub eulerian_rate: Vec<f64>,
    /// The advective rate interpolated to the time nodes
    /// (`material_advective_rate`).
    pub advective_rate: Vec<f64>,
    /// The material rate interpolated to the time nodes
    /// (`material_deformation`).
    pub material_deformation: Vec<f64>,
    /// Duration-weighted Eulerian index `Σ rate·Δt / duration`
    /// (`material_eulerian_rate_index`).
    pub eulerian_rate_index: f64,
    /// Duration-weighted advective index (`material_advective_rate_index`).
    pub advective_rate_index: f64,
    /// Duration-weighted material index (`material_deformation_index`).
    pub material_deformation_index: f64,
    /// `max_j |eulerian_rate_interval[j] − baseline temporal deformation over
    /// interval j|` (`material_eulerian_consistency_error`). Both series
    /// evaluate the same formula, so this certifies the two code paths agree to
    /// machine-epsilon scale.
    pub eulerian_consistency_error: f64,
}

/// Interpolates a per-interval series to the time nodes, matching the reference
/// `interpolate_interval_series_to_nodes`: each interval value is attributed to
/// the interval's temporal midpoint, and the nodes are linearly interpolated
/// between midpoints (NumPy `interp`), extrapolating as a constant before the
/// first midpoint and after the last.
pub fn interpolate_interval_series_to_nodes(
    times: &[f64],
    interval_values: &[f64],
) -> Result<Vec<f64>> {
    if times.len() < 2
    {
        return Err(ItdError::TooFewPoints(
            "interval interpolation needs at least two time samples".into(),
        ));
    }
    if !times.iter().all(|t| t.is_finite())
    {
        return Err(ItdError::NonFinite("time samples".into()));
    }
    if !times.windows(2).all(|w| w[1] > w[0])
    {
        return Err(ItdError::InvalidGeometry(
            "time samples must be strictly increasing".into(),
        ));
    }
    let m = times.len() - 1;
    if interval_values.len() != m
    {
        return Err(ItdError::ShapeMismatch(format!(
            "interval series has length {}, expected {m}",
            interval_values.len()
        )));
    }
    if !interval_values.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite("interval series".into()));
    }

    let midpoints: Vec<f64> = times.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let out = times
        .iter()
        .map(|&t| {
            if t <= midpoints[0]
            {
                return interval_values[0];
            }
            if t >= midpoints[m - 1]
            {
                return interval_values[m - 1];
            }
            // Find k with midpoints[k] <= t < midpoints[k + 1], then apply
            // NumPy's lerp form.
            let mut k = 0;
            while k + 2 < m && midpoints[k + 1] <= t
            {
                k += 1;
            }
            let slope =
                (interval_values[k + 1] - interval_values[k]) / (midpoints[k + 1] - midpoints[k]);
            interval_values[k] + slope * (t - midpoints[k])
        })
        .collect();
    Ok(out)
}

/// Runs the eulerian simulation and adds the material-derivative diagnostic,
/// advecting the vorticity by the scenario's own velocity field (the reference
/// default `advection_velocity_function = None`).
///
/// See [`simulate_material_deformation_with_advection`] for the general form
/// and the parameter documentation.
#[allow(clippy::too_many_arguments)]
pub fn simulate_material_deformation<VF, CF>(
    name: &str,
    velocity: VF,
    curvature: CF,
    xc: &[f64],
    yc: &[f64],
    times: &[f64],
    geometry: &Geometry,
    characteristic_length: f64,
    config: &SimConfig,
) -> Result<MaterialDeformation>
where
    VF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
    CF: Fn(&[f64], &[f64], f64) -> Field2,
{
    run_material_deformation(
        name,
        &velocity,
        curvature,
        &velocity,
        AdvectionSource::VelocityField,
        xc,
        yc,
        times,
        geometry,
        characteristic_length,
        config,
    )
}

/// Runs the eulerian simulation and adds the material-derivative diagnostic,
/// advecting the vorticity by a **separately supplied** velocity field (the
/// reference `advection_velocity_function`), for when the transport is known
/// from another source than the scenario's velocity.
///
/// `velocity` and `curvature` drive the baseline run exactly as in
/// [`simulate`](crate::simulate::simulate)); `advection` is evaluated once per
/// interval at the interval's midpoint time. The diagnostic is reported
/// alongside the baseline and is not injected into the structural signature.
#[allow(clippy::too_many_arguments)]
pub fn simulate_material_deformation_with_advection<VF, CF, AF>(
    name: &str,
    velocity: VF,
    curvature: CF,
    advection: AF,
    xc: &[f64],
    yc: &[f64],
    times: &[f64],
    geometry: &Geometry,
    characteristic_length: f64,
    config: &SimConfig,
) -> Result<MaterialDeformation>
where
    VF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
    CF: Fn(&[f64], &[f64], f64) -> Field2,
    AF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
{
    run_material_deformation(
        name,
        &velocity,
        curvature,
        &advection,
        AdvectionSource::AdvectionVelocityField,
        xc,
        yc,
        times,
        geometry,
        characteristic_length,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_material_deformation<VF, CF, AF>(
    name: &str,
    velocity: &VF,
    curvature: CF,
    advection: &AF,
    advection_source: AdvectionSource,
    xc: &[f64],
    yc: &[f64],
    times: &[f64],
    geometry: &Geometry,
    characteristic_length: f64,
    config: &SimConfig,
) -> Result<MaterialDeformation>
where
    VF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
    CF: Fn(&[f64], &[f64], f64) -> Field2,
    AF: Fn(&[f64], &[f64], f64) -> (Field2, Field2),
{
    // The baseline run performs the full time-grid / field validation.
    let baseline = simulate(
        name,
        velocity,
        curvature,
        xc,
        yc,
        times,
        geometry,
        characteristic_length,
        config,
    )?;

    let n = times.len();
    let m = n - 1;
    let mut eulerian_rate_interval = Vec::with_capacity(m);
    let mut advective_rate_interval = Vec::with_capacity(m);
    let mut material_deformation_interval = Vec::with_capacity(m);

    let mut previous: Option<(Field2, f64)> = None;
    for &t in times
    {
        let (vx, vy) = velocity(xc, yc, t);
        let omega = vorticity(&vx, &vy, geometry, config.boundary)?;

        if let Some((previous_omega, previous_time)) = previous.take()
        {
            let delta_time = t - previous_time;
            let midpoint_time = 0.5 * (t + previous_time);
            let (midpoint_vx, midpoint_vy) = advection(xc, yc, midpoint_time);
            let interval = material_vorticity_interval(
                &previous_omega,
                &omega,
                &midpoint_vx,
                &midpoint_vy,
                geometry,
                delta_time,
                config.boundary,
            )?;
            eulerian_rate_interval.push(interval.eulerian_rate);
            advective_rate_interval.push(interval.advective_rate);
            material_deformation_interval.push(interval.material_rate);
        }
        previous = Some((omega, t));
    }

    let duration = times[n - 1] - times[0];
    let interval_dt: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let integrate = |rates: &[f64]| -> f64 {
        let mut acc = 0.0;
        for j in 0..m
        {
            acc += rates[j] * interval_dt[j];
        }
        acc / duration
    };

    // The baseline's temporal deformation at node j + 1 covers interval j; in
    // eulerian mode it is the same formula as the diagnostic's Eulerian rate.
    let mut eulerian_consistency_error = 0.0f64;
    for j in 0..m
    {
        let discrepancy = (eulerian_rate_interval[j] - baseline.temporal_deformation[j + 1]).abs();
        eulerian_consistency_error = eulerian_consistency_error.max(discrepancy);
    }

    Ok(MaterialDeformation {
        eulerian_rate: interpolate_interval_series_to_nodes(times, &eulerian_rate_interval)?,
        advective_rate: interpolate_interval_series_to_nodes(times, &advective_rate_interval)?,
        material_deformation: interpolate_interval_series_to_nodes(
            times,
            &material_deformation_interval,
        )?,
        eulerian_rate_index: integrate(&eulerian_rate_interval),
        advective_rate_index: integrate(&advective_rate_interval),
        material_deformation_index: integrate(&material_deformation_interval),
        baseline,
        advection_source,
        eulerian_rate_interval,
        advective_rate_interval,
        material_deformation_interval,
        eulerian_consistency_error,
    })
}

/// Runs [`simulate_material_deformation`] on a canonical scenario with the
/// grid, time horizon and shared curvature weighting given by `config`
/// (mirroring [`simulate_canonical`](crate::simulate::simulate_canonical)).
pub fn simulate_canonical_material(
    scenario: Scenario,
    config: &Config,
    sim: &SimConfig,
) -> Result<MaterialDeformation> {
    let xc = config.coordinates();
    let yc = xc.clone();
    let times = config.times();
    let geometry = Geometry::isotropic(config.spacing())?;
    simulate_material_deformation(
        scenario.name(),
        |x, y, t| scenario.velocity(x, y, t),
        curvature_field,
        &xc,
        &yc,
        &times,
        &geometry,
        config.characteristic_length,
        sim,
    )
}
