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

use crate::error::{ItdError, Result};
use crate::field::Field2;
use crate::geometry::{BoundaryMode, Geometry};
use crate::operators::{gradient, spatial_mean};
use crate::signature::ZERO_THRESHOLD;

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
