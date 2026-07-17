//! Covariance of the field problem under spatial dilations and moving frames.
//!
//! These are the coordinate and field laws the reference simulator uses to check
//! that its diagnostics transform correctly under two families of change of
//! variables.
//!
//! **Spatial dilation** by a factor `a > 0` about an origin `o` maps
//! `x' = o + a(x − o)`, so the source coordinates of a target point are
//! `x = o + (x' − o)/a`. A velocity field scales as `v_a = a·v` and a curvature
//! field as `R_a = R/a²`; a uniform grid's spacings scale by `a` and a
//! rectilinear grid's coordinates dilate about `o`.
//!
//! **Frame changes** shift the coordinates in time. A Galilean boost by a
//! constant frame velocity `c` gives source coordinates `x = x' + c(t − t₀)` and
//! velocity `v' = v − c`. A general time-dependent translation by a displacement
//! `b(t)` gives `x = x' + b(t)` and `v' = v − ḃ(t)`. In both cases the velocity
//! law is the same subtraction of a constant 2-vector at a fixed time, provided
//! here by [`subtract_frame_velocity`].

use crate::error::{ItdError, Result};
use crate::field::Field2;
use crate::geometry::Geometry;

/// Validates a spatial scale factor: finite and strictly positive.
fn validate_scale_factor(a: f64) -> Result<f64> {
    if !a.is_finite() || a <= 0.0
    {
        return Err(ItdError::InvalidGeometry(format!(
            "spatial scale factor must be finite and strictly positive (got {a})"
        )));
    }
    Ok(a)
}

/// Scales a non-negative length by `a`: `a · length`.
pub fn scale_length(length: f64, a: f64) -> Result<f64> {
    let a = validate_scale_factor(a)?;
    if !length.is_finite() || length < 0.0
    {
        return Err(ItdError::InvalidGeometry(format!(
            "length must be finite and non-negative (got {length})"
        )));
    }
    Ok(a * length)
}

/// The source coordinates of a spatial dilation: `x = o + (x' − o)/a`.
///
/// `x` and `y` must have equal length; the result has the same length.
pub fn inverse_scale_coordinates(
    x: &[f64],
    y: &[f64],
    a: f64,
    origin: [f64; 2],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let a = validate_scale_factor(a)?;
    check_origin(origin)?;
    if x.len() != y.len()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "coordinate arrays differ in length: {} vs {}",
            x.len(),
            y.len()
        )));
    }
    check_finite(x, "x coordinates")?;
    check_finite(y, "y coordinates")?;
    let sx = x.iter().map(|&v| origin[0] + (v - origin[0]) / a).collect();
    let sy = y.iter().map(|&v| origin[1] + (v - origin[1]) / a).collect();
    Ok((sx, sy))
}

/// Dilates a grid by `a` about `origin`: a uniform grid's spacings become
/// `a·dx, a·dy`; a rectilinear grid's coordinates become `o + a(coord − o)`.
pub fn scale_geometry(geometry: &Geometry, a: f64, origin: [f64; 2]) -> Result<Geometry> {
    let a = validate_scale_factor(a)?;
    check_origin(origin)?;
    match geometry
    {
        Geometry::Uniform { dx, dy } => Geometry::uniform(a * dx, a * dy),
        Geometry::Rectilinear { x, y } =>
        {
            let sx = x.iter().map(|&v| origin[0] + a * (v - origin[0])).collect();
            let sy = y.iter().map(|&v| origin[1] + a * (v - origin[1])).collect();
            Geometry::rectilinear(sx, sy)
        },
    }
}

/// The source coordinates of a Galilean boost by frame velocity `c` between
/// `reference_time` `t₀` and `time` `t`: `x = x' + c(t − t₀)`.
pub fn galilean_source_coordinates(
    x: &[f64],
    y: &[f64],
    time: f64,
    frame_velocity: [f64; 2],
    reference_time: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if !time.is_finite() || !reference_time.is_finite()
    {
        return Err(ItdError::NonFinite("Galilean time".into()));
    }
    if !frame_velocity.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite("frame velocity".into()));
    }
    if x.len() != y.len()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "coordinate arrays differ in length: {} vs {}",
            x.len(),
            y.len()
        )));
    }
    check_finite(x, "x coordinates")?;
    check_finite(y, "y coordinates")?;
    let elapsed = time - reference_time;
    let sx = x.iter().map(|&v| v + frame_velocity[0] * elapsed).collect();
    let sy = y.iter().map(|&v| v + frame_velocity[1] * elapsed).collect();
    Ok((sx, sy))
}

/// The source coordinates of a time-dependent translation by displacement
/// `b`: `x = x' + b`.
pub fn translating_frame_source_coordinates(
    x: &[f64],
    y: &[f64],
    displacement: [f64; 2],
) -> Result<(Vec<f64>, Vec<f64>)> {
    if !displacement.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite("frame displacement".into()));
    }
    if x.len() != y.len()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "coordinate arrays differ in length: {} vs {}",
            x.len(),
            y.len()
        )));
    }
    check_finite(x, "x coordinates")?;
    check_finite(y, "y coordinates")?;
    let sx = x.iter().map(|&v| v + displacement[0]).collect();
    let sy = y.iter().map(|&v| v + displacement[1]).collect();
    Ok((sx, sy))
}

/// The transformed velocity in a moving frame: `v' = v − frame_velocity`, the
/// common law behind both the Galilean boost (`frame_velocity = c`) and the
/// time-dependent translation (`frame_velocity = ḃ(t)`).
pub fn subtract_frame_velocity(
    vx: &Field2,
    vy: &Field2,
    frame_velocity: [f64; 2],
) -> Result<(Field2, Field2)> {
    if !frame_velocity.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite("frame velocity".into()));
    }
    if vx.shape() != vy.shape()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "velocity components differ: {:?} vs {:?}",
            vx.shape(),
            vy.shape()
        )));
    }
    Ok((
        vx.map(|v| v - frame_velocity[0]),
        vy.map(|v| v - frame_velocity[1]),
    ))
}

fn check_origin(origin: [f64; 2]) -> Result<()> {
    if !origin.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite("origin".into()));
    }
    Ok(())
}

fn check_finite(values: &[f64], name: &str) -> Result<()> {
    if !values.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite(name.into()));
    }
    Ok(())
}
