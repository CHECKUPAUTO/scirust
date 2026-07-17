//! A genuinely curvilinear chart of flat Minkowski spacetime, and transforms
//! between it and the standard Cartesian chart.
//!
//! [`CylindricalMinkowski`] describes the *same* flat spacetime as
//! `scirust_relativity::Minkowski`, just in cylindrical coordinates `(t, r,
//! phi, z)`. Unlike the Cartesian chart, its connection is not identically
//! zero, so it is useful for checking whether a computation's disagreement
//! between charts shrinks under refinement and under geometric transport —
//! see `examples/coordinate_covariance.rs`. It is not a distinct physical
//! background.

use crate::{NonlocalRelativityError, NonlocalResult};
use scirust_relativity::{Connection, Metric};

/// Flat Minkowski spacetime expressed in cylindrical coordinates `(t, r,
/// phi, z)`, signature `(-,+,+,+)`.
///
/// This is the same flat spacetime as `scirust_relativity::Minkowski`, in a
/// chart whose connection is not identically zero. The chart is regular only
/// for `r > 0`; `r = 0` is a coordinate singularity of the chart, not a
/// physical singularity of flat spacetime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CylindricalMinkowski;

impl CylindricalMinkowski {
    /// Determine whether coordinates lie in the regular region of this
    /// chart.
    #[must_use]
    pub fn is_regular(coordinates: &[f64; 4]) -> bool {
        coordinates.iter().all(|value| value.is_finite()) && coordinates[1] > 0.0
    }
}

impl Metric<4> for CylindricalMinkowski {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let radius = coordinates[1];

        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, radius * radius, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl Connection<4> for CylindricalMinkowski {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        let radius = coordinates[1];
        let mut symbols = [[[0.0_f64; 4]; 4]; 4];

        symbols[1][2][2] = -radius;
        symbols[2][1][2] = 1.0 / radius;
        symbols[2][2][1] = 1.0 / radius;

        symbols
    }
}

/// Convert Cartesian Minkowski coordinates `(t, x, y, z)` into cylindrical
/// Minkowski coordinates `(t, r, phi, z)`, with `r = sqrt(x^2 + y^2)` and
/// `phi = atan2(y, x)`.
pub fn cartesian_to_cylindrical_coordinates(coordinates: [f64; 4]) -> NonlocalResult<[f64; 4]> {
    let [time, x, y, z] = coordinates;
    let radius = (x * x + y * y).sqrt();
    let angle = y.atan2(x);
    let converted = [time, radius, angle, z];
    validate_chart_coordinates(&converted)?;
    Ok(converted)
}

/// Convert cylindrical Minkowski coordinates `(t, r, phi, z)` into Cartesian
/// Minkowski coordinates `(t, x, y, z)`, with `x = r cos(phi)` and
/// `y = r sin(phi)`.
pub fn cylindrical_to_cartesian_coordinates(coordinates: [f64; 4]) -> NonlocalResult<[f64; 4]> {
    let [time, radius, angle, z] = coordinates;
    validate_cylindrical_radius(radius)?;
    let converted = [time, radius * angle.cos(), radius * angle.sin(), z];
    validate_chart_coordinates(&converted)?;
    Ok(converted)
}

/// Convert a contravariant velocity from the Cartesian Minkowski chart to
/// the cylindrical Minkowski chart at `coordinates` (given in the Cartesian
/// chart), using the exact Jacobian
/// `u^r = (x u^x + y u^y) / r`, `u^phi = (x u^y - y u^x) / r^2`.
pub fn cartesian_to_cylindrical_velocity(
    coordinates: [f64; 4],
    velocity: [f64; 4],
) -> NonlocalResult<[f64; 4]> {
    let [_, x, y, _] = coordinates;
    let [u_time, u_x, u_y, u_z] = velocity;
    let radius_squared = x * x + y * y;
    let radius = radius_squared.sqrt();
    validate_cylindrical_radius(radius)?;

    let u_radius = (x * u_x + y * u_y) / radius;
    let u_angle = (x * u_y - y * u_x) / radius_squared;
    let converted = [u_time, u_radius, u_angle, u_z];
    validate_chart_velocity(&converted)?;
    Ok(converted)
}

/// Convert a contravariant velocity from the cylindrical Minkowski chart to
/// the Cartesian Minkowski chart at `coordinates` (given in the cylindrical
/// chart), using the exact Jacobian
/// `u^x = u^r cos(phi) - r sin(phi) u^phi`,
/// `u^y = u^r sin(phi) + r cos(phi) u^phi`.
pub fn cylindrical_to_cartesian_velocity(
    coordinates: [f64; 4],
    velocity: [f64; 4],
) -> NonlocalResult<[f64; 4]> {
    let [_, radius, angle, _] = coordinates;
    validate_cylindrical_radius(radius)?;
    let [u_time, u_radius, u_angle, u_z] = velocity;
    let (sin_angle, cos_angle) = angle.sin_cos();

    let converted = [
        u_time,
        u_radius * cos_angle - radius * sin_angle * u_angle,
        u_radius * sin_angle + radius * cos_angle * u_angle,
        u_z,
    ];
    validate_chart_velocity(&converted)?;
    Ok(converted)
}

fn validate_cylindrical_radius(radius: f64) -> NonlocalResult<()> {
    if !radius.is_finite() || radius <= 0.0
    {
        return Err(NonlocalRelativityError::InvalidCylindricalRadius(radius));
    }

    Ok(())
}

fn validate_chart_coordinates(coordinates: &[f64; 4]) -> NonlocalResult<()> {
    for (component, value) in coordinates.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteChartCoordinate { component, value });
        }
    }

    Ok(())
}

fn validate_chart_velocity(velocity: &[f64; 4]) -> NonlocalResult<()> {
    for (component, value) in velocity.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteChartVelocity { component, value });
        }
    }

    Ok(())
}
