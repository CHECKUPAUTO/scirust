//! Parallel transport of a vector along a coordinate path.
//!
//! Given any [`Connection`], this module transports a contravariant vector
//! `V^rho` along a piecewise-linear coordinate path by integrating the
//! parallel-transport equation
//!
//! ```text
//! dV^rho/ds + Gamma^rho_(mu nu)(x(s)) (dx^mu/ds) V^nu = 0
//! ```
//!
//! along each straight coordinate segment `x(s) = start + s (end - start)`,
//! `s in [0, 1]`, with the deterministic fixed-step RK4 integrator from
//! `scirust-sim` (the same engine [`crate::GeodesicSystem`] uses — no separate
//! integrator is introduced). `substeps` sets the RK4 resolution per segment.
//!
//! ## Validated properties
//!
//! - **Flatness / exactness.** In Minkowski Cartesian coordinates the
//!   Christoffel symbols vanish, so transport is the identity, returned
//!   bit-for-bit.
//! - **Metric compatibility.** For the Levi-Civita connection the metric inner
//!   product `g_(mu nu) V^mu W^nu` of transported vectors is preserved along
//!   the path (exactly in the continuum; numerically to the RK4 tolerance).
//! - **Holonomy.** Transport around a closed loop returns the vector unchanged
//!   in flat spacetime (zero holonomy), and in a curved spacetime by a defect
//!   that is `-R^rho_(sigma mu nu) V^sigma` times the enclosed coordinate area
//!   to leading order in the loop size — the standard holonomy/curvature
//!   identity, checked in this crate's tests against the numerical Riemann
//!   tensor from [`crate::CurvatureTensors`].
//!
//! This is the reusable geometry-core transport primitive. Covector/tensor
//! transport, adaptive resolution, and per-interval error estimates are natural
//! extensions layered on this same segment integrator.

use crate::{Connection, RelativityError};
use scirust_sim::{System, simulate};

/// A [`System`] whose state is the transported vector along one straight
/// coordinate segment `x(s) = start + s * delta`, `s in [0, 1]`.
struct SegmentTransport<'a, C, const D: usize> {
    connection: &'a C,
    start: [f64; D],
    delta: [f64; D],
}

impl<C, const D: usize> System for SegmentTransport<'_, C, D>
where
    C: Connection<D>,
{
    fn dim(&self) -> usize {
        D
    }

    fn derivatives(&self, parameter: f64, state: &[f64], output: &mut [f64]) {
        debug_assert_eq!(state.len(), D);
        debug_assert_eq!(output.len(), D);

        let mut coordinates = self.start;
        for (axis, coordinate) in coordinates.iter_mut().enumerate()
        {
            *coordinate += parameter * self.delta[axis];
        }

        let christoffel = self.connection.christoffel(&coordinates);

        // dV^rho/ds = - Gamma^rho_(mu nu) (dx^mu/ds) V^nu, dx^mu/ds = delta^mu.
        for (rho, rate) in output.iter_mut().enumerate().take(D)
        {
            let mut value = 0.0;
            for (mu, christoffel_row) in christoffel[rho].iter().enumerate()
            {
                let delta_mu = self.delta[mu];
                for (coefficient, velocity) in christoffel_row.iter().zip(state.iter())
                {
                    value -= coefficient * delta_mu * velocity;
                }
            }
            *rate = value;
        }
    }
}

pub(crate) fn validate_finite_coordinates<const D: usize>(
    coordinates: &[f64; D],
) -> Result<(), RelativityError> {
    if let Some((index, _)) = coordinates
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RelativityError::NonFiniteCoordinate(index));
    }
    Ok(())
}

pub(crate) fn validate_finite_vector<const D: usize>(
    vector: &[f64; D],
) -> Result<(), RelativityError> {
    if vector.iter().any(|value| !value.is_finite())
    {
        return Err(RelativityError::NonFiniteTransportedVector);
    }
    Ok(())
}

/// Parallel-transport `vector` from `start` to `end` along the straight
/// coordinate segment between them, using `substeps` RK4 steps.
///
/// Returns a typed [`RelativityError`] for non-finite coordinates or vector, a
/// zero substep count, or any non-finite intermediate/result; it never panics.
///
/// # Example
///
/// In flat Minkowski spacetime the Christoffel symbols vanish, so parallel
/// transport is the identity — the vector is returned unchanged.
///
/// ```
/// use scirust_relativity::{Minkowski, transport_along_segment};
///
/// let vector = [0.3, -0.5, 0.2, 0.1];
/// let transported = transport_along_segment(
///     &Minkowski,
///     &[0.0, 0.0, 0.0, 0.0],
///     &[1.0, 2.0, -1.0, 0.5],
///     &vector,
///     4,
/// )
/// .expect("flat transport");
/// assert_eq!(transported, vector);
/// ```
pub fn transport_along_segment<C, const D: usize>(
    connection: &C,
    start: &[f64; D],
    end: &[f64; D],
    vector: &[f64; D],
    substeps: usize,
) -> Result<[f64; D], RelativityError>
where
    C: Connection<D>,
{
    validate_finite_coordinates(start)?;
    validate_finite_coordinates(end)?;
    validate_finite_vector(vector)?;
    if substeps == 0
    {
        return Err(RelativityError::InvalidTransportResolution);
    }

    let mut delta = [0.0_f64; D];
    for (axis, component) in delta.iter_mut().enumerate()
    {
        *component = end[axis] - start[axis];
    }

    let system = SegmentTransport {
        connection,
        start: *start,
        delta,
    };
    let step = 1.0 / substeps as f64;

    let trajectory = simulate(&system, vector, 0.0, 1.0, step)
        .map_err(|_| RelativityError::NonFiniteTransportedVector)?;
    let final_state = trajectory
        .last_state()
        .ok_or(RelativityError::NonFiniteTransportedVector)?;

    let mut transported = [0.0_f64; D];
    for (axis, component) in transported.iter_mut().enumerate()
    {
        let value = final_state[axis];
        if !value.is_finite()
        {
            return Err(RelativityError::NonFiniteTransportedVector);
        }
        *component = value;
    }
    Ok(transported)
}

/// Parallel-transport `vector` along the polyline through `path`, transporting
/// across each consecutive segment with `substeps` RK4 steps per segment.
///
/// A `path` with fewer than two points transports across nothing and returns
/// the (validated) vector unchanged.
pub fn transport_along_polyline<C, const D: usize>(
    connection: &C,
    path: &[[f64; D]],
    vector: &[f64; D],
    substeps: usize,
) -> Result<[f64; D], RelativityError>
where
    C: Connection<D>,
{
    validate_finite_vector(vector)?;
    if substeps == 0
    {
        return Err(RelativityError::InvalidTransportResolution);
    }

    let mut transported = *vector;
    for segment in path.windows(2)
    {
        transported =
            transport_along_segment(connection, &segment[0], &segment[1], &transported, substeps)?;
    }
    Ok(transported)
}

/// The holonomy defect of transporting `vector` around the closed loop through
/// `loop_path` (whose last point should coincide with its first): the
/// transported vector minus the original. Zero (to RK4 tolerance) in flat
/// spacetime; in a curved spacetime it encodes the enclosed curvature.
pub fn holonomy_defect<C, const D: usize>(
    connection: &C,
    loop_path: &[[f64; D]],
    vector: &[f64; D],
    substeps: usize,
) -> Result<[f64; D], RelativityError>
where
    C: Connection<D>,
{
    let transported = transport_along_polyline(connection, loop_path, vector, substeps)?;
    let mut defect = [0.0_f64; D];
    for (axis, component) in defect.iter_mut().enumerate()
    {
        *component = transported[axis] - vector[axis];
    }
    Ok(defect)
}
