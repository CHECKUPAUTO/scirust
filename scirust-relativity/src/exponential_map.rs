//! Geodesic exponential and logarithm maps.
//!
//! The exponential map `exp_p(v)` follows the geodesic leaving `p` with initial
//! tangent `v` for unit affine parameter and returns the endpoint. The
//! logarithm map `log_p(q)` is its local inverse: the tangent `v` at `p` with
//! `exp_p(v) = q`, found by Newton shooting (the exponential map's Jacobian is
//! evaluated by central finite differences and inverted with the crate's
//! deterministic Gauss-Jordan routine).
//!
//! In flat spacetime `exp_p(v) = p + v` and `log_p(q) = q - p` exactly; in a
//! curved spacetime they are local inverses, `exp_p(log_p(q)) = q` and
//! `log_p(exp_p(v)) = v` for `q` within the geodesically convex neighbourhood of
//! `p`, which this crate's tests verify as a round-trip identity. Both reuse the
//! [`crate::GeodesicSystem`] integrator (the same deterministic RK4 engine as
//! the rest of the crate); no new geodesic solver is introduced.

use crate::{Connection, GeodesicSystem, RelativityError, invert_metric};
use scirust_sim::simulate;

/// Evaluate the exponential map `exp_p(v)`: the endpoint of the geodesic from
/// `position` with initial tangent `velocity` at unit affine parameter,
/// integrated with RK4 step `step`.
///
/// Returns a typed [`RelativityError`] for a non-finite position, an invalid
/// step, or a geodesic that cannot be integrated to a finite endpoint; it never
/// panics.
///
/// # Example
///
/// In flat spacetime the exponential map is a translation: `exp_p(v) = p + v`.
///
/// ```
/// use scirust_relativity::{geodesic_exponential, Minkowski};
///
/// let position = [0.0, 1.0, 2.0, 0.5];
/// let velocity = [1.0, 0.3, -0.2, 0.1];
/// let image = geodesic_exponential(&Minkowski, &position, &velocity, 0.05)
///     .expect("flat exponential map");
///
/// assert!((image[1] - (position[1] + velocity[1])).abs() < 1.0e-12);
/// ```
pub fn geodesic_exponential<B, const D: usize>(
    background: &B,
    position: &[f64; D],
    velocity: &[f64; D],
    step: f64,
) -> Result<[f64; D], RelativityError>
where
    B: Connection<D> + Copy,
{
    if let Some((index, _)) = position
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RelativityError::NonFiniteCoordinate(index));
    }
    if !step.is_finite() || step <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(step));
    }

    let mut initial = vec![0.0_f64; 2 * D];
    initial[..D].copy_from_slice(position);
    initial[D..].copy_from_slice(velocity);

    let system = GeodesicSystem::<B, D>::new(*background);
    let trajectory = simulate(&system, &initial, 0.0, 1.0, step)
        .map_err(|_| RelativityError::ExponentialMapIntegrationFailed)?;
    let endpoint = trajectory
        .last_state()
        .ok_or(RelativityError::ExponentialMapIntegrationFailed)?;

    let mut result = [0.0_f64; D];
    for (component, target) in result.iter_mut().enumerate()
    {
        let value = endpoint[component];
        if !value.is_finite()
        {
            return Err(RelativityError::ExponentialMapIntegrationFailed);
        }
        *target = value;
    }
    Ok(result)
}

/// Evaluate the logarithm map `log_p(q)`: the tangent `v` at `position` with
/// `exp_p(v) = target`, by Newton shooting from the flat-space guess
/// `v_0 = target - position`.
///
/// `step` is the geodesic RK4 step, `jacobian_step` the central-difference step
/// for the exponential map's Jacobian, `tolerance` the Euclidean convergence
/// tolerance on `exp_p(v) - target`, and `max_iterations` the Newton iteration
/// cap. Returns [`RelativityError::LogarithmMapDidNotConverge`] if the tolerance
/// is not met in time, or a singular-Jacobian / integration error; it never
/// panics.
#[allow(clippy::too_many_arguments)]
pub fn geodesic_logarithm<B, const D: usize>(
    background: &B,
    position: &[f64; D],
    target: &[f64; D],
    step: f64,
    jacobian_step: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<[f64; D], RelativityError>
where
    B: Connection<D> + Copy,
{
    for (index, value) in position.iter().chain(target.iter()).enumerate()
    {
        if !value.is_finite()
        {
            return Err(RelativityError::NonFiniteCoordinate(index % D));
        }
    }
    if !jacobian_step.is_finite() || jacobian_step <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(jacobian_step));
    }
    if !tolerance.is_finite() || tolerance <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(tolerance));
    }

    // Flat-space initial guess.
    let mut tangent = [0.0_f64; D];
    for (component, slot) in tangent.iter_mut().enumerate()
    {
        *slot = target[component] - position[component];
    }

    for _ in 0..max_iterations
    {
        let image = geodesic_exponential(background, position, &tangent, step)?;

        let mut residual = [0.0_f64; D];
        let mut residual_norm_squared = 0.0;
        for (component, slot) in residual.iter_mut().enumerate()
        {
            *slot = image[component] - target[component];
            residual_norm_squared += *slot * *slot;
        }
        if residual_norm_squared.sqrt() < tolerance
        {
            return Ok(tangent);
        }

        // Jacobian J[i][j] = d exp_p(v)_i / d v_j by central differences.
        let mut jacobian = [[0.0_f64; D]; D];
        for j in 0..D
        {
            let mut forward = tangent;
            let mut backward = tangent;
            forward[j] += jacobian_step;
            backward[j] -= jacobian_step;
            let image_forward = geodesic_exponential(background, position, &forward, step)?;
            let image_backward = geodesic_exponential(background, position, &backward, step)?;
            for (i, row) in jacobian.iter_mut().enumerate()
            {
                row[j] = (image_forward[i] - image_backward[i]) / (2.0 * jacobian_step);
            }
        }

        // Newton update v <- v - J^{-1} residual.
        let inverse = invert_metric(&jacobian)?;
        for (i, slot) in tangent.iter_mut().enumerate()
        {
            let mut delta = 0.0;
            for (k, &residual_k) in residual.iter().enumerate()
            {
                delta += inverse[i][k] * residual_k;
            }
            *slot -= delta;
        }
    }

    Err(RelativityError::LogarithmMapDidNotConverge)
}
