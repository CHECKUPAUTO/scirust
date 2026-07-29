//! Geodesic deviation (Jacobi fields).
//!
//! A Jacobi field `xi^mu(tau)` measures the infinitesimal separation of a
//! one-parameter family of geodesics. Along a geodesic with tangent `u^mu` it
//! obeys the geodesic-deviation (Jacobi) equation
//!
//! ```text
//! D^2 xi^mu / dtau^2 = - R^mu_(nu rho sigma) u^nu xi^rho u^sigma,
//! ```
//!
//! where `D/dtau` is the covariant derivative along the geodesic. This module
//! integrates the coupled first-order system for the geodesic and its
//! deviation, `(x, u, xi, w)` with `w = D xi / dtau`, using the deterministic
//! RK4 integrator from `scirust-sim` and the Riemann tensor from
//! [`crate::CurvatureTensors`] evaluated along the path.
//!
//! The convention (which slot of the Riemann tensor each of `u`, `xi`, `u`
//! contracts into, and the overall sign) is fixed to reproduce the actual
//! coordinate separation of two nearby geodesics — the convention-free ground
//! truth checked in this crate's tests. In flat spacetime the deviation grows
//! linearly (`xi = xi_0 + tau xi_dot_0`); in a curved spacetime it focuses or
//! defocuses according to the curvature.

use crate::{Connection, CurvatureTensors, Metric, RelativityError};
use scirust_sim::{System, simulate};

/// One sample of a Jacobi field along a geodesic: the affine parameter, the
/// geodesic position, and the deviation vector there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JacobiSample<const D: usize> {
    /// Affine parameter `tau` of this sample.
    pub affine_parameter: f64,
    /// Geodesic position `x^mu(tau)`.
    pub position: [f64; D],
    /// Deviation vector `xi^mu(tau)`.
    pub deviation: [f64; D],
}

/// The coupled geodesic + Jacobi system with state `[x, u, xi, w]`
/// (`w = D xi / dtau`), of dimension `4 D`.
struct JacobiSystem<'a, B, const D: usize> {
    background: &'a B,
    curvature_step: f64,
}

impl<B, const D: usize> System for JacobiSystem<'_, B, D>
where
    B: Metric<D> + Connection<D>,
{
    fn dim(&self) -> usize {
        4 * D
    }

    fn derivatives(&self, _parameter: f64, state: &[f64], output: &mut [f64]) {
        let mut position = [0.0_f64; D];
        let mut velocity = [0.0_f64; D];
        let mut deviation = [0.0_f64; D];
        let mut deviation_rate = [0.0_f64; D];
        position.copy_from_slice(&state[..D]);
        velocity.copy_from_slice(&state[D..2 * D]);
        deviation.copy_from_slice(&state[2 * D..3 * D]);
        deviation_rate.copy_from_slice(&state[3 * D..4 * D]);

        let christoffel = self.background.christoffel(&position);
        let riemann =
            match CurvatureTensors::compute(self.background, &position, self.curvature_step)
            {
                Ok(tensors) => *tensors.riemann(),
                // Propagate failure as non-finite derivatives; the integrator then
                // reports a non-finite state rather than silently continuing.
                Err(_) =>
                {
                    output.iter_mut().for_each(|value| *value = f64::NAN);
                    return;
                },
            };

        // dx^mu/dtau = u^mu.
        output[..D].copy_from_slice(&velocity);

        for rho in 0..D
        {
            // du^rho/dtau = - Gamma^rho_(mu nu) u^mu u^nu (geodesic).
            let mut acceleration = 0.0;
            // dxi^rho/dtau = w^rho - Gamma^rho_(mu nu) u^mu xi^nu.
            let mut deviation_velocity = deviation_rate[rho];
            // Covariant correction for dw^rho/dtau: - Gamma^rho_(mu nu) u^mu w^nu.
            let mut rate_correction = 0.0;
            for (mu, christoffel_mu) in christoffel[rho].iter().enumerate()
            {
                let velocity_mu = velocity[mu];
                for nu in 0..D
                {
                    let symbol = christoffel_mu[nu];
                    acceleration -= symbol * velocity_mu * velocity[nu];
                    deviation_velocity -= symbol * velocity_mu * deviation[nu];
                    rate_correction -= symbol * velocity_mu * deviation_rate[nu];
                }
            }

            // Jacobi source J^rho = - R^rho_(nu rho' sigma) u^nu xi^rho' u^sigma.
            let mut jacobi_source = 0.0;
            for (nu, riemann_nu) in riemann[rho].iter().enumerate()
            {
                let velocity_nu = velocity[nu];
                for (rho_prime, riemann_rho) in riemann_nu.iter().enumerate()
                {
                    let deviation_rho = deviation[rho_prime];
                    for sigma in 0..D
                    {
                        jacobi_source -=
                            riemann_rho[sigma] * velocity_nu * deviation_rho * velocity[sigma];
                    }
                }
            }

            output[D + rho] = acceleration;
            output[2 * D + rho] = deviation_velocity;
            output[3 * D + rho] = jacobi_source + rate_correction;
        }
    }
}

/// Contract the Christoffel symbols: `Gamma^rho_(mu nu) a^mu b^nu`.
fn christoffel_contraction<const D: usize>(
    christoffel: &[[[f64; D]; D]; D],
    left: &[f64; D],
    right: &[f64; D],
) -> [f64; D] {
    let mut result = [0.0_f64; D];
    for (rho, value) in result.iter_mut().enumerate()
    {
        let mut accumulator = 0.0;
        for (mu, christoffel_mu) in christoffel[rho].iter().enumerate()
        {
            for nu in 0..D
            {
                accumulator += christoffel_mu[nu] * left[mu] * right[nu];
            }
        }
        *value = accumulator;
    }
    result
}

fn validate_finite<const D: usize>(
    vector: &[f64; D],
    error: RelativityError,
) -> Result<(), RelativityError> {
    if vector.iter().any(|value| !value.is_finite())
    {
        return Err(error);
    }
    Ok(())
}

/// Integrate a Jacobi field along the geodesic from `position` with tangent
/// `velocity`, given the initial deviation `deviation` and its initial
/// coordinate rate `deviation_velocity = dxi/dtau` at `tau = 0`, out to affine
/// parameter `affine_length` with RK4 step `step`. The Riemann tensor in the
/// Jacobi source is evaluated by central differences with `curvature_step`.
///
/// Returns the sampled `(tau, x, xi)` triples. Errors are typed
/// ([`RelativityError`]); it never panics and never returns a non-finite
/// deviation.
///
/// # Example
///
/// In flat spacetime a Jacobi field grows exactly linearly,
/// `xi(tau) = xi_0 + tau xi_dot_0`.
///
/// ```
/// use scirust_relativity::{integrate_geodesic_deviation, Minkowski};
///
/// let samples = integrate_geodesic_deviation(
///     &Minkowski,
///     &[0.0, 0.0, 0.0, 0.0],       // position
///     &[1.0, 0.0, 0.0, 0.0],       // geodesic tangent
///     &[0.0, 1.0, 0.0, 0.0],       // initial deviation
///     &[0.0, 0.5, 0.0, 0.0],       // initial deviation rate
///     2.0,                          // affine length
///     0.01,
///     1.0e-4,
/// )
/// .expect("flat Jacobi field");
///
/// let end = samples.last().unwrap().deviation;
/// assert!((end[1] - (1.0 + 2.0 * 0.5)).abs() < 1.0e-9); // 1 + tau * 0.5 = 2.0
/// ```
#[allow(clippy::too_many_arguments)]
pub fn integrate_geodesic_deviation<B, const D: usize>(
    background: &B,
    position: &[f64; D],
    velocity: &[f64; D],
    deviation: &[f64; D],
    deviation_velocity: &[f64; D],
    affine_length: f64,
    step: f64,
    curvature_step: f64,
) -> Result<Vec<JacobiSample<D>>, RelativityError>
where
    B: Metric<D> + Connection<D>,
{
    validate_finite(position, RelativityError::NonFiniteCoordinate(0))?;
    validate_finite(velocity, RelativityError::NonFiniteDeviationVector)?;
    validate_finite(deviation, RelativityError::NonFiniteDeviationVector)?;
    validate_finite(
        deviation_velocity,
        RelativityError::NonFiniteDeviationVector,
    )?;
    if !step.is_finite() || step <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(step));
    }
    if !curvature_step.is_finite() || curvature_step <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(curvature_step));
    }
    if !affine_length.is_finite() || affine_length <= 0.0
    {
        return Err(RelativityError::InvalidAffineLength(affine_length));
    }

    // The initial covariant rate is w_0 = dxi/dtau + Gamma(u, xi), so that the
    // integrator's deviation reproduces the coordinate separation of two
    // geodesics whose initial velocities differ by `deviation_velocity`.
    let christoffel = background.christoffel(position);
    let covariant_rate = christoffel_contraction(&christoffel, velocity, deviation);

    let mut initial = vec![0.0_f64; 4 * D];
    initial[..D].copy_from_slice(position);
    initial[D..2 * D].copy_from_slice(velocity);
    initial[2 * D..3 * D].copy_from_slice(deviation);
    for rho in 0..D
    {
        initial[3 * D + rho] = deviation_velocity[rho] + covariant_rate[rho];
    }

    let system = JacobiSystem {
        background,
        curvature_step,
    };
    let trajectory = simulate(&system, &initial, 0.0, affine_length, step)
        .map_err(|_| RelativityError::NonFiniteDeviationVector)?;

    let mut samples = Vec::with_capacity(trajectory.len());
    for (parameter, state) in trajectory.t.iter().zip(trajectory.y.iter())
    {
        let mut sample_position = [0.0_f64; D];
        let mut sample_deviation = [0.0_f64; D];
        sample_position.copy_from_slice(&state[..D]);
        sample_deviation.copy_from_slice(&state[2 * D..3 * D]);
        validate_finite(&sample_deviation, RelativityError::NonFiniteDeviationVector)?;
        samples.push(JacobiSample {
            affine_parameter: *parameter,
            position: sample_position,
            deviation: sample_deviation,
        });
    }

    Ok(samples)
}
