//! Exact closed-form parallel transport along a circular equatorial geodesic
//! orbit in Schwarzschild, exploiting a different structural fact than the
//! flat-spacetime oracle: not path-independence (curvature is nonzero here),
//! but the *constancy* of the transport generator along this one special
//! family of paths.
//!
//! Schwarzschild is stationary and axisymmetric, so its Christoffel symbols
//! depend only on `r` and `theta`
//! (see [`scirust_relativity::Schwarzschild::christoffel`]). Along a circular
//! equatorial orbit both are fixed (`r = r_0`, `theta = pi/2`), and the
//! orbit's four-velocity `u = (u^t, 0, 0, u^phi)` is itself constant. The
//! parallel transport equation `dV^mu/dlambda = -Gamma^mu_(alpha beta) u^alpha
//! V^beta` therefore reduces, along this specific path, to a *linear,
//! constant-coefficient* ODE `dV/dlambda = -A V` for the fixed generator
//! matrix `A^mu_beta = Gamma^mu_(alpha beta) u^alpha`, which has the exact
//! closed-form solution `V(lambda) = exp(-lambda A) V(0)`.
//!
//! This is **not** a general bitensor propagator: it is exact only for the
//! one-parameter family of circular equatorial geodesic orbits, exactly as
//! [`crate::exact_cylindrical_minkowski_transport`] is exact only for flat
//! spacetime. It must never be described as valid for a general curved path,
//! an eccentric or inclined orbit, or any other trajectory.

use crate::{NonlocalRelativityError, NonlocalResult};
use scirust_relativity::{Connection, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

const MATRIX_EXPONENTIAL_TAYLOR_TERMS: usize = 24;

/// Angular velocity `d(phi)/d(t)` of a circular equatorial geodesic orbit at
/// Schwarzschild-coordinate radius `radius`: `sqrt(M / r^3)`, the
/// general-relativistic form of Kepler's third law (exact in these
/// coordinates for circular equatorial orbits, not merely a weak-field
/// approximation).
///
/// `radius` must be finite and strictly exceed `3 M`, the existence bound
/// for a circular equatorial timelike geodesic (below it no circular
/// geodesic is timelike; the innermost *stable* circular orbit is a
/// separate, larger bound at `6 M` that this function does not enforce,
/// since stability is irrelevant to evaluating the transport along a
/// mathematically valid orbit).
pub fn schwarzschild_circular_orbit_angular_velocity(
    background: &Schwarzschild,
    radius: f64,
) -> NonlocalResult<f64> {
    validate_circular_orbit_radius(background, radius)?;
    let mass = background.mass();
    Ok((mass / (radius * radius * radius)).sqrt())
}

/// Four-velocity `u = (u^t, 0, 0, u^phi)` of a circular equatorial geodesic
/// orbit at Schwarzschild-coordinate radius `radius`, normalized so that
/// `g(u, u) = -1` (the orbit's affine parameter is its own proper time).
///
/// See [`schwarzschild_circular_orbit_angular_velocity`] for the radius
/// validity bound.
pub fn schwarzschild_circular_orbit_four_velocity(
    background: &Schwarzschild,
    radius: f64,
) -> NonlocalResult<[f64; 4]> {
    validate_circular_orbit_radius(background, radius)?;
    let mass = background.mass();
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let angular_velocity = (mass / (radius * radius * radius)).sqrt();
    let time_component = 1.0 / denominator;
    let angular_component = angular_velocity / denominator;
    let four_velocity = [time_component, 0.0, 0.0, angular_component];

    for (component, value) in four_velocity.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteTransportedVector {
                retained_index: 0,
                component,
                value,
            });
        }
    }

    Ok(four_velocity)
}

/// Exact closed-form parallel transport of `vector` along a circular
/// equatorial geodesic orbit at Schwarzschild-coordinate radius `radius`,
/// advancing the affine parameter (equal to proper time along this orbit) by
/// `delta_lambda`, which may be negative to transport backward along the
/// orbit.
///
/// This evaluates `V(delta_lambda) = exp(-delta_lambda * A) V(0)` for the
/// constant generator `A` described in this module's documentation, using a
/// deterministic scaling-and-squaring matrix exponential (fixed-length
/// Taylor series, no data-dependent term count). The Christoffel symbols are
/// taken directly from [`scirust_relativity::Schwarzschild::christoffel`],
/// the same already-validated implementation used everywhere else in this
/// crate; this function does not re-derive them.
///
/// `radius` must satisfy the bound documented on
/// [`schwarzschild_circular_orbit_angular_velocity`]; `vector` and
/// `delta_lambda` must be finite.
pub fn exact_schwarzschild_circular_orbit_transport(
    background: &Schwarzschild,
    radius: f64,
    vector: [f64; 4],
    delta_lambda: f64,
) -> NonlocalResult<[f64; 4]> {
    let four_velocity = schwarzschild_circular_orbit_four_velocity(background, radius)?;

    if !delta_lambda.is_finite()
    {
        return Err(NonlocalRelativityError::InvalidTransportSegmentStep(
            delta_lambda,
        ));
    }

    for (component, value) in vector.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteTransportedVector {
                retained_index: 0,
                component,
                value,
            });
        }
    }

    let coordinates = [0.0, radius, FRAC_PI_2, 0.0];
    let christoffel = background.christoffel(&coordinates);

    for (rho, rho_values) in christoffel.iter().enumerate()
    {
        for (mu, mu_values) in rho_values.iter().enumerate()
        {
            for (nu, value) in mu_values.iter().copied().enumerate()
            {
                if !value.is_finite()
                {
                    return Err(NonlocalRelativityError::NonFiniteTransportChristoffel {
                        rho,
                        mu,
                        nu,
                        value,
                    });
                }
            }
        }
    }

    let mut generator = [[0.0_f64; 4]; 4];
    for (rho, generator_row) in generator.iter_mut().enumerate()
    {
        for (nu, generator_entry) in generator_row.iter_mut().enumerate()
        {
            let mut sum = 0.0;
            for mu in 0..4
            {
                sum += christoffel[rho][mu][nu] * four_velocity[mu];
            }
            *generator_entry = sum;
        }
    }

    let evolution_generator = matrix_scale_4(&generator, -delta_lambda);
    let evolution_operator = matrix_exponential_4(&evolution_generator);
    let transported = matrix_vector_multiply_4(&evolution_operator, &vector);

    for (component, value) in transported.iter().copied().enumerate()
    {
        if !value.is_finite()
        {
            return Err(NonlocalRelativityError::NonFiniteTransportedVector {
                retained_index: 0,
                component,
                value,
            });
        }
    }

    Ok(transported)
}

fn validate_circular_orbit_radius(background: &Schwarzschild, radius: f64) -> NonlocalResult<()> {
    if !radius.is_finite() || radius <= 3.0 * background.mass()
    {
        return Err(NonlocalRelativityError::InvalidCircularOrbitRadius(radius));
    }

    Ok(())
}

const IDENTITY_4: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn matrix_infinity_norm(matrix: &[[f64; 4]; 4]) -> f64 {
    let mut max_row_sum = 0.0_f64;

    for row in matrix
    {
        let row_sum: f64 = row.iter().map(|value| value.abs()).sum();
        if row_sum > max_row_sum
        {
            max_row_sum = row_sum;
        }
    }

    max_row_sum
}

fn matrix_multiply_4(left: &[[f64; 4]; 4], right: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut result = [[0.0_f64; 4]; 4];

    for (row, result_row) in result.iter_mut().enumerate()
    {
        for (column, result_entry) in result_row.iter_mut().enumerate()
        {
            let mut sum = 0.0;
            for inner in 0..4
            {
                sum += left[row][inner] * right[inner][column];
            }
            *result_entry = sum;
        }
    }

    result
}

fn matrix_vector_multiply_4(matrix: &[[f64; 4]; 4], vector: &[f64; 4]) -> [f64; 4] {
    let mut result = [0.0_f64; 4];

    for (row, result_entry) in result.iter_mut().enumerate()
    {
        let mut sum = 0.0;
        for column in 0..4
        {
            sum += matrix[row][column] * vector[column];
        }
        *result_entry = sum;
    }

    result
}

fn matrix_scale_4(matrix: &[[f64; 4]; 4], scalar: f64) -> [[f64; 4]; 4] {
    let mut result = [[0.0_f64; 4]; 4];

    for (row, result_row) in result.iter_mut().enumerate()
    {
        for (column, result_entry) in result_row.iter_mut().enumerate()
        {
            *result_entry = matrix[row][column] * scalar;
        }
    }

    result
}

fn matrix_add_4(left: &[[f64; 4]; 4], right: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut result = [[0.0_f64; 4]; 4];

    for (row, result_row) in result.iter_mut().enumerate()
    {
        for (column, result_entry) in result_row.iter_mut().enumerate()
        {
            *result_entry = left[row][column] + right[row][column];
        }
    }

    result
}

/// Deterministic matrix exponential of a 4x4 real matrix via scaling and
/// squaring with a fixed-length Taylor series.
///
/// This is a standard numerical linear algebra primitive (the same strategy
/// used by widely deployed `expm` implementations), not a physics-specific
/// construction. The scaling exponent is chosen from the matrix's own
/// infinity norm so the truncated Taylor series always evaluates a
/// small-norm argument; the term count is fixed regardless of input,
/// preserving bit-for-bit determinism.
fn matrix_exponential_4(generator: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let norm = matrix_infinity_norm(generator);
    let scaling_exponent: u32 = if norm <= 0.5
    {
        0
    }
    else
    {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    };
    let scale_factor = 2.0_f64.powi(scaling_exponent as i32);
    let scaled = matrix_scale_4(generator, 1.0 / scale_factor);

    let mut term = IDENTITY_4;
    let mut sum = IDENTITY_4;

    for term_index in 1..=MATRIX_EXPONENTIAL_TAYLOR_TERMS
    {
        term = matrix_scale_4(&matrix_multiply_4(&term, &scaled), 1.0 / term_index as f64);
        sum = matrix_add_4(&sum, &term);
    }

    let mut result = sum;
    for _ in 0..scaling_exponent
    {
        result = matrix_multiply_4(&result, &result);
    }

    result
}
