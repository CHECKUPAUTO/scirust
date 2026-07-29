//! Validation of Synge's world function and its gradient bitensors.
//!
//! - Flat spacetime: `sigma = (1/2) eta(Δ, Δ)`, `sigma^{mu'} = -Δ`,
//!   `sigma^mu = +Δ` exactly (`Δ = x - x'`).
//! - Curved spacetimes (Schwarzschild, de Sitter, anti-de Sitter): the
//!   convention-free identities hold to the shooting tolerance — base/field
//!   symmetry `sigma(x', x) = sigma(x, x')`, the field-point fundamental identity
//!   `2 sigma = g(field) sigma^mu sigma^mu`, and the gradient round trip
//!   `exp_base(-sigma^{mu'}) = field`.
//! - Coincidence: `sigma` and both gradients vanish when the points coincide.
//! - Inputs are validated with typed errors.

use scirust_relativity::{
    AntiDeSitter, Connection, DeSitter, Metric, Minkowski, RelativityError, Schwarzschild,
    WorldFunction, WorldFunctionSettings, geodesic_exponential, metric_norm, world_function,
    world_function_with_gradients,
};
use std::f64::consts::FRAC_PI_2;

fn settings() -> WorldFunctionSettings {
    WorldFunctionSettings::default()
}

fn assert_close_vector(actual: &[f64; 4], expected: &[f64; 4], tolerance: f64) {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (a - e).abs() <= tolerance,
            "component {i}: actual={a:.12e} expected={e:.12e}"
        );
    }
}

// --------------------------------------------------------------------------
// Flat spacetime: exact
// --------------------------------------------------------------------------

#[test]
fn flat_world_function_and_gradients_are_exact() {
    let base = [0.0, 1.0, 2.0, 0.5];
    let field = [2.0, 1.5, 1.0, 0.9];
    let separation = [
        field[0] - base[0],
        field[1] - base[1],
        field[2] - base[2],
        field[3] - base[3],
    ];

    let WorldFunction {
        sigma,
        gradient_base,
        gradient_field,
    } = world_function_with_gradients(&Minkowski, &base, &field, &settings()).unwrap();

    // sigma = (1/2) eta(Δ, Δ) = (1/2)(-4 + 0.25 + 1 + 0.16) = -1.295.
    assert!((sigma - (-1.295)).abs() < 1.0e-12, "sigma = {sigma}");

    // sigma^{mu'} = -Δ at the base point; sigma^mu = +Δ at the field point.
    let negated = separation.map(|component| -component);
    assert_close_vector(&gradient_base, &negated, 1.0e-12);
    assert_close_vector(&gradient_field, &separation, 1.0e-12);

    // The scalar-only entry point agrees with the struct.
    let sigma_scalar = world_function(&Minkowski, &base, &field, &settings()).unwrap();
    assert_eq!(sigma_scalar.to_bits(), sigma.to_bits());
}

// --------------------------------------------------------------------------
// Curved spacetimes: convention-free identities (shooting tolerance)
// --------------------------------------------------------------------------

fn check_curved_identities<B>(background: &B, base: [f64; 4], field: [f64; 4])
where
    B: Metric<4> + Connection<4> + Copy,
{
    let result = world_function_with_gradients(background, &base, &field, &settings()).unwrap();

    // Base/field symmetry: sigma(x', x) = sigma(x, x') via independent shootings.
    let sigma_reversed = world_function(background, &field, &base, &settings()).unwrap();
    assert!(
        (result.sigma - sigma_reversed).abs() < 1.0e-8,
        "symmetry gap: {} vs {}",
        result.sigma,
        sigma_reversed
    );

    // Field-point fundamental identity: 2 sigma = g(field) sigma^mu sigma^mu.
    let metric_field = background.components(&field);
    let field_norm = metric_norm(&metric_field, &result.gradient_field);
    assert!(
        (2.0 * result.sigma - field_norm).abs() < 1.0e-8,
        "fundamental-identity gap: 2 sigma = {}, g(field)(grad, grad) = {field_norm}",
        2.0 * result.sigma
    );

    // Gradient round trip: exp_base(-sigma^{mu'}) = field, exp_field(-sigma^mu) = base.
    let recovered_field = geodesic_exponential(
        background,
        &base,
        &result.gradient_base.map(|c| -c),
        settings().step,
    )
    .unwrap();
    assert_close_vector(&recovered_field, &field, 1.0e-8);
    let recovered_base = geodesic_exponential(
        background,
        &field,
        &result.gradient_field.map(|c| -c),
        settings().step,
    )
    .unwrap();
    assert_close_vector(&recovered_base, &base, 1.0e-8);
}

#[test]
fn curved_identities_schwarzschild() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    check_curved_identities(
        &background,
        [0.0, 12.0, FRAC_PI_2, 0.0],
        [0.15, 12.4, FRAC_PI_2 + 0.03, 0.05],
    );
}

#[test]
fn curved_identities_de_sitter() {
    let background = DeSitter::try_new(0.05).unwrap();
    check_curved_identities(
        &background,
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
    );
}

#[test]
fn curved_identities_anti_de_sitter() {
    let background = AntiDeSitter::try_new(0.05).unwrap();
    check_curved_identities(
        &background,
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
    );
}

// --------------------------------------------------------------------------
// Coincidence limit
// --------------------------------------------------------------------------

#[test]
fn coincidence_limit_is_zero() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let point = [0.0, 10.0, FRAC_PI_2, 0.0];
    let result = world_function_with_gradients(&background, &point, &point, &settings()).unwrap();
    assert_eq!(result.sigma, 0.0);
    assert_eq!(result.gradient_base, [0.0; 4]);
    assert_eq!(result.gradient_field, [0.0; 4]);
}

// --------------------------------------------------------------------------
// Error paths
// --------------------------------------------------------------------------

#[test]
fn world_function_reports_typed_errors() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let base = [0.0, 10.0, FRAC_PI_2, 0.0];
    let field = [0.1, 10.4, FRAC_PI_2 + 0.05, 0.03];

    // Non-finite base coordinate propagates from the logarithm shooting.
    assert_eq!(
        world_function(
            &background,
            &[f64::NAN, 10.0, FRAC_PI_2, 0.0],
            &field,
            &settings()
        ),
        Err(RelativityError::NonFiniteCoordinate(0)),
    );

    // A one-iteration cap cannot reach the tolerance on a curved background.
    let starved = WorldFunctionSettings {
        max_iterations: 1,
        ..WorldFunctionSettings::default()
    };
    assert_eq!(
        world_function(&background, &base, &field, &starved),
        Err(RelativityError::LogarithmMapDidNotConverge),
    );
}
