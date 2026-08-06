//! Validation of the linearized-gravity module against the design-note oracles
//! (`docs/LAYER_2_COVARIANT_GRAVITY.md`):
//!
//! 1. weak-field Schwarzschild is linearized-vacuum (`G^(1) = 0`);
//! 2. the Newtonian limit reproduces the Poisson equation (`G^(1)_00 = 2 nabla^2 Phi`);
//! 3. the linearized Riemann tensor is gauge invariant;
//! 4. the linearized Ricci scalar matches the nonlinear curvature to `O(h^2)`.

use scirust_relativity::{
    Connection, CurvatureTensors, LinearizedField, Metric, RelativityError, numerical_christoffel,
};

const STEP: f64 = 1.0e-4;

/// Newtonian perturbation from a potential `Phi = a r^2`:
/// `h_00 = -2 Phi`, `h_ij = -2 Phi delta_ij` (`r^2 = x1^2 + x2^2 + x3^2`).
fn newtonian(amplitude: f64) -> impl Fn(&[f64; 4]) -> [[f64; 4]; 4] {
    move |x| {
        let phi = amplitude * (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
        let mut h = [[0.0; 4]; 4];
        h[0][0] = -2.0 * phi;
        h[1][1] = -2.0 * phi;
        h[2][2] = -2.0 * phi;
        h[3][3] = -2.0 * phi;
        h
    }
}

/// Far-field Schwarzschild perturbation: `h_00 = 2M/r`, `h_ij = (2M/r) delta_ij`.
fn weak_schwarzschild(mass: f64) -> impl Fn(&[f64; 4]) -> [[f64; 4]; 4] {
    move |x| {
        let r = (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();
        let factor = 2.0 * mass / r;
        let mut h = [[0.0; 4]; 4];
        h[0][0] = factor;
        h[1][1] = factor;
        h[2][2] = factor;
        h[3][3] = factor;
        h
    }
}

/// A pure-gauge perturbation `Delta h_(mu nu) = d_mu xi_nu + d_nu xi_mu` for the
/// cubic field `xi = ((x1)^3, (x2)^3, 0, 0)`: `Delta h_01 = 3 x1^2`,
/// `Delta h_12 = 3 x2^2`. Cubic `xi` makes the second derivatives of `Delta h`
/// non-trivial, so gauge invariance is a real cancellation, not `0 = 0`.
fn gauge(x: &[f64; 4]) -> [[f64; 4]; 4] {
    let mut h = [[0.0; 4]; 4];
    let time_space = 3.0 * x[1] * x[1];
    let space_space = 3.0 * x[2] * x[2];
    h[0][1] = time_space;
    h[1][0] = time_space;
    h[1][2] = space_space;
    h[2][1] = space_space;
    h
}

fn add(left: &[[f64; 4]; 4], right: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut sum = [[0.0; 4]; 4];
    for (row, (left_row, right_row)) in sum.iter_mut().zip(left.iter().zip(right.iter()))
    {
        for (value, (l, r)) in row.iter_mut().zip(left_row.iter().zip(right_row.iter()))
        {
            *value = l + r;
        }
    }
    sum
}

fn max_abs_matrix(matrix: &[[f64; 4]; 4]) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
}

fn max_abs_riemann(riemann: &[[[[f64; 4]; 4]; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for block in riemann
    {
        for plane in block
        {
            for row in plane
            {
                for &value in row
                {
                    worst = worst.max(value.abs());
                }
            }
        }
    }
    worst
}

fn max_riemann_difference(left: &[[[[f64; 4]; 4]; 4]; 4], right: &[[[[f64; 4]; 4]; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for (left_block, right_block) in left.iter().zip(right.iter())
    {
        for (left_plane, right_plane) in left_block.iter().zip(right_block.iter())
        {
            for (left_row, right_row) in left_plane.iter().zip(right_plane.iter())
            {
                for (l, r) in left_row.iter().zip(right_row.iter())
                {
                    worst = worst.max((l - r).abs());
                }
            }
        }
    }
    worst
}

#[test]
fn newtonian_limit_reproduces_poisson() {
    // Phi = a r^2 has nabla^2 Phi = 6a, so G^(1)_00 = 2 nabla^2 Phi = 12a. The
    // perturbation is quadratic, so the central differences are exact.
    let amplitude = 1.0e-3;
    let field = LinearizedField::compute(newtonian(amplitude), &[0.3, 1.0, -2.0, 0.5], STEP)
        .expect("finite linearized field");
    assert!(
        (field.einstein[0][0] - 12.0 * amplitude).abs() < 1.0e-8,
        "G_00 = {}, expected 12a = {}",
        field.einstein[0][0],
        12.0 * amplitude
    );
    // The time-space Einstein components vanish for this static field.
    for spatial in 1..4
    {
        assert!(field.einstein[0][spatial].abs() < 1.0e-8);
    }
}

#[test]
fn weak_schwarzschild_is_linearized_vacuum() {
    // Off-axis point: h ~ 1/r is not polynomial, so G^(1) = 0 holds to the
    // finite-difference truncation (~1e-8 here), not exactly.
    let field = LinearizedField::compute(weak_schwarzschild(1.0), &[0.0, 6.0, 3.0, 2.0], STEP)
        .expect("finite linearized field");
    assert!(
        max_abs_matrix(&field.einstein) < 1.0e-6,
        "max|G^(1)| = {}",
        max_abs_matrix(&field.einstein)
    );
}

#[test]
fn linearized_riemann_is_gauge_invariant() {
    let point = [0.3, 1.0, -2.0, 0.5];

    // A pure-gauge perturbation has zero linearized Riemann (exactly, since it is
    // polynomial).
    let pure = LinearizedField::compute(gauge, &point, STEP).expect("finite linearized field");
    assert!(
        max_abs_riemann(&pure.riemann) < 1.0e-10,
        "pure-gauge max|R^(1)| = {}",
        max_abs_riemann(&pure.riemann)
    );

    // Adding a gauge term to a real perturbation leaves the Riemann unchanged.
    let base = weak_schwarzschild(1.0);
    let base_field =
        LinearizedField::compute(&base, &point, STEP).expect("finite linearized field");
    let combined_field = LinearizedField::compute(|x| add(&base(x), &gauge(x)), &point, STEP)
        .expect("finite linearized field");
    assert!(
        max_riemann_difference(&base_field.riemann, &combined_field.riemann) < 1.0e-10,
        "gauge-shift Riemann change = {}",
        max_riemann_difference(&base_field.riemann, &combined_field.riemann)
    );
}

#[test]
fn linearized_scalar_matches_nonlinear_to_second_order() {
    // For g = eta + eps h, the nonlinear Ricci scalar is eps R^(1)(h) + O(eps^2):
    // the linearized scalar is the exact leading term, and the residual falls
    // quadratically as eps halves.
    let amplitude = 1.0e-3;
    let point = [0.0, 1.0, -2.0, 0.5];
    let linear_unit = LinearizedField::compute(newtonian(amplitude), &point, STEP)
        .expect("finite linearized field")
        .ricci_scalar;

    let mut previous_residual = f64::INFINITY;
    for epsilon in [0.1_f64, 0.05, 0.025]
    {
        let background = ScaledNewtonian { epsilon, amplitude };
        let nonlinear = CurvatureTensors::compute(&background, &point, STEP)
            .expect("finite curvature")
            .ricci_scalar();
        let linear = epsilon * linear_unit;
        let residual = (nonlinear - linear).abs();

        // The linear term dominates: the residual is a small fraction of it.
        assert!(
            residual < 0.1 * linear.abs(),
            "eps={epsilon}: residual {residual} not << linear {linear}"
        );
        // Quadratic convergence: halving eps roughly quarters the residual.
        assert!(
            residual < 0.35 * previous_residual,
            "eps={epsilon}: residual {residual} did not fall quadratically from {previous_residual}"
        );
        previous_residual = residual;
    }
}

#[test]
fn reports_typed_errors() {
    assert!(matches!(
        LinearizedField::compute(newtonian(1.0e-3), &[f64::NAN, 1.0, 2.0, 0.5], STEP),
        Err(RelativityError::NonFiniteCoordinate(0)),
    ));
    assert!(matches!(
        LinearizedField::compute(newtonian(1.0e-3), &[0.0, 1.0, 2.0, 0.5], 0.0),
        Err(RelativityError::InvalidDifferenceStep(_)),
    ));
}

/// `g = eta + eps * h_newtonian` as a `Metric + Connection` background, for the
/// nonlinear cross-check (oracle 4).
#[derive(Clone, Copy)]
struct ScaledNewtonian {
    epsilon: f64,
    amplitude: f64,
}

impl Metric<4> for ScaledNewtonian {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let phi = self.amplitude
            * (coordinates[1] * coordinates[1]
                + coordinates[2] * coordinates[2]
                + coordinates[3] * coordinates[3]);
        let entry = -2.0 * self.epsilon * phi;
        let mut g = [[0.0; 4]; 4];
        g[0][0] = -1.0 + entry;
        g[1][1] = 1.0 + entry;
        g[2][2] = 1.0 + entry;
        g[3][3] = 1.0 + entry;
        g
    }
}

impl Connection<4> for ScaledNewtonian {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        numerical_christoffel(self, coordinates, STEP).unwrap_or([[[f64::NAN; 4]; 4]; 4])
    }
}
