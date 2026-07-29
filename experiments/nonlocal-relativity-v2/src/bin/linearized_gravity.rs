//! Linearized-gravity checks: the weak-field Einstein equations and their
//! oracles (the opening slice of Layer 2, the Covariant Gravity Workbench).
//!
//! Writing `g = eta + h` with `|h| << 1`, the curvature is linear in `h`. This
//! experiment reports the four design-note oracles
//! (`docs/LAYER_2_COVARIANT_GRAVITY.md`):
//!
//! - **Newtonian Poisson limit.** For `Phi = a r^2`, `G^(1)_00 = 2 nabla^2 Phi
//!   = 12 a` exactly (the perturbation is quadratic, so the differences are
//!   exact).
//! - **Weak-field Schwarzschild is linearized-vacuum.** `h_00 = 2M/r`,
//!   `h_ij = (2M/r) delta_ij` gives `G^(1) = 0` to the finite-difference
//!   truncation, which shrinks as the field point moves outward.
//! - **Gauge invariance.** The linearized Riemann of a pure-gauge perturbation
//!   is zero, and adding a gauge term leaves a real perturbation's Riemann
//!   unchanged.
//! - **O(h^2) cross-check.** For `g = eta + eps h`, the nonlinear Ricci scalar
//!   equals `eps R^(1)(h)` up to `O(eps^2)`; the residual falls quadratically.
//!
//! Established general relativity only; built on the geometry core's
//! finite-difference curvature machinery.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    Connection, CurvatureTensors, LinearizedField, Metric, numerical_christoffel,
};

const STEP: f64 = 1.0e-4;

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

fn main() -> Result<(), String> {
    print_experiment_header(
        "Linearized-gravity weak-field checks",
        "scirust-relativity geometry core (established general relativity)",
        "weak-field Einstein equations from a metric perturbation; established GR, no new physics.",
    );

    // Part A: Newtonian Poisson limit, G_00 = 12a, position-independent.
    println!("# Part A: Newtonian Phi = a r^2 -- G^(1)_00 vs 2 nabla^2 Phi = 12a");
    println!("amplitude,point,g00,expected_12a,abs_error");
    for &amplitude in &[1.0e-3, 5.0e-3]
    {
        for point in [[0.3, 1.0, -2.0, 0.5], [0.0, 4.0, 1.0, -3.0]]
        {
            let field = LinearizedField::compute(newtonian(amplitude), &point, STEP)
                .map_err(|e| e.to_string())?;
            let expected = 12.0 * amplitude;
            let error = (field.einstein[0][0] - expected).abs();
            require_finite(&[("g00", field.einstein[0][0]), ("error", error)])?;
            println!(
                "{amplitude:.1e},{point:?},{:.6e},{expected:.6e},{error:.3e}",
                field.einstein[0][0]
            );
        }
    }

    // Part B: weak Schwarzschild is linearized-vacuum to the truncation floor.
    println!("#");
    println!("# Part B: weak Schwarzschild h_00 = 2M/r -- max|G^(1)| at the truncation floor");
    println!("radius,max_abs_einstein");
    for &radius in &[6.0, 8.0, 12.0, 20.0]
    {
        // A non-symmetric spatial direction (0.6, 0.7, sqrt(0.15)) of unit norm.
        let point = [0.0, radius * 0.6, radius * 0.7, radius * 0.387_298_33];
        let field = LinearizedField::compute(weak_schwarzschild(1.0), &point, STEP)
            .map_err(|e| e.to_string())?;
        let residual = max_abs_matrix(&field.einstein);
        require_finite(&[("residual", residual)])?;
        println!("{radius:.1},{residual:.3e}");
    }

    // Part C: gauge invariance of the linearized Riemann.
    println!("#");
    println!("# Part C: gauge invariance -- pure-gauge Riemann and gauge-shift change");
    println!("quantity,value");
    let point = [0.3, 1.0, -2.0, 0.5];
    let pure = LinearizedField::compute(gauge, &point, STEP).map_err(|e| e.to_string())?;
    let base = weak_schwarzschild(1.0);
    let base_field = LinearizedField::compute(&base, &point, STEP).map_err(|e| e.to_string())?;
    let combined = LinearizedField::compute(
        |x| {
            let mut h = base(x);
            let g = gauge(x);
            for mu in 0..4
            {
                for nu in 0..4
                {
                    h[mu][nu] += g[mu][nu];
                }
            }
            h
        },
        &point,
        STEP,
    )
    .map_err(|e| e.to_string())?;
    let pure_gauge = max_abs_riemann(&pure.riemann);
    let mut gauge_shift = 0.0_f64;
    for (base_block, combined_block) in base_field.riemann.iter().zip(combined.riemann.iter())
    {
        for (base_plane, combined_plane) in base_block.iter().zip(combined_block.iter())
        {
            for (base_row, combined_row) in base_plane.iter().zip(combined_plane.iter())
            {
                for (base_value, combined_value) in base_row.iter().zip(combined_row.iter())
                {
                    gauge_shift = gauge_shift.max((base_value - combined_value).abs());
                }
            }
        }
    }
    require_finite(&[("pure_gauge", pure_gauge), ("gauge_shift", gauge_shift)])?;
    println!("pure_gauge_riemann_max,{pure_gauge:.3e}");
    println!("gauge_shift_riemann_change,{gauge_shift:.3e}");

    // Part D: O(h^2) cross-check against the nonlinear curvature.
    println!("#");
    println!("# Part D: g = eta + eps h -- nonlinear Ricci scalar vs eps R^(1)(h)");
    println!("epsilon,nonlinear,linear,residual,residual_over_eps_squared");
    let amplitude = 1.0e-3;
    let oracle_point = [0.0, 1.0, -2.0, 0.5];
    let linear_unit = LinearizedField::compute(newtonian(amplitude), &oracle_point, STEP)
        .map_err(|e| e.to_string())?
        .ricci_scalar;
    for &epsilon in &[0.1, 0.05, 0.025, 0.0125]
    {
        let background = ScaledNewtonian { epsilon, amplitude };
        let nonlinear = CurvatureTensors::compute(&background, &oracle_point, STEP)
            .map_err(|e| e.to_string())?
            .ricci_scalar();
        let linear = epsilon * linear_unit;
        let residual = (nonlinear - linear).abs();
        require_finite(&[("nonlinear", nonlinear), ("residual", residual)])?;
        println!(
            "{epsilon:.4},{nonlinear:.6e},{linear:.6e},{residual:.3e},{:.4e}",
            residual / (epsilon * epsilon)
        );
    }

    println!("# interpretation: G^(1)_00 reproduces the Newtonian Poisson source exactly, weak");
    println!("# Schwarzschild is linearized-vacuum to the finite-difference truncation floor (the");
    println!("# 1/r field is not polynomial), the linearized Riemann is gauge invariant, and the");
    println!("# linearized Ricci scalar matches the nonlinear one to O(eps^2) (residual/eps^2 ->");
    println!("# const). Established GR, not a model.");
    Ok(())
}

/// `g = eta + eps * h_newtonian`, for the nonlinear cross-check.
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
