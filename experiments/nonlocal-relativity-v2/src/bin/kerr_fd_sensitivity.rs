//! Phase 9 experiment: Kerr finite-difference Christoffel sensitivity.
//!
//! Unlike the analytic backgrounds, `Kerr`'s connection is evaluated by central
//! finite differences (`numerical_christoffel`), which carries a
//! difference-step-dependent truncation error. At spin `a = 0` the Kerr metric
//! reduces to Schwarzschild exactly, so the finite-difference symbols can be
//! compared against Schwarzschild's *exact analytic* Christoffel symbols. This
//! experiment sweeps the difference step and reports the maximum absolute
//! Christoffel-component discrepancy, exposing the classic central-difference
//! trade-off: the truncation error falls as the step shrinks (order `h^2`)
//! until floating-point cancellation begins to dominate at very small steps.
//!
//! This quantifies the numerical error of the finite-difference connection, an
//! honest disclosed limitation of the Kerr background; it is not a physical
//! result.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::print_common_header;
use scirust_relativity::{Connection, Kerr, Schwarzschild, numerical_christoffel};
use std::f64::consts::FRAC_PI_2;

const MASS: f64 = 1.0;
const RADIUS: f64 = 8.0;
const DIFFERENCE_STEPS: [f64; 7] = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 1.0e-5, 1.0e-6];

fn max_component_difference(left: &[[[f64; 4]; 4]; 4], right: &[[[f64; 4]; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for rho in 0..4
    {
        for mu in 0..4
        {
            for nu in 0..4
            {
                let difference = (left[rho][mu][nu] - right[rho][mu][nu]).abs();
                if difference > worst
                {
                    worst = difference;
                }
            }
        }
    }
    worst
}

fn main() -> Result<(), String> {
    // Spin zero: Kerr reduces to Schwarzschild exactly, giving an exact analytic
    // reference for the finite-difference symbols.
    let kerr = Kerr::try_new(MASS, 0.0).ok_or_else(|| "invalid Kerr parameters".to_string())?;
    let schwarzschild =
        Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;
    let coordinates = [0.0, RADIUS, FRAC_PI_2, 0.0];
    let analytic = schwarzschild.christoffel(&coordinates);

    print_common_header("Kerr finite-difference Christoffel sensitivity (spin = 0)");
    println!(
        "# reference: Schwarzschild exact analytic Christoffel symbols at r = {RADIUS}, theta = pi/2"
    );
    println!(
        "# Kerr at spin = 0 reduces to Schwarzschild, so the difference is pure finite-difference error"
    );
    println!("difference_step,max_abs_christoffel_error");

    for &step in &DIFFERENCE_STEPS
    {
        let numerical =
            numerical_christoffel(&kerr, &coordinates, step).map_err(|e| e.to_string())?;
        let error = max_component_difference(&numerical, &analytic);
        if !error.is_finite()
        {
            return Err(format!("non-finite Christoffel error at step {step}"));
        }
        println!("{step:.0e},{error:.6e}");
    }

    println!(
        "# interpretation: the finite-difference Christoffel error against the exact analytic"
    );
    println!(
        "# symbols falls with the difference step (central differences are second order) toward a"
    );
    println!(
        "# roundoff-limited floor. This is the disclosed truncation cost of the Kerr connection;"
    );
    println!("# every other background in this crate uses exact analytic symbols and avoids it.");
    Ok(())
}
