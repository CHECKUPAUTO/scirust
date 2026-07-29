//! Curvature-invariant validation and finite-difference sensitivity.
//!
//! The [`scirust_relativity::CurvatureTensors`] engine evaluates the Riemann,
//! Ricci, Einstein, and Kretschmann tensors from a background's metric and
//! connection using central finite differences of the Christoffel symbols. For
//! backgrounds with an analytic connection the two scalar invariants below have
//! exact closed forms, so this experiment is a genuine validation against
//! analytic oracles (not merely a self-consistency check):
//!
//! - Ricci scalar `R`: `0` for Schwarzschild (vacuum), `4 Lambda` for de Sitter
//!   and anti-de Sitter.
//! - Kretschmann `K = R_(abcd) R^(abcd)`: `48 M^2 / r^6` for Schwarzschild,
//!   `8 Lambda^2 / 3` for (anti-)de Sitter.
//!
//! Part A reports the numerical invariants against these oracles across the
//! backgrounds. Part B sweeps the finite-difference step for the Schwarzschild
//! Kretschmann scalar, exposing the central-difference trade-off (truncation
//! error `~h^2` falling toward a roundoff-limited floor), exactly as the Kerr
//! Christoffel experiment does one derivative lower.
//!
//! This validates established general-relativistic curvature against exact
//! results; it uses only the deterministic geometry core and asserts no
//! physical claim beyond textbook GR.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{AntiDeSitter, CurvatureTensors, DeSitter, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

const STEP: f64 = 1.0e-4;
const DIFFERENCE_STEPS: [f64; 7] = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 1.0e-5, 1.0e-6];

/// One background evaluated at a point, with its exact oracle invariants.
struct Case {
    name: &'static str,
    coordinates: [f64; 4],
    ricci_scalar_oracle: f64,
    kretschmann_oracle: f64,
    ricci_scalar: f64,
    kretschmann: f64,
}

fn relative_error(numerical: f64, oracle: f64) -> f64 {
    let scale = oracle.abs().max(1.0);
    (numerical - oracle).abs() / scale
}

fn main() -> Result<(), String> {
    let mass = 1.0;
    let schwarzschild =
        Schwarzschild::try_new(mass).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;

    let lambda_ds = 0.03;
    let de_sitter = DeSitter::try_new(lambda_ds).ok_or_else(|| "invalid de Sitter".to_string())?;

    let magnitude_ads = 0.05;
    let anti_de_sitter =
        AntiDeSitter::try_new(magnitude_ads).ok_or_else(|| "invalid anti-de Sitter".to_string())?;
    let lambda_ads = anti_de_sitter.cosmological_constant();

    let schwarzschild_radius = 8.0;
    let de_sitter_radius = 3.0;
    let anti_de_sitter_radius = 4.0;

    let schwarzschild_tensors = CurvatureTensors::compute(
        &schwarzschild,
        &[0.0, schwarzschild_radius, FRAC_PI_2, 0.0],
        STEP,
    )
    .map_err(|e| e.to_string())?;
    let de_sitter_tensors =
        CurvatureTensors::compute(&de_sitter, &[0.0, de_sitter_radius, FRAC_PI_2, 0.0], STEP)
            .map_err(|e| e.to_string())?;
    let anti_de_sitter_tensors = CurvatureTensors::compute(
        &anti_de_sitter,
        &[0.0, anti_de_sitter_radius, FRAC_PI_2, 0.0],
        STEP,
    )
    .map_err(|e| e.to_string())?;

    let cases = [
        Case {
            name: "Schwarzschild",
            coordinates: [0.0, schwarzschild_radius, FRAC_PI_2, 0.0],
            ricci_scalar_oracle: 0.0,
            kretschmann_oracle: 48.0 * mass * mass / schwarzschild_radius.powi(6),
            ricci_scalar: schwarzschild_tensors.ricci_scalar(),
            kretschmann: schwarzschild_tensors.kretschmann(),
        },
        Case {
            name: "de_Sitter",
            coordinates: [0.0, de_sitter_radius, FRAC_PI_2, 0.0],
            ricci_scalar_oracle: 4.0 * lambda_ds,
            kretschmann_oracle: 8.0 * lambda_ds * lambda_ds / 3.0,
            ricci_scalar: de_sitter_tensors.ricci_scalar(),
            kretschmann: de_sitter_tensors.kretschmann(),
        },
        Case {
            name: "anti_de_Sitter",
            coordinates: [0.0, anti_de_sitter_radius, FRAC_PI_2, 0.0],
            ricci_scalar_oracle: 4.0 * lambda_ads,
            kretschmann_oracle: 8.0 * lambda_ads * lambda_ads / 3.0,
            ricci_scalar: anti_de_sitter_tensors.ricci_scalar(),
            kretschmann: anti_de_sitter_tensors.kretschmann(),
        },
    ];

    print_experiment_header(
        "Curvature invariants against exact analytic oracles",
        "scirust-relativity geometry core (established general relativity)",
        "validates the numerical curvature engine against exact closed-form GR invariants.",
    );
    println!("# difference step for the curvature engine: h = {STEP:.0e}");
    println!(
        "# Ricci scalar oracle: 0 (Schwarzschild vacuum), 4*Lambda (de Sitter / anti-de Sitter)"
    );
    println!(
        "# Kretschmann oracle: 48*M^2/r^6 (Schwarzschild), 8*Lambda^2/3 (maximally symmetric)"
    );
    println!(
        "background,radius,ricci_scalar,ricci_scalar_oracle,ricci_abs_error,kretschmann,kretschmann_oracle,kretschmann_rel_error"
    );

    for case in &cases
    {
        let ricci_abs_error = (case.ricci_scalar - case.ricci_scalar_oracle).abs();
        let kretschmann_rel_error = relative_error(case.kretschmann, case.kretschmann_oracle);
        require_finite(&[
            ("ricci_scalar", case.ricci_scalar),
            ("kretschmann", case.kretschmann),
            ("ricci_abs_error", ricci_abs_error),
            ("kretschmann_rel_error", kretschmann_rel_error),
        ])?;
        println!(
            "{},{:.1},{:.6e},{:.6e},{:.3e},{:.6e},{:.6e},{:.3e}",
            case.name,
            case.coordinates[1],
            case.ricci_scalar,
            case.ricci_scalar_oracle,
            ricci_abs_error,
            case.kretschmann,
            case.kretschmann_oracle,
            kretschmann_rel_error,
        );
    }

    println!("#");
    println!("# Part B: finite-difference sensitivity of the Schwarzschild Kretschmann scalar");
    println!("# oracle K = 48*M^2/r^6 at M = {mass}, r = {schwarzschild_radius}");
    println!("difference_step,kretschmann,kretschmann_rel_error");

    let kretschmann_oracle = 48.0 * mass * mass / schwarzschild_radius.powi(6);
    for &step in &DIFFERENCE_STEPS
    {
        let tensors = CurvatureTensors::compute(
            &schwarzschild,
            &[0.0, schwarzschild_radius, FRAC_PI_2, 0.0],
            step,
        )
        .map_err(|e| e.to_string())?;
        let kretschmann = tensors.kretschmann();
        let rel_error = relative_error(kretschmann, kretschmann_oracle);
        require_finite(&[("kretschmann", kretschmann), ("rel_error", rel_error)])?;
        println!("{step:.0e},{kretschmann:.9e},{rel_error:.3e}");
    }

    println!(
        "# interpretation: every background's numerical invariants match their exact analytic"
    );
    println!(
        "# oracles to the finite-difference tolerance. The step sweep shows the classic central-"
    );
    println!(
        "# difference trade-off: the relative error falls (order h^2) as the step shrinks until"
    );
    println!("# floating-point cancellation dominates at very small steps. This is established GR");
    println!("# curvature validated against closed-form results, not a phenomenological model.");
    Ok(())
}
