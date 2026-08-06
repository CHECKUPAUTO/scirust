//! Exponential/logarithm map round-trip accuracy.
//!
//! The geodesic exponential and logarithm maps are local inverses:
//! `log_p(exp_p(v)) = v`. This experiment reports the Euclidean round-trip error
//! `|log_p(exp_p(v)) - v|` versus the tangent magnitude across backgrounds. In
//! flat spacetime the maps are exact translations, so the error is at the
//! roundoff floor for every magnitude; in curved spacetimes the Newton
//! logarithm inverts the exponential to the requested tolerance while the
//! geodesics stay within the convex neighbourhood, so the round-trip error
//! stays at or below that tolerance across displacements.
//!
//! Established general relativity only; the maps reuse the deterministic
//! geodesic integrator.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    AntiDeSitter, Connection, DeSitter, Minkowski, Schwarzschild, geodesic_exponential,
    geodesic_logarithm,
};
use std::f64::consts::FRAC_PI_2;

const STEP: f64 = 0.01;
const JACOBIAN_STEP: f64 = 1.0e-5;
const TOLERANCE: f64 = 1.0e-11;
const MAX_ITERATIONS: usize = 60;
const SCALES: [f64; 4] = [0.25, 0.5, 1.0, 2.0];

fn round_trip_error<B: Connection<4> + Copy>(
    background: &B,
    position: [f64; 4],
    tangent: [f64; 4],
) -> Result<f64, String> {
    let image =
        geodesic_exponential(background, &position, &tangent, STEP).map_err(|e| e.to_string())?;
    let recovered = geodesic_logarithm(
        background,
        &position,
        &image,
        STEP,
        JACOBIAN_STEP,
        TOLERANCE,
        MAX_ITERATIONS,
    )
    .map_err(|e| e.to_string())?;
    let mut error_squared = 0.0;
    for i in 0..4
    {
        error_squared += (recovered[i] - tangent[i]).powi(2);
    }
    Ok(error_squared.sqrt())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "Exponential/logarithm map round-trip accuracy",
        "scirust-relativity geometry core (established general relativity)",
        "checks exp and log are local inverses (log(exp(v)) = v); established GR.",
    );
    println!(
        "# round-trip error |log_p(exp_p(v)) - v| vs tangent scale; flat is exact to roundoff"
    );
    println!("background,tangent_scale,tangent_norm,round_trip_error");

    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let de_sitter = DeSitter::try_new(0.05).ok_or_else(|| "invalid de Sitter".to_string())?;
    let anti_de_sitter =
        AntiDeSitter::try_new(0.05).ok_or_else(|| "invalid anti-de Sitter".to_string())?;

    let base_tangent = [0.2, 0.15, 0.03, 0.02];
    let cases: [(&str, [f64; 4]); 4] = [
        ("Minkowski", [0.0, 2.0, 1.0, 0.5]),
        ("Schwarzschild", [0.0, 12.0, FRAC_PI_2, 0.0]),
        ("de_Sitter", [0.0, 3.0, FRAC_PI_2, 0.0]),
        ("anti_de_Sitter", [0.0, 3.0, FRAC_PI_2, 0.0]),
    ];

    for &scale in &SCALES
    {
        let tangent = [
            base_tangent[0] * scale,
            base_tangent[1] * scale,
            base_tangent[2] * scale,
            base_tangent[3] * scale,
        ];
        let tangent_norm = tangent.iter().map(|v| v * v).sum::<f64>().sqrt();

        for &(label, position) in &cases
        {
            let error = match label
            {
                "Minkowski" => round_trip_error(&Minkowski, position, tangent)?,
                "Schwarzschild" => round_trip_error(&schwarzschild, position, tangent)?,
                "de_Sitter" => round_trip_error(&de_sitter, position, tangent)?,
                _ => round_trip_error(&anti_de_sitter, position, tangent)?,
            };
            require_finite(&[("round_trip_error", error)])?;
            println!("{label},{scale:.2},{tangent_norm:.4e},{error:.3e}");
        }
    }

    println!("# interpretation: flat spacetime round-trips exactly (error at the roundoff floor);");
    println!("# in curved spacetimes the Newton logarithm inverts the exponential map to the");
    println!("# requested tolerance, so the round-trip error stays at or below that tolerance");
    println!("# across displacements. The maps are local inverses of established GR, not a model.");
    Ok(())
}
