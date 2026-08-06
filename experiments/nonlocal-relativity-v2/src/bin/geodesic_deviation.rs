//! Geodesic deviation (tidal focusing) via the Jacobi field.
//!
//! Integrates a Jacobi field along a geodesic and reports the invariant proper
//! magnitude of the deviation vector, `sqrt(g_(mu nu) xi^mu xi^nu)`, versus
//! affine parameter. Flat spacetime gives exactly linear growth (a quantitative
//! anchor); positive curvature (de Sitter) defocuses geodesics so the magnitude
//! grows faster, while negative curvature (anti-de Sitter) focuses them so the
//! magnitude is turned back. Schwarzschild shows the radial-vs-transverse tidal
//! asymmetry.
//!
//! The Jacobi integrator is validated against the actual separation of two
//! nearby geodesics in this crate's tests; here it is exercised as a physics
//! demonstration. Established general relativity only.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    AntiDeSitter, Connection, DeSitter, Metric, Minkowski, Schwarzschild,
    integrate_geodesic_deviation, metric_norm,
};
use std::f64::consts::FRAC_PI_2;

const STEP: f64 = 0.002;
const CURVATURE_STEP: f64 = 1.0e-4;
const CHECKPOINTS: [f64; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];

/// Proper magnitude `sqrt(|g_(mu nu) xi^mu xi^nu|)` of the deviation at the
/// endpoint of a Jacobi integration to affine length `affine_length`.
#[allow(clippy::too_many_arguments)]
fn proper_deviation_magnitude<B: Metric<4> + Connection<4>>(
    background: &B,
    position: [f64; 4],
    velocity: [f64; 4],
    deviation: [f64; 4],
    deviation_velocity: [f64; 4],
    affine_length: f64,
) -> Result<f64, String> {
    let samples = integrate_geodesic_deviation(
        background,
        &position,
        &velocity,
        &deviation,
        &deviation_velocity,
        affine_length,
        STEP,
        CURVATURE_STEP,
    )
    .map_err(|e| e.to_string())?;
    let endpoint = samples.last().ok_or("empty Jacobi trajectory")?;
    let squared = metric_norm(
        &background.components(&endpoint.position),
        &endpoint.deviation,
    );
    Ok(squared.abs().sqrt())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "Geodesic deviation (tidal focusing) via the Jacobi field",
        "scirust-relativity geometry core (established general relativity)",
        "exercises the Jacobi integrator (validated against geodesic flow in tests); established GR.",
    );

    // Part A: flat spacetime -- exactly linear growth (proper magnitude anchor).
    // xi_dot_0 parallel to xi_0 makes the proper magnitude linear: |xi| = 0.1 + 0.05 tau.
    println!(
        "# Part A: flat Minkowski, deviation parallel to its rate: |xi| = 0.1 + 0.05 tau (exact)"
    );
    println!("affine_parameter,proper_magnitude,linear_oracle,abs_error");
    for &tau in &CHECKPOINTS
    {
        let magnitude = proper_deviation_magnitude(
            &Minkowski,
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0],
            tau,
        )?;
        let oracle = 0.1 + 0.05 * tau;
        require_finite(&[("magnitude", magnitude)])?;
        println!(
            "{tau:.1},{magnitude:.9e},{oracle:.9e},{:.3e}",
            (magnitude - oracle).abs()
        );
    }

    // Part B: de Sitter (defocusing) vs anti-de Sitter (focusing), same setup.
    println!("#");
    println!("# Part B: de Sitter (Lambda>0, defocusing) vs anti-de Sitter (Lambda<0, focusing)");
    println!("# same initial spatial deviation; proper magnitude grows faster / is turned back");
    println!("background,affine_parameter,proper_magnitude");
    let de_sitter = DeSitter::try_new(0.2).ok_or_else(|| "invalid de Sitter".to_string())?;
    let anti_de_sitter =
        AntiDeSitter::try_new(0.2).ok_or_else(|| "invalid anti-de Sitter".to_string())?;
    for &tau in &CHECKPOINTS
    {
        let de_sitter_magnitude = proper_deviation_magnitude(
            &de_sitter,
            [0.0, 2.0, FRAC_PI_2, 0.0],
            [1.0, 0.0, 0.0, 0.02],
            [0.0, 0.1, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            tau,
        )?;
        let anti_de_sitter_magnitude = proper_deviation_magnitude(
            &anti_de_sitter,
            [0.0, 2.0, FRAC_PI_2, 0.0],
            [1.0, 0.0, 0.0, 0.02],
            [0.0, 0.1, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            tau,
        )?;
        require_finite(&[
            ("de_sitter_magnitude", de_sitter_magnitude),
            ("anti_de_sitter_magnitude", anti_de_sitter_magnitude),
        ])?;
        println!("de_Sitter,{tau:.1},{de_sitter_magnitude:.9e}");
        println!("anti_de_Sitter,{tau:.1},{anti_de_sitter_magnitude:.9e}");
    }

    // Part C: Schwarzschild radial vs transverse tidal deviation.
    println!("#");
    println!("# Part C: Schwarzschild tidal deviation, radial vs transverse initial deviation");
    println!("orientation,affine_parameter,proper_magnitude");
    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let position = [0.0, 15.0, FRAC_PI_2, 0.0];
    let velocity = [1.05, 0.0, 0.0, 0.015];
    for &(label, deviation) in &[
        ("radial", [0.0, 0.1, 0.0, 0.0]),
        ("transverse", [0.0, 0.0, 0.1, 0.0]),
    ]
    {
        for &tau in &CHECKPOINTS
        {
            let magnitude = proper_deviation_magnitude(
                &schwarzschild,
                position,
                velocity,
                deviation,
                [0.0, 0.0, 0.0, 0.0],
                tau,
            )?;
            require_finite(&[("magnitude", magnitude)])?;
            println!("{label},{tau:.1},{magnitude:.9e}");
        }
    }

    println!("# interpretation: flat-spacetime deviation grows exactly linearly (the oracle); de");
    println!(
        "# Sitter defocuses (proper magnitude grows faster), anti-de Sitter focuses (magnitude"
    );
    println!("# is turned back), and Schwarzschild stretches radial while compressing transverse");
    println!("# deviations -- the tidal signature of the Riemann tensor. Established GR only.");
    Ok(())
}
