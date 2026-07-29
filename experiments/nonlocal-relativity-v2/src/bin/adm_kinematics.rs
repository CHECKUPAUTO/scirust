//! ADM 3+1 kinematics checks: lapse, shift, spatial curvature, extrinsic
//! curvature, and the Hamiltonian and momentum constraints across four exact
//! foliations (the fourth Layer 2 slice; see `docs/LAYER_2_ADM.md`).
//!
//! For each background and point the experiment decomposes the 4-metric on the
//! constant-time slice and reports the lapse, radial shift, spatial Ricci scalar,
//! mean curvature, `K_ij K^ij`, and the Gauss-Codazzi constraint residuals, which
//! vanish for these vacuum-with-Lambda solutions. Painlevé–Gullstrand is the
//! non-zero-shift, spatially-varying-K oracle.
//!
//! Established general relativity. The ADM variables are numerical
//! approximations (a metric-only nested difference for R^(3), a time difference
//! for the extrinsic curvature, a spatial difference for the momentum
//! constraint); this evolves nothing and asserts no new physics.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::adm::{AdmSettings, adm_constraints, adm_decomposition};
use scirust_relativity::{
    DeSitter, ExponentialScaleFactor, Flrw, Metric, PainleveGullstrand, Schwarzschild,
};
use std::f64::consts::FRAC_PI_2;

fn emit_row<B: Metric<4>>(
    name: &str,
    background: &B,
    coordinates: &[f64; 4],
    cosmological_constant: f64,
) -> Result<(), String> {
    let settings = AdmSettings {
        time_step: 1.0e-3,
        spatial_step: 1.0e-3,
    };
    let decomposition =
        adm_decomposition(background, coordinates, &settings).map_err(|e| e.to_string())?;
    let constraints = adm_constraints(background, coordinates, cosmological_constant, &settings)
        .map_err(|e| e.to_string())?;
    let momentum_max = constraints
        .momentum
        .iter()
        .fold(0.0_f64, |acc, &m| acc.max(m.abs()));
    require_finite(&[
        ("lapse", decomposition.lapse),
        ("spatial_ricci_scalar", decomposition.spatial_ricci_scalar),
        ("mean_curvature", decomposition.mean_curvature),
        (
            "extrinsic_curvature_norm",
            decomposition.extrinsic_curvature_norm,
        ),
        ("hamiltonian", constraints.hamiltonian),
        ("momentum_max", momentum_max),
    ])?;
    println!(
        "{name},{:.3},{cosmological_constant},{:.6},{:+.6},{:+.4e},{:+.6},{:.6},{:.3e},{:.3e},ok",
        coordinates[1],
        decomposition.lapse,
        decomposition.shift[0],
        decomposition.spatial_ricci_scalar,
        decomposition.mean_curvature,
        decomposition.extrinsic_curvature_norm,
        constraints.hamiltonian,
        momentum_max,
    );
    Ok(())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "ADM 3+1 kinematics and the Gauss-Codazzi constraints",
        "scirust-relativity Layer 2 (established general relativity)",
        "lapse/shift/spatial-metric/extrinsic-curvature and the Hamiltonian/momentum constraints; numerical, not a bound.",
    );

    println!(
        "background,radius,cosmological_constant,lapse,shift_radial,spatial_ricci_scalar,\
         mean_curvature,extrinsic_curvature_norm,hamiltonian_residual,momentum_max,status"
    );

    let schwarzschild = Schwarzschild::try_new(1.0).ok_or("invalid Schwarzschild")?;
    for &radius in &[4.0, 6.0, 10.0]
    {
        emit_row(
            "schwarzschild",
            &schwarzschild,
            &[0.0, radius, FRAC_PI_2, 0.0],
            0.0,
        )?;
    }

    let lambda = 0.03;
    let de_sitter = DeSitter::try_new(lambda).ok_or("invalid de Sitter")?;
    for &radius in &[2.0, 3.0, 4.0]
    {
        emit_row(
            "de_sitter",
            &de_sitter,
            &[0.0, radius, FRAC_PI_2, 0.0],
            lambda,
        )?;
    }

    let hubble = 0.5;
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).ok_or("invalid H")?);
    let flrw_lambda = 3.0 * hubble * hubble;
    // Exponential FLRW is de Sitter: K = -3H is constant in cosmic time.
    emit_row("flrw", &flrw, &[0.0, 0.1, 0.2, 0.3], flrw_lambda)?;

    let painleve = PainleveGullstrand::try_new(1.0).ok_or("invalid Painleve-Gullstrand")?;
    for &radius in &[3.0, 4.0, 6.0]
    {
        emit_row(
            "painleve_gullstrand",
            &painleve,
            &[0.0, radius, FRAC_PI_2, 0.0],
            0.0,
        )?;
    }

    println!("# interpretation: the lapse/shift/spatial-metric reconstruct each 4-metric; the");
    println!(
        "# spatial Ricci scalar is 0 (Schwarzschild, flat FLRW and Painleve-Gullstrand slices)"
    );
    println!("# or 2*Lambda (static de Sitter); the extrinsic curvature is zero for the static");
    println!(
        "# slicings and K = -3H for FLRW; and the Hamiltonian and momentum constraints vanish"
    );
    println!(
        "# for every one of these exact vacuum-with-Lambda foliations. Numerical, not a bound."
    );
    Ok(())
}
