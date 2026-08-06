//! FLRW cosmological curvature against the exact Friedmann oracles.
//!
//! Validates the [`scirust_relativity`] spatially flat FLRW background:
//!
//! - **Exponential** scale factor `a = exp(H t)` is de Sitter space, so its
//!   Ricci scalar `R = 12 H^2` and Kretschmann `K = 24 H^4` are constant in
//!   cosmic time and its Kretschmann agrees with the static de Sitter chart at
//!   `Lambda = 3 H^2` (a coordinate-independence cross-check).
//! - **Power-law** scale factor `a = (t/t_ref)^p` is a genuinely time-dependent
//!   geometry: its `R` and `K` match the general flat-FLRW formulas
//!   `R = 6 (a_ddot/a + (a_dot/a)^2)`, `K = 12 ((a_ddot/a)^2 + (a_dot/a)^4)`.
//!   Radiation (`p = 1/2`) is traceless (`R = 0`) while `K > 0`.
//!
//! Established general relativity validated against exact closed forms; no
//! phenomenological or speculative model appears.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    CurvatureTensors, DeSitter, ExponentialScaleFactor, Flrw, PowerLawScaleFactor, ScaleFactor,
};
use std::f64::consts::FRAC_PI_2;

const STEP: f64 = 1.0e-4;

fn absolute_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs()
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "FLRW cosmological curvature against the Friedmann oracles",
        "scirust-relativity geometry core (established general relativity)",
        "validates the FLRW curvature engine against exact Friedmann formulas; established GR.",
    );

    // Part A: exponential (de Sitter) FLRW -- constant curvature.
    let hubble = 0.5;
    let lambda = 3.0 * hubble * hubble;
    let de_sitter_flrw = Flrw::new(
        ExponentialScaleFactor::try_new(hubble).ok_or_else(|| "invalid Hubble".to_string())?,
    );
    println!(
        "# Part A: exponential FLRW a = exp(H t), H = {hubble} (de Sitter, Lambda = {lambda})"
    );
    println!(
        "# oracle: R = 12 H^2 = {:.6}, K = 24 H^4 = {:.6}",
        12.0 * hubble * hubble,
        24.0 * hubble.powi(4)
    );
    println!("cosmic_time,ricci_scalar,ricci_abs_error,kretschmann,kretschmann_abs_error");
    let ricci_oracle_a = 12.0 * hubble * hubble;
    let kretschmann_oracle_a = 24.0 * hubble.powi(4);
    for &time in &[-1.0, 0.0, 1.0, 2.0]
    {
        let tensors = CurvatureTensors::compute(&de_sitter_flrw, &[time, 0.1, -0.2, 0.3], STEP)
            .map_err(|e| e.to_string())?;
        let ricci_error = absolute_error(tensors.ricci_scalar(), ricci_oracle_a);
        let kretschmann_error = absolute_error(tensors.kretschmann(), kretschmann_oracle_a);
        require_finite(&[
            ("ricci_scalar", tensors.ricci_scalar()),
            ("kretschmann", tensors.kretschmann()),
        ])?;
        println!(
            "{time:.1},{:.6e},{ricci_error:.3e},{:.6e},{kretschmann_error:.3e}",
            tensors.ricci_scalar(),
            tensors.kretschmann()
        );
    }

    // Part B: coordinate independence against the static de Sitter chart.
    println!("#");
    println!("# Part B: de Sitter Kretschmann in FLRW slicing vs the static chart (must agree)");
    println!("chart,kretschmann,oracle_8_lambda_sq_over_3,abs_gap");
    let static_de_sitter =
        DeSitter::try_new(lambda).ok_or_else(|| "invalid de Sitter".to_string())?;
    let flrw_k = CurvatureTensors::compute(&de_sitter_flrw, &[0.7, 0.0, 0.0, 0.0], STEP)
        .map_err(|e| e.to_string())?
        .kretschmann();
    let static_k = CurvatureTensors::compute(&static_de_sitter, &[0.0, 3.0, FRAC_PI_2, 0.0], STEP)
        .map_err(|e| e.to_string())?
        .kretschmann();
    let oracle_b = 8.0 * lambda * lambda / 3.0;
    require_finite(&[("flrw_k", flrw_k), ("static_k", static_k)])?;
    println!(
        "FLRW_flat_slicing,{flrw_k:.9e},{oracle_b:.9e},{:.3e}",
        absolute_error(flrw_k, oracle_b)
    );
    println!(
        "static,{static_k:.9e},{oracle_b:.9e},{:.3e}",
        absolute_error(static_k, oracle_b)
    );
    println!(
        "# cross-chart gap: {:.3e}",
        absolute_error(flrw_k, static_k)
    );

    // Part C: power-law FLRW -- time-dependent curvature.
    println!("#");
    println!("# Part C: power-law FLRW a = (t/t_ref)^p, t_ref = 1 (time-dependent curvature)");
    println!(
        "# oracle: R = 6(a''/a + (a'/a)^2), K = 12((a''/a)^2 + (a'/a)^4); radiation p=1/2 has R=0"
    );
    println!(
        "exponent,cosmic_time,ricci_scalar,ricci_oracle,ricci_abs_error,kretschmann,kretschmann_oracle,kretschmann_rel_error"
    );
    for &(label, exponent) in &[("radiation_1_2", 0.5), ("matter_2_3", 2.0 / 3.0)]
    {
        let scale_factor = PowerLawScaleFactor::try_new(exponent, 1.0)
            .ok_or_else(|| "invalid power".to_string())?;
        let background = Flrw::new(scale_factor);
        for &time in &[1.5, 2.0, 4.0]
        {
            let tensors = CurvatureTensors::compute(&background, &[time, 0.0, 0.0, 0.0], STEP)
                .map_err(|e| e.to_string())?;
            let scale = scale_factor.value(time);
            let acceleration_ratio = scale_factor.second_derivative(time) / scale;
            let hubble_ratio = scale_factor.first_derivative(time) / scale;
            let ricci_oracle = 6.0 * (acceleration_ratio + hubble_ratio * hubble_ratio);
            let kretschmann_oracle =
                12.0 * (acceleration_ratio * acceleration_ratio + hubble_ratio.powi(4));
            let kretschmann_rel = absolute_error(tensors.kretschmann(), kretschmann_oracle)
                / kretschmann_oracle.abs().max(1.0e-300);
            require_finite(&[
                ("ricci_scalar", tensors.ricci_scalar()),
                ("kretschmann", tensors.kretschmann()),
                ("kretschmann_rel", kretschmann_rel),
            ])?;
            println!(
                "{label},{time:.1},{:.6e},{ricci_oracle:.6e},{:.3e},{:.6e},{kretschmann_oracle:.6e},{kretschmann_rel:.3e}",
                tensors.ricci_scalar(),
                absolute_error(tensors.ricci_scalar(), ricci_oracle),
                tensors.kretschmann(),
            );
        }
    }

    println!("# interpretation: exponential FLRW reproduces de Sitter with constant R and K that");
    println!("# match both the Friedmann oracle and the static-chart Kretschmann (coordinate");
    println!("# independence); power-law FLRW matches the time-dependent Friedmann formulas, with");
    println!("# the radiation era's traceless R = 0 recovered numerically. Established GR only.");
    Ok(())
}
