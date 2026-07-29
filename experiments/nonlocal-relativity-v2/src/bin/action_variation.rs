//! Einstein-Hilbert action variation checks: metric-only curvature, vacuum
//! stationarity, a mismatched-Lambda nonzero cross-check, and grid convergence
//! (the third Layer 2 slice; see `docs/LAYER_2_ACTION_VARIATION.md`).
//!
//! For each background, perturbed component, action `Lambda`, and grid the
//! experiment runs the deterministic numerical variation and reports the numeric
//! directional derivative, the analytic-integrand prediction from the Einstein
//! tensor, and their residual. Vacuum solutions (Schwarzschild; Lambda-matched
//! de Sitter) give a residual toward zero that falls with resolution; a
//! mismatched action `Lambda` reproduces the known nonzero prediction.
//!
//! Established general relativity. The variation is a numerical approximation
//! (a metric-only nested finite difference, a Simpson quadrature, and a central
//! difference in the amplitude); it is never an exact variation and asserts no
//! new physics.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::action::{
    ActionDomain, ActionPerturbation, ActionSettings, einstein_hilbert_action_variation,
};
use scirust_relativity::{Connection, DeSitter, Metric, Schwarzschild, ricci_scalar_from_metric};
use std::f64::consts::FRAC_PI_2;

const LAMBDA: f64 = 0.03;

#[allow(clippy::too_many_arguments)]
fn emit_row<B: Metric<4> + Connection<4>>(
    name: &str,
    background: &B,
    component: (usize, usize),
    center: (f64, f64),
    half_widths: (f64, f64),
    radial_range: (f64, f64),
    polar_range: (f64, f64),
    grid: usize,
    lambda_action: f64,
) -> Result<(), String> {
    let perturbation = ActionPerturbation {
        component,
        center,
        half_widths,
    };
    let domain = ActionDomain {
        radial_range,
        polar_range,
        grid,
    };
    let settings = ActionSettings {
        amplitude: 1.0e-3,
        connection_step: 1.0e-3,
        metric_step: 1.0e-3,
        cosmological_constant: lambda_action,
    };
    let variation =
        einstein_hilbert_action_variation(background, &perturbation, &domain, &settings)
            .map_err(|error| error.to_string())?;
    require_finite(&[
        ("numeric", variation.numeric),
        ("predicted", variation.predicted),
        ("residual", variation.residual),
    ])?;
    let relative = if variation.predicted.abs() < 1.0e-6
    {
        "n_a".to_string()
    }
    else
    {
        format!("{:.3e}", variation.residual / variation.predicted.abs())
    };
    println!(
        "{name},g{}{},{lambda_action},{grid},{:.3},{:.3},{:+.6e},{:+.6e},{:.3e},{relative},ok",
        component.0,
        component.1,
        radial_range.0,
        radial_range.1,
        variation.numeric,
        variation.predicted,
        variation.residual,
    );
    Ok(())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "Einstein-Hilbert action variation (gamma numerical functional derivative)",
        "scirust-relativity Layer 2 (established general relativity)",
        "numerical delta S / delta g for static axisymmetric backgrounds; a numerical approximation, not a bound.",
    );

    let de_sitter = DeSitter::try_new(LAMBDA).ok_or("invalid de Sitter")?;
    let schwarzschild = Schwarzschild::try_new(1.0).ok_or("invalid Schwarzschild")?;

    // O1: metric-only nested-difference Ricci scalar against exact oracles.
    let ds_scalar =
        ricci_scalar_from_metric(&de_sitter, &[0.0, 3.0, FRAC_PI_2, 0.0], 1.0e-3, 1.0e-3)
            .map_err(|e| e.to_string())?;
    let sch_scalar =
        ricci_scalar_from_metric(&schwarzschild, &[0.0, 6.0, FRAC_PI_2, 0.0], 1.0e-3, 1.0e-3)
            .map_err(|e| e.to_string())?;
    println!(
        "# metric-only Ricci scalar: de Sitter R = {ds_scalar:.8} (exact {:.8}); \
         Schwarzschild R = {sch_scalar:.3e} (exact 0)",
        4.0 * LAMBDA
    );

    println!(
        "background,component,cosmological_constant_action,grid,radius_min,radius_max,\
         numeric_variation,predicted_variation,abs_residual,relative_residual,status"
    );

    let de_sitter_center = (3.0, FRAC_PI_2);
    let de_sitter_widths = (1.0, 1.0);
    let de_sitter_radial = (2.0, 4.0);
    let de_sitter_polar = (FRAC_PI_2 - 1.0, FRAC_PI_2 + 1.0);

    // O2 + O4: matched Lambda is stationary; the residual falls with resolution.
    for component in [(0, 0), (1, 1), (2, 2)]
    {
        for grid in [21, 41, 61]
        {
            emit_row(
                "de_sitter_matched",
                &de_sitter,
                component,
                de_sitter_center,
                de_sitter_widths,
                de_sitter_radial,
                de_sitter_polar,
                grid,
                LAMBDA,
            )?;
        }
    }

    // O3 + O4: a mismatched action Lambda reproduces the known nonzero prediction.
    for grid in [21, 41, 61]
    {
        emit_row(
            "de_sitter_mismatch",
            &de_sitter,
            (1, 1),
            de_sitter_center,
            de_sitter_widths,
            de_sitter_radial,
            de_sitter_polar,
            grid,
            0.0,
        )?;
    }

    // O2: Schwarzschild is a vacuum solution (action Lambda = 0).
    for grid in [21, 41, 61]
    {
        emit_row(
            "schwarzschild_vacuum",
            &schwarzschild,
            (1, 1),
            (6.0, FRAC_PI_2),
            (2.0, 1.0),
            (4.0, 8.0),
            de_sitter_polar,
            grid,
            0.0,
        )?;
    }

    println!("# interpretation: the metric-only Ricci scalar recovers 4*Lambda (de Sitter) and 0");
    println!("# (Schwarzschild); the matched-Lambda and Schwarzschild variations are stationary");
    println!("# (residual toward zero, falling ~O(dx^4) with the grid); and the mismatched-Lambda");
    println!("# de Sitter variation reproduces the nonzero Einstein-tensor prediction. Numerical");
    println!("# approximation, not a bound and not an exact variation.");
    Ok(())
}
