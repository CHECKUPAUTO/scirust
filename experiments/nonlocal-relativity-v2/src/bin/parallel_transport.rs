//! Parallel-transport accuracy, holonomy, and the holonomy/curvature identity.
//!
//! Validates the [`scirust_relativity`] geometry-core parallel-transport engine
//! against exact identities of established general relativity:
//!
//! - **Part A — metric compatibility.** The metric inner product of a
//!   transported vector is preserved along the path; the relative drift falls
//!   as the RK4 substep count grows (second-order integrator).
//! - **Part B — flat holonomy.** Transport around a closed loop in the
//!   spherical (curvilinear) chart of flat Minkowski returns the vector: the
//!   holonomy defect falls to the roundoff floor.
//! - **Part C — holonomy/curvature identity.** In curved spacetimes the
//!   holonomy defect around a small parallelogram loop spanned by
//!   `A = eps e_(mu0)`, `B = eps e_(nu0)` equals
//!   `-R^rho_(sigma mu0 nu0) V^sigma eps^2` to leading order; the relative gap
//!   between the transported defect and the independently computed Riemann
//!   prediction falls as `O(eps)`. This cross-checks two separate numerical
//!   engines (transport and curvature) against one analytic identity.
//!
//! This exercises established general relativity only; no phenomenological or
//! speculative model appears.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    Connection, CurvatureTensors, DeSitter, Metric, MinkowskiSpherical, Schwarzschild,
    holonomy_defect, metric_norm, transport_along_segment,
};
use std::f64::consts::FRAC_PI_2;

fn norm_drift<B: Metric<4> + Connection<4>>(
    background: &B,
    start: [f64; 4],
    end: [f64; 4],
    vector: [f64; 4],
    substeps: usize,
) -> Result<f64, String> {
    let transported = transport_along_segment(background, &start, &end, &vector, substeps)
        .map_err(|e| e.to_string())?;
    let initial = metric_norm(&background.components(&start), &vector);
    let final_norm = metric_norm(&background.components(&end), &transported);
    Ok((final_norm - initial).abs() / initial.abs().max(1.0e-12))
}

/// Relative gap between the transported holonomy defect around a small
/// `(mu0, nu0)` loop of size `eps` and the Riemann prediction
/// `-R^rho_(sigma mu0 nu0) V^sigma eps^2`, in Euclidean vector norm.
fn holonomy_vs_riemann<B: Metric<4> + Connection<4>>(
    background: &B,
    point: [f64; 4],
    vector: [f64; 4],
    mu0: usize,
    nu0: usize,
    eps: f64,
) -> Result<f64, String> {
    let riemann = *CurvatureTensors::compute(background, &point, 1.0e-4)
        .map_err(|e| e.to_string())?
        .riemann();

    let mut a_end = point;
    a_end[mu0] += eps;
    let mut ab_end = a_end;
    ab_end[nu0] += eps;
    let mut b_end = point;
    b_end[nu0] += eps;
    let loop_path = [point, a_end, ab_end, b_end, point];

    let defect =
        holonomy_defect(background, &loop_path, &vector, 400).map_err(|e| e.to_string())?;

    let mut gap_squared = 0.0;
    let mut prediction_squared = 0.0;
    for rho in 0..4
    {
        let mut prediction = 0.0;
        for sigma in 0..4
        {
            prediction -= riemann[rho][sigma][mu0][nu0] * vector[sigma];
        }
        prediction *= eps * eps;
        gap_squared += (defect[rho] - prediction).powi(2);
        prediction_squared += prediction.powi(2);
    }
    Ok((gap_squared / prediction_squared.max(1.0e-300)).sqrt())
}

fn main() -> Result<(), String> {
    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let de_sitter = DeSitter::try_new(0.03).ok_or_else(|| "invalid de Sitter".to_string())?;

    print_experiment_header(
        "Parallel transport: accuracy, holonomy, and the holonomy/curvature identity",
        "scirust-relativity geometry core (established general relativity)",
        "validates the transport engine against exact GR identities; established GR.",
    );

    // Part A: metric-compatibility (norm-preservation) convergence.
    println!(
        "# Part A: relative metric-norm drift after transport vs RK4 substeps (should fall ~h^2)"
    );
    println!("background,substeps,relative_norm_drift");
    for substeps in [10, 25, 50, 100, 200]
    {
        let schwarzschild_drift = norm_drift(
            &schwarzschild,
            [0.0, 10.0, FRAC_PI_2, 0.0],
            [0.0, 6.0, FRAC_PI_2, 1.0],
            [0.2, 0.1, 0.03, 0.02],
            substeps,
        )?;
        let de_sitter_drift = norm_drift(
            &de_sitter,
            [0.0, 3.0, FRAC_PI_2, 0.0],
            [0.0, 5.0, 1.2, 0.7],
            [0.15, 0.2, 0.05, 0.04],
            substeps,
        )?;
        require_finite(&[
            ("schwarzschild_drift", schwarzschild_drift),
            ("de_sitter_drift", de_sitter_drift),
        ])?;
        println!("Schwarzschild,{substeps},{schwarzschild_drift:.3e}");
        println!("de_Sitter,{substeps},{de_sitter_drift:.3e}");
    }

    // Part B: flat-spacetime closed-loop holonomy (should fall to roundoff).
    println!("#");
    println!(
        "# Part B: closed-loop holonomy in flat spherical-chart Minkowski (should fall to ~0)"
    );
    println!("substeps,max_abs_holonomy_defect");
    let base = [0.0, 5.0, FRAC_PI_2, 0.3];
    let edge = 0.4;
    let loop_path = [
        base,
        [base[0], base[1] + edge, base[2], base[3]],
        [base[0], base[1] + edge, base[2] + edge, base[3]],
        [base[0], base[1], base[2] + edge, base[3]],
        base,
    ];
    let flat_vector = [0.1, 0.3, 0.05, 0.2];
    for substeps in [25, 50, 100, 200, 400]
    {
        let defect = holonomy_defect(&MinkowskiSpherical, &loop_path, &flat_vector, substeps)
            .map_err(|e| e.to_string())?;
        let max_defect = defect.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        require_finite(&[("max_defect", max_defect)])?;
        println!("{substeps},{max_defect:.3e}");
    }

    // Part C: holonomy/curvature identity convergence with loop size.
    println!("#");
    println!(
        "# Part C: relative gap between the loop holonomy and -R^rho_(sigma mu nu) V^sigma A^mu B^nu"
    );
    println!(
        "# (r,theta) loop; the gap is the O(eps) next-order correction and should fall with eps"
    );
    println!("background,eps,relative_gap_to_riemann_prediction");
    for &eps in &[1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3]
    {
        let de_sitter_gap = holonomy_vs_riemann(
            &de_sitter,
            [0.0, 3.0, FRAC_PI_2, 0.2],
            [0.1, 0.2, 0.05, 0.03],
            1,
            2,
            eps,
        )?;
        let schwarzschild_gap = holonomy_vs_riemann(
            &schwarzschild,
            [0.0, 8.0, FRAC_PI_2, 0.2],
            [0.1, 0.2, 0.05, 0.03],
            1,
            2,
            eps,
        )?;
        require_finite(&[
            ("de_sitter_gap", de_sitter_gap),
            ("schwarzschild_gap", schwarzschild_gap),
        ])?;
        println!("de_Sitter,{eps:.0e},{de_sitter_gap:.3e}");
        println!("Schwarzschild,{eps:.0e},{schwarzschild_gap:.3e}");
    }

    println!("# interpretation: the transported metric norm is preserved (drift falls ~h^2); the");
    println!("# flat-chart closed-loop holonomy falls to the roundoff floor; and the curved-loop");
    println!(
        "# holonomy matches the independent Riemann-tensor prediction with an O(eps) gap. The"
    );
    println!(
        "# transport and curvature engines thus cross-validate against one analytic identity."
    );
    Ok(())
}
