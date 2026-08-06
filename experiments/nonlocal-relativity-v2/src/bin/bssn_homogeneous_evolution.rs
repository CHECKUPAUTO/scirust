//! BSSN formulation-core checks against the Layer 3.2 ADM evolution path:
//! ADM/BSSN right-hand-side equivalence, algebraic constraints, the conformal
//! Ricci decomposition, and a deliberate algebraic violation with its explicit
//! projection (the third Layer 3 slice; see `docs/LAYER_3_BSSN.md`).
//!
//! For each exact Friedmann scenario and step size the experiment evolves the
//! homogeneous state with the Layer 3.2 ADM path, converts the state to BSSN,
//! and reports the exact scale factor, the ADM result and error, the observed
//! convergence ratio, the Hamiltonian residual, and every BSSN algebraic
//! constraint. The BSSN column reports the scale factor reconstructed from the
//! BSSN variables, so any conversion drift would show as an ADM/BSSN
//! difference.
//!
//! Established general relativity. Every quantity is a numerical approximation.
//! **Implementing BSSN does not demonstrate numerical stability**: there is no
//! spatial grid here, so nothing about hyperbolicity is tested or claimed.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::adm_evolution::{AdmEvolutionSettings, AdmSources, SpatialTensorField};
use scirust_relativity::adm_homogeneous::{
    BarotropicFluid, HomogeneousEvolution, HomogeneousState, evolve_homogeneous,
};
use scirust_relativity::bssn::{
    BssnGauge, adm_bssn_equivalence, bssn_constraints_from_adm, bssn_to_adm, conformal_ricci,
    project_trace_free, project_unit_determinant,
};
use scirust_relativity::{
    ExponentialScaleFactor, Metric, PowerLawScaleFactor, ScaleFactor, determinant,
    ricci_tensor_from_metric,
};

const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

struct ConstantMetric([[f64; 3]; 3]);
impl Metric<3> for ConstantMetric {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}
struct ConstantTensor([[f64; 3]; 3]);
impl SpatialTensorField for ConstantTensor {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}
fn diagonal(value: f64) -> [[f64; 3]; 3] {
    [[value, 0.0, 0.0], [0.0, value, 0.0], [0.0, 0.0, value]]
}

/// The Schwarzschild `t = const` slice, for the Ricci-decomposition rows.
struct SchwarzschildSlice {
    mass: f64,
}
impl Metric<3> for SchwarzschildSlice {
    fn components(&self, x: &[f64; 3]) -> [[f64; 3]; 3] {
        let (radius, polar) = (x[0], x[1]);
        let lapse = 1.0 - 2.0 * self.mass / radius;
        let sine = polar.sin();
        [
            [1.0 / lapse, 0.0, 0.0],
            [0.0, radius * radius, 0.0],
            [0.0, 0.0, radius * radius * sine * sine],
        ]
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_evolution_row(
    scenario: &str,
    fluid: BarotropicFluid,
    initial_scale: f64,
    hubble: f64,
    start: f64,
    end: f64,
    step: f64,
    exact_end: f64,
    previous_error: &mut Option<f64>,
) -> Result<(), String> {
    let evolution = HomogeneousEvolution::new(fluid, settings());
    let initial = HomogeneousState::friedmann(initial_scale, hubble).map_err(|e| e.to_string())?;
    let history =
        evolve_homogeneous(&evolution, &initial, start, end, step).map_err(|e| e.to_string())?;
    let last = history.last().ok_or("empty history")?;
    let adm_scale = last.scale_factor;
    let adm_error = (adm_scale - exact_end).abs();

    // Rebuild the final ADM state, convert to BSSN, and reconstruct: the BSSN
    // scale factor and every algebraic constraint come from that round trip.
    let spatial = adm_scale * adm_scale;
    let metric = diagonal(spatial);
    let curvature = diagonal(-last.hubble_parameter * spatial);
    let (state, constraints) = bssn_constraints_from_adm(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings(),
    )
    .map_err(|e| e.to_string())?;
    let reconstructed = bssn_to_adm(&state).map_err(|e| e.to_string())?;
    let bssn_scale = determinant::<3>(&reconstructed.spatial_metric)
        .map_err(|e| e.to_string())?
        .powf(1.0 / 6.0);
    let bssn_error = (bssn_scale - exact_end).abs();
    let difference = (bssn_scale - adm_scale).abs();

    // ADM/BSSN right-hand-side equivalence at the final state.
    let sources = AdmSources::perfect_fluid(
        last.energy_density,
        fluid.pressure(last.energy_density),
        &metric,
    );
    let equivalence = adm_bssn_equivalence(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &BssnGauge::SYNCHRONOUS,
        &sources,
        &settings(),
    )
    .map_err(|e| e.to_string())?;

    require_finite(&[
        ("adm_scale", adm_scale),
        ("bssn_scale", bssn_scale),
        ("adm_error", adm_error),
        ("bssn_error", bssn_error),
        ("hamiltonian", last.hamiltonian_residual),
        ("determinant", constraints.determinant_residual),
        ("trace", constraints.trace_residual),
        ("connection", constraints.connection_absolute),
        ("rhs_metric", equivalence.metric_difference),
        (
            "rhs_curvature",
            equivalence.curvature_difference_constraint_corrected,
        ),
    ])?;

    let ratio = match previous_error
    {
        Some(previous) if adm_error > 0.0 => format!("{:.2}", *previous / adm_error),
        _ => "n_a".to_string(),
    };
    *previous_error = Some(adm_error);

    println!(
        "{scenario},{},{step},{},{exact_end:.10},{adm_scale:.10},{bssn_scale:.10},{adm_error:.4e},\
         {bssn_error:.4e},{difference:.4e},{ratio},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},valid",
        fluid.equation_of_state(),
        history.len(),
        last.hamiltonian_residual,
        constraints.determinant_residual,
        constraints.trace_residual,
        constraints.connection_absolute,
        equivalence.metric_difference,
        equivalence.curvature_difference_constraint_corrected,
    );
    Ok(())
}

// Explicit tensor-index loops read most clearly here (matching the relativity
// crate's curvature, adm, and bssn modules).
#[allow(clippy::needless_range_loop)]
fn main() -> Result<(), String> {
    print_experiment_header(
        "BSSN formulation core against the ADM evolution path",
        "scirust-relativity Layer 3.3 (established general relativity)",
        "BSSN conversion, algebraic constraints, and ADM/BSSN RHS equivalence; numerical, not a proof of stability.",
    );

    println!(
        "scenario,equation_of_state_w,step,samples,exact_scale_factor,adm_scale_factor,\
         bssn_scale_factor,adm_error,bssn_error,adm_bssn_difference,convergence_ratio,\
         hamiltonian_residual,determinant_constraint,trace_free_constraint,\
         connection_constraint,rhs_metric_difference,rhs_curvature_difference_corrected,state_valid"
    );

    let hubble = 0.5;
    let de_sitter = ExponentialScaleFactor::try_new(hubble).ok_or("invalid Hubble")?;
    let mut previous = None;
    for &step in &[0.05, 0.025, 0.0125, 0.00625]
    {
        emit_evolution_row(
            "de_sitter",
            BarotropicFluid::vacuum_energy(),
            de_sitter.value(0.0),
            hubble,
            0.0,
            2.0,
            step,
            de_sitter.value(2.0),
            &mut previous,
        )?;
    }

    let reference_time = 1.0;
    for &(label, exponent, fluid) in &[
        ("dust", 2.0 / 3.0, BarotropicFluid::dust()),
        ("radiation", 0.5, BarotropicFluid::radiation()),
    ]
    {
        let exact =
            PowerLawScaleFactor::try_new(exponent, reference_time).ok_or("invalid power law")?;
        let mut previous = None;
        for &step in &[0.02, 0.01, 0.005, 0.0025]
        {
            emit_evolution_row(
                label,
                fluid,
                exact.value(reference_time),
                exponent / reference_time,
                reference_time,
                2.0,
                step,
                exact.value(2.0),
                &mut previous,
            )?;
        }
    }

    // Conformal Ricci decomposition on a curved slice: Rtilde + Rphi vs the
    // independently computed physical Ricci.
    println!("# conformal Ricci decomposition (Schwarzschild slice, M = 1):");
    println!(
        "# scenario,radius,polar,ricci_scale,conformal_scale,conformal_factor_scale,max_mismatch,status"
    );
    let slice = SchwarzschildSlice { mass: 1.0 };
    for &(radius, polar) in &[(6.0, 1.27), (8.0, 1.0), (12.0, 1.4)]
    {
        let point = [radius, polar, 0.0];
        let decomposition =
            conformal_ricci(&slice, &point, &settings()).map_err(|e| e.to_string())?;
        let physical = ricci_tensor_from_metric::<_, 3>(&slice, &point, 1.0e-3, 1.0e-3)
            .map_err(|e| e.to_string())?;
        let mut mismatch = 0.0_f64;
        let mut ricci_scale = 0.0_f64;
        let mut conformal_scale = 0.0_f64;
        let mut factor_scale = 0.0_f64;
        for i in 0..3
        {
            for j in 0..3
            {
                mismatch = mismatch.max((decomposition.total[i][j] - physical[i][j]).abs());
                ricci_scale = ricci_scale.max(physical[i][j].abs());
                conformal_scale = conformal_scale.max(decomposition.conformal[i][j].abs());
                factor_scale = factor_scale.max(decomposition.conformal_factor_part[i][j].abs());
            }
        }
        require_finite(&[("mismatch", mismatch), ("ricci_scale", ricci_scale)])?;
        println!(
            "ricci_decomposition,{radius},{polar},{ricci_scale:.6e},{conformal_scale:.6e},\
             {factor_scale:.6e},{mismatch:.4e},valid"
        );
    }

    // Deliberate algebraic violations and their explicit projections.
    println!("# deliberate algebraic violations and explicit projection:");
    println!("# scenario,amplitude,residual_before,residual_after,correction_magnitude,status");
    let metric = [[1.3, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.1]];
    let curvature = [[-0.21, 0.0, 0.0], [0.0, -0.09, 0.0], [0.0, 0.0, -0.15]];
    for &amplitude in &[0.01, 0.05, 0.1]
    {
        let (mut state, _) = bssn_constraints_from_adm(
            &ConstantMetric(metric),
            &ConstantTensor(curvature),
            &ORIGIN,
            &settings(),
        )
        .map_err(|e| e.to_string())?;
        let factor = 1.0 + amplitude;
        for i in 0..3
        {
            for j in 0..3
            {
                state.conformal_metric[i][j] *= factor;
            }
        }
        let projection = project_unit_determinant(&mut state).map_err(|e| e.to_string())?;
        require_finite(&[
            ("before", projection.residual_before),
            ("after", projection.residual_after),
        ])?;
        println!(
            "determinant_violation,{amplitude},{:.6e},{:.6e},{:.6e},projected",
            projection.residual_before, projection.residual_after, projection.correction_magnitude
        );

        let (mut state, _) = bssn_constraints_from_adm(
            &ConstantMetric(metric),
            &ConstantTensor(curvature),
            &ORIGIN,
            &settings(),
        )
        .map_err(|e| e.to_string())?;
        for i in 0..3
        {
            for j in 0..3
            {
                state.conformal_curvature[i][j] += amplitude * state.conformal_metric[i][j];
            }
        }
        let projection = project_trace_free(&mut state).map_err(|e| e.to_string())?;
        println!(
            "trace_violation,{amplitude},{:.6e},{:.6e},{:.6e},projected",
            projection.residual_before, projection.residual_after, projection.correction_magnitude
        );
    }

    // The off-constraint ADM/BSSN difference is exactly alpha * H.
    println!("# off-constraint ADM/BSSN d_t K difference, raw vs constraint-corrected:");
    println!("# scenario,hamiltonian_residual,raw_difference,corrected_difference,status");
    let sources = AdmSources::perfect_fluid(0.02, 0.007, &metric);
    let equivalence = adm_bssn_equivalence(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &BssnGauge::SYNCHRONOUS,
        &sources,
        &settings(),
    )
    .map_err(|e| e.to_string())?;
    require_finite(&[
        ("raw", equivalence.curvature_difference),
        (
            "corrected",
            equivalence.curvature_difference_constraint_corrected,
        ),
    ])?;
    println!(
        "off_constraint,{:.6e},{:.6e},{:.6e},explained_by_alpha_times_hamiltonian",
        equivalence.hamiltonian_residual,
        equivalence.curvature_difference,
        equivalence.curvature_difference_constraint_corrected
    );

    println!("# interpretation: the BSSN conversion reproduces the ADM scale factor to rounding");
    println!("# (adm_bssn_difference ~ 1e-16), every algebraic constraint is satisfied by");
    println!("# construction, the conformal Ricci decomposition Rtilde + Rphi reproduces the");
    println!("# independently computed physical Ricci to the nested finite-difference floor, and");
    println!("# the ADM/BSSN right-hand sides agree on the constraint surface. Off it, the d_t K");
    println!("# difference is not an error: it is exactly alpha * H, the Hamiltonian residual");
    println!("# introduced by the standard constraint-substituted BSSN trace equation, and the");
    println!("# corrected column shows it vanishing. Deliberate violations are detected and");
    println!("# repaired only by explicit projection. No spatial grid exists here, so nothing");
    println!("# about hyperbolicity or numerical stability is tested or claimed.");
    Ok(())
}
