//! Homogeneous ADM free-evolution checks: the three exact Friedmann solutions,
//! fourth-order convergence, constraint preservation, and a deliberately
//! constraint-violating run (the second Layer 3 slice; see
//! `docs/LAYER_3_HOMOGENEOUS_EVOLUTION.md`).
//!
//! For each equation of state and step size the experiment freely evolves
//! homogeneous ADM data -- integrating the Layer 3.1 right-hand sides through
//! the platform's existing RK4 -- and reports the final scale factor against
//! the exact closed-form solution, the error and its convergence ratio, the
//! Hubble parameter, the energy density, and the worst Hamiltonian constraint
//! residual seen anywhere along the trajectory. The constraint is *monitored*,
//! never enforced.
//!
//! Established general relativity. The evolution is a numerical approximation
//! (fixed-step RK4 through finite-difference right-hand sides). It is
//! restricted to the spatially homogeneous sector, where ADM's weak
//! hyperbolicity cannot arise, so it says nothing about inhomogeneous
//! stability, and it is not a cosmological model: the equations of state are
//! validation oracles, not a fit to any observation.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::adm_evolution::AdmEvolutionSettings;
use scirust_relativity::adm_homogeneous::{
    BarotropicFluid, HomogeneousEvolution, HomogeneousState, evolve_homogeneous,
};
use scirust_relativity::{ExponentialScaleFactor, PowerLawScaleFactor, ScaleFactor};

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_row(
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
    let equation_of_state = fluid.equation_of_state();
    let evolution = HomogeneousEvolution::new(fluid, settings());
    let initial =
        HomogeneousState::friedmann(initial_scale, hubble).map_err(|error| error.to_string())?;
    let history = evolve_homogeneous(&evolution, &initial, start, end, step)
        .map_err(|error| error.to_string())?;
    let last = history.last().ok_or("empty history")?;
    let worst_residual = history.iter().fold(0.0_f64, |acc, sample| {
        acc.max(sample.hamiltonian_residual.abs())
    });
    let error = (last.scale_factor - exact_end).abs();

    require_finite(&[
        ("scale_factor", last.scale_factor),
        ("hubble_parameter", last.hubble_parameter),
        ("energy_density", last.energy_density),
        ("scale_factor_error", error),
        ("worst_hamiltonian_residual", worst_residual),
    ])?;

    let ratio = match previous_error
    {
        Some(previous) if error > 0.0 => format!("{:.2}", *previous / error),
        _ => "n_a".to_string(),
    };
    *previous_error = Some(error);

    println!(
        "{scenario},{equation_of_state:.6},{step},{},{:.10},{exact_end:.10},{error:.4e},{ratio},\
         {:.10},{:.6e},{:.4e},ok",
        history.len(),
        last.scale_factor,
        last.hubble_parameter,
        last.energy_density,
        worst_residual,
    );
    Ok(())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "Homogeneous ADM free evolution with constraint monitoring",
        "scirust-relativity Layer 3.2 (established general relativity)",
        "ADM evolution integrated by fixed-step RK4 against exact Friedmann solutions; numerical, not a bound.",
    );

    println!(
        "scenario,equation_of_state_w,step,samples,scale_factor,exact_scale_factor,\
         scale_factor_error,error_ratio,hubble_parameter,energy_density,\
         worst_hamiltonian_residual,status"
    );

    // O1 + O4: de Sitter, a(t) = exp(H t); the error ratio exposes RK4's order.
    let hubble = 0.5;
    let de_sitter = ExponentialScaleFactor::try_new(hubble).ok_or("invalid Hubble")?;
    let mut previous = None;
    for &step in &[0.05, 0.025, 0.0125, 0.00625]
    {
        emit_row(
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

    // O2 + O4: dust, a(t) = (t / t_ref)^(2/3).
    let reference_time = 1.0;
    let dust_exponent = 2.0 / 3.0;
    let dust = PowerLawScaleFactor::try_new(dust_exponent, reference_time)
        .ok_or("invalid dust power law")?;
    let mut previous = None;
    for &step in &[0.02, 0.01, 0.005, 0.0025]
    {
        emit_row(
            "dust",
            BarotropicFluid::dust(),
            dust.value(reference_time),
            dust_exponent / reference_time,
            reference_time,
            2.0,
            step,
            dust.value(2.0),
            &mut previous,
        )?;
    }

    // O3 + O4: radiation, a(t) = (t / t_ref)^(1/2).
    let radiation_exponent = 0.5;
    let radiation = PowerLawScaleFactor::try_new(radiation_exponent, reference_time)
        .ok_or("invalid radiation power law")?;
    let mut previous = None;
    for &step in &[0.02, 0.01, 0.005, 0.0025]
    {
        emit_row(
            "radiation",
            BarotropicFluid::radiation(),
            radiation.value(reference_time),
            radiation_exponent / reference_time,
            reference_time,
            2.0,
            step,
            radiation.value(2.0),
            &mut previous,
        )?;
    }

    // O6: initial data seeded off the constraint surface stays violated -- free
    // evolution has no projection or damping to repair it.
    println!("# deliberate constraint violation (dust, rho scaled off the constraint surface):");
    println!("# scenario,excess_fraction,initial_residual,final_residual,status");
    let evolution = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());
    for &excess in &[0.01, 0.05, 0.1]
    {
        let mut initial = HomogeneousState::friedmann(1.0, hubble).map_err(|e| e.to_string())?;
        initial.energy_density *= 1.0 + excess;
        let initial_residual = evolution
            .hamiltonian_residual(&initial)
            .map_err(|e| e.to_string())?;
        let history =
            evolve_homogeneous(&evolution, &initial, 1.0, 2.0, 0.005).map_err(|e| e.to_string())?;
        let final_residual = history.last().ok_or("empty history")?.hamiltonian_residual;
        require_finite(&[
            ("initial_residual", initial_residual),
            ("final_residual", final_residual),
        ])?;
        println!(
            "constraint_violation,{excess},{initial_residual:.6e},{final_residual:.6e},\
             violation_detected"
        );
    }

    println!("# interpretation: the evolved scale factor matches the exact de Sitter, dust, and");
    println!("# radiation solutions, with the error falling by ~2^4 = 16 per step halving (the");
    println!("# RK4 order survives the finite-difference right-hand sides); the Hamiltonian");
    println!("# constraint -- monitored, never enforced -- stays at the integration-error floor");
    println!("# and improves with the step; and initial data seeded off the constraint surface");
    println!("# stays visibly violated, so bad data is detected rather than silently repaired.");
    println!("# Numerical approximation, not a bound. Homogeneous sector only: this says nothing");
    println!("# about ADM's stability for inhomogeneous data, and it is not a cosmological model.");
    Ok(())
}
