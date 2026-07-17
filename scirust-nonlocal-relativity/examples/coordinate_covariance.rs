//! Compares the same timelike, memory-coupled motion computed in two charts
//! of flat Minkowski spacetime: Cartesian coordinates (zero connection) and
//! cylindrical coordinates (non-zero connection).
//!
//! The Caputo velocity memory used by this crate is evaluated componentwise
//! in whatever chart the background supplies, so it is coordinate-dependent
//! by construction (see `scirust-nonlocal-relativity/README.md`). This
//! example is a controlled numerical check, not a proof, that:
//!
//! - transporting retained history vectors with [`DiscreteConnectionTransport`]
//!   before the Caputo evaluation reduces the disagreement between the
//!   Cartesian and cylindrical computations, compared to raw componentwise
//!   ("coordinate") memory in the same cylindrical chart;
//! - both disagreements shrink under step refinement.
//!
//! It does not claim exact equality between charts: the discretization
//! itself is chart-dependent, so residual disagreement is expected even for
//! the transported pipeline.

use scirust_nonlocal_relativity::{
    CaputoCoordinateMemory, CompleteUniformHistory, CylindricalMinkowski,
    DiscreteConnectionTransport, HeunPeceStepper, IdentityHistoryTransport, NonlocalConfig,
    NonlocalSimulationPolicy, NonlocalTrajectory, WorldlineState,
    cartesian_to_cylindrical_coordinates, cartesian_to_cylindrical_velocity, coordinate_l2_norm,
    cylindrical_to_cartesian_coordinates, cylindrical_to_cartesian_velocity,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Minkowski;
use std::error::Error;

fn vector_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut difference = [0.0_f64; 4];

    for component in 0..4
    {
        difference[component] = left[component] - right[component];
    }

    coordinate_l2_norm(&difference)
}

fn print_row(
    memory_method: &str,
    refinement_label: &str,
    step: f64,
    cartesian_final: &WorldlineState<4>,
    cylindrical_trajectory: &NonlocalTrajectory<4>,
) -> Result<(), Box<dyn Error>> {
    let final_state = *cylindrical_trajectory
        .final_state()
        .expect("non-empty trajectory");
    let final_diagnostics = *cylindrical_trajectory
        .final_diagnostics()
        .expect("non-empty trajectory");
    let samples_used = cylindrical_trajectory
        .history_diagnostics()
        .last()
        .expect("non-empty trajectory")
        .used_samples;

    let converted_coordinates = cylindrical_to_cartesian_coordinates(final_state.coordinates)?;
    let converted_velocity =
        cylindrical_to_cartesian_velocity(final_state.coordinates, final_state.velocity)?;

    let position_disagreement =
        vector_distance(&cartesian_final.coordinates, &converted_coordinates);
    let velocity_disagreement = vector_distance(&cartesian_final.velocity, &converted_velocity);

    println!(
        "{memory_method},{refinement_label},{step:.12e},{position_disagreement:.12e},\
         {velocity_disagreement:.12e},{:.12e},{:.12e},{samples_used}",
        final_diagnostics.orthogonality_residual, final_diagnostics.metric_norm_drift,
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cartesian_initial = WorldlineState::new([0.0, 3.0, 4.0, 0.0], [1.2, 0.3, -0.2, 0.1]);
    let cylindrical_coordinates =
        cartesian_to_cylindrical_coordinates(cartesian_initial.coordinates)?;
    let cylindrical_velocity = cartesian_to_cylindrical_velocity(
        cartesian_initial.coordinates,
        cartesian_initial.velocity,
    )?;
    let cylindrical_initial = WorldlineState::new(cylindrical_coordinates, cylindrical_velocity);
    assert!(
        CylindricalMinkowski::is_regular(&cylindrical_initial.coordinates),
        "initial cylindrical coordinates must lie in the regular chart region"
    );

    let alpha = 0.5;
    let coupling = 0.4;
    let metric_norm_floor = 1.0e-8;
    let base_step = 0.02;
    let base_steps = 40;

    println!(
        "memory_method,refinement_level,step,final_position_disagreement,\
         final_velocity_disagreement,orthogonality_residual,metric_norm_drift,samples_used"
    );

    for (refinement_label, factor) in [("h", 1usize), ("h/2", 2usize), ("h/4", 4usize)]
    {
        let step = base_step / factor as f64;
        let steps = base_steps * factor;
        let config = NonlocalConfig::new(alpha, coupling, step, steps, metric_norm_floor)?;

        let cartesian_trajectory = simulate_nonlocal_worldline_with_policy(
            &Minkowski,
            cartesian_initial,
            config,
            NonlocalSimulationPolicy::new(
                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                CaputoCoordinateMemory,
                IdentityHistoryTransport,
                HeunPeceStepper,
            ),
        )?;
        let cartesian_final = *cartesian_trajectory
            .final_state()
            .expect("non-empty trajectory");

        let coordinate_memory_trajectory = simulate_nonlocal_worldline_with_policy(
            &CylindricalMinkowski,
            cylindrical_initial,
            config,
            NonlocalSimulationPolicy::new(
                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                CaputoCoordinateMemory,
                IdentityHistoryTransport,
                HeunPeceStepper,
            ),
        )?;
        let transported_memory_trajectory = simulate_nonlocal_worldline_with_policy(
            &CylindricalMinkowski,
            cylindrical_initial,
            config,
            NonlocalSimulationPolicy::new(
                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                CaputoCoordinateMemory,
                DiscreteConnectionTransport,
                HeunPeceStepper,
            ),
        )?;

        print_row(
            "coordinate",
            refinement_label,
            step,
            &cartesian_final,
            &coordinate_memory_trajectory,
        )?;
        print_row(
            "transported",
            refinement_label,
            step,
            &cartesian_final,
            &transported_memory_trajectory,
        )?;
    }

    Ok(())
}
