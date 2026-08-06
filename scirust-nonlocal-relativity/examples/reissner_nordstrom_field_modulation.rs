//! Compares unmodulated and electromagnetic-field-modulated velocity memory
//! on a Reissner-Nordström exterior worldline, across two refinement levels.
//!
//! `ReissnerNordstromFieldModulator` is an explicitly experimental,
//! phenomenological weight `q = 1 + beta * L^2 * |F^2|` applied to each
//! retained history sample before the Caputo evaluation, where
//! `F^2 = F_(mu nu) F^(mu nu) = 2 Q^2 / r^4` is the electromagnetic field
//! invariant of the Reissner-Nordström background's radial Coulomb field —
//! not a curvature invariant, and not the Kretschmann scalar used by
//! `SchwarzschildKretschmannModulator`. This is **not** a consequence of
//! general relativity or electromagnetism, **not** a quantum-field-theory
//! prediction, **not** an experimentally derived law, and **not** a
//! modification of the Einstein or Maxwell equations — see
//! `scirust-nonlocal-relativity/README.md` and
//! `docs/EXPERIMENTAL_NONLOCAL_RELATIVITY.md`.

use scirust_nonlocal_relativity::{
    CompleteUniformHistory, HeunPeceStepper, HistoryEntry, HistoryModulator,
    IdentityHistoryTransport, ModulatedCaputoCoordinateMemory, NonlocalConfig,
    NonlocalSimulationPolicy, NonlocalTrajectory, ReissnerNordstromFieldModulator, WorldlineState,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::ReissnerNordstrom;
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn approximately_circular_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn print_rows(
    beta_label: &str,
    refinement_label: &str,
    modulator: &ReissnerNordstromFieldModulator,
    trajectory: &NonlocalTrajectory<4>,
    baseline_trajectory: &NonlocalTrajectory<4>,
    stride: usize,
) {
    for index in (0..trajectory.len()).step_by(stride)
    {
        let state = trajectory.states()[index];
        let diagnostics = trajectory.diagnostics()[index];
        let baseline_state = baseline_trajectory.states()[index];
        let radial_deviation = state.coordinates[1] - baseline_state.coordinates[1];

        let probe_entry = HistoryEntry::new(
            state.coordinates,
            state.velocity,
            diagnostics.affine_parameter,
        );
        let weight = modulator
            .weight(&probe_entry)
            .expect("regular exterior sample yields a finite weight");
        let radius = state.coordinates[1];
        let charge = modulator.background().charge();
        let field_invariant = 2.0 * charge * charge / radius.powi(4);

        println!(
            "{beta_label},{refinement_label},{:.12e},{radius:.12},{field_invariant:.12e},\
             {weight:.12e},{:.12e},{:.12e},{:.12e},{radial_deviation:.12e}",
            diagnostics.affine_parameter,
            diagnostics.memory_l2_norm,
            diagnostics.memory_force_l2_norm,
            diagnostics.metric_norm_drift,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let charge = 0.4;
    let background = ReissnerNordstrom::try_new(mass, charge).expect("sub-extremal parameters");
    let mut initial = approximately_circular_state(mass, 10.0);
    initial.velocity[1] = -0.01;

    let alpha = 0.55;
    let coupling = 0.02;
    let metric_norm_floor = 1.0e-8;
    let reference_length = mass;
    let base_step = 0.02;
    let base_steps = 80;

    println!(
        "beta,refinement_level,parameter,radius,field_invariant,modulation_weight,\
         memory_l2_norm,memory_force_l2_norm,metric_norm_drift,radial_deviation"
    );

    for (refinement_label, factor) in [("h", 1usize), ("h/2", 2usize)]
    {
        let step = base_step / factor as f64;
        let steps = base_steps * factor;
        let config = NonlocalConfig::new(alpha, coupling, step, steps, metric_norm_floor)?;
        let stride = 8 * factor;

        let baseline_modulator =
            ReissnerNordstromFieldModulator::try_new(background, reference_length, 0.0)?;
        let modulated_modulator =
            ReissnerNordstromFieldModulator::try_new(background, reference_length, 0.05)?;

        let baseline_trajectory = simulate_nonlocal_worldline_with_policy(
            &background,
            initial,
            config,
            NonlocalSimulationPolicy::new(
                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                ModulatedCaputoCoordinateMemory::new(baseline_modulator),
                IdentityHistoryTransport,
                HeunPeceStepper,
            ),
        )?;
        let modulated_trajectory = simulate_nonlocal_worldline_with_policy(
            &background,
            initial,
            config,
            NonlocalSimulationPolicy::new(
                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                ModulatedCaputoCoordinateMemory::new(modulated_modulator),
                IdentityHistoryTransport,
                HeunPeceStepper,
            ),
        )?;

        print_rows(
            "0.00",
            refinement_label,
            &baseline_modulator,
            &baseline_trajectory,
            &baseline_trajectory,
            stride,
        );
        print_rows(
            "0.05",
            refinement_label,
            &modulated_modulator,
            &modulated_trajectory,
            &baseline_trajectory,
            stride,
        );
    }

    Ok(())
}
