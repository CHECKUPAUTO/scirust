//! Compares unmodulated and Schwarzschild-Kretschmann-modulated velocity
//! memory on a Schwarzschild exterior worldline, across two refinement
//! levels and two transport strategies.
//!
//! `SchwarzschildKretschmannModulator` is an explicitly experimental,
//! phenomenological weight `q = 1 + beta * L^4 * K` applied to each retained
//! history sample before the Caputo evaluation, where `K = 48 M^2 / r^6` is
//! the Schwarzschild Kretschmann scalar. This is **not** a consequence of
//! general relativity, **not** a quantum-gravity prediction, **not** an
//! experimentally derived law, and **not** a modification of the Einstein
//! field equations — see `scirust-nonlocal-relativity/README.md` and
//! `docs/EXPERIMENTAL_NONLOCAL_RELATIVITY.md`.

use scirust_nonlocal_relativity::{
    CompleteUniformHistory, DiscreteConnectionTransport, HeunPeceStepper, HistoryEntry,
    HistoryModulator, IdentityHistoryTransport, ModulatedCaputoCoordinateMemory, NonlocalConfig,
    NonlocalSimulationPolicy, NonlocalTrajectory, SchwarzschildKretschmannModulator,
    WorldlineState, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn circular_timelike_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

#[allow(clippy::too_many_arguments)]
fn print_rows(
    beta_label: &str,
    refinement_label: &str,
    transport_label: &str,
    modulator: &SchwarzschildKretschmannModulator,
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
        let kretschmann = 48.0 * modulator.mass() * modulator.mass() / radius.powi(6);

        println!(
            "{beta_label},{refinement_label},{transport_label},complete,heun_pece,\
             {:.12e},{radius:.12},{kretschmann:.12e},{weight:.12e},{:.12e},{:.12e},{:.12e},{radial_deviation:.12e}",
            diagnostics.affine_parameter,
            diagnostics.memory_l2_norm,
            diagnostics.memory_force_l2_norm,
            diagnostics.metric_norm_drift,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).expect("positive mass");
    let mut initial = circular_timelike_state(mass, 10.0);
    initial.velocity[1] = -0.01;

    let alpha = 0.55;
    let coupling = 0.02;
    let metric_norm_floor = 1.0e-8;
    let reference_length = mass;
    let base_step = 0.02;
    let base_steps = 80;

    println!(
        "beta,refinement_level,transport_mode,history_backend,integrator,parameter,radius,\
         kretschmann,modulation_weight,memory_l2_norm,memory_force_l2_norm,metric_norm_drift,\
         radial_deviation"
    );

    for (refinement_label, factor) in [("h", 1usize), ("h/2", 2usize)]
    {
        let step = base_step / factor as f64;
        let steps = base_steps * factor;
        let config = NonlocalConfig::new(alpha, coupling, step, steps, metric_norm_floor)?;
        let stride = 8 * factor;

        for (transport_label, discrete_transport) in [("coordinate", false), ("transported", true)]
        {
            let baseline_modulator =
                SchwarzschildKretschmannModulator::try_new(mass, reference_length, 0.0)?;
            let modulated_modulator =
                SchwarzschildKretschmannModulator::try_new(mass, reference_length, 0.05)?;

            let baseline_trajectory = if discrete_transport
            {
                simulate_nonlocal_worldline_with_policy(
                    &background,
                    initial,
                    config,
                    NonlocalSimulationPolicy::new(
                        CompleteUniformHistory::<4>::with_capacity(steps + 1),
                        ModulatedCaputoCoordinateMemory::new(baseline_modulator),
                        DiscreteConnectionTransport,
                        HeunPeceStepper,
                    ),
                )?
            }
            else
            {
                simulate_nonlocal_worldline_with_policy(
                    &background,
                    initial,
                    config,
                    NonlocalSimulationPolicy::new(
                        CompleteUniformHistory::<4>::with_capacity(steps + 1),
                        ModulatedCaputoCoordinateMemory::new(baseline_modulator),
                        IdentityHistoryTransport,
                        HeunPeceStepper,
                    ),
                )?
            };

            let modulated_trajectory = if discrete_transport
            {
                simulate_nonlocal_worldline_with_policy(
                    &background,
                    initial,
                    config,
                    NonlocalSimulationPolicy::new(
                        CompleteUniformHistory::<4>::with_capacity(steps + 1),
                        ModulatedCaputoCoordinateMemory::new(modulated_modulator),
                        DiscreteConnectionTransport,
                        HeunPeceStepper,
                    ),
                )?
            }
            else
            {
                simulate_nonlocal_worldline_with_policy(
                    &background,
                    initial,
                    config,
                    NonlocalSimulationPolicy::new(
                        CompleteUniformHistory::<4>::with_capacity(steps + 1),
                        ModulatedCaputoCoordinateMemory::new(modulated_modulator),
                        IdentityHistoryTransport,
                        HeunPeceStepper,
                    ),
                )?
            };

            print_rows(
                "0.00",
                refinement_label,
                transport_label,
                &baseline_modulator,
                &baseline_trajectory,
                &baseline_trajectory,
                stride,
            );
            print_rows(
                "0.05",
                refinement_label,
                transport_label,
                &modulated_modulator,
                &modulated_trajectory,
                &baseline_trajectory,
                stride,
            );
        }
    }

    Ok(())
}
