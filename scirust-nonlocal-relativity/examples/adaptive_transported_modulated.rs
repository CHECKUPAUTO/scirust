//! Demonstrates composing the adaptive-step integrator
//! ([`simulate_nonlocal_worldline_adaptive_with_policy`]) with geometric
//! transport ([`DiscreteConnectionTransport`]) and curvature modulation
//! ([`SchwarzschildKretschmannModulator`]) together, on the same
//! Schwarzschild trajectory and error tolerance.
//!
//! `simulate_nonlocal_worldline_adaptive` (the plain entry point) is the
//! identity-transport, unmodulated special case of this same function; this
//! example shows that swapping in a real transport and/or a real modulator
//! changes the numerical result relative to that baseline while every
//! combination remains well-behaved (finite, deterministic, reaching the
//! same target affine parameter).

use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, AdaptiveSimulationPolicy, CompleteUniformHistory,
    DiscreteConnectionTransport, HistoryModulator, HistoryTransport, IdentityHistoryTransport,
    SchwarzschildKretschmannModulator, WorldlineState,
    simulate_nonlocal_worldline_adaptive_with_policy,
};
use scirust_relativity::Schwarzschild;
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn run<T, M>(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    config: AdaptiveNonlocalConfig,
    transport: T,
    modulator: M,
) -> Result<(usize, f64, f64), Box<dyn Error>>
where
    T: HistoryTransport<4>,
    M: HistoryModulator<4>,
{
    let trajectory = simulate_nonlocal_worldline_adaptive_with_policy(
        background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(CompleteUniformHistory::<4>::new(), transport, modulator),
    )?;
    let final_state = trajectory.final_state().expect("non-empty trajectory");
    let final_diagnostics = trajectory
        .final_diagnostics()
        .expect("non-empty trajectory");

    Ok((
        trajectory.len() - 1,
        final_state.coordinates[1],
        final_diagnostics.memory_l2_norm,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).expect("positive mass");
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;

    let config = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.02, 0.0001, 0.1, 1.0e-9, 1.0e-8, 0.8, 5_000, 40,
    )?;
    let identity_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.0)?;
    let coupled_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.3)?;

    println!("transport,modulation,accepted_steps,final_radius,final_memory_l2_norm");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        IdentityHistoryTransport,
        identity_modulator,
    )?;
    println!("identity,none,{steps},{radius:.12e},{memory:.12e}");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        DiscreteConnectionTransport,
        identity_modulator,
    )?;
    println!("discrete,none,{steps},{radius:.12e},{memory:.12e}");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        IdentityHistoryTransport,
        coupled_modulator,
    )?;
    println!("identity,kretschmann,{steps},{radius:.12e},{memory:.12e}");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        DiscreteConnectionTransport,
        coupled_modulator,
    )?;
    println!("discrete,kretschmann,{steps},{radius:.12e},{memory:.12e}");

    // Sanity anchor: the plain entry point must match the first row exactly.
    let plain = scirust_nonlocal_relativity::simulate_nonlocal_worldline_adaptive(
        &background,
        initial,
        config,
    )?;
    let plain_final = plain.final_state().expect("non-empty trajectory");
    println!(
        "plain_entry_point,none,{},{:.12e},{:.12e}",
        plain.len() - 1,
        plain_final.coordinates[1],
        plain
            .final_diagnostics()
            .expect("non-empty trajectory")
            .memory_l2_norm
    );

    Ok(())
}
