//! Demonstrates the step-doubling adaptive integrator
//! ([`simulate_nonlocal_worldline_adaptive_with_stepper_policy`]), which
//! genuinely composes [`scirust_nonlocal_relativity::MemoryLaw`] and
//! [`scirust_nonlocal_relativity::SemiImplicitEulerStepper`] (a real
//! [`scirust_nonlocal_relativity::WorldlineStepper`] implementation) with
//! geometric transport and curvature modulation, unlike
//! `simulate_nonlocal_worldline_adaptive_with_policy`'s embedded Heun-Euler
//! controller (see `adaptive_stepper.rs`'s module documentation for exactly
//! why that controller cannot reuse either trait).
//!
//! `simulate_nonlocal_worldline_adaptive_with_stepper` (the plain entry
//! point) is the `NonuniformCaputoCoordinateMemory` +
//! `IdentityHistoryTransport` special case of this same function; this
//! example shows that swapping in `NonuniformModulatedCaputoCoordinateMemory`
//! and/or `DiscreteConnectionTransport` changes the numerical result
//! relative to that baseline while every combination remains well-behaved
//! (finite, deterministic, reaching the same target affine parameter).

use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, AdaptiveStepperPolicy, CompleteUniformHistory,
    DiscreteConnectionTransport, HistoryTransport, IdentityHistoryTransport, MemoryLaw,
    NonuniformCaputoCoordinateMemory, NonuniformModulatedCaputoCoordinateMemory,
    SchwarzschildKretschmannModulator, WorldlineState,
    simulate_nonlocal_worldline_adaptive_with_stepper_policy,
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

fn run<L, T>(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    config: AdaptiveNonlocalConfig,
    memory_law: L,
    transport: T,
) -> Result<(usize, f64, f64), Box<dyn Error>>
where
    L: MemoryLaw<4>,
    T: HistoryTransport<4>,
{
    let trajectory = simulate_nonlocal_worldline_adaptive_with_stepper_policy(
        background,
        initial,
        config,
        AdaptiveStepperPolicy::new(CompleteUniformHistory::<4>::new(), memory_law, transport),
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
        0.55, 0.02, 0.02, 0.0001, 0.1, 1.0e-9, 1.0e-8, 1.0, 5_000, 40,
    )?;
    let identity_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.0)?;
    let coupled_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.3)?;

    println!("memory_law,transport,accepted_steps,final_radius,final_memory_l2_norm");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        NonuniformCaputoCoordinateMemory,
        IdentityHistoryTransport,
    )?;
    println!("nonuniform_caputo,identity,{steps},{radius:.12e},{memory:.12e}");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        NonuniformCaputoCoordinateMemory,
        DiscreteConnectionTransport,
    )?;
    println!("nonuniform_caputo,discrete,{steps},{radius:.12e},{memory:.12e}");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        NonuniformModulatedCaputoCoordinateMemory::new(identity_modulator),
        IdentityHistoryTransport,
    )?;
    println!("nonuniform_modulated(beta=0),identity,{steps},{radius:.12e},{memory:.12e}");

    let (steps, radius, memory) = run(
        &background,
        initial,
        config,
        NonuniformModulatedCaputoCoordinateMemory::new(coupled_modulator),
        DiscreteConnectionTransport,
    )?;
    println!("nonuniform_modulated(kretschmann),discrete,{steps},{radius:.12e},{memory:.12e}");

    // Sanity anchor: the plain entry point must match the first row exactly.
    let plain = scirust_nonlocal_relativity::simulate_nonlocal_worldline_adaptive_with_stepper(
        &background,
        initial,
        config,
    )?;
    let plain_final = plain.final_state().expect("non-empty trajectory");
    println!(
        "plain_entry_point,identity,{},{:.12e},{:.12e}",
        plain.len() - 1,
        plain_final.coordinates[1],
        plain
            .final_diagnostics()
            .expect("non-empty trajectory")
            .memory_l2_norm
    );

    Ok(())
}
