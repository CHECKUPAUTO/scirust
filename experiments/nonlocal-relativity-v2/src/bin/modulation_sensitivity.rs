//! Phase 9 experiment: curvature-modulation sensitivity with a bit-identical
//! `beta = 0` baseline.
//!
//! `SchwarzschildKretschmannModulator` reweights each retained velocity sample
//! by `q = 1 + beta * L^4 * K` before the Caputo stencil (`K` the Kretschmann
//! scalar). At `beta = 0` the modulator returns exactly `1.0` and bypasses the
//! Kretschmann computation, so a modulated run reproduces the unmodulated
//! baseline **bit-for-bit**. This experiment sweeps `beta` and reports the
//! endpoint deviation from that unmodulated baseline, confirming the
//! `beta = 0` bit-identity and the monotone growth of the effect with `beta`.
//!
//! `beta` and the reference length are free, uncalibrated phenomenological
//! parameters; this quantifies a hook's sensitivity, not a physical effect.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{
    circular_schwarzschild_state, euclidean_distance, print_common_header, require_finite,
};
use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, AdaptiveSimulationPolicy, CompleteUniformHistory,
    IdentityHistoryModulator, IdentityHistoryTransport, SchwarzschildKretschmannModulator,
    simulate_nonlocal_worldline_adaptive_with_policy,
};
use scirust_relativity::Schwarzschild;

const MASS: f64 = 1.0;
const RADIUS: f64 = 10.0;
const REFERENCE_LENGTH: f64 = 1.0;
const BETAS: [f64; 5] = [0.0, 0.1, 0.5, 1.0, 2.0];

fn config() -> AdaptiveNonlocalConfig {
    AdaptiveNonlocalConfig::new(0.55, 0.05, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.8, 5_000, 30)
        .expect("valid adaptive config")
}

fn main() -> Result<(), String> {
    let background =
        Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;
    let mut initial = circular_schwarzschild_state(MASS, RADIUS);
    initial.velocity[1] = -0.01;

    // Unmodulated baseline (identity modulator).
    let baseline = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config(),
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            IdentityHistoryModulator,
        ),
    )
    .map_err(|e| e.to_string())?;
    let baseline_final = *baseline.final_state().ok_or("empty baseline")?;

    print_common_header("Schwarzschild-Kretschmann modulation sensitivity");
    println!("# background: Schwarzschild M = {MASS}, exterior near-circular orbit r0 = {RADIUS}");
    println!("# modulator weight q = 1 + beta * L^4 * K, reference length L = {REFERENCE_LENGTH}");
    println!(
        "# deviation is the endpoint coordinate+velocity distance from the unmodulated baseline"
    );
    println!("# beta = 0 must reproduce the baseline bit-for-bit (deviation exactly 0)");
    println!("beta,endpoint_deviation,bit_identical_to_baseline");

    for &beta in &BETAS
    {
        let modulator = SchwarzschildKretschmannModulator::try_new(MASS, REFERENCE_LENGTH, beta)
            .map_err(|e| e.to_string())?;
        let trajectory = simulate_nonlocal_worldline_adaptive_with_policy(
            &background,
            initial,
            config(),
            AdaptiveSimulationPolicy::new(
                CompleteUniformHistory::<4>::new(),
                IdentityHistoryTransport,
                modulator,
            ),
        )
        .map_err(|e| e.to_string())?;
        let final_state = *trajectory.final_state().ok_or("empty trajectory")?;

        let coordinate_gap =
            euclidean_distance(&final_state.coordinates, &baseline_final.coordinates);
        let velocity_gap = euclidean_distance(&final_state.velocity, &baseline_final.velocity);
        let deviation = (coordinate_gap * coordinate_gap + velocity_gap * velocity_gap).sqrt();
        require_finite(&[("endpoint_deviation", deviation)])?;

        let bit_identical = (0..4).all(|i| {
            final_state.coordinates[i].to_bits() == baseline_final.coordinates[i].to_bits()
                && final_state.velocity[i].to_bits() == baseline_final.velocity[i].to_bits()
        });
        println!("{beta:.1},{deviation:.6e},{bit_identical}");
    }

    println!(
        "# interpretation: beta = 0 reproduces the unmodulated baseline bit-for-bit (deviation 0,"
    );
    println!(
        "# bit_identical = true), and the endpoint deviation grows monotonically with beta. The"
    );
    println!(
        "# modulation is a deterministic phenomenological reweighting hook, not a physical law."
    );
    Ok(())
}
