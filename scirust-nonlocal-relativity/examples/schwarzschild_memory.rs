use scirust_nonlocal_relativity::{NonlocalConfig, WorldlineState, simulate_nonlocal_worldline};
use scirust_relativity::Schwarzschild;
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn circular_timelike_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let radius = 10.0;
    let background = Schwarzschild::try_new(mass).expect("positive mass");
    let mut initial = circular_timelike_state(mass, radius);
    initial.velocity[1] = -0.015;

    let step = 0.02;
    let steps = 160;
    let baseline_config = NonlocalConfig::new(0.55, 0.0, step, steps, 1.0e-8)?;
    let coupled_config = NonlocalConfig::new(0.55, 0.015, step, steps, 1.0e-8)?;

    let baseline = simulate_nonlocal_worldline(&background, initial, baseline_config)?;
    let coupled = simulate_nonlocal_worldline(&background, initial, coupled_config)?;

    println!(
        "lambda,baseline_radius,coupled_radius,baseline_metric_drift,coupled_metric_drift,coupled_memory_force_norm,radial_deviation"
    );

    for index in (0..baseline.len()).step_by(16)
    {
        let base_state = baseline.states()[index];
        let coupled_state = coupled.states()[index];
        let base_diag = baseline.diagnostics()[index];
        let coupled_diag = coupled.diagnostics()[index];
        let radial_deviation = coupled_state.coordinates[1] - base_state.coordinates[1];

        println!(
            "{:.6},{:.12},{:.12},{:.12e},{:.12e},{:.12e},{:.12e}",
            coupled_diag.affine_parameter,
            base_state.coordinates[1],
            coupled_state.coordinates[1],
            base_diag.metric_norm_drift,
            coupled_diag.metric_norm_drift,
            coupled_diag.memory_force_l2_norm,
            radial_deviation
        );
    }

    Ok(())
}
