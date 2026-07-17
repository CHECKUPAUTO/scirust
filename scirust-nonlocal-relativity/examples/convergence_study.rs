use scirust_nonlocal_relativity::{
    ConvergenceStudy, NonlocalConfig, RefinementEndpoint, WorldlineIntegrator, WorldlineState,
    run_convergence_study, schwarzschild_invariants,
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

fn endpoint_state(endpoint: &RefinementEndpoint<4>) -> WorldlineState<4> {
    WorldlineState::new(endpoint.coordinates, endpoint.velocity)
}

fn format_optional(value: Option<f64>) -> String {
    match value
    {
        Some(value) => format!("{value:.12e}"),
        None => "NA".to_string(),
    }
}

fn print_endpoint(
    background: &Schwarzschild,
    initial: &WorldlineState<4>,
    study: &ConvergenceStudy<4>,
    refinement: &str,
    endpoint: &RefinementEndpoint<4>,
    coordinate_error_to_next: Option<f64>,
    velocity_error_to_next: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let initial_invariants = schwarzschild_invariants(background, initial)?;
    let endpoint_state = endpoint_state(endpoint);
    let endpoint_invariants = schwarzschild_invariants(background, &endpoint_state)?;
    let energy_drift = endpoint_invariants.specific_energy - initial_invariants.specific_energy;
    let angular_momentum_drift = endpoint_invariants.azimuthal_angular_momentum
        - initial_invariants.azimuthal_angular_momentum;

    println!(
        "{},{},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{}",
        study.integrator,
        refinement,
        endpoint.step,
        endpoint.steps,
        endpoint.final_affine_parameter,
        endpoint.coordinates[1],
        endpoint.coordinates[3],
        endpoint.metric_norm_drift,
        endpoint.memory_force_l2_norm,
        energy_drift,
        angular_momentum_drift,
        format_optional(coordinate_error_to_next),
        format_optional(velocity_error_to_next),
        format_optional(study.coordinate_self_convergence_ratio),
        format_optional(study.velocity_self_convergence_ratio),
    );

    Ok(())
}

fn print_study(
    background: &Schwarzschild,
    initial: &WorldlineState<4>,
    study: &ConvergenceStudy<4>,
) -> Result<(), Box<dyn Error>> {
    print_endpoint(
        background,
        initial,
        study,
        "h",
        &study.coarse,
        Some(study.endpoint_coordinate_error_h_h2),
        Some(study.endpoint_velocity_error_h_h2),
    )?;
    print_endpoint(
        background,
        initial,
        study,
        "h/2",
        &study.fine,
        Some(study.endpoint_coordinate_error_h2_h4),
        Some(study.endpoint_velocity_error_h2_h4),
    )?;
    print_endpoint(background, initial, study, "h/4", &study.finest, None, None)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let radius = 10.0;
    let background = Schwarzschild::try_new(mass).expect("positive mass");
    let mut initial = circular_timelike_state(mass, radius);
    initial.velocity[1] = -0.01;

    let config = NonlocalConfig::new(0.55, 0.01, 0.04, 40, 1.0e-8)?;
    let euler = run_convergence_study(
        &background,
        initial,
        config,
        WorldlineIntegrator::SemiImplicitEuler,
    )?;
    let heun = run_convergence_study(&background, initial, config, WorldlineIntegrator::HeunPece)?;

    println!(
        "integrator,refinement,step,steps,lambda,radius,phi,metric_norm_drift,\
         memory_force_norm,specific_energy_drift,azimuthal_angular_momentum_drift,\
         coordinate_error_to_next,velocity_error_to_next,coordinate_ratio,velocity_ratio"
    );
    print_study(&background, &initial, &euler)?;
    print_study(&background, &initial, &heun)?;

    Ok(())
}
