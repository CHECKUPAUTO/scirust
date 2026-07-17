use scirust_fractional::FractionalOrder;
use scirust_nonlocal_relativity::{
    BoundedShortMemoryHistory, CaputoCoordinateMemory, CompleteUniformHistory,
    DefaultNonlocalSimulationPolicy, HeunPeceStepper, HistoryApproximation, HistoryBackend,
    IdentityHistoryTransport, NonlocalConfig, NonlocalRelativityError, NonlocalTrajectory,
    SemiImplicitEulerStepper, StepDiagnostics, WorldlineIntegrator, WorldlineState,
    caputo_velocity_memory, coordinate_l2_norm, gr_acceleration, lower_index,
    projected_memory_force, run_convergence_study, schwarzschild_azimuthal_angular_momentum,
    schwarzschild_invariants, schwarzschild_metric_norm, schwarzschild_specific_energy,
    simulate_nonlocal_worldline, simulate_nonlocal_worldline_with_components,
    simulate_nonlocal_worldline_with_integrator, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::{Connection, Metric, Minkowski, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

#[derive(Debug, Clone, Copy)]
struct UniformAccelerationBackground;

impl Metric<4> for UniformAccelerationBackground {
    fn components(&self, _coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        Minkowski.components(&[0.0; 4])
    }
}

impl Connection<4> for UniformAccelerationBackground {
    fn christoffel(&self, _coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        let mut symbols = [[[0.0_f64; 4]; 4]; 4];
        symbols[1][0][0] = -0.02;
        symbols
    }
}

#[derive(Debug, Clone, Copy)]
struct SmoothVelocityBackground;

impl Metric<4> for SmoothVelocityBackground {
    fn components(&self, _coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        Minkowski.components(&[0.0; 4])
    }
}

impl Connection<4> for SmoothVelocityBackground {
    fn christoffel(&self, _coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        let mut symbols = [[[0.0_f64; 4]; 4]; 4];
        symbols[1][0][1] = -0.018;
        symbols[1][1][0] = -0.018;
        symbols[2][0][2] = 0.011;
        symbols[2][2][0] = 0.011;
        symbols
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;

    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn assert_finite_trajectory(trajectory: &NonlocalTrajectory<4>) {
    for state in trajectory.states()
    {
        for value in state.coordinates
        {
            assert!(value.is_finite(), "non-finite coordinate {value}");
        }

        for value in state.velocity
        {
            assert!(value.is_finite(), "non-finite velocity {value}");
        }
    }

    for diagnostics in trajectory.diagnostics()
    {
        assert!(diagnostics.affine_parameter.is_finite());
        assert!(diagnostics.metric_norm.is_finite());
        assert!(diagnostics.metric_norm_drift.is_finite());
        assert!(diagnostics.memory_l2_norm.is_finite());
        assert!(diagnostics.memory_force_l2_norm.is_finite());
        assert!(diagnostics.orthogonality_residual.is_finite());
        assert!(diagnostics.gr_acceleration_l2_norm.is_finite());
    }
}

fn assert_bit_identical(left: &NonlocalTrajectory<4>, right: &NonlocalTrajectory<4>) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.diagnostics().len(), right.diagnostics().len());

    for (left_state, right_state) in left.states().iter().zip(right.states())
    {
        for (left_value, right_value) in left_state.coordinates.iter().zip(right_state.coordinates)
        {
            assert_eq!(left_value.to_bits(), right_value.to_bits());
        }

        for (left_value, right_value) in left_state.velocity.iter().zip(right_state.velocity)
        {
            assert_eq!(left_value.to_bits(), right_value.to_bits());
        }
    }

    for (left_diagnostics, right_diagnostics) in left.diagnostics().iter().zip(right.diagnostics())
    {
        assert_diagnostics_bit_identical(left_diagnostics, right_diagnostics);
    }
}

fn vector_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut difference = [0.0_f64; 4];

    for component in 0..4
    {
        difference[component] = left[component] - right[component];
    }

    coordinate_l2_norm(&difference)
}

fn exact_uniform_acceleration_state(
    initial: WorldlineState<4>,
    final_affine_parameter: f64,
) -> WorldlineState<4> {
    let background = UniformAccelerationBackground;
    let acceleration = gr_acceleration(
        &background.christoffel(&initial.coordinates),
        &initial.velocity,
    );
    let mut coordinates = [0.0_f64; 4];
    let mut velocity = [0.0_f64; 4];

    for component in 0..4
    {
        velocity[component] =
            initial.velocity[component] + final_affine_parameter * acceleration[component];
        coordinates[component] = initial.coordinates[component]
            + final_affine_parameter * initial.velocity[component]
            + 0.5 * final_affine_parameter * final_affine_parameter * acceleration[component];
    }

    WorldlineState::new(coordinates, velocity)
}

fn assert_diagnostics_bit_identical(left: &StepDiagnostics, right: &StepDiagnostics) {
    assert_eq!(
        left.affine_parameter.to_bits(),
        right.affine_parameter.to_bits()
    );
    assert_eq!(left.metric_norm.to_bits(), right.metric_norm.to_bits());
    assert_eq!(
        left.metric_norm_drift.to_bits(),
        right.metric_norm_drift.to_bits()
    );
    assert_eq!(
        left.memory_l2_norm.to_bits(),
        right.memory_l2_norm.to_bits()
    );
    assert_eq!(
        left.memory_force_l2_norm.to_bits(),
        right.memory_force_l2_norm.to_bits()
    );
    assert_eq!(
        left.orthogonality_residual.to_bits(),
        right.orthogonality_residual.to_bits()
    );
    assert_eq!(
        left.gr_acceleration_l2_norm.to_bits(),
        right.gr_acceleration_l2_norm.to_bits()
    );
}

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

#[test]
fn configuration_rejects_invalid_inputs() {
    assert!(matches!(
        NonlocalConfig::new(0.0, 0.0, 0.1, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidFractionalOrder(0.0))
    ));
    assert!(matches!(
        NonlocalConfig::new(1.0, 0.0, 0.1, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidFractionalOrder(1.0))
    ));
    assert!(matches!(
        NonlocalConfig::new(f64::NAN, 0.0, 0.1, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidFractionalOrder(_))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, -1.0, 0.1, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidCoupling(-1.0))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, f64::INFINITY, 0.1, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidCoupling(_))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, 0.0, 0.0, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidStep(0.0))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, 0.0, -0.1, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidStep(-0.1))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, 0.0, f64::NAN, 1, 1.0e-12),
        Err(NonlocalRelativityError::InvalidStep(_))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, 0.0, 0.1, 0, 1.0e-12),
        Err(NonlocalRelativityError::InvalidStepCount(0))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, 0.0, 0.1, 1, 0.0),
        Err(NonlocalRelativityError::InvalidMetricNormFloor(0.0))
    ));
    assert!(matches!(
        NonlocalConfig::new(0.5, 0.0, 0.1, 1, f64::INFINITY),
        Err(NonlocalRelativityError::InvalidMetricNormFloor(_))
    ));

    let config = NonlocalConfig::new(0.5, 0.0, 0.1, 1, 1.0e-12).unwrap();
    assert!(matches!(
        simulate_nonlocal_worldline(
            &Minkowski,
            WorldlineState::new([f64::NAN, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            config,
        ),
        Err(NonlocalRelativityError::NonFiniteInitialCoordinate { index: 0, .. })
    ));
    assert!(matches!(
        simulate_nonlocal_worldline(
            &Minkowski,
            WorldlineState::new([0.0; 4], [1.0, f64::NAN, 0.0, 0.0]),
            config,
        ),
        Err(NonlocalRelativityError::NonFiniteInitialVelocity { index: 1, .. })
    ));
}

#[test]
fn invalid_integrator_configuration_is_rejected() {
    assert!(matches!(
        WorldlineIntegrator::try_from_name("fractional_adams"),
        Err(NonlocalRelativityError::InvalidIntegratorConfiguration { name })
            if name == "fractional_adams"
    ));
}

#[test]
fn default_simulation_matches_explicit_euler_bit_for_bit() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.25, 0.07, 0.01, 0.0]);
    let config = NonlocalConfig::new(0.47, 0.018, 0.025, 24, 1.0e-12).unwrap();

    let default = simulate_nonlocal_worldline(&background, initial, config).unwrap();
    let explicit = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::SemiImplicitEuler,
    )
    .unwrap();

    assert_bit_identical(&default, &explicit);
}

#[test]
fn default_advanced_policy_matches_compatibility_api_bit_for_bit() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.25, 0.07, 0.01, 0.0]);
    let config = NonlocalConfig::new(0.47, 0.018, 0.025, 24, 1.0e-12).unwrap();

    let compatibility = simulate_nonlocal_worldline(&background, initial, config).unwrap();
    let advanced = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        DefaultNonlocalSimulationPolicy::<4>::default(),
    )
    .unwrap();

    assert_bit_identical(&compatibility, &advanced);
    assert_eq!(advanced.history_diagnostics().len(), advanced.len());
    assert!(advanced.history_diagnostics().iter().all(|diagnostics| {
        diagnostics.approximation == HistoryApproximation::Exact
            && diagnostics.retained_samples == diagnostics.used_samples
    }));
}

#[test]
fn explicit_exact_backend_matches_public_integrator_api_bit_for_bit() {
    let background = SmoothVelocityBackground;
    let initial = WorldlineState::new([0.0; 4], [1.35, 0.08, -0.04, 0.0]);
    let config = NonlocalConfig::new(0.52, 0.025, 0.02, 32, 1.0e-12).unwrap();

    let public = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::HeunPece,
    )
    .unwrap();
    let exact_architecture = simulate_nonlocal_worldline_with_components(
        &background,
        initial,
        config,
        CompleteUniformHistory::<4>::new(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        HeunPeceStepper,
    )
    .unwrap();

    assert_bit_identical(&public, &exact_architecture);
    assert!(
        exact_architecture
            .history_diagnostics()
            .iter()
            .all(|diagnostics| diagnostics.approximation == HistoryApproximation::Exact)
    );
}

#[test]
fn both_steppers_operate_through_advanced_architecture() {
    let background = SmoothVelocityBackground;
    let initial = WorldlineState::new([0.0; 4], [1.3, 0.05, -0.03, 0.0]);
    let config = NonlocalConfig::new(0.58, 0.015, 0.04, 20, 1.0e-12).unwrap();

    let euler = simulate_nonlocal_worldline_with_components(
        &background,
        initial,
        config,
        CompleteUniformHistory::<4>::new(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    )
    .unwrap();
    let heun = simulate_nonlocal_worldline_with_components(
        &background,
        initial,
        config,
        CompleteUniformHistory::<4>::new(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        HeunPeceStepper,
    )
    .unwrap();

    assert_finite_trajectory(&euler);
    assert_finite_trajectory(&heun);
    assert_eq!(euler.len(), config.steps() + 1);
    assert_eq!(heun.len(), config.steps() + 1);
}

#[test]
fn minkowski_zero_coupling_keeps_constant_velocity_and_linear_coordinates() {
    let initial = WorldlineState::new([1.0, -2.0, 3.0, -4.0], [2.0, 0.25, -0.5, 0.75]);
    let step = 0.125;
    let steps = 32;
    let config = NonlocalConfig::new(0.5, 0.0, step, steps, 1.0e-12).unwrap();

    let trajectory = simulate_nonlocal_worldline(&Minkowski, initial, config).unwrap();

    assert_eq!(trajectory.len(), steps + 1);
    assert_eq!(trajectory.diagnostics().len(), steps + 1);

    for (index, state) in trajectory.states().iter().enumerate()
    {
        let lambda = index as f64 * step;

        for component in 0..4
        {
            assert_eq!(
                state.velocity[component].to_bits(),
                initial.velocity[component].to_bits()
            );
            assert_close(
                state.coordinates[component],
                initial.coordinates[component] + lambda * initial.velocity[component],
                1.0e-14,
            );
        }
    }

    for diagnostics in trajectory.diagnostics()
    {
        assert_eq!(
            diagnostics.memory_force_l2_norm.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(diagnostics.memory_l2_norm.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            diagnostics.gr_acceleration_l2_norm.to_bits(),
            0.0_f64.to_bits()
        );
    }
}

#[test]
fn heun_pece_runs_are_bit_identical() {
    let background = SmoothVelocityBackground;
    let initial = WorldlineState::new([0.0; 4], [1.35, 0.08, -0.04, 0.0]);
    let config = NonlocalConfig::new(0.52, 0.025, 0.02, 48, 1.0e-12).unwrap();

    let first = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::HeunPece,
    )
    .unwrap();
    let second = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::HeunPece,
    )
    .unwrap();

    assert_bit_identical(&first, &second);
}

#[test]
fn heun_pece_has_lower_endpoint_error_than_euler_on_smooth_oracle() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.2, 0.1, 0.0, 0.0]);
    let step = 0.08;
    let steps = 20;
    let config = NonlocalConfig::new(0.5, 0.0, step, steps, 1.0e-12).unwrap();
    let expected = exact_uniform_acceleration_state(initial, step * steps as f64);

    let euler = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::SemiImplicitEuler,
    )
    .unwrap();
    let heun = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::HeunPece,
    )
    .unwrap();

    let euler_coordinate_error = vector_distance(
        &euler.final_state().unwrap().coordinates,
        &expected.coordinates,
    );
    let heun_coordinate_error = vector_distance(
        &heun.final_state().unwrap().coordinates,
        &expected.coordinates,
    );

    assert!(
        heun_coordinate_error < euler_coordinate_error,
        "heun error {heun_coordinate_error:.17e}, euler error {euler_coordinate_error:.17e}"
    );
}

#[test]
fn refinement_study_endpoint_errors_decrease() {
    let background = SmoothVelocityBackground;
    let initial = WorldlineState::new([0.0; 4], [1.3, 0.05, -0.03, 0.0]);
    let config = NonlocalConfig::new(0.58, 0.015, 0.04, 32, 1.0e-12).unwrap();

    let study =
        run_convergence_study(&background, initial, config, WorldlineIntegrator::HeunPece).unwrap();

    assert!(
        study.endpoint_coordinate_error_h_h2 > study.endpoint_coordinate_error_h2_h4,
        "coordinate errors did not decrease: h/h2={:.17e}, h2/h4={:.17e}",
        study.endpoint_coordinate_error_h_h2,
        study.endpoint_coordinate_error_h2_h4
    );
    assert!(
        study.endpoint_velocity_error_h_h2 > study.endpoint_velocity_error_h2_h4,
        "velocity errors did not decrease: h/h2={:.17e}, h2/h4={:.17e}",
        study.endpoint_velocity_error_h_h2,
        study.endpoint_velocity_error_h2_h4
    );
    assert!(study.coordinate_self_convergence_ratio.unwrap() > 1.0);
    assert!(study.velocity_self_convergence_ratio.unwrap() > 1.0);
}

#[test]
fn constant_velocity_history_has_zero_caputo_memory() {
    let order = FractionalOrder::new(0.37).unwrap();
    let history = vec![[1.25, -0.5, 0.75, 2.0]; 96];

    let memory = caputo_velocity_memory(&history, 0.05, order).unwrap();

    assert_eq!(coordinate_l2_norm(&memory).to_bits(), 0.0_f64.to_bits());
    for value in memory
    {
        assert_eq!(value.to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn short_memory_constant_history_has_exact_zero_memory_force() {
    let initial = WorldlineState::new([1.0, -2.0, 3.0, -4.0], [2.0, 0.25, -0.5, 0.75]);
    let config = NonlocalConfig::new(0.5, 0.3, 0.125, 16, 1.0e-12).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_components(
        &Minkowski,
        initial,
        config,
        BoundedShortMemoryHistory::<4>::new(4).unwrap(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    )
    .unwrap();

    for diagnostics in trajectory.diagnostics()
    {
        assert_eq!(diagnostics.memory_l2_norm.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            diagnostics.memory_force_l2_norm.to_bits(),
            0.0_f64.to_bits()
        );
    }
    assert!(trajectory.history_diagnostics().iter().all(|diagnostics| {
        diagnostics.approximation == HistoryApproximation::Approximate
            && diagnostics.retained_samples == diagnostics.used_samples
            && diagnostics.retained_samples <= 4
    }));
}

#[test]
fn short_memory_deviation_is_finite_measurable_and_bounded() {
    let background = SmoothVelocityBackground;
    let initial = WorldlineState::new([0.0; 4], [1.28, 0.06, -0.02, 0.0]);
    let config = NonlocalConfig::new(0.54, 0.03, 0.025, 72, 1.0e-12).unwrap();

    let exact = simulate_nonlocal_worldline_with_components(
        &background,
        initial,
        config,
        CompleteUniformHistory::<4>::new(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    )
    .unwrap();
    let short = simulate_nonlocal_worldline_with_components(
        &background,
        initial,
        config,
        BoundedShortMemoryHistory::<4>::new(5).unwrap(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    )
    .unwrap();

    assert_finite_trajectory(&short);
    let coordinate_deviation = vector_distance(
        &exact.final_state().unwrap().coordinates,
        &short.final_state().unwrap().coordinates,
    );
    let velocity_deviation = vector_distance(
        &exact.final_state().unwrap().velocity,
        &short.final_state().unwrap().velocity,
    );

    assert!(
        coordinate_deviation > 1.0e-10,
        "coordinate deviation was too small: {coordinate_deviation:.17e}"
    );
    assert!(
        coordinate_deviation < 0.02,
        "coordinate deviation was unexpectedly large: {coordinate_deviation:.17e}"
    );
    assert!(
        velocity_deviation > 1.0e-10,
        "velocity deviation was too small: {velocity_deviation:.17e}"
    );
    assert!(
        velocity_deviation < 0.02,
        "velocity deviation was unexpectedly large: {velocity_deviation:.17e}"
    );
}

#[test]
fn short_memory_history_window_accounting_is_correct() {
    let initial = WorldlineState::new([0.0; 4], [2.0, 0.25, -0.5, 0.75]);
    let config = NonlocalConfig::new(0.5, 0.0, 0.1, 5, 1.0e-12).unwrap();
    let trajectory = simulate_nonlocal_worldline_with_components(
        &Minkowski,
        initial,
        config,
        BoundedShortMemoryHistory::<4>::new(3).unwrap(),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    )
    .unwrap();

    let retained: Vec<usize> = trajectory
        .history_diagnostics()
        .iter()
        .map(|diagnostics| diagnostics.retained_samples)
        .collect();
    let used: Vec<usize> = trajectory
        .history_diagnostics()
        .iter()
        .map(|diagnostics| diagnostics.used_samples)
        .collect();

    assert_eq!(retained, vec![1, 2, 3, 3, 3, 3]);
    assert_eq!(used, retained);
    assert!(
        trajectory
            .history_diagnostics()
            .iter()
            .all(|diagnostics| { diagnostics.approximation == HistoryApproximation::Approximate })
    );
}

#[test]
fn invalid_short_memory_windows_are_rejected() {
    assert!(matches!(
        BoundedShortMemoryHistory::<4>::new(0),
        Err(NonlocalRelativityError::InvalidHistoryWindow { window_samples: 0 })
    ));
    assert!(matches!(
        BoundedShortMemoryHistory::<4>::new(1),
        Err(NonlocalRelativityError::InvalidHistoryWindow { window_samples: 1 })
    ));

    let backend = BoundedShortMemoryHistory::<4>::new(2).unwrap();
    assert_eq!(backend.window_samples(), 2);
}

#[test]
fn history_backend_samples_are_returned_by_value() {
    let mut backend = BoundedShortMemoryHistory::<4>::new(2).unwrap();
    backend.push_velocity([1.0, 2.0, 3.0, 4.0]).unwrap();

    let mut copied_sample = backend.sample(0).unwrap();
    copied_sample[0] = 99.0;

    assert_eq!(copied_sample, [99.0, 2.0, 3.0, 4.0]);
    assert_eq!(backend.sample(0).unwrap(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(backend.retained_samples(), 1);
}

#[test]
fn projected_memory_force_is_orthogonal_to_timelike_velocity() {
    let metric = Minkowski.components(&[0.0; 4]);
    let velocity = [2.0, 0.3, -0.4, 0.2];
    let memory = [0.15, -0.5, 0.25, 0.4];
    let lowered = lower_index(&metric, &velocity);
    let metric_norm = -velocity[0] * velocity[0]
        + velocity[1] * velocity[1]
        + velocity[2] * velocity[2]
        + velocity[3] * velocity[3];

    let force = projected_memory_force(&velocity, &lowered, metric_norm, &memory, 0.7);
    let residual = lowered
        .iter()
        .zip(force)
        .fold(0.0, |sum, (lowered_component, force_component)| {
            sum + *lowered_component * force_component
        });

    assert!(
        residual.abs() <= 2.0e-16,
        "orthogonality residual was {residual:.17e}"
    );
}

#[test]
fn zero_coupling_follows_ordinary_geodesic_acceleration_path() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.2, 0.1, 0.0, 0.0]);
    let step = 0.05;
    let steps = 12;
    let config = NonlocalConfig::new(0.5, 0.0, step, steps, 1.0e-12).unwrap();

    let trajectory = simulate_nonlocal_worldline(&background, initial, config).unwrap();
    let mut expected = initial;

    for step_index in 0..steps
    {
        let symbols = background.christoffel(&expected.coordinates);
        let acceleration = gr_acceleration(&symbols, &expected.velocity);
        let mut next_velocity = [0.0_f64; 4];
        let mut next_coordinates = [0.0_f64; 4];

        for component in 0..4
        {
            next_velocity[component] =
                expected.velocity[component] + step * acceleration[component];
            next_coordinates[component] =
                expected.coordinates[component] + step * next_velocity[component];
        }

        expected = WorldlineState::new(next_coordinates, next_velocity);
        let actual = trajectory.states()[step_index + 1];

        for component in 0..4
        {
            assert_eq!(
                actual.velocity[component].to_bits(),
                expected.velocity[component].to_bits()
            );
            assert_eq!(
                actual.coordinates[component].to_bits(),
                expected.coordinates[component].to_bits()
            );
        }
    }

    for diagnostics in trajectory.diagnostics()
    {
        assert_eq!(
            diagnostics.memory_force_l2_norm.to_bits(),
            0.0_f64.to_bits()
        );
    }
}

#[test]
fn heun_zero_coupling_remains_geodesic_baseline() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.2, 0.1, 0.0, 0.0]);
    let step = 0.05;
    let steps = 16;
    let config = NonlocalConfig::new(0.5, 0.0, step, steps, 1.0e-12).unwrap();
    let expected = exact_uniform_acceleration_state(initial, step * steps as f64);

    let trajectory = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::HeunPece,
    )
    .unwrap();

    let final_state = trajectory.final_state().unwrap();
    let coordinate_error = vector_distance(&final_state.coordinates, &expected.coordinates);
    let velocity_error = vector_distance(&final_state.velocity, &expected.velocity);
    assert!(
        coordinate_error <= 1.0e-14,
        "coordinate error {coordinate_error:.17e}"
    );
    assert!(
        velocity_error <= 1.0e-14,
        "velocity error {velocity_error:.17e}"
    );

    for diagnostics in trajectory.diagnostics()
    {
        assert_eq!(
            diagnostics.memory_force_l2_norm.to_bits(),
            0.0_f64.to_bits()
        );
    }
}

#[test]
fn identical_runs_are_bit_identical() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.4, 0.05, -0.02, 0.0]);
    let config = NonlocalConfig::new(0.52, 0.03, 0.025, 40, 1.0e-12).unwrap();

    let first = simulate_nonlocal_worldline(&background, initial, config).unwrap();
    let second = simulate_nonlocal_worldline(&background, initial, config).unwrap();

    assert_bit_identical(&first, &second);
}

#[test]
fn heun_values_remain_finite_for_alpha_near_supported_bounds() {
    let background = SmoothVelocityBackground;
    let initial = WorldlineState::new([0.0; 4], [1.25, 0.04, -0.02, 0.0]);

    for alpha in [1.0e-6, 1.0 - 1.0e-6]
    {
        let config = NonlocalConfig::new(alpha, 0.01, 0.015, 24, 1.0e-12).unwrap();
        let trajectory = simulate_nonlocal_worldline_with_integrator(
            &background,
            initial,
            config,
            WorldlineIntegrator::HeunPece,
        )
        .unwrap();

        assert_finite_trajectory(&trajectory);
    }
}

#[test]
fn null_or_nearly_null_initial_velocity_is_rejected() {
    let config = NonlocalConfig::new(0.5, 0.0, 0.1, 1, 1.0e-8).unwrap();

    let null = simulate_nonlocal_worldline(
        &Minkowski,
        WorldlineState::new([0.0; 4], [1.0, 1.0, 0.0, 0.0]),
        config,
    );
    assert!(matches!(
        null,
        Err(NonlocalRelativityError::MetricNormBelowFloor { step: 0, .. })
    ));

    let nearly_null = simulate_nonlocal_worldline(
        &Minkowski,
        WorldlineState::new([0.0; 4], [1.0, 1.0 + 1.0e-12, 0.0, 0.0]),
        config,
    );
    assert!(matches!(
        nearly_null,
        Err(NonlocalRelativityError::MetricNormBelowFloor { step: 0, .. })
    ));
}

#[test]
fn schwarzschild_smoke_test_remains_finite_exterior_and_deterministic() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let initial = circular_schwarzschild_state(1.0, 10.0);
    let config = NonlocalConfig::new(0.55, 0.01, 0.01, 32, 1.0e-8).unwrap();

    let first = simulate_nonlocal_worldline(&background, initial, config).unwrap();
    let second = simulate_nonlocal_worldline(&background, initial, config).unwrap();

    assert_finite_trajectory(&first);
    assert_bit_identical(&first, &second);

    for state in first.states()
    {
        assert!(
            background.is_in_exterior(&state.coordinates),
            "left Schwarzschild exterior at coordinates {:?}",
            state.coordinates
        );
    }
}

#[test]
fn schwarzschild_energy_and_angular_momentum_drift_are_measured() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    let config = NonlocalConfig::new(0.55, 0.008, 0.01, 48, 1.0e-8).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_integrator(
        &background,
        initial,
        config,
        WorldlineIntegrator::HeunPece,
    )
    .unwrap();
    let final_state = *trajectory.final_state().unwrap();
    let initial_invariants = schwarzschild_invariants(&background, &initial).unwrap();
    let final_invariants = schwarzschild_invariants(&background, &final_state).unwrap();
    let energy_drift = final_invariants.specific_energy - initial_invariants.specific_energy;
    let angular_momentum_drift =
        final_invariants.azimuthal_angular_momentum - initial_invariants.azimuthal_angular_momentum;

    assert!(energy_drift.is_finite());
    assert!(angular_momentum_drift.is_finite());
    assert_close(
        schwarzschild_specific_energy(&background, &final_state).unwrap(),
        final_invariants.specific_energy,
        1.0e-15,
    );
    assert_close(
        schwarzschild_azimuthal_angular_momentum(&background, &final_state).unwrap(),
        final_invariants.azimuthal_angular_momentum,
        1.0e-15,
    );
    assert_close(
        schwarzschild_metric_norm(&background, &final_state).unwrap(),
        trajectory.final_diagnostics().unwrap().metric_norm,
        1.0e-15,
    );
}

#[test]
fn small_coupling_has_finite_bounded_measurable_deviation() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.3, 0.05, 0.0, 0.0]);
    let baseline_config = NonlocalConfig::new(0.5, 0.0, 0.04, 50, 1.0e-12).unwrap();
    let repeated_baseline_config = NonlocalConfig::new(0.5, 0.0, 0.04, 50, 1.0e-12).unwrap();
    let coupled_config = NonlocalConfig::new(0.5, 0.025, 0.04, 50, 1.0e-12).unwrap();

    let baseline = simulate_nonlocal_worldline(&background, initial, baseline_config).unwrap();
    let repeated_baseline =
        simulate_nonlocal_worldline(&background, initial, repeated_baseline_config).unwrap();
    let coupled = simulate_nonlocal_worldline(&background, initial, coupled_config).unwrap();

    assert_bit_identical(&baseline, &repeated_baseline);
    assert_finite_trajectory(&coupled);

    let radial_deviation = coupled.final_state().unwrap().coordinates[1]
        - baseline.final_state().unwrap().coordinates[1];
    let max_force = coupled
        .diagnostics()
        .iter()
        .map(|diagnostics| diagnostics.memory_force_l2_norm)
        .fold(0.0_f64, f64::max);

    assert!(
        radial_deviation.abs() > 1.0e-9,
        "deviation was too small: {radial_deviation:.17e}"
    );
    assert!(
        radial_deviation.abs() < 0.1,
        "deviation was unexpectedly large: {radial_deviation:.17e}"
    );
    assert!(max_force > 0.0);
    assert!(max_force < 0.1);
}
