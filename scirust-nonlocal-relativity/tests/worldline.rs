use scirust_fractional::FractionalOrder;
use scirust_nonlocal_relativity::{
    NonlocalConfig, NonlocalRelativityError, NonlocalTrajectory, StepDiagnostics, WorldlineState,
    caputo_velocity_memory, coordinate_l2_norm, gr_acceleration, lower_index,
    projected_memory_force, simulate_nonlocal_worldline,
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
fn identical_runs_are_bit_identical() {
    let background = UniformAccelerationBackground;
    let initial = WorldlineState::new([0.0; 4], [1.4, 0.05, -0.02, 0.0]);
    let config = NonlocalConfig::new(0.52, 0.03, 0.025, 40, 1.0e-12).unwrap();

    let first = simulate_nonlocal_worldline(&background, initial, config).unwrap();
    let second = simulate_nonlocal_worldline(&background, initial, config).unwrap();

    assert_bit_identical(&first, &second);
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
