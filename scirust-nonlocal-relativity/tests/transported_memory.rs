use scirust_nonlocal_relativity::{
    BoundedShortMemoryHistory, CaputoCoordinateMemory, CompleteUniformHistory,
    CylindricalMinkowski, DiscreteConnectionTransport, HeunPeceStepper, HistoryApproximation,
    HistoryBackend, HistoryEntry, IdentityHistoryTransport, NonlocalConfig,
    NonlocalRelativityError, NonlocalSimulationPolicy, ParameterizationMode,
    SemiImplicitEulerStepper, WorldlineState, cartesian_to_cylindrical_coordinates,
    cartesian_to_cylindrical_velocity, coordinate_l2_norm, cylindrical_to_cartesian_coordinates,
    cylindrical_to_cartesian_velocity, simulate_nonlocal_worldline_with_mode,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::{Connection, Metric, Minkowski};

/// A Euclidean-signature `(+,+,+,+)` background, used only to exercise the
/// proper-time mode's signature rejection. It is not part of the crate's
/// public API and does not represent a physical spacetime.
#[derive(Debug, Clone, Copy)]
struct EuclideanBackground;

impl Metric<4> for EuclideanBackground {
    fn components(&self, _coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl Connection<4> for EuclideanBackground {
    fn christoffel(&self, _coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        [[[0.0; 4]; 4]; 4]
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

fn cylindrical_experiment_initial() -> (WorldlineState<4>, WorldlineState<4>) {
    let cartesian = WorldlineState::new([0.0, 3.0, 4.0, 0.0], [1.2, 0.3, -0.2, 0.1]);
    let cylindrical_coordinates =
        cartesian_to_cylindrical_coordinates(cartesian.coordinates).unwrap();
    let cylindrical_velocity =
        cartesian_to_cylindrical_velocity(cartesian.coordinates, cartesian.velocity).unwrap();
    (
        cartesian,
        WorldlineState::new(cylindrical_coordinates, cylindrical_velocity),
    )
}

/// Run the coordinate-covariance controlled experiment at a given
/// refinement factor, returning `(position_disagreement, velocity_disagreement)`
/// for the raw coordinate-memory pipeline and the transported pipeline,
/// mirroring `examples/coordinate_covariance.rs`.
fn chart_disagreement_at(factor: usize) -> ((f64, f64), (f64, f64)) {
    let (cartesian_initial, cylindrical_initial) = cylindrical_experiment_initial();
    let base_step = 0.02;
    let base_steps = 40;
    let step = base_step / factor as f64;
    let steps = base_steps * factor;
    let config = NonlocalConfig::new(0.5, 0.4, step, steps, 1.0e-8).unwrap();

    let cartesian_final = *simulate_nonlocal_worldline_with_policy(
        &Minkowski,
        cartesian_initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap()
    .final_state()
    .unwrap();

    let coordinate_trajectory = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();
    let transported_trajectory = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let disagreement_of = |trajectory: &scirust_nonlocal_relativity::NonlocalTrajectory<4>| {
        let final_state = *trajectory.final_state().unwrap();
        let coordinates = cylindrical_to_cartesian_coordinates(final_state.coordinates).unwrap();
        let velocity =
            cylindrical_to_cartesian_velocity(final_state.coordinates, final_state.velocity)
                .unwrap();
        (
            vector_distance(&cartesian_final.coordinates, &coordinates),
            vector_distance(&cartesian_final.velocity, &velocity),
        )
    };

    (
        disagreement_of(&coordinate_trajectory),
        disagreement_of(&transported_trajectory),
    )
}

#[test]
fn identity_transport_reproduces_coordinate_memory_baseline_bit_for_bit() {
    let background = CylindricalMinkowski;
    let transport = IdentityHistoryTransport;
    let velocities = [
        [1.20, 0.10, 0.05, -0.02],
        [1.19, 0.11, 0.04, -0.01],
        [1.21, 0.09, 0.06, -0.03],
        [1.20, 0.10, 0.05, -0.02],
    ];
    let coordinates = [
        [0.00, 5.000, 0.9000, 0.0000],
        [0.02, 5.002, 0.9010, -0.0002],
        [0.04, 5.0035, 0.9025, -0.0004],
        [0.06, 5.006, 0.9040, -0.0006],
    ];

    let mut legacy = CompleteUniformHistory::<4>::new();
    let mut via_entry = CompleteUniformHistory::<4>::new();

    for (index, velocity) in velocities.iter().enumerate()
    {
        legacy.push_velocity(*velocity).unwrap();
        via_entry
            .push_entry(
                &background,
                &transport,
                HistoryEntry::new(coordinates[index], *velocity, index as f64 * 0.02),
            )
            .unwrap();
    }

    assert_eq!(legacy.retained_samples(), via_entry.retained_samples());

    for index in 0..velocities.len()
    {
        let legacy_sample = legacy.sample(index).unwrap();
        let entry_sample = via_entry.sample(index).unwrap();

        for component in 0..4
        {
            assert_eq!(
                legacy_sample[component].to_bits(),
                entry_sample[component].to_bits()
            );
        }
    }
}

#[test]
fn discrete_connection_transport_is_deterministic_bit_for_bit() {
    let (_, cylindrical_initial) = cylindrical_experiment_initial();
    let config = NonlocalConfig::new(0.5, 0.4, 0.02, 24, 1.0e-8).unwrap();

    let policy = || {
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(25),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            HeunPeceStepper,
        )
    };

    let first = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        policy(),
    )
    .unwrap();
    let second = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        policy(),
    )
    .unwrap();

    assert_eq!(first.len(), second.len());

    for (left, right) in first.states().iter().zip(second.states())
    {
        for component in 0..4
        {
            assert_eq!(
                left.coordinates[component].to_bits(),
                right.coordinates[component].to_bits()
            );
            assert_eq!(
                left.velocity[component].to_bits(),
                right.velocity[component].to_bits()
            );
        }
    }

    for (left, right) in first.diagnostics().iter().zip(second.diagnostics())
    {
        assert_eq!(
            left.orthogonality_residual.to_bits(),
            right.orthogonality_residual.to_bits()
        );
        assert_eq!(
            left.metric_norm_drift.to_bits(),
            right.metric_norm_drift.to_bits()
        );
    }
}

#[test]
fn transported_vectors_remain_finite() {
    let background = CylindricalMinkowski;
    let transport = DiscreteConnectionTransport;
    let mut backend = CompleteUniformHistory::<4>::new();
    let mut coordinates = [0.0_f64, 5.0, 0.7, 0.0];
    let mut velocity = [1.2_f64, 0.15, 0.08, -0.05];

    for step_index in 0..12
    {
        backend
            .push_entry(
                &background,
                &transport,
                HistoryEntry::new(coordinates, velocity, step_index as f64 * 0.03),
            )
            .unwrap();

        for retained_index in 0..backend.retained_samples()
        {
            let sample = backend.sample(retained_index).unwrap();
            assert!(
                sample.iter().all(|value| value.is_finite()),
                "non-finite transported sample at retained index {retained_index}: {sample:?}"
            );
            let entry = backend.entry(retained_index).unwrap();
            assert!(entry.velocity.iter().all(|value| value.is_finite()));
            assert!(entry.coordinates.iter().all(|value| value.is_finite()));
        }

        coordinates[1] += 0.01;
        coordinates[2] += 0.02;
        velocity[1] *= 1.01;
    }
}

#[test]
fn zero_connection_leaves_transported_vector_unchanged() {
    let background = Minkowski;
    let transport = DiscreteConnectionTransport;
    let mut backend = CompleteUniformHistory::<4>::new();
    let first_velocity = [1.3_f64, 0.2, -0.1, 0.05];
    let second_velocity = [1.25_f64, 0.18, -0.08, 0.04];

    backend
        .push_entry(
            &background,
            &transport,
            HistoryEntry::new([0.0, 1.0, 2.0, 3.0], first_velocity, 0.0),
        )
        .unwrap();
    backend
        .push_entry(
            &background,
            &transport,
            HistoryEntry::new([0.05, 1.02, 1.98, 3.01], second_velocity, 0.05),
        )
        .unwrap();

    let transported_first = backend.sample(0).unwrap();

    for component in 0..4
    {
        assert_eq!(
            transported_first[component].to_bits(),
            first_velocity[component].to_bits(),
            "zero connection must leave transported vector bit-identical"
        );
    }
}

#[test]
fn chart_disagreement_decreases_under_refinement() {
    let (_, transported_h) = chart_disagreement_at(1);
    let (_, transported_h2) = chart_disagreement_at(2);
    let (_, transported_h4) = chart_disagreement_at(4);

    assert!(
        transported_h2.0 < transported_h.0,
        "position disagreement did not shrink: h={:.6e}, h/2={:.6e}",
        transported_h.0,
        transported_h2.0
    );
    assert!(
        transported_h4.0 < transported_h2.0,
        "position disagreement did not shrink further: h/2={:.6e}, h/4={:.6e}",
        transported_h2.0,
        transported_h4.0
    );
    assert!(
        transported_h2.1 < transported_h.1,
        "velocity disagreement did not shrink: h={:.6e}, h/2={:.6e}",
        transported_h.1,
        transported_h2.1
    );
    assert!(
        transported_h4.1 < transported_h2.1,
        "velocity disagreement did not shrink further: h/2={:.6e}, h/4={:.6e}",
        transported_h2.1,
        transported_h4.1
    );
}

#[test]
fn transported_memory_reduces_disagreement_versus_coordinate_memory() {
    for factor in [1usize, 2, 4]
    {
        let (coordinate, transported) = chart_disagreement_at(factor);

        assert!(
            transported.0 < coordinate.0,
            "factor {factor}: transported position disagreement {:.6e} was not below \
             coordinate disagreement {:.6e}",
            transported.0,
            coordinate.0
        );
        assert!(
            transported.1 < coordinate.1,
            "factor {factor}: transported velocity disagreement {:.6e} was not below \
             coordinate disagreement {:.6e}",
            transported.1,
            coordinate.1
        );
    }
}

#[test]
fn proper_time_mode_accepts_normalized_timelike_minkowski_trajectory() {
    let spatial = [0.2_f64, 0.1, 0.05];
    let spatial_squared: f64 = spatial.iter().map(|value| value * value).sum();
    let time_component = (1.0 + spatial_squared).sqrt();
    let initial = WorldlineState::new(
        [0.0, 0.0, 0.0, 0.0],
        [time_component, spatial[0], spatial[1], spatial[2]],
    );
    let config = NonlocalConfig::new(0.5, 0.0, 0.05, 16, 1.0e-12).unwrap();
    let policy = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(17),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    );

    let result = simulate_nonlocal_worldline_with_mode(
        &Minkowski,
        initial,
        config,
        policy,
        ParameterizationMode::NormalizedTimelikeProperTime { tolerance: 1.0e-6 },
    );

    assert!(result.is_ok(), "expected acceptance, got {result:?}");
}

#[test]
fn proper_time_mode_rejects_wrong_signature() {
    let initial = WorldlineState::new([0.0; 4], [1.0, 0.0, 0.0, 0.0]);
    let config = NonlocalConfig::new(0.5, 0.0, 0.05, 4, 1.0e-12).unwrap();
    let policy = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(5),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    );

    let result = simulate_nonlocal_worldline_with_mode(
        &EuclideanBackground,
        initial,
        config,
        policy,
        ParameterizationMode::NormalizedTimelikeProperTime { tolerance: 1.0e-6 },
    );

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::ProperTimeNormDrift { step: 0, .. })
    ));
}

#[test]
fn proper_time_mode_rejects_null_state() {
    let initial = WorldlineState::new([0.0; 4], [1.0, 1.0, 0.0, 0.0]);
    let config = NonlocalConfig::new(0.5, 0.0, 0.05, 4, 1.0e-12).unwrap();
    let policy = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(5),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    );

    let result = simulate_nonlocal_worldline_with_mode(
        &Minkowski,
        initial,
        config,
        policy,
        ParameterizationMode::NormalizedTimelikeProperTime { tolerance: 1.0e-6 },
    );

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::ProperTimeNormDrift { step: 0, .. })
    ));
}

#[test]
fn proper_time_mode_rejects_spacelike_state() {
    let initial = WorldlineState::new([0.0; 4], [0.1, 1.0, 0.0, 0.0]);
    let config = NonlocalConfig::new(0.5, 0.0, 0.05, 4, 1.0e-12).unwrap();
    let policy = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(5),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    );

    let result = simulate_nonlocal_worldline_with_mode(
        &Minkowski,
        initial,
        config,
        policy,
        ParameterizationMode::NormalizedTimelikeProperTime { tolerance: 1.0e-6 },
    );

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::ProperTimeNormDrift { step: 0, .. })
    ));
}

#[test]
fn proper_time_mode_rejects_excessive_norm_drift() {
    let mass = 1.0_f64;
    let background = scirust_relativity::Schwarzschild::try_new(mass).unwrap();
    let radius = 6.0_f64;
    let lapse = 1.0 - 2.0 * mass / radius;
    let u_r = 0.15_f64;
    let u_phi = 0.05_f64;
    let spatial_term = u_r * u_r / lapse + radius * radius * u_phi * u_phi;
    let u_t = ((1.0 + spatial_term) / lapse).sqrt();
    let initial = WorldlineState::new(
        [0.0, radius, std::f64::consts::FRAC_PI_2, 0.0],
        [u_t, u_r, 0.0, u_phi],
    );
    let config = NonlocalConfig::new(0.5, 0.1, 0.05, 200, 1.0e-8).unwrap();
    let policy = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(201),
        CaputoCoordinateMemory,
        IdentityHistoryTransport,
        SemiImplicitEulerStepper,
    );

    let result = simulate_nonlocal_worldline_with_mode(
        &background,
        initial,
        config,
        policy,
        ParameterizationMode::NormalizedTimelikeProperTime { tolerance: 1.0e-4 },
    );

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::ProperTimeNormDrift { .. })
    ));
    if let Err(NonlocalRelativityError::ProperTimeNormDrift { step, .. }) = result
    {
        assert!(
            step > 0,
            "drift should be detected after the initial sample, not at it"
        );
    }
}

#[test]
fn affine_mode_remains_compatible_with_existing_api() {
    let (_, cylindrical_initial) = cylindrical_experiment_initial();
    let config = NonlocalConfig::new(0.5, 0.02, 0.02, 20, 1.0e-8).unwrap();

    let policy_direct = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(21),
        CaputoCoordinateMemory,
        DiscreteConnectionTransport,
        HeunPeceStepper,
    );
    let direct = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        policy_direct,
    )
    .unwrap();

    let policy_via_mode = NonlocalSimulationPolicy::new(
        CompleteUniformHistory::<4>::with_capacity(21),
        CaputoCoordinateMemory,
        DiscreteConnectionTransport,
        HeunPeceStepper,
    );
    let via_mode = simulate_nonlocal_worldline_with_mode(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        policy_via_mode,
        ParameterizationMode::AffineParameter,
    )
    .unwrap();

    assert_eq!(direct.len(), via_mode.len());

    for (left, right) in direct.states().iter().zip(via_mode.states())
    {
        for component in 0..4
        {
            assert_eq!(
                left.coordinates[component].to_bits(),
                right.coordinates[component].to_bits()
            );
            assert_eq!(
                left.velocity[component].to_bits(),
                right.velocity[component].to_bits()
            );
        }
    }
}

#[test]
fn orthogonality_residual_stays_controlled_under_transported_memory() {
    let (_, cylindrical_initial) = cylindrical_experiment_initial();
    let config = NonlocalConfig::new(0.5, 0.4, 0.02, 80, 1.0e-8).unwrap();
    let trajectory = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(81),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    for diagnostics in trajectory.diagnostics()
    {
        assert!(
            diagnostics.orthogonality_residual.abs() < 1.0e-9,
            "orthogonality residual grew too large: {:.6e}",
            diagnostics.orthogonality_residual
        );
    }
}

#[test]
fn complete_history_remains_exact_oracle_under_transported_pipeline() {
    let (_, cylindrical_initial) = cylindrical_experiment_initial();
    let config = NonlocalConfig::new(0.5, 0.1, 0.02, 16, 1.0e-8).unwrap();
    let trajectory = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(17),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap();

    assert_eq!(trajectory.history_diagnostics().len(), trajectory.len());
    assert!(trajectory.history_diagnostics().iter().all(|diagnostics| {
        diagnostics.approximation == HistoryApproximation::Exact
            && diagnostics.retained_samples == diagnostics.used_samples
    }));
}

#[test]
fn short_memory_backend_remains_explicitly_approximate_under_transported_pipeline() {
    let (_, cylindrical_initial) = cylindrical_experiment_initial();
    let config = NonlocalConfig::new(0.5, 0.1, 0.02, 20, 1.0e-8).unwrap();
    let trajectory = simulate_nonlocal_worldline_with_policy(
        &CylindricalMinkowski,
        cylindrical_initial,
        config,
        NonlocalSimulationPolicy::new(
            BoundedShortMemoryHistory::<4>::new(5).unwrap(),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap();

    assert!(trajectory.history_diagnostics().iter().all(|diagnostics| {
        diagnostics.approximation == HistoryApproximation::Approximate
            && diagnostics.retained_samples <= 5
    }));
}
