use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, AdaptiveSimulationPolicy, AdaptiveTolerance, BoundedShortMemoryHistory,
    CaputoCoordinateMemory, CompleteUniformHistory, DiscreteConnectionTransport, HeunPeceStepper,
    IdentityHistoryModulator, IdentityHistoryTransport, NonlocalConfig, NonlocalRelativityError,
    NonlocalSimulationPolicy, ReissnerNordstromFieldModulator, SchwarzschildKretschmannModulator,
    WorldlineState, simulate_nonlocal_worldline_adaptive,
    simulate_nonlocal_worldline_adaptive_with_policy, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::{Minkowski, ReissnerNordstrom, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;

    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

#[test]
fn constructor_rejects_invalid_parameters() {
    assert!(matches!(
        AdaptiveNonlocalConfig::new(1.5, 0.02, 0.05, 0.001, 0.5, 1.0e-9, 1.0e-8, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidFractionalOrder(_))
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, -0.1, 0.05, 0.001, 0.5, 1.0e-9, 1.0e-8, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidCoupling(_))
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.0, 0.5, 1.0e-9, 1.0e-8, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
            field: "min_step",
            ..
        })
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.1, 0.05, 1.0e-9, 1.0e-8, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
            field: "max_step",
            ..
        })
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 1.0, 0.001, 0.5, 1.0e-9, 1.0e-8, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
            field: "initial_step",
            ..
        })
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.5, 0.0, 1.0e-8, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
            field: "error_tolerance",
            ..
        })
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.5, 1.0e-9, 0.0, 2.0, 500, 20),
        Err(NonlocalRelativityError::InvalidMetricNormFloor(_))
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.5, 1.0e-9, 1.0e-8, 0.0, 500, 20),
        Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
            field: "target_affine_parameter",
            ..
        })
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.5, 1.0e-9, 1.0e-8, 2.0, 0, 20),
        Err(NonlocalRelativityError::InvalidStepCount(0))
    ));
    assert!(matches!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.5, 1.0e-9, 1.0e-8, 2.0, 500, 0),
        Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
            field: "max_rejections_per_step",
            ..
        })
    ));
    assert!(
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.5, 1.0e-9, 1.0e-8, 2.0, 500, 20)
            .is_ok()
    );
}

#[test]
fn adaptive_simulation_reaches_the_target_affine_parameter() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let target = 1.6;
    let config = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.02, 0.001, 0.2, 1.0e-9, 1.0e-8, target, 5_000, 30,
    )
    .unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();

    let final_parameter = trajectory.final_diagnostics().unwrap().affine_parameter;
    assert_close(final_parameter, target, 1.0e-9);
}

#[test]
fn adaptive_trajectory_uses_non_uniform_steps() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    // A deliberately generous initial step relative to the tolerance forces
    // early rejections and shrinkage.
    let config = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.15, 0.00005, 0.2, 1.0e-8, 1.0e-8, 1.0, 5_000, 60,
    )
    .unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();
    let parameters: Vec<f64> = trajectory
        .diagnostics()
        .iter()
        .map(|d| d.affine_parameter)
        .collect();

    assert!(parameters.len() > 2, "expected more than two samples");

    let mut increments: Vec<f64> = parameters.windows(2).map(|w| w[1] - w[0]).collect();
    increments.dedup_by(|a, b| (*a - *b).abs() < 1.0e-15);

    assert!(
        increments.len() > 1,
        "expected non-uniform step sizes, got a single repeated increment"
    );
}

#[test]
fn adaptive_is_deterministic_bit_for_bit() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.8, 5_000, 30)
            .unwrap();

    let first = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();
    let second = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();

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
}

#[test]
fn constant_velocity_in_flat_spacetime_has_exactly_zero_memory() {
    let initial = WorldlineState::new([1.0, -2.0, 3.0, -4.0], [2.0, 0.25, -0.5, 0.75]);
    let config =
        AdaptiveNonlocalConfig::new(0.5, 0.03, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 1.0, 5_000, 30)
            .unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive(&Minkowski, initial, config).unwrap();

    for state in trajectory.states()
    {
        for component in 0..4
        {
            assert_eq!(
                state.velocity[component].to_bits(),
                initial.velocity[component].to_bits()
            );
        }
    }

    for diagnostics in trajectory.diagnostics()
    {
        assert_eq!(diagnostics.memory_l2_norm.to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn step_budget_exhaustion_is_reported_not_silently_truncated() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    // Deliberately too few accepted steps to reach the target.
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.02, 0.001, 0.02, 1.0e-9, 1.0e-8, 5.0, 3, 30)
            .unwrap();

    let result = simulate_nonlocal_worldline_adaptive(&background, initial, config);

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::AdaptiveStepBudgetExhausted {
            accepted_steps: 3,
            ..
        })
    ));
}

#[test]
fn rejects_non_finite_initial_state() {
    let background = Minkowski;
    let initial = WorldlineState::new([f64::NAN, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    let config =
        AdaptiveNonlocalConfig::new(0.5, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 1.0, 500, 20)
            .unwrap();

    let result = simulate_nonlocal_worldline_adaptive(&background, initial, config);

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::NonFiniteInitialCoordinate { .. })
    ));
}

#[test]
fn adaptive_matches_fine_fixed_step_heun_pece_closely() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let alpha = 0.55;
    let coupling = 0.02;
    let target = 0.8;

    let adaptive_config = AdaptiveNonlocalConfig::new(
        alpha, coupling, 0.01, 0.00002, 0.05, 1.0e-8, 1.0e-8, target, 20_000, 60,
    )
    .unwrap();
    let adaptive_trajectory =
        simulate_nonlocal_worldline_adaptive(&background, initial, adaptive_config).unwrap();

    let fine_step = 0.005;
    let fine_steps = (target / fine_step).round() as usize;
    let fixed_config = NonlocalConfig::new(alpha, coupling, fine_step, fine_steps, 1.0e-8).unwrap();
    let fixed_trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        fixed_config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(fine_steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let adaptive_final = adaptive_trajectory.final_state().unwrap();
    let fixed_final = fixed_trajectory.final_state().unwrap();

    for component in 0..4
    {
        assert_close(
            adaptive_final.coordinates[component],
            fixed_final.coordinates[component],
            1.0e-4,
        );
        assert_close(
            adaptive_final.velocity[component],
            fixed_final.velocity[component],
            1.0e-4,
        );
    }
}

/// Golden bit-for-bit regression anchor for the deterministic output of
/// `simulate_nonlocal_worldline_adaptive`.
///
/// These values were recomputed for the Phase 1 scaled local-error norm. The
/// previous golden (353 samples) was captured before the adaptive controllers
/// switched from the unscaled sum of coordinate and velocity L2 differences
/// against one absolute tolerance to the componentwise scaled RMS norm
/// (`scaled_local_error_norm`); that change is a deliberate, documented
/// correctness improvement, not a regression, so it legitimately changes which
/// steps are accepted (here 353 -> 176 samples) and therefore the endpoint
/// bits. The target affine parameter is still reached exactly (`0.8`), and the
/// identity-policy and plain-entry-point paths still agree bit-for-bit (see
/// `adaptive_with_identity_policy_matches_plain_entry_point_bit_for_bit`); this
/// test guards the scaled-norm controller against unintended future drift.
#[test]
fn adaptive_scaled_norm_golden_values_bit_for_bit() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.8, 5_000, 30)
            .unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();

    assert_eq!(trajectory.len(), 176);

    let final_state = trajectory.final_state().unwrap();
    let expected_coordinates_bits: [u64; 4] = [
        0x3fee99d33872904c,
        0x4023fbe788e8f7f6,
        0x3ff921fb54442d18,
        0x3f9efcc3603d09df,
    ];
    let expected_velocity_bits: [u64; 4] = [
        0x3ff3209f6d6aa13e,
        0xbf84797d4428fc0c,
        0x3bf46a69ccd61f69,
        0x3fa361e17609ee9b,
    ];
    for component in 0..4
    {
        assert_eq!(
            final_state.coordinates[component].to_bits(),
            expected_coordinates_bits[component],
            "coordinate {component}"
        );
        assert_eq!(
            final_state.velocity[component].to_bits(),
            expected_velocity_bits[component],
            "velocity {component}"
        );
    }

    let final_diagnostics = trajectory.final_diagnostics().unwrap();
    assert_eq!(
        final_diagnostics.affine_parameter.to_bits(),
        0x3fe999999999999a
    );
    assert_eq!(
        final_diagnostics.memory_l2_norm.to_bits(),
        0x3f34510b8cc1bd6f
    );
}

#[test]
fn adaptive_with_identity_policy_matches_plain_entry_point_bit_for_bit() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.8, 5_000, 30)
            .unwrap();

    let plain = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();
    let explicit = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            IdentityHistoryModulator,
        ),
    )
    .unwrap();

    assert_eq!(plain.len(), explicit.len());
    for (left, right) in plain.states().iter().zip(explicit.states())
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
fn adaptive_composes_with_discrete_connection_transport() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.6, 5_000, 30)
            .unwrap();

    let transported = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            DiscreteConnectionTransport,
            IdentityHistoryModulator,
        ),
    )
    .unwrap();

    for state in transported.states()
    {
        assert!(state.coordinates.iter().all(|value| value.is_finite()));
        assert!(state.velocity.iter().all(|value| value.is_finite()));
    }

    let identity = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            IdentityHistoryModulator,
        ),
    )
    .unwrap();

    // DiscreteConnectionTransport must actually change the numerical result
    // here (Schwarzschild is curved), not silently fall back to identity
    // behavior.
    let transported_final = transported.final_state().unwrap();
    let identity_final = identity.final_state().unwrap();
    let differs = (0..4).any(|component| {
        transported_final.coordinates[component] != identity_final.coordinates[component]
    });
    assert!(differs, "transported and identity results were identical");
}

#[test]
fn adaptive_composes_with_schwarzschild_kretschmann_modulator() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.6, 5_000, 30)
            .unwrap();

    let baseline_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.0).unwrap();
    let coupled_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.5).unwrap();

    let baseline = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            baseline_modulator,
        ),
    )
    .unwrap();
    let coupled = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            coupled_modulator,
        ),
    )
    .unwrap();

    for state in coupled.states()
    {
        assert!(state.coordinates.iter().all(|value| value.is_finite()));
    }

    let baseline_final = baseline.final_state().unwrap();
    let coupled_final = coupled.final_state().unwrap();
    let differs = (0..4).any(|component| {
        baseline_final.coordinates[component] != coupled_final.coordinates[component]
    });
    assert!(differs, "modulated and baseline results were identical");
}

#[test]
fn adaptive_beta_zero_modulator_matches_identity_modulator_bit_for_bit() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.6, 5_000, 30)
            .unwrap();
    let beta_zero_modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.0).unwrap();

    let with_beta_zero = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            beta_zero_modulator,
        ),
    )
    .unwrap();
    let with_identity = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            IdentityHistoryModulator,
        ),
    )
    .unwrap();

    assert_eq!(with_beta_zero.len(), with_identity.len());
    for (left, right) in with_beta_zero.states().iter().zip(with_identity.states())
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
fn adaptive_composes_with_reissner_nordstrom_field_modulator() {
    let mass = 1.0;
    let charge = 0.4;
    let background = ReissnerNordstrom::try_new(mass, charge).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.6, 5_000, 30)
            .unwrap();

    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.5).unwrap();

    let modulated = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            IdentityHistoryTransport,
            modulator,
        ),
    )
    .unwrap();

    for state in modulated.states()
    {
        assert!(state.coordinates.iter().all(|value| value.is_finite()));
        assert!(state.velocity.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn adaptive_composes_with_both_transport_and_modulation_together() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.6, 5_000, 30)
            .unwrap();
    let modulator = SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.3).unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            CompleteUniformHistory::<4>::new(),
            DiscreteConnectionTransport,
            modulator,
        ),
    )
    .unwrap();

    for state in trajectory.states()
    {
        assert!(state.coordinates.iter().all(|value| value.is_finite()));
        assert!(state.velocity.iter().all(|value| value.is_finite()));
    }
    for diagnostics in trajectory.diagnostics()
    {
        assert!(diagnostics.memory_l2_norm.is_finite());
        assert!(diagnostics.memory_force_l2_norm.is_finite());
    }
}

#[test]
fn adaptive_composes_with_bounded_short_memory() {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.05, 0.001, 0.2, 1.0e-9, 1.0e-8, 0.4, 5_000, 30)
            .unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive_with_policy(
        &background,
        initial,
        config,
        AdaptiveSimulationPolicy::new(
            BoundedShortMemoryHistory::<4>::new(6).unwrap(),
            IdentityHistoryTransport,
            IdentityHistoryModulator,
        ),
    )
    .unwrap();

    for state in trajectory.states()
    {
        assert!(state.coordinates.iter().all(|value| value.is_finite()));
    }
    assert!(
        trajectory
            .history_diagnostics()
            .iter()
            .all(|diagnostics| diagnostics.retained_samples <= 6),
        "bounded short memory retained more than its window"
    );
}

#[test]
fn with_tolerance_constructor_exposes_asymmetric_scaled_tolerance() {
    let tolerance = AdaptiveTolerance::new(1.0e-6, 1.0e-8, 1.0e-10).unwrap();
    let config = AdaptiveNonlocalConfig::with_tolerance(
        0.55, 0.02, 0.05, 0.001, 0.2, tolerance, 1.0e-8, 0.8, 5_000, 30,
    )
    .unwrap();

    assert_eq!(
        config.tolerance().relative().to_bits(),
        1.0e-6_f64.to_bits()
    );
    assert_eq!(
        config.tolerance().coordinate_absolute().to_bits(),
        1.0e-8_f64.to_bits()
    );
    assert_eq!(
        config.tolerance().velocity_absolute().to_bits(),
        1.0e-10_f64.to_bits()
    );

    // The configuration still integrates to the target parameter.
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    let trajectory = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();
    assert_close(
        trajectory.final_diagnostics().unwrap().affine_parameter,
        0.8,
        1.0e-9,
    );
}

/// Integration counterpart to the norm-level scale-invariance unit test: the
/// Schwarzschild time coordinate `t` has a purely gauge origin (the metric is
/// static, so `t -> t + C` is an exact symmetry of the physics). Adding a
/// large constant to that arbitrary origin changes the numerical *scale* of
/// the `t` coordinate from order 1 to order 1e6 without changing anything
/// physically relevant. Because the scaled RMS norm holds each component to a
/// relative accuracy, the accepted-step count stays essentially unchanged
/// rather than being driven by the absolute magnitude of an arbitrarily
/// offset coordinate. (Observed with the shipped parameters: 73 vs 70 steps.)
#[test]
fn scaled_norm_step_count_is_robust_to_arbitrary_coordinate_scale() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let tolerance = AdaptiveTolerance::new(1.0e-7, 1.0e-9, 1.0e-9).unwrap();
    let config = AdaptiveNonlocalConfig::with_tolerance(
        0.55, 0.02, 0.05, 0.001, 0.2, tolerance, 1.0e-8, 0.8, 20_000, 30,
    )
    .unwrap();

    let mut base = circular_schwarzschild_state(1.0, 10.0);
    base.velocity[1] = -0.01;

    let mut offset = base;
    offset.coordinates[0] = 1.0e6;

    let base_steps = simulate_nonlocal_worldline_adaptive(&background, base, config)
        .unwrap()
        .len();
    let offset_steps = simulate_nonlocal_worldline_adaptive(&background, offset, config)
        .unwrap()
        .len();

    assert!(base_steps > 10 && offset_steps > 10);
    // "Not catastrophic": the two accepted-step counts stay within 25% of
    // each other despite a 1e6-fold change in the t-coordinate's magnitude.
    let larger = base_steps.max(offset_steps);
    let smaller = base_steps.min(offset_steps);
    assert!(
        (larger - smaller) * 4 <= larger,
        "step counts diverged too far: base={base_steps}, offset={offset_steps}"
    );
}

// ---- Phase 2: rejection-budget enforcement (embedded Heun-Euler controller) ----

#[test]
fn embedded_rejection_budget_of_one_fails_on_first_rejection() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    // Generous initial step vs a tight tolerance forces the first trial to be
    // rejected; a budget of one is then exhausted immediately, before the step
    // ever shrinks below the (tiny) min_step.
    let config = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.15, 0.00005, 0.2, 1.0e-8, 1.0e-8, 1.0, 5_000, 1,
    )
    .unwrap();

    let result = simulate_nonlocal_worldline_adaptive(&background, initial, config);

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::AdaptiveRejectionBudgetExhausted {
            accepted_step: 0,
            rejections: 1,
            ..
        })
    ));
}

#[test]
fn embedded_minimum_step_exhaustion_is_a_distinct_error() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    // A high min_step (close to initial_step) and a large rejection budget: the
    // proposed shrink crosses min_step before the retry count is reached, so
    // this must be reported as minimum-step exhaustion, not rejection-budget
    // exhaustion.
    let config =
        AdaptiveNonlocalConfig::new(0.55, 0.02, 0.15, 0.1, 0.2, 1.0e-9, 1.0e-8, 1.0, 5_000, 100)
            .unwrap();

    let result = simulate_nonlocal_worldline_adaptive(&background, initial, config);

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::AdaptiveMinimumStepExhausted {
            accepted_step: 0,
            min_step,
            ..
        }) if (min_step - 0.1).abs() < 1.0e-15
    ));
}

#[test]
fn embedded_larger_budget_allows_shrinkage_and_acceptance() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    // Same generous initial step as the budget-of-one case, but a budget large
    // enough to shrink into tolerance and complete the whole trajectory.
    let config = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.15, 0.00005, 0.2, 1.0e-8, 1.0e-8, 1.0, 5_000, 60,
    )
    .unwrap();

    let trajectory = simulate_nonlocal_worldline_adaptive(&background, initial, config).unwrap();
    assert_close(
        trajectory.final_diagnostics().unwrap().affine_parameter,
        1.0,
        1.0e-9,
    );
}

/// The rejection counter must be per accepted step, not cumulative across the
/// run. With this config the hardest accepted step needs two rejections
/// (budget 2 fails, budget 3 succeeds), yet the whole run accepts hundreds of
/// steps — so the total rejection count far exceeds 3. A cumulative counter
/// would be exhausted within the first few accepted steps; completing the run
/// proves the counter resets on every acceptance.
#[test]
fn embedded_rejection_counter_resets_after_each_accepted_step() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;

    let too_small = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.2, 0.00005, 0.2, 1.0e-9, 1.0e-8, 1.5, 50_000, 2,
    )
    .unwrap();
    assert!(matches!(
        simulate_nonlocal_worldline_adaptive(&background, initial, too_small),
        Err(NonlocalRelativityError::AdaptiveRejectionBudgetExhausted { .. })
    ));

    let sufficient = AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.2, 0.00005, 0.2, 1.0e-9, 1.0e-8, 1.5, 50_000, 3,
    )
    .unwrap();
    let trajectory =
        simulate_nonlocal_worldline_adaptive(&background, initial, sufficient).unwrap();
    assert!(
        trajectory.len() > 50,
        "expected many accepted steps, got {}",
        trajectory.len()
    );
    assert_close(
        trajectory.final_diagnostics().unwrap().affine_parameter,
        1.5,
        1.0e-9,
    );
}
