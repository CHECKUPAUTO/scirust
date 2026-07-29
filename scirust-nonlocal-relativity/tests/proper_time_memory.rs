use scirust_fractional::{FractionalOrder, caputo_l1_nonuniform};
use scirust_nonlocal_relativity::{
    CaputoCoordinateMemory, CompleteUniformHistory, HeunPeceStepper, IdentityHistoryTransport,
    NonlocalConfig, NonlocalRelativityError, NonlocalSimulationPolicy, WorldlineState,
    affine_trajectory_proper_time, proper_time_caputo_velocity_memory,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::{Connection, Metric, Minkowski, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

/// A spacelike-signature background used only to exercise the non-timelike
/// rejection path. Not part of the crate's public API and not a physical
/// spacetime.
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

#[test]
fn proper_time_memory_is_finite_for_a_valid_timelike_trajectory() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.02;
    let config = NonlocalConfig::new(0.55, 0.02, 0.02, 40, 1.0e-8).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let memory =
        proper_time_caputo_velocity_memory(&trajectory, config.step(), config.fractional_order())
            .unwrap();

    assert!(memory.iter().all(|value| value.is_finite()));
}

#[test]
fn proper_time_memory_is_exactly_zero_for_constant_velocity() {
    let initial = WorldlineState::new([1.0, -2.0, 3.0, -4.0], [2.0, 0.25, -0.5, 0.75]);
    let config = NonlocalConfig::new(0.5, 0.0, 0.1, 24, 1.0e-12).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &Minkowski,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(25),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    // Zero coupling in flat spacetime keeps velocity exactly constant, so the
    // proper-time axis is uniform and every retained sample is identical:
    // the Caputo L1 stencil sees an exact zero first difference everywhere,
    // on any grid.
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

    let memory =
        proper_time_caputo_velocity_memory(&trajectory, config.step(), config.fractional_order())
            .unwrap();

    for component in memory
    {
        assert_eq!(component.to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn proper_time_memory_rejects_non_timelike_trajectory() {
    let initial = WorldlineState::new([0.0; 4], [0.1, 1.0, 0.0, 0.0]);
    let config = NonlocalConfig::new(0.5, 0.0, 0.05, 8, 1.0e-8).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &EuclideanBackground,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(9),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let result =
        proper_time_caputo_velocity_memory(&trajectory, config.step(), config.fractional_order());

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::NonTimelikeMetricNorm { .. })
    ));
}

#[test]
fn proper_time_memory_is_deterministic_bit_for_bit() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.02;
    let config = NonlocalConfig::new(0.55, 0.03, 0.02, 40, 1.0e-8).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let first =
        proper_time_caputo_velocity_memory(&trajectory, config.step(), config.fractional_order())
            .unwrap();
    let second =
        proper_time_caputo_velocity_memory(&trajectory, config.step(), config.fractional_order())
            .unwrap();

    for component in 0..4
    {
        assert_eq!(first[component].to_bits(), second[component].to_bits());
    }
}

#[test]
fn proper_time_memory_matches_manual_composition_bit_for_bit() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.02;
    let config = NonlocalConfig::new(0.55, 0.03, 0.02, 40, 1.0e-8).unwrap();
    let order: FractionalOrder = config.fractional_order();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let increments = affine_trajectory_proper_time(&trajectory, config.step()).unwrap();
    let mut proper_times = vec![0.0_f64];
    for entry in &increments
    {
        proper_times.push(entry.cumulative_proper_time);
    }

    let mut expected = [0.0_f64; 4];
    for (component, expected_component) in expected.iter_mut().enumerate()
    {
        let samples: Vec<f64> = trajectory
            .states()
            .iter()
            .map(|state| state.velocity[component])
            .collect();
        *expected_component = caputo_l1_nonuniform(&samples, &proper_times, order).unwrap();
    }

    let actual =
        proper_time_caputo_velocity_memory(&trajectory, config.step(), config.fractional_order())
            .unwrap();

    for component in 0..4
    {
        assert_eq!(actual[component].to_bits(), expected[component].to_bits());
    }
}
