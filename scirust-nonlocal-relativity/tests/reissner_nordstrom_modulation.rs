use scirust_nonlocal_relativity::{
    CaputoCoordinateMemory, CompleteUniformHistory, DiscreteConnectionTransport, HeunPeceStepper,
    HistoryEntry, HistoryModulator, IdentityHistoryTransport, ModulatedCaputoCoordinateMemory,
    NonlocalConfig, NonlocalRelativityError, NonlocalSimulationPolicy,
    ReissnerNordstromFieldModulator, WorldlineState, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::ReissnerNordstrom;
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

fn approximately_circular_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn reissner_nordstrom_setup() -> (ReissnerNordstrom, WorldlineState<4>, NonlocalConfig) {
    let mass = 1.0;
    let charge = 0.3;
    let background = ReissnerNordstrom::try_new(mass, charge).unwrap();
    let mut initial = approximately_circular_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config = NonlocalConfig::new(0.55, 0.02, 0.02, 40, 1.0e-8).unwrap();
    (background, initial, config)
}

#[test]
fn weight_matches_hand_computed_field_invariant() {
    let mass = 1.0;
    let charge = 0.6;
    let background = ReissnerNordstrom::try_new(mass, charge).unwrap();
    let reference_length = 2.0;
    let beta = 0.7;
    let modulator =
        ReissnerNordstromFieldModulator::try_new(background, reference_length, beta).unwrap();
    let radius = 9.0;
    let entry = HistoryEntry::new([0.0, radius, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);

    let field_invariant = 2.0 * charge * charge / radius.powi(4);
    let expected_weight = 1.0 + beta * reference_length.powi(2) * field_invariant;

    assert_close(modulator.weight(&entry).unwrap(), expected_weight, 1.0e-14);
}

#[test]
fn zero_charge_gives_weight_one_regardless_of_beta() {
    let background = ReissnerNordstrom::try_new(1.0, 0.0).unwrap();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 5.0).unwrap();
    let entry = HistoryEntry::new([0.0, 8.0, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);

    assert_eq!(
        modulator.weight(&entry).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn beta_zero_bypasses_computation_and_returns_exactly_one() {
    let background = ReissnerNordstrom::try_new(1.0, 0.9).unwrap();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.0).unwrap();
    // A radius inside the horizon would normally be rejected; beta = 0.0
    // bypasses the radius check entirely, exactly like
    // SchwarzschildKretschmannModulator.
    let entry = HistoryEntry::new([0.0, 0.5, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);

    assert_eq!(
        modulator.weight(&entry).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn invalid_reference_length_is_rejected() {
    let background = ReissnerNordstrom::try_new(1.0, 0.3).unwrap();
    for length in [0.0, -2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
    {
        assert!(matches!(
            ReissnerNordstromFieldModulator::try_new(background, length, 0.1),
            Err(NonlocalRelativityError::InvalidModulationReferenceLength(_))
        ));
    }
}

#[test]
fn invalid_beta_is_rejected() {
    let background = ReissnerNordstrom::try_new(1.0, 0.3).unwrap();
    for beta in [-0.1, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
    {
        assert!(matches!(
            ReissnerNordstromFieldModulator::try_new(background, 1.0, beta),
            Err(NonlocalRelativityError::InvalidModulationBeta(_))
        ));
    }
}

#[test]
fn invalid_radius_is_rejected() {
    let background = ReissnerNordstrom::try_new(1.0, 0.3).unwrap();
    let horizon = background.outer_horizon_radius();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.1).unwrap();

    for radius in [horizon, horizon - 0.5, 0.0, -3.0, f64::NAN, f64::INFINITY]
    {
        let entry = HistoryEntry::new([0.0, radius, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);
        assert!(
            matches!(
                modulator.weight(&entry),
                Err(NonlocalRelativityError::InvalidModulationRadius(_))
            ),
            "radius {radius} should have been rejected"
        );
    }
}

#[test]
fn weight_decreases_as_radius_increases() {
    let background = ReissnerNordstrom::try_new(1.0, 0.5).unwrap();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.5).unwrap();
    let near = HistoryEntry::new([0.0, 6.0, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);
    let mid = HistoryEntry::new([0.0, 10.0, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);
    let far = HistoryEntry::new([0.0, 20.0, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);

    let weight_near = modulator.weight(&near).unwrap();
    let weight_mid = modulator.weight(&mid).unwrap();
    let weight_far = modulator.weight(&far).unwrap();

    assert!(weight_near > weight_mid, "{weight_near} <= {weight_mid}");
    assert!(weight_mid > weight_far, "{weight_mid} <= {weight_far}");
    assert!(weight_far > 1.0, "{weight_far} <= 1.0");
}

#[test]
fn accessors_return_constructed_values() {
    let background = ReissnerNordstrom::try_new(1.5, 0.4).unwrap();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 2.5, 0.3).unwrap();

    assert_close(modulator.background().mass(), 1.5, 0.0);
    assert_close(modulator.background().charge(), 0.4, 0.0);
    assert_close(modulator.reference_length(), 2.5, 0.0);
    assert_close(modulator.beta(), 0.3, 0.0);
}

#[test]
fn beta_zero_reproduces_baseline_bit_for_bit() {
    let (background, initial, config) = reissner_nordstrom_setup();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.0).unwrap();

    let baseline = simulate_nonlocal_worldline_with_policy(
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
    let modulated = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(modulator),
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    assert_eq!(baseline.len(), modulated.len());

    for (left, right) in baseline.states().iter().zip(modulated.states())
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
fn modulated_results_are_deterministic_bit_for_bit() {
    let (background, initial, config) = reissner_nordstrom_setup();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.2).unwrap();

    let policy = || {
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(modulator),
            DiscreteConnectionTransport,
            HeunPeceStepper,
        )
    };

    let first =
        simulate_nonlocal_worldline_with_policy(&background, initial, config, policy()).unwrap();
    let second =
        simulate_nonlocal_worldline_with_policy(&background, initial, config, policy()).unwrap();

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
fn small_positive_beta_produces_finite_bounded_measurable_deviation() {
    let (background, initial, config) = reissner_nordstrom_setup();
    let baseline_modulator =
        ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.0).unwrap();
    let coupled_modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.5).unwrap();

    let baseline = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(baseline_modulator),
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();
    let coupled = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(coupled_modulator),
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    let radial_deviation = coupled.final_state().unwrap().coordinates[1]
        - baseline.final_state().unwrap().coordinates[1];

    assert!(radial_deviation.is_finite());
    assert!(
        radial_deviation.abs() > 0.0,
        "deviation was exactly zero, not measurable"
    );
    assert!(
        radial_deviation.abs() < 1.0e-4,
        "deviation was unexpectedly large: {radial_deviation:.6e}"
    );
}

#[test]
fn modulation_composes_with_discrete_transport() {
    let (background, initial, config) = reissner_nordstrom_setup();
    let modulator = ReissnerNordstromFieldModulator::try_new(background, 1.0, 0.1).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(modulator),
            DiscreteConnectionTransport,
            HeunPeceStepper,
        ),
    )
    .unwrap();

    for state in trajectory.states()
    {
        assert!(state.coordinates.iter().all(|value| value.is_finite()));
        assert!(state.velocity.iter().all(|value| value.is_finite()));
    }
}
