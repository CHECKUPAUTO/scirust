use scirust_nonlocal_relativity::{
    BoundedShortMemoryHistory, CaputoCoordinateMemory, CompleteUniformHistory,
    DiscreteConnectionTransport, HeunPeceStepper, HistoryApproximation, HistoryEntry,
    HistoryModulator, IdentityHistoryTransport, ModulatedCaputoCoordinateMemory, NonlocalConfig,
    NonlocalRelativityError, NonlocalSimulationPolicy, SchwarzschildKretschmannModulator,
    SemiImplicitEulerStepper, WorldlineState, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;
use std::f64::consts::FRAC_PI_2;

fn circular_timelike_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn schwarzschild_setup() -> (Schwarzschild, WorldlineState<4>, NonlocalConfig) {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let mut initial = circular_timelike_state(mass, 10.0);
    initial.velocity[1] = -0.01;
    let config = NonlocalConfig::new(0.55, 0.02, 0.02, 40, 1.0e-8).unwrap();
    (background, initial, config)
}

#[test]
fn beta_zero_reproduces_baseline_bit_for_bit() {
    let (background, initial, config) = schwarzschild_setup();
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.0).unwrap();

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

    for (left, right) in baseline.diagnostics().iter().zip(modulated.diagnostics())
    {
        assert_eq!(
            left.memory_l2_norm.to_bits(),
            right.memory_l2_norm.to_bits()
        );
        assert_eq!(
            left.memory_force_l2_norm.to_bits(),
            right.memory_force_l2_norm.to_bits()
        );
    }
}

#[test]
fn invalid_mass_is_rejected() {
    for mass in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
    {
        assert!(matches!(
            SchwarzschildKretschmannModulator::try_new(mass, 1.0, 0.1),
            Err(NonlocalRelativityError::InvalidModulationMass(_))
        ));
    }
}

#[test]
fn invalid_reference_length_is_rejected() {
    for length in [0.0, -2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
    {
        assert!(matches!(
            SchwarzschildKretschmannModulator::try_new(1.0, length, 0.1),
            Err(NonlocalRelativityError::InvalidModulationReferenceLength(_))
        ));
    }
}

#[test]
fn invalid_beta_is_rejected() {
    for beta in [-0.1, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
    {
        assert!(matches!(
            SchwarzschildKretschmannModulator::try_new(1.0, 1.0, beta),
            Err(NonlocalRelativityError::InvalidModulationBeta(_))
        ));
    }
}

#[test]
fn invalid_radius_is_rejected() {
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.1).unwrap();
    assert_eq!(modulator.horizon_radius(), 2.0);

    for radius in [2.0, 1.5, 0.0, -3.0, f64::NAN, f64::INFINITY]
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
fn non_finite_weight_is_rejected() {
    let huge = 1.0e100;
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, huge, huge).unwrap();
    let entry = HistoryEntry::new([0.0, 10.0, FRAC_PI_2, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0);

    assert!(matches!(
        modulator.weight(&entry),
        Err(NonlocalRelativityError::NonFiniteModulationWeight(_))
    ));
}

#[test]
fn weight_decreases_as_radius_increases() {
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.5).unwrap();
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
fn modulated_results_are_deterministic_bit_for_bit() {
    let (background, initial, config) = schwarzschild_setup();
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.1).unwrap();

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
    let (background, initial, config) = schwarzschild_setup();
    let baseline_modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.0).unwrap();
    let coupled_modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.5).unwrap();

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
        radial_deviation.abs() < 1.0e-6,
        "deviation was unexpectedly large: {radial_deviation:.6e}"
    );
}

#[test]
fn modulation_composes_with_discrete_transport() {
    let (background, initial, config) = schwarzschild_setup();
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.1).unwrap();

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

#[test]
fn modulation_composes_with_exact_history() {
    let (background, initial, config) = schwarzschild_setup();
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.1).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(modulator),
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap();

    assert!(trajectory.history_diagnostics().iter().all(|diagnostics| {
        diagnostics.approximation == HistoryApproximation::Exact
            && diagnostics.retained_samples == diagnostics.used_samples
    }));
}

#[test]
fn modulation_composes_with_short_memory() {
    let (background, initial, config) = schwarzschild_setup();
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.1).unwrap();

    let trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            BoundedShortMemoryHistory::<4>::new(5).unwrap(),
            ModulatedCaputoCoordinateMemory::new(modulator),
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap();

    assert!(trajectory.history_diagnostics().iter().all(|diagnostics| {
        diagnostics.approximation == HistoryApproximation::Approximate
            && diagnostics.retained_samples <= 5
    }));
}

#[test]
fn modulation_composes_with_both_integrators() {
    let (background, initial, config) = schwarzschild_setup();
    let modulator = SchwarzschildKretschmannModulator::try_new(1.0, 1.0, 0.1).unwrap();

    let euler = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(41),
            ModulatedCaputoCoordinateMemory::new(modulator),
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap();
    let heun = simulate_nonlocal_worldline_with_policy(
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

    assert_eq!(euler.len(), config.steps() + 1);
    assert_eq!(heun.len(), config.steps() + 1);
    assert!(
        euler
            .states()
            .iter()
            .all(|state| state.coordinates.iter().all(|value| value.is_finite()))
    );
    assert!(
        heun.states()
            .iter()
            .all(|state| state.coordinates.iter().all(|value| value.is_finite()))
    );
}

/// Mechanically enforces the non-negotiable scientific boundary: no source
/// file in this crate declares a struct, enum, trait, or function whose name
/// suggests a modified field equation, Einstein tensor, or stress-energy
/// structure. Comment lines (which may legitimately *disclaim* such things in
/// prose) are skipped; only actual item declarations are checked.
#[test]
fn no_modified_field_equation_structure_is_introduced() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_dir = std::path::Path::new(manifest_dir).join("src");
    let forbidden_substrings = [
        "einsteintensor",
        "einstein_tensor",
        "fieldequation",
        "field_equation",
        "stressenergy",
        "stress_energy",
        "modifiedeinstein",
        "modified_einstein",
        "fractionaleinstein",
        "fractional_einstein",
    ];
    let declaration_keywords = ["struct ", "enum ", "trait ", "fn "];

    let mut checked_files = 0usize;

    for entry in std::fs::read_dir(&src_dir).expect("src directory exists")
    {
        let path = entry.expect("readable directory entry").path();

        if path.extension().and_then(|extension| extension.to_str()) != Some("rs")
        {
            continue;
        }

        checked_files += 1;
        let contents = std::fs::read_to_string(&path).expect("readable source file");

        for line in contents.lines()
        {
            let trimmed = line.trim_start();

            if trimmed.starts_with("//")
            {
                continue;
            }

            let lower = line.to_ascii_lowercase();
            let declares_item = declaration_keywords
                .iter()
                .any(|keyword| lower.contains(keyword));

            if !declares_item
            {
                continue;
            }

            for forbidden in forbidden_substrings
            {
                assert!(
                    !lower.contains(forbidden),
                    "file {path:?} declares an item suggesting a modified field equation: {line}"
                );
            }
        }
    }

    assert!(
        checked_files > 0,
        "expected to check at least one source file"
    );
}
