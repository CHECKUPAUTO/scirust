//! Oracles for homogeneous ADM time evolution (Layer 3.2; see
//! `docs/LAYER_3_HOMOGENEOUS_EVOLUTION.md`).
//!
//! O1-O3 check the evolved scale factor against the three exact Friedmann
//! solutions, each read from the repository's **already-validated**
//! [`ScaleFactor`] implementations rather than a hand-retyped formula. O4
//! checks that halving the step reduces the error by ~2^4, confirming the RK4
//! order survives the full finite-difference evaluator stack. O5 checks
//! constraint preservation along the whole trajectory (free evolution: the
//! Hamiltonian constraint is monitored, never enforced). O6 checks that data
//! seeded deliberately *off* the constraint surface is actually detected.
//!
//! Established general relativity; a numerical approximation validated against
//! exact closed-form solutions. Restricted to the homogeneous sector, where
//! ADM's weak hyperbolicity cannot arise -- these tests say nothing about
//! inhomogeneous stability.

use scirust_relativity::adm_evolution::AdmEvolutionSettings;
use scirust_relativity::adm_homogeneous::{
    BarotropicFluid, HomogeneousEvolution, HomogeneousEvolutionError, HomogeneousState,
    evolve_homogeneous,
};
use scirust_relativity::{ExponentialScaleFactor, PowerLawScaleFactor, ScaleFactor};

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

/// The scale-factor error at `end` for a given step, plus the worst Hamiltonian
/// residual seen anywhere along the trajectory.
fn evolve_and_compare(
    fluid: BarotropicFluid,
    initial_scale: f64,
    hubble: f64,
    start: f64,
    end: f64,
    step: f64,
    exact_end: f64,
) -> (f64, f64) {
    let evolution = HomogeneousEvolution::new(fluid, settings());
    let initial = HomogeneousState::friedmann(initial_scale, hubble).expect("valid initial data");
    let history = evolve_homogeneous(&evolution, &initial, start, end, step).expect("evolves");
    let last = history.last().expect("non-empty history");
    let worst_residual = history.iter().fold(0.0_f64, |acc, sample| {
        acc.max(sample.hamiltonian_residual.abs())
    });
    ((last.scale_factor - exact_end).abs(), worst_residual)
}

// ---------------------------------------------------------------------------
// O1-O3 -- the three exact Friedmann solutions.
// ---------------------------------------------------------------------------

#[test]
fn o1_de_sitter_matches_the_exact_exponential_solution() {
    let hubble = 0.5;
    let end = 2.0;
    // Exact oracle from the repository's already-validated scale factor.
    let exact = ExponentialScaleFactor::try_new(hubble).expect("valid Hubble");
    let expected = exact.value(end);

    let (error, worst_residual) = evolve_and_compare(
        BarotropicFluid::vacuum_energy(),
        exact.value(0.0),
        hubble,
        0.0,
        end,
        0.005,
        expected,
    );
    assert!(
        error < 1.0e-9,
        "scale-factor error {error} (expected {expected})"
    );
    assert!(
        worst_residual < 1.0e-9,
        "worst Hamiltonian residual {worst_residual}"
    );
}

#[test]
fn o2_dust_matches_the_exact_two_thirds_power_law() {
    // Dust: a = (t / t_ref)^(2/3), so H(t_ref) = (2/3) / t_ref.
    let reference_time = 1.0;
    let exponent = 2.0 / 3.0;
    let end = 2.0;
    let exact = PowerLawScaleFactor::try_new(exponent, reference_time).expect("valid power law");

    let (error, worst_residual) = evolve_and_compare(
        BarotropicFluid::dust(),
        exact.value(reference_time),
        exponent / reference_time,
        reference_time,
        end,
        0.0025,
        exact.value(end),
    );
    assert!(error < 1.0e-9, "scale-factor error {error}");
    assert!(
        worst_residual < 1.0e-9,
        "worst Hamiltonian residual {worst_residual}"
    );
}

#[test]
fn o3_radiation_matches_the_exact_square_root_power_law() {
    // Radiation: a = (t / t_ref)^(1/2), so H(t_ref) = (1/2) / t_ref.
    let reference_time = 1.0;
    let exponent = 0.5;
    let end = 2.0;
    let exact = PowerLawScaleFactor::try_new(exponent, reference_time).expect("valid power law");

    let (error, worst_residual) = evolve_and_compare(
        BarotropicFluid::radiation(),
        exact.value(reference_time),
        exponent / reference_time,
        reference_time,
        end,
        0.0025,
        exact.value(end),
    );
    assert!(error < 1.0e-9, "scale-factor error {error}");
    assert!(
        worst_residual < 1.0e-9,
        "worst Hamiltonian residual {worst_residual}"
    );
}

// ---------------------------------------------------------------------------
// O4 -- fourth-order convergence through the full evaluator stack.
// ---------------------------------------------------------------------------

#[test]
fn o4_convergence_is_fourth_order() {
    // Halving the step must shrink the error by ~2^4 = 16. A tolerance band of
    // [10, 24] around 16 accommodates the finite-difference floor of the
    // right-hand sides without admitting a lower-order method (8 would be
    // third order, 32 fifth).
    let hubble = 0.5;
    let end = 2.0;
    let exact = ExponentialScaleFactor::try_new(hubble).expect("valid Hubble");
    let expected = exact.value(end);

    let mut previous_error = None;
    for &step in &[0.05, 0.025, 0.0125]
    {
        let (error, _) = evolve_and_compare(
            BarotropicFluid::vacuum_energy(),
            1.0,
            hubble,
            0.0,
            end,
            step,
            expected,
        );
        if let Some(previous) = previous_error
        {
            let ratio: f64 = previous / error;
            assert!(
                (10.0..=24.0).contains(&ratio),
                "step {step}: error ratio {ratio} is not consistent with fourth order (expected ~16)"
            );
        }
        previous_error = Some(error);
    }
}

// ---------------------------------------------------------------------------
// O5 -- constraint preservation under free evolution.
// ---------------------------------------------------------------------------

#[test]
fn o5_hamiltonian_constraint_is_preserved_and_improves_with_the_step() {
    let hubble = 0.5;
    let evolution = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());

    let mut previous_worst: Option<f64> = None;
    for &step in &[0.02, 0.01, 0.005]
    {
        let initial = HomogeneousState::friedmann(1.0, hubble).expect("valid initial data");
        let history = evolve_homogeneous(&evolution, &initial, 1.0, 3.0, step).expect("evolves");
        assert!(history.len() > 10, "expected a sampled trajectory");
        let worst = history.iter().fold(0.0_f64, |acc, sample| {
            acc.max(sample.hamiltonian_residual.abs())
        });
        // The constraint is monitored, never enforced: it stays satisfied only
        // because the evolution equations preserve it.
        assert!(worst < 1.0e-6, "step {step}: worst residual {worst}");
        if let Some(previous) = previous_worst
        {
            assert!(
                worst <= previous,
                "step {step}: residual {worst} did not improve on {previous}"
            );
        }
        previous_worst = Some(worst);
    }
}

#[test]
fn o5_initial_data_starts_exactly_on_the_constraint_surface() {
    // `friedmann` sets rho from the Hamiltonian constraint, so the residual at
    // t = 0 is at the rounding floor before any integration happens.
    let evolution = HomogeneousEvolution::new(BarotropicFluid::radiation(), settings());
    let initial = HomogeneousState::friedmann(1.0, 0.75).expect("valid initial data");
    let residual = evolution.hamiltonian_residual(&initial).expect("residual");
    assert!(residual.abs() < 1.0e-12, "initial residual {residual}");
}

// ---------------------------------------------------------------------------
// O6 -- deliberate constraint violation is detected, not silently accepted.
// ---------------------------------------------------------------------------

#[test]
fn o6_off_constraint_initial_data_is_detected() {
    let hubble = 0.5;
    let evolution = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());

    let mut previous = 0.0_f64;
    for &excess in &[0.01, 0.02, 0.05, 0.1]
    {
        let mut initial = HomogeneousState::friedmann(1.0, hubble).expect("valid initial data");
        // Seed rho off the constraint surface: H^2 = 8 pi rho / 3 no longer holds.
        initial.energy_density *= 1.0 + excess;
        let residual = evolution
            .hamiltonian_residual(&initial)
            .expect("residual")
            .abs();
        // The Hamiltonian term is -16 pi rho, so an excess of `excess * rho`
        // shows up as exactly 16 pi rho excess = 6 H^2 * excess.
        let expected = 6.0 * hubble * hubble * excess;
        assert!(
            (residual - expected).abs() < 1.0e-9,
            "excess {excess}: residual {residual} vs expected {expected}"
        );
        assert!(
            residual > previous,
            "residual should grow with the violation"
        );
        previous = residual;
    }
}

#[test]
fn o6_violated_initial_data_stays_violated_under_free_evolution() {
    // Free evolution does not repair bad initial data -- there is no projection
    // or damping, so a violated constraint must remain visibly violated.
    let hubble = 0.5;
    let evolution = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());
    let mut initial = HomogeneousState::friedmann(1.0, hubble).expect("valid initial data");
    initial.energy_density *= 1.1;

    let history = evolve_homogeneous(&evolution, &initial, 1.0, 2.0, 0.01).expect("evolves");
    let last = history.last().expect("non-empty");
    assert!(
        last.hamiltonian_residual.abs() > 1.0e-3,
        "violation should persist, got {}",
        last.hamiltonian_residual
    );
}

// ---------------------------------------------------------------------------
// Physical consistency, determinism, and rejection.
// ---------------------------------------------------------------------------

#[test]
fn vacuum_energy_density_stays_constant_while_dust_dilutes() {
    let hubble = 0.5;
    // w = -1: rho' = -3H(rho + p) = 0 exactly.
    let vacuum = HomogeneousEvolution::new(BarotropicFluid::vacuum_energy(), settings());
    let initial = HomogeneousState::friedmann(1.0, hubble).expect("valid");
    let history = evolve_homogeneous(&vacuum, &initial, 0.0, 2.0, 0.01).expect("evolves");
    let first = history.first().expect("non-empty");
    let last = history.last().expect("non-empty");
    assert!(
        (last.energy_density - first.energy_density).abs() < 1.0e-12,
        "vacuum energy drifted: {} -> {}",
        first.energy_density,
        last.energy_density
    );

    // Dust: rho ~ a^-3, so rho * a^3 is conserved.
    let dust = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());
    let initial = HomogeneousState::friedmann(1.0, hubble).expect("valid");
    let history = evolve_homogeneous(&dust, &initial, 1.0, 2.0, 0.005).expect("evolves");
    let first = history.first().expect("non-empty");
    let last = history.last().expect("non-empty");
    assert!(
        last.energy_density < first.energy_density,
        "dust should dilute"
    );
    let first_invariant = first.energy_density * first.scale_factor.powi(3);
    let last_invariant = last.energy_density * last.scale_factor.powi(3);
    assert!(
        (last_invariant - first_invariant).abs() / first_invariant < 1.0e-9,
        "rho * a^3 should be conserved for dust: {first_invariant} -> {last_invariant}"
    );
}

#[test]
fn hubble_parameter_tracks_the_exact_solution() {
    // De Sitter has constant H; the evolved Hubble parameter must stay put.
    let hubble = 0.5;
    let evolution = HomogeneousEvolution::new(BarotropicFluid::vacuum_energy(), settings());
    let initial = HomogeneousState::friedmann(1.0, hubble).expect("valid");
    let history = evolve_homogeneous(&evolution, &initial, 0.0, 3.0, 0.01).expect("evolves");
    for sample in &history
    {
        assert!(
            (sample.hubble_parameter - hubble).abs() < 1.0e-9,
            "H drifted to {} at t = {}",
            sample.hubble_parameter,
            sample.time
        );
    }
}

#[test]
fn evolution_is_deterministic() {
    let evolution = HomogeneousEvolution::new(BarotropicFluid::radiation(), settings());
    let initial = HomogeneousState::friedmann(1.0, 0.5).expect("valid");
    let first = evolve_homogeneous(&evolution, &initial, 1.0, 2.0, 0.01).expect("evolves");
    let second = evolve_homogeneous(&evolution, &initial, 1.0, 2.0, 0.01).expect("evolves");
    assert_eq!(first, second);
}

#[test]
fn rejects_invalid_requests() {
    let evolution = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());
    let initial = HomogeneousState::friedmann(1.0, 0.5).expect("valid");

    // Non-finite / non-positive equation of state parameter.
    assert!(matches!(
        BarotropicFluid::try_new(f64::NAN),
        Err(HomogeneousEvolutionError::InvalidEquationOfState(_))
    ));

    // Malformed time spans.
    assert!(matches!(
        evolve_homogeneous(&evolution, &initial, 2.0, 1.0, 0.01),
        Err(HomogeneousEvolutionError::InvalidTimeSpan { .. })
    ));
    assert!(matches!(
        evolve_homogeneous(&evolution, &initial, 0.0, 1.0, 0.0),
        Err(HomogeneousEvolutionError::InvalidTimeSpan { .. })
    ));
    assert!(matches!(
        evolve_homogeneous(&evolution, &initial, f64::NAN, 1.0, 0.01),
        Err(HomogeneousEvolutionError::InvalidTimeSpan { .. })
    ));

    // Negative energy density.
    let mut negative = initial;
    negative.energy_density = -1.0;
    assert!(matches!(
        evolve_homogeneous(&evolution, &negative, 0.0, 1.0, 0.01),
        Err(HomogeneousEvolutionError::InvalidEnergyDensity(_))
    ));

    // Singular spatial metric.
    let mut singular = initial;
    singular.spatial_metric = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(matches!(
        evolve_homogeneous(&evolution, &singular, 0.0, 1.0, 0.01),
        Err(HomogeneousEvolutionError::InvalidSpatialMetric)
    ));

    // Invalid initial data constructors.
    assert!(matches!(
        HomogeneousState::friedmann(0.0, 0.5),
        Err(HomogeneousEvolutionError::InvalidSpatialMetric)
    ));
    assert!(matches!(
        HomogeneousState::friedmann(1.0, f64::INFINITY),
        Err(HomogeneousEvolutionError::InvalidEnergyDensity(_))
    ));
}
