//! `L2` reproduction verdicts over *real* solver output.
//!
//! The unit tests in `verdict.rs` construct certificates by hand to pin the
//! comparison rules exactly. This file does the opposite: it runs the actual
//! adaptive backends twice and asks whether the second run reproduces the
//! first, which is the question `sos-repro` defers to this crate.
//!
//! That distinction matters for one property in particular — two adaptive
//! runs at *different* tolerances really do choose different step sequences,
//! so the endpoint comparison is not merely defensive coding against a
//! hypothetical. A test below shows the grids genuinely differ.

use sos_core::{DeterminismLevel, HashAlgo, ObjectId, SemVer};
use sos_repro::{NodeClaim, NodeVerdict, verify_node};
use sos_scirust::ode::{AdaptiveOdeConfig, Dopri5OdeSimulator};
use sos_scirust::quadrature::{QuadratureConfig, QuadratureSimulator};
use sos_scirust::verdict::{
    DEFAULT_SIGMA, VerdictError, certify_estimate_in_distribution, certify_integral,
    certify_trajectory_endpoint, certify_trajectory_pointwise,
};
use sos_simulation::{SimDescriptor, Simulate};

/// `dy/dt = -y`, exact solution `y(t) = e^(-t)`.
fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}

fn ode(rtol: f64, atol: f64) -> sos_scirust::ode::CertifiedTrajectory {
    let sim = Dopri5OdeSimulator::new(
        SimDescriptor::new("scirust-solvers/ode-dopri5", SemVer::new(1, 0, 0)),
        decay as fn(f64, &[f64], &mut [f64]),
    );
    let config = AdaptiveOdeConfig::new(0.0, 1.0, vec![1.0], rtol, atol, 0.05);
    sim.run(&config, 0)
        .expect("the integration must succeed")
        .output
}

fn integral(tol: f64) -> sos_scirust::quadrature::CertifiedIntegral {
    let sim = QuadratureSimulator::new(
        SimDescriptor::new("scirust-solvers/quadrature", SemVer::new(1, 0, 0)),
        (|x: f64| x.sin()) as fn(f64) -> f64,
    );
    let config = QuadratureConfig::new(0.0, std::f64::consts::PI, tol, 50);
    sim.run(&config, 0)
        .expect("the quadrature must succeed")
        .output
}

#[test]
fn an_identical_rerun_reproduces() {
    // The easy case, but it has to hold: same backend, same config, same
    // seed. Anything else would mean the verdict is broken, not the solver.
    let original = ode(1e-8, 1e-10);
    let rerun = ode(1e-8, 1e-10);
    let agreement = certify_trajectory_endpoint(&original, &rerun).unwrap();
    assert!(agreement.agrees);
    assert_eq!(agreement.deviation, 0.0, "DOPRI5 is deterministic here");
    // And it reproduces pointwise too, since the grids are identical.
    assert!(
        certify_trajectory_pointwise(&original, &rerun)
            .unwrap()
            .agrees
    );
}

#[test]
fn a_tighter_rerun_reproduces_the_looser_original_at_its_endpoint() {
    // The case the summed-tolerance rule exists for: both runs honoured
    // their own certificates, and they differ by more than either one alone
    // allows in the tight direction.
    let loose = ode(1e-5, 1e-7);
    let tight = ode(1e-10, 1e-12);
    let agreement = certify_trajectory_endpoint(&loose, &tight).unwrap();
    assert!(
        agreement.agrees,
        "deviation {} exceeded the summed bound {}",
        agreement.deviation, agreement.allowed
    );
    // Both are genuinely close to e^-1 — the verdict is not passing because
    // the solver produced nothing.
    let exact = std::f64::consts::E.recip();
    for c in [&loose, &tight]
    {
        assert!((c.trajectory.last().unwrap().1[0] - exact).abs() < 1e-4);
    }
}

#[test]
fn runs_at_different_tolerances_really_do_pick_different_grids() {
    // This is why `certify_trajectory_endpoint` exists as a separate entry
    // point rather than being a defensive branch inside a pointwise compare.
    let loose = ode(1e-4, 1e-6);
    let tight = ode(1e-11, 1e-13);
    assert_ne!(
        loose.trajectory.len(),
        tight.trajectory.len(),
        "adaptive runs at different tolerances should differ in step count"
    );
    assert!(matches!(
        certify_trajectory_pointwise(&loose, &tight),
        Err(VerdictError::NotComparable(_))
    ));
    // ...while the endpoint comparison still answers the real question.
    assert!(certify_trajectory_endpoint(&loose, &tight).unwrap().agrees);
}

#[test]
fn a_rerun_of_a_different_problem_does_not_reproduce() {
    // The verdict has to be able to say no. Same solver and tolerances, but
    // integrating from a different initial condition.
    let original = ode(1e-9, 1e-11);
    let sim = Dopri5OdeSimulator::new(
        SimDescriptor::new("scirust-solvers/ode-dopri5", SemVer::new(1, 0, 0)),
        decay as fn(f64, &[f64], &mut [f64]),
    );
    let different = sim
        .run(
            &AdaptiveOdeConfig::new(0.0, 1.0, vec![2.0], 1e-9, 1e-11, 0.05),
            0,
        )
        .unwrap()
        .output;
    let agreement = certify_trajectory_endpoint(&original, &different).unwrap();
    assert!(!agreement.agrees);
    assert!(agreement.deviation > agreement.allowed);
}

#[test]
fn a_quadrature_rerun_at_a_different_tolerance_reproduces() {
    let loose = integral(1e-6);
    let tight = integral(1e-11);
    let agreement = certify_integral(&loose, &tight).unwrap();
    assert!(agreement.agrees);
    // Both really did compute ∫₀^π sin = 2.
    for c in [&loose, &tight]
    {
        assert!((c.value - 2.0).abs() < 1e-5);
    }
}

#[test]
fn the_verdict_reaches_the_reproduction_contract_for_a_real_run() {
    // End to end: a real L2 re-run, certified by this backend, decided by
    // `sos-repro`'s contract — the seam that crate declined to fill itself.
    let claim = NodeClaim::new(
        ObjectId::compute(HashAlgo::Sha256, b"t", b"decay-node"),
        DeterminismLevel::L2,
    );
    let verdict = certify_trajectory_endpoint(&ode(1e-6, 1e-8), &ode(1e-9, 1e-11))
        .unwrap()
        .verdict();
    assert_eq!(verify_node(&claim, &verdict), NodeVerdict::Reproduced);
}

#[test]
fn an_l1_estimate_from_the_nested_mc_backend_certifies_against_itself() {
    // `NestedMcEigEstimator` reports a real, computed standard error, which
    // is exactly what the in-distribution test consumes. Under the same seed
    // it is bit-reproducible, so the two-sample test — the *weaker* check —
    // must certainly pass.
    use scirust_stats::{DiscreteDistribution, Poisson};
    use sos_scirust::nmc::NestedMcEigEstimator;

    let estimator = NestedMcEigEstimator::uniform(2, 4_096).expect("a valid two-hypothesis setup");
    let (lo, hi) = (Poisson::new(2.0), Poisson::new(6.0));
    let models: Vec<&dyn DiscreteDistribution> = vec![&lo, &hi];

    let a = estimator
        .estimate(&models, 0xC0FFEE)
        .expect("estimation must succeed");
    let b = estimator
        .estimate(&models, 0xC0FFEE)
        .expect("estimation must succeed");
    assert_eq!(a.level, DeterminismLevel::L1);
    assert!(
        a.se_milli > 0,
        "the standard error must be real, not asserted"
    );
    assert!(
        certify_estimate_in_distribution(&a, &b, DEFAULT_SIGMA)
            .unwrap()
            .agrees
    );

    // A different seed draws different samples, and the two must still be
    // indistinguishable at 3 sigma — that is what L1 reproducibility means.
    let c = estimator
        .estimate(&models, 0xBEEF)
        .expect("estimation must succeed");
    let across_seeds = certify_estimate_in_distribution(&a, &c, DEFAULT_SIGMA).unwrap();
    assert!(
        across_seeds.agrees,
        "deviation {} exceeded the {} allowed",
        across_seeds.deviation, across_seeds.allowed
    );
}
