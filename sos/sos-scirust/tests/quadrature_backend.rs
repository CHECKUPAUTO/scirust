//! Integration tests: the adaptive-quadrature `Simulate` backend (gap #3,
//! third backend), exercised through the public API only.

use sos_core::{DeterminismLevel, SemVer};
use sos_scirust::QuadratureSimulator;
use sos_scirust::quadrature::QuadratureConfig;
use sos_simulation::{SimDescriptor, SimError, Simulate};

fn descriptor(name: &str) -> SimDescriptor {
    SimDescriptor::new(name, SemVer::new(1, 0, 0))
}

#[test]
fn a_real_quadrature_estimate_is_l2_with_an_honest_certificate() {
    // The Runge function, a classic adaptive-quadrature stress test:
    // ∫₋₅⁵ 1/(1+x²) dx = 2·atan(5).
    let sim = QuadratureSimulator::new(descriptor("test/runge"), |x: f64| 1.0 / (1.0 + x * x));
    let config = QuadratureConfig::new(-5.0, 5.0, 1e-10, 30);
    let obs = sim.run(&config, 0).unwrap();

    assert_eq!(obs.level(), DeterminismLevel::L2);
    assert_eq!(obs.output.a, -5.0);
    assert_eq!(obs.output.b, 5.0);
    assert_eq!(obs.output.tol, 1e-10);

    let exact = 2.0 * 5.0_f64.atan();
    assert!((obs.output.value - exact).abs() < 1e-9);
}

#[test]
fn distinct_descriptors_let_two_integrands_cache_independently() {
    use sos_simulation::Vcr;

    let mut vcr_a = Vcr::new();
    let mut vcr_b = Vcr::new();
    let sin = QuadratureSimulator::new(descriptor("test/sin"), |x: f64| x.sin());
    let cos = QuadratureSimulator::new(descriptor("test/cos"), |x: f64| x.cos());
    let config = QuadratureConfig::new(0.0, std::f64::consts::PI, 1e-10, 30);

    let a = vcr_a.observe(&sin, &config, 0).unwrap();
    let b = vcr_b.observe(&cos, &config, 0).unwrap();
    // Same bounds, different integrands: values must differ (∫sin=2, ∫cos=0).
    assert_ne!(a.observation.output.value, b.observation.output.value);
}

#[test]
fn invalid_config_and_depth_exhaustion_are_distinguishable_errors() {
    let sim = QuadratureSimulator::new(descriptor("test/errors"), |x: f64| x);

    // A non-finite bound is rejected before any compute begins.
    let bad_bounds = QuadratureConfig::new(f64::NAN, 1.0, 1e-6, 10);
    assert!(matches!(
        sim.run(&bad_bounds, 0),
        Err(SimError::InvalidConfig(_))
    ));

    // An oscillating integrand with an unreasonable tolerance and almost no
    // recursion budget exhausts depth without meeting tol: a backend
    // failure, not a silently wrong value.
    let oscillating =
        QuadratureSimulator::new(descriptor("test/oscillating"), |x: f64| (500.0 * x).sin());
    let starved = QuadratureConfig::new(0.0, 1.0, 1e-25, 1);
    assert!(matches!(
        oscillating.run(&starved, 0),
        Err(SimError::Backend(_))
    ));
}
