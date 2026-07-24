//! Integration tests: the RK4 `Simulate` backend (gap #3, first backend),
//! exercised through the public API only — a caller with an ODE right-hand
//! side, a config, and (for one test) a `Vcr`.

use sos_core::{DeterminismLevel, SemVer};
use sos_scirust::Rk4OdeSimulator;
use sos_scirust::ode::OdeConfig;
use sos_simulation::{SimDescriptor, SimError, Simulate, Vcr};

fn descriptor(name: &str) -> SimDescriptor {
    SimDescriptor::new(name, SemVer::new(1, 0, 0))
}

#[test]
fn a_real_ode_integration_is_l3_and_seed_independent_in_value() {
    // dy/dt = -y, y(0) = 2: exact solution y(t) = 2 e^{-t}.
    let sim = Rk4OdeSimulator::new(descriptor("test/decay"), |_t, y, dy| dy[0] = -y[0]);
    let config = OdeConfig::new(0.0, 2.0, vec![2.0], 0.001);

    let obs_a = sim.run(&config, 0).unwrap();
    let obs_b = sim.run(&config, 999).unwrap();

    assert_eq!(obs_a.level(), DeterminismLevel::L3);
    // Seedless-deterministic: the physics doesn't depend on the seed at all.
    assert_eq!(obs_a.output, obs_b.output);

    let (t_final, y_final) = obs_a.output.last().unwrap();
    assert!((t_final - 2.0).abs() < 1e-9);
    assert!((y_final[0] - 2.0 * (-2.0_f64).exp()).abs() < 1e-5);
}

#[test]
fn distinct_descriptors_let_two_models_cache_independently() {
    let mut vcr_a = Vcr::new();
    let mut vcr_b = Vcr::new();

    let decay = Rk4OdeSimulator::new(descriptor("test/model-a"), |_t, y, dy| dy[0] = -y[0]);
    let growth = Rk4OdeSimulator::new(descriptor("test/model-b"), |_t, y, dy| dy[0] = y[0]);
    let config = OdeConfig::new(0.0, 1.0, vec![1.0], 0.01);

    let a = vcr_a.observe(&decay, &config, 0).unwrap();
    let b = vcr_b.observe(&growth, &config, 0).unwrap();
    // Same config, opposite dynamics: outputs must differ.
    assert_ne!(a.observation.output, b.observation.output);
}

#[test]
fn invalid_and_failing_configs_are_distinguishable_errors() {
    let sim = Rk4OdeSimulator::new(descriptor("test/errors"), |_t, y, dy| dy[0] = -y[0]);

    // t_end < t0 is rejected before any compute begins.
    let bad_interval = OdeConfig::new(1.0, 0.0, vec![1.0], 0.1);
    assert!(matches!(
        sim.run(&bad_interval, 0),
        Err(SimError::InvalidConfig(_))
    ));

    // A right-hand side that blows up mid-integration is a backend failure.
    let blowup = Rk4OdeSimulator::new(descriptor("test/blowup"), |_t, _y, dy| {
        dy[0] = f64::MAX;
    });
    let config = OdeConfig::new(0.0, 1.0, vec![1.0], 1e-3);
    assert!(matches!(blowup.run(&config, 0), Err(SimError::Backend(_))));
}
