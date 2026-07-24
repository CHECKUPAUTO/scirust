//! Integration tests: the adaptive DOPRI5 `Simulate` backend (gap #3, second
//! backend), exercised through the public API only.

use sos_core::{DeterminismLevel, SemVer};
use sos_scirust::Dopri5OdeSimulator;
use sos_scirust::ode::AdaptiveOdeConfig;
use sos_simulation::{SimDescriptor, SimError, Simulate};

fn descriptor(name: &str) -> SimDescriptor {
    SimDescriptor::new(name, SemVer::new(1, 0, 0))
}

#[test]
fn a_real_adaptive_integration_is_l2_with_a_real_certificate() {
    // The pendulum: dy0/dt = y1, dy1/dt = -g sin(y0). No closed form, but
    // energy is conserved, so energy drift bounds the integration's honesty.
    let g = 9.81;
    let sim = Dopri5OdeSimulator::new(descriptor("test/pendulum"), move |_t, y, dy| {
        dy[0] = y[1];
        dy[1] = -g * y[0].sin();
    });
    let config = AdaptiveOdeConfig::new(0.0, 10.0, vec![0.5, 0.0], 1e-8, 1e-10, 0.05);
    let obs = sim.run(&config, 0).unwrap();

    assert_eq!(obs.level(), DeterminismLevel::L2);
    assert_eq!(obs.output.rtol, 1e-8);
    assert_eq!(obs.output.atol, 1e-10);
    assert!(obs.output.accepted_steps > 0);

    let e0 = 0.5_f64.cos().mul_add(-g, g);
    let (_, y_final) = obs.output.trajectory.last().unwrap();
    let e_final = y_final[1].powi(2) / 2.0 + g * (1.0 - y_final[0].cos());
    assert!(
        (e_final - e0).abs() / e0 < 1e-6,
        "energy drift too large: e0={e0}, e={e_final}"
    );
}

#[test]
fn tighter_tolerance_still_converges_and_stays_bit_reproducible() {
    let sim = Dopri5OdeSimulator::new(descriptor("test/tight-tol"), |_t, y, dy| dy[0] = -y[0]);
    let loose = AdaptiveOdeConfig::new(0.0, 3.0, vec![1.0], 1e-4, 1e-6, 0.1);
    let tight = AdaptiveOdeConfig::new(0.0, 3.0, vec![1.0], 1e-10, 1e-12, 0.1);

    let a = sim.run(&loose, 7).unwrap();
    let b = sim.run(&tight, 7).unwrap();

    // Both converge to the true solution, the tighter one more closely.
    let exact = (-3.0_f64).exp();
    let loose_err = (a.output.trajectory.last().unwrap().1[0] - exact).abs();
    let tight_err = (b.output.trajectory.last().unwrap().1[0] - exact).abs();
    assert!(tight_err <= loose_err);

    // Re-running the tight config is bit-reproducible.
    let b2 = sim.run(&tight, 7).unwrap();
    assert_eq!(b, b2);
}

#[test]
fn invalid_tolerances_and_intervals_are_clean_config_errors() {
    let sim = Dopri5OdeSimulator::new(descriptor("test/dopri5-errors"), |_t, y, dy| {
        dy[0] = -y[0];
    });

    // t_end == t0 is rejected (dopri5 requires t_end > t0, strictly).
    let bad_interval = AdaptiveOdeConfig::new(1.0, 1.0, vec![1.0], 1e-6, 1e-9, 0.1);
    assert!(matches!(
        sim.run(&bad_interval, 0),
        Err(SimError::InvalidConfig(_))
    ));

    // A negative tolerance is rejected too.
    let bad_tol = AdaptiveOdeConfig::new(0.0, 1.0, vec![1.0], -1.0, 1e-9, 0.1);
    assert!(matches!(
        sim.run(&bad_tol, 0),
        Err(SimError::InvalidConfig(_))
    ));
}
