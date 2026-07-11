//! Bridge exposing this crate's glucose-insulin dynamics as a
//! [`scirust_sim::System`] (enabled by the optional `sim` feature).
//!
//! The [`barrier`](super::barrier) module models the *plant* the safety filter
//! controls — the affine glucose dynamics `dG/dt = -a·(G - G_b) - k·u` — but
//! only ever evaluates its instantaneous derivative inside the CBF constraint.
//! This module wraps those same parameters ([`GlucoseModel`]) into a
//! [`GlucoseSystem`] that implements the shared `y' = f(t, y)` trait, so
//! `scirust-sim`'s engine (RK4, Dormand–Prince, …) can integrate the plant
//! forward in time directly.
//!
//! It is the "reverse direction" of the simulation layer: instead of
//! `scirust-sim` re-declaring a vertical's physics, the vertical exposes its
//! own model through the shared trait. The default build is unaffected — the
//! `scirust-sim` dependency is pulled in only under the `sim` feature.
//!
//! The same non-clinical-use caveat as [`super::barrier`] applies: this is a
//! one-compartment pedagogical model, not a validated physiological simulator.
//!
//! ```
//! # // (requires the `sim` feature)
//! use scirust_biomed::control::GlucoseModel;
//! use scirust_biomed::control::sim::GlucoseSystem;
//! use scirust_sim::simulate;
//!
//! // Reversion a = 0.1/min, basal G_b = 120, sensitivity k = 1.0.
//! let model = GlucoseModel {
//!     reversion_rate: 0.1,
//!     basal_target: 120.0,
//!     insulin_sensitivity: 1.0,
//! };
//! // A constant 2.0 units/rate infusion pulls the steady state down to
//! // G* = G_b - (k/a)·u = 120 - 10·2 = 100.
//! let sys = GlucoseSystem::new(model, 2.0);
//! let traj = simulate(&sys, &[180.0], 0.0, 200.0, 0.01).expect("integrates");
//! assert!((traj.last_state().unwrap()[0] - 100.0).abs() < 1e-3);
//! ```

use super::barrier::GlucoseModel;
use scirust_sim::System;

/// The affine glucose dynamics `dG/dt = -a·(G - G_b) - k·u` as a steppable
/// [`System`], driven by a constant exogenous insulin infusion `u`. State is
/// the single glucose level `y = [G]`.
///
/// Wrapping the plant this way lets `scirust-biomed`'s own physiological model
/// be integrated by the shared `scirust-sim` engine, rather than having
/// `scirust-sim` re-declare it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlucoseSystem {
    /// The plant parameters (reversion rate `a`, basal target `G_b`, insulin
    /// sensitivity `k`).
    pub model: GlucoseModel,
    /// The constant insulin infusion rate `u ≥ 0` held over the integration.
    pub insulin_rate: f64,
}

impl GlucoseSystem {
    /// Wrap a [`GlucoseModel`] as a steppable system driven by a constant
    /// insulin infusion rate.
    pub fn new(model: GlucoseModel, insulin_rate: f64) -> Self {
        Self {
            model,
            insulin_rate,
        }
    }

    /// The steady-state glucose `G* = G_b - (k/a)·u` the system relaxes to
    /// (well-defined when the reversion rate `a` is non-zero).
    pub fn steady_state(&self) -> f64 {
        self.model.basal_target
            - self.model.insulin_sensitivity / self.model.reversion_rate * self.insulin_rate
    }

    /// The exact closed-form solution `G(t) = G* + (G0 - G*)·e^{-a·t}` from an
    /// initial glucose `g0` — the linear ODE integrates in closed form, which
    /// the tests use as an oracle for the numerical integrator.
    pub fn exact(&self, g0: f64, t: f64) -> f64 {
        let g_star = self.steady_state();
        g_star + (g0 - g_star) * (-self.model.reversion_rate * t).exp()
    }
}

impl System for GlucoseSystem {
    fn dim(&self) -> usize {
        1
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.model.reversion_rate * (y[0] - self.model.basal_target)
            - self.model.insulin_sensitivity * self.insulin_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_sim::simulate;

    fn model() -> GlucoseModel {
        GlucoseModel {
            reversion_rate: 0.1,
            basal_target: 120.0,
            insulin_sensitivity: 1.0,
        }
    }

    #[test]
    fn matches_the_closed_form_solution_under_constant_insulin() {
        // Constant u = 2.0 ⇒ steady state G* = 120 - (1.0/0.1)·2 = 100.
        let sys = GlucoseSystem::new(model(), 2.0);
        assert!((sys.steady_state() - 100.0).abs() < 1e-12);

        let traj = simulate(&sys, &[180.0], 0.0, 120.0, 0.001).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            assert!(
                (row[0] - sys.exact(180.0, *t)).abs() < 1e-6,
                "t = {t}: {} vs {}",
                row[0],
                sys.exact(180.0, *t)
            );
        }
        // And it has essentially reached the steady state by the end.
        assert!((traj.last_state().unwrap()[0] - 100.0).abs() < 1e-2);
    }

    #[test]
    fn zero_insulin_relaxes_to_the_basal_target() {
        // With u = 0 the plant reverts to G_b regardless of where it starts.
        let sys = GlucoseSystem::new(model(), 0.0);
        assert!((sys.steady_state() - 120.0).abs() < 1e-12);

        let traj = simulate(&sys, &[200.0], 0.0, 120.0, 0.001).unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - 120.0).abs() < 1e-2, "G(120) = {}", last[0]);
        // Monotone decay from above (no insulin ⇒ glucose only falls toward G_b).
        assert!(
            traj.y.windows(2).all(|w| w[1][0] <= w[0][0] + 1e-9),
            "not monotone"
        );
    }

    #[test]
    fn derivative_has_the_expected_shape() {
        let sys = GlucoseSystem::new(model(), 2.0);
        assert_eq!(sys.dim(), 1);
        let mut d = [0.0];
        // At the steady state the derivative vanishes.
        sys.derivatives(0.0, &[sys.steady_state()], &mut d);
        assert!(d[0].abs() < 1e-12, "dG/dt at G* = {}", d[0]);
        // Above the steady state the glucose falls.
        sys.derivatives(0.0, &[sys.steady_state() + 10.0], &mut d);
        assert!(d[0] < 0.0);
    }
}
