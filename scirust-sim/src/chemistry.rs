//! Chemical kinetics models: consecutive first-order reactions (with the
//! Bateman closed form as the oracle) and a reversible reaction relaxing to
//! its equilibrium constant.
//!
//! These models are non-stiff by construction. Genuinely stiff kinetics
//! (e.g. the Robertson problem, with rate constants spanning nine orders of
//! magnitude) should be adapted to `scirust-stiff`'s implicit integrators
//! instead — see the crate-level interoperability notes.

use crate::engine::{SimError, System};

fn check_rate(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite and positive"
        )))
    }
}

/// Consecutive first-order reactions `A →(k₁) B →(k₂) C`, state
/// `y = [a, b, c]` in concentrations:
///
/// `a' = -k₁·a`, `b' = k₁·a - k₂·b`, `c' = k₂·b`.
///
/// The closed-form (Bateman) solution is exposed by
/// [`exact`](ConsecutiveReactions::exact) for `k₁ ≠ k₂`; total mass
/// `a + b + c` is a linear invariant.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsecutiveReactions {
    k1: f64,
    k2: f64,
}

impl ConsecutiveReactions {
    /// Create the model; both rate constants must be finite and positive.
    pub fn new(k1: f64, k2: f64) -> Result<Self, SimError> {
        check_rate("k1", k1)?;
        check_rate("k2", k2)?;
        Ok(ConsecutiveReactions { k1, k2 })
    }

    /// The Bateman closed form `[a(t), b(t), c(t)]` from `a(0) = a0`,
    /// `b(0) = c(0) = 0`, or `None` when `a0` is not finite and non-negative
    /// or `k₁ = k₂` (the formula has a removable singularity there).
    pub fn exact(&self, a0: f64, t: f64) -> Option<[f64; 3]> {
        if !a0.is_finite() || a0 < 0.0 || self.k1 == self.k2
        {
            return None;
        }
        let a = a0 * (-self.k1 * t).exp();
        let b = a0 * self.k1 / (self.k2 - self.k1) * ((-self.k1 * t).exp() - (-self.k2 * t).exp());
        Some([a, b, a0 - a - b])
    }
}

impl System for ConsecutiveReactions {
    fn dim(&self) -> usize {
        3
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.k1 * y[0];
        dydt[1] = self.k1 * y[0] - self.k2 * y[1];
        dydt[2] = self.k2 * y[1];
    }
}

/// A reversible first-order reaction `A ⇌ B` with forward rate `k_f` and
/// backward rate `k_r`, state `y = [a, b]`:
///
/// `a' = -k_f·a + k_r·b`, `b' = k_f·a - k_r·b`.
///
/// The system relaxes exponentially at rate `k_f + k_r` toward the
/// equilibrium ratio `b/a = k_f/k_r` (the equilibrium constant).
#[derive(Debug, Clone, PartialEq)]
pub struct ReversibleReaction {
    kf: f64,
    kr: f64,
}

impl ReversibleReaction {
    /// Create the model; both rate constants must be finite and positive.
    pub fn new(kf: f64, kr: f64) -> Result<Self, SimError> {
        check_rate("kf", kf)?;
        check_rate("kr", kr)?;
        Ok(ReversibleReaction { kf, kr })
    }

    /// The equilibrium constant `K = k_f / k_r`.
    pub fn equilibrium_constant(&self) -> f64 {
        self.kf / self.kr
    }

    /// The closed-form solution `[a(t), b(t)]` from `a(0) = a0`,
    /// `b(0) = b0`, or `None` when an initial concentration is not finite
    /// and non-negative.
    pub fn exact(&self, a0: f64, b0: f64, t: f64) -> Option<[f64; 2]> {
        if !a0.is_finite() || a0 < 0.0 || !b0.is_finite() || b0 < 0.0
        {
            return None;
        }
        let total = a0 + b0;
        let a_eq = self.kr / (self.kf + self.kr) * total;
        let a = a_eq + (a0 - a_eq) * (-(self.kf + self.kr) * t).exp();
        Some([a, total - a])
    }
}

impl System for ReversibleReaction {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let net = self.kf * y[0] - self.kr * y[1];
        dydt[0] = -net;
        dydt[1] = net;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn consecutive_reactions_match_the_bateman_solution() {
        let sys = ConsecutiveReactions::new(2.0, 0.5).unwrap();
        let a0 = 1.5;
        let traj = simulate(&sys, &[a0, 0.0, 0.0], 0.0, 8.0, 0.001).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = sys.exact(a0, *t).unwrap();
            for k in 0..3
            {
                assert!(
                    (row[k] - exact[k]).abs() < 1e-8,
                    "t = {t}, component {k}: {} vs {}",
                    row[k],
                    exact[k]
                );
            }
        }
    }

    #[test]
    fn total_mass_is_conserved_and_intermediate_peaks() {
        let sys = ConsecutiveReactions::new(1.0, 3.0).unwrap();
        let traj = simulate(&sys, &[1.0, 0.0, 0.0], 0.0, 10.0, 0.005).unwrap();
        for row in &traj.y
        {
            assert!((row[0] + row[1] + row[2] - 1.0).abs() < 1e-12);
        }
        // The intermediate B rises then falls: its peak (analytically at
        // t* = ln(k2/k1)/(k2-k1)) is interior, and B ends near zero.
        let b = traj.column(1).unwrap();
        let peak = b.iter().cloned().fold(0.0, f64::max);
        assert!(peak > *b.first().unwrap() && peak > *b.last().unwrap());
        assert!(b.last().unwrap() < &1e-3);
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn reversible_reaction_relaxes_to_the_equilibrium_constant() {
        let sys = ReversibleReaction::new(1.2, 0.4).unwrap();
        let traj = simulate(&sys, &[1.0, 0.0], 0.0, 30.0, 0.005).unwrap();
        // Trajectory matches the closed form along the way.
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = sys.exact(1.0, 0.0, *t).unwrap();
            assert!((row[0] - exact[0]).abs() < 1e-9 && (row[1] - exact[1]).abs() < 1e-9);
        }
        // And the final ratio b/a is the equilibrium constant kf/kr = 3.
        let last = traj.last_state().unwrap();
        assert!((last[1] / last[0] - sys.equilibrium_constant()).abs() < 1e-6);
    }

    #[test]
    fn constructors_and_closed_forms_reject_bad_inputs() {
        assert!(ConsecutiveReactions::new(0.0, 1.0).is_err());
        assert!(ConsecutiveReactions::new(1.0, f64::NAN).is_err());
        assert!(ReversibleReaction::new(-1.0, 1.0).is_err());
        let sys = ConsecutiveReactions::new(1.0, 1.0).unwrap();
        assert!(sys.exact(1.0, 0.5).is_none()); // k1 = k2: removable singularity
        let sys = ConsecutiveReactions::new(1.0, 2.0).unwrap();
        assert!(sys.exact(-1.0, 0.5).is_none());
        let rev = ReversibleReaction::new(1.0, 1.0).unwrap();
        assert!(rev.exact(f64::NAN, 0.0, 1.0).is_none());
    }
}
