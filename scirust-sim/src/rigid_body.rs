//! Free rigid-body rotation: Euler's equations for a torque-free body spun
//! about its principal axes. State `y = [ω₁, ω₂, ω₃]` (the body-frame angular
//! velocity):
//!
//! `I₁·ω₁' = (I₂-I₃)·ω₂·ω₃`, `I₂·ω₂' = (I₃-I₁)·ω₃·ω₁`, `I₃·ω₃' = (I₁-I₂)·ω₁·ω₂`.
//!
//! With no external torque the rotational kinetic energy and the magnitude of
//! the angular momentum are both exact invariants — the oracles the tests use.
//! The model also reproduces two textbook phenomena: steady precession of a
//! symmetric top (closed form), and the intermediate-axis instability (the
//! tennis-racket / Dzhanibekov effect).

use crate::engine::{SimError, System};

/// A rigid body characterised by its three principal moments of inertia.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody {
    i1: f64,
    i2: f64,
    i3: f64,
}

impl RigidBody {
    /// Create the body; all three principal moments of inertia must be finite
    /// and positive.
    pub fn new(i1: f64, i2: f64, i3: f64) -> Result<Self, SimError> {
        for (name, value) in [("i1", i1), ("i2", i2), ("i3", i3)]
        {
            if !(value.is_finite() && value > 0.0)
            {
                return Err(SimError::BadInput(format!(
                    "{name} = {value} must be finite and positive"
                )));
            }
        }
        Ok(RigidBody { i1, i2, i3 })
    }

    /// Rotational kinetic energy `½·(I₁·ω₁² + I₂·ω₂² + I₃·ω₃²)` of an angular
    /// velocity `[ω₁, ω₂, ω₃]`, or `None` when the slice is not length 3.
    /// Conserved along any free-rotation trajectory.
    pub fn kinetic_energy(&self, omega: &[f64]) -> Option<f64> {
        let [w1, w2, w3] = *omega
        else
        {
            return None;
        };
        Some(0.5 * (self.i1 * w1 * w1 + self.i2 * w2 * w2 + self.i3 * w3 * w3))
    }

    /// Squared angular-momentum magnitude `I₁²·ω₁² + I₂²·ω₂² + I₃²·ω₃²`, or
    /// `None` when the slice is not length 3. Conserved along any
    /// free-rotation trajectory.
    pub fn angular_momentum_squared(&self, omega: &[f64]) -> Option<f64> {
        let [w1, w2, w3] = *omega
        else
        {
            return None;
        };
        Some(
            self.i1 * self.i1 * w1 * w1 + self.i2 * self.i2 * w2 * w2 + self.i3 * self.i3 * w3 * w3,
        )
    }

    /// The precession rate `Ω = (I₃-I₁)·ω₃/I₁` of a symmetric top (`I₁ = I₂`):
    /// the transverse angular velocity `(ω₁, ω₂)` rotates at this rate while
    /// `ω₃` stays constant. Returns `None` when the body is not symmetric
    /// about axis 3 (`I₁ ≠ I₂`).
    pub fn symmetric_precession_rate(&self, omega3: f64) -> Option<f64> {
        if self.i1 != self.i2
        {
            return None;
        }
        Some((self.i3 - self.i1) * omega3 / self.i1)
    }
}

impl System for RigidBody {
    fn dim(&self) -> usize {
        3
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = (self.i2 - self.i3) * y[1] * y[2] / self.i1;
        dydt[1] = (self.i3 - self.i1) * y[2] * y[0] / self.i2;
        dydt[2] = (self.i1 - self.i2) * y[0] * y[1] / self.i3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn free_rotation_conserves_energy_and_angular_momentum() {
        // A fully asymmetric top spun about a generic axis.
        let body = RigidBody::new(1.0, 2.0, 3.0).unwrap();
        let w0 = [1.0, 0.5, 0.3];
        let traj = simulate(&body, &w0, 0.0, 20.0, 0.001).unwrap();
        let e0 = body.kinetic_energy(&w0).unwrap();
        let l0 = body.angular_momentum_squared(&w0).unwrap();
        for row in &traj.y
        {
            let e = body.kinetic_energy(row).unwrap();
            let l = body.angular_momentum_squared(row).unwrap();
            assert!((e - e0).abs() < 1e-8 * e0, "energy drifted to {e}");
            assert!((l - l0).abs() < 1e-8 * l0, "|L|² drifted to {l}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn symmetric_top_precesses_at_the_closed_form_rate() {
        // I1 = I2 ≠ I3: ω3 is constant and (ω1, ω2) rotate at Ω.
        let body = RigidBody::new(1.0, 1.0, 2.0).unwrap();
        let omega3 = 1.5;
        let rate = body.symmetric_precession_rate(omega3).unwrap();
        assert!((rate - 1.5).abs() < 1e-12);
        let period = 2.0 * std::f64::consts::PI / rate.abs();
        let w0 = [0.3, 0.0, omega3];
        let traj = simulate(&body, &w0, 0.0, period, period / 8000.0).unwrap();
        // Transverse magnitude and ω3 are constant throughout.
        for row in &traj.y
        {
            assert!((row[0] * row[0] + row[1] * row[1] - 0.09).abs() < 1e-7);
            assert!((row[2] - omega3).abs() < 1e-9);
        }
        // After exactly one precession period the state returns.
        let last = traj.last_state().unwrap();
        assert!(
            (last[0] - 0.3).abs() < 1e-5 && last[1].abs() < 1e-5,
            "{last:?}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn intermediate_axis_is_unstable_but_the_extreme_axes_are_stable() {
        // I1 < I2 < I3: axis 2 is the intermediate axis.
        let body = RigidBody::new(1.0, 2.0, 3.0).unwrap();
        let perturb = 0.05;
        let max_off = |w0: [f64; 3], off: [usize; 2]| {
            let traj = simulate(&body, &w0, 0.0, 20.0, 0.002).unwrap();
            traj.y
                .iter()
                .map(|r| r[off[0]].abs().max(r[off[1]].abs()))
                .fold(0.0, f64::max)
        };
        // Spin about the intermediate axis: the tiny off-axis components grow
        // to order one — the body tumbles.
        let unstable = max_off([perturb, 2.0, perturb], [0, 2]);
        assert!(
            unstable > 0.5,
            "intermediate axis did not tumble: {unstable}"
        );
        // Spin about the minimum axis (1): off-axis components stay small.
        let stable_min = max_off([2.0, perturb, perturb], [1, 2]);
        assert!(stable_min < 0.15, "min axis was not stable: {stable_min}");
        // Spin about the maximum axis (3): off-axis components stay small.
        let stable_max = max_off([perturb, perturb, 2.0], [0, 1]);
        assert!(stable_max < 0.15, "max axis was not stable: {stable_max}");
    }

    #[test]
    fn constructors_and_helpers_reject_bad_inputs() {
        assert!(RigidBody::new(0.0, 2.0, 3.0).is_err());
        assert!(RigidBody::new(1.0, -2.0, 3.0).is_err());
        assert!(RigidBody::new(1.0, 2.0, f64::NAN).is_err());
        let body = RigidBody::new(1.0, 2.0, 3.0).unwrap();
        assert!(body.kinetic_energy(&[1.0, 2.0]).is_none());
        assert!(
            body.angular_momentum_squared(&[1.0, 2.0, 3.0, 4.0])
                .is_none()
        );
        // Not symmetric about axis 3: no single precession rate.
        assert!(body.symmetric_precession_rate(1.0).is_none());
        let symmetric = RigidBody::new(1.0, 1.0, 2.0).unwrap();
        assert!(symmetric.symmetric_precession_rate(1.0).is_some());
    }
}
