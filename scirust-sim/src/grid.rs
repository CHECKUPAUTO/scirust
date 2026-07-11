//! A power-system plant: the **swing equation** of a synchronous machine tied
//! to an infinite bus — the electromechanical dynamics behind transient
//! stability, which the `scirust-grid` vertical monitors (RoCoF, angle). Here
//! it is a simulator.
//!
//! Rotor-angle `δ` (rad), a [`SecondOrderSystem`](crate::engine::SecondOrderSystem):
//!
//! `δ'' = (ω_s / 2H)·(P_m − P_max·sin δ) − (D / 2H)·δ'`
//!
//! with synchronous speed `ω_s = 2π·f₀`, inertia constant `H` (s), mechanical
//! power `P_m` and peak electrical power `P_max` (per-unit), and damping `D`.
//! The stable equilibrium is `δ* = asin(P_m/P_max)`; small perturbations there
//! oscillate at `ω_n = √((ω_s/2H)·P_max·cos δ*)` (the ~1 Hz electromechanical
//! mode). With `D = 0` the transient energy
//! `E = ½·δ'² − (ω_s/2H)·(P_m·δ + P_max·cos δ)` is conserved — the tests'
//! oracles. When `P_m > P_max` no equilibrium exists (loss of synchronism).

use crate::engine::{SecondOrderSystem, SimError};

/// Single-machine–infinite-bus swing-equation model.
#[derive(Debug, Clone, PartialEq)]
pub struct SwingEquation {
    omega_s: f64,
    inertia_h: f64,
    damping: f64,
    p_mech: f64,
    p_max: f64,
}

impl SwingEquation {
    /// Create the model. `frequency_hz` and `inertia_h` must be finite and
    /// positive, `p_max` finite and positive, `p_mech` finite, `damping`
    /// finite and non-negative.
    pub fn new(
        frequency_hz: f64,
        inertia_h: f64,
        damping: f64,
        p_mech: f64,
        p_max: f64,
    ) -> Result<Self, SimError> {
        if !(frequency_hz.is_finite() && frequency_hz > 0.0)
        {
            return Err(SimError::BadInput(format!(
                "frequency_hz = {frequency_hz} must be finite and positive"
            )));
        }
        if !(inertia_h.is_finite() && inertia_h > 0.0)
        {
            return Err(SimError::BadInput(format!(
                "inertia_h = {inertia_h} must be finite and positive"
            )));
        }
        if !(damping.is_finite() && damping >= 0.0)
        {
            return Err(SimError::BadInput(format!(
                "damping = {damping} must be finite and non-negative"
            )));
        }
        if !p_mech.is_finite()
        {
            return Err(SimError::BadInput(format!(
                "p_mech = {p_mech} must be finite"
            )));
        }
        if !(p_max.is_finite() && p_max > 0.0)
        {
            return Err(SimError::BadInput(format!(
                "p_max = {p_max} must be finite and positive"
            )));
        }
        Ok(SwingEquation {
            omega_s: 2.0 * std::f64::consts::PI * frequency_hz,
            inertia_h,
            damping,
            p_mech,
            p_max,
        })
    }

    /// The stable equilibrium rotor angle `δ* = asin(P_m/P_max) ∈ [−π/2, π/2]`,
    /// or `None` when `|P_m| > P_max` (no synchronous operating point).
    pub fn equilibrium_angle(&self) -> Option<f64> {
        let ratio = self.p_mech / self.p_max;
        if ratio.abs() > 1.0
        {
            return None;
        }
        Some(ratio.asin())
    }

    /// The small-signal oscillation frequency `ω_n` (rad/s) about the stable
    /// equilibrium, or `None` when there is no equilibrium.
    pub fn small_signal_frequency(&self) -> Option<f64> {
        let delta_eq = self.equilibrium_angle()?;
        Some((self.omega_s / (2.0 * self.inertia_h) * self.p_max * delta_eq.cos()).sqrt())
    }

    /// The conserved transient energy `½·δ'² − (ω_s/2H)·(P_m·δ + P_max·cos δ)`
    /// of a state `q = [δ]`, `v = [δ']` (an invariant only when `D = 0`), or
    /// `None` when either slice is not length 1.
    pub fn energy(&self, q: &[f64], v: &[f64]) -> Option<f64> {
        let ([delta], [omega]) = (<[f64; 1]>::try_from(q).ok()?, <[f64; 1]>::try_from(v).ok()?);
        let k = self.omega_s / (2.0 * self.inertia_h);
        Some(0.5 * omega * omega - k * (self.p_mech * delta + self.p_max * delta.cos()))
    }
}

impl SecondOrderSystem for SwingEquation {
    fn dof(&self) -> usize {
        1
    }

    fn acceleration(&self, _t: f64, q: &[f64], v: &[f64], acc: &mut [f64]) {
        let k = self.omega_s / (2.0 * self.inertia_h);
        acc[0] = k * (self.p_mech - self.p_max * q[0].sin())
            - self.damping / (2.0 * self.inertia_h) * v[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FirstOrderForm, simulate, simulate_second_order};

    fn machine(damping: f64) -> SwingEquation {
        // 50 Hz, H = 5 s, P_m = 1.0 pu into a P_max = 2.0 pu line.
        SwingEquation::new(50.0, 5.0, damping, 1.0, 2.0).unwrap()
    }

    #[test]
    fn equilibrium_is_a_fixed_point_and_its_frequency_is_physical() {
        let sys = machine(0.5);
        let delta_eq = sys.equilibrium_angle().unwrap();
        assert!((delta_eq - std::f64::consts::FRAC_PI_6).abs() < 1e-12); // asin(0.5) = π/6
        // The acceleration vanishes at (δ*, 0).
        let mut acc = [0.0];
        sys.acceleration(0.0, &[delta_eq], &[0.0], &mut acc);
        assert!(acc[0].abs() < 1e-12, "acc at equilibrium = {}", acc[0]);
        // ω_n ≈ 7.38 rad/s (~1.17 Hz electromechanical mode).
        let wn = sys.small_signal_frequency().unwrap();
        assert!((wn - 7.376).abs() < 0.01, "ω_n = {wn}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn undamped_small_oscillation_conserves_energy_and_returns() {
        let sys = machine(0.0);
        let delta_eq = sys.equilibrium_angle().unwrap();
        let wn = sys.small_signal_frequency().unwrap();
        let period = 2.0 * std::f64::consts::PI / wn;
        let q0 = [delta_eq + 0.02];
        let traj = simulate_second_order(&sys, &q0, &[0.0], 0.0, period, period / 8000.0).unwrap();
        // Energy is an exact invariant of the undamped swing equation.
        let e0 = sys.energy(&q0, &[0.0]).unwrap();
        for row in &traj.y
        {
            let e = sys.energy(&row[..1], &row[1..]).unwrap();
            assert!(
                (e - e0).abs() < 1e-6 * e0.abs().max(1.0),
                "energy drifted to {e}"
            );
        }
        // After one small-signal period the angle is back near its start.
        let last = traj.last_state().unwrap();
        assert!((last[0] - q0[0]).abs() < 2e-3, "δ(T) = {}", last[0]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn damping_settles_the_rotor_to_the_equilibrium() {
        let sys = machine(12.0);
        let delta_eq = sys.equilibrium_angle().unwrap();
        // Kick the rotor and let damping bring it back.
        let traj = simulate(
            &FirstOrderForm(&sys),
            &[delta_eq + 0.4, 0.0],
            0.0,
            20.0,
            0.001,
        )
        .unwrap();
        let last = traj.last_state().unwrap();
        assert!(
            (last[0] - delta_eq).abs() < 1e-3,
            "δ = {}, δ* = {delta_eq}",
            last[0]
        );
        assert!(last[1].abs() < 1e-3, "ω = {}", last[1]);
    }

    #[test]
    fn loss_of_synchronism_and_validation() {
        // P_m > P_max: no synchronous operating point.
        let unstable = SwingEquation::new(50.0, 5.0, 1.0, 3.0, 2.0).unwrap();
        assert!(unstable.equilibrium_angle().is_none());
        assert!(unstable.small_signal_frequency().is_none());
        // Constructor validation.
        assert!(SwingEquation::new(0.0, 5.0, 1.0, 1.0, 2.0).is_err());
        assert!(SwingEquation::new(50.0, -5.0, 1.0, 1.0, 2.0).is_err());
        assert!(SwingEquation::new(50.0, 5.0, -1.0, 1.0, 2.0).is_err());
        assert!(SwingEquation::new(50.0, 5.0, 1.0, 1.0, 0.0).is_err());
        assert!(SwingEquation::new(50.0, 5.0, 1.0, f64::NAN, 2.0).is_err());
        let sys = machine(0.5);
        assert!(sys.energy(&[0.1, 0.2], &[0.0]).is_none());
    }
}
