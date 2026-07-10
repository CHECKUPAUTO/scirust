//! Electrical circuit models: RC charging (closed form) and the series RLC
//! circuit (underdamped closed form + passivity of the stored energy).

use crate::engine::{SimError, System};

fn check_positive(name: &str, value: f64) -> Result<(), SimError> {
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

fn check_non_negative(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value >= 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite and non-negative"
        )))
    }
}

/// An RC circuit charging from a constant source:
/// `v_C' = (V - v_C)/(R·C)`, state `y = [v_C]`, with the closed form
/// `v_C(t) = V + (v₀ - V)·e^{-t/RC}`.
#[derive(Debug, Clone, PartialEq)]
pub struct RcCircuit {
    resistance: f64,
    capacitance: f64,
    v_source: f64,
}

impl RcCircuit {
    /// Create the model; `resistance` and `capacitance` must be finite and
    /// positive, `v_source` finite.
    pub fn new(resistance: f64, capacitance: f64, v_source: f64) -> Result<Self, SimError> {
        check_positive("resistance", resistance)?;
        check_positive("capacitance", capacitance)?;
        if !v_source.is_finite()
        {
            return Err(SimError::BadInput(format!(
                "v_source = {v_source} must be finite"
            )));
        }
        Ok(RcCircuit {
            resistance,
            capacitance,
            v_source,
        })
    }

    /// The time constant `τ = R·C`.
    pub fn time_constant(&self) -> f64 {
        self.resistance * self.capacitance
    }

    /// The closed-form capacitor voltage at time `t` from `v0`, or `None`
    /// when `v0` is not finite.
    pub fn exact(&self, v0: f64, t: f64) -> Option<f64> {
        if !v0.is_finite()
        {
            return None;
        }
        Some(self.v_source + (v0 - self.v_source) * (-t / self.time_constant()).exp())
    }
}

impl System for RcCircuit {
    fn dim(&self) -> usize {
        1
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = (self.v_source - y[0]) / self.time_constant();
    }
}

/// A series RLC circuit driven by a constant source:
/// `q' = i`, `i' = (V - R·i - q/C)/L`, state `y = [q, i]` (charge, current).
///
/// With `V = 0` this is the damped harmonic oscillator of circuit theory;
/// the stored energy `q²/(2C) + L·i²/2` can only be dissipated in the
/// resistor (`dE/dt = -R·i² ≤ 0`), the passivity oracle used in the tests.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesRlc {
    resistance: f64,
    inductance: f64,
    capacitance: f64,
    v_source: f64,
}

impl SeriesRlc {
    /// Create the model; `inductance` and `capacitance` must be finite and
    /// positive, `resistance` finite and non-negative, `v_source` finite.
    pub fn new(
        resistance: f64,
        inductance: f64,
        capacitance: f64,
        v_source: f64,
    ) -> Result<Self, SimError> {
        check_non_negative("resistance", resistance)?;
        check_positive("inductance", inductance)?;
        check_positive("capacitance", capacitance)?;
        if !v_source.is_finite()
        {
            return Err(SimError::BadInput(format!(
                "v_source = {v_source} must be finite"
            )));
        }
        Ok(SeriesRlc {
            resistance,
            inductance,
            capacitance,
            v_source,
        })
    }

    /// Undamped natural frequency `ω₀ = 1/√(L·C)`.
    pub fn natural_frequency(&self) -> f64 {
        1.0 / (self.inductance * self.capacitance).sqrt()
    }

    /// Damping ratio `ζ = (R/2)·√(C/L)`.
    pub fn damping_ratio(&self) -> f64 {
        0.5 * self.resistance * (self.capacitance / self.inductance).sqrt()
    }

    /// Stored energy `q²/(2C) + L·i²/2` of a state `[q, i]`, or `None` when
    /// the state does not have length 2.
    pub fn energy(&self, state: &[f64]) -> Option<f64> {
        let [q, i] = *state
        else
        {
            return None;
        };
        Some(q * q / (2.0 * self.capacitance) + 0.5 * self.inductance * i * i)
    }
}

impl System for SeriesRlc {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[1];
        dydt[1] =
            (self.v_source - self.resistance * y[1] - y[0] / self.capacitance) / self.inductance;
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
    fn rc_charging_matches_the_closed_form() {
        let rc = RcCircuit::new(1_000.0, 1e-4, 5.0).unwrap(); // τ = 0.1 s
        assert!((rc.time_constant() - 0.1).abs() < 1e-15);
        let traj = simulate(&rc, &[0.0], 0.0, 0.6, 1e-4).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = rc.exact(0.0, *t).unwrap();
            assert!(
                (row[0] - exact).abs() < 1e-9,
                "t = {t}: {} vs {exact}",
                row[0]
            );
        }
        // After 6 time constants the capacitor is charged to within 0.25%.
        assert!((traj.last_state().unwrap()[0] - 5.0).abs() < 0.0125);
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn underdamped_rlc_matches_the_damped_oscillator_closed_form() {
        // R = 0.4 Ω, L = 1 H, C = 0.25 F: ω₀ = 2, ζ = 0.1 (underdamped).
        let rlc = SeriesRlc::new(0.4, 1.0, 0.25, 0.0).unwrap();
        let (w0, zeta) = (rlc.natural_frequency(), rlc.damping_ratio());
        assert!((w0 - 2.0).abs() < 1e-12 && (zeta - 0.1).abs() < 1e-12);
        let wd = w0 * (1.0 - zeta * zeta).sqrt();
        let q0 = 1.0;
        let traj = simulate(&rlc, &[q0, 0.0], 0.0, 10.0, 0.001).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = (-zeta * w0 * t).exp()
                * (q0 * (wd * t).cos() + zeta * w0 * q0 / wd * (wd * t).sin());
            assert!(
                (row[0] - exact).abs() < 1e-6,
                "t = {t}: {} vs {exact}",
                row[0]
            );
        }
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn rlc_stored_energy_never_increases_without_a_source() {
        let rlc = SeriesRlc::new(0.5, 1.0, 0.5, 0.0).unwrap();
        let traj = simulate(&rlc, &[1.0, 0.0], 0.0, 20.0, 0.001).unwrap();
        let energies: Vec<f64> = traj.y.iter().map(|row| rlc.energy(row).unwrap()).collect();
        assert!(
            energies.windows(2).all(|w| w[1] <= w[0] + 1e-12),
            "passivity violated"
        );
        // And with R > 0 it actually dissipates.
        assert!(energies.last().unwrap() < &(0.01 * energies[0]));
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn lossless_lc_conserves_energy() {
        let lc = SeriesRlc::new(0.0, 1.0, 0.25, 0.0).unwrap();
        let traj = simulate(&lc, &[1.0, 0.0], 0.0, 20.0, 0.001).unwrap();
        let e0 = lc.energy(&traj.y[0]).unwrap();
        for row in &traj.y
        {
            let e = lc.energy(row).unwrap();
            assert!((e - e0).abs() < 1e-9 * e0, "energy drifted to {e}");
        }
    }

    #[test]
    fn constructors_and_helpers_reject_bad_inputs() {
        assert!(RcCircuit::new(0.0, 1e-4, 5.0).is_err());
        assert!(RcCircuit::new(1_000.0, 1e-4, f64::NAN).is_err());
        assert!(SeriesRlc::new(-0.1, 1.0, 0.25, 0.0).is_err());
        assert!(SeriesRlc::new(0.4, 0.0, 0.25, 0.0).is_err());
        assert!(SeriesRlc::new(0.4, 1.0, f64::INFINITY, 0.0).is_err());
        let rc = RcCircuit::new(1.0, 1.0, 1.0).unwrap();
        assert!(rc.exact(f64::NAN, 1.0).is_none());
        let rlc = SeriesRlc::new(0.4, 1.0, 0.25, 0.0).unwrap();
        assert!(rlc.energy(&[1.0]).is_none());
    }
}
