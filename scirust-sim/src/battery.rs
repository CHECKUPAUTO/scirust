//! A lithium-ion battery plant: the Thévenin equivalent-circuit model (one RC
//! polarization branch) coupled to a lumped self-heating thermal state. This
//! is the plant the `scirust-bms` vertical estimates on; here it is a
//! *simulator* — feed it a constant current and integrate.
//!
//! State `y = [soc, v_rc, temp]`:
//!
//! - `soc' = -I / (capacity·3600)` — coulomb counting (`capacity` in A·h, `I`
//!   in A, positive on discharge);
//! - `v_rc' = -v_rc/(R₁·C₁) + I/C₁` — the polarization overpotential, which
//!   relaxes to `I·R₁` with time constant `τ = R₁·C₁`;
//! - `temp' = (P_gen − (temp − T_amb)/R_th) / C_th` — Newtonian cooling driven
//!   by the ohmic + polarization heat `P_gen = I²·R₀ + v_rc·I`.
//!
//! The terminal voltage is `OCV(soc) − I·R₀ − v_rc` with a linear open-circuit
//! curve. Coulomb counting is a linear invariant (RK4 integrates it exactly),
//! `v_rc` and (once `v_rc` settles) `temp` have closed-form relaxations — the
//! oracles the tests use.

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

fn check_finite(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite"
        )))
    }
}

/// A Thévenin (1-RC) battery equivalent-circuit model with a self-heating
/// thermal state, discharged (or charged, `current < 0`) at constant current.
#[derive(Debug, Clone, PartialEq)]
pub struct TheveninBattery {
    capacity_ah: f64,
    r0: f64,
    r1: f64,
    c1: f64,
    current: f64,
    ocv_min: f64,
    ocv_max: f64,
    r_th: f64,
    c_th: f64,
    t_ambient: f64,
}

/// Parameters for [`TheveninBattery::new`], grouped so the constructor is not
/// a wall of positional `f64`s.
#[derive(Debug, Clone, PartialEq)]
pub struct BatteryParams {
    /// Nominal capacity in ampere-hours (> 0).
    pub capacity_ah: f64,
    /// Ohmic (series) resistance in ohms (> 0).
    pub r0: f64,
    /// Polarization resistance in ohms (> 0).
    pub r1: f64,
    /// Polarization capacitance in farads (> 0).
    pub c1: f64,
    /// Constant load current in amperes; positive discharges, negative charges.
    pub current: f64,
    /// Open-circuit voltage at `soc = 0` (finite).
    pub ocv_min: f64,
    /// Open-circuit voltage at `soc = 1` (finite, ≥ `ocv_min`).
    pub ocv_max: f64,
    /// Thermal resistance to ambient in K/W (> 0).
    pub r_th: f64,
    /// Thermal capacitance in J/K (> 0).
    pub c_th: f64,
    /// Ambient temperature in the same unit as the thermal state (finite).
    pub t_ambient: f64,
}

impl TheveninBattery {
    /// Create the model, validating every parameter.
    pub fn new(p: BatteryParams) -> Result<Self, SimError> {
        check_positive("capacity_ah", p.capacity_ah)?;
        check_positive("r0", p.r0)?;
        check_positive("r1", p.r1)?;
        check_positive("c1", p.c1)?;
        check_finite("current", p.current)?;
        check_finite("ocv_min", p.ocv_min)?;
        check_finite("ocv_max", p.ocv_max)?;
        check_positive("r_th", p.r_th)?;
        check_positive("c_th", p.c_th)?;
        check_finite("t_ambient", p.t_ambient)?;
        if p.ocv_max < p.ocv_min
        {
            return Err(SimError::BadInput(format!(
                "ocv_max = {} must be ≥ ocv_min = {}",
                p.ocv_max, p.ocv_min
            )));
        }
        Ok(TheveninBattery {
            capacity_ah: p.capacity_ah,
            r0: p.r0,
            r1: p.r1,
            c1: p.c1,
            current: p.current,
            ocv_min: p.ocv_min,
            ocv_max: p.ocv_max,
            r_th: p.r_th,
            c_th: p.c_th,
            t_ambient: p.t_ambient,
        })
    }

    /// The initial state `[soc0, 0, temp0]` (polarization starts relaxed).
    pub fn initial_state(&self, soc0: f64, temp0: f64) -> [f64; 3] {
        [soc0, 0.0, temp0]
    }

    /// Open-circuit voltage at a state of charge, linear between `ocv_min`
    /// (soc = 0) and `ocv_max` (soc = 1).
    pub fn open_circuit_voltage(&self, soc: f64) -> f64 {
        self.ocv_min + (self.ocv_max - self.ocv_min) * soc
    }

    /// Terminal voltage for a state `[soc, v_rc, _temp]`, `OCV(soc) − I·R₀ −
    /// v_rc`, or `None` when the state is not length 3.
    pub fn terminal_voltage(&self, state: &[f64]) -> Option<f64> {
        let [soc, v_rc, _temp] = *state
        else
        {
            return None;
        };
        Some(self.open_circuit_voltage(soc) - self.current * self.r0 - v_rc)
    }

    /// The polarization time constant `τ = R₁·C₁`.
    pub fn polarization_time_constant(&self) -> f64 {
        self.r1 * self.c1
    }

    /// The steady-state polarization overpotential `I·R₁`.
    pub fn steady_state_overpotential(&self) -> f64 {
        self.current * self.r1
    }

    /// The steady-state temperature once the polarization has settled:
    /// `T_amb + P_gen·R_th` with `P_gen = I²·R₀ + (I·R₁)·I`.
    pub fn steady_state_temperature(&self) -> f64 {
        let p_gen = self.current * self.current * self.r0
            + self.steady_state_overpotential() * self.current;
        self.t_ambient + p_gen * self.r_th
    }

    /// The closed-form state of charge at time `t` from `soc0` under the
    /// constant current: `soc0 − I·t/(capacity·3600)`.
    pub fn soc_at(&self, soc0: f64, t: f64) -> f64 {
        soc0 - self.current * t / (self.capacity_ah * 3600.0)
    }

    /// The closed-form polarization overpotential at time `t` from `v_rc = 0`:
    /// `I·R₁·(1 − e^{−t/τ})`.
    pub fn overpotential_at(&self, t: f64) -> f64 {
        self.steady_state_overpotential() * (1.0 - (-t / self.polarization_time_constant()).exp())
    }
}

impl System for TheveninBattery {
    fn dim(&self) -> usize {
        3
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let (_soc, v_rc, temp) = (y[0], y[1], y[2]);
        dydt[0] = -self.current / (self.capacity_ah * 3600.0);
        dydt[1] = -v_rc / (self.r1 * self.c1) + self.current / self.c1;
        let p_gen = self.current * self.current * self.r0 + v_rc * self.current;
        dydt[2] = (p_gen - (temp - self.t_ambient) / self.r_th) / self.c_th;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    fn sample() -> TheveninBattery {
        TheveninBattery::new(BatteryParams {
            capacity_ah: 2.0,
            r0: 0.02,
            r1: 0.01,
            c1: 2000.0,
            current: 4.0, // 2C discharge
            ocv_min: 3.0,
            ocv_max: 4.2,
            r_th: 5.0,
            c_th: 40.0,
            t_ambient: 25.0,
        })
        .unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn coulomb_counting_is_exact_and_overpotential_relaxes() {
        let bat = sample();
        let traj = simulate(&bat, &bat.initial_state(1.0, 25.0), 0.0, 100.0, 0.01).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            // SoC is a linear invariant: RK4 tracks it to round-off.
            assert!((row[0] - bat.soc_at(1.0, *t)).abs() < 1e-12, "soc t = {t}");
            // Polarization matches the closed-form relaxation.
            assert!(
                (row[1] - bat.overpotential_at(*t)).abs() < 1e-9,
                "v_rc t = {t}"
            );
        }
        // τ = 20 s, so by t = 100 s (5τ) the overpotential is within 1% of I·R₁.
        let last = traj.last_state().unwrap();
        assert!(
            (last[1] - bat.steady_state_overpotential()).abs()
                < 0.01 * bat.steady_state_overpotential()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn temperature_rises_to_the_steady_state() {
        let bat = sample();
        // Integrate well past both the RC (20 s) and thermal (C_th·R_th = 200 s)
        // time constants.
        let traj = simulate(&bat, &bat.initial_state(1.0, 25.0), 0.0, 2000.0, 0.05).unwrap();
        let temp = traj.last_state().unwrap()[2];
        let expected = bat.steady_state_temperature();
        assert!(
            (temp - expected).abs() < 1e-3,
            "T = {temp}, expected {expected}"
        );
        // Self-heating raised it above ambient.
        assert!(temp > 25.0 + 1.0);
    }

    #[test]
    fn terminal_voltage_drops_under_load_and_helpers_validate() {
        let bat = sample();
        // At rest (soc = 1, v_rc = 0) the terminal voltage is OCV − I·R₀.
        let v0 = bat.terminal_voltage(&[1.0, 0.0, 25.0]).unwrap();
        assert!((v0 - (4.2 - 4.0 * 0.02)).abs() < 1e-12);
        // Under polarization it sags further.
        let v1 = bat
            .terminal_voltage(&[1.0, bat.steady_state_overpotential(), 25.0])
            .unwrap();
        assert!(v1 < v0);
        assert!(bat.terminal_voltage(&[1.0, 0.0]).is_none());
        assert!((bat.polarization_time_constant() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn charging_current_raises_soc() {
        // Negative current = charging: SoC increases.
        let mut p = BatteryParams {
            capacity_ah: 2.0,
            r0: 0.02,
            r1: 0.01,
            c1: 2000.0,
            current: -2.0,
            ocv_min: 3.0,
            ocv_max: 4.2,
            r_th: 5.0,
            c_th: 40.0,
            t_ambient: 25.0,
        };
        let bat = TheveninBattery::new(p.clone()).unwrap();
        assert!(bat.soc_at(0.5, 3600.0) > 0.5);
        // Bad parameters are rejected.
        p.capacity_ah = 0.0;
        assert!(TheveninBattery::new(p.clone()).is_err());
        p.capacity_ah = 2.0;
        p.ocv_max = 2.0; // < ocv_min
        assert!(TheveninBattery::new(p.clone()).is_err());
        p.ocv_max = 4.2;
        p.current = f64::NAN;
        assert!(TheveninBattery::new(p).is_err());
    }
}
