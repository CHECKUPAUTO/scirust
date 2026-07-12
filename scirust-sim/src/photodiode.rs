//! A photodiode detector: the optoelectronic *receiver* that complements the
//! [`laser`](crate::laser) emitter. Incident optical power is converted to a
//! **photocurrent** by the diode's spectral responsivity, and that current
//! charges the junction capacitance through the load resistance — a first-order
//! `RC` low-pass that sets the detector's bandwidth. Here it is a *simulator*:
//! set the optical power and integrate the output voltage.
//!
//! State `y = [v]` (the load voltage):
//!
//! - `v' = (I_ph − v/R_L) / C_j` — the photocurrent `I_ph = ℛ·P_opt + I_dark`
//!   charging the junction capacitance `C_j` through the load `R_L`.
//!
//! The physics gives clean closed forms — the oracles the tests use: the
//! **responsivity** `ℛ = η·q·λ/(h·c)` (rising linearly with wavelength), the
//! steady-state voltage `v_ss = I_ph·R_L`, the `−3 dB` bandwidth
//! `f = 1/(2π·R_L·C_j)`, and the exponential step response with time constant
//! `τ = R_L·C_j`.

use crate::engine::{SimError, System};

/// Elementary charge (C).
const Q: f64 = 1.602_176_634e-19;
/// Planck constant (J·s).
const H: f64 = 6.626_070_15e-34;
/// Speed of light in vacuum (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;

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

fn check_nonnegative(name: &str, value: f64) -> Result<(), SimError> {
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

/// The spectral **responsivity** `ℛ = η·q·λ/(h·c)` (amperes per watt) of an
/// ideal photodiode of quantum efficiency `quantum_efficiency` at wavelength
/// `wavelength` (metres): each absorbed photon of energy `h·c/λ` yields `η`
/// electrons, so responsivity rises linearly with wavelength.
pub fn responsivity(quantum_efficiency: f64, wavelength: f64) -> f64 {
    quantum_efficiency * Q * wavelength / (H * C_LIGHT)
}

/// A photodiode driven by constant optical power into a resistive load with
/// junction capacitance.
#[derive(Debug, Clone, PartialEq)]
pub struct Photodiode {
    responsivity: f64,
    dark_current: f64,
    r_load: f64,
    c_junction: f64,
    optical_power: f64,
}

/// Parameters for [`Photodiode::new`], grouped so the constructor is not a wall
/// of positional `f64`s.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotodiodeParams {
    /// Spectral responsivity `ℛ` in A/W (> 0); see [`responsivity`].
    pub responsivity: f64,
    /// Dark (leakage) current in amperes (≥ 0).
    pub dark_current: f64,
    /// Load resistance in ohms (> 0).
    pub r_load: f64,
    /// Junction capacitance in farads (> 0).
    pub c_junction: f64,
    /// Incident optical power in watts (≥ 0).
    pub optical_power: f64,
}

impl Photodiode {
    /// Create the model, validating every parameter.
    pub fn new(p: PhotodiodeParams) -> Result<Self, SimError> {
        check_positive("responsivity", p.responsivity)?;
        check_nonnegative("dark_current", p.dark_current)?;
        check_positive("r_load", p.r_load)?;
        check_positive("c_junction", p.c_junction)?;
        check_nonnegative("optical_power", p.optical_power)?;
        Ok(Photodiode {
            responsivity: p.responsivity,
            dark_current: p.dark_current,
            r_load: p.r_load,
            c_junction: p.c_junction,
            optical_power: p.optical_power,
        })
    }

    /// The initial state `[v0]`.
    pub fn initial_state(&self, v0: f64) -> [f64; 1] {
        [v0]
    }

    /// The total **photocurrent** `I_ph = ℛ·P_opt + I_dark`.
    pub fn photocurrent(&self) -> f64 {
        self.responsivity * self.optical_power + self.dark_current
    }

    /// The steady-state load voltage `v_ss = I_ph·R_L`.
    pub fn steady_state_voltage(&self) -> f64 {
        self.photocurrent() * self.r_load
    }

    /// The `RC` time constant `τ = R_L·C_j`.
    pub fn time_constant(&self) -> f64 {
        self.r_load * self.c_junction
    }

    /// The `−3 dB` electrical **bandwidth** `f = 1/(2π·R_L·C_j)`.
    pub fn bandwidth(&self) -> f64 {
        1.0 / (2.0 * std::f64::consts::PI * self.time_constant())
    }
}

impl System for Photodiode {
    fn dim(&self) -> usize {
        1
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let v = y[0];
        dydt[0] = (self.photocurrent() - v / self.r_load) / self.c_junction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    fn sample() -> Photodiode {
        // ℛ = 1 A/W, R_L = 1 kΩ, C_j = 1 nF ⇒ τ = 1 µs, v_ss = 1 V at 1 mW.
        Photodiode::new(PhotodiodeParams {
            responsivity: 1.0,
            dark_current: 0.0,
            r_load: 1.0e3,
            c_junction: 1.0e-9,
            optical_power: 1.0e-3,
        })
        .unwrap()
    }

    #[test]
    fn responsivity_follows_the_closed_form_and_scales_with_wavelength() {
        // At 1.55 µm with unit quantum efficiency, ℛ ≈ 1.25 A/W.
        let r = responsivity(1.0, 1.55e-6);
        assert!((r - Q * 1.55e-6 / (H * C_LIGHT)).abs() < 1e-15);
        assert!((r - 1.2497).abs() < 1e-3, "responsivity {r}");
        // Linear in wavelength, linear in quantum efficiency.
        assert!((responsivity(1.0, 3.1e-6) - 2.0 * r).abs() < 1e-12);
        assert!((responsivity(0.5, 1.55e-6) - 0.5 * r).abs() < 1e-15);
    }

    #[test]
    fn photocurrent_steady_state_and_bandwidth_closed_forms() {
        let pd = sample();
        assert!((pd.photocurrent() - 1.0e-3).abs() < 1e-15);
        assert!((pd.steady_state_voltage() - 1.0).abs() < 1e-12);
        assert!((pd.time_constant() - 1.0e-6).abs() < 1e-18);
        let expected_bw = 1.0 / (2.0 * std::f64::consts::PI * 1.0e-6);
        assert!((pd.bandwidth() - expected_bw).abs() < 1e-3);
    }

    #[test]
    fn dark_current_sets_the_no_light_floor() {
        let pd = Photodiode::new(PhotodiodeParams {
            responsivity: 1.0,
            dark_current: 2.0e-6,
            r_load: 1.0e3,
            c_junction: 1.0e-9,
            optical_power: 0.0,
        })
        .unwrap();
        // No light: the output floor is the dark current across the load.
        assert!((pd.photocurrent() - 2.0e-6).abs() < 1e-15);
        assert!((pd.steady_state_voltage() - 2.0e-3).abs() < 1e-12);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // integrates an ODE — too slow under Miri
    fn step_response_charges_with_the_rc_time_constant() {
        let pd = sample();
        let (v_ss, tau) = (pd.steady_state_voltage(), pd.time_constant());
        // Integrate ten time constants from a dark start; h = τ/100.
        let traj = simulate(&pd, &pd.initial_state(0.0), 0.0, 10.0 * tau, tau / 100.0).unwrap();
        let v = traj.column(0).unwrap();
        // At one time constant the RC law gives v = v_ss·(1 − 1/e).
        let at_tau = v[100];
        assert!(
            (at_tau - v_ss * (1.0 - (-1.0_f64).exp())).abs() < 1e-3,
            "v(τ) = {at_tau}"
        );
        // After ten time constants it has settled to v_ss.
        assert!((v.last().unwrap() - v_ss).abs() < 1e-3);
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(
            Photodiode::new(PhotodiodeParams {
                responsivity: 0.0,
                dark_current: 0.0,
                r_load: 1.0e3,
                c_junction: 1.0e-9,
                optical_power: 1.0e-3,
            })
            .is_err()
        );
        assert!(
            Photodiode::new(PhotodiodeParams {
                responsivity: 1.0,
                dark_current: 0.0,
                r_load: 1.0e3,
                c_junction: 0.0,
                optical_power: 1.0e-3,
            })
            .is_err()
        );
    }
}
