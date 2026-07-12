//! A semiconductor-laser plant: the single-mode **rate equations** coupling the
//! carrier density `n` and photon density `s`. This is the canonical
//! optoelectronic device model — the dynamics behind a laser diode's threshold,
//! its linear light–current curve, and its relaxation-oscillation ringing on
//! turn-on. Here it is a *simulator*: set the pump and integrate.
//!
//! State `y = [n, s]` (carrier density, photon density):
//!
//! - `n' = J − n/τ_n − g₀·(n − n_t)·s` — pumping `J`, spontaneous carrier
//!   recombination `n/τ_n`, and stimulated recombination into the mode;
//! - `s' = Γ·g₀·(n − n_t)·s − s/τ_p + Γ·β·n/τ_n` — modal gain, cavity loss
//!   `s/τ_p`, and the spontaneous-emission seed `Γ·β·n/τ_n`.
//!
//! With `β = 0` the model has clean closed forms — the oracles the tests use:
//! the gain clamps the carrier density at threshold `n_th = n_t + 1/(Γ·g₀·τ_p)`,
//! the pump threshold is `J_th = n_th/τ_n`, and above threshold the photon
//! density is the linear light–current law `s = Γ·τ_p·(J − J_th)`. Small
//! perturbations ring at the relaxation-oscillation frequency
//! `f_r = (1/2π)·√(g₀·s_ss/τ_p)`.

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

/// A single-mode semiconductor-laser rate-equation model driven by a constant
/// pump.
#[derive(Debug, Clone, PartialEq)]
pub struct SemiconductorLaser {
    g0: f64,
    n_t: f64,
    tau_n: f64,
    tau_p: f64,
    gamma: f64,
    beta: f64,
    pump: f64,
}

/// Parameters for [`SemiconductorLaser::new`], grouped so the constructor is not
/// a wall of positional `f64`s.
#[derive(Debug, Clone, PartialEq)]
pub struct LaserParams {
    /// Differential gain `g₀` (> 0).
    pub g0: f64,
    /// Transparency carrier density `n_t` (≥ 0).
    pub n_t: f64,
    /// Carrier lifetime `τ_n` in seconds (> 0).
    pub tau_n: f64,
    /// Photon lifetime `τ_p` in seconds (> 0).
    pub tau_p: f64,
    /// Optical confinement factor `Γ` (> 0, typically ≤ 1).
    pub gamma: f64,
    /// Spontaneous-emission coupling factor `β` (≥ 0, typically ≪ 1).
    pub beta: f64,
    /// Pump rate `J` (carrier density per unit time, ≥ 0).
    pub pump: f64,
}

impl SemiconductorLaser {
    /// Create the model, validating every parameter.
    pub fn new(p: LaserParams) -> Result<Self, SimError> {
        check_positive("g0", p.g0)?;
        check_nonnegative("n_t", p.n_t)?;
        check_positive("tau_n", p.tau_n)?;
        check_positive("tau_p", p.tau_p)?;
        check_positive("gamma", p.gamma)?;
        check_nonnegative("beta", p.beta)?;
        check_nonnegative("pump", p.pump)?;
        Ok(SemiconductorLaser {
            g0: p.g0,
            n_t: p.n_t,
            tau_n: p.tau_n,
            tau_p: p.tau_p,
            gamma: p.gamma,
            beta: p.beta,
            pump: p.pump,
        })
    }

    /// The initial state `[n0, s0]`.
    pub fn initial_state(&self, n0: f64, s0: f64) -> [f64; 2] {
        [n0, s0]
    }

    /// The **threshold carrier density** `n_th = n_t + 1/(Γ·g₀·τ_p)`, where the
    /// modal gain equals the cavity loss and the carrier density clamps.
    pub fn threshold_density(&self) -> f64 {
        self.n_t + 1.0 / (self.gamma * self.g0 * self.tau_p)
    }

    /// The **threshold pump** `J_th = n_th/τ_n` (the `β → 0` limit): below it the
    /// laser stays dark, above it the output rises linearly.
    pub fn threshold_pump(&self) -> f64 {
        self.threshold_density() / self.tau_n
    }

    /// The steady-state **photon density** (`β → 0` limit): the linear
    /// light–current law `s_ss = Γ·τ_p·(J − J_th)` above threshold, else `0`.
    pub fn steady_state_photon_density(&self) -> f64 {
        let excess = self.pump - self.threshold_pump();
        if excess > 0.0
        {
            self.gamma * self.tau_p * excess
        }
        else
        {
            0.0
        }
    }

    /// The steady-state **carrier density** (`β → 0` limit): clamped at `n_th`
    /// above threshold, else `J·τ_n` (pure spontaneous recombination).
    pub fn steady_state_carrier_density(&self) -> f64 {
        if self.pump > self.threshold_pump()
        {
            self.threshold_density()
        }
        else
        {
            self.pump * self.tau_n
        }
    }

    /// The **relaxation-oscillation frequency** `f_r = (1/2π)·√(g₀·s_ss/τ_p)`
    /// about the above-threshold steady state; `0` below threshold.
    pub fn relaxation_frequency(&self) -> f64 {
        let s_ss = self.steady_state_photon_density();
        if s_ss > 0.0
        {
            (self.g0 * s_ss / self.tau_p).sqrt() / (2.0 * std::f64::consts::PI)
        }
        else
        {
            0.0
        }
    }
}

impl System for SemiconductorLaser {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let (n, s) = (y[0], y[1]);
        let gain = self.g0 * (n - self.n_t);
        dydt[0] = self.pump - n / self.tau_n - gain * s;
        dydt[1] = self.gamma * gain * s - s / self.tau_p + self.gamma * self.beta * n / self.tau_n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{simulate, simulate_adaptive};

    /// Normalised parameters (τ_n = 1, τ_p = 0.01, g₀ = n_t = Γ = 1) so the
    /// closed forms are round numbers: n_th = 101, J_th = 101.
    fn laser(pump: f64, beta: f64) -> SemiconductorLaser {
        SemiconductorLaser::new(LaserParams {
            g0: 1.0,
            n_t: 1.0,
            tau_n: 1.0,
            tau_p: 0.01,
            gamma: 1.0,
            beta,
            pump,
        })
        .unwrap()
    }

    #[test]
    fn threshold_and_light_current_closed_forms() {
        let l = laser(150.0, 0.0);
        assert!((l.threshold_density() - 101.0).abs() < 1e-12);
        assert!((l.threshold_pump() - 101.0).abs() < 1e-12);
        // Above threshold: s_ss = Γ·τ_p·(J − J_th) = 0.01·49 = 0.49, n clamped.
        assert!((l.steady_state_photon_density() - 0.49).abs() < 1e-12);
        assert!((l.steady_state_carrier_density() - 101.0).abs() < 1e-12);
        // Below threshold: dark, carriers accumulate to J·τ_n.
        let dark = laser(50.0, 0.0);
        assert_eq!(dark.steady_state_photon_density(), 0.0);
        assert!((dark.steady_state_carrier_density() - 50.0).abs() < 1e-12);
    }

    #[test]
    fn light_current_curve_is_linear_above_threshold() {
        // s_ss ∝ (J − J_th): doubling the pump excess doubles the output.
        let (a, b) = (laser(150.0, 0.0), laser(199.0, 0.0)); // excess 49 vs 98
        let ratio = b.steady_state_photon_density() / a.steady_state_photon_density();
        assert!((ratio - 2.0).abs() < 1e-12, "L-I slope not linear: {ratio}");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // integrates a stiff-ish ODE — too slow under Miri
    fn turn_on_converges_to_the_closed_form_steady_state() {
        let l = laser(150.0, 0.0);
        // Start dark with a tiny photon seed (β = 0 needs a seed to turn on).
        let traj =
            simulate_adaptive(&l, &l.initial_state(0.0, 1.0e-6), 0.0, 40.0, 1e-9, 1e-12).unwrap();
        let end = traj.last_state().unwrap();
        assert!(
            (end[0] - l.steady_state_carrier_density()).abs() < 1e-2,
            "n {}",
            end[0]
        );
        assert!(
            (end[1] - l.steady_state_photon_density()).abs() < 1e-4,
            "s {}",
            end[1]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn below_threshold_stays_dark() {
        let l = laser(50.0, 0.0);
        let traj =
            simulate_adaptive(&l, &l.initial_state(0.0, 1.0e-6), 0.0, 40.0, 1e-9, 1e-12).unwrap();
        let end = traj.last_state().unwrap();
        // The seed photon decays away; carriers settle at J·τ_n = 50.
        assert!(end[1] < 1e-6, "should stay dark: s = {}", end[1]);
        assert!((end[0] - 50.0).abs() < 1e-2, "n = {}", end[0]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn small_perturbation_rings_at_the_relaxation_frequency() {
        let l = laser(150.0, 0.0);
        let s_ss = l.steady_state_photon_density();
        let n_ss = l.steady_state_carrier_density();
        // Kick the photon density down 30 % from steady state and watch it ring.
        let traj = simulate(&l, &[n_ss, 0.7 * s_ss], 0.0, 3.0, 5.0e-4).unwrap();
        let s = traj.column(1).unwrap();
        // Zero crossings of (s − s_ss) mark half-periods of the oscillation.
        let mut crossings = Vec::new();
        for i in 1..s.len()
        {
            let (prev, cur) = (s[i - 1] - s_ss, s[i] - s_ss);
            if (prev <= 0.0 && cur > 0.0) || (prev >= 0.0 && cur < 0.0)
            {
                crossings.push(traj.t[i]);
            }
        }
        assert!(
            crossings.len() >= 3,
            "no ringing: {} crossings",
            crossings.len()
        );
        // One full period spans two half-periods.
        let period = crossings[2] - crossings[0];
        let expected = 1.0 / l.relaxation_frequency();
        assert!(
            (period - expected).abs() / expected < 0.1,
            "period {period} vs expected {expected}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn spontaneous_emission_seeds_turn_on_without_a_photon_seed() {
        // With β > 0 the laser turns on from s = 0 exactly (spontaneous seed).
        let l = laser(150.0, 1.0e-4);
        let traj =
            simulate_adaptive(&l, &l.initial_state(0.0, 0.0), 0.0, 40.0, 1e-9, 1e-12).unwrap();
        let end = traj.last_state().unwrap();
        // Reaches near the β → 0 output (spontaneous emission adds a small offset).
        assert!(
            end[1] > 0.4,
            "spontaneous emission failed to seed lasing: s = {}",
            end[1]
        );
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(
            SemiconductorLaser::new(LaserParams {
                g0: -1.0,
                n_t: 1.0,
                tau_n: 1.0,
                tau_p: 0.01,
                gamma: 1.0,
                beta: 0.0,
                pump: 150.0,
            })
            .is_err()
        );
        assert!(
            SemiconductorLaser::new(LaserParams {
                g0: 1.0,
                n_t: 1.0,
                tau_n: 1.0,
                tau_p: 0.0,
                gamma: 1.0,
                beta: 0.0,
                pump: 150.0,
            })
            .is_err()
        );
    }
}
