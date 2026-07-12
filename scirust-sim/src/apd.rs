//! An avalanche photodiode (APD) — the high-sensitivity optoelectronic receiver
//! behind lidar and laser rangefinders. Unlike the plain [`photodiode`], an APD
//! runs its junction in avalanche breakdown so each primary photo-electron
//! triggers an impact-ionization cascade of average **gain** `M`. That gain
//! amplifies the weak signal photocurrent above the downstream thermal noise —
//! but the cascade is random, so it adds **excess noise** through the McIntyre
//! factor `F(M) = k·M + (1 − k)·(2 − 1/M)`, where `k` is the ionization ratio.
//!
//! The result is the defining APD design tension, captured here in closed form:
//! shot noise grows with `M²·F(M)` while the multiplied signal grows with `M²`,
//! so raising the gain helps only until the excess noise overtakes the thermal
//! floor — there is an **optimal gain** that maximises the signal-to-noise
//! ratio. This module is a static receiver-analysis model (the RC output
//! dynamics are the same first-order low-pass as [`photodiode`]).
//!
//! [`photodiode`]: crate::photodiode

use crate::engine::SimError;

/// Elementary charge (C).
const Q: f64 = 1.602_176_634e-19;
/// Boltzmann constant (J/K).
const K_B: f64 = 1.380_649e-23;

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

/// The McIntyre **excess noise factor** `F(M) = k·M + (1 − k)·(2 − 1/M)` for
/// avalanche gain `gain` (`M ≥ 1`) and ionization ratio `ionization_ratio`
/// (`k ∈ [0, 1]`). `F(1) = 1` (no multiplication, no excess noise); as `M` grows
/// it tends to `2` for `k = 0` (electron-only multiplication) and to `M` for
/// `k = 1` (the noisiest case). Lower `k` ⇒ a quieter APD.
pub fn excess_noise_factor(gain: f64, ionization_ratio: f64) -> f64 {
    let (m, k) = (gain, ionization_ratio);
    k * m + (1.0 - k) * (2.0 - 1.0 / m)
}

/// An avalanche-photodiode receiver operating at a fixed gain, analysed for its
/// signal current and noise.
#[derive(Debug, Clone, PartialEq)]
pub struct Apd {
    responsivity: f64,
    gain: f64,
    ionization_ratio: f64,
    dark_current: f64,
    r_load: f64,
    temperature: f64,
    bandwidth: f64,
    optical_power: f64,
}

/// Parameters for [`Apd::new`], grouped so the constructor is not a wall of
/// positional `f64`s.
#[derive(Debug, Clone, PartialEq)]
pub struct ApdParams {
    /// Unity-gain spectral responsivity `ℛ` in A/W (> 0).
    pub responsivity: f64,
    /// Average avalanche gain `M` (≥ 1).
    pub gain: f64,
    /// Ionization ratio `k` in `[0, 1]` — lower is quieter.
    pub ionization_ratio: f64,
    /// Primary (unmultiplied) dark current in amperes (≥ 0).
    pub dark_current: f64,
    /// Load resistance in ohms (> 0).
    pub r_load: f64,
    /// Receiver temperature in kelvin (> 0).
    pub temperature: f64,
    /// Noise-equivalent bandwidth in hertz (> 0).
    pub bandwidth: f64,
    /// Incident optical power in watts (≥ 0).
    pub optical_power: f64,
}

impl Apd {
    /// Create the model, validating every parameter (`gain ≥ 1`,
    /// `0 ≤ ionization_ratio ≤ 1`).
    pub fn new(p: ApdParams) -> Result<Self, SimError> {
        check_positive("responsivity", p.responsivity)?;
        check_positive("r_load", p.r_load)?;
        check_positive("temperature", p.temperature)?;
        check_positive("bandwidth", p.bandwidth)?;
        check_nonnegative("dark_current", p.dark_current)?;
        check_nonnegative("optical_power", p.optical_power)?;
        if !(p.gain.is_finite() && p.gain >= 1.0)
        {
            return Err(SimError::BadInput(format!(
                "gain = {} must be finite and ≥ 1",
                p.gain
            )));
        }
        if !(p.ionization_ratio.is_finite() && (0.0..=1.0).contains(&p.ionization_ratio))
        {
            return Err(SimError::BadInput(format!(
                "ionization_ratio = {} must be in [0, 1]",
                p.ionization_ratio
            )));
        }
        Ok(Apd {
            responsivity: p.responsivity,
            gain: p.gain,
            ionization_ratio: p.ionization_ratio,
            dark_current: p.dark_current,
            r_load: p.r_load,
            temperature: p.temperature,
            bandwidth: p.bandwidth,
            optical_power: p.optical_power,
        })
    }

    /// The primary (pre-multiplication) photocurrent `ℛ·P_opt + I_dark`.
    pub fn primary_photocurrent(&self) -> f64 {
        self.responsivity * self.optical_power + self.dark_current
    }

    /// The multiplied output current `M·(ℛ·P_opt + I_dark)`.
    pub fn multiplied_photocurrent(&self) -> f64 {
        self.gain * self.primary_photocurrent()
    }

    /// The multiplied **signal** current `M·ℛ·P_opt` (excludes dark current).
    pub fn signal_current(&self) -> f64 {
        self.gain * self.responsivity * self.optical_power
    }

    /// The McIntyre excess noise factor `F(M)` at this APD's gain and `k`.
    pub fn excess_noise(&self) -> f64 {
        excess_noise_factor(self.gain, self.ionization_ratio)
    }

    /// The multiplied **shot-noise** current variance
    /// `2·q·I_primary·M²·F(M)·B`.
    pub fn shot_noise_variance(&self) -> f64 {
        2.0 * Q
            * self.primary_photocurrent()
            * self.gain
            * self.gain
            * self.excess_noise()
            * self.bandwidth
    }

    /// The **thermal (Johnson) noise** current variance `4·k_B·T·B / R_L`.
    pub fn thermal_noise_variance(&self) -> f64 {
        4.0 * K_B * self.temperature * self.bandwidth / self.r_load
    }

    /// The electrical **signal-to-noise ratio** `I_signal² / (σ²_shot +
    /// σ²_thermal)`. Rises with gain while thermal-limited, then falls once the
    /// excess noise dominates — hence an optimal gain.
    pub fn snr(&self) -> f64 {
        let signal = self.signal_current();
        signal * signal / (self.shot_noise_variance() + self.thermal_noise_variance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receiver(gain: f64) -> Apd {
        // A weak-signal Si APD receiver: 1 nW, R_L = 10 kΩ, 1 GHz, low k.
        Apd::new(ApdParams {
            responsivity: 0.8,
            gain,
            ionization_ratio: 0.02,
            dark_current: 1.0e-10,
            r_load: 1.0e4,
            temperature: 300.0,
            bandwidth: 1.0e9,
            optical_power: 1.0e-9,
        })
        .unwrap()
    }

    #[test]
    fn excess_noise_factor_hits_its_limits() {
        // F(1) = 1 for any k (no multiplication ⇒ no excess noise).
        for k in [0.0, 0.3, 1.0]
        {
            assert!(
                (excess_noise_factor(1.0, k) - 1.0).abs() < 1e-12,
                "F(1,{k})"
            );
        }
        // k = 0 ⇒ F → 2 − 1/M; k = 1 ⇒ F = M.
        assert!((excess_noise_factor(100.0, 0.0) - (2.0 - 1.0 / 100.0)).abs() < 1e-12);
        assert!((excess_noise_factor(100.0, 1.0) - 100.0).abs() < 1e-12);
        // Monotonic increasing in gain, and larger k is noisier.
        assert!(excess_noise_factor(50.0, 0.02) > excess_noise_factor(10.0, 0.02));
        assert!(excess_noise_factor(50.0, 0.5) > excess_noise_factor(50.0, 0.02));
    }

    #[test]
    fn currents_and_noise_match_closed_forms() {
        let apd = receiver(20.0);
        assert!((apd.primary_photocurrent() - (0.8 * 1e-9 + 1e-10)).abs() < 1e-20);
        assert!((apd.multiplied_photocurrent() - 20.0 * apd.primary_photocurrent()).abs() < 1e-18);
        assert!((apd.signal_current() - 20.0 * 0.8 * 1e-9).abs() < 1e-18);
        // Shot / thermal variances from their closed forms.
        let shot = 2.0 * Q * apd.primary_photocurrent() * 400.0 * apd.excess_noise() * 1e9;
        assert!((apd.shot_noise_variance() - shot).abs() < shot * 1e-12);
        let thermal = 4.0 * K_B * 300.0 * 1e9 / 1e4;
        assert!((apd.thermal_noise_variance() - thermal).abs() < thermal * 1e-12);
    }

    #[test]
    fn snr_peaks_at_an_intermediate_gain() {
        // Unity gain is thermal-noise-limited (poor SNR); very high gain is
        // excess-noise-limited (poor SNR); a moderate gain wins.
        let (low, mid, high) = (
            receiver(1.0).snr(),
            receiver(50.0).snr(),
            receiver(1000.0).snr(),
        );
        assert!(mid > low, "gain should help vs unity: {mid} vs {low}");
        assert!(
            mid > high,
            "excess noise should hurt at high gain: {mid} vs {high}"
        );
    }

    #[test]
    fn rejects_bad_parameters() {
        // Gain below unity is unphysical.
        assert!(
            Apd::new(ApdParams {
                responsivity: 0.8,
                gain: 0.5,
                ionization_ratio: 0.02,
                dark_current: 1.0e-10,
                r_load: 1.0e4,
                temperature: 300.0,
                bandwidth: 1.0e9,
                optical_power: 1.0e-9,
            })
            .is_err()
        );
        // Ionization ratio outside [0, 1].
        assert!(
            Apd::new(ApdParams {
                responsivity: 0.8,
                gain: 20.0,
                ionization_ratio: 1.5,
                dark_current: 1.0e-10,
                r_load: 1.0e4,
                temperature: 300.0,
                bandwidth: 1.0e9,
                optical_power: 1.0e-9,
            })
            .is_err()
        );
    }
}
