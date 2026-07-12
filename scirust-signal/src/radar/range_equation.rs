//! The radar range equation — SNR and detection range versus target RCS.
//!
//! [`super::swerling`] gives the SNR a target needs for a required detection
//! probability; this module supplies the other half of the link budget — the SNR
//! a radar actually *delivers* on a target of a given radar cross-section at a
//! given range, and, inverting it, the **maximum detection range**. Signal power
//! falls as `1/R⁴` (two-way spreading), so range is a fourth-root function of
//! everything else — doubling the transmit power buys only a 19 % range
//! increase.
//!
//! A [`RadarLink`] bundles the radar and system parameters; its methods take the
//! target RCS and the range (or the minimum SNR from
//! [`super::swerling`]). Dependency-free.

/// Boltzmann constant `k_B` (J/K).
const BOLTZMANN: f64 = 1.380_649e-23;

/// A monostatic radar link budget: transmitter, antenna, and receiver-noise
/// parameters, in SI/linear units (gain, noise figure and losses are linear
/// ratios, not dB).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarLink {
    /// Peak transmit power `P_t` (W).
    pub peak_power: f64,
    /// Antenna gain `G` (linear), used for both transmit and receive.
    pub gain: f64,
    /// Wavelength `λ` (m).
    pub wavelength: f64,
    /// Receiver noise bandwidth `B` (Hz).
    pub bandwidth: f64,
    /// Receiver noise figure `F` (linear).
    pub noise_figure: f64,
    /// System noise reference temperature `T` (K).
    pub temperature: f64,
    /// Total system losses `L` (linear, ≥ 1).
    pub losses: f64,
}

impl RadarLink {
    /// The receiver **noise power** `N = k_B·T·B·F` (W).
    pub fn noise_power(&self) -> f64 {
        BOLTZMANN * self.temperature * self.bandwidth * self.noise_figure
    }

    /// The **received echo power** from a target of cross-section `rcs` (m²) at
    /// `range` (m), by the monostatic radar equation
    /// `P_r = P_t·G²·λ²·σ / ((4π)³·R⁴·L)`. `0.0` for a non-positive range.
    pub fn received_power(&self, rcs: f64, range: f64) -> f64 {
        if range <= 0.0 || self.losses <= 0.0
        {
            return 0.0;
        }
        let four_pi_cubed = (4.0 * std::f64::consts::PI).powi(3);
        let num = self.peak_power * self.gain * self.gain * self.wavelength * self.wavelength * rcs;
        num / (four_pi_cubed * range.powi(4) * self.losses)
    }

    /// The single-pulse **signal-to-noise ratio** (linear) on a target of
    /// cross-section `rcs` at `range`: received power over noise power.
    pub fn snr_at_range(&self, rcs: f64, range: f64) -> f64 {
        let noise = self.noise_power();
        if noise <= 0.0
        {
            return 0.0;
        }
        self.received_power(rcs, range) / noise
    }

    /// The **maximum detection range** (m) at which a target of cross-section
    /// `rcs` still yields at least `snr_min` (linear) — the required SNR from
    /// [`super::swerling`] for a chosen detection probability. Inverts the radar
    /// equation: `R_max = [P_t·G²·λ²·σ / ((4π)³·N·L·SNR_min)]^{1/4}`. `0.0` if
    /// `snr_min` or the noise is non-positive.
    pub fn max_range(&self, rcs: f64, snr_min: f64) -> f64 {
        let noise = self.noise_power();
        if snr_min <= 0.0 || noise <= 0.0 || self.losses <= 0.0
        {
            return 0.0;
        }
        let four_pi_cubed = (4.0 * std::f64::consts::PI).powi(3);
        let num = self.peak_power * self.gain * self.gain * self.wavelength * self.wavelength * rcs;
        (num / (four_pi_cubed * noise * self.losses * snr_min)).powf(0.25)
    }
}

#[cfg(test)]
mod tests {
    use super::super::swerling::albersheim_snr;
    use super::*;

    fn link() -> RadarLink {
        RadarLink {
            peak_power: 1.0e6,  // 1 MW
            gain: 3.16e4,       // ~45 dBi
            wavelength: 0.03,   // 10 GHz (X-band)
            bandwidth: 1.0e6,   // 1 MHz
            noise_figure: 3.16, // ~5 dB
            temperature: 290.0, // reference
            losses: 3.16,       // ~5 dB
        }
    }

    #[test]
    fn received_power_follows_the_inverse_fourth_power_law() {
        let l = link();
        let (rcs, r) = (1.0, 50_000.0);
        let near = l.received_power(rcs, r);
        let far = l.received_power(rcs, 2.0 * r);
        // Doubling the range drops the echo power by 2⁴ = 16.
        assert!((near / far - 16.0).abs() < 1e-6, "{}", near / far);
        assert_eq!(l.received_power(rcs, 0.0), 0.0);
    }

    #[test]
    fn noise_power_is_ktbf() {
        let l = link();
        let expect = BOLTZMANN * l.temperature * l.bandwidth * l.noise_figure;
        assert!((l.noise_power() - expect).abs() < 1e-30);
    }

    #[test]
    fn snr_scales_with_rcs_and_falls_with_range() {
        let l = link();
        // SNR ∝ σ.
        let s1 = l.snr_at_range(1.0, 40_000.0);
        let s2 = l.snr_at_range(2.0, 40_000.0);
        assert!((s2 / s1 - 2.0).abs() < 1e-9);
        // SNR ∝ 1/R⁴.
        let near = l.snr_at_range(1.0, 30_000.0);
        let far = l.snr_at_range(1.0, 60_000.0);
        assert!((near / far - 16.0).abs() < 1e-6);
    }

    #[test]
    fn max_range_is_the_range_where_snr_meets_the_minimum() {
        // The headline consistency check: at the computed maximum range, the
        // delivered SNR equals the required minimum.
        let l = link();
        let (rcs, snr_min) = (2.0, 20.0);
        let r_max = l.max_range(rcs, snr_min);
        assert!(r_max > 0.0);
        assert!((l.snr_at_range(rcs, r_max) - snr_min).abs() / snr_min < 1e-9);
    }

    #[test]
    fn max_range_scales_as_the_fourth_root_of_rcs() {
        let l = link();
        let snr_min = 15.0;
        let r1 = l.max_range(1.0, snr_min);
        let r16 = l.max_range(16.0, snr_min);
        // 16× the RCS ⇒ 2× the range (16^{1/4} = 2).
        assert!((r16 / r1 - 2.0).abs() < 1e-9, "{}", r16 / r1);
        assert_eq!(l.max_range(1.0, 0.0), 0.0);
    }

    #[test]
    fn integrates_with_swerling_detection_range() {
        // The link budget closes with the detection statistics: the required SNR
        // for a target P_d/P_fa (Albersheim, steady target) sets the detection
        // range — and demanding a higher P_d (more SNR) shortens it.
        let l = link();
        let rcs = 1.0;
        let snr_easy = 10.0_f64.powf(albersheim_snr(0.5, 1e-6, 1) / 10.0);
        let snr_hard = 10.0_f64.powf(albersheim_snr(0.9, 1e-6, 1) / 10.0);
        let r_easy = l.max_range(rcs, snr_easy);
        let r_hard = l.max_range(rcs, snr_hard);
        assert!(r_easy > 0.0 && r_hard > 0.0);
        assert!(
            r_hard < r_easy,
            "higher P_d must shorten range: {r_hard} vs {r_easy}"
        );
    }
}
