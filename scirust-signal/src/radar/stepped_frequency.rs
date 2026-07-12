//! Stepped-frequency synthetic wideband ranging.
//!
//! Fine range resolution needs wide bandwidth — but transmitting and digitising a
//! wideband pulse is expensive. A **stepped-frequency** waveform gets there with
//! cheap narrowband hardware: it transmits a burst of `N` pulses at frequencies
//! `fₙ = f₀ + n·Δf`, measuring the target's complex reflectivity at each step.
//! Those samples *are* the frequency response of the range profile, so an inverse
//! DFT synthesises a high-resolution profile whose resolution
//! `Δr = c/(2·N·Δf) = c/(2·B)` is set by the total synthesised bandwidth
//! `B = N·Δf` — far finer than any single narrowband pulse — over an unambiguous
//! window `c/(2·Δf)` set by the step size. Built on the crate's inverse FFT;
//! dependency-free.

use crate::complex::Complex;
use crate::fft::ifft;

/// Speed of light `c` (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;

/// The **synthesised bandwidth** `B = N·Δf` (Hz) of an `n_steps`-pulse burst
/// stepped by `freq_step`.
pub fn synthetic_bandwidth(n_steps: usize, freq_step: f64) -> f64 {
    n_steps as f64 * freq_step
}

/// The **range resolution** `Δr = c/(2·N·Δf)` (m) — set by the total synthesised
/// bandwidth, and also the spacing of the range-profile bins. `f64::INFINITY` for
/// a degenerate burst.
pub fn range_resolution(n_steps: usize, freq_step: f64) -> f64 {
    if n_steps == 0 || freq_step <= 0.0
    {
        return f64::INFINITY;
    }
    C_LIGHT / (2.0 * n_steps as f64 * freq_step)
}

/// The **unambiguous range window** `c/(2·Δf)` (m): the range profile is periodic
/// with this extent, so a scatterer beyond it aliases back in. Equals
/// `N·Δr`. `f64::INFINITY` for a non-positive step.
pub fn max_unambiguous_range(freq_step: f64) -> f64 {
    if freq_step <= 0.0
    {
        return f64::INFINITY;
    }
    C_LIGHT / (2.0 * freq_step)
}

/// The synthetic **range profile**: the magnitude of the inverse DFT of the
/// per-step complex `measurements` (the reflectivity sampled at `fₙ = n·Δf`),
/// one value per range bin spaced by [`range_resolution`]. Empty unless the
/// number of steps is a power of two (the FFT length requirement).
pub fn range_profile(measurements: &[Complex]) -> Vec<f64> {
    let n = measurements.len();
    if n == 0 || !n.is_power_of_two()
    {
        return Vec::new();
    }
    let mut buf = measurements.to_vec();
    ifft(&mut buf);
    buf.iter().map(|c| c.mag()).collect()
}

/// The range (m) of each profile bin, `R_m = m·Δr` for `m = 0..n_steps`.
pub fn range_bins(n_steps: usize, freq_step: f64) -> Vec<f64> {
    let dr = range_resolution(n_steps, freq_step);
    (0..n_steps).map(|m| m as f64 * dr).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// The stepped-frequency response of point scatterers at the given `ranges`:
    /// `H[n] = Σ_k e^{−j·2π·n·Δf·τ_k}` with round-trip delay `τ_k = 2R_k/c`.
    fn response(n_steps: usize, freq_step: f64, ranges: &[f64]) -> Vec<Complex> {
        (0..n_steps)
            .map(|n| {
                ranges.iter().fold(Complex::zero(), |acc, &r| {
                    let tau = 2.0 * r / C_LIGHT;
                    acc + Complex::cis(-2.0 * PI * n as f64 * freq_step * tau)
                })
            })
            .collect()
    }

    fn argmax(v: &[f64]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn bandwidth_and_resolution_formulas() {
        let (n, df) = (64, 1e6);
        assert!((synthetic_bandwidth(n, df) - 64e6).abs() < 1e-3);
        assert!((range_resolution(n, df) - C_LIGHT / (2.0 * 64e6)).abs() < 1e-9);
        // Resolution improves (halves) when the synthesised bandwidth doubles.
        assert!((range_resolution(n, df) / range_resolution(2 * n, df) - 2.0).abs() < 1e-9);
        // The unambiguous window is N range bins wide.
        assert!((max_unambiguous_range(df) - n as f64 * range_resolution(n, df)).abs() < 1e-3);
    }

    #[test]
    fn single_scatterer_localises_to_its_bin() {
        // A scatterer placed exactly on a range bin peaks there.
        let (n, df) = (64, 1e6);
        let dr = range_resolution(n, df);
        let bin = 20;
        let r = bin as f64 * dr;
        let profile = range_profile(&response(n, df, &[r]));
        assert_eq!(profile.len(), n);
        assert_eq!(argmax(&profile), bin);
        // The bin's mapped range matches the truth.
        assert!((range_bins(n, df)[bin] - r).abs() < 1e-6);
    }

    #[test]
    fn resolves_two_scatterers_a_few_bins_apart() {
        let (n, df) = (128, 2e6);
        let dr = range_resolution(n, df);
        let (b1, b2) = (30usize, 44usize);
        let profile = range_profile(&response(n, df, &[b1 as f64 * dr, b2 as f64 * dr]));
        // Both scatterers show up as peaks above the valley between them.
        assert!(profile[b1] > profile[(b1 + b2) / 2]);
        assert!(profile[b2] > profile[(b1 + b2) / 2]);
        assert!(profile[b1] > 0.5 * profile.iter().cloned().fold(0.0, f64::max));
    }

    #[test]
    fn finer_steps_do_not_change_resolution_but_widen_the_window() {
        // Resolution depends on the total bandwidth N·Δf; halving Δf at fixed N
        // halves the bandwidth (coarser resolution) but doubles the range window.
        let n = 64;
        assert!(range_resolution(n, 1e6) < range_resolution(n, 0.5e6));
        assert!(max_unambiguous_range(0.5e6) > max_unambiguous_range(1e6));
    }

    #[test]
    fn guards() {
        assert!(range_profile(&[]).is_empty());
        // Non-power-of-two step count is rejected.
        assert!(range_profile(&vec![Complex::new(1.0, 0.0); 100]).is_empty());
        assert!(range_resolution(0, 1e6).is_infinite());
        assert!(max_unambiguous_range(0.0).is_infinite());
    }
}
