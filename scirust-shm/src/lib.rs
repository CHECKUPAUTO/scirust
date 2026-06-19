//! # scirust-shm — Structural Health Monitoring
//!
//! Pure-Rust, deterministic modal analysis and damage prognosis for civil,
//! wind and aerospace structures:
//!
//! - [`magnitude_spectrum`] / [`natural_frequencies`] — modal frequencies from a
//!   vibration response.
//! - [`damping_half_power`] — modal damping by the half-power bandwidth method.
//! - [`damage_index`] — relative natural-frequency drop (damage softens a structure).
//! - [`paris_cycles_to_failure`] — Paris-law crack-growth life.

use scirust_signal::{Complex, fft_real, hanning};

/// Hann-windowed magnitude half-spectrum of a vibration response (`signal.len()`
/// a power of two).
pub fn magnitude_spectrum(signal: &[f64]) -> Vec<f64> {
    let win = hanning(signal.len());
    let windowed: Vec<f64> = signal.iter().zip(&win).map(|(&x, &w)| x * w).collect();
    fft_real(&windowed).iter().map(Complex::mag).collect()
}

/// Natural frequencies (Hz) = spectral local maxima at least `min_rel_height`
/// times the global peak, returned in ascending order with parabolic sub-bin
/// interpolation.
pub fn natural_frequencies(
    mag: &[f64],
    sample_rate: f64,
    n_fft: usize,
    min_rel_height: f64,
) -> Vec<f64> {
    let bin_hz = sample_rate / n_fft as f64;
    let global = mag.iter().cloned().fold(0.0_f64, f64::max);
    if global <= 0.0
    {
        return Vec::new();
    }
    let threshold = min_rel_height * global;
    let mut out = Vec::new();
    for b in 1..mag.len().saturating_sub(1)
    {
        let (m0, m1, m2) = (mag[b - 1], mag[b], mag[b + 1]);
        if m1 >= threshold && m1 > m0 && m1 >= m2
        {
            let denom = m0 - 2.0 * m1 + m2;
            let delta = if denom.abs() > 1e-20
            {
                0.5 * (m0 - m2) / denom
            }
            else
            {
                0.0
            };
            out.push((b as f64 + delta) * bin_hz);
        }
    }
    out
}

/// Modal damping ratio `ζ` of the mode near `peak_hz`, by the half-power
/// (`−3 dB`, amplitude `1/√2`) bandwidth method: `ζ = (f₂ − f₁) / (2·f_n)`.
/// Returns `None` if the half-power points cannot be bracketed.
pub fn damping_half_power(
    mag: &[f64],
    sample_rate: f64,
    n_fft: usize,
    peak_hz: f64,
) -> Option<f64> {
    let bin_hz = sample_rate / n_fft as f64;
    let peak_bin = (peak_hz / bin_hz).round() as usize;
    if peak_bin == 0 || peak_bin >= mag.len()
    {
        return None;
    }
    let peak = mag[peak_bin];
    let half = peak / core::f64::consts::SQRT_2;

    // Walk left and right to the half-power crossings, linearly interpolated.
    let left = crossing_below(mag, peak_bin, half, -1)?;
    let right = crossing_below(mag, peak_bin, half, 1)?;
    let f1 = left * bin_hz;
    let f2 = right * bin_hz;
    let f_n = peak_bin as f64 * bin_hz;
    Some((f2 - f1) / (2.0 * f_n))
}

/// Fractional bin position where `mag` first falls to `level` walking in
/// `step` (`-1` left, `+1` right) from `start`, linearly interpolated.
fn crossing_below(mag: &[f64], start: usize, level: f64, step: isize) -> Option<f64> {
    let mut b = start as isize;
    loop
    {
        let next = b + step;
        if next < 0 || next as usize >= mag.len()
        {
            return None;
        }
        let (mb, mn) = (mag[b as usize], mag[next as usize]);
        if mn <= level && mb > level
        {
            // Interpolate between b and next.
            let frac = (mb - level) / (mb - mn);
            return Some(b as f64 + step as f64 * frac);
        }
        b = next;
    }
}

/// Damage index = relative natural-frequency drop `(f_baseline − f_current) /
/// f_baseline`. Positive means a softer (damaged) structure; ~0 means healthy.
pub fn damage_index(baseline_hz: f64, current_hz: f64) -> f64 {
    if baseline_hz <= 0.0
    {
        return 0.0;
    }
    (baseline_hz - current_hz) / baseline_hz
}

/// Paris-law fatigue life (number of cycles) for a crack growing from `a0` to
/// `a_crit` under stress range `delta_sigma`, geometry factor `y`:
/// `da/dN = C·(ΔK)^m`, `ΔK = Y·Δσ·√(π·a)`. Closed form for `m ≠ 2`.
pub fn paris_cycles_to_failure(
    c: f64,
    m: f64,
    delta_sigma: f64,
    y: f64,
    a0: f64,
    a_crit: f64,
) -> f64 {
    let coeff = c * (y * delta_sigma * core::f64::consts::PI.sqrt()).powf(m);
    if coeff <= 0.0 || a_crit <= a0
    {
        return f64::INFINITY;
    }
    if (m - 2.0).abs() < 1e-9
    {
        // m = 2: N = ln(a_crit/a0) / coeff.
        (a_crit / a0).ln() / coeff
    }
    else
    {
        let p = 1.0 - m / 2.0;
        (a_crit.powf(p) - a0.powf(p)) / (coeff * p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic Lorentzian magnitude spectrum peaked at `f_n` with
    /// half-width-at-half-max `hwhm` (Hz).
    fn lorentzian(n_fft: usize, sample_rate: f64, f_n: f64, hwhm: f64, amp: f64) -> Vec<f64> {
        let bin_hz = sample_rate / n_fft as f64;
        (0..=n_fft / 2)
            .map(|b| {
                let f = b as f64 * bin_hz;
                let x = (f - f_n) / hwhm;
                amp / (1.0 + x * x).sqrt()
            })
            .collect()
    }

    #[test]
    fn finds_natural_frequencies() {
        let (n, sr) = (4096usize, 4096.0);
        // Three modes; superpose narrow Lorentzians.
        let mut mag = vec![0.0; n / 2 + 1];
        for &(f, a) in &[(10.0, 1.0), (25.0, 0.8), (40.0, 0.6)]
        {
            for (b, v) in lorentzian(n, sr, f, 0.5, a).iter().enumerate()
            {
                mag[b] += v;
            }
        }
        let modes = natural_frequencies(&mag, sr, n, 0.3);
        assert_eq!(modes.len(), 3, "found {modes:?}");
        for (got, want) in modes.iter().zip(&[10.0, 25.0, 40.0])
        {
            assert!((got - want).abs() < 0.2, "mode {got} vs {want}");
        }
    }

    #[test]
    fn damping_from_half_power_bandwidth() {
        let (n, sr) = (8192usize, 4096.0); // fine resolution
        let f_n = 50.0;
        let zeta = 0.02;
        let hwhm = zeta * f_n; // Lorentzian HWHM = ζ·f_n
        let mag = lorentzian(n, sr, f_n, hwhm, 1.0);
        let z = damping_half_power(&mag, sr, n, f_n).expect("bracketed");
        assert!((z - zeta).abs() < 2e-3, "damping {z} (want {zeta})");
    }

    #[test]
    fn damage_index_detects_frequency_drop() {
        // 5% natural-frequency drop.
        assert!((damage_index(10.0, 9.5) - 0.05).abs() < 1e-12);
        assert!(damage_index(10.0, 10.0).abs() < 1e-12); // healthy
    }

    #[test]
    fn paris_law_matches_numerical_integral() {
        let (c, m, ds, y, a0, ac) = (1e-11, 3.0, 100.0, 1.0, 1e-3, 1e-2);
        let closed = paris_cycles_to_failure(c, m, ds, y, a0, ac);
        // Trapezoidal integral of dN/da = 1 / (C (Y Δσ √(π a))^m).
        let steps = 200_000;
        let da = (ac - a0) / steps as f64;
        let integrand = |a: f64| 1.0 / (c * (y * ds * (core::f64::consts::PI * a).sqrt()).powf(m));
        let mut num = 0.0;
        for i in 0..steps
        {
            let a = a0 + (i as f64 + 0.5) * da;
            num += integrand(a) * da;
        }
        let rel = (closed - num).abs() / num;
        assert!(rel < 1e-3, "closed {closed} vs numeric {num} (rel {rel})");
    }

    #[test]
    fn spectrum_peaks_at_excited_tone() {
        let (n, sr) = (4096usize, 4096.0);
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * core::f64::consts::PI * 30.0 * i as f64 / sr).sin())
            .collect();
        let mag = magnitude_spectrum(&sig);
        let modes = natural_frequencies(&mag, sr, n, 0.5);
        assert!(modes.iter().any(|f| (f - 30.0).abs() < 0.2), "{modes:?}");
    }
}
