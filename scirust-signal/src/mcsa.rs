//! Motor Current Signature Analysis (MCSA).
//!
//! Detects rotor/electrical faults from the **stator current** spectrum rather
//! than vibration — every motor already has a current sensor, so MCSA reaches
//! machines an accelerometer cannot. The hallmark of **broken rotor bars** is a
//! pair of sidebands around the supply fundamental at `(1 ± 2·k·s)·f_supply`,
//! where `s` is the per-unit slip and `k = 1, 2, …`; their amplitude relative
//! to the fundamental (in dB) grades the severity.
//!
//! A low-leakage Hann window is applied before the FFT: for an on-bin supply
//! fundamental the window confines its leakage to the two adjacent bins, so the
//! close, low-level fault sidebands are not swamped by the fundamental.

use crate::complex::Complex;
use crate::fft::fft_real;
use crate::windows::hanning;

/// Severity of a broken-rotor-bar signature, from the sideband-to-fundamental
/// level (dB). Thresholds follow common MCSA practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarSeverity {
    /// Sideband `< −60 dB`: no significant rotor-bar signature.
    Healthy,
    /// `−60 … −50 dB`: developing / possible defect, monitor.
    Developing,
    /// `−50 … −45 dB`: broken bar likely.
    Broken,
    /// `≥ −45 dB`: multiple broken bars / severe.
    Severe,
}

impl BarSeverity {
    fn from_db(db: f64) -> Self {
        if db < -60.0
        {
            BarSeverity::Healthy
        }
        else if db < -50.0
        {
            BarSeverity::Developing
        }
        else if db < -45.0
        {
            BarSeverity::Broken
        }
        else
        {
            BarSeverity::Severe
        }
    }
}

/// Outcome of a broken-rotor-bar MCSA evaluation at harmonic order `k`.
#[derive(Debug, Clone)]
pub struct BrokenBarResult {
    /// Supply fundamental amplitude (linear spectrum magnitude).
    pub fundamental: f64,
    /// Lower sideband `(1 − 2ks)·f` amplitude (linear).
    pub lower_sideband: f64,
    /// Upper sideband `(1 + 2ks)·f` amplitude (linear).
    pub upper_sideband: f64,
    /// Lower-sideband frequency (Hz).
    pub lower_hz: f64,
    /// Upper-sideband frequency (Hz).
    pub upper_hz: f64,
    /// Strongest sideband relative to the fundamental, in dB (`≤ 0`).
    pub sideband_db: f64,
    /// Severity verdict derived from `sideband_db`.
    pub severity: BarSeverity,
}

/// Per-unit slip `s = (n_sync − n_rotor) / n_sync`, with synchronous mechanical
/// speed `n_sync = f_supply / pole_pairs` (Hz). Clamped to `[0, 1]`.
pub fn slip(supply_hz: f64, pole_pairs: u32, rotor_speed_hz: f64) -> f64 {
    let n_sync = supply_hz / pole_pairs.max(1) as f64;
    if n_sync <= 0.0
    {
        return 0.0;
    }
    ((n_sync - rotor_speed_hz) / n_sync).clamp(0.0, 1.0)
}

/// Peak magnitude within `±half_window` bins of `target_hz` in a half-spectrum.
fn peak_mag_near(
    spec: &[Complex],
    n_fft: usize,
    sample_rate: f64,
    target_hz: f64,
    half_window: usize,
) -> f64 {
    if spec.is_empty() || target_hz <= 0.0
    {
        return 0.0;
    }
    let bin = (target_hz * n_fft as f64 / sample_rate).round() as isize;
    let lo = (bin - half_window as isize).max(0) as usize;
    let hi = ((bin + half_window as isize).max(0) as usize).min(spec.len() - 1);
    if lo > hi
    {
        return 0.0;
    }
    spec[lo..=hi]
        .iter()
        .map(Complex::mag)
        .fold(0.0_f64, f64::max)
}

/// Broken-rotor-bar MCSA on a stator-current signal.
///
/// `current.len()` must be a power of two. `slip` is per-unit; `k_harmonic` is
/// the sideband order (`1` for the primary `(1 ± 2s)f` pair). The fundamental
/// is located within ±2 bins of `supply_hz`; each sideband within ±2 bins of
/// its predicted frequency.
pub fn analyze_broken_bar(
    current: &[f64],
    sample_rate: f64,
    supply_hz: f64,
    slip: f64,
    k_harmonic: u32,
) -> BrokenBarResult {
    let n = current.len();
    let win = hanning(n);
    let windowed: Vec<f64> = current.iter().zip(&win).map(|(&x, &w)| x * w).collect();
    let spec = fft_real(&windowed);

    let fundamental = peak_mag_near(&spec, n, sample_rate, supply_hz, 2);

    let offset = 2.0 * k_harmonic as f64 * slip;
    let lower_hz = supply_hz * (1.0 - offset);
    let upper_hz = supply_hz * (1.0 + offset);
    // Sidebands are searched with a tight ±1-bin window: a Hann-windowed
    // on-bin fundamental leaks to its immediate neighbours at ~−6 dB, so a
    // wider window would capture that leak instead of the real sideband.
    let lower_sideband = peak_mag_near(&spec, n, sample_rate, lower_hz, 1);
    let upper_sideband = peak_mag_near(&spec, n, sample_rate, upper_hz, 1);

    let strongest = lower_sideband.max(upper_sideband);
    let sideband_db = if fundamental > 0.0 && strongest > 0.0
    {
        20.0 * (strongest / fundamental).log10()
    }
    else
    {
        f64::NEG_INFINITY
    };

    BrokenBarResult {
        fundamental,
        lower_sideband,
        upper_sideband,
        lower_hz,
        upper_hz,
        sideband_db,
        severity: BarSeverity::from_db(sideband_db),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Synthesize a stator current: fundamental at `supply` plus broken-bar
    /// sidebands at `(1 ± 2s)·supply`, each at `sideband_db` relative level.
    fn synth(n: usize, sample_rate: f64, supply: f64, slip: f64, sideband_db: f64) -> Vec<f64> {
        let ratio = 10f64.powf(sideband_db / 20.0);
        let off = 2.0 * slip;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let f = (2.0 * PI * supply * t).sin();
                let lo = ratio * (2.0 * PI * supply * (1.0 - off) * t).sin();
                let hi = ratio * (2.0 * PI * supply * (1.0 + off) * t).sin();
                f + lo + hi
            })
            .collect()
    }

    #[test]
    fn detects_broken_bar_at_injected_level() {
        // 1 Hz bin resolution; sidebands at 47/53 Hz (3 bins from 50).
        let (n, sr, supply, s) = (4096usize, 4096.0, 50.0, 0.03);
        let current = synth(n, sr, supply, s, -50.0);
        let r = analyze_broken_bar(&current, sr, supply, s, 1);

        assert!((r.lower_hz - 47.0).abs() < 1e-6 && (r.upper_hz - 53.0).abs() < 1e-6);
        // Recovered sideband level is within a couple dB of the injected −50 dB.
        assert!(
            (r.sideband_db - (-50.0)).abs() < 2.0,
            "sideband_db = {} (want ~ -50)",
            r.sideband_db
        );
        assert_eq!(r.severity, BarSeverity::Broken);
    }

    #[test]
    fn healthy_motor_has_no_sidebands() {
        let (n, sr, supply, s) = (4096usize, 4096.0, 50.0, 0.03);
        // -90 dB sidebands ≈ healthy (well below the -60 dB floor).
        let current = synth(n, sr, supply, s, -90.0);
        let r = analyze_broken_bar(&current, sr, supply, s, 1);
        assert!(r.sideband_db < -60.0, "sideband_db = {}", r.sideband_db);
        assert_eq!(r.severity, BarSeverity::Healthy);
    }

    #[test]
    fn slip_matches_definition() {
        // 50 Hz, 2 pole pairs -> n_sync = 25 Hz; rotor at 24.25 Hz -> s = 0.03.
        let s = slip(50.0, 2, 24.25);
        assert!((s - 0.03).abs() < 1e-9, "slip = {s}");
        // No load: rotor at synchronous speed -> s = 0.
        assert_eq!(slip(50.0, 2, 25.0), 0.0);
    }
}
