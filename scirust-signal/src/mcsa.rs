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

/// Outcome of an air-gap eccentricity MCSA evaluation.
#[derive(Debug, Clone)]
pub struct EccentricityResult {
    /// Supply fundamental amplitude (linear).
    pub fundamental: f64,
    /// Strongest of the `f ± k·f_rotor` sidebands relative to the fundamental (dB).
    pub sideband_db: f64,
    /// Whether the sideband exceeds `threshold_db` (eccentricity flagged).
    pub eccentric: bool,
}

/// Mixed air-gap eccentricity MCSA: sidebands appear around the supply at
/// `f_supply ± k·f_rotor`, where `f_rotor` is the rotor mechanical frequency
/// (Hz). The strongest sideband relative to the fundamental (dB) flags
/// eccentricity above `threshold_db` (e.g. `−50`). These lines sit far from the
/// fundamental, so a ±2-bin search is safe.
pub fn analyze_eccentricity(
    current: &[f64],
    sample_rate: f64,
    supply_hz: f64,
    rotor_freq_hz: f64,
    k_harmonic: u32,
    threshold_db: f64,
) -> EccentricityResult {
    let n = current.len();
    let win = hanning(n);
    let windowed: Vec<f64> = current.iter().zip(&win).map(|(&x, &w)| x * w).collect();
    let spec = fft_real(&windowed);

    let fundamental = peak_mag_near(&spec, n, sample_rate, supply_hz, 2);
    let off = k_harmonic as f64 * rotor_freq_hz;
    let lower = peak_mag_near(&spec, n, sample_rate, supply_hz - off, 2);
    let upper = peak_mag_near(&spec, n, sample_rate, supply_hz + off, 2);
    let strongest = lower.max(upper);
    let sideband_db = if fundamental > 0.0 && strongest > 0.0
    {
        20.0 * (strongest / fundamental).log10()
    }
    else
    {
        f64::NEG_INFINITY
    };
    EccentricityResult {
        fundamental,
        sideband_db,
        eccentric: sideband_db >= threshold_db,
    }
}

/// Dominant motor fault from a unified MCSA diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorFault {
    /// No fault signature above threshold.
    Healthy,
    /// Broken rotor bar(s).
    BrokenBar,
    /// Air-gap eccentricity.
    Eccentricity,
}

/// Combined MCSA diagnosis: runs the broken-bar and eccentricity analyses and
/// reports the dominant fault.
#[derive(Debug, Clone)]
pub struct MotorDiagnosis {
    pub broken_bar: BrokenBarResult,
    pub eccentricity: EccentricityResult,
    pub dominant: MotorFault,
}

/// Diagnose a motor from one stator-current capture. `slip` is per-unit;
/// `rotor_freq_hz` is the rotor mechanical frequency. A fault is declared when
/// its primary sideband exceeds `fault_db` (e.g. `−50`); the stronger of the two
/// signatures wins.
pub fn diagnose_motor(
    current: &[f64],
    sample_rate: f64,
    supply_hz: f64,
    slip: f64,
    rotor_freq_hz: f64,
    fault_db: f64,
) -> MotorDiagnosis {
    let broken_bar = analyze_broken_bar(current, sample_rate, supply_hz, slip, 1);
    let eccentricity =
        analyze_eccentricity(current, sample_rate, supply_hz, rotor_freq_hz, 1, fault_db);
    let bb_fault = broken_bar.sideband_db >= fault_db;
    let ecc_fault = eccentricity.sideband_db >= fault_db;
    let dominant = if bb_fault && (!ecc_fault || broken_bar.sideband_db >= eccentricity.sideband_db)
    {
        MotorFault::BrokenBar
    }
    else if ecc_fault
    {
        MotorFault::Eccentricity
    }
    else
    {
        MotorFault::Healthy
    };
    MotorDiagnosis {
        broken_bar,
        eccentricity,
        dominant,
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

    #[test]
    fn detects_air_gap_eccentricity() {
        // Supply 50 Hz, rotor 12.5 Hz -> sidebands at 37.5 / 62.5 Hz.
        let (n, sr, supply, f_rot) = (4096usize, 4096.0, 50.0, 12.5);
        let ratio = 10f64.powf(-45.0 / 20.0);
        let current: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * supply * t).sin()
                    + ratio * (2.0 * PI * (supply - f_rot) * t).sin()
                    + ratio * (2.0 * PI * (supply + f_rot) * t).sin()
            })
            .collect();
        let r = analyze_eccentricity(&current, sr, supply, f_rot, 1, -50.0);
        assert!(r.eccentric, "sideband_db = {}", r.sideband_db);
        assert!(
            (r.sideband_db - (-45.0)).abs() < 2.0,
            "sideband_db = {}",
            r.sideband_db
        );

        // Healthy motor: no eccentricity sidebands.
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * supply * i as f64 / sr).sin())
            .collect();
        let h = analyze_eccentricity(&clean, sr, supply, f_rot, 1, -50.0);
        assert!(!h.eccentric, "false eccentricity, db = {}", h.sideband_db);
    }

    #[test]
    fn unified_diagnosis_distinguishes_faults() {
        let (n, sr, supply) = (4096usize, 4096.0, 50.0);
        let (s, f_rot) = (0.03, 12.5); // broken-bar sidebands 47/53, ecc 37.5/62.5
        let bb_ratio = 10f64.powf(-45.0 / 20.0);
        let ecc_ratio = 10f64.powf(-45.0 / 20.0);

        // Broken-bar-only current.
        let bb_sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * supply * t).sin()
                    + bb_ratio * (2.0 * PI * supply * (1.0 - 2.0 * s) * t).sin()
                    + bb_ratio * (2.0 * PI * supply * (1.0 + 2.0 * s) * t).sin()
            })
            .collect();
        let d = diagnose_motor(&bb_sig, sr, supply, s, f_rot, -50.0);
        assert_eq!(d.dominant, MotorFault::BrokenBar, "{:?}", d.dominant);

        // Eccentricity-only current.
        let ecc_sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * supply * t).sin()
                    + ecc_ratio * (2.0 * PI * (supply - f_rot) * t).sin()
                    + ecc_ratio * (2.0 * PI * (supply + f_rot) * t).sin()
            })
            .collect();
        let d2 = diagnose_motor(&ecc_sig, sr, supply, s, f_rot, -50.0);
        assert_eq!(d2.dominant, MotorFault::Eccentricity, "{:?}", d2.dominant);

        // Clean current.
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * supply * i as f64 / sr).sin())
            .collect();
        let d3 = diagnose_motor(&clean, sr, supply, s, f_rot, -50.0);
        assert_eq!(d3.dominant, MotorFault::Healthy, "{:?}", d3.dominant);
    }
}
