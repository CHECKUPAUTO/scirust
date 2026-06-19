//! ECG R-peak detection and rhythm classification.
//!
//! A Pan–Tompkins-style pipeline — derivative, squaring, moving-window
//! integration, adaptive threshold with a physiological refractory period —
//! locates the R peaks, from which heart rate, RR intervals and a coarse rhythm
//! class (normal / bradycardia / tachycardia / irregular) are derived. Pure
//! deterministic `f64`.

use serde::{Deserialize, Serialize};

/// Detect R-peak sample indices in an ECG `signal` sampled at `sample_rate` Hz.
pub fn detect_r_peaks(signal: &[f64], sample_rate: f64) -> Vec<usize> {
    let n = signal.len();
    if n < 3
    {
        return Vec::new();
    }
    // 1. Derivative (central difference) then square.
    let mut sq = vec![0.0; n];
    for i in 1..n - 1
    {
        let d = (signal[i + 1] - signal[i - 1]) * 0.5;
        sq[i] = d * d;
    }
    // 2. Moving-window integration (~120 ms window).
    let win = ((0.12 * sample_rate).round() as usize).max(1);
    let mut mwi = vec![0.0; n];
    let mut acc = 0.0;
    for i in 0..n
    {
        acc += sq[i];
        if i >= win
        {
            acc -= sq[i - win];
        }
        mwi[i] = acc / win as f64;
    }
    // 3. Adaptive threshold + refractory peak picking on the MWI.
    let peak_mwi = mwi.iter().cloned().fold(0.0_f64, f64::max);
    if peak_mwi <= 0.0
    {
        return Vec::new();
    }
    let threshold = 0.3 * peak_mwi;
    let refractory = (0.2 * sample_rate).round() as usize; // 200 ms

    // Region-based picking: each contiguous run above threshold is one beat.
    // The R peak is the raw-signal max over the run, expanded back by `win` to
    // undo the trailing moving-window lag.
    let mut peaks: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < n
    {
        if mwi[i] < threshold
        {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && mwi[i] >= threshold
        {
            i += 1;
        }
        let lo = start.saturating_sub(win);
        let mut best = lo;
        for j in lo..i
        {
            if signal[j] > signal[best]
            {
                best = j;
            }
        }
        if peaks
            .last()
            .map(|&p| best.saturating_sub(p) >= refractory)
            .unwrap_or(true)
        {
            peaks.push(best);
        }
    }
    peaks
}

/// RR intervals (seconds) from R-peak indices.
pub fn rr_intervals(peaks: &[usize], sample_rate: f64) -> Vec<f64> {
    peaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 / sample_rate)
        .collect()
}

/// Mean heart rate (beats per minute) from R-peak indices.
pub fn heart_rate_bpm(peaks: &[usize], sample_rate: f64) -> f64 {
    let rr = rr_intervals(peaks, sample_rate);
    if rr.is_empty()
    {
        return 0.0;
    }
    let mean_rr = rr.iter().sum::<f64>() / rr.len() as f64;
    if mean_rr > 0.0 { 60.0 / mean_rr } else { 0.0 }
}

/// Coarse rhythm class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhythmClass {
    /// Regular rhythm, 60–100 bpm.
    Normal,
    /// Slow: < 60 bpm.
    Bradycardia,
    /// Fast: > 100 bpm.
    Tachycardia,
    /// Irregular RR (high beat-to-beat variability), e.g. atrial fibrillation.
    Irregular,
}

/// Classify rhythm from RR intervals: irregularity (coefficient of variation
/// `> 0.15`) takes precedence, then rate.
pub fn classify_rhythm(rr: &[f64]) -> RhythmClass {
    if rr.is_empty()
    {
        return RhythmClass::Normal;
    }
    let mean = rr.iter().sum::<f64>() / rr.len() as f64;
    if mean <= 0.0
    {
        return RhythmClass::Normal;
    }
    let var = rr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / rr.len() as f64;
    let cv = var.sqrt() / mean;
    if cv > 0.15
    {
        return RhythmClass::Irregular;
    }
    let hr = 60.0 / mean;
    if hr < 60.0
    {
        RhythmClass::Bradycardia
    }
    else if hr > 100.0
    {
        RhythmClass::Tachycardia
    }
    else
    {
        RhythmClass::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Synthetic ECG: a sharp Gaussian QRS at each beat plus a slow baseline.
    fn synth_ecg(beats: &[usize], n: usize, sample_rate: f64) -> Vec<f64> {
        let qrs_sd = 0.01 * sample_rate; // ~10 ms
        (0..n)
            .map(|i| {
                let baseline = 0.05 * (2.0 * PI * 0.3 * i as f64 / sample_rate).sin();
                let qrs: f64 = beats
                    .iter()
                    .map(|&b| {
                        let d = (i as f64 - b as f64) / qrs_sd;
                        (-0.5 * d * d).exp()
                    })
                    .sum();
                baseline + qrs
            })
            .collect()
    }

    #[test]
    fn detects_r_peaks_at_known_locations() {
        let sr = 250.0;
        let n = 2500; // 10 s
        // 75 bpm -> RR = 0.8 s = 200 samples, starting at 150.
        let beats: Vec<usize> = (0..12).map(|k| 150 + k * 200).filter(|&b| b < n).collect();
        let ecg = synth_ecg(&beats, n, sr);
        let peaks = detect_r_peaks(&ecg, sr);
        assert_eq!(peaks.len(), beats.len(), "got {peaks:?}");
        for (p, b) in peaks.iter().zip(&beats)
        {
            assert!(
                (*p as isize - *b as isize).abs() <= 5,
                "peak {p} vs beat {b}"
            );
        }
        let hr = heart_rate_bpm(&peaks, sr);
        assert!((hr - 75.0).abs() < 2.0, "HR {hr}");
    }

    #[test]
    fn rhythm_classification() {
        // Regular 75 bpm.
        let rr_normal = vec![0.8; 10];
        assert_eq!(classify_rhythm(&rr_normal), RhythmClass::Normal);
        // Regular 50 bpm.
        assert_eq!(classify_rhythm(&[1.2; 10]), RhythmClass::Bradycardia);
        // Regular 120 bpm.
        assert_eq!(classify_rhythm(&[0.5; 10]), RhythmClass::Tachycardia);
        // Irregular RR (AFib-like): alternating long/short.
        let rr_afib = vec![0.6, 1.0, 0.5, 1.1, 0.7, 0.95, 0.55, 1.05];
        assert_eq!(classify_rhythm(&rr_afib), RhythmClass::Irregular);
    }
}
