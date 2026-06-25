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

    #[test]
    fn rr_intervals_convert_samples_to_seconds() {
        // Peaks one second apart at 250 Hz are 250 samples apart -> RR = 1.0 s.
        let rr = rr_intervals(&[0, 250, 500, 750], 250.0);
        assert_eq!(rr.len(), 3);
        for v in &rr
        {
            assert!((v - 1.0).abs() < 1e-12, "RR {v}");
        }
        // Uneven spacing at 200 Hz: gaps 100 and 300 samples -> 0.5 s and 1.5 s.
        let rr2 = rr_intervals(&[0, 100, 400], 200.0);
        assert!(
            (rr2[0] - 0.5).abs() < 1e-12 && (rr2[1] - 1.5).abs() < 1e-12,
            "{rr2:?}"
        );
        // Fewer than two peaks yields no intervals.
        assert!(rr_intervals(&[42], 250.0).is_empty());
        assert!(rr_intervals(&[], 250.0).is_empty());
    }

    #[test]
    fn heart_rate_from_one_second_rr_is_exactly_60_bpm() {
        // R-peaks spaced exactly 1.0 s apart (250 samples at fs = 250) -> 60 bpm.
        let peaks: Vec<usize> = (0..10).map(|k| k * 250).collect();
        let hr = heart_rate_bpm(&peaks, 250.0);
        assert!((hr - 60.0).abs() < 1e-12, "HR {hr}");
    }

    #[test]
    fn heart_rate_uses_mean_rr_not_per_beat_average() {
        // Gaps 0.5 s and 1.5 s (mean RR = 1.0 s) -> 60/mean = 60 bpm, even though
        // the per-beat rates (120 and 40 bpm) average to 80. This pins the
        // "60 / mean(RR)" definition.
        let hr = heart_rate_bpm(&[0, 100, 400], 200.0);
        assert!((hr - 60.0).abs() < 1e-12, "HR {hr}");
        // A single peak has no interval -> 0 bpm (not a panic / NaN).
        assert_eq!(heart_rate_bpm(&[7], 250.0), 0.0);
        assert_eq!(heart_rate_bpm(&[], 250.0), 0.0);
    }

    #[test]
    fn detects_sixty_bpm_with_exact_indices() {
        // RR = 1.0 s at fs = 250 (250 samples); first beat at 250.
        let sr = 250.0;
        let n = 2500;
        let beats: Vec<usize> = (0..9).map(|k| 250 + k * 250).filter(|&b| b < n).collect();
        let ecg = synth_ecg(&beats, n, sr);
        let peaks = detect_r_peaks(&ecg, sr);
        assert_eq!(peaks, beats, "exact peak indices");
        let hr = heart_rate_bpm(&peaks, sr);
        assert!((hr - 60.0).abs() < 0.5, "HR {hr}");
    }

    #[test]
    fn refractory_period_rejects_too_close_peaks() {
        // Two QRS complexes 152 ms apart (38 samples at 250 Hz) are inside the
        // 200 ms refractory window, so the detector keeps only the first of the
        // pair; a later well-separated beat is kept.
        let sr = 250.0;
        let beats = [200usize, 238, 700];
        let ecg = synth_ecg(&beats, 1200, sr);
        let peaks = detect_r_peaks(&ecg, sr);
        assert_eq!(peaks, vec![200, 700], "refractory rejection, got {peaks:?}");
    }

    #[test]
    fn degenerate_signals_yield_no_peaks() {
        // Too short to differentiate.
        assert!(detect_r_peaks(&[1.0, 2.0], 250.0).is_empty());
        assert!(detect_r_peaks(&[], 250.0).is_empty());
        // Flat (constant) signal: zero derivative everywhere -> no QRS energy.
        assert!(detect_r_peaks(&[0.7; 500], 250.0).is_empty());
    }

    #[test]
    fn classify_rhythm_edge_cases_and_exact_cv_threshold() {
        // Empty input is reported as Normal (no rhythm to flag).
        assert_eq!(classify_rhythm(&[]), RhythmClass::Normal);
        // Two-value alternation a/b has coefficient of variation |a-b|/(a+b).
        // a=0.6, b=0.9 -> CV = 0.3/1.5 = 0.20 > 0.15 -> Irregular (mean 0.75 s
        // would otherwise be a normal 80 bpm rate, so irregularity must win).
        assert_eq!(
            classify_rhythm(&[0.6, 0.9, 0.6, 0.9]),
            RhythmClass::Irregular
        );
        // a=0.72, b=0.88 -> CV = 0.16/1.6 = 0.10 < 0.15, mean 0.8 s = 75 bpm.
        assert_eq!(
            classify_rhythm(&[0.72, 0.88, 0.72, 0.88]),
            RhythmClass::Normal
        );
    }
}
