//! Control-loop performance monitoring: oscillation detection.
//!
//! A persistently oscillating loop (bad tuning, valve stiction) shows a regular,
//! periodic error. This detector measures the regularity of the error's
//! zero-crossings: regularly spaced crossings ⇒ sustained oscillation.

use serde::{Deserialize, Serialize};

/// Oscillation diagnosis for an error trace.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OscillationReport {
    /// Whether a regular oscillation was detected.
    pub oscillating: bool,
    /// Estimated oscillation period in samples (0 if undetected).
    pub period_samples: f64,
    /// Crossing-interval regularity in `[0, 1]` (1 = perfectly periodic).
    pub regularity: f64,
}

/// Detect a sustained oscillation in a control `error` trace. Requires at least
/// `min_crossings` mean-crossings, and a crossing-interval regularity at or
/// above `regularity_thresh` (e.g. `0.8`).
pub fn detect_oscillation(
    error: &[f64],
    min_crossings: usize,
    regularity_thresh: f64,
) -> OscillationReport {
    let n = error.len();
    let none = OscillationReport {
        oscillating: false,
        period_samples: 0.0,
        regularity: 0.0,
    };
    if n < 3
    {
        return none;
    }
    let mean = error.iter().sum::<f64>() / n as f64;
    let mut crossings = Vec::new();
    for i in 1..n
    {
        let prev = error[i - 1] - mean;
        let cur = error[i] - mean;
        if prev == 0.0
        {
            continue;
        }
        if (prev < 0.0) != (cur < 0.0) && cur != 0.0
        {
            crossings.push(i as f64);
        }
    }
    if crossings.len() < min_crossings
    {
        return none;
    }
    let intervals: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
    let imean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if imean <= 0.0
    {
        return none;
    }
    let var = intervals.iter().map(|&x| (x - imean).powi(2)).sum::<f64>() / intervals.len() as f64;
    let cv = var.sqrt() / imean;
    let regularity = (1.0 - cv).clamp(0.0, 1.0);

    // Sustained-amplitude check: a decaying transient also has regular
    // crossings, so require the late-window RMS to be a real fraction of the
    // early-window RMS (the oscillation does not die out).
    let qn = (n / 4).max(1);
    let rms = |slice: &[f64]| -> f64 {
        (slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64).sqrt()
    };
    let first_rms = rms(&error[..qn]);
    let last_rms = rms(&error[n - qn..]);
    let sustained = last_rms > 1e-9 && last_rms >= 0.5 * first_rms;

    OscillationReport {
        oscillating: regularity >= regularity_thresh && sustained,
        period_samples: 2.0 * imean,
        regularity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn detects_a_regular_oscillation() {
        // Error oscillating with period 40 samples.
        let period = 40.0;
        let err: Vec<f64> = (0..800)
            .map(|i| (2.0 * PI * i as f64 / period).sin())
            .collect();
        let r = detect_oscillation(&err, 6, 0.8);
        assert!(r.oscillating, "regularity {}", r.regularity);
        assert!(
            (r.period_samples - period).abs() < 3.0,
            "period {}",
            r.period_samples
        );
    }

    #[test]
    fn ignores_a_settled_loop() {
        // Decaying transient then flat: not a sustained oscillation.
        let err: Vec<f64> = (0..800)
            .map(|i| (-0.05 * i as f64).exp() * (2.0 * PI * i as f64 / 40.0).sin())
            .collect();
        let r = detect_oscillation(&err, 8, 0.8);
        assert!(!r.oscillating, "should not flag a settled loop");
    }
}
