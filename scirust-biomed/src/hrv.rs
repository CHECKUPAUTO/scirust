//! Heart-Rate Variability (HRV) time-domain metrics.
//!
//! From the RR-interval series, the standard clinical descriptors of autonomic
//! tone: **SDNN** (overall variability), **RMSSD** and **pNN50** (short-term /
//! parasympathetic), plus the mean heart rate. Durations are reported in
//! milliseconds, the clinical convention.

use serde::{Deserialize, Serialize};

/// HRV time-domain metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HrvMetrics {
    /// Mean heart rate (bpm).
    pub mean_hr_bpm: f64,
    /// Standard deviation of NN intervals (ms).
    pub sdnn_ms: f64,
    /// Root mean square of successive differences (ms).
    pub rmssd_ms: f64,
    /// Percentage of successive NN differences greater than 50 ms.
    pub pnn50: f64,
}

/// Compute HRV time-domain metrics from RR intervals (seconds).
pub fn compute_hrv(rr_seconds: &[f64]) -> HrvMetrics {
    let n = rr_seconds.len();
    if n == 0
    {
        return HrvMetrics {
            mean_hr_bpm: 0.0,
            sdnn_ms: 0.0,
            rmssd_ms: 0.0,
            pnn50: 0.0,
        };
    }
    let mean = rr_seconds.iter().sum::<f64>() / n as f64;
    let mean_hr_bpm = if mean > 0.0 { 60.0 / mean } else { 0.0 };
    let var = rr_seconds.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let sdnn_ms = var.sqrt() * 1000.0;

    let diffs: Vec<f64> = rr_seconds.windows(2).map(|w| w[1] - w[0]).collect();
    let (rmssd_ms, pnn50) = if diffs.is_empty()
    {
        (0.0, 0.0)
    }
    else
    {
        let ms = (diffs.iter().map(|&d| d * d).sum::<f64>() / diffs.len() as f64).sqrt() * 1000.0;
        let over = diffs.iter().filter(|&&d| d.abs() > 0.05).count();
        (ms, 100.0 * over as f64 / diffs.len() as f64)
    };

    HrvMetrics {
        mean_hr_bpm,
        sdnn_ms,
        rmssd_ms,
        pnn50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfectly_regular_rhythm_has_zero_variability() {
        let m = compute_hrv(&[0.8; 20]);
        assert!((m.mean_hr_bpm - 75.0).abs() < 1e-9);
        assert!(m.sdnn_ms < 1e-9 && m.rmssd_ms < 1e-9 && m.pnn50 < 1e-9);
    }

    #[test]
    fn alternating_rr_matches_hand_computation() {
        // 0.8 / 0.9 s alternating: std = 50 ms, successive diffs ±100 ms.
        let rr: Vec<f64> = (0..20)
            .map(|k| if k % 2 == 0 { 0.8 } else { 0.9 })
            .collect();
        let m = compute_hrv(&rr);
        assert!((m.sdnn_ms - 50.0).abs() < 1e-6, "SDNN {}", m.sdnn_ms);
        assert!((m.rmssd_ms - 100.0).abs() < 1e-6, "RMSSD {}", m.rmssd_ms);
        assert!((m.pnn50 - 100.0).abs() < 1e-6, "pNN50 {}", m.pnn50);
    }

    #[test]
    fn small_variations_do_not_trigger_pnn50() {
        // ±20 ms successive diffs are below the 50 ms threshold.
        let rr: Vec<f64> = (0..20)
            .map(|k| if k % 2 == 0 { 0.80 } else { 0.82 })
            .collect();
        let m = compute_hrv(&rr);
        assert!(m.pnn50 < 1e-9, "pNN50 {}", m.pnn50);
        assert!(m.rmssd_ms > 0.0);
    }
}
