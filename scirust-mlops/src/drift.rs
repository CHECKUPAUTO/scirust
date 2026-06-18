use serde::{Deserialize, Serialize};

/// Type of drift detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftType {
    NoDrift,
    DataDrift,
    ModelDrift,
    Both,
}

/// A drift detection report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub drift_type: DriftType,
    pub data_drift_score: f64,
    pub model_drift_score: f64,
    pub threshold: f64,
    pub details: Vec<(String, f64)>,
    pub sample_count: u64,
}

/// Statistical test for detecting data distribution shift.
///
/// Uses a simplified Population Stability Index (PSI) comparing
/// a reference distribution (baseline) against recent observations.
///
/// PSI < 0.1: no significant drift
/// 0.1 <= PSI < 0.25: moderate drift — monitor
/// PSI >= 0.25: significant drift — retrain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDriftDetector {
    /// Reference (baseline) histogram
    pub reference_hist: Vec<u64>,
    /// Number of bins
    pub n_bins: usize,
    /// Data range [min, max]
    pub data_min: f64,
    pub data_max: f64,
    /// Current window of observations
    window: Vec<f64>,
    /// Window size before evaluating drift
    pub window_size: usize,
    /// Drift threshold (PSI)
    pub threshold: f64,
    /// Total samples observed
    sample_count: u64,
}

impl DataDriftDetector {
    /// Create a drift detector from a reference dataset.
    pub fn from_reference(
        reference: &[f64],
        n_bins: usize,
        window_size: usize,
        threshold: f64,
    ) -> Self {
        if reference.is_empty() || n_bins == 0
        {
            return Self::empty(n_bins, window_size, threshold);
        }
        let min = reference.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = reference.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;
        if range < f64::EPSILON
        {
            return Self::empty(n_bins, window_size, threshold);
        }
        let bin_width = range / n_bins as f64;
        let mut hist = vec![0u64; n_bins];
        for &v in reference
        {
            let mut bin = ((v - min) / bin_width) as usize;
            if bin >= n_bins
            {
                bin = n_bins - 1;
            }
            hist[bin] += 1;
        }
        Self {
            reference_hist: hist,
            n_bins,
            data_min: min,
            data_max: max,
            window: Vec::with_capacity(window_size),
            window_size,
            threshold,
            sample_count: 0,
        }
    }

    fn empty(n_bins: usize, window_size: usize, threshold: f64) -> Self {
        Self {
            reference_hist: vec![0; n_bins],
            n_bins,
            data_min: 0.0,
            data_max: 0.0,
            window: Vec::new(),
            window_size,
            threshold,
            sample_count: 0,
        }
    }

    /// Add a new observation.
    pub fn add_sample(&mut self, value: f64) {
        self.window.push(value);
        self.sample_count += 1;
    }

    /// Check for drift. Returns `Some(DriftReport)` if the window is full.
    pub fn check(&mut self) -> Option<DriftReport> {
        if self.window.len() < self.window_size
        {
            return None;
        }
        // Build current histogram
        let range = self.data_max - self.data_min;
        if range < f64::EPSILON
        {
            return None;
        }
        let bin_width = range / self.n_bins as f64;
        let mut current_hist = vec![0u64; self.n_bins];
        for &v in &self.window
        {
            let mut bin = ((v - self.data_min) / bin_width) as usize;
            if bin >= self.n_bins
            {
                bin = self.n_bins - 1;
            }
            current_hist[bin] += 1;
        }
        let psi = self.compute_psi(&current_hist);
        let drifted = psi > self.threshold;
        // Clear window after evaluation
        self.window.clear();
        let drift_type = if drifted
        {
            DriftType::DataDrift
        }
        else
        {
            DriftType::NoDrift
        };
        Some(DriftReport {
            drift_type,
            data_drift_score: psi,
            model_drift_score: 0.0,
            threshold: self.threshold,
            details: vec![
                ("psi".to_string(), psi),
                ("window_size".to_string(), self.window_size as f64),
            ],
            sample_count: self.sample_count,
        })
    }

    /// Compute Population Stability Index.
    fn compute_psi(&self, current: &[u64]) -> f64 {
        let ref_total = self.reference_hist.iter().sum::<u64>() as f64;
        let cur_total = current.iter().sum::<u64>() as f64;
        if ref_total < 1.0 || cur_total < 1.0
        {
            return 0.0;
        }
        self.reference_hist
            .iter()
            .zip(current.iter())
            .take(self.n_bins)
            .map(|(r, c)| {
                let ref_pct = (*r as f64 + 0.5) / ref_total;
                let cur_pct = (*c as f64 + 0.5) / cur_total;
                (cur_pct - ref_pct) * (cur_pct / ref_pct).ln()
            })
            .sum()
    }

    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub fn window_fill(&self) -> usize {
        self.window.len()
    }
}

/// Model drift detector.
///
/// Tracks the divergence between predicted and actual outcomes over time.
/// Uses a rolling window to compute accuracy / error metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDriftDetector {
    /// Window of (prediction, actual) pairs
    window: Vec<(f64, f64)>,
    pub window_size: usize,
    /// Baseline error metric
    pub baseline_error: f64,
    /// Drift threshold (relative increase factor)
    pub threshold_factor: f64,
    sample_count: u64,
}

impl ModelDriftDetector {
    pub fn new(window_size: usize, baseline_error: f64, threshold_factor: f64) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            baseline_error,
            threshold_factor,
            sample_count: 0,
        }
    }

    /// Add a prediction/actual pair.
    pub fn add_observation(&mut self, prediction: f64, actual: f64) {
        self.window.push((prediction, actual));
        self.sample_count += 1;
    }

    /// Check for model drift. Returns `Some` if window is full.
    pub fn check(&mut self) -> Option<DriftReport> {
        if self.window.len() < self.window_size
        {
            return None;
        }
        let mae: f64 =
            self.window.iter().map(|(p, a)| (p - a).abs()).sum::<f64>() / self.window.len() as f64;
        let relative_change = if self.baseline_error > f64::EPSILON
        {
            mae / self.baseline_error
        }
        else
        {
            1.0
        };
        let drifted = relative_change > self.threshold_factor;
        self.window.clear();
        let drift_type = if drifted
        {
            DriftType::ModelDrift
        }
        else
        {
            DriftType::NoDrift
        };
        Some(DriftReport {
            drift_type,
            data_drift_score: 0.0,
            model_drift_score: relative_change,
            threshold: self.threshold_factor,
            details: vec![
                ("current_mae".to_string(), mae),
                ("baseline_mae".to_string(), self.baseline_error),
                ("relative_change".to_string(), relative_change),
            ],
            sample_count: self.sample_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psi_no_drift() {
        let reference: Vec<f64> = (0..1000).map(|_| rand::random::<f64>() * 10.0).collect();
        let mut det = DataDriftDetector::from_reference(&reference, 10, 200, 0.25);
        // Feed similar data
        for _ in 0..200
        {
            det.add_sample(rand::random::<f64>() * 10.0);
        }
        let report = det.check().unwrap();
        assert!(
            report.data_drift_score < 0.25,
            "PSI too high: {}",
            report.data_drift_score
        );
        assert_eq!(report.drift_type, DriftType::NoDrift);
    }

    #[test]
    fn test_psi_drift_detected() {
        let reference: Vec<f64> = (0..1000).map(|_| rand::random::<f64>() * 10.0).collect();
        let mut det = DataDriftDetector::from_reference(&reference, 10, 200, 0.25);
        // Feed shifted data (mean shifted by +20)
        for _ in 0..200
        {
            det.add_sample(20.0 + rand::random::<f64>() * 10.0);
        }
        let report = det.check().unwrap();
        assert!(
            report.data_drift_score > 0.25,
            "PSI too low: {}",
            report.data_drift_score
        );
        assert_eq!(report.drift_type, DriftType::DataDrift);
    }

    #[test]
    fn test_model_drift_no_drift() {
        let mut det = ModelDriftDetector::new(50, 0.1, 2.0);
        for _ in 0..50
        {
            det.add_observation(10.0, 10.0 + (rand::random::<f64>() - 0.5) * 0.1);
        }
        let report = det.check().unwrap();
        assert_eq!(report.drift_type, DriftType::NoDrift);
    }

    #[test]
    fn test_model_drift_detected() {
        let mut det = ModelDriftDetector::new(50, 0.1, 2.0);
        // Predictions are way off (error ~5x baseline)
        for _ in 0..50
        {
            det.add_observation(10.0, 12.0);
        }
        let report = det.check().unwrap();
        assert_eq!(report.drift_type, DriftType::ModelDrift);
        assert!(report.model_drift_score > 2.0);
    }

    #[test]
    fn test_window_not_full_returns_none() {
        let reference: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut det = DataDriftDetector::from_reference(&reference, 10, 50, 0.25);
        det.add_sample(5.0);
        assert!(det.check().is_none());
    }

    #[test]
    fn test_sample_count() {
        let mut det = ModelDriftDetector::new(10, 0.1, 2.0);
        for _ in 0..5
        {
            det.add_observation(1.0, 2.0);
        }
        assert_eq!(det.sample_count, 5);
    }
}
