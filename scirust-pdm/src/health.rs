use serde::{Deserialize, Serialize};

/// Health state classification per ISO 13374.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthState {
    /// Normal operation
    Good,
    /// Slight degradation detected — monitor
    Degraded,
    /// Significant degradation — plan maintenance
    Warning,
    /// Critical degradation — immediate action
    Critical,
    /// Component failed
    Failed,
}

impl HealthState {
    pub fn from_index(hi: f64) -> Self {
        if hi >= 0.9
        {
            HealthState::Good
        }
        else if hi >= 0.7
        {
            HealthState::Degraded
        }
        else if hi >= 0.5
        {
            HealthState::Warning
        }
        else if hi >= 0.2
        {
            HealthState::Critical
        }
        else
        {
            HealthState::Failed
        }
    }

    pub fn label(&self) -> &'static str {
        match self
        {
            HealthState::Good => "Good",
            HealthState::Degraded => "Degraded",
            HealthState::Warning => "Warning",
            HealthState::Critical => "Critical",
            HealthState::Failed => "Failed",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            HealthState::Good => "Bon",
            HealthState::Degraded => "Dégradé",
            HealthState::Warning => "Alerte",
            HealthState::Critical => "Critique",
            HealthState::Failed => "Défaillant",
        }
    }
}

/// Health Index estimator.
///
/// Combines multiple feature indicators into a single 0..1 health score,
/// where 1.0 = perfect, 0.0 = failed.
///
/// Uses a weighted distance from a baseline (healthy) reference, normalized
/// by the distance to a failure threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndex {
    /// Baseline (healthy) feature values
    pub baseline: Vec<f64>,
    /// Failure threshold feature values
    pub failure_threshold: Vec<f64>,
    /// Weight per feature (must sum to ~1.0)
    pub weights: Vec<f64>,
    /// Smoothing EMA alpha (0..1)
    pub ema_alpha: f64,
    /// Current smoothed HI value
    current: f64,
    /// Number of updates so far
    update_count: u64,
}

impl HealthIndex {
    /// Create a new Health Index estimator.
    ///
    /// For each feature i, the contribution is:
    ///   w_i * clamp((threshold_i - value_i) / (threshold_i - baseline_i), 0, 1)
    pub fn new(
        baseline: Vec<f64>,
        failure_threshold: Vec<f64>,
        weights: Vec<f64>,
        ema_alpha: f64,
    ) -> Self {
        assert_eq!(
            baseline.len(),
            failure_threshold.len(),
            "baseline and threshold must match length"
        );
        assert_eq!(
            baseline.len(),
            weights.len(),
            "weights must match features length"
        );
        Self {
            baseline,
            failure_threshold,
            weights,
            ema_alpha: ema_alpha.clamp(0.0, 1.0),
            current: 1.0,
            update_count: 0,
        }
    }

    /// Update the Health Index with a new feature vector.
    ///
    /// Returns the new HI value (0.0 = failed, 1.0 = healthy).
    pub fn update(&mut self, features: &[f64]) -> f64 {
        assert_eq!(
            features.len(),
            self.baseline.len(),
            "feature vector length mismatch"
        );
        let mut hi_raw = 0.0;
        for (i, (&b, (&t, &v))) in self
            .baseline
            .iter()
            .zip(self.failure_threshold.iter().zip(features.iter()))
            .enumerate()
        {
            let range = t - b;
            if range.abs() < f64::EPSILON
            {
                hi_raw += self.weights[i]; // no degradation possible
                continue;
            }
            // Normalize: if v == b → 1.0 (healthy), if v == t → 0.0 (failed)
            let normalized = (t - v) / range;
            let clamped = normalized.clamp(0.0, 1.0);
            hi_raw += self.weights[i] * clamped;
        }

        // EMA smoothing
        if self.update_count == 0
        {
            self.current = hi_raw;
        }
        else
        {
            self.current = self.ema_alpha * hi_raw + (1.0 - self.ema_alpha) * self.current;
        }
        self.update_count += 1;
        self.current
    }

    /// Current smoothed HI.
    pub fn value(&self) -> f64 {
        self.current
    }

    /// Current health state.
    pub fn state(&self) -> HealthState {
        HealthState::from_index(self.current)
    }

    /// Number of updates received.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_state_mapping() {
        assert_eq!(HealthState::from_index(0.95), HealthState::Good);
        assert_eq!(HealthState::from_index(0.80), HealthState::Degraded);
        assert_eq!(HealthState::from_index(0.60), HealthState::Warning);
        assert_eq!(HealthState::from_index(0.35), HealthState::Critical);
        assert_eq!(HealthState::from_index(0.10), HealthState::Failed);
    }

    #[test]
    fn test_hi_healthy() {
        let mut hi = HealthIndex::new(
            vec![0.5, 1.0],  // baseline (healthy)
            vec![5.0, 10.0], // failure threshold
            vec![0.5, 0.5],  // weights
            1.0,             // no smoothing
        );
        // Feed baseline → should return 1.0
        let h = hi.update(&[0.5, 1.0]);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hi_failed() {
        let mut hi = HealthIndex::new(vec![0.5, 1.0], vec![5.0, 10.0], vec![0.5, 0.5], 1.0);
        let h = hi.update(&[5.0, 10.0]);
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hi_midpoint() {
        let mut hi = HealthIndex::new(vec![0.0], vec![10.0], vec![1.0], 1.0);
        // At midpoint → HI should be 0.5
        let h = hi.update(&[5.0]);
        assert!((h - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hi_ema_smoothing() {
        let mut hi = HealthIndex::new(
            vec![0.0],
            vec![10.0],
            vec![1.0],
            0.5, // alpha=0.5
        );
        let h1 = hi.update(&[0.0]); // raw=1.0, current=1.0 (first update)
        assert!((h1 - 1.0).abs() < 1e-10);
        let h2 = hi.update(&[10.0]); // raw=0.0, current=0.5*0 + 0.5*1.0 = 0.5
        assert!((h2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hi_clamping() {
        let mut hi = HealthIndex::new(vec![0.0], vec![10.0], vec![1.0], 1.0);
        // Beyond baseline → should clamp to 1.0
        let h = hi.update(&[-5.0]);
        assert!((h - 1.0).abs() < 1e-10);
        // Beyond threshold → should clamp to 0.0
        let h2 = hi.update(&[15.0]);
        assert!((h2 - 0.0).abs() < 1e-10);
    }
}
