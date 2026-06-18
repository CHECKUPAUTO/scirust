use serde::{Deserialize, Serialize};

/// Comparison metric for shadow deployment evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ComparisonMetric {
    MeanAbsoluteError,
    MeanSquaredError,
    Accuracy,
    F1Score,
}

/// Result of a shadow deployment comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowResult {
    pub production_metric: f64,
    pub shadow_metric: f64,
    pub delta: f64,
    pub improvement: bool,
    pub sample_count: u64,
    pub metric_type: ComparisonMetric,
    pub recommendation: DeploymentRecommendation,
}

/// Deployment recommendation based on shadow comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentRecommendation {
    /// Shadow model is better → promote to production
    Promote,
    /// Production model is better → keep current
    Keep,
    /// Inconclusive → collect more data
    Inconclusive,
}

/// Shadow deployment runner.
///
/// Runs a new "shadow" model alongside the production model,
/// collecting predictions from both. After sufficient samples,
/// compares metrics and recommends whether to promote the shadow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowDeployment {
    pub window_size: usize,
    pub metric: ComparisonMetric,
    /// Minimum relative improvement to recommend promotion (e.g. 0.05 = 5%)
    pub min_improvement: f64,
    /// Production predictions: (prediction, actual)
    production_obs: Vec<(f64, f64)>,
    /// Shadow predictions: (prediction, actual)
    shadow_obs: Vec<(f64, f64)>,
    sample_count: u64,
}

impl ShadowDeployment {
    pub fn new(window_size: usize, metric: ComparisonMetric, min_improvement: f64) -> Self {
        Self {
            window_size,
            metric,
            min_improvement,
            production_obs: Vec::with_capacity(window_size),
            shadow_obs: Vec::with_capacity(window_size),
            sample_count: 0,
        }
    }

    /// Record predictions from both models for the same input.
    pub fn add_observation(&mut self, production_pred: f64, shadow_pred: f64, actual: f64) {
        self.production_obs.push((production_pred, actual));
        self.shadow_obs.push((shadow_pred, actual));
        self.sample_count += 1;
    }

    /// Evaluate and return comparison result. Returns None if insufficient data.
    pub fn evaluate(&mut self) -> Option<ShadowResult> {
        if self.production_obs.len() < self.window_size
        {
            return None;
        }
        let prod_metric = self.compute_metric(&self.production_obs);
        let shadow_metric = self.compute_metric(&self.shadow_obs);

        let delta = match self.metric
        {
            ComparisonMetric::MeanAbsoluteError | ComparisonMetric::MeanSquaredError =>
            {
                // Lower is better
                prod_metric - shadow_metric
            },
            ComparisonMetric::Accuracy | ComparisonMetric::F1Score =>
            {
                // Higher is better
                shadow_metric - prod_metric
            },
        };

        let improvement = delta > 0.0;
        let relative_improvement = if prod_metric.abs() > f64::EPSILON
        {
            delta / prod_metric.abs()
        }
        else
        {
            0.0
        };

        let recommendation = if relative_improvement > self.min_improvement
        {
            DeploymentRecommendation::Promote
        }
        else if relative_improvement < -self.min_improvement
        {
            DeploymentRecommendation::Keep
        }
        else
        {
            DeploymentRecommendation::Inconclusive
        };

        self.production_obs.clear();
        self.shadow_obs.clear();

        Some(ShadowResult {
            production_metric: prod_metric,
            shadow_metric,
            delta,
            improvement,
            sample_count: self.sample_count,
            metric_type: self.metric,
            recommendation,
        })
    }

    fn compute_metric(&self, obs: &[(f64, f64)]) -> f64 {
        if obs.is_empty()
        {
            return 0.0;
        }
        match self.metric
        {
            ComparisonMetric::MeanAbsoluteError =>
            {
                obs.iter().map(|(p, a)| (p - a).abs()).sum::<f64>() / obs.len() as f64
            },
            ComparisonMetric::MeanSquaredError =>
            {
                obs.iter().map(|(p, a)| (p - a).powi(2)).sum::<f64>() / obs.len() as f64
            },
            ComparisonMetric::Accuracy =>
            {
                let correct = obs.iter().filter(|(p, a)| (p - a).abs() < 0.5).count();
                correct as f64 / obs.len() as f64
            },
            ComparisonMetric::F1Score =>
            {
                // Binary F1: positive if pred > 0.5
                let tp = obs.iter().filter(|(p, a)| *p > 0.5 && *a > 0.5).count();
                let fp = obs.iter().filter(|(p, a)| *p > 0.5 && *a <= 0.5).count();
                let fn_ = obs.iter().filter(|(p, a)| *p <= 0.5 && *a > 0.5).count();
                let precision = if tp + fp > 0
                {
                    tp as f64 / (tp + fp) as f64
                }
                else
                {
                    0.0
                };
                let recall = if tp + fn_ > 0
                {
                    tp as f64 / (tp + fn_) as f64
                }
                else
                {
                    0.0
                };
                if precision + recall > 0.0
                {
                    2.0 * precision * recall / (precision + recall)
                }
                else
                {
                    0.0
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_promote_better_model() {
        let mut shadow = ShadowDeployment::new(50, ComparisonMetric::MeanAbsoluteError, 0.05);
        // Production: error ~1.0, Shadow: error ~0.1
        for _ in 0..50
        {
            shadow.add_observation(11.0, 10.1, 10.0); // prod err=1, shadow err=0.1
        }
        let result = shadow.evaluate().unwrap();
        assert!(result.improvement);
        assert_eq!(result.recommendation, DeploymentRecommendation::Promote);
    }

    #[test]
    fn test_shadow_keep_better_model() {
        let mut shadow = ShadowDeployment::new(50, ComparisonMetric::MeanAbsoluteError, 0.05);
        for _ in 0..50
        {
            shadow.add_observation(10.1, 12.0, 10.0); // prod err=0.1, shadow err=2
        }
        let result = shadow.evaluate().unwrap();
        assert!(!result.improvement);
        assert_eq!(result.recommendation, DeploymentRecommendation::Keep);
    }

    #[test]
    fn test_shadow_inconclusive() {
        let mut shadow = ShadowDeployment::new(50, ComparisonMetric::MeanAbsoluteError, 0.20);
        for _ in 0..50
        {
            shadow.add_observation(10.5, 10.4, 10.0); // nearly identical
        }
        let result = shadow.evaluate().unwrap();
        assert_eq!(
            result.recommendation,
            DeploymentRecommendation::Inconclusive
        );
    }

    #[test]
    fn test_shadow_insufficient_data() {
        let mut shadow = ShadowDeployment::new(50, ComparisonMetric::MeanAbsoluteError, 0.05);
        for _ in 0..10
        {
            shadow.add_observation(1.0, 2.0, 1.5);
        }
        assert!(shadow.evaluate().is_none());
    }

    #[test]
    fn test_accuracy_metric() {
        let mut shadow = ShadowDeployment::new(10, ComparisonMetric::Accuracy, 0.05);
        for _ in 0..10
        {
            shadow.add_observation(1.0, 1.0, 1.0); // both correct
        }
        let result = shadow.evaluate().unwrap();
        assert!((result.production_metric - 1.0).abs() < 1e-10);
        assert_eq!(
            result.recommendation,
            DeploymentRecommendation::Inconclusive
        );
    }
}
