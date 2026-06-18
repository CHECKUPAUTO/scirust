use serde::{Deserialize, Serialize};

/// RUL (Remaining Useful Life) prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulPrediction {
    /// Estimated remaining useful life in hours
    pub remaining_hours: f64,
    /// Lower 95% confidence bound
    pub lower_bound_hours: f64,
    /// Upper 95% confidence bound
    pub upper_bound_hours: f64,
    /// Current health index (0..1)
    pub health_index: f64,
    /// Timestamp of prediction (hours since start)
    pub timestamp_hours: f64,
    /// Prediction method used
    pub method: String,
}

/// Trait for RUL estimation methods.
pub trait RulEstimator {
    /// Update the estimator with a new Health Index observation.
    ///
    /// `hi`: current health index (0..1)
    /// `timestamp_hours`: time since monitoring start
    fn update(&mut self, hi: f64, timestamp_hours: f64);

    /// Predict remaining useful life.
    fn predict(&self) -> RulPrediction;

    /// Reset the estimator state.
    fn reset(&mut self);
}

/// Linear RUL estimator.
///
/// Fits a linear regression `HI(t) = a*t + b` to recent observations.
/// RUL is the time until HI reaches 0, i.e. `RUL = -b / a - current_time`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRulEstimator {
    /// Window of recent (time, HI) observations
    window: Vec<(f64, f64)>,
    /// Maximum window size
    window_size: usize,
    /// Minimum observations before predicting
    min_observations: usize,
    /// Last fit parameters
    slope: f64,
    intercept: f64,
    last_time: f64,
    last_hi: f64,
    /// Confidence band width factor (standard deviations)
    confidence_factor: f64,
}

impl LinearRulEstimator {
    pub fn new(window_size: usize, min_observations: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            min_observations,
            slope: 0.0,
            intercept: 1.0,
            last_time: 0.0,
            last_hi: 1.0,
            confidence_factor: 1.96, // 95% CI
        }
    }

    fn fit_linear(&mut self) {
        let n = self.window.len();
        if n < self.min_observations
        {
            return;
        }
        let sum_t: f64 = self.window.iter().map(|p| p.0).sum();
        let sum_h: f64 = self.window.iter().map(|p| p.1).sum();
        let sum_tt: f64 = self.window.iter().map(|p| p.0 * p.0).sum();
        let sum_th: f64 = self.window.iter().map(|p| p.0 * p.1).sum();
        let n_f = n as f64;
        let denom = n_f * sum_tt - sum_t * sum_t;
        if denom.abs() < f64::EPSILON
        {
            return;
        }
        self.slope = (n_f * sum_th - sum_t * sum_h) / denom;
        self.intercept = (sum_h - self.slope * sum_t) / n_f;
    }

    fn residuals_std(&self) -> f64 {
        if self.window.len() < 2
        {
            return 100.0; // large default uncertainty
        }
        let residuals: Vec<f64> = self
            .window
            .iter()
            .map(|(t, h)| {
                let predicted = self.slope * t + self.intercept;
                h - predicted
            })
            .collect();
        let mean: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let variance: f64 =
            residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / residuals.len() as f64;
        f64::sqrt(variance)
    }
}

impl RulEstimator for LinearRulEstimator {
    fn update(&mut self, hi: f64, timestamp_hours: f64) {
        self.window.push((timestamp_hours, hi));
        if self.window.len() > self.window_size
        {
            self.window.remove(0);
        }
        self.last_time = timestamp_hours;
        self.last_hi = hi;
        self.fit_linear();
    }

    fn predict(&self) -> RulPrediction {
        let rul_mean = if self.slope.abs() < f64::EPSILON
        {
            // No degradation trend — very large RUL
            100_000.0
        }
        else if self.slope >= 0.0
        {
            // HI is increasing (getting healthier) — no failure predicted
            100_000.0
        }
        else
        {
            // Time until HI reaches 0: t = -intercept / slope
            let t_failure = -self.intercept / self.slope;
            (t_failure - self.last_time).max(0.0)
        };

        let r_sigma = self.residuals_std();
        // Convert HI residual to time uncertainty: dt = dHI / |slope|
        let time_uncertainty = if self.slope.abs() > f64::EPSILON
        {
            self.confidence_factor * r_sigma / self.slope.abs()
        }
        else
        {
            1000.0
        };

        RulPrediction {
            remaining_hours: rul_mean,
            lower_bound_hours: (rul_mean - time_uncertainty).max(0.0),
            upper_bound_hours: rul_mean + time_uncertainty,
            health_index: self.last_hi,
            timestamp_hours: self.last_time,
            method: "linear_regression".to_string(),
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.slope = 0.0;
        self.intercept = 1.0;
        self.last_time = 0.0;
        self.last_hi = 1.0;
    }
}

/// Exponential (Paris-law style) RUL estimator.
///
/// Fits `HI(t) = exp(-lambda * t)` and predicts time until HI reaches
/// a failure threshold (default 0.05).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialRulEstimator {
    window: Vec<(f64, f64)>,
    window_size: usize,
    min_observations: usize,
    lambda: f64,
    last_time: f64,
    last_hi: f64,
    failure_threshold: f64,
    confidence_factor: f64,
}

impl ExponentialRulEstimator {
    pub fn new(window_size: usize, min_observations: usize, failure_threshold: f64) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            min_observations,
            lambda: 0.0,
            last_time: 0.0,
            last_hi: 1.0,
            failure_threshold,
            confidence_factor: 1.96,
        }
    }

    fn fit_exponential(&mut self) {
        // Fit ln(HI) = -lambda * t via linear regression on ln(HI)
        let n = self.window.len();
        if n < self.min_observations
        {
            return;
        }
        let log_window: Vec<(f64, f64)> = self
            .window
            .iter()
            .filter(|(_, h)| *h > f64::EPSILON)
            .map(|(t, h)| (*t, h.ln()))
            .collect();
        let m = log_window.len();
        if m < 2
        {
            return;
        }
        let sum_t: f64 = log_window.iter().map(|p| p.0).sum();
        let sum_lh: f64 = log_window.iter().map(|p| p.1).sum();
        let sum_tt: f64 = log_window.iter().map(|p| p.0 * p.0).sum();
        let sum_t_lh: f64 = log_window.iter().map(|p| p.0 * p.1).sum();
        let m_f = m as f64;
        let denom = m_f * sum_tt - sum_t * sum_t;
        if denom.abs() < f64::EPSILON
        {
            return;
        }
        let slope = (m_f * sum_t_lh - sum_t * sum_lh) / denom;
        self.lambda = -slope; // lambda = -slope (since ln(HI) = -lambda*t)
    }
}

impl RulEstimator for ExponentialRulEstimator {
    fn update(&mut self, hi: f64, timestamp_hours: f64) {
        self.window.push((timestamp_hours, hi));
        if self.window.len() > self.window_size
        {
            self.window.remove(0);
        }
        self.last_time = timestamp_hours;
        self.last_hi = hi;
        self.fit_exponential();
    }

    fn predict(&self) -> RulPrediction {
        let rul_mean = if self.lambda <= f64::EPSILON
        {
            100_000.0
        }
        else
        {
            // HI(t) = exp(-lambda * t)
            // RUL = (ln(HI_current) - ln(threshold)) / lambda
            let ln_hi = if self.last_hi > f64::EPSILON
            {
                self.last_hi.ln()
            }
            else
            {
                f64::MIN.ln()
            };
            let ln_threshold = self.failure_threshold.ln();
            ((ln_hi - ln_threshold) / self.lambda).max(0.0)
        };

        RulPrediction {
            remaining_hours: rul_mean,
            lower_bound_hours: (rul_mean * 0.7).max(0.0),
            upper_bound_hours: rul_mean * 1.3,
            health_index: self.last_hi,
            timestamp_hours: self.last_time,
            method: "exponential".to_string(),
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.lambda = 0.0;
        self.last_time = 0.0;
        self.last_hi = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_rul_linear_degradation() {
        let mut est = LinearRulEstimator::new(50, 5);
        // HI decreases linearly: HI = 1.0 - 0.01*t (reaches 0 at t=100)
        for t in 0..20
        {
            est.update(1.0 - 0.01 * t as f64, t as f64);
        }
        let pred = est.predict();
        // At t=19, HI=0.81, RUL should be ~81 hours
        assert!(
            (pred.remaining_hours - 81.0).abs() < 5.0,
            "expected RUL ~81, got {}",
            pred.remaining_hours
        );
        assert!(pred.lower_bound_hours <= pred.remaining_hours);
        assert!(pred.upper_bound_hours >= pred.remaining_hours);
    }

    #[test]
    fn test_linear_rul_no_degradation() {
        let mut est = LinearRulEstimator::new(50, 3);
        for t in 0..10
        {
            est.update(0.95, t as f64); // constant HI
        }
        let pred = est.predict();
        // No degradation → very large RUL
        assert!(pred.remaining_hours > 10_000.0);
    }

    #[test]
    fn test_exponential_rul() {
        let mut est = ExponentialRulEstimator::new(50, 5, 0.05);
        // HI = exp(-0.01*t), threshold=0.05
        // At t=19, HI=exp(-0.19)=0.827
        // RUL = (ln(0.827) - ln(0.05)) / 0.01 ≈ 280.6
        for t in 0..20
        {
            est.update((-0.01 * t as f64).exp(), t as f64);
        }
        let pred = est.predict();
        assert!(pred.remaining_hours > 0.0, "RUL should be positive");
        // At t=19, HI=0.827, RUL = (ln(0.827/0.05))/0.01 ≈ 280.6
        assert!(
            (pred.remaining_hours - 280.0).abs() < 20.0,
            "expected RUL ~280, got {}",
            pred.remaining_hours
        );
    }

    #[test]
    fn test_rul_reset() {
        let mut est = LinearRulEstimator::new(50, 3);
        est.update(0.5, 100.0);
        est.reset();
        assert!(est.window.is_empty());
    }

    #[test]
    fn test_rul_min_observations() {
        let mut est = LinearRulEstimator::new(50, 10);
        for t in 0..5
        {
            est.update(1.0 - 0.1 * t as f64, t as f64);
        }
        // With only 5 obs (min=10), slope should still be 0 (not enough data)
        let pred = est.predict();
        // Should not crash; will return large RUL since slope=0
        assert!(pred.remaining_hours >= 0.0);
    }
}
