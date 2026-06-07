//! Time series forecasting and event prediction.

pub mod nbeats;

/// Simple Exponential Smoothing (SES)
/// Formula: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
pub fn simple_exponential_smoothing(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let mut smoothed = Vec::with_capacity(data.len());
    let mut s = data[0];
    smoothed.push(s);

    for &x in data.iter().skip(1) {
        s = alpha * x + (1.0 - alpha) * s;
        smoothed.push(s);
    }
    smoothed
}

/// Holt's Linear Trend method for forecasting
/// Returns (level, trend)
pub fn holt_linear_trend(data: &[f64], alpha: f64, beta: f64, horizon: usize) -> Vec<f64> {
    if data.len() < 2 {
        return vec![];
    }
    let mut l = data[0];
    let mut b = data[1] - data[0];

    for &x in data.iter().skip(1) {
        let prev_l = l;
        l = alpha * x + (1.0 - alpha) * (l + b);
        b = beta * (l - prev_l) + (1.0 - beta) * b;
    }

    let mut forecasts = Vec::with_capacity(horizon);
    for h in 1..=horizon {
        forecasts.push(l + (h as f64) * b);
    }
    forecasts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ses() {
        let data = vec![10.0, 12.0, 14.0];
        let alpha = 0.5;
        let result = simple_exponential_smoothing(&data, alpha);
        assert_eq!(result.len(), 3);
        // s0 = 10
        // s1 = 0.5 * 12 + 0.5 * 10 = 11
        // s2 = 0.5 * 14 + 0.5 * 11 = 12.5
        assert_eq!(result[0], 10.0);
        assert_eq!(result[1], 11.0);
        assert_eq!(result[2], 12.5);
    }

    #[test]
    fn test_holt_linear() {
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let alpha = 0.5;
        let beta = 0.5;
        let forecasts = holt_linear_trend(&data, alpha, beta, 3);
        assert_eq!(forecasts.len(), 3);
        // Data is perfectly linear with trend 1.0.
        // Forecasts should continue the trend.
        assert!(forecasts[0] > 14.0);
        assert!(forecasts[1] > forecasts[0]);
        assert!(forecasts[2] > forecasts[1]);
    }
}
