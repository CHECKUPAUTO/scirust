//! Cryptocurrency trading indicators.

/// Exponential Moving Average (EMA)
/// Formula: EMA_t = Price_t * (2/(N+1)) + EMA_{t-1} * (1 - 2/(N+1))
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0
    {
        return vec![];
    }
    let mut ema_values = Vec::with_capacity(data.len());
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut current_ema = data[0];
    ema_values.push(current_ema);

    for &price in data.iter().skip(1)
    {
        current_ema = price * alpha + current_ema * (1.0 - alpha);
        ema_values.push(current_ema);
    }
    ema_values
}

/// Relative Strength Index (RSI)
/// period: typically 14
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() <= period
    {
        return vec![];
    }
    let mut rsi_values = vec![0.0; data.len()];
    let mut gains = 0.0;
    let mut losses = 0.0;

    for i in 1..=period
    {
        let diff = data[i] - data[i - 1];
        if diff >= 0.0
        {
            gains += diff;
        }
        else
        {
            losses -= diff;
        }
    }

    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;

    if avg_loss == 0.0
    {
        rsi_values[period] = 100.0;
    }
    else
    {
        let rs = avg_gain / avg_loss;
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    for i in (period + 1)..data.len()
    {
        let diff = data[i] - data[i - 1];
        let (gain, loss) = if diff >= 0.0
        {
            (diff, 0.0)
        }
        else
        {
            (0.0, -diff)
        };

        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        if avg_loss == 0.0
        {
            rsi_values[i] = 100.0;
        }
        else
        {
            let rs = avg_gain / avg_loss;
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    rsi_values[period..].to_vec()
}

/// Bollinger Bands
/// Returns (Upper Band, Middle Band, Lower Band)
pub fn bollinger_bands(data: &[f64], period: usize, k: f64) -> Vec<(f64, f64, f64)> {
    if data.len() < period
    {
        return vec![];
    }
    let mut results = Vec::with_capacity(data.len() - period + 1);

    for i in 0..=(data.len() - period)
    {
        let window = &data[i..(i + period)];
        let middle_band: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window
            .iter()
            .map(|&x| (x - middle_band).powi(2))
            .sum::<f64>()
            / period as f64;
        let std_dev = variance.sqrt();
        results.push((
            middle_band + k * std_dev,
            middle_band,
            middle_band - k * std_dev,
        ));
    }
    results
}

/// Moving Average Convergence Divergence (MACD)
/// Returns (MACD Line, Signal Line, Histogram)
pub fn macd(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Vec<(f64, f64, f64)> {
    let ema_fast = ema(data, fast_period);
    let ema_slow = ema(data, slow_period);

    let mut macd_line = Vec::with_capacity(data.len());
    for (f, s) in ema_fast.iter().zip(ema_slow.iter())
    {
        macd_line.push(f - s);
    }

    let signal_line = ema(&macd_line, signal_period);
    let mut results = Vec::with_capacity(data.len());

    for (m, s) in macd_line.iter().zip(signal_line.iter())
    {
        results.push((*m, *s, *m - *s));
    }
    results
}

/// Kelly Criterion for position sizing
/// win_prob: probability of winning (0.0 to 1.0)
/// win_loss_ratio: ratio of average win to average loss (b in formula)
/// Returns the fraction of the capital to bet.
pub fn kelly_criterion(win_prob: f64, win_loss_ratio: f64) -> f64 {
    if win_loss_ratio <= 0.0
    {
        return 0.0;
    }
    let f = (win_prob * (win_loss_ratio + 1.0) - 1.0) / win_loss_ratio;
    f.max(0.0)
}

/// Value at Risk (VaR) using the historical simulation method
/// data: historical returns
/// confidence_level: e.g., 0.95 for 95% confidence
pub fn value_at_risk(returns: &[f64], confidence_level: f64) -> f64 {
    if returns.is_empty()
    {
        return 0.0;
    }
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let index = ((alpha * sorted_returns.len() as f64) + 1e-10).floor() as usize;

    if index >= sorted_returns.len()
    {
        return 0.0;
    }
    -sorted_returns[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let data = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let period = 3;
        let result = ema(&data, period);
        assert_eq!(result.len(), 5);
        // Alpha = 2 / (3 + 1) = 0.5
        // EMA0 = 10
        // EMA1 = 12 * 0.5 + 10 * 0.5 = 11
        // EMA2 = 14 * 0.5 + 11 * 0.5 = 12.5
        assert!((result[1] - 11.0).abs() < 1e-10);
        assert!((result[2] - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let data = vec![
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
            45.61, 46.28, 46.28, 46.00,
        ];
        let period = 14;
        let result = rsi(&data, period);
        assert_eq!(result.len(), 2);
        assert!(result[0] > 0.0 && result[0] < 100.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let data = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let period = 3;
        let k = 2.0;
        let result = bollinger_bands(&data, period, k);
        assert_eq!(result.len(), 3);
        // Window [10, 12, 14]: Mean = 12, StdDev = sqrt((4+0+4)/3) = sqrt(2.666...) = 1.633
        // Upper = 12 + 2*1.633 = 15.266
        // Lower = 12 - 2*1.633 = 8.734
        assert!((result[0].1 - 12.0).abs() < 1e-10);
        assert!(result[0].0 > 12.0);
        assert!(result[0].2 < 12.0);
    }

    #[test]
    fn test_macd() {
        let data = vec![10.0; 30];
        let result = macd(&data, 12, 26, 9);
        assert_eq!(result.len(), 30);
        // Constant data should result in 0 MACD
        for (m, s, h) in result
        {
            assert!(m.abs() < 1e-10);
            assert!(s.abs() < 1e-10);
            assert!(h.abs() < 1e-10);
        }
    }

    #[test]
    fn test_kelly_criterion() {
        // Example: win_prob = 0.6, win_loss_ratio = 1.0 (even money)
        // f = (0.6 * (1 + 1) - 1) / 1 = (1.2 - 1) / 1 = 0.2
        let f = kelly_criterion(0.6, 1.0);
        assert!((f - 0.2).abs() < 1e-10);

        // Negative expectation should return 0
        let f_neg = kelly_criterion(0.4, 1.0);
        assert_eq!(f_neg, 0.0);
    }

    #[test]
    fn test_value_at_risk() {
        let returns = vec![
            -0.05, -0.02, 0.01, 0.03, 0.05, -0.10, 0.02, 0.04, -0.01, 0.01,
        ];
        // Sorted: [-0.10, -0.05, -0.02, -0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05]
        // Len = 10. confidence = 0.90 => alpha = 0.10. Index = floor(0.1 * 10) = 1
        // returns[1] = -0.05. VaR = 0.05
        let var = value_at_risk(&returns, 0.90);
        assert!((var - 0.05).abs() < 1e-10);
    }
}
