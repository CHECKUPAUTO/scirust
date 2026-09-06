//! High-value indicator gaps not covered by the core indicator catalogue.
//!
//! The functions here are deterministic forward reductions and keep the same
//! NaN-padding convention as `indicators.rs`. They focus on families that add
//! materially different information rather than aliases of existing formulas.

/// Kaufman's Efficiency Ratio over `period` bars.
pub fn kaufman_efficiency_ratio(values: &[f32], period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    for i in period..n {
        let change = (values[i] - values[i - period]).abs();
        let mut volatility = 0.0f32;
        for j in (i + 1 - period)..=i {
            volatility += (values[j] - values[j - 1]).abs();
        }
        out[i] = if volatility > 1e-12 {
            (change / volatility).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    out
}

/// Kaufman's Adaptive Moving Average (KAMA).
///
/// The smoothing constant interpolates between the caller-declared fast and
/// slow EMA constants using the squared Efficiency Ratio convention.
pub fn kama(values: &[f32], er_period: usize, fast_period: usize, slow_period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if er_period == 0
        || fast_period == 0
        || slow_period == 0
        || fast_period >= slow_period
        || n <= er_period
    {
        return out;
    }
    let er = kaufman_efficiency_ratio(values, er_period);
    let fast = 2.0 / (fast_period as f32 + 1.0);
    let slow = 2.0 / (slow_period as f32 + 1.0);
    let mut prev = values[er_period];
    out[er_period] = prev;
    for i in (er_period + 1)..n {
        let efficiency = er[i];
        let sc = (efficiency * (fast - slow) + slow).powi(2);
        prev += sc * (values[i] - prev);
        out[i] = prev;
    }
    out
}

/// Rolling annualized Parkinson volatility from high/low ranges.
///
/// `periods_per_year` expresses the sampling frequency explicitly (for example,
/// 365 for daily crypto bars). The function returns decimal volatility, not %.
pub fn parkinson_volatility(
    highs: &[f32],
    lows: &[f32],
    period: usize,
    periods_per_year: f32,
) -> Vec<f32> {
    let n = highs.len();
    let mut out = vec![f32::NAN; n];
    if lows.len() != n || period == 0 || n < period || !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return out;
    }
    let denom = 4.0 * std::f32::consts::LN_2;
    for i in (period - 1)..n {
        let mut sum = 0.0f32;
        let mut valid = true;
        for j in (i + 1 - period)..=i {
            if !highs[j].is_finite() || !lows[j].is_finite() || highs[j] <= 0.0 || lows[j] <= 0.0 || highs[j] < lows[j] {
                valid = false;
                break;
            }
            let r = (highs[j] / lows[j]).ln();
            sum += r * r;
        }
        if valid {
            let variance = sum / (period as f32 * denom);
            out[i] = (variance.max(0.0) * periods_per_year).sqrt();
        }
    }
    out
}

/// Rolling annualized Garman–Klass volatility from OHLC bars.
///
/// Returns decimal volatility. Invalid/non-positive OHLC bars leave the output
/// at NaN for the affected rolling window rather than silently repairing data.
pub fn garman_klass_volatility(
    opens: &[f32],
    highs: &[f32],
    lows: &[f32],
    closes: &[f32],
    period: usize,
    periods_per_year: f32,
) -> Vec<f32> {
    let n = opens.len();
    let mut out = vec![f32::NAN; n];
    if highs.len() != n
        || lows.len() != n
        || closes.len() != n
        || period == 0
        || n < period
        || !periods_per_year.is_finite()
        || periods_per_year <= 0.0
    {
        return out;
    }
    let close_coeff = 2.0 * std::f32::consts::LN_2 - 1.0;
    for i in (period - 1)..n {
        let mut sum = 0.0f32;
        let mut valid = true;
        for j in (i + 1 - period)..=i {
            let (o, h, l, c) = (opens[j], highs[j], lows[j], closes[j]);
            if !o.is_finite()
                || !h.is_finite()
                || !l.is_finite()
                || !c.is_finite()
                || o <= 0.0
                || h <= 0.0
                || l <= 0.0
                || c <= 0.0
                || h < l
            {
                valid = false;
                break;
            }
            let hl = (h / l).ln();
            let co = (c / o).ln();
            sum += 0.5 * hl * hl - close_coeff * co * co;
        }
        if valid {
            let variance = (sum / period as f32).max(0.0);
            out[i] = (variance * periods_per_year).sqrt();
        }
    }
    out
}

/// Chaikin Money Flow over a trailing `period`.
pub fn chaikin_money_flow(
    highs: &[f32],
    lows: &[f32],
    closes: &[f32],
    volumes: &[f32],
    period: usize,
) -> Vec<f32> {
    let n = highs.len();
    let mut out = vec![f32::NAN; n];
    if lows.len() != n || closes.len() != n || volumes.len() != n || period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let mut money_flow_volume = 0.0f32;
        let mut volume_sum = 0.0f32;
        let mut valid = true;
        for j in (i + 1 - period)..=i {
            let range = highs[j] - lows[j];
            if !highs[j].is_finite()
                || !lows[j].is_finite()
                || !closes[j].is_finite()
                || !volumes[j].is_finite()
                || volumes[j] < 0.0
                || range < 0.0
            {
                valid = false;
                break;
            }
            let multiplier = if range > 1e-12 {
                ((closes[j] - lows[j]) - (highs[j] - closes[j])) / range
            } else {
                0.0
            };
            money_flow_volume += multiplier * volumes[j];
            volume_sum += volumes[j];
        }
        if valid {
            out[i] = if volume_sum > 1e-12 {
                money_flow_volume / volume_sum
            } else {
                0.0
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn efficiency_ratio_distinguishes_trend_from_chop() {
        let trend: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let er = kaufman_efficiency_ratio(&trend, 10);
        assert!((er[19] - 1.0).abs() < 1e-6);

        let chop = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        let er2 = kaufman_efficiency_ratio(&chop, 10);
        assert!(er2[10] < 0.2);
    }

    #[test]
    fn kama_is_deterministic_and_tracks_monotonic_input() {
        let values: Vec<f32> = (1..=40).map(|i| i as f32).collect();
        let a = kama(&values, 10, 2, 30);
        let b = kama(&values, 10, 2, 30);
        assert_eq!(a[39].to_bits(), b[39].to_bits());
        assert!(a[39] > a[20]);
        assert!(a[39] < values[39]);
    }

    #[test]
    fn range_estimators_are_zero_for_constant_bars() {
        let x = vec![100.0; 20];
        let p = parkinson_volatility(&x, &x, 10, 365.0);
        let g = garman_klass_volatility(&x, &x, &x, &x, 10, 365.0);
        assert_eq!(p[19], 0.0);
        assert_eq!(g[19], 0.0);
    }

    #[test]
    fn range_estimators_are_positive_for_variable_bars() {
        let o = vec![100.0; 20];
        let h = vec![105.0; 20];
        let l = vec![95.0; 20];
        let c = vec![101.0; 20];
        assert!(parkinson_volatility(&h, &l, 10, 365.0)[19] > 0.0);
        assert!(garman_klass_volatility(&o, &h, &l, &c, 10, 365.0)[19] > 0.0);
    }

    #[test]
    fn chaikin_flow_sign_reflects_close_location() {
        let highs = vec![10.0; 10];
        let lows = vec![0.0; 10];
        let volumes = vec![100.0; 10];
        let near_high = vec![9.0; 10];
        let near_low = vec![1.0; 10];
        assert!(chaikin_money_flow(&highs, &lows, &near_high, &volumes, 5)[9] > 0.0);
        assert!(chaikin_money_flow(&highs, &lows, &near_low, &volumes, 5)[9] < 0.0);
    }
}
