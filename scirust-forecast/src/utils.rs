//! Series-transformation utilities: differencing and moving averages.

use crate::error::ForecastError;

/// Lag-`lag` differencing: `out[i] = series[i + lag] - series[i]`.
///
/// The result has length `series.len().saturating_sub(lag)`; if `lag` is at
/// least the series length the result is empty. A `lag` of zero yields a run of
/// zeros the same length as the input.
pub fn difference(series: &[f64], lag: usize) -> Vec<f64> {
    (lag..series.len())
        .map(|i| series[i] - series[i - lag])
        .collect()
}

/// Trailing simple moving average with the given `window`.
///
/// The result has length `series.len() - window + 1`; entry `i` is the mean of
/// `series[i..i + window]`.
///
/// Returns [`ForecastError::EmptySeries`] on an empty series and
/// [`ForecastError::InvalidWindow`] when `window` is zero or larger than the
/// series.
pub fn moving_average(series: &[f64], window: usize) -> Result<Vec<f64>, ForecastError> {
    if series.is_empty() {
        return Err(ForecastError::EmptySeries);
    }
    if window == 0 || window > series.len() {
        return Err(ForecastError::InvalidWindow { window });
    }
    let inv = 1.0 / window as f64;
    let out = (0..=series.len() - window)
        .map(|i| series[i..i + window].iter().sum::<f64>() * inv)
        .collect();
    Ok(out)
}
