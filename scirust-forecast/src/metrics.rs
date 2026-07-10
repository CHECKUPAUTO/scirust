//! Forecast-accuracy metrics: MAE, RMSE and MAPE.

use crate::error::ForecastError;

/// Validate that two slices are non-empty and equal in length.
fn check_pair(actual: &[f64], pred: &[f64]) -> Result<(), ForecastError> {
    if actual.is_empty() || pred.is_empty()
    {
        return Err(ForecastError::EmptySeries);
    }
    if actual.len() != pred.len()
    {
        return Err(ForecastError::LengthMismatch {
            left: actual.len(),
            right: pred.len(),
        });
    }
    Ok(())
}

/// Mean absolute error between `actual` and `pred`.
///
/// Returns [`ForecastError::EmptySeries`] if either slice is empty and
/// [`ForecastError::LengthMismatch`] if the lengths differ.
pub fn mae(actual: &[f64], pred: &[f64]) -> Result<f64, ForecastError> {
    check_pair(actual, pred)?;
    let sum: f64 = actual.iter().zip(pred).map(|(a, p)| (a - p).abs()).sum();
    Ok(sum / actual.len() as f64)
}

/// Root mean squared error between `actual` and `pred`.
///
/// Returns [`ForecastError::EmptySeries`] if either slice is empty and
/// [`ForecastError::LengthMismatch`] if the lengths differ.
pub fn rmse(actual: &[f64], pred: &[f64]) -> Result<f64, ForecastError> {
    check_pair(actual, pred)?;
    let sum: f64 = actual
        .iter()
        .zip(pred)
        .map(|(a, p)| {
            let d = a - p;
            d * d
        })
        .sum();
    Ok((sum / actual.len() as f64).sqrt())
}

/// Mean absolute percentage error, expressed on a 0-100 percentage scale.
///
/// Returns [`ForecastError::EmptySeries`] if either slice is empty,
/// [`ForecastError::LengthMismatch`] if the lengths differ and
/// [`ForecastError::ZeroActual`] if any actual value is zero (which would make
/// the percentage undefined).
pub fn mape(actual: &[f64], pred: &[f64]) -> Result<f64, ForecastError> {
    check_pair(actual, pred)?;
    let mut sum = 0.0;
    for (index, (&a, &p)) in actual.iter().zip(pred).enumerate()
    {
        if a == 0.0
        {
            return Err(ForecastError::ZeroActual { index });
        }
        sum += ((a - p) / a).abs();
    }
    Ok(100.0 * sum / actual.len() as f64)
}
