//! Exponential-smoothing forecasters: simple (SES), double (Holt) and triple
//! (Holt-Winters, additive & multiplicative).

use crate::error::ForecastError;

/// Validate that a smoothing parameter lies in the closed interval `[0, 1]`.
fn check_smoothing(name: &'static str, value: f64) -> Result<(), ForecastError> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ForecastError::InvalidSmoothing { name, value })
    }
}

/// A fitted simple exponential smoothing (SES) model.
///
/// SES tracks a single smoothed *level* `l_t = alpha * y_t + (1 - alpha) *
/// l_{t-1}`. Because there is no trend or seasonal term, every future
/// observation is forecast with the same (flat) final level.
#[derive(Debug, Clone, PartialEq)]
pub struct Ses {
    level: f64,
    fitted: Vec<f64>,
}

impl Ses {
    /// The final smoothed level, i.e. the value every horizon is forecast at.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// The in-sample one-step-ahead fitted values (same length as the series).
    pub fn fitted(&self) -> Vec<f64> {
        self.fitted.clone()
    }

    /// Flat forecast for the next `h` steps (all equal to the final level).
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        vec![self.level; h]
    }
}

/// Fit a simple exponential smoothing model to `series` with parameter `alpha`.
///
/// Returns [`ForecastError::EmptySeries`] on an empty series and
/// [`ForecastError::InvalidSmoothing`] when `alpha` is outside `[0, 1]`.
pub fn simple_exp_smoothing(series: &[f64], alpha: f64) -> Result<Ses, ForecastError> {
    if series.is_empty() {
        return Err(ForecastError::EmptySeries);
    }
    check_smoothing("alpha", alpha)?;

    let mut level = series[0];
    let mut fitted = Vec::with_capacity(series.len());
    fitted.push(series[0]);
    for &y in &series[1..] {
        // The forecast of `y` made one step earlier is the current level.
        fitted.push(level);
        level = alpha * y + (1.0 - alpha) * level;
    }

    Ok(Ses { level, fitted })
}

/// A fitted Holt (double exponential smoothing) model.
///
/// Holt smoothing tracks a *level* and a *trend*; the `h`-step forecast is the
/// linear extrapolation `level + h * trend`.
#[derive(Debug, Clone, PartialEq)]
pub struct Holt {
    level: f64,
    trend: f64,
    fitted: Vec<f64>,
}

impl Holt {
    /// The final smoothed level.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// The final smoothed trend (per-step slope).
    pub fn trend(&self) -> f64 {
        self.trend
    }

    /// The in-sample one-step-ahead fitted values (same length as the series).
    pub fn fitted(&self) -> Vec<f64> {
        self.fitted.clone()
    }

    /// Forecast the next `h` steps as `level + i * trend` for `i = 1..=h`.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        (1..=h).map(|i| self.level + i as f64 * self.trend).collect()
    }
}

/// Fit a Holt double-exponential-smoothing model to `series`.
///
/// `alpha` smooths the level and `beta` smooths the trend; both must lie in
/// `[0, 1]`. The series needs at least two observations so the initial trend
/// `series[1] - series[0]` is defined.
///
/// Returns [`ForecastError::EmptySeries`],
/// [`ForecastError::SeriesTooShort`] (fewer than two points) or
/// [`ForecastError::InvalidSmoothing`] on bad input.
pub fn holt(series: &[f64], alpha: f64, beta: f64) -> Result<Holt, ForecastError> {
    if series.is_empty() {
        return Err(ForecastError::EmptySeries);
    }
    if series.len() < 2 {
        return Err(ForecastError::SeriesTooShort {
            got: series.len(),
            need: 2,
        });
    }
    check_smoothing("alpha", alpha)?;
    check_smoothing("beta", beta)?;

    let mut level = series[0];
    let mut trend = series[1] - series[0];
    let mut fitted = Vec::with_capacity(series.len());
    fitted.push(series[0]);

    for &y in &series[1..] {
        // One-step-ahead forecast for `y` using the pre-update state.
        fitted.push(level + trend);
        let new_level = alpha * y + (1.0 - alpha) * (level + trend);
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        level = new_level;
        trend = new_trend;
    }

    Ok(Holt {
        level,
        trend,
        fitted,
    })
}

/// The kind of seasonal component used by [`holt_winters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seasonality {
    /// Seasonal effects are *added* to the level-plus-trend baseline.
    Additive,
    /// Seasonal effects *multiply* the level-plus-trend baseline.
    Multiplicative,
}

/// A fitted Holt-Winters (triple exponential smoothing) model.
///
/// Extends [`Holt`] with a seasonal component of a fixed `period`. The stored
/// seasonals are the most recent full cycle, applied cyclically when
/// forecasting.
#[derive(Debug, Clone, PartialEq)]
pub struct HoltWinters {
    level: f64,
    trend: f64,
    seasonals: Vec<f64>,
    period: usize,
    seasonality: Seasonality,
}

impl HoltWinters {
    /// The final smoothed level.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// The final smoothed trend (per-step slope).
    pub fn trend(&self) -> f64 {
        self.trend
    }

    /// The most recent full cycle of seasonal components (length `period`).
    pub fn seasonals(&self) -> &[f64] {
        &self.seasonals
    }

    /// The seasonal period.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Forecast the next `h` steps, applying the seasonal cycle repeatedly.
    ///
    /// For step `i` (`1..=h`) the baseline `level + i * trend` is combined with
    /// seasonal component `(i - 1) mod period`, either additively or
    /// multiplicatively depending on the fitted [`Seasonality`].
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        (1..=h)
            .map(|i| {
                let baseline = self.level + i as f64 * self.trend;
                let seasonal = self.seasonals[(i - 1) % self.period];
                match self.seasonality {
                    Seasonality::Additive => baseline + seasonal,
                    Seasonality::Multiplicative => baseline * seasonal,
                }
            })
            .collect()
    }
}

/// Fit a Holt-Winters triple-exponential-smoothing model.
///
/// `alpha`, `beta` and `gamma` smooth the level, trend and seasonal components
/// respectively and must each lie in `[0, 1]`. `period` is the length of one
/// season and must be at least one; the series must contain at least two full
/// seasons (`2 * period` observations) so the level, trend and seasonal terms
/// can be initialised in the standard way.
///
/// Returns [`ForecastError::EmptySeries`], [`ForecastError::InvalidPeriod`],
/// [`ForecastError::SeriesTooShort`] or [`ForecastError::InvalidSmoothing`] on
/// bad input.
pub fn holt_winters(
    series: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: usize,
    seasonality: Seasonality,
) -> Result<HoltWinters, ForecastError> {
    if series.is_empty() {
        return Err(ForecastError::EmptySeries);
    }
    if period == 0 {
        return Err(ForecastError::InvalidPeriod { period });
    }
    let need = 2 * period;
    if series.len() < need {
        return Err(ForecastError::SeriesTooShort {
            got: series.len(),
            need,
        });
    }
    check_smoothing("alpha", alpha)?;
    check_smoothing("beta", beta)?;
    check_smoothing("gamma", gamma)?;

    let m = period;
    let n = series.len();

    // Initial level: mean of the first season.
    let mut level = series[..m].iter().sum::<f64>() / m as f64;

    // Initial trend: average per-position slope between the first two seasons.
    let mut trend =
        (0..m).map(|i| (series[m + i] - series[i]) / m as f64).sum::<f64>() / m as f64;

    // Initial seasonals from the first season, relative to the initial level.
    // `season[t]` holds the seasonal component estimated at time `t`.
    let mut season: Vec<f64> = (0..m)
        .map(|i| match seasonality {
            Seasonality::Additive => series[i] - level,
            Seasonality::Multiplicative => series[i] / level,
        })
        .collect();

    for (t, &y) in series.iter().enumerate().skip(m) {
        let s_prev = season[t - m];
        let baseline = level + trend;
        let new_level = match seasonality {
            Seasonality::Additive => alpha * (y - s_prev) + (1.0 - alpha) * baseline,
            Seasonality::Multiplicative => alpha * (y / s_prev) + (1.0 - alpha) * baseline,
        };
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        let new_season = match seasonality {
            Seasonality::Additive => gamma * (y - baseline) + (1.0 - gamma) * s_prev,
            Seasonality::Multiplicative => gamma * (y / baseline) + (1.0 - gamma) * s_prev,
        };
        season.push(new_season);
        level = new_level;
        trend = new_trend;
    }

    // The most recent full cycle of seasonal components.
    let seasonals = season[n - m..].to_vec();

    Ok(HoltWinters {
        level,
        trend,
        seasonals,
        period: m,
        seasonality,
    })
}
