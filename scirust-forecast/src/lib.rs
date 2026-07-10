//! `scirust-forecast` — classical time-series forecasting on `&[f64]` series.
//!
//! This crate provides deterministic, dependency-free implementations of the
//! workhorse univariate forecasting methods:
//!
//! - [`simple_exp_smoothing`] — simple exponential smoothing ([`Ses`]).
//! - [`holt`] — double exponential smoothing with a trend ([`Holt`]).
//! - [`holt_winters`] — triple exponential smoothing with additive or
//!   multiplicative seasonality ([`HoltWinters`], [`Seasonality`]).
//! - [`ar_fit`] — autoregressive AR(p) fitting via the Yule-Walker equations
//!   and the Levinson-Durbin recursion ([`ArModel`]).
//! - [`difference`] / [`moving_average`] — series-transformation helpers.
//! - [`metrics`] — [`mae`](metrics::mae), [`rmse`](metrics::rmse) and
//!   [`mape`](metrics::mape) accuracy scores.
//!
//! Every fallible entry point returns [`ForecastError`], which implements
//! [`std::fmt::Display`] and [`std::error::Error`].
//!
//! # Example
//!
//! ```
//! use scirust_forecast::{holt, simple_exp_smoothing};
//!
//! // A clean linear trend: y_t = 2 + 3 t.
//! let series: Vec<f64> = (0..10).map(|t| 2.0 + 3.0 * t as f64).collect();
//!
//! // Holt recovers the slope and extrapolates the next three points exactly.
//! let model = holt(&series, 0.5, 0.5).unwrap();
//! let fc = model.forecast(3);
//! assert!((fc[0] - 32.0).abs() < 1e-6); // y_10 = 2 + 3*10
//! assert!((fc[1] - 35.0).abs() < 1e-6); // y_11
//! assert!((fc[2] - 38.0).abs() < 1e-6); // y_12
//!
//! // Simple exponential smoothing produces a flat forecast at the final level.
//! let flat = simple_exp_smoothing(&series, 0.4).unwrap();
//! assert_eq!(flat.forecast(2).len(), 2);
//! ```
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod autoreg;
pub mod error;
pub mod metrics;
pub mod smoothing;
pub mod utils;

pub use autoreg::{ArModel, ar_fit};
pub use error::ForecastError;
pub use metrics::{mae, mape, rmse};
pub use smoothing::{
    Holt, HoltWinters, Seasonality, Ses, holt, holt_winters, simple_exp_smoothing,
};
pub use utils::{difference, moving_average};

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // --- Deterministic pseudo-random innovations (no `rand` crate). ---

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform draw in `[-1, 1)`.
    fn next_unit(state: &mut u64) -> f64 {
        let bits = splitmix64(state) >> 11; // 53 significant bits
        let f = bits as f64 / (1u64 << 53) as f64; // [0, 1)
        2.0 * f - 1.0
    }

    // ---------- Simple exponential smoothing ----------

    #[test]
    fn ses_constant_series_forecasts_constant() {
        let series = [7.0; 12];
        let model = simple_exp_smoothing(&series, 0.3).unwrap();
        assert!(approx(model.level(), 7.0, TOL));
        for v in model.forecast(5)
        {
            assert!(approx(v, 7.0, TOL));
        }
        assert_eq!(model.fitted().len(), series.len());
    }

    #[test]
    fn ses_flat_forecast_equals_level() {
        let series = [3.0, 5.0, 2.0, 8.0, 6.0];
        let model = simple_exp_smoothing(&series, 0.5).unwrap();
        let fc = model.forecast(4);
        assert_eq!(fc.len(), 4);
        assert!(fc.iter().all(|&v| approx(v, model.level(), TOL)));
    }

    // ---------- Holt ----------

    #[test]
    fn holt_constant_series() {
        let series = [42.0; 8];
        let model = holt(&series, 0.5, 0.5).unwrap();
        assert!(approx(model.trend(), 0.0, TOL));
        for v in model.forecast(4)
        {
            assert!(approx(v, 42.0, TOL));
        }
    }

    #[test]
    fn holt_recovers_linear_trend() {
        // y_t = 2 + 3 t.
        let a = 2.0;
        let b = 3.0;
        let series: Vec<f64> = (0..20).map(|t| a + b * t as f64).collect();
        let model = holt(&series, 0.5, 0.5).unwrap();
        assert!(approx(model.trend(), b, 1e-6));
        let fc = model.forecast(5);
        for (i, v) in fc.iter().enumerate()
        {
            let t = 20 + i; // next indices
            let truth = a + b * t as f64;
            assert!(approx(*v, truth, 1e-3), "h={i}: {v} vs {truth}");
        }
    }

    // ---------- Holt-Winters ----------

    #[test]
    fn holt_winters_additive_reproduces_season() {
        let m = 4;
        let pattern = [10.0, -5.0, 3.0, -8.0]; // sums to zero
        let level = 100.0;
        let series: Vec<f64> = (0..32).map(|t| level + pattern[t % m]).collect();
        let model = holt_winters(&series, 0.3, 0.1, 0.3, m, Seasonality::Additive).unwrap();

        // Forecast one full period ahead should match the underlying truth.
        let fc = model.forecast(m);
        for (i, v) in fc.iter().enumerate()
        {
            let truth = level + pattern[(32 + i) % m];
            assert!(approx(*v, truth, 1e-6), "h={i}: {v} vs {truth}");
        }
        // The recovered seasonal cycle matches the pattern.
        for (i, s) in model.seasonals().iter().enumerate()
        {
            assert!(approx(*s, pattern[(32 - m + i) % m], 1e-6));
        }
    }

    #[test]
    fn holt_winters_multiplicative_reproduces_season() {
        let m = 4;
        let factors = [1.2, 0.8, 1.1, 0.9]; // mean 1.0
        let level = 50.0;
        let series: Vec<f64> = (0..32).map(|t| level * factors[t % m]).collect();
        let model = holt_winters(&series, 0.3, 0.1, 0.3, m, Seasonality::Multiplicative).unwrap();

        let fc = model.forecast(m);
        for (i, v) in fc.iter().enumerate()
        {
            let truth = level * factors[(32 + i) % m];
            assert!(approx(*v, truth, 1e-6), "h={i}: {v} vs {truth}");
        }
    }

    // ---------- Autoregression ----------

    #[test]
    fn ar_constant_series_forecasts_constant() {
        let series = [9.0; 30];
        let model = ar_fit(&series, 2).unwrap();
        for v in model.forecast(5)
        {
            assert!(approx(v, 9.0, 1e-6));
        }
    }

    #[test]
    fn ar_recovers_ar1_coefficient() {
        // x_t = 0.7 x_{t-1} + 0.5 * u_t, u_t ~ Uniform[-1, 1).
        let phi = 0.7;
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let n = 2000;
        let mut x = vec![0.0; n];
        for t in 1..n
        {
            x[t] = phi * x[t - 1] + 0.5 * next_unit(&mut state);
        }
        let model = ar_fit(&x, 1).unwrap();
        // Yule-Walker estimate of an AR(1) coefficient. Tolerance ~1e-1.
        assert!(
            approx(model.coefficients()[0], phi, 0.1),
            "phi_hat = {}",
            model.coefficients()[0]
        );

        // One-step forecast reconstructs the recurrence from the last value.
        let last = *x.last().unwrap();
        let expected = model.intercept() + model.coefficients()[0] * last;
        let fc = model.forecast(1);
        assert!(approx(fc[0], expected, 1e-9));
        // ... and is close to the true conditional mean phi * x_last.
        assert!(approx(fc[0], phi * last, 0.1 * (last.abs() + 1.0) + 0.2));
    }

    // ---------- Utilities ----------

    #[test]
    fn difference_hand_verified() {
        let series = [1.0, 2.0, 4.0, 7.0, 11.0];
        assert_eq!(difference(&series, 1), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(difference(&series, 2), vec![3.0, 5.0, 7.0]);
        assert!(difference(&series, 5).is_empty());
        assert!(difference(&series, 9).is_empty());
    }

    #[test]
    fn moving_average_hand_verified() {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(moving_average(&series, 3).unwrap(), vec![2.0, 3.0, 4.0]);
        assert_eq!(moving_average(&series, 1).unwrap(), series.to_vec());
        assert_eq!(moving_average(&series, 5).unwrap(), vec![3.0]);
    }

    // ---------- Metrics ----------

    #[test]
    fn metrics_hand_verified() {
        let actual = [2.0, 4.0, 6.0, 8.0];
        let pred = [1.0, 5.0, 5.0, 10.0];
        assert!(approx(mae(&actual, &pred).unwrap(), 1.25, TOL));
        assert!(approx(rmse(&actual, &pred).unwrap(), 1.75_f64.sqrt(), TOL));
        // (0.5 + 0.25 + 1/6 + 0.25) / 4 * 100.
        assert!(approx(mape(&actual, &pred).unwrap(), 29.166_666_666, 1e-6));
    }

    // ---------- Error paths ----------

    #[test]
    fn error_empty_series() {
        assert_eq!(
            simple_exp_smoothing(&[], 0.3).unwrap_err(),
            ForecastError::EmptySeries
        );
        assert_eq!(ar_fit(&[], 1).unwrap_err(), ForecastError::EmptySeries);
        assert_eq!(
            moving_average(&[], 2).unwrap_err(),
            ForecastError::EmptySeries
        );
        assert_eq!(mae(&[], &[]).unwrap_err(), ForecastError::EmptySeries);
    }

    #[test]
    fn error_bad_smoothing_parameters() {
        assert!(matches!(
            simple_exp_smoothing(&[1.0, 2.0], 1.5).unwrap_err(),
            ForecastError::InvalidSmoothing { name: "alpha", .. }
        ));
        assert!(matches!(
            simple_exp_smoothing(&[1.0, 2.0], -0.1).unwrap_err(),
            ForecastError::InvalidSmoothing { .. }
        ));
        assert!(matches!(
            holt(&[1.0, 2.0, 3.0], 0.5, 2.0).unwrap_err(),
            ForecastError::InvalidSmoothing { name: "beta", .. }
        ));
        // NaN is rejected too.
        assert!(matches!(
            simple_exp_smoothing(&[1.0, 2.0], f64::NAN).unwrap_err(),
            ForecastError::InvalidSmoothing { .. }
        ));
    }

    #[test]
    fn error_series_too_short() {
        assert!(matches!(
            holt(&[1.0], 0.5, 0.5).unwrap_err(),
            ForecastError::SeriesTooShort { got: 1, need: 2 }
        ));
        // Period longer than series (needs two full seasons).
        assert!(matches!(
            holt_winters(
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                0.3,
                0.1,
                0.3,
                7,
                Seasonality::Additive
            )
            .unwrap_err(),
            ForecastError::SeriesTooShort { .. }
        ));
        assert!(matches!(
            ar_fit(&[1.0, 2.0], 3).unwrap_err(),
            ForecastError::SeriesTooShort { got: 2, need: 4 }
        ));
    }

    #[test]
    fn error_invalid_period_window_order() {
        assert!(matches!(
            holt_winters(
                &[1.0, 2.0, 3.0, 4.0],
                0.3,
                0.1,
                0.3,
                0,
                Seasonality::Additive
            )
            .unwrap_err(),
            ForecastError::InvalidPeriod { period: 0 }
        ));
        assert!(matches!(
            moving_average(&[1.0, 2.0], 0).unwrap_err(),
            ForecastError::InvalidWindow { window: 0 }
        ));
        assert!(matches!(
            moving_average(&[1.0, 2.0], 5).unwrap_err(),
            ForecastError::InvalidWindow { window: 5 }
        ));
        assert!(matches!(
            ar_fit(&[1.0, 2.0], 0).unwrap_err(),
            ForecastError::InvalidOrder { order: 0 }
        ));
    }

    #[test]
    fn error_gamma_out_of_range() {
        let series: Vec<f64> = (0..8).map(|t| t as f64).collect();
        assert!(matches!(
            holt_winters(&series, 0.3, 0.1, 2.0, 2, Seasonality::Additive).unwrap_err(),
            ForecastError::InvalidSmoothing { name: "gamma", .. }
        ));
    }

    #[test]
    fn error_metric_length_and_zero() {
        assert!(matches!(
            mae(&[1.0, 2.0], &[1.0]).unwrap_err(),
            ForecastError::LengthMismatch { left: 2, right: 1 }
        ));
        assert!(matches!(
            rmse(&[1.0], &[1.0, 2.0]).unwrap_err(),
            ForecastError::LengthMismatch { .. }
        ));
        assert!(matches!(
            mape(&[1.0, 0.0], &[1.0, 1.0]).unwrap_err(),
            ForecastError::ZeroActual { index: 1 }
        ));
    }

    #[test]
    fn error_display_is_nonempty() {
        let e = ForecastError::ZeroActual { index: 3 };
        assert!(!format!("{e}").is_empty());
        let _: &dyn std::error::Error = &e;
    }
}
