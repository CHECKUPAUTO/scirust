//! ARIMA(p, d, q) modelling: `d`-th order differencing (the "I") composed
//! with a joint AR(p) + MA(q) fit via the two-stage Hannan-Rissanen
//! procedure (Hannan & Rissanen, 1982, *Recursive estimation of mixed
//! autoregressive-moving average order*, Biometrika 69(1)):
//!
//! 1. Fit a long autoregression to the differenced series (reusing
//!    [`crate::ar_fit`]) and take its in-sample one-step residuals as a
//!    consistent estimate of the true innovations.
//! 2. Regress the differenced series on `p` of its own lags *and* `q` lags
//!    of those estimated innovations, by ordinary least squares — this
//!    single joint regression gives both the AR and MA coefficients.
//!
//! This avoids full nonlinear (state-space/Kalman-filter) maximum-likelihood
//! estimation while still handling the MA component, which a pure
//! Yule-Walker fit ([`crate::ar_fit`]) cannot represent at all. Consistent
//! with the crate's "pure Rust, no dependencies" guarantee, the OLS solve
//! (a small, self-contained linear system) is implemented from scratch here
//! rather than pulling in `scirust-solvers`.

use crate::error::ForecastError;
use crate::utils::difference;

/// A fitted ARIMA(p, d, q) model.
#[derive(Debug, Clone, PartialEq)]
pub struct ArimaModel {
    d: usize,
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    intercept: f64,
    /// Last `p` values of the `d`-times-differenced series, oldest first.
    history: Vec<f64>,
    /// Last `q` one-step-ahead residuals of the fitted model, oldest first.
    resid_history: Vec<f64>,
    /// Last value of the series at each differencing level `0..d`, needed to
    /// integrate forecasts of the differenced process back to the original
    /// scale (level 0 is the original series' last value).
    undiff_tails: Vec<f64>,
}

impl ArimaModel {
    /// The autoregressive coefficients `phi_1 .. phi_p` (`phi_0` multiplies
    /// the most recent lag of the differenced series).
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coeffs
    }

    /// The moving-average coefficients `theta_1 .. theta_q` (`theta_0`
    /// multiplies the most recent one-step residual).
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coeffs
    }

    /// The regression intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The differencing order `d`.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Forecast the next `h` observations, on the original (undifferenced)
    /// scale.
    ///
    /// Future innovations are taken to be `0` (their expectation), the
    /// standard convention for point forecasts from an ARMA model — so the
    /// MA terms' contribution fades out once the horizon exceeds `q`.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        let p = self.ar_coeffs.len();
        let q = self.ma_coeffs.len();
        let mut w = self.history.clone(); // running tail of the differenced series
        let mut resid = self.resid_history.clone(); // running tail of residuals
        let mut w_forecast = Vec::with_capacity(h);

        for _ in 0..h
        {
            let mut pred = self.intercept;
            for (i, &phi) in self.ar_coeffs.iter().enumerate()
            {
                pred += phi * w[p - 1 - i];
            }
            for (j, &theta) in self.ma_coeffs.iter().enumerate()
            {
                pred += theta * resid[q - 1 - j];
            }
            w_forecast.push(pred);
            if p > 0
            {
                w.remove(0);
                w.push(pred);
            }
            // Future innovations are 0 in expectation.
            if q > 0
            {
                resid.remove(0);
                resid.push(0.0);
            }
        }

        // Integrate d times back to the original scale: level_{k} forecasts
        // are the cumulative sum of level_{k+1} forecasts, seeded by the
        // last observed value at level k.
        let mut level = w_forecast;
        for &tail in self.undiff_tails.iter().rev()
        {
            let mut acc = tail;
            for v in level.iter_mut()
            {
                acc += *v;
                *v = acc;
            }
        }
        level
    }
}

/// Heuristic order for the Hannan-Rissanen long autoregression: comfortably
/// above `p + q` (so the estimated residuals are a good proxy for the true
/// innovations) while leaving enough trailing observations for the OLS
/// stage to be well-posed.
fn long_ar_order(n: usize, p: usize, q: usize) -> usize {
    let adaptive = (n as f64).sqrt().ceil() as usize;
    (p + q + 5).max(adaptive).min(n / 3).max(1)
}

/// Solve the small dense linear system `a * x = b` (`a` is `k x k`,
/// row-major) via Gaussian elimination with partial pivoting. Self-contained
/// (no dependency on `scirust-solvers`) since `k` here is always `1 + p + q`,
/// a handful of unknowns.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let k = b.len();
    for col in 0..k
    {
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        for row in (col + 1)..k
        {
            if a[row][col].abs() > pivot_val
            {
                pivot_val = a[row][col].abs();
                pivot_row = row;
            }
        }
        if pivot_val < 1e-300
        {
            return None;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot = a[col][col];
        for row in (col + 1)..k
        {
            let factor = a[row][col] / pivot;
            if factor == 0.0
            {
                continue;
            }
            for c in col..k
            {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0; k];
    for row in (0..k).rev()
    {
        let mut s = b[row];
        for c in (row + 1)..k
        {
            s -= a[row][c] * x[c];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

/// Fit an ARIMA(p, d, q) model to `series` via `d`-th order differencing
/// followed by a joint AR(p)/MA(q) fit (Hannan-Rissanen).
///
/// `p` and `q` may each be `0` (a pure MA(q) or AR(p) model on the
/// differenced series, respectively; `p = q = 0` fits just an intercept —
/// the differenced series' mean, i.e. a random walk with drift when
/// `d = 1`).
///
/// Returns [`ForecastError::EmptySeries`] on an empty series, or
/// [`ForecastError::SeriesTooShort`] when there are not enough observations
/// left after differencing and reserving data for the long autoregression
/// and the joint regression.
pub fn arima_fit(
    series: &[f64],
    p: usize,
    d: usize,
    q: usize,
) -> Result<ArimaModel, ForecastError> {
    if series.is_empty()
    {
        return Err(ForecastError::EmptySeries);
    }

    // Difference d times, keeping the last value at each level to integrate
    // forecasts back to the original scale.
    let mut level = series.to_vec();
    let mut undiff_tails = Vec::with_capacity(d);
    for _ in 0..d
    {
        if level.len() < 2
        {
            return Err(ForecastError::SeriesTooShort {
                got: series.len(),
                need: series.len() + 1,
            });
        }
        undiff_tails.push(*level.last().unwrap());
        level = difference(&level, 1);
    }
    let w = level;
    let n = w.len();

    let m = long_ar_order(n, p, q);
    let min_needed = m + 1 + p.max(q) + p + q + 2;
    if n <= m || n < min_needed
    {
        return Err(ForecastError::SeriesTooShort {
            got: series.len(),
            need: series.len() + (min_needed.saturating_sub(n)),
        });
    }

    // Stage 1: long AR fit, then its in-sample one-step residuals as a
    // proxy for the true innovations.
    let long_ar = crate::autoreg::ar_fit(&w, m)?;
    let long_coeffs = long_ar.coefficients();
    let long_intercept = long_ar.intercept();
    let mut resid = vec![0.0; n];
    for t in m..n
    {
        let mut pred = long_intercept;
        for (i, &c) in long_coeffs.iter().enumerate()
        {
            pred += c * w[t - 1 - i];
        }
        resid[t] = w[t] - pred;
    }

    // Stage 2: joint OLS regression of w[t] on p lags of w and q lags of the
    // stage-1 residuals. Valid rows need both sets of lags available.
    let start = p.max(m + q);
    let k = 1 + p + q; // intercept + AR + MA coefficients
    let mut ata = vec![vec![0.0; k]; k];
    let mut atb = vec![0.0; k];
    let mut rows = 0usize;
    for t in start..n
    {
        let mut x_row = vec![1.0; k];
        for i in 0..p
        {
            x_row[1 + i] = w[t - 1 - i];
        }
        for j in 0..q
        {
            x_row[1 + p + j] = resid[t - 1 - j];
        }
        for r in 0..k
        {
            atb[r] += x_row[r] * w[t];
            for c in 0..k
            {
                ata[r][c] += x_row[r] * x_row[c];
            }
        }
        rows += 1;
    }
    if rows < k
    {
        return Err(ForecastError::SeriesTooShort {
            got: series.len(),
            need: series.len() + (k - rows),
        });
    }

    let beta = solve_linear_system(ata, atb).ok_or(ForecastError::SeriesTooShort {
        got: series.len(),
        need: series.len() + 1,
    })?;
    let intercept = beta[0];
    let ar_coeffs = beta[1..1 + p].to_vec();
    let ma_coeffs = beta[1 + p..1 + p + q].to_vec();

    // Final in-sample residuals, self-consistent with the fitted (AR, MA)
    // coefficients: computed by a single forward recursive pass, treating
    // pre-sample lags (of both w and the residuals themselves) as 0 — the
    // standard convention, whose influence on the *last* few residuals
    // (all that forecasting needs) is negligible for a stationary,
    // invertible fit.
    let mut final_resid = vec![0.0; n];
    for t in 0..n
    {
        let mut pred = intercept;
        for (i, &phi) in ar_coeffs.iter().enumerate()
        {
            if t > i
            {
                pred += phi * w[t - 1 - i];
            }
        }
        for (j, &theta) in ma_coeffs.iter().enumerate()
        {
            if t > j
            {
                pred += theta * final_resid[t - 1 - j];
            }
        }
        final_resid[t] = w[t] - pred;
    }

    let history = if p == 0
    {
        Vec::new()
    }
    else
    {
        w[n - p..].to_vec()
    };
    let resid_history = if q == 0
    {
        Vec::new()
    }
    else
    {
        final_resid[n - q..].to_vec()
    };

    Ok(ArimaModel {
        d,
        ar_coeffs,
        ma_coeffs,
        intercept,
        history,
        resid_history,
        undiff_tails,
    })
}
