//! Autoregressive AR(p) modelling via the Yule-Walker equations solved with
//! the Levinson-Durbin recursion.

use crate::error::ForecastError;

/// A fitted autoregressive AR(p) model.
///
/// The model has the form
/// `y_t = intercept + sum_i coeff_i * y_{t-i-1} + e_t`,
/// where the coefficients are obtained from the (biased) sample
/// autocovariances by solving the Yule-Walker system with the Levinson-Durbin
/// recursion, and the intercept is derived from the series mean.
#[derive(Debug, Clone, PartialEq)]
pub struct ArModel {
    coeffs: Vec<f64>,
    intercept: f64,
    /// The last `p` observations, oldest first, used to seed forecasting.
    history: Vec<f64>,
}

impl ArModel {
    /// The autoregressive coefficients `coeff_0 .. coeff_{p-1}`, where
    /// `coeff_0` multiplies the most recent lag `y_{t-1}`.
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    /// The model intercept (`mean * (1 - sum(coeffs))`).
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The autoregressive order `p`.
    pub fn order(&self) -> usize {
        self.coeffs.len()
    }

    /// Forecast the next `h` steps by iterating the AR recurrence, feeding the
    /// model's own predictions back in for multi-step horizons.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        let p = self.coeffs.len();
        let mut recent = self.history.clone();
        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            let mut pred = self.intercept;
            // `coeffs[i]` multiplies the (i+1)-th most recent value.
            for (i, &c) in self.coeffs.iter().enumerate() {
                pred += c * recent[p - 1 - i];
            }
            out.push(pred);
            recent.remove(0);
            recent.push(pred);
        }
        out
    }
}

/// Fit an AR(p) model to `series` via the Yule-Walker / Levinson-Durbin method.
///
/// The intercept is taken from the series mean and the coefficients from the
/// biased sample autocovariances. A perfectly constant series (zero variance)
/// yields zero coefficients and an intercept equal to that constant.
///
/// Returns [`ForecastError::EmptySeries`], [`ForecastError::InvalidOrder`]
/// (when `p == 0`) or [`ForecastError::SeriesTooShort`] (when the series has
/// `p` or fewer observations).
pub fn ar_fit(series: &[f64], p: usize) -> Result<ArModel, ForecastError> {
    if series.is_empty() {
        return Err(ForecastError::EmptySeries);
    }
    if p == 0 {
        return Err(ForecastError::InvalidOrder { order: p });
    }
    if series.len() <= p {
        return Err(ForecastError::SeriesTooShort {
            got: series.len(),
            need: p + 1,
        });
    }

    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;

    // Biased sample autocovariances r_0 .. r_p.
    let autocov = |k: usize| -> f64 {
        (0..n - k)
            .map(|t| (series[t] - mean) * (series[t + k] - mean))
            .sum::<f64>()
            / n as f64
    };
    let r: Vec<f64> = (0..=p).map(autocov).collect();

    let history = series[n - p..].to_vec();

    // Degenerate (constant) series: zero variance -> flat forecast at the mean.
    if r[0] <= f64::EPSILON {
        return Ok(ArModel {
            coeffs: vec![0.0; p],
            intercept: mean,
            history,
        });
    }

    // Levinson-Durbin recursion on the normalised autocorrelations.
    let rho: Vec<f64> = r.iter().map(|&rk| rk / r[0]).collect();
    let mut phi = vec![0.0_f64; p];
    phi[0] = rho[1];
    let mut err = 1.0 - rho[1] * rho[1];

    for k in 2..=p {
        // Reflection coefficient for order k.
        let mut acc = rho[k];
        for j in 1..k {
            acc -= phi[j - 1] * rho[k - j];
        }
        let refl = if err.abs() > f64::EPSILON {
            acc / err
        } else {
            0.0
        };

        let prev: Vec<f64> = phi[..k - 1].to_vec();
        for j in 0..k - 1 {
            phi[j] = prev[j] - refl * prev[k - 2 - j];
        }
        phi[k - 1] = refl;
        err *= 1.0 - refl * refl;
    }

    let intercept = mean * (1.0 - phi.iter().sum::<f64>());

    Ok(ArModel {
        coeffs: phi,
        intercept,
        history,
    })
}
