//! Distribution fitting and non-normal capability by the percentile method
//! (ISO 22514-2 / Q-DAS style).
//!
//! [`crate::nonnormal`] lifts the normal assumption from the *moments*
//! (Cornish–Fisher, Clements). The complementary industrial approach — the one
//! Q-DAS qs-STAT automates over eight laws — is to **fit a distribution family**
//! to the data, then read capability from its percentiles:
//!
//! ```text
//! Cp = (USL − LSL) / (X_{0.99865} − X_{0.00135}) ,
//! Cpk = min( (USL − X_{0.5})/(X_{0.99865} − X_{0.5}),
//!            (X_{0.5} − LSL)/(X_{0.5} − X_{0.00135}) ) ,
//! ```
//!
//! the `6σ`-equivalent percentile spread of the *fitted* law (exact `Cp`/`Cpk`
//! for a normal, where the spread is `6σ`).
//!
//! [`FittedDistribution`] covers the [normal](FittedDistribution::Normal),
//! [lognormal](FittedDistribution::Lognormal),
//! [Rayleigh](FittedDistribution::Rayleigh) and
//! [Weibull](FittedDistribution::Weibull) laws (MLE / median-rank regression);
//! [`best_fit`] picks the family of highest log-likelihood, and
//! [`percentile_capability`] reads the indices off it. Fits are validated in
//! `fuzz_crosscheck` by parameter recovery on simulated samples and by the
//! `CDF(quantile(p)) = p` round-trip.

use crate::nonnormal::ClementsCapability;
use crate::special::{inv_normal_cdf, normal_cdf};
use serde::{Deserialize, Serialize};

/// A fitted continuous distribution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FittedDistribution {
    /// Normal `N(mean, sd²)`.
    Normal {
        /// Mean.
        mean: f64,
        /// Standard deviation.
        sd: f64,
    },
    /// Lognormal: `ln X ~ N(mu, sigma²)`.
    Lognormal {
        /// Mean of `ln X`.
        mu: f64,
        /// Standard deviation of `ln X`.
        sigma: f64,
    },
    /// Rayleigh with scale `sigma` (support `x ≥ 0`).
    Rayleigh {
        /// Scale parameter.
        sigma: f64,
    },
    /// Weibull with `shape` `k` and `scale` `λ` (support `x ≥ 0`).
    Weibull {
        /// Shape parameter `k`.
        shape: f64,
        /// Scale parameter `λ`.
        scale: f64,
    },
}

impl FittedDistribution {
    /// The `p`-quantile (inverse CDF), `p ∈ (0, 1)`.
    pub fn quantile(&self, p: f64) -> f64 {
        match *self
        {
            FittedDistribution::Normal { mean, sd } => mean + sd * inv_normal_cdf(p),
            FittedDistribution::Lognormal { mu, sigma } => (mu + sigma * inv_normal_cdf(p)).exp(),
            FittedDistribution::Rayleigh { sigma } => sigma * (-2.0 * (1.0 - p).ln()).sqrt(),
            FittedDistribution::Weibull { shape, scale } =>
            {
                scale * (-(1.0 - p).ln()).powf(1.0 / shape)
            },
        }
    }

    /// The CDF `P(X ≤ x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        match *self
        {
            FittedDistribution::Normal { mean, sd } => normal_cdf((x - mean) / sd),
            FittedDistribution::Lognormal { mu, sigma } =>
            {
                if x <= 0.0
                {
                    0.0
                }
                else
                {
                    normal_cdf((x.ln() - mu) / sigma)
                }
            },
            FittedDistribution::Rayleigh { sigma } =>
            {
                if x <= 0.0
                {
                    0.0
                }
                else
                {
                    1.0 - (-x * x / (2.0 * sigma * sigma)).exp()
                }
            },
            FittedDistribution::Weibull { shape, scale } =>
            {
                if x <= 0.0
                {
                    0.0
                }
                else
                {
                    1.0 - (-(x / scale).powf(shape)).exp()
                }
            },
        }
    }

    /// The probability density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        let inv_sqrt_2pi = 1.0 / (std::f64::consts::TAU).sqrt();
        match *self
        {
            FittedDistribution::Normal { mean, sd } =>
            {
                let z = (x - mean) / sd;
                inv_sqrt_2pi / sd * (-0.5 * z * z).exp()
            },
            FittedDistribution::Lognormal { mu, sigma } =>
            {
                if x <= 0.0
                {
                    return 0.0;
                }
                let z = (x.ln() - mu) / sigma;
                inv_sqrt_2pi / (x * sigma) * (-0.5 * z * z).exp()
            },
            FittedDistribution::Rayleigh { sigma } =>
            {
                if x < 0.0
                {
                    return 0.0;
                }
                x / (sigma * sigma) * (-x * x / (2.0 * sigma * sigma)).exp()
            },
            FittedDistribution::Weibull { shape, scale } =>
            {
                if x < 0.0
                {
                    return 0.0;
                }
                let r = x / scale;
                shape / scale * r.powf(shape - 1.0) * (-r.powf(shape)).exp()
            },
        }
    }

    /// Total log-likelihood of `data` under this distribution (`−∞` if any point
    /// has zero density, e.g. a non-positive value under a positive-support law).
    pub fn log_likelihood(&self, data: &[f64]) -> f64 {
        let mut ll = 0.0;
        for &x in data
        {
            let d = self.pdf(x);
            if d <= 0.0 || !d.is_finite()
            {
                return f64::NEG_INFINITY;
            }
            ll += d.ln();
        }
        ll
    }
}

fn mean_sd(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    // Unbiased (n − 1) sample standard deviation.
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

/// Fit a normal by its sample mean and (unbiased) standard deviation. `None`
/// for fewer than two points.
pub fn fit_normal(data: &[f64]) -> Option<FittedDistribution> {
    if data.len() < 2
    {
        return None;
    }
    let (mean, sd) = mean_sd(data);
    Some(FittedDistribution::Normal { mean, sd })
}

/// Fit a lognormal (normal on `ln x`); `None` unless every point is positive.
pub fn fit_lognormal(data: &[f64]) -> Option<FittedDistribution> {
    if data.len() < 2 || data.iter().any(|&x| x <= 0.0)
    {
        return None;
    }
    let logs: Vec<f64> = data.iter().map(|x| x.ln()).collect();
    let (mu, sigma) = mean_sd(&logs);
    Some(FittedDistribution::Lognormal { mu, sigma })
}

/// Fit a Rayleigh by MLE `σ̂ = √( (1/2n) Σ xᵢ² )`; `None` unless every point is
/// non-negative.
pub fn fit_rayleigh(data: &[f64]) -> Option<FittedDistribution> {
    if data.is_empty() || data.iter().any(|&x| x < 0.0)
    {
        return None;
    }
    let s2 = data.iter().map(|x| x * x).sum::<f64>() / (2.0 * data.len() as f64);
    if s2 <= 0.0
    {
        return None;
    }
    Some(FittedDistribution::Rayleigh { sigma: s2.sqrt() })
}

/// Fit a Weibull by median-rank regression (a Weibull-plot least-squares fit);
/// `None` unless every point is positive and a slope is resolvable.
pub fn fit_weibull(data: &[f64]) -> Option<FittedDistribution> {
    let n = data.len();
    if n < 3 || data.iter().any(|&x| x <= 0.0)
    {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Regress Y = ln(−ln(1−F)) on X = ln(x), F = median rank (i−0.3)/(n+0.4).
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (i, &x) in sorted.iter().enumerate()
    {
        let f = (i as f64 + 1.0 - 0.3) / (n as f64 + 0.4);
        let xx = x.ln();
        let yy = (-(1.0 - f).ln()).ln();
        sx += xx;
        sy += yy;
        sxx += xx * xx;
        sxy += xx * yy;
    }
    let nf = n as f64;
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-14
    {
        return None;
    }
    let k = (nf * sxy - sx * sy) / denom; // slope = shape
    if k <= 0.0
    {
        return None;
    }
    let c = (sy - k * sx) / nf; // intercept = −k·ln(λ)
    let scale = (-c / k).exp();
    if !scale.is_finite() || scale <= 0.0
    {
        return None;
    }
    Some(FittedDistribution::Weibull { shape: k, scale })
}

/// Fit every applicable family and return the one of highest log-likelihood
/// (the best-fitting model, Q-DAS style). Normal always applies; the
/// positive-support laws are tried only when the data admit them. `None` for
/// fewer than two points.
pub fn best_fit(data: &[f64]) -> Option<FittedDistribution> {
    let candidates = [
        fit_normal(data),
        fit_lognormal(data),
        fit_rayleigh(data),
        fit_weibull(data),
    ];
    candidates
        .into_iter()
        .flatten()
        .map(|d| (d.log_likelihood(data), d))
        .filter(|(ll, _)| ll.is_finite())
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, d)| d)
}

/// Capability of a fitted distribution against `[lsl, usl]` by the ISO 22514-2
/// percentile method — the `0.135 % / 50 % / 99.865 %` quantiles of the fitted
/// law standing in for `±3σ` and the median. Reduces to the classical `Cp`/`Cpk`
/// for a [`FittedDistribution::Normal`].
pub fn percentile_capability(dist: &FittedDistribution, lsl: f64, usl: f64) -> ClementsCapability {
    let up = dist.quantile(0.998_65);
    let lp = dist.quantile(0.001_35);
    let m = dist.quantile(0.5);
    let spread = up - lp;
    let cp = if spread > 0.0
    {
        (usl - lsl) / spread
    }
    else
    {
        f64::INFINITY
    };
    let cpu = if up > m
    {
        (usl - m) / (up - m)
    }
    else
    {
        f64::INFINITY
    };
    let cpl = if m > lp
    {
        (m - lsl) / (m - lp)
    }
    else
    {
        f64::INFINITY
    };
    ClementsCapability {
        cp,
        cpk: cpu.min(cpl),
        cpu,
        cpl,
        median: m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{cp, cpk};
    use approx::assert_relative_eq;

    #[test]
    fn cdf_inverts_quantile_for_every_family() {
        let dists = [
            FittedDistribution::Normal { mean: 3.0, sd: 1.2 },
            FittedDistribution::Lognormal {
                mu: 0.5,
                sigma: 0.4,
            },
            FittedDistribution::Rayleigh { sigma: 2.0 },
            FittedDistribution::Weibull {
                shape: 1.8,
                scale: 5.0,
            },
        ];
        for d in dists
        {
            for &p in &[0.05, 0.25, 0.5, 0.75, 0.95]
            {
                assert_relative_eq!(d.cdf(d.quantile(p)), p, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn percentile_capability_reduces_to_classical_for_normal() {
        let (mean, sd, lsl, usl) = (10.5, 1.0, 7.0, 13.0);
        let d = FittedDistribution::Normal { mean, sd };
        let c = percentile_capability(&d, lsl, usl);
        assert_relative_eq!(c.cp, cp(sd, lsl, usl), epsilon = 1e-3);
        assert_relative_eq!(c.cpk, cpk(mean, sd, lsl, usl), epsilon = 1e-3);
    }

    #[test]
    fn fitters_recover_parameters() {
        // Rayleigh MLE on values with known Σx².
        let r = fit_rayleigh(&[1.0, 2.0, 3.0]).unwrap();
        if let FittedDistribution::Rayleigh { sigma } = r
        {
            // σ² = (1+4+9)/(2·3) = 14/6.
            assert_relative_eq!(sigma, (14.0f64 / 6.0).sqrt(), epsilon = 1e-12);
        }
        else
        {
            panic!("expected Rayleigh");
        }
        // Lognormal recovers mean/sd of the logs.
        let ln = fit_lognormal(&[1.0, std::f64::consts::E, std::f64::consts::E.powi(2)]).unwrap();
        if let FittedDistribution::Lognormal { mu, .. } = ln
        {
            assert_relative_eq!(mu, 1.0, epsilon = 1e-12); // mean of {0,1,2}
        }
        else
        {
            panic!("expected Lognormal");
        }
    }

    #[test]
    fn positive_support_fits_reject_nonpositive() {
        assert!(fit_lognormal(&[1.0, -2.0, 3.0]).is_none());
        assert!(fit_weibull(&[1.0, 0.0, 3.0, 4.0]).is_none());
        // Normal always fits.
        assert!(fit_normal(&[-1.0, 0.0, 1.0]).is_some());
    }

    #[test]
    fn best_fit_prefers_the_true_family() {
        // Strongly right-skewed positive data ⇒ a skewed law beats the normal.
        let data = [0.2, 0.3, 0.35, 0.5, 0.6, 0.9, 1.2, 1.8, 2.5, 4.0, 6.0];
        let best = best_fit(&data).unwrap();
        assert!(!matches!(best, FittedDistribution::Normal { .. }));
    }
}
