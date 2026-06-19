//! Distribution-free RUL prediction intervals.
//!
//! The base estimators in [`crate::rul`] emit a point Remaining-Useful-Life
//! plus a *heuristic* Gaussian confidence band (`±k·σ`) — which only holds if
//! the residuals are Gaussian, and degradation residuals rarely are.
//!
//! [`ConformalRul`] replaces that band with a **split-conformal** interval
//! calibrated on real RUL residuals `|RUL_true − RUL_pred|` from run-to-failure
//! history: `[r̂ − q̂, r̂ + q̂]` has marginal coverage `≥ 1 − α` with **no
//! distributional assumption** and a finite-sample guarantee. Built on the
//! audited [`scirust_core::nn::conformal::ConformalRegressor`], so the
//! maintenance-planning interval and the model-side conformal predictors share
//! one guarantee.

use crate::rul::RulPrediction;
use scirust_core::nn::conformal::ConformalRegressor;
use serde::{Deserialize, Serialize};

/// Split-conformal RUL interval calibrated on absolute RUL residuals (hours).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalRul {
    half_width: f64,
    alpha: f64,
    n_calib: usize,
}

impl ConformalRul {
    /// Calibrate on absolute RUL residuals `|RUL_true − RUL_pred|` (hours)
    /// gathered from run-to-failure trajectories. `alpha ∈ (0,1)` is the target
    /// miscoverage; the resulting intervals cover the true RUL with probability
    /// `≥ 1 − α`. With too few calibration points the half-width is `+∞` (the
    /// interval is `[0, ∞)` — uninformative but never under-covers).
    pub fn calibrate(abs_residuals_hours: &[f64], alpha: f64) -> Self {
        let r: Vec<f32> = abs_residuals_hours.iter().map(|&x| x as f32).collect();
        let reg = ConformalRegressor::calibrate(&r, alpha as f32);
        Self {
            half_width: reg.half_width() as f64,
            alpha,
            n_calib: abs_residuals_hours.len(),
        }
    }

    /// Conformal half-width `q̂` (hours).
    pub fn half_width(&self) -> f64 {
        self.half_width
    }

    /// Target miscoverage `α` (coverage `≥ 1 − α`).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of calibration residuals used.
    pub fn n_calib(&self) -> usize {
        self.n_calib
    }

    /// Guaranteed-coverage interval around a point RUL prediction (hours),
    /// clamped at 0 since remaining life is non-negative.
    pub fn interval(&self, rul_hat_hours: f64) -> (f64, f64) {
        (
            (rul_hat_hours - self.half_width).max(0.0),
            rul_hat_hours + self.half_width,
        )
    }

    /// Whether the interval around `rul_hat` covers the realized `rul_true`.
    pub fn covers(&self, rul_hat_hours: f64, rul_true_hours: f64) -> bool {
        let (lo, hi) = self.interval(rul_hat_hours);
        rul_true_hours >= lo && rul_true_hours <= hi
    }

    /// Return a copy of `pred` whose bounds are the conformal interval (the
    /// `method` string is tagged `+conformal`).
    pub fn apply(&self, pred: &RulPrediction) -> RulPrediction {
        let (lo, hi) = self.interval(pred.remaining_hours);
        let mut out = pred.clone();
        out.lower_bound_hours = lo;
        out.upper_bound_hours = hi;
        out.method = format!("{}+conformal", pred.method);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic splitmix64 with a heavy-tailed (two-sided exponential /
    /// Laplace) residual draw — deliberately NON-Gaussian, so a `±kσ` band
    /// would mis-cover but the conformal interval must not.
    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn u01(&mut self) -> f64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn laplace(&mut self, scale: f64) -> f64 {
            let mag = -scale * self.u01().max(1e-12).ln();
            if self.u01() < 0.5 { -mag } else { mag }
        }
    }

    #[test]
    fn coverage_is_distribution_free() {
        let mut rng = Rng::new(0x5EED);
        let alpha = 0.1;
        let cal: Vec<f64> = (0..2000).map(|_| rng.laplace(50.0).abs()).collect();
        let guard = ConformalRul::calibrate(&cal, alpha);
        assert!(guard.half_width().is_finite() && guard.half_width() > 0.0);

        let n = 8000;
        let mut covered = 0usize;
        for _ in 0..n
        {
            let rul_true = 100.0 + 500.0 * rng.u01();
            let resid = rng.laplace(50.0);
            let rul_hat = rul_true - resid;
            if guard.covers(rul_hat, rul_true)
            {
                covered += 1;
            }
        }
        let cov = covered as f64 / n as f64;
        // Guaranteed ≥ 1−α; calibrated (not vacuously 1.0) on a non-Gaussian law.
        assert!(
            (0.87..=0.97).contains(&cov),
            "coverage {cov} outside [0.87, 0.97] for 1-alpha = {}",
            1.0 - alpha
        );
    }

    #[test]
    fn apply_replaces_band_and_tags_method() {
        let guard = ConformalRul::calibrate(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 0.2);
        let pred = RulPrediction {
            remaining_hours: 200.0,
            lower_bound_hours: 190.0,
            upper_bound_hours: 210.0,
            health_index: 0.6,
            timestamp_hours: 100.0,
            method: "linear".to_string(),
        };
        let out = guard.apply(&pred);
        let (lo, hi) = guard.interval(200.0);
        assert_eq!(out.lower_bound_hours, lo);
        assert_eq!(out.upper_bound_hours, hi);
        assert_eq!(out.method, "linear+conformal");
    }

    #[test]
    fn too_few_points_never_under_covers() {
        // n=3, alpha=0.01 -> ceil(4*0.99)=4 > 3 -> q̂ = +inf -> covers anything.
        let guard = ConformalRul::calibrate(&[1.0, 2.0, 3.0], 0.01);
        assert!(guard.half_width().is_infinite());
        assert!(guard.covers(100.0, 0.0));
        assert!(guard.covers(100.0, 1e9));
    }
}
