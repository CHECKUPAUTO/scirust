//! Deterministic split-conformal prediction intervals (direction C).
//!
//! The benchmark produces point predictions and promote/hold decisions; a
//! deployment also needs **calibrated uncertainty**. Split-conformal prediction
//! turns any point predictor into an interval with a *distribution-free,
//! finite-sample* marginal coverage guarantee (Vovk; Lei et al., 2018): given a
//! point predictor `ŷ` and an **exchangeable calibration set** of residuals, the
//! symmetric band `ŷ ± q` covers a fresh exchangeable target with probability at
//! least `level`, for **any** residual distribution — no Gaussianity, no
//! variance estimate, no RNG.
//!
//! The half-width `q` is the conformal quantile of the absolute calibration
//! residuals: with `n` calibration points and nonconformity scores
//! `sᵢ = |yᵢ − ŷᵢ|`, `q` is the `⌈(n+1)·level⌉`-th smallest score (the `+1` is the
//! finite-sample correction that makes the guarantee exact under exchangeability).
//! When `⌈(n+1)·level⌉ > n` no finite band can meet the level; that is reported as
//! [`ConformalError::CalibrationTooSmall`] rather than silently returning `+∞`.
//!
//! This is exactly the honest counterpart to the program's heavy-tailed findings:
//! under heavy tails a `±σ` band lies, but a conformal band is valid by
//! construction, and a *robust* point predictor (smaller bulk residuals) yields a
//! *tighter* valid band — which the accompanying `industrial-obd2-conformal`
//! binary measures on real data. Determinism: a sort and an index; no RNG.

use core::fmt;

/// A fitted split-conformal band: one symmetric half-width `q` at a nominal level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitConformal {
    half_width: f64,
    level: f64,
    /// Calibration points used (recorded for reporting).
    calibration_count: usize,
}

/// Typed split-conformal errors.
#[derive(Clone, Debug, PartialEq)]
pub enum ConformalError {
    /// The calibration set was empty.
    EmptyCalibration,
    /// The requested level was not in the open interval `(0, 1)`.
    InvalidLevel {
        /// The rejected level.
        level: f64,
    },
    /// A calibration residual was NaN or infinite.
    NonFiniteResidual,
    /// `⌈(n+1)·level⌉ > n`: too few calibration points for a finite band at this
    /// level (need at least `⌈level / (1 − level)⌉` points).
    CalibrationTooSmall {
        /// Calibration points supplied.
        count: usize,
        /// Requested level.
        level: f64,
    },
}

impl fmt::Display for ConformalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyCalibration => write!(formatter, "the calibration set is empty"),
            Self::InvalidLevel { level } =>
            {
                write!(
                    formatter,
                    "level {level} must lie in the open interval (0, 1)"
                )
            },
            Self::NonFiniteResidual =>
            {
                write!(formatter, "a calibration residual is not finite")
            },
            Self::CalibrationTooSmall { count, level } =>
            {
                write!(
                    formatter,
                    "{count} calibration points are too few for a finite band at level {level}"
                )
            },
        }
    }
}

impl std::error::Error for ConformalError {}

impl SplitConformal {
    /// Fits the band from calibration residuals `yᵢ − ŷᵢ` at nominal coverage
    /// `level ∈ (0, 1)`.
    ///
    /// # Errors
    ///
    /// [`ConformalError::EmptyCalibration`] / [`ConformalError::InvalidLevel`] /
    /// [`ConformalError::NonFiniteResidual`] for malformed input, and
    /// [`ConformalError::CalibrationTooSmall`] when `⌈(n+1)·level⌉ > n`.
    pub fn fit(calibration_residuals: &[f64], level: f64) -> Result<Self, ConformalError> {
        if calibration_residuals.is_empty()
        {
            return Err(ConformalError::EmptyCalibration);
        }

        if !(level > 0.0 && level < 1.0)
        {
            return Err(ConformalError::InvalidLevel { level });
        }

        let mut scores: Vec<f64> = Vec::with_capacity(calibration_residuals.len());

        for &residual in calibration_residuals
        {
            if !residual.is_finite()
            {
                return Err(ConformalError::NonFiniteResidual);
            }

            scores.push(residual.abs());
        }

        scores.sort_by(f64::total_cmp);

        let n = scores.len();
        // The finite-sample conformal rank; `+1` is the exchangeability correction.
        let rank = (((n + 1) as f64) * level).ceil() as usize;

        if rank > n
        {
            return Err(ConformalError::CalibrationTooSmall { count: n, level });
        }

        Ok(Self {
            half_width: scores[rank - 1],
            level,
            calibration_count: n,
        })
    }

    /// The symmetric half-width `q`.
    pub fn half_width(&self) -> f64 {
        self.half_width
    }

    /// The full interval width `2q`.
    pub fn width(&self) -> f64 {
        2.0 * self.half_width
    }

    /// The nominal coverage level the band was fitted at.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Calibration points used.
    pub fn calibration_count(&self) -> usize {
        self.calibration_count
    }

    /// The prediction interval `[ŷ − q, ŷ + q]` around a point prediction.
    pub fn interval(&self, prediction: f64) -> (f64, f64) {
        (prediction - self.half_width, prediction + self.half_width)
    }

    /// Whether the band around `prediction` covers `actual` (`|actual − ŷ| ≤ q`).
    pub fn covers(&self, prediction: f64, actual: f64) -> bool {
        (actual - prediction).abs() <= self.half_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_width_is_the_conformal_order_statistic() {
        // Residuals 1..=100 (all positive → scores == 1..=100). At level 0.9,
        // rank = ceil(101 * 0.9) = 91, so q = the 91st smallest score = 91.
        let residuals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let band = SplitConformal::fit(&residuals, 0.9).unwrap();
        assert!((band.half_width() - 91.0).abs() < 1e-12);
        assert!((band.width() - 182.0).abs() < 1e-12);
        assert_eq!(band.calibration_count(), 100);
    }

    #[test]
    fn covers_respects_the_half_width() {
        let residuals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let band = SplitConformal::fit(&residuals, 0.9).unwrap();
        assert!(band.covers(0.0, 91.0));
        assert!(band.covers(10.0, 10.0 - 91.0));
        assert!(!band.covers(0.0, 91.5));
        assert_eq!(band.interval(5.0), (5.0 - 91.0, 5.0 + 91.0));
    }

    #[test]
    fn in_sample_coverage_meets_the_level() {
        // The band's own guarantee: at least a `level` fraction of the calibration
        // residuals fall within q (treating each as an actual around prediction 0).
        let residuals: Vec<f64> = (0..200).map(|i| ((i * 37) % 101) as f64 - 50.0).collect();
        for &level in &[0.8, 0.9, 0.95]
        {
            let band = SplitConformal::fit(&residuals, level).unwrap();
            let covered = residuals.iter().filter(|&&r| band.covers(0.0, r)).count();
            let empirical = covered as f64 / residuals.len() as f64;
            assert!(
                empirical >= level,
                "empirical {empirical} below nominal {level}"
            );
        }
    }

    #[test]
    fn heavier_calibration_tail_widens_the_band() {
        let light: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let heavy: Vec<f64> = (1..=100)
            .map(|i| if i == 100 { 1000.0 } else { i as f64 })
            .collect();
        let light_band = SplitConformal::fit(&light, 0.99).unwrap();
        let heavy_band = SplitConformal::fit(&heavy, 0.99).unwrap();
        assert!(heavy_band.half_width() >= light_band.half_width());
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(
            SplitConformal::fit(&[], 0.9),
            Err(ConformalError::EmptyCalibration)
        );
        assert_eq!(
            SplitConformal::fit(&[1.0, 2.0], 1.0),
            Err(ConformalError::InvalidLevel { level: 1.0 })
        );
        assert_eq!(
            SplitConformal::fit(&[1.0, f64::NAN], 0.9),
            Err(ConformalError::NonFiniteResidual)
        );
    }

    #[test]
    fn too_few_calibration_points_is_a_typed_error() {
        // n = 5, level 0.9 → rank = ceil(6 * 0.9) = 6 > 5 → no finite band.
        assert_eq!(
            SplitConformal::fit(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.9),
            Err(ConformalError::CalibrationTooSmall {
                count: 5,
                level: 0.9
            })
        );
        // Nine points suffice at 0.9: rank = ceil(10 * 0.9) = 9 = n.
        assert!(SplitConformal::fit(&[1.0; 9], 0.9).is_ok());
    }
}
