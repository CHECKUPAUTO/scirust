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
    /// The conformalized-quantile calibration slices had unequal lengths.
    MismatchedCalibration {
        /// Lower-prediction slice length.
        lower: usize,
        /// Upper-prediction slice length.
        upper: usize,
        /// Actual-value slice length.
        actual: usize,
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
            Self::MismatchedCalibration {
                lower,
                upper,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "conformalized-quantile calibration slices differ in length: \
lower {lower}, upper {upper}, actual {actual}"
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

/// A conformalized-quantile-regression adjustment (Romano, Patterson & Candès, 2019).
///
/// Split conformal makes a *constant*-width band; native quantile regression makes
/// an *adaptive* band with no coverage guarantee. CQR unites them: fit lower/upper
/// quantile predictors, then on a calibration set compute the nonconformity score
/// `Eᵢ = max(q_lo(xᵢ) − yᵢ, yᵢ − q_hi(xᵢ))` — how far `yᵢ` falls outside the native
/// interval (negative when comfortably inside). The single conformal offset `Q` is
/// the `⌈(n+1)·level⌉`-th smallest `Eᵢ`, and the adjusted interval
/// `[q_lo(x) − Q, q_hi(x) + Q]` has finite-sample coverage `≥ level` for **any**
/// distribution, **while keeping the adaptive shape** of the quantile band. `Q` may
/// be negative — a too-wide native interval is *tightened*. No RNG.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalizedQuantile {
    offset: f64,
    level: f64,
    calibration_count: usize,
}

impl ConformalizedQuantile {
    /// Fits the offset from calibration triples: lower/upper quantile predictions
    /// and the observed targets, all aligned and of equal length.
    ///
    /// # Errors
    ///
    /// [`ConformalError::MismatchedCalibration`] for unequal slice lengths;
    /// [`ConformalError::EmptyCalibration`] / [`ConformalError::InvalidLevel`] /
    /// [`ConformalError::NonFiniteResidual`] for malformed input; and
    /// [`ConformalError::CalibrationTooSmall`] when `⌈(n+1)·level⌉ > n`.
    pub fn fit(
        lower: &[f64],
        upper: &[f64],
        actual: &[f64],
        level: f64,
    ) -> Result<Self, ConformalError> {
        if lower.len() != upper.len() || lower.len() != actual.len()
        {
            return Err(ConformalError::MismatchedCalibration {
                lower: lower.len(),
                upper: upper.len(),
                actual: actual.len(),
            });
        }

        if lower.is_empty()
        {
            return Err(ConformalError::EmptyCalibration);
        }

        if !(level > 0.0 && level < 1.0)
        {
            return Err(ConformalError::InvalidLevel { level });
        }

        let mut scores: Vec<f64> = Vec::with_capacity(lower.len());

        for ((&low, &high), &target) in lower.iter().zip(upper).zip(actual)
        {
            if !low.is_finite() || !high.is_finite() || !target.is_finite()
            {
                return Err(ConformalError::NonFiniteResidual);
            }

            scores.push((low - target).max(target - high));
        }

        scores.sort_by(f64::total_cmp);

        let n = scores.len();
        let rank = (((n + 1) as f64) * level).ceil() as usize;

        if rank > n
        {
            return Err(ConformalError::CalibrationTooSmall { count: n, level });
        }

        Ok(Self {
            offset: scores[rank - 1],
            level,
            calibration_count: n,
        })
    }

    /// The conformal offset `Q` (added to `q_hi`, subtracted from `q_lo`; may be
    /// negative).
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// The nominal coverage level the offset was fitted at.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Calibration points used.
    pub fn calibration_count(&self) -> usize {
        self.calibration_count
    }

    /// The adjusted interval `[q_lo − Q, q_hi + Q]` for one prediction pair.
    pub fn interval(&self, lower_prediction: f64, upper_prediction: f64) -> (f64, f64) {
        (
            lower_prediction - self.offset,
            upper_prediction + self.offset,
        )
    }

    /// Whether the adjusted interval covers `actual`.
    pub fn covers(&self, lower_prediction: f64, upper_prediction: f64, actual: f64) -> bool {
        let (low, high) = self.interval(lower_prediction, upper_prediction);
        low <= actual && actual <= high
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

    #[test]
    fn cqr_guarantees_in_sample_coverage() {
        // Native interval [0, 1] for every point; actuals spread well outside it.
        let actual: Vec<f64> = (0..100)
            .map(|i| ((i * 7) % 21) as f64 * 0.25 - 2.0)
            .collect();
        let lower = vec![0.0; 100];
        let upper = vec![1.0; 100];
        for &level in &[0.8, 0.9, 0.95]
        {
            let cqr = ConformalizedQuantile::fit(&lower, &upper, &actual, level).unwrap();
            let covered = (0..100)
                .filter(|&i| cqr.covers(lower[i], upper[i], actual[i]))
                .count();
            let empirical = covered as f64 / 100.0;
            assert!(empirical >= level, "empirical {empirical} below {level}");
        }
    }

    #[test]
    fn cqr_offset_tightens_a_too_wide_interval() {
        // Native interval [-10, 10] is far too wide; actuals sit near zero, so the
        // conformal offset is negative and the adjusted interval is narrower.
        let actual: Vec<f64> = (0..50).map(|i| (i % 3) as f64 - 1.0).collect();
        let lower = vec![-10.0; 50];
        let upper = vec![10.0; 50];
        let cqr = ConformalizedQuantile::fit(&lower, &upper, &actual, 0.9).unwrap();
        assert!(
            cqr.offset() < 0.0,
            "offset {} should be negative",
            cqr.offset()
        );
        let (low, high) = cqr.interval(-10.0, 10.0);
        assert!(
            high - low < 20.0,
            "adjusted width {} not tighter",
            high - low
        );
    }

    #[test]
    fn cqr_rejects_mismatched_lengths() {
        assert_eq!(
            ConformalizedQuantile::fit(&[1.0], &[1.0, 2.0], &[1.0], 0.9),
            Err(ConformalError::MismatchedCalibration {
                lower: 1,
                upper: 2,
                actual: 1
            })
        );
    }
}
