//! Conditional and temporal conformal prediction (phase 4E.6).
//!
//! [`SplitConformal`](crate::conformal::SplitConformal) gives a *marginal*
//! coverage guarantee under *exchangeability*. Two industrial realities break
//! each assumption, and this module answers each honestly:
//!
//! - **Heterogeneous sub-populations** — a marginal band can hold 90 % *overall*
//!   while badly under-covering a hard sub-group and over-covering an easy one.
//!   [`MondrianConformal`] calibrates a separate band per group, so coverage holds
//!   *conditional on group membership* (Vovk's Mondrian conformal). This is
//!   coverage conditional on a **discrete grouping**, which is achievable
//!   distribution-free; full coverage conditional on a continuous `x` is provably
//!   impossible distribution-free (Vovk 2012; Lei & Wasserman 2014; Barber et al.
//!   2020) and is **not** claimed.
//!
//! - **Distribution drift over time** — once the stream is non-exchangeable a
//!   fixed conformal band silently loses coverage. [`AdaptiveConformal`]
//!   (Adaptive Conformal Inference; Gibbs & Candès 2021) adjusts the working
//!   miscoverage `αₜ` online from realized coverage errors. It guarantees
//!   **long-run (time-average) coverage → `level`** for an *arbitrary, even
//!   adversarial* sequence — no exchangeability needed. It does **not** guarantee
//!   per-step or finite-sample marginal coverage, and the band can widen sharply
//!   (or, when `αₜ` saturates, degenerate); those costs are surfaced, not hidden.
//!
//! Determinism throughout: per-group sorts and an online recursion, `total_cmp`
//! ordering, no RNG.

use core::fmt;
use std::collections::VecDeque;

/// A clamp keeping the working miscoverage strictly inside `(0, 1)` so the
/// derived coverage level is always a valid quantile target.
const ALPHA_FLOOR: f64 = 1.0e-6;

/// Typed errors for the conditional / temporal conformal calibrators.
#[derive(Clone, Debug, PartialEq)]
pub enum ConditionalConformalError {
    /// The calibration set was empty.
    EmptyCalibration,
    /// A level or target coverage was not in the open interval `(0, 1)`.
    InvalidLevel {
        /// The rejected level.
        level: f64,
    },
    /// A calibration residual or score was `NaN` or `±∞`.
    NonFiniteScore,
    /// The group-key and residual slices had different lengths.
    LengthMismatch {
        /// Group-key count.
        keys: usize,
        /// Residual count.
        residuals: usize,
    },
    /// The learning rate was not finite and positive.
    InvalidLearningRate {
        /// The rejected rate.
        rate: f64,
    },
    /// The rolling-window capacity was zero.
    ZeroCapacity,
}

impl fmt::Display for ConditionalConformalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyCalibration => formatter.write_str("the calibration set is empty"),
            Self::InvalidLevel { level } =>
            {
                write!(
                    formatter,
                    "level {level} must lie in the open interval (0, 1)"
                )
            },
            Self::NonFiniteScore => formatter.write_str("a calibration score is not finite"),
            Self::LengthMismatch { keys, residuals } => write!(
                formatter,
                "group-key count {keys} does not match residual count {residuals}"
            ),
            Self::InvalidLearningRate { rate } =>
            {
                write!(
                    formatter,
                    "learning rate {rate} must be finite and positive"
                )
            },
            Self::ZeroCapacity =>
            {
                formatter.write_str("the rolling-window capacity must be positive")
            },
        }
    }
}

impl std::error::Error for ConditionalConformalError {}

/// The finite-sample conformal half-width of a set of absolute scores at a
/// coverage `level`: the `⌈(n+1)·level⌉`-th smallest score, or the maximum score
/// when that rank exceeds `n` (a conservative, still-finite fallback). `scores`
/// must be non-empty and already validated finite.
fn conformal_quantile(scores: &mut [f64], level: f64) -> f64 {
    scores.sort_by(f64::total_cmp);
    let n = scores.len();
    let rank = (((n + 1) as f64) * level).ceil() as usize;
    scores[rank.clamp(1, n) - 1]
}

/// One group's calibrated band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupBand {
    /// The group key.
    pub key: u64,
    /// The symmetric half-width for this group.
    pub half_width: f64,
    /// Calibration points in this group.
    pub calibration_count: usize,
    /// Whether this group had enough points for its own finite-sample guarantee
    /// (`false` means it borrowed the pooled marginal band).
    pub conditionally_valid: bool,
}

/// Group-conditional (Mondrian) split-conformal bands.
#[derive(Clone, Debug, PartialEq)]
pub struct MondrianConformal {
    level: f64,
    bands: Vec<GroupBand>,
    pooled_half_width: f64,
}

impl MondrianConformal {
    /// Fits one band per group from aligned `group_keys` and calibration
    /// `residuals` (`yᵢ − ŷᵢ`). A group whose own calibration count is too small
    /// for a finite band at `level` borrows the pooled marginal band and is
    /// flagged `conditionally_valid = false`.
    ///
    /// # Errors
    ///
    /// [`ConditionalConformalError`] on length mismatch, empty input, an
    /// out-of-range level, or a non-finite residual.
    pub fn fit(
        group_keys: &[u64],
        residuals: &[f64],
        level: f64,
    ) -> Result<Self, ConditionalConformalError> {
        if group_keys.len() != residuals.len()
        {
            return Err(ConditionalConformalError::LengthMismatch {
                keys: group_keys.len(),
                residuals: residuals.len(),
            });
        }
        if residuals.is_empty()
        {
            return Err(ConditionalConformalError::EmptyCalibration);
        }
        if !(level > 0.0 && level < 1.0)
        {
            return Err(ConditionalConformalError::InvalidLevel { level });
        }
        for &residual in residuals
        {
            if !residual.is_finite()
            {
                return Err(ConditionalConformalError::NonFiniteScore);
            }
        }

        // Pooled (marginal) band over every residual — the fallback for small groups.
        let mut pooled: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        let pooled_half_width = conformal_quantile(&mut pooled, level);

        // Collect absolute scores per group in canonical key order.
        let mut keys_sorted: Vec<u64> = group_keys.to_vec();
        keys_sorted.sort_unstable();
        keys_sorted.dedup();

        let smallest_sufficient = smallest_sufficient_count(level);

        let mut bands = Vec::with_capacity(keys_sorted.len());
        for key in keys_sorted
        {
            let mut scores: Vec<f64> = group_keys
                .iter()
                .zip(residuals)
                .filter(|(k, _)| **k == key)
                .map(|(_, r)| r.abs())
                .collect();
            let count = scores.len();
            if count >= smallest_sufficient
            {
                let half_width = conformal_quantile(&mut scores, level);
                bands.push(GroupBand {
                    key,
                    half_width,
                    calibration_count: count,
                    conditionally_valid: true,
                });
            }
            else
            {
                bands.push(GroupBand {
                    key,
                    half_width: pooled_half_width,
                    calibration_count: count,
                    conditionally_valid: false,
                });
            }
        }

        Ok(Self {
            level,
            bands,
            pooled_half_width,
        })
    }

    /// The calibrated bands, in ascending key order.
    pub fn bands(&self) -> &[GroupBand] {
        &self.bands
    }

    /// The nominal coverage level.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// The pooled marginal half-width (used for unseen or too-small groups).
    pub fn pooled_half_width(&self) -> f64 {
        self.pooled_half_width
    }

    /// The half-width for `group`, or the pooled marginal half-width when the
    /// group was never seen in calibration.
    pub fn half_width(&self, group: u64) -> f64 {
        self.bands
            .iter()
            .find(|band| band.key == group)
            .map_or(self.pooled_half_width, |band| band.half_width)
    }

    /// The interval `[ŷ − q, ŷ + q]` for `group` around a point prediction.
    pub fn interval(&self, group: u64, prediction: f64) -> (f64, f64) {
        let half_width = self.half_width(group);
        (prediction - half_width, prediction + half_width)
    }

    /// Whether the `group` band around `prediction` covers `actual`.
    pub fn covers(&self, group: u64, prediction: f64, actual: f64) -> bool {
        (actual - prediction).abs() <= self.half_width(group)
    }
}

/// The smallest calibration count for which `⌈(n+1)·level⌉ ≤ n` (a finite
/// conformal band exists at `level`).
fn smallest_sufficient_count(level: f64) -> usize {
    // ceil((n+1) level) <= n  <=>  n >= level / (1 - level).
    (level / (1.0 - level)).ceil() as usize
}

/// Adaptive Conformal Inference: an online conformal calibrator for drifting
/// (non-exchangeable) streams (Gibbs & Candès 2021).
///
/// At each step the working miscoverage `αₜ` sets the band radius as the
/// `1 − αₜ` conformal quantile of a rolling window of recent absolute scores.
/// After the true score is observed, `αₜ` moves against the realized coverage
/// error: `α_{t+1} = clamp(αₜ + γ · (α − errₜ))`, where `errₜ = 1` iff the point
/// fell outside the band. A miss widens the next band; a hit narrows it. The
/// time-average coverage converges to `level` for any sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveConformal {
    target_alpha: f64,
    gamma: f64,
    alpha_t: f64,
    window: VecDeque<f64>,
    capacity: usize,
    steps: usize,
    miscovered: usize,
}

impl AdaptiveConformal {
    /// Creates a calibrator at nominal coverage `level`, learning rate `gamma`,
    /// and rolling-window `capacity` (most recent scores kept).
    ///
    /// # Errors
    ///
    /// [`ConditionalConformalError`] on an out-of-range level, a non-positive
    /// learning rate, or zero capacity.
    pub fn new(level: f64, gamma: f64, capacity: usize) -> Result<Self, ConditionalConformalError> {
        if !(level > 0.0 && level < 1.0)
        {
            return Err(ConditionalConformalError::InvalidLevel { level });
        }
        if !(gamma.is_finite() && gamma > 0.0)
        {
            return Err(ConditionalConformalError::InvalidLearningRate { rate: gamma });
        }
        if capacity == 0
        {
            return Err(ConditionalConformalError::ZeroCapacity);
        }
        let target_alpha = 1.0 - level;
        Ok(Self {
            target_alpha,
            gamma,
            alpha_t: target_alpha,
            window: VecDeque::with_capacity(capacity),
            capacity,
            steps: 0,
            miscovered: 0,
        })
    }

    /// The current band half-width: the `1 − αₜ` conformal quantile of the
    /// window. `f64::INFINITY` before any score has been seen (no data yet).
    pub fn radius(&self) -> f64 {
        if self.window.is_empty()
        {
            return f64::INFINITY;
        }
        let mut scores: Vec<f64> = self.window.iter().copied().collect();
        conformal_quantile(&mut scores, 1.0 - self.alpha_t)
    }

    /// Records the observed absolute nonconformity score for the current step:
    /// updates `αₜ` from the coverage error at the *current* radius, then appends
    /// the score to the rolling window (evicting the oldest if full).
    ///
    /// # Errors
    ///
    /// [`ConditionalConformalError::NonFiniteScore`] if `score` is not finite.
    pub fn observe(&mut self, score: f64) -> Result<(), ConditionalConformalError> {
        if !score.is_finite()
        {
            return Err(ConditionalConformalError::NonFiniteScore);
        }
        let radius = self.radius();
        let miscovered = score > radius;
        let error = f64::from(u8::from(miscovered));
        self.alpha_t = (self.alpha_t + self.gamma * (self.target_alpha - error))
            .clamp(ALPHA_FLOOR, 1.0 - ALPHA_FLOOR);

        if self.window.len() == self.capacity
        {
            self.window.pop_front();
        }
        self.window.push_back(score);

        self.steps += 1;
        if miscovered
        {
            self.miscovered += 1;
        }
        Ok(())
    }

    /// The current working miscoverage `αₜ`.
    pub fn current_alpha(&self) -> f64 {
        self.alpha_t
    }

    /// The realized time-average coverage so far (`1 − miscovered / steps`);
    /// `1.0` before any step.
    pub fn realized_coverage(&self) -> f64 {
        if self.steps == 0
        {
            return 1.0;
        }
        1.0 - self.miscovered as f64 / self.steps as f64
    }

    /// Steps observed so far.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The nominal coverage level.
    pub fn level(&self) -> f64 {
        1.0 - self.target_alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mondrian_calibrates_each_group_separately() {
        // Group 0: residuals ±1; group 1: residuals ±10. A marginal band would be
        // wide for group 0 and (relatively) tight for group 1; Mondrian separates.
        let mut keys = Vec::new();
        let mut residuals = Vec::new();
        for i in 0..100
        {
            keys.push(0);
            residuals.push(((i % 3) as f64 - 1.0) * 1.0); // -1, 0, 1
        }
        for i in 0..100
        {
            keys.push(1);
            residuals.push(((i % 3) as f64 - 1.0) * 10.0); // -10, 0, 10
        }
        let mondrian = MondrianConformal::fit(&keys, &residuals, 0.9).unwrap();
        assert_eq!(mondrian.bands().len(), 2);
        let g0 = mondrian.half_width(0);
        let g1 = mondrian.half_width(1);
        assert!(
            g0 < g1,
            "group 0 band {g0} should be tighter than group 1 {g1}"
        );
        assert!(mondrian.bands().iter().all(|b| b.conditionally_valid));
    }

    #[test]
    fn mondrian_beats_marginal_on_per_group_coverage() {
        // Two groups with very different spreads. The marginal band under-covers
        // the wide group; Mondrian restores per-group coverage.
        let mut keys = Vec::new();
        let mut residuals = Vec::new();
        for i in 0..200
        {
            keys.push(0u64);
            residuals.push(((i * 7) % 5) as f64 - 2.0); // small spread [-2, 2]
        }
        for i in 0..200
        {
            keys.push(1u64);
            residuals.push((((i * 11) % 41) as f64 - 20.0) * 2.0); // large spread
        }

        let mondrian = MondrianConformal::fit(&keys, &residuals, 0.9).unwrap();

        // Per-group empirical coverage of the Mondrian band (prediction 0).
        for group in [0u64, 1u64]
        {
            let covered = keys
                .iter()
                .zip(&residuals)
                .filter(|(k, _)| **k == group)
                .filter(|(_, r)| mondrian.covers(group, 0.0, **r))
                .count();
            let total = keys.iter().filter(|&&k| k == group).count();
            let empirical = covered as f64 / total as f64;
            assert!(
                empirical >= 0.9,
                "group {group} coverage {empirical} below nominal 0.9"
            );
        }
    }

    #[test]
    fn small_groups_fall_back_to_the_pooled_band() {
        // Group 7 has only two points — too few for a finite band at 0.9
        // (needs ceil(0.9 / 0.1) = 9). It must borrow the pooled band.
        let mut keys: Vec<u64> = vec![0; 50];
        let mut residuals: Vec<f64> = (0..50).map(|i| (i % 7) as f64 - 3.0).collect();
        keys.push(7);
        residuals.push(2.0);
        keys.push(7);
        residuals.push(-2.0);

        let mondrian = MondrianConformal::fit(&keys, &residuals, 0.9).unwrap();
        let small = mondrian.bands().iter().find(|b| b.key == 7).unwrap();
        assert!(!small.conditionally_valid);
        assert_eq!(small.half_width, mondrian.pooled_half_width());
    }

    #[test]
    fn unseen_group_uses_the_pooled_band() {
        let keys = vec![0u64; 20];
        let residuals: Vec<f64> = (0..20).map(|i| i as f64 - 10.0).collect();
        let mondrian = MondrianConformal::fit(&keys, &residuals, 0.9).unwrap();
        // Group 99 never appeared → pooled band.
        assert_eq!(mondrian.half_width(99), mondrian.pooled_half_width());
    }

    #[test]
    fn mondrian_rejects_malformed_input() {
        assert_eq!(
            MondrianConformal::fit(&[0, 1], &[1.0], 0.9),
            Err(ConditionalConformalError::LengthMismatch {
                keys: 2,
                residuals: 1
            })
        );
        // Equal-length empty slices pass the length check and hit the emptiness one.
        assert_eq!(
            MondrianConformal::fit(&[], &[], 0.9),
            Err(ConditionalConformalError::EmptyCalibration)
        );
        assert_eq!(
            MondrianConformal::fit(&[0], &[f64::NAN], 0.9),
            Err(ConditionalConformalError::NonFiniteScore)
        );
        assert_eq!(
            MondrianConformal::fit(&[0, 0], &[1.0, 2.0], 1.5),
            Err(ConditionalConformalError::InvalidLevel { level: 1.5 })
        );
    }

    #[test]
    fn aci_recovers_coverage_under_drift() {
        // A scale-drifting stream: the nonconformity score grows over time. A
        // fixed band calibrated early would under-cover late; ACI adapts and holds
        // long-run coverage near the nominal 0.9.
        let mut aci = AdaptiveConformal::new(0.9, 0.05, 200).unwrap();
        // Warm up on early (small) scores.
        for i in 0..200
        {
            aci.observe(((i * 13) % 17) as f64 * 0.1).unwrap();
        }
        // Drift: scores grow linearly with time.
        for t in 0..2000
        {
            let scale = 1.0 + t as f64 * 0.01;
            let score = (((t * 29) % 23) as f64 * 0.1) * scale;
            aci.observe(score).unwrap();
        }
        let coverage = aci.realized_coverage();
        assert!(
            coverage >= 0.85,
            "ACI long-run coverage {coverage} fell well below nominal 0.9 under drift"
        );
    }

    #[test]
    fn aci_alpha_moves_the_right_way() {
        // Feed scores that are always inside a huge window band → never miscovered
        // → alpha_t should climb toward its ceiling (bands narrow).
        let mut aci = AdaptiveConformal::new(0.9, 0.1, 50).unwrap();
        aci.observe(1.0).unwrap(); // seed
        let start = aci.current_alpha();
        for _ in 0..100
        {
            aci.observe(0.5).unwrap(); // always well inside → covered
        }
        assert!(
            aci.current_alpha() > start,
            "alpha_t {} should rise when everything is covered",
            aci.current_alpha()
        );
        assert!(aci.realized_coverage() > 0.9);
    }

    #[test]
    fn aci_is_deterministic() {
        let run = || {
            let mut aci = AdaptiveConformal::new(0.9, 0.05, 100).unwrap();
            for i in 0..500
            {
                aci.observe(((i * 31) % 19) as f64 * 0.2).unwrap();
            }
            (aci.current_alpha(), aci.realized_coverage(), aci.radius())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn aci_rejects_malformed_construction_and_scores() {
        assert_eq!(
            AdaptiveConformal::new(1.0, 0.1, 10),
            Err(ConditionalConformalError::InvalidLevel { level: 1.0 })
        );
        assert_eq!(
            AdaptiveConformal::new(0.9, 0.0, 10),
            Err(ConditionalConformalError::InvalidLearningRate { rate: 0.0 })
        );
        assert_eq!(
            AdaptiveConformal::new(0.9, 0.1, 0),
            Err(ConditionalConformalError::ZeroCapacity)
        );
        let mut aci = AdaptiveConformal::new(0.9, 0.1, 10).unwrap();
        assert_eq!(
            aci.observe(f64::INFINITY),
            Err(ConditionalConformalError::NonFiniteScore)
        );
    }

    #[test]
    fn empty_window_radius_is_infinite() {
        let aci = AdaptiveConformal::new(0.9, 0.1, 10).unwrap();
        assert_eq!(aci.radius(), f64::INFINITY);
        assert_eq!(aci.realized_coverage(), 1.0);
        assert_eq!(aci.steps(), 0);
    }
}
