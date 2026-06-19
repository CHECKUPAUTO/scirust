//! Distribution-free anomaly guard for predictive maintenance.
//!
//! The fault detectors in [`crate::detectors`] and the [`crate::health`] index
//! emit continuous *anomaly scores* (vibration RMS, band power, fault
//! confidence, inverted health …). Turning a score into an alarm needs a
//! threshold — and a fixed hand-tuned threshold has no guarantee on its
//! false-alarm rate.
//!
//! [`ConformalGuard`] calibrates that threshold with **split conformal
//! prediction**: given a set of NORMAL-condition scores, it picks the
//! `⌈(n+1)(1−α)⌉`-th smallest as the threshold, so that the probability a
//! future *normal* sample is flagged (a false alarm) is bounded by `α`, with
//! **no distributional assumption** and a finite-sample guarantee. Convention:
//! higher score = more anomalous.
//!
//! This reuses the audited [`scirust_core::nn::conformal::conformal_quantile`]
//! primitive — the same machinery as the model-side conformal predictors — so
//! the industrial alarm and the ML uncertainty story share one guarantee.

use scirust_core::nn::conformal::conformal_quantile;
use serde::{Deserialize, Serialize};

/// Verdict for a single anomaly score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardVerdict {
    /// Score is within the calibrated normal envelope.
    Normal,
    /// Score exceeds the conformal threshold — raise a maintenance alarm.
    Anomaly,
}

/// Split-conformal anomaly guard calibrated on normal-condition scores.
///
/// Guarantees a marginal false-alarm rate `≤ α` on future normal samples drawn
/// exchangeably with the calibration set, independent of the score distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalGuard {
    threshold: f32,
    alpha: f32,
    n_calib: usize,
}

impl ConformalGuard {
    /// Calibrate on NORMAL-condition anomaly scores (higher = more anomalous).
    ///
    /// `alpha` is the target false-alarm rate, in `(0, 1)`. With too few
    /// calibration points to guarantee the level, the threshold is `+∞`
    /// (nothing is ever flagged — a safe, alarm-free default).
    pub fn calibrate(normal_scores: &[f32], alpha: f32) -> Self {
        Self {
            threshold: conformal_quantile(normal_scores, alpha),
            alpha,
            n_calib: normal_scores.len(),
        }
    }

    /// The conformal threshold; scores strictly greater are flagged.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Target (and guaranteed upper bound on the) false-alarm rate.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Number of calibration scores used.
    pub fn n_calib(&self) -> usize {
        self.n_calib
    }

    /// Classify a single anomaly score.
    pub fn check(&self, score: f32) -> GuardVerdict {
        if score > self.threshold
        {
            GuardVerdict::Anomaly
        }
        else
        {
            GuardVerdict::Normal
        }
    }

    /// Empirical false-alarm rate on a held-out NORMAL set: the fraction of
    /// scores flagged as anomalies. By the conformal guarantee this is `≈ α`
    /// (in expectation `≤ α`).
    pub fn false_alarm_rate(&self, normal_scores: &[f32]) -> f32 {
        if normal_scores.is_empty()
        {
            return 0.0;
        }
        let flagged = normal_scores
            .iter()
            .filter(|&&s| self.check(s) == GuardVerdict::Anomaly)
            .count();
        flagged as f32 / normal_scores.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform-[0,1) stream (splitmix64) — the conformal
    /// guarantee is distribution-free, so a uniform draw is a valid and
    /// tight probe of the false-alarm bound.
    struct Uniform01 {
        state: u64,
    }
    impl Uniform01 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next(&mut self) -> f32 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // top 24 bits -> [0,1)
            ((z >> 40) as f32) / ((1u64 << 24) as f32)
        }
        fn sample(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    #[test]
    fn false_alarm_rate_is_bounded_by_alpha() {
        let mut rng = Uniform01::new(0xA11CE);
        let calib = rng.sample(2000);
        let held_out_normal = rng.sample(8000);
        let alpha = 0.1;

        let guard = ConformalGuard::calibrate(&calib, alpha);
        let far = guard.false_alarm_rate(&held_out_normal);

        // With 8000 held-out points the empirical FAR concentrates tightly
        // around alpha (~0.1); the guarantee bounds it near/below alpha.
        assert!(
            (0.07..=0.13).contains(&far),
            "FAR {far} not ~alpha {alpha} (threshold {})",
            guard.threshold()
        );
    }

    #[test]
    fn flags_a_developing_fault_but_not_the_baseline() {
        // Normal bearing vibration RMS (mm/s) around a baseline, with noise.
        let mut rng = Uniform01::new(0xBEA21);
        let normal: Vec<f32> = rng.sample(1000).iter().map(|u| 1.0 + 0.4 * u).collect(); // [1.0, 1.4]
        let guard = ConformalGuard::calibrate(&normal, 0.05);

        // A healthy reading near the baseline is not flagged.
        assert_eq!(guard.check(1.15), GuardVerdict::Normal);
        // A developing outer-race fault drives RMS well past the envelope.
        assert_eq!(guard.check(3.5), GuardVerdict::Anomaly);
        assert_eq!(guard.check(7.0), GuardVerdict::Anomaly);
    }

    #[test]
    fn too_few_points_never_false_alarms() {
        // n=5, alpha=0.01 -> ceil(6*0.99)=6 > 5 -> threshold = +inf.
        let guard = ConformalGuard::calibrate(&[0.1, 0.2, 0.3, 0.4, 0.5], 0.01);
        assert!(guard.threshold().is_infinite());
        assert_eq!(guard.check(1e9), GuardVerdict::Normal);
        assert_eq!(guard.false_alarm_rate(&[10.0, 20.0, 30.0]), 0.0);
    }
}
