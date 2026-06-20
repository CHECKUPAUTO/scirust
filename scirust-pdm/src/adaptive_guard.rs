//! Streaming anomaly guard with drift adaptation (Adaptive Conformal Inference).
//!
//! A machine's "normal" baseline drifts — warm-up, load changes, seasonal
//! temperature. A *static* conformal threshold then over- or under-alarms.
//! [`AdaptiveGuard`] wraps [`scirust_core::nn::conformal::AdaptiveConformal`]:
//! it keeps a sliding window of recent scores as the calibration set and adapts
//! the effective level `αₜ` online, so the long-run false-alarm rate stays near
//! `α` **through** distribution shifts. Deterministic (fixed-order `f32`).

use crate::conformal_guard::GuardVerdict;
use scirust_core::nn::conformal::conformal_quantile;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Online, drift-adaptive anomaly guard (Adaptive Conformal Inference).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveGuard {
    target_alpha: f32,
    gamma: f32,
    alpha_t: f32,
    window: VecDeque<f32>,
    window_size: usize,
}

impl AdaptiveGuard {
    /// Target false-alarm rate `target_alpha`, adaptation rate `gamma`, and the
    /// sliding-window length used as the rolling calibration set.
    pub fn new(target_alpha: f32, gamma: f32, window_size: usize) -> Self {
        assert!(
            target_alpha > 0.0 && target_alpha < 1.0,
            "target_alpha in (0,1)"
        );
        assert!(gamma > 0.0, "gamma must be positive");
        Self {
            target_alpha,
            gamma,
            alpha_t: target_alpha,
            window: VecDeque::with_capacity(window_size + 1),
            window_size: window_size.max(1),
        }
    }

    /// Current effective miscoverage level `αₜ`.
    pub fn alpha(&self) -> f32 {
        self.alpha_t
    }

    /// Process the next anomaly score: classify it against the current rolling
    /// envelope (covered ⇒ Normal), adapt `αₜ` by the ACI update, and slide the
    /// window.
    pub fn check(&mut self, score: f32) -> GuardVerdict {
        let scores: Vec<f32> = self.window.iter().copied().collect();
        let q = if self.alpha_t <= 0.0
        {
            f32::INFINITY
        }
        else if self.alpha_t >= 1.0
        {
            f32::NEG_INFINITY
        }
        else
        {
            conformal_quantile(&scores, self.alpha_t)
        };
        let covered = score <= q;
        let err = if covered { 0.0 } else { 1.0 };
        self.alpha_t = (self.alpha_t + self.gamma * (self.target_alpha - err)).clamp(0.0, 1.0);

        self.window.push_back(score);
        if self.window.len() > self.window_size
        {
            self.window.pop_front();
        }
        if covered
        {
            GuardVerdict::Normal
        }
        else
        {
            GuardVerdict::Anomaly
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::conformal::conformal_quantile;

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn u01(&mut self) -> f32 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 + 0.5) / ((1u64 << 24) as f32)
        }
        fn normal(&mut self, mean: f32, sd: f32) -> f32 {
            let (u1, u2) = (self.u01().max(1e-6), self.u01());
            mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos()
        }
    }

    #[test]
    fn holds_coverage_under_drift_where_static_fails() {
        let target = 0.1_f32;
        let mut guard = AdaptiveGuard::new(target, 0.02, 300);
        let mut rng = Rng::new(0xD21F7);

        // Build a static threshold from the first 300 (pre-drift) scores.
        let mut warmup = Vec::new();
        let t = 5000;
        let mut adaptive_alarms = 0usize;
        let mut drifted_scores = Vec::new();

        for k in 0..t
        {
            // Normal baseline drifts upward over the run (mean 1 -> ~4).
            let mean = 1.0 + 3.0 * (k as f32 / t as f32);
            let score = rng.normal(mean, 0.3).abs();
            if k < 300
            {
                warmup.push(score);
            }
            if guard.check(score) == GuardVerdict::Anomaly
            {
                adaptive_alarms += 1;
            }
            if k >= t / 2
            {
                drifted_scores.push(score);
            }
        }
        let adaptive_far = adaptive_alarms as f64 / t as f64;
        // ACI keeps the long-run false-alarm rate near the target.
        assert!(
            (adaptive_far - target as f64).abs() < 0.05,
            "adaptive FAR {adaptive_far} vs target {target}"
        );

        // A static threshold frozen on the warmup over-alarms badly once the
        // baseline has drifted up.
        let static_thr = conformal_quantile(&warmup, target);
        let static_alarms = drifted_scores.iter().filter(|&&s| s > static_thr).count();
        let static_far = static_alarms as f64 / drifted_scores.len() as f64;
        assert!(
            static_far > 3.0 * target as f64,
            "static FAR {static_far} should blow up under drift"
        );
    }
}
