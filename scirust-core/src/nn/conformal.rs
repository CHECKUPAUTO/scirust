//! **Split conformal prediction** (Vovk et al.; Angelopoulos & Bates,
//! *A Gentle Introduction to Conformal Prediction*, 2021).
//!
//! Conformal prediction turns *any* point predictor into one that emits
//! **prediction sets/intervals with a distribution-free coverage guarantee**:
//! for a chosen miscoverage `α`, the set contains the truth with probability
//! `≥ 1 − α`, with **no assumption** on the data distribution or the model —
//! only that calibration and test points are exchangeable. This is exactly
//! scirust's "certifiable AI" thesis: a guarantee you can *test* (empirical
//! coverage) rather than hope for.
//!
//! Method (split/inductive conformal): on a held-out calibration set compute a
//! nonconformity score per point, take the finite-sample quantile
//! `q̂ = ⌈(n+1)(1−α)⌉`-th smallest score, then
//! - **regression**: interval `[ŷ − q̂, ŷ + q̂]` from absolute-residual scores;
//! - **classification** (LAC/threshold): set `{c : p_c ≥ 1 − q̂}` from
//!   `1 − p_true` scores.
//!
//! Everything is pure, deterministic arithmetic; the coverage guarantee is
//! validated by sampling in the tests.

/// Finite-sample conformal quantile of calibration nonconformity `scores` at
/// miscoverage `alpha`: the `⌈(n+1)(1−α)⌉`-th smallest score (the `+1`/ceil is
/// the exact finite-sample correction). Returns `+∞` when the calibration set
/// is too small to guarantee the level (then any interval/set is "everything").
pub fn conformal_quantile(scores: &[f32], alpha: f32) -> f32 {
    assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1)");
    let n = scores.len();
    if n == 0
    {
        return f32::INFINITY;
    }
    let mut s = scores.to_vec();
    s.sort_by(f32::total_cmp);
    let k = (((n + 1) as f32) * (1.0 - alpha)).ceil() as usize; // 1-indexed rank
    if (1..=n).contains(&k)
    {
        s[k - 1]
    }
    else
    {
        f32::INFINITY
    }
}

/// Split-conformal **regression**: calibrate on absolute residuals `|y − ŷ|`,
/// then emit symmetric intervals `[ŷ − q̂, ŷ + q̂]` with marginal coverage
/// `≥ 1 − α`.
pub struct ConformalRegressor {
    q: f32,
    alpha: f32,
}

impl ConformalRegressor {
    /// Calibrate from the calibration residuals `|y − ŷ|`.
    pub fn calibrate(residuals: &[f32], alpha: f32) -> Self {
        Self {
            q: conformal_quantile(residuals, alpha),
            alpha,
        }
    }

    /// The prediction interval around a point estimate.
    pub fn interval(&self, y_hat: f32) -> (f32, f32) {
        (y_hat - self.q, y_hat + self.q)
    }

    /// Half-width `q̂` of every interval (the certified radius).
    pub fn half_width(&self) -> f32 {
        self.q
    }

    /// Whether the interval around `y_hat` covers `y_true`.
    pub fn covers(&self, y_hat: f32, y_true: f32) -> bool {
        (y_true - y_hat).abs() <= self.q
    }

    /// The target miscoverage `α` (coverage is `≥ 1 − α`).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

/// Split-conformal **classification** (LAC / threshold rule): nonconformity is
/// `1 − p_true`; the prediction set is `{c : p_c ≥ 1 − q̂}`, with marginal
/// coverage `≥ 1 − α`.
pub struct ConformalClassifier {
    q: f32,
}

impl ConformalClassifier {
    /// Calibrate from per-example softmax probabilities and true labels.
    pub fn calibrate(cal_probs: &[Vec<f32>], cal_labels: &[usize], alpha: f32) -> Self {
        assert_eq!(
            cal_probs.len(),
            cal_labels.len(),
            "conformal: probs/labels length mismatch"
        );
        let scores: Vec<f32> = cal_probs
            .iter()
            .zip(cal_labels)
            .map(|(p, &y)| 1.0 - p[y])
            .collect();
        Self {
            q: conformal_quantile(&scores, alpha),
        }
    }

    /// The prediction set: every class whose probability clears `1 − q̂`.
    pub fn predict_set(&self, probs: &[f32]) -> Vec<usize> {
        probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| 1.0 - p <= self.q)
            .map(|(i, _)| i)
            .collect()
    }

    /// Whether the prediction set for `probs` contains `y_true`.
    pub fn covers(&self, probs: &[f32], y_true: usize) -> bool {
        1.0 - probs[y_true] <= self.q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    #[test]
    fn conformal_quantile_rank_is_finite_sample_correct() {
        // n=4, alpha=0.5 → k = ceil(5·0.5) = 3 → 3rd smallest = 0.3.
        let q = conformal_quantile(&[0.4, 0.1, 0.3, 0.2], 0.5);
        assert!((q - 0.3).abs() < 1e-6, "q = {q}");
    }

    #[test]
    fn conformal_quantile_infinite_when_calibration_too_small() {
        // n=2, alpha=0.1 → k = ceil(3·0.9) = 3 > 2 → +∞ (can't guarantee 90%).
        assert!(conformal_quantile(&[0.5, 0.7], 0.1).is_infinite());
    }

    /// **The guarantee, tested**: split-conformal regression achieves the target
    /// marginal coverage on fresh data, distribution-free. Residuals are |noise|;
    /// calibration and test draws are i.i.d. from the same seeded stream.
    #[test]
    fn conformal_regression_hits_target_coverage() {
        let mut rng = PcgEngine::new(42);
        // Heavy-ish noise so the interval is non-trivial.
        let noise = |rng: &mut PcgEngine| (rng.float_signed() + rng.float_signed()).abs();

        let cal: Vec<f32> = (0..2000).map(|_| noise(&mut rng)).collect();
        let alpha = 0.1; // target coverage 90%
        let reg = ConformalRegressor::calibrate(&cal, alpha);
        assert!(reg.half_width().is_finite() && reg.half_width() > 0.0);

        let n_test = 5000;
        let covered = (0..n_test)
            .filter(|_| {
                let e = noise(&mut rng);
                reg.covers(0.0, e) // ŷ = 0, y_true = residual e
            })
            .count();
        let coverage = covered as f32 / n_test as f32;
        // ≥ 1−α (with finite-sample slack); and not vacuous (< ~99%).
        assert!(
            coverage >= 0.86,
            "coverage {coverage} below target {}",
            1.0 - alpha
        );
        assert!(
            coverage <= 0.99,
            "interval is vacuous (coverage {coverage})"
        );
    }

    /// Conformal classification reaches the target coverage on fresh data.
    #[test]
    fn conformal_classification_hits_target_coverage() {
        let mut rng = PcgEngine::new(7);
        let classes = 4usize;
        // Synthesise a "decent but imperfect" classifier: logits with a bump on
        // the true class, then softmax.
        let make = |rng: &mut PcgEngine| -> (Vec<f32>, usize) {
            let y = (rng.float() * classes as f32) as usize % classes;
            let mut logits: Vec<f32> = (0..classes).map(|_| rng.float_signed()).collect();
            logits[y] += 1.3; // model usually—but not always—favours the truth
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
            let z: f32 = exps.iter().sum();
            (exps.iter().map(|&e| e / z).collect(), y)
        };

        let mut cal_probs = Vec::new();
        let mut cal_labels = Vec::new();
        for _ in 0..2000
        {
            let (p, y) = make(&mut rng);
            cal_probs.push(p);
            cal_labels.push(y);
        }
        let alpha = 0.1;
        let clf = ConformalClassifier::calibrate(&cal_probs, &cal_labels, alpha);

        let n_test = 5000;
        let covered = (0..n_test)
            .filter(|_| {
                let (p, y) = make(&mut rng);
                clf.covers(&p, y)
            })
            .count();
        let coverage = covered as f32 / n_test as f32;
        assert!(coverage >= 0.86, "coverage {coverage} below target 0.90");
    }
}
