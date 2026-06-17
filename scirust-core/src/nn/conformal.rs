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

/// **Conformalized Quantile Regression (CQR)** — Romano, Patterson & Candès
/// (NeurIPS 2019). Plain split-conformal regression ([`ConformalRegressor`]) adds
/// a **constant** half-width `q̂` to every point, so its intervals cannot adapt to
/// input-dependent (heteroscedastic) noise. CQR instead starts from a **quantile**
/// regressor — estimates `q_lo(x)` and `q_hi(x)` of the conditional `α/2` and
/// `1−α/2` quantiles — and conformalizes the *signed* score
/// `Eᵢ = max(q_lo(xᵢ) − yᵢ, yᵢ − q_hi(xᵢ))` (how far **outside** the predicted band
/// the truth fell; negative when comfortably inside). The calibrated interval is
/// `[q_lo(x) − Q, q_hi(x) + Q]` with `Q` the finite-sample conformal quantile of
/// the scores. This keeps the **distribution-free marginal coverage `≥ 1−α`** of
/// split conformal while letting the width **vary with `x`** (and `Q` may be
/// negative, *shrinking* over-wide base bands). The conformal step is agnostic to
/// how `q_lo`/`q_hi` were fit. Pure, deterministic arithmetic.
pub struct ConformalQuantileRegressor {
    q: f32,
    alpha: f32,
}

impl ConformalQuantileRegressor {
    /// Calibrate from the base quantile predictions `q_lo`, `q_hi` and the truths
    /// `y` on a held-out set, scoring `Eᵢ = max(q_lo(xᵢ) − yᵢ, yᵢ − q_hi(xᵢ))`.
    pub fn calibrate(q_lo: &[f32], q_hi: &[f32], y: &[f32], alpha: f32) -> Self {
        assert_eq!(q_lo.len(), q_hi.len(), "CQR: q_lo/q_hi length mismatch");
        assert_eq!(
            q_lo.len(),
            y.len(),
            "CQR: predictions/targets length mismatch"
        );
        let scores: Vec<f32> = q_lo
            .iter()
            .zip(q_hi)
            .zip(y)
            .map(|((&lo, &hi), &yi)| (lo - yi).max(yi - hi))
            .collect();
        Self {
            q: conformal_quantile(&scores, alpha),
            alpha,
        }
    }

    /// The **adaptive** prediction interval `[q_lo(x) − Q, q_hi(x) + Q]`.
    pub fn interval(&self, q_lo_x: f32, q_hi_x: f32) -> (f32, f32) {
        (q_lo_x - self.q, q_hi_x + self.q)
    }

    /// Whether the calibrated interval around `(q_lo(x), q_hi(x))` covers `y_true`.
    pub fn covers(&self, q_lo_x: f32, q_hi_x: f32, y_true: f32) -> bool {
        y_true >= q_lo_x - self.q && y_true <= q_hi_x + self.q
    }

    /// The conformal correction `Q` (subtracted from `q_lo`, added to `q_hi`).
    /// May be negative when the base quantiles already over-cover.
    pub fn correction(&self) -> f32 {
        self.q
    }

    /// The target miscoverage `α` (coverage is `≥ 1 − α`).
    pub fn alpha(&self) -> f32 {
        self.alpha
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

    /// CQR's score `Eᵢ = max(q_lo − y, y − q_hi)` and interval semantics, on an
    /// exactly hand-computable case. Base band `[-1, 1]`; truths give scores
    /// `[-1, 1, -0.5, 2]`; sorted `[-1, -0.5, 1, 2]`; `α = 0.5 → k = ⌈5·0.5⌉ = 3`,
    /// so `Q` = 3rd smallest = `1`. The band widens to `[-2, 2]`.
    #[test]
    fn cqr_score_and_interval_semantics() {
        let q_lo = [-1.0f32, -1.0, -1.0, -1.0];
        let q_hi = [1.0f32, 1.0, 1.0, 1.0];
        let y = [0.0f32, 2.0, 0.5, 3.0];
        let cqr = ConformalQuantileRegressor::calibrate(&q_lo, &q_hi, &y, 0.5);
        assert!(
            (cqr.correction() - 1.0).abs() < 1e-6,
            "Q = {}",
            cqr.correction()
        );
        let (lo, hi) = cqr.interval(-1.0, 1.0);
        assert!(
            (lo + 2.0).abs() < 1e-6 && (hi - 2.0).abs() < 1e-6,
            "interval ({lo}, {hi})"
        );
        assert!(cqr.covers(-1.0, 1.0, 1.8) && !cqr.covers(-1.0, 1.0, 2.5));
        assert!((cqr.alpha() - 0.5).abs() < 1e-6);
    }

    /// **The CQR guarantee + its headline property, tested.** On heteroscedastic
    /// data (noise std `σ(x) = 0.1 + x` grows with `x`) with a quantile model whose
    /// band scales as `±c·σ(x)`, CQR (1) hits the distribution-free marginal
    /// coverage `≥ 1−α` on fresh draws and (2) stays **adaptive** — intervals in
    /// the high-noise region are far wider than in the low-noise region (a constant
    /// half-width could not do both). Bit-for-bit deterministic across runs.
    #[test]
    fn cqr_hits_coverage_and_is_adaptive() {
        let run = || -> (f32, f32, f32, f32) {
            let mut rng = PcgEngine::new(123);
            // Symmetric noise ξ = u1+u2+u3, u ~ U(−1,1) (std 1); σ(x) = 0.1 + x.
            let noise = |r: &mut PcgEngine| r.float_signed() + r.float_signed() + r.float_signed();
            let sigma = |x: f32| 0.1 + x;
            let alpha = 0.1f32;
            let c = 1.4f32; // base band a touch narrow ⇒ CQR corrects upward
            let q_lo = |x: f32| -c * sigma(x);
            let q_hi = |x: f32| c * sigma(x);

            let (mut clo, mut chi, mut cy) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..3000
            {
                let x = rng.float();
                let y = sigma(x) * noise(&mut rng);
                clo.push(q_lo(x));
                chi.push(q_hi(x));
                cy.push(y);
            }
            let cqr = ConformalQuantileRegressor::calibrate(&clo, &chi, &cy, alpha);

            let n_test = 8000usize;
            let (mut covered, mut wlow, mut nlow, mut whigh, mut nhigh) =
                (0usize, 0.0f32, 0usize, 0.0f32, 0usize);
            for _ in 0..n_test
            {
                let x = rng.float();
                let y = sigma(x) * noise(&mut rng);
                let (lo, hi) = cqr.interval(q_lo(x), q_hi(x));
                if y >= lo && y <= hi
                {
                    covered += 1;
                }
                let w = hi - lo;
                if x < 0.2
                {
                    wlow += w;
                    nlow += 1;
                }
                else if x > 0.8
                {
                    whigh += w;
                    nhigh += 1;
                }
            }
            (
                covered as f32 / n_test as f32,
                wlow / nlow as f32,
                whigh / nhigh as f32,
                cqr.correction(),
            )
        };
        let (coverage, wlow, whigh, q) = run();
        // (1) Marginal coverage ≥ 1−α (with finite-sample slack), not vacuous.
        assert!(coverage >= 0.87, "CQR coverage {coverage} below 0.90");
        assert!(
            coverage <= 0.985,
            "CQR interval vacuous (coverage {coverage})"
        );
        // (2) Adaptive: high-noise intervals clearly wider than low-noise ones.
        assert!(
            whigh > 1.5 * wlow,
            "CQR not adaptive: wlow = {wlow}, whigh = {whigh}"
        );
        // (3) Determinism.
        assert_eq!((coverage, wlow, whigh, q), run());
    }
}
