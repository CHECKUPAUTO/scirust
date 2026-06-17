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

/// Cumulative APS/RAPS score `s(x, c)` for **every** class `c` of one softmax row
/// `probs`: classes are ranked by descending probability (ascending-index
/// tie-break) and `s(x,c)` is the cumulative mass of all classes ranked at or
/// above `c` (including `c`). The optional RAPS regularization adds
/// `λ·max(0, rank(c) − k_reg)` (rank 1-indexed) to penalise — and thereby trim —
/// low-probability tail classes.
fn aps_cumulative(probs: &[f32], k_reg: usize, lam: f32) -> Vec<f32> {
    let n = probs.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Descending probability, ascending index for a deterministic tie-break.
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    let mut s = vec![0.0f32; n];
    let mut cum = 0.0f32;
    for (pos, &c) in order.iter().enumerate()
    {
        cum += probs[c];
        let penalty = lam * ((pos as f32 + 1.0) - k_reg as f32).max(0.0);
        s[c] = cum + penalty;
    }
    s
}

/// **Adaptive Prediction Sets (APS)** for conformal **classification** — Romano,
/// Sesia & Candès (NeurIPS 2020), with the optional **RAPS** regularization of
/// Angelopoulos et al. (2021). Where the threshold rule ([`ConformalClassifier`])
/// keeps every class above a fixed probability, APS builds the set by walking
/// classes from most to least probable and accumulating their mass: the
/// nonconformity score of `(x, y)` is the cumulative mass of all classes at least
/// as probable as the truth, `s(x,y)`. After calibrating `q̂` as the finite-sample
/// quantile of those scores, the prediction set is `{c : s(x,c) ≤ q̂}`. This gives
/// **distribution-free marginal coverage `≥ 1−α`** with **adaptive set size** —
/// confident inputs get small sets, ambiguous ones get larger sets. **RAPS** adds
/// `λ·max(0, rank − k_reg)` to the score, trimming unlikely tail classes for
/// smaller, more stable sets at the same coverage. The score is shared by
/// calibration and prediction, so the guarantee holds for any fixed `(k_reg, λ)`.
/// Pure, deterministic arithmetic.
pub struct AdaptivePredictionSets {
    q: f32,
    k_reg: usize,
    lam: f32,
}

impl AdaptivePredictionSets {
    /// Calibrate plain **APS** (no regularization) from softmax probabilities and
    /// true labels.
    pub fn calibrate(cal_probs: &[Vec<f32>], cal_labels: &[usize], alpha: f32) -> Self {
        Self::calibrate_raps(cal_probs, cal_labels, alpha, 0, 0.0)
    }

    /// Calibrate **RAPS** with regularization `λ·max(0, rank − k_reg)`
    /// (`k_reg = 0, lam = 0` recovers plain APS).
    pub fn calibrate_raps(
        cal_probs: &[Vec<f32>],
        cal_labels: &[usize],
        alpha: f32,
        k_reg: usize,
        lam: f32,
    ) -> Self {
        assert_eq!(
            cal_probs.len(),
            cal_labels.len(),
            "APS: probs/labels length mismatch"
        );
        let scores: Vec<f32> = cal_probs
            .iter()
            .zip(cal_labels)
            .map(|(p, &y)| aps_cumulative(p, k_reg, lam)[y])
            .collect();
        Self {
            q: conformal_quantile(&scores, alpha),
            k_reg,
            lam,
        }
    }

    /// The adaptive prediction set `{c : s(x,c) ≤ q̂}` for one softmax row.
    pub fn predict_set(&self, probs: &[f32]) -> Vec<usize> {
        let s = aps_cumulative(probs, self.k_reg, self.lam);
        (0..probs.len()).filter(|&c| s[c] <= self.q).collect()
    }

    /// Whether the prediction set for `probs` contains `y_true`.
    pub fn covers(&self, probs: &[f32], y_true: usize) -> bool {
        aps_cumulative(probs, self.k_reg, self.lam)[y_true] <= self.q
    }

    /// The calibrated score threshold `q̂`.
    pub fn threshold(&self) -> f32 {
        self.q
    }
}

/// **Adaptive Conformal Inference (ACI)** — Gibbs & Candès (NeurIPS 2021). Plain
/// split conformal assumes exchangeability; under **distribution shift** its fixed
/// quantile silently loses coverage. ACI restores it **online**: it tracks an
/// effective miscoverage level `αₜ` and, after each observation, nudges it by the
/// coverage error — `αₜ₊₁ = αₜ + γ·(α − errₜ)` where `errₜ = 1` iff the truth fell
/// outside the level-`(1−αₜ)` interval. This feedback drives the long-run miss
/// rate to `α` (so coverage to `1−α`) for **any** score stream, shifting or not.
/// Paired with a sliding window of recent nonconformity scores (so the quantile
/// itself tracks the current distribution), it holds `≈ 1−α` coverage through
/// shifts where static conformal collapses. Pure `f32`, fixed order ⇒
/// **deterministic**.
pub struct AdaptiveConformal {
    target_alpha: f32,
    gamma: f32,
    alpha_t: f32,
}

impl AdaptiveConformal {
    /// New ACI at target miscoverage `target_alpha ∈ (0,1)` and adaptation rate
    /// `gamma > 0` (the step size of the `αₜ` update).
    pub fn new(target_alpha: f32, gamma: f32) -> Self {
        assert!(
            target_alpha > 0.0 && target_alpha < 1.0,
            "ACI: target_alpha must be in (0,1)"
        );
        assert!(gamma > 0.0, "ACI: gamma must be positive");
        Self {
            target_alpha,
            gamma,
            alpha_t: target_alpha,
        }
    }

    /// One online step: given the recent calibration `scores` (e.g. a sliding
    /// window) and the new point's nonconformity `score`, report whether the
    /// current level-`(1−αₜ)` interval **covers** it (`score ≤ q̂`), then update
    /// `αₜ`. `αₜ ≤ 0` ⇒ the interval is everything (always covers); `αₜ ≥ 1` ⇒
    /// empty (never covers).
    pub fn step(&mut self, scores: &[f32], score: f32) -> bool {
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
            conformal_quantile(scores, self.alpha_t)
        };
        let covered = score <= q;
        let err = if covered { 0.0 } else { 1.0 };
        self.alpha_t = (self.alpha_t + self.gamma * (self.target_alpha - err)).clamp(0.0, 1.0);
        covered
    }

    /// The current effective miscoverage level `αₜ`.
    pub fn alpha(&self) -> f32 {
        self.alpha_t
    }

    /// The target miscoverage `α` (long-run coverage is `≈ 1 − α`).
    pub fn target_alpha(&self) -> f32 {
        self.target_alpha
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

    /// APS cumulative score orders classes by descending mass; RAPS adds the
    /// rank penalty. probs sorted: idx1(0.5) > idx3(0.3) > idx0(0.15) > idx2(0.05).
    #[test]
    fn aps_cumulative_score_orders_by_mass() {
        let p = [0.15f32, 0.5, 0.05, 0.3];
        let s = aps_cumulative(&p, 0, 0.0);
        assert!((s[1] - 0.5).abs() < 1e-6, "s1 = {}", s[1]); // rank1
        assert!((s[3] - 0.8).abs() < 1e-6, "s3 = {}", s[3]); // rank2: 0.5+0.3
        assert!((s[0] - 0.95).abs() < 1e-6, "s0 = {}", s[0]); // rank3: +0.15
        assert!((s[2] - 1.0).abs() < 1e-6, "s2 = {}", s[2]); // rank4: +0.05
        // RAPS penalty λ=0.1, k_reg=1: rank r adds 0.1·max(0, r−1).
        let sr = aps_cumulative(&p, 1, 0.1);
        assert!((sr[1] - 0.5).abs() < 1e-6); // rank1: +0
        assert!((sr[3] - 0.9).abs() < 1e-6); // rank2: 0.8 + 0.1
        assert!((sr[2] - 1.3).abs() < 1e-6); // rank4: 1.0 + 0.3
    }

    /// Build a mixed-difficulty classification stream: with probability ½ an
    /// **easy** (peaked) example, else a **hard** (near-flat) one. Returns
    /// `(probs, label, is_easy)`.
    fn make_clf_example(r: &mut PcgEngine, classes: usize) -> (Vec<f32>, usize, bool) {
        let y = (r.float() * classes as f32) as usize % classes;
        let easy = r.float() < 0.5;
        let bump = if easy { 3.2 } else { 0.4 };
        let mut logits: Vec<f32> = (0..classes).map(|_| r.float_signed()).collect();
        logits[y] += bump;
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
        let z: f32 = exps.iter().sum();
        (exps.iter().map(|&e| e / z).collect(), y, easy)
    }

    /// **The APS guarantee + adaptivity, tested.** APS hits the distribution-free
    /// marginal coverage `≥ 1−α` on fresh data, and its set size **adapts** —
    /// ambiguous (hard) inputs get substantially larger sets than confident (easy)
    /// ones. Bit-for-bit deterministic across runs.
    #[test]
    fn aps_hits_coverage_and_adapts() {
        let run = || -> (f32, f32, f32) {
            let mut rng = PcgEngine::new(11);
            let classes = 6usize;
            let (mut cp, mut cl) = (Vec::new(), Vec::new());
            for _ in 0..3000
            {
                let (p, y, _) = make_clf_example(&mut rng, classes);
                cp.push(p);
                cl.push(y);
            }
            let aps = AdaptivePredictionSets::calibrate(&cp, &cl, 0.1);
            let n_test = 6000usize;
            let (mut cov, mut esz, mut en, mut hsz, mut hn) =
                (0usize, 0usize, 0usize, 0usize, 0usize);
            for _ in 0..n_test
            {
                let (p, y, easy) = make_clf_example(&mut rng, classes);
                if aps.covers(&p, y)
                {
                    cov += 1;
                }
                let sz = aps.predict_set(&p).len();
                if easy
                {
                    esz += sz;
                    en += 1;
                }
                else
                {
                    hsz += sz;
                    hn += 1;
                }
            }
            (
                cov as f32 / n_test as f32,
                esz as f32 / en as f32,
                hsz as f32 / hn as f32,
            )
        };
        let (cov, easy, hard) = run();
        assert!(cov >= 0.87, "APS coverage {cov} below 0.90");
        assert!(
            hard > easy + 0.5,
            "APS not adaptive: easy set {easy}, hard set {hard}"
        );
        assert_eq!((cov, easy, hard), run());
    }

    /// **RAPS** (regularized APS) yields **smaller** average prediction sets than
    /// plain APS while keeping marginal coverage `≥ 1−α` — its headline property.
    /// Demonstrated on a decent classifier over **many** classes (a long
    /// probability tail that APS pads sets with and RAPS trims). Same data for both.
    #[test]
    fn raps_shrinks_sets_vs_aps_at_coverage() {
        let classes = 40usize;
        // Decent classifier (true class usually top) over many classes ⇒ long tail.
        let draw = |r: &mut PcgEngine| -> (Vec<f32>, usize) {
            let y = (r.float() * classes as f32) as usize % classes;
            let mut logits: Vec<f32> = (0..classes).map(|_| r.float_signed()).collect();
            logits[y] += 2.5;
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
            let z: f32 = exps.iter().sum();
            (exps.iter().map(|&e| e / z).collect(), y)
        };
        let mut rng = PcgEngine::new(5);
        let (mut cp, mut cl) = (Vec::new(), Vec::new());
        for _ in 0..3000
        {
            let (p, y) = draw(&mut rng);
            cp.push(p);
            cl.push(y);
        }
        let alpha = 0.1;
        let aps = AdaptivePredictionSets::calibrate(&cp, &cl, alpha);
        let raps = AdaptivePredictionSets::calibrate_raps(&cp, &cl, alpha, 2, 0.1);

        let n_test = 6000usize;
        let mut tp = Vec::new();
        let mut tl = Vec::new();
        for _ in 0..n_test
        {
            let (p, y) = draw(&mut rng);
            tp.push(p);
            tl.push(y);
        }
        let eval = |m: &AdaptivePredictionSets| -> (f32, f32) {
            let mut cov = 0usize;
            let mut size = 0usize;
            for (p, &y) in tp.iter().zip(&tl)
            {
                if m.covers(p, y)
                {
                    cov += 1;
                }
                size += m.predict_set(p).len();
            }
            (cov as f32 / n_test as f32, size as f32 / n_test as f32)
        };
        let (cov_aps, size_aps) = eval(&aps);
        let (cov_raps, size_raps) = eval(&raps);
        assert!(cov_aps >= 0.87, "APS coverage {cov_aps} below 0.90");
        assert!(cov_raps >= 0.87, "RAPS coverage {cov_raps} below 0.90");
        assert!(
            size_raps < size_aps,
            "RAPS did not shrink sets: APS {size_aps}, RAPS {size_raps}"
        );
    }

    /// ACI's `αₜ` feedback: a miss decreases `αₜ` by `γ(1−α)` (widening), a cover
    /// increases it by `γα` (narrowing) — the exact update toward miss-rate `= α`.
    #[test]
    fn aci_alpha_update_rule() {
        let window: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect(); // [0,…,1.9]
        let mut aci = AdaptiveConformal::new(0.1, 0.05);
        // q at α=0.1: ⌈21·0.9⌉=19 → 19th smallest = 1.8; score 100 > 1.8 ⇒ miss.
        assert!(!aci.step(&window, 100.0));
        assert!(
            (aci.alpha() - 0.055).abs() < 1e-6,
            "after miss α={}",
            aci.alpha()
        );
        // score −100 ⇒ cover ⇒ α += 0.05·0.1 = 0.005.
        assert!(aci.step(&window, -100.0));
        assert!(
            (aci.alpha() - 0.06).abs() < 1e-6,
            "after cover α={}",
            aci.alpha()
        );
        assert!((aci.target_alpha() - 0.1).abs() < 1e-6);
    }

    /// **The ACI guarantee, tested.** On a stream with a mid-stream variance shift,
    /// adaptive conformal (sliding window + `αₜ` feedback) holds ≈ `1−α` coverage,
    /// while static conformal (a fixed quantile from the initial calibration)
    /// collapses after the shift. Bit-for-bit deterministic.
    #[test]
    fn aci_maintains_coverage_under_shift() {
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(20);
            let noise =
                |r: &mut PcgEngine, scale: f32| scale * (r.float_signed() + r.float_signed()).abs();
            let target = 0.1f32;
            let w = 200usize;
            let mut window: Vec<f32> = (0..w).map(|_| noise(&mut rng, 1.0)).collect();
            let q_static = conformal_quantile(&window, target);
            let mut aci = AdaptiveConformal::new(target, 0.05);
            let n = 4000usize;
            let (mut acov, mut scov) = (0usize, 0usize);
            for t in 0..n
            {
                let scale = if t < n / 2 { 1.0 } else { 3.0 }; // variance shift halfway
                let s = noise(&mut rng, scale);
                if aci.step(&window, s)
                {
                    acov += 1;
                }
                if s <= q_static
                {
                    scov += 1;
                }
                window.remove(0);
                window.push(s);
            }
            (acov as f32 / n as f32, scov as f32 / n as f32)
        };
        let (aci_cov, static_cov) = run();
        assert!(
            (aci_cov - 0.9).abs() < 0.05,
            "ACI coverage {aci_cov} not ≈ 0.9"
        );
        assert!(
            static_cov < aci_cov - 0.1,
            "static conformal ({static_cov}) not clearly worse than ACI ({aci_cov})"
        );
        assert_eq!(run(), (aci_cov, static_cov)); // determinism
    }
}
