//! **Lipschitz-based certified robustness** (GloRo — Leino, Wang & Fredrikson,
//! *Globally-Robust Neural Networks*, ICML 2021).
//!
//! A function with global L2 Lipschitz constant `L` cannot change its output by
//! more than `L·‖δ‖` under an input perturbation `δ`. For a classifier this yields
//! a **provable robustness radius** with no search and no sampling: at an input
//! whose top-vs-runner-up logit **margin** is `m`, the prediction is certified
//! constant within `‖δ‖₂ ≤ m / (√2·L)` (the `√2` because the margin functional
//! `f_A − f_B = (e_A − e_B)ᵀ f` has Lipschitz `≤ √2·L`). The network's `L` is
//! upper-bounded by the **product of the layers' spectral norms** (largest
//! singular values) when the activations are 1-Lipschitz (ReLU, etc.).
//!
//! # Soundness: use an *upper* bound, not an estimate
//!
//! A certificate is only sound if `L` is a genuine **upper** bound. Power
//! iteration ([`spectral_norm`]) converges to `σ_max` **from below**, so with a
//! finite iteration count it *under*-estimates — plugging it into the radius
//! makes the ball too large (unsound). The certified [`GloroClassifier`]
//! therefore uses [`spectral_norm_upper_bound`] (the always-valid
//! `√(‖W‖₁·‖W‖∞)` bound) for the radius; the power-iteration value is exposed
//! only as a tighter *non-certified* estimate (fine for spectral normalization
//! during training). The `√(‖W‖₁·‖W‖∞)` bound is conservative (it can be loose
//! for well-conditioned matrices); a tighter *rigorous* a-posteriori bound is
//! future work.
//!
//! Here: [`spectral_norm`] (deterministic power iteration, an estimate),
//! [`spectral_norm_upper_bound`] (a guaranteed upper bound),
//! [`spectral_normalize`] (the 1-Lipschitz-constrained layer of GloRo), and
//! [`GloroClassifier`] (a linear classifier with a sound certified radius). Pure
//! `f32`, fixed order ⇒ **bit-for-bit deterministic**.

use crate::error::{Result, SciRustError};
use std::f32::consts::SQRT_2;

/// Largest singular value `‖W‖₂` of a `rows×cols` row-major matrix by **power
/// iteration** on `WᵀW` (deterministic: fixed all-ones start, fixed `iters`).
pub fn spectral_norm(w: &[f32], rows: usize, cols: usize, iters: usize) -> f32 {
    assert_eq!(w.len(), rows * cols, "spectral_norm: size mismatch");
    if rows == 0 || cols == 0
    {
        return 0.0;
    }
    let mut v = vec![1.0f32 / (cols as f32).sqrt(); cols];
    let mut sigma = 0.0f32;
    for _ in 0..iters
    {
        // u = W v   (rows)
        let mut u = vec![0.0f32; rows];
        for (i, ui) in u.iter_mut().enumerate()
        {
            let row = &w[i * cols..(i + 1) * cols];
            *ui = row.iter().zip(&v).map(|(&a, &b)| a * b).sum();
        }
        sigma = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
        // v ← normalize(Wᵀ u)   (cols)
        let mut vn = vec![0.0f32; cols];
        for (i, &ui) in u.iter().enumerate()
        {
            let row = &w[i * cols..(i + 1) * cols];
            for (vj, &wij) in vn.iter_mut().zip(row)
            {
                *vj += wij * ui;
            }
        }
        let nrm = vn.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if nrm <= 0.0
        {
            return 0.0;
        }
        for x in vn.iter_mut()
        {
            *x /= nrm;
        }
        v = vn;
    }
    sigma
}

/// A **guaranteed upper bound** on the spectral norm `‖W‖₂`, valid for *any*
/// matrix: `‖W‖₂ ≤ √(‖W‖₁ · ‖W‖∞)`, where `‖W‖₁` is the maximum column
/// absolute-sum and `‖W‖∞` the maximum row absolute-sum. Unlike
/// [`spectral_norm`] (power iteration, which converges to `σ_max` *from below*
/// and only *estimates* it), this never under-estimates — so it is the value
/// that must back a *sound* Lipschitz certificate.
pub fn spectral_norm_upper_bound(w: &[f32], rows: usize, cols: usize) -> f32 {
    assert_eq!(
        w.len(),
        rows * cols,
        "spectral_norm_upper_bound: size mismatch"
    );
    if rows == 0 || cols == 0
    {
        return 0.0;
    }
    // Accumulate the row/column absolute-sums in f64 so the sums themselves are
    // not rounded below the true values, then apply a small relative safety
    // margin before narrowing to f32. Without this, f32 rounding of a large sum
    // can make the returned bound land a few ulps *below* the true σ_max (e.g. a
    // 7×7 all-ones matrix scaled by 2^23+1), which would make the certificate
    // unsound; the margin (≫ f32 epsilon) keeps it a genuine upper bound.
    let mut col_sums = vec![0.0f64; cols];
    let mut max_row = 0.0f64;
    for i in 0..rows
    {
        let row = &w[i * cols..(i + 1) * cols];
        let mut row_sum = 0.0f64;
        for (j, &wij) in row.iter().enumerate()
        {
            let a = wij.abs() as f64;
            row_sum += a;
            col_sums[j] += a;
        }
        if row_sum > max_row
        {
            max_row = row_sum;
        }
    }
    let max_col = col_sums.into_iter().fold(0.0f64, f64::max);
    // 1e-6 ≫ f32 machine epsilon (~1.2e-7), so the result stays ≥ σ_max even
    // after the round-to-nearest f32 narrowing.
    let bound = (max_row * max_col).sqrt() * (1.0 + 1e-6);
    bound as f32
}

/// A **spectrally-normalized** copy of `w` (`W / ‖W‖₂`), so the result has spectral
/// norm ≈ 1 — a 1-Lipschitz-constrained linear layer (GloRo). A zero matrix is
/// returned unchanged.
///
/// Note: this divides by the power-iteration *estimate*, so the result's norm is
/// only *approximately* 1 (it may exceed 1 slightly when the estimate has not
/// fully converged). It is a training-time constraint, not a certified bound;
/// certification goes through [`spectral_norm_upper_bound`].
pub fn spectral_normalize(w: &[f32], rows: usize, cols: usize, iters: usize) -> Vec<f32> {
    let sn = spectral_norm(w, rows, cols, iters);
    if sn <= 0.0
    {
        return w.to_vec();
    }
    w.iter().map(|&x| x / sn).collect()
}

/// A linear classifier `f(x) = W·x` (`W` is `num_classes × in_features`,
/// row-major) with a **GloRo** certified L2 radius. The global Lipschitz bound of
/// the margin functional is `√2·‖W‖₂`, so the certified radius at `x` is
/// `margin(x) / (√2·‖W‖₂)`. For a linear classifier this is *sound* (and tight up
/// to the `√2` versus the exact per-pair distance).
pub struct GloroClassifier {
    w: Vec<f32>,
    num_classes: usize,
    in_features: usize,
    /// Certified Lipschitz bound `√2·upper_bound(‖W‖₂)` — a genuine upper bound,
    /// so the radius it produces is sound.
    lip: f32,
    /// Tighter but *non-certified* `√2·power_iteration(‖W‖₂)`, for reference.
    lip_estimate: f32,
}

impl GloroClassifier {
    /// Fallible constructor for a certifiable linear classifier.
    ///
    /// Certification requires at least two classes, a positive input dimension,
    /// an exactly matching row-major weight matrix, and finite weights. It also
    /// rejects a model whose derived Lipschitz values cannot be represented as
    /// finite `f32` values.
    pub fn try_new_linear(
        w: Vec<f32>,
        num_classes: usize,
        in_features: usize,
        iters: usize,
    ) -> Result<Self> {
        if num_classes < 2
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo certification requires at least two classes".to_string(),
            ));
        }
        if in_features == 0
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo in_features must be > 0".to_string(),
            ));
        }
        let expected = num_classes.checked_mul(in_features).ok_or_else(|| {
            SciRustError::InvalidConfig("GloRo weight dimensions overflow usize".to_string())
        })?;
        if w.len() != expected
        {
            return Err(SciRustError::InvalidConfig(format!(
                "GloRo weight size mismatch: expected {expected}, got {}",
                w.len()
            )));
        }
        if !w.iter().all(|value| value.is_finite())
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo weights must be finite".to_string(),
            ));
        }

        let ub = spectral_norm_upper_bound(&w, num_classes, in_features);
        let est = spectral_norm(&w, num_classes, in_features, iters);
        let lip = SQRT_2 * ub;
        let lip_estimate = SQRT_2 * est;
        if !lip.is_finite() || !lip_estimate.is_finite()
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo Lipschitz bound/estimate is non-finite".to_string(),
            ));
        }
        // A valid upper bound never lies below a from-below estimate.
        debug_assert!(
            ub + 1e-4 >= est,
            "upper bound {ub} below power-iteration estimate {est}"
        );
        Ok(Self {
            w,
            num_classes,
            in_features,
            lip,
            lip_estimate,
        })
    }

    /// Build from the weight matrix. The certified `lip = √2·upper_bound(‖W‖₂)`
    /// uses the guaranteed [`spectral_norm_upper_bound`] so the radius is sound;
    /// `iters` steps of power iteration give the tighter non-certified estimate
    /// available via [`Self::lipschitz_estimate`].
    ///
    /// # Panics
    ///
    /// Panics when the model cannot satisfy the certification preconditions.
    /// Prefer [`Self::try_new_linear`] for caller-controlled model data.
    pub fn new_linear(w: Vec<f32>, num_classes: usize, in_features: usize, iters: usize) -> Self {
        Self::try_new_linear(w, num_classes, in_features, iters)
            .unwrap_or_else(|error| panic!("GloroClassifier::new_linear: {error}"))
    }

    /// Fallible logits `W·x` with exact dimension and finiteness validation.
    pub fn try_logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_features
        {
            return Err(SciRustError::InvalidConfig(format!(
                "GloRo input size mismatch: expected {}, got {}",
                self.in_features,
                x.len()
            )));
        }
        if !x.iter().all(|value| value.is_finite())
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo input must contain only finite values".to_string(),
            ));
        }
        let logits: Vec<f32> = (0..self.num_classes)
            .map(|c| {
                let row = &self.w[c * self.in_features..(c + 1) * self.in_features];
                row.iter().zip(x).map(|(&wij, &xj)| wij * xj).sum()
            })
            .collect();
        if !logits.iter().all(|value| value.is_finite())
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo logits became non-finite".to_string(),
            ));
        }
        Ok(logits)
    }

    /// Logits `W·x`.
    ///
    /// # Panics
    ///
    /// Panics when `x` has the wrong length, contains a non-finite value, or the
    /// dot products overflow to a non-finite logit. Prefer [`Self::try_logits`]
    /// for caller-controlled inputs.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        self.try_logits(x)
            .unwrap_or_else(|error| panic!("GloroClassifier::logits: {error}"))
    }

    /// Fallible `(top class, certified L2 radius)`.
    pub fn try_certify(&self, x: &[f32]) -> Result<(usize, f32)> {
        let logits = self.try_logits(x)?;
        let mut top = 0usize;
        for c in 1..self.num_classes
        {
            if logits[c] > logits[top]
            {
                top = c;
            }
        }
        let mut runner = f32::NEG_INFINITY;
        for (c, &l) in logits.iter().enumerate()
        {
            if c != top && l > runner
            {
                runner = l;
            }
        }
        let margin = logits[top] - runner;
        if !margin.is_finite()
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo top-vs-runner-up margin is non-finite".to_string(),
            ));
        }
        let radius = if self.lip > 0.0
        {
            margin / self.lip
        }
        else
        {
            0.0
        };
        if !radius.is_finite()
        {
            return Err(SciRustError::InvalidConfig(
                "GloRo certified radius is non-finite".to_string(),
            ));
        }
        Ok((top, radius.max(0.0)))
    }

    /// `(top class, certified L2 radius)` where the radius is
    /// `(f_top − max_{B≠top} f_B) / (√2·‖W‖₂)` (0 if the top two logits tie).
    ///
    /// # Panics
    ///
    /// Panics when the input cannot satisfy the certification preconditions.
    /// Prefer [`Self::try_certify`] for caller-controlled inputs.
    pub fn certify(&self, x: &[f32]) -> (usize, f32) {
        self.try_certify(x)
            .unwrap_or_else(|error| panic!("GloroClassifier::certify: {error}"))
    }

    /// The **certified** global Lipschitz bound `√2·upper_bound(‖W‖₂)` used in the
    /// (sound) certificate.
    pub fn lipschitz(&self) -> f32 {
        self.lip
    }

    /// The tighter **non-certified** power-iteration estimate `√2·σ̂(W)`. Do not
    /// use for certification — it can under-estimate the true Lipschitz constant.
    pub fn lipschitz_estimate(&self) -> f32 {
        self.lip_estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// Spectral norm = largest singular value. For a diagonal matrix it is the
    /// largest `|diagonal|`; for a rectangular matrix with orthogonal rows it is
    /// the largest row norm.
    #[test]
    fn spectral_norm_known_values() {
        // diag(3, -5, 2) → 5.
        let d = vec![3.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 2.0];
        assert!((spectral_norm(&d, 3, 3, 100) - 5.0).abs() < 1e-3);
        // [[1,0,0],[0,2,0]] (2×3) → singular values {2,1} → 2.
        let r = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        assert!((spectral_norm(&r, 2, 3, 100) - 2.0).abs() < 1e-3);
    }

    /// The guaranteed upper bound never falls below the true spectral norm
    /// (whereas power iteration can, from below).
    #[test]
    fn upper_bound_dominates_spectral_norm() {
        let mut rng = PcgEngine::new(11);
        for &(rows, cols) in &[(3usize, 3usize), (5, 7), (8, 4)]
        {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.float_signed() * 3.0).collect();
            let ub = spectral_norm_upper_bound(&w, rows, cols);
            let sn = spectral_norm(&w, rows, cols, 200);
            assert!(ub + 1e-4 >= sn, "ub {ub} < spectral norm {sn}");
        }
        // Exact on diag(3,-5,2): ‖·‖₁ = ‖·‖∞ = 5 ⇒ √(5·5) = 5 = σ_max.
        let d = vec![3.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 2.0];
        assert!((spectral_norm_upper_bound(&d, 3, 3) - 5.0).abs() < 1e-4);
    }

    /// Soundness at a large scale where naive f32 summation would round the
    /// bound *below* σ_max: a 7×7 all-ones matrix scaled by 2²³+1 has exact
    /// σ_max = 7·(2²³+1); the upper bound must not fall under it.
    #[test]
    fn upper_bound_is_sound_at_large_scale() {
        let s = 8388609.0f32; // 2^23 + 1, exactly representable
        let w = vec![s; 49]; // 7×7 all-ones × s (rank-1 ⇒ σ_max = 7·s exactly)
        let sigma_max = 7.0f32 * s;
        let ub = spectral_norm_upper_bound(&w, 7, 7);
        assert!(
            ub >= sigma_max,
            "upper bound {ub} fell below true σ_max {sigma_max}"
        );
    }

    /// After spectral normalization the spectral norm is ≈ 1 (the 1-Lipschitz
    /// constrained layer).
    #[test]
    fn spectral_normalize_gives_unit_norm() {
        let mut rng = PcgEngine::new(4);
        let (rows, cols) = (5usize, 7usize);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.float_signed() * 2.0).collect();
        let wn = spectral_normalize(&w, rows, cols, 100);
        assert!(
            (spectral_norm(&wn, rows, cols, 100) - 1.0).abs() < 1e-3,
            "normalized spectral norm = {}",
            spectral_norm(&wn, rows, cols, 100)
        );
    }

    #[test]
    fn fallible_gloro_boundary_rejects_invalid_model_or_input() {
        assert!(GloroClassifier::try_new_linear(vec![], 0, 2, 20).is_err());
        assert!(GloroClassifier::try_new_linear(vec![1.0, 2.0], 1, 2, 20).is_err());
        assert!(GloroClassifier::try_new_linear(vec![], 2, 0, 20).is_err());
        assert!(GloroClassifier::try_new_linear(vec![1.0, f32::NAN, 0.0, 1.0], 2, 2, 20).is_err());

        let clf = GloroClassifier::try_new_linear(vec![1.0, 0.0, 0.0, 1.0], 2, 2, 20).unwrap();
        assert!(clf.try_logits(&[1.0]).is_err());
        assert!(clf.try_logits(&[1.0, 2.0, 3.0]).is_err());
        assert!(clf.try_certify(&[1.0, f32::INFINITY]).is_err());
    }

    #[test]
    fn fallible_gloro_boundary_accepts_valid_input() {
        let clf = GloroClassifier::try_new_linear(vec![1.0, 0.0, 0.0, 1.0], 2, 2, 20).unwrap();
        assert_eq!(clf.try_logits(&[2.0, 1.0]).unwrap(), vec![2.0, 1.0]);
        let (class, radius) = clf.try_certify(&[2.0, 1.0]).unwrap();
        assert_eq!(class, 0);
        assert!(radius > 0.0 && radius.is_finite());
    }

    /// **The GloRo certificate, tested for soundness and conservativeness.** For a
    /// linear classifier the certified radius `m/(√2‖W‖)` is (1) **sound** — the
    /// worst-case perturbation of that size does not flip the prediction — and
    /// (2) **conservative** — it never exceeds the exact L2 distance to the nearest
    /// decision boundary `min_B (f_top−f_B)/‖W_top−W_B‖`. Deterministic.
    #[test]
    fn gloro_radius_is_sound_and_conservative() {
        let mut rng = PcgEngine::new(8);
        let (nc, inf) = (4usize, 6usize);
        let w: Vec<f32> = (0..nc * inf).map(|_| rng.float_signed()).collect();
        let clf = GloroClassifier::new_linear(w.clone(), nc, inf, 80);
        let x: Vec<f32> = (0..inf).map(|_| rng.float_signed()).collect();
        let (top, r) = clf.certify(&x);
        assert!(r > 0.0, "expected a positive certified radius");

        // (1) Soundness: the worst-case perturbation toward each boundary at
        // radius r keeps `top` the argmax.
        let logits = clf.logits(&x);
        for b in 0..nc
        {
            if b == top
            {
                continue;
            }
            // d = W_top − W_b; worst δ = −0.999·r·d/‖d‖.
            let d: Vec<f32> = (0..inf)
                .map(|j| w[top * inf + j] - w[b * inf + j])
                .collect();
            let dn = d.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let xp: Vec<f32> = x
                .iter()
                .zip(&d)
                .map(|(&xj, &dj)| xj - 0.999 * r * dj / dn)
                .collect();
            let lp = clf.logits(&xp);
            let mut amax = 0usize;
            for c in 1..nc
            {
                if lp[c] > lp[amax]
                {
                    amax = c;
                }
            }
            assert_eq!(amax, top, "GloRo radius not sound toward class {b}");

            // (2) Conservativeness: r ≤ exact distance to the A-vs-b boundary.
            let exact = (logits[top] - logits[b]) / dn;
            assert!(
                r <= exact + 1e-5,
                "GloRo radius {r} exceeds exact boundary distance {exact} (class {b})"
            );
        }

        // Determinism.
        let clf2 = GloroClassifier::new_linear(w, nc, inf, 80);
        assert_eq!(clf.certify(&x), clf2.certify(&x));
    }
}
