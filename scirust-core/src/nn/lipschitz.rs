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
    let mut col_sums = vec![0.0f32; cols];
    let mut max_row = 0.0f32;
    for i in 0..rows
    {
        let row = &w[i * cols..(i + 1) * cols];
        let mut row_sum = 0.0f32;
        for (j, &wij) in row.iter().enumerate()
        {
            let a = wij.abs();
            row_sum += a;
            col_sums[j] += a;
        }
        if row_sum > max_row
        {
            max_row = row_sum;
        }
    }
    let max_col = col_sums.into_iter().fold(0.0f32, f32::max);
    (max_row * max_col).sqrt()
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
    /// Build from the weight matrix. The certified `lip = √2·upper_bound(‖W‖₂)`
    /// uses the guaranteed [`spectral_norm_upper_bound`] so the radius is sound;
    /// `iters` steps of power iteration give the tighter non-certified estimate
    /// available via [`Self::lipschitz_estimate`].
    pub fn new_linear(w: Vec<f32>, num_classes: usize, in_features: usize, iters: usize) -> Self {
        assert_eq!(w.len(), num_classes * in_features, "GloRo: size mismatch");
        let ub = spectral_norm_upper_bound(&w, num_classes, in_features);
        let est = spectral_norm(&w, num_classes, in_features, iters);
        // A valid upper bound never lies below a from-below estimate.
        debug_assert!(
            ub + 1e-4 >= est,
            "upper bound {ub} below power-iteration estimate {est}"
        );
        Self {
            w,
            num_classes,
            in_features,
            lip: SQRT_2 * ub,
            lip_estimate: SQRT_2 * est,
        }
    }

    /// Logits `W·x`.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        (0..self.num_classes)
            .map(|c| {
                let row = &self.w[c * self.in_features..(c + 1) * self.in_features];
                row.iter().zip(x).map(|(&wij, &xj)| wij * xj).sum()
            })
            .collect()
    }

    /// `(top class, certified L2 radius)` where the radius is
    /// `(f_top − max_{B≠top} f_B) / (√2·‖W‖₂)` (0 if the top two logits tie).
    pub fn certify(&self, x: &[f32]) -> (usize, f32) {
        let logits = self.logits(x);
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
        let radius = if self.lip > 0.0
        {
            margin / self.lip
        }
        else
        {
            0.0
        };
        (top, radius.max(0.0))
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
