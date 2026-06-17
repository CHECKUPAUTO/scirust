//! **Kolmogorov–Arnold Networks (KAN)** (Liu et al., 2024, arXiv:2404.19756),
//! realised with the **RBF basis** of *FastKAN* (Li, 2024, arXiv:2405.06721).
//!
//! Where an MLP puts a fixed nonlinearity on the **nodes** and learns the edge
//! **weights**, a KAN puts a **learnable univariate function** on each **edge**:
//! `y_j = Σ_i φ_{ij}(x_i)`. Each `φ` is a small learnable function — a weighted
//! sum of fixed basis functions plus a `SiLU` base term:
//! `φ(x) = w_b·SiLU(x) + Σ_k c_k·B_k(x)`. The Kolmogorov–Arnold theorem says any
//! continuous multivariate function is a sum of univariate functions, so a single
//! KAN layer represents **additive** targets exactly. Here `B_k` are Gaussian RBFs
//! on a fixed grid (FastKAN) — smoother and free of the B-spline boundary cases,
//! and an output that is **linear in the coefficients**, so fitting is a convex
//! least-squares problem solved by deterministic gradient descent.
//!
//! Pure `f32` in a fixed order ⇒ **bit-for-bit deterministic**.

use crate::nn::PcgEngine;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `SiLU(x) = x·σ(x)` — the smooth base term added to every edge function.
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// A single **KAN layer** `(in_features → out_features)` with a learnable
/// univariate function on each edge, expressed in a fixed Gaussian-RBF basis plus
/// a `SiLU` base term. The output is linear in the trainable coefficients, so
/// [`fit`](Self::fit) is a convex least-squares gradient descent.
pub struct KanLayer {
    in_features: usize,
    out_features: usize,
    num_basis: usize,
    grid_lo: f32,
    grid_hi: f32,
    coeffs: Vec<f32>, // in·out·num_basis (spline weights)
    base: Vec<f32>,   // in·out (SiLU base weights)
}

impl KanLayer {
    /// New layer with `num_basis` RBF centres evenly spaced on `[grid_lo, grid_hi]`
    /// and small seeded initial coefficients.
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_basis: usize,
        grid_lo: f32,
        grid_hi: f32,
        rng: &mut PcgEngine,
    ) -> Self {
        assert!(
            num_basis >= 2 && grid_hi > grid_lo,
            "KAN: need ≥2 basis, grid_hi>grid_lo"
        );
        let scale = 0.1f32;
        let coeffs = (0..in_features * out_features * num_basis)
            .map(|_| rng.float_signed() * scale)
            .collect();
        let base = (0..in_features * out_features)
            .map(|_| rng.float_signed() * scale)
            .collect();
        Self {
            in_features,
            out_features,
            num_basis,
            grid_lo,
            grid_hi,
            coeffs,
            base,
        }
    }

    /// The `num_basis` Gaussian RBF activations of a scalar `x` (centres on the
    /// grid; width = the grid spacing, so neighbours overlap).
    fn rbf(&self, x: f32) -> Vec<f32> {
        let step = (self.grid_hi - self.grid_lo) / (self.num_basis as f32 - 1.0);
        let inv_h = 1.0 / step;
        (0..self.num_basis)
            .map(|k| {
                let centre = self.grid_lo + k as f32 * step;
                let z = (x - centre) * inv_h;
                (-z * z).exp()
            })
            .collect()
    }

    /// Forward `y_j = Σ_i (base_{ij}·SiLU(x_i) + Σ_k coeff_{ijk}·RBF_k(x_i))`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features, "KAN: input width mismatch");
        let mut y = vec![0.0f32; self.out_features];
        for (i, &xi) in x.iter().enumerate()
        {
            let phi = self.rbf(xi);
            let s = silu(xi);
            for (j, yj) in y.iter_mut().enumerate()
            {
                let off = (i * self.out_features + j) * self.num_basis;
                let mut acc = self.base[i * self.out_features + j] * s;
                for (k, &bk) in phi.iter().enumerate()
                {
                    acc += self.coeffs[off + k] * bk;
                }
                *yj += acc;
            }
        }
        y
    }

    /// Gradient-descent fit on a dataset (`xs`: rows of length `in_features`,
    /// `ts`: rows of length `out_features`). Convex (linear in the coefficients);
    /// returns the final mean squared error.
    pub fn fit(&mut self, xs: &[Vec<f32>], ts: &[Vec<f32>], steps: usize, lr: f32) -> f32 {
        let s = xs.len();
        assert!(s > 0 && ts.len() == s, "KAN: empty / mismatched dataset");
        // Precompute basis features (they do not depend on the coefficients).
        let feats: Vec<(Vec<Vec<f32>>, Vec<f32>)> = xs
            .iter()
            .map(|x| {
                let rbf: Vec<Vec<f32>> = x.iter().map(|&xi| self.rbf(xi)).collect();
                let silus: Vec<f32> = x.iter().map(|&xi| silu(xi)).collect();
                (rbf, silus)
            })
            .collect();
        let inv_s = 1.0 / s as f32;
        let mut mse = 0.0f32;
        for _ in 0..steps
        {
            let mut g_coeffs = vec![0.0f32; self.coeffs.len()];
            let mut g_base = vec![0.0f32; self.base.len()];
            mse = 0.0;
            for (si, (rbf, silus)) in feats.iter().enumerate()
            {
                // Forward from the precomputed features.
                let mut y = vec![0.0f32; self.out_features];
                for i in 0..self.in_features
                {
                    for (j, yj) in y.iter_mut().enumerate()
                    {
                        let off = (i * self.out_features + j) * self.num_basis;
                        let mut acc = self.base[i * self.out_features + j] * silus[i];
                        for (k, &bk) in rbf[i].iter().enumerate()
                        {
                            acc += self.coeffs[off + k] * bk;
                        }
                        *yj += acc;
                    }
                }
                for j in 0..self.out_features
                {
                    let e = y[j] - ts[si][j];
                    mse += e * e;
                    let ge = 2.0 * e * inv_s;
                    for i in 0..self.in_features
                    {
                        g_base[i * self.out_features + j] += ge * silus[i];
                        let off = (i * self.out_features + j) * self.num_basis;
                        for (k, &bk) in rbf[i].iter().enumerate()
                        {
                            g_coeffs[off + k] += ge * bk;
                        }
                    }
                }
            }
            mse *= inv_s;
            for (c, g) in self.coeffs.iter_mut().zip(&g_coeffs)
            {
                *c -= lr * g;
            }
            for (b, g) in self.base.iter_mut().zip(&g_base)
            {
                *b -= lr * g;
            }
        }
        mse
    }

    /// Number of trainable parameters (`in·out·(num_basis + 1)`).
    pub fn num_params(&self) -> usize {
        self.coeffs.len() + self.base.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the grid dataset for `f(x0,x1) = sin(2·x0) + x1²` on `[-1,1]²`.
    fn additive_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n = 11usize;
        let mut xs = Vec::new();
        let mut ts = Vec::new();
        for a in 0..n
        {
            for b in 0..n
            {
                let x0 = -1.0 + 2.0 * a as f32 / (n as f32 - 1.0);
                let x1 = -1.0 + 2.0 * b as f32 / (n as f32 - 1.0);
                xs.push(vec![x0, x1]);
                ts.push(vec![(2.0 * x0).sin() + x1 * x1]);
            }
        }
        (xs, ts)
    }

    /// Best-fit **linear** model `y = w0·x0 + w1·x1 + b` MSE (gradient descent
    /// baseline) — what a KAN must beat on a nonlinear target.
    fn linear_baseline_mse(xs: &[Vec<f32>], ts: &[Vec<f32>]) -> f32 {
        let (mut w0, mut w1, mut b) = (0.0f32, 0.0f32, 0.0f32);
        let s = xs.len() as f32;
        let mut mse = 0.0;
        for _ in 0..5000
        {
            let (mut g0, mut g1, mut gb) = (0.0f32, 0.0f32, 0.0f32);
            mse = 0.0;
            for (x, t) in xs.iter().zip(ts)
            {
                let e = w0 * x[0] + w1 * x[1] + b - t[0];
                mse += e * e;
                g0 += 2.0 * e * x[0] / s;
                g1 += 2.0 * e * x[1] / s;
                gb += 2.0 * e / s;
            }
            mse /= s;
            w0 -= 0.3 * g0;
            w1 -= 0.3 * g1;
            b -= 0.3 * gb;
        }
        mse
    }

    /// **The KAN thesis, tested.** A single KAN layer fits the nonlinear additive
    /// target `sin(2·x0) + x1²` to a small error — far below the best linear model,
    /// which cannot represent the `sin`/`square` shapes. (Both fit the same data;
    /// the KAN's learnable per-edge activations are what make the difference.)
    #[test]
    fn kan_fits_additive_target_and_beats_linear() {
        let (xs, ts) = additive_dataset();
        let mut rng = PcgEngine::new(7);
        let mut kan = KanLayer::new(2, 1, 12, -1.2, 1.2, &mut rng);
        let mse = kan.fit(&xs, &ts, 4000, 0.3);
        let lin = linear_baseline_mse(&xs, &ts);
        assert!(mse < 0.02, "KAN did not fit: MSE {mse}");
        assert!(
            mse < 0.2 * lin,
            "KAN ({mse}) did not clearly beat the linear baseline ({lin})"
        );
    }

    /// The RBF basis is localised: each basis peaks (=1) at its own centre and is
    /// small far away; the fit and forward are bit-for-bit deterministic.
    #[test]
    fn kan_basis_localised_and_deterministic() {
        let rng = &mut PcgEngine::new(1);
        let kan = KanLayer::new(1, 1, 5, -1.0, 1.0, rng);
        // Centre 0 is at grid_lo = -1; its RBF peaks there.
        let at_centre = kan.rbf(-1.0);
        assert!(
            (at_centre[0] - 1.0).abs() < 1e-6,
            "RBF centre not 1: {}",
            at_centre[0]
        );
        assert!(at_centre[4] < 0.02, "far RBF not small: {}", at_centre[4]);

        let (xs, ts) = additive_dataset();
        let fit_once = || {
            let mut r = PcgEngine::new(3);
            let mut k = KanLayer::new(2, 1, 8, -1.2, 1.2, &mut r);
            let m = k.fit(&xs, &ts, 500, 0.2);
            (m, k.forward(&[0.3, -0.4]))
        };
        let (m1, y1) = fit_once();
        let (m2, y2) = fit_once();
        assert_eq!(m1.to_bits(), m2.to_bits());
        assert_eq!(y1[0].to_bits(), y2[0].to_bits());
        assert_eq!(kan.num_params(), 6); // in·out·num_basis + in·out = 5 + 1
    }
}
