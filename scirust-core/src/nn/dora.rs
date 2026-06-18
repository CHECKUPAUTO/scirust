//! **DoRA — Weight-Decomposed Low-Rank Adaptation** (Liu et al., ICML 2024,
//! arXiv:2402.09353).
//!
//! LoRA adapts a frozen weight `W₀` by an additive low-rank term `BA`, which
//! couples *how much* and *in which direction* each column changes. DoRA instead
//! decomposes the weight into a **magnitude** and a **direction** and adapts them
//! separately: it normalises each column of `V = W₀ + BA` to unit length and
//! rescales it by a learned per-column magnitude `m`,
//!
//! ```text
//! W' = m ⊙ (W₀ + BA) / ‖W₀ + BA‖_col
//! ```
//!
//! so the direction is steered by the (low-rank) `BA` while the magnitude is a
//! separate, cheap, learnable vector. Initialised at `B = 0` and `m = ‖W₀‖_col`,
//! DoRA reproduces the frozen layer **exactly** (tested), so adaptation starts from
//! the pretrained function. Only `m`, `A`, `B` train; `W₀` stays frozen.
//!
//! The column normalisation is differentiated in closed form and the gradients are
//! finite-difference-checked. Pure, deterministic `f32` arithmetic; `(d, k)` is
//! `(out, in)`, the LoRA rank is `r`.

use crate::nn::rng::PcgEngine;

/// Column-normalisation epsilon (guards an all-zero column of `V`).
const DORA_EPS: f32 = 1e-12;

/// `B(d×r) · A(r×k) → (d×k)`, row-major.
fn matmul(b: &[f32], a: &[f32], d: usize, r: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d * k];
    for i in 0..d
    {
        for p in 0..r
        {
            let bip = b[i * r + p];
            for j in 0..k
            {
                out[i * k + j] += bip * a[p * k + j];
            }
        }
    }
    out
}

/// The DoRA effective weight `W' = m ⊙ (W₀+BA)/‖W₀+BA‖_col` (`d×k`, row-major),
/// alongside `V = W₀+BA` and the column norms — the latter two are reused by the
/// backward pass.
fn effective_weight(
    w0: &[f32],
    a: &[f32],
    b: &[f32],
    m: &[f32],
    d: usize,
    k: usize,
    r: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ba = matmul(b, a, d, r, k);
    let mut v = vec![0.0f32; d * k];
    for i in 0..d * k
    {
        v[i] = w0[i] + ba[i];
    }
    // Column L2 norms.
    let mut norm = vec![0.0f32; k];
    for j in 0..k
    {
        let mut s = 0.0f32;
        for i in 0..d
        {
            s += v[i * k + j] * v[i * k + j];
        }
        norm[j] = s.sqrt().max(DORA_EPS);
    }
    // W'[i,j] = m_j · V[i,j] / norm_j.
    let mut w = vec![0.0f32; d * k];
    for i in 0..d
    {
        for j in 0..k
        {
            w[i * k + j] = m[j] * v[i * k + j] / norm[j];
        }
    }
    (w, v, norm)
}

/// **DoRA-adapted linear weight**: a frozen `W₀ (d×k)` decomposed into a learnable
/// per-column magnitude `m (k)` and a low-rank direction update `B (d×r)·A (r×k)`.
/// Only `m`, `A`, `B` are trainable. See the [module docs](self).
pub struct DoraLinear {
    /// Frozen pretrained weight, `d×k` row-major.
    pub w0: Vec<f32>,
    /// LoRA `A`, `r×k`.
    pub a: Vec<f32>,
    /// LoRA `B`, `d×r` (initialised to zero ⇒ starts equal to `W₀`).
    pub b: Vec<f32>,
    /// Per-column magnitude, length `k` (initialised to the column norms of `W₀`).
    pub m: Vec<f32>,
    /// Output dimension.
    pub d: usize,
    /// Input dimension.
    pub k: usize,
    /// LoRA rank.
    pub r: usize,
}

impl DoraLinear {
    /// New DoRA layer over a frozen `w0` (`d×k`). `m` is initialised to the column
    /// norms of `W₀`, `A` to small seeded noise, and `B` to zero — so the initial
    /// effective weight equals `W₀` exactly.
    pub fn new(w0: &[f32], d: usize, k: usize, rank: usize, rng: &mut PcgEngine) -> Self {
        assert_eq!(w0.len(), d * k, "DoRA: w0 must be d*k");
        let mut m = vec![0.0f32; k];
        for j in 0..k
        {
            let mut s = 0.0f32;
            for i in 0..d
            {
                s += w0[i * k + j] * w0[i * k + j];
            }
            m[j] = s.sqrt();
        }
        // A ~ small noise, B = 0 (LoRA-style init: zero perturbation).
        let scale = 1.0 / (k as f32).sqrt();
        let a: Vec<f32> = (0..rank * k).map(|_| rng.float_signed() * scale).collect();
        let b = vec![0.0f32; d * rank];
        Self {
            w0: w0.to_vec(),
            a,
            b,
            m,
            d,
            k,
            r: rank,
        }
    }

    /// The current effective weight `W'` (`d×k`).
    pub fn effective_weight(&self) -> Vec<f32> {
        effective_weight(&self.w0, &self.a, &self.b, &self.m, self.d, self.k, self.r).0
    }

    /// Apply the layer to an input `x` (length `k`): `y = W' · x` (length `d`).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.k, "DoRA forward: x must be length k");
        let w = self.effective_weight();
        let mut y = vec![0.0f32; self.d];
        for i in 0..self.d
        {
            let mut s = 0.0f32;
            for j in 0..self.k
            {
                s += w[i * self.k + j] * x[j];
            }
            y[i] = s;
        }
        y
    }

    /// Closed-form gradients of a scalar loss w.r.t. the trainable parameters,
    /// given the upstream gradient `gw = ∂L/∂W'` (`d×k`). Returns `(dm, dA, dB)`
    /// (shapes `k`, `r×k`, `d×r`). Differentiates the column normalisation:
    /// with `u = V/‖V‖_col` and `s_j = Σ_i gw[i,j] u[i,j]`,
    /// `∂L/∂V[i,j] = (m_j/‖V‖_j)·(gw[i,j] − u[i,j]·s_j)` and `∂L/∂m_j = s_j`.
    pub fn grads(&self, gw: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(gw.len(), self.d * self.k, "DoRA grads: gw must be d*k");
        let (d, k, r) = (self.d, self.k, self.r);
        let (_, v, norm) = effective_weight(&self.w0, &self.a, &self.b, &self.m, d, k, r);
        // u = V / norm (column-normalised), s_j = Σ_i gw·u, dm_j = s_j.
        let mut dm = vec![0.0f32; k];
        for j in 0..k
        {
            let mut s = 0.0f32;
            for i in 0..d
            {
                s += gw[i * k + j] * v[i * k + j] / norm[j];
            }
            dm[j] = s;
        }
        // dV[i,j] = (m_j/norm_j)·(gw[i,j] − u[i,j]·s_j).
        let mut dv = vec![0.0f32; d * k];
        for i in 0..d
        {
            for j in 0..k
            {
                let u = v[i * k + j] / norm[j];
                dv[i * k + j] = (self.m[j] / norm[j]) * (gw[i * k + j] - u * dm[j]);
            }
        }
        // dB = dV·Aᵀ (d×r); dA = Bᵀ·dV (r×k).
        let mut db = vec![0.0f32; d * r];
        for i in 0..d
        {
            for p in 0..r
            {
                let mut s = 0.0f32;
                for j in 0..k
                {
                    s += dv[i * k + j] * self.a[p * k + j];
                }
                db[i * r + p] = s;
            }
        }
        let mut da = vec![0.0f32; r * k];
        for p in 0..r
        {
            for j in 0..k
            {
                let mut s = 0.0f32;
                for i in 0..d
                {
                    s += self.b[i * r + p] * dv[i * k + j];
                }
                da[p * k + j] = s;
            }
        }
        (dm, da, db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **DoRA starts equal to the frozen layer.** With `B = 0` and
    /// `m = ‖W₀‖_col`, the effective weight reproduces `W₀` exactly, so adaptation
    /// begins from the pretrained function.
    #[test]
    fn init_reproduces_base_weight() {
        let (d, k, r) = (4usize, 3usize, 2usize);
        let mut rng = PcgEngine::new(1);
        let w0: Vec<f32> = (0..d * k).map(|_| rng.float_signed()).collect();
        let dora = DoraLinear::new(&w0, d, k, r, &mut rng);
        let w = dora.effective_weight();
        for (a, b) in w.iter().zip(&w0)
        {
            assert!((a - b).abs() < 1e-5, "init weight {a} ≠ base {b}");
        }
        // And the forward map matches the base linear map.
        let x: Vec<f32> = (0..k).map(|i| i as f32 - 1.0).collect();
        let y = dora.forward(&x);
        for (i, &yi) in y.iter().enumerate()
        {
            let want: f32 = (0..k).map(|j| w0[i * k + j] * x[j]).sum();
            assert!((yi - want).abs() < 1e-5);
        }
    }

    /// **The DoRA gradients are correct**: the closed-form `(dm, dA, dB)` for
    /// `L = Σ W'²` match central finite differences through the column
    /// normalisation, at generic (non-zero `B`) parameters.
    #[test]
    fn gradient_check() {
        let (d, k, r) = (4usize, 3usize, 2usize);
        let mut rng = PcgEngine::new(7);
        let w0: Vec<f32> = (0..d * k).map(|_| rng.float_signed()).collect();
        let a: Vec<f32> = (0..r * k).map(|_| rng.float_signed() * 0.5).collect();
        let b: Vec<f32> = (0..d * r).map(|_| rng.float_signed() * 0.5).collect();
        let m: Vec<f32> = (0..k).map(|_| 0.5 + rng.float() * 0.5).collect();
        let dora = DoraLinear {
            w0: w0.clone(),
            a: a.clone(),
            b: b.clone(),
            m: m.clone(),
            d,
            k,
            r,
        };

        let loss_of = |a: &[f32], b: &[f32], m: &[f32]| -> f32 {
            let (w, _, _) = effective_weight(&w0, a, b, m, d, k, r);
            w.iter().map(|&x| x * x).sum()
        };
        let w = dora.effective_weight();
        let gw: Vec<f32> = w.iter().map(|&x| 2.0 * x).collect(); // ∂(ΣW'²)/∂W'
        let (dm, da, db) = dora.grads(&gw);

        let eps = 1e-3f32;
        let check = |analytic: &[f32], base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic[i]).abs() < 3e-2,
                    "grad {i}: numeric {num}, analytic {}",
                    analytic[i]
                );
            }
        };
        check(&dm, &m, &|p| loss_of(&a, &b, p));
        check(&da, &a, &|p| loss_of(p, &b, &m));
        check(&db, &b, &|p| loss_of(&a, p, &m));
    }

    /// **DoRA trains.** Gradient descent on `(m, A, B)` (with `W₀` frozen) recovers
    /// a DoRA-generated target weight, driving the fitting loss down by orders of
    /// magnitude — and the run is bit-for-bit deterministic.
    #[test]
    fn fits_target_and_is_deterministic() {
        let run = || -> (f32, f32) {
            let (d, k, r) = (5usize, 4usize, 2usize);
            let mut rng = PcgEngine::new(21);
            let w0: Vec<f32> = (0..d * k).map(|_| rng.float_signed()).collect();
            // A reachable target: another DoRA's effective weight.
            let a_t: Vec<f32> = (0..r * k).map(|_| rng.float_signed() * 0.6).collect();
            let b_t: Vec<f32> = (0..d * r).map(|_| rng.float_signed() * 0.6).collect();
            let m_t: Vec<f32> = (0..k).map(|_| 0.5 + rng.float()).collect();
            let (target, _, _) = effective_weight(&w0, &a_t, &b_t, &m_t, d, k, r);

            let mut dora = DoraLinear::new(&w0, d, k, r, &mut rng);
            let loss = |dora: &DoraLinear| -> f32 {
                dora.effective_weight()
                    .iter()
                    .zip(&target)
                    .map(|(&w, &t)| (w - t) * (w - t))
                    .sum()
            };
            let init = loss(&dora);
            let lr = 0.2f32;
            for _ in 0..4000
            {
                let w = dora.effective_weight();
                let gw: Vec<f32> = w
                    .iter()
                    .zip(&target)
                    .map(|(&wi, &ti)| 2.0 * (wi - ti))
                    .collect();
                let (dm, da, db) = dora.grads(&gw);
                for (mi, g) in dora.m.iter_mut().zip(&dm)
                {
                    *mi -= lr * g;
                }
                for (ai, g) in dora.a.iter_mut().zip(&da)
                {
                    *ai -= lr * g;
                }
                for (bi, g) in dora.b.iter_mut().zip(&db)
                {
                    *bi -= lr * g;
                }
            }
            (init, loss(&dora))
        };
        let (init, final_loss) = run();
        assert!(
            final_loss < 0.01 * init,
            "DoRA did not fit: {init} → {final_loss}"
        );
        assert_eq!(run().1.to_bits(), final_loss.to_bits());
    }
}
