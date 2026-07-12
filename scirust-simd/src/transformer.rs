//! # Bloc Transformer décodeur (assemblage end-to-end, x86_64)
//!
//! Démonstration que les primitives de `scirust-simd` s'imbriquent en un vrai
//! bloc décodeur **pre-norm** (style LLaMA/GPT-NeoX) :
//!
//! ```text
//! h  = RMSNorm(x, γ₁)
//! q,k,v = h·Wq, h·Wk, h·Wv           (projections, GEMM tuilé)
//! q,k   = RoPE(q), RoPE(k)            (par tête)
//! a  = MultiHeadAttention(q,k,v, causal)
//! x  = x + a·Wo                       (projection de sortie + résidu)
//! h₂ = RMSNorm(x, γ₂)
//! f  = SiLU(h₂·W₁ + b₁)·W₂            (FFN, dense fusionné + GEMM)
//! x  = x + f                          (résidu)
//! ```
//!
//! Chaque étape réutilise le noyau dédié : [`crate::gemm::sgemm_tiled`] et
//! [`crate::gemm::sgemm_bias_act`], [`crate::attention::multi_head_attention`],
//! [`crate::norm::rmsnorm`], et le RoPE par tête ci-dessous. Le tout hérite du
//! dispatch runtime (AVX-512 → … → scalaire) et du repli portable.

use crate::attention::multi_head_attention;
use crate::gemm::{Activation, sgemm_bias_act, sgemm_tiled};
use crate::matrix::view::{MatrixView, MatrixViewMut};
use crate::norm::rmsnorm;

/// Poids d'un bloc décodeur (tous empruntés, row-major).
///
/// Conventions de formes (`d = d_model`, `dff = d_ff`) :
/// * `wq`,`wk`,`wv`,`wo` : `d×d` (projection `y = x·W`).
/// * `w1` : `d×dff`, `b1` : `dff`, `w2` : `dff×d` (FFN).
/// * `norm1`,`norm2` : `d` (gains RMSNorm).
pub struct TransformerBlock<'a> {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub norm1: &'a [f32],
    pub norm2: &'a [f32],
    pub eps: f32,
    pub rope_base: f32,
    pub causal: bool,
}

/// Applique RoPE **par tête** à `x` (`s×(h·d_head)`, têtes concaténées) en place.
/// Chaque tête occupe `d_head` colonnes contiguës ; la position d'une ligne est
/// son indice `r`. `d_head` doit être pair.
pub fn rope_apply_heads(x: &mut [f32], s: usize, h: usize, d_head: usize, base: f32) {
    assert_eq!(d_head % 2, 0, "rope_apply_heads: d_head doit être pair");
    let dm = h * d_head;
    assert_eq!(x.len(), s * dm, "rope_apply_heads: shape mismatch");
    let half = d_head / 2;
    for r in 0..s
    {
        let pos = r as f32;
        for hh in 0..h
        {
            let off = r * dm + hh * d_head;
            for i in 0..half
            {
                let theta = base.powf(-2.0 * i as f32 / d_head as f32);
                let (sin, cos) = (pos * theta).sin_cos();
                let a = x[off + 2 * i];
                let b = x[off + 2 * i + 1];
                x[off + 2 * i] = a * cos - b * sin;
                x[off + 2 * i + 1] = a * sin + b * cos;
            }
        }
    }
}

impl TransformerBlock<'_> {
    /// Forward en place : `x` est le flux résiduel `s×d_model` (row-major),
    /// mis à jour par le bloc.
    pub fn forward(&self, x: &mut [f32], s: usize) {
        let d = self.d_model;
        let h = self.n_heads;
        let dff = self.d_ff;
        assert_eq!(x.len(), s * d, "forward: x shape");
        assert_eq!(d % h, 0, "forward: d_model non divisible par n_heads");
        let dh = d / h;

        // ---- Sous-bloc attention (pre-norm) ----
        let mut hn = x.to_vec();
        rmsnorm(&mut hn, s, d, self.norm1, self.eps);

        let mut q = vec![0.0f32; s * d];
        let mut k = vec![0.0f32; s * d];
        let mut v = vec![0.0f32; s * d];
        proj(&hn, s, d, self.wq, d, &mut q);
        proj(&hn, s, d, self.wk, d, &mut k);
        proj(&hn, s, d, self.wv, d, &mut v);

        rope_apply_heads(&mut q, s, h, dh, self.rope_base);
        rope_apply_heads(&mut k, s, h, dh, self.rope_base);

        let scale = 1.0 / (dh as f32).sqrt();
        let mut attn = vec![0.0f32; s * d];
        multi_head_attention(&q, s, s, h, dh, &k, &v, scale, self.causal, &mut attn);

        let mut o = vec![0.0f32; s * d];
        proj(&attn, s, d, self.wo, d, &mut o);
        for (xi, oi) in x.iter_mut().zip(&o)
        {
            *xi += *oi; // résidu
        }

        // ---- Sous-bloc FFN (pre-norm) ----
        let mut hn2 = x.to_vec();
        rmsnorm(&mut hn2, s, d, self.norm2, self.eps);

        // f1 = SiLU(hn2·W1 + b1)  (s×dff), dense fusionné.
        let mut f1 = vec![0.0f32; s * dff];
        sgemm_bias_act(
            1.0,
            MatrixView::new(&hn2, s, d),
            MatrixView::new(self.w1, d, dff),
            self.b1,
            Activation::Silu,
            MatrixViewMut::new(&mut f1, s, dff),
        );
        // f2 = f1·W2  (s×d).
        let mut f2 = vec![0.0f32; s * d];
        proj(&f1, s, dff, self.w2, d, &mut f2);
        for (xi, fi) in x.iter_mut().zip(&f2)
        {
            *xi += *fi; // résidu
        }
    }
}

/// Projection linéaire `out = a(rows×k)·w(k×n)` via le GEMM tuilé.
fn proj(a: &[f32], rows: usize, k: usize, w: &[f32], n: usize, out: &mut [f32]) {
    sgemm_tiled(
        1.0,
        MatrixView::new(a, rows, k),
        MatrixView::new(w, k, n),
        0.0,
        MatrixViewMut::new(out, rows, n),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Référence scalaire entièrement indépendante ----

    fn matmul(a: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a[i * k + p] * w[p * n + j];
                }
                o[i * n + j] = acc;
            }
        }
        o
    }

    fn rmsnorm_ref(x: &[f32], s: usize, d: usize, g: &[f32], eps: f32) -> Vec<f32> {
        let mut o = vec![0.0f32; s * d];
        for r in 0..s
        {
            let row = &x[r * d..r * d + d];
            let ss: f32 = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            for j in 0..d
            {
                o[r * d + j] = row[j] * inv * g[j];
            }
        }
        o
    }

    fn rope_ref(x: &mut [f32], s: usize, h: usize, dh: usize, base: f32) {
        let dm = h * dh;
        for r in 0..s
        {
            let pos = r as f32;
            for hh in 0..h
            {
                let off = r * dm + hh * dh;
                for i in 0..dh / 2
                {
                    let theta = base.powf(-2.0 * i as f32 / dh as f32);
                    let (sin, cos) = (pos * theta).sin_cos();
                    let a = x[off + 2 * i];
                    let b = x[off + 2 * i + 1];
                    x[off + 2 * i] = a * cos - b * sin;
                    x[off + 2 * i + 1] = a * sin + b * cos;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mha_causal_ref(q: &[f32], k: &[f32], v: &[f32], s: usize, h: usize, dh: usize) -> Vec<f32> {
        let dm = h * dh;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut out = vec![0.0f32; s * dm];
        for hh in 0..h
        {
            let off = hh * dh;
            for i in 0..s
            {
                // scores causaux 0..=i.
                let mut row = vec![0.0f32; i + 1];
                for (j, r) in row.iter_mut().enumerate()
                {
                    let mut acc = 0.0f32;
                    for e in 0..dh
                    {
                        acc += q[i * dm + off + e] * k[j * dm + off + e];
                    }
                    *r = scale * acc;
                }
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for r in row.iter_mut()
                {
                    *r = (*r - m).exp();
                    sum += *r;
                }
                for e in 0..dh
                {
                    let mut acc = 0.0f32;
                    for (j, &p) in row.iter().enumerate()
                    {
                        acc += p * v[j * dm + off + e];
                    }
                    out[i * dm + off + e] = acc / sum;
                }
            }
        }
        out
    }

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    #[test]
    fn transformer_block_matches_scalar_reference() {
        let (s, d, h, dff) = (6usize, 8usize, 2usize, 16usize);
        let dh = d / h;
        let eps = 1e-5f32;
        let base = 10000.0f32;

        // Poids déterministes.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
                .collect()
        };
        let wq = mk(d * d, 1.0);
        let wk = mk(d * d, 2.0);
        let wv = mk(d * d, 3.0);
        let wo = mk(d * d, 4.0);
        let w1 = mk(d * dff, 5.0);
        let b1 = mk(dff, 6.0);
        let w2 = mk(dff * d, 7.0);
        let norm1: Vec<f32> = (0..d).map(|i| 1.0 + i as f32 * 0.01).collect();
        let norm2: Vec<f32> = (0..d).map(|i| 0.9 + i as f32 * 0.02).collect();

        let x0: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.05).cos()).collect();

        // --- Sous le test ---
        let block = TransformerBlock {
            d_model: d,
            n_heads: h,
            d_ff: dff,
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            w1: &w1,
            b1: &b1,
            w2: &w2,
            norm1: &norm1,
            norm2: &norm2,
            eps,
            rope_base: base,
            causal: true,
        };
        let mut got = x0.clone();
        block.forward(&mut got, s);

        // --- Référence scalaire indépendante ---
        let mut x = x0.clone();
        // attention
        let hn = rmsnorm_ref(&x, s, d, &norm1, eps);
        let mut q = matmul(&hn, s, d, &wq, d);
        let mut k = matmul(&hn, s, d, &wk, d);
        let v = matmul(&hn, s, d, &wv, d);
        rope_ref(&mut q, s, h, dh, base);
        rope_ref(&mut k, s, h, dh, base);
        let a = mha_causal_ref(&q, &k, &v, s, h, dh);
        let o = matmul(&a, s, d, &wo, d);
        for i in 0..s * d
        {
            x[i] += o[i];
        }
        // FFN
        let hn2 = rmsnorm_ref(&x, s, d, &norm2, eps);
        let mut f1 = matmul(&hn2, s, d, &w1, dff);
        for i in 0..s
        {
            for j in 0..dff
            {
                f1[i * dff + j] = silu(f1[i * dff + j] + b1[j]);
            }
        }
        let f2 = matmul(&f1, s, dff, &w2, d);
        for i in 0..s * d
        {
            x[i] += f2[i];
        }

        // --- Comparaison ---
        for i in 0..s * d
        {
            let tol = 2e-3 * (1.0 + x[i].abs());
            assert!(
                (got[i] - x[i]).abs() <= tol,
                "idx {i}: got {} vs ref {}",
                got[i],
                x[i]
            );
        }
    }

    #[test]
    fn rope_apply_heads_is_per_head_rotation() {
        let (s, h, dh) = (4usize, 3usize, 4usize);
        let dm = h * dh;
        let x0: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut got = x0.clone();
        rope_apply_heads(&mut got, s, h, dh, 10000.0);
        let mut want = x0.clone();
        rope_ref(&mut want, s, h, dh, 10000.0);
        for i in 0..s * dm
        {
            assert!((got[i] - want[i]).abs() <= 1e-5, "idx {i}");
        }
    }
}
