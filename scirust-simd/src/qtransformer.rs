//! # Bloc décodeur **quantifié int8 (W8A8)** — inférence AMX (x86_64)
//!
//! Branche la couche linéaire quantifiée ([`crate::amx`]) dans le bloc décodeur
//! de [`crate::transformer`]. Les **six grosses projections** (`Wq`/`Wk`/`Wv`/
//! `Wo` de l'attention, `W1`/`W2` du FFN) — qui concentrent l'essentiel des
//! FLOPs — passent en **int8** :
//!
//! * **poids** quantifiés une fois, symétriquement, **par canal de sortie**
//!   (`scale_w[j] = max_i|W[i,j]| / 127`) → int8 + `scale_w` ;
//! * **activations** quantifiées dynamiquement à l'exécution, **par token
//!   (ligne)** (`scale_x[i] = max_j|X[i,j]| / 127`) → int8 + `scale_x` ;
//! * produit `int8·int8 → i32` par le **GEMM AMX** ([`crate::amx::amx_matmul_i8`],
//!   `_tile_dpbssd` sur silicium), puis **déquantification**
//!   `Y[i,j] = scale_x[i]·scale_w[j]·acc[i,j] (+ biais[j])`.
//!
//! Les parties **non-GEMM** (RMSNorm, RoPE, softmax d'attention, résidus,
//! activation SiLU) restent en `f32` — c'est le schéma d'inférence quantifiée
//! usuel : on quantifie les matmuls, pas les normalisations ni la softmax.
//!
//! Le bloc quantifié est comparé au bloc `f32` de référence
//! ([`crate::transformer::TransformerBlock`]) : sortie **proche** (tolérance de
//! quantification, ~1 %/matmul), pour un stockage des poids **÷4** (int8 vs f32)
//! et un GEMM accéléré AMX.

use crate::amx::{amx_int8_usable, amx_matmul_i8, amx_matmul_i8_prepacked, prepack_b_i8};
use crate::attention::multi_head_attention;
use crate::norm::rmsnorm;
use crate::transformer::rope_apply_heads;

/// Quantifie `x` (`m×k`, row-major) en int8 **par ligne** (symétrique) : pour
/// chaque ligne `i`, `scale[i] = max_j|x[i,j]| / 127`, `xq[i,j] =
/// round(x[i,j]/scale[i])`. Renvoie `(xq, scale)`. Ligne nulle ⇒ `scale = 1`.
pub fn quantize_i8_per_row(x: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(x.len(), m * k, "quantize_i8_per_row: shape");
    let mut xq = vec![0i8; m * k];
    let mut scale = vec![0f32; m];
    for i in 0..m
    {
        let row = &x[i * k..i * k + k];
        let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv = 1.0 / s;
        for (q, &v) in xq[i * k..i * k + k].iter_mut().zip(row)
        {
            *q = quant_round(v * inv);
        }
        scale[i] = s;
    }
    (xq, scale)
}

/// Quantifie `w` (`k×n`, row-major) en int8 **par colonne** (canal de sortie,
/// symétrique) : `scale[j] = max_i|w[i,j]| / 127`, `wq[i,j] =
/// round(w[i,j]/scale[j])`. Renvoie `(wq, scale)`. Colonne nulle ⇒ `scale = 1`.
pub fn quantize_i8_per_col(w: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(w.len(), k * n, "quantize_i8_per_col: shape");
    let mut scale = vec![0f32; n];
    for j in 0..n
    {
        let mut amax = 0f32;
        for i in 0..k
        {
            amax = amax.max(w[i * n + j].abs());
        }
        scale[j] = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    }
    let mut wq = vec![0i8; k * n];
    for i in 0..k
    {
        for j in 0..n
        {
            wq[i * n + j] = quant_round(w[i * n + j] / scale[j]);
        }
    }
    (wq, scale)
}

/// Arrondi au plus proche + saturation dans `[-127, 127]` (int8 symétrique).
#[inline]
fn quant_round(v: f32) -> i8 {
    let r = v.round();
    if r >= 127.0
    {
        127
    }
    else if r <= -127.0
    {
        -127
    }
    else
    {
        r as i8
    }
}

/// **Couche linéaire quantifiée** à poids int8 pré-quantifiés (par canal). Le
/// `forward` quantifie dynamiquement les activations (par token), fait le GEMM
/// AMX int8, puis déquantifie (+ biais optionnel).
pub struct QuantizedLinear {
    /// Poids int8 `k×n` (row-major), quantifiés par colonne (stockage canonique
    /// ÷4 vs `f32`).
    pub wq: Vec<i8>,
    /// Échelle de déquantification par colonne (`n`).
    pub w_scale: Vec<f32>,
    /// Biais `f32` optionnel (`n`), ajouté après déquantification.
    pub bias: Option<Vec<f32>>,
    pub k: usize,
    pub n: usize,
    /// `B` pré-empaqueté en disposition tuile AMX (cache d'accélération, construit
    /// seulement si AMX est utilisable) : évite de re-packer les poids à chaque
    /// `forward`. `None` ⇒ chemin `amx_matmul_i8` (AMX per-appel ou scalaire).
    b_packed: Option<Vec<i8>>,
}

impl QuantizedLinear {
    /// Quantifie une matrice de poids `f32` `k×n` (par colonne) + biais optionnel.
    pub fn from_f32(w: &[f32], k: usize, n: usize, bias: Option<&[f32]>) -> Self {
        let (wq, w_scale) = quantize_i8_per_col(w, k, n);
        let b_packed = if amx_int8_usable()
        {
            Some(prepack_b_i8(&wq, k, n))
        }
        else
        {
            None
        };
        QuantizedLinear {
            wq,
            w_scale,
            bias: bias.map(|b| b.to_vec()),
            k,
            n,
            b_packed,
        }
    }

    /// `Y[m×n] = dequant(Xq · Wq) (+ biais)`, activations quantifiées par ligne.
    /// `x` est `m×k` (row-major `f32`).
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "QuantizedLinear::forward: x shape");
        let (xq, x_scale) = quantize_i8_per_row(x, m, self.k);
        let acc = match &self.b_packed
        {
            Some(bp) => amx_matmul_i8_prepacked(&xq, bp, m, self.k, self.n),
            None => amx_matmul_i8(&xq, &self.wq, m, self.k, self.n),
        };
        let mut y = vec![0f32; m * self.n];
        for i in 0..m
        {
            let sx = x_scale[i];
            let arow = &acc[i * self.n..i * self.n + self.n];
            let yrow = &mut y[i * self.n..i * self.n + self.n];
            for j in 0..self.n
            {
                let mut val = sx * self.w_scale[j] * (arow[j] as f32);
                if let Some(b) = &self.bias
                {
                    val += b[j];
                }
                yrow[j] = val;
            }
        }
        y
    }
}

/// Bloc décodeur pre-norm à **projections quantifiées int8** (attention + FFN).
/// Miroir de [`crate::transformer::TransformerBlock`] : mêmes étapes, mais
/// `Wq`/`Wk`/`Wv`/`Wo`/`W1`/`W2` sont des [`QuantizedLinear`].
pub struct QuantizedTransformerBlock {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub wq: QuantizedLinear,
    pub wk: QuantizedLinear,
    pub wv: QuantizedLinear,
    pub wo: QuantizedLinear,
    pub w1: QuantizedLinear,
    pub w2: QuantizedLinear,
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub eps: f32,
    pub rope_base: f32,
    pub causal: bool,
}

impl QuantizedTransformerBlock {
    /// Construit le bloc quantifié depuis les mêmes poids `f32` qu'un
    /// [`crate::transformer::TransformerBlock`] (quantification des 6 matrices).
    #[allow(clippy::too_many_arguments)]
    pub fn from_f32(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        norm1: &[f32],
        norm2: &[f32],
        eps: f32,
        rope_base: f32,
        causal: bool,
    ) -> Self {
        let d = d_model;
        QuantizedTransformerBlock {
            d_model,
            n_heads,
            d_ff,
            wq: QuantizedLinear::from_f32(wq, d, d, None),
            wk: QuantizedLinear::from_f32(wk, d, d, None),
            wv: QuantizedLinear::from_f32(wv, d, d, None),
            wo: QuantizedLinear::from_f32(wo, d, d, None),
            w1: QuantizedLinear::from_f32(w1, d, d_ff, Some(b1)),
            w2: QuantizedLinear::from_f32(w2, d_ff, d, None),
            norm1: norm1.to_vec(),
            norm2: norm2.to_vec(),
            eps,
            rope_base,
            causal,
        }
    }

    /// Forward en place (prefill), analogue à
    /// [`crate::transformer::TransformerBlock::forward`] mais projections int8.
    pub fn forward(&self, x: &mut [f32], s: usize) {
        let d = self.d_model;
        let h = self.n_heads;
        assert_eq!(x.len(), s * d, "forward: x shape");
        assert_eq!(d % h, 0, "forward: d_model non divisible par n_heads");
        let dh = d / h;

        // ---- Attention (pre-norm) ----
        let mut hn = x.to_vec();
        rmsnorm(&mut hn, s, d, &self.norm1, self.eps);

        let mut q = self.wq.forward(&hn, s);
        let mut k = self.wk.forward(&hn, s);
        let v = self.wv.forward(&hn, s);

        rope_apply_heads(&mut q, s, h, dh, self.rope_base);
        rope_apply_heads(&mut k, s, h, dh, self.rope_base);

        let scale = 1.0 / (dh as f32).sqrt();
        let mut attn = vec![0.0f32; s * d];
        multi_head_attention(&q, s, s, h, dh, &k, &v, scale, self.causal, &mut attn);

        let o = self.wo.forward(&attn, s);
        for (xi, oi) in x.iter_mut().zip(&o)
        {
            *xi += *oi; // résidu
        }

        // ---- FFN (pre-norm) ----
        let mut hn2 = x.to_vec();
        rmsnorm(&mut hn2, s, d, &self.norm2, self.eps);

        // f1 = SiLU(hn2·W1 + b1) — GEMM int8 (biais fusionné à la déquant), puis
        // SiLU scalaire (non-linéarité conservée en f32).
        let mut f1 = self.w1.forward(&hn2, s);
        for v in f1.iter_mut()
        {
            *v = crate::activations::silu_scalar(*v);
        }
        let f2 = self.w2.forward(&f1, s);
        for (xi, fi) in x.iter_mut().zip(&f2)
        {
            *xi += *fi; // résidu
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::transformer::TransformerBlock;

    // AMX est touché indirectement (via qlinear) ⇒ sérialiser (cf. amx::tests).
    static AMX_LOCK: Mutex<()> = Mutex::new(());

    fn mk(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
            .collect()
    }

    #[test]
    fn quant_dequant_roundtrip_is_close() {
        // La quantification par ligne puis déquantification approche l'identité
        // à ~1 lsb près (erreur relative bornée par 1/127 sur l'amplitude max).
        let (m, k) = (5usize, 40usize);
        let x = mk(m * k, 1.0);
        let (xq, sc) = quantize_i8_per_row(&x, m, k);
        for i in 0..m
        {
            for j in 0..k
            {
                let approx = sc[i] * (xq[i * k + j] as f32);
                assert!(
                    (approx - x[i * k + j]).abs() <= sc[i] * 0.5 + 1e-6,
                    "i={i} j={j}"
                );
            }
        }
    }

    #[test]
    fn quantized_block_close_to_f32_block() {
        let _guard = AMX_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (s, d, h, dff) = (6usize, 32usize, 4usize, 64usize);
        let eps = 1e-5f32;
        let base = 10000.0f32;
        let wq = mk(d * d, 1.0);
        let wk = mk(d * d, 2.0);
        let wv = mk(d * d, 3.0);
        let wo = mk(d * d, 4.0);
        let w1 = mk(d * dff, 5.0);
        let b1 = mk(dff, 6.0);
        let w2 = mk(dff * d, 7.0);
        let norm1: Vec<f32> = (0..d).map(|i| 1.0 + i as f32 * 0.01).collect();
        let norm2: Vec<f32> = (0..d).map(|i| 0.9 + i as f32 * 0.005).collect();
        let x0: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.05).cos()).collect();

        // Référence f32.
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
        let mut f32_out = x0.clone();
        block.forward(&mut f32_out, s);

        // Bloc quantifié.
        let qblock = QuantizedTransformerBlock::from_f32(
            d, h, dff, &wq, &wk, &wv, &wo, &w1, &b1, &w2, &norm1, &norm2, eps, base, true,
        );
        let mut q_out = x0.clone();
        qblock.forward(&mut q_out, s);

        // Erreur relative globale (RMS) < quelques % : l'int8 W8A8 préserve la
        // sortie du bloc à la tolérance de quantification près.
        let mut num = 0f32;
        let mut den = 0f32;
        for i in 0..s * d
        {
            num += (q_out[i] - f32_out[i]).powi(2);
            den += f32_out[i].powi(2);
        }
        let rel = (num / den).sqrt();
        assert!(rel < 0.05, "erreur relative RMS trop grande : {rel}");
    }
}
