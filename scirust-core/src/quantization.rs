//! Quantification int8/int4 pour inférence sur matériel modeste.
//!
//! Implémente la quantification symétrique par canal.

/// Quantifie un tenseur fp32 en int8 par canal.
///
/// Retourne les valeurs quantifiées et le scale utilisé.
pub fn quantize_tensor(fp32: &[f32], scale: f32) -> Vec<i8> {
    fp32.iter()
        .map(|&x| {
            let q = (x / scale).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Déquantifie un tenseur int8 en fp32.
pub fn dequantize_tensor(int8: &[i8], scale: f32) -> Vec<f32> {
    int8.iter().map(|&x| x as f32 * scale).collect()
}

/// Calcule un scale optimal pour quantification symétrique.
pub fn compute_scale(fp32: &[f32]) -> f32 {
    let max_abs = fp32.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 }
}

/// Matmul int8 × int8 → i32.
pub fn matmul_int8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut result = vec![0i32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            // Accumulate in i64: each i8*i8 product fits in i32, but summing
            // more than ~133k of them overflows i32 (a debug-mode panic and a
            // silent wraparound → wrong result in release). i64 holds any
            // realistic contraction length; the final value is saturated into
            // the i32 output range instead of wrapping.
            let mut sum = 0i64;
            for kk in 0..k
            {
                sum += a[i * k + kk] as i64 * b[kk * n + j] as i64;
            }
            result[i * n + j] = sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let original: Vec<f32> = vec![-1.5, 0.0, 0.5, 2.3, -0.8];
        let scale = compute_scale(&original);
        let quantized = quantize_tensor(&original, scale);
        let recovered = dequantize_tensor(&quantized, scale);

        for (orig, rec) in original.iter().zip(recovered.iter())
        {
            let error = (orig - rec).abs();
            assert!(error < scale * 1.5, "error {} exceeds threshold", error);
        }
    }

    #[test]
    fn test_quantize_clamping() {
        let large_values: Vec<f32> = vec![500.0, -500.0, 0.0];
        let scale = compute_scale(&large_values);
        let quantized = quantize_tensor(&large_values, scale);
        // Symmetric quantization maps ±max_abs to ±127 and zero to zero.
        assert_eq!(quantized, vec![127, -127, 0]);
    }

    #[test]
    fn test_matmul_int8() {
        // 2x3 * 3x2 = 2x2
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let b: Vec<i8> = vec![7, 8, 9, 10, 11, 12];
        let result = matmul_int8(&a, &b, 2, 3, 2);
        // 1*7 + 2*9 + 3*11 = 58, 1*8 + 2*10 + 3*12 = 64
        // 4*7 + 5*9 + 6*11 = 139, 4*8 + 5*10 + 6*12 = 154
        assert_eq!(result, vec![58, 64, 139, 154]);
    }
}

// ----- Extension embarquee : per-channel + Linear quantifie deterministe -----

/// Quantifie une matrice de poids Linear (in_features, out_features), row-major,
/// avec un scale par canal de sortie (colonne o). Schema symetrique int8.
/// Retourne (poids_int8, scales) avec scales.len() == out_features.
pub fn quantize_per_channel(
    weight: &[f32],
    in_features: usize,
    out_features: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut scales = vec![1.0f32; out_features];
    for o in 0..out_features
    {
        let mut max_abs = 0.0f32;
        for i in 0..in_features
        {
            let v = weight[i * out_features + o].abs();
            if v > max_abs
            {
                max_abs = v;
            }
        }
        scales[o] = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    }
    let mut q = vec![0i8; in_features * out_features];
    for i in 0..in_features
    {
        for o in 0..out_features
        {
            let val = (weight[i * out_features + o] / scales[o]).round();
            q[i * out_features + o] = val.clamp(-128.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// Forward d'une couche Linear quantifiee, calque exact de
/// input.matmul(weight).add_bias(bias).
/// input (batch, in_features) f32 ; w_q (in_features, out_features) int8 ;
/// w_scales per-canal (len out_features) ; bias (out_features) f32.
/// Schema W8A8-dynamique : entree quantifiee per-tensor symetrique a la volee.
/// Deterministe : matmul entier en ordre fixe + ops f32 en ordre fixe.
pub fn quantized_linear_forward(
    input: &[f32],
    batch: usize,
    in_features: usize,
    w_q: &[i8],
    w_scales: &[f32],
    bias: &[f32],
    out_features: usize,
) -> Vec<f32> {
    // Validate lengths up front (matching the sibling quantized kernels) so a
    // caller mismatch is a clear error rather than an opaque out-of-bounds panic
    // deep inside matmul_int8 / the scale loop.
    assert_eq!(
        input.len(),
        batch * in_features,
        "quantized_linear_forward: input.len() {} != batch*in_features {}",
        input.len(),
        batch * in_features
    );
    assert_eq!(
        w_q.len(),
        in_features * out_features,
        "quantized_linear_forward: w_q.len() {} != in_features*out_features {}",
        w_q.len(),
        in_features * out_features
    );
    assert_eq!(
        w_scales.len(),
        out_features,
        "quantized_linear_forward: w_scales.len() {} != out_features {}",
        w_scales.len(),
        out_features
    );
    assert_eq!(
        bias.len(),
        out_features,
        "quantized_linear_forward: bias.len() {} != out_features {}",
        bias.len(),
        out_features
    );
    let scale_in = compute_scale(input);
    let x_q = quantize_tensor(input, scale_in);
    let acc = matmul_int8(&x_q, w_q, batch, in_features, out_features);
    let mut out = vec![0.0f32; batch * out_features];
    for b in 0..batch
    {
        for o in 0..out_features
        {
            out[b * out_features + o] =
                acc[b * out_features + o] as f32 * scale_in * w_scales[o] + bias[o];
        }
    }
    out
}

#[cfg(test)]
mod tests_quant_linear {
    use super::*;

    #[test]
    fn test_quantize_per_channel_scales() {
        let w = vec![1.0, -2.0, 0.5, 4.0, 1.0, -0.25]; // (in=2, out=3)
        let (q, scales) = quantize_per_channel(&w, 2, 3);
        assert_eq!(q.len(), 6);
        assert_eq!(scales.len(), 3);
        assert!((scales[0] - 4.0 / 127.0).abs() < 1e-9);
        assert!((scales[1] - 2.0 / 127.0).abs() < 1e-9);
        assert!((scales[2] - 0.5 / 127.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "input.len()")]
    fn quantized_linear_forward_rejects_length_mismatch() {
        // input should be batch*in_features = 2*4 = 8, but we pass 6.
        let input = vec![0.0f32; 6];
        let w_q = vec![0i8; 12]; // 4*3
        let w_scales = vec![1.0f32; 3];
        let bias = vec![0.0f32; 3];
        let _ = quantized_linear_forward(&input, 2, 4, &w_q, &w_scales, &bias, 3);
    }

    #[test]
    fn test_quantized_linear_matches_fp32() {
        let (in_f, out_f, batch) = (4usize, 3usize, 2usize);
        let weight: Vec<f32> = vec![
            0.10, -0.20, 0.30, 0.40, 0.05, -0.10, -0.30, 0.20, 0.15, 0.25, -0.05, 0.10,
        ];
        let bias: Vec<f32> = vec![0.01, -0.02, 0.03];
        let input: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.4, 0.0, 0.6];
        let mut reference = vec![0.0f32; batch * out_f];
        for b in 0..batch
        {
            for o in 0..out_f
            {
                let mut s = 0.0f32;
                for i in 0..in_f
                {
                    s += input[b * in_f + i] * weight[i * out_f + o];
                }
                reference[b * out_f + o] = s + bias[o];
            }
        }
        let (w_q, w_scales) = quantize_per_channel(&weight, in_f, out_f);
        let out = quantized_linear_forward(&input, batch, in_f, &w_q, &w_scales, &bias, out_f);
        for (r, o) in reference.iter().zip(out.iter())
        {
            assert!((r - o).abs() < 0.05, "ecart trop grand: ref={} q={}", r, o);
        }
    }

    #[test]
    fn test_quantized_linear_deterministic() {
        let (in_f, out_f, batch) = (8usize, 5usize, 3usize);
        let weight: Vec<f32> = (0..in_f * out_f).map(|k| (k as f32 * 0.13).sin()).collect();
        let bias: Vec<f32> = (0..out_f).map(|o| o as f32 * 0.01).collect();
        let input: Vec<f32> = (0..batch * in_f).map(|k| (k as f32 * 0.27).cos()).collect();
        let (w_q, w_scales) = quantize_per_channel(&weight, in_f, out_f);
        let a = quantized_linear_forward(&input, batch, in_f, &w_q, &w_scales, &bias, out_f);
        let b = quantized_linear_forward(&input, batch, in_f, &w_q, &w_scales, &bias, out_f);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

// ----- SmoothQuant : lissage activation→poids pour quantification int8 --------

/// **SmoothQuant** (Xiao et al. 2022) : facteurs de lissage par canal d'entrée
/// `s_j = max|X_:,j|^α / max|W_j,:|^(1-α)`. On migre la difficulté de
/// quantification des activations (souvent à valeurs aberrantes) vers les poids,
/// qui se quantifient mieux. `x` est `(tokens × in)`, `w` est `(in × out)`,
/// `alpha ∈ [0,1]` (0.5 typique). Retourne les `in` facteurs.
pub fn smoothquant_scales(
    x: &[f32],
    tokens: usize,
    in_f: usize,
    w: &[f32],
    out_f: usize,
    alpha: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), tokens * in_f, "smoothquant: x size");
    assert_eq!(w.len(), in_f * out_f, "smoothquant: w size");
    let mut s = vec![1.0f32; in_f];
    for (j, sj) in s.iter_mut().enumerate()
    {
        let mut xmax = 0.0f32;
        for t in 0..tokens
        {
            xmax = xmax.max(x[t * in_f + j].abs());
        }
        let mut wmax = 0.0f32;
        for o in 0..out_f
        {
            wmax = wmax.max(w[j * out_f + o].abs());
        }
        let num = xmax.powf(alpha);
        let den = wmax.powf(1.0 - alpha);
        // Keep s_j = 1 for empty channels (avoids div-by-zero / NaNs).
        if num > 1e-12 && den > 1e-12
        {
            *sj = num / den;
        }
    }
    s
}

/// Apply SmoothQuant in place: `X[:,j] /= s_j`, `W[j,:] *= s_j`. This is exactly
/// difficulty-preserving — `X·W` is unchanged — so it can run before int8
/// quantization without altering the (full-precision) result.
pub fn apply_smoothquant(
    x: &mut [f32],
    tokens: usize,
    in_f: usize,
    w: &mut [f32],
    out_f: usize,
    s: &[f32],
) {
    assert_eq!(s.len(), in_f, "smoothquant: scale len");
    for t in 0..tokens
    {
        for j in 0..in_f
        {
            x[t * in_f + j] /= s[j];
        }
    }
    for j in 0..in_f
    {
        for o in 0..out_f
        {
            w[j * out_f + o] *= s[j];
        }
    }
}

#[cfg(test)]
mod smoothquant_tests {
    use super::*;

    fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m
        {
            for p in 0..k
            {
                for j in 0..n
                {
                    out[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        out
    }

    /// SmoothQuant preserves the full-precision product `X·W`.
    #[test]
    fn smoothquant_preserves_product() {
        let (tokens, in_f, out_f) = (3usize, 4, 2);
        let mut x: Vec<f32> = (0..tokens * in_f)
            .map(|i| (i as f32 * 0.3 - 1.0).sin())
            .collect();
        // Outlier input channel 0 (large activations) — the case SmoothQuant targets.
        for t in 0..tokens
        {
            x[t * in_f] *= 50.0;
        }
        let mut w: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.2).cos()).collect();

        let y = matmul_f32(&x, &w, tokens, in_f, out_f);
        let s = smoothquant_scales(&x, tokens, in_f, &w, out_f, 0.5);
        apply_smoothquant(&mut x, tokens, in_f, &mut w, out_f, &s);
        let y2 = matmul_f32(&x, &w, tokens, in_f, out_f);

        for (a, b) in y.iter().zip(&y2)
        {
            assert!((a - b).abs() < 1e-2, "product changed: {a} vs {b}");
        }
    }

    /// SmoothQuant flattens the per-channel activation ranges: the outlier
    /// channel's range shrinks, so the spread across channels is far smaller.
    #[test]
    fn smoothquant_reduces_activation_outliers() {
        let (tokens, in_f, out_f) = (4usize, 4, 3);
        let mut x: Vec<f32> = (0..tokens * in_f)
            .map(|i| (i as f32 * 0.17 + 0.5).sin())
            .collect();
        for t in 0..tokens
        {
            x[t * in_f] *= 40.0; // outlier channel 0
        }
        let mut w: Vec<f32> = (0..in_f * out_f)
            .map(|i| (i as f32 * 0.11 - 0.3).cos())
            .collect();

        let chan_spread = |x: &[f32]| -> f32 {
            let maxes: Vec<f32> = (0..in_f)
                .map(|j| {
                    (0..tokens)
                        .map(|t| x[t * in_f + j].abs())
                        .fold(0.0, f32::max)
                })
                .collect();
            maxes.iter().cloned().fold(0.0, f32::max)
                / maxes
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min)
                    .max(1e-9)
        };

        let before = chan_spread(&x);
        let s = smoothquant_scales(&x, tokens, in_f, &w, out_f, 0.5);
        apply_smoothquant(&mut x, tokens, in_f, &mut w, out_f, &s);
        let after = chan_spread(&x);
        assert!(
            after < before * 0.5,
            "outliers not smoothed: {before} -> {after}"
        );
    }
}

// ----- GPTQ : quantification post-entraînement par feedback d'erreur ----------

/// **GPTQ** (Frantar et al. 2022) — proxy de Hessienne `H = XᵀX` sur les
/// activations de calibration. `x` est `(samples × in_features)` row-major
/// (même disposition que l'entrée de `quantized_linear_forward`). Retourne la
/// matrice symétrique `in_features × in_features` (row-major).
pub fn gptq_hessian(x: &[f32], samples: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(x.len(), samples * in_features, "gptq_hessian: x size");
    let d = in_features;
    let mut h = vec![0f32; d * d];
    for t in 0..samples
    {
        let row = &x[t * d..t * d + d];
        for a in 0..d
        {
            let xa = row[a];
            if xa == 0.0
            {
                continue;
            }
            for b in a..d
            {
                h[a * d + b] += xa * row[b];
            }
        }
    }
    // Symmetrise (we only filled the upper triangle).
    for a in 0..d
    {
        for b in (a + 1)..d
        {
            h[b * d + a] = h[a * d + b];
        }
    }
    h
}

/// Inverse d'une matrice symétrique définie positive `d×d` (row-major) par
/// Cholesky `H = L Lᵀ` puis `H⁻¹ = L⁻ᵀ L⁻¹`. Calcul interne en `f64` pour la
/// stabilité ; déterministe (ordre fixe). Suppose `h` déjà amortie (definie
/// positive) ; un pivot non positif est planché pour rester fini.
fn spd_inverse_f64(h: &[f32], d: usize) -> Vec<f64> {
    let mut l = vec![0f64; d * d];
    for i in 0..d
    {
        for j in 0..=i
        {
            let mut s = h[i * d + j] as f64;
            for k in 0..j
            {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j
            {
                l[i * d + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            }
            else
            {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    // L⁻¹ (lower) par substitution avant, colonne par colonne.
    let mut linv = vec![0f64; d * d];
    for col in 0..d
    {
        linv[col * d + col] = 1.0 / l[col * d + col];
        for i in (col + 1)..d
        {
            let mut s = 0f64;
            for k in col..i
            {
                s += l[i * d + k] * linv[k * d + col];
            }
            linv[i * d + col] = -s / l[i * d + i];
        }
    }
    // H⁻¹ = L⁻ᵀ L⁻¹ : Hinv[a,b] = Σ_k linv[k,a]·linv[k,b].
    let mut hinv = vec![0f64; d * d];
    for a in 0..d
    {
        for b in a..d
        {
            let mut s = 0f64;
            for k in a.max(b)..d
            {
                s += linv[k * d + a] * linv[k * d + b];
            }
            hinv[a * d + b] = s;
            hinv[b * d + a] = s;
        }
    }
    hinv
}

/// **GPTQ** : quantification int8 des poids par feedback d'erreur d'ordre 2.
/// `weight` est `(in_features × out_features)` row-major (canal de sortie `o` en
/// colonne, comme `quantize_per_channel`). `hessian` est le proxy
/// `in_features × in_features` issu de `gptq_hessian`. Pour chaque canal `o`, on
/// quantifie les poids d'entrée séquentiellement et on **propage l'erreur** sur
/// les poids non encore quantifiés via la Hessienne inverse (OBQ/GPTQ, ordre
/// naturel). Scale symétrique par canal de sortie. Déterministe (ordre fixe).
///
/// `percdamp` amortit la diagonale (`λ = percdamp·moyenne(diag H)`, 0.01 typique)
/// pour la stabilité numérique de l'inverse.
pub fn quantize_gptq(
    weight: &[f32],
    in_features: usize,
    out_features: usize,
    hessian: &[f32],
    percdamp: f32,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(
        weight.len(),
        in_features * out_features,
        "gptq: weight size"
    );
    assert_eq!(
        hessian.len(),
        in_features * in_features,
        "gptq: hessian size"
    );
    let d = in_features;

    // Scale par canal de sortie, figé sur les poids d'origine.
    let mut scales = vec![1f32; out_features];
    for (o, so) in scales.iter_mut().enumerate()
    {
        let mut m = 0f32;
        for i in 0..d
        {
            m = m.max(weight[i * out_features + o].abs());
        }
        *so = if m == 0.0 { 1.0 } else { m / 127.0 };
    }

    // Hessienne amortie puis inversée (en f64).
    let mut h = hessian.to_vec();
    let mut diag_mean = 0f32;
    for i in 0..d
    {
        diag_mean += h[i * d + i];
    }
    diag_mean /= d as f32;
    let damp = percdamp * diag_mean.max(1e-8);
    for i in 0..d
    {
        if h[i * d + i] <= 0.0
        {
            h[i * d + i] = 1.0; // colonne « morte » : se quantifiera vers 0
        }
        h[i * d + i] += damp;
    }
    let mut hinv = spd_inverse_f64(&h, d);

    let mut q = vec![0i8; d * out_features];
    let mut w = weight.to_vec(); // copie modifiée par le feedback d'erreur
    let mut err = vec![0f64; out_features];
    for i in 0..d
    {
        let di = hinv[i * d + i];
        let inv_di = if di.abs() > 1e-12 { 1.0 / di } else { 0.0 };
        // Quantifier la colonne d'entrée i pour tous les canaux de sortie.
        for o in 0..out_features
        {
            let s = scales[o];
            let wv = w[i * out_features + o];
            let qv = (wv / s).round().clamp(-128.0, 127.0);
            q[i * out_features + o] = qv as i8;
            err[o] = (wv as f64 - (qv * s) as f64) * inv_di;
        }
        // Propager l'erreur sur les poids non encore quantifiés (colonnes j>i).
        for j in (i + 1)..d
        {
            let hij = hinv[i * d + j];
            if hij == 0.0
            {
                continue;
            }
            for o in 0..out_features
            {
                w[j * out_features + o] -= (err[o] * hij) as f32;
            }
        }
        // Complément de Schur sur le bloc restant de H⁻¹ (déterministe).
        if inv_di != 0.0
        {
            for a in (i + 1)..d
            {
                let f = hinv[a * d + i] * inv_di;
                if f == 0.0
                {
                    continue;
                }
                for b in (i + 1)..d
                {
                    hinv[a * d + b] -= f * hinv[i * d + b];
                }
            }
        }
    }
    (q, scales)
}

#[cfg(test)]
mod gptq_tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// Reconstruction error weighted by the calibration covariance:
    /// `Σ_o (Δw_o)ᵀ H (Δw_o)` where `Δw_o` is the per-output-channel weight
    /// error — exactly the objective GPTQ minimises.
    fn weighted_err(w: &[f32], wq: &[f32], in_f: usize, out_f: usize, h: &[f32]) -> f64 {
        let mut e = 0f64;
        for o in 0..out_f
        {
            for a in 0..in_f
            {
                let da = (wq[a * out_f + o] - w[a * out_f + o]) as f64;
                if da == 0.0
                {
                    continue;
                }
                for b in 0..in_f
                {
                    let db = (wq[b * out_f + o] - w[b * out_f + o]) as f64;
                    e += da * h[a * in_f + b] as f64 * db;
                }
            }
        }
        e
    }

    fn dequant(q: &[i8], scales: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
        let mut w = vec![0f32; in_f * out_f];
        for i in 0..in_f
        {
            for o in 0..out_f
            {
                w[i * out_f + o] = q[i * out_f + o] as f32 * scales[o];
            }
        }
        w
    }

    /// On **correlated** calibration data, GPTQ's error feedback yields a
    /// strictly lower calibration-weighted reconstruction error than plain
    /// round-to-nearest (`quantize_per_channel`) at the same int8 budget — the
    /// whole point of the method. Also checks soundness (never worse).
    #[test]
    fn gptq_beats_rtn_on_calibration_error() {
        let (in_f, out_f, samples, latent) = (8usize, 4usize, 96usize, 3usize);
        let mut rng = PcgEngine::new(7);

        // Correlated activations: x[t,:] = A·z[t] + small noise (rank-`latent`).
        let a: Vec<f32> = (0..in_f * latent).map(|_| rng.float_signed()).collect();
        let mut x = vec![0f32; samples * in_f];
        for t in 0..samples
        {
            let z: Vec<f32> = (0..latent).map(|_| rng.float_signed()).collect();
            for i in 0..in_f
            {
                let mut v = 0.1 * rng.float_signed();
                for (l, &zl) in z.iter().enumerate()
                {
                    v += a[i * latent + l] * zl;
                }
                x[t * in_f + i] = v;
            }
        }
        let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
        let h = gptq_hessian(&x, samples, in_f);

        let (qg, sg) = quantize_gptq(&w, in_f, out_f, &h, 0.01);
        let wg = dequant(&qg, &sg, in_f, out_f);
        let (qr, sr) = quantize_per_channel(&w, in_f, out_f);
        let wr = dequant(&qr, &sr, in_f, out_f);

        let eg = weighted_err(&w, &wg, in_f, out_f, &h);
        let er = weighted_err(&w, &wr, in_f, out_f, &h);
        // Sound: GPTQ is never worse than RTN on the objective it optimises.
        assert!(eg <= er + 1e-3, "GPTQ worse than RTN: {eg} vs {er}");
        // Meaningful: on correlated data GPTQ is clearly better.
        assert!(
            eg < 0.9 * er,
            "GPTQ not meaningfully better: gptq={eg} rtn={er}"
        );
    }

    /// GPTQ is deterministic: identical inputs ⇒ bit-identical codes and scales.
    #[test]
    fn gptq_deterministic() {
        let (in_f, out_f) = (6usize, 3usize);
        let w: Vec<f32> = (0..in_f * out_f).map(|k| (k as f32 * 0.21).sin()).collect();
        let x: Vec<f32> = (0..40 * in_f).map(|k| (k as f32 * 0.13).cos()).collect();
        let h = gptq_hessian(&x, 40, in_f);
        let (q1, s1) = quantize_gptq(&w, in_f, out_f, &h, 0.01);
        let (q2, s2) = quantize_gptq(&w, in_f, out_f, &h, 0.01);
        assert_eq!(q1, q2);
        assert_eq!(
            s1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            s2.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

// ----- AWQ : quantification consciente des activations (scaling par recherche) -

/// **AWQ** (Lin et al. 2023) — importance d'activation par canal d'entrée :
/// `a_j = moyenne_t |x[t,j]|`. Les canaux à grande activation sont « saillants »
/// et méritent d'être protégés à la quantification. `x` est `(samples × in_features)`
/// row-major. Retourne les `in_features` magnitudes moyennes.
pub fn awq_act_scale(x: &[f32], samples: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(x.len(), samples * in_features, "awq_act_scale: x size");
    let mut a = vec![0f32; in_features];
    for t in 0..samples
    {
        for (j, aj) in a.iter_mut().enumerate()
        {
            *aj += x[t * in_features + j].abs();
        }
    }
    let inv = 1.0 / samples as f32;
    for aj in a.iter_mut()
    {
        *aj *= inv;
    }
    a
}

/// Résultat AWQ : poids int8, scales int8 par canal de sortie, facteurs d'échelle
/// par canal d'entrée appliqués avant quantification, et l'exposant `alpha` retenu.
#[derive(Clone, Debug)]
pub struct AwqResult {
    /// Poids quantifiés int8 `(in_features × out_features)` row-major (sur les
    /// poids **mis à l'échelle** `W·diag(s)`).
    pub q: Vec<i8>,
    /// Scale int8 symétrique par canal de sortie (len `out_features`).
    pub w_scales: Vec<f32>,
    /// Facteur d'échelle par canal d'entrée `s_j` (len `in_features`).
    pub channel_scales: Vec<f32>,
    /// Exposant `alpha` retenu par la recherche (`s_j = a_j^alpha`, moyenne géom. 1).
    pub alpha: f32,
}

impl AwqResult {
    /// Déquantifie vers les poids **d'origine** `Ŵ[j,:] = dequant(q)[j,:] / s_j`,
    /// row-major `(in_features × out_features)`.
    pub fn dequantize(&self, in_features: usize, out_features: usize) -> Vec<f32> {
        let mut w = vec![0f32; in_features * out_features];
        for j in 0..in_features
        {
            let inv = 1.0 / self.channel_scales[j];
            for o in 0..out_features
            {
                w[j * out_features + o] =
                    self.q[j * out_features + o] as f32 * self.w_scales[o] * inv;
            }
        }
        w
    }
}

/// **AWQ** : quantification int8 des poids **consciente des activations** par
/// recherche d'échelle. On protège les canaux d'entrée saillants en les mettant à
/// l'échelle (`W[j,:] ·= s_j`, avec `s_j = a_j^alpha` normalisé à moyenne
/// géométrique unité) avant la quantification int8 per-canal de sortie ; l'équivalence est
/// préservée en divisant les activations correspondantes (`x[:,j] /= s_j`), ce qui
/// se replie dans la couche précédente au déploiement. `alpha` est choisi par
/// **grille** (`grid` points dans `[0,1]`, `alpha=0` ⇒ round-to-nearest) en
/// minimisant l'erreur de sortie pondérée par la calibration `Σ_o Δw_oᵀ H Δw_o`.
/// `w` est `(in_features × out_features)`, `x` est `(samples × in_features)`.
/// Déterministe (grille et ordre fixes).
pub fn awq_quantize(
    w: &[f32],
    in_features: usize,
    out_features: usize,
    x: &[f32],
    samples: usize,
    grid: usize,
) -> AwqResult {
    assert_eq!(w.len(), in_features * out_features, "awq: weight size");
    assert!(
        grid >= 2,
        "awq: grid must be >= 2 (includes alpha=0 and alpha=1)"
    );
    let act = awq_act_scale(x, samples, in_features);
    let h = gptq_hessian(x, samples, in_features); // métrique d'erreur pondérée
    let log_act: Vec<f32> = act.iter().map(|&a| a.max(1e-12).ln()).collect();

    // Erreur de sortie pondérée par la calibration pour des poids déquantifiés.
    let werr = |wq: &[f32]| -> f64 {
        let mut e = 0f64;
        for o in 0..out_features
        {
            for a in 0..in_features
            {
                let da = (wq[a * out_features + o] - w[a * out_features + o]) as f64;
                if da == 0.0
                {
                    continue;
                }
                for b in 0..in_features
                {
                    let db = (wq[b * out_features + o] - w[b * out_features + o]) as f64;
                    e += da * h[a * in_features + b] as f64 * db;
                }
            }
        }
        e
    };

    let mut best: Option<AwqResult> = None;
    let mut best_err = f64::INFINITY;
    for g in 0..grid
    {
        let alpha = g as f32 / (grid - 1) as f32;
        // s_j = a_j^alpha normalisé en moyenne géométrique 1 (neutre en magnitude).
        let mean_log = log_act.iter().sum::<f32>() / in_features as f32;
        let s: Vec<f32> = log_act
            .iter()
            .map(|&la| (alpha * (la - mean_log)).exp())
            .collect();
        // Poids mis à l'échelle, puis quantification int8 per-canal de sortie.
        let mut ws = vec![0f32; in_features * out_features];
        for j in 0..in_features
        {
            for o in 0..out_features
            {
                ws[j * out_features + o] = w[j * out_features + o] * s[j];
            }
        }
        let (q, w_scales) = quantize_per_channel(&ws, in_features, out_features);
        let cand = AwqResult {
            q,
            w_scales,
            channel_scales: s,
            alpha,
        };
        let err = werr(&cand.dequantize(in_features, out_features));
        if err < best_err
        {
            best_err = err;
            best = Some(cand);
        }
    }
    best.expect("awq: grid >= 2 guarantees a candidate")
}

#[cfg(test)]
mod awq_tests {
    use super::*;
    use crate::nn::PcgEngine;

    fn weighted_err(w: &[f32], wq: &[f32], in_f: usize, out_f: usize, h: &[f32]) -> f64 {
        let mut e = 0f64;
        for o in 0..out_f
        {
            for a in 0..in_f
            {
                let da = (wq[a * out_f + o] - w[a * out_f + o]) as f64;
                if da == 0.0
                {
                    continue;
                }
                for b in 0..in_f
                {
                    let db = (wq[b * out_f + o] - w[b * out_f + o]) as f64;
                    e += da * h[a * in_f + b] as f64 * db;
                }
            }
        }
        e
    }

    /// With a few **salient** (high-activation) input channels, AWQ's search picks
    /// a scaling (`alpha > 0`) that strictly lowers the calibration output error
    /// versus plain round-to-nearest (which AWQ contains as the `alpha = 0` point).
    #[test]
    fn awq_protects_salient_channels_and_beats_rtn() {
        let (in_f, out_f, samples) = (12usize, 6usize, 96usize);
        let mut rng = PcgEngine::new(11);
        // Activations: most channels small, a few salient (×20). These dominate Y.
        let salient = [2usize, 5, 9];
        let mut x = vec![0f32; samples * in_f];
        for t in 0..samples
        {
            for j in 0..in_f
            {
                let base = rng.float_signed();
                x[t * in_f + j] = if salient.contains(&j)
                {
                    20.0 * base
                }
                else
                {
                    base
                };
            }
        }
        let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
        let h = gptq_hessian(&x, samples, in_f);

        let res = awq_quantize(&w, in_f, out_f, &x, samples, 21);
        let wq = res.dequantize(in_f, out_f);
        let e_awq = weighted_err(&w, &wq, in_f, out_f, &h);

        // alpha = 0 is exactly per-channel round-to-nearest.
        let (qr, sr) = quantize_per_channel(&w, in_f, out_f);
        let mut wr = vec![0f32; in_f * out_f];
        for j in 0..in_f
        {
            for o in 0..out_f
            {
                wr[j * out_f + o] = qr[j * out_f + o] as f32 * sr[o];
            }
        }
        let e_rtn = weighted_err(&w, &wr, in_f, out_f, &h);

        assert!(
            res.alpha > 0.0,
            "AWQ should scale salient channels (alpha>0)"
        );
        assert!(
            e_awq < e_rtn,
            "AWQ not better than RTN: awq={e_awq} rtn={e_rtn} (alpha={})",
            res.alpha
        );
    }

    /// AWQ is deterministic: identical inputs ⇒ identical codes, scales, alpha.
    #[test]
    fn awq_deterministic() {
        let (in_f, out_f, samples) = (8usize, 4usize, 50usize);
        let w: Vec<f32> = (0..in_f * out_f).map(|k| (k as f32 * 0.17).sin()).collect();
        let x: Vec<f32> = (0..samples * in_f)
            .map(|k| (k as f32 * 0.09).cos())
            .collect();
        let r1 = awq_quantize(&w, in_f, out_f, &x, samples, 11);
        let r2 = awq_quantize(&w, in_f, out_f, &x, samples, 11);
        assert_eq!(r1.q, r2.q);
        assert_eq!(r1.alpha.to_bits(), r2.alpha.to_bits());
        assert_eq!(
            r1.channel_scales
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            r2.channel_scales
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

// ----- BitNet b1.58 : poids ternaires {−1,0,1}, matmul sans multiplication ----

/// **BitNet b1.58** (Ma et al. 2024) — quantification **ternaire** des poids vers
/// `{−1, 0, +1}` avec une échelle par tenseur (« absmean » : `scale = moyenne|W|`).
/// Chaque poids devient `round(W/scale)` borné à `[−1, 1]`. Retourne les codes
/// ternaires (`i8 ∈ {−1,0,1}`) et le scale.
pub fn ternary_quantize(w: &[f32]) -> (Vec<i8>, f32) {
    let n = w.len().max(1);
    let scale = w.iter().map(|x| x.abs()).sum::<f32>() / n as f32;
    let inv = if scale > 1e-12 { 1.0 / scale } else { 0.0 };
    let q: Vec<i8> = w
        .iter()
        .map(|&x| (x * inv).round().clamp(-1.0, 1.0) as i8)
        .collect();
    (q, scale)
}

/// **Multiplication-free** ternary matmul `y = x · (scale · W_ternary)` with
/// `W_ternary ∈ {−1,0,1}` — `x` is `(batch × in)` row-major, `w_q` is
/// `(in × out)` row-major. Because the weights are ±1/0, each accumulation is an
/// **add / subtract / skip** (no multiply); a single `scale` multiply is applied
/// at the end. This equals the dequantised product **bit-for-bit** (the test
/// oracle), demonstrating BitNet's "matmul without multiplications".
pub fn ternary_matmul(
    x: &[f32],
    batch: usize,
    w_q: &[i8],
    in_f: usize,
    out_f: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), batch * in_f, "ternary_matmul: x size");
    assert_eq!(w_q.len(), in_f * out_f, "ternary_matmul: w size");
    let mut out = vec![0f32; batch * out_f];
    for b in 0..batch
    {
        let xrow = &x[b * in_f..b * in_f + in_f];
        for o in 0..out_f
        {
            let mut acc = 0f32;
            for i in 0..in_f
            {
                match w_q[i * out_f + o]
                {
                    1 => acc += xrow[i],
                    -1 => acc -= xrow[i],
                    _ =>
                    {},
                }
            }
            out[b * out_f + o] = acc * scale;
        }
    }
    out
}

#[cfg(test)]
mod bitnet_tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// `ternary_quantize` maps to `{−1,0,1}` only, and the **multiplication-free**
    /// `ternary_matmul` equals the dequantised float product **bit-for-bit**.
    #[test]
    fn ternary_matmul_equals_dequant_bit_exact() {
        let (in_f, out_f, batch) = (12usize, 6usize, 4usize);
        let mut rng = PcgEngine::new(3);
        let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
        let x: Vec<f32> = (0..batch * in_f).map(|_| rng.float_signed()).collect();

        let (wq, scale) = ternary_quantize(&w);
        // Only ternary values.
        assert!(wq.iter().all(|&v| v == -1 || v == 0 || v == 1));

        let mut sign_sum = vec![0f32; batch * out_f]; // (Σ ±xᵢ)·scale — same order as mf
        let mut dequant = vec![0f32; batch * out_f]; // Σ xᵢ·(±scale) — ordinary multiply
        for b in 0..batch
        {
            for o in 0..out_f
            {
                let (mut s, mut d) = (0f32, 0f32);
                for i in 0..in_f
                {
                    let q = wq[i * out_f + o] as f32;
                    s += q * x[b * in_f + i]; // ±xᵢ (or 0)
                    d += x[b * in_f + i] * (q * scale);
                }
                sign_sum[b * out_f + o] = s * scale;
                dequant[b * out_f + o] = d;
            }
        }
        let mf = ternary_matmul(&x, batch, &wq, in_f, out_f, scale);
        // The multiplication-free path is exactly (Σ ±xᵢ)·scale (bit-for-bit).
        for (a, r) in mf.iter().zip(&sign_sum)
        {
            assert_eq!(
                a.to_bits(),
                r.to_bits(),
                "ternary matmul not the sign-sum form: {a} vs {r}"
            );
        }
        // …and equals the dequantised product up to floating-point reassociation.
        for (a, d) in mf.iter().zip(&dequant)
        {
            assert!(
                (a - d).abs() < 1e-4,
                "ternary matmul off from dequant: {a} vs {d}"
            );
        }
    }

    /// Deterministic: identical input ⇒ identical codes and scale.
    #[test]
    fn ternary_quantize_deterministic() {
        let w: Vec<f32> = (0..40).map(|k| (k as f32 * 0.21).sin()).collect();
        let (q1, s1) = ternary_quantize(&w);
        let (q2, s2) = ternary_quantize(&w);
        assert_eq!(q1, q2);
        assert_eq!(s1.to_bits(), s2.to_bits());
    }
}

// ----- NF4 : NormalFloat 4-bit (QLoRA, Dettmers et al. 2023) -------------------

/// The **NF4** (4-bit NormalFloat) code values, from QLoRA / bitsandbytes: 16
/// levels that are the quantiles of a standard normal, normalised so the extreme
/// magnitudes map to ±1 and with an **exact 0**. NF4 is information-theoretically
/// (near-)optimal for the roughly-Gaussian weights of a trained network.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Quantise a weight block to **NF4**: divide by the block's `absmax` (so values
/// lie in `[−1, 1]`), then map each to the nearest [`NF4_LEVELS`] entry. Returns
/// the 4-bit codes (`0..16`) and the `absmax` scale. Deterministic.
pub fn nf4_quantize(w: &[f32]) -> (Vec<u8>, f32) {
    let absmax = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let inv = if absmax > 1e-12 { 1.0 / absmax } else { 0.0 };
    let codes: Vec<u8> = w
        .iter()
        .map(|&x| {
            let v = x * inv;
            // Nearest NF4 level (16 entries; linear scan, deterministic order).
            let mut best = 0usize;
            let mut bd = (v - NF4_LEVELS[0]).abs();
            for (k, &lvl) in NF4_LEVELS.iter().enumerate().skip(1)
            {
                let d = (v - lvl).abs();
                if d < bd
                {
                    bd = d;
                    best = k;
                }
            }
            best as u8
        })
        .collect();
    (codes, absmax)
}

/// Dequantise NF4 codes back to `f32`: `NF4_LEVELS[code] · absmax`.
///
/// NF4 codes are 4-bit; only the low nibble of each byte is meaningful. A
/// corrupted or untrusted packed-weights buffer may carry bytes ≥ 16, which
/// would index `NF4_LEVELS` (16 entries) out of bounds and panic (DoS). Masking
/// to the low nibble decodes an out-of-range byte deterministically instead of
/// aborting the process.
pub fn nf4_dequantize(codes: &[u8], absmax: f32) -> Vec<f32> {
    codes
        .iter()
        .map(|&c| NF4_LEVELS[(c & 0x0F) as usize] * absmax)
        .collect()
}

#[cfg(test)]
mod nf4_tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// On **Gaussian** weights, NF4's quantile-matched levels give a strictly
    /// lower reconstruction error than uniform 4-bit (16 evenly-spaced levels) —
    /// the whole point of the NormalFloat type. Also checks determinism.
    #[test]
    fn nf4_beats_uniform_4bit_on_gaussian() {
        let mut rng = PcgEngine::new(5);
        // Box–Muller: standard-normal weights from uniform (0,1) draws.
        let w: Vec<f32> = (0..4096)
            .map(|_| {
                let u1 = rng.float().max(1e-7);
                let u2 = rng.float();
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect();

        let (codes, absmax) = nf4_quantize(&w);
        assert!(codes.iter().all(|&c| (c as usize) < 16));
        let nf4 = nf4_dequantize(&codes, absmax);

        // Uniform 4-bit: 16 evenly-spaced levels in [−1, 1], same nearest mapping.
        let ulevels: Vec<f32> = (0..16).map(|k| -1.0 + 2.0 * k as f32 / 15.0).collect();
        let inv = 1.0 / absmax;
        let uni: Vec<f32> = w
            .iter()
            .map(|&x| {
                let v = x * inv;
                let mut best = ulevels[0];
                let mut bd = (v - ulevels[0]).abs();
                for &lvl in &ulevels[1..]
                {
                    let d = (v - lvl).abs();
                    if d < bd
                    {
                        bd = d;
                        best = lvl;
                    }
                }
                best * absmax
            })
            .collect();

        let err = |q: &[f32]| -> f64 {
            w.iter()
                .zip(q)
                .map(|(&a, &b)| (a - b).abs() as f64)
                .sum::<f64>()
        };
        let (e_nf4, e_uni) = (err(&nf4), err(&uni));
        assert!(
            e_nf4 < e_uni,
            "NF4 not better than uniform-4bit: {e_nf4} vs {e_uni}"
        );

        // Determinism.
        let (c2, s2) = nf4_quantize(&w);
        assert_eq!(codes, c2);
        assert_eq!(absmax.to_bits(), s2.to_bits());
    }

    /// Round-trip on values that are exactly NF4 levels is exact.
    #[test]
    fn nf4_round_trip_on_level_values() {
        let w: Vec<f32> = NF4_LEVELS.iter().map(|&l| l * 3.0).collect(); // absmax = 3
        let (codes, absmax) = nf4_quantize(&w);
        assert!((absmax - 3.0).abs() < 1e-6);
        let back = nf4_dequantize(&codes, absmax);
        for (a, b) in w.iter().zip(&back)
        {
            assert!((a - b).abs() < 1e-5, "NF4 round-trip off: {a} vs {b}");
        }
    }
}

// ----- SqueezeLLM : sensitivity-weighted non-uniform quantization (#66) --------

/// **SqueezeLLM** (Kim et al. 2023, arXiv:2306.07629) — non-uniform weight
/// quantization. Instead of a uniform grid (round-to-nearest), SqueezeLLM fits a
/// `2^bits`-entry **codebook** to the weights by **weighted k-means**, where each
/// value's weight is its **sensitivity** — a proxy for the Hessian diagonal
/// `∂²L/∂wᵢ²`. This places code points where they cut the *loss* most (important
/// weights quantized finely), not merely where weights are dense, giving markedly
/// lower error than RTN at the same bit-width — especially under the sensitivity
/// metric it optimises. Deterministic: quantile initialisation + a fixed number of
/// Lloyd iterations, pure `f32` in a fixed order. (The orthogonal "sparse" outlier
/// branch of the paper is not modelled here.)
pub struct SqueezeLlmCodebook {
    levels: Vec<f32>,
}

impl SqueezeLlmCodebook {
    /// Fit a `2^bits` codebook to `weights` with per-weight `sensitivity` (≥ 0;
    /// pass all-ones for unweighted k-means), running `iters` Lloyd iterations
    /// after a deterministic quantile initialisation.
    pub fn fit(weights: &[f32], sensitivity: &[f32], bits: u32, iters: usize) -> Self {
        assert_eq!(
            weights.len(),
            sensitivity.len(),
            "SqueezeLLM: weights/sensitivity length mismatch"
        );
        assert!((1..=8).contains(&bits), "SqueezeLLM: bits must be in 1..=8");
        let k = 1usize << bits;
        let n = weights.len();
        let mut levels = vec![0.0f32; k];
        if n == 0
        {
            return Self { levels };
        }
        // Deterministic init: centroids at evenly spaced quantiles of the weights.
        let mut sorted = weights.to_vec();
        sorted.sort_by(f32::total_cmp);
        for (j, lvl) in levels.iter_mut().enumerate()
        {
            let pos = ((j as f32 + 0.5) / k as f32) * (n as f32 - 1.0);
            *lvl = sorted[pos.round() as usize];
        }
        // Weighted Lloyd iterations: assign to nearest centroid, then move each
        // centroid to the sensitivity-weighted mean of its members (the optimum).
        for _ in 0..iters
        {
            let mut sum = vec![0.0f32; k];
            let mut wsum = vec![0.0f32; k];
            for (&w, &s) in weights.iter().zip(sensitivity)
            {
                let a = nearest_level(&levels, w);
                let s = s.max(0.0);
                sum[a] += s * w;
                wsum[a] += s;
            }
            for (lvl, (&sj, &wj)) in levels.iter_mut().zip(sum.iter().zip(&wsum))
            {
                if wj > 0.0
                {
                    *lvl = sj / wj;
                }
                // Empty cluster: keep its previous value (deterministic).
            }
        }
        // Sort for a reproducible, scan-friendly codebook representation.
        levels.sort_by(f32::total_cmp);
        Self { levels }
    }

    /// Index of the nearest codebook entry to `w`.
    pub fn quantize_index(&self, w: f32) -> usize {
        nearest_level(&self.levels, w)
    }

    /// Centroid value for a codebook index.
    pub fn dequantize(&self, idx: usize) -> f32 {
        self.levels[idx]
    }

    /// Round-trip: the nearest codebook value to `w`.
    pub fn quantize(&self, w: f32) -> f32 {
        self.levels[nearest_level(&self.levels, w)]
    }

    /// The codebook (sorted centroid values).
    pub fn levels(&self) -> &[f32] {
        &self.levels
    }
}

/// Index of the nearest value in `levels` to `w` (linear scan; ties go to the
/// lower index — deterministic).
fn nearest_level(levels: &[f32], w: f32) -> usize {
    let mut best = 0usize;
    let mut bd = (w - levels[0]).abs();
    for (i, &l) in levels.iter().enumerate().skip(1)
    {
        let d = (w - l).abs();
        if d < bd
        {
            bd = d;
            best = i;
        }
    }
    best
}

/// Sensitivity-weighted quantization error `Σ sᵢ·(wᵢ − q(wᵢ))²` — the objective
/// SqueezeLLM minimises and the metric on which it beats round-to-nearest.
pub fn weighted_quant_error(weights: &[f32], sensitivity: &[f32], quantized: &[f32]) -> f32 {
    weights
        .iter()
        .zip(sensitivity)
        .zip(quantized)
        .map(|((&w, &s), &q)| s.max(0.0) * (w - q) * (w - q))
        .sum()
}

#[cfg(test)]
mod tests_squeezellm {
    use super::*;
    use crate::nn::PcgEngine;

    /// Uniform round-to-nearest baseline: `2^bits` evenly spaced levels over
    /// `[min, max]`, nearest mapping.
    fn rtn(weights: &[f32], bits: u32) -> Vec<f32> {
        let k = 1usize << bits;
        let lo = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step = (hi - lo) / (k as f32 - 1.0);
        weights
            .iter()
            .map(|&w| {
                let idx = ((w - lo) / step).round().clamp(0.0, (k - 1) as f32);
                lo + idx * step
            })
            .collect()
    }

    /// **The SqueezeLLM claim, tested.** On Gaussian weights with non-uniform
    /// sensitivity, the sensitivity-weighted k-means codebook achieves strictly
    /// (and meaningfully) lower weighted quantization error than uniform
    /// round-to-nearest at the same bit-width.
    #[test]
    fn squeezellm_beats_rtn_on_weighted_error() {
        let mut rng = PcgEngine::new(9);
        let n = 2000usize;
        // Gaussian-ish weights (sum of uniforms); heteroscedastic sensitivity.
        let w: Vec<f32> = (0..n)
            .map(|_| rng.float_signed() + rng.float_signed() + rng.float_signed())
            .collect();
        let s: Vec<f32> = (0..n).map(|_| rng.float().abs() + 0.05).collect();
        let bits = 3;
        let cb = SqueezeLlmCodebook::fit(&w, &s, bits, 12);
        let q_sq: Vec<f32> = w.iter().map(|&x| cb.quantize(x)).collect();
        let q_rtn = rtn(&w, bits);
        let e_sq = weighted_quant_error(&w, &s, &q_sq);
        let e_rtn = weighted_quant_error(&w, &s, &q_rtn);
        assert!(
            e_sq < e_rtn,
            "SqueezeLLM not better than RTN: sq={e_sq} rtn={e_rtn}"
        );
        assert!(
            e_sq < 0.85 * e_rtn,
            "SqueezeLLM only marginally better: sq={e_sq} rtn={e_rtn}"
        );
    }

    /// The fit is deterministic, the codebook is sorted with `2^bits` entries, and
    /// round-trip on a codebook value is exact (it maps to itself).
    #[test]
    fn squeezellm_roundtrip_and_determinism() {
        let mut rng = PcgEngine::new(3);
        let n = 500usize;
        let w: Vec<f32> = (0..n).map(|_| rng.float_signed() * 2.0).collect();
        let s = vec![1.0f32; n];
        let fit = || SqueezeLlmCodebook::fit(&w, &s, 4, 10).levels().to_vec();
        assert_eq!(fit(), fit()); // determinism

        let cb = SqueezeLlmCodebook::fit(&w, &s, 4, 10);
        assert_eq!(cb.levels().len(), 16);
        for pair in cb.levels().windows(2)
        {
            assert!(pair[0] <= pair[1], "codebook not sorted");
        }
        for &l in cb.levels()
        {
            assert!(
                (cb.quantize(l) - l).abs() < 1e-6,
                "codebook value not fixed: {l}"
            );
        }
    }
}

// ----- SpQR : sparse-quantized representation (outliers en fp) (#67) -----------

/// **SpQR** — Sparse-Quantized Representation (Dettmers et al. 2023,
/// arXiv:2306.03078). Weight-quantization error is **heavy-tailed**: a small
/// fraction of "outlier" weights account for most of the loss. SpQR keeps that
/// fraction in **full precision** (a sparse side-channel) and quantizes the rest
/// densely, so a ~1 % outlier budget removes a large share of the error at a small
/// memory overhead. This models the sparse-outlier core (the paper's bi-level
/// grouped scales are orthogonal). Deterministic: outliers are the largest
/// dense-quantization errors, ties broken by index.
pub struct SpqrOutliers {
    indices: Vec<usize>,
    values: Vec<f32>,
}

impl SpqrOutliers {
    /// Extract the `outlier_frac` (in `[0,1]`) fraction of weights whose dense
    /// quantization error `|w − q|` is largest, storing their exact fp values.
    pub fn extract(weights: &[f32], quantized: &[f32], outlier_frac: f32) -> Self {
        assert_eq!(
            weights.len(),
            quantized.len(),
            "SpQR: weights/quantized length mismatch"
        );
        assert!(
            (0.0..=1.0).contains(&outlier_frac),
            "SpQR: outlier_frac must be in [0,1]"
        );
        let n = weights.len();
        let k = ((n as f32) * outlier_frac).round() as usize;
        // Rank indices by descending |w − q| (ties → lower index for determinism).
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            let ea = (weights[a] - quantized[a]).abs();
            let eb = (weights[b] - quantized[b]).abs();
            eb.total_cmp(&ea).then(a.cmp(&b))
        });
        let mut indices: Vec<usize> = idx.into_iter().take(k).collect();
        indices.sort_unstable();
        let values = indices.iter().map(|&i| weights[i]).collect();
        Self { indices, values }
    }

    /// Reconstruct: the dense `quantized` weights with the stored outliers
    /// overwritten by their exact fp values.
    pub fn reconstruct(&self, quantized: &[f32]) -> Vec<f32> {
        let mut out = quantized.to_vec();
        for (&i, &v) in self.indices.iter().zip(&self.values)
        {
            out[i] = v;
        }
        out
    }

    /// Number of full-precision outliers kept.
    pub fn num_outliers(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests_spqr {
    use super::*;
    use crate::nn::PcgEngine;

    /// Dense baseline: uniform int-`bits` round-to-nearest **clipped** to a fixed
    /// `[lo, hi]` range (so out-of-range outliers incur a large clamp error).
    fn rtn_clip(w: &[f32], bits: u32, lo: f32, hi: f32) -> Vec<f32> {
        let k = 1usize << bits;
        let step = (hi - lo) / (k as f32 - 1.0);
        w.iter()
            .map(|&x| {
                let idx = ((x.clamp(lo, hi) - lo) / step).round();
                lo + idx * step
            })
            .collect()
    }

    /// **The SpQR insight, tested.** Quantization error is heavy-tailed: a few
    /// large-magnitude weights, clamped by the dense grid, dominate the squared
    /// error. Keeping just **1 %** of weights (the largest errors) in full
    /// precision cuts the total error by far more than 1 %.
    #[test]
    fn spqr_outliers_slash_heavy_tailed_error() {
        let mut rng = PcgEngine::new(13);
        let n = 2000usize;
        // Bulk ~ sum of two uniforms ∈ [−2, 2]; inject 20 large outliers (±12).
        let mut w: Vec<f32> = (0..n)
            .map(|_| rng.float_signed() + rng.float_signed())
            .collect();
        for j in 0..(n / 100)
        {
            w[j * 100] = if j % 2 == 0 { 12.0 } else { -12.0 };
        }
        let q = rtn_clip(&w, 4, -3.0, 3.0);
        let e_dense: f32 = w.iter().zip(&q).map(|(&a, &b)| (a - b) * (a - b)).sum();
        let spqr = SpqrOutliers::extract(&w, &q, 0.01);
        let recon = spqr.reconstruct(&q);
        let e_spqr: f32 = w.iter().zip(&recon).map(|(&a, &b)| (a - b) * (a - b)).sum();
        assert_eq!(spqr.num_outliers(), 20);
        assert!(
            e_spqr < 0.3 * e_dense,
            "SpQR did not cut heavy-tailed error: dense={e_dense}, spqr={e_spqr}"
        );
    }

    /// Reconstruction overwrites outliers with exact fp (error never increases),
    /// and extraction is deterministic.
    #[test]
    fn spqr_reconstruction_and_determinism() {
        let w: Vec<f32> = (0..50).map(|i| (i as f32 * 0.3).sin() * 3.0).collect();
        let q: Vec<f32> = w.iter().map(|&x| x.round()).collect();
        let a = SpqrOutliers::extract(&w, &q, 0.2);
        let b = SpqrOutliers::extract(&w, &q, 0.2);
        assert_eq!(a.reconstruct(&q), b.reconstruct(&q)); // determinism
        assert_eq!(a.num_outliers(), 10);
        let recon = a.reconstruct(&q);
        let e_dense: f32 = w.iter().zip(&q).map(|(&x, &y)| (x - y) * (x - y)).sum();
        let e_spqr: f32 = w.iter().zip(&recon).map(|(&x, &y)| (x - y) * (x - y)).sum();
        assert!(
            e_spqr <= e_dense,
            "SpQR increased error: {e_dense} -> {e_spqr}"
        );
    }
}

// ----- KVQuant : quantification du KV-cache (clés per-canal) (#68) -------------

/// Symmetric `bits`-bit quantize→dequantize of a `rows×cols` matrix with one
/// absmax scale per **row** (`per_row = true`) or per **column** (`per_row =
/// false`). A zero row/column is left as zeros.
fn kv_quant_dequant_axis(
    m: &[f32],
    rows: usize,
    cols: usize,
    bits: u32,
    per_row: bool,
) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32; // symmetric signed range
    let mut out = vec![0.0f32; rows * cols];
    let q1 = |x: f32, scale: f32| (x / scale).round().clamp(-qmax, qmax) * scale;
    if per_row
    {
        for (row, orow) in m.chunks_exact(cols).zip(out.chunks_exact_mut(cols))
        {
            let amax = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let scale = if amax > 0.0 { amax / qmax } else { 1.0 };
            for (o, &x) in orow.iter_mut().zip(row)
            {
                *o = q1(x, scale);
            }
        }
    }
    else
    {
        for j in 0..cols
        {
            let mut amax = 0.0f32;
            for i in 0..rows
            {
                amax = amax.max(m[i * cols + j].abs());
            }
            let scale = if amax > 0.0 { amax / qmax } else { 1.0 };
            for i in 0..rows
            {
                out[i * cols + j] = q1(m[i * cols + j], scale);
            }
        }
    }
    out
}

/// **KVQuant** (Hooper et al., NeurIPS 2024): quantize the attention **KV cache**
/// at the granularity matching its outlier structure — **Keys per-channel** (per
/// feature column, where key outliers concentrate) and **Values per-token** (per
/// row). Returns `(k_hat, v_hat)` reconstructed at `bits`-bit. This is far more
/// faithful than a single per-tensor scale, which a few large key channels would
/// dominate (crushing the resolution of all the others). `k`/`v` are `(seq, d)`.
pub fn kvquant_kv(k: &[f32], v: &[f32], seq: usize, d: usize, bits: u32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(k.len(), seq * d, "KVQuant: K size mismatch");
    assert_eq!(v.len(), seq * d, "KVQuant: V size mismatch");
    assert!((2..=8).contains(&bits), "KVQuant: bits must be in 2..=8");
    let k_hat = kv_quant_dequant_axis(k, seq, d, bits, false); // keys per-channel
    let v_hat = kv_quant_dequant_axis(v, seq, d, bits, true); // values per-token
    (k_hat, v_hat)
}

#[cfg(test)]
mod tests_kvquant {
    use super::*;
    use crate::nn::PcgEngine;

    /// Single-scale (per-tensor) symmetric quantize→dequantize baseline.
    fn per_tensor(m: &[f32], bits: u32) -> Vec<f32> {
        let qmax = ((1i32 << (bits - 1)) - 1) as f32;
        let amax = m.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let scale = if amax > 0.0 { amax / qmax } else { 1.0 };
        m.iter()
            .map(|&x| (x / scale).round().clamp(-qmax, qmax) * scale)
            .collect()
    }

    /// Tiny attention forward `softmax(Q·Kᵀ/√d)·V` (no mask), for the oracle.
    fn attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize) -> Vec<f32> {
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; seq * d];
        for i in 0..seq
        {
            let mut scores = vec![0.0f32; seq];
            for (j, sj) in scores.iter_mut().enumerate()
            {
                let mut s = 0.0f32;
                for t in 0..d
                {
                    s += q[i * d + t] * k[j * d + t];
                }
                *sj = s * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&z| (z - mx).exp()).collect();
            let z: f32 = exps.iter().sum();
            for (j, &e) in exps.iter().enumerate()
            {
                let w = e / z;
                for t in 0..d
                {
                    out[i * d + t] += w * v[j * d + t];
                }
            }
        }
        out
    }

    /// **The KVQuant insight, tested.** Keys carry **per-channel outliers** (a few
    /// feature columns are much larger). Per-tensor quantization wastes resolution
    /// on the outlier channels, corrupting `Q·Kᵀ`; KVQuant's per-channel K (and
    /// per-token V) keeps every channel well-resolved, so the attention output is
    /// far closer to fp16. Deterministic.
    #[test]
    fn kvquant_beats_per_tensor_on_attention() {
        let mut rng = PcgEngine::new(17);
        let (seq, d) = (8usize, 16usize);
        let mut k = vec![0.0f32; seq * d];
        for i in 0..seq
        {
            for j in 0..d
            {
                let outlier = if j == 0 || j == 5 { 12.0 } else { 1.0 };
                k[i * d + j] = rng.float_signed() * 0.5 * outlier;
            }
        }
        let v: Vec<f32> = (0..seq * d).map(|_| rng.float_signed()).collect();
        let q: Vec<f32> = (0..seq * d).map(|_| rng.float_signed()).collect();
        let bits = 4;

        let fp = attention(&q, &k, &v, seq, d);
        let (kc, vt) = kvquant_kv(&k, &v, seq, d, bits);
        let kvq = attention(&q, &kc, &vt, seq, d);
        let pt = attention(&q, &per_tensor(&k, bits), &per_tensor(&v, bits), seq, d);

        let err = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
        };
        let (e_kvq, e_pt) = (err(&kvq, &fp), err(&pt, &fp));
        assert!(
            e_kvq < e_pt,
            "KVQuant {e_kvq} not better than per-tensor {e_pt}"
        );
        assert!(
            e_kvq < 0.6 * e_pt,
            "KVQuant not meaningfully better: {e_kvq} vs {e_pt}"
        );
    }

    /// Per-channel quantization resolves a non-outlier column accurately even when
    /// another column is a huge outlier — exactly where per-tensor fails — and the
    /// reconstruction is deterministic.
    #[test]
    fn kvquant_per_channel_resolves_small_columns() {
        let (seq, d) = (6usize, 4usize);
        // Column 0 is a large outlier; columns 1..3 are small.
        let mut m = vec![0.0f32; seq * d];
        for i in 0..seq
        {
            m[i * d] = 50.0 + i as f32; // outlier column
            for j in 1..d
            {
                m[i * d + j] = 0.1 * (i + j) as f32;
            }
        }
        let per_chan = kv_quant_dequant_axis(&m, seq, d, 4, false);
        let per_tens = per_tensor(&m, 4);
        // Error on the small columns (j ≥ 1).
        let small_err = |r: &[f32]| -> f32 {
            let mut e = 0.0f32;
            for i in 0..seq
            {
                for j in 1..d
                {
                    e += (m[i * d + j] - r[i * d + j]).powi(2);
                }
            }
            e
        };
        assert!(
            small_err(&per_chan) < 0.1 * small_err(&per_tens),
            "per-channel did not resolve small columns: {} vs {}",
            small_err(&per_chan),
            small_err(&per_tens)
        );
        assert_eq!(per_chan, kv_quant_dequant_axis(&m, seq, d, 4, false)); // determinism
    }
}

// ----- LLM.int8() : décomposition mixte int8 / fp16 (#71) ----------------------

/// **LLM.int8()** (Dettmers et al., NeurIPS 2022): a mixed-precision matmul
/// `X·W` (`X` is `m×k`, `W` is `k×n`). Transformer activations have a few
/// **outlier feature columns** of huge magnitude; quantizing them with everything
/// else to int8 inflates the scale and crushes the resolution of the normal
/// features. LLM.int8() keeps those columns (and the matching `W` rows) in **full
/// precision** and quantizes the rest to **int8**:
/// `X·W = X_normal·W_normal (int8) + X_outlier·W_outlier (fp32)`. A column is an
/// outlier if any `|X[i,j]|` exceeds `threshold` (the paper's default is 6.0). The
/// int8 scale, computed with the outliers removed, then resolves the bulk finely,
/// so the result is close to fp while most of the matmul stays integer.
pub fn int8_mixed_matmul(
    x: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    threshold: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "int8_mixed_matmul: X size mismatch");
    assert_eq!(w.len(), k * n, "int8_mixed_matmul: W size mismatch");
    // Outlier feature columns of X (any row exceeds the threshold).
    let outlier: Vec<bool> = (0..k)
        .map(|j| (0..m).any(|i| x[i * k + j].abs() > threshold))
        .collect();

    // int8 path: X and W with the outlier columns / rows zeroed (so the scale
    // reflects the normal range and the zeros contribute nothing to the product).
    let mut x_int = x.to_vec();
    let mut w_int = w.to_vec();
    for (j, &is_out) in outlier.iter().enumerate()
    {
        if is_out
        {
            for i in 0..m
            {
                x_int[i * k + j] = 0.0;
            }
            for l in 0..n
            {
                w_int[j * n + l] = 0.0;
            }
        }
    }
    let sx = compute_scale(&x_int);
    let sw = compute_scale(&w_int);
    let acc = matmul_int8(
        &quantize_tensor(&x_int, sx),
        &quantize_tensor(&w_int, sw),
        m,
        k,
        n,
    );
    let mut out: Vec<f32> = acc.iter().map(|&a| a as f32 * sx * sw).collect();

    // fp32 path: add the exact contribution of the outlier columns.
    for (j, &is_out) in outlier.iter().enumerate()
    {
        if is_out
        {
            for i in 0..m
            {
                let xij = x[i * k + j];
                for l in 0..n
                {
                    out[i * n + l] += xij * w[j * n + l];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests_llm_int8 {
    use super::*;
    use crate::nn::PcgEngine;

    fn naive_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m
        {
            for l in 0..n
            {
                let mut s = 0.0f32;
                for j in 0..k
                {
                    s += x[i * k + j] * w[j * n + l];
                }
                out[i * n + l] = s;
            }
        }
        out
    }

    /// **The LLM.int8() insight, tested.** With a few outlier feature columns,
    /// plain int8 (quantize everything) loses huge accuracy — the outliers set the
    /// scale and crush the bulk. The mixed decomposition keeps the outliers in fp32
    /// and is far closer to the fp result. Deterministic.
    #[test]
    fn int8_mixed_beats_plain_int8() {
        let mut rng = PcgEngine::new(23);
        let (m, k, n) = (6usize, 16usize, 8usize);
        let mut x = vec![0.0f32; m * k];
        for i in 0..m
        {
            for j in 0..k
            {
                let v = rng.float_signed();
                // Columns 3 and 10 are outlier features (≈ ×75 the bulk).
                x[i * k + j] = if j == 3 || j == 10 { v * 30.0 } else { v * 0.4 };
            }
        }
        let w: Vec<f32> = (0..k * n).map(|_| rng.float_signed()).collect();

        let fp = naive_matmul(&x, &w, m, k, n);
        let mixed = int8_mixed_matmul(&x, &w, m, k, n, 6.0);

        // Plain int8: quantize all of X and all of W.
        let (sx, sw) = (compute_scale(&x), compute_scale(&w));
        let plain_acc = matmul_int8(&quantize_tensor(&x, sx), &quantize_tensor(&w, sw), m, k, n);
        let plain: Vec<f32> = plain_acc.iter().map(|&a| a as f32 * sx * sw).collect();

        let err = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(&p, &q)| (p - q) * (p - q)).sum()
        };
        let (e_mixed, e_plain) = (err(&mixed, &fp), err(&plain, &fp));
        assert!(
            e_mixed < 0.5 * e_plain,
            "LLM.int8 not better than plain int8: {e_mixed} vs {e_plain}"
        );
        // Determinism.
        assert_eq!(mixed, int8_mixed_matmul(&x, &w, m, k, n, 6.0));
    }

    /// With no outliers below the threshold, LLM.int8() reduces to a pure int8
    /// matmul (no fp32 correction added) — and still approximates fp well.
    #[test]
    fn int8_mixed_no_outliers_is_plain_int8() {
        let (m, k, n) = (4usize, 8usize, 4usize);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.2).cos()).collect();
        let mixed = int8_mixed_matmul(&x, &w, m, k, n, 6.0);
        let (sx, sw) = (compute_scale(&x), compute_scale(&w));
        let plain_acc = matmul_int8(&quantize_tensor(&x, sx), &quantize_tensor(&w, sw), m, k, n);
        let plain: Vec<f32> = plain_acc.iter().map(|&a| a as f32 * sx * sw).collect();
        for (a, b) in mixed.iter().zip(&plain)
        {
            assert!((a - b).abs() < 1e-6, "no-outlier mismatch: {a} vs {b}");
        }
    }
}

// ----- NEON int8 (aarch64) : matmul entier accelere, bit-exact vs scalaire -----

/// Produit scalaire int8 sur k elements contigus, accumulation i32 (NEON aarch64).
#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_neon(a: *const i8, b: *const i8, k: usize) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        let mut kk = 0usize;
        while kk + 16 <= k
        {
            let va = vld1q_s8(a.add(kk));
            let vb = vld1q_s8(b.add(kk));
            let lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            let hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
            kk += 16;
        }
        let mut sum = vaddvq_s32(acc);
        while kk < k
        {
            sum += (*a.add(kk)) as i32 * (*b.add(kk)) as i32;
            kk += 1;
        }
        sum
    }
}

/// Matmul int8 accelere NEON (aarch64). b est transpose en interne pour des acces
/// contigus. Resultat i32 strictement identique a matmul_int8 (somme entiere =>
/// independante de l'ordre), donc deterministe et bit-exact vs le scalaire.
#[cfg(target_arch = "aarch64")]
pub fn matmul_int8_neon(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut bt = vec![0i8; n * k];
    for kk in 0..k
    {
        for j in 0..n
        {
            bt[j * k + kk] = b[kk * n + j];
        }
    }
    let mut out = vec![0i32; m * n];
    for i in 0..m
    {
        let arow = a[i * k..i * k + k].as_ptr();
        for j in 0..n
        {
            let brow = bt[j * k..j * k + k].as_ptr();
            out[i * n + j] = unsafe { dot_i8_neon(arow, brow, k) };
        }
    }
    out
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests_neon {
    use super::*;

    #[test]
    fn neon_matches_scalar_bit_exact() {
        let (m, k, n) = (7usize, 50usize, 9usize); // k % 16 != 0 -> teste le tail
        let mut s: u64 = 0x1234;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 56) as i64 - 128) as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| nxt()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| nxt()).collect();
        assert_eq!(
            matmul_int8(&a, &b, m, k, n),
            matmul_int8_neon(&a, &b, m, k, n),
            "NEON != scalaire (doit etre bit-exact)"
        );
    }
}

// ===== OmniQuant — learnable weight clipping (Shao et al., ICLR 2024) ==========

/// Result of [`omniquant_quantize`]: per-output-channel symmetric codes with the
/// learned clip factors and scales.
pub struct OmniQuantResult {
    /// Quantized weights `(in_features, out_features)` row-major, codes in
    /// `[−qmax, qmax]`.
    pub codes: Vec<i8>,
    /// Per-output-channel dequantization scale.
    pub scales: Vec<f32>,
    /// Per-output-channel learned clip factor `γ ∈ (0, 1]` (`1` = round-to-nearest).
    pub clips: Vec<f32>,
    /// Input features.
    pub in_features: usize,
    /// Output features.
    pub out_features: usize,
}

impl OmniQuantResult {
    /// Reconstruct the dequantized weight matrix `(in_features, out_features)`.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut w = vec![0.0f32; self.in_features * self.out_features];
        for i in 0..self.in_features
        {
            for o in 0..self.out_features
            {
                w[i * self.out_features + o] =
                    self.codes[i * self.out_features + o] as f32 * self.scales[o];
            }
        }
        w
    }
}

/// Symmetric quantization error of a channel at clip factor `gamma` (range
/// `γ·max|w|`): `Σ (w − q·scale)²` with `q = clamp(round(w/scale), ±qmax)`.
fn omniquant_channel_error(col: &[f32], maxabs: f32, qmax: f32, gamma: f32) -> (f32, f32) {
    let scale = gamma * maxabs / qmax;
    if scale == 0.0
    {
        return (0.0, 1.0);
    }
    let mut err = 0.0f32;
    for &w in col
    {
        let q = (w / scale).round().clamp(-qmax, qmax);
        let d = w - q * scale;
        err += d * d;
    }
    (err, scale)
}

/// **OmniQuant** weight quantization with **Learnable Weight Clipping** (Shao et
/// al., ICLR 2024, arXiv:2308.13137). Round-to-nearest quantizes each output
/// channel over its full range `[−max|w|, max|w|]`; with heavy-tailed weights that
/// wastes most code levels on rare outliers. OmniQuant instead learns a per-channel
/// clip factor `γ ∈ (0, 1]` that **shrinks** the range to `γ·max|w|`, trading a
/// little clipping error on the outliers for much finer steps on the bulk — found
/// here by a deterministic search over a grid that **includes `γ = 1`** (plain
/// RTN), so the result is never worse than RTN and strictly better whenever
/// outliers are present. `bits ≤ 8`, symmetric; weights are `(in, out)` row-major.
pub fn omniquant_quantize(
    weight: &[f32],
    in_features: usize,
    out_features: usize,
    bits: u32,
    grid: usize,
) -> OmniQuantResult {
    assert_eq!(weight.len(), in_features * out_features, "weight size");
    assert!((2..=8).contains(&bits), "omniquant: bits in 2..=8");
    assert!(grid >= 1, "omniquant: grid >= 1");
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut scales = vec![1.0f32; out_features];
    let mut clips = vec![1.0f32; out_features];
    let mut codes = vec![0i8; in_features * out_features];

    let mut col = vec![0.0f32; in_features];
    for o in 0..out_features
    {
        for i in 0..in_features
        {
            col[i] = weight[i * out_features + o];
        }
        let maxabs = col.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if maxabs == 0.0
        {
            continue;
        }
        // Search γ over the grid (g/grid for g=1..=grid ⇒ includes γ=1 = RTN).
        let (mut best_err, mut best_gamma, mut best_scale) = (f32::INFINITY, 1.0, maxabs / qmax);
        for g in 1..=grid
        {
            let gamma = g as f32 / grid as f32;
            let (err, scale) = omniquant_channel_error(&col, maxabs, qmax, gamma);
            if err < best_err
            {
                best_err = err;
                best_gamma = gamma;
                best_scale = scale;
            }
        }
        scales[o] = best_scale;
        clips[o] = best_gamma;
        for i in 0..in_features
        {
            let q = (col[i] / best_scale).round().clamp(-qmax, qmax);
            codes[i * out_features + o] = q as i8;
        }
    }
    OmniQuantResult {
        codes,
        scales,
        clips,
        in_features,
        out_features,
    }
}

#[cfg(test)]
mod omniquant_tests {
    use super::*;
    use crate::nn::rng::PcgEngine;

    /// Round-to-nearest reconstruction error over the **full** range (γ = 1).
    fn rtn_error(weight: &[f32], in_f: usize, out_f: usize, bits: u32) -> f32 {
        let qmax = ((1i32 << (bits - 1)) - 1) as f32;
        let mut err = 0.0f32;
        for o in 0..out_f
        {
            let mut maxabs = 0.0f32;
            for i in 0..in_f
            {
                maxabs = maxabs.max(weight[i * out_f + o].abs());
            }
            let scale = if maxabs == 0.0 { 1.0 } else { maxabs / qmax };
            for i in 0..in_f
            {
                let w = weight[i * out_f + o];
                let q = (w / scale).round().clamp(-qmax, qmax);
                let d = w - q * scale;
                err += d * d;
            }
        }
        err
    }

    fn omniquant_error(res: &OmniQuantResult, weight: &[f32]) -> f32 {
        res.dequantize()
            .iter()
            .zip(weight)
            .map(|(&d, &w)| (d - w) * (d - w))
            .sum()
    }

    /// **OmniQuant beats round-to-nearest** on heavy-tailed weights: learning a
    /// per-channel clip that ignores the rare outliers shrinks the step size on the
    /// bulk, cutting the reconstruction error well below RTN — and at least one
    /// channel actually clips (`γ < 1`).
    #[test]
    fn omniquant_beats_rtn_on_heavy_tailed_weights() {
        let mut rng = PcgEngine::new(7);
        let (in_f, out_f) = (64usize, 8usize);
        let mut w = vec![0.0f32; in_f * out_f];
        for v in w.iter_mut()
        {
            // Bulk ~ small Gaussian-ish; rare large outliers.
            let mut x = (rng.float_signed() + rng.float_signed()) * 0.1;
            if rng.float() < 0.03
            {
                x += rng.float_signed() * 5.0;
            }
            *v = x;
        }
        let res = omniquant_quantize(&w, in_f, out_f, 4, 64);
        let e_omni = omniquant_error(&res, &w);
        let e_rtn = rtn_error(&w, in_f, out_f, 4);
        assert!(
            e_omni < e_rtn,
            "OmniQuant not better than RTN: omni={e_omni} rtn={e_rtn}"
        );
        assert!(
            res.clips.iter().any(|&g| g < 1.0),
            "no channel learned to clip"
        );
    }

    /// OmniQuant is **never worse than RTN** (the grid includes `γ = 1`): on
    /// outlier-free near-uniform weights it falls back to RTN, and it is
    /// deterministic.
    #[test]
    fn omniquant_never_worse_than_rtn_and_deterministic() {
        let mut rng = PcgEngine::new(3);
        let (in_f, out_f) = (32usize, 4usize);
        let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
        let res = omniquant_quantize(&w, in_f, out_f, 4, 32);
        let e_omni = omniquant_error(&res, &w);
        let e_rtn = rtn_error(&w, in_f, out_f, 4);
        assert!(
            e_omni <= e_rtn + 1e-6,
            "omni {e_omni} worse than rtn {e_rtn}"
        );
        // Determinism: same codes/scales on a repeat.
        let res2 = omniquant_quantize(&w, in_f, out_f, 4, 32);
        assert_eq!(res.codes, res2.codes);
        assert_eq!(
            res.scales.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            res2.scales.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
        );
    }
}

// ====================================================================
// AQLM — Additive Quantization for Language Models (Egiazarian et al.,
// ICML 2024, arXiv:2401.06118). Roadmap #70.
// ====================================================================

/// Result of [`quantize_aqlm`]: `num_codebooks` learned codebooks (each
/// `codebook_size` codewords of dimension `group_size`) plus, per weight group, one
/// code index into **each** codebook. The group is reconstructed by **summing** the
/// chosen codewords — *additive* quantization.
pub struct AqlmResult {
    codebooks: Vec<Vec<f32>>, // M codebooks, each K·g floats (row-major K×g)
    codes: Vec<usize>,        // n_groups · M, codes[i·M + m] = index into codebook m
    group_size: usize,
    num_codebooks: usize,
    codebook_size: usize,
    n_weights: usize,
}

impl AqlmResult {
    /// Reconstruct the weights: each group is `Σ_m codebook_m[code_m]`.
    pub fn dequantize(&self) -> Vec<f32> {
        let (g, m) = (self.group_size, self.num_codebooks);
        let n_groups = self.codes.len().checked_div(m).unwrap_or(0);
        let mut out = vec![0.0f32; n_groups * g];
        for i in 0..n_groups
        {
            for (cb, codebook) in self.codebooks.iter().enumerate()
            {
                let a = self.codes[i * m + cb];
                for t in 0..g
                {
                    out[i * g + t] += codebook[a * g + t];
                }
            }
        }
        out.truncate(self.n_weights);
        out
    }

    /// The flat code indices (`n_groups · num_codebooks`).
    pub fn codes(&self) -> &[usize] {
        &self.codes
    }

    /// Effective bits per weight: `num_codebooks·log₂(codebook_size)/group_size`
    /// (codebook storage is amortised over all groups).
    pub fn bits_per_weight(&self) -> f32 {
        self.num_codebooks as f32 * (self.codebook_size as f32).log2() / self.group_size as f32
    }
}

/// Index of the nearest codeword in a flat `K×g` codebook to vector `v` (squared
/// L2; ties to the lower index — deterministic).
fn nearest_codeword(codebook: &[f32], k: usize, g: usize, v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for j in 0..k
    {
        let mut d = 0.0f32;
        for t in 0..g
        {
            let e = v[t] - codebook[j * g + t];
            d += e * e;
        }
        if d < bd
        {
            bd = d;
            best = j;
        }
    }
    best
}

/// Deterministic vector k-means: `k` centroids of dimension `g` fitted to `data`
/// (evenly-spaced initialisation, `iters` Lloyd steps; empty clusters keep their
/// previous value). Returns a flat `K×g` codebook.
fn vec_kmeans(data: &[Vec<f32>], k: usize, g: usize, iters: usize) -> Vec<f32> {
    let n = data.len();
    let mut cents = vec![0.0f32; k * g];
    if n == 0
    {
        return cents;
    }
    for j in 0..k
    {
        let idx = (j * n / k).min(n - 1);
        cents[j * g..j * g + g].copy_from_slice(&data[idx][..g]);
    }
    for _ in 0..iters
    {
        let mut sum = vec![0.0f32; k * g];
        let mut cnt = vec![0usize; k];
        for v in data
        {
            let a = nearest_codeword(&cents, k, g, v);
            for t in 0..g
            {
                sum[a * g + t] += v[t];
            }
            cnt[a] += 1;
        }
        for j in 0..k
        {
            if cnt[j] > 0
            {
                for t in 0..g
                {
                    cents[j * g + t] = sum[j * g + t] / cnt[j] as f32;
                }
            }
        }
    }
    cents
}

/// **AQLM** — additive (multi-codebook) quantization. The weights are split into
/// groups of `group_size`; each group vector is approximated by the **sum** of one
/// codeword taken from each of `num_codebooks` learned codebooks (each with
/// `codebook_size` entries). Codebooks are initialised by **residual k-means** and
/// then refined by **alternating optimisation** — repeatedly re-encode every group
/// (greedy residual assignment) and re-fit each codebook by least squares given the
/// other codebooks' current contributions (AQLM's beam search is simplified here to
/// greedy residual assignment). Because the codewords are *vectors*, additive
/// quantization captures cross-dimension structure that scalar round-to-nearest
/// cannot, so it reconstructs better at the same low bit budget. Deterministic.
pub fn quantize_aqlm(
    w: &[f32],
    group_size: usize,
    num_codebooks: usize,
    codebook_size: usize,
    iters: usize,
) -> AqlmResult {
    let (g, m, k) = (group_size, num_codebooks, codebook_size);
    assert!(g >= 1 && m >= 1 && k >= 1, "AQLM: sizes must be ≥ 1");
    let n_weights = w.len();
    let n_groups = n_weights.div_ceil(g);
    // Split (zero-padding the final partial group).
    let mut groups: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
    for i in 0..n_groups
    {
        let mut grp = vec![0.0f32; g];
        for (t, gt) in grp.iter_mut().enumerate()
        {
            let idx = i * g + t;
            if idx < n_weights
            {
                *gt = w[idx];
            }
        }
        groups.push(grp);
    }
    // Initialise codebooks by residual k-means.
    let mut codebooks: Vec<Vec<f32>> = Vec::with_capacity(m);
    let mut residual: Vec<Vec<f32>> = groups.clone();
    for _ in 0..m
    {
        let cents = vec_kmeans(&residual, k, g, iters);
        for grp in residual.iter_mut()
        {
            let a = nearest_codeword(&cents, k, g, grp);
            for t in 0..g
            {
                grp[t] -= cents[a * g + t];
            }
        }
        codebooks.push(cents);
    }
    // Greedy residual encoding of every group across all codebooks.
    let encode_all = |codebooks: &[Vec<f32>]| -> Vec<usize> {
        let mut codes = vec![0usize; n_groups * m];
        for (i, grp) in groups.iter().enumerate()
        {
            let mut r = grp.clone();
            for (cb, codebook) in codebooks.iter().enumerate()
            {
                let a = nearest_codeword(codebook, k, g, &r);
                codes[i * m + cb] = a;
                for t in 0..g
                {
                    r[t] -= codebook[a * g + t];
                }
            }
        }
        codes
    };
    // Alternating refinement: re-encode, then re-fit each codebook by LS on the
    // partial residual (group minus the other codebooks' contributions).
    for _ in 0..iters
    {
        let codes = encode_all(&codebooks);
        for cb in 0..m
        {
            let mut sum = vec![0.0f32; k * g];
            let mut cnt = vec![0usize; k];
            for (i, grp) in groups.iter().enumerate()
            {
                let mut pr = grp.clone();
                for (mm, codebook) in codebooks.iter().enumerate()
                {
                    if mm == cb
                    {
                        continue;
                    }
                    let a = codes[i * m + mm];
                    for t in 0..g
                    {
                        pr[t] -= codebook[a * g + t];
                    }
                }
                let a = codes[i * m + cb];
                for t in 0..g
                {
                    sum[a * g + t] += pr[t];
                }
                cnt[a] += 1;
            }
            for j in 0..k
            {
                if cnt[j] > 0
                {
                    for t in 0..g
                    {
                        codebooks[cb][j * g + t] = sum[j * g + t] / cnt[j] as f32;
                    }
                }
            }
        }
    }
    let codes = encode_all(&codebooks);
    AqlmResult {
        codebooks,
        codes,
        group_size: g,
        num_codebooks: m,
        codebook_size: k,
        n_weights,
    }
}

#[cfg(test)]
mod aqlm_tests {
    use super::*;
    use crate::nn::rng::PcgEngine;

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            / a.len() as f32
    }

    /// Scalar symmetric round-to-nearest error at `bits` over the full range.
    fn rtn_mse(w: &[f32], bits: u32) -> f32 {
        let qmax = ((1i32 << (bits - 1)) - 1).max(1) as f32;
        let maxabs = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let scale = if maxabs == 0.0 { 1.0 } else { maxabs / qmax };
        let deq: Vec<f32> = w
            .iter()
            .map(|&x| (x / scale).round().clamp(-qmax, qmax) * scale)
            .collect();
        mse(w, &deq)
    }

    /// AQLM beats scalar RTN at a matched ~2-bit budget on **structured** weights
    /// (groups built from a few prototype directions) — additive *vector* codebooks
    /// capture cross-dimension structure scalar quantisation cannot.
    #[test]
    fn aqlm_beats_rtn_on_structured_weights() {
        let (g, m, k) = (4usize, 2usize, 16usize); // 2·log2(16)/4 = 2 bits/weight
        let n_groups = 256usize;
        let mut rng = PcgEngine::new(20);
        // A few prototype directions; each group ≈ scaled prototype + small noise.
        let n_proto = 6usize;
        let protos: Vec<Vec<f32>> = (0..n_proto)
            .map(|_| (0..g).map(|_| rng.float_signed()).collect())
            .collect();
        let mut w = Vec::with_capacity(n_groups * g);
        for i in 0..n_groups
        {
            let p = &protos[i % n_proto];
            let scale = 0.5 + (i % 5) as f32 * 0.4;
            for &pt in p.iter()
            {
                w.push(scale * pt + 0.03 * rng.float_signed());
            }
        }
        let res = quantize_aqlm(&w, g, m, k, 8);
        let deq = res.dequantize();
        assert_eq!(deq.len(), w.len());
        let e_aqlm = mse(&w, &deq);
        let e_rtn = rtn_mse(&w, 2);
        assert!(
            e_aqlm < 0.7 * e_rtn,
            "AQLM {e_aqlm} not better than 0.7·RTN {e_rtn}"
        );
        assert!((res.bits_per_weight() - 2.0).abs() < 1e-6);
    }

    /// Round-trip dimensions (incl. a non-divisible length) and bit-exact determinism.
    #[test]
    fn aqlm_roundtrip_shape_and_determinism() {
        let mut rng = PcgEngine::new(21);
        let w: Vec<f32> = (0..103).map(|_| rng.float_signed()).collect(); // not a multiple of g
        let res = quantize_aqlm(&w, 4, 2, 8, 5);
        assert_eq!(res.dequantize().len(), w.len());
        let res2 = quantize_aqlm(&w, 4, 2, 8, 5);
        assert_eq!(res.codes(), res2.codes());
        let (d1, d2) = (res.dequantize(), res2.dequantize());
        assert_eq!(
            d1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            d2.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

// ====================================================================
// QuIP# — incoherence processing + E8 lattice codebook (Tseng et al.,
// ICML 2024, arXiv:2402.04396). Roadmap #64.
// ====================================================================

/// In-place length-8 **fast Walsh-Hadamard transform**, normalised by `1/√8` so the
/// transform is **orthogonal** and **involutory** (`FWHT(FWHT(x)) = x`).
fn fwht8(a: &mut [f32; 8]) {
    let mut len = 1usize;
    while len < 8
    {
        let mut i = 0usize;
        while i < 8
        {
            for j in i..i + len
            {
                let (x, y) = (a[j], a[j + len]);
                a[j] = x + y;
                a[j + len] = x - y;
            }
            i += len * 2;
        }
        len *= 2;
    }
    let inv = 1.0 / (8.0f32).sqrt();
    for v in a.iter_mut()
    {
        *v *= inv;
    }
}

/// Nearest point of the integer lattice **D8** (`{z ∈ ℤ⁸ : Σ zᵢ even}`) to `y`:
/// round to nearest integers; if the coordinate sum is odd, flip the parity by
/// re-rounding the single coordinate with the largest rounding error.
fn nearest_d8(y: &[f32; 8]) -> [f32; 8] {
    let mut f = [0.0f32; 8];
    let mut sum = 0i64;
    let (mut worst, mut worst_err) = (0usize, -1.0f32);
    for (i, fi) in f.iter_mut().enumerate()
    {
        let r = y[i].round();
        *fi = r;
        sum += r as i64;
        let e = (y[i] - r).abs();
        if e > worst_err
        {
            worst_err = e;
            worst = i;
        }
    }
    if sum % 2 != 0
    {
        // Re-round the worst coordinate to its second-nearest integer (flips parity).
        f[worst] += if y[worst] - f[worst] >= 0.0
        {
            1.0
        }
        else
        {
            -1.0
        };
    }
    f
}

/// Nearest point of the **E8 lattice** (`D8 ∪ (D8 + ½·1)`) to `x` — the densest
/// lattice packing in 8 dimensions, with a lower quantization error than the cubic
/// (integer) grid at the **same** point density. Closed-form Conway-Sloane decoder:
/// take the nearer of the best `D8` point and the best `D8 + ½` point.
pub fn nearest_e8(x: &[f32; 8]) -> [f32; 8] {
    let a = nearest_d8(x);
    let mut xs = *x;
    for v in xs.iter_mut()
    {
        *v -= 0.5;
    }
    let mut b = nearest_d8(&xs);
    for v in b.iter_mut()
    {
        *v += 0.5;
    }
    let dist = |p: &[f32; 8]| -> f32 { (0..8).map(|i| (x[i] - p[i]).powi(2)).sum() };
    if dist(&a) <= dist(&b) { a } else { b }
}

/// Apply the **random Hadamard transform** (the QuIP# incoherence step) block-wise
/// over groups of 8: flip signs by a seeded `±1` diagonal, then a length-8 FWHT. The
/// map is **orthogonal**; its inverse is [`inverse_random_hadamard_transform`]. Pads to a
/// multiple of 8 with zeros. `seed` selects the (reproducible) sign pattern.
pub fn random_hadamard_transform(w: &[f32], seed: u64) -> Vec<f32> {
    let n = w.len();
    let nb = n.div_ceil(8);
    let mut rng = crate::nn::PcgEngine::new(seed);
    let signs: Vec<f32> = (0..nb * 8)
        .map(|_| if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut out = vec![0.0f32; nb * 8];
    for b in 0..nb
    {
        let mut blk = [0.0f32; 8];
        for (t, bt) in blk.iter_mut().enumerate()
        {
            let idx = b * 8 + t;
            let v = if idx < n { w[idx] } else { 0.0 };
            *bt = v * signs[idx];
        }
        fwht8(&mut blk);
        out[b * 8..b * 8 + 8].copy_from_slice(&blk);
    }
    out
}

/// Inverse of [`random_hadamard_transform`] for the same `seed`: the length-8 FWHT
/// (its own inverse) then the `±1` sign flip. Returns `len` values (drops the padding).
pub fn inverse_random_hadamard_transform(wr: &[f32], seed: u64, len: usize) -> Vec<f32> {
    let nb = wr.len() / 8;
    let mut rng = crate::nn::PcgEngine::new(seed);
    let signs: Vec<f32> = (0..nb * 8)
        .map(|_| if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut out = vec![0.0f32; nb * 8];
    for b in 0..nb
    {
        let mut blk = [0.0f32; 8];
        blk.copy_from_slice(&wr[b * 8..b * 8 + 8]);
        fwht8(&mut blk);
        for (t, bt) in blk.iter().enumerate()
        {
            out[b * 8 + t] = bt * signs[b * 8 + t];
        }
    }
    out.truncate(len);
    out
}

/// Result of [`quantize_quip`]: the E8 lattice codes (stored as `2×` coordinates so
/// half-integers fit an integer), the shared scale, the seed (to regenerate the
/// sign pattern) and the original length.
pub struct QuipResult {
    codes: Vec<i16>, // 2× E8 coordinates, per (padded) weight
    scale: f32,
    seed: u64,
    n_weights: usize,
}

impl QuipResult {
    /// Reconstruct the weights: scale the E8 lattice points back, then invert the
    /// random Hadamard transform.
    pub fn dequantize(&self) -> Vec<f32> {
        let wr: Vec<f32> = self
            .codes
            .iter()
            .map(|&c| (c as f32 / 2.0) * self.scale)
            .collect();
        inverse_random_hadamard_transform(&wr, self.seed, self.n_weights)
    }

    /// The E8 codes (`2×` coordinates).
    pub fn codes(&self) -> &[i16] {
        &self.codes
    }
}

/// **QuIP#** — quantize weights with **incoherence processing** + an **E8 lattice**
/// codebook. First a [`random_hadamard_transform`] rotates the weights: this spreads
/// outliers across coordinates (incoherence), shrinking the dynamic range so the
/// fixed `2^bits` levels resolve the bulk far better at the **same** bit budget.
/// The rotated weights are then quantized **block-wise in 8-D** to the nearest
/// [`nearest_e8`] lattice point (scaled, bounded to `±qmax`), which has lower
/// quantization error than the per-coordinate cubic grid at equal density.
/// Deterministic. (QuIP#'s larger global Hadamard and curated E8P codebook are
/// simplified here to a per-8-block Hadamard and the plain E8 lattice.)
pub fn quantize_quip(w: &[f32], bits: u32, seed: u64) -> QuipResult {
    assert!((1..=8).contains(&bits), "QuIP#: bits must be in 1..=8");
    let n = w.len();
    let wr = random_hadamard_transform(w, seed);
    let qmax = ((1i32 << (bits - 1)) - 1).max(1) as f32;
    let maxabs = wr.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let scale = if maxabs == 0.0 { 1.0 } else { maxabs / qmax };
    let nb = wr.len() / 8;
    let mut codes = vec![0i16; wr.len()];
    for b in 0..nb
    {
        let mut y = [0.0f32; 8];
        for (t, yt) in y.iter_mut().enumerate()
        {
            *yt = (wr[b * 8 + t] / scale).clamp(-qmax, qmax);
        }
        let e8 = nearest_e8(&y);
        for t in 0..8
        {
            // Bound to the ±qmax box and store 2× (E8 coords are in ½ℤ).
            let c = e8[t].clamp(-qmax, qmax);
            codes[b * 8 + t] = (c * 2.0).round() as i16;
        }
    }
    QuipResult {
        codes,
        scale,
        seed,
        n_weights: n,
    }
}

#[cfg(test)]
mod quip_tests {
    use super::*;
    use crate::nn::PcgEngine;

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            / a.len() as f32
    }

    /// The random Hadamard transform is **orthogonal** (round-trips to the input) and
    /// **reduces the dynamic range** of an outlier-heavy weight (incoherence).
    #[test]
    fn rht_orthogonal_and_reduces_outliers() {
        let mut w = vec![0.02f32; 64];
        w[5] = 3.0; // a big outlier
        w[40] = -2.5;
        let wr = random_hadamard_transform(&w, 7);
        let back = inverse_random_hadamard_transform(&wr, 7, w.len());
        for (a, b) in back.iter().zip(&w)
        {
            assert!((a - b).abs() < 1e-5, "RHT not orthogonal: {a} vs {b}");
        }
        let max_w = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let max_wr = wr.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        assert!(
            max_wr < 0.6 * max_w,
            "incoherence did not reduce range: {max_wr} vs {max_w}"
        );
    }

    /// The E8 decoder returns a **valid** lattice point (all-integer or all-half-integer
    /// coords, even integer-sum) and, **on average**, quantizes better than the cubic
    /// (integer) grid — the E8 packing gain.
    #[test]
    fn nearest_e8_valid_and_beats_cubic() {
        let mut rng = PcgEngine::new(31);
        let (mut e8_err, mut cubic_err) = (0.0f32, 0.0f32);
        for _ in 0..4000
        {
            let mut x = [0.0f32; 8];
            for v in x.iter_mut()
            {
                *v = 2.5 * rng.float_signed();
            }
            let p = nearest_e8(&x);
            // Validity: coords all-even-or-all-odd in 2× form (all integer or all
            // half-integer) with even coordinate sum (⇔ 2·Σ ≡ 0 mod 4).
            let two_c: Vec<i64> = p.iter().map(|&c| (2.0 * c).round() as i64).collect();
            let all_even = two_c.iter().all(|&t| t % 2 == 0);
            let all_odd = two_c.iter().all(|&t| t % 2 != 0);
            assert!(all_even || all_odd, "E8 point not lattice-aligned: {p:?}");
            assert!(
                two_c.iter().sum::<i64>() % 4 == 0,
                "E8 even-sum violated: {p:?}"
            );
            e8_err += (0..8).map(|i| (x[i] - p[i]).powi(2)).sum::<f32>();
            let cubic: Vec<f32> = x.iter().map(|v| v.round()).collect();
            cubic_err += (0..8).map(|i| (x[i] - cubic[i]).powi(2)).sum::<f32>();
        }
        assert!(
            e8_err < 0.95 * cubic_err,
            "E8 lattice gain absent: E8 {e8_err} vs cubic {cubic_err}"
        );
    }

    /// End-to-end: QuIP# reconstructs **outlier-heavy** weights better than scalar RTN
    /// at a matched 2-bit budget (incoherence shrinks the range the fixed levels must
    /// cover; the E8 lattice adds packing gain). Plus determinism.
    #[test]
    fn quip_beats_rtn_at_2bit() {
        let mut rng = PcgEngine::new(33);
        // Mostly small weights with sparse large outliers (the hard case for RTN).
        let mut w: Vec<f32> = (0..512).map(|_| 0.05 * rng.float_signed()).collect();
        for i in (0..512).step_by(37)
        {
            w[i] = 2.0 + rng.float_signed();
        }
        let res = quantize_quip(&w, 2, 5);
        let deq = res.dequantize();
        assert_eq!(deq.len(), w.len());
        let e_quip = mse(&w, &deq);
        // Scalar RTN at 2 bits over the full range.
        let qmax = 1.0f32;
        let maxabs = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let s = maxabs / qmax;
        let rtn: Vec<f32> = w
            .iter()
            .map(|&x| (x / s).round().clamp(-qmax, qmax) * s)
            .collect();
        let e_rtn = mse(&w, &rtn);
        assert!(e_quip < e_rtn, "QuIP# {e_quip} not better than RTN {e_rtn}");
        // Determinism.
        let res2 = quantize_quip(&w, 2, 5);
        assert_eq!(res.codes(), res2.codes());
    }
}
