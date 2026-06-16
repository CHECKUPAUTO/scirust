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
            let mut sum = 0i32;
            for kk in 0..k
            {
                sum += a[i * k + kk] as i32 * b[kk * n + j] as i32;
            }
            result[i * n + j] = sum;
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
