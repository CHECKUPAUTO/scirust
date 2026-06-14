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
