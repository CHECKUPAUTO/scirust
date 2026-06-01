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
    let max_abs = fp32
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 127.0
    }
}

/// Matmul int8 × int8 → i32.
pub fn matmul_int8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut result = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for kk in 0..k {
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

        for (orig, rec) in original.iter().zip(recovered.iter()) {
            let error = (orig - rec).abs();
            assert!(error < scale * 1.5, "error {} exceeds threshold", error);
        }
    }

    #[test]
    fn test_quantize_clamping() {
        let large_values: Vec<f32> = vec![500.0, -500.0, 0.0];
        let scale = compute_scale(&large_values);
        let quantized = quantize_tensor(&large_values, scale);
        assert!(quantized.iter().all(|&x| x >= -128 && x <= 127));
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
    for o in 0..out_features {
        let mut max_abs = 0.0f32;
        for i in 0..in_features {
            let v = weight[i * out_features + o].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        scales[o] = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    }
    let mut q = vec![0i8; in_features * out_features];
    for i in 0..in_features {
        for o in 0..out_features {
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
    for b in 0..batch {
        for o in 0..out_features {
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
        for b in 0..batch {
            for o in 0..out_f {
                let mut s = 0.0f32;
                for i in 0..in_f {
                    s += input[b * in_f + i] * weight[i * out_f + o];
                }
                reference[b * out_f + o] = s + bias[o];
            }
        }
        let (w_q, w_scales) = quantize_per_channel(&weight, in_f, out_f);
        let out = quantized_linear_forward(&input, batch, in_f, &w_q, &w_scales, &bias, out_f);
        for (r, o) in reference.iter().zip(out.iter()) {
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
