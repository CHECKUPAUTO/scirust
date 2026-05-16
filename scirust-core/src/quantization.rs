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
