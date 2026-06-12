//! # Quantification int8 avec décompression SIMD on-the-fly (Pilier 5)
//!
//! Implémente la quantification symétrique int8 pour les tenseurs de modèle,
//! avec décompression et calcul enregistreur directe.
//!
//! ## Précision
//!
//! La quantification int8 symétrique utilise l'intervalle [-127, 127]
//! (pas [-128, 127]) pour éviter l'asymétrie due à -128 ≠ 127.
//!
//! ## Format de stockage
//!
//! | Composant | f32 | int8 | Gain |
//! |-----|-  ----------|-  -----|-  ----------|
//! | Bandwidth | 4 bytes/elt | 1 byte/elt | 4× |
//! | Cache L2 | 4× moins d'entrées | 1 ligne/cache-line | 4× |
//! | DDR usage | N × 4 bytes | N × 1 byte | 4× |

/// Quantifie un tenseur fp32 en int8 par canal (channel-wise).
///
/// Retourne les valeurs quantifiées et le scale utilisé.
/// Le scale = max(|data|) / 127.0 pour la quantification symétrique.
pub fn quantize_tensor_f32_to_i8(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()));

    let scale = if max_abs < 1e-8 {
        1.0
    } else {
        max_abs / 127.0
    };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| {
            let q = (x / scale).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();

    (quantized, scale)
}

/// Déquantifie un tenseur int8 en fp32.
///
/// Utilisé pour les opérations qui nécessitent fp32 (optimizers, evaluation).
pub fn dequantize_i8_to_f32(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x as f32 * scale).collect()
}

/// Déquantise int8 → fp32 directement dans des registres SIMD (8-lanes AVX2).
///
/// Cette fonction est optimisée pour être appelée dans une boucle de matmul
/// où la déquantification et le produit sont fusionnés.
#[inline]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn dequantize_i8_simd(avx2: bool, a_i8: &[i8], scale: f32, out: &mut [f32]) {
    let n = a_i8.len();
    let mut i = 0;

    if avx2 && n >= 8 {
        use std::arch::x86_64::*;

        let scale_vec = _mm256_set1_ps(scale);

        while i + 8 <= n {
            // Charge 8 int8
            let lo = _mm_loadu_si128(a_i8.as_ptr().add(i) as *const __m128i);
            let hi = _mm_loadu_si128(a_i8.as_ptr().add(i + 16) as *const __m128i);

            // Sign-extend int8 → int16
            let lo16 = _mm_cvtepi8_epi16(lo);
            let hi16 = _mm_cvtepi8_epi16(hi);

            // Combine en int32 (2 × int16 → 1 × int32)
            let lo32 = _mm_cvtepi16_epi32(lo16);
            let hi32_lo = _mm_cvtepi16_epi32(_mm_shuffle_epi32(lo16, 0x55));
            let hi32_hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(hi16, 0x55));
            let hi32 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(hi16, 0xAA));
            let lo32_hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(lo16, 0xAA));

            // Convertir int32 → float32 puis × scale
            let lo_f = _mm256_cvtepi32_ps(lo32);
            let hi_f = _mm256_cvtepi32_ps(hi32);

            let result_lo = _mm256_mul_ps(lo_f, scale_vec);
            let result_hi = _mm256_mul_ps(hi_f, scale_vec);

            _mm256_storeu_ps(out.as_mut_ptr().add(i), result_lo);
            _mm256_storeu_ps(out.as_mut_ptr().add(i + 8), result_hi);

            i += 16;
        }
    }

    // Fallback scalar
    while i < n {
        out[i] = a_i8[i] as f32 * scale;
        i += 1;
    }
}

/// Matmul int8 × int8 → fp32 avec accumulation en int32 puis déquantification.
///
/// Utilise les accumulateurs int32 pour éviter le overflow de int16.
/// La déquantification se fait après l'accumulation complète.
pub fn matmul_int8_f32(
    a: &[i8],
    b: &[i8],
    scale_a: f32,
    scale_b: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let result_rows = m;
    let result_cols = n;
    let mut result = vec![0.0f32; result_rows * result_cols];

    // Accumulation en int32
    let mut acc_i32 = vec![0i32; result_rows * result_cols];

    for i in 0..result_rows {
        for j in 0..result_cols {
            let mut sum = 0i32;
            for kk in 0..k {
                sum += a[i * k + kk] as i32 * b[kk * n + j] as i32;
            }
            acc_i32[i * result_cols + j] = sum;
        }
    }

    // Déquantification: fp32 = int32 * scale_a * scale_b
    let combined_scale = scale_a * scale_b;
    for (i, &val) in acc_i32.iter().enumerate() {
        result[i] = val as f32 * combined_scale;
    }

    result
}

/// Matmul int8 × int8 → fp32 avec SIMD AVX2 pour la boulette intérieure.
#[inline]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn matmul_int8_avx2_f32(
    a: &[i8],
    b: &[i8],
    scale_a: f32,
    scale_b: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let b_col = j;
            let mut sum = [0i32; 8]; // 8 accumulateurs SIMD
            let mut ii = 0;

            // Boulette intérieure avec SIMD: 8 accumulateurs en parallèle
            while ii + 8 <= k {
                let a_row = _mm_loadu_si128(a.as_ptr().add(i * k + ii) as *const __m128i);

                // Charge 8 éléments de la colonne j de B
                let mut col_vec = [0i8; 8];
                for c in 0..8 {
                    col_vec[c] = b[(ii + c) * n + b_col];
                }
                let b_col_vec = _mm_loadu_si128(col_vec.as_ptr() as *const __m128i);

                // Produit int8 → int32 (8 × 8 = 64 produits)
                // Utiliser _mm_maddubs_epi16 + _mm_madd_epi16
                let a_sign = _mm_cmpgt_epi8(_mm_setzero_si128(), a_row); // masque de signe
                let b_sign = _mm_cmpgt_epi8(_mm_setzero_si128(), _mm_loadu_si128(b_col_vec as *const __m128i));

                // Simplification: accumuler en int16 puis int32
                let a_u = _mm_and_si128(a_row, _mm_set1_epi8(0x7F));
                let b_u = _mm_and_si128(b_col_vec, _mm_set1_epi8(0x7F));
                let prod = _mm_maddubs_epi16(a_u, b_u);
                let acc = _mm_madd_epi16(prod, _mm_set1_epi16(1));

                sum[0] += _mm_cvtsi128_si32(acc);
                ii += 8;
            }

            // Scalar pour le reste
            for kk in ii..k {
                result[i * n + j] += a[i * k + kk] as f32 * b[kk * n + j] as f32;
            }

            result[i * n + j] *= scale_a * scale_b;
        }
    }

    result
}
