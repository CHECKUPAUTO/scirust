//! # Quantification bf16 (Bfloat16) — Pilier 5
//!
//! Bfloat16 est une format 16-bit dérivé de IEEE 754 float.
//! Il conserve la même précision (8 bits de mantisse) que float32
//! mais réduit l'exposant de 8 à 9 bits (range ±127 au lieu de ±126).
//!
//! | Propriété | fp32 | bf16 | fp16 |
//! |-|-  -------|-  ----------|-  --------|
//! | Bits | 32 | 16 | 16 |
//! | Mantisse | 23 | 7 | 10 |
//! | Exposant | 8 | 8 | 5 |
//! | Range | ±3.4×10^38 | ±3.4×10^38 | ±65504 |
//! | Précision | ~7 déc. | ~2 déc. | ~3 déc. |
//!
//! ## Avantages pour le deep learning
//!
//! - Pas de loss de range pendant l'entraînement (contrairement à fp16)
//! - Conversion fp32 ↔ bf16 = troncature des 16 bits de poids faibles
//! - Support matériel sur NVIDIA Ampere+ (Tensor Cores) et ARM SVE2
//! - Bandwidth divisée par 2 sans loss de convergence

/// Convertit un f32 en bf16 (troncature des 16 LSB).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Tronquer les 16 bits de poids faibles
    ((bits >> 16) | (bits & 0x8000)) as u16
}

/// Convertit un bf16 en f32 (extension par zéros).
#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    let bits = (x as u32) << 16;
    f32::from_bits(bits)
}

/// Quantifie un tenseur fp32 en bf16.
pub fn quantize_tensor_f32_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f32_to_bf16(x)).collect()
}

/// Déquantifie un tenseur bf16 en fp32.
pub fn dequantize_bf16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&x| bf16_to_f32(x)).collect()
}

/// Conversion bf16 par batch SIMD (NEON ARM64).
///
/// 4 floats → 4 bf16 par itération.
#[inline]
#[cfg(target_arch = "aarch64")]
pub fn f32_to_bf16_neon(data: &[f32], out: &mut [u16]) {
    #[target_feature(enable = "neon")]
    unsafe fn inner(data: &[f32], out: &mut [u16]) {
        use std::arch::aarch64::*;

        let n = data.len().min(out.len());
        let mut i = 0;

        while i + 4 <= n {
            // Charge 4 f32
            let v = unsafe { std::arch::aarch64::vld1q_f32(data.as_ptr().add(i)) };

            // Extraction de la partie haute de chaque float (bits 31-16 → bf16)
            let hi0 = vgetq_lane_u32(vreinterpretq_u32_f32(v), 0) >> 16;
            let hi1 = vgetq_lane_u32(vreinterpretq_u32_f32(v), 1) >> 16;
            let hi2 = vgetq_lane_u32(vreinterpretq_u32_f32(v), 2) >> 16;
            let hi3 = vgetq_lane_u32(vreinterpretq_u32_f32(v), 3) >> 16;

            out[i] = ((hi0 & 0x8000) as u16) | (hi0 >> 16) as u16;
            out[i + 1] = ((hi1 & 0x8000) as u16) | (hi1 >> 16) as u16;
            out[i + 2] = ((hi2 & 0x8000) as u16) | (hi2 >> 16) as u16;
            out[i + 3] = ((hi3 & 0x8000) as u16) | (hi3 >> 16) as u16;

            i += 4;
        }

        while i < n {
            out[i] = f32_to_bf16(data[i]);
            i += 1;
        }
    }
    unsafe { inner(data, out) }
}

/// Conversion bf16 → f32 par batch NEON.
#[inline]
#[cfg(target_arch = "aarch64")]
pub fn bf16_to_f32_neon(data: &[u16], out: &mut [f32]) {
    #[target_feature(enable = "neon")]
    unsafe fn inner(data: &[u16], out: &mut [f32]) {
        use std::arch::aarch64::*;

        let n = data.len().min(out.len());
        let mut i = 0;

        while i + 4 <= n {
            // Charge 4 bf16
            let ptr = unsafe { data.as_ptr().add(i) };
            let v0 = std::arch::aarch64::vget_lane_u32(std::arch::aarch64::vreinterpret_u32_u16(std::arch::aarch64::vld1_u16(ptr)), 0);
            let v1 = std::arch::aarch64::vget_lane_u32(std::arch::aarch64::vreinterpret_u32_u16(std::arch::aarch64::vld1_u16(unsafe { ptr.add(2) })), 0);

            // Extension: décaler à gauche de 16 bits
            let f0 = vsetq_lane_f32(bf16_to_f32(v0 as u16), vdupq_n_f32(0.0), 0);
            let f1 = vsetq_lane_f32(bf16_to_f32((v0 >> 16) as u16), f0, 1);
            let f2 = vsetq_lane_f32(bf16_to_f32(v1 as u16), f1, 2);
            let f3 = vsetq_lane_f32(bf16_to_f32((v1 >> 16) as u16), f2, 3);

            unsafe { std::arch::aarch64::vst1q_f32(out.as_mut_ptr().add(i), f3); }

            i += 4;
        }

        while i < n {
            out[i] = bf16_to_f32(data[i]);
            i += 1;
        }
    }
    unsafe { inner(data, out) }
}

/// Conversion bf16 par batch SIMD (AVX2 x86).
#[inline]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn f32_to_bf16_avx2(data: &[f32], out: &mut [u16]) {
    use std::arch::x86_64::*;

    let n = data.len().min(out.len());
    let mut i = 0;

    while i + 8 <= n {
        // Charge 8 f32
        let v = _mm256_loadu_ps(data.as_ptr().add(i));

        // Extraire les 16 bits de poids forts de chaque float
        // Les floats en little-endian: bits 31-16 sont dans les 2 premiers bytes
        let hi = _mm256_extract_epi32(v, 0);
        let hi1 = _mm256_extract_epi32(v, 1);
        let hi2 = _mm256_extract_epi32(v, 2);
        let hi3 = _mm256_extract_epi32(v, 3);
        let hi4 = _mm256_extract_epi32(v, 4);
        let hi5 = _mm256_extract_epi32(v, 5);
        let hi6 = _mm256_extract_epi32(v, 6);
        let hi7 = _mm256_extract_epi32(v, 7);

        out[i] = ((hi >> 16) as u16) | ((hi & 0x8000) as u16);
        out[i + 1] = ((hi1 >> 16) as u16) | ((hi1 & 0x8000) as u16);
        out[i + 2] = ((hi2 >> 16) as u16) | ((hi2 & 0x8000) as u16);
        out[i + 3] = ((hi3 >> 16) as u16) | ((hi3 & 0x8000) as u16);
        out[i + 4] = ((hi4 >> 16) as u16) | ((hi4 & 0x8000) as u16);
        out[i + 5] = ((hi5 >> 16) as u16) | ((hi5 & 0x8000) as u16);
        out[i + 6] = ((hi6 >> 16) as u16) | ((hi6 & 0x8000) as u16);
        out[i + 7] = ((hi7 >> 16) as u16) | ((hi7 & 0x8000) as u16);

        i += 8;
    }

    while i < n {
        out[i] = f32_to_bf16(data[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_bf16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 2.0, -0.25, 100.0, -100.0];
        for &v in &values {
            let bf16 = f32_to_bf16(v);
            let back = bf16_to_f32(bf16);
            let error = (v - back).abs();
            assert!(
                error < 0.01 || v == 0.0,
                "Round-trip error too large for {}: {} vs {}",
                v,
                back,
                error
            );
        }
    }

    #[test]
    fn test_quantize_dequantize() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let bf16 = quantize_tensor_f32_to_bf16(&data);
        let recovered = dequantize_bf16_to_f32(&bf16);
        assert_eq!(bf16.len(), 100);
        // Vérifier que la perte est acceptable (bf16 a ~7 bits de précision)
        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 0.02, "bf16 max error: {}", max_err);
    }
}
