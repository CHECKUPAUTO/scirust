//! # Quantification int4 (Packed) — Pilier 5
//!
//! Quantification symétrique int4 avec packing: 2 valeurs int4 par byte.
//! L'intervalle est [-7, 7] (8 niveaux positifs + 8 négatifs + 0).
//!
//! | Propriété | fp32 | int8 | int4 |
//! |-|-  ----------|-  ------|-  ------|
//! | Bits/elt | 32 | 8 | 4 |
//! | Niveaux | 2^23 | 255 | 15 |
//! | Compression | 1× | 4× | 8× |
//! | Usage | Reference | Standard | Edge |
//!
//! ## Packing
//!
//! Chaque byte contient 2 valeurs int4:
//! ```
//! byte[0] = high_nibble << 4 | low_nibble
//! high = (byte >> 4) & 0xF
//! low  = byte & 0xF
//! ```
//!
//! ## Signed int4
//!
//! Utilise l'encodage biasé: signed_val = unsigned_val - 7
//! unsigned range: [0, 15] → signed range: [-7, 8]

/// Quantifie un tenseur fp32 en int4 packed.
///
/// Retourne les bytes packés et le scale.
pub fn quantize_tensor_f32_to_i4(data: &[f32]) -> (Vec<u8>, f32) {
    let max_abs = data
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()));

    let scale = if max_abs < 1e-8 {
        1.0
    } else {
        max_abs / 7.0 // 7 est le max absolu pour int4 signed (7 bits)
    };

    let packed: Vec<u8> = data
        .chunks(2)
        .map(|chunk| {
            let lo = if chunk.first().is_some() {
                quantize_i4(chunk[0], scale)
            } else {
                0
            };
            let hi = if chunk.len() > 1 {
                quantize_i4(chunk[1], scale)
            } else {
                0
            };
            (hi << 4) | lo
        })
        .collect();

    (packed, scale)
}

/// Déquantifie un tenseur int4 packed en fp32.
pub fn dequantize_i4_to_f32(data: &[u8], scale: f32) -> Vec<f32> {
    let len = data.len() * 2;
    let mut result = Vec::with_capacity(len);

    for (i, &byte) in data.iter().enumerate() {
        let lo = (byte & 0x0F) as i8;
        let hi = ((byte >> 4) & 0x0F) as i8;

        // Signed decode: bias 7
        result.push((lo as i8).wrapping_sub(7) as f32 * scale);
        if i * 2 + 1 < len {
            result.push((hi as i8).wrapping_sub(7) as f32 * scale);
        }
    }

    result
}

/// Quantifie une valeur fp32 → int4 unsigned.
fn quantize_i4(x: f32, scale: f32) -> u8 {
    let q = (x / scale).round().clamp(-7.0, 7.0);
    ((q as i8).wrapping_add(7) as u8) & 0x0F
}

/// Unpack int4 → int8 (pour le calcul SIMD).
pub fn unpack_i4_to_i8(packed: &[u8]) -> Vec<i8> {
    let len = packed.len() * 2;
    let mut result = Vec::with_capacity(len);

    for &byte in packed {
        let lo = (byte & 0x0F).wrapping_sub(7);
        let hi = ((byte >> 4) & 0x0F).wrapping_sub(7);
        result.push(lo as i8);
        result.push(hi as i8);
    }

    result
}

/// Matmul int4 packed × int4 packed → fp32.
///
/// Décode les int4, fait le produit en int32, déquantifie le résultat.
pub fn matmul_int4_f32(
    a_packed: &[u8],
    b_packed: &[u8],
    scale_a: f32,
    scale_b: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let a = unpack_i4_to_i8(a_packed);
    let b = unpack_i4_to_i8(b_packed);

    let mut result = vec![0.0f32; m * n];
    let ak = k / 2; // packed dim
    #[allow(unused_variables)]
    let bk = k / 2;

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for kk in 0..ak {
                // Les données sont déja unpackées
                let ai = a[i * k + kk] as i32;
                let bi = b[kk * n + j] as i32;
                sum += ai * bi;
            }
            result[i * n + j] = sum as f32 * scale_a * scale_b;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i4_quantize_dequantize() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let (packed, scale) = quantize_tensor_f32_to_i4(&data);
        let recovered = dequantize_i4_to_f32(&packed, scale);
        assert_eq!(recovered.len(), 100);
        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 0.1, "int4 max error: {}", max_err);
    }

    #[test]
    fn test_i4_packing() {
        let packed = vec![0b_0101_1010]; // hi=5, lo=10 (unsigned)
        let a = unpack_i4_to_i8(&packed);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], 3); // 5 - 7 = -2 → sign bit is 1 so it wraps to 6, 6-7=-1
        assert_eq!(a[1], -2); // 10 - 7 = 3 → 1010 = -1+7 = 6, 6-7 = -1

        // Vérifier l'encodage correct
        let (packed2, _) = quantize_tensor_f32_to_i4(&[0.0, 7.0]);
        assert_eq!(packed2.len(), 1);
        // 0.0 → 7 (bias) → 0x07 low nibble
        // 7.0 → 7 (max) → 0x07 high nibble
        assert_eq!(packed2[0], (0x07 << 4) | 0x07);
    }
}
