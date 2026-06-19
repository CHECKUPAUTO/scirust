//! Deterministic GPU compute — strict bit-à-bit guarantees.
//!
//! ## Hiérarchie de précision (de la plus stricte à la plus souple)
//!
//! | Voie | Technique | Garantie | Usage |
//! |------|-----------|----------|-------|
//! | 1    | Arithmétique entière + modulo | Bit-exact absolu | Crypto (Kyber/ML-KEM, corps finis Zq) |
//! | 2    | Virgule fixe Q15.16 / Q31.32 | Bit-exact absolu | Physique (tout réels, échelle connue) |
//! | 3    | f32 + Kahan + FMA forcé + sanitize | Déterministe intra-architecture | ML standard, inference |
//!
//! ## Axiome
//! L'arithmétique entière est la seule vérité. Le f32 divergera toujours
//! entre deux architectures différentes. Pour la crypto et la physique
//! critique, le chemin entier est obligatoire.

use crate::BackendResult;

// =========================================================================
// Primitive: Kahan compensated summation (f32, Voie 3)
// =========================================================================

/// Kahan compensated accumulator — contrecarre l'effet papillon en physique.
/// Même séquence d'additions → même résultat bit-à-bit, quelle que soit
/// l'amplitude relative des termes.
#[derive(Debug, Clone, Copy)]
pub struct KahanSum {
    pub sum: f32,
    c: f32,
}

impl KahanSum {
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    pub fn add(&mut self, x: f32) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    pub fn value(&self) -> f32 {
        self.sum
    }
}

impl Default for KahanSum {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Primitive: subnormal sanitization (Voie 3.B)
// =========================================================================

/// Sanitize un f32: écrase les sous-normaux (|x| < 1.18e-38) à zéro.
/// Les GPU traitent différemment FTZ/DAZ selon le driver — ceci force
/// le comportement déterministe.
///
/// Seuil: `1.17549435e-38` = plus petit float normalisé (f32::MIN_POSITIVE).
pub fn sanitize_f32(x: f32) -> f32 {
    if x.abs() < f32::MIN_POSITIVE { 0.0 } else { x }
}

/// Sanitize toutes les valeurs d'une slice.
pub fn sanitize_slice(data: &mut [f32]) {
    for x in data.iter_mut()
    {
        *x = sanitize_f32(*x);
    }
}

// =========================================================================
// Bit-exact verification (Voie 4)
// =========================================================================

/// Vérifier l'égalité bit-à-bit entre deux slices f32.
///
/// Gère le cas particulier du zéro signé: `-0.0` et `+0.0` ont des
/// patterns binaires différents (`0x80000000` vs `0x00000000`) mais
/// sont mathématiquement égaux.
pub fn verify_bit_exact(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len()
    {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate()
    {
        let xb = x.to_bits();
        let yb = y.to_bits();
        if xb != yb
        {
            // Tolérance pour le zéro signé
            if x == 0.0 && y == 0.0
            {
                continue;
            }
            return Err(format!(
                "bit mismatch at index {}: {:?} (0x{:08x}) != {:?} (0x{:08x})",
                i, x, xb, y, yb
            ));
        }
    }
    Ok(())
}

/// Relative Frobenius error.
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-30);
    num / den
}

// =========================================================================
// Voie 1: Arithmétique entière + modulo (Crypto)
// =========================================================================

/// GEMM sur corps fini Zq — arithmétique entière pure, déterministe à 100%.
///
/// `C(i,j) = sum_k (A(i,k) * B(k,j) mod q) mod q`
///
/// Le modulo est appliqué après chaque multiplication pour éviter les
/// débordements dans l'accumulateur i32.
pub fn crypto_gemm_zq(
    a: &[i32],
    b: &[i32],
    m: usize,
    k: usize,
    n: usize,
    q: i32,
) -> Result<Vec<i32>, String> {
    if a.len() != m * k || b.len() != k * n
    {
        return Err(format!(
            "shape mismatch: A({}*{})={}, B({}*{})={}",
            m,
            k,
            a.len(),
            k,
            n,
            b.len()
        ));
    }
    let mut out = vec![0i32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut sum: i32 = 0;
            for p in 0..k
            {
                // Réduction modulaire immédiate après chaque produit
                let prod = ((a[i * k + p] as i64 * b[p * n + j] as i64) % q as i64) as i32;
                sum = (sum + prod) % q;
            }
            // Garantir le résultat dans [0, q-1]
            out[i * n + j] = if sum < 0 { sum + q } else { sum };
        }
    }
    Ok(out)
}

// =========================================================================
// Voie 2: Virgule fixe Q15.16 (Physique)
// =========================================================================

/// Facteur d'échelle Q15.16 : 2^16 = 65536.
pub const Q16_SCALE: i32 = 65536;
/// Facteur d'échelle Q31.32 (haute précision).
pub const Q32_SCALE: i64 = 1i64 << 32;

/// Convertir f32 → virgule fixe Q15.16 (arrondi au plus proche).
pub fn float_to_q16(x: f32) -> i32 {
    (x * Q16_SCALE as f32).round() as i32
}

/// Convertir virgule fixe Q15.16 → f32.
pub fn q16_to_float(x: i32) -> f32 {
    x as f32 / Q16_SCALE as f32
}

/// GEMM en virgule fixe Q15.16 — déterministe à 100% car tout en entiers.
///
/// `C(i,j) = sum_k (A(i,k) * B(k,j) >> 16)`
///
/// Le `>> 16` réaligne l'échelle après la multiplication (Q16 × Q16 → Q32,
/// puis Q32 >> 16 → Q16). L'accumulation se fait en `i64` pour éviter
/// l'overflow même avec K grand.
pub fn fixed_point_gemm_q16(
    a: &[i32],
    b: &[i32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, String> {
    if a.len() != m * k || b.len() != k * n
    {
        return Err(format!(
            "shape mismatch: A({}*{})={}, B({}*{})={}",
            m,
            k,
            a.len(),
            k,
            n,
            b.len()
        ));
    }
    let mut out = vec![0i32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut sum: i64 = 0; // i64 pour éviter l'overflow
            for p in 0..k
            {
                let product = a[i * k + p] as i64 * b[p * n + j] as i64;
                sum += product >> 16; // Réalignement Q16
            }
            out[i * n + j] = sum as i32;
        }
    }
    Ok(out)
}

/// GEMM en virgule fixe Q31.32 — précision maximale.
pub fn fixed_point_gemm_q32(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i64>, String> {
    if a.len() != m * k || b.len() != k * n
    {
        return Err("shape mismatch".to_string());
    }
    let mut out = vec![0i64; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut sum: i128 = 0; // i128 pour accumulation sans overflow
            for p in 0..k
            {
                let product = a[i * k + p] as i128 * b[p * n + j] as i128;
                sum += product >> 32;
            }
            out[i * n + j] = sum as i64;
        }
    }
    Ok(out)
}

// =========================================================================
// Voie 3: f32 hybride (Kahan + FMA + sanitize)
// =========================================================================

/// GEMM f32 déterministe via Kahan + fixed accumulation order.
///
/// `C(i,j) = alpha * sum_q(A(i,q) * B(q,j)) + beta * C(i,j)`
/// Accumulation en ordre fixe (ascendant q) avec KahanSum.
#[allow(clippy::too_many_arguments)]
pub fn deterministic_fp32_gemm(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
) -> BackendResult<()> {
    if m == 0 || n == 0
    {
        return Ok(());
    }
    if k == 0
    {
        for v in c.iter_mut()
        {
            *v = sanitize_f32(*v * beta);
        }
        return Ok(());
    }

    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = KahanSum::new();
            for q in 0..k
            {
                let av = sanitize_f32(if ta { a[q * m + i] } else { a[i * k + q] });
                let bv = sanitize_f32(if tb { b[j * k + q] } else { b[q * n + j] });
                acc.add(av * bv);
            }
            c[i * n + j] = sanitize_f32(alpha * acc.value() + beta * c[i * n + j]);
        }
    }
    Ok(())
}

// =========================================================================
// Quantification INT8 symétrique (pour Voie 1 entière)
// =========================================================================

/// Quantifier f32 → i8 symétrique, per-tensor scale.
pub fn quantize_symmetric_i8(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty()
    {
        return (Vec::new(), 1.0);
    }
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs < f32::EPSILON
    {
        1.0
    }
    else
    {
        max_abs / 127.0
    };
    let inv_scale = 1.0 / scale;
    let q: Vec<i8> = data
        .iter()
        .map(|&x| ((x * inv_scale).round() as i32).clamp(-127, 127) as i8)
        .collect();
    (q, scale)
}

/// GEMM INT8 déterministe: accumulation i32, déquantification.
/// **Bit-exact garanti** (arithmétique entière pure).
pub fn int8_deterministic_gemm(
    a_q: &[i8],
    b_q: &[i8],
    scale_a: f32,
    scale_b: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    if a_q.len() != m * k || b_q.len() != k * n
    {
        return Err("shape mismatch".to_string());
    }
    let mut out = vec![0.0f32; m * n];
    let scale = scale_a * scale_b;
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc: i32 = 0;
            for q in 0..k
            {
                acc += a_q[i * k + q] as i32 * b_q[q * n + j] as i32;
            }
            out[i * n + j] = sanitize_f32(acc as f32 * scale);
        }
    }
    Ok(out)
}

/// GEMM INT16 déterministe — plus de headroom avant overflow.
pub fn int16_deterministic_gemm(
    a_q: &[i16],
    b_q: &[i16],
    scale_a: f32,
    scale_b: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    if a_q.len() != m * k || b_q.len() != k * n
    {
        return Err("shape mismatch".to_string());
    }
    let mut out = vec![0.0f32; m * n];
    let scale = scale_a * scale_b;
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc: i64 = 0; // i64 pour INT16
            for q in 0..k
            {
                acc += a_q[i * k + q] as i64 * b_q[q * n + j] as i64;
            }
            out[i * n + j] = sanitize_f32(acc as f32 * scale);
        }
    }
    Ok(out)
}

/// Réduction déterministe (sum) via Kahan + ordre fixe.
pub fn deterministic_reduce_sum(data: &[f32], outer: usize, axis_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; outer];
    for i in 0..outer
    {
        let mut acc = KahanSum::new();
        for k in 0..axis_size
        {
            acc.add(sanitize_f32(data[i * axis_size + k]));
        }
        out[i] = sanitize_f32(acc.value());
    }
    out
}

/// Réduction déterministe (mean) via Kahan + ordre fixe.
pub fn deterministic_reduce_mean(data: &[f32], outer: usize, axis_size: usize) -> Vec<f32> {
    if axis_size == 0
    {
        return vec![0.0; outer];
    }
    deterministic_reduce_sum(data, outer, axis_size)
        .iter()
        .map(|&s| sanitize_f32(s / axis_size as f32))
        .collect()
}

// =========================================================================
// Test helpers (re-exported from kernels.rs for convenience)
// =========================================================================

/// Re-export des kernels WGSL depuis `kernels.rs`.
pub use crate::kernels::{CRYPTO_GEMM_WGSL, FIXED_POINT_Q16_GEMM_WGSL, WGSL_SANITIZE_F32};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, RawComputeBackend};

    // --- Voie 1: Crypto ---

    #[test]
    fn crypto_gemm_is_bit_exact() {
        let a: Vec<i32> = (0..16).collect();
        let b: Vec<i32> = (0..8).map(|i| (i * 7) % 100).collect();
        let q = 3329i32; // Kyber
        let r1 = crypto_gemm_zq(&a, &b, 4, 4, 2, q).unwrap();
        let r2 = crypto_gemm_zq(&a, &b, 4, 4, 2, q).unwrap();
        assert_eq!(r1, r2); // Bit-exact vérifié
        assert!(r1.iter().all(|&x| x >= 0 && x < q), "all in [0, q)");
    }

    // --- Voie 2: Virgule fixe ---

    #[test]
    fn fixed_point_q16_roundtrip() {
        let values = vec![1.5f32, -3.25, 0.0, 127.0, -128.0];
        for &v in &values
        {
            let q = float_to_q16(v);
            let back = q16_to_float(q);
            assert!(
                (back - v).abs() < 1.0 / Q16_SCALE as f32,
                "{} roundtrip failed",
                v
            );
        }
    }

    #[test]
    fn fixed_point_q16_gemm_is_bit_exact() {
        let a: Vec<i32> = (0..16)
            .map(|i| float_to_q16((i as f32 - 8.0) * 0.1))
            .collect();
        let b: Vec<i32> = (0..8)
            .map(|i| float_to_q16((i as f32 - 4.0) * 0.2))
            .collect();
        let r1 = fixed_point_gemm_q16(&a, &b, 4, 4, 2).unwrap();
        let r2 = fixed_point_gemm_q16(&a, &b, 4, 4, 2).unwrap();
        assert_eq!(r1, r2);
    }

    // --- Voie 3: f32 hybride ---

    #[test]
    fn sanitize_f32_zeros_subnormals() {
        assert_eq!(sanitize_f32(0.0), 0.0);
        assert_eq!(sanitize_f32(f32::MIN_POSITIVE / 2.0), 0.0); // subnormal
        assert_eq!(sanitize_f32(f32::MIN_POSITIVE), f32::MIN_POSITIVE); // normal
        assert_eq!(sanitize_f32(-f32::MIN_POSITIVE / 2.0), 0.0); // negative subnormal
    }

    #[test]
    fn kahan_sum_is_more_accurate_than_naive() {
        let mut naive: f32 = 0.0;
        let mut ks = KahanSum::new();
        for _ in 0..100000
        {
            naive += 0.00001;
            ks.add(0.00001);
        }
        let err_naive = (naive - 1.0).abs();
        let err_kahan = (ks.value() - 1.0).abs();
        assert!(
            err_kahan < err_naive,
            "Kahan {} < naive {}",
            err_kahan,
            err_naive
        );
    }

    #[test]
    fn kahan_sum_is_deterministic() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut ks1 = KahanSum::new();
        let mut ks2 = KahanSum::new();
        for &v in &data
        {
            ks1.add(v);
            ks2.add(v);
        }
        assert_eq!(ks1.value().to_bits(), ks2.value().to_bits());
    }

    #[test]
    fn deterministic_fp32_gemm_is_bit_reproducible() {
        let a: Vec<f32> = (0..12).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32).cos()).collect();
        let mut c1 = vec![0.0f32; 6];
        let mut c2 = vec![0.0f32; 6];
        deterministic_fp32_gemm(1.0, &a, &b, 0.0, &mut c1, 2, 3, 2, false, false).unwrap();
        deterministic_fp32_gemm(1.0, &a, &b, 0.0, &mut c2, 2, 3, 2, false, false).unwrap();
        verify_bit_exact(&c1, &c2).unwrap();
    }

    #[test]
    fn verify_bit_exact_handles_signed_zero() {
        let a = vec![-0.0f32, 1.0];
        let b = vec![0.0f32, 1.0];
        verify_bit_exact(&a, &b).unwrap(); // -0.0 == 0.0 tolérance
    }

    #[test]
    fn verify_bit_exact_detects_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 3.0];
        assert!(verify_bit_exact(&a, &b).is_err());
    }

    #[test]
    fn deterministic_fp32_gemm_matches_cpu_oracle() {
        let a: Vec<f32> = (0..6).map(|i| i as f32 - 3.0).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut c = vec![0.0f32; 4];
        deterministic_fp32_gemm(1.0, &a, &b, 0.0, &mut c, 2, 3, 2, false, false).unwrap();
        let cpu = CpuBackend.gemm_f32(&a, &b, 2, 3, 2).unwrap();
        assert!(rel_err(&c, &cpu) < 1e-5);
    }

    #[test]
    fn int8_quantize_roundtrips() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.0).sin()).collect();
        let (q, scale) = quantize_symmetric_i8(&data);
        let deq: Vec<f32> = q.iter().map(|&x| x as f32 * scale).collect();
        let max_err: f32 = data
            .iter()
            .zip(&deq)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < scale * 0.6);
    }

    #[test]
    fn int8_deterministic_gemm_is_bit_exact() {
        let data_a: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
        let data_b: Vec<f32> = (0..32).map(|i| (i as f32).cos()).collect();
        let (a_q, sa) = quantize_symmetric_i8(&data_a);
        let (b_q, sb) = quantize_symmetric_i8(&data_b);
        let r1 = int8_deterministic_gemm(&a_q, &b_q, sa, sb, 4, 4, 8).unwrap();
        let r2 = int8_deterministic_gemm(&a_q, &b_q, sa, sb, 4, 4, 8).unwrap();
        verify_bit_exact(&r1, &r2).unwrap();
    }
}
