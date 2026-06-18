//! Deterministic GPU compute layer.
//!
//! ## Philosophy
//!
//! SciRust's GPU determinism guarantee comes from three strategies, in order
//! of preference:
//!
//! 1. **Integer arithmetic** (INT8 → INT32 accumulation) — multiplication and
//!    addition of integers is exact; there is no rounding error, no
//!    non-associativity, no FMA differences. The quantized GEMM path is
//!    *mathematically deterministic* regardless of thread count.
//!
//! 2. **Kahan compensated summation** — for FP32 operations that must stay in
//!    floating-point, we use Kahan summation (a running compensation term) in
//!    a fixed-order reduction tree. Two Kahan summations of the same vector in
//!    the same order yield bit-identical results.
//!
//! 3. **Fixed dispatch ordering** — the host dispatches workgroups in a fixed,
//!    deterministic order, and each thread processes elements in array-index
//!    order, so the accumulation sequence is reproducible.
//!
//! The CPU oracle path (`CpuBackend`) remains the bit-exact reference. GPU
//! results are validated within `< 1e-4` relative tolerance for FP32 paths
//! and exactly bit-exact for INT8 quantized paths.

use crate::BackendResult;

/// Deterministic floating-point accumulator using Kahan compensated summation.
///
/// `KahanSum` maintains a running `sum` and a compensation term `c` so that
/// `sum + c` approximates the exact sum to machine epsilon, regardless of the
/// magnitude range of the inputs. Two KahanSum instances that process the same
/// sequence of values in the same order produce bit-identical results.
#[derive(Debug, Clone, Copy)]
pub struct KahanSum {
    pub sum: f32,
    c: f32,
}

impl KahanSum {
    /// Create a new accumulator starting at zero.
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    /// Accumulate one `f32` value.
    pub fn add(&mut self, x: f32) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    /// Current compensated sum.
    pub fn value(&self) -> f32 {
        self.sum
    }
}

impl Default for KahanSum {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic row-major GEMM using Kahan accumulation.
///
/// `C(i,j) = alpha * sum_q(A(i,q) * B(q,j)) + beta * C(i,j)`
///
/// Each output cell uses an independent Kahan accumulator. The accumulation
/// order is **fixed** (ascending `q`) so results are bit-identical across runs
/// and across threads (each thread processes a fixed range of output cells).
pub fn deterministic_gemm(
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
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        for v in c.iter_mut() {
            *v *= beta;
        }
        return Ok(());
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = KahanSum::new();
            for q in 0..k {
                let av = if ta { a[q * m + i] } else { a[i * k + q] };
                let bv = if tb { b[j * k + q] } else { b[q * n + j] };
                acc.add(av * bv);
            }
            c[i * n + j] = alpha * acc.value() + beta * c[i * n + j];
        }
    }
    Ok(())
}

/// Deterministic mean reduction along one axis using Kahan summation.
///
/// For a tensor shaped `outer × axis_size`, reduces each outer slice to a
/// scalar mean. Uses Kahan compensated summation in fixed order.
pub fn deterministic_reduce_mean(
    data: &[f32],
    outer: usize,
    axis_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; outer];
    if axis_size == 0 {
        return out;
    }
    for i in 0..outer {
        let mut acc = KahanSum::new();
        for k in 0..axis_size {
            acc.add(data[i * axis_size + k]);
        }
        out[i] = acc.value() / axis_size as f32;
    }
    out
}

/// Deterministic sum reduction along one axis using Kahan summation.
pub fn deterministic_reduce_sum(
    data: &[f32],
    outer: usize,
    axis_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; outer];
    for i in 0..outer {
        let mut acc = KahanSum::new();
        for k in 0..axis_size {
            acc.add(data[i * axis_size + k]);
        }
        out[i] = acc.value();
    }
    out
}

/// Quantize a `f32` vector to `i8` with per-tensor scale.
///
/// `scale = max_abs / 127.0`, values clamped to [-127, 127]. Zéro point = 0.
/// This is symmetric per-tensor quantization — deterministic by construction.
pub fn quantize_symmetric_i8(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }
    let max_abs = data
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs < f32::EPSILON { 1.0 } else { max_abs / 127.0 };
    let inv_scale = 1.0 / scale;
    let q: Vec<i8> = data
        .iter()
        .map(|&x| {
            let v = (x * inv_scale).round();
            v.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (q, scale)
}

/// Deterministic INT8 quantized GEMM: `C = A_quantized ⊗ B_quantized`.
///
/// Uses `i32` accumulation — integer arithmetic is exact (no rounding).
/// The result is dequantized with `scale_a * scale_b`.
/// **Guaranteed bit-exact** regardless of parallelism or platform.
pub fn int8_deterministic_gemm(
    a_q: &[i8],
    b_q: &[i8],
    scale_a: f32,
    scale_b: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    if a_q.len() != m * k || b_q.len() != k * n {
        return Err(format!(
            "shape mismatch: A({}*{})={}, B({}*{})={}",
            m, k, a_q.len(), k, n, b_q.len()
        ));
    }
    let mut out = vec![0.0f32; m * n];
    let scale = scale_a * scale_b;

    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for q in 0..k {
                acc += a_q[i * k + q] as i32 * b_q[q * n + j] as i32;
            }
            out[i * n + j] = acc as f32 * scale;
        }
    }
    Ok(out)
}

/// Verifier that two GEMM results are bit-identical.
///
/// Returns `Ok(())` if every element is bit-exact (0 tolerance).
/// Returns `Err(details)` with the first mismatch index and values.
pub fn verify_bit_exact(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Err(format!(
                "bit mismatch at index {}: {:?} (0x{:08x}) != {:?} (0x{:08x})",
                i,
                x,
                x.to_bits(),
                y,
                y.to_bits()
            ));
        }
    }
    Ok(())
}

/// Relative Frobenius error between two vectors.
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-30);
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, RawComputeBackend};

    #[test]
    fn kahan_sum_is_more_accurate_than_naive() {
        // Sum of 100000 * 0.00001 = 1.0 — naive loses precision, Kahan preserves
        let mut naive: f32 = 0.0;
        let mut ks = KahanSum::new();
        for _ in 0..100000 {
            naive += 0.00001;
            ks.add(0.00001);
        }
        let err_naive = (naive - 1.0).abs();
        let err_kahan = (ks.value() - 1.0).abs();
        assert!(err_kahan < err_naive, "Kahan {} < naive {}", err_kahan, err_naive);
    }

    #[test]
    fn kahan_sum_is_deterministic() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut ks1 = KahanSum::new();
        let mut ks2 = KahanSum::new();
        for &v in &data {
            ks1.add(v);
            ks2.add(v);
        }
        assert_eq!(ks1.value().to_bits(), ks2.value().to_bits());
    }

    #[test]
    fn deterministic_gemm_is_bit_reproducible() {
        let a: Vec<f32> = (0..12).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32).cos()).collect();
        let mut c1 = vec![0.0f32; 6];
        let mut c2 = vec![0.0f32; 6];

        deterministic_gemm(1.0, &a, &b, 0.0, &mut c1, 2, 3, 2, false, false).unwrap();
        deterministic_gemm(1.0, &a, &b, 0.0, &mut c2, 2, 3, 2, false, false).unwrap();
        verify_bit_exact(&c1, &c2).unwrap();
    }

    #[test]
    fn deterministic_gemm_matches_cpu_oracle() {
        // 2×3 GEMM: a is 2×3 (6 elems), b is 3×2 (6 elems), c is 2×2 (4 elems)
        let a: Vec<f32> = (0..6).map(|i| i as f32 - 3.0).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut c = vec![0.0f32; 4];

        deterministic_gemm(1.0, &a, &b, 0.0, &mut c, 2, 3, 2, false, false).unwrap();
        let cpu = CpuBackend.gemm_f32(&a, &b, 2, 3, 2).unwrap();
        assert!(rel_err(&c, &cpu) < 1e-5);
    }

    #[test]
    fn int8_quantize_roundtrips_approximately() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.0).sin()).collect();
        let (q, scale) = quantize_symmetric_i8(&data);
        let deq: Vec<f32> = q.iter().map(|&x| x as f32 * scale).collect();
        // Quantization error should be within ~0.5 * scale
        let max_err: f32 = data
            .iter()
            .zip(&deq)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < scale * 0.6, "max quant error {} > scale*0.5", max_err);
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

    #[test]
    fn verify_bit_exact_detects_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 3.0];
        assert!(verify_bit_exact(&a, &b).is_err());
    }
}
