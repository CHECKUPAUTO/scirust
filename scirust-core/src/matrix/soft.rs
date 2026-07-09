// scirust-core/src/matrix/soft.rs
// Soft-Float Matrix Engine for Cross-Platform Determinism

/// Deterministic, scalar-fallback computational kernel for GEMM using integer emulation.
/// alpha_scale is used to interpret integers as fixed-point (e.g., 1e6).
#[allow(clippy::too_many_arguments)]
pub fn soft_gemm(
    alpha: i32,
    alpha_scale: i32,
    a: &[i32],
    b: &[i32],
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) {
    // `alpha_scale` is the fixed-point denominator; 0 would be an integer
    // divide-by-zero (process abort). Reject it explicitly with a clear message
    // rather than crashing deep inside the inner loop.
    assert!(
        alpha_scale != 0,
        "soft_gemm: alpha_scale must be non-zero (fixed-point denominator)"
    );
    // Isolated from FPU: strictly integer arithmetic
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0i64;
            for p in 0..k
            {
                // Using i64 for intermediate accumulation to prevent overflow
                acc += a[i * k + p] as i64 * b[p * n + j] as i64;
            }
            // Apply alpha and scale back
            let val = (acc * alpha as i64) / alpha_scale as i64;
            c[i * n + j] = val as i32;
        }
    }
}
