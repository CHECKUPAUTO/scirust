// scirust-core/src/matrix/soft.rs
// Soft-Float Matrix Engine for Cross-Platform Determinism

/// Deterministic, scalar-fallback computational kernel for GEMM using integer emulation.
/// alpha_scale is used to interpret integers as fixed-point (e.g., 1e6).
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
    // Isolated from FPU: strictly integer arithmetic
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for p in 0..k {
                // Using i64 for intermediate accumulation to prevent overflow
                acc += a[i * k + p] as i64 * b[p * n + j] as i64;
            }
            // Apply alpha and scale back
            let val = (acc * alpha as i64) / alpha_scale as i64;
            c[i * n + j] = val as i32;
        }
    }
}
