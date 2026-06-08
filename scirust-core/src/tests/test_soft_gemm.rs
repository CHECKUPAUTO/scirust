// scirust-core/src/tests/test_soft_gemm.rs
#[cfg(test)]
mod tests {
    use crate::matrix::soft::soft_gemm;

    #[test]
    fn test_soft_gemm_determinism() {
        let (m, n, k) = (2, 2, 2);
        let a = vec![1000, 2000, 3000, 4000]; // 1.0, 2.0, 3.0, 4.0 with scale 1000
        let b = vec![1000, 0, 0, 1000];       // Identity with scale 1000
        let mut c = vec![0; 4];
        let alpha = 1; // alpha = 1.0
        let scale = 1000;

        soft_gemm(alpha, scale, &a, &b, &mut c, m, n, k);

        // Expected: (a * b) / 1000
        // c[0] = (1000*1000 + 2000*0) * 1 / 1000 = 1000
        assert_eq!(c, vec![1000, 2000, 3000, 4000]);

        // Verify bit-identical results by running again
        let mut c2 = vec![0; 4];
        soft_gemm(alpha, scale, &a, &b, &mut c2, m, n, k);
        assert_eq!(c, c2);
    }
}
