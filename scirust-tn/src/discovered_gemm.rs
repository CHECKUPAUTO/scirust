//! Algorithme decouvert automatiquement par forge (FunSearch/AlphaEvolve-style).
//! Injecte le 2026-06-05T09:57:09Z.
//!
//! model = hf.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q8_0
//! latency_ns = 90517
//! baseline_ns = 178662
//! speedup = 1.97x
//! bytes = 480
//! verified_holdout = true
//!
//! NE PAS editer a la main : regenere par le binaire `inject_elite`.

#[inline(never)]
pub fn compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    // Pré-zéroer la matrice C pour éviter les accumulations inutiles
    for x in c.iter_mut()
    {
        *x = 0.0;
    }

    // Utilisation de l'algorithme de multiplication matricielle optimisé (i-k-j)
    for i in 0..n
    {
        for k in 0..n
        {
            let aik = a[i * n + k];
            for j in 0..n
            {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
}

#[cfg(test)]
mod forge_tests {
    use super::*;
    #[test]
    fn gemm_matches_reference() {
        let n = 5usize;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.7).cos()).collect();
        let mut got = vec![0.0f64; n * n];
        compute_kernel(&mut got, &a, &b, n);
        let mut want = vec![0.0f64; n * n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut s = 0.0f64;
                for k in 0..n
                {
                    s += a[i * n + k] * b[k * n + j];
                }
                want[i * n + j] = s;
            }
        }
        for i in 0..n * n
        {
            assert!((got[i] - want[i]).abs() < 1e-9, "mismatch at {i}");
        }
    }
}
