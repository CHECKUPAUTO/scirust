//! Algorithme decouvert automatiquement par forge (FunSearch/AlphaEvolve-style).
//! Injecte le 2026-06-05T18:17:10Z.
//!
//! model = hf.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q8_0
//! domain = simd_gemm
//! latency_ns = 83006
//! baseline_ns = 178995
//! speedup = 2.15x
//! bytes = 391
//! verified_holdout = true
//! gate = C=A*B durci (c pre-rempli 1e30)
//!
//! NE PAS editer a la main : regenere par le binaire `inject_elite`.

#[inline(never)]
pub fn compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    // SECURITY GUARD (added by audit — port this to the `inject_elite` generator
    // template so it survives regeneration): this is a *safe* `pub fn` but the
    // loop below reads/writes `a`, `b`, `c` at indices up to `n*n - 1` through
    // raw pointers with no bounds check. Without this guard, a caller passing a
    // slice shorter than `n*n` (or an `n*n` that overflows `usize`) triggers
    // out-of-bounds reads/writes (UB) from 100% safe code.
    let nn = n
        .checked_mul(n)
        .expect("discovered_gemm::compute_kernel: n*n overflows usize");
    assert!(
        a.len() >= nn && b.len() >= nn && c.len() >= nn,
        "discovered_gemm::compute_kernel: a/b/c must each hold at least n*n = {nn} elements"
    );
    // Pre-zeroage de C : produit C = A*B (ecrasement, pas accumulation)
    for elem in c.iter_mut()
    {
        *elem = 0.0;
    }
    // i-k-j : boucle interne sur j contigue en memoire -> vectorisable
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();
        for i in 0..n
        {
            for k in 0..n
            {
                let a_ik = *a_ptr.add(i * n + k);
                for j in 0..n
                {
                    let b_kj = *b_ptr.add(k * n + j);
                    let c_ij = *c_ptr.add(i * n + j) + a_ik * b_kj;
                    *c_ptr.add(i * n + j) = c_ij;
                }
            }
        }
    }
}

#[cfg(test)]
mod forge_tests {
    use super::*;

    fn naive_ref(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; n * n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for k in 0..n
                {
                    acc += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_reference_and_overwrites() {
        let n = 17; // non multiple de 4 : aucune hypothese de taille
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for x in a.iter_mut()
        {
            *x = rng();
        }
        for x in b.iter_mut()
        {
            *x = rng();
        }
        let expected = naive_ref(&a, &b, n);
        let mut c = vec![999.0f64; n * n]; // bruit : le kernel DOIT ecraser
        compute_kernel(&mut c, &a, &b, n);
        for i in 0..n * n
        {
            assert!(
                (c[i] - expected[i]).abs() < 1e-9,
                "mismatch @{i}: {} vs {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn gemm_identity_and_exact_2x2() {
        // A · I = A.
        let n = 3;
        let a: Vec<f64> = (0..9).map(|i| i as f64 + 1.0).collect();
        let mut ident = vec![0.0; 9];
        for i in 0..n
        {
            ident[i * n + i] = 1.0;
        }
        let mut c = vec![0.0; 9];
        compute_kernel(&mut c, &a, &ident, n);
        assert_eq!(c, a);

        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]].
        let mut c2 = vec![0.0; 4];
        compute_kernel(&mut c2, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2);
        assert_eq!(c2, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
