//! Matmul adossé à cuBLAS pour scirust-gpu (feature = "cuda").
//!
//! Row-major C = A·B via cuBLAS (qui est column-major) : on calcule
//! (Bᵀ·Aᵀ) = (A·B)ᵀ en passant B et A inversés. Le handle (contexte CUDA +
//! cuBLAS) est mis en cache par thread pour éviter une ré-initialisation
//! coûteuse à chaque appel.
#![cfg(feature = "cuda")]

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaStream};

thread_local! {
    static HANDLE: RefCell<Option<(Arc<CudaContext>, Arc<CudaStream>, CudaBlas)>> =
        RefCell::new(None);
}

/// Produit matriciel row-major `C = A·B` exécuté sur GPU via cuBLAS (FP32).
///
/// `a` est `m×k`, `b` est `k×n`, résultat `m×n` row-major — même convention
/// que le matmul CPU du framework. Panique si les tailles ne concordent pas.
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A doit faire m*k");
    assert_eq!(b.len(), k * n, "B doit faire k*n");

    HANDLE.with(|h| {
        let mut slot = h.borrow_mut();
        if slot.is_none() {
            let ctx = CudaContext::new(0).expect("init contexte CUDA");
            let stream = ctx.default_stream();
            let blas = CudaBlas::new(stream.clone()).expect("init cuBLAS");
            *slot = Some((ctx, stream, blas));
        }
        let (_ctx, stream, blas) = slot.as_ref().unwrap();

        let a_dev = stream.memcpy_stod(a).expect("htod A");
        let b_dev = stream.memcpy_stod(b).expect("htod B");
        let mut c_dev = stream.alloc_zeros::<f32>(m * n).expect("alloc C");

        // C row-major = A·B  =>  cuBLAS col-major calcule (Bᵀ·Aᵀ) = (A·B)ᵀ
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0,
            ldc: n as i32,
        };
        unsafe {
            blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)
                .expect("cublas sgemm");
        }
        stream.synchronize().expect("sync");
        stream.memcpy_dtov(&c_dev).expect("dtoh C")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let av = a[i * k + kk];
                for j in 0..n {
                    c[i * n + j] += av * b[kk * n + j];
                }
            }
        }
        c
    }

    /// Erreur absolue (partout) et relative (uniquement la ou |ref| est
    /// significatif). En non-carre, des annulations produisent des entrees
    /// quasi nulles : l'erreur relative y explose alors que l'absolue reste
    /// au niveau du bruit FP32 — d'ou la separation des deux metriques.
    fn errs(g: &[f32], c: &[f32]) -> (f32, f32) {
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (x, y) in g.iter().zip(c) {
            let e = (x - y).abs();
            if e > max_abs {
                max_abs = e;
            }
            if y.abs() > 1e-2 {
                let r = e / y.abs();
                if r > max_rel {
                    max_rel = r;
                }
            }
        }
        (max_abs, max_rel)
    }

    #[test]
    fn cublas_matches_cpu_square_512() {
        let (m, k, n) = (512, 512, 512);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let (ma, mr) = errs(&matmul_f32(&a, &b, m, k, n), &matmul_cpu(&a, &b, m, k, n));
        assert!(ma < 1e-2 && mr < 1e-3, "carre 512: abs={:.2e} rel={:.2e}", ma, mr);
    }

    #[test]
    fn cublas_matches_cpu_non_square() {
        // m != k != n : valide le swap m/n et lda/ldb/ldc col-major
        let (m, k, n) = (64, 128, 32);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 9) as f32 - 4.0) * 0.05).collect();
        let (ma, mr) = errs(&matmul_f32(&a, &b, m, k, n), &matmul_cpu(&a, &b, m, k, n));
        assert!(ma < 1e-2 && mr < 1e-3, "non-carre 64x128x32: abs={:.2e} rel={:.2e}", ma, mr);
    }
}
