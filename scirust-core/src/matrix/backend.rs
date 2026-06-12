// scirust-core/src/matrix/backend.rs
//
// Trait SimdBackend + implémentations :
//   - ScalarBackend   : référence pure Rust, toujours compilable
//   - PortableSimdBackend : std::simd, nightly
//
// L'utilisateur choisit le backend à la compilation ou à l'exécution.
// À terme : BlasBackend pourra déléguer à matrixmultiply/netlib.

use crate::matrix::view::{MatrixShape, MatrixView, MatrixViewMut};

// ------------------------------------------------------------------ //
//  Trait central                                                       //
// ------------------------------------------------------------------ //

/// Opérations de base sur matrices/vecteurs, abstraites du backend.
/// Toutes les opérations supposent des layouts compatibles (row-major).
pub trait SimdBackend: Send + Sync {
    /// AXPY : y = alpha * x + y
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]);
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]);

    /// Produit scalaire
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32;
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64;

    /// GEMV : y = alpha * A * x + beta * y
    /// A : (m × k), x : (k,), y : (m,)
    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]);

    /// GEMM : C = alpha * A * B + beta * C
    /// A : (m × k), B : (k × n), C : (m × n)
    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        c: MatrixViewMut<f32>,
    );

    /// ReLU in-place
    fn relu_f32(&self, v: &mut [f32]);

    /// Décomposition de Cholesky : A = L * L^T (A doit être symétrique définie positive)
    /// Remplace A par sa partie triangulaire inférieure L.
    #[allow(clippy::needless_range_loop)]
    fn cholesky_f64(&self, a: &mut [Vec<f64>]) -> Option<()>;

    fn name(&self) -> &'static str;
}

// ------------------------------------------------------------------ //
//  BlasBackend — backend utilisant BLAS (OpenBLAS/MKL)                //
// ------------------------------------------------------------------ //

#[cfg(feature = "blas")]
pub struct BlasBackend;

#[cfg(feature = "blas")]
impl SimdBackend for BlasBackend {
    fn name(&self) -> &'static str {
        "blas"
    }

    #[inline]
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe {
            blas::saxpy(x.len() as i32, alpha, x, 1, y, 1);
        }
    }

    #[inline]
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe {
            blas::daxpy(x.len() as i32, alpha, x, 1, y, 1);
        }
    }

    #[inline]
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        unsafe { blas::sdot(x.len() as i32, x, 1, y, 1) }
    }

    #[inline]
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        unsafe { blas::ddot(x.len() as i32, x, 1, y, 1) }
    }

    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        let (m, k) = a.shape();
        assert_eq!(x.len(), k);
        assert_eq!(y.len(), m);

        // BLAS needs contiguous storage.
        if a.col_stride() == 1
        {
            unsafe {
                blas::sgemv(
                    b'T',
                    k as i32,
                    m as i32,
                    alpha,
                    std::slice::from_raw_parts(a.as_ptr(), m * a.row_stride()),
                    a.row_stride() as i32,
                    x,
                    1,
                    beta,
                    y,
                    1,
                );
            }
        }
        else
        {
            // Fallback for non-contiguous views
            ScalarBackend.sgemv_f32(alpha, a, x, beta, y);
        }
    }

    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        mut c: MatrixViewMut<f32>,
    ) {
        let (m, k) = a.shape();
        let (_k, n) = b.shape();
        assert_eq!(k, _k);
        assert_eq!(c.shape(), (m, n));

        if a.col_stride() == 1 && b.col_stride() == 1 && c.col_stride() == 1
        {
            unsafe {
                blas::sgemm(
                    b'N',
                    b'N',
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    std::slice::from_raw_parts(b.as_ptr(), b.rows() * b.row_stride()),
                    b.row_stride() as i32,
                    std::slice::from_raw_parts(a.as_ptr(), a.rows() * a.row_stride()),
                    a.row_stride() as i32,
                    beta,
                    std::slice::from_raw_parts_mut(c.as_mut_ptr(), c.rows() * c.row_stride()),
                    c.row_stride() as i32,
                );
            }
        }
        else
        {
            ScalarBackend.sgemm_f32(alpha, a, b, beta, c);
        }
    }

    fn relu_f32(&self, v: &mut [f32]) {
        for x in v.iter_mut()
        {
            if *x < 0.0
            {
                *x = 0.0;
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn cholesky_f64(&self, a: &mut [Vec<f64>]) -> Option<()> {
        ScalarBackend.cholesky_f64(a)
    }
}

// ------------------------------------------------------------------ //
//  ScalarBackend — référence portable, stable toolchain               //
// ------------------------------------------------------------------ //

pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "scalar"
    }

    #[inline]
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        for (yi, xi) in y.iter_mut().zip(x.iter())
        {
            *yi += alpha * xi;
        }
    }

    #[inline]
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (yi, xi) in y.iter_mut().zip(x.iter())
        {
            *yi += alpha * xi;
        }
    }

    #[inline]
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    #[inline]
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        let (m, k) = a.shape();
        assert_eq!(x.len(), k);
        assert_eq!(y.len(), m);
        for i in 0..m
        {
            let mut acc = 0.0f32;
            for j in 0..k
            {
                acc += a[(i, j)] * x[j];
            }
            y[i] = alpha * acc + beta * y[i];
        }
    }

    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        mut c: MatrixViewMut<f32>,
    ) {
        let (m, k) = a.shape();
        let (_k, n) = b.shape();
        assert_eq!(k, _k);
        assert_eq!(c.shape(), (m, n));

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            // Safety check for parallelization:
            // We need to ensure c is contiguous for the simple pointer arithmetic used.
            if c.col_stride() == 1
            {
                // Wrap MatrixView to ensure they are moved and accessible
                #[derive(Copy, Clone)]
                struct SendView<'a>(MatrixView<'a, f32>);
                unsafe impl<'a> Send for SendView<'a> {}
                unsafe impl<'a> Sync for SendView<'a> {}

                let ptr_c_raw = c.as_mut_ptr() as usize;
                let view_a = SendView(a);
                let view_b = SendView(b);
                let row_stride_c = c.row_stride();

                (0..m).into_par_iter().for_each(move |i| {
                    let a = view_a.0;
                    let b = view_b.0;
                    unsafe {
                        let row_c_ptr = (ptr_c_raw as *mut f32).add(i * row_stride_c);
                        for j in 0..n
                        {
                            let mut acc = 0.0f32;
                            for p in 0..k
                            {
                                acc += a[(i, p)] * b[(p, j)];
                            }
                            let val_c = row_c_ptr.add(j);
                            *val_c = alpha * acc + beta * *val_c;
                        }
                    }
                });
                return;
            }
        }

        // Sequential fallback
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a[(i, p)] * b[(p, j)];
                }
                c[(i, j)] = alpha * acc + beta * c[(i, j)];
            }
        }
    }

    fn relu_f32(&self, v: &mut [f32]) {
        for x in v.iter_mut()
        {
            *x = x.max(0.0);
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn cholesky_f64(&self, a: &mut [Vec<f64>]) -> Option<()> {
        let n = a.len();
        for i in 0..n
        {
            for j in 0..=i
            {
                let mut sum = a[i][j];
                for k in 0..j
                {
                    sum -= a[i][k] * a[j][k];
                }

                if i == j
                {
                    if sum <= 0.0
                    {
                        return None; // Non définie positive
                    }
                    a[i][j] = sum.sqrt();
                }
                else
                {
                    a[i][j] = sum / a[j][j];
                }
            }
        }
        // Nettoyage de la partie supérieure
        for i in 0..n
        {
            for j in (i + 1)..n
            {
                a[i][j] = 0.0;
            }
        }
        Some(())
    }
}

// ------------------------------------------------------------------ //
//  PortableSimdBackend — nightly std::simd                            //
// ------------------------------------------------------------------ //

#[cfg(feature = "portable-simd")]
pub struct PortableSimdBackend;

#[cfg(feature = "portable-simd")]
impl SimdBackend for PortableSimdBackend {
    fn name(&self) -> &'static str {
        "portable-simd"
    }

    #[inline]
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        // y += alpha * x, vectorisé
        use std::simd::f32x8;
        let splat = f32x8::splat(alpha);
        let (pre_x, mid_x, suf_x) = x.as_simd::<8>();
        let (pre_y, mid_y, suf_y) = y.as_simd_mut::<8>();

        for (yi, xi) in pre_y.iter_mut().zip(pre_x.iter())
        {
            *yi += alpha * xi;
        }
        for (vy, vx) in mid_y.iter_mut().zip(mid_x.iter())
        {
            *vy = splat.mul_add(*vx, *vy);
        }
        let offset = pre_x.len() + mid_x.len() * 8;
        for (yi, xi) in suf_y.iter_mut().zip(suf_x.iter())
        {
            *yi += alpha * xi;
        }
        let _ = offset; // silence unused
    }

    #[inline]
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        use std::simd::f64x4;
        let splat = f64x4::splat(alpha);
        let (pre_x, mid_x, _) = x.as_simd::<4>();
        let (pre_y, mid_y, suf_y) = y.as_simd_mut::<4>();

        for (yi, xi) in pre_y.iter_mut().zip(pre_x.iter())
        {
            *yi += alpha * xi;
        }
        for (vy, vx) in mid_y.iter_mut().zip(mid_x.iter())
        {
            *vy = splat.mul_add(*vx, *vy);
        }
        let offset = pre_x.len() + mid_x.len() * 4;
        for (yi, xi) in suf_y.iter_mut().zip(x[offset..].iter())
        {
            *yi += alpha * xi;
        }
    }

    #[inline]
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::simd_ops::dot_f32(x, y)
    }

    #[inline]
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        crate::simd_ops::dot_f64(x, y)
    }

    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        // Chaque ligne de A est un produit scalaire avec x
        let (m, _) = a.shape();
        for i in 0..m
        {
            let row = a.row_slice(i).expect("row_slice nécessite col_stride=1");
            let dot = self.sdot_f32(row, x);
            y[i] = alpha * dot + beta * y[i];
        }
    }

    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        mut c: MatrixViewMut<f32>,
    ) {
        // GEMM par blocs (tiling L1) + SIMD sur les produits scalaires internes
        // Taille de bloc : 64 éléments ≈ ligne de cache typique
        const BLOCK: usize = 64;
        let (m, k) = a.shape();
        let (_k, n) = b.shape();
        assert_eq!(k, _k);

        // Pré-scale C par beta
        for i in 0..m
        {
            if let Some(row) = c.row_slice_mut(i)
            {
                for x in row.iter_mut()
                {
                    *x *= beta;
                }
            }
        }

        // Boucle tuilée : i-p-j avec SIMD sur la dimension intérieure j
        let mut i = 0;
        while i < m
        {
            let ib = (i + BLOCK).min(m);
            let mut p = 0;
            while p < k
            {
                let pb = (p + BLOCK).min(k);
                let mut j = 0;
                while j < n
                {
                    let jb = (j + BLOCK).min(n);
                    // Bloc (ib-i) × (jb-j) de C accumulé depuis A[:,p:pb] * B[p:pb,:]
                    for ii in i..ib
                    {
                        for jj in j..jb
                        {
                            let a_row = &a.row_slice(ii).unwrap()[p..pb];
                            // Colonne jj de B : accès non contigu — tampon local
                            let b_col: Vec<f32> = (p..pb).map(|pp| b[(pp, jj)]).collect();
                            c[(ii, jj)] += alpha * self.sdot_f32(a_row, &b_col);
                        }
                    }
                    j += BLOCK;
                }
                p += BLOCK;
            }
            i += BLOCK;
        }
    }

    fn relu_f32(&self, v: &mut [f32]) {
        use std::simd::{SimdFloat, f32x8};
        let zero = f32x8::splat(0.0);
        let (pre, mid, suf) = v.as_simd_mut::<8>();
        for x in pre.iter_mut()
        {
            *x = x.max(0.0);
        }
        for vx in mid.iter_mut()
        {
            *vx = vx.simd_max(zero);
        }
        for x in suf.iter_mut()
        {
            *x = x.max(0.0);
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn cholesky_f64(&self, a: &mut [Vec<f64>]) -> Option<()> {
        ScalarBackend.cholesky_f64(a)
    }
}

// ------------------------------------------------------------------ //
//  Sélection automatique du meilleur backend dispo                    //
// ------------------------------------------------------------------ //

/// Renvoie le backend le plus performant disponible à la compilation.
pub fn best_backend() -> &'static dyn SimdBackend {
    #[cfg(feature = "blas")]
    {
        &BlasBackend
    }

    #[cfg(all(not(feature = "blas"), feature = "portable-simd"))]
    {
        &PortableSimdBackend
    }

    #[cfg(all(not(feature = "blas"), not(feature = "portable-simd")))]
    {
        &ScalarBackend
    }
}

// ------------------------------------------------------------------ //
//  Tests d'intégration backend                                         //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::view::{MatrixView, MatrixViewMut};

    fn backend() -> &'static dyn SimdBackend {
        &ScalarBackend
    }

    #[test]
    fn test_saxpy() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![1.0f32; 4];
        backend().saxpy_f32(2.0, &x, &mut y);
        assert_eq!(y, vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sgemv() {
        // A = [[1,2],[3,4]], x = [1,1], y = [0,0]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = MatrixView::from_slice(&a_data, 2, 2);
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        backend().sgemv_f32(1.0, a, &x, 0.0, &mut y);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_sgemm_identity() {
        // I * A = A
        let id = vec![1.0f32, 0.0, 0.0, 1.0]; // 2x2 identité
        let a = vec![3.0f32, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 4];
        let ia = MatrixView::from_slice(&id, 2, 2);
        let av = MatrixView::from_slice(&a, 2, 2);
        let cv = MatrixViewMut::from_slice(&mut c, 2, 2);
        backend().sgemm_f32(1.0, ia, av, 0.0, cv);
        assert!((c[0] - 3.0).abs() < 1e-6);
        assert!((c[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_cholesky() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        let mut a = vec![
            vec![4.0, 12.0, -16.0],
            vec![12.0, 37.0, -43.0],
            vec![-16.0, -43.0, 98.0],
        ];
        backend().cholesky_f64(&mut a).unwrap();

        assert_eq!(a[0], vec![2.0, 0.0, 0.0]);
        assert_eq!(a[1], vec![6.0, 1.0, 0.0]);
        assert_eq!(a[2], vec![-8.0, 5.0, 3.0]);
    }
}
