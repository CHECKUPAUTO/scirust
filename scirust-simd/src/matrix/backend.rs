// Backend types and traits for scirust-simd matrix operations.

use crate::matrix::view::{MatrixView, MatrixViewMut};

// ------------------------------------------------------------------ //
//  SimdBackend trait — interface commune à tous les backends          //
// ------------------------------------------------------------------ //

pub trait SimdBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]);
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]);
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32;
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64;
    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]);
    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        c: MatrixViewMut<f32>,
    );
    fn relu_f32(&self, v: &mut [f32]);
}

// ------------------------------------------------------------------ //
//  ScalarBackend — implémentation scalaire de référence               //
// ------------------------------------------------------------------ //

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        for i in 0..x.len()
        {
            y[i] += alpha * x[i];
        }
    }

    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for i in 0..x.len()
        {
            y[i] += alpha * x[i];
        }
    }

    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }

    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }

    #[allow(clippy::needless_range_loop)]
    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        let m = a.rows();
        for i in 0..m
        {
            let row = a.row_slice(i).expect("row_slice");
            let dot: f32 = row.iter().zip(x).map(|(a, b)| a * b).sum();
            y[i] = alpha * dot + beta * y[i];
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        mut c: MatrixViewMut<f32>,
    ) {
        // Naive triple-loop GEMM: C = alpha * A(m×k) * B(k×n) + beta * C(m×n).
        // (Was an empty no-op, so every backend that delegates here silently left
        // C unchanged — matrix products were wrong.)
        let m = a.rows();
        let k = a.cols();
        let n = b.cols();
        for i in 0..m
        {
            let a_row = a.row_slice(i).expect("A row");
            let c_row = c.row_slice_mut(i).expect("C row");
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a_row[p] * b.row_slice(p).expect("B row")[j];
                }
                c_row[j] = alpha * acc + beta * c_row[j];
            }
        }
    }

    fn relu_f32(&self, v: &mut [f32]) {
        for x in v
        {
            *x = x.max(0.0);
        }
    }
}

// ------------------------------------------------------------------ //
//  Backends spécialisés (marqueurs + target_feature)                  //
// ------------------------------------------------------------------ //

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortableSimdBackend;

impl SimdBackend for PortableSimdBackend {
    fn name(&self) -> &'static str {
        "portable_simd"
    }
    fn saxpy_f32(&self, a: f32, x: &[f32], y: &mut [f32]) {
        ScalarBackend.saxpy_f32(a, x, y);
    }
    fn daxpy_f64(&self, a: f64, x: &[f64], y: &mut [f64]) {
        ScalarBackend.daxpy_f64(a, x, y);
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        ScalarBackend.sdot_f32(x, y)
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        ScalarBackend.ddot_f64(x, y)
    }
    fn sgemv_f32(&self, a: f32, m: MatrixView<f32>, x: &[f32], b: f32, y: &mut [f32]) {
        ScalarBackend.sgemv_f32(a, m, x, b, y);
    }
    fn sgemm_f32(
        &self,
        a: f32,
        ma: MatrixView<f32>,
        mb: MatrixView<f32>,
        b: f32,
        mc: MatrixViewMut<f32>,
    ) {
        ScalarBackend.sgemm_f32(a, ma, mb, b, mc);
    }
    fn relu_f32(&self, v: &mut [f32]) {
        ScalarBackend.relu_f32(v);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Avx2Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sse2Backend;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::view::{MatrixView, MatrixViewMut};

    #[test]
    fn scalar_sgemm_computes_alpha_ab_plus_beta_c() {
        // A (2x3), B (3x2), C (2x2). A*B = [[58,64],[139,154]].
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [1.0f32, 1.0, 1.0, 1.0];
        ScalarBackend.sgemm_f32(
            2.0,
            MatrixView::new(&a, 2, 3),
            MatrixView::new(&b, 3, 2),
            3.0,
            MatrixViewMut::new(&mut c, 2, 2),
        );
        // 2*A*B + 3*C = [[119,131],[281,311]] (pre-fix this stayed [1,1,1,1]).
        assert_eq!(c, [119.0, 131.0, 281.0, 311.0]);
    }
}
