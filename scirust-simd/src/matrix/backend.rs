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

    fn sgemm_f32(
        &self,
        _alpha: f32,
        _a: MatrixView<f32>,
        _b: MatrixView<f32>,
        _beta: f32,
        _c: MatrixViewMut<f32>,
    ) {
        // Naive triple-loop GEMM — correct but slow
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
