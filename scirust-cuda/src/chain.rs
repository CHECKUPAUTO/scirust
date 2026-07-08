//! The resident CUDA op-chain (feature `cuda`). B1: device/stream/cuBLASLt
//! plumbing + the first bf16 Tensor-core GEMM, gradient-checked against the CPU.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;

/// A resident row-major `rows × cols` matrix in VRAM, stored in **bf16** (the
/// Tensor-core input type). The fp32 → bf16 rounding happens on upload; fp32
/// accumulation happens inside the GEMM.
pub struct CudaMatrix {
    buf: CudaSlice<bf16>,
    rows: usize,
    cols: usize,
}

impl CudaMatrix {
    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// The CUDA backend handle: a device context, its default stream, and a cuBLASLt
/// handle. Mirrors `scirust_gpu::GpuChain`'s role so `ResidentModel` can ride on
/// either backend once the op surface is complete (Route B, phases B2–B4).
pub struct CudaChain {
    // Held to keep the device context alive for the stream's lifetime.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
}

impl CudaChain {
    /// Acquire GPU 0, its default stream, and a cuBLASLt handle. Returns `None`
    /// if no CUDA device is available (so callers can fall back exactly like the
    /// wgpu path's `GpuChain::new`).
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone()).ok()?;
        Some(Self { ctx, stream, blas })
    }

    /// Upload a row-major `rows × cols` fp32 matrix to VRAM, rounding to bf16.
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> CudaMatrix {
        assert_eq!(data.len(), rows * cols, "upload: data len != rows*cols");
        let bf: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
        let buf = self.stream.clone_htod(&bf).expect("cuda htod");
        CudaMatrix { buf, rows, cols }
    }

    /// Download a resident bf16 matrix to a row-major fp32 `Vec`.
    pub fn download(&self, m: &CudaMatrix) -> Vec<f32> {
        let bf: Vec<bf16> = self.stream.clone_dtoh(&m.buf).expect("cuda dtoh");
        bf.iter().map(|x| x.to_f32()).collect()
    }

    /// `C = A · B` on Tensor cores: `a` is `m×k`, `b` is `k×n`, result `m×n`
    /// (row-major), bf16 in / fp32 accumulate / bf16 out.
    ///
    /// cuBLASLt is **column-major**; a row-major `M×N` buffer *is* a column-major
    /// `N×M` one, so to get row-major `C = A·B` we compute the column-major
    /// `Cᵀ = Bᵀ·Aᵀ` — i.e. pass `B` as the first operand and `A` as the second
    /// with `m`/`n` swapped. No data is transposed; only the descriptor changes.
    pub fn matmul(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        assert_eq!(b.rows, k, "matmul: inner dims disagree ({}x{} · {}x{})", a.rows, a.cols, b.rows, b.cols);
        let mut c = self.stream.alloc_zeros::<bf16>(m * n).expect("cuda alloc C");
        let cfg = MatmulConfig {
            transa: false,
            transb: false,
            transc: false,
            m: n as u64,
            n: m as u64,
            k: k as u64,
            alpha: 1.0,
            lda: n as i64,
            ldb: k as i64,
            beta: 0.0,
            ldc: n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        // SAFETY: shapes/leading-dims are consistent with the buffers above; the
        // bias/activation epilogues are unused.
        unsafe {
            self.blas
                .matmul(cfg, &b.buf, &a.buf, &mut c, None, None)
                .expect("cublasLt bf16 matmul");
        }
        CudaMatrix { buf: c, rows: m, cols: n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-30);
        num / den
    }

    /// The bf16 Tensor-core GEMM matches a CPU fp32 matmul within a
    /// bf16-appropriate relative tolerance (~8-bit mantissa ⇒ a few percent). This
    /// is B1's gradient-check: it confirms the cuBLASLt plumbing, the fp32→bf16
    /// round-trip, and the row-major/column-major layout are all correct. Skips
    /// cleanly with no CUDA device.
    #[test]
    fn bf16_matmul_matches_cpu_within_tol() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping bf16 matmul parity");
            return;
        };
        let (m, k, n) = (4usize, 3usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1 - 0.3).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.2 + 0.1).cos()).collect();

        let ga = chain.upload(&a, m, k);
        let gb = chain.upload(&b, k, n);
        let gc = chain.matmul(&ga, &gb);
        assert_eq!((gc.rows(), gc.cols()), (m, n), "output shape");
        let got = chain.download(&gc);
        let want = cpu_matmul(&a, &b, m, k, n);

        let e = rel_err(&got, &want);
        assert!(e < 5e-2, "bf16 matmul rel_err {e} too large\n got  {got:?}\n want {want:?}");
        eprintln!("bf16 Tensor-core matmul vs CPU fp32: rel_err {e:.3e} — PASS");
    }
}
