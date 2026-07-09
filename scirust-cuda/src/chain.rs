//! The resident CUDA op-chain (feature `cuda`). B1: device/stream/cuBLASLt
//! plumbing + the first bf16 Tensor-core GEMM, gradient-checked against the CPU.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

/// Custom device kernels, compiled once at runtime via NVRTC (no build-time
/// nvcc). B2 starts with element-wise `add` (the residual adds); the rest of the
/// validated WGSL ops (RMSNorm, RoPE, SwiGLU, softmax, slice/place, embed) port
/// here the same way.
///
/// bf16 is handled **header-free**: a bf16 value is exactly the top 16 bits of an
/// fp32, so we widen with `<<16` (via `__uint_as_float`) and round back to nearest-
/// even with the standard bias — using only `__uint_as_float`/`__float_as_uint`,
/// which every arch has and NVRTC compiles with no include path. This sidesteps
/// `<cuda_bf16.h>` (NVRTC's usual friction) while keeping the fp32-accumulate
/// contract. The buffers are `CudaSlice<half::bf16>` (2 bytes); the kernel views
/// them as `unsigned short` — byte-identical.
const KERNELS_SRC: &str = r#"
extern "C" __global__ void add_kernel(
    unsigned short* c, const unsigned short* a, const unsigned short* b, const size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float fa = __uint_as_float(((unsigned int)a[i]) << 16);
        float fb = __uint_as_float(((unsigned int)b[i]) << 16);
        unsigned int s = __float_as_uint(fa + fb);
        unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);  // round to nearest even
        c[i] = (unsigned short)((s + bias) >> 16);
    }
}
"#;

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
    // NVRTC-compiled custom kernels. `None` if compilation failed — GEMM (cuBLASLt)
    // still works, so B1 stays independently testable; only `add` then errors.
    add_fn: Option<CudaFunction>,
}

impl CudaChain {
    /// Acquire GPU 0, its default stream, and a cuBLASLt handle. Returns `None`
    /// if no CUDA device is available (so callers can fall back exactly like the
    /// wgpu path's `GpuChain::new`).
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone()).ok()?;
        // Compile the custom kernels once (non-fatal: GEMM works regardless). Any
        // NVRTC error is surfaced to stderr so a kernel issue is diagnosable
        // rather than silently disabling `add`.
        let add_fn = match compile_ptx(KERNELS_SRC)
        {
            Ok(ptx) => ctx
                .load_module(ptx)
                .and_then(|m| m.load_function("add_kernel"))
                .map_err(|e| eprintln!("scirust-cuda: load add_kernel failed: {e}"))
                .ok(),
            Err(e) =>
            {
                eprintln!("scirust-cuda: NVRTC compile failed: {e}");
                None
            },
        };
        Some(Self {
            ctx,
            stream,
            blas,
            add_fn,
        })
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
        assert_eq!(
            b.rows, k,
            "matmul: inner dims disagree ({}x{} · {}x{})",
            a.rows, a.cols, b.rows, b.cols
        );
        let mut c = self
            .stream
            .alloc_zeros::<bf16>(m * n)
            .expect("cuda alloc C");
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
        CudaMatrix {
            buf: c,
            rows: m,
            cols: n,
        }
    }

    /// Element-wise `C = A + B` (equal shapes) via the NVRTC `add_kernel` — the
    /// residual add. Proves the custom-kernel pipeline (compile → load → launch)
    /// end to end; the remaining resident ops follow the same pattern.
    pub fn add(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        assert_eq!(
            (a.rows, a.cols),
            (b.rows, b.cols),
            "add: shape mismatch {}x{} vs {}x{}",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        );
        let n = a.rows * a.cols;
        let mut c = self.stream.alloc_zeros::<bf16>(n).expect("cuda alloc add");
        let f = self.add_fn.as_ref().expect("add_kernel failed to compile");
        let n_arg = n; // size_t (usize)
        let mut builder = self.stream.launch_builder(f);
        builder.arg(&mut c);
        builder.arg(&a.buf);
        builder.arg(&b.buf);
        builder.arg(&n_arg);
        // SAFETY: arg order/types match `add_kernel`; the launch grid covers `n`.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("launch add_kernel");
        }
        CudaMatrix {
            buf: c,
            rows: a.rows,
            cols: a.cols,
        }
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
        let num: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
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
        assert!(
            e < 5e-2,
            "bf16 matmul rel_err {e} too large\n got  {got:?}\n want {want:?}"
        );
        eprintln!("bf16 Tensor-core matmul vs CPU fp32: rel_err {e:.3e} — PASS");
    }

    /// The kernel source compiles under NVRTC — surfaces the compiler log verbatim
    /// on failure (so a broken kernel is diagnosable, not a silent `None`). Does
    /// not need a device to launch, but NVRTC needs the CUDA runtime, so it still
    /// only runs on the Thor.
    #[test]
    fn nvrtc_kernels_compile() {
        match compile_ptx(KERNELS_SRC)
        {
            Ok(_) => eprintln!("NVRTC compiled scirust-cuda kernels — PASS"),
            Err(e) => panic!("NVRTC failed to compile kernels:\n{e}"),
        }
    }

    /// The NVRTC `add_kernel` computes element-wise `A + B` in bf16, matching a CPU
    /// fp32 add within bf16 tolerance. This is B2's check — it exercises the whole
    /// custom-kernel pipeline (runtime compile, module load, launch). Skips with no
    /// device.
    #[test]
    fn bf16_add_matches_cpu_within_tol() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping bf16 add parity");
            return;
        };
        let (rows, cols) = (3usize, 7usize);
        let a: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.23 - 0.5).sin())
            .collect();
        let b: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.11 + 0.2).cos())
            .collect();

        let ga = chain.upload(&a, rows, cols);
        let gb = chain.upload(&b, rows, cols);
        let got = chain.download(&chain.add(&ga, &gb));
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

        let e = rel_err(&got, &want);
        assert!(
            e < 5e-2,
            "bf16 add rel_err {e} too large\n got  {got:?}\n want {want:?}"
        );
        eprintln!("bf16 add_kernel (NVRTC) vs CPU fp32: rel_err {e:.3e} — PASS");
    }
}
