//! The resident CUDA op-chain (feature `cuda`).
//!
//! - B1: device/stream/cuBLASLt plumbing + the bf16 Tensor-core GEMM.
//! - B2: NVRTC custom-kernel pipeline + the element-wise/normalisation ops
//!   (`add`, `mul`, `swiglu`, `rms_norm`) — each gradient-checked against the CPU.
//!
//! Every op takes/returns bf16 [`CudaMatrix`] and computes in fp32 (accumulate /
//! reductions in fp32, round to bf16 on write), matching the mixed-precision
//! contract. Results agree with the fp32 CPU oracle to a bf16 tolerance (~`5e-2`).

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

/// Custom device kernels, compiled once at runtime via NVRTC (no build-time nvcc).
///
/// bf16 is handled **header-free**: a bf16 value is exactly the top 16 bits of an
/// fp32, so `b2f` widens with `<<16` (via `__uint_as_float`) and `f2b` rounds back
/// to nearest-even with the standard bias — using only `__uint_as_float` /
/// `__float_as_uint` (every arch, no include path). This sidesteps `<cuda_bf16.h>`
/// (NVRTC's usual friction) while keeping the fp32-compute contract. Buffers are
/// `CudaSlice<half::bf16>` (2 bytes); kernels view them as `unsigned short` —
/// byte-identical. The remaining resident ops (RoPE, softmax, slice/place, embed)
/// port here the same way toward the B3 forward.
const KERNELS_SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int s = __float_as_uint(f);
    unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);  // round to nearest even
    return (unsigned short)((s + bias) >> 16);
}

extern "C" __global__ void add_kernel(
    unsigned short* c, const unsigned short* a, const unsigned short* b, const size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = f2b(b2f(a[i]) + b2f(b[i]));
}

extern "C" __global__ void mul_kernel(
    unsigned short* c, const unsigned short* a, const unsigned short* b, const size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = f2b(b2f(a[i]) * b2f(b[i]));
}

// SwiGLU activation: silu(gate) * up, silu(x) = x * sigmoid(x) = x / (1 + e^-x).
extern "C" __global__ void swiglu_kernel(
    unsigned short* c, const unsigned short* g, const unsigned short* u, const size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = b2f(g[i]);
        float silu = x / (1.0f + __expf(-x));
        c[i] = f2b(silu * b2f(u[i]));
    }
}

// Row-wise RMSNorm: out[r,j] = x[r,j] / sqrt(mean_j(x[r,:]^2) + eps) * w[j].
// One thread per row; the sum of squares accumulates in fp32.
extern "C" __global__ void rmsnorm_kernel(
    unsigned short* out, const unsigned short* x, const unsigned short* w,
    const size_t rows, const size_t cols, const float eps)
{
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float ss = 0.0f;
        for (size_t j = 0; j < cols; j++) { float v = b2f(x[r*cols+j]); ss += v*v; }
        float inv = 1.0f / sqrtf(ss / (float)cols + eps);
        for (size_t j = 0; j < cols; j++)
            out[r*cols+j] = f2b(b2f(x[r*cols+j]) * inv * b2f(w[j]));
    }
}

// Gather columns [col_start, col_start+ncols) — a pure bf16 copy (no math).
extern "C" __global__ void slice_cols_kernel(
    unsigned short* out, const unsigned short* x,
    const size_t rows, const size_t src_cols, const size_t col_start, const size_t ncols)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * ncols) {
        size_t r = idx / ncols, c = idx % ncols;
        out[idx] = x[r * src_cols + col_start + c];
    }
}

// Scatter a narrow block into a zero-padded wide matrix at col_start.
extern "C" __global__ void place_cols_kernel(
    unsigned short* out, const unsigned short* x,
    const size_t rows, const size_t ncols, const size_t col_start, const size_t dst_cols)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * dst_cols) {
        size_t r = idx / dst_cols, c = idx % dst_cols;
        out[idx] = (c >= col_start && c < col_start + ncols)
                     ? x[r * ncols + (c - col_start)] : (unsigned short)0;
    }
}

// Row-wise softmax, max-subtracted for stability. One thread per row.
extern "C" __global__ void softmax_kernel(
    unsigned short* out, const unsigned short* x, const size_t rows, const size_t cols)
{
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float mx = -3.0e38f;
        for (size_t j = 0; j < cols; j++) { float v = b2f(x[r*cols+j]); if (v > mx) mx = v; }
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) sum += __expf(b2f(x[r*cols+j]) - mx);
        for (size_t j = 0; j < cols; j++)
            out[r*cols+j] = f2b(__expf(b2f(x[r*cols+j]) - mx) / sum);
    }
}

// Scale a t×t score matrix by `scale`, and (if causal) mask j>i to a large
// negative so softmax drives it to ~0.
extern "C" __global__ void scale_mask_kernel(
    unsigned short* out, const unsigned short* x,
    const size_t rows, const size_t cols, const float scale, const int causal)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        size_t i = idx / cols, j = idx % cols;
        float v = b2f(x[idx]) * scale;
        if (causal && j > i) v = -1.0e30f;
        out[idx] = f2b(v);
    }
}

// RoPE: interleaved-pair rotation. pos = (row mod seq_len) + offset,
// freq_p = theta^(-2p/dim), angle = pos*freq_p; one thread per (row, pair).
extern "C" __global__ void rope_kernel(
    unsigned short* out, const unsigned short* x, const size_t rows, const size_t dim,
    const size_t seq_len, const size_t offset, const float theta)
{
    size_t pairs = dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * pairs) {
        size_t r = idx / pairs, p = idx % pairs;
        float pos = (float)((r % seq_len) + offset);
        float freq = powf(theta, -2.0f * (float)p / (float)dim);
        float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        float x0 = b2f(x[r*dim + 2*p]);
        float x1 = b2f(x[r*dim + 2*p + 1]);
        out[r*dim + 2*p]     = f2b(x0 * c - x1 * s);
        out[r*dim + 2*p + 1] = f2b(x0 * s + x1 * c);
    }
}

// Embedding gather: out row i = table row tokens[i]. Pure copy.
extern "C" __global__ void embed_kernel(
    unsigned short* out, const unsigned int* tokens, const unsigned short* table,
    const size_t n_tokens, const size_t d)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tokens * d) {
        size_t i = idx / d, j = idx % d;
        out[idx] = table[(size_t)tokens[i] * d + j];
    }
}
"#;

/// A resident row-major `rows × cols` matrix in VRAM, stored in **bf16** (the
/// Tensor-core input type). The fp32 → bf16 rounding happens on upload; fp32
/// accumulation happens inside each op.
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

/// The NVRTC-compiled kernel handles. Loaded together; if compilation fails the
/// chain keeps working for cuBLASLt GEMM (so B1 stays independently testable) and
/// only the kernel ops error.
struct Kernels {
    add: CudaFunction,
    mul: CudaFunction,
    swiglu: CudaFunction,
    rmsnorm: CudaFunction,
    slice_cols: CudaFunction,
    place_cols: CudaFunction,
    softmax: CudaFunction,
    scale_mask: CudaFunction,
    rope: CudaFunction,
    embed: CudaFunction,
}

/// The CUDA backend handle: a device context, its default stream, a cuBLASLt
/// handle, and the custom kernels. Mirrors `scirust_gpu::GpuChain`'s role so
/// `ResidentModel` can ride on either backend once the op surface is complete
/// (Route B, phases B2–B4).
pub struct CudaChain {
    // Held to keep the device context alive for the stream's lifetime.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    kernels: Option<Kernels>,
}

impl CudaChain {
    /// Acquire GPU 0, its default stream, and a cuBLASLt handle, and compile the
    /// custom kernels. Returns `None` if no CUDA device is available (so callers
    /// fall back exactly like the wgpu path's `GpuChain::new`).
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone()).ok()?;
        let kernels = Self::compile_kernels(&ctx);
        Some(Self {
            ctx,
            stream,
            blas,
            kernels,
        })
    }

    /// Compile + load the NVRTC kernels (non-fatal: any error is surfaced to stderr
    /// and leaves the GEMM path intact).
    fn compile_kernels(ctx: &Arc<CudaContext>) -> Option<Kernels> {
        let ptx = compile_ptx(KERNELS_SRC)
            .map_err(|e| eprintln!("scirust-cuda: NVRTC compile failed: {e}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| eprintln!("scirust-cuda: load_module failed: {e}"))
            .ok()?;
        let f = |name: &str| module.load_function(name).expect("load kernel");
        Some(Kernels {
            add: f("add_kernel"),
            mul: f("mul_kernel"),
            swiglu: f("swiglu_kernel"),
            rmsnorm: f("rmsnorm_kernel"),
            slice_cols: f("slice_cols_kernel"),
            place_cols: f("place_cols_kernel"),
            softmax: f("softmax_kernel"),
            scale_mask: f("scale_mask_kernel"),
            rope: f("rope_kernel"),
            embed: f("embed_kernel"),
        })
    }

    fn kernels(&self) -> &Kernels {
        self.kernels
            .as_ref()
            .expect("scirust-cuda kernels failed to compile")
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
    /// `Cᵀ = Bᵀ·Aᵀ` — pass `B` first and `A` second with `m`/`n` swapped. No data
    /// is transposed; only the descriptor changes.
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

    /// Launch an element-wise binary kernel `c = f(a, b)` on equal-shaped inputs.
    fn binary(&self, a: &CudaMatrix, b: &CudaMatrix, f: &CudaFunction, op: &str) -> CudaMatrix {
        assert_eq!(
            (a.rows, a.cols),
            (b.rows, b.cols),
            "{op}: shape mismatch {}x{} vs {}x{}",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        );
        let n = a.rows * a.cols;
        let mut c = self.stream.alloc_zeros::<bf16>(n).expect("cuda alloc");
        let n_arg = n;
        let mut builder = self.stream.launch_builder(f);
        builder.arg(&mut c);
        builder.arg(&a.buf);
        builder.arg(&b.buf);
        builder.arg(&n_arg);
        // SAFETY: arg order/types match the kernel; the grid covers `n`.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("launch binary kernel");
        }
        CudaMatrix {
            buf: c,
            rows: a.rows,
            cols: a.cols,
        }
    }

    /// Element-wise `C = A + B` (residual add).
    pub fn add(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        self.binary(a, b, &self.kernels().add, "add")
    }

    /// Element-wise `C = A ⊙ B` (Hadamard product).
    pub fn mul(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        self.binary(a, b, &self.kernels().mul, "mul")
    }

    /// SwiGLU activation `silu(gate) ⊙ up` (equal shapes) — the MLP nonlinearity.
    pub fn swiglu(&self, gate: &CudaMatrix, up: &CudaMatrix) -> CudaMatrix {
        self.binary(gate, up, &self.kernels().swiglu, "swiglu")
    }

    /// Row-wise RMSNorm: `x / sqrt(mean(x²) + eps) · weight`. `weight` is a
    /// `cols`-length gain vector (any shape whose element count is `x.cols`).
    /// One thread per row; the sum of squares accumulates in fp32.
    pub fn rms_norm(&self, x: &CudaMatrix, weight: &CudaMatrix, eps: f32) -> CudaMatrix {
        assert_eq!(
            weight.rows * weight.cols,
            x.cols,
            "rms_norm: weight has {} elems, expected cols = {}",
            weight.rows * weight.cols,
            x.cols
        );
        let (rows, cols) = (x.rows, x.cols);
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a, eps_a) = (rows, cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels().rmsnorm);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&weight.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&eps_a);
        // SAFETY: arg order/types match `rmsnorm_kernel`; one thread per row.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(rows as u32))
                .expect("launch rmsnorm_kernel");
        }
        CudaMatrix {
            buf: out,
            rows,
            cols,
        }
    }

    /// `C = A · Bᵀ` on Tensor cores: `a` is `m×k`, `b` is `n×k`, result `m×n`
    /// (row-major). The tied LM head is `normed · Eᵀ` (E is `vocab×d`).
    ///
    /// Column-major identity: `Cᵀ = B·Aᵀ`. My row-major `b` viewed column-major is
    /// `bᵀ`, so `transa=true` recovers `b`; my row-major `a` viewed column-major is
    /// already `aᵀ`. Hence `matmul(transa=true, transb=false, m=n, n=m, k)` over
    /// `(b, a)` yields row-major `A·Bᵀ`.
    pub fn matmul_bt(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        let (m, k, n) = (a.rows, a.cols, b.rows);
        assert_eq!(
            b.cols, k,
            "matmul_bt: inner dims disagree ({}x{} · ({}x{})ᵀ)",
            a.rows, a.cols, b.rows, b.cols
        );
        let mut c = self
            .stream
            .alloc_zeros::<bf16>(m * n)
            .expect("cuda alloc C");
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n as u64,
            n: m as u64,
            k: k as u64,
            alpha: 1.0,
            lda: k as i64,
            ldb: k as i64,
            beta: 0.0,
            ldc: n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        // SAFETY: shapes/leading-dims match the buffers; epilogues unused.
        unsafe {
            self.blas
                .matmul(cfg, &b.buf, &a.buf, &mut c, None, None)
                .expect("cublasLt bf16 matmul_bt");
        }
        CudaMatrix {
            buf: c,
            rows: m,
            cols: n,
        }
    }

    /// Gather columns `[col_start, col_start+ncols)` into a `rows × ncols` matrix
    /// (one head's slice of a full-width projection).
    pub fn slice_cols(&self, x: &CudaMatrix, col_start: usize, ncols: usize) -> CudaMatrix {
        assert!(
            col_start + ncols <= x.cols,
            "slice_cols: [{col_start}, {}) out of {} cols",
            col_start + ncols,
            x.cols
        );
        let (rows, src_cols) = (x.rows, x.cols);
        let total = rows * ncols;
        let mut out = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc");
        let (rows_a, src_a, start_a, ncols_a) = (rows, src_cols, col_start, ncols);
        let mut builder = self.stream.launch_builder(&self.kernels().slice_cols);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&src_a);
        builder.arg(&start_a);
        builder.arg(&ncols_a);
        // SAFETY: arg order/types match `slice_cols_kernel`; grid covers rows*ncols.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch slice_cols_kernel");
        }
        CudaMatrix {
            buf: out,
            rows,
            cols: ncols,
        }
    }

    /// Scatter a `rows × ncols` block into a zero-padded `rows × dst_cols` matrix
    /// at `col_start` (place a head's context back into its `d_model` slot).
    pub fn place_cols(&self, x: &CudaMatrix, col_start: usize, dst_cols: usize) -> CudaMatrix {
        assert!(
            col_start + x.cols <= dst_cols,
            "place_cols: block [{col_start}, {}) does not fit in {dst_cols}",
            col_start + x.cols
        );
        let (rows, ncols) = (x.rows, x.cols);
        let total = rows * dst_cols;
        let mut out = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc");
        let (rows_a, ncols_a, start_a, dst_a) = (rows, ncols, col_start, dst_cols);
        let mut builder = self.stream.launch_builder(&self.kernels().place_cols);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&ncols_a);
        builder.arg(&start_a);
        builder.arg(&dst_a);
        // SAFETY: arg order/types match `place_cols_kernel`; grid covers rows*dst_cols.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch place_cols_kernel");
        }
        CudaMatrix {
            buf: out,
            rows,
            cols: dst_cols,
        }
    }

    /// Row-wise softmax (max-subtracted), result resident. One thread per row.
    pub fn softmax(&self, x: &CudaMatrix) -> CudaMatrix {
        let (rows, cols) = (x.rows, x.cols);
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().softmax);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        // SAFETY: arg order/types match `softmax_kernel`; one thread per row.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(rows as u32))
                .expect("launch softmax_kernel");
        }
        CudaMatrix {
            buf: out,
            rows,
            cols,
        }
    }

    /// Scale a `t×t` score matrix by `scale` and (optionally) apply the causal
    /// mask (`j>i` → large negative so softmax zeroes it), result resident.
    pub fn scale_causal_mask(&self, x: &CudaMatrix, scale: f32, causal: bool) -> CudaMatrix {
        let (rows, cols) = (x.rows, x.cols);
        let total = rows * cols;
        let mut out = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc");
        let (rows_a, cols_a, scale_a) = (rows, cols, scale);
        let causal_a: i32 = causal as i32;
        let mut builder = self.stream.launch_builder(&self.kernels().scale_mask);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&scale_a);
        builder.arg(&causal_a);
        // SAFETY: arg order/types match `scale_mask_kernel`; grid covers rows*cols.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch scale_mask_kernel");
        }
        CudaMatrix {
            buf: out,
            rows,
            cols,
        }
    }

    /// RoPE: interleaved-pair rotation of a `rows × dim` matrix, `pos = (row mod
    /// seq_len) + offset`, `freq_p = theta^(-2p/dim)`. `dim` must be even. One
    /// thread per `(row, pair)`.
    pub fn rope(&self, x: &CudaMatrix, seq_len: usize, offset: usize, theta: f32) -> CudaMatrix {
        assert_eq!(x.cols % 2, 0, "rope: dim must be even, got {}", x.cols);
        let (rows, dim) = (x.rows, x.cols);
        let total = rows * (dim / 2);
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * dim)
            .expect("cuda alloc");
        let (rows_a, dim_a, seq_a, off_a, theta_a) = (rows, dim, seq_len, offset, theta);
        let mut builder = self.stream.launch_builder(&self.kernels().rope);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&dim_a);
        builder.arg(&seq_a);
        builder.arg(&off_a);
        builder.arg(&theta_a);
        // SAFETY: arg order/types match `rope_kernel`; one thread per (row, pair).
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch rope_kernel");
        }
        CudaMatrix {
            buf: out,
            rows,
            cols: dim,
        }
    }

    /// Token embedding gather: build a `tokens.len() × d` matrix whose row `i` is
    /// row `tokens[i]` of the `vocab × d` `table`.
    pub fn embed(&self, tokens: &[u32], table: &CudaMatrix) -> CudaMatrix {
        let (n, d) = (tokens.len(), table.cols);
        let toks = self.stream.clone_htod(tokens).expect("cuda htod tokens");
        let total = n * d;
        let mut out = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc");
        let (n_a, d_a) = (n, d);
        let mut builder = self.stream.launch_builder(&self.kernels().embed);
        builder.arg(&mut out);
        builder.arg(&toks);
        builder.arg(&table.buf);
        builder.arg(&n_a);
        builder.arg(&d_a);
        // SAFETY: arg order/types match `embed_kernel`; grid covers n*d.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch embed_kernel");
        }
        CudaMatrix {
            buf: out,
            rows: n,
            cols: d,
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

    fn cpu_rms_norm(x: &[f32], w: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows
        {
            let ss: f32 = (0..cols).map(|j| x[r * cols + j].powi(2)).sum();
            let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
            for j in 0..cols
            {
                out[r * cols + j] = x[r * cols + j] * inv * w[j];
            }
        }
        out
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

    /// The kernel source compiles under NVRTC — surfaces the compiler log verbatim
    /// on failure (so a broken kernel is diagnosable, not a silent `None`). NVRTC
    /// needs the CUDA runtime, so it still only runs on the Thor.
    #[test]
    fn nvrtc_kernels_compile() {
        match compile_ptx(KERNELS_SRC)
        {
            Ok(_) => eprintln!("NVRTC compiled scirust-cuda kernels — PASS"),
            Err(e) => panic!("NVRTC failed to compile kernels:\n{e}"),
        }
    }

    /// The bf16 Tensor-core GEMM matches a CPU fp32 matmul within a bf16 tolerance
    /// — B1's check (cuBLASLt plumbing, fp32→bf16 round-trip, and the row/column-
    /// major layout). Skips with no device.
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
        let gc = chain.matmul(&chain.upload(&a, m, k), &chain.upload(&b, k, n));
        assert_eq!((gc.rows(), gc.cols()), (m, n), "output shape");
        let e = rel_err(&chain.download(&gc), &cpu_matmul(&a, &b, m, k, n));
        assert!(e < 5e-2, "bf16 matmul rel_err {e} too large");
        eprintln!("bf16 Tensor-core matmul vs CPU fp32: rel_err {e:.3e} — PASS");
    }

    /// The element-wise / normalisation kernels (`add`, `mul`, `swiglu`,
    /// `rms_norm`) each match their CPU fp32 reference within bf16 tolerance —
    /// B2's checks over the whole compile→load→launch pipeline. Skips with no
    /// device.
    #[test]
    fn bf16_elementwise_and_rmsnorm_match_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping bf16 kernel parity");
            return;
        };
        let (rows, cols) = (3usize, 8usize);
        let n = rows * cols;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23 - 0.5).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11 + 0.2).cos()).collect();
        let ga = chain.upload(&a, rows, cols);
        let gb = chain.upload(&b, rows, cols);

        let add = chain.download(&chain.add(&ga, &gb));
        let want_add: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let e_add = rel_err(&add, &want_add);
        assert!(e_add < 5e-2, "add rel_err {e_add}");

        let mul = chain.download(&chain.mul(&ga, &gb));
        let want_mul: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        let e_mul = rel_err(&mul, &want_mul);
        assert!(e_mul < 5e-2, "mul rel_err {e_mul}");

        let swi = chain.download(&chain.swiglu(&ga, &gb));
        let want_swi: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x / (1.0 + (-x).exp())) * y)
            .collect();
        let e_swi = rel_err(&swi, &want_swi);
        assert!(e_swi < 5e-2, "swiglu rel_err {e_swi}");

        let w: Vec<f32> = (0..cols).map(|j| 0.5 + 0.1 * j as f32).collect();
        let gw = chain.upload(&w, 1, cols);
        let rn = chain.download(&chain.rms_norm(&ga, &gw, 1e-5));
        let want_rn = cpu_rms_norm(&a, &w, rows, cols, 1e-5);
        let e_rn = rel_err(&rn, &want_rn);
        assert!(e_rn < 5e-2, "rms_norm rel_err {e_rn}");

        eprintln!(
            "bf16 kernels vs CPU — add {e_add:.2e}, mul {e_mul:.2e}, swiglu {e_swi:.2e}, rms_norm {e_rn:.2e} — PASS"
        );
    }

    /// `matmul_bt` computes `A·Bᵀ` (the tied LM head shape) within bf16 tolerance —
    /// confirms the transpose config + column-major layout. Skips with no device.
    #[test]
    fn bf16_matmul_bt_matches_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping matmul_bt parity");
            return;
        };
        // a: m×k, b: n×k, want a·bᵀ : m×n.
        let (m, k, n) = (4usize, 6usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13 - 0.4).sin()).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.09 + 0.2).cos()).collect();
        let got =
            chain.download(&chain.matmul_bt(&chain.upload(&a, m, k), &chain.upload(&b, n, k)));
        // CPU a·bᵀ : out[i,j] = Σ_p a[i,p]·b[j,p].
        let mut want = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                want[i * n + j] = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
            }
        }
        let e = rel_err(&got, &want);
        assert!(
            e < 5e-2,
            "matmul_bt rel_err {e}\n got {got:?}\n want {want:?}"
        );
        eprintln!("bf16 matmul_bt (A·Bᵀ) vs CPU fp32: rel_err {e:.3e} — PASS");
    }

    /// `slice_cols` then `place_cols` round-trips a head slice back into its slot
    /// (zeros elsewhere) — the per-head attention split/merge. Exact (pure copy),
    /// so compares bit-for-bit against the bf16-rounded input. Skips with no device.
    #[test]
    fn bf16_slice_place_cols_round_trip() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping slice/place parity");
            return;
        };
        let (rows, cols) = (4usize, 12usize);
        let (start, ncols) = (4usize, 3usize); // a "head" at columns [4, 7)
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.17 - 1.0).sin())
            .collect();
        let gx = chain.upload(&x, rows, cols);

        let sliced = chain.slice_cols(&gx, start, ncols);
        assert_eq!((sliced.rows(), sliced.cols()), (rows, ncols), "slice shape");
        let sliced_h = chain.download(&sliced);
        // Reference: the bf16-rounded input's slice (pure copy ⇒ bit-exact).
        let x_bf: Vec<f32> = x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
        let mut want_slice = Vec::with_capacity(rows * ncols);
        for r in 0..rows
        {
            for c in 0..ncols
            {
                want_slice.push(x_bf[r * cols + start + c]);
            }
        }
        assert_eq!(sliced_h, want_slice, "slice_cols must be an exact gather");

        let placed = chain.place_cols(&sliced, start, cols);
        assert_eq!((placed.rows(), placed.cols()), (rows, cols), "place shape");
        let placed_h = chain.download(&placed);
        let mut want_place = vec![0.0f32; rows * cols];
        for r in 0..rows
        {
            for c in 0..ncols
            {
                want_place[r * cols + start + c] = x_bf[r * cols + start + c];
            }
        }
        assert_eq!(
            placed_h, want_place,
            "place_cols must scatter into a zero slot"
        );
        eprintln!("bf16 slice_cols/place_cols round-trip — PASS");
    }

    fn cpu_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows
        {
            let row = &x[r * cols..(r + 1) * cols];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..cols
            {
                out[r * cols + j] = exps[j] / sum;
            }
        }
        out
    }

    /// `softmax` and the `scale_causal_mask → softmax` attention-score pipeline
    /// match their CPU references within bf16 tolerance. Skips with no device.
    #[test]
    fn bf16_softmax_and_causal_mask_match_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping softmax/mask parity");
            return;
        };
        let (rows, cols) = (5usize, 5usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.3 - 2.0).sin())
            .collect();
        let gx = chain.upload(&x, rows, cols);

        // Plain row softmax.
        let sm = chain.download(&chain.softmax(&gx));
        let e_sm = rel_err(&sm, &cpu_softmax(&x, rows, cols));
        assert!(e_sm < 5e-2, "softmax rel_err {e_sm}");

        // Scaled + causally-masked softmax (the attention score path).
        let scale = 0.5f32;
        let masked = chain.softmax(&chain.scale_causal_mask(&gx, scale, true));
        let got = chain.download(&masked);
        // CPU reference: scale, mask upper triangle to -inf, softmax.
        let mut ref_in = vec![0.0f32; rows * cols];
        for i in 0..rows
        {
            for j in 0..cols
            {
                ref_in[i * cols + j] = if j > i
                {
                    f32::NEG_INFINITY
                }
                else
                {
                    x[i * cols + j] * scale
                };
            }
        }
        let e_mask = rel_err(&got, &cpu_softmax(&ref_in, rows, cols));
        assert!(e_mask < 5e-2, "causal-masked softmax rel_err {e_mask}");
        eprintln!("bf16 softmax {e_sm:.2e} + causal-masked softmax {e_mask:.2e} vs CPU — PASS");
    }

    fn cpu_rope(
        x: &[f32],
        rows: usize,
        dim: usize,
        seq_len: usize,
        off: usize,
        theta: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows
        {
            let pos = ((r % seq_len) + off) as f32;
            for p in 0..dim / 2
            {
                let freq = theta.powf(-2.0 * p as f32 / dim as f32);
                let (s, c) = (pos * freq).sin_cos();
                let x0 = x[r * dim + 2 * p];
                let x1 = x[r * dim + 2 * p + 1];
                out[r * dim + 2 * p] = x0 * c - x1 * s;
                out[r * dim + 2 * p + 1] = x0 * s + x1 * c;
            }
        }
        out
    }

    /// `rope` matches a CPU interleaved-pair rotation within bf16 tolerance (the
    /// convention is confirmed end-to-end by the B3 forward parity). Skips with no
    /// device.
    #[test]
    fn bf16_rope_matches_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping rope parity");
            return;
        };
        let (rows, dim, seq_len, theta) = (6usize, 8usize, 6usize, 10_000.0f32);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.21 - 0.7).cos())
            .collect();
        let got = chain.download(&chain.rope(&chain.upload(&x, rows, dim), seq_len, 0, theta));
        let e = rel_err(&got, &cpu_rope(&x, rows, dim, seq_len, 0, theta));
        assert!(e < 5e-2, "rope rel_err {e}");
        eprintln!("bf16 rope vs CPU interleaved rotation: rel_err {e:.3e} — PASS");
    }

    /// `embed` gathers table rows by token id — exact (pure copy) against the
    /// bf16-rounded table. Skips with no device.
    #[test]
    fn bf16_embed_gathers_rows() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping embed parity");
            return;
        };
        let (vocab, d) = (10usize, 4usize);
        let table: Vec<f32> = (0..vocab * d)
            .map(|i| (i as f32 * 0.19 - 0.5).sin())
            .collect();
        let gtab = chain.upload(&table, vocab, d);
        let tokens = [3u32, 0, 7, 3, 9];
        let got = chain.download(&chain.embed(&tokens, &gtab));

        let tbf: Vec<f32> = table.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
        let mut want = Vec::with_capacity(tokens.len() * d);
        for &t in &tokens
        {
            for j in 0..d
            {
                want.push(tbf[t as usize * d + j]);
            }
        }
        assert_eq!(got, want, "embed must gather the exact table rows");
        eprintln!("bf16 embed gather — PASS");
    }
}
