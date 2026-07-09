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

// ---- backward adjoints (B4) ----

// Softmax backward: dx = y ⊙ (dy − Σⱼ dyⱼyⱼ) per row. One thread per row.
extern "C" __global__ void softmax_bwd_kernel(
    unsigned short* dx, const unsigned short* y, const unsigned short* dy,
    const size_t rows, const size_t cols)
{
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float s = 0.0f;
        for (size_t j = 0; j < cols; j++) s += b2f(dy[r*cols+j]) * b2f(y[r*cols+j]);
        for (size_t j = 0; j < cols; j++) {
            float yv = b2f(y[r*cols+j]);
            dx[r*cols+j] = f2b(yv * (b2f(dy[r*cols+j]) - s));
        }
    }
}

// SwiGLU backward of c = silu(a) ⊙ b: da = dc·silu'(a)·b, db = dc·silu(a),
// silu'(x) = σ(x)·(1 + x·(1−σ(x))). Elementwise, two outputs.
extern "C" __global__ void swiglu_bwd_kernel(
    unsigned short* da, unsigned short* db,
    const unsigned short* a, const unsigned short* b, const unsigned short* dc,
    const size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float av = b2f(a[i]);
        float sig = 1.0f / (1.0f + __expf(-av));
        float silu = av * sig;
        float dsilu = sig * (1.0f + av * (1.0f - sig));
        float dcv = b2f(dc[i]);
        da[i] = f2b(dcv * dsilu * b2f(b[i]));
        db[i] = f2b(dcv * silu);
    }
}

// RMSNorm input backward: dx_j = dy_j·w_j/rms − x_j·(Σₖ dyₖwₖxₖ)/(cols·rms³),
// rms = √(mean(x²)+eps). One thread per row (matches rmsnorm_kernel's reduction).
extern "C" __global__ void rmsnorm_bwd_kernel(
    unsigned short* dx, const unsigned short* x, const unsigned short* w,
    const unsigned short* dy, const size_t rows, const size_t cols, const float eps)
{
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float ss = 0.0f;
        for (size_t j = 0; j < cols; j++) { float v = b2f(x[r*cols+j]); ss += v*v; }
        float ms = ss / (float)cols + eps;
        float rms = sqrtf(ms);
        float dot = 0.0f;
        for (size_t j = 0; j < cols; j++)
            dot += b2f(dy[r*cols+j]) * b2f(w[j]) * b2f(x[r*cols+j]);
        float coef = dot / ((float)cols * ms * rms);
        for (size_t j = 0; j < cols; j++)
            dx[r*cols+j] = f2b(b2f(dy[r*cols+j]) * b2f(w[j]) / rms - b2f(x[r*cols+j]) * coef);
    }
}

// RMSNorm gain backward: dw_j = Σ_r dy[r,j]·x[r,j]/rms[r]. One thread per column;
// each recomputes the per-row rms (self-contained, no shared per-row buffer).
extern "C" __global__ void rmsnorm_gain_bwd_kernel(
    unsigned short* dw, const unsigned short* x, const unsigned short* dy,
    const size_t rows, const size_t cols, const float eps)
{
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float acc = 0.0f;
        for (size_t r = 0; r < rows; r++) {
            float ss = 0.0f;
            for (size_t k = 0; k < cols; k++) { float v = b2f(x[r*cols+k]); ss += v*v; }
            float rms = sqrtf(ss / (float)cols + eps);
            acc += b2f(dy[r*cols+j]) * b2f(x[r*cols+j]) / rms;
        }
        dw[j] = f2b(acc);
    }
}

// Scale + causal-mask backward: din = scale·dout at kept positions, 0 above the
// diagonal (masked keys carry no gradient). Elementwise.
extern "C" __global__ void scale_mask_bwd_kernel(
    unsigned short* din, const unsigned short* dout,
    const size_t rows, const size_t cols, const float scale, const int causal)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        size_t i = idx / cols, j = idx % cols;
        float v = (causal && j > i) ? 0.0f : b2f(dout[idx]) * scale;
        din[idx] = f2b(v);
    }
}

// RoPE backward — the adjoint (transpose) rotation, same pos/freq as rope_kernel:
// dx[2p] = c·dy[2p] + s·dy[2p+1], dx[2p+1] = −s·dy[2p] + c·dy[2p+1].
extern "C" __global__ void rope_bwd_kernel(
    unsigned short* dx, const unsigned short* dy, const size_t rows, const size_t dim,
    const size_t seq_len, const size_t offset, const float theta)
{
    size_t pairs = dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * pairs) {
        size_t r = idx / pairs, p = idx % pairs;
        float pos = (float)((r % seq_len) + offset);
        float freq = powf(theta, -2.0f * (float)p / (float)dim);
        float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        float ge = b2f(dy[r*dim + 2*p]);
        float go = b2f(dy[r*dim + 2*p + 1]);
        dx[r*dim + 2*p]     = f2b(c * ge + s * go);
        dx[r*dim + 2*p + 1] = f2b(-s * ge + c * go);
    }
}

// Embedding gather backward (scatter-add): dtable[v,c] = Σ_i (clamp(tok_i)==v)·dout[i,c].
// One thread per (vocab row, col) output element — atomic-free and deterministic
// (each output owns its full reduction over the token stream). Token ids are
// clamped to vocab-1, matching cpu_embed_backward.
extern "C" __global__ void embed_bwd_kernel(
    unsigned short* dtable, const unsigned int* tokens, const unsigned short* dout,
    const size_t n_tokens, const size_t d, const size_t vocab)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab * d) {
        size_t v = idx / d, c = idx % d;
        float acc = 0.0f;
        for (size_t i = 0; i < n_tokens; i++) {
            size_t t = (size_t)tokens[i];
            if (t >= vocab) t = vocab - 1;
            if (t == v) acc += b2f(dout[i * d + c]);
        }
        dtable[idx] = f2b(acc);
    }
}

// Cross-entropy gradient w.r.t. logits: d = (softmax(logits) − onehot(target))/rows.
// One thread per row (fp32 max-subtracted softmax); target clamped to cols-1.
extern "C" __global__ void ce_grad_kernel(
    unsigned short* d, const unsigned short* logits, const unsigned int* targets,
    const size_t rows, const size_t cols)
{
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float mx = -3.0e38f;
        for (size_t j = 0; j < cols; j++) { float v = b2f(logits[r*cols+j]); if (v > mx) mx = v; }
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) sum += __expf(b2f(logits[r*cols+j]) - mx);
        size_t tgt = (size_t)targets[r];
        if (tgt >= cols) tgt = cols - 1;
        float inv = 1.0f / (float)rows;
        for (size_t j = 0; j < cols; j++) {
            float p = __expf(b2f(logits[r*cols+j]) - mx) / sum;
            if (j == tgt) p -= 1.0f;
            d[r*cols+j] = f2b(p * inv);
        }
    }
}

// AdamW step (mixed precision): fp32 master `param`, fp32 moments `m`/`v`, bf16
// `grad`. Updates param/m/v in fp32 in place and writes a fresh bf16 `param_bf`
// view for the next forward. bc1/bc2 = 1 − beta^step are host-computed. Matches
// cpu_adamw_step exactly (decoupled weight decay).
extern "C" __global__ void adamw_kernel(
    float* param, float* m, float* v, const unsigned short* grad, unsigned short* param_bf,
    const size_t n, const float lr, const float b1, const float b2,
    const float eps, const float wd, const float bc1, const float bc2, const float grad_scale)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = b2f(grad[i]) * grad_scale;
        float mi = b1 * m[i] + (1.0f - b1) * g;
        float vi = b2 * v[i] + (1.0f - b2) * g * g;
        m[i] = mi;
        v[i] = vi;
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        float p = param[i];
        p -= lr * (mhat / (sqrtf(vhat) + eps) + wd * p);
        param[i] = p;
        param_bf[i] = f2b(p);
    }
}

// Sum of squares of a bf16 buffer, accumulated (fp32) into accum[0] — the building
// block for the global gradient L2 norm (grad clipping). Block-local reduction in
// shared memory, then one atomicAdd per block. Launch with block_dim = 256 (the
// shared array size). `accum` must be zeroed before the first launch of a step.
extern "C" __global__ void sumsq_kernel(
    const unsigned short* g, const size_t n, float* accum)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (i < n) { float x = b2f(g[i]); v = x * x; }
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(accum, sdata[0]);
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

/// A resident **fp32** vector in VRAM — the mixed-precision contract's *master*
/// storage: master weights and the AdamW moments (`m`, `v`) live here in full
/// precision, while their bf16 [`CudaMatrix`] views feed the Tensor-core GEMMs.
/// The optimizer update happens on these fp32 buffers; only the forward/backward
/// see bf16.
pub struct CudaF32 {
    buf: CudaSlice<f32>,
    len: usize,
}

impl CudaF32 {
    /// Element count.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    softmax_bwd: CudaFunction,
    swiglu_bwd: CudaFunction,
    rmsnorm_bwd: CudaFunction,
    rmsnorm_gain_bwd: CudaFunction,
    scale_mask_bwd: CudaFunction,
    rope_bwd: CudaFunction,
    embed_bwd: CudaFunction,
    ce_grad: CudaFunction,
    adamw: CudaFunction,
    sumsq: CudaFunction,
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
            softmax_bwd: f("softmax_bwd_kernel"),
            swiglu_bwd: f("swiglu_bwd_kernel"),
            rmsnorm_bwd: f("rmsnorm_bwd_kernel"),
            rmsnorm_gain_bwd: f("rmsnorm_gain_bwd_kernel"),
            scale_mask_bwd: f("scale_mask_bwd_kernel"),
            rope_bwd: f("rope_bwd_kernel"),
            embed_bwd: f("embed_bwd_kernel"),
            ce_grad: f("ce_grad_kernel"),
            adamw: f("adamw_kernel"),
            sumsq: f("sumsq_kernel"),
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

    /// Upload an fp32 vector to VRAM **without** rounding (master-weight / moment
    /// storage — see [`CudaF32`]).
    pub fn upload_f32(&self, data: &[f32]) -> CudaF32 {
        let buf = self.stream.clone_htod(data).expect("cuda htod f32");
        CudaF32 {
            buf,
            len: data.len(),
        }
    }

    /// Download an fp32 resident vector.
    pub fn download_f32(&self, x: &CudaF32) -> Vec<f32> {
        self.stream.clone_dtoh(&x.buf).expect("cuda dtoh f32")
    }

    /// A zero-filled fp32 resident vector of length `n` (fresh AdamW moments).
    pub fn zeros_f32(&self, n: usize) -> CudaF32 {
        let buf = self.stream.alloc_zeros::<f32>(n).expect("cuda alloc f32");
        CudaF32 { buf, len: n }
    }

    /// Round an fp32 master vector down to a bf16 [`CudaMatrix`] view of shape
    /// `rows × cols` (the Tensor-core input produced from the master weight).
    pub fn to_bf16(&self, x: &CudaF32, rows: usize, cols: usize) -> CudaMatrix {
        assert_eq!(x.len, rows * cols, "to_bf16: len != rows*cols");
        let host = self.download_f32(x);
        self.upload(&host, rows, cols)
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

    /// `C = Aᵀ · B` on Tensor cores: `a` is `k×m`, `b` is `k×n`, result `m×n`
    /// (row-major). The weight-gradient GEMM: with [`Self::matmul_bt`], the two give
    /// the full matmul VJP — `dA = dC·Bᵀ` (`matmul_bt(dC, B)`), `dB = Aᵀ·dC`
    /// (`matmul_at(A, dC)`).
    ///
    /// Column-major identity: `Cᵀ = Bᵀ·A`. My row-major `b` viewed column-major is
    /// `bᵀ`, used as-is; my row-major `a` viewed column-major is `aᵀ`, so
    /// `transb=true` recovers `a`. Hence `matmul(transa=false, transb=true, m=n,
    /// n=m, k)` over `(b, a)` yields row-major `Aᵀ·B`.
    pub fn matmul_at(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        let (k, m, n) = (a.rows, a.cols, b.cols);
        assert_eq!(
            b.rows, k,
            "matmul_at: outer dims disagree (({}x{})ᵀ · {}x{})",
            a.rows, a.cols, b.rows, b.cols
        );
        let mut c = self
            .stream
            .alloc_zeros::<bf16>(m * n)
            .expect("cuda alloc C");
        let cfg = MatmulConfig {
            transa: false,
            transb: true,
            transc: false,
            m: n as u64,
            n: m as u64,
            k: k as u64,
            alpha: 1.0,
            lda: n as i64,
            ldb: m as i64,
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
                .expect("cublasLt bf16 matmul_at");
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

    // ---- backward adjoints (B4) ----

    /// Softmax backward: given the forward output `y` and upstream grad `dy`,
    /// `dx = y ⊙ (dy − Σⱼ dyⱼyⱼ)` per row. One thread per row.
    pub fn softmax_backward(&self, y: &CudaMatrix, dy: &CudaMatrix) -> CudaMatrix {
        assert_eq!(
            (y.rows, y.cols),
            (dy.rows, dy.cols),
            "softmax_backward: y {}x{} vs dy {}x{}",
            y.rows,
            y.cols,
            dy.rows,
            dy.cols
        );
        let (rows, cols) = (y.rows, y.cols);
        let mut dx = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().softmax_bwd);
        builder.arg(&mut dx);
        builder.arg(&y.buf);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        // SAFETY: arg order/types match `softmax_bwd_kernel`; one thread per row.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(rows as u32))
                .expect("launch softmax_bwd_kernel");
        }
        CudaMatrix {
            buf: dx,
            rows,
            cols,
        }
    }

    /// SwiGLU backward of `c = silu(a) ⊙ b`: returns `(da, db)` with
    /// `da = dc·silu'(a)·b`, `db = dc·silu(a)`. Elementwise.
    pub fn swiglu_backward(
        &self,
        a: &CudaMatrix,
        b: &CudaMatrix,
        dc: &CudaMatrix,
    ) -> (CudaMatrix, CudaMatrix) {
        assert_eq!(
            (a.rows, a.cols),
            (b.rows, b.cols),
            "swiglu_backward: a/b shape"
        );
        assert_eq!(
            (a.rows, a.cols),
            (dc.rows, dc.cols),
            "swiglu_backward: a/dc shape"
        );
        let (rows, cols) = (a.rows, a.cols);
        let n = rows * cols;
        let mut da = self.stream.alloc_zeros::<bf16>(n).expect("cuda alloc");
        let mut db = self.stream.alloc_zeros::<bf16>(n).expect("cuda alloc");
        let n_a = n;
        let mut builder = self.stream.launch_builder(&self.kernels().swiglu_bwd);
        builder.arg(&mut da);
        builder.arg(&mut db);
        builder.arg(&a.buf);
        builder.arg(&b.buf);
        builder.arg(&dc.buf);
        builder.arg(&n_a);
        // SAFETY: arg order/types match `swiglu_bwd_kernel`; grid covers n.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("launch swiglu_bwd_kernel");
        }
        (
            CudaMatrix {
                buf: da,
                rows,
                cols,
            },
            CudaMatrix {
                buf: db,
                rows,
                cols,
            },
        )
    }

    /// RMSNorm **input** backward: `dx_j = dy_j·w_j/rms − x_j·(Σₖ dyₖwₖxₖ)/(cols·rms³)`
    /// per row, `rms = √(mean(x²)+eps)`. `weight` is a `cols`-length gain. One thread
    /// per row.
    pub fn rms_norm_backward(
        &self,
        x: &CudaMatrix,
        weight: &CudaMatrix,
        dy: &CudaMatrix,
        eps: f32,
    ) -> CudaMatrix {
        assert_eq!(
            (x.rows, x.cols),
            (dy.rows, dy.cols),
            "rms_norm_backward: x/dy shape"
        );
        assert_eq!(
            weight.rows * weight.cols,
            x.cols,
            "rms_norm_backward: weight {} elems, expected {}",
            weight.rows * weight.cols,
            x.cols
        );
        let (rows, cols) = (x.rows, x.cols);
        let mut dx = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a, eps_a) = (rows, cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels().rmsnorm_bwd);
        builder.arg(&mut dx);
        builder.arg(&x.buf);
        builder.arg(&weight.buf);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&eps_a);
        // SAFETY: arg order/types match `rmsnorm_bwd_kernel`; one thread per row.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(rows as u32))
                .expect("launch rmsnorm_bwd_kernel");
        }
        CudaMatrix {
            buf: dx,
            rows,
            cols,
        }
    }

    /// RMSNorm **gain** backward: `dw_j = Σ_r dy[r,j]·x[r,j]/rms[r]`; result is a
    /// `1×cols` row vector. One thread per column (each recomputes the per-row rms).
    pub fn rms_norm_gain_backward(&self, x: &CudaMatrix, dy: &CudaMatrix, eps: f32) -> CudaMatrix {
        assert_eq!(
            (x.rows, x.cols),
            (dy.rows, dy.cols),
            "rms_norm_gain_backward: x/dy shape"
        );
        let (rows, cols) = (x.rows, x.cols);
        let mut dw = self.stream.alloc_zeros::<bf16>(cols).expect("cuda alloc");
        let (rows_a, cols_a, eps_a) = (rows, cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels().rmsnorm_gain_bwd);
        builder.arg(&mut dw);
        builder.arg(&x.buf);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&eps_a);
        // SAFETY: arg order/types match `rmsnorm_gain_bwd_kernel`; one thread per col.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(cols as u32))
                .expect("launch rmsnorm_gain_bwd_kernel");
        }
        CudaMatrix {
            buf: dw,
            rows: 1,
            cols,
        }
    }

    /// Scale + causal-mask backward: `din = scale·dout` at kept positions, `0` above
    /// the diagonal (masked keys carry no gradient). Elementwise.
    pub fn scale_causal_mask_backward(
        &self,
        dout: &CudaMatrix,
        scale: f32,
        causal: bool,
    ) -> CudaMatrix {
        let (rows, cols) = (dout.rows, dout.cols);
        let total = rows * cols;
        let mut din = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc");
        let (rows_a, cols_a, scale_a) = (rows, cols, scale);
        let causal_a: i32 = causal as i32;
        let mut builder = self.stream.launch_builder(&self.kernels().scale_mask_bwd);
        builder.arg(&mut din);
        builder.arg(&dout.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&scale_a);
        builder.arg(&causal_a);
        // SAFETY: arg order/types match `scale_mask_bwd_kernel`; grid covers rows*cols.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch scale_mask_bwd_kernel");
        }
        CudaMatrix {
            buf: din,
            rows,
            cols,
        }
    }

    /// RoPE backward — the adjoint (transpose) rotation, same `pos`/`freq` as
    /// [`Self::rope`]. `dim` (cols) must be even. One thread per `(row, pair)`.
    pub fn rope_backward(
        &self,
        dy: &CudaMatrix,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> CudaMatrix {
        assert_eq!(
            dy.cols % 2,
            0,
            "rope_backward: dim must be even, got {}",
            dy.cols
        );
        let (rows, dim) = (dy.rows, dy.cols);
        let total = rows * (dim / 2);
        let mut dx = self
            .stream
            .alloc_zeros::<bf16>(rows * dim)
            .expect("cuda alloc");
        let (rows_a, dim_a, seq_a, off_a, theta_a) = (rows, dim, seq_len, offset, theta);
        let mut builder = self.stream.launch_builder(&self.kernels().rope_bwd);
        builder.arg(&mut dx);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&dim_a);
        builder.arg(&seq_a);
        builder.arg(&off_a);
        builder.arg(&theta_a);
        // SAFETY: arg order/types match `rope_bwd_kernel`; one thread per (row, pair).
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch rope_bwd_kernel");
        }
        CudaMatrix {
            buf: dx,
            rows,
            cols: dim,
        }
    }

    /// Embedding-gather backward (scatter-add): accumulate `dout` (`tokens.len()×d`)
    /// into a `vocab×d` table gradient — row `v` sums the `dout` rows whose (clamped)
    /// token id is `v`. Atomic-free (one thread owns each output element's reduction).
    pub fn embed_backward(&self, tokens: &[u32], dout: &CudaMatrix, vocab: usize) -> CudaMatrix {
        let (n, d) = (tokens.len(), dout.cols);
        assert_eq!(
            dout.rows, n,
            "embed_backward: dout rows {} != tokens {}",
            dout.rows, n
        );
        let toks = self.stream.clone_htod(tokens).expect("cuda htod tokens");
        let total = vocab * d;
        let mut dtable = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc");
        let (n_a, d_a, vocab_a) = (n, d, vocab);
        let mut builder = self.stream.launch_builder(&self.kernels().embed_bwd);
        builder.arg(&mut dtable);
        builder.arg(&toks);
        builder.arg(&dout.buf);
        builder.arg(&n_a);
        builder.arg(&d_a);
        builder.arg(&vocab_a);
        // SAFETY: arg order/types match `embed_bwd_kernel`; grid covers vocab*d.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch embed_bwd_kernel");
        }
        CudaMatrix {
            buf: dtable,
            rows: vocab,
            cols: d,
        }
    }

    /// Cross-entropy gradient w.r.t. the logits: `(softmax(logits) − onehot(target))
    /// / rows`, one row per target. The loss itself is computed host-side from the
    /// downloaded logits (as Route A does), so only the grad is resident here.
    pub fn cross_entropy_grad(&self, logits: &CudaMatrix, targets: &[u32]) -> CudaMatrix {
        let (rows, cols) = (logits.rows, logits.cols);
        assert_eq!(
            targets.len(),
            rows,
            "cross_entropy_grad: {} targets != {rows} rows",
            targets.len()
        );
        let tgt = self.stream.clone_htod(targets).expect("cuda htod targets");
        let mut d = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().ce_grad);
        builder.arg(&mut d);
        builder.arg(&logits.buf);
        builder.arg(&tgt);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        // SAFETY: arg order/types match `ce_grad_kernel`; one thread per row.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(rows as u32))
                .expect("launch ce_grad_kernel");
        }
        CudaMatrix { buf: d, rows, cols }
    }

    /// One in-place AdamW step at `step` (1-based) in the mixed-precision regime:
    /// fp32 master `param`, fp32 moments `m`/`v`, a bf16 `grad`. `grad_scale`
    /// multiplies the gradient before the moment update (the global-norm clip
    /// factor; pass `1.0` for no clipping). Updates `param`/`m`/`v` in fp32 and
    /// refreshes `param_bf16` (the bf16 view fed to the next forward/backward GEMM).
    /// `bc1`/`bc2` bias corrections are computed host-side. With `grad_scale = 1.0`
    /// this matches `cpu_adamw_step` exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &self,
        param: &mut CudaF32,
        m: &mut CudaF32,
        v: &mut CudaF32,
        grad: &CudaMatrix,
        param_bf16: &mut CudaMatrix,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        step: u32,
        grad_scale: f32,
    ) {
        let n = param.len;
        assert_eq!(m.len, n, "adamw_step: m len {} != param {n}", m.len);
        assert_eq!(v.len, n, "adamw_step: v len {} != param {n}", v.len);
        assert_eq!(
            grad.rows * grad.cols,
            n,
            "adamw_step: grad has {} elems != {n}",
            grad.rows * grad.cols
        );
        assert_eq!(
            param_bf16.rows * param_bf16.cols,
            n,
            "adamw_step: param_bf16 has {} elems != {n}",
            param_bf16.rows * param_bf16.cols
        );
        let (b1, b2) = betas;
        let bc1 = 1.0 - b1.powi(step as i32);
        let bc2 = 1.0 - b2.powi(step as i32);
        let (n_a, lr_a, b1_a, b2_a, eps_a, wd_a, bc1_a, bc2_a, gs_a) =
            (n, lr, b1, b2, eps, weight_decay, bc1, bc2, grad_scale);
        let mut builder = self.stream.launch_builder(&self.kernels().adamw);
        builder.arg(&mut param.buf);
        builder.arg(&mut m.buf);
        builder.arg(&mut v.buf);
        builder.arg(&grad.buf);
        builder.arg(&mut param_bf16.buf);
        builder.arg(&n_a);
        builder.arg(&lr_a);
        builder.arg(&b1_a);
        builder.arg(&b2_a);
        builder.arg(&eps_a);
        builder.arg(&wd_a);
        builder.arg(&bc1_a);
        builder.arg(&bc2_a);
        builder.arg(&gs_a);
        // SAFETY: arg order/types match `adamw_kernel`; grid covers n.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("launch adamw_kernel");
        }
    }

    /// The global L2 norm `sqrt(Σᵢ ‖gᵢ‖²)` over a set of gradient matrices — for
    /// gradient clipping. Each grad's sum-of-squares is reduced on-device (fp32) and
    /// accumulated into one scalar, downloaded once. The atomic accumulation order
    /// varies run-to-run, but only below the bf16 noise floor (the grads are already
    /// bf16), so this doesn't add meaningful non-determinism. Returns `+inf`/`NaN`
    /// faithfully if any grad is non-finite (so the caller can skip the step).
    pub fn global_grad_norm(&self, grads: &[&CudaMatrix]) -> f32 {
        let mut accum = self.stream.alloc_zeros::<f32>(1).expect("cuda alloc accum");
        for g in grads
        {
            let n = g.rows * g.cols;
            if n == 0
            {
                continue;
            }
            let block = 256u32;
            let grid = (n as u32).div_ceil(block);
            let n_a = n;
            let mut builder = self.stream.launch_builder(&self.kernels().sumsq);
            builder.arg(&g.buf);
            builder.arg(&n_a);
            builder.arg(&mut accum);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: arg order/types match `sumsq_kernel`; block_dim 256 = sdata size.
            unsafe {
                builder.launch(cfg).expect("launch sumsq_kernel");
            }
        }
        let host = self.stream.clone_dtoh(&accum).expect("cuda dtoh accum");
        host[0].sqrt()
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

    /// `matmul_at` computes `Aᵀ·B` (the weight-gradient GEMM) within bf16 tolerance.
    /// With `matmul_bt` this is the full matmul VJP: for `C = A·B`, `dA = dC·Bᵀ =
    /// matmul_bt(dC, B)` and `dB = Aᵀ·dC = matmul_at(A, dC)`. Skips with no device.
    #[test]
    fn bf16_matmul_at_matches_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping matmul_at parity");
            return;
        };
        // a: k×m, b: k×n, want aᵀ·b : m×n.
        let (k, m, n) = (6usize, 4usize, 5usize);
        let a: Vec<f32> = (0..k * m).map(|i| (i as f32 * 0.11 - 0.4).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07 + 0.3).cos()).collect();
        let got =
            chain.download(&chain.matmul_at(&chain.upload(&a, k, m), &chain.upload(&b, k, n)));
        // CPU aᵀ·b : out[i,j] = Σ_p a[p,i]·b[p,j].
        let mut want = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                want[i * n + j] = (0..k).map(|p| a[p * m + i] * b[p * n + j]).sum();
            }
        }
        let e = rel_err(&got, &want);
        assert!(
            e < 5e-2,
            "matmul_at rel_err {e}\n got {got:?}\n want {want:?}"
        );
        eprintln!("bf16 matmul_at (Aᵀ·B) vs CPU fp32: rel_err {e:.3e} — PASS");
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

    /// The B4 backward adjoints (`softmax_backward`, `swiglu_backward`,
    /// `rms_norm_backward`, `rms_norm_gain_backward`, `scale_causal_mask_backward`,
    /// `rope_backward`) each match their CPU fp32 oracle within bf16 tolerance —
    /// these are the VJP building blocks the CudaModel backward composes. Skips with
    /// no device.
    #[test]
    fn bf16_backward_adjoints_match_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping backward-adjoint parity");
            return;
        };
        let (rows, cols) = (4usize, 8usize);
        let n = rows * cols;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13 - 0.6).sin()).collect();
        let dy: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.09 + 0.2).cos() * 0.5)
            .collect();
        let gx = chain.upload(&x, rows, cols);
        let gdy = chain.upload(&dy, rows, cols);

        // softmax backward: dx = y ⊙ (dy − Σⱼ dyⱼyⱼ), y = softmax(x).
        let y = cpu_softmax(&x, rows, cols);
        let gy = chain.upload(&y, rows, cols);
        let sm_bwd = chain.download(&chain.softmax_backward(&gy, &gdy));
        let mut want_sm = vec![0.0f32; n];
        for r in 0..rows
        {
            let base = r * cols;
            let s: f32 = (0..cols).map(|j| dy[base + j] * y[base + j]).sum();
            for j in 0..cols
            {
                want_sm[base + j] = y[base + j] * (dy[base + j] - s);
            }
        }
        let e_sm = rel_err(&sm_bwd, &want_sm);
        assert!(e_sm < 5e-2, "softmax_backward rel_err {e_sm}");

        // SwiGLU backward of c = silu(a) ⊙ b, with a = x, b = another signal.
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17 + 0.4).sin()).collect();
        let gb = chain.upload(&b, rows, cols);
        let (da, db) = chain.swiglu_backward(&gx, &gb, &gdy);
        let (da, db) = (chain.download(&da), chain.download(&db));
        let mut want_da = vec![0.0f32; n];
        let mut want_db = vec![0.0f32; n];
        for i in 0..n
        {
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let silu = x[i] * sig;
            let dsilu = sig * (1.0 + x[i] * (1.0 - sig));
            want_da[i] = dy[i] * dsilu * b[i];
            want_db[i] = dy[i] * silu;
        }
        let e_da = rel_err(&da, &want_da);
        let e_db = rel_err(&db, &want_db);
        assert!(e_da < 5e-2, "swiglu_backward da rel_err {e_da}");
        assert!(e_db < 5e-2, "swiglu_backward db rel_err {e_db}");

        // RMSNorm input + gain backward.
        let eps = 1e-5f32;
        let w: Vec<f32> = (0..cols).map(|j| 0.5 + 0.1 * j as f32).collect();
        let gw = chain.upload(&w, 1, cols);
        let rn_dx = chain.download(&chain.rms_norm_backward(&gx, &gw, &gdy, eps));
        let rn_dw = chain.download(&chain.rms_norm_gain_backward(&gx, &gdy, eps));
        let mut want_dx = vec![0.0f32; n];
        let mut want_dw = vec![0.0f32; cols];
        for r in 0..rows
        {
            let base = r * cols;
            let ms = x[base..base + cols].iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps;
            let rms = ms.sqrt();
            let dot: f32 = (0..cols).map(|j| dy[base + j] * w[j] * x[base + j]).sum();
            let coef = dot / (cols as f32 * ms * rms);
            for j in 0..cols
            {
                want_dx[base + j] = dy[base + j] * w[j] / rms - x[base + j] * coef;
                want_dw[j] += dy[base + j] * x[base + j] / rms;
            }
        }
        let e_dx = rel_err(&rn_dx, &want_dx);
        let e_dw = rel_err(&rn_dw, &want_dw);
        assert!(e_dx < 5e-2, "rms_norm_backward dx rel_err {e_dx}");
        assert!(e_dw < 5e-2, "rms_norm_gain_backward dw rel_err {e_dw}");

        // Scale + causal-mask backward on a square t×t score grad.
        let t = 5usize;
        let scale = 0.5f32;
        let sg: Vec<f32> = (0..t * t).map(|i| (i as f32 * 0.23 - 1.0).cos()).collect();
        let smask_bwd = chain.download(&chain.scale_causal_mask_backward(
            &chain.upload(&sg, t, t),
            scale,
            true,
        ));
        let mut want_smask = vec![0.0f32; t * t];
        for i in 0..t
        {
            for j in 0..t
            {
                want_smask[i * t + j] = if j > i { 0.0 } else { sg[i * t + j] * scale };
            }
        }
        let e_smask = rel_err(&smask_bwd, &want_smask);
        assert!(
            e_smask < 5e-2,
            "scale_causal_mask_backward rel_err {e_smask}"
        );

        // RoPE backward — the transpose rotation.
        let (rrows, dim, seq_len, theta) = (6usize, 8usize, 6usize, 10_000.0f32);
        let rdy: Vec<f32> = (0..rrows * dim)
            .map(|i| (i as f32 * 0.15 - 0.3).sin())
            .collect();
        let rope_bwd = chain.download(&chain.rope_backward(
            &chain.upload(&rdy, rrows, dim),
            seq_len,
            0,
            theta,
        ));
        let mut want_rope = vec![0.0f32; rrows * dim];
        for r in 0..rrows
        {
            let base = r * dim;
            let pos = (r % seq_len) as f32;
            for p in 0..dim / 2
            {
                let freq = theta.powf(-2.0 * p as f32 / dim as f32);
                let (s, c) = (pos * freq).sin_cos();
                let ge = rdy[base + 2 * p];
                let go = rdy[base + 2 * p + 1];
                want_rope[base + 2 * p] = c * ge + s * go;
                want_rope[base + 2 * p + 1] = -s * ge + c * go;
            }
        }
        let e_rope = rel_err(&rope_bwd, &want_rope);
        assert!(e_rope < 5e-2, "rope_backward rel_err {e_rope}");

        eprintln!(
            "bf16 backward adjoints vs CPU — softmax {e_sm:.2e}, swiglu(da {e_da:.2e}, db {e_db:.2e}), \
             rms(dx {e_dx:.2e}, dw {e_dw:.2e}), scale_mask {e_smask:.2e}, rope {e_rope:.2e} — PASS"
        );
    }

    /// The loss-adjacent backward kernels — `embed_backward` (scatter-add table
    /// gradient) and `cross_entropy_grad` ((softmax − onehot)/rows) — match their CPU
    /// oracles within bf16 tolerance. These close the two ends of the backward chain.
    /// Skips with no device.
    #[test]
    fn bf16_embed_and_ce_grad_backward_match_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping embed/ce-grad backward parity");
            return;
        };

        // embed_backward: scatter-add dout rows into their token's table row.
        let (vocab, d) = (6usize, 4usize);
        let tokens = [3u32, 0, 5, 3, 0, 3];
        let n = tokens.len();
        let dout: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.21 - 0.5).sin()).collect();
        let dtable =
            chain.download(&chain.embed_backward(&tokens, &chain.upload(&dout, n, d), vocab));
        let mut want_dt = vec![0.0f32; vocab * d];
        for (i, &t) in tokens.iter().enumerate()
        {
            let v = (t as usize).min(vocab - 1);
            for c in 0..d
            {
                want_dt[v * d + c] += dout[i * d + c];
            }
        }
        let e_embed = rel_err(&dtable, &want_dt);
        assert!(e_embed < 5e-2, "embed_backward rel_err {e_embed}");

        // cross_entropy_grad: (softmax(logits) − onehot(target)) / rows.
        let (rows, cols) = (4usize, 7usize);
        let logits: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.17 - 1.0).cos())
            .collect();
        let targets = [2u32, 6, 0, 5];
        let ce =
            chain.download(&chain.cross_entropy_grad(&chain.upload(&logits, rows, cols), &targets));
        let mut want_ce = cpu_softmax(&logits, rows, cols);
        let inv = 1.0f32 / rows as f32;
        for (i, &t) in targets.iter().enumerate()
        {
            want_ce[i * cols + (t as usize).min(cols - 1)] -= 1.0;
        }
        for v in want_ce.iter_mut()
        {
            *v *= inv;
        }
        let e_ce = rel_err(&ce, &want_ce);
        assert!(e_ce < 5e-2, "cross_entropy_grad rel_err {e_ce}");

        eprintln!(
            "bf16 embed_backward {e_embed:.2e} + cross_entropy_grad {e_ce:.2e} vs CPU — PASS"
        );
    }

    /// The mixed-precision AdamW step updates fp32 master `param`/`m`/`v` exactly like
    /// `cpu_adamw_step` (fed the same bf16-rounded grad) and refreshes the bf16 view.
    /// The fp32 master path is checked tightly (fp32 arithmetic, rel_err ~1e-3); the
    /// refreshed bf16 view at bf16 tolerance. Skips with no device.
    #[test]
    fn adamw_step_matches_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping adamw parity");
            return;
        };
        let n = 32usize;
        let param0: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.13 - 0.5).sin() * 0.2)
            .collect();
        let grad_f: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.29 + 0.1).cos() * 0.05)
            .collect();
        let (lr, betas, eps, wd) = (1e-3f32, (0.9f32, 0.999f32), 1e-8f32, 0.01f32);

        // Resident state: fp32 master + moments, bf16 grad + bf16 view.
        let mut param = chain.upload_f32(&param0);
        let mut m = chain.zeros_f32(n);
        let mut v = chain.zeros_f32(n);
        let ggrad = chain.upload(&grad_f, 1, n);
        // The grad the kernel actually sees is bf16-rounded — feed the same to the
        // CPU oracle so only the AdamW arithmetic is under test.
        let grad_bf = chain.download(&ggrad);
        let mut param_bf = chain.upload(&param0, 1, n);

        // Two steps so bias correction (bc1/bc2) is exercised at step ≥ 2.
        let mut cpu_p = param0.clone();
        let mut cpu_m = vec![0.0f32; n];
        let mut cpu_v = vec![0.0f32; n];
        for step in 1..=2u32
        {
            chain.adamw_step(
                &mut param,
                &mut m,
                &mut v,
                &ggrad,
                &mut param_bf,
                lr,
                betas,
                eps,
                wd,
                step,
                1.0, // grad_scale = 1.0 → no clipping, matches cpu_adamw_step
            );
            // cpu_adamw_step, inlined (same as scirust_gpu::ops::cpu_adamw_step).
            let (b1, b2) = betas;
            let bc1 = 1.0 - b1.powi(step as i32);
            let bc2 = 1.0 - b2.powi(step as i32);
            for i in 0..n
            {
                let g = grad_bf[i];
                cpu_m[i] = b1 * cpu_m[i] + (1.0 - b1) * g;
                cpu_v[i] = b2 * cpu_v[i] + (1.0 - b2) * g * g;
                let mhat = cpu_m[i] / bc1;
                let vhat = cpu_v[i] / bc2;
                cpu_p[i] -= lr * (mhat / (vhat.sqrt() + eps) + wd * cpu_p[i]);
            }
        }

        let got_p = chain.download_f32(&param);
        let got_m = chain.download_f32(&m);
        let got_v = chain.download_f32(&v);
        let e_p = rel_err(&got_p, &cpu_p);
        let e_m = rel_err(&got_m, &cpu_m);
        let e_v = rel_err(&got_v, &cpu_v);
        assert!(e_p < 1e-3, "adamw param rel_err {e_p}");
        assert!(e_m < 1e-3, "adamw m rel_err {e_m}");
        assert!(e_v < 1e-3, "adamw v rel_err {e_v}");
        // The refreshed bf16 view must equal the fp32 master, rounded.
        let e_bf = rel_err(&chain.download(&param_bf), &cpu_p);
        assert!(e_bf < 5e-2, "adamw bf16 view rel_err {e_bf}");
        eprintln!(
            "adamw fp32 master (param {e_p:.2e}, m {e_m:.2e}, v {e_v:.2e}) + bf16 view {e_bf:.2e} vs CPU — PASS"
        );
    }

    /// `global_grad_norm` computes `sqrt(Σ ‖gᵢ‖²)` over several matrices, matching a
    /// CPU fp32 reference (over the bf16-rounded values). The clip building block.
    /// Skips with no device.
    #[test]
    fn global_grad_norm_matches_cpu() {
        let Some(chain) = CudaChain::new()
        else
        {
            eprintln!("cuda: no device, skipping grad-norm parity");
            return;
        };
        // A few grads of different shapes (like a block's weights).
        let shapes = [(7usize, 5usize), (32, 1), (3, 40), (256, 1)];
        let mats: Vec<(Vec<f32>, usize, usize)> = shapes
            .iter()
            .enumerate()
            .map(|(k, &(r, c))| {
                let data: Vec<f32> = (0..r * c)
                    .map(|i| ((i + k) as f32 * 0.037 - 0.5).sin() * 0.3)
                    .collect();
                (data, r, c)
            })
            .collect();
        let uploaded: Vec<CudaMatrix> = mats
            .iter()
            .map(|(d, r, c)| chain.upload(d, *r, *c))
            .collect();
        let refs: Vec<&CudaMatrix> = uploaded.iter().collect();
        let got = chain.global_grad_norm(&refs);
        // CPU reference over the bf16-rounded values (what the kernel sees).
        let mut ss = 0.0f64;
        for (d, _, _) in &mats
        {
            for &x in d
            {
                let bx = bf16::from_f32(x).to_f32();
                ss += (bx as f64) * (bx as f64);
            }
        }
        let want = ss.sqrt() as f32;
        let rel = (got - want).abs() / want.max(1e-30);
        assert!(
            rel < 5e-2,
            "global_grad_norm {got} vs cpu {want} (rel {rel})"
        );
        eprintln!("global_grad_norm {got:.4} vs CPU {want:.4} — rel {rel:.2e} — PASS");
    }
}
