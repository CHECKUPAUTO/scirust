//! Batch-one CUDA decode runtime for latency-sensitive autoregressive inference.
//!
//! This module is deliberately isolated from [`crate::CudaChain`], which remains the
//! Route-B training/parity implementation.  The decode runtime is allowed to fuse
//! operations aggressively without changing the kernels used by an active training
//! run.  Its main primitive combines head-local RoPE, fixed-capacity KV-cache writes,
//! single-query GQA score/softmax/context, and head assembly in one CUDA launch.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

const DECODE_KERNELS_SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int s = __float_as_uint(f);
    unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);
    return (unsigned short)((s + bias) >> 16);
}

extern "C" __global__ void decode_add_kernel(
    unsigned short* out, const unsigned short* a, const unsigned short* b, const size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2b(b2f(a[i]) + b2f(b[i]));
}

extern "C" __global__ void decode_embed_token_kernel(
    unsigned short* out, const unsigned short* table,
    const size_t token, const size_t vocab, const size_t d)
{
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d) {
        const size_t row = token < vocab ? token : vocab - 1;
        out[c] = table[row * d + c];
    }
}

extern "C" __global__ void decode_rmsnorm_kernel(
    unsigned short* out, const unsigned short* x, const unsigned short* w,
    const size_t cols, const float eps)
{
    __shared__ float red[256];
    const unsigned int tid = threadIdx.x;
    float ss = 0.0f;
    for (size_t c = tid; c < cols; c += blockDim.x) {
        const float v = b2f(x[c]);
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)cols + eps);
    for (size_t c = tid; c < cols; c += blockDim.x)
        out[c] = f2b(b2f(x[c]) * inv * b2f(w[c]));
}

// Input is [gate | up], each d_ff wide.  Keeping the two projections adjacent lets
// one GEMM replace the historical gate+up pair before this elementwise fusion.
extern "C" __global__ void decode_swiglu_split_kernel(
    unsigned short* out, const unsigned short* gate_up, const size_t d_ff)
{
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d_ff) {
        const float g = b2f(gate_up[c]);
        const float u = b2f(gate_up[d_ff + c]);
        const float silu = g / (1.0f + __expf(-g));
        out[c] = f2b(silu * u);
    }
}

// One block owns one query head.  The launch fuses the entire incremental GQA
// attention path:
//   raw Q/K/V row -> head-local RoPE -> fixed KV append -> scores -> scale ->
//   softmax -> deterministic left-to-right context -> assembled d_model row.
// Historical K/V are already bf16 in the fixed caches.  The current K/V are read
// directly from qkv so no inter-block synchronization is required after the one
// designated writer per KV head stores them for the *next* token.
extern "C" __global__ void decode_gqa_kernel(
    unsigned short* out,
    const unsigned short* qkv,
    unsigned short* kcache,
    unsigned short* vcache,
    const size_t pos,
    const size_t capacity,
    const size_t d_model,
    const size_t kv_dim,
    const size_t n_heads,
    const size_t n_kv_heads,
    const float theta,
    const float scale)
{
    const size_t head = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= n_heads || pos >= capacity) return;

    const size_t dh = d_model / n_heads;
    const size_t repeat = n_heads / n_kv_heads;
    const size_t kv = head / repeat;
    const size_t seq = pos + 1;

    extern __shared__ unsigned char smem_raw[];
    float* scores = (float*)smem_raw;              // seq floats
    float* qrot = scores + seq;                    // dh floats
    float* kcur = qrot + dh;                       // dh floats
    float* red = kcur + dh;                        // 256 floats

    // RoPE frequency restarts in every logical head, matching semantics-v2.
    const size_t pairs = dh / 2;
    for (size_t p = tid; p < pairs; p += blockDim.x) {
        const float freq = powf(theta, -2.0f * (float)p / (float)dh);
        const float angle = (float)pos * freq;
        const float co = cosf(angle);
        const float si = sinf(angle);

        const size_t qc = head * dh + 2 * p;
        const float q0 = b2f(qkv[qc]);
        const float q1 = b2f(qkv[qc + 1]);
        qrot[2 * p] = b2f(f2b(q0 * co - q1 * si));
        qrot[2 * p + 1] = b2f(f2b(q0 * si + q1 * co));

        const size_t kc = d_model + kv * dh + 2 * p;
        const float k0 = b2f(qkv[kc]);
        const float k1 = b2f(qkv[kc + 1]);
        kcur[2 * p] = b2f(f2b(k0 * co - k1 * si));
        kcur[2 * p + 1] = b2f(f2b(k0 * si + k1 * co));
    }
    __syncthreads();

    // Exactly one query-head block writes each KV-head slice.  Other query heads in
    // the GQA group use kcur/raw-V for the current position and only need the cache
    // from the next launch onward.
    if ((head % repeat) == 0) {
        for (size_t c = tid; c < dh; c += blockDim.x) {
            const size_t dst = pos * kv_dim + kv * dh + c;
            kcache[dst] = f2b(kcur[c]);
            vcache[dst] = qkv[d_model + kv_dim + kv * dh + c];
        }
    }

    // QK^T for one query.  Preserve the two bf16 round boundaries used by the
    // legacy path: GEMM output first, then scaled-score output before softmax.
    for (size_t j = tid; j < seq; j += blockDim.x) {
        float acc = 0.0f;
        for (size_t c = 0; c < dh; ++c) {
            const float kval = j == pos
                ? kcur[c]
                : b2f(kcache[j * kv_dim + kv * dh + c]);
            acc += qrot[c] * kval;
        }
        const float score_bf = b2f(f2b(acc));
        scores[j] = b2f(f2b(score_bf * scale));
    }
    __syncthreads();

    // Same 256-way max/sum reduction shape as the Route-B fast softmax kernel.
    float mx = -3.0e38f;
    for (size_t j = tid; j < seq; j += blockDim.x)
        mx = fmaxf(mx, scores[j]);
    red[tid] = mx;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    mx = red[0];

    float sum = 0.0f;
    for (size_t j = tid; j < seq; j += blockDim.x)
        sum += __expf(scores[j] - mx);
    red[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    sum = red[0];

    // Once every score has been consumed by the reduction, reuse the score storage
    // for bf16 probabilities.  This makes the context accumulation see the same
    // bf16 boundary as the legacy softmax -> attention_context pipeline.
    unsigned short* probs = (unsigned short*)scores;
    for (size_t j = tid; j < seq; j += blockDim.x)
        probs[j] = f2b(__expf(scores[j] - mx) / sum);
    __syncthreads();

    // B49 parity rule: every output channel owns a strict left-to-right fp32
    // accumulation over sequence positions, followed by one bf16 rounding.
    for (size_t c = tid; c < dh; c += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < seq; ++j) {
            const float vv = j == pos
                ? b2f(qkv[d_model + kv_dim + kv * dh + c])
                : b2f(vcache[j * kv_dim + kv * dh + c]);
            acc += b2f(probs[j]) * vv;
        }
        out[head * dh + c] = f2b(acc);
    }
}
"#;

/// Device-resident bf16 row-major matrix used only by the decode runtime.
pub struct CudaDecodeMatrix {
    buf: CudaSlice<bf16>,
    rows: usize,
    cols: usize,
}

impl CudaDecodeMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Fixed-capacity resident KV storage.  `pos` is supplied by the caller; no cache
/// reallocation or historical-row copy occurs during decode.
pub struct CudaDecodeKvCache {
    buf: CudaSlice<bf16>,
    capacity: usize,
    cols: usize,
}

impl CudaDecodeKvCache {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

struct DecodeKernels {
    add: CudaFunction,
    embed_token: CudaFunction,
    rmsnorm: CudaFunction,
    swiglu_split: CudaFunction,
    gqa: CudaFunction,
}

/// Dedicated CUDA runtime for one-token autoregressive decode.
pub struct CudaDecodeRuntime {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    kernels: DecodeKernels,
}

impl CudaDecodeRuntime {
    pub fn new() -> Option<Self> {
        if !cuda_libraries_available() || !nvrtc_available() {
            return None;
        }
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone()).ok()?;
        let ptx = compile_ptx(DECODE_KERNELS_SRC)
            .map_err(|e| eprintln!("scirust-cuda decode: NVRTC compile failed: {e}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| eprintln!("scirust-cuda decode: module load failed: {e}"))
            .ok()?;
        let f = |name: &str| module.load_function(name).expect("load decode kernel");
        let kernels = DecodeKernels {
            add: f("decode_add_kernel"),
            embed_token: f("decode_embed_token_kernel"),
            rmsnorm: f("decode_rmsnorm_kernel"),
            swiglu_split: f("decode_swiglu_split_kernel"),
            gqa: f("decode_gqa_kernel"),
        };
        Some(Self {
            _ctx: ctx,
            stream,
            blas,
            kernels,
        })
    }

    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> CudaDecodeMatrix {
        assert_eq!(data.len(), rows * cols, "decode upload shape mismatch");
        let bf: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
        let buf = self.stream.clone_htod(&bf).expect("decode CUDA upload");
        CudaDecodeMatrix { buf, rows, cols }
    }

    pub fn kv_cache(&self, capacity: usize, cols: usize) -> CudaDecodeKvCache {
        assert!(capacity > 0 && cols > 0, "decode KV cache must be non-empty");
        let buf = self
            .stream
            .alloc_zeros::<bf16>(capacity * cols)
            .expect("decode CUDA KV allocation");
        CudaDecodeKvCache {
            buf,
            capacity,
            cols,
        }
    }

    pub fn download(&self, matrix: &CudaDecodeMatrix) -> Vec<f32> {
        let host: Vec<bf16> = self
            .stream
            .clone_dtoh(&matrix.buf)
            .expect("decode CUDA download");
        host.iter().map(|x| x.to_f32()).collect()
    }

    pub fn embed_token(&self, token: u32, table: &CudaDecodeMatrix) -> CudaDecodeMatrix {
        assert!(table.rows > 0 && table.cols > 0, "decode embedding table empty");
        let d = table.cols;
        let mut out = self.stream.alloc_zeros::<bf16>(d).expect("decode embed alloc");
        let (token_a, vocab_a, d_a) = (token as usize, table.rows, d);
        let mut builder = self.stream.launch_builder(&self.kernels.embed_token);
        builder.arg(&mut out);
        builder.arg(&table.buf);
        builder.arg(&token_a);
        builder.arg(&vocab_a);
        builder.arg(&d_a);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(d as u32))
                .expect("decode embed launch");
        }
        CudaDecodeMatrix {
            buf: out,
            rows: 1,
            cols: d,
        }
    }

    pub fn add(&self, a: &CudaDecodeMatrix, b: &CudaDecodeMatrix) -> CudaDecodeMatrix {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols), "decode add shape mismatch");
        let n = a.rows * a.cols;
        let mut out = self.stream.alloc_zeros::<bf16>(n).expect("decode add alloc");
        let n_a = n;
        let mut builder = self.stream.launch_builder(&self.kernels.add);
        builder.arg(&mut out);
        builder.arg(&a.buf);
        builder.arg(&b.buf);
        builder.arg(&n_a);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("decode add launch");
        }
        CudaDecodeMatrix {
            buf: out,
            rows: a.rows,
            cols: a.cols,
        }
    }

    pub fn rms_norm(
        &self,
        x: &CudaDecodeMatrix,
        weight: &CudaDecodeMatrix,
        eps: f32,
    ) -> CudaDecodeMatrix {
        assert_eq!(x.rows, 1, "decode RMSNorm is batch-one only");
        assert_eq!(weight.rows * weight.cols, x.cols, "decode RMSNorm weight shape");
        let cols = x.cols;
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(cols)
            .expect("decode RMSNorm alloc");
        let (cols_a, eps_a) = (cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels.rmsnorm);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&weight.buf);
        builder.arg(&cols_a);
        builder.arg(&eps_a);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg).expect("decode RMSNorm launch") };
        CudaDecodeMatrix {
            buf: out,
            rows: 1,
            cols,
        }
    }

    pub fn swiglu_split(&self, gate_up: &CudaDecodeMatrix) -> CudaDecodeMatrix {
        assert_eq!(gate_up.rows, 1, "decode SwiGLU is batch-one only");
        assert_eq!(gate_up.cols % 2, 0, "decode gate/up width must be even");
        let d_ff = gate_up.cols / 2;
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(d_ff)
            .expect("decode SwiGLU alloc");
        let d_ff_a = d_ff;
        let mut builder = self.stream.launch_builder(&self.kernels.swiglu_split);
        builder.arg(&mut out);
        builder.arg(&gate_up.buf);
        builder.arg(&d_ff_a);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(d_ff as u32))
                .expect("decode SwiGLU launch");
        }
        CudaDecodeMatrix {
            buf: out,
            rows: 1,
            cols: d_ff,
        }
    }

    pub fn matmul(&self, a: &CudaDecodeMatrix, b: &CudaDecodeMatrix) -> CudaDecodeMatrix {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        assert_eq!(b.rows, k, "decode matmul inner dimensions");
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(m * n)
            .expect("decode matmul alloc");
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
        unsafe {
            self.blas
                .matmul(cfg, &b.buf, &a.buf, &mut out, None, None)
                .expect("decode cuBLASLt matmul");
        }
        CudaDecodeMatrix {
            buf: out,
            rows: m,
            cols: n,
        }
    }

    pub fn matmul_bt(&self, a: &CudaDecodeMatrix, b: &CudaDecodeMatrix) -> CudaDecodeMatrix {
        let (m, k, n) = (a.rows, a.cols, b.rows);
        assert_eq!(b.cols, k, "decode matmul_bt inner dimensions");
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(m * n)
            .expect("decode matmul_bt alloc");
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
        unsafe {
            self.blas
                .matmul(cfg, &b.buf, &a.buf, &mut out, None, None)
                .expect("decode cuBLASLt matmul_bt");
        }
        CudaDecodeMatrix {
            buf: out,
            rows: m,
            cols: n,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gqa_decode(
        &self,
        qkv: &CudaDecodeMatrix,
        kcache: &mut CudaDecodeKvCache,
        vcache: &mut CudaDecodeKvCache,
        pos: usize,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        theta: f32,
    ) -> CudaDecodeMatrix {
        assert_eq!(qkv.rows, 1, "decode GQA is single-query only");
        assert!(n_heads > 0 && n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads));
        assert!(d_model.is_multiple_of(n_heads));
        let dh = d_model / n_heads;
        let kv_dim = n_kv_heads * dh;
        assert_eq!(qkv.cols, d_model + 2 * kv_dim, "decode fused QKV width");
        assert_eq!(kcache.cols, kv_dim, "decode K-cache width");
        assert_eq!(vcache.cols, kv_dim, "decode V-cache width");
        assert_eq!(kcache.capacity, vcache.capacity, "decode KV capacity mismatch");
        assert!(pos < kcache.capacity, "decode position exceeds KV capacity");

        let mut out = self
            .stream
            .alloc_zeros::<bf16>(d_model)
            .expect("decode GQA output alloc");
        let capacity = kcache.capacity;
        let (pos_a, cap_a, d_a, kv_a, nh_a, nkv_a, theta_a) = (
            pos,
            capacity,
            d_model,
            kv_dim,
            n_heads,
            n_kv_heads,
            theta,
        );
        let scale_a = 1.0f32 / (dh as f32).sqrt();
        let mut builder = self.stream.launch_builder(&self.kernels.gqa);
        builder.arg(&mut out);
        builder.arg(&qkv.buf);
        builder.arg(&mut kcache.buf);
        builder.arg(&mut vcache.buf);
        builder.arg(&pos_a);
        builder.arg(&cap_a);
        builder.arg(&d_a);
        builder.arg(&kv_a);
        builder.arg(&nh_a);
        builder.arg(&nkv_a);
        builder.arg(&theta_a);
        builder.arg(&scale_a);

        // scores + rotated Q + current rotated K + 256-float reduction scratch.
        let seq = pos + 1;
        let shared_floats = seq + 2 * dh + 256;
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32,
        };
        unsafe { builder.launch(cfg).expect("decode fused GQA launch") };
        CudaDecodeMatrix {
            buf: out,
            rows: 1,
            cols: d_model,
        }
    }
}

fn cuda_libraries_available() -> bool {
    unsafe { cudarc::driver::sys::is_culib_present() && cudarc::cublaslt::sys::is_culib_present() }
}

fn nvrtc_available() -> bool {
    unsafe { cudarc::nvrtc::sys::is_culib_present() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_kernel_source_compiles_when_nvrtc_is_available() {
        if !nvrtc_available() {
            eprintln!("cuda decode: NVRTC unavailable, skipping compile test");
            return;
        }
        compile_ptx(DECODE_KERNELS_SRC).expect("decode NVRTC kernel compilation");
    }
}
