//! Reconstruction-free batch-one ElasticKV CUDA decode runtime.
//!
//! Unlike the dense I250-A runtime, this path receives Q/K/V already projected into
//! latent coordinates. Historical K/V stay latent, attention is evaluated directly
//! in that space, and the caller applies an absorbed latent-to-model output matrix.
//! The initial kernel supports only native complete RoPE-pair coordinates, including
//! full identity. General learned bases require a projected rotary operator and are
//! deliberately rejected by the higher-level SCIAGENT planner until implemented.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

const ELASTIC_DECODE_KERNELS_SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int s = __float_as_uint(f);
    unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);
    return (unsigned short)((s + bias) >> 16);
}

extern "C" __global__ void elastic_add_kernel(
    unsigned short* out, const unsigned short* a, const unsigned short* b, const size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2b(b2f(a[i]) + b2f(b[i]));
}

extern "C" __global__ void elastic_embed_token_kernel(
    unsigned short* out, const unsigned short* table,
    const size_t token, const size_t vocab, const size_t d)
{
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d) {
        const size_t row = token < vocab ? token : vocab - 1;
        out[c] = table[row * d + c];
    }
}

extern "C" __global__ void elastic_rmsnorm_kernel(
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

extern "C" __global__ void elastic_swiglu_split_kernel(
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

// qkv layout:
//   [n_heads * key_rank | n_kv_heads * key_rank | n_kv_heads * value_rank]
//
// One block owns one query head. Q/K coordinates are a prefix of complete native
// RoPE pairs. Frequencies therefore use the ORIGINAL dense d_head denominator, not
// key_rank. This preserves semantics-v2 inside the retained subspace.
extern "C" __global__ void elastic_gqa_kernel(
    unsigned short* out_latent,
    const unsigned short* qkv,
    unsigned short* kcache,
    unsigned short* vcache,
    const size_t pos,
    const size_t capacity,
    const size_t dense_d_head,
    const size_t key_rank,
    const size_t value_rank,
    const size_t n_heads,
    const size_t n_kv_heads,
    const float theta,
    const float scale)
{
    const size_t head = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= n_heads || pos >= capacity) return;

    const size_t repeat = n_heads / n_kv_heads;
    const size_t kv = head / repeat;
    const size_t seq = pos + 1;
    const size_t q_width = n_heads * key_rank;
    const size_t k_width = n_kv_heads * key_rank;
    const size_t v_width = n_kv_heads * value_rank;
    const size_t k_offset = q_width;
    const size_t v_offset = q_width + k_width;

    extern __shared__ unsigned char smem_raw[];
    float* scores = (float*)smem_raw;
    float* qrot = scores + seq;
    float* kcur = qrot + key_rank;
    float* red = kcur + key_rank;

    const size_t pairs = key_rank / 2;
    for (size_t p = tid; p < pairs; p += blockDim.x) {
        const float freq = powf(theta, -2.0f * (float)p / (float)dense_d_head);
        const float angle = (float)pos * freq;
        const float co = cosf(angle);
        const float si = sinf(angle);

        const size_t qc = head * key_rank + 2 * p;
        const float q0 = b2f(qkv[qc]);
        const float q1 = b2f(qkv[qc + 1]);
        qrot[2 * p] = b2f(f2b(q0 * co - q1 * si));
        qrot[2 * p + 1] = b2f(f2b(q0 * si + q1 * co));

        const size_t kc = k_offset + kv * key_rank + 2 * p;
        const float k0 = b2f(qkv[kc]);
        const float k1 = b2f(qkv[kc + 1]);
        kcur[2 * p] = b2f(f2b(k0 * co - k1 * si));
        kcur[2 * p + 1] = b2f(f2b(k0 * si + k1 * co));
    }
    __syncthreads();

    if ((head % repeat) == 0) {
        for (size_t c = tid; c < key_rank; c += blockDim.x)
            kcache[pos * k_width + kv * key_rank + c] = f2b(kcur[c]);
        for (size_t c = tid; c < value_rank; c += blockDim.x)
            vcache[pos * v_width + kv * value_rank + c]
                = qkv[v_offset + kv * value_rank + c];
    }

    // Reconstruction-free key scoring. Preserve the legacy explicit bf16 score
    // boundary followed by a second bf16 boundary after scaling.
    for (size_t j = tid; j < seq; j += blockDim.x) {
        float acc = 0.0f;
        for (size_t c = 0; c < key_rank; ++c) {
            const float kval = j == pos
                ? kcur[c]
                : b2f(kcache[j * k_width + kv * key_rank + c]);
            acc += qrot[c] * kval;
        }
        const float score_bf = b2f(f2b(acc));
        scores[j] = b2f(f2b(score_bf * scale));
    }
    __syncthreads();

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

    for (size_t j = tid; j < seq; j += blockDim.x)
        scores[j] = b2f(f2b(__expf(scores[j] - mx) / sum));
    __syncthreads();

    // Reconstruction-free value aggregation. One thread owns one latent value
    // channel and preserves B49's strict left-to-right sequence accumulation.
    for (size_t c = tid; c < value_rank; c += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < seq; ++j) {
            const float vv = j == pos
                ? b2f(qkv[v_offset + kv * value_rank + c])
                : b2f(vcache[j * v_width + kv * value_rank + c]);
            acc += scores[j] * vv;
        }
        out_latent[head * value_rank + c] = f2b(acc);
    }
}
"#;

/// Device-resident bf16 matrix for the Elastic decode runtime.
pub struct CudaElasticMatrix {
    buf: CudaSlice<bf16>,
    rows: usize,
    cols: usize,
}

impl CudaElasticMatrix {
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

/// Fixed resident latent coefficient ring for one packed set of KV heads.
pub struct CudaElasticKvCache {
    buf: CudaSlice<bf16>,
    capacity: usize,
    cols: usize,
}

impl CudaElasticKvCache {
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

struct ElasticDecodeKernels {
    add: CudaFunction,
    embed_token: CudaFunction,
    rmsnorm: CudaFunction,
    swiglu_split: CudaFunction,
    gqa: CudaFunction,
}

/// Dedicated CUDA stream/runtime for reconstruction-free ElasticKV decode.
pub struct CudaElasticDecodeRuntime {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    kernels: ElasticDecodeKernels,
}

impl CudaElasticDecodeRuntime {
    #[must_use]
    pub fn new() -> Option<Self> {
        if !cuda_libraries_available() || !nvrtc_available()
        {
            return None;
        }
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone()).ok()?;
        let ptx = compile_ptx(ELASTIC_DECODE_KERNELS_SRC)
            .map_err(|error| eprintln!("scirust-cuda elastic decode: NVRTC failed: {error}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| eprintln!("scirust-cuda elastic decode: module load failed: {error}"))
            .ok()?;
        let function = |name: &str| module.load_function(name).expect("load elastic decode kernel");
        let kernels = ElasticDecodeKernels {
            add: function("elastic_add_kernel"),
            embed_token: function("elastic_embed_token_kernel"),
            rmsnorm: function("elastic_rmsnorm_kernel"),
            swiglu_split: function("elastic_swiglu_split_kernel"),
            gqa: function("elastic_gqa_kernel"),
        };
        Some(Self {
            _ctx: ctx,
            stream,
            blas,
            kernels,
        })
    }

    #[must_use]
    pub fn matrix(&self, rows: usize, cols: usize) -> CudaElasticMatrix {
        assert!(rows > 0 && cols > 0, "elastic decode matrix must be non-empty");
        let buf = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("elastic decode CUDA matrix allocation");
        CudaElasticMatrix { buf, rows, cols }
    }

    #[must_use]
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> CudaElasticMatrix {
        assert_eq!(data.len(), rows * cols, "elastic decode upload shape mismatch");
        let bf: Vec<bf16> = data.iter().map(|&value| bf16::from_f32(value)).collect();
        let buf = self
            .stream
            .clone_htod(&bf)
            .expect("elastic decode CUDA upload");
        CudaElasticMatrix { buf, rows, cols }
    }

    #[must_use]
    pub fn kv_cache(&self, capacity: usize, cols: usize) -> CudaElasticKvCache {
        assert!(capacity > 0 && cols > 0, "elastic KV cache must be non-empty");
        let buf = self
            .stream
            .alloc_zeros::<bf16>(capacity * cols)
            .expect("elastic decode CUDA KV allocation");
        CudaElasticKvCache {
            buf,
            capacity,
            cols,
        }
    }

    #[must_use]
    pub fn download(&self, matrix: &CudaElasticMatrix) -> Vec<f32> {
        let host: Vec<bf16> = self
            .stream
            .clone_dtoh(&matrix.buf)
            .expect("elastic decode CUDA download");
        host.iter().map(|value| value.to_f32()).collect()
    }

    pub fn embed_token_into(
        &self,
        token: u32,
        table: &CudaElasticMatrix,
        out: &mut CudaElasticMatrix,
    ) {
        assert!(table.rows > 0 && table.cols > 0, "elastic embedding table empty");
        assert_eq!((out.rows, out.cols), (1, table.cols));
        let (token_arg, vocab_arg, d_arg) = (token as usize, table.rows, table.cols);
        let mut builder = self.stream.launch_builder(&self.kernels.embed_token);
        builder.arg(&mut out.buf);
        builder.arg(&table.buf);
        builder.arg(&token_arg);
        builder.arg(&vocab_arg);
        builder.arg(&d_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(table.cols as u32))
                .expect("elastic decode embed launch");
        }
    }

    pub fn add_into(
        &self,
        a: &CudaElasticMatrix,
        b: &CudaElasticMatrix,
        out: &mut CudaElasticMatrix,
    ) {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        assert_eq!((out.rows, out.cols), (a.rows, a.cols));
        let n = a.rows * a.cols;
        let n_arg = n;
        let mut builder = self.stream.launch_builder(&self.kernels.add);
        builder.arg(&mut out.buf);
        builder.arg(&a.buf);
        builder.arg(&b.buf);
        builder.arg(&n_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("elastic decode add launch");
        }
    }

    pub fn rms_norm_into(
        &self,
        x: &CudaElasticMatrix,
        weight: &CudaElasticMatrix,
        eps: f32,
        out: &mut CudaElasticMatrix,
    ) {
        assert_eq!(x.rows, 1, "elastic RMSNorm is batch-one only");
        assert_eq!(weight.rows * weight.cols, x.cols);
        assert_eq!((out.rows, out.cols), (1, x.cols));
        let (cols_arg, eps_arg) = (x.cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels.rmsnorm);
        builder.arg(&mut out.buf);
        builder.arg(&x.buf);
        builder.arg(&weight.buf);
        builder.arg(&cols_arg);
        builder.arg(&eps_arg);
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(config).expect("elastic decode RMSNorm launch") };
    }

    pub fn swiglu_split_into(
        &self,
        gate_up: &CudaElasticMatrix,
        out: &mut CudaElasticMatrix,
    ) {
        assert_eq!(gate_up.rows, 1);
        assert_eq!(gate_up.cols % 2, 0);
        let d_ff = gate_up.cols / 2;
        assert_eq!((out.rows, out.cols), (1, d_ff));
        let d_ff_arg = d_ff;
        let mut builder = self.stream.launch_builder(&self.kernels.swiglu_split);
        builder.arg(&mut out.buf);
        builder.arg(&gate_up.buf);
        builder.arg(&d_ff_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(d_ff as u32))
                .expect("elastic decode SwiGLU launch");
        }
    }

    pub fn matmul_into(
        &self,
        a: &CudaElasticMatrix,
        b: &CudaElasticMatrix,
        out: &mut CudaElasticMatrix,
    ) {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        assert_eq!(b.rows, k);
        assert_eq!((out.rows, out.cols), (m, n));
        let config = MatmulConfig {
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
                .matmul(config, &b.buf, &a.buf, &mut out.buf, None, None)
                .expect("elastic decode cuBLASLt matmul");
        }
    }

    pub fn matmul_bt_into(
        &self,
        a: &CudaElasticMatrix,
        b: &CudaElasticMatrix,
        out: &mut CudaElasticMatrix,
    ) {
        let (m, k, n) = (a.rows, a.cols, b.rows);
        assert_eq!(b.cols, k);
        assert_eq!((out.rows, out.cols), (m, n));
        let config = MatmulConfig {
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
                .matmul(config, &b.buf, &a.buf, &mut out.buf, None, None)
                .expect("elastic decode cuBLASLt matmul_bt");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn latent_gqa_into(
        &self,
        qkv: &CudaElasticMatrix,
        kcache: &mut CudaElasticKvCache,
        vcache: &mut CudaElasticKvCache,
        pos: usize,
        dense_d_head: usize,
        key_rank: usize,
        value_rank: usize,
        n_heads: usize,
        n_kv_heads: usize,
        theta: f32,
        out_latent: &mut CudaElasticMatrix,
    ) {
        assert_eq!(qkv.rows, 1);
        assert!(dense_d_head > 0 && key_rank > 0 && value_rank > 0);
        assert!(key_rank <= dense_d_head && key_rank.is_multiple_of(2));
        assert!(n_heads > 0 && n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads));
        let q_width = n_heads * key_rank;
        let k_width = n_kv_heads * key_rank;
        let v_width = n_kv_heads * value_rank;
        assert_eq!(qkv.cols, q_width + k_width + v_width);
        assert_eq!(kcache.cols, k_width);
        assert_eq!(vcache.cols, v_width);
        assert_eq!(kcache.capacity, vcache.capacity);
        assert!(pos < kcache.capacity);
        assert_eq!((out_latent.rows, out_latent.cols), (1, n_heads * value_rank));

        let capacity = kcache.capacity;
        let (
            pos_arg,
            capacity_arg,
            dense_d_head_arg,
            key_rank_arg,
            value_rank_arg,
            n_heads_arg,
            n_kv_heads_arg,
            theta_arg,
        ) = (
            pos,
            capacity,
            dense_d_head,
            key_rank,
            value_rank,
            n_heads,
            n_kv_heads,
            theta,
        );
        let scale_arg = 1.0f32 / (dense_d_head as f32).sqrt();
        let mut builder = self.stream.launch_builder(&self.kernels.gqa);
        builder.arg(&mut out_latent.buf);
        builder.arg(&qkv.buf);
        builder.arg(&mut kcache.buf);
        builder.arg(&mut vcache.buf);
        builder.arg(&pos_arg);
        builder.arg(&capacity_arg);
        builder.arg(&dense_d_head_arg);
        builder.arg(&key_rank_arg);
        builder.arg(&value_rank_arg);
        builder.arg(&n_heads_arg);
        builder.arg(&n_kv_heads_arg);
        builder.arg(&theta_arg);
        builder.arg(&scale_arg);

        let seq = pos + 1;
        let shared_floats = seq + 2 * key_rank + 256;
        let config = LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (shared_floats * core::mem::size_of::<f32>()) as u32,
        };
        unsafe { builder.launch(config).expect("elastic latent GQA launch") };
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
    fn elastic_decode_kernel_source_compiles_when_nvrtc_is_available() {
        if !nvrtc_available()
        {
            eprintln!("cuda elastic decode: NVRTC unavailable, skipping compile test");
            return;
        }
        compile_ptx(ELASTIC_DECODE_KERNELS_SRC).expect("elastic decode NVRTC compilation");
    }
}
