//! Batch-one CUDA decode runtime for latency-sensitive autoregressive inference.
//!
//! This path is isolated from [`crate::CudaChain`] so inference experiments cannot
//! perturb Route-B training semantics. It owns fixed-capacity KV storage, fused
//! single-query GQA, fused QKV/gate-up consumers, and a greedy device-feedback path.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

use crate::bf16_gemv::CudaBf16Gemv;

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

extern "C" __global__ void decode_embed_feedback_kernel(
    unsigned short* out, const unsigned short* table,
    const unsigned int* token, const size_t vocab, const size_t d)
{
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d) {
        const size_t raw = (size_t)token[0];
        const size_t row = raw < vocab ? raw : vocab - 1;
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

// Greedy tie rule matches sample_row: lower token index wins an equal logit.
extern "C" __global__ void decode_argmax_feedback_kernel(
    const unsigned short* logits,
    const size_t vocab,
    unsigned int* current_token,
    unsigned int* generated,
    const size_t generated_index)
{
    __shared__ float best_values[256];
    __shared__ unsigned int best_indices[256];
    const unsigned int tid = threadIdx.x;
    if (vocab == 0) return;

    // CPU greedy initializes from row[0]. If it is NaN every `v > best` is false.
    if (isnan(b2f(logits[0]))) {
        if (tid == 0) {
            current_token[0] = 0u;
            generated[generated_index] = 0u;
        }
        return;
    }

    float local_value = -__int_as_float(0x7f800000);
    unsigned int local_index = 0xffffffffu;
    for (size_t i = tid; i < vocab; i += blockDim.x) {
        const float value = b2f(logits[i]);
        if (isnan(value)) continue;
        const unsigned int index = (unsigned int)i;
        if (value > local_value || (value == local_value && index < local_index)) {
            local_value = value;
            local_index = index;
        }
    }
    best_values[tid] = local_value;
    best_indices[tid] = local_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float rhs_value = best_values[tid + stride];
            const unsigned int rhs_index = best_indices[tid + stride];
            const float lhs_value = best_values[tid];
            const unsigned int lhs_index = best_indices[tid];
            if (rhs_value > lhs_value ||
                (rhs_value == lhs_value && rhs_index < lhs_index)) {
                best_values[tid] = rhs_value;
                best_indices[tid] = rhs_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        current_token[0] = best_indices[0];
        generated[generated_index] = best_indices[0];
    }
}

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
    float* scores = (float*)smem_raw;
    float* qrot = scores + seq;
    float* kcur = qrot + dh;
    float* red = kcur + dh;

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

    if ((head % repeat) == 0) {
        for (size_t c = tid; c < dh; c += blockDim.x) {
            const size_t dst = pos * kv_dim + kv * dh + c;
            kcache[dst] = f2b(kcur[c]);
            vcache[dst] = qkv[d_model + kv_dim + kv * dh + c];
        }
    }

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

    for (size_t c = tid; c < dh; c += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < seq; ++j) {
            const float vv = j == pos
                ? b2f(qkv[d_model + kv_dim + kv * dh + c])
                : b2f(vcache[j * kv_dim + kv * dh + c]);
            acc += scores[j] * vv;
        }
        out[head * dh + c] = f2b(acc);
    }
}
"#;

pub struct CudaDecodeMatrix {
    buf: CudaSlice<bf16>,
    rows: usize,
    cols: usize,
}

impl CudaDecodeMatrix {
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

pub struct CudaDecodeKvCache {
    buf: CudaSlice<bf16>,
    capacity: usize,
    cols: usize,
}

impl CudaDecodeKvCache {
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

pub struct CudaDecodeGreedyFeedback {
    current_token: CudaSlice<u32>,
    generated: CudaSlice<u32>,
    capacity: usize,
}

impl CudaDecodeGreedyFeedback {
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

struct DecodeKernels {
    add: CudaFunction,
    embed_token: CudaFunction,
    embed_feedback: CudaFunction,
    rmsnorm: CudaFunction,
    swiglu_split: CudaFunction,
    argmax_feedback: CudaFunction,
    gqa: CudaFunction,
}

pub struct CudaDecodeRuntime {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlasLT,
    gemv: CudaBf16Gemv,
    kernels: DecodeKernels,
}

impl CudaDecodeRuntime {
    #[must_use]
    pub fn new() -> Option<Self> {
        if !cuda_libraries_available() || !nvrtc_available()
        {
            return None;
        }
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone()).ok()?;
        let gemv = CudaBf16Gemv::from_context(ctx.clone(), stream.clone())?;
        let ptx = compile_ptx(DECODE_KERNELS_SRC)
            .map_err(|error| eprintln!("scirust-cuda decode: NVRTC compile failed: {error}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| eprintln!("scirust-cuda decode: module load failed: {error}"))
            .ok()?;
        let function = |name: &str| module.load_function(name).expect("load decode kernel");
        let kernels = DecodeKernels {
            add: function("decode_add_kernel"),
            embed_token: function("decode_embed_token_kernel"),
            embed_feedback: function("decode_embed_feedback_kernel"),
            rmsnorm: function("decode_rmsnorm_kernel"),
            swiglu_split: function("decode_swiglu_split_kernel"),
            argmax_feedback: function("decode_argmax_feedback_kernel"),
            gqa: function("decode_gqa_kernel"),
        };
        Some(Self {
            _ctx: ctx,
            stream,
            blas,
            gemv,
            kernels,
        })
    }

    #[must_use]
    pub fn matrix(&self, rows: usize, cols: usize) -> CudaDecodeMatrix {
        assert!(rows > 0 && cols > 0, "decode matrix must be non-empty");
        let buf = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("decode CUDA matrix allocation");
        CudaDecodeMatrix { buf, rows, cols }
    }

    #[must_use]
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> CudaDecodeMatrix {
        assert_eq!(data.len(), rows * cols, "decode upload shape mismatch");
        let bf: Vec<bf16> = data.iter().map(|&value| bf16::from_f32(value)).collect();
        let buf = self.stream.clone_htod(&bf).expect("decode CUDA upload");
        CudaDecodeMatrix { buf, rows, cols }
    }

    #[must_use]
    pub fn kv_cache(&self, capacity: usize, cols: usize) -> CudaDecodeKvCache {
        assert!(
            capacity > 0 && cols > 0,
            "decode KV cache must be non-empty"
        );
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

    #[must_use]
    pub fn greedy_feedback(&self, capacity: usize) -> CudaDecodeGreedyFeedback {
        assert!(capacity > 0, "greedy feedback capacity must be non-zero");
        let current_token = self
            .stream
            .alloc_zeros::<u32>(1)
            .expect("decode greedy token allocation");
        let generated = self
            .stream
            .alloc_zeros::<u32>(capacity)
            .expect("decode greedy output allocation");
        CudaDecodeGreedyFeedback {
            current_token,
            generated,
            capacity,
        }
    }

    #[must_use]
    pub fn download(&self, matrix: &CudaDecodeMatrix) -> Vec<f32> {
        let host: Vec<bf16> = self
            .stream
            .clone_dtoh(&matrix.buf)
            .expect("decode CUDA download");
        host.iter().map(|value| value.to_f32()).collect()
    }

    #[must_use]
    pub fn download_feedback(&self, feedback: &CudaDecodeGreedyFeedback) -> Vec<u32> {
        self.stream
            .clone_dtoh(&feedback.generated)
            .expect("decode greedy feedback download")
    }

    pub fn embed_token_into(
        &self,
        token: u32,
        table: &CudaDecodeMatrix,
        out: &mut CudaDecodeMatrix,
    ) {
        assert!(
            table.rows > 0 && table.cols > 0,
            "decode embedding table empty"
        );
        assert_eq!(
            (out.rows, out.cols),
            (1, table.cols),
            "decode embed output shape"
        );
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
                .expect("decode embed launch");
        }
    }

    pub fn embed_feedback_into(
        &self,
        feedback: &CudaDecodeGreedyFeedback,
        table: &CudaDecodeMatrix,
        out: &mut CudaDecodeMatrix,
    ) {
        assert_eq!(
            (out.rows, out.cols),
            (1, table.cols),
            "decode feedback embed shape"
        );
        let (vocab_arg, d_arg) = (table.rows, table.cols);
        let mut builder = self.stream.launch_builder(&self.kernels.embed_feedback);
        builder.arg(&mut out.buf);
        builder.arg(&table.buf);
        builder.arg(&feedback.current_token);
        builder.arg(&vocab_arg);
        builder.arg(&d_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(table.cols as u32))
                .expect("decode feedback embed launch");
        }
    }

    pub fn greedy_argmax_into(
        &self,
        logits: &CudaDecodeMatrix,
        feedback: &mut CudaDecodeGreedyFeedback,
        generated_index: usize,
    ) {
        assert_eq!(logits.rows, 1, "greedy argmax expects one logits row");
        assert!(
            generated_index < feedback.capacity,
            "greedy feedback index overflow"
        );
        let (vocab_arg, index_arg) = (logits.cols, generated_index);
        let mut builder = self.stream.launch_builder(&self.kernels.argmax_feedback);
        builder.arg(&logits.buf);
        builder.arg(&vocab_arg);
        builder.arg(&mut feedback.current_token);
        builder.arg(&mut feedback.generated);
        builder.arg(&index_arg);
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(config).expect("decode greedy argmax launch") };
    }

    pub fn add_into(&self, a: &CudaDecodeMatrix, b: &CudaDecodeMatrix, out: &mut CudaDecodeMatrix) {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols), "decode add input shape");
        assert_eq!(
            (out.rows, out.cols),
            (a.rows, a.cols),
            "decode add output shape"
        );
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
                .expect("decode add launch");
        }
    }

    pub fn rms_norm_into(
        &self,
        x: &CudaDecodeMatrix,
        weight: &CudaDecodeMatrix,
        eps: f32,
        out: &mut CudaDecodeMatrix,
    ) {
        assert_eq!(x.rows, 1, "decode RMSNorm is batch-one only");
        assert_eq!(
            weight.rows * weight.cols,
            x.cols,
            "decode RMSNorm weight shape"
        );
        assert_eq!(
            (out.rows, out.cols),
            (1, x.cols),
            "decode RMSNorm output shape"
        );
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
        unsafe { builder.launch(config).expect("decode RMSNorm launch") };
    }

    /// Fused batch-one gate/up projection + SwiGLU using the decode-native GEMV kernel.
    pub fn swiglu_gemv_into(
        &self,
        input: &CudaDecodeMatrix,
        gate_up_weight: &CudaDecodeMatrix,
        out: &mut CudaDecodeMatrix,
    ) {
        assert_eq!(input.rows, 1, "decode fused SwiGLU is batch-one only");
        assert_eq!(
            gate_up_weight.rows, input.cols,
            "decode fused SwiGLU input width"
        );
        assert_eq!(
            gate_up_weight.cols,
            out.cols * 2,
            "decode fused SwiGLU weight width"
        );
        assert_eq!(out.rows, 1, "decode fused SwiGLU output rows");
        self.gemv.swiglu_kn_into(
            &input.buf,
            &gate_up_weight.buf,
            &mut out.buf,
            input.cols,
            out.cols,
        );
    }

    pub fn swiglu_split_into(&self, gate_up: &CudaDecodeMatrix, out: &mut CudaDecodeMatrix) {
        assert_eq!(gate_up.rows, 1, "decode SwiGLU is batch-one only");
        assert_eq!(gate_up.cols % 2, 0, "decode gate/up width must be even");
        let d_ff = gate_up.cols / 2;
        assert_eq!(
            (out.rows, out.cols),
            (1, d_ff),
            "decode SwiGLU output shape"
        );
        let d_ff_arg = d_ff;
        let mut builder = self.stream.launch_builder(&self.kernels.swiglu_split);
        builder.arg(&mut out.buf);
        builder.arg(&gate_up.buf);
        builder.arg(&d_ff_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(d_ff as u32))
                .expect("decode SwiGLU launch");
        }
    }

    pub fn matmul_into(
        &self,
        a: &CudaDecodeMatrix,
        b: &CudaDecodeMatrix,
        out: &mut CudaDecodeMatrix,
    ) {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        assert_eq!(b.rows, k, "decode matmul inner dimensions");
        assert_eq!((out.rows, out.cols), (m, n), "decode matmul output shape");
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
                .expect("decode cuBLASLt matmul");
        }
    }

    pub fn matmul_bt_into(
        &self,
        a: &CudaDecodeMatrix,
        b: &CudaDecodeMatrix,
        out: &mut CudaDecodeMatrix,
    ) {
        let (m, k, n) = (a.rows, a.cols, b.rows);
        assert_eq!(b.cols, k, "decode matmul_bt inner dimensions");
        assert_eq!(
            (out.rows, out.cols),
            (m, n),
            "decode matmul_bt output shape"
        );
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
                .expect("decode cuBLASLt matmul_bt");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gqa_decode_into(
        &self,
        qkv: &CudaDecodeMatrix,
        kcache: &mut CudaDecodeKvCache,
        vcache: &mut CudaDecodeKvCache,
        pos: usize,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        theta: f32,
        out: &mut CudaDecodeMatrix,
    ) {
        assert_eq!(qkv.rows, 1, "decode GQA is single-query only");
        assert!(n_heads > 0 && n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads));
        assert!(d_model.is_multiple_of(n_heads));
        let dh = d_model / n_heads;
        let kv_dim = n_kv_heads * dh;
        assert_eq!(qkv.cols, d_model + 2 * kv_dim, "decode fused QKV width");
        assert_eq!(kcache.cols, kv_dim, "decode K-cache width");
        assert_eq!(vcache.cols, kv_dim, "decode V-cache width");
        assert_eq!(
            kcache.capacity, vcache.capacity,
            "decode KV capacity mismatch"
        );
        assert!(pos < kcache.capacity, "decode position exceeds KV capacity");
        assert_eq!(
            (out.rows, out.cols),
            (1, d_model),
            "decode GQA output shape"
        );

        let capacity = kcache.capacity;
        let (pos_arg, cap_arg, d_arg, kv_arg, heads_arg, kv_heads_arg, theta_arg) =
            (pos, capacity, d_model, kv_dim, n_heads, n_kv_heads, theta);
        let scale_arg = 1.0f32 / (dh as f32).sqrt();
        let mut builder = self.stream.launch_builder(&self.kernels.gqa);
        builder.arg(&mut out.buf);
        builder.arg(&qkv.buf);
        builder.arg(&mut kcache.buf);
        builder.arg(&mut vcache.buf);
        builder.arg(&pos_arg);
        builder.arg(&cap_arg);
        builder.arg(&d_arg);
        builder.arg(&kv_arg);
        builder.arg(&heads_arg);
        builder.arg(&kv_heads_arg);
        builder.arg(&theta_arg);
        builder.arg(&scale_arg);

        let seq = pos + 1;
        let shared_floats = seq + 2 * dh + 256;
        let config = LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (shared_floats * core::mem::size_of::<f32>()) as u32,
        };
        unsafe { builder.launch(config).expect("decode fused GQA launch") };
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
        if !nvrtc_available()
        {
            eprintln!("cuda decode: NVRTC unavailable, skipping compile test");
            return;
        }
        compile_ptx(DECODE_KERNELS_SRC).expect("decode NVRTC kernel compilation");
    }
}
