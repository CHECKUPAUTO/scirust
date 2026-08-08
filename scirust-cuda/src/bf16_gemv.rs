//! Batch-one BF16 matrix-vector kernels for decode-dominant projections.
//!
//! cuBLASLt is the general Route-B GEMM implementation. Autoregressive decode has a
//! different geometry: `m = 1`, large row-major weights, and the same weights are
//! streamed once per generated token. These kernels make that memory-streaming shape
//! explicit instead of padding it into a general matrix-matrix problem.
//!
//! Two primitives are provided:
//!
//! - [`CudaBf16Gemv::gemv_kn_into`] computes `[1,K] × [K,N] -> [1,N]` with one
//!   CUDA thread per output column. Threads in a warp therefore read adjacent BF16
//!   weights from every input row.
//! - [`CudaBf16Gemv::swiglu_kn_into`] consumes a fused `[gate | up]` weight matrix
//!   `[K, 2*Dff]`, accumulates both projections, preserves their explicit BF16
//!   boundaries, applies SiLU and writes only the final `Dff` BF16 activations. The
//!   intermediate gate/up row is never materialized in global memory.
//!
//! Both kernels accumulate strictly left-to-right over `K` in FP32. The generic
//! GEMV rounds once at its output boundary. Fused SwiGLU separately rounds gate and
//! up to BF16 before the nonlinearity, then rounds the final activation to BF16.
//! Those boundaries mirror the existing dense decoder; token-stream parity against
//! the historical Route-B decoder remains a separate promotion gate because cuBLASLt
//! may use a different FP32 reduction order.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

const BF16_GEMV_SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int s = __float_as_uint(f);
    unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);
    return (unsigned short)((s + bias) >> 16);
}

extern "C" __global__ void scirust_bf16_gemv_kn_kernel(
    unsigned short* out,
    const unsigned short* input,
    const unsigned short* weight,
    const size_t k,
    const size_t n)
{
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    float acc = 0.0f;
    for (size_t row = 0; row < k; ++row)
        acc += b2f(input[row]) * b2f(weight[row * n + col]);
    out[col] = f2b(acc);
}

extern "C" __global__ void scirust_bf16_swiglu_kn_kernel(
    unsigned short* out,
    const unsigned short* input,
    const unsigned short* gate_up_weight,
    const size_t k,
    const size_t d_ff)
{
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= d_ff) return;

    const size_t width = d_ff * 2;
    float gate = 0.0f;
    float up = 0.0f;
    for (size_t row = 0; row < k; ++row) {
        const float x = b2f(input[row]);
        const size_t base = row * width + col;
        gate += x * b2f(gate_up_weight[base]);
        up += x * b2f(gate_up_weight[base + d_ff]);
    }

    // Preserve the same visible boundaries as dense BF16 projection -> SwiGLU.
    const float gate_bf = b2f(f2b(gate));
    const float up_bf = b2f(f2b(up));
    const float silu = gate_bf / (1.0f + __expf(-gate_bf));
    out[col] = f2b(silu * up_bf);
}
"#;

/// Decode-specialized BF16 GEMV kernels bound to one CUDA context/stream.
pub struct CudaBf16Gemv {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    gemv_kn: CudaFunction,
    swiglu_kn: CudaFunction,
}

impl CudaBf16Gemv {
    /// Standalone constructor, primarily for focused benchmarks/tests.
    #[must_use]
    pub fn new() -> Option<Self> {
        if !cuda_libraries_available() || !nvrtc_available()
        {
            return None;
        }
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        Self::from_context(ctx, stream)
    }

    /// Bind to an existing CUDA context/stream so the kernels can share resident
    /// decode allocations without copies or a second execution stream.
    #[must_use]
    pub fn from_context(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Option<Self> {
        let ptx = compile_ptx(BF16_GEMV_SRC)
            .map_err(|error| eprintln!("scirust-cuda bf16 gemv: NVRTC failed: {error}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| eprintln!("scirust-cuda bf16 gemv: module load failed: {error}"))
            .ok()?;
        let gemv_kn = module
            .load_function("scirust_bf16_gemv_kn_kernel")
            .ok()?;
        let swiglu_kn = module
            .load_function("scirust_bf16_swiglu_kn_kernel")
            .ok()?;
        Some(Self {
            _ctx: ctx,
            stream,
            gemv_kn,
            swiglu_kn,
        })
    }

    /// `[1,K] × [K,N] -> [1,N]`, all buffers resident BF16.
    pub fn gemv_kn_into(
        &self,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        k: usize,
        n: usize,
    ) {
        assert!(k > 0 && n > 0, "bf16 gemv dimensions must be non-zero");
        assert_eq!(input.len(), k, "bf16 gemv input length");
        assert_eq!(weight.len(), k * n, "bf16 gemv weight length");
        assert_eq!(output.len(), n, "bf16 gemv output length");

        let (k_arg, n_arg) = (k, n);
        let mut builder = self.stream.launch_builder(&self.gemv_kn);
        builder.arg(output);
        builder.arg(input);
        builder.arg(weight);
        builder.arg(&k_arg);
        builder.arg(&n_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("bf16 gemv launch");
        }
    }

    /// Fused `[1,K] × [K,2*Dff] -> gate/up -> SiLU(gate)*up -> [1,Dff]`.
    pub fn swiglu_kn_into(
        &self,
        input: &CudaSlice<bf16>,
        gate_up_weight: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        k: usize,
        d_ff: usize,
    ) {
        assert!(k > 0 && d_ff > 0, "bf16 SwiGLU dimensions must be non-zero");
        assert_eq!(input.len(), k, "bf16 SwiGLU input length");
        assert_eq!(
            gate_up_weight.len(),
            k * 2 * d_ff,
            "bf16 SwiGLU fused weight length"
        );
        assert_eq!(output.len(), d_ff, "bf16 SwiGLU output length");

        let (k_arg, d_ff_arg) = (k, d_ff);
        let mut builder = self.stream.launch_builder(&self.swiglu_kn);
        builder.arg(output);
        builder.arg(input);
        builder.arg(gate_up_weight);
        builder.arg(&k_arg);
        builder.arg(&d_ff_arg);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(d_ff as u32))
                .expect("bf16 fused SwiGLU GEMV launch");
        }
    }
}

fn cuda_libraries_available() -> bool {
    unsafe { cudarc::driver::sys::is_culib_present() }
}

fn nvrtc_available() -> bool {
    unsafe { cudarc::nvrtc::sys::is_culib_present() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_gemv_kernel_source_compiles_when_nvrtc_is_available() {
        if !nvrtc_available()
        {
            eprintln!("cuda bf16 gemv: NVRTC unavailable, skipping compile test");
            return;
        }
        compile_ptx(BF16_GEMV_SRC).expect("bf16 GEMV NVRTC compilation");
    }
}
