//! Fused batch-one tied-LM-head projection and greedy argmax.
//!
//! The standard greedy I250 path currently computes all vocabulary logits into a
//! BF16 row and then launches a separate argmax kernel. For a tied embedding matrix
//! `[V,K]`, that means materializing `V` BF16 logits even though greedy generation
//! consumes only the winning token id.
//!
//! This module fuses the projection boundary with greedy selection:
//!
//! 1. one warp computes one vocabulary-row dot product from resident BF16 hidden
//!    state and tied embedding weights;
//! 2. the FP32 dot is rounded to BF16 exactly at the visible LM-head boundary;
//! 3. each CUDA block emits only its best `(logit, token)` candidate;
//! 4. one deterministic final reduction writes the winning token directly into the
//!    device-feedback buffers.
//!
//! Ties select the lower token id. NaNs after token 0 are ignored, matching the CPU
//! greedy loop's `v > best` comparison. If token 0 itself is NaN, token 0 wins
//! unconditionally, also matching the CPU initialization rule. The warp reduction
//! association differs from cuBLASLt, so generated-token parity remains a mandatory
//! promotion gate.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: usize = 32;
const WARPS_PER_BLOCK: usize = THREADS_PER_BLOCK as usize / WARP_SIZE;

const BF16_LM_HEAD_ARGMAX_SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int s = __float_as_uint(f);
    unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);
    return (unsigned short)((s + bias) >> 16);
}

extern "C" __global__ void scirust_bf16_lm_head_candidates_kernel(
    float* candidate_values,
    unsigned int* candidate_indices,
    const unsigned short* hidden,
    const unsigned short* embedding,
    const size_t vocab,
    const size_t k)
{
    __shared__ float warp_values[8];
    __shared__ unsigned int warp_indices[8];
    __shared__ int token_zero_nan;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const size_t token = (size_t)blockIdx.x * 8u + warp;

    if (tid == 0) token_zero_nan = 0;
    __syncthreads();

    float acc = 0.0f;
    if (token < vocab) {
        const size_t row = token * k;
        for (size_t col = lane; col < k; col += 32u)
            acc += b2f(hidden[col]) * b2f(embedding[row + col]);
    }

    for (unsigned int offset = 16u; offset > 0u; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);

    if (lane == 0) {
        if (token < vocab) {
            const float logit = b2f(f2b(acc));
            if (token == 0 && isnan(logit)) token_zero_nan = 1;
            if (token != 0 && isnan(logit)) {
                warp_values[warp] = -__int_as_float(0x7f800000);
                warp_indices[warp] = 0xffffffffu;
            } else {
                warp_values[warp] = logit;
                warp_indices[warp] = (unsigned int)token;
            }
        } else {
            warp_values[warp] = -__int_as_float(0x7f800000);
            warp_indices[warp] = 0xffffffffu;
        }
    }
    __syncthreads();

    if (tid == 0) {
        if (blockIdx.x == 0 && token_zero_nan) {
            candidate_values[0] = __int_as_float(0x7fc00000);
            candidate_indices[0] = 0u;
            return;
        }

        float best_value = -__int_as_float(0x7f800000);
        unsigned int best_index = 0xffffffffu;
        for (unsigned int i = 0; i < 8u; ++i) {
            const float value = warp_values[i];
            const unsigned int index = warp_indices[i];
            if (index == 0xffffffffu) continue;
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        candidate_values[blockIdx.x] = best_value;
        candidate_indices[blockIdx.x] = best_index;
    }
}

extern "C" __global__ void scirust_bf16_lm_head_argmax_kernel(
    const float* candidate_values,
    const unsigned int* candidate_indices,
    const size_t candidate_count,
    unsigned int* current_token,
    unsigned int* generated,
    const size_t generated_index)
{
    __shared__ float best_values[256];
    __shared__ unsigned int best_indices[256];
    const unsigned int tid = threadIdx.x;

    if (candidate_count == 0) return;
    if (isnan(candidate_values[0])) {
        if (tid == 0) {
            current_token[0] = 0u;
            generated[generated_index] = 0u;
        }
        return;
    }

    float local_value = -__int_as_float(0x7f800000);
    unsigned int local_index = 0xffffffffu;
    for (size_t i = tid; i < candidate_count; i += blockDim.x) {
        const float value = candidate_values[i];
        const unsigned int index = candidate_indices[i];
        if (index == 0xffffffffu || isnan(value)) continue;
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
"#;

/// Persistent candidate storage for fused LM-head greedy selection.
pub struct CudaBf16LmHeadArgmaxWorkspace {
    candidate_values: CudaSlice<f32>,
    candidate_indices: CudaSlice<u32>,
    candidate_count: usize,
}

impl CudaBf16LmHeadArgmaxWorkspace {
    #[must_use]
    pub const fn candidate_count(&self) -> usize {
        self.candidate_count
    }
}

/// Fused tied-LM-head projection and deterministic greedy argmax.
pub struct CudaBf16LmHeadArgmax {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    candidates: CudaFunction,
    argmax: CudaFunction,
}

impl CudaBf16LmHeadArgmax {
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

    #[must_use]
    pub fn from_context(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Option<Self> {
        let ptx = compile_ptx(BF16_LM_HEAD_ARGMAX_SRC)
            .map_err(|error| eprintln!("scirust-cuda lm-head argmax: NVRTC failed: {error}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| eprintln!("scirust-cuda lm-head argmax: module load failed: {error}"))
            .ok()?;
        let candidates = module
            .load_function("scirust_bf16_lm_head_candidates_kernel")
            .ok()?;
        let argmax = module
            .load_function("scirust_bf16_lm_head_argmax_kernel")
            .ok()?;
        Some(Self {
            _ctx: ctx,
            stream,
            candidates,
            argmax,
        })
    }

    #[must_use]
    pub fn workspace(&self, vocab: usize) -> CudaBf16LmHeadArgmaxWorkspace {
        assert!(vocab > 0, "LM-head vocabulary must be non-zero");
        let candidate_count = vocab.div_ceil(WARPS_PER_BLOCK);
        let candidate_values = self
            .stream
            .alloc_zeros::<f32>(candidate_count)
            .expect("LM-head candidate-value allocation");
        let candidate_indices = self
            .stream
            .alloc_zeros::<u32>(candidate_count)
            .expect("LM-head candidate-index allocation");
        CudaBf16LmHeadArgmaxWorkspace {
            candidate_values,
            candidate_indices,
            candidate_count,
        }
    }

    /// Compute tied-embedding logits and write only the greedy winner.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_into(
        &self,
        hidden: &CudaSlice<bf16>,
        embedding: &CudaSlice<bf16>,
        workspace: &mut CudaBf16LmHeadArgmaxWorkspace,
        current_token: &mut CudaSlice<u32>,
        generated: &mut CudaSlice<u32>,
        generated_index: usize,
        k: usize,
        vocab: usize,
    ) {
        assert!(k > 0 && vocab > 0, "LM-head dimensions must be non-zero");
        assert_eq!(hidden.len(), k, "LM-head hidden width");
        assert_eq!(embedding.len(), vocab * k, "LM-head embedding shape");
        assert_eq!(current_token.len(), 1, "LM-head current-token length");
        assert!(generated_index < generated.len(), "LM-head generated index");
        assert_eq!(
            workspace.candidate_count,
            vocab.div_ceil(WARPS_PER_BLOCK),
            "LM-head candidate workspace shape"
        );

        let (vocab_arg, k_arg) = (vocab, k);
        let mut candidate_builder = self.stream.launch_builder(&self.candidates);
        candidate_builder.arg(&mut workspace.candidate_values);
        candidate_builder.arg(&mut workspace.candidate_indices);
        candidate_builder.arg(hidden);
        candidate_builder.arg(embedding);
        candidate_builder.arg(&vocab_arg);
        candidate_builder.arg(&k_arg);
        let candidate_config = LaunchConfig {
            grid_dim: (workspace.candidate_count as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            candidate_builder
                .launch(candidate_config)
                .expect("LM-head candidate launch");
        }

        let (candidate_count_arg, generated_index_arg) =
            (workspace.candidate_count, generated_index);
        let mut argmax_builder = self.stream.launch_builder(&self.argmax);
        argmax_builder.arg(&workspace.candidate_values);
        argmax_builder.arg(&workspace.candidate_indices);
        argmax_builder.arg(&candidate_count_arg);
        argmax_builder.arg(current_token);
        argmax_builder.arg(generated);
        argmax_builder.arg(&generated_index_arg);
        let argmax_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            argmax_builder
                .launch(argmax_config)
                .expect("LM-head argmax launch");
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
    fn candidate_count_matches_warp_packing() {
        assert_eq!(32_768usize.div_ceil(WARPS_PER_BLOCK), 4096);
        assert_eq!(48usize.div_ceil(WARPS_PER_BLOCK), 6);
    }

    #[test]
    fn lm_head_argmax_kernel_source_compiles_when_nvrtc_is_available() {
        if !nvrtc_available()
        {
            eprintln!("cuda LM-head argmax: NVRTC unavailable, skipping compile test");
            return;
        }
        compile_ptx(BF16_LM_HEAD_ARGMAX_SRC).expect("LM-head argmax NVRTC compilation");
    }
}
