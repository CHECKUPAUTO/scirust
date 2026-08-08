//! Deterministic tiled BF16 GEMV for narrower batch-one decode projections.
//!
//! The simple decode-native GEMV assigns one thread to one output column. That is a
//! good fit for wide projections such as `gate+up` (`N=5632` in SCIAGENT 350m), but
//! narrower matrices such as `down`, `Wo`, and fused QKV expose too few blocks to
//! keep a large GPU busy.
//!
//! This implementation splits the reduction dimension `K` into fixed tiles. Stage 1
//! computes one FP32 partial sum for each `(K tile, output column)`; stage 2 reduces
//! those partials in strictly increasing tile order and rounds once to BF16. No
//! atomics are used. The result is deterministic for a fixed tile size and launch
//! geometry, though its FP32 association intentionally differs from both cuBLASLt and
//! the one-thread full-K GEMV and therefore requires a token-parity promotion gate.
//!
//! A second stage-2 kernel can fuse the residual add. It first rounds the reduced
//! projection to BF16, then adds the resident BF16 residual in FP32, then rounds the
//! final sum to BF16. Those are exactly the two visible boundaries of the existing
//! `matmul -> BF16 temporary -> add -> BF16 output` path, without materializing the
//! projection temporary or launching a separate add kernel.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::bf16;

const DEFAULT_K_TILE: usize = 256;
const THREADS_PER_BLOCK: u32 = 256;

const BF16_TILED_GEMV_SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2b(float f) {
    unsigned int s = __float_as_uint(f);
    unsigned int bias = 0x00007FFFu + ((s >> 16) & 1u);
    return (unsigned short)((s + bias) >> 16);
}

extern "C" __global__ void scirust_bf16_tiled_gemv_partial_kernel(
    float* partials,
    const unsigned short* input,
    const unsigned short* weight,
    const size_t k,
    const size_t n,
    const size_t k_tile)
{
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tile = (size_t)blockIdx.y;
    if (col >= n) return;

    const size_t begin = tile * k_tile;
    const size_t end = min(begin + k_tile, k);
    float acc = 0.0f;
    for (size_t row = begin; row < end; ++row)
        acc += b2f(input[row]) * b2f(weight[row * n + col]);
    partials[tile * n + col] = acc;
}

extern "C" __global__ void scirust_bf16_tiled_gemv_reduce_kernel(
    unsigned short* out,
    const float* partials,
    const size_t tiles,
    const size_t n)
{
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    float acc = 0.0f;
    for (size_t tile = 0; tile < tiles; ++tile)
        acc += partials[tile * n + col];
    out[col] = f2b(acc);
}

extern "C" __global__ void scirust_bf16_tiled_gemv_reduce_add_kernel(
    unsigned short* out,
    const float* partials,
    const unsigned short* residual,
    const size_t tiles,
    const size_t n)
{
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    float acc = 0.0f;
    for (size_t tile = 0; tile < tiles; ++tile)
        acc += partials[tile * n + col];

    // Preserve projection BF16, then residual-add BF16, as separate boundaries.
    const float projected = b2f(f2b(acc));
    out[col] = f2b(b2f(residual[col]) + projected);
}
"#;

/// Persistent FP32 partial-sum storage for one tiled GEMV shape.
pub struct CudaBf16TiledGemvWorkspace {
    partials: CudaSlice<f32>,
    k_tiles: usize,
    n: usize,
}

impl CudaBf16TiledGemvWorkspace {
    #[must_use]
    pub const fn k_tiles(&self) -> usize {
        self.k_tiles
    }

    #[must_use]
    pub const fn output_width(&self) -> usize {
        self.n
    }

    #[must_use]
    pub const fn partial_scalar_count(&self) -> usize {
        self.k_tiles * self.n
    }
}

/// Two-stage deterministic BF16 GEMV bound to one CUDA context/stream.
pub struct CudaBf16TiledGemv {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    partial: CudaFunction,
    reduce: CudaFunction,
    reduce_add: CudaFunction,
    k_tile: usize,
}

impl CudaBf16TiledGemv {
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
        Self::from_context_with_k_tile(ctx, stream, DEFAULT_K_TILE)
    }

    #[must_use]
    pub fn from_context_with_k_tile(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        k_tile: usize,
    ) -> Option<Self> {
        if k_tile == 0
        {
            return None;
        }
        let ptx = compile_ptx(BF16_TILED_GEMV_SRC)
            .map_err(|error| eprintln!("scirust-cuda tiled gemv: NVRTC failed: {error}"))
            .ok()?;
        let module = ctx
            .load_module(ptx)
            .map_err(|error| eprintln!("scirust-cuda tiled gemv: module load failed: {error}"))
            .ok()?;
        let partial = module
            .load_function("scirust_bf16_tiled_gemv_partial_kernel")
            .ok()?;
        let reduce = module
            .load_function("scirust_bf16_tiled_gemv_reduce_kernel")
            .ok()?;
        let reduce_add = module
            .load_function("scirust_bf16_tiled_gemv_reduce_add_kernel")
            .ok()?;
        Some(Self {
            _ctx: ctx,
            stream,
            partial,
            reduce,
            reduce_add,
            k_tile,
        })
    }

    #[must_use]
    pub const fn k_tile(&self) -> usize {
        self.k_tile
    }

    #[must_use]
    pub fn workspace(&self, k: usize, n: usize) -> CudaBf16TiledGemvWorkspace {
        assert!(k > 0 && n > 0, "tiled GEMV dimensions must be non-zero");
        let k_tiles = k.div_ceil(self.k_tile);
        let partials = self
            .stream
            .alloc_zeros::<f32>(k_tiles * n)
            .expect("tiled GEMV partial workspace allocation");
        CudaBf16TiledGemvWorkspace {
            partials,
            k_tiles,
            n,
        }
    }

    /// `[1,K] × [K,N] -> [1,N]` with deterministic tiled FP32 reduction.
    pub fn gemv_kn_into(
        &self,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        workspace: &mut CudaBf16TiledGemvWorkspace,
        output: &mut CudaSlice<bf16>,
        k: usize,
        n: usize,
    ) {
        self.validate(input, weight, workspace, output, k, n);
        self.launch_partials(input, weight, workspace, k, n);

        let (tiles_arg, n_arg) = (workspace.k_tiles, n);
        let mut reduce_builder = self.stream.launch_builder(&self.reduce);
        reduce_builder.arg(output);
        reduce_builder.arg(&workspace.partials);
        reduce_builder.arg(&tiles_arg);
        reduce_builder.arg(&n_arg);
        unsafe {
            reduce_builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("tiled GEMV reduction launch");
        }
    }

    /// `[1,K] × [K,N]`, BF16 projection boundary, then BF16 residual add.
    #[allow(clippy::too_many_arguments)] // Mirrors the explicit GEMV + residual contract.
    pub fn gemv_kn_add_into(
        &self,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        workspace: &mut CudaBf16TiledGemvWorkspace,
        residual: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        k: usize,
        n: usize,
    ) {
        self.validate(input, weight, workspace, output, k, n);
        assert_eq!(residual.len(), n, "tiled GEMV residual length");
        self.launch_partials(input, weight, workspace, k, n);

        let (tiles_arg, n_arg) = (workspace.k_tiles, n);
        let mut reduce_builder = self.stream.launch_builder(&self.reduce_add);
        reduce_builder.arg(output);
        reduce_builder.arg(&workspace.partials);
        reduce_builder.arg(residual);
        reduce_builder.arg(&tiles_arg);
        reduce_builder.arg(&n_arg);
        unsafe {
            reduce_builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("tiled GEMV residual reduction launch");
        }
    }

    fn validate(
        &self,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        workspace: &CudaBf16TiledGemvWorkspace,
        output: &CudaSlice<bf16>,
        k: usize,
        n: usize,
    ) {
        assert!(k > 0 && n > 0, "tiled GEMV dimensions must be non-zero");
        assert_eq!(input.len(), k, "tiled GEMV input length");
        assert_eq!(weight.len(), k * n, "tiled GEMV weight length");
        assert_eq!(output.len(), n, "tiled GEMV output length");
        assert_eq!(workspace.n, n, "tiled GEMV workspace output width");
        assert_eq!(
            workspace.k_tiles,
            k.div_ceil(self.k_tile),
            "tiled GEMV workspace K tile count"
        );
    }

    fn launch_partials(
        &self,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        workspace: &mut CudaBf16TiledGemvWorkspace,
        k: usize,
        n: usize,
    ) {
        let (k_arg, n_arg, k_tile_arg) = (k, n, self.k_tile);
        let mut partial_builder = self.stream.launch_builder(&self.partial);
        partial_builder.arg(&mut workspace.partials);
        partial_builder.arg(input);
        partial_builder.arg(weight);
        partial_builder.arg(&k_arg);
        partial_builder.arg(&n_arg);
        partial_builder.arg(&k_tile_arg);
        let x_blocks = (n as u32).div_ceil(THREADS_PER_BLOCK);
        let partial_config = LaunchConfig {
            grid_dim: (x_blocks, workspace.k_tiles as u32, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            partial_builder
                .launch(partial_config)
                .expect("tiled GEMV partial launch");
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
    fn tile_count_is_exact_for_decode_shapes() {
        assert_eq!(1024usize.div_ceil(DEFAULT_K_TILE), 4);
        assert_eq!(2816usize.div_ceil(DEFAULT_K_TILE), 11);
    }

    #[test]
    fn tiled_gemv_kernel_source_compiles_when_nvrtc_is_available() {
        if !nvrtc_available()
        {
            eprintln!("cuda tiled gemv: NVRTC unavailable, skipping compile test");
            return;
        }
        compile_ptx(BF16_TILED_GEMV_SRC).expect("tiled GEMV NVRTC compilation");
    }
}
