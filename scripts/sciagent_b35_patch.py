#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:160]!r}")
    return text.replace(old, new, count)


OLD_KERNEL = r'''// Sum of squares of a bf16 buffer, accumulated (fp32) into accum[0] — the building
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
'''

NEW_KERNEL = r'''// Deterministic two-stage sum-of-squares for global grad clipping. Stage 1 writes
// exactly one fp32 partial per CUDA block; stage 2 reads those partials in a fixed
// strided order and one thread adds the final scalar to the running accumulator.
// There are no atomics, so identical bf16 gradients produce bit-identical norms.
extern "C" __global__ void sumsq_partials_kernel(
    const unsigned short* g, const size_t n, float* partials)
{
    __shared__ float sdata[256];
    const unsigned int tid = threadIdx.x;
    const size_t i = (size_t)blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    if (i < n) { const float x = b2f(g[i]); v = x * x; }
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

extern "C" __global__ void sumsq_finish_kernel(
    const float* partials, const size_t n_partials, float* accum)
{
    __shared__ float sdata[256];
    const unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (size_t i = tid; i < n_partials; i += blockDim.x)
        v += partials[i];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) accum[0] += sdata[0];
}
'''

OLD_METHOD = r'''    /// Accumulate the global gradient sum-of-squares into a resident fp32 scalar.
    /// This is the asynchronous half of global gradient clipping.
    pub fn global_grad_sumsq(&self, grads: &[&CudaMatrix]) -> CudaF32 {
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
            // SAFETY: argument layout matches sumsq_kernel; block size is 256.
            unsafe { builder.launch(cfg).expect("launch sumsq_kernel") };
        }
        CudaF32 { buf: accum, len: 1 }
    }
'''

NEW_METHOD = r'''    /// Accumulate the global gradient sum-of-squares into a resident fp32 scalar.
    /// The reduction tree and tensor order are fixed: no atomics, bit-identical for
    /// identical bf16 gradients, while remaining asynchronous with respect to host.
    pub fn global_grad_sumsq(&self, grads: &[&CudaMatrix]) -> CudaF32 {
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
            let mut partials = self
                .stream
                .alloc_zeros::<f32>(grid as usize)
                .expect("cuda alloc grad partials");
            let n_a = n;
            let mut stage1 = self.stream.launch_builder(&self.kernels().sumsq_partials);
            stage1.arg(&g.buf);
            stage1.arg(&n_a);
            stage1.arg(&mut partials);
            let cfg1 = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: each block writes exactly partials[blockIdx.x].
            unsafe {
                stage1
                    .launch(cfg1)
                    .expect("launch sumsq_partials_kernel");
            }

            let n_partials = grid as usize;
            let mut stage2 = self.stream.launch_builder(&self.kernels().sumsq_finish);
            stage2.arg(&partials);
            stage2.arg(&n_partials);
            stage2.arg(&mut accum);
            let cfg2 = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: one block reduces the fixed partial sequence and one thread
            // updates accum; grad tensors are launched serially in caller order.
            unsafe {
                stage2
                    .launch(cfg2)
                    .expect("launch sumsq_finish_kernel");
            }
        }
        CudaF32 { buf: accum, len: 1 }
    }
'''


def patch(text: str) -> str:
    if "sumsq_partials_kernel" in text:
        raise SystemExit("chain already B35 patched")
    text = must_replace(text, OLD_KERNEL, NEW_KERNEL)
    text = must_replace(text, "    sumsq: CudaFunction,\n", "    sumsq_partials: CudaFunction,\n    sumsq_finish: CudaFunction,\n")
    text = must_replace(text, '            sumsq: f("sumsq_kernel"),\n', '            sumsq_partials: f("sumsq_partials_kernel"),\n            sumsq_finish: f("sumsq_finish_kernel"),\n')
    text = must_replace(text, OLD_METHOD, NEW_METHOD)
    text = must_replace(
        text,
        '''    /// The global L2 norm `sqrt(Σᵢ ‖gᵢ‖²)` over a set of gradient matrices — for\n    /// gradient clipping. Each grad's sum-of-squares is reduced on-device (fp32) and\n    /// accumulated into one scalar, downloaded once. The atomic accumulation order\n    /// varies run-to-run, but only below the bf16 noise floor (the grads are already\n    /// bf16), so this doesn't add meaningful non-determinism. Returns `+inf`/`NaN`\n    /// faithfully if any grad is non-finite (so the caller can skip the step).\n''',
        '''    /// The global L2 norm `sqrt(Σᵢ ‖gᵢ‖²)` over a set of gradient matrices — for\n    /// gradient clipping. The two-stage fp32 reduction has a fixed tree and tensor\n    /// order, so identical bf16 gradients produce a bit-identical scalar.\n''')
    return text


CHAIN.write_text(patch(CHAIN.read_text()))
print("patched B35 deterministic grad-norm reduction")
