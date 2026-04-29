# GPU — Activate GPU compute

## Goal

Move expensive operations (Conv2d, large matmuls) to GPU using one line of code,
without rewriting your model.

## Prerequisites

- A GPU compatible with WebGPU (most modern GPUs since 2020)
- Build with `--features wgpu`

## Step 1 — Enable the feature

```toml
# Cargo.toml
[dependencies]
scirust-gpu = { path = "...", features = ["wgpu"] }
```

```bash
cargo run --release --features wgpu
```

## Step 2 — Initialize the GPU context

```rust
use std::sync::Arc;
use scirust_gpu::gpu_tensor::GpuContext;
use scirust_gpu::gpu_conv::ConvGpuPipelines;
use scirust_gpu::global::set_global_gpu_context;

let ctx = GpuContext::try_init()
    .expect("No GPU adapter found");

// Build pipelines once (compilation is expensive)
let conv_pipelines = Arc::new(ConvGpuPipelines::build(&ctx.device));

// Install the global context (used by the autograd backward pass)
set_global_gpu_context(ctx.clone()).unwrap();
```

## Step 3 — Activate GPU on a layer

```rust
use scirust_core::nn::conv2d::Conv2d;

let conv = Conv2d::new(in_c, out_c, 3, 1, Padding::Same,
    &KaimingNormal, Some(&Zeros), &mut rng)
    .input_dims(32, 32)
    .on_gpu(ctx.clone(), conv_pipelines.clone());   // ← that's it
```

The `.on_gpu()` builder activates GPU routing for this layer:
- The forward pass runs on GPU (im2col + sgemm + reshape)
- The im2col tensor is cached in VRAM for backward
- Output is returned to CPU automatically (compatible with CPU-only layers
  in the rest of the model)

## Three GPU modes

The `Conv2d` module supports three backends:

```rust
use scirust_core::nn::conv2d::ConvBackend;

// Default: CPU only
let conv = Conv2d::new(...);

// GPU forward, output downloaded to RAM (compatible with CPU layers after)
let conv = Conv2d::new(...).on_gpu_descend(ctx, pipelines);

// GPU forward, output stays in VRAM (for chains: Conv → ReLU GPU → Conv)
let conv = Conv2d::new(...).on_gpu(ctx, pipelines);

// Switch back to CPU
let conv = conv.on_cpu();
```

## Building a full GPU pipeline

To minimize CPU↔GPU transfers, keep activations in VRAM between layers:

```rust
use scirust_gpu::gpu_elementwise::{ElementwisePipelines, relu_gpu};

let ew_pipelines = Arc::new(ElementwisePipelines::build(&ctx.device));

let mut conv1 = Conv2d::new(...).on_gpu(ctx.clone(), conv_pipelines.clone());
let mut conv2 = Conv2d::new(...).on_gpu(ctx.clone(), conv_pipelines.clone());

let tape = Tape::new();
let xv = tape.input(x_cpu);
let xv_gpu = xv.to_gpu(&ctx);                          // upload once
let h1 = conv1.forward(&tape, xv_gpu);
let h1_relu = h1.relu_gpu(&ctx, &ew_pipelines);        // GPU → GPU
let h2 = conv2.forward(&tape, h1_relu);
let h2_relu = h2.relu_gpu(&ctx, &ew_pipelines);        // GPU → GPU
let out = h2_relu.to_cpu(&ctx);                        // download once
```

This is **Pattern 3** in our architecture: the activations live in VRAM,
and only the input/output cross the bus.

## When GPU is *slower*

GPU overhead (kernel launch, memory transfers) means GPU is **not always faster**:

| Workload | GPU benefit |
|---|---|
| Conv2d on small images (8×8, batch 4) | Probably slower than CPU |
| Conv2d on medium images (32×32, batch 32) | 2-5× faster |
| Conv2d on large images (224×224, batch 64) | 10-50× faster |
| MatMul on small matrices (< 256×256) | Marginal benefit |
| MatMul on large matrices (1024×1024+) | Massive speedup |

Rule of thumb: GPU is worth it when the compute per byte transferred is high.

## Limitations

- **Backward pass**: forward runs on GPU, but backward currently performs
  some matmul operations on CPU. Forward speedup is realized; full
  end-to-end speedup is partial.
- **Single GPU only**: the global context system assumes one GPU. Multi-GPU
  support is on the roadmap.
- **No fp16**: all GPU compute is f32. Mixed precision is planned for v10.

## Troubleshooting

**`No GPU adapter found`** — Your system doesn't have a wgpu-compatible
adapter. Check that you have a recent enough graphics driver. On headless
servers, you may need to set up a software adapter or skip GPU.

**`set_global_gpu_context` panic** — You've installed it twice. The global
context is set once per program (it's a `OnceLock`). For testing scenarios
where you need different contexts, structure your tests as separate processes.

**Outputs differ from CPU** — Should be < 1e-3 relative error. If significantly
larger, file an issue with a minimal reproducer.

## Performance tips

1. **Build pipelines once** at startup, share via `Arc`. Compiling shaders
   is expensive (~ms per pipeline).
2. **Batch as much as possible** — GPU loves big batches, hates many small ones.
3. **Use `--release`** — Debug builds disable optimization in `wgpu`.
4. **Profile transfers**, not just compute — `cargo run --features wgpu --release`
   and look at the bench output, which counts CPU↔GPU transfers explicitly.
