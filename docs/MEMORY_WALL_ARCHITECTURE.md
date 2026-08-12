# SciRust Memory Wall — Architecture of the 5 Pillars

## Overview

This document describes the complete architecture for overcoming the **Memory Wall** (memory-bandwidth bottleneck) in SciRust, targeting ARM64 (Jetson AGX Thor) and x86 (64 cores) architectures.

## Problem statement

In SSMs (Mamba), LLMs, and high-frequency trading:

| Metric | Before | Target |
|----------|-------|-------|
| Allocations/intermediates | 1-3 per layer norm | 0 (fusion) |
| CPU↔GPU copies per matmul | 2 (h2d + d2h) | 0 (zero-copy pinned) |
| Arena alloc latency | O(n) linear scan | O(1) pointer bump |
| L2 cache hit rate (tiled matmul) | ~65% | >95% |
| Effective bandwidth (int8 quant) | ~1× | ~4× |

## Module structure

```
scirust/
├── scirust-arena/                    # Pillar 3: Arena Allocators
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                    # Arena public API
│       ├── allocator.rs              # PinnedArena impl
│       ├── slab.rs                   # Slab allocator for SSM states
│       └── aligned.rs                # AlignedVec + alignment utilities
├── scirust-fusion/                   # Pillar 1: AST Kernel Fusion
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                    # Public API
│       ├── graph.rs                  # OpGraph — dependency graph
│       ├── fusion.rs                 # FusionPass — detection + fusion
│       ├── kernel.rs                 # FusedKernel — code generation
│       └── patterns.rs               # Built-in fusion patterns
├── scirust-core/ (modified)
│   └── src/
│       ├── tensor/
│       │   ├── pinned.rs             # Pillar 2: PinnedMemory
│       │   └── mem_pool.rs           # Pillar 2: MemoryPool
│       ├── simd/
│       │   ├── tiling.rs             # Pillar 4: Cache-Aware Tiling
│       │   └── neon.rs               # Pillar 4: ARM64 NEON/SVE
│       ├── quant/
│       │   ├── mod.rs                # Pillar 5: quantization module
│       │   ├── int8.rs               # Pillar 5: int8 quant + dequant SIMD
│       │   ├── bf16.rs               # Pillar 5: bf16 <-> f32
│       │   └── int4.rs               # Pillar 5: int4 unpacking
│       └── nn/
│           └── fused_ops.rs          # Pillar 1: Fused matmul+silu+layernorm
└── scirust-simd/ (modified)
    └── src/
        ├── sve.rs                    # Pillar 4: ARM SVE intrinsics
        └── matrix/tiling_dispatch.rs # Pillar 4: Tiling dispatch
```

## Per-pillar details

### Pillar 1: AST Kernel Fusion

**Goal**: Avoid RAM round-trips by fusing consecutive operators.

**Detection algorithm**:
1. Build an `OpGraph` from the MIR (via the rustc driver) or from the forward pass (via the tracing runtime)
2. Look for canonical patterns:
   - `MatMul → SiLU` (linear activation)
   - `MatMul → SiLU → LayerNorm` (MLP block)
   - `MatMul → LayerNorm` (pre-LN transformer)
   - `MatMul → MatMul → Add` (two-layer MLP)
   - `Conv2d → ReLU → Pool` (conv block)
3. For each detected pattern, generate a `FusedKernel` that:
   - Computes mean/var in a single pass (LayerNorm)
   - Applies SiLU/GELU with no intermediate
   - Accumulates matrix products in accum registers

### Pillar 2: Unified Memory (Zero-Copy)

**PinnedMemory** — 64-byte aligned memory, pinned in user space:
- On ARM64 (Jetson): uses `mmap(MAP_ANONYMOUS | MAP_POPULATE)` with `mlock()`
- On x86: uses `posix_memalign` + `mlock()`
- Compatible with CUDA Unified Memory (`cudaHostRegister`) and GPU Direct

**MemoryPool** — fixed-size tensor pool:
- Reduces fragmentation for constant-size batches
- Reuses already-allocated blocks

### Pillar 3: Arena Allocators

**PinnedArena** — bump-pointer allocation:
- Pre-allocates one large block (128-byte aligned)
- `alloc::<T>()` = pointer bump (O(1))
- `reset()` = pointer reset (O(1))
- **No Drop, no free** — all allocations are deallocated together

**Slab** — for SSM states:
- Stores the hidden states (c, h̃) of Mamba cells
- Indexed access (O(1))
- Supports mark-and-sweep garbage collection for variable-length sequences

### Pillar 4: "Cache-Aware" SIMD Auto-Vectorization

**Tiling** — for matmul:
- Analyzes the L2 cache size of the target machine
- Adapts the tile sizes so they fit in L2
- x86: AVX-512 (16x f32 per tile), AVX2 (8x f32)
- ARM64: NEON (4x f32), SVE (scalable vector length)

**Cache profiling**:
- Detects L2 size at runtime via `/sys/devices/system/cpu/cpu0/cache/`
- On Jetson AGX Thor: L2 = 4MB per cluster
- On x86: L2 ≈ 256KB-1MB per core

### Pillar 5: Native Quantization Primitives

**QuantizedTensor** — quantized storage:
- `int8`: 4× compression (f32 → int8)
- `bf16`: 2× compression (f32 → bf16)
- `int4`: 8× compression (f32 → int4 packed)

**On-the-fly decompression**:
- int8 → f32 in SIMD registers (AVX2: 8-lanes, NEON: 4-lanes)
- bf16 → f32: direct conversion
- int4 packed → int8 → f32: unpack + sign-extend

**Computing in quantized form**:
- int8 × int8 → int32 (accumulate in int32)
- Fused dequant + matmul: dequantize directly into the product

## Compatibility

| Platform | SIMD | Arena | Fusion | Quant | Pinned |
|------------|------|-------|--------|-------|--------|
| x86_64 (AVX-512) | AVX-512 | ✓ | ✓ | ✓ | ✓ |
| x86_64 (AVX2) | AVX2 | ✓ | ✓ | ✓ | ✓ |
| ARM64 (NEON) | NEON | ✓ | ✓ | ✓ | ✓ |
| ARM64 (SVE) | SVE | ✓ | ✓ | ✓ | ✓ |
| Jetson AGX Thor | NEON+SVE | ✓ | ✓ | ✓ | ✓ |
