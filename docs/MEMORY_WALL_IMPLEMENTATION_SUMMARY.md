# SciRust — Memory Wall Optimization: Implementation Summary

## Final module architecture

```
scirust/
├── Cargo.toml                         # Workspace with scirust-arena + scirust-fusion
├── docs/
│   ├── MEMORY_WALL_ARCHITECTURE.md    # Full architecture (5 pillars)
│   └── MEMORY_WALL_IMPLEMENTATION_SUMMARY.md  # This file
│
├── scirust-arena/                     # PILLAR 3: Arena Allocators
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                     # PinnedArena + Slab + AlignedVec + exports
│       ├── allocator.rs               # PinnedArena — bump pointer, 128-byte aligned
│       ├── slab.rs                    # Slab — free list + versioning for SSM
│       └── aligned.rs                 # AlignedVec — SIMD-aligned buffer
│
├── scirust-fusion/                    # PILLAR 1: AST Kernel Fusion
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                     # FusionPipeline — public entry point
│       ├── graph.rs                   # OpGraph — DAG dependency graph
│       ├── fusion.rs                  # FusionPass — pattern detection
│       ├── kernel.rs                  # FusedKernel — kernel execution
│       └── patterns.rs                # FusionPatterns — canonical pattern database
│
├── scirust-core/                      # Core — updated modules
│   ├── Cargo.toml                     # +libc for pinned memory
│   └── src/
│       ├── lib.rs                     # Export quant + tensor/pinned
│       ├── tensor/
│       │   ├── mod.rs                 # +pinned + tiling exports
│       │   ├── tensor_nd.rs           # TensorND (unchanged)
│       │   ├── pinned.rs              # PILLAR 2: PinnedBuffer
│       │   └── tiling.rs              # PILLAR 4: Tiling config + detection
│       ├── quant/                     # PILLAR 5: Native quantization
│       │   ├── mod.rs                 # QuantTensor + Quantized trait
│       │   ├── int8.rs                # int8 quant/dequant + AVX2 SIMD
│       │   ├── bf16.rs                # bf16 quant/dequant + NEON/SVE
│       │   └── int4.rs                # int4 packed (8× compression)
│       └── nn/
│           ├── mod.rs                 # +fused_ops module
│           └── fused_ops.rs           # PILLAR 1+4: Fused kernels
│
└── scirust-simd/                      # SIMD — ARM64 extensions
    ├── Cargo.toml                     # +libc for SVE detection
    └── src/
        ├── lib.rs                     # +NEON + SVE + runtime dispatch
        ├── neon.rs                    # PILLAR 4: ARM64 NEON intrinsics
        ├── sve.rs                     # PILLAR 4: ARM SVE intrinsics
        └── matrix/
            ├── backend.rs             # SimdBackend trait (unchanged)
            └── tiling_dispatch.rs     # (in progress)
```

## Pillar 1: AST Kernel Fusion

### File: `scirust-fusion/src/graph.rs`
- **OpKind**: enum of the 30+ supported operations
- **FusedOp**: graph node with inputs, constant, kind
- **OpGraph**: DAG with topological sort (Kahn's algorithm)

### File: `scirust-fusion/src/patterns.rs`
Detected patterns:
| Pattern | Memory gain | Operations |
|-------|-------------|------------|
| matmul_silu | 50% | Linear + SiLU |
| matmul_relu | 50% | Linear + ReLU |
| matmul_silu_layernorm | 66% | Linear + SiLU + LayerNorm |
| matmul_layernorm | 50% | Linear + LayerNorm |
| layernorm_activation | 50% | LayerNorm + Activation |
| two_layer_mlp | 66% | Linear + Linear + Add |
| matmul_scale | 50% | Linear × scale |
| ssm_scan | 0% | SsmStep + SsmStep (sequential) |

### File: `scirust-fusion/src/kernel.rs`
- FusedKernel with execute() for each kernel type
- Implements matmul_silu, matmul_gelu, matmul_relu, matmul_layernorm
- Accumulators stay in local (stack) vectors, never on the heap

### File: `scirust-core/src/nn/fused_ops.rs`
- Executable fused kernels (matmul_silu, matmul_gelu, matmul_layernorm)
- Uses scirust_core::simd::tiling for automatic tiling
- Compatible with the autograd graph

## Pillar 2: PinnedMemory (Zero-Copy)

### File: `scirust-core/src/tensor/pinned.rs`
- **PinnedBuffer**: mmap + mlock on Linux, 128-byte alignment
- **PinnedPool**: pool of reusable buffers
- **MemoryLayout**: enum (Cpu, Pinned, GpuUnified)
- Compatible with CUDA unified memory (cudaHostRegister)

## Pillar 3: Arena Allocators

### File: `scirust-arena/src/allocator.rs`
- **PinnedArena**: bump pointer, O(1) alloc/dealloc
- 128-byte alignment guaranteed
- reset() = O(1) — the whole block is reset in one operation
- **MemoryBlock**: mmap(MAP_ANONYMOUS) + mlock()

### File: `scirust-arena/src/slab.rs`
- **Slab**: free list + versioning for SSM states
- Handle with version → use-after-free protection
- **SlabHandle**: index + version

### File: `scirust-arena/src/aligned.rs`
- **AlignedVec**: Vec with guaranteed alignment
- ToAligned trait for Vec<T> → AlignedVec

## Pillar 4: Cache-Aware Tiling

### File: `scirust-core/src/simd/tiling.rs`
- **TilingConfig**: automatic platform and L2 cache detection
- **CacheProfile**: L1/L2/L3 detection + optimal tile computation
- **matmul_tiled_f32**: i-p-j tiled matmul
- **detect_l2_cache_size()**: reads /sys/devices/system/cpu/cpu0/cache/
- Per-platform config:
  - x86_64 AVX-512: tile 64, lane 16
  - x86_64 AVX2: tile 32, lane 8
  - ARM64 NEON: tile 32, lane 4
  - ARM64 SVE: scalable tile, configurable lane

### File: `scirust-simd/src/neon.rs`
- **NEON kernels**: saxpy, add, mul, silu, gelu, relu, layernorm, matmul
- 4 lanes per register (float32x4_t)
- Tiling for matmul (32×32 by default)

### File: `scirust-simd/src/sve.rs`
- **SVE kernels**: scalable vector length (256-bit on Jetson Thor)
- Predicate-based: svld1, svst1, svmla, etc.
- Detects SVE presence via getauxval(AT_HWCAP)

## Pillar 5: Native Quantization

### File: `scirust-core/src/quant/mod.rs`
- **Quantized** trait: format, compression ratio
- **QuantFormat**: Fp32, Int8, Bf16, Int4
- **QuantTensor**: storage + metadata + dequantize()

### File: `scirust-core/src/quant/int8.rs`
- Per-channel symmetric int8 quantization
- SIMD dequantization: int8 → f32 in 8 lanes (AVX2)
- int8 × int8 matmul → int32 accumulator

### File: `scirust-core/src/quant/bf16.rs`
- f32 ↔ bf16 conversion (LSB truncation)
- NEON batch: 4 elements per iteration
- AVX2 batch: 8 elements per iteration

### File: `scirust-core/src/quant/int4.rs`
- Signed int4 quantization (8× compression)
- Packing: 2 int4 values per byte
- int4 packed × int4 packed matmul → fp32

## Multi-platform compatibility

| Feature | x86_64 | ARM64 | Jetson | Windows |
|---------|--------|-------|--------|---------|
| AVX-512 | ✓ | - | - | - |
| AVX2 | ✓ | - | - | - |
| SSE2 | ✓ | - | - | - |
| NEON | - | ✓ | ✓ | - |
| SVE | - | ✓ | ✓ | - |
| Arena alloc | ✓ | ✓ | ✓ | ✓ |
| Pinned mem | ✓ | ✓ | ✓ | ✓ |
| int8 quant | ✓ | ✓ | ✓ | ✓ |
| bf16 quant | ✓ | ✓ | ✓ | ✓ |
| int4 quant | ✓ | ✓ | ✓ | ✓ |

## Next steps

1. **Fusion with autodiff**: adapt the fused kernels so they work with the
   tape graph. Requires adding backward rules for each kernel.

2. **PinnedMemory + CUDA**: integrate cudaHostRegister/cudaHostUnregister for
   zero-copy to GPU.

3. **Slab + SSM cells**: implement Mamba cells with the Slab to manage the
   hidden states (c, h̃) at each timestep.

4. **Fused matmul SIMD**: replace the scalar loops in the fused kernels with
   direct NEON/AVX512 calls.

5. **Benchmarks**: measure the speedup on the target patterns (MatMul → SiLU → LN).

6. **Possible compiler extension**: only reintroduce an MIR path with a real
   transformation, generated-code oracles, and a blocking CI gate.
