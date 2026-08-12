# Architecture — `scirust-simd`

This document maps the `scirust-simd` crate: a **portable BLAS + Transformer
foundation**, from the x86_64 datacenter to embedded ARM, built around a single
guiding idea — **one binary, backend selection at runtime, guaranteed scalar
fallback**.

It is the fruit of an incremental effort (≈ 20 PRs) starting from simple
`add`/`mul` kernels and ending with a complete inference **and** training
pipeline.

## Toolchains and SIMD features

The default build is compatible with Rust stable and with the **MSRV 1.89**. It
includes the scalar fallback, SSE2/AVX2/AVX-512 on x86_64 and NEON on aarch64.
Extensions that still use unstable `core::arch` APIs are explicit:

- `nightly-simd` enables Intel AMX, ARM SVE/SME and the ARM int8 paths
  `dotprod`/`i8mm`; this feature requires the nightly pinned by the repository;
- `portable-simd` enables the generic `std::simd` kernels and also requires the
  pinned nightly.

Both features are disabled by default. They have their own CI jobs; they
therefore are not part of the stable MSRV contract.

---

## 1. The guiding thread: runtime dispatch + portability

All the performance rests on a **single abstraction layer** that selects, on the
first call, the best instruction set available on the current CPU, then caches
it:

```
                 detect_backend()  (OnceLock, amortized cost = 1 atomic load)
                        │
     ┌──────────┬───────┴────┬──────────┬──────┬──────┬──────────┐
   AVX-512    AVX2/FMA      SSE2       SVE    NEON        Scalar
  (x86_64)   (x86_64)     (x86_64)  (aarch64)(aarch64)   (everywhere)
```

- **x86_64**: `AVX-512F` (+ `VNNI`/`BW` depending on the kernels) → `AVX2+FMA` →
  `SSE2`. Detection via `std::is_x86_feature_detected!`.
- **aarch64**: **`NEON`** (ARMv8 baseline) on stable. With `nightly-simd`,
  **`SVE`** (scalable `saxpy`/`sdot`/`sscal`/`sgemm` kernels, runtime vector
  length — Graviton 3/A64FX/Neoverse V2) is chosen when present, and the int8
  `dotprod`/`i8mm` paths are enabled. The backend is chosen by
  `detect_backend()` (SVE before NEON) and serves the whole `SimdBackend`
  trait.
- **Everywhere else**: scalar path, always correct.

Consequence: **the same source code** runs from an AVX-512 server to a Jetson /
Raspberry Pi / Rockchip RK3588, without conditional recompilation by the
developer. The scalar fallback is the **correctness reference** against which
all vector kernels are tested.

Modules involved: [`dispatch`](src/dispatch.rs), [`matrix`](src/matrix/),
[`portable`](src/portable.rs).

---

## 2. Layer map

From the bottom (silicon) to the top (application):

| Layer | Module(s) | Content |
|---|---|---|
| **Dispatch / backends** | `dispatch`, `matrix`, `portable` | CPU detection, `SimdBackend` trait (saxpy/sdot/…), AVX-512/AVX2/SSE2/NEON/scalar backends; **SVE** with `nightly-simd` |
| **BLAS — GEMM** | `gemm` | Tiled/packed SGEMM (`f32`) & DGEMM (`f64`), multi-threaded, fused GEMM `act(A·B+b)` |
| **Activations** | `activations` | Vectorized `exp` (range-reduction + `scalef`) → `sigmoid`/`tanh`/`GELU`/`SiLU` |
| **Quantization** | `quant` | int8 dot `u8·i8→i32` (VNNI and ARM `i8mm` USDOT with `nightly-simd`), `i8·i8→i32` (ARM `dotprod` SDOT with `nightly-simd` / AVX-512BW), bf16 (native `avx512bf16` or widening) — **cross-arch** |
| **Matrix accelerators** | `amx`, `sme` | With `nightly-simd`: tiled **Intel AMX** GEMM int8 (`_tile_dpbssd`) & bf16 (`_tile_dpbf16ps`); **ARM SME** (probe + rank-1 reference, awaiting ZA intrinsics) |
| **Scalable vectors** | `sve` | With `nightly-simd`: SVE kernels `saxpy`/`sdot`/`sscal` **and packed register-blocked SGEMM** (`MR×VL` tile, C amortized over K), runtime vector length (aarch64) |
| **Advanced x86 kernels** | `x86_ext` | `k` masks (conditional axpy), NT-stores, software prefetch |
| **Attention** | `attention`, `kv_cache`, `qkv_cache` | naive, **flash**, **causal**, **multi-head**, **KV cache** (`f32` **and int8** ÷4 memory) |
| **Normalizations** | `norm` | RMSNorm, LayerNorm (vectorized), RoPE |
| **Assembly** | `transformer`, `model` | Pre-norm decoder block (prefill **+** decode `f32` **and int8**), multi-layer model + generation (`generate_hidden`/`_quant`) |
| **Training** | `grad` | Backward of all kernels, validated by **gradcheck** |
| **Application** | `scirust-learning::simd_nn` | Trainable `DenseLayer`/`Mlp`, **AdamW** optimizer |

---

## 3. BLAS: the GEMM, performance core

The matrix product feeds everything else (projections, FFN, backward). Three
properties make it fast:

1. **Cache blocking** (`MC`/`KC`/`NC`) BLIS-style: the worked panels fit in
   L2/L1.
2. **Explicit packing** of `A` and `B` into contiguous buffers → the micro-kernel
   reads them in **unit stride** (optimal hardware prefetch).
3. **8×16 register-blocked micro-kernel**: 8 `zmm` accumulators held in
   registers across the whole `KC` dimension; edges handled by `k` mask.
4. **Parallelism**: splitting of the `M` dimension into disjoint blocks of rows
   via `std::thread::scope` (no external dependency).

The **same packed GEMM is ported to NEON** (aarch64): identical structure
(`MC_N`/`KC_N`/`NC_N` blocking, packing of `A`/`B`), `8×8` register tile (16
`float32x4_t` accumulators, buffer epilogue because NEON has no masked store).
Consequence: **the whole stack — Transformer and training — goes from scalar to
vectorized on embedded ARM** (Jetson, Raspberry Pi, RK3588), and no longer only
the `saxpy`/`sdot` primitives. The kernel is validated bit-to-tolerance against
the scalar **under `qemu-aarch64`** (real execution, not just compilation).

### Measured performance (AVX-512 machine, 4 cores)

| Kernel | Throughput | Gain vs naive |
|---|---:|---:|
| SGEMM 1024³, 1 thread | 56.8 GFLOP/s | ~84× |
| SGEMM 1024³, 4 threads | 110 GFLOP/s | ~163× |
| DGEMM 1024³, 4 threads | 127 GFLOP/s | — |
| Fused dense layer 4096×1024×1024 (ReLU) | 53.9 GFLOP/s | ~86× |
| **AMX int8** 512³ (`_tile_dpbssd`, silicon) | 47.5 GOP/s | ~23× |
| **AMX bf16** 512³ (`_tile_dpbf16ps`, silicon) | 44 GFLOP/s | ~21× |
| **int8 W8A8 decoder block** (s=128, d=1024, d_ff=4096) | ×1.97 vs `f32` | weights ÷4, RMS error 0.01 % |

The **fused GEMM** (`sgemm_bias_act`) computes `act(α·A·B + bias)`: `A·B` by the
tiled GEMM (any `k`), then a vectorized bias+activation epilogue in a single
`O(m·n)` pass. Its **int8 quantized** analogue
([`amx::qlinear_i8`](src/amx.rs)) dequantizes `X·W` (AMX GEMM) per channel +
bias.

The **quantized int8 (W8A8) decoder block** ([`qtransformer`](src/qtransformer.rs))
routes the six projections (`Wq`/`Wk`/`Wv`/`Wo`/`W1`/`W2`) onto the AMX GEMM:
weights quantized **per channel** and **pre-packed** once into tile layout
(`amx::prepack_b_i8`, VNNI packing out of the hot path), activations quantized
**per token** at runtime. Measured result (silicon): faster than `f32`,
weights ÷4, faithful output (the non-quantized residual dilutes the int8 noise)
— cf. [`examples/qtransformer_bench.rs`](examples/qtransformer_bench.rs).

The AMX figures are measured **on AMX silicon** (the machine exposes
`amx_tile`/`amx_int8`/`amx_bf16`) — cf. [`examples/amx_bench.rs`](examples/amx_bench.rs).
Two packing optimizations: (1) tile buffers are only zeroed for partial `K`
blocks (full ones rewrite them); (2) `A` panels are packed **once per block of
rows** then reused over all `N` panels (eliminates the `n/16`× redundancy,
~+25 % throughput), and the static weights are **pre-packed** out of the hot
path (`prepack_b_i8`). AMX GEMMs are **single-threaded by design**: the
multi-threaded variant corrupts the tile state on context switch on a
virtualized platform (~0.1 %); reliable parallelism goes through the `f32` GEMM
[`sgemm_parallel`](src/gemm.rs). See also
[`examples/bench.rs`](examples/bench.rs).

---

## 4. Transformer pipeline

All the bricks of a decoder block, chainable:

```
RMSNorm → Q,K,V = proj(·)      (tiled GEMM)
        → RoPE(Q,K)            (per head)
        → Causal multi-head attention
        → + proj·Wo  (residual)
RMSNorm → FFN : SiLU(·W₁+b₁)·W₂ (fused GEMM + GEMM)
        → + (residual)
```

- **Attention** ([`attention`](src/attention.rs)): naive, **flash** (online
  softmax, `O(d)` memory per query), **causal** (triangle, ~2× less work),
  **multi-head** versions.
- **KV cache** ([`kv_cache`](src/kv_cache.rs)): incremental autoregressive
  decoding, `O(t·d)` per token instead of recomputing the prefix.
- **Block & model** ([`transformer`](src/transformer.rs),
  [`model`](src/model.rs)): two regimes — *prefill* (whole sequence) and
  *decode* (token by token via cache). A key invariant guarantees their
  consistency: **`prefill ≡ decode`** line by line, propagated through the
  whole stack.

---

## 5. Training

[`grad`](src/grad.rs) provides the **backpropagation** of all kernels:

- `linear_backward` (reuses the tiled GEMM), `relu`/`silu`/`gelu_backward`,
  `rmsnorm`/`layernorm_backward`, `softmax_backward`, and
  **`attention_backward`** (chain `dV = Pᵀ·dO`, `dScores = softmax'`, `dQ`, `dK`).

On the application side, [`scirust-learning::simd_nn`](../scirust-learning/src/simd_nn.rs):
trainable `DenseLayer` and `Mlp` (fused forward + chained backward), and the
**`AdamW`** optimizer (moments, bias correction, decoupled weight decay).

---

## 6. Testing philosophy

Every claim is verified mechanically:

- **Vector correctness**: each SIMD kernel is compared to the scalar fallback
  (often bit-for-bit or with a tight tolerance), on all lengths (including the
  masked `1..15` epilogues).
- **Gradients**: all backward passes are validated by **centered finite
  differences** (gradcheck) against an independent reference forward.
- **Structural equivalences**: `prefill ≡ decode` (block and stack),
  `incremental ≡ batch` (KV cache), `AdamW < SGD` at equal budget.
- **Portability**: the workspace compiles under
  `RUSTFLAGS="-D warnings" cargo +nightly-2026-07-02 check --workspace --all-targets --features scirust-simd/nightly-simd --target aarch64-unknown-linux-gnu`.

---

## 7. Module index

| Module | Role |
|---|---|
| [`dispatch`](src/dispatch.rs) | CPU detection + arch-specific backends |
| [`matrix`](src/matrix/) | `SimdBackend` trait, matrix views |
| [`gemm`](src/gemm.rs) | Tiled, parallel, fused SGEMM/DGEMM |
| [`activations`](src/activations.rs) | Vectorized `exp`/`sigmoid`/`tanh`/`GELU`/`SiLU` |
| [`quant`](src/quant.rs) | int8 (VNNI/USDOT/SDOT), bf16 mixed-precision — cross-arch x86/ARM |
| [`amx`](src/amx.rs) | `nightly-simd` — tiled Intel AMX GEMM, int8/bf16, pre-packed weights, `qlinear_i8` (x86) |
| [`qtransformer`](src/qtransformer.rs) | `nightly-simd` — **quantized int8 W8A8** decoder block (AMX projections) (x86) |
| [`sve`](src/sve.rs) | `nightly-simd` — scalable SVE kernels `saxpy`/`sdot`/`sscal` + packed `MR×VL` SGEMM (aarch64) |
| [`sme`](src/sme.rs) | `nightly-simd` — ARM SME, probe + rank-1 reference (ZA accumulator), aarch64 |
| [`x86_ext`](src/x86_ext.rs) | `k` masks, NT-stores, prefetch (x86) |
| [`attention`](src/attention.rs) | Naive/flash/causal/multi-head attention |
| [`kv_cache`](src/kv_cache.rs) | `f32` KV cache, incremental decoding |
| [`qkv_cache`](src/qkv_cache.rs) | **int8** KV cache (÷4 memory, int8 dot VNNI/SDOT scores) |
| [`norm`](src/norm.rs) | RMSNorm, LayerNorm, RoPE |
| [`transformer`](src/transformer.rs) | Decoder block (prefill + decode) |
| [`model`](src/model.rs) | Multi-block model + generation |
| [`grad`](src/grad.rs) | Backward of all kernels (gradcheck) |
| [`complex`](src/complex.rs) | SIMD complex arithmetic |

---

*This document is alive: it tracks the evolution of the crate. The guiding
thread stays constant — fine-grained mastery of x86_64 hardware **and**
coverage of the whole platform grid, behind a single abstraction.*
