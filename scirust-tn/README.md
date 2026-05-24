# scirust-tn — Tensor Networks for SciRust

Tensor-Train (TT) decomposition and Matrix Product State (MPS) primitives that
integrate with `scirust-core`'s tape-based autograd.

## What this crate gives you

- **`auto_factorize(n, d)`** — balanced integer factorization helper.
- **`TensorND`** — minimal N-dimensional tensor for offline decomposition logic.
- **`truncated_svd(...)`** — truncated SVD via nalgebra (Phase 1 CPU).
- **`tt_decompose_tensor(...)`** — Oseledets TT-SVD on generic d-mode tensors.
- **`tt_decompose_matrix(...)`** — Novikov 2015 TT-Linear decomposition of a
  weight matrix `(in, out)` into `d` cores of shape `(r_k, I_k * O_k, r_{k+1})`.
- **`TTLinear`** *(feature `core`)* — drop-in replacement for
  `scirust_core::nn::Linear` whose weight is stored as TT-cores. Implements
  `Module`.
- **`tt_decompose(linear, ...)`** *(feature `core`)* — manual factorization API.
- **`tt_decompose_auto(linear, n_factors, ...)`** *(feature `core`)* —
  automatic balanced factorization API.

## Quick start

```rust
use scirust_core::nn::Linear;
use scirust_tn::tt_decompose_auto;

let linear = Linear::new(768, 3072);          // a transformer FFN projection
let tt = tt_decompose_auto(&linear,
                           /*n_factors=*/ 3,
                           /*max_rank=*/ 32,
                           /*tolerance=*/ 1e-4);
println!("compression: {:.2}x   params: {} → {}",
         tt.compression_ratio(),
         tt.dense_params(),
         tt.num_params());
```

## Building

```bash
# Standalone (decomposition algorithms only, no scirust-core needed):
cargo check
cargo test

# With autograd-integrated TTLinear (requires scirust-core in workspace):
cargo check --features core
cargo test --features core

# Benchmark on transformer-style shapes:
cargo run --release --example transformer_compress --features core
```

## Phase 1 scope and limitations

### What works
- Decomposition of `Linear` layers into TT-cores with controllable rank/tolerance.
- Memory compression: 8-30× typical for transformer FFN projections.
- Inference through `TTLinear::forward` (reconstructs `W` from cores, then `x @ W + b`).
- Re-decomposition workflow: train a dense `Linear`, periodically call
  `tt_decompose` to refresh the TT representation.

### What's deferred to Phase 2
- **Native TT contraction during forward** (compute savings, not just memory).
  Requires a tensor-permutation op on `Var` to handle the in/out interleaving
  inherent in the Novikov 2015 layout. Once `Var::permute` lands in
  `scirust-core`, `TTLinear::forward` can do `d` sequential matmuls of much
  smaller shape instead of reconstructing the full `W`.
- **TT-aware training** (autograd through the cores). Depends on the same
  permute op.
- **GPU backend**. CPU-only via `nalgebra` for now. Once `cudarc` 0.19+ is
  wired into `scirust-gpu` with CUDA 13 / sm_110 support, this crate gets a
  feature flag for GPU SVD via cuSOLVER `gesvdj`.

## Architectural decisions worth knowing

1. **`Tensor` stays 2D-only in scirust-core.** The TN code never asks for
   N-dimensional tensors from the core. Instead, `TensorND` (this crate's
   own struct) handles offline decomposition logic; on the autograd side
   everything goes through 2D `Tensor` + `Var::reshape(&[usize])`. Net effect:
   zero changes required in `scirust-core` to add tensor-network support.
2. **Interleaved layout.** TT-Linear treats `W (in, out)` as a tensor of
   shape `(I_0, O_0, I_1, O_1, ..., I_{d-1}, O_{d-1})`, with each `(I_k, O_k)`
   pair grouped into a single mode of size `I_k * O_k`. This is the layout
   from Novikov et al. 2015 ("Tensorizing Neural Networks", NeurIPS).
3. **Full SVD then truncate.** Phase 1 uses nalgebra's full SVD and truncates
   afterwards. For Thor-scale weights (≤ 4096 × 4096) this is fine; for
   larger matrices a randomized SVD (Halko 2011) would be more efficient.

## Test coverage

```
38 tests total:
  30 unit tests
   4 round-trip integration tests
   4 forward-match integration tests
```

All passing. Run `cargo test --features core` to verify on your environment.

## References

- I. V. Oseledets. *Tensor-Train Decomposition*. SIAM J. Sci. Comput. 33(5), 2011.
- A. Novikov et al. *Tensorizing Neural Networks*. NeurIPS 2015.
- N. Halko, P.-G. Martinsson, J. A. Tropp. *Finding structure with randomness*. SIAM Rev. 53(2), 2011.

## License

MIT OR Apache-2.0
