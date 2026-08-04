# Exclusions (Profile 1.0)

Operators and behaviors deliberately excluded from SciRust Tensor Parity
Profile 1.0, with reasons. A row here can move into scope in a later profile
by writing the profile change first (this file + SCOPE.md) and then the code.

## By dtype / data class

| Exclusion | Reason | Re-entry trigger |
| --- | --- | --- |
| Complex dtypes (`complex64/128`) | No complex storage in any tensor type; would require new storage paths everywhere | A storage layer with `Complex` dtype |
| Integer arithmetic kernels (`add/mul/...` on `int64/uint8/...`) | `i64` used only as indices; integer promotion rules not implemented | Registry rows with integer dtypes |
| Quantized dtypes (qint8/quint8/qint32 + observers) | No quantized tensor type; `fake_quantize_ste` exists in autodiff only | A `QuantizedTensor` type |
| `float8_*`, `float16` arithmetic (storage-only bf16) | Hardware-dependent numerics; bf16 is storage/promotion-only in v1 | bf16 arithmetic rows with tolerances |

## By operator family

| Exclusion | Reason |
| --- | --- |
| `torch.sparse` ops (spmv/spmm/solve autograd) | Sparse storage exists (`CooMatrix`/`CsrMatrix`/`CscMatrix`, `SparseLu` solve) but no sparse autograd or sparse-sparse kernels in v1 |
| `torch.linalg` beyond what exists (no `svd`/`eig` general, no `qr`/`lstsq` public) | `lanczos_eigen_symmetric` exists in simd but is experimental; no public parity surface |
| `torch.fft` (all) | No FFT implementation anywhere in the workspace |
| `torch.fft` / `torch.signal` / `torch.fft` window functions | no kernel |
| Distributions (`torch.distributions`) | Out of scope by program charter (tensor ops only) |
| RNG stream bit-exactness | `set_seed` exists and is deterministic; stream equality with torch MT19937 is not required (see SCOPE.md) |
| In-place ops and `out=` variants | Registry only covers out-of-place forms where not already present |
| `torch.compile` / export / AOTAutograd | No graph compiler compatibility target in v1 |
| Sparse autograd (all) | see sparse row above |

## By device

| Exclusion | Reason |
| --- | --- |
| CUDA parity claims | `scirust-cuda`/`scirust-gpu` kernels tracked as `experimental` until a differential run exists |
| MPS / other backends | Not implemented |

## By behavior

| Exclusion | Reason |
| --- | --- |
| Double backward / higher-order autograd | `Tape`/`NdTape` are first-order; `Dual` is forward-mode-only |
| Non-deterministic reduction order (CPU) | SciRust *guarantees* deterministic order where rows say so; where PyTorch is non-deterministic SciRust keeps determinism and the row says `deterministic = "stricter-than-torch"` |
| NaN propagation nuances | Tolerances cover finite domain; NaN semantics documented per row, not per-torch |
| Thread-count-dependent results | Multi-threaded runs pin the thread count in tests; results must not depend on it |
| bf16 promotion | storage-only in v1: any arithmetic op on bf16 storage either promotes to f32 (documented) or the row is `unsupported` |

## Governance

Exclusions change only by PR editing this file AND `SCOPE.md` together with a
written rationale and (for in-scope moves) a profile row addition. The CI
coverage gate fails if a registry row claims `parity` for an excluded
(operator, dtype, device) combination.
