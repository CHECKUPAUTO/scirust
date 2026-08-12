# Coverage audit — RepDL (Microsoft) vs SciRust

> Date: 2026-07-10 · Branch: `claude/repdl-scirust-audit-81gcb8`
> Subject: verify that SciRust covers the features of
> [microsoft/RepDL](https://github.com/microsoft/RepDL) (semi-concurrent
> library), close the closable gaps, and guarantee **zero copyright risk** in
> the process.

---

## 1. Executive summary

**Verdict: near-complete functional coverage, in a different and documented
guarantee regime.** Of the 23 public API elements of RepDL, SciRust already
covered 18 before this audit (operations, layers, gradients, optimizer,
trainable examples). Three real, closable gaps are **closed by this PR**
(AMSGrad, SHA-256 hashing of tensors/parameters, exp/ln via `f64` promotion).
Two elements are **not applicable by design** (conversion of PyTorch modules)
or **covered by composition** (4D reductions).

**Copyright risk: nil.**
- No RepDL code exists in this repository (§3); the only occurrences of
  "RepDL" are documentary citations (scientific positioning, workstream 108).
- This audit was conducted on **specification only**: API surface, README and
  arXiv abstract; the three added implementations derive from published
  algorithms (Reddi et al. 2018; FIPS 180-4 via the `sha2` crate already a
  dependency; double promotion — folkloric technique), not from RepDL code.

The only axis where RepDL remains objectively stronger — f32
**cross-platform** reproducibility via correct rounding — was already
identified and recorded as future work in `paper/RELATED_WORK.md`
(workstream 108). This PR makes an honest step toward it (§6.3) without
over-promising.

---

## 2. RepDL fact sheet (state as of 2026-07-10)

| Field | Value |
|---|---|
| Repository | github.com/microsoft/RepDL (≈13 commits, research project) |
| License | **MIT** |
| Reference | Xie, Zhang & Chen, *RepDL: Bit-level Reproducible Deep Learning Training and Inference*, arXiv:2510.09180 (2025) |
| Nature | **PyTorch** (Python) wrapper + C++ (OpenMP) and CUDA backends |
| Promise | **Bit-identical cross-platform results** (training and inference), f32 only |
| Techniques | (a) fixed floating-point operation order (sequential summations, fixed-accumulation-order GEMM, `fmaf`); (b) avoidance of non-IEEE-754 instructions; (c) "correctly rounded" math functions via **double promotion** (`exp`, `log`, `sqrt` computed in f64 then rounded to f32) |
| Claimed limits | "Only a subset of functions and modules is available"; no low precision (bf16/int8); report without numerical evaluation |

Complete public API surface:

- `repdl.ops`: `mm` (optional transposes), `div`, `sqrt`, `softmax`,
  `sum1d`, `sum2d_dim0`, `sum2d_dim1`, `sum4d_dim023`, `conv2d`,
  `conv2d_grad_input`, `conv2d_grad_kernel`, `cross_entropy`
- `repdl.func` (with backward autograd): `expand_as`, `mean1d`,
  `mean2d_dim0`, `mean4d_dim023`
- `repdl.nn`: `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`,
  `CrossEntropyLoss` (+ corresponding `nn.functional`)
- `repdl.optim`: `Adam` (AMSGrad option)
- `repdl.from_torch_module(m)`: recursive conversion of a PyTorch module
- `repdl.utils`: `get_hash` / `print_hash` (SHA-256 of a tensor or of a
  module's parameters — the reproducibility-verification tool)

## 3. Methodology and intellectual property

**Sources consulted**: repository page, README, tree, API inventories
(signatures + semantics in prose), arXiv abstract. The backend algorithms
were characterized **in prose** (accumulation order, double promotion) to
locate the guarantee class — no code was copied, translated or adapted.

**Finding on the SciRust repository** (exhaustive search):

- `grep -ri "repdl|sum2d_dim|sum4d_dim|mean4d_dim|from_torch_module"` over the
  whole tree: **0 match in source code**. The 7 touched files are all
  documentary (`README.md`, `CHANGELOG.md`, `LIVESTATE.md`,
  `paper/RELATED_WORK.md`, `paper/PAPER_PLAN.md`, `docs/INDUSTRIAL_ROADMAP.md`,
  `docs/DOSSIER_FINANCEURS.md`) and fall under **scientific citation**
  (allowed and desirable — it is the honesty work of workstream 108).
- The pre-existing SciRust implementations (GEMM, conv2d im2col/col2im,
  Demmel–Nguyen/Shewchuk summation, Kahan, PCG…) are architecturally
  different from RepDL and predate this audit: **no derivative-work risk**.

**License positions**: RepDL is MIT-licensed — reusing code would be *legal*
provided the Microsoft copyright notice is preserved, but it would create an
attribution obligation in a PolyForm Noncommercial repository and a
confusion risk. **Policy retained (zero risk): never copy or translate RepDL
code; reimplement only from public specs/papers.** This PR complies; any
future contribution touching reproducibility should follow the same rule.

## 4. Coverage matrix, element by element

Statuses: ✅ covered · ✅➕ covered by composition · 🆕 closed by this PR ·
Ⓝ not applicable by design.

| RepDL API | SciRust equivalent | Status | Proof |
|---|---|---|---|
| `ops.mm` (transA/transB) | `Var::matmul` + `Op::MatMul`; GEMM with transposition flags (internal) | ✅ | `scirust-core/src/autodiff/reverse.rs:31-43,412,868-892,1253` |
| `ops.div` | `Op::Div` / `Op::DivBroadcast` (autograd) — IEEE-754 division is correctly rounded by the standard | ✅ | `reverse.rs:606,610,1065,1202` |
| `ops.sqrt` | `Op::Sqrt` (autograd) — IEEE-754 `sqrt` correctly rounded by the standard (RepDL goes through f64, identical result) | ✅ | `reverse.rs:620,1410` |
| `ops.softmax` | `Op::Softmax` (+ `LogSoftmax`) 2-D, and last-axis softmax on the N-D tape | ✅ | `reverse.rs:383,649,1648`; `nd.rs:190` |
| `ops.sum1d` | `Op::Sum` | ✅ | `reverse.rs:639,1534` |
| `ops.sum2d_dim0/dim1` | `Op::SumAxis(axis ∈ {0,1})` | ✅ | `reverse.rs:640,1539` |
| `ops.sum4d_dim023` | composition `transpose().reshape([C, N·H·W])` + axis-1 reduction (real use: per-channel BatchNorm2d statistics) | ✅➕ | `scirust-core/src/nn/batch_norm_2d.rs:86-113` |
| `ops.conv2d` | `Op::Conv2dForward` (+ `ConvTranspose2d`) | ✅ | `reverse.rs:724-736`; `nn/conv2d.rs:120` |
| `ops.conv2d_grad_input` | `Conv2dForward` backward: `dcol = Wᵀ·dout` then `col2im` | ✅ | `reverse.rs:2188-2193` |
| `ops.conv2d_grad_kernel` | `Conv2dForward` backward: `dw = dout·colᵀ` | ✅ | `reverse.rs:2184-2186`; test `test_conv_grad.rs:11-102` |
| `ops.cross_entropy` | stable log-softmax + NLL (one-hot and indices) | ✅ | `nn/loss/cross_entropy.rs:27,63` |
| `func.expand_as` | `Op::Broadcast` (backward = reduction) on the tape's 2-D regime | ✅➕ | `reverse.rs:644,1620,3142` |
| `func.mean1d` | `Var::mean` / `MeanAxis` | ✅ | `reverse.rs:641,1544` |
| `func.mean2d_dim0` | `Var::mean_axis(0)` | ✅ | `reverse.rs:3156` |
| `func.mean4d_dim023` | composition (see `sum4d_dim023`) | ✅➕ | `batch_norm_2d.rs:101-103` |
| `nn.Linear` | `nn::Linear` (autograd, state_dict) | ✅ | `nn/linear.rs:69-137` |
| `nn.Conv2d` | `nn::Conv2d` | ✅ | `nn/conv2d.rs` |
| `nn.BatchNorm1d` | `nn::BatchNorm` (train/eval, running stats) | ✅ | `nn/batch_norm.rs:49-134` |
| `nn.BatchNorm2d` | `nn::BatchNorm2d` (train/eval, running stats) | ✅ | `nn/batch_norm_2d.rs:54-120` |
| `nn.CrossEntropyLoss` | `nn::loss::CrossEntropyLoss` (gradient verified = softmax − target) | ✅ | `nn/loss/cross_entropy.rs:179-207` |
| `optim.Adam` | `autodiff::optim::Adam` (betas, eps, weight decay, bias correction) + `NdAdam`/AdamW | ✅ | `autodiff/optim.rs`; `nn/nd_optim.rs` |
| `optim.Adam(amsgrad=True)` | **added**: `Adam::with_amsgrad()` (running max of the 2nd moment, bias-corrected) + 2 tests (convergence oracle, anti-spike property) | 🆕 | `autodiff/optim.rs` (this PR) |
| `utils.get_hash`/`print_hash` | **added**: `scirust_runtime::hash::{sha256_hex_f32, sha256_hex_tensor, sha256_hex_state_dict}` (platform-independent LE encoding, sorted keys) + 5 tests | 🆕 | `scirust-runtime/src/hash.rs` (this PR) |
| `from_torch_module` | Ⓝ SciRust is not a PyTorch wrapper — equivalents: **safetensors** reader (HF/PyTorch weights), deterministic SRT1 format, ONNX-JSON export/import, per-layer `state_dict`/`load_state_dict` | Ⓝ | `scirust-core/src/io/safetensors.rs:138`; `scirust-runtime/src/lib.rs`; `scirust-onnx/src/lib.rs:295` |
| Transcendentals via f64 promotion (`exp2d`, `log1d`) | **added**: `reproducible::{exp_via_f64, ln_via_f64}` — same technique class, honest documentation of the guarantee class | 🆕 | `scirust-core/src/reproducible.rs` (this PR) |

The `mnist_classifier` and `cifar10_classifier` examples genuinely train
models with these building blocks (complete forward/backward/step loops,
>90 % accuracy criterion on MNIST) — the equivalent of RepDL's
`examples/mnist_training.py`.

## 5. Determinism axis — compared guarantees

This is where the two projects really differ (a finding consistent with the
position already recorded in `paper/RELATED_WORK.md`):

| Guarantee | RepDL | SciRust |
|---|---|---|
| f32 bit-exact **cross-platform** (CPU↔GPU, x86↔ARM) | ✅ (its raison d'être; not numerically evaluated in its report) | ❌ assumed out of scope for the f32 path (`scirust-runtime/README.md:34-35`); **recorded future work** |
| f32 bit-exact **intra-architecture**, invariant to thread count | (implicit) | ✅ tested: identical fingerprint on 1/2/4/8/16/64 threads, 0 divergence over 5,120 logits (`tests/fingerprint_thread_invariance.rs`, report §6.2) |
| Low-precision deterministic (int8/int16/fixed-point) **cross-platform by construction** | ❌ out of scope | ✅ int8/int16/Q16/Q32/Zq GEMM, NEON == scalar bit-exact (`quantization.rs:1959`), GPU == CPU bit-exact integer paths (`scirust-gpu/src/deterministic_gpu.rs`) |
| Order-independent reproducible summation | fixed sequential order (depends on traversal) | ✅ stronger: **correctly rounded sum of the multiset** (Demmel–Nguyen + Shewchuk), bit-identical under permutation (`reproducible.rs`) |
| Parallel reductions with fixed order | OpenMP, fixed order | ✅ worker/rank-order aggregation, tested bit-exact (`data_parallel.rs`, `distributed.rs`) |
| Fingerprint (hash) verification | `get_hash` SHA-256 | 🆕 `runtime::hash` (this PR) + existing FNV-1a + SHA-256 attestation chain (`attest.rs`) |
| Verifiability beyond the hash | ❌ | ✅ Freivalds/GF(p) (`vinfer.rs`), DiFR error envelope (`difr.rs`) — no RepDL equivalent |
| TCB | PyTorch + libtorch (millions of C++ lines) | 100 % auditable Rust, zero FFI in the compute path |

Known and unchanged watch points (already tracked elsewhere): the aarch64 CI
job only runs `cargo check` (real ARM execution lives on Jetson, outside CI);
the SIMD/GPU **floating-point** reductions remain equal-within-tolerance to
the scalar, not bit-exact — only the integer path is.

## 6. Gaps closed by this PR

### 6.1 `Adam::with_amsgrad()` — parity with `optim.Adam(amsgrad=True)`
`v_max` buffer (running max of the 2nd moment, bias-corrected like `v`),
implemented from Reddi, Kale & Kumar (ICLR 2018). Two tests: convergence
oracle on a quadratic, and the defining property (after a gradient spike,
AMSGrad steps stay < 10 % of Adam steps).

### 6.2 `scirust_runtime::hash` — parity with `utils.get_hash`/`print_hash`
Hex SHA-256 fingerprints of f32 slices, of tensors (shape included) and of
complete `state_dict`s (sorted keys ⇒ independent of insertion order).
Little-endian encoding of the IEEE-754 bits ⇒ identical fingerprint on any
platform for bit-identical data. This is the tool that lets a user *observe*
reproducibility (two machines, same hash).

### 6.3 `reproducible::{exp_via_f64, ln_via_f64}` — parity with `exp2d`/`log1d`
Same technique class as RepDL (double promotion). The documentation states
the guarantee class without over-promising: faithfully rounded (and correctly
rounded outside table-maker's-dilemma cases), deterministic on a given
binary, cross-platform identity very likely but not proven — provably
correctly rounded transcendentals in pure Rust remain the future work
recorded in workstream 108.

## 7. Gaps not retained (justified)

- **`sum4d_dim023` / `mean4d_dim023` as a dedicated op**: the real need
  (per-channel BatchNorm2d statistics) is covered by composition
  (`batch_norm_2d.rs:86-113`). A fused op would be a performance
  optimization, not a functional gap.
- **General N-D `expand_as`**: the tape's 2-D regime is covered by
  `Op::Broadcast`; the N-D tape has not needed it to date.
- **`from_torch_module`**: not applicable — SciRust is a standalone
  framework, not a PyTorch wrapper; external weight import goes through
  safetensors.
- **AMSGrad on `NdAdam`**: not added (the `NdAdam` targets AdamW for N-D
  decoders); to do if a transformer use case requires it.

## 8. Recommendations

1. **P1 — written IP policy**: record (done here, §3) the rule "no
   copy/translation of RepDL code; reimplementation on public specs only"
   for all future reproducibility work.
2. **P2 — wire up the new building blocks**: publish the
   `sha256_hex_state_dict` hash in the existing audit artifacts (test
   protocol, Jetson reports) alongside the FNV fingerprints; consider
   `exp_via_f64` in the tape's softmax if f32 portability becomes a product
   goal.
3. **P2 — native ARM CI**: the day an aarch64 runner is available, run (not
   just `cargo check`) the invariance tests and compare the x86/ARM
   fingerprints of the integer paths — would turn "bit-exact cross-platform
   by construction" into "…and tested in CI".

---

## Post-script (same day, same PR)

Following up on the §5 recommendation: the **portable f32 path** was
implemented on the spot — `scirust-core/src/portable_f32.rs` (`exp_f32`,
`ln_f32`, `softmax_f32`, `dot_f32`, `gemm_f32`), pure Rust without libm, only
basic IEEE-754 operations in fixed order ⇒ **bit-exact cross-platform by
construction** (faithfully rounded; *proven* correct rounding remains outside
the claim). The bitwise goldens and FNV fingerprints of the full f32-space
sweep are committed in the tests: these are the contracts to verify on ARM.
Clean-room implementation (argument reduction + Taylor/atanh series — public
mathematical methods; no fdlibm/musl/RepDL code consulted). This closes, in
the "by construction" regime, the last axis where RepDL was stronger; the
correct-rounding proof and ARM execution in CI remain the next two steps.
