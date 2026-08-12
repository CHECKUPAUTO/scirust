# Empirical study of "dead guards" — dead epsilon guards in public numerical codebases

Campaign date: 2026-07-10. Tool: `epsilon-audit --mine` (crate
`scirust-sigma` v0.1.0, std-only binary, multi-language lexical parsing).
Manual review: every candidate emitted by the tool was re-read in context in
the cloned tree before classification.

## 1. Research question

Does the "**dead epsilon guard**" bug class — an f32 guard constant so
small that the guard does not protect — exist in real numerical codebases,
and at what prevalence?

Two death mechanisms are detected (detailed in
`scirust-sigma/src/mine.rs`):

- **M1 (flush)**: f32 literal with `0 < |v| < 1.17549435e-38`
  (`f32::MIN_POSITIVE`). Such a literal is **subnormal**: under FTZ/DAZ
  (fast-math, GPU drivers, CPU modes) it is flushed to `0` — `x.max(g)` becomes
  `x.max(0)` and the guard no longer exists.
- **M2 (inversion)**: f32 literal with `0 < |v| < 2.938736e-39`
  (= `1/f32::MAX`). Even without FTZ: if `x.max(g)` equals `g`, then
  `1.0/(x.max(g))` overflows to `inf`. The M2 range is included in the M1
  range; both are classified separately (M2 = the strongest mechanism).

## 2. Exact methodology

1. **Cloning**: `git clone --depth 1 <url>` into `/tmp/mining/` (sparse
   clone `--filter=blob:none --sparse` limited to the indicated subdirectories
   for giant repositories). Commit SHA recorded for each repository.
2. **Scan**: `epsilon-audit --mine /tmp/mining/<repo> --out reports/<repo>.md`
   (binary compiled in release from this commit of the SciRust repository). Scanned
   extensions: `.rs`, `.c`, `.h`, `.cpp`, `.hpp`, `.cc`, `.cu`, `.cuh`, `.cl`,
   `.metal`, `.wgsl`, `.glsl`, `.comp`. Exclusions: `test*/` directories,
   `bench*/`, `benchmark*/`, `third_party/`, `3rdparty/`, `vendor/`,
   `external/`, build artifacts (`target/`, `build/`), `*_test.*` /
   `test_*.*` files.
3. **f32 typing (documented lexical heuristic)**:
   - Rust: `f32` suffix or `f32` on the line → CONFIRMED-F32; `f64` →
     out of scope; otherwise UNCERTAIN (never counted).
   - C/CUDA/OpenCL family: `f`/`F` suffix → CONFIRMED-F32; bare literal on a
     line containing `float` → PROBABLE-F32; otherwise UNCERTAIN (a bare literal
     is `double` in C — never counted as a finding).
   - Shaders (WGSL/GLSL/Metal/compute): floats are f32 by default →
     CONFIRMED-F32 (and GPUs very commonly flush subnormals).
   - The threshold comparison is done on the value **rounded to f32**
     (materialization semantics: `1.17549435e-38` rounds exactly to
     `f32::MIN_POSITIVE` → legal guard, not captured).
4. **Fast-math**: grep for the flags `-ffast-math`, `use_fast_math`
   (covers `--use_fast_math`/`-use_fast_math`), `-funsafe-math-optimizations`
   and `ftz` (case-insensitive) in build files (CMakeLists,
   `*.cmake`, Makefile*, `*.mk`, build.rs, setup.py, meson.build, BUILD,
   `*.bzl`/`*.bazel`, `*.gn`/`*.gni`) → "Probable FTZ" column.
5. **Mandatory manual review**: each candidate re-read in context
   (full file in the cloned tree) and classified `CONFIRMED_DEAD_GUARD` /
   `BENIGN` / `UNCERTAIN`. A candidate is CONFIRMED only if: f32 typing
   established **and** guard usage established (`.max(`, denominator, `fmaxf`,
   threshold protecting division/log/sqrt) **and** mechanism M1 or M2 applies.

The per-repository reports (Markdown sealed with SHA-256) are reproducible
bit-for-bit on an identical tree; the tool never modifies cloned repositories.

## 3. Corpus — 22 repositories scanned, 0 clone failures

| Repository | SHA | Scanned scope | Files | LOC | Candidates | Confirmed | Uncertain | Probable FTZ |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ggml-org/llama.cpp | `8f114a9b` | root | 1 550 | 609 678 | 0 | 0 | 0 | yes (7) |
| ggml-org/ggml | `524f974b` | root | 1 098 | 416 806 | 0 | 0 | 0 | yes (7) |
| huggingface/candle | `31f35b14` | root | 772 | 247 523 | 0 | 0 | 0 | yes (2) |
| tracel-ai/burn | `105b0e9b` | root | 1 142 | 287 987 | 0 | 0 | 0 | no |
| pytorch/pytorch | `3bda7431` | `aten/src`, `c10`, `caffe2/utils/math` | 2 599 | 709 503 | 0 | 0 | 0 | yes (5) |
| tensorflow/tensorflow | `bb8ff7dc` | `tensorflow/core/kernels` | 1 565 | 309 452 | 0 | 0 | 0 | yes (3, ftz) |
| microsoft/onnxruntime | `f4aa2b44` | `onnxruntime/core` | 2 718 | 763 846 | 0 | 0 | 0 | no |
| OpenMathLib/OpenBLAS | `7c991951` | `kernel`, `lapack` | 1 279 | 419 995 | 0 | 0 | 0 | yes (1) |
| libeigen/eigen (GitLab) | `26f009db` | `Eigen/src` | 409 | 183 137 | 0 | 0 | 0 | no |
| NVIDIA/cutlass | `e6233cba` | `include` | 785 | 674 100 | 0 | 0 | 0 | no |
| rust-ndarray/ndarray | `bd3ade99` | root | 113 | 33 147 | 12 | 0 | 0 | no |
| dimforge/nalgebra | `3320ecca` | root | 279 | 73 273 | 0 | 0 | 0 | no |
| sarah-quinones/faer-rs | `0539947f` | root | 163 | 120 623 | 0 | 0 | 9 | no |
| sonos/tract | `26edc98e` | root | 818 | 207 109 | 0 | 0 | 0 | yes (1) |
| gfx-rs/wgpu | `48904f8e` | root | 683 | 318 099 | 2 | 0 | 0 | no |
| bitshifter/glam-rs | `16e0d32f` | root | 225 | 226 782 | 0 | 0 | 0 | no |
| Tencent/ncnn | `13b6d531` | `src` | 1 747 | 883 793 | 0 | 0 | 0 | yes (1) |
| alibaba/MNN | `785907a8` | `source` | 2 092 | 1 613 017 | 0 | 0 | 0 | no |
| apache/tvm | `67bd1ea1` | `src` | 874 | 275 047 | 0 | 0 | 0 | no |
| ggml-org/whisper.cpp | `6fc7c33b` | root | 1 329 | 628 589 | 0 | 0 | 0 | yes (3) |
| leejet/stable-diffusion.cpp | `cc734292` | root | 161 | 141 339 | 0 | 0 | 0 | no |
| webonnx/wonnx | `c62f5d33` | root | 49 | 18 003 | 0 | 0 | 0 | no |
| **Total** | | | **22 450** | **9 160 848** | **14** | **0** | **9** | **9/22 repositories** |

## 4. Manual review of the 14 candidates

### 4.1 ndarray@bd3ade99 — 12 candidates, all BENIGN (test context)

`src/array_approx.rs:199-234` — `#[cfg(test)] mod tests` module **inline in
`src/`** (file lines 181-182), therefore not covered by the path
exclusion. Six assertions of the type:

```rust
// Check epsilon.
assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
```

- f32 typing: established (`f32` suffixes). Mechanism M2 applicable in range.
- Guard usage: **not established** — `1e-40`/`1e-41` are assertion tolerances
  **deliberately** chosen subnormal to test the semantics of `approx`
  comparisons around zero. No division, no `.max(`, no protection threshold.
- Classification: **BENIGN** (test context; value not used as a guard). Same
  for lines 216-217 (`assert_relative_*`) and 233-234 (`assert_ulps_*`).

### 4.2 wgpu@48904f8e — 2 candidates, all BENIGN (deliberate test constants)

`naga/src/front/wgsl/parse/lexer.rs:854-856` — `#[cfg(test)]` section of the
WGSL lexer:

```rust
const SMALLEST_POSITIVE_SUBNORMAL_F32: f32 = 1e-45;
const LARGEST_SUBNORMAL_F32: f32 = 1.1754942e-38;
```

- f32 typing: established (`: f32` annotation). Mechanisms M2/M1 applicable in
  range.
- Guard usage: **not established** — constants explicitly named
  `SUBNORMAL`, used to verify that the WGSL lexer parses subnormal literals
  correctly. Subnormality is the very subject of the test.
- Classification: **BENIGN** (test context; subnormality intentional and
  documented by the name).

### 4.3 faer-rs@0539947f — 9 UNCERTAIN literals (not counted)

`1e-200`/`1e-250` in norm test loops (`for factor in [...,
1e-250]`) and f128 helpers (`eigen-bench-setup/eigen.cpp`,
`faer-ffi/quad.hpp`). f64/f128 contexts: outside the f32 range by typing —
correctly excluded from the count by the heuristic.

### 4.4 Fast-math / FTZ flags ("Probable FTZ" column)

Mechanism M1 assumes an FTZ/DAZ environment. The campaign confirms that this
environment is **real and widespread**: 9 of the 22 repositories enable
fast-math or FTZ in their build files, notably:

- `ggml`/`llama.cpp`/`whisper.cpp`: `-ffast-math` (CPU), `--use_fast_math`
  (CUDA), HIP/MUSA (`src/ggml-cpu/CMakeLists.txt:720`,
  `src/ggml-cuda/CMakeLists.txt:197`);
- `pytorch`: `-ffast-math` in QNNPACK
  (`aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl`);
- `tensorflow`: explicit `ftz` control of generated kernels
  (`tensorflow/core/kernels/mlir_generated/build_defs.bzl:169`);
- `candle`: `--use_fast_math` (flash-attn, `candle-flash-attn/build.rs:141`);
- `OpenBLAS` (`Makefile.power:147`), `ncnn` (`src/CMakeLists.txt:379`),
  `tract` (bench).

The *threat model* (subnormals flushed in GPU/fast-math production) is
therefore confirmed; it is the *bug class itself* that was not observed in
the corpus.

## 5. Limitations

- **Lexical parsing, not semantic**: no type inference (typing relies on
  suffixes and type mentions on the line), no macro expansion, no constant
  propagation (`f32::from_bits`, named constants reused elsewhere escape the
  scan), hexadecimal float literals (`0x1p-149`) not covered.
- **Incomplete test exclusion by construction**: the exclusions are path
  rules; *inline* test modules (`#[cfg(test)]` in `src/`) are scanned — all
  14 campaign candidates came from such modules and were set aside during
  manual review, not by the tool.
- **Scope**: giant repositories are scanned on the core subdirectories
  indicated in the table (sparse clone), not on the whole tree; shaders
  embedded in strings (WGSL inline in Rust, as in practice in wgpu/wonnx)
  are not scanned — the scanner skips strings.
- **Range bias**: only literals `< f32::MIN_POSITIVE` are captured. A guard
  that is "too small for its scale" but normal (e.g. `1e-30` against squares
  of values ~`1e-5`) is a neighboring, real bug class, but outside the
  mechanical M1/M2 criterion — not measured here.
- **A snapshot**: one commit per repository, at the campaign date.

## 6. Decision rule applied and verdict

Rule (fixed before the campaign):

- ≥ 3 `CONFIRMED_DEAD_GUARD` in ≥ 2 distinct known repositories → **GO**:
  the study becomes the paper's motivation section, with bug reports
  written (never posted).
- Otherwise → **NO-GO**: negative result recorded honestly; the paper
  positions itself without this section.

Numbers: 22 repositories scanned (acceptance threshold: ≥ 20 — reached),
9 160 848 lines, 14 raw candidates, **0 CONFIRMED_DEAD_GUARD**,
14 BENIGN, 0 UNCERTAIN remaining after review (the 9 uncertain literals in
faer-rs are f64/f128 contexts, excluded by typing).

### Verdict: **NO-GO**

The "dead epsilon guard" bug class (in the strict M1/M2 sense: a subnormal
f32 literal used as a guard) **was not observed** in the corpus of 22 major
numerical repositories (~9.2 M LOC). The only subnormal f32 literals found
outside excluded tests were deliberate test values (ndarray `approx`
tolerances, naga lexer constants).

Honest reading of the negative result:

1. **Prevalence in mature, widely reviewed code is ≈ 0** on this
   mechanical criterion. Practitioners choose normal guards
   (`1e-6`…`1e-12` typical in f32) — the "subnormal guard" error apparently
   does not survive in projects of this maturity.
2. **The result does not refute the usefulness of the internal σ gate**: SciRust's
   `epsilon-audit --check` gate is *preventive* (it blocks the
   introduction of such a guard in CI on the sanitized path, where
   `sanitize_f32` flushes any subnormal by construction); the campaign
   shows that the targeted FTZ environment is widespread (9/22 repositories), not
   that the error is frequent.
3. **Consequence for the paper (Lot 3)**: no "measured prevalence of the bug
   class" section; the argument refocuses on the measured cost of
   determinism and the evidence architecture. The negative study
   can be cited in one sentence (method + numbers) as due diligence.

No bug report is therefore written (the GO branch is not taken), no
issue/PR has been opened, no external contact has occurred; the quoted
excerpts are ≤ 3 lines per finding (public repositories, analysis use).
