# Paper plan — determinism as certification evidence

> Lot 3 of the 2026-07-10 mission. Centerpiece: the claims →
> evidence table (§4) — the paper will contain **no claim not proven by
> execution**; any claim without executable evidence is removed or
> marked `TODO-EVIDENCE` with the test to be written.

## 1. Title

Working title:

> **Determinism as Certification Evidence: a Fully Auditable Rust Stack for
> Bit-Reproducible Training and Quantized Edge Inference**

Proposed variants:

1. *Bit-Reproducible by Construction, Auditable by Design: Deterministic
   Training and Int8 Edge Inference in Pure Rust* — emphasizes the
   double property (construction + audit).
2. *From Best-Effort to Evidence: a Zero-FFI Rust Stack Where Every
   Determinism Claim Ships With Its Test* — emphasizes the measured-claims
   discipline, which is the real differentiator against RepDL (whose
   report contains no evaluation).

## 2. Target venues and recommendation

### Option A — JOSS (Journal of Open Source Software), software paper

- **For**: short artifact-centered format; our strengths match the
  criteria (substantial tests, CI, multi-language documentation, reproducible
  install); feasible in the short term; citable DOI.
- **Against / BLOCKING as it stands**: JOSS requires an **OSI-approved** license.
  The repository is under **PolyForm Noncommercial 1.0.0** (dual licensing,
  `LICENSING.md`) — non-OSI. Without a re-license (global or of a published
  subset, e.g. `scirust-core` + `scirust-runtime` + `scirust-sigma` under
  MIT/Apache-2.0), a JOSS submission is **inadmissible**. Human
  decision required.

### Option B — correctness / reproducibility workshop (research version)

- Examples of targeted families: "Correctness" workshops (SC), the
  reproducibility/artifact sessions of systems-ML conferences. (The precise
  choice of edition/year is a human decision; no deadline is
  assumed here.)
- **For**: the research contribution is real but targeted — (i) determinism
  *as a piece of certification evidence* (fingerprints,
  hash-chained journals, manifest) rather than as an execution property;
  (ii) the **measured cost of determinism on ARM edge**; (iii) the
  "one claim = one test" discipline. No license conflict.
- **Against**: less artifact-software recognition than a JOSS; the O1
  overhead benchmark is delivered on the x86 side (see §4), the Jetson
  leg remains.

### Argued recommendation

**Option B first.** Two reasons: (1) the PolyForm license makes JOSS
inadmissible today — the re-license is a proprietary decision, not an
editorial one; (2) since the **NO-GO** verdict of the "dead guards" study
(`docs/DEAD_GUARDS_STUDY.md`: 22 repositories, ~9,2 M LOC, 0 confirmed dead
guard), the "widespread bug class" motivation is not available —
the paper must position itself on the **measured cost of determinism** and
the **evidence architecture**, an angle that suits a correctness/reproducibility
workshop better than a software journal. JOSS remains the natural target
*afterwards* if the re-license of a subset is decided.

## 3. Section plan

1. **Introduction** — determinism treated as certification
   evidence, not as an execution mode; the contract "every claim is
   proven by an execution of the repository".
2. **Related work** — draw on `paper/RELATED_WORK.md` (Goldberg,
   Monniaux, ReproBLAS; PyTorch deterministic mode, EasyScale, RepDL —
   pivot paragraph; cross-vendor GPU divergences). One sentence of
   due diligence on the negative "dead guards" study (method + figures).
3. **Architecture** — three numerical regimes: integer/fixed-point
   (bit-exact cross-platform), f32 *sanitized* (deterministic
   intra-architecture, σ = `f32::MIN_POSITIVE`), raw f32 (measured, not
   guaranteed); zero FFI in the compute path; TCB = rustc + std.
4. **Bit-reproducible training** — fixed worker-order reduction;
   1/2/4/8-thread invariance == sequential, composed over a multi-step SGD
   loop; design comparison with ReproBLAS (fixed order vs
   order-insensitive sum) and EasyScale.
5. **Inference as an audit artifact** — 64-bit fingerprint,
   reconstruction by manifest + SRT1/QSR1, regression lock,
   tamper detection, hash-chained journals.
6. **Deterministic Int8 for the embedded** — fully integer pipeline,
   NEON kernel bit-exact against the scalar reference, Jetson validation
   (aarch64).
7. **Evaluation: the cost of determinism** — measured overhead of the
   deterministic choices on edge and x86 (see O1); bounded p99/p50 latency;
   accuracy (MNIST/CIFAR) unchanged.
8. **Limitations** — intra-architecture f32 path (vs RepDL's correct
   rounding); ops scope; wgpu GPU validated on a software adapter in
   CI; wall-clock outside CI by nature.
9. **Conclusion and future work** — correctly rounded
   transcendentals in pure Rust; multi-node extension with a fixed reduction tree.

## 4. Claims → evidence table (centerpiece)

Rule: each claim below is linked to the **exact** test/benchmark of the repository
that proves it, with the command. `[CI]` = executed by
`.github/workflows/ci.yml`; `[protocol]` = documented execution
(`docs/TEST_PROTOCOL.md`), reproducible but outside CI (wall-clock or specific
hardware). Any claim without evidence is marked `TODO-EVIDENCE`.

| # | Paper claim | Exact evidence | Command |
|---|---|---|---|
| T1 | A multi-threaded training batch is bit-identical for 1/2/4/8 threads and equal to the sequential one, with order-sensitive contributions (±1e16) | `train_batch_threaded_is_thread_count_invariant`, `scirust-core/src/autodiff/data_parallel.rs:399` [CI] | `cargo test -p scirust-core train_batch_threaded_is_thread_count_invariant` |
| T2 | The invariance holds with a true autograd backward (ParallelTape) | `parallel_tape_training_is_deterministic_across_threads`, `scirust-core/src/autodiff/data_parallel.rs:424` [CI] | `cargo test -p scirust-core parallel_tape_training` |
| T3 | The invariance composes over a multi-step SGD loop (bit-identical weight trajectory at 1/2/4 threads) | `multi_step_training_is_thread_count_invariant`, `scirust-core/src/autodiff/data_parallel.rs:458` [CI] | `cargo test -p scirust-core multi_step_training` |
| T4 | The all-reduce averages bit-exactly regardless of the number of threads | `all_reduce_averages_across_threads_bit_exactly`, `scirust-core/src/distributed.rs:251` [CI] | `cargo test -p scirust-core all_reduce_averages` |
| R1 | An emitted artifact is reconstructed from the manifest and any tampering is detected | `emit_then_verify_roundtrip_and_tamper_detection`, `scirust-runtime/tests/verify_roundtrip.rs:30` [CI] | `cargo test -p scirust-runtime --test verify_roundtrip` |
| R2 | The quantized QModel/QSR1 artifact does a deterministic round trip | `qmodel_roundtrip_and_deterministic`, `scirust-runtime/src/quant.rs:285` [CI] | `cargo test -p scirust-runtime qmodel_roundtrip` |
| R3 | A verifiable inference soundly rejects tampering (output, model commitment, substitution) | `vinfer_rejects_tampering_soundly` + 2 neighboring tests, `scirust-runtime/src/vinfer.rs:219-257` [CI] | `cargo test -p scirust-runtime vinfer` |
| R4 | The 64-bit forward fingerprint is identical across threads **and across processes** (0 divergence over 5120 logits) | threads leg: `forward_fingerprint_is_thread_count_invariant`, `scirust-runtime/tests/fingerprint_thread_invariance.rs` (rayon pools 1/2/4/8, synthetic batches) [CI]; inter-process leg: `scirust-runtime` binary (fn `fingerprint`, `src/main.rs:14`) + technical report §6.2 [protocol] | `cargo test -p scirust-runtime --test fingerprint_thread_invariance` |
| Q1 | The quantized int8 linear layer reproduces fp32 within tolerance and is deterministic | `test_quantized_linear_matches_fp32` (`scirust-core/src/quantization.rs:214`), `test_quantized_linear_deterministic` (`:243`) [CI] | `cargo test -p scirust-core quantized_linear` |
| Q2 | The multiplication-free BitNet ternary matmul equals the dequantized reference bit-exactly | `ternary_matmul_equals_dequant_bit_exact`, `scirust-core/src/quantization.rs:1033` [CI] | `cargo test -p scirust-core ternary_matmul` |
| Q3 | The int8 NEON kernel (aarch64) is bit-exact against the scalar reference | `quantization::tests_neon::neon_matches_scalar_bit_exact`, `scirust-core/src/quantization.rs:1959` [protocol: **executed on target 2026-07-10**, Jetson AGX Thor, commit `014795f` — `ok. 1 passed`; re-runnable via `scripts/bench-o1-jetson.sh`] | `cargo test --release -p scirust-core --lib neon_matches_scalar_bit_exact` (on ARM) |
| S1 | The `sanitize_f32` threshold (sanitized GPU path) is exactly σ = `f32::MIN_POSITIVE`, aligned by test | `sanitize_threshold_matches_sigma_sanitized_f32`, `scirust-sigma/tests/sanitize_alignment.rs:21` [CI] | `cargo test -p scirust-sigma --test sanitize_alignment` |
| S2 | No f32 guard below σ can enter `scirust-gpu/src` without breaking the build | gate `epsilon-audit --check`, wired as CI job `epsilon-audit` (`.github/workflows/ci.yml`) [CI] | `cargo run -q -p scirust-sigma --bin epsilon-audit -- --root . --check` |
| S3 | The "dead guards" miner correctly classifies M1/M2/f64/tests on synthetic fixtures | 27 unit tests, `scirust-sigma/src/mine.rs` (module `tests`) [CI] | `cargo test -p scirust-sigma mine` |
| G1 | The WGSL f32 GEMM (wgpu) equals the CPU oracle, on a software Vulkan adapter | CI job `gpu-wgpu` (Mesa lavapipe), `scirust-gpu` tests with feature `wgpu` [CI] | `cargo test -p scirust-gpu --features wgpu` |
| A1 | The hash-chained audit journal detects the tampering of a link | `test_chain_tamper_detection`, `scirust-func-safety/src/audit.rs:236` [CI]; twin chain `scirust-ids/src/hashchain.rs` | `cargo test -p scirust-func-safety chain_tamper` |
| O1 | Cost of determinism: measured overhead of fixed-order reductions vs arrival order (x86 + Jetson) | benchmark `scirust-core/src/bin/bench_reduction_overhead.rs` (indexed slots + reduction in order 0..n — the `train_batch_threaded` pattern — vs per-channel arrival-order accumulation; ±1e16 magnitudes to make the order observable; bit-by-bit fingerprints per repetition). **Measured x86 (4 cores, 2026-07-10, release, dim=100 352, 30 reps)**: fixed/arrival overhead = 0,930× (1 thr), 0,895× (2), 0,756× (4), 0,846× (8) — determinism is *free here, the fixed order is even faster* (contention-free slots vs channel); a single fixed fingerprint at each n over 30 reps; the "arrival" baseline produced **3 distinct fingerprints** at 8 threads (non-determinism observed in practice). **Measured Jetson AGX Thor (aarch64, 14 cores, L4T R38.4.0, MAXN, 2026-07-10, commit `0c2f1bf`, 3 runs × 30 reps; reconfirmed at commit `014795f` with `--pin-clocks` operational: 0,93-1,01× at 1-4 thr, 1,06-1,10× at 8, same fingerprints)**: fixed/arrival overhead ≈ 0,99× (1 thr), 0,93-0,95× (2), 1,01-1,03× (4), 1,06-1,11× (8) — free up to 2 threads, ~1-3 % at 4, ~6-11 % at 8; the "arrival" baseline is non-deterministic there too (2 distinct fingerprints at 8 threads). **Key result: the 4 "fixed" fingerprints are bit-identical between x86_64 and aarch64** (`0x60daf62c…`, `0x9bf7c3f3…`, `0xd5b8e15f…`, `0x7e99a9d0…`) — the fixed-order f32 reduction (IEEE add/mul, without FMA or reassociation) is reproducible **cross-platform**, measured and not merely expected. Wall-clock ⇒ [protocol], never CI | x86: `cargo run -q --release -p scirust-core --bin bench_reduction_overhead`; Jetson: `sudo scripts/bench-o1-jetson.sh --pin-clocks` |
| O2 | Bounded latency: p99/p50 ≈ 1,15 (MLP), 1,20 (CNN) | `scirust-runtime/src/bin/bench_latency.rs` + technical report §6 [protocol] | `cargo run -p scirust-runtime --bin bench_latency --release` |
| P1 | 1718+ workspace tests, 0 failure (x86); 1884/1886 executed x86/Jetson | CI job `build-test` + `docs/TEST_PROTOCOL.md` [CI + protocol] | `cargo test --workspace` |

Claims **excluded** from the paper by the rule ("no claim not proven by
execution"): the historical figure "~63 TFLOPS BF16 on Jetson Thor"
(archived CUDA path, not reproducible from the current build — already
marked as such in the README); any claim of absolute uniqueness ("only
framework to…") — replaced by the wording "to our knowledge +
self-contained scope" agreed at Lot 1.

## 5. Anticipated reviewer weaknesses, and answers

- **(a) "Your f32 path is only deterministic intra-architecture; RepDL
  does cross-platform bit-for-bit via correct rounding."** Assumed
  answer: that is exact, and said as such in the paper (§8). SciRust offers
  a **spectrum of regimes** — integer/fixed-point already bit-exact
  cross-platform, f32 sanitized intra-architecture — and the explicit
  roadmap: **correctly rounded transcendentals in pure Rust** to
  make the f32 path converge, without reintroducing a C++/Python TCB. The
  counterpart that RepDL does not offer: a zero-FFI auditable stack, integer
  int8, evidence artifacts, and every guarantee tested in CI (the RepDL
  report has no evaluation section).
- **(b) "What is the research question?"** Answer: *how much does
  determinism cost, measured, on edge?* The repository's existing figures
  (p99/p50 = 1,15/1,20; int8 NEON ~10× vs scalar at equal bit-exactness;
  int8 4× smaller without loss — technical report §6,
  `docs/TEST_PROTOCOL.md`) form the basis; the O1 benchmark delivers the first
  direct measurement: **on x86, the fixed-order reduction is even
  faster than arrival-order accumulation** (0,76–0,93×), while
  the non-deterministic baseline truly diverges (3 distinct fingerprints
  at 8 threads) — "determinism of the reduction pattern is free"
  is a measured answer, not a slogan. The Jetson leg completes it before
  submission. Second angle: the **determinism-as-evidence** architecture
  (fingerprint + hash-chain + manifest)
  as a reusable design object for certification.
- **(c) "Scope of covered ops?"** Answer: the scope is
  deliberately a subset accepted *op by op* against an oracle
  (the "one op = one gradient check / one bit-exact test" policy), listed in
  the technical report; the paper publishes the coverage table rather than
  hiding it, and positions SciRust as an auditable reference
  framework, not as a production competitor of PyTorch (already the line
  of the README).

## 6. Decisions — status as of 2026-07-10

Agreed decisions (accepted recommendations):

- **External bug reports**: closed — zero external contact (NO-GO verdict;
  the negative result is cited in one sentence of due diligence in the paper).
- **Venue**: Option B (correctness/reproducibility workshop); **no
  re-license** for JOSS — the PolyForm license is a strategic choice,
  JOSS does not drive it.
- **Paper**: conditional GO engaged — S2 wired in CI, R4 locked in CI,
  O1 benchmark delivered with x86 figures (see table §4).

Decisions — all agreed on 2026-07-10:

1. ~~Choice of the precise workshop~~ — **done**: **Correctness '26** (10th
   International Workshop on Software Correctness for HPC Applications,
   SC26, Chicago). Submission deadline: **23 July 2026**; notification:
   1 September 2026. ACM sigconf format, regular paper 7-8 pages excluding
   references (fallback: short paper 4 pages). CFP:
   correctness-workshop.github.io/2026. Paper evaluation platform:
   **Jetson AGX Thor** (user decision).
2. ~~Execution of the Jetson/aarch64 leg of the O1 protocol~~ — **done 2026-07-10**
   (AGX Thor, see O1 row: near-free determinism, and fixed reduction
   bit-identical x86_64 ↔ aarch64).
3. ~~Triggering of the full paper writing~~ — **done**: complete
   draft in `paper/correctness26/` (`main.tex` ACM sigconf +
   verified `references.bib` + build README). Remaining TODOs marked
   in the tex: affiliation, reviewer artifact link, CFP anonymity
   check, length control after compilation.

## 7. Final status — ARCHIVED (user decision of 2026-07-11)

**Submission postponed: no submission in 2026.** The workstream is closed
as it stands, ready to resume for a future edition (Correctness is
annual; the '27 edition at SC27 will likely have an analogous CFP around
June-July 2027, to be re-checked at that time).

What remains valid as-is: the draft `paper/correctness26/main.tex`
(structure, argument, claims → evidence table), `paper/RELATED_WORK.md`,
and all the `[CI]` claims of table §4 — they are re-tested on every
commit and do not go stale.

What must be refreshed before any future submission:
- re-measure the O1 protocol (x86 + Jetson, `scripts/bench-o1-jetson.sh`)
  on the submission commit and update table §4 and the tex;
- re-verify the bibliographic metadata and the state of the art (RepDL may
  have evolved; the portable f32 path added since — see RELATED_WORK §3 —
  probably deserves a section in the paper);
- take up the 4 submission TODOs listed in
  `paper/correctness26/README.md`.

Raw evidence archived in `docs/evidence/` (sealed mining reports
SHA-256 + O1 x86/Jetson outputs with provenance notes).
