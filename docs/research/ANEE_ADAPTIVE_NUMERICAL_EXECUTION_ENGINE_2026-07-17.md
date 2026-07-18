# Adaptive Numerical Execution Engine (ANEE) — Scientific Evaluation

**Date:** 2026-07-17
**Status:** Research investigation (phase 4). Design + evidence; implementation deferred
pending the recommendation in §13.
**Companion documents:** `TSA_TRANSFORMED_SCALAR_ALGORITHMS_2026-07-16.md` **[TSA]**,
`ATRA_ADAPTIVE_TRANSFORMED_REPRESENTATIONS_2026-07-16.md` **[ATRA]**,
`CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md` **[CANR]** in this directory. Their
conclusions are presupposed, not revisited. This phase starts from CANR's endpoint: certified,
*jointly* selected (representation, operator) pairs were found to be a real but narrow
engineering contribution ("novelty ladder" level 6 — a new tool niche, not a research
program). ANEE asks whether extending the selected tuple from two axes to seven —
`P = (R, O, A, T, Q, M, H)` — and adding a representation graph, portable execution-plan
artifacts, propagating numerical contracts, and a learned plan cache, changes that verdict.
**Method:** falsificationist, as in [TSA]/[ATRA]/[CANR]: definitions restricted until
falsifiable, propositions proved or refuted, claims checked against a reproducible experiment
script (`docs/research/anee_experiments/anee_experiments.py`, pure stdlib, fixed seeds,
Decimal prec-60 reference where exactness is checked) or against primary literature sources
verified online. Every external system named below was independently researched with
verified primary sources (paper venue/year, or official documentation); uncertain claims are
flagged as such rather than asserted.

---

## 0. Verdict

> **Update, same day:** the §13 Phase C prototype below was built and benchmarked after this
> verdict was first written. It survived its own pre-registered kill criterion on its first
> kernel (3 of 3 tested workload families, Addendum 1) — the one candidate this report could not
> close (§10.7) was confirmed, narrowly, for one task. A second kernel (quaternion orientation
> averaging, Addendum 2) was then tried as the honest replication check the first addendum itself
> called for, and the **replication failed** (1 of 3 conditions, short of the pre-registered bar)
> — a genuine, informative negative result showing the finding is conditional on the task, not
> general. Combined, the two addenda now **actively recommend against** generalizing kernel 1's
> win into any broader claim or tool. A third round (Addendum 3) then attacked two of this
> report's *own* remaining products: the Addendum-2 boundary heuristic **survived a prospective
> dose-response test (13/15 cells)** and is now a validated decision rule — with a mandatory
> absolute-error-floor guard learned from its two noise-floor failures — while the §4
> "representation graph," tested for the first time *as a graph* (two-hop paths), had its wins
> **deflated by a labeled post-hoc diagnostic to dictionary densification**: composition
> generates useful new dictionary members (with certificates via the exact κ product law, now
> unit-tested in code), but path structure per se showed no irreplaceable value. None of the
> addenda changes the verdict below, which predates all three and is left as originally written.
> See the three addenda, before Appendix A.

**ANEE, as stated, does not survive falsification as a new research object — but one narrow,
precisely-locatable piece of it survives as a genuinely open question worth a small, bounded
engineering prototype.** This is the fourth consecutive phase of this research program
([TSA]→[ATRA]→[CANR]→ANEE) to reach that shape of conclusion, and the evidence base behind it is
unusually deep: every one of the ~25 external systems the mission named for comparison, plus a
dozen more surfaced along the way, was independently verified against primary sources, and every
one of SciRust's own five-plus relevant internal components was read directly.

- **The core formulation is Rice's Algorithm Selection Problem (1976), not new.**
  `P=(R,O,A,T,Q,M,H), P*=argmin J(P)`, restricted to declared finite dictionaries (the
  restriction the mission itself omits, and which Lemma ANEE-1 shows must be imported unchanged
  from [ATRA]'s own prior correction of the identical vacuity problem), is exactly Rice's
  problem/feature/algorithm/performance-space framework, already extended to parametrized
  configuration spaces by CASH (2013) and solved by SMAC/irace (already verified in [CANR]).
  Rice's own research group already built a feature→algorithm selector for scientific numerical
  routines — PYTHIA, 1996 — three decades before this mission (§2.4).
- **The "Execution planner" architecture is a relabeling of a 27-year-old pattern.** FFTW (1998)
  and ATLAS (1998) already ran Generator→Benchmark→cached-Dispatcher for one kernel family each;
  SPIRAL (2005) already added a genuine (if narrow, linearity-exploiting) Static-Analyzer/
  Certificate-Engine pair — the *only* general-purpose certificate engine found anywhere in this
  investigation; AutoTVM (2018) and Ansor (2020) already add a learned cost model and a
  persisted, versioned, community-shared `(workload, target)→schedule` cache (TopHub) that is
  functionally ANEE's plan database, missing only a distribution-aware key and a confidence
  field; MLIR's Transform dialect (2022) already makes a schedule a program-independent, portable
  IR (§2.1, §2.3).
- **"Composition instead of optimization" is not new.** Equality saturation (Tate et al., POPL
  2009; egg, POPL 2021, a Distinguished Paper) already *is* the composition algebra the mission
  asks for — a space of equivalent realizations closed under rewrite, with extraction as
  optimization over the whole space, not any single hand-picked realization — matured, general,
  and already applied to two of ANEE's own pipeline stages jointly (Diospyros, ASPLOS 2021:
  operator selection + data layout, for one fixed hardware target, with no accuracy certificate).
  A parallel formalization of the *search-strategy* algebra (Elevate/Rise, ICFP 2020) exists too,
  and the two have already been fused (Guided Equality Saturation, POPL 2024) (§2.6, §6).
- **Cross-hardware plan portability is a live, unsolved problem, not a baseline ANEE could
  inherit.** Quantitatively disproven for every existing (schedule-only, easier-than-ANEE's)
  plan cache checked — FFTW's own documentation, SPIRAL's own platform-tuning experiment,
  PetaBricks' own Table 1 (up to 2.35× slowdown moving a tuned config across architectures) — and
  confirmed to remain an actively-worked, unsolved 2022–2026 research frontier (§2.3, §2.4).
- **Two independent, primary-source systems say, in their own words, that ANEE's central axis is
  a known, named, currently open gap.** MetaSchedule's own documentation states dtype is a
  cache-*matching key*, never a search variable; TVM's own Relay paper explicitly names
  accuracy-vs-performance search (via Herbie/STOKE/posits) as unimplemented future work; Ansor's
  own Limitations section names mixed/low-precision support as something it "comes short of"
  (§2.1).
- **Joint multi-axis search is mathematically necessary and combinatorially infeasible at once —
  a known tension, not a discovery.** This report's own Z1/Z2 experiments show the joint plan
  space blows up combinatorially (`k⁷` vs. `7k`) while per-axis-independent search provably fails
  once axes interact (extending [CANR]'s own H3 finding from two axes to a constructed,
  genuinely non-separable case) — exactly the tension the algorithm-configuration field (SMAC,
  irace) was built to manage, decades ago (§8).
- **Numerical contracts compose mechanically, per-property, using classical rules — the chain
  rule, a lattice meet, logical AND, and the Data Processing Inequality — but no tool bundles
  them across an arbitrary composed plan.** Genuine forward-error certificate composition across
  independently-verified, pipelined kernels is confirmed to remain an open research problem as
  of a 2026 PhD dissertation on exactly this question (§2.5, §7).

**What survives, narrowly.** The representation graph — formats as nodes, certified conversions
as edges, multi-hop selection as shortest-path search, exact by construction (Proposition ANEE-2,
an elasticity chain-rule identity validated to Decimal prec-60 by experiment Z3) — and a
`(kernel, distribution, hardware)`-keyed plan cache (rather than the field's universal
`(kernel, hardware)` key, which suffices for schedule choices precisely because a schedule's
optimality doesn't depend on the data's values, but does not suffice for representation/
precision choices, whose *correctness* does) were both searched for directly and not found
anywhere in adjacent literature: ML quantization's HAQ/HAWQ solve the structurally different
node-labeling problem on a fixed graph; no autotuning cache examined anywhere keys on data
distribution. This is an absence-of-evidence finding, not a proof, and this report cannot close
it (§10.7, §12).

**Recommendation (§13, §14): no internal runtime, no new crate ecosystem, no research paper on
ANEE as stated.** Two phases of ordinary, valuable engineering (connect SciRust's already-
existing, currently-uncoordinated pieces; enforce, rather than merely propose, a shared benchmark
schema — the second a direct response to this repository's own confirmed failure to adopt
[CANR]'s prior, structurally identical proposal, §3/§9.6) and one narrow, falsifiable, roughly
two-to-three-week prototype (the representation graph plus distribution-aware caching, benchmarked
directly against [CANR]'s existing pipeline with a pre-registered kill criterion). On the
novelty ladder established in [CANR §12]: level 1–5 for the formulation, the architecture, and
the formal machinery derived this phase — all traceable to known theory; level 5–6 at best, and
only if the Phase C prototype survives its own benchmark, for the one surviving candidate. No
level 7–8 item (a new theorem, a new algorithm class, a new field) was found — the same ceiling
[TSA], [ATRA], and [CANR] each independently reached before it.

---

## 1. Restricted formal framework (deliverable 3: formal definition of ANEE)

### 1.1 The tuple, and the vacuity trap it inherits from [ATRA]

The mission states the hypothesis as a single optimization problem

```
P = (R, O, A, T, Q, M, H)          P* = argmin_P J(P)
```

subject to accuracy, reproducibility, latency, throughput, energy, memory, stability, and
determinism constraints. Read literally — R, O, A, T, Q, M, H ranging over *all* representations,
*all* operator implementations, *all* accumulation strategies, *all* transformations, *all*
precisions, *all* execution models, *all* hardware specializations — this is exactly the trap
[ATRA]'s Lemma 1 already named and closed: **the unrestricted space of 7-tuples is the space of
all programs achieving a given result**, because any algorithm A can be written as an instance
of the tuple with degenerate choices at every axis except one ("no transform," "the operator is
A itself," "one hardware target"). An unrestricted P has no empirical content and cannot be
falsified.

**Lemma ANEE-1 (non-vacuity is conditional, not automatic).** `P = (R, O, A, T, Q, M, H)` is a
scientific (falsifiable) object only if each axis is restricted, in advance, to a *declared*
dictionary — a finite set, or a finite-dimensional parametrized family — exactly the corrective
move [CANR §1] already had to make to ATRA's original quadruple (`ℐ = (ℛ, ℱ, S, J, C, P)`, with
`ℛ`, `ℱ` explicitly finite declared sets). The mission statement for this phase does not state
this restriction anywhere; it must be imported, unchanged, from the immediately preceding phase
of this same research program. *This is the first falsification finding of this report*: ANEE's
core formulation, as literally written, re-opens a vacuity trap this research program had
already closed one phase earlier, and closes it the same way, not a new way.

**Corrected object of study.** An *admissible ANEE instance* is
`𝒜 = (𝓡, 𝓞, 𝓐, 𝓣, 𝓠, 𝓜, 𝓗, J, C, P)` with `𝓡, 𝓞, 𝓐, 𝓣, 𝓠, 𝓜, 𝓗` declared finite (or
finite-dimensional parametrized) dictionaries, `J` a stated objective, `C` a stated constraint
set (accuracy/reproducibility/latency/…), and `P` a stated evaluation protocol (dev/held-out
split, seeds — [CANR §1]'s `P` component, reused verbatim). Under this restriction, `𝒫 = 𝓡 × 𝓞 ×
𝓐 × 𝓣 × 𝓠 × 𝓜 × 𝓗` is a finite Cartesian product and `P* = argmin_{P ∈ 𝒫} J(P)` s.t. `C` is a
completely standard, well-posed **combinatorial / algorithm-configuration problem** — see §2.4
for exactly which one, and how old.

### 1.2 Single-node vs. per-node: the mission conflates two different objects

The mission's own composition diagram —

```
Representation → Transformation → Operator → Accumulation → Reduction → Reconstruction
→ Hardware specialization → Result
```

— is a labeled path (a chain). But a real computation is a graph of many operations, not one
(e.g. GEMM has two input operands; a CNN layer has a convolution, a bias add, and an activation,
each potentially wanting its own representation/hardware choice). This forces a sharpening the
mission text does not make explicit:

**Definition (execution plan, general form).** An execution plan is a pair `π = (G, ℓ)` where
`G = (V, E)` is a DAG of computation nodes (a standard compute graph, as used by every compiler
IR reviewed in §2), and `ℓ : V → 𝓡 × 𝓞 × 𝓐 × 𝓣 × 𝓠 × 𝓜 × 𝓗` labels every node with its own local
7-tuple choice — different nodes MAY choose different representations/hardware.

The mission's single global tuple `P` is exactly the special case `|V| = 1`. This matters because
it is precisely the **many-node** case — where a per-node choice interacts with its neighbors'
choices (a representation chosen for node `v` constrains which representations are cheap to feed
into node `v`'s successor) — where the interesting and hard content of "composition" actually
lives (§5, §6), and where the comparison to real compute-graph IRs (MLIR, TVM, Halide, XLA — §2)
must be made. Restated at |V|=1, ANEE risks comparing itself only to single-kernel autotuners
(FFTW, ATLAS) and missing the more demanding comparison class it actually resembles once `|V|>1`
is taken seriously: whole-graph compilers.

## 2. Literature review and prior-art map (deliverables 1–2)

Every system below was independently researched this phase with verified primary sources
(paper venue/year confirmed via publisher/Crossref/official docs, or official documentation
directly fetched); uncertain sub-claims are marked *(unverified)*. [CANR §2] already verified
Precimonious (SC 2013), FPTuner (POPL 2017), Herbie (PLDI 2015), Daisy/Rosa, FPTaylor
(TOPLAS 2018), Gappa (IEEE TC 2011), FFTW3 (Proc. IEEE 2005), ATLAS (Parallel Comput. 2001),
SMAC (LION 2011), and irace (2016) — not repeated here except where this phase adds new depth.

### 2.1 Compiler IRs and auto-scheduling DSLs

| System | Primary source (verified) | Plan/schedule as portable artifact? | Precision a search axis? | Joint operator+accum+hw search? | Certified bounds? |
|---|---|---|---|---|---|
| LLVM | Lattner & Adve, CGO 2004 | Pass-pipeline string is inspectable/reusable, but fixed per `-O` level, never searched or benchmarked | No — binary precise/fast-math flags only | No — vectorizer/reduction strategy varies *by target* (AArch64/RISC-V ordered reductions) but as independent hardcoded heuristics | None (Alive-FP, CIRE, Gappa exist but are disconnected external tools) |
| MLIR | Lattner et al., CGO 2021 (arXiv preprint differently titled) | **Yes — the Transform dialect** represents a schedule as ordinary, program-independent, serializable MLIR IR ("reification of pass options into IR"); the single closest verified match to ANEE's "plan artifact" found in this investigation | Only in downstream vendor compilers (TPU-MLIR's `search_qtable`/`search_threshold`), not core MLIR; validation is empirical (cosine similarity), never certified | **Partial yes** — `linalg` + Transform dialect + autotuners (e.g. PEAK *(unverified in full)*) jointly search tiling/fusion/vectorization + GPU hardware mapping; accumulation-strategy-as-search-axis unconfirmed | None found anywhere in the ecosystem |
| Triton | Tillet, Kung, Cox, MAPL 2019 | `triton.Config` winners are empirically benchmarked and cache to disk (JSON), keyed on args/dtype/compiler version/backend — reusable on one setup, not cross-hardware-portable | No — dtype only keys the cache, never searched; cross-hardware strategy change requires a manual rewrite in "Gluon" | No — accumulation is hardcoded in kernel source; hardware backends share one pipeline with backend-specific passes | None |
| CUTLASS / CuTe | No paper — `CITATION.cff` types it `software` | CuTe's layout algebra is a first-class *compile-time* abstraction, not a runtime/portable artifact; the Profiler's CSV output is a real offline empirical-search record | **Yes, uniquely among this cluster** — the Profiler explicitly "schmoos" over accumulator/input precision, but by brute-force racing of a pre-built library reaching "millions" of kernel variants, not a cost-model planner | Partial — Profiler sweeps tile/stage/cluster/architecture/precision jointly, but as combinatorial enumeration, not principled search | None |
| Tensor Comprehensions | Vasilache et al., arXiv:1802.04730 (2018) — **arXiv-only, no peer-reviewed venue found; project archived by Meta, April 2023, dead since 2018** | `MappingOptions` (tile/fusion/GPU-mapping) is a first-class, protobuf-serialized, reusable artifact — but sits *above* the actual ISL polyhedral schedule tree, which stays fully compiler-internal and is never exposed | No — search space is loop/parallelization structure only | Partial — genetic autotuner co-optimizes tiling/fusion with GPU thread/block/shared-memory mapping in one loop; accumulation strategy absent entirely | None — pure wall-clock benchmarking fitness |
| XLA (HLO / StableHLO) | No single canonical paper (TensorFlow OSDI 2016 venue *(unverified — usenix.org 403'd)* is the closest ancestor citation) | Scheduling/fusion decisions are baked into the HLO graph, not exposed as a separate plan object. **StableHLO is explicitly not a scheduling mechanism** — verified as "a portability layer... between frameworks and compilers," pure op-set interchange. The GPU `AutotunerPass`, however, persists a real serializable cache (`xla/autotune_results.proto`, keyed on device/HLO-fingerprint/result) — portable as a *file*, but "only works well if the autotune cache contains results generated on the same type of GPU," verified directly from source | No — `PrecisionConfig` (multi-pass bf16 emulation) is hand-set by the caller/framework (`jax.default_matmul_precision`), never searched | Partial — the `AutotunerPass` jointly, empirically benchmarks candidate operator implementations (cuDNN/cuBLAS/Triton kernels) *per instruction* against the fixed, already-chosen hardware target; precision and accumulation strategy are separate, uncoordinated, externally-fixed inputs to this search, not part of it | None — all verification found is empirical/differential (`run_hlo_module` against a reference interpreter) |
| Halide | Ragan-Kelley et al., PLDI 2013 (+ CACM 2018 retrospective; Mullapudi et al. TOG/SIGGRAPH 2016; Adams et al. TOG/SIGGRAPH 2019) | **The verified historical origin of "schedule as a first-class artifact pulled entirely outside the compiler."** Real, inspectable, but the *scheduling language* is portable across x86/ARM/CUDA/Hexagon — a *tuned instance* is not (the original paper itself quantifies up to 16× slowdown across resolutions, 7× remapping GPU→CPU) | **No — confirmed absent across all three autoscheduler generations** (2013 genetic/stochastic, 2016 greedy cost-model, 2019 learned beam-search); dtype is fixed in the algorithm, never a scheduled dimension | No — reduction order is a fixed, associativity-gated programmer annotation; the SOTA (2019) learned autoscheduler explicitly **excludes** `rfactor`/reduction-factorization from automatic search by name | None — correctness is an empirical reference-image sanity check, not a bound |
| TVM (+ Relay/AutoTVM/Ansor/MetaSchedule) | Chen et al., OSDI 2018 (+ Roesch et al. MAPL 2019 [Relay]; AutoTVM/Ansor already tabulated in §2.3) | Schedules are Halide-derived (TVM's own paper credits Halide by name), serialized as JSON tuning records; **MetaSchedule's `(workload-structural-hash, target) → best-schedule` database is a genuine, functioning, real-world instance of ANEE's cached plan store** — restricted to the schedule dimension | **No — confirmed by three independent sources**, most decisively MetaSchedule's own docs: "identifies workloads by their structural hash... shape, dtype, and computation" — **dtype is a cache-*matching key*, never a search variable.** Relay's own paper explicitly proposes accuracy-vs-performance search (via Herbie/STOKE/posits) but frames it as *unimplemented future work* | Partial — Ansor jointly searches tiling/fusion/hardware target *with* `rfactor` (reduction parallelization, itself a Halide-originated primitive), but `rfactor` only **parallelizes a fixed summation**; no compensated/Kahan/Neumaier/pairwise choice ever appears in the schedule space | None anywhere in the lineage — TVM's own Table 1 compares automation methods purely on runtime-prediction properties |

**Cluster synthesis (compiler IRs).** The strongest piece of prior art for ANEE's "execution
plan as a portable, inspectable artifact" claim is MLIR's Transform dialect (program-independent
by design, per its own CGO 2025 paper) *and*, historically prior to it by a decade, Halide's
original 2013 schedule/algorithm split (explicitly the acknowledged ancestor of TVM's, and hence
also of AutoTVM/Ansor/MetaSchedule's, schedule representation). MetaSchedule's
`(workload-hash, target) → schedule` database is, concretely, a real, shipping instance of
exactly the caching shape ANEE's "Learning execution plans" section proposes — for the schedule
axis only. **Across every system in this entire cluster — LLVM, MLIR, Triton, CUTLASS, Tensor
Comprehensions, XLA, Halide, TVM — precision/representation is either fixed upstream by the
programmer/frontend type system or, at best (CUTLASS's Profiler), brute-force enumerated over a
precompiled library; not one treats it as a jointly co-searched axis alongside schedule and
hardware.** Two independent, primary-source-verified systems (MetaSchedule's docs; TVM's own
Relay paper) state *explicitly*, in their own words, that precision search is a known, named,
currently-unaddressed gap — direct, first-party confirmation, not inference, that this specific
piece of ANEE's ambition is recognized as open by the field's own leading systems. Certified
numerical bounds are absent from every system in this cluster without exception (only SPIRAL, in
the separate autotuning cluster of §2.3, has one — and only by exploiting DSP-transform
linearity).

### 2.2 Numerical libraries and performance-portability runtimes

| System | Primary source (verified) | Dispatch mechanism | Precision a search axis? | Autotuning / plan-cache present? | Portable plan artifact? |
|---|---|---|---|---|---|
| oneDNN | UXL Foundation docs (no core-architecture paper; CGO 2024 paper covers only its separate Graph Compiler) | Runtime CPUID/device-triggered **JIT** (Xbyak/nGEN) kernel generation, chosen per shape/ISA | No — caller-set `data_type`; int8 scale/zero-point supplied by caller | Primitive cache = pure memoization (avoids recompiling), not search; `convolution_auto` picks among 2–3 algorithms by a static heuristic, not benchmarking | No — persistent-cache blobs are opaque, version/commit-locked |
| Intel MKL | Official docs (closed source, no architecture paper) | Deterministic CPUID dispatch across **precompiled** ISA-specific paths (not JIT), except a narrow true-JIT path for small (≤16×16×16) GEMM | No — distinct entry points (sgemm/dgemm/mixed) chosen by caller | Small-GEMM JIT uses a static performance *predictor*, not a bake-off; no general autotuning documented | No |
| OpenBLAS | No architecture paper (GotoBLAS2 lineage: Goto & van de Geijn, ACM TOMS 2008) | Load-time CPUID sets one global function-pointer struct from 40+ hand-written per-microarchitecture kernels | No — BLAS naming (sgemm/dgemm/…) bakes in precision | None — offline hand-tuning by maintainers, not a build- or run-time search | No |
| Eigen | Software citation only, no paper | **Compile-time only** — ISA macros set by the caller's `-march` flags select a fixed `PacketMath` header for the whole translation unit; *no runtime CPU dispatch at all* | No — `Scalar` is a compile-time template parameter | None | No |
| Kokkos | Edwards, Trott & Sunderland, *J. Parallel Distrib. Comput.* 2014; Trott et al., *IEEE TPDS* 2022 | Compile-time `ExecutionSpace` template parameter; no runtime backend auto-selection | No | **Yes — the closest library-level analogue of a plan cache found in this investigation**: the Kokkos Tuning API (core hooks) + the third-party APEX tool, which does *online, in-situ* empirical search over kernel-**launch parameters** (team/tile size, occupancy) and persists a converged result to an on-disk YAML file, reused on subsequent runs | Partial — the YAML is a real, inspectable, cross-run artifact, but keys only launch parameters, never (R,O,A,T,Q,M,H); *(exact key schema unverified)* |
| RAJA | Beckingsale et al., IEEE/ACM P3HPC 2019 | Compile-time execution-policy template parameter, even more strictly than Kokkos | No | **None** — RAJA's own feature index lists no autotuning; RAJAPerf is a manual, human-read benchmark suite, not an optimizer; the one adaptive feature (Proteus JIT) is explicitly experimental and unrelated to plan search | No |
| SYCL / DPC++ / oneAPI | Khronos SYCL 2020 spec | JIT (SPIR-V, backend-compiled at runtime) or opt-in AOT — either way, for **one fixed, programmer-written kernel**, never among algorithmic alternatives | No — `float`/`double`/`sycl::half`/`joint_matrix` precisions are explicit source-level types | Persistent JIT cache (`SYCL_CACHE_PERSISTENT`) and the new SYCLBIN container are pure memoization of *one kernel's* compiled forms — "exactly ANEE's compiled-artifact-≠-execution-plan distinction" | Real on-disk artifacts (AOT binaries, SYCLBIN) exist, but package compilation targets of a fixed kernel, not algorithmic alternatives; no confidence score or certificate |

**Cluster synthesis (numerical libraries):** this cluster falsifies, cleanly and repeatedly, any
suggestion that joint (R,O,A,T,Q,M,H) search is already standard practice in production
numerical libraries. Every library in the table fixes precision by caller-chosen type with
*zero* library-side search; hardware dispatch is either compile-time (Eigen, Kokkos, RAJA) or a
deterministic runtime lookup/JIT for *one* algorithm (oneDNN, MKL, OpenBLAS, SYCL) — never a
choice among *algorithmically different* strategies per target. The one real "plan-cache"
analogue — Kokkos plus the third-party APEX tool's on-disk converged-tuning YAML — is
third-party, online-search, and scoped to kernel-launch parameters only, never touching
representation or precision. No certified error bound is attached to any dispatch decision in
this entire cluster.

### 2.3 Autotuning and self-tuning numerical systems — the architecturally closest cluster

This cluster is where ANEE's "Execution planner" (Representation DB → Plan Generator → Static
Analyzer → Certificate Engine → Empirical Benchmark → Plan Optimizer → Runtime Dispatcher →
Execution) and "Learning execution plans" ((kernel, distribution, hardware) → plan + confidence
+ certificate + history) sections must be checked most carefully against.

| System | Primary source (verified, full text read except ATLAS) | Search space | Cached/learned plan store | Precision as a search axis? | Certificate or purely empirical? | Cross-hardware portability |
|---|---|---|---|---|---|---|
| OpenTuner | Ansel et al., PACT 2014 | Generic, user-defined (bandit-weighted ensemble of DE/Nelder-Mead/hill-climbing) | In-run SQL DB only; no persistent cross-run/cross-hardware store | No — domain-agnostic, not demonstrated in any of its 7 case studies | Purely empirical | No — paper frames portability as "re-run on new machines" |
| AutoTVM | Chen et al., NeurIPS 2018 | Per-operator hand-written schedule template; billions of configs | **Yes — JSON tuning records (target, workload, config, cost) in versioned per-backend "TopHub" repositories, nearest-shape fallback for unseen workloads.** No confidence/history field | No — dtype fixed in the input expression; paper lists this as future work | Purely empirical | Only the cost model's *feature representation* transfers (2–10× faster search, flagged by the authors as unsolved); the schedule itself does not |
| Ansor | Zheng et al., OSDI 2020 | Auto-derived hierarchical sketches, no manual template | Cost model retrained from scratch every session — explicitly **not** a persistent cross-session store | **No — explicitly named as a gap**: "comes short of utilizing... mixed-precision and low-precision operators" (Limitations section) | Dependency-analysis checks legality of a rewrite, not numerical accuracy | No — every target tuned fully independently |
| ATLAS | Whaley & Dongarra, SC'98; Whaley/Petitet/Dongarra, *Parallel Comput.* 2001 | Blocking/unrolling/loop-order/scheduling for a generated mini-GEMM | Install-time only, one machine; shipped defaults run 10–15% slower than a real search (ATLAS's own FAQ) | No — S/D/C/Z generated and tuned as separate, fixed routines | Purely empirical ("testers and timers") | No — install/search redone per machine |
| FFTW | Frigo & Johnson, ICASSP 1998 (wisdom formalized in FFTW3, 2005) | Recursive Cooley–Tukey factorizations + codelets, DP-pruned | **Yes — "wisdom," serialized to disk, keyed by (size, direction, rank/strides, precision, alignment).** The 1998-vintage direct precedent for a persistent plan cache | No — precision fixed by which compiled library variant is linked | Purely empirical timing selection | **No, directly stated in FFTW's own docs**: wisdom from one processor risks "sub-optimal plans" on another |
| SPIRAL | Püschel et al., *Proc. IEEE* 93(2), 2005 | Formula space over **distinct algorithms** (100+ breakdown rules: Cooley–Tukey, Rader, lifting…), not just schedules | Offline-learned regression-tree cost model per (transform-family, platform) — near-optimal formulas with **zero further search**; static, no confidence/history field | **Yes — uniquely among this whole cluster**: accuracy can replace runtime as the objective, with a *closed-form, algebraically derived* worst-case error bound (exploiting DSP-transform linearity), plus automated bit-width search for constant multipliers | **Yes — the only real certificate engine found anywhere in this investigation**: exact symbolic (GAP computer-algebra) verification that a generated formula equals the transform's exact matrix | **No, quantitatively shown**: SSE2-tuned code run as plain SSE is up to 320% slower; Pentium-III-tuned code on a binary-compatible Pentium 4 up to 50% slower |
| PetaBricks | Ansel et al., PLDI 2009 | Choice spans **whole alternative algorithms** (sort variants, Poisson solvers, eigensolvers) + recursive cutoffs ("algorithmic selectors"), stitched into one poly-algorithm | Per-program choice-configuration file; program- and machine-specific, no cross-program database | Partial, but not ANEE's sense — a "variable accuracy" mode picks the fastest *algorithm/iteration-count* for an accuracy target (never a numeric type/bit-width choice) | "Not provably correct... good testing coverage" — explicitly non-certified | **No, quantitatively shown**: up to 2.35× slowdown, 1.68× average, moving a tuned config across 4 architectures (Table 1) |

**Cluster synthesis.** Every individual box in ANEE's proposed pipeline has a decades-old working
precedent *somewhere* in this cluster: FFTW (1998) and ATLAS (1998) already run
Generator→Benchmark→cached-Dispatcher for one narrow kernel family; SPIRAL (2005) already adds a
genuine Static-Analyzer/Certificate-Engine pair (exact symbolic verification, closed-form error
bounds) — but only by exploiting the linearity of DSP transforms specifically, a property general
numerical kernels do not share; AutoTVM (2018) and Ansor (2020) already add a learned cost model
and a persisted, versioned, community-shared plan database (TopHub) that is functionally
ANEE's `(kernel, hardware) → plan` cache (missing only the workload-*distribution* key
component and any confidence/history field); PetaBricks (2009) already treats **algorithm choice
itself** as the autotuned object, closest in spirit to "composition instead of optimization" —
but its choice grammar never includes representation, precision, or hardware, only algorithm and
some parallelism/blocking parameters. **No system in this cluster jointly searches more than two
of ANEE's seven axes at once**; every one optimizes schedule, or algorithm, or fixed-point
bit-width, while holding the rest fixed by construction — and **cross-hardware portability is
disproven, quantitatively, by three separate primary sources** (FFTW's docs, SPIRAL's own
platform-tuning experiment, PetaBricks' Table 1), not merely unaddressed. ANEE's pipeline
architecture is therefore not new in shape; its claim to jointly search all seven axes at
once, with a *general* (not linearity-exploiting) certificate engine, would be new in *degree
and generality* — but nothing in this cluster's evidence shows that scaling joint search from
the one-or-two axes every existing system handles up to seven is even tractable (§8 already
quantifies why: `k⁷` vs. `7k`).

### 2.4 Algorithm-selection theory, representation graphs, and hardware-adaptive dispatch

**Rice's Algorithm Selection Problem is confirmed as the load-bearing prior theory.** John R.
Rice, "The Algorithm Selection Problem," *Advances in Computers* 15:65–118 (1976) formalizes
exactly ANEE's outer shell: a problem space, a feature space, an algorithm space, a
multi-criterion performance space, and a selection mapping — verified via Kotthoff's 2014 *AI
Magazine* survey, which reproduces Rice's original diagram. **Devastatingly for any novelty
claim resting on "instance-adaptive selection of a numerical strategy": Rice's own research
group already built exactly this for scientific computing, in 1996** — Weerawarana, Houstis,
Rice, Joshi & Houstis, "PYTHIA: A Knowledge-Based System to Select Scientific Algorithms," *ACM
TOMS* 22(4):447–468 (1996), a feature→algorithm selector for scientific numerical routines,
three decades before this mission. The modern algorithm-*configuration* literature (CASH —
Thornton, Hutter, Hoos & Leyton-Brown, KDD 2013; solved by SMAC, already verified in [CANR]) is
the direct extension of Rice's framework to *parameterized* algorithm families — i.e., exactly
a Cartesian-product configuration space of the kind §1.1 restricts `P` to. **`P = (R,O,A,T,Q,M,H)`
reads, without strain, as a 7-axis CASH instance.**

What this 50-year-old theory and its descendants do *not* natively carry — and where a
defensible, narrower novelty claim would have to live — is confirmed by Kotthoff's own survey,
verbatim: the field is "self-consciously statistical" ("even the designer of an algorithm does
not have a general model of its performance"), with no machine-checked *error certificate* as a
selector output; no *bit-reproducibility/determinism level* as an axis of the performance space
anywhere in Rice's framework or its descendants; and Kotthoff explicitly names exploitation of
*heterogeneous hardware as a retargeting axis* (rather than a fixed instance feature) as
unexplored as of 2014.

**The representation graph is, on current evidence, ANEE's single most original-looking claim.**
HAQ (Wang, Liu, Lin, Lin & Han, CVPR 2019) and HAWQ (Dong, Yao, Gholami, Mahoney & Keutzer, ICCV
2019; HAWQ-V2, NeurIPS 2020) — [ATRA]'s own cited hardware-aware-quantization prior art — are
confirmed, precisely, to solve a **node-labeling** problem: the graph (a neural network's fixed
layer structure) is *given*, and the only free variable per node is a precision label from a
small ordered set, chosen by RL/Hessian-sensitivity search. This generalizes across every
adjacent literature checked — GNN quantization (DegreeQuant, A²Q, SGQuant), database column-store
encoding selection (BtrBlocks, ClickHouse) — all fixed-structure, per-location labeling. **No
published instance of ANEE's alternative shape was found**: nodes *are* representations/formats
themselves, multi-attribute edges are format-to-format conversions (§4's `G`), and the problem is
**path search**, not per-location labeling, across targeted searches spanning ML quantization, DB
type-conversion, and codec/transcoding literature. This is reported as an honest negative search
result, not proof of absence — but the two problem shapes are structurally different, and no
adjacent field appears to have posed ANEE's version of the question.

**Cross-hardware plan portability is confirmed to be a live, unsolved research frontier, not a
solved problem — for existing (schedule-only) plans, let alone representation-inclusive ones.**
AutoTVM's own TopHub is hardware-keyed by construction; secondary sources report "adapting
AutoTVM to new hardware would require months of effort." Real but explicitly partial transfer
work is active through 2022–2026: Transfer-Tuning (Gibson & Cano, PACT 2022) transfers across
*models* on the *same* hardware only; Verma et al. (2023, extreme-heterogeneity workshop)
demonstrate genuine cross-hardware transfer but frame it as warm-started fine-tuning, not
zero-shot, with 25–90% search-time reduction (not elimination); TenSet (Zheng et al., NeurIPS
2021 Datasets & Benchmarks, 52M records/6 platforms) exists specifically because the field
lacked data to study this at all; and TCL (arXiv:2604.12891, April 2026, an unreviewed preprint
at time of writing — flagged) states plainly that "most existing methods are restricted to
one-to-one adaptation." **No zero-target-search full cross-hardware transfer was found anywhere**
— this remains open for the field in general in mid-2026, which reframes mission question 10
("can execution plans become portable between architectures?") from "does ANEE solve this" to
"ANEE would inherit a currently-open research problem, not resolve one."

**Hardware dispatch beyond vectorization exists, but is the minority pattern.** The dominant,
most-cited mechanisms — GCC/Clang function multiversioning (`target_clones`/`ifunc`), ISPC,
Google Highway's `HWY_DYNAMIC_DISPATCH` — are, by construction, re-vectorizations of *identical*
source per target. Real counterexamples were found and verified directly: Highway's own
design-philosophy doc states plainly that "common idioms for one platform can be inefficient on
others" (e.g. horizontal-sum vs. shuffle-based reduction) and explicitly permits genuinely
divergent per-target code paths; glibc's `ifunc`-dispatched `memcpy`/`memset` ship not just
SIMD-width variants but an entirely different mechanism (hardware `rep movsb/stosb`) as one of
the dispatch targets (verified directly against the glibc source tree); and SPIRAL (§2.3) is the
cleanest fully-published case of true algorithm-level (not just width-level) hardware adaptation
— its Formula Generator produces genuinely different recursive breakdown rules per platform, not
just re-vectorized code. Production BLAS (OpenBLAS/BLIS `KERNEL.CORENAME` files) remains the more
common pattern: blocking/vectorization change per target, the underlying blocked-GEMM algorithm
does not.

### 2.5 Precision tuning, verified numerics, and numerical contracts — deepened beyond [CANR]

[CANR §2] already verified Precimonious (SC 2013), FPTuner (POPL 2017), Herbie (PLDI 2015),
Daisy/Rosa, FPTaylor (TOPLAS 2018), and Gappa (IEEE TC 2011). This phase went deeper on four
specific open questions:

**FPBench** (fpbench.org) is a real, verified, actively-maintained interchange schema (FPCore
2.0 + Metadata 2.0 + a "Measures" standardized error-*measurement* methodology, tracing to a 2016
NSV workshop paper). Its `:pre`/`:spec`/`:precision`/`:round` fields genuinely constitute a
narrow numerical contract — domain, correctness target, representation, and rounding discipline
— for **one flat expression**. Confirmed **absent** entirely: accumulation/summation strategy,
hardware target (beyond a free-text libm-provenance string), reproducibility/determinism level,
SIMD compatibility, energy/memory/throughput, conditioning, backward error, invertibility,
monotonicity, information loss, and even a numeric acceptance threshold (the "Measures" spec
standardizes *how* error is computed for cross-tool comparability, not *how much is tolerable*).

**Certificate composition — the load-bearing question for mission question 9 — is confirmed
unsolved in general.** FPTaylor, Gappa, and the original PRECiSA (Moscato, Titolo, Dutle & Muñoz,
SAFECOMP 2017) all analyze one flat/monolithic expression. The few genuine 2023–2025 advances
toward modularity are narrowly scoped: Abbasi & Darulova, "Modular Optimization-Based Roundoff
Error Analysis of Floating-Point Programs" (SAS 2023) bills itself as the *first* modular
analysis for FP programs, but only for one analyzer's own non-recursive procedure calls —
intra-tool, not a mechanism for combining independently-produced certificates for pipelined
kernels A→B. PRECiSA 4.0 (2024) adds an analogous, equally intra-tool abstraction. Zhang & Aiken,
"Automatic Verification of Floating-Point Accumulation Networks" (CAV 2025) is a genuine positive
result but scoped to accumulation networks built from a small closed primitive set (two-sum,
fast-two-sum), not arbitrary chained kernels. A June 2026 PhD dissertation (Abbasi, RPTU
Kaiserslautern-Landau, advised by Darulova) is titled, as of this writing, around exactly this
open problem — circumstantial but strong confirmation the general case remains open. Manual
composition remains possible (Gappa's hint mechanism; Appel's hand-glued VST+Gappa+Coq proofs)
but is not automatic propagation.

**Joint precision + one other axis exists in 2023–2026 work, but never beyond two axes.**
Heldens & van Werkhoven, "Accuracy-Aware Mixed-Precision GPU Auto-Tuning" (*IEEE TPDS*, 2026)
jointly tunes precision alongside GPU block-size/tiling/vector-width and reports joint tuning
outperforming isolated approaches — the closest match to ANEE's *spirit* found in this
investigation, but 2 axes, not 7. El Arar, Filip, Mary & Riccietti (arXiv:2503.15568, 2025)
combine precision and accumulation, but strictly sequentially, not simultaneously. No paper
found jointly searches three or more of `(R,O,A,T,Q,M,H)` at once.

**No general-purpose "numerical contract" type system spanning all of ANEE's listed properties
was found.** Every rigorous formal technique certifies essentially one property (almost always
rounding/backward error): Martel's dependent-type system (ESOP 2002), Numerical Fuzz (Kellison
& Hsu, PLDI 2024) — compositional *by typing*, but scoped to rounding error alone — and PRECiSA
throughout its whole 2017→2024 evolution. General-purpose, backend-agnostic contract *syntax*
does exist (ACSL/Frama-C, Why3/WhyML — arbitrary first-order pre/postconditions, with ACSL
modeling a float as an (actual, exact, model) triple), but the only automated proof backend
wired to any of them is still Gappa: one property, straight-line reasoning. **The syntactic
shell for a unified numerical contract already exists and is backend-agnostic; the automation
behind it does not extend past rounding error, and nothing propagates such a contract through a
composed, hardware-specific execution plan.**

### 2.6 Composition algebra: e-graphs, superoptimization, strategy languages

**This section directly confirms Proposition ANEE-3 (§6).** Equality saturation — Tate, Stepp,
Tatlock & Lerner, "Equality Saturation: a New Approach to Optimization," **POPL 2009** (not
PLDI, as this investigation's own research prompt mis-stated — a correction the literature-review
agent caught and flagged) — is, on the primary source's own terms, exactly a composition algebra:
an e-graph is a set of terms closed under a congruence relation and under rewrite-generated
equalities (rewrites only ever *add* equalities, never delete, so the structure accumulates every
rule-reachable equivalent composition simultaneously — the paper notes a saturated e-graph "can be
cyclic, representing infinitely many expressions" in polynomial-size shared structure), with
*extraction* a bottom-up traversal minimizing a user-supplied cost function over that entire
equivalence-closed space. This is matured, general, and reusable engineering, not a one-off
research idea: **egg** (Willsey, Nandi, Wang, Flatt, Tatlock & Panchekha, *PACMPL* 5(POPL),
2021 — a POPL **Distinguished Paper**) is a general-purpose, domain-agnostic e-graph library
whose own three headline case studies are Herbie (floating-point accuracy — already verified in
[CANR]), Spores (linear algebra), and CAD program synthesis.

**Diospyros** (VanHattum, Nigam, Lee, Bornholt & Sampson, ASPLOS 2021 — verified; the mission
research prompt's guessed authorship was also corrected by the agent) is direct, concrete proof
that this exact machinery already spans *two* of ANEE's pipeline stages jointly for a real
domain: it uses egg to equality-saturate small DSP linear-algebra kernels, searching **vector
instruction selection (Operator) jointly with data/memory layout** (a sub-case of Representation
— though, contrary to this investigation's working hypothesis, over `float`, not fixed-point
bit-width), beating hand-tuned DSP libraries by 3.1× average — for **one fixed target ISA**
(Tensilica Fusion G3), with no accumulation-strategy axis and no accuracy certificate (extraction
optimizes performance only). **STOKE** (Schkufza, Sharma & Aiken, ASPLOS 2013) shows a
non-e-graph alternative (MCMC stochastic search over instruction-level program mutations,
correctness checked by testing/SMT) reaching the same *kind* of result for one narrower stage
(instruction sequencing, fixed representation, fixed hardware) — narrower than Diospyros, and not
itself phrased as an algebra.

The "algebra of composition" the mission asks for has, in fact, been formalized *twice over*:
once for the *space of equivalent realizations* (e-graphs, above), and independently for the
*space of search strategies themselves* — **Elevate/Rise** (Hagedorn, Lenfers, Koehler, Qin,
Gorlatch & Steuwer, *PACMPL* 4(ICFP), 2020) makes optimization strategies first-class values with
a genuine, verified combinator algebra (`seq`, `leftChoice`, `try`, `repeat`, `topDown`/
`bottomUp` traversals — a Kleisli arrow algebra over a `RewriteResult` monad, explicitly
descended from Stratego's strategy combinators). The two lines are not competitors but already
fused in current work: **Guided Equality Saturation** (Koehler, Goens, Bhat, Grosser, Trinder &
Steuwer, POPL 2024) uses Elevate-style strategies to guide e-graph search, and **Shoggoth** (Qin,
O'Connor, van Glabbeek, Hoefner, Kammar & Steuwer, POPL 2024) gives strategic rewriting its own
formal semantic foundation. **Exo** (Ikarashi, Bernstein, Reinking, Genc & Ragan-Kelley, PLDI
2022) shows the complementary, non-automatic end of this spectrum: schedules as explicit,
human-authored, compiler-*verified* (not compiler-*searched*) composable rewrites, with
retargetability across genuinely different hardware (an embedded accelerator and AVX-512) —
proof-checked composition without automated search, the mirror image of egg's
searched-but-only-empirically-costed composition.

**Cluster verdict.** "Compose, don't optimize one component" is not new; it has been the explicit
design thesis of equality saturation since 2009, matured into shipped, general infrastructure by
2021, extended with a formal strategy-combinator algebra by 2020, and fused with that algebra as
recently as 2024. What is confirmed **absent** anywhere in this cluster: a single published rule
set plus cost function spanning Representation×Transformation×Operator×Accumulation×Hardware
*jointly*, gated by a *certified* (not merely empirical) error bound. Diospyros gets two axes,
one fixed target, no certificate; every other system in this cluster is narrower still on at
least one of those dimensions.

## 3. The SciRust baseline: what already exists internally

Before asking whether ANEE is new *to the field*, it must clear what already exists *in this
repository* — some of it built directly in response to [CANR]'s own recommendations. Verified
by direct source reading this phase:

| Component | File | What it does | What it does *not* do |
|---|---|---|---|
| Certificate-gated representation selection | `scirust-core/src/transform_search.rs` | Single-axis (`R` only): accepts/rejects each candidate transform by a machine-checked round-trip certificate, ranks survivors by cost | No operator/accumulation/hardware axis; no search — a gate, not an optimizer |
| Generic dev/held-out autotuning harness | `scirust-core/src/transform_autotune.rs` | `autotune_by<C, D>(dev, eval, candidates, score, baseline)` — domain-agnostic over candidate type `C`; already reused verbatim for two *different, single-axis* problems | Never called with a Cartesian-product candidate type spanning multiple axes at once; each call optimizes one categorical axis, holding all others fixed |
| Accumulation-strategy autotuning | `scirust-core/src/autotune_accumulate.rs` | Reuses the harness above for the `A` axis: {naive, pairwise, Neumaier, Klein, stochastic-rounding} | Independent of the `R`-axis selector — the two never run jointly |
| VST-kind autotuning | `scirust-signal/src/denoise/autotune.rs` | Reuses the same harness a third time, for denoiser transform choice | Same limitation |
| Graph-IR compiler with fusion | `scirust-tensor-compile` (`TensorGraph`/`TensorOp` → `FusedKernel`/`FusedOp`) | A real, if minimal, compute-graph IR with elementwise-operator fusion — structurally the same *kind* of object as MLIR/Halide/TVM's compute graphs (§2.1), at far smaller scope | No representation/precision axis, no hardware-specialization axis, no search — fusion is the only optimization |
| Register-machine plan executor | `scirust-tensor-runtime` | Executes a compiled `FusedKernel` against named tensor registers | Executes exactly one, already-fixed plan; no comparison among plans |
| Contraction planner | `scirust-tensor-contraction::ContractionPlan` | A literal type named "Plan": greedily orders pairwise einsum contractions by a result-size cost heuristic | Single axis (contraction order only); no representation/hardware axis, no empirical benchmarking, no certificate |
| Hardware backend dispatch | `scirust-simd/src/dispatch.rs::BackendKind` | Safe, construction-enforced runtime CPU-feature detection choosing among {Scalar, SSE2, AVX2, AVX-512, NEON, SVE, PortableSimd} for a *fixed* kernel | Picks a backend for one predetermined algorithm; does not change which algorithm/representation/accumulator runs per target |
| GPU/CUDA backend abstractions | `scirust-gpu::RawComputeBackend`, `scirust-cuda` (feature-gated) | Each defines its **own**, separate backend-selection notion | **Confirmed uncoordinated**: neither imports nor references `scirust-simd::BackendKind` — SciRust today has *three* independent hardware-dispatch abstractions (CPU-SIMD, GPU-generic, CUDA-specific), not one |
| Fixed-point representation | `scirust-simd/src/fixed/types.rs::Fixed<I, const FRAC: u32>` | A real deterministic fixed-point type | **`FRAC` is a compile-time const generic, not a runtime value** — precision for this representation is fixed by monomorphization at compile time, and is confirmed **not wired into** any autotuning code. This is a genuine feasibility constraint on any `Q`-axis (quantization/precision) *runtime* planner: Rust's type system makes fixed-point width a build-time, not run-time, decision unless the plan search is pushed into codegen/build scripts |
| Reproducible benchmarking | `scirust-tdi-bench` (10 binaries); ~16 separate `criterion` bench targets across the workspace | Real, if scattered, benchmarking infrastructure | **[CANR §9]'s own proposed shared JSON result schema (`{kernel, dataset, method, seed, metric, value, ci, cert}`) was never implemented anywhere** — every actual harness (tdi-bench, `vst_bench.rs`'s ad hoc `BenchRow`, the criterion targets) rolled its own incompatible output shape. This is direct, in-repository evidence that a *design-level* recommendation for unification, made one phase ago, was not enough to produce unification in practice |

**Reading.** Every *structural piece* ANEE's "Execution planner" architecture calls for —
a representation database (`transform_search.rs`'s dictionary), a certificate engine
(`transform_search.rs`'s gate), an empirical benchmark stage (`transform_autotune.rs`), a
graph-IR plan generator (`scirust-tensor-compile`), a runtime dispatcher (`scirust-tensor-runtime`,
`BackendKind`) — **already exists somewhere in this repository**, built independently, one axis
or one crate at a time, over three prior research phases plus ordinary engineering. None of them
talk to each other. This is the single most concrete, load-bearing fact this phase's
investigation rests on: **the SciRust-internal question ANEE actually poses is an integration
question — "should these five-plus already-built pieces be unified into one typed plan object
and one planner?" — not an invention question.** Whether that integration is itself a *scientific*
contribution, as opposed to routine (if valuable) software-engineering consolidation, is
answered by how it compares to the external literature in §2 and the formal analysis in §§4–8.

## 4. Representation graph formalization (deliverable 4)

**Definition (representation graph).** `G = (V, E)` where `V` is a declared, finite dictionary of
representations (e.g. SciRust's own seven: `Log, Log1p, SignedLog, Power(λ), MuLaw(μ), Logit,
Anscombe`, plus `f32/f64/fixed⟨W,F⟩/BF16/FP8/quaternion/...`), and an edge `u → v` exists exactly
when a conversion map `ψ_{u→v}` is defined. Each edge carries a label
`(cost, Bound{ulps}, invertible?, determinism-level)` — reusing [CANR]'s own `Bound` and
determinism-ladder types verbatim, not inventing new ones.

**Proposition ANEE-2 (exact multiplicative composition of `κ_rt`).** For a chain of strictly
monotone, C¹ representation maps `u → v → w`, the round-trip condition number of the *composed*
map equals the **exact** product of the two hops' condition numbers:
`κ_rt(w∘v-hop then u∘v-hop) = κ_rt(v→w)|at ψ(u→v)(x) · κ_rt(u→v)|at x`.

*Proof.* `κ_rt(x) = |φ(x)/(x·φ′(x))|` ([CANR §3.1]) is exactly the **elasticity** (logarithmic
derivative `d(ln φ)/d(ln x)`) of `φ`. Elasticities compose exactly multiplicatively under
function composition — an immediate consequence of the ordinary chain rule applied to
`d(ln(g∘f))/d(ln x) = d(ln g)/d(ln f)|_{f(x)} · d(ln f)/d(ln x)|_x`, with no first-order
truncation anywhere. ∎

This is *not* the same claim as [ATRA]'s Proposition 4 (`|x̂−x|/|x| ≈ ε·κ_rt(x)`), which
translates a condition number into an *actual rounding-error bound* and is explicitly a
first-order approximation, validated only "within the first-order factor" in [ATRA]'s own
measurements. Proposition ANEE-2 is about condition numbers composing with each other, which is
exact.

**Experiment Z3** (`anee_experiments.py`, Decimal prec-60 reference) validates this by direct
computation at seven `(x, λ)` points spanning 42 orders of magnitude, composing `power(λ)` then
`log`: the measured relative difference between the composed map's *directly computed* `κ_rt`
and the product of the two hops' individually computed `κ_rt` is **at most 5.46 × 10⁻⁶⁰** —
Decimal-arithmetic-of-the-proof-itself noise, confirming an exact identity rather than an
approximation.

**Corollary (representation selection is certified shortest path).** Tracking `(log κ, B)` pairs
(multiplicative `κ`, additive ulp-bound `B`, per [CANR §3.2]'s exact composition formula) along
a path makes total path cost additive, so representation selection over a *multi-hop* graph is
Dijkstra/Bellman–Ford-solvable directly; since every real conversion has non-negative cost and
error, there are no negative-cost cycles to worry about (a lossy round trip like `f64→f32→f64` is
simply a non-negative-cost cycle, safely ignored by any shortest-path algorithm).

**Falsification of the "graph" framing's novelty.** The corollary is a legitimate, useful
*engineering* abstraction (call a library shortest-path routine instead of hand-composing
bounds) — but the underlying mathematics is 100% inherited: elasticity composition is ordinary
calculus, and propagating certified bounds through a DAG of operations is *exactly* what
FPTaylor/Daisy/Gappa already do for a program's computation graph ([CANR §2], already verified).
The distinction that survives scrutiny: those tools' graphs have *program values* as nodes and
*operations* as edges (error propagation through **one fixed program**); ANEE's representation
graph has *formats themselves* as nodes and *conversions* as edges (a search over **which format
to route data through**, orthogonal to and layered on top of the program's own computation
graph). This orthogonal-layering distinction is real and — per §2.4's dedicated search across ML
quantization (HAQ/HAWQ's node-labeling formulation, which is structurally different), database
encoding selection, and codec/transcoding literature — not found stated explicitly anywhere in
prior art. Every mathematical brick it's built from (elasticity chain rule, Dijkstra,
interval/DAG error propagation) is textbook; the specific assembly is, on the evidence gathered,
not.

## 5. Execution plan formalization (deliverable 5)

§1.2 already established the general form: `π = (G, ℓ)`, a DAG of computation nodes each labeled
with a local `(R, O, A, T, Q, M, H)` choice, of which the mission's single global tuple `P` is the
`|V| = 1` special case. Three further formal points, checked against SciRust's own code
(§3) and the compiler-IR/autotuning literature (§2.1, §2.3):

**5.1 A plan is a refinement of a compute graph, not a new IR.** `G` in `π = (G, ℓ)` is, node-
and edge-for-edge, the same object every compiler IR in §2.1 already builds (LLVM's SSA graph,
MLIR's payload IR, Halide's and TVM's compute DAG). What ANEE adds is only the *label alphabet*
`ℓ: V → 𝓡×𝓞×𝓐×𝓣×𝓠×𝓜×𝓗` — i.e., ANEE proposes no new *graph*, only a numerics-specific *type
system for node attributes* layered onto an existing, well-studied kind of graph. §2.1 settles
how rich any existing IR's attribute alphabet already is: MLIR's `quant` dialect carries
precision-as-type-annotation (in downstream vendor compilers, not core MLIR); TVM's MetaSchedule
and Relay both confirm dtype is fixed *before* scheduling, never a schedulable node attribute;
Halide's and Ansor's `rfactor` mechanism confirms reduction/accumulation order is a programmer
annotation or a parallelization detail, never a node-labeled *choice* among numerically distinct
accumulators anywhere in that lineage. **No existing IR examined carries accumulation-strategy or
reproducibility-level as a first-class, searchable node attribute** — this is a real, if modest,
gap in the label alphabet, not in the underlying graph structure.

**5.2 Plans compose associatively but only trivially so.** Sequential composition of plan
fragments (`π₁ ; π₂` = graph concatenation) is associative by construction (`(π₁;π₂);π₃ =
π₁;(π₂;π₃)` — literally the same graph either way this is parenthesized) — a free, not deep,
algebraic fact. The *hard* algebraic content is in §6.

**5.3 Reproducibility as a scientific artifact (mission question 8).** A plan `π` *is* already a
reproducible scientific artifact exactly to the extent its node labels carry [CANR]'s
determinism ladder (D0–D3) and its certificates (§7) — this was already built and validated in
[CANR] for the `|V|=1` case (the `Bound`/`CandidateVerdict`/`SelectionReport` types in
`transform_search.rs`). Extending it to `|V|>1` is a data-structure generalization (a `Vec` of
per-node reports instead of one), not a new concept. **Portability across architectures**
(mission question 10) is a materially different and harder claim, and §2.3/§2.4 settle it with
primary-source evidence, not speculation: no plan-cache examined anywhere in this investigation
— Kokkos+APEX's launch-parameter YAML, Triton's disk cache, FFTW's wisdom, SPIRAL's learned cost
model, or AutoTVM/Ansor's tuning records — transfers to different hardware without re-search; all
are keyed in a way that includes the specific hardware target, and FFTW's own documentation,
SPIRAL's own platform-tuning experiment, and PetaBricks' own Table 1 all *quantify* the resulting
slowdown when a tuned plan is moved anyway.

## 6. Composition algebra (deliverable 6)

The mission's central claim is that *the composition itself*, not any one stage, is the right
object of optimization. Made precise:

**Derivation.** Any search over compositions needs, at minimum: (i) a way to **sequence** two
plan fragments (`;`), (ii) a way to represent **choice** among alternative fragments achieving
the same sub-result (`+`), and (iii) a **cost/extraction** function picking the best element of a
choice. `;` is associative (§5.2). `+` must be associative and commutative (the order in which
alternatives are considered cannot change the *set* of alternatives) and idempotent
(`a + a = a`: listing the same alternative twice adds nothing) — which makes `(𝒮, +)` exactly an
**idempotent commutative monoid**, i.e. a **join-semilattice**, with `cost` an extraction
function over it.

**Proposition ANEE-3.** Any composition algebra satisfying the minimal requirements above is,
up to presentation, the algebraic skeleton of **equality saturation**: `+`-equivalence classes
are e-graph equivalence classes (implemented efficiently via union-find), `;`-sequenced
fragments are e-nodes, and `cost`-based selection is e-graph extraction. **This independent
derivation is directly confirmed by §2.6's literature review**: equality saturation (Tate et al.,
POPL 2009) is, on the primary source's own terms, precisely this — a set of terms closed under a
congruence relation and rewrite-generated equalities, with extraction as cost-based optimization
over the whole equivalence-closed space — matured into general, reusable, domain-agnostic
infrastructure by egg (Willsey et al., POPL 2021, a Distinguished Paper) and shown, by Diospyros
(VanHattum et al., ASPLOS 2021), to already span two of ANEE's own pipeline stages (operator
selection jointly with data layout) for a real domain.

**Confirmed implication.** "Composition algebra" is not a new mathematical object — it is, at
most, a fresh *domain-specific rule set* (representation-conversion equivalences,
accumulation-strategy equivalences, hardware-specialization equivalences) poured into an
existing, general, mature, fifteen-year-old optimization technique that already has a *second*,
independently-formalized companion algebra for search *strategies themselves* (Elevate/Rise,
ICFP 2020) and a demonstrated fusion of the two (Guided Equality Saturation, POPL 2024) — exactly
mirroring this report's finding for the representation graph (§4) and the SciRust-internal
finding (§3): **the pattern repeating across every formal component examined in this
investigation is "integration of known parts," not invention of new parts.** What §2.6 confirms
is genuinely absent is narrower and more specific than "a composition algebra": a single
published numerics-flavored rule set (spanning representation conversions, accumulation-strategy
equivalences, and hardware-specialization choices at once) with a *certified*, not merely
empirical, cost/extraction function — nobody has assembled that particular e-graph rule set yet,
which is an engineering gap, not a mathematical one.

## 7. Numerical contract model (deliverable 7)

A contract for primitive `f` is the property tuple the mission lists: domain, overflow behavior,
conditioning, forward error, backward error, reproducibility level, SIMD compatibility,
invertibility, monotonicity, information loss. The mission's question 9 asks whether contracts
compose automatically. Answered property-by-property, each via a *different*, already-classical
composition rule — this heterogeneity is itself the main finding:

| Property | Composition rule for `g∘f` | Status |
|---|---|---|
| Domain | `D_{g∘f} = f⁻¹(D_g) ∩ D_f` | Mechanical for interval/box domains (interval arithmetic); undecidable in general for data-dependent regions |
| Conditioning (`κ_rt`) | **Exact product** (§4, Proposition ANEE-2) | Classical (elasticity chain rule) |
| Forward/backward error | Weighted sum, `e_{g∘f} ≈ κ_g·e_f + e_g` | Classical (already derived and *numerically validated* across a 3-stage pipeline in [CANR §1]'s error decomposition, Y1–Y6) |
| Reproducibility/determinism level | **Meet (infimum)** in the D0 < D1/D2 < … partial order, *not* an average | Newly stated precisely for [CANR]'s ladder by this report; the underlying "weakest-link" pattern is not new mathematics. **Experiment Z4** confirms it directly: piping a stage that is exactly order-invariant across 20 random permutations of its own input (0 of 20 trials differ) into an ordinary order-dependent float64 accumulation stage yields an overall pipeline whose output is **not** thereby made order-invariant — the D1 stage's guarantee does not upgrade the D0 stage |
| SIMD compatibility, invertibility, monotonicity | Logical AND / preservation rules | Mechanical, boolean |
| Information loss | **One-directional inequality only** (loss can only stay equal or increase through composition) | This is exactly the classical **Data Processing Inequality** from information theory — pre-existing, not new |

**Verdict on question 9.** §2.5's dedicated literature check confirms this directly: contracts
compose *mechanically*, property by property, using rules that are all independently classical
(chain rule, lattice meet, logical AND, Data Processing Inequality) — but **no single rule**
applies to *all* properties at once, so a real "numerical contract" object must bundle several
structurally different composition procedures behind one interface. Whether that bundling has
been done before, for an *arbitrary* composed plan (not just [CANR]'s fixed 3-stage `D(F(E(x)))`
shape), is answered by §2.5: **no.** FPTaylor/Gappa/original PRECiSA certify one flat expression;
the few genuine 2023–2025 modularity advances (Abbasi & Darulova, SAS 2023; PRECiSA 4.0) are
intra-tool only; a positive result exists for one narrow primitive family (Zhang & Aiken, CAV
2025, accumulation networks); and a June 2026 PhD dissertation is titled around exactly this
still-open general problem. Bundling several classical composition rules behind one interface,
for arbitrary composed plans including hardware-specific ones, was not found assembled anywhere.

## 8. Complexity analysis (deliverable 8)

**Experiment Z1** (`anee_experiments.py`) computes the actual combinatorial cost of joint search
using SciRust's *own, currently-real* dictionary sizes wherever they exist (not hypothetical
numbers): `|R| = 7` (`transform_search::Representation`), `|A| = 5` (`autotune_accumulate::
AccumMethod`), `|H| = 7` (`dispatch::BackendKind`), with small, explicitly-labeled illustrative
placeholders for the three axes SciRust does not yet catalogue at all (`O`, `Q`, `M` — confirmed
absent from the codebase in §3). Even at these deliberately conservative sizes:

```
sum of per-axis sizes  (7 independent single-axis autotune_by calls):     38
product of axis sizes  (one joint search over the full plan space):   94,080
blow-up factor:                                                     ~2,476x
```

and, more tellingly, for `k` candidates per axis uniformly: independent cost is `7k` (linear),
joint cost is `k⁷`. At `k = 10` — a plausible size once `T`, `Q`, `M` are properly catalogued —
that is `70` vs. `10,000,000`. **Exhaustive joint search over a realistic ANEE plan space is not
computationally feasible**; this is not a criticism specific to ANEE, it is the generic
combinatorial-explosion fact that motivated the entire algorithm-configuration research area
(§2.4's Rice/CASH/SMAC lineage) decades ago.

**Experiment Z2** checks whether cheaper alternatives to exhaustive joint search — the two
approaches actually available to SciRust today (independent per-axis `autotune_by` calls; or a
cheap iterative coordinate-descent wrapper around them) — are adequate substitutes, using a
constructed two-basin, genuinely non-separable objective (a mixture of two Gaussian reward bumps
at different centers — provably *not* decomposable as `f(x₁) + g(x₂)` for any `f, g`, unlike this
report's first, buggy attempt at the same experiment, which accidentally used a separable
interaction term and — correctly, but uninformatively — showed no gap at all). Three regimes,
all produced by the committed script:

- **No interaction** (λ ≤ 3): independent, coordinate-descent, and exhaustive joint search tie
  exactly (0 gap) — the converse check that independent per-axis search is only exact when the
  objective *is* separable.
- **Mild interaction** (λ = 6–10): one-shot independent search combines a piece of each basin's
  answer into a point near *neither* basin (gap = 9.996 of a −10 optimum) — a direct,
  constructed generalization of [CANR §1]'s H3 finding that representation and operator "must be
  selected as pairs" to arbitrary interacting axis pairs. Coordinate descent still recovers the
  true optimum here (0 gap): cheap local search compensates for mild interaction.
- **Strong interaction** (λ = 12–15): the second basin overtakes the first as the *true* global
  optimum, but coordinate descent's trajectory — captured early by the now-suboptimal first
  basin — no longer reaches it either (gap = 2.0, then 5.0). Only exhaustive search is correct in
  every regime tested, and Z1 already showed exhaustive search is infeasible at realistic scale.

**Reading.** These two results together are this report's sharpest, self-contained technical
finding: **joint search over the ANEE plan space is both (a) mathematically necessary in
general** (per-axis-independent search provably fails once axes interact, which [CANR] already
showed happens for real representation/operator pairs) **and (b) combinatorially infeasible to
perform exhaustively**, which means any working "Plan Optimizer" component of ANEE's proposed
architecture is *not optional engineering plumbing* — it would have to reimplement genuine
search technology (Bayesian optimization, cost models, successive halving, or similar) of the
kind the algorithm-configuration literature (SMAC, irace — already verified in [CANR]) and
modern auto-schedulers (§2.3 — AutoTVM's XGBoost/TreeRNN cost model, Ansor's evolutionary search)
were built specifically to provide. This alone is enough to place the "Plan Optimizer" firmly in
*applied algorithm-configuration engineering*, not new optimization theory.

## 9. Counterexamples (deliverable 9)

Concrete facts that directly refute a literal reading of the mission's framing:

1. *"Determine whether numerical representation itself may become an optimization variable"*
   (implying open) — **refuted**: SPIRAL (Püschel et al. 2005) already optimizes over
   fixed-point representation with a *closed-form, algebraically derived* worst-case error bound,
   twenty years before this mission, for the class of linear DSP transforms.
2. *"The execution plan itself may change depending on hardware"* (implying novel) —
   **refuted**: SPIRAL's Formula Generator already produces genuinely different recursive
   algorithm breakdowns per platform (not just re-vectorization); glibc's `ifunc`-dispatched
   `memcpy` already ships a fundamentally different mechanism (hardware `rep movsb`) as one of
   several dispatch targets, in production, for decades.
3. *"The engine may remember: (kernel, distribution, hardware) → best execution plan →
   confidence → certificate → benchmark history"* (implying absent) — **refuted in shape,
   confirmed in specifics**: AutoTVM's TopHub is exactly `(target, workload, config, cost)`,
   versioned and community-maintained, in production since 2018 — missing only a confidence
   field and, critically (§12), a *distribution*-aware key.
4. *"Can execution plans become reproducible scientific artifacts?"* (implying unaddressed) —
   **refuted**: FFTW's "wisdom" (1998) already serializes chosen plans to disk; MLIR's Transform
   dialect (2022) already makes a schedule program-independent, serializable IR.
5. *Implicit framing that P=argmin J(P) over a structured multi-axis space is a new
   formulation* — **refuted**: this is Rice's Algorithm Selection Problem (1976), and Rice's own
   research group already built a feature→algorithm selector for scientific numerical routines
   (PYTHIA, 1996) three decades before this mission; the parametrized-configuration generalization
   is CASH (2013), already solved by SMAC (2011, verified in [CANR]).
6. *Implicit framing that this research program's design proposals reliably become adopted
   engineering* — **refuted by this repository's own history**: [CANR §9] proposed a shared
   benchmark-result JSON schema one phase ago; §3 confirms it was never implemented by any of the
   ~16 actual benchmark harnesses subsequently written in this codebase, each of which rolled its
   own incompatible shape instead.

## 10. Falsification attempts and kill criteria (deliverable 10)

Following [CANR §10]/[ATRA §7]'s convention — standing criteria a claim must survive:

1. **Killed**: "the P=(R,O,A,T,Q,M,H) formulation is a new optimization object." It is Rice
   (1976) / CASH (2013) instantiated on a 7-axis numerics-specific space (§2.4).
2. **Killed**: "execution plans are portable artifacts across hardware architectures."
   Quantitatively disproven by three independent primary sources (FFTW's docs, SPIRAL's own
   platform-tuning experiment, PetaBricks' Table 1: up to 2.35× slowdown) and confirmed to remain
   an open, actively-worked research problem through at least April 2026 (§2.4).
3. **Killed**: "the Representation DB → Generator → Analyzer → Certificate Engine → Benchmark →
   Optimizer → Dispatcher architecture is new." FFTW/ATLAS (1998), SPIRAL (2005), AutoTVM/Ansor
   (2018/2020) already instantiate this shape piecewise, and MLIR's Transform dialect (2022)
   already provides the portable-schedule-as-IR component (§2.1, §2.3).
4. **Downgraded to engineering, not research**: "joint multi-axis search is the right approach."
   Z1/Z2 (§8) show it is both mathematically necessary under interaction (extending [CANR]'s own
   H3 finding) and combinatorially infeasible to run exhaustively — which is exactly why the
   algorithm-configuration field (SMAC/irace) exists, not evidence of a new problem.
5. **Killed as stated, narrow variant survives**: "numerical contracts compose automatically
   through arbitrary composed plans." Confirmed absent as an integrated system anywhere in the
   literature (§2.5); what composes automatically is each *individual* property, via classical,
   mutually different rules (§7) — a real but narrower claim.
6. **Killed**: "composition itself is a new optimization object." Confirmed to already be
   equality saturation's explicit design thesis since 2009, matured into general infrastructure
   by 2021 (§2.6, §6).
7. **Not falsified — the strongest surviving claim**: "a representation graph, with formats as
   nodes and certified conversions as edges, generalizing single-hop certificate gating (as in
   `transform_search.rs`) to multi-hop shortest-path search, and keyed for caching by data
   *distribution* as well as kernel and hardware, is a genuinely unaddressed formulation."
   Targeted search across ML quantization (HAQ/HAWQ), database encoding selection, and codec
   literature found no precedent (§2.4); this is reported as an absence-of-evidence finding, not
   proof, and would be falsified immediately by discovery of prior art — but none surfaced despite
   deliberate search. **This is the one candidate this report cannot close.**

## 11. Components already known (deliverable 11)

Organized by the mission's own "Execution planner" architecture, listing the earliest/clearest
precedent found for each box — every box has at least one:

- **Representation database**: `transform_search.rs`'s own dictionary (this repo); FPBench's
  interchange schema (2016 onward); MLIR's `quant` dialect type annotations.
- **Plan generator**: `scirust-tensor-compile`'s `TensorGraph` (this repo, elementwise-fusion
  only); MLIR's Transform dialect (2022); Halide/TVM/Ansor's schedule/sketch generators
  (2013–2020); **SPIRAL's Formula Generator (2005) is the only one found generating distinct
  algorithms, not just schedules — the closest match to ANEE's actual ambition here.**
- **Static analyzer**: AutoTVM/Ansor's learned cost models (predict *performance*, never
  correctness); XLA's HLO passes.
- **Certificate engine**: **SPIRAL's exact symbolic (GAP computer-algebra) verification plus
  closed-form error bound (2005) is the only general-purpose certificate engine found anywhere
  in this entire investigation** — and it works only by exploiting the linearity of DSP
  transforms, a property general numerical kernels lack. `transform_search.rs`'s own κ_rt gate
  (this repo, [CANR]) is the same idea at `|V|=1` scope.
- **Empirical benchmark**: universal across the whole autotuning cluster — FFTW, ATLAS, OpenTuner,
  AutoTVM, Ansor, CUTLASS's Profiler, `transform_autotune.rs` (this repo) — the single
  best-precedented box in the entire architecture.
- **Plan optimizer**: OpenTuner's search-technique ensemble (2014); SMAC (2011)/irace (2016,
  both verified in [CANR]); Ansor's evolutionary search plus cost model (2020).
- **Runtime dispatcher**: `scirust-tensor-runtime`'s register machine plus `BackendKind` (this
  repo); FFTW's plan executor; AutoTVM's compiled-schedule application.
- **Plan cache keyed by (kernel, hardware)**: FFTW's wisdom (1998); AutoTVM's TopHub (2018);
  SPIRAL's offline-learned cost model (2005); Kokkos+APEX's converged-tuning YAML (2020s,
  launch-parameters only).
- **Composition as the optimization object**: equality saturation (Tate et al., POPL 2009; egg,
  POPL 2021); Diospyros (ASPLOS 2021) for two of ANEE's own stages jointly; PetaBricks (2009) for
  whole-algorithm composition specifically.
- **Non-separability of jointly-chosen axes**: [CANR §1] H3 (log representation needs its
  matching operator) is itself already an instance; the entire algorithm-configuration field
  exists because of this fact.
- **κ_rt composing exactly multiplicatively under composition**: a direct corollary of [ATRA
  Prop. 4] plus Higham's standard condition-number chain rule (already cited three times over in
  this program) — made explicit and validated exactly, at Decimal prec-60, this phase (§4, Z3).
- **Determinism composing as a lattice meet, not an average**: implicit already in [CANR §6.1]'s
  SIMD-width-dependent-reduction discussion; stated precisely and validated empirically this
  phase (§7, Z4).

## 12. Components that appear potentially original (deliverable 12)

Four candidates survive this phase's scrutiny, ranked by how much of their surrounding
machinery is genuinely absent from prior art (vs. built from classical or already-precedented
parts):

1. **The representation graph as an explicit path-search problem over format-nodes with
   certified, multi-attribute conversion edges** (§4, §10.7). Structurally sound (built from an
   exact, proven identity), practically unprecedented on the evidence gathered. The single
   strongest candidate in this report.
2. **Distribution-aware plan caching.** A *schedule's* optimality depends only on problem shape
   and hardware — a pure, cacheable, data-independent property, which is exactly why
   FFTW/AutoTVM/SPIRAL's caches key on `(kernel-shape, hardware)` and work. A
   *representation/precision* choice's **correctness** (not just its speed) depends on the data's
   numeric distribution (range, conditioning, sparsity) — [CANR]'s own H1–H3 findings are
   precisely about this. No cache examined in this investigation keys on data distribution; all
   key on shape and/or hardware alone. Extending the cache key from `(kernel, hardware)` to
   `(kernel, distribution, hardware)` is a small-sounding but structurally real change with no
   found precedent — because for every *other* system in §2.3, it would be pointless (their cached
   choice doesn't depend on data values), and only becomes necessary once representation/precision
   joins the cached axis set.
3. **A unified numerical-contract object spanning all of the mission's listed properties**, with
   one interface hiding several structurally different composition rules (§7's chain
   rule/meet/AND/Data-Processing-Inequality table). No individual rule is new; the *bundling*,
   for an arbitrary composed plan rather than one flat expression, was not found assembled
   anywhere (§2.5).
4. **Determinism/reproducibility level as a first-class axis of an algorithm-selection
   performance space.** Confirmed absent from Rice's original framework and from Kotthoff's 2014
   survey of the entire field (§2.4) — every other axis of ANEE's `C` (accuracy, latency,
   throughput, energy, memory) has a well-worn place in the algorithm-configuration literature;
   determinism does not.

**None of these four is validated by this report** — each is an absence-of-precedent finding
(§10.7's caveat applies to all four), not a demonstrated result. §13 recommends how to find out
cheaply.

## 13. Recommendations for SciRust (deliverable 13)

The internal baseline (§3) and external comparison (§2) point to the same shape of answer
[CANR §11]–[§12] already reached one phase earlier: real, narrow, staged engineering value, no
research program, and — this phase's addition — an explicit warning drawn from this repository's
own recent history (§3, §9.6: [CANR]'s own unification proposal was not adopted in practice)
against recommending unification-by-design-document again.

**Phase A — connect what already exists (low risk, no new research).** SciRust already has, in
five-plus independent pieces (§3), essentially every box of ANEE's architecture at `|V|≈1`
scope. Before any new abstraction: (i) make `transform_search.rs`'s certificate gate and
`autotune_accumulate.rs`'s accumulator autotuner runnable *jointly* for the two axes [CANR]'s own
H3 already proved interact (representation × accumulation), using Z2's coordinate-descent
approach (§8) as the cheap default and exhaustive-pair search as the fallback for small
dictionaries; (ii) document (at minimum) and ideally unify the three currently-uncoordinated
hardware-backend abstractions (`scirust-simd::BackendKind`, `scirust-gpu::RawComputeBackend`,
`scirust-cuda`'s feature gates) confirmed in §3 — this is real, bounded, valuable engineering
regardless of ANEE's fate.

**Phase B — enforce, don't just propose, a shared benchmark artifact schema.** Re-attempt
[CANR §9]'s JSON schema, but as a shared crate/trait new harnesses must import (making
divergence a compile error, not a documentation lapse) — a direct, specific response to §9.6's
finding that the design-only version did not stick.

**Phase C — one narrow, falsifiable prototype, not a crate ecosystem.** Build the representation
graph (§4) as literal shortest-path search over SciRust's *existing* seven-member `Representation`
dictionary, with a plan cache keyed by `(kernel, distribution-summary, hardware)` — testing §12's
two strongest candidates together. Benchmark it directly against [CANR]'s existing S1–S4 pipeline
run once per axis (the honest baseline, per this phase's own Z2 experiment) on the same
[CANR §9] canonical benchmark suite. **Kill criterion, stated in advance per this program's own
established discipline**: if the graph/distribution-aware cache does not beat sequential
per-axis `autotune_by` calls by a pre-registered margin on held-out data, close this line
exactly as [TSA]'s Γ-transform and [ATRA]'s hypercomplex-transform directions were closed.

**Explicitly not recommended**: a general 7-axis joint planner; a general numerical-contract type
system spanning arbitrary plans; a new "ANEE" crate or crate ecosystem; treating any of this as
a systems-research paper topic beyond the narrow, falsifiable claim in Phase C.

## 14. Final decision (deliverable 14) and novelty ladder

Reusing [CANR §12]'s ladder (established at level 6 = "a new tool niche," levels 7–8 unreached
by any of the three prior phases):

- The `P=(R,O,A,T,Q,M,H)` formulation, taken as stated: **level 1** (Rice 1976, known theory),
  packaged for numerics at roughly the level [CANR] already reached for two axes.
- The "Execution planner" architecture (DB → Generator → Analyzer → Certificate → Benchmark →
  Optimizer → Dispatcher): **level 2–3** (FFTW/ATLAS/SPIRAL/AutoTVM/Ansor/MLIR-Transform already
  built every box; SciRust already has independent, unconnected instances of most boxes — §3).
- "Composition instead of optimization" (§6): **level 2** (equality saturation, POPL 2009/2021,
  already formalizes and ships exactly this; Diospyros, ASPLOS 2021, already applies it to two
  of ANEE's own axes jointly).
- The non-separability/combinatorial-infeasibility analysis (§8, Z1/Z2) and the exact κ_rt/meet
  composition rules (§4/§7, Z3/Z4): **level 3** (known mathematics — chain rule, lattice meet,
  Data Processing Inequality — freshly and precisely stated for this program's own certificate/
  determinism types, validated numerically, not new theorems).
- Numerical contracts as a *bundled, heterogeneous-rule* object spanning an arbitrary composed
  plan: **level 4–5** (a real, useful implementation/integration contribution if built — no
  individual rule is new, no prior bundling was found, §2.5).
- Distribution-aware `(kernel, distribution, hardware) → plan` caching: **level 4–5**, same
  character — a small, precise, apparently-unaddressed extension of a well-understood caching
  pattern.
- The representation graph as certified path search over format-nodes: **level 5–6 at best, if
  built and shown to beat the CANR baseline** (§13 Phase C) — potentially a new tool niche
  exactly the size and shape of [CANR]'s own level-6 finding, generalized from 2 axes with
  single-hop selection to N axes with multi-hop search. **Level 7–8 (new theorem, new algorithm
  class, new field): none found, consistent with all three prior phases of this program.**

**Decision: no internal runtime, no new crate ecosystem, no research paper on the full ANEE
vision as stated.** Phase A/B (§13) as ordinary, valuable engineering. Phase C as one narrow,
falsifiable, benchmarked prototype — worth perhaps two to three weeks of effort against a
pre-registered kill criterion — and *only if it survives that benchmark* does a focused paper
become defensible, on the specific, narrow claim ("distribution-aware representation-graph
caching beats sequential per-axis autotuning for numerical kernel selection"), not on "ANEE" as
a named field or framework. This mirrors, and extends one step further in the same direction as,
[TSA]'s "survey + engineering track + focused open questions" and [CANR]'s "engineering module +
benchmark tool, novelty ceiling reached" verdicts — the fourth consecutive phase of this research
program to reach a "real but narrow engineering value, not a new research field" conclusion.

## 15. The mission's scientific questions, answered directly

1. **Does ANEE reduce to compiler optimization?** Mostly, for the plan-as-artifact and
   hardware-specialization pieces (MLIR Transform dialect, XLA's `AutotunerPass`, SPIRAL) — not
   for the representation axis or contract composition, which compiler-optimization literature
   does not cover.
2. **Does it reduce to autotuning?** Architecturally, yes — the search-generate-benchmark-cache
   pipeline shape is FFTW/ATLAS/AutoTVM/Ansor/SPIRAL, decades old. The specific *axis set*
   (representation/precision searched jointly with schedule and hardware) is not fully covered by
   any single existing autotuner, though every individual axis is covered by some autotuner.
3. **Does it reduce to mixed precision?** Partially — the `Q` axis is well-covered (Precimonious/
   FPTuner/Daisy, and 2023–2026 joint precision+one-other-axis work); no source found searches
   precision jointly with more than one other axis at once.
4. **Does it reduce to numerical DSL scheduling?** Largely, for plan generation (Halide/TVM/MLIR
   Transform-dialect lineage) — these DSLs do not treat representation/precision as a scheduled
   axis.
5. **Does it reduce to existing runtime systems?** For dispatch, yes (SciRust's own
   `tensor-runtime`+`BackendKind`; TVM's/XLA's runtimes); no runtime system found also carries a
   certificate/contract layer.
6. **Is the representation graph genuinely useful?** Structurally sound and exact (§4,
   Proposition ANEE-2, validated at Decimal prec-60 by Z3); practically unprecedented as a
   path-search formulation (§2.4, §10.7, §12.1) — plausibly useful, **not yet demonstrated**.
   This is this report's most open, most falsifiable-in-the-future answer.
7. **Is composition itself a new optimization object?** No — confirmed directly: the minimal
   algebra any composition-with-choice construction needs (§6) is, on the primary source's own
   terms, exactly what equality saturation formalized in 2009 and shipped as general
   infrastructure by 2021 (§2.6); PetaBricks (2009) already frames "compose whole algorithmic
   choices" explicitly, just without ANEE's representation/precision axes.
8. **Can execution plans become reproducible scientific artifacts?** Yes, and this is close to
   already done: extending [CANR]'s validated per-instance certificate/determinism reporting from
   `|V|=1` to `|V|>1` (§5.3) is a data-structure generalization, not a new concept.
9. **Can certificates compose automatically?** Mechanically, per-property, yes — using classical,
   *different* rules per property (chain rule / lattice meet / logical AND / Data Processing
   Inequality, §7) — but no tool found composes across an arbitrary plan for *all* properties at
   once; genuine forward-error certificate composition across independently-verified pipelined
   kernels is confirmed to still be an open research problem as of a 2026 PhD thesis (§2.5).
10. **Can execution plans become portable between architectures?** No — quantitatively refuted
    for every existing schedule-only plan-cache checked (FFTW/SPIRAL/PetaBricks/AutoTVM, §2.3–2.4)
    and confirmed to remain an actively-worked, unsolved research frontier through at least
    April 2026. ANEE would inherit this open problem, not resolve it.

---

## Addendum 1 (2026-07-17, same day): Phase C prototype — results

§13's Phase C prototype was built and benchmarked the same day this report was first drafted.
This addendum reports the outcome without altering §§0–15 above, which remain the record of
what was known and recommended *before* the prototype existed — matching this program's
established practice of appending updates rather than rewriting prior verdicts.

**What was built.** `scirust-core/src/representation_graph.rs` (library: a `RepresentationChoice`
node type generalizing [`Representation`] with an explicit `Identity` member, a `PlanCache` keyed
by `(kernel, DistributionSummary, BackendKind)`, and `sequential_baseline`/`joint_search`
functions) plus `scirust-core/examples/anee_phase_c_prototype.rs` (the benchmark). Both build
clean and pass `cargo clippy -D warnings` / `cargo fmt --check` on their crates, and the new
module's tests (structural invariants, not the empirical finding itself — see below) pass
alongside all 880 pre-existing `scirust-core` tests. Two small, well-justified changes to
existing code were needed to "connect what already exists," exactly as §13 Phase A recommended:
`transform_autotune::UniformQuantizer` was widened from private to `pub(crate)` (reused as-is,
not re-derived), and `scirust_simd::dispatch::BackendKind` gained a derived `Hash` impl (needed
to use it as a cache-key component).

**The task, concretely.** A "compress, store, later aggregate" pipeline for positive sensor-style
readings: encode via a representation `R` (dictionary: `Identity`, `Log`, `Log1p`, `Power(0.5)`,
`Anscombe`), quantize/dequantize at a fixed 64 levels (matching [CANR]'s own convention;
deliberately not a searched axis, keeping the prototype narrow), decode, narrow to `f32`, then
accumulate via strategy `A` (the existing 5-member `AccumMethod` dictionary). Objective: held-out
relative error of the accumulated total against the exact sum of the *original* readings. This
concretely generalizes [CANR §1]'s H3 finding — "representations must be selected as pairs with
operators" — from `(representation, operator)` to `(representation, accumulation)`.

**Methodology strengthening beyond the pre-registration.** The pre-registered criterion (§13)
used a single dev/eval draw per family. Before treating any number as final, the benchmark was
extended to re-score both chosen (fixed) plans on 3 additional fresh held-out seeds never used
for selection — matching `certified_numerics.rs`'s own established "select on one seed, validate
on fresh seeds" convention — and the kill-criterion decision was based on the 3-seed mean, not
the single draw. This mattered: on the single eval draw, the "benign" family showed joint search
*losing* by 68.6% (an artifact of both approaches already sitting near the quantization noise
floor, where single-draw RNG noise dominates); the 3-seed mean revealed the true, smaller, but
still real 23.1% joint win. Reporting the single-draw number without this check would have been
methodologically unsound and is exactly the failure mode CANR's own held-out discipline exists to
catch.

**Results** (`cargo run -p scirust-core --example anee_phase_c_prototype --release`, seeds fixed
in-source, x86_64/AVX-512 backend detected at runtime — reproducible from the committed code):

| Family | Sequential baseline (plan, 3-seed mean rel. err.) | Joint search (plan, 3-seed mean rel. err.) | Relative reduction | Kill criterion (≥20%) |
|---|---|---|---|---|
| benign | `identity+PairwiseF32`, 7.69×10⁻⁵ | `anscombe+NaiveF32`, 5.92×10⁻⁵ | 23.1% | MET (narrowly; noisy) |
| wide-range | `identity+NaiveF32`, 1.66×10⁻¹ | `power(0.5)+PairwiseF32`, 1.24×10⁻³ | 99.3% | MET (decisively) |
| stagnation-prone | `identity+NaiveF32`, 3.22×10⁻² | `anscombe+StochasticF32`, 2.63×10⁻⁴ | 99.2% | MET (decisively) |

**Verdict: the pre-registered kill criterion is MET — 3 of 3 families, exceeding the "at least 2
of 3" bar.** Reported with the honesty this program requires: the two wins on wide-range and
stagnation-prone are decisive and unambiguous (every one of the 3 fresh-seed scores for the joint
plan beats every one of the 3 fresh-seed scores for the sequential plan — no overlap at all); the
win on benign is real but small and noisy (the fresh-seed score ranges overlap substantially,
and only the *mean* clears the 20% bar) — this task's control family, as designed, shows
representation/accumulation choice barely matters once both approaches are already near the
quantization noise floor, exactly as expected going in.

**Why sequential loses.** The mechanism is fully explainable from the code, not a black box:
`sequential_baseline`'s S1 step picks the *cheapest* certified-safe representation, and every
member of the 5-representation dictionary is certified-safe on all three tested families (no
domain/invalid-region rejections occur here) — so the cost-based tie-break deterministically
picks `Identity` (cost 0) every time, regardless of whether identity actually serves the
*downstream* quantization+accumulation objective well. This is [CANR §1]'s H3 finding, made
concrete: a certificate-cost-only selector cannot see the downstream task, and only joint search
(or an objective-aware S1) can.

**An important caveat about certificates, surfaced by this run.** The joint search's *winning*
representation on `stagnation-prone` (`Anscombe`) carries a *much looser* analytic round-trip
certificate (3011.93 ulps, vs. 8.00 ulps for `Identity` — Anscombe's `κ_rt = 2 + 0.75/x` diverges
for the small values in that family) and yet **empirically wins by 99.2%** on the real objective.
This means a purely analytic, certificate-only extension of the representation graph (ranking
candidates by tightest round-trip bound, as [CANR]'s own S1 gate does) would **not** have found
this winning plan — reinforcing, with a fresh concrete example, [CANR §8]'s own finding that "S1
is sound but conservative" and the empirical S3/S4 layer is not optional. The representation
graph's *shortest-path-by-certificate* framing (§4) is validated as a sound *filter* (nothing
certified-unsafe was ever selected, in this run or any prior phase's), not as a sufficient
*ranking* for real task performance.

**Distribution-aware cache.** [`PlanCache`] correctly hit on a fresh sample from the *same* family
in all 3 cases (e.g., for `wide-range`, the cached plan scored 4.78×10⁻⁴ on a fresh draw,
consistent with the 1.24×10⁻³ 3-seed mean — the same order of magnitude, not a degraded reuse).
The distribution-mismatch demo — applying the plan cached for `benign` to a `stagnation-prone`
sample, as a `(kernel, hardware)`-only cache (every existing system surveyed in §2) would be
forced to — produced a relative error **3793.5× worse** than that family's own properly-searched
plan. This is concrete, quantitative support for §12.2's argued gap: existing autotuning caches
key on kernel shape and/or hardware because a *schedule's* optimum doesn't depend on data values;
a *representation* choice's correctness does, and this run shows exactly how badly that
assumption breaks when violated.

**What this does and does not establish.** This is one task (a specific compress-then-aggregate
pipeline), one small pair of dictionaries (5 representations × 5 accumulators = 25 joint
candidates), and one fixed quantization level count — deliberately narrow, per §13's own
instruction not to build a general planner. It does **not** establish that joint search wins in
general, on other kernels, at other dictionary sizes, or under the combinatorial pressure §8's
`k⁷` analysis warns about. What it does establish: on a first real, non-synthetic test, the
specific, narrow, previously-unvalidated claim from §12.1/§12.2 — that a representation graph
with a distribution-aware cache beats sequential per-axis selection — **survived its own
pre-registered falsification bar**, with an honestly-reported mix of a decisive mechanism (S1's
cost-blindness to the downstream objective) and a caveat (certificates alone would not have found
the winning plan; empirical validation remains load-bearing).

**Updated novelty-ladder status (§14).** The representation-graph-plus-distribution-aware-cache
candidate moves from "level 5–6 at best, if built and shown to beat the CANR baseline" (§14,
original text, unconfirmed) to **confirmed at level 6 for this one task** — a genuine, working,
narrow tool-niche result, on the same footing as [CANR]'s own level-6 finding, not yet
generalized. Per §13's own discipline, the next step is *not* to broaden scope (no new crate, no
general planner) but to replicate on 1–2 more kernels from [CANR §9]'s canonical benchmark
families before any claim stronger than "this specific prototype survived this specific
benchmark" is warranted.

## Addendum 2 (2026-07-17, same day): Phase C kernel 2 replication attempt — results

The first addendum's own closing line named the honest next step: replicate on 1–2 more kernels
before generalizing. This addendum reports that attempt. As with Addendum 1, §§0–15 and Addendum
1 are left unaltered.

**What was tested.** Hypercomplex orientation averaging, mirroring [ATRA]'s own X5 experiment:
average `N=100` noisy unit-quaternion observations (isotropic Gaussian-angle noise around a
random axis, `σ ∈ {0.2, 0.8, 1.5}` rad — [ATRA X5]'s exact three levels) of a fixed true
orientation, over 20 trials per condition, and measure angular error in degrees. Implemented in
`scirust-core/src/representation_graph_quaternion.rs`, using
`scirust_simd::geometry::quaternion::Quaternion<f64>` — a real, generic, deterministic quaternion
type already in this workspace (`slerp`/`nlerp`/`to_axis_angle`/`from_axis_angle`/`normalize`),
not reimplemented. This module and its companion example
(`examples/anee_phase_c_kernel2_quaternion.rs`) are gated behind the `portable-simd` feature
(`scirust_simd::geometry` is itself feature-gated) — built and tested via `cargo test -p
scirust-core --lib --features portable-simd` and `cargo run -p scirust-core --features
portable-simd --example anee_phase_c_kernel2_quaternion --release`. A `required-features`
`[[example]]` entry was added to `scirust-core/Cargo.toml` (matching `scirust-simd/Cargo.toml`'s
own established pattern for its `transformer-inference`-gated examples) so default-feature
`cargo build/clippy --examples` skips this example rather than failing on it.

Two charts (`R`) were compared: [`Chart::Componentwise`] (ambient `ℝ⁴` mean + renormalize —
[ATRA X5]'s "componentwise mean + renorm") and [`Chart::LogChart`] (a fixed 2-iteration
Karcher-style tangent-space mean via log/exp maps — [ATRA X5]'s "Karcher mean"). [ATRA X5]'s
third method (the chordal/Markley mean, requiring a symmetric eigensolver this prototype had no
independent reason to add) was **not** replicated — an honest partial (2-of-3-method), not full,
replication of ATRA X5's own comparison. Accumulation (`A`) reused the unmodified 5-member
`AccumMethod` dictionary from kernel 1.

**Pre-registered kill criterion: the same bar as kernel 1**, deliberately not tuned per kernel —
joint `(Chart, A)` search must reduce mean angular error vs. sequential (always `Componentwise`,
mirroring kernel 1's `Identity`-always-wins cost tie-break, then `A` autotuned) by ≥20% relative
on ≥2 of the 3 tested noise levels, validated on 3 fresh held-out seeds beyond the dev/eval draw.

**Result: the kill criterion is NOT MET.** Only 1 of 3 noise levels (`σ = 1.5`) shows the required
≥20% reduction — short of the pre-registered "≥2 of 3" bar. **This is a genuine non-replication,
reported as such, not explained away:**

| σ (rad) | Sequential (plan, 3-seed mean) | Joint (plan, 3-seed mean) | Relative reduction | Criterion |
|---|---|---|---|---|
| 0.2 | `componentwise+NaiveF32`, 1.061° | `componentwise+NaiveF32`, 1.061° | 0.0% | not met |
| 0.8 | `componentwise+StochasticF32`, 4.392° | `componentwise+StochasticF32`, 4.392° | 0.0% | not met |
| 1.5 | `componentwise+NaiveF32`, 17.939° | `log-chart+StochasticF32`, 9.579° | 46.6% | MET |

**The mechanism differs from kernel 1, and that difference is the interesting finding.** At
`σ = 0.2` and `σ = 0.8`, sequential and joint search select the **identical** plan — not because
joint search failed to explore, but because a direct chart-only ablation (holding accumulation
fixed at `NeumaierF32`) shows `Componentwise` genuinely is as good as or better than `LogChart` at
these noise levels (1.066° vs. 1.069° at `σ=0.2`; 4.077° vs. 4.210° at `σ=0.8` — `LogChart`
slightly *worse* both times). Unlike kernel 1, where the sequential baseline's cost-based tie-break
was demonstrably *myopic* to the downstream objective (§13/Addendum 1's central mechanism), here
the cheap default (`Componentwise`) is not myopic at low/medium noise — it already is the right
choice, so there is no exploitable `(R,A)` interaction for joint search to find. Only at `σ = 1.5`
does `LogChart` pull ahead (13.267° vs. 10.204° in the same chart-only ablation), and **most of
that win comes from the chart choice alone** — accumulation strategy adds a smaller further
refinement (10.204° → 9.579–9.920° across held-out seeds), unlike kernel 1 where both axes were
jointly essential.

**Independent cross-validation of the implementation.** Three structural unit tests pass
(zero-noise exact recovery to `<10⁻³°` for both charts; hemisphere-sign-flip robustness; log/exp
maps verified as exact inverses to `<10⁻⁶°`), and — more importantly — the `LogChart` result at
`σ = 1.5` (9.58–9.92° across held-out seeds) lands close to [ATRA X5]'s own, independently
computed Karcher-mean reference value (**9.959°**, different RNG, different implementation
language) — a reassuring consistency signal that this module's quaternion math is correct, even
though the overall pre-registered replication claim did not survive.

**What this changes about the report's conclusions.** Kernel 1 (§13 Phase C, Addendum 1) is
unaffected on its own terms — it is a real, reproducible result for that specific task. But taken
together with kernel 2's failure to replicate, the honest combined finding is sharper and more
useful than either result alone: **"distribution-aware joint `(R,A)` search beats sequential
per-axis selection" is real but conditional, not general** — it wins specifically when the
default/cheapest choice is *objective-blind and empirically wrong* for the task at hand (kernel
1's `Identity` always winning a cost tie-break regardless of downstream quantization behavior),
and has nothing to offer when the default happens to already be good (kernel 2's `Componentwise`
at low/medium noise). This yields a concrete, actionable, falsifiable heuristic that itself falls
out of the replication attempt: **before investing in a full joint search for a new kernel, run a
cheap single-axis ablation (exactly this addendum's "chart-only comparison" table) to check
whether the default representation is already near-optimal for the target data regime — only
invest in joint search where it is not.**

**Updated novelty-ladder status.** With a 1-of-2-kernel replication rate (3/3 conditions on
kernel 1; 1/3 on kernel 2, criterion not met), this report now actively **recommends against**
generalizing the kernel-1 result into any broader claim, tool, or crate — reinforcing, with direct
evidence rather than only caution, [CANR §12]/[ANEE §13]'s original instruction not to broaden
scope before replication. The representation-graph-plus-distribution-aware-cache candidate's
confirmed status is narrowed accordingly: **level 6, for kernel 1's specific task only; not
demonstrated to generalize, and one kernel's worth of evidence now weighs against assuming it
does by default.** No further kernel replications or scope expansion are recommended as a next
step from this report; the boundary condition identified above (joint search only pays when the
cheap default is objective-blind) is the more valuable and more narrowly defensible takeaway than
either individual kernel's win/loss.

## Addendum 3 (2026-07-17, same day): two self-falsification attempts on our own claims — results

After Addenda 1–2, two of this report's *own* products remained untested: the Addendum-2
boundary heuristic ("run a cheap single-axis ablation first; only invest in joint search where
the cheap default is objective-blind and empirically wrong") had never been tested
*prospectively*, and the §4 "representation graph" had been formalized (Proposition ANEE-2,
validated by Z3) but its actual *graph structure* — multi-hop paths — had never been exercised:
both Phase C kernels searched flat, single-hop dictionaries. This addendum attacks both. As
before, all prior text is left unaltered; both protocols and criteria were written into the
committed benchmark sources before any run.

### Avenue 1 — dose-response test of the Addendum-2 heuristic

**Protocol** (`examples/anee_phase_c_dose_response.rs`): kernel 1's quantizer level count —
fixed at 64 in all published results — becomes an environmental dose knob,
`L ∈ {8, 16, 64, 256, 1024}`, across kernel 1's three workload families = 15 cells. Per cell,
*first* the cheap prospective predictor (dev data only: the R-axis-only ablation gap at fixed
Neumaier accumulation; predict "joint pays" iff gap ≥ 20%), *then* the outcome (kernel 1's
exact sequential-vs-joint protocol, 3 fresh held-out seeds, 20% bar). **P1 (decisive,
pre-registered): predictor/outcome agreement in ≥ 12 of 15 cells, else the heuristic is
falsified as a decision rule.** Three secondary, descriptive predictions (P2–P4) were also
pre-registered.

**Result: P1 MET — 13/15 agreement. The heuristic survives as a decision rule, with a sharp,
newly-learned caveat.** Both disagreements are *false positives at the noise floor* (benign,
L = 256 and L = 1024): the dev-only ablation showed large relative gaps (79.7%, 37.4%) on
absolute errors already down at 10⁻⁶–10⁻⁵, joint search chased them, and its picks *lost* on
held-out data (−87.8%, −34.2%). The predictor produced **zero false negatives** — it never
missed a real win in any of the 13 cells where joint search genuinely paid. Practical
refinement that follows directly: **the ablation gap must be guarded by an absolute-error
floor** — ignore relative gaps when the default's absolute error already meets the target;
relative improvement over an already-negligible error is noise, and chasing it is actively
harmful (the joint picks at those two cells were *worse* than the default).

**Secondary predictions: 1 held, 2 failed — both failures are author-model errors, reported as
such.** P3 (wide-range reduction ≥ 20% at every L) held: 94.5–99.8% across the full range,
confirming that for heavy-tailed data uniform quantization stays wrong at every tested level
count, exactly as high-resolution quantization theory predicts (the uniform-vs-companded gap is
roughly level-independent). P2 (stagnation-prone reduction decreasing in L) **failed** — the
reduction stays at 79.6–99.6% across all L with no downward trend. The author's model error:
assuming quantization error stops dominating for bimodal 6-decade data somewhere below L = 1024;
in fact the small-value mode is crushed by a uniform quantizer at *every* tested L, the default
stays empirically wrong throughout, and — consistently — the *primary* heuristic classified all
five of those cells correctly (PAYS/PAYS); only the dose-response *shape* prediction was wrong.
P4 (benign reduction < 20% for L ≥ 64) **failed at L = 64 by construction — a pre-registration
authoring error**: kernel 1 had *already published* 23.1% at exactly that cell (Addendum 1),
and P4 as written contradicted a known data point. At the two genuinely new cells (L = 256,
1024) the prediction held (reductions negative). Recorded as a specification mistake by the
author, not as evidence about the heuristic.

### Avenue 2 — is the "representation graph" really a graph?

**Protocol** (`examples/anee_phase_c_two_hop.rs`): `RepresentationChoice` gained a
`Composed(a, b)` variant — encode `x ↦ b(a(x))`, condition number computed by the **exact
product law of Proposition ANEE-2** (the elasticity chain rule, validated at Decimal prec-60 by
Z3, and now additionally unit-tested in Rust: `composed_kappa_is_the_exact_product_of_hops`
verifies κ(power(½)∘log) ≡ |ln x| to 10⁻¹²) — its first executable use. All 20 ordered two-hop
compositions of the dictionary compete against the 5 single hops on 6 cells (3 families ×
L ∈ {8, 64}), same selection + 3-fresh-seed protocol. **Pre-registered criterion (existential):
the graph structure earns its name iff ≥ 1 cell shows a two-hop win (pairs winner ≥ 20% better
than singles winner); zero wins ⇒ the "graph" is a flat dictionary wearing a graph's name.
Author's declared prior, stated in the committed source before running: zero wins expected.**

**Result: criterion MET — 2 of 6 cells are two-hop wins — and the author's declared prior was
therefore falsified.** Wide-range L = 8: `anscombe∘anscombe` beats the best single
(`power(½)`) by 87.8%; wide-range L = 64: `power(½)∘power(½)` beats it by 42.6%. The composed
encode gate worked as designed throughout (16–17 of 20 compositions admitted per cell; e.g.
log-then-Anscombe correctly rejected on data crossing 1, where the intermediate goes negative).

**Post-hoc diagnostic (explicitly labeled as such in the committed source; added after the
pre-registered run) — the wins are dictionary densification, not path structure.** Both winning
compositions are, mathematically, just *stronger companders the single dictionary lacked*:
`power(½)∘power(½)` **is** `power(¼)`, and `anscombe∘anscombe` behaves like a fourth-root-type
curve. Re-running the *flat singles* search with `power(¼)` and `power(⅛)` added: at L = 64 the
densified flat dictionary reproduces the two-hop winner's error **bit-for-bit** (7.1251×10⁻⁴ —
confirming the `power(¼)` identity computationally), and at L = 8 it *beats* the two-hop winner
(8.67×10⁻³ vs. 9.68×10⁻³). **Refined conclusion, both layers reported: the pre-registered
existential claim survived (composition demonstrably reaches better plans than the base
dictionary), but the mechanism is that composition acts as a *generator of new dictionary
members* — a flat dictionary enriched with the same generated curves does as well or better.
Path search over a representation graph has, on all evidence in this program, no demonstrated
value beyond that generative role.**

### What Addendum 3 changes

1. **The Addendum-2 heuristic is promoted from "derived observation" to "prospectively
   validated decision rule (13/15), with a mandatory absolute-error-floor guard"** — its two
   observed failures are both of one type (chasing relative gaps at the noise floor), and the
   guard eliminates exactly that type. Zero false negatives observed.
2. **§4's "representation graph" framing is deflated, by our own experiment, to "compositional
   closure as dictionary generator."** The recommendation for any future work changes
   accordingly: enrich the flat dictionary with composition-generated members (e.g. a λ-grid of
   powers, generated companders with certificates via the exact κ product law — which is
   precisely what the `Composed` variant provides), and do **not** build shortest-path
   machinery, for which no evidence of need exists after direct testing.
3. **Three author predictions failed this round** (P2's regime model, P4's specification against
   known data, avenue 2's zero-win prior) and are recorded as such. The program's protocol —
   criteria locked in committed sources before running — is what made these failures visible
   and cheap rather than silent.
4. No scope expansion: no new crates, no planner, no further kernels. The operational takeaway
   for SciRust is one selector policy (guarded ablation-first) and one dictionary-enrichment
   mechanism (certified composition), both already implemented in `representation_graph.rs`.

## Appendix A — Experiment index

`docs/research/anee_experiments/anee_experiments.py` (pure stdlib Python 3, deterministic, fixed
seeds, Decimal prec-60 where exactness is checked): **Z1** combinatorial search-space size, using
SciRust's own current dictionary sizes where they exist; **Z2** independent vs. coordinate-descent
vs. exhaustive joint search under a constructed, genuinely non-separable two-basin objective;
**Z3** exact multiplicative composition of `κ_rt` (elasticity chain rule) validated to Decimal
prec-60 across seven points spanning 42 orders of magnitude; **Z4** empirical confirmation that
determinism composes as a lattice meet (weakest link), not an average, across 20 random
permutations per stage. All numbers quoted in §§4, 7, 8 came from this committed script's output
on 2026-07-17.

**Phase C prototype, kernel 1** (see Addendum 1): `scirust-core/src/representation_graph.rs`
(library) and `scirust-core/examples/anee_phase_c_prototype.rs` (benchmark, run via
`cargo run -p scirust-core --example anee_phase_c_prototype --release`), Rust, deterministic
(fixed seeds), part of the `scirust-core` crate's normal `cargo test`/`cargo clippy` surface. All
numbers quoted in Addendum 1 came from this committed code's output on 2026-07-17.

**Phase C prototype, kernel 2 (replication attempt)** (see Addendum 2):
`scirust-core/src/representation_graph_quaternion.rs` (library) and
`scirust-core/examples/anee_phase_c_kernel2_quaternion.rs` (benchmark), Rust, deterministic
(fixed seeds), gated behind the `portable-simd` feature (`scirust_simd::geometry::quaternion` is
itself feature-gated) — run via `cargo test -p scirust-core --lib --features portable-simd` and
`cargo run -p scirust-core --features portable-simd --example anee_phase_c_kernel2_quaternion
--release`. All numbers quoted in Addendum 2 came from this committed code's output on
2026-07-17.

**Addendum 3 self-falsification benchmarks** (see Addendum 3):
`scirust-core/examples/anee_phase_c_dose_response.rs` (avenue 1 — prospective dose-response
test of the Addendum-2 heuristic over quantizer levels L ∈ {8, 16, 64, 256, 1024}) and
`scirust-core/examples/anee_phase_c_two_hop.rs` (avenue 2 — two-hop compositions vs. the flat
single-hop dictionary, including the labeled post-hoc densified-singles diagnostic), both
running against `representation_graph.rs`'s `*_with_levels` entry points and the new
`RepresentationChoice::Composed` variant (exact κ product law per Proposition ANEE-2, unit
test `composed_kappa_is_the_exact_product_of_hops`). Rust, deterministic (fixed seeds), default
features — `cargo run -p scirust-core --example anee_phase_c_dose_response --release` and
`cargo run -p scirust-core --example anee_phase_c_two_hop --release`. All numbers quoted in
Addendum 3 came from this committed code's output on 2026-07-17.

## Appendix B — Verified sources (this phase, not already in [TSA]/[ATRA]/[CANR] Appendix B)

**Foundational theory.** Rice, "The Algorithm Selection Problem," *Advances in Computers*
15:65–118 (1976). Weerawarana, Houstis, Rice, Joshi, Houstis, "PYTHIA: A Knowledge-Based System
to Select Scientific Algorithms," *ACM TOMS* 22(4):447–468 (1996). Thornton, Hutter, Hoos,
Leyton-Brown, "Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification
Algorithms" (CASH), KDD 2013. Kotthoff, "Algorithm Selection for Combinatorial Search Problems:
A Survey," *AI Magazine* 35(3), 2014.

**Compiler IRs and scheduling.** Lattner, Adve, "LLVM: A Compilation Framework for Lifelong
Program Analysis & Transformation," CGO 2004. Lattner et al., "MLIR: Scaling Compiler
Infrastructure for Domain Specific Computation," CGO 2021 (arXiv:2002.11054, preprint titled
"...for the End of Moore's Law"); Zinenko et al., MLIR Transform dialect RFC, March 2022; Lücke
et al., "The MLIR Transform Dialect," CGO 2025 (arXiv:2409.03864). Ragan-Kelley, Barnes, Adams,
Paris, Durand, Amarasinghe, "Halide," PLDI 2013; CACM retrospective, Jan 2018; Mullapudi, Adams,
Sharlet, Ragan-Kelley, Fatahalian, TOG/SIGGRAPH 2016; Adams et al., "Learning to Optimize Halide
with Tree Search and Random Programs," TOG/SIGGRAPH 2019. Chen et al., "TVM," OSDI 2018; Roesch
et al., "Relay: A New IR for Machine Learning Frameworks," MAPL 2019 (arXiv:1810.00952). Tillet,
Kung, Cox, "Triton," MAPL 2019. Vasilache et al., "Tensor Comprehensions," arXiv:1802.04730
(2018; archived, no peer-reviewed venue). Abadi et al., "TensorFlow," OSDI 2016
(arXiv:1605.08695); OpenXLA/StableHLO project documentation (openxla.org, github.com/openxla).
CUTLASS `CITATION.cff` (software, no paper); NVIDIA CUTLASS 3.x design docs (CuTe, Profiler).

**Autotuning and self-tuning systems.** Ansel, Kamil, Veeramachaneni, Ragan-Kelley, Bosboom,
O'Reilly, Amarasinghe, "OpenTuner," PACT 2014. Chen, Zheng, Yan, Jiang, Moreau, Ceze, Guestrin,
Krishnamurthy, "Learning to Optimize Tensor Programs" (AutoTVM), NeurIPS 2018
(arXiv:1805.08166). Zheng et al., "Ansor," OSDI 2020 (arXiv:2006.06762). Whaley, Dongarra, SC'98;
Whaley, Petitet, Dongarra, "Automated Empirical Optimization of Software and the ATLAS Project,"
*Parallel Computing* 27(1–2):3–35 (2001). Frigo, Johnson, "FFTW," ICASSP 1998. Püschel et al.,
"SPIRAL: Code Generation for DSP Transforms," *Proc. IEEE* 93(2):232–275 (2005). Ansel, Chan,
Wong, Olszewski, Zhao, Edelman, Amarasinghe, "PetaBricks," PLDI 2009.

**Numerical libraries and performance portability.** UXL Foundation oneDNN documentation; Li et
al., "oneDNN Graph Compiler," CGO 2024. Intel oneMKL developer reference. OpenBLAS project
documentation (GotoBLAS2 lineage: Goto, van de Geijn, *ACM TOMS* 34(3), 2008). Eigen project
documentation. Edwards, Trott, Sunderland, "Kokkos," *J. Parallel Distrib. Comput.* 74(12):
3202–3216 (2014); Trott et al., "Kokkos 3," *IEEE TPDS* 33(4):805–817 (2022). Beckingsale et al.,
"RAJA," P3HPC 2019. Khronos SYCL 2020 specification; Intel DPC++/oneAPI documentation.

**Hardware-aware quantization and dispatch.** Wang, Liu, Lin, Lin, Han, "HAQ," CVPR 2019. Dong,
Yao, Gholami, Mahoney, Keutzer, "HAWQ," ICCV 2019; HAWQ-V2, NeurIPS 2020. Google Highway
design-philosophy documentation; glibc `sysdeps/x86_64/multiarch` source (ifunc dispatch).

**Precision tuning, verified numerics, and contracts.** FPBench/FPCore/Metadata specifications
(fpbench.org). Moscato, Titolo, Dutle, Muñoz, PRECiSA, SAFECOMP 2017 (+ PRECiSA 4.0, 2024).
Abbasi, Darulova, "Modular Optimization-Based Roundoff Error Analysis of Floating-Point
Programs," SAS 2023. Zhang, Aiken, "Automatic Verification of Floating-Point Accumulation
Networks," CAV 2025. Heldens, van Werkhoven, "Accuracy-Aware Mixed-Precision GPU Auto-Tuning,"
*IEEE TPDS* (2026). El Arar, Filip, Mary, Riccietti, arXiv:2503.15568 (2025). Martel, dependent
type system for roundoff error, ESOP 2002. Kellison, Hsu, "Numerical Fuzz," PLDI 2024.

**Composition algebra.** Tate, Stepp, Tatlock, Lerner, "Equality Saturation," POPL 2009. Willsey,
Nandi, Wang, Flatt, Tatlock, Panchekha, "egg," *PACMPL* 5(POPL), 2021. VanHattum, Nigam, Lee,
Bornholt, Sampson, "Diospyros," ASPLOS 2021. Schkufza, Sharma, Aiken, "STOKE," ASPLOS 2013.
Hagedorn, Lenfers, Koehler, Qin, Gorlatch, Steuwer, "Elevate," *PACMPL* 4(ICFP), 2020. Koehler,
Goens, Bhat, Grosser, Trinder, Steuwer, "Guided Equality Saturation," POPL 2024. Qin, O'Connor,
van Glabbeek, Hoefner, Kammar, Steuwer, "Shoggoth," POPL 2024. Ikarashi, Bernstein, Reinking,
Genc, Ragan-Kelley, "Exo," PLDI 2022.

## Appendix C — Cumulative search log (phase 4 additions)

| Claimed gap / verification target | Closest prior work found |
|---|---|
| Is `P=(R,O,A,T,Q,M,H)` a new optimization formulation? | Rice's Algorithm Selection Problem (1976); PYTHIA (1996) already applied it to scientific algorithm selection; CASH (2013)/SMAC own the parametrized-configuration generalization |
| Representation graph as path search over format-nodes with weighted conversion edges | Not found. HAQ/HAWQ and adjacent ML-quantization, DB-encoding, and codec literature all solve the structurally different node-labeling-on-a-fixed-graph problem instead |
| Execution plan as a portable, inspectable artifact | MLIR Transform dialect (2022); Halide's original schedule/algorithm split (2013); FFTW's wisdom (1998) |
| (kernel, distribution, hardware) → plan cache with confidence/history | AutoTVM's TopHub (2018) owns `(target, workload, config, cost)`; no cache found anywhere keys on data *distribution* specifically |
| Cross-hardware plan portability | Confirmed unsolved/open as of TCL (arXiv:2604.12891, April 2026); FFTW/SPIRAL/PetaBricks all quantitatively disprove naive portability |
| Certificate/error-bound engine for general (non-DSP-linear) numerical kernels | SPIRAL (2005) owns it for linear DSP transforms specifically; no general-purpose equivalent found |
| Certificate composition across independently-verified pipelined kernels | Confirmed open; Abbasi–Darulova (SAS 2023) and PRECiSA 4.0 (2024) are intra-tool only; a June 2026 PhD thesis is titled around this exact open problem |
| "Composition instead of optimization" as a distinct optimization object | Equality saturation (Tate et al., POPL 2009) owns it; egg (POPL 2021) ships it generally; Diospyros (ASPLOS 2021) already applies it to two of ANEE's own axes jointly |
| Numerical contract type system spanning all listed properties | Not found as an integration; each property has its own separate, classical treatment (chain rule / lattice meet / logical AND / Data Processing Inequality) |
| Hardware dispatch that changes the algorithm, not just vector width | Mostly absent (FMV/ISPC/Highway re-vectorize); SPIRAL's Formula Generator and glibc's `ifunc` `memcpy` (hardware `rep movsb` path) are confirmed counterexamples |
