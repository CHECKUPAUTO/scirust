# FLAT-ATTENTION + ElasticAutoTuner — Master Engineering Roadmap

Status baseline: SciRust `master` at `be8203179b566727a557ef93321b748c5a47c177` (2026-08-10), after merged PR #1140. FLAT-ATTENTION `main` at `c971359fe0e03c97382d536745cbdd28aa4924ae`, with PR #23 open for portable ALiBi WGPU qualification. SciRust PR #1141 is open for the opt-in SciAgent FLAT M15 decode boundary.

This document is the authoritative cross-repository execution plan for turning FLAT-ATTENTION into SciRust's complete high-performance attention engine and ElasticAutoTuner into SciRust's universal deterministic kernel autotuner.

The plan starts from the code that already exists. It is not a green-field design.

---

## 0. Absolute completion rule

FLAT-ATTENTION is **not complete** when a fast forward kernel exists. It is complete only when every mandatory capability, integration, correctness gate, numerical gate, device gate, benchmark gate, fallback path, cache contract, backward path, tuning path, documentation contract, and production-selection gate in this document is closed.

A milestone is not complete because code exists. It is complete only when its acceptance evidence exists and is reproducible.

No performance claim is valid without real-device evidence. No optimized path may become default because it looks structurally faster. No path may silently fall back to CPU when a GPU execution mode was requested.

The project is considered **100% complete** only after the final qualification matrix in §19 is fully green.

---

## 1. Non-negotiable engineering constraints

### 1.1 Sovereignty

- Rust is the host implementation language.
- No project-authored C or C++ implementation layer.
- No project-authored C ABI bridge.
- No mandatory CUDA C++, `nvcc`, WMMA/WGMMA API, CUTLASS, cuDNN, or equivalent vendor SDK in the core architecture.
- Portable GPU execution is based on open shader/IR paths first.
- Hardware-specific acceleration is capability-gated and must preserve a qualified portable path.
- Dependencies must be pinned/controlled in SciRust where cross-repository reproducibility requires it.

### 1.2 Correctness hierarchy

1. mathematical definition;
2. deterministic scalar Rust oracle;
3. portable fused device implementation;
4. optimized portable device implementation;
5. generated/specialized implementation;
6. backend/device-specific open acceleration path.

Every lower level must be qualified against the level above it before promotion.

### 1.3 Determinism

Deterministic mode must explicitly define:

- accumulation type;
- reduction topology;
- exponentiation/normalization policy;
- plan-selection policy;
- autotuner cache/version semantics;
- same-device repeatability guarantees;
- what is and is not promised across different devices/backends.

### 1.4 CI/merge discipline

Every implementation slice uses a dedicated branch/PR. Required CI must be green on the exact final head SHA before merge. Mergeability alone is not acceptance evidence.

---

## 2. Verified work already completed

The following work is already present and must be preserved as the foundation.

### 2.1 FLAT-ATTENTION repository

Completed/merged capability chain includes:

- M1 deterministic scalar online-softmax oracle and first fused WGSL forward;
- M2 real WGPU executor with resident buffers and no hidden CPU fallback;
- M3 device parity matrix;
- M4 Q4 multi-query-row K/V tile reuse;
- M5 subgroup-assisted reductions with explicit capability policy;
- M6 vec4 memory path for common dimensions;
- M7 opt-in double-buffered K/V staging candidate;
- M8 packed IEEE binary16 IO with FP32 accumulation/LSE;
- M9 explicit numerical policy and deterministic WGPU mode;
- M10 native MHA/GQA/MQA without K/V replication;
- R1 fused RoPE + native grouped attention;
- R2 caller-owned WGPU encoder and projection-layout zero-copy boundary;
- M11 rectangular Q/KV lengths, GQA/MQA decode geometry and cross-attention geometry;
- M12 variable-length padded batch scalar and WGPU contracts;
- M13 additive attention-bias scalar oracle merged;
- M15 pre-rotated resident-K decode mode merged;
- reproducible decode benchmark protocol exists;
- current open FLAT PR #23 extends M13 ALiBi semantics to WGPU.

### 2.2 SciRust repository

Already available and relevant:

- deterministic/scalar attention components;
- transformer inference stack;
- fixed-point deterministic attention, RoPE, RMSNorm, KV cache and Transformer block;
- batched GEMM paths;
- attention gradient infrastructure;
- quantized KV-cache work;
- resident WGPU infrastructure;
- pinned FLAT source allow-list;
- FLAT M11 resident bridge merged;
- pre-rotated FLAT K-cache bridge merged in PR #1140;
- current open SciRust PR #1141 establishes the opt-in SciAgent FLAT M15 decode feature boundary.

These are baseline assets. They are not to be reimplemented gratuitously.

---

## 3. Target architecture

```text
SciAgent / SciRust Transformer
        |
        v
AttentionRequest
        |
        v
FLAT Planner -------------------------+
        |                              |
        v                              v
ElasticAutoTuner                deterministic policy
        |                              |
        +------------+-----------------+
                     v
              ExecutionPlan
                     |
        +------------+-------------+
        |            |             |
        v            v             v
     Prefill       Decode       Backward
        |            |             |
        +------+-----+------+------+ 
               |            |
               v            v
        FLAT Kernel IR / generated variants
               |
        +------+------+----------------+
        |             |                |
        v             v                v
   portable WGSL   subgroup path   open matrix path
        |
        v
   SciRust-owned WGPU device/queue/buffers
```

FLAT owns attention algorithms and execution plans. ElasticAutoTuner owns search, measurement, evidence, plan selection and plan persistence. SciRust owns the higher-level tensor/model runtime and device graph.

---

# PROGRAM A — Finish the currently active integration chain

## A1 — Complete FLAT M13 WGPU bias/ALiBi qualification

Current anchor: FLAT PR #23.

Required closure:

- dense additive bias WGPU path;
- ALiBi WGPU path;
- raw-K and pre-rotated-K modes;
- causal/non-causal combinations where semantically valid;
- GQA/MQA cardinality preservation;
- variable-length compatibility;
- explicit finite/cardinality/overflow validation;
- no new score/probability storage matrix;
- Naga validation;
- mandatory WGPU parity;
- CI green on exact head;
- merge to FLAT `main`.

## A2 — Complete SciAgent M15 routing

Current anchor: SciRust PR #1141.

Required follow-up slices:

1. merge feature boundary only after full CI;
2. route actual `ResidentModel::decode_step` through FLAT behind opt-in feature;
3. preserve current resident attention as fallback/oracle;
4. prove parity token-by-token for growing cache lengths;
5. verify pre-rotated K is never rotated twice;
6. prove no host Q/K/V round-trip in resident mode;
7. prove no per-token K/V cache copy;
8. benchmark paired same-adapter old-vs-FLAT decode;
9. only then consider selection-policy promotion.

## A3 — SciRust pinned-revision discipline

Every FLAT integration promotion must:

- pin an immutable merged FLAT SHA;
- refresh canonical lockfile through Cargo;
- keep `cargo-deny` source constraints precise;
- qualify MSRV/workspace/all-features compatibility;
- never float against FLAT `main` in production integration.

---

# PROGRAM B — Complete inference attention semantics

## B1 — Resident KV-cache contract completion

Implement/qualify:

- append-only resident K/V;
- logical length and capacity;
- reset/reuse semantics;
- GQA/MQA physical KV-head storage;
- raw-K and pre-rotated-K representation metadata;
- page/block abstraction boundary;
- cache format/version ID;
- overflow and stale-data rejection;
- zero-copy FLAT consumption.

## B2 — Dedicated decode kernel family

Do not rely permanently on a generic rectangular kernel.

Create specialized variants for:

- `q_len=1`;
- short speculative windows (`q_len=2..K`);
- GQA/MQA reuse across query-head groups;
- short/medium/long KV lengths;
- split-KV reduction when beneficial;
- causal absolute-position semantics;
- pre-rotated K caches;
- optional ALiBi/bias;
- packed-f16 storage path;
- deterministic path.

Acceptance requires real-device paired evidence against generic M11 and current SciRust decode.

## B3 — Prefill kernel family

Create separate prefill policies for:

- short sequences;
- medium sequences;
- long contexts;
- D32/D64/D80/D96/D128 minimum;
- MHA/GQA/MQA;
- causal/non-causal;
- local/sliding attention;
- variable-length batches;
- packed-f16 and f32;
- RoPE-fused and pre-rotated modes where meaningful.

## B4 — Sliding/local attention

Add first-class local attention rather than treating it as arbitrary masks:

- left window;
- optional right window for non-causal models;
- causal sliding window;
- exact boundary semantics;
- oracle;
- dedicated tile pruning so out-of-window K/V tiles are never staged;
- GQA/MQA compatibility;
- variable lengths;
- prefill/decode variants.

## B5 — Cross attention

Finish rectangular non-causal support for encoder/decoder workloads:

- independent RoPE/no-RoPE policy;
- independent Q/KV sequence origins;
- bias/mask combination matrix;
- variable lengths;
- benchmark as its own workload class.

## B6 — Packed/ragged variable-length execution

M12 padded variable-length execution is not the end state.

Research and qualify:

- packed Q representation;
- packed KV representation;
- sequence offset tables;
- block ownership;
- output scattering without host repack;
- padded-vs-packed cost crossover;
- deterministic metadata ordering;
- fail-closed shape validation.

Promote packed/ragged only where benchmarks demonstrate benefit.

## B7 — Paged KV-cache

Implement project-owned page-table semantics:

- fixed page size policy;
- logical token -> physical page mapping;
- page allocation/free/reuse;
- fragmentation telemetry;
- batch request isolation;
- page-boundary decode correctness;
- sliding-window page retirement;
- chunked prefill integration;
- no vendor library dependency.

---

# PROGRAM C — Training completion

## C1 — Scalar backward oracle

Implement deterministic Rust dQ/dK/dV for all mandatory semantics:

- MHA/GQA/MQA;
- causal/non-causal;
- rectangular Q/KV;
- variable lengths;
- local/sliding attention;
- additive bias/ALiBi where gradients are defined;
- RoPE interaction;
- FP32 oracle first.

Use finite differences and independent small-shape references.

## C2 — Recompute backward GPU kernel

Requirements:

- consume Q/K/V, O and LSE;
- recompute scores/probabilities;
- never store full `N x N` P;
- dQ/dK/dV;
- explicit race-free accumulation strategy;
- no hidden host reduction;
- deterministic fallback topology;
- resident output buffers.

## C3 — Backward optimization families

Specialize independently for:

- head dimensions;
- causal/non-causal;
- GQA/MQA;
- prefill shapes;
- mixed precision;
- subgroup/vectorized paths;
- recomputation tile sizes;
- dK/dV ownership/splitting.

## C4 — Training integration in SciRust

Integrate FLAT backward into SciRust autograd behind feature/selection policy, retaining existing attention backward as oracle/fallback until parity and benchmark gates pass.

---

# PROGRAM D — FLAT Kernel IR and generated kernel architecture

## D1 — Define FLAT Kernel IR

IR must represent at minimum:

- logical tensors/layouts;
- tile dimensions;
- vector width;
- staged loads/stores;
- reductions;
- online-softmax state transition;
- masks/bias;
- RoPE transforms;
- barriers;
- subgroup operations;
- precision conversions;
- matrix fragments;
- accumulator ownership;
- capability requirements;
- workgroup memory budget;
- deterministic serialization/hash.

## D2 — IR verifier

Reject statically where possible:

- illegal barriers;
- invalid tile dimensions;
- workgroup storage overflow;
- impossible vector alignment;
- unsupported precision/capabilities;
- inconsistent GQA mappings;
- invalid page/block assumptions;
- race-prone ownership contracts.

## D3 — Deterministic WGSL emitter

Requirements:

- byte-stable output for identical IR/config;
- generated-source hash;
- Naga validation;
- source cache;
- generated-vs-handwritten parity;
- debug emission mode.

## D4 — Replace handwritten specialization explosion

Once generated kernels equal or beat handwritten versions in correctness and performance, migrate specializations progressively behind exact regression gates. Do not delete qualified handwritten baselines until generated replacements are proven.

---

# PROGRAM E — Open high-performance matrix paths

## E1 — Capability inventory

Build a factual device capability layer covering:

- workgroup limits;
- storage limits;
- subgroup support/range;
- shader f16 support;
- cooperative/subgroup matrix availability where exposed through open paths;
- backend type;
- adapter/driver identity;
- timestamp-query support;
- dispatch/binding limits.

## E2 — Open cooperative matrix research

Investigate only paths consistent with sovereignty policy:

- Vulkan/SPIR-V cooperative matrix support;
- WGSL/WebGPU subgroup-matrix support as available;
- project-owned IR lowering;
- no mandatory vendor SDK.

No path is promoted without real-device proof.

## E3 — Fragment scheduler

Design architecture-independent fragment scheduling for D64/D128 first:

- Q/K fragment mapping;
- score fragment accumulation;
- transition from fragment results into row max/sum;
- P/V accumulation mapping;
- register pressure model;
- occupancy/resource estimator.

---

# PROGRAM F — ElasticAutoTuner foundation

## F1 — Crate and public contract

Create reusable `elastic-autotuner` component, not FLAT-private logic.

Core public concepts:

- `ElasticAutoTuner`;
- `ElasticConfig`;
- `ElasticObjective`;
- `ElasticHardwareProfile`;
- `ElasticProblemClass`;
- `ElasticSearchSpace`;
- `ElasticCandidate`;
- `ElasticExecutionPlan`;
- `ElasticMeasurement`;
- `ElasticEvidence`;
- `ElasticPlanCache`;
- `ElasticCostModel`.

FLAT becomes the first major client.

## F2 — Hardware fingerprint

Fingerprint must include stable execution-relevant facts, not marketing names alone:

- backend;
- adapter/vendor/device identifiers where exposed;
- driver/version where exposed;
- limits/capabilities;
- subgroup range;
- precision support;
- timestamp-query availability;
- FLAT/IR schema version.

Serialize deterministically.

## F3 — Problem classification

Do not key every exact sequence length.

Classify workloads by meaningful domains:

- operation: prefill/decode/backward/cross;
- dtype/storage;
- head dimension;
- q-heads / kv-heads ratio;
- q-length class;
- kv-length class;
- causal/local/global;
- bias mode;
- KV layout/page mode;
- batch/variable-length class.

Learn validity regions rather than exploding one cache entry per exact geometry.

## F4 — Search parameters

Initial search dimensions:

- Q tile rows;
- KV tile rows;
- workgroup size;
- vector width;
- subgroup policy;
- double/triple buffering;
- memory layout/swizzle;
- load ordering;
- reduction topology;
- GQA head-group reuse;
- split-KV factor;
- precision/storage choice;
- kernel family;
- pipeline specialization;
- generated IR scheduling options.

## F5 — Static constraint solver

Before benchmarking, reject candidates exceeding:

- workgroup size;
- workgroup storage;
- binding limits;
- alignment constraints;
- capability requirements;
- known invalid subgroup assumptions;
- estimated register/resource bounds where available;
- semantic incompatibilities.

## F6 — Analytical cost model

Estimate at minimum:

- logical bytes loaded/stored;
- arithmetic work;
- reuse factor;
- synchronization count;
- dispatch count;
- expected occupancy pressure;
- temporary memory;
- split/reduction overhead.

The cost model ranks candidates; it does not replace measurements.

## F7 — Deterministic candidate generation

Same hardware fingerprint + same problem class + same tuner version + same policy must produce the same ordered candidate set.

---

# PROGRAM G — ElasticAutoTuner measurement and search

## G1 — Correctness-before-timing gate

Every candidate must pass numerical parity before its timing is accepted. Incorrect or unstable candidates are permanently rejected for the applicable fingerprint/schema.

## G2 — Measurement protocol

Define:

- warmups;
- measured iterations;
- synchronization boundary;
- resident vs transfer-inclusive modes;
- median;
- p95/p99 where useful;
- variance/MAD;
- outlier policy;
- device timestamp vs host wall-clock source;
- thermal/load metadata where obtainable.

## G3 — Search strategy progression

Stage 1: exhaustive small discrete spaces.

Stage 2: constrained top-K from analytical model.

Stage 3: deterministic successive-halving/racing.

Stage 4: learned cost model trained only from retained evidence.

Stage 5: optional evolutionary/Bayesian-style search only if deterministic replay/versioning can be preserved.

## G4 — Multi-objective tuning

Support explicit objectives:

- minimum latency;
- maximum throughput;
- minimum temporary memory;
- balanced latency/memory;
- deterministic-only;
- energy-aware only if reliable measurements exist.

Never hide objective changes.

## G5 — Tuner operating modes

- `Cold`: safe qualified heuristic plan immediately;
- `Learn`: controlled exploration;
- `Locked`: no exploration, use only validated persisted plans;
- `Audit`: replay measurements/validation without changing production selection.

Production inference should be able to run fully locked.

## G6 — Persistent evidence cache

Every plan record contains:

- schema/version;
- hardware fingerprint;
- problem-class bounds;
- kernel/IR hash;
- parameters;
- correctness evidence hash;
- measurement statistics;
- sample count;
- objective;
- selected/not-selected state;
- timestamp/provenance;
- invalidation dependencies.

Corruption/version mismatch must fail safely.

---

# PROGRAM H — FLAT + ElasticAutoTuner integration

## H1 — FLAT planner

Convert an attention request into:

1. semantic validation;
2. problem class;
3. hardware capability query;
4. eligible kernel families;
5. Elastic plan query;
6. deterministic fallback plan if no tuned evidence exists;
7. execution.

## H2 — Plan boundaries

Plans should cover validity regions such as KV-length ranges, not single exact shapes where avoidable. Boundary transitions require regression tests.

## H3 — Online safe learning

Never benchmark destructive/random candidates in the middle of latency-sensitive production work unless explicitly enabled. Learning must be controlled and isolatable.

## H4 — SciRust integration

Expose tuning policy from SciRust without leaking low-level FLAT internals into model code. SciAgent requests attention; planner/tuner decide execution.

---

# PROGRAM I — Benchmarking and observability

## I1 — Canonical benchmark matrix

At minimum cover:

- B: 1, 2, 4, 8 and representative production batches;
- Q lengths: 1, 2, 4, 8, 16, 64, 128, 512, 1K, 4K, 16K and larger where hardware allows;
- KV lengths: 16 through long-context limits;
- D: 32, 64, 80, 96, 128;
- MHA/GQA/MQA;
- causal/non-causal/local;
- f32/packed-f16 and future qualified formats;
- raw-K/pre-rotated-K;
- padded/packed variable batches;
- contiguous/paged KV;
- forward/backward.

## I2 — Metrics

Record:

- median/p95/p99 latency;
- tokens/s;
- batch throughput;
- effective logical bandwidth;
- estimated arithmetic intensity;
- allocations/temporary bytes;
- dispatch count;
- read/write bytes when measurable;
- compile/pipeline creation time separately;
- tuner search cost;
- plan-cache hit rate;
- numerical error statistics.

## I3 — Baselines

Compare, where technically available and policy-compatible for measurement:

- deterministic scalar oracle;
- existing SciRust composed attention;
- generic FLAT portable baseline;
- each optimized FLAT family;
- historical qualified FLAT variants.

External competitors may be measured as external reference points, but FLAT must not depend on them.

## I4 — Regression thresholds

Define per-workload allowed regression envelopes. CI/bench infrastructure should flag statistically meaningful regressions rather than single noisy samples.

---

# PROGRAM J — Numerical robustness campaign

## J1 — Adversarial corpus

Include:

- equal scores;
- near ties;
- large positive/negative finite scores;
- cancellation-heavy Q/K;
- all-zero rows;
- very long rows;
- causal boundaries;
- local-window boundaries;
- padded poisoned regions;
- page boundaries;
- GQA group boundaries;
- unusual D values;
- scale extremes still within contract.

## J2 — Non-finite policy

Define and test treatment/rejection for NaN/Inf inputs and invalid scales before device dispatch where possible.

## J3 — Repeatability

For deterministic modes, repeat exact calls and compare bit patterns under the documented same-device/backend guarantee.

---

# PROGRAM K — Reliability and stress qualification

## K1 — Long-duration stress

Run sustained prefill/decode loops to detect:

- leaks;
- resource growth;
- stale buffers/pages;
- cache corruption;
- synchronization bugs;
- sporadic numerical divergence.

## K2 — Boundary/fuzz testing

Property/fuzz inputs for:

- shapes;
- offsets;
- capacities;
- GQA ratios;
- page tables;
- variable-length metadata;
- bias cardinality;
- tuner candidate schemas;
- serialized plan caches.

## K3 — Recovery/fail-closed behavior

Test:

- unsupported adapter capability;
- invalid cached plan;
- corrupted cache;
- unavailable optimized pipeline;
- driver capability drift;
- partial feature support.

Never silently return a numerically different mode.

---

# PROGRAM L — SciAgent production qualification

## L1 — Prefill integration

Route SciAgent prefill through FLAT behind opt-in selection. Preserve prior path as oracle until parity/performance evidence supports promotion.

## L2 — Decode integration

Complete current M15 chain and later dedicated decode-kernel promotion.

## L3 — KV-cache integration

Use resident contiguous first, then paged KV once qualified.

## L4 — End-to-end generation parity

Compare complete token generation trajectories, not only isolated attention tensors, under deterministic decoding settings.

## L5 — End-to-end performance

Measure:

- time-to-first-token;
- decode inter-token latency;
- tokens/s;
- memory footprint;
- long-context scaling;
- batch scaling.

No production-default promotion before this gate.

---

# PROGRAM M — ElasticAutoTuner beyond FLAT

After FLAT integration is mature, reuse the tuner for:

- SciRust GEMM;
- fused projection kernels;
- MLP;
- RMSNorm;
- RoPE;
- convolution and other kernel families where a meaningful discrete search space exists.

FLAT remains the first proving ground. Do not generalize prematurely if it slows FLAT completion.

---

## 18. Ordered implementation sequence from current state

This is the mandatory near-term order unless a failing dependency requires a local detour:

1. finish and merge FLAT PR #23;
2. finish and merge SciRust PR #1141;
3. route SciAgent real resident decode through opt-in M15;
4. add paired same-adapter decode parity/latency harness;
5. complete M13 combinations with variable-length/pre-rotated paths if not already fully covered;
6. implement dedicated decode kernel family;
7. implement local/sliding attention oracle + portable kernel;
8. complete resident KV contract and page-ready metadata;
9. implement packed/ragged variable-length study and candidate;
10. implement paged KV groundwork;
11. implement backward scalar oracle;
12. implement portable recomputation backward;
13. optimize backward;
14. define FLAT Kernel IR;
15. implement verifier + deterministic WGSL emitter;
16. migrate selected kernel families to generated IR;
17. create `elastic-autotuner` crate and core schemas;
18. hardware fingerprint + problem classifier;
19. deterministic candidate generator + constraint solver;
20. cost model + benchmark engine;
21. persistent evidence/plan cache;
22. integrate ElasticAutoTuner into FLAT planner;
23. tune prefill/decode/backward families;
24. open cooperative-matrix research path and prototype where standards permit;
25. complete SciAgent prefill/decode/paged-KV integration;
26. run exhaustive numerical/stress/performance campaigns;
27. lock production plan set for supported reference hardware classes;
28. complete documentation, compatibility matrix and release qualification;
29. only then declare FLAT-ATTENTION 1.0 / 100% complete.

---

## 19. Final 100% completion gate

FLAT-ATTENTION may be declared complete only when **all** of the following are true:

### Functional

- MHA complete;
- GQA complete;
- MQA complete;
- causal complete;
- non-causal complete;
- rectangular/cross-attention complete;
- local/sliding complete;
- additive bias/ALiBi complete;
- variable-length padded complete;
- packed/ragged decision completed and implemented where justified;
- prefill complete;
- decode complete;
- speculative short-window decode qualified;
- resident contiguous KV complete;
- paged KV complete;
- raw-K and pre-rotated-K cache modes complete;
- forward complete;
- backward complete;
- recomputation complete;
- mixed-precision qualified paths complete for every format claimed.

### Performance architecture

- portable baseline retained;
- vectorized path qualified;
- subgroup path qualified;
- buffering path qualified or explicitly rejected by evidence;
- generated FLAT IR path qualified;
- open matrix acceleration path either qualified or explicitly documented unavailable on current supported standards/devices;
- no unmeasured optimized path is default.

### ElasticAutoTuner

- deterministic hardware fingerprint;
- workload classifier;
- candidate generator;
- constraint solver;
- analytical cost model;
- benchmark engine;
- correctness-before-timing gate;
- persistent evidence cache;
- plan invalidation;
- Cold/Learn/Locked/Audit modes;
- multi-objective selection;
- FLAT planner integration;
- reproducible plan replay.

### Correctness and reliability

- scalar oracle matrix green;
- real-device parity matrix green;
- gradient checks green;
- deterministic repeatability gates green where promised;
- adversarial numerical corpus green;
- page-boundary and cache-reset stress green;
- long-duration stress green;
- fuzz/property gates green;
- no hidden CPU fallback;
- no full score/probability matrix in fused execution paths.

### SciRust/SciAgent

- pinned immutable FLAT integration;
- resident prefill integration;
- resident decode integration;
- contiguous and paged KV integration;
- end-to-end deterministic generation parity;
- end-to-end TTFT/decode throughput benchmarks;
- production selection uses qualified Elastic plans;
- old fallback retained until deprecation is separately justified.

### Documentation/release

- public API documented;
- numerical guarantees documented;
- hardware/backend capability matrix documented;
- benchmark methodology documented;
- tuning cache format/version documented;
- integration guide documented;
- release checklist reproducible;
- all required repository CI green on release head.

If one mandatory item above is missing, FLAT-ATTENTION is **not** 100% complete.

---

## 20. Definition of success

The final system must satisfy this statement truthfully:

> FLAT-ATTENTION is SciRust's complete Rust-native high-performance attention engine. It supports modern inference and training attention semantics, resident and paged KV execution, deterministic reference/portable modes, generated optimized kernels, and hardware/workload-specific execution chosen by ElasticAutoTuner from validated reproducible evidence, without making a vendor SDK part of the core architecture.

That statement is the project finish line, not the starting marketing claim.
