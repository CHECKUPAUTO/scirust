# SciRust algogen tensor IR — V2 audit (Phase 0)

Mission: strengthen the `scirust-algogen` tensor-program representation into a
general-purpose, deterministic, auditable **scientific algorithm discovery IR**.

This document audits the infrastructure exactly as it exists at mission-start
master commit `d3e435d690e1a2a95e5b4379c681bdb8954cfbb0`. It records what is
genuinely strong, what is weak, where semantics are implicitly assumed, and
which limitations block the ADA / scientific-discovery research program.
Nothing here assumes the current design is correct merely because its tests
pass.

Audited surface:

| Module | Lines | Role |
|---|---|---|
| `tensor/ir.rs` | 75 | IR enum + program struct |
| `tensor/verify.rs` | 416 | causality, shape inference, resource limits |
| `tensor/interpreter.rs` | 795 | safe deterministic execution |
| `tensor/active.rs` | 38 | backward liveness |
| `tensor/canonicalize.rs` | 208 | dead-code elimination |
| `tensor/canonical.rs` | 177 | canonical bytes + FNV-1a fingerprint |
| `tensor/digest.rs` | 672 | versioned SHA-256 archive encoding |
| `tensor/cost.rs` | 484 | structural cost model + peak-live analysis |
| `tensor/generate.rs` | 522 | valid-by-construction generator |
| `tensor/mutate.rs` | 841 | verify-then-select mutation |
| `tensor/crossover.rs` | 329 | prune-concatenate-join crossover |
| `tensor/fitness.rs` | 322 | MSE loss + failure penalty + Pareto inputs |
| `tensor/population.rs` | 922 | ranking, fronts, tournament, evolution |
| `tensor/experiment.rs` | 416 | deterministic experiment runner |
| `tensor/archive.rs` | 1352 | hall of fame, archive, verification, replay |
| `tensor/problem.rs` | 755 | serialisable problems + exact-oracle benchmarks |
| `tensor/dataset.rs` | 245 | validated case datasets |
| `tensor/rng.rs` | 128 | SplitMix64 deterministic stream |
| `tensor/report.rs` | 157 | text/JSON reporting |

Downstream callers: no crate outside `scirust-algogen` references
`scirust_algogen::tensor` (verified by grep over all `*.rs` and `Cargo.toml`
in the workspace). The public compatibility surface is therefore entirely
inside this crate, which materially lowers the risk of a clean V2 boundary.

---

## 1. Current strengths (must be preserved)

These properties are the reason SciRust research is trustworthy. The V2 design
must not regress any of them.

1. **Determinism as an invariant, not a hope.**
   - Single seeded SplitMix64 stream (`rng.rs`) drives generation, mutation,
     crossover, tournaments. No OS entropy, no wall-clock stopping in the core,
     no HashMap iteration in ranking or serialization order.
   - `rank()` is a strict total order: Pareto front → lexicographic objectives →
     authoritative canonical bytes → population index. Equal-fingerprint
     collisions are explicitly tested to *not* collapse ranking.
   - Rayon evaluation restores input order; sequential/parallel bit-equality is
     tested (`rayon` feature).
2. **Verification before execution.** `verify_program` enforces strict
   causality (`source < node`), per-op shape rules, rank limits, checked shape
   products, per-tensor and total register element budgets, output bounds.
   Invalid candidates never reach evaluation.
3. **Safe interpreter with explicit non-finite policy.** No FFI, no panics on
   hostile inputs, layout/stride validation of inputs, dead-code never executed
   or validated, NaN/Inf inputs and results rejected deterministically with
   precise error variants (`NonFiniteInput`, `NonFiniteResult`,
   `NonFiniteScaleFactor`). Overflow-to-infinity is tested.
4. **Identity discipline.** Canonical bytes are authoritative identity;
   the 128-bit FNV-1a fingerprint is documented as a collidable hint and never
   used for equality, dedup, or ordering. Archive digests are SHA-256 over an
   explicit, versioned, domain-separated byte encoding — never JSON, never
   `Debug` formatting.
5. **Versioning culture.** `ARCHIVE_SCHEMA_VERSION`, `DIGEST_FORMAT_VERSION`
   exist; unknown versions are rejected (`UnsupportedFormat`,
   `UnsupportedSchemaVersion`), never silently interpreted.
6. **Valid-by-construction generation** plus defensive re-verification; bounded
   candidate enumeration in mutation (cap 512) with mandatory re-verification of
   every mutant; crossover has a documented parent-fallback instead of invalid
   offspring.
7. **Structural cost model** that is pure, saturating, and honest about being a
   lower bound on interpreter residency (`peak_live_elements` vs
   `total_active_elements` distinction is documented and tested with exact
   oracles).
8. **Exact-oracle benchmark problems** implemented independently of the IR under
   test; multi-case so constant-output programs cannot pass.
9. **Archive integrity + replay separation**: `verify_archive` checks stored
   consistency; `replay_experiment` reruns generations from the archived seed.
10. **Excellent negative-test hygiene**: nearly every verifier rule, degenerate
    size, pathological bound (`usize::MAX` tournaments), and non-finite path has
    a dedicated test.

## 2. Current weaknesses

### 2.1 Operator vocabulary is not a scientific vocabulary

The entire IR is six opcodes: `Input`, `Add`, `MatMul`, `Transpose2d`, `Relu`,
`Scale { f32 }`.

Consequences:

- No subtraction, multiplication, division, negation as first-class ops
  (`Scale` only multiplies by a compile-time constant).
- No elementary functions (`exp`, `log`, `sqrt`, …) → softmax-family,
  normalization, attention score transforms are **unrepresentable**.
- No min/max/clamp → stable online softmax (`max` state) unrepresentable.
- No comparisons or selection → conditional numerical algorithms,
  masking, threshold logic unrepresentable.
- No constants other than `Scale` factors → compensated summation (Kahan)
  needs the constant `0.0`; Welford needs constants; neither can be written.
- `Relu` is the sole nonlinearity; it is an ML activation, not a scientific
  primitive, and it cannot express soft-thresholding, clamping, or masking.

### 2.2 Reductions do not exist

There is no sum/max/min/product over any axis. A reduction as trivial as
"sum of a vector" cannot be expressed. Mean/variance, moments, normalization,
loss aggregation, attention denominators: all unrepresentable. This alone makes
the current IR unfit for the stated research program.

### 2.3 Exact-shape-only arithmetic; no shape algebra

`Add` requires bitwise-equal shapes; there is no broadcasting, reshape,
squeeze/unsqueeze, general permutation, broadcast_to, concat, or slicing.
Scalar × tensor exists only via `Scale` with a baked constant. Rank-0 values
exist only incidentally (empty shape). Stencil-like and block algorithms need
shape manipulation; none is possible today.

### 2.4 Single dtype, hard-wired `f32`

`TensorND` is `f32`-only and the whole pipeline assumes it (cost model even
hard-codes `BYTES_PER_ELEMENT = 4`). No `f64` for numerically sensitive
discovery, no Boolean mask type (comparisons don't exist anyway), no integer/
index type, no extension point for bf16/f16 beyond `scirust-compute::DType`
(which exists but is unused here).

### 2.5 Single output; no state; no recurrence

`TensorProgram.output` is one register. There is no notion of state, previous
state, scan/fold, iteration count, or multiple observable outputs. Therefore:
online-softmax recurrences, Welford moments, Kahan compensation, Newton-style
iterative updates, EMA filters — the core objects of ADA A1/A6/A8 — are not
expressible at any cost. This is the single largest capability gap.

### 2.6 Canonicalization = DCE only; identity is syntactic, not semantic

- `canonicalize.rs` performs dead-code pruning with remapping. That's all.
- No CSE: `(x+0)` computed twice stays duplicated; semantically identical
  programs built in different orders have different identities forever.
- No constant folding, no identity simplification (`x+0`, `x*1`),
  no commutative operand normalization, no register renumbering beyond DCE
  compaction.
- Consequence: the search engine cannot recognize that two candidates compute
  the same function; archives accumulate syntactic duplicates; diversity
  metrics overcount.

### 2.7 Cost model lacks operator-class resolution

`estimated_flops` treats `Relu`, `Add`, `Scale` identically (1/elem) and has no
notion of transcendental weight (exp/log), division cost, comparison/select,
reductions, or expression depth / critical-path proxy. For algorithm discovery,
where "fewer exps" vs "fewer adds" is a real trade-off axis, this is too coarse.

### 2.8 No range/constant analysis layer

Nothing tracks known constants, known-nonnegative values, or finiteness
possibilities. This blocks both safer rewrites (see 2.9) and earlier rejection.

### 2.9 Rewrite-safety is unexamined because no rewrites exist

Once rewrites are introduced they must state their validity domain: e.g.
`x * 0 → 0` is invalid for IEEE NaN/±inf operands; float addition must **not**
be treated as associative or as commutative-with-rounding-neutrality unless the
regime says so. Today nothing documents any numerical regime at all; the only
implicit regime is "reject anything non-finite at runtime".

### 2.10 Grammar is a fixed struct; search-space control is minimal

`OperatorSet { add, matmul, transpose, relu, scale }` booleans. There is no way
to say "allow exp but at most 2", "forbid division", "maximum expression
depth 8", "only scalar recurrence state", "no recurrence". With a richer
operator set, unconstrained enumeration would explode combinatorially; budget
control must be first-class.

### 2.11 Counterexamples are discarded

`FitnessReport` aggregates failures into a count and penalty. The failing case
id, inputs, expected vs actual outputs, absolute/relative errors are dropped.
Autonomous falsification loops (ADA) need bounded counterexample retention.

### 2.12 Smaller defects and risks found during audit

- `ProgramError` and `VerificationLimits` are not `Serialize`/`Deserialize`
  (problems mirror limits by hand in `ProblemLimits`); fine today, but V2
  should make limits serde-native to avoid drift between mirror structs.
- `CostReport.bloat_ratio` is `f64` inside a `Copy` struct used in Pareto
  objectives — `total_cmp` handles NaN, but a NaN bloat ratio would be silent
  nonsense; it is derived from usize division so currently unreachable.
- `peak_live_elements` is O(n²) in instruction count (fine ≤1024, worth noting
  for larger budgets).
- `crossover` joins parents with `Add` when output shapes match — semantically
  arbitrary but deterministic and verified; acceptable, but with typed V2 the
  junction should be chosen among type-compatible joins.
- `mutate.rs` enumerates up to 512 candidates then draws one — the cap keeps a
  fixed prefix, so determinism holds, but candidate order is load-bearing and
  must stay documented.
- `digest.rs` rejects non-finite floats in fields "where they are not valid"
  while deliberately distinguishing ±0.0 — good; but `FitnessReport.loss` can
  legitimately be enormous (penalty 1e12), still finite, OK.
- `archive.rs` `HallOfFameFingerprintCollision` issue exists — good — but the
  hall-of-fame `consider()` is O(capacity) per insert with full sort each time;
  acceptable at capacity 16–64.
- `interpreter.rs` special-cases zero-length dimensions around the einsum
  backend (backend panics otherwise). A V2 interpreter should avoid delegating
  semantics to a backend that can panic; implement reductions/matmuls directly
  in safe Rust with checked indexing.
- `problem.rs` benchmarks cover identity/scale/relu/transpose/add/matmul/affine
  — all expressible precisely because the operator set is tiny. None of them
  exercises discovery of a *composed* numerical idiom requiring several
  interacting primitives.

## 3. Hidden coupling and semantic assumptions

1. **Interpreter ↔ verifier shape contract.** The interpreter re-derives shapes
   via `verify_program` and indexes `left.shape[0]` etc. assuming verifier
   guarantees. Any new op must keep this pairing total (every op: verifier rule
   + interpreter implementation + tests), or the guarantee evaporates.
2. **Dead-code ↔ numerical-validation coupling.** Only *active* inputs are
   validated for finiteness/layout. This is deliberate and documented, but it
   means liveness is part of the numerical contract, not just an optimization.
3. **Register-index arithmetic in three places.** Insert/delete remapping
   (mutation), offset shifting (crossover), compaction (DCE) each reimplement
   source rewriting. A V2 with named `ValueId`s and structural maps removes
   this triplication risk.
4. **`Scale` factor participates in identity by bits** (canonical bytes use
   `factor.to_bits()`), so `-0.0 ≠ +0.0` in identity. Any folding of constants
   must preserve bit-exact float semantics or bump the canonicalization
   version.
5. **Penalty constant `CASE_FAILURE_PENALTY = 1e12`** is load-bearing across
   fitness, problem criteria, and tests; changing it changes experiment
   outcomes. It must remain pinned and documented.
6. **Archive digest covers `crate_version`** — rebuilding with a different
   crate version breaks archive digests by design (provenance), meaning
   "same content, different build" is detectable but also non-replayable across
   versions. Keep this behavior explicit in V2.
7. **`OperatorSet::all()` default** means adding an operator to V1 would change
   existing experiments' behavior. V1 is frozen; V2 grammar profiles must be
   explicit, versioned configuration instead of ambient defaults.

## 4. Backward-compatibility constraints

- Public API: everything under `scirust_algogen::tensor::*` is exported
  (`mod.rs`). No external workspace crate uses it today, but the module is
  public API of the crate; breaking it gratuitously is unacceptable.
- Serialized artifacts: `ExperimentArchive` (schema v1) embeds V1 programs and
  configs; old archives must continue to verify against the V1 code paths.
- Deterministic identities: V1 fingerprints/canonical bytes must not change at
  all (they encode raw opcode tags). Any V1 change would invalidate archived
  research. ⇒ **V1 code must remain byte-stable; V2 is additive.**

## 5. Search-space limitations (summary)

| Limitation | Blocks |
|---|---|
| 6 operators, no exp/log/min/max/div | stable softmax, attention scores, normalization |
| no reductions | moments, sums, denominators, pooling |
| no broadcasting/reshape | stencils, block/tile algorithms, scalar-tensor mixing |
| single output | multi-objective recurrences (m, l, o) |
| no state/recurrence | online/streaming algorithms entirely |
| no bool masks | conditional numerical algorithms, masking (A2/A9) |
| f32 only | high-precision oracles, index-valued results |
| no gather/slice | sparse/indexed algorithms (A2/A4/A5) |

## 6. Numerical-semantics limitations

- One implicit regime: reject non-finite inputs/intermediates/results.
  No declared regime concept, so rewrite validity domains cannot even be
  stated today.
- Signed zero is distinguished in identity (good) but nothing else reasons
  about it.
- Division, sqrt/log of negatives, exp overflow, 0×∞ are unrepresentable
  operations today — which is "safe" but vacuous; once added they need exact
  documented IEEE behavior plus regime-gated rejection.

## 7. Recurrence / streaming limitations

None present. Nothing can iterate. Static trip counts, declared state
signatures, step functions referencing `prev(state)` — all absent. See §2.5.

## 8. Lowering limitations

- The IR conflates "what" and "how" mildly: `Transpose2d`/`MatMul` delegate to
  einsum with zero-dim workarounds in the interpreter. There is no separation
  of semantic IR from execution plan; no lowering boundary exists. Not blocking
  for discovery, but V2 must define the boundary (semantic IR → optimization
  passes → execution plan) so future CPU/SIMD/GPU backends never contaminate
  canonical identity.

## 9. Conclusion

The V1 substrate is disciplined, deterministic, and well-tested, but it is a
tiny affine/relu network DSL, not a scientific algorithm IR. The deficiencies
are architectural (single output, no types, no sections, no recurrence), not
mere operator-count gaps. The correct move is a **new, typed, sectioned,
multi-output IR with an explicit recurrence construct**, living alongside V1
behind a compatibility adapter, reusing V1's proven patterns: seeded streams,
verify-before-execute, canonical-bytes identity, versioned digests, Pareto
ranking with structural tiebreaks.

The architecture specification for that IR follows in
`SCIRUST_ALGOGEN_IR_V2_ARCHITECTURE.md`.

---

## 10. V2 implementation follow-up and hostile self-review

The sections above remain the mission-start V1 audit. After implementing the
additive V2 namespace, a second source-level review found and corrected several
issues that ordinary positive tests did not expose:

1. The initial V2 init-section reference decoder accidentally rejected
   `Local`, despite init being a straight-line section. This prevented a tensor
   state initializer from using `Const` followed by `BroadcastTo`. Init locals
   are now legal, causal, inferred, and regression-tested.
2. `ValueTensor` has public serde fields. Its constructor validated shape/data,
   binary32 exactness, and Boolean encoding, but a struct literal/deserialized
   payload could bypass the constructor. Every external input/item and
   counterexample tensor is now revalidated at the interpreter/search trust
   boundary with structured payload errors.
3. Early V2 canonical bytes versioned the encoding but did not include IR,
   canonicalization, and numerical-regime identifiers. All three are now in
   the canonical header; changing a regime changes identity.
4. The first rewrite framework treated several floating identities as if one
   domain covered all uses. Rules now have stable ids, ordered application
   records, and explicit Strict/finite/real applicability. A new negative test
   caught and fixed a leak that briefly allowed floating commutative sorting in
   Strict IEEE while Boolean sorting was being enabled.
5. The initial structural cost draft scaled step work inside the section tally
   and again at aggregation. It also extracted matrix dimensions incorrectly.
   Exact recurrence and `m*k*n` tests now pin scaling, per-step FLOPs, and state
   footprint.
6. Verified root extraction used `expect`. Although verification and liveness
   made the condition unreachable, the interpreter now uses checked lookups and
   a structured `MissingBinding` defensive error. Generated artifacts cannot
   make a root lookup panic.
7. Rich operator enumeration can grow cubically for ternary operations. The
   grammar has a whole-program proposal cap, per-class/depth/expensive budgets,
   deterministic truncation/rejection counters, and typed filtering before
   selection. Exhaustion is a visible error, not an implicit budget increase.

### Hostile questions and current answers

- **Did V2 become a generic programming language?** No. It has three closed
  straight-line sections and one fixed-trip state fold. There are no functions,
  arbitrary control flow, recursion, effects, dynamic allocation requests, or
  dynamic shapes.
- **Is recurrence expressive enough?** It represents multiple scalar/tensor
  state components and multiple per-step items. Online softmax, Welford,
  compensated sum, bounded Newton, and `(m,l,o)` attention folds verify and
  execute. It deliberately lacks early exit and scan-output materialization.
- **Are target algorithms hidden in the IR/search?** No magic operations exist,
  and search does not import the reference constructors. Profiles contain only
  class/budget constraints.
- **Can malformed programs reach interpreter panics?** Verification rejects
  malformed refs, shapes, axes, state, and budgets before execution; external
  payloads are revalidated; defensive register/root failures are structured.
  Safe internal kernels rely on verified checked element products. No unsafe
  code or host effect is reachable.
- **Are broadcasting/type rules ambiguous?** No: concrete shapes and NumPy
  trailing-axis rules are implemented once in shared helpers and used by
  inference. Select allows mask broadcasting but requires exact branch shape.
- **Are rewrites treating floats as reals?** Strict does not reorder float
  add/mul or remove float identities sensitive to exceptional/signed-zero
  values. No regime reassociates, distributes, or contracts FMA.
- **Can hash collisions collapse candidates?** No. FNV/SHA are hints/labels;
  canonical bytes are retained and compared for deduplication and archive
  identity. A forced same-digest/different-bytes test covers the policy.
- **Can recurrence allocate without bound?** Trip count, state count, signature
  elements, register elements, and conservative host bytes are verifier
  budgets. The fold retains state only and reports zero materialized sequence
  elements.
- **Do mutation/crossover produce garbage?** Mutation enumerates compatible
  changes then whole-program verifies; crossover exchanges section+binding
  units only across identical semantic/state contexts. Both are seeded and
  tested for replay.
- **Has the grammar become unusably broad?** GeneralScientific is broad but
  bounded; narrower named curricula are the intended starting point. Proposal,
  rejection, duplicate, interpreter, and archive counters expose pressure.
  Useful large-attention search performance is not claimed.
- **Can parallelism change ordering?** The new V2 experiment runner is
  sequential. V1's optional parallel evaluator restores indexed input order.
  V2 introduces no schedule-dependent archive update.
- **Do old bytes silently change meaning?** V1 is untouched. V2 JSON requires a
  versioned envelope and rejects bare/mismatched payloads. No automatic V1
  archive migration occurs.
- **Is ADA readiness asserted or executed?** A1–A10 map to verifier-backed
  programs; core recurrences run against numerical fixtures. The document
  explicitly distinguishes representability from discovery/proof/integration.

### Security/trust result

The semantic IR has no opcode capable of shell, filesystem, network, FFI,
dynamic Rust, unsafe callback, pointer, or host-object access. Indices in V2 are
static `Narrow` attributes checked against concrete dimensions. There is no
unbounded loop or recursive evaluator. Resource budgets cover signatures and
produced registers before kernels allocate. This is a bounded scientific IR,
not a sandbox for an otherwise general language.

### Remaining audit risks

- `ValueTensor` exposes a convenience `to_f32_tensor` compatibility conversion
  that asserts its dtype; generated programs cannot call Rust APIs, and the
  interpreter never uses it, but a future API cleanup should add a fallible
  public conversion.
- Transcendental last-bit portability depends on Rust target/toolchain libm.
- The conservative resident-byte verifier counts signature and register
  carriers but is not a proof of allocator peak during every cloned kernel
  operand. Structural liveness provides a better phase estimate; a future
  interpreter can borrow operands to tighten the bound.
- Dynamic sparse/index operations, casts/mixed precision, and formal range
  proof are absent rather than weakly specified.
- Bounded tests and replay establish implementation evidence, not mathematical
  equivalence or discovery completeness.
