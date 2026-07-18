# SciRust-Hypermemory — Phase 1 Research Specification

> **Status:** Phase 1 (deterministic exact-memory foundation).
> **Positioning:** A deterministic associative-memory *research* subsystem using
> 16-dimensional hypercomplex representations and explicitly parenthesized
> sedenion compositions, evaluated against ordinary real-vector baselines.
> **Nothing here is claimed to be state-of-the-art, cognitive, or superior to a
> plain 16-real-number index. Every claim below is backed by a test, a
> benchmark, a derivation, or is labelled as a hypothesis.**

This document is the authoritative contract for the `scirust-hypermemory`
crate. Where the code and this document disagree, that disagreement is a bug to
be reconciled — either the code or the spec is wrong.

---

## 1. Motivation

SciRust already contains a register-resident 16-dimensional hypercomplex
algebra (`scirust_simd::hypercomplex::SedenionSimd`, the fourth Cayley–Dickson
iteration ℝ → ℂ → ℍ → 𝕆 → 𝕊). It is exact, tested, and exhibits the two
properties that make sedenions mathematically interesting and practically
dangerous:

* **non-associativity** — `(a·b)·c ≠ a·(b·c)` in general; and
* **zero divisors** — non-zero `x, y` with `x·y = 0`, e.g. the exact identity
  `(e₁ + e₁₀)·(e₄ − e₁₅) = 0`.

There is a recurring hypothesis in associative-memory research that a richer
algebra over a small fixed dimension can encode *structured relations* (ordered,
parenthesized compositions) more faithfully than a flat real vector of the same
width. Phase 1 does **not** try to prove that hypothesis. It builds the
*minimal, exact, deterministic* substrate on which the hypothesis can later be
tested or falsified, and it wires up the real-vector baseline that any such
claim must beat.

The concrete question Phase 1 makes answerable is:

> Given the *same 16 scalar components*, does routing them through the sedenion
> algebra provide any measurable advantage over treating them as 16 plain real
> numbers — for exact similarity retrieval, and for representing parenthesized
> relations?

Phase 1's honest, tested answer for **retrieval** is *no* (see §9 and §10): an
exact similarity search over the normalized effective representation is
bit-identical whether the 16 lanes are interpreted as a sedenion or as a real
vector. The only place the algebra does anything a real vector cannot is
**relation composition** (§5), and even there Phase 1 only provides the exact,
auditable machinery — not a claim that it is useful.

## 2. Exact scope (what Phase 1 *is*)

Phase 1 delivers a deterministic, non-panicking, exact **oracle**:

1. **Generation-safe identifiers** (`ConceptId`) — stale ids never resolve to a
   reused slot.
2. **Exact payload separation** (`ConceptRecord`) — the sedenion is *not* the
   sole source of truth; the exact byte payload, an immutable anchor, an
   optional bounded residual, metadata, an insertion sequence, access metadata,
   importance, and a content digest are all preserved.
3. **Stable concept store** (`S16Store`) — slot/generation array with
   deterministic iteration and generation-safe slot reuse.
4. **Exact exhaustive index** (`S16ExactIndex`) — brute-force top-k over the
   effective 16-lane representations, with a deterministic total order and
   `ConceptId` tie-breaking. This is the reference against which every future
   approximate index will be measured.
5. **Explicitly parenthesized relations** (`S16Expr` / `S16Relation`) — the
   parenthesization is stored *as data*; relations are reconstructed from the
   stored expression, never by inverting a product.
6. **Zero-divisor instrumentation** (`ProductDiagnostics`) — every relation
   product is measured for the near-zero-divisor condition; no silent NaN, no
   silent zero.
7. **Retention baseline** — metadata and a `RetentionPolicy` interface
   sufficient for later bounded-memory experiments, using a caller-supplied
   logical tick (never wall-clock).
8. **Real-vector baseline** (`Real16Index`) — the same 16 components, the same
   ids, payloads, metric, tie-break, and corpus order, interpreted as a plain
   real vector.

## 3. Non-goals (what Phase 1 is *not*)

Explicitly **out of scope** for Phase 1 (deferred to later phases):

HNSW; approximate nearest-neighbour search; dynamic PCA; loose-boundary trees;
KD-trees; distributed storage; lock-free concurrency; GPU kernels; persistence
through mmap; autonomous agents; natural-language generation; global
continual-learning loops; automatic movement of stored concepts; production
cryptography; reverse-mode differentiation over the full store; and any residual
*learning* loop (Phase 1 stores a residual but never auto-updates it — learning
is disabled).

**Scientific non-claims.** Phase 1 does **not** claim that: 16 dimensions
constitute conventional hyperdimensional computing; non-associativity
automatically stores or reconstructs syntax trees; sedenion multiplication is
collision-free; automatic differentiation establishes causality; all sedenion
operations remain register-resident on every CPU; the memory replaces
databases, RAG, vector databases, or neural networks; or the architecture is
cognitively equivalent to biological memory.

## 4. Algebraic foundation

The representation layer is reused verbatim from `scirust-simd`; Phase 1 adds no
new algebra. The relevant facts (all tested in `scirust-simd`) are:

* `SedenionSimd` is a transparent wrapper over a 16-lane `f32` register,
  `#[repr(C, align(64))]`, `size_of == align_of == 64`.
* Cayley–Dickson product `(a,b)·(c,d) = (a·c − d̄·b, d·a + b·c̄)` over octonions.
* `norm_sqr(s) = Σ sᵢ²`, and — crucially — **𝕊 is not a composition algebra**:
  `‖x·y‖² ≠ ‖x‖²·‖y‖²` in general, which is exactly why zero divisors exist.
* `s̄·s = s·s̄ = ‖s‖²·1` at every Cayley–Dickson level, so *every* non-zero
  sedenion is two-sided invertible — yet zero divisors still exist, because the
  usual "invertible ⇒ no zero divisor" argument needs associativity, which 𝕊
  lacks.

**Effective representation.** Each concept's search vector is

```text
effective = normalize(anchor + bounded_residual)
```

computed once at insertion (anchor and residual are immutable in Phase 1, so the
cached value is stable). Validation rules (see §7 invariants):

* the sum must be lane-wise finite, else `NonFiniteRepresentation`;
* `‖sum‖²` must be finite (no overflow) and strictly positive, else
  `NonFiniteRepresentation` / `ZeroNormRepresentation`;
* the normalized result is re-checked finite;
* normalization is `s.scale(1/√‖s‖²)`, a fixed scalar multiply — no library call
  that could reassociate.

**Relation atoms use the anchor, not the effective vector.** Relation
composition (§5) multiplies the *raw immutable anchor* codes, not the normalized
effective vectors. This is deliberate: normalizing every operand to unit norm
would hide the norm dynamics that produce zero divisors. Keeping the anchor
faithful is what makes `(e₁+e₁₀)·(e₄−e₁₅) = 0` observable through the public
relation machinery.

## 5. Why the expression tree stays explicit

Non-associativity is represented as **data**, never as an emergent property of a
scalar code. A relation is a binary tree

```rust
enum S16Expr { Atom(ConceptId), Product { left: Box<S16Expr>, right: Box<S16Expr> } }
```

* `(a·b)·c` and `a·(b·c)` are *independently representable* and produce
  different trees, different digests, and (in general) different sedenion codes.
* **Reconstruction reads the stored tree.** We never attempt to invert a
  sedenion product to recover its operands — that is ill-posed in the presence
  of zero divisors and non-associativity, and Phase 1 makes no such claim.
* Evaluation is **iterative** (an explicit heap stack, two-stack post-order), so
  a deep tree cannot exhaust the native call stack.
* Two independent guards prevent pathological input: `max_depth` and `max_size`,
  both enforced during traversal, both surfaced as typed errors
  (`ExpressionDepthLimit`, `ExpressionSizeLimit`). `max_depth` is additionally
  clamped to a hard crate ceiling.
* An atom that references a missing, vacant, or stale concept fails cleanly
  (`MissingAtom` / `StaleId` / `VacantSlot`), never a panic.

The **expression digest** is a domain-separated SHA-256 over a prefix-free
pre-order serialization of the tree (node-type tag + fixed-width atom bytes), so
it is a stable, order-sensitive identity for the *structure*, independent of the
floating-point code it evaluates to.

## 6. Zero-divisor risk

Zero divisors are treated as a first-class engineering hazard, not a curiosity.
For every product evaluated through the relation machinery, `ProductDiagnostics`
records:

* `lhs_norm_sqr`, `rhs_norm_sqr`, `result_norm_sqr`;
* `near_zero_divisor` — true iff **both** operands are strictly non-zero
  (`lhs_norm_sqr > 0 ∧ rhs_norm_sqr > 0`) **and** `result_norm_sqr ≤ threshold`;
* `finite` — true iff the result is lane-wise finite.

The **threshold** is an explicit, documented, deterministic parameter
(`ZeroDivisorThreshold`, default `1e-12` on the *squared* norm). Comparison is
`result_norm_sqr ≤ threshold` (inclusive), evaluated with ordinary IEEE-754
`<=`. Low-norm products are **not** rejected by default — the diagnostic is
informational; the caller decides. Normalization-safety
(`result_norm_sqr > 0 ∧ finite`) is exposed separately so a caller can avoid
dividing by (near-)zero.

The canonical test reuses the SciRust identity: inserting concepts with anchors
`e₁+e₁₀` and `e₄−e₁₅`, the relation product is exactly the zero sedenion, both
operand norms are `2`, the diagnostics flag `near_zero_divisor = true`, nothing
panics, and the original relation structure is still recoverable from the stored
expression.

## 7. Deterministic store architecture

`S16Store` is a `Vec<Slot>` plus an explicit free list (`Vec<u32>`):

```text
Slot::Occupied { generation, record }   Slot::Vacant { generation }   Slot::Retired
```

**Invariants (enforced and tested):**

* **I1 — no stale resolution.** A `ConceptId{slot, generation}` resolves to a
  record only if the slot is `Occupied` with the *same* generation. Removal
  bumps the slot's generation, so every previously issued id for that slot
  becomes stale (`StaleId`) and can never resolve to the next occupant.
* **I2 — generation-safe reuse.** A removed slot is pushed to the free list with
  an incremented generation and reused before any new slot is appended. If the
  generation would overflow `u32::MAX`, the slot is **retired** (never reused)
  rather than risk an ABA collision.
* **I3 — deterministic iteration.** Iteration is in ascending slot-index order,
  yielding only occupied records. No `HashMap`, no iteration-order
  nondeterminism anywhere in the observable API.
* **I4 — bounded id space.** Slot indices are `u32`; appending past `u32::MAX`
  slots returns `IdSpaceExhausted`. An optional capacity returns
  `CapacityExhausted`.
* **I5 — invariant-protected mutation.** No public path hands out a raw
  `&mut ConceptRecord`. `get_mut` returns `&mut ConceptMetadata` only; because
  `ConceptMetadata` has private fields and validating setters, the payload,
  anchor, residual, effective vector, and content digest can never be mutated
  after insertion. This keeps the content digest and the cached effective
  representation consistent for the record's entire lifetime.
* **I6 — validated insertion.** Every inserted concept has a finite anchor, a
  finite residual within `residual_bound`, a finite non-negative importance, and
  a computable effective representation. A record that violates any of these is
  never stored (the error is returned instead).

**Comparison with `scirust_retrieval::BoundedSemanticMemory`.**
`BoundedSemanticMemory` is the workspace's existing bounded store: it carries
per-document importance / written-at / access-count / last-access and evicts by
an importance + recency score. Phase 1 deliberately *does not duplicate its
eviction policy*. Two design differences are intentional:

* **Time model.** `BoundedSemanticMemory` uses caller-supplied `f64`
  timestamps. Phase 1 core logic uses a **logical `u64` tick** so the retention
  interface contains no floating-point time and is bit-reproducible; a
  `RetentionPolicy` trait is defined but Phase 1 ships only `NoForgetting` (keep
  everything) and a deterministic integer `LinearDecay` example. No automatic
  eviction runs in Phase 1 — only explicit `remove`.
* **Identity model.** `BoundedSemanticMemory` keys on a raw `u64` id with no
  generation; Phase 1 uses generation-safe `ConceptId`. When Phase 2 adds
  eviction, it will implement `RetentionPolicy` over `S16Store` rather than
  re-deriving `BoundedSemanticMemory`'s policy.

## 8. Exact-index oracle

`S16ExactIndex` stores a **structure-of-arrays**: `Vec<ConceptId>` parallel to a
contiguous `Vec<SedenionSimd>` of effective vectors (64-byte-aligned lanes, not
`Vec<Vec<f32>>`). Search is exhaustive and exact:

* **Metrics.** `SquaredEuclidean` (lower is better) and `Cosine` (higher is
  better). Because effective vectors are unit-norm, cosine is the dot product
  and `‖q−e‖² = 2 − 2·⟨q,e⟩`, so the two metrics are **rank-equivalent** on
  Phase 1 data — a tested property, not an assumption.
* **Reduction order.** Similarity is computed by a scalar, index-order
  (`0..16`, left-to-right) accumulation over `to_array()`, mirroring
  `scirust_retrieval::vector::dot`. The SIMD `reduce_sum` path exists in
  `SedenionSimd` for hot algebra, but the *index* uses the auditable scalar
  order so a run is trivially reproducible and the oracle is beyond dispute.
* **Total order.** Ranking uses `f32::total_cmp` (a total order — no
  `partial_cmp` unwrap, NaN-safe), then `ConceptId` ascending as the
  deterministic tie-break.
* **Degenerate inputs.** `k == 0` → `Ok([])`; empty index → `Ok([])`; a
  zero-norm or non-finite query with `k > 0` → typed `Err` (never a silently
  empty result that would conceal invalid input).

## 9. Baseline methodology

`Real16Index` is `S16ExactIndex`'s twin over `[f32; 16]` interpreted as a plain
real vector. It uses the **same** `ConceptId`s, payloads, queries,
`SimilarityMetric`, tie-break, and corpus order. The effective 16 components fed
to both are identical.

The tests assert that `S16ExactIndex` and `Real16Index` return **bit-identical**
rankings and scores on the same corpus. Where practical, both are additionally
cross-checked against `scirust_retrieval::DenseIndex` (cosine), the workspace's
independent audited exact dense index, to confirm the oracle agrees with a
second implementation.

This is the crux of honest positioning: *for retrieval, the sedenion algebra is
provably not doing anything a real vector isn't.* Any future claim of retrieval
advantage must therefore come from something Phase 1 does not yet have (learned
residuals, relation-aware scoring, structured queries), and must beat this
baseline to be believed.

## 10. Falsification criteria

Phase 1 is built to be *disproved*. The subsystem should be considered to have
**failed its core hypothesis** if any of the following hold (each maps to a
concrete measurement):

* **F1 — no retrieval advantage.** Sedenion-based exact retrieval produces no
  measurable ranking or quality advantage over the `Real16` baseline on the same
  components. *Phase 1 status: confirmed — they are bit-identical by
  construction (test `index_matches_real16_baseline`). Retrieval alone does not
  justify the algebra.*
* **F2 — collisions / near-zero products too frequent.** If, over a
  representative corpus of relation products, the near-zero-divisor rate is high
  enough that relation codes routinely collapse toward zero, parenthesized
  relations cannot reliably encode structure. Measured by
  `ProductDiagnostics::near_zero_divisor` frequency (the `experiments` module and
  the `hypermemory-falsify` binary). *Result: **not triggered** for generic
  operands — 0 near-zero divisors over 100 000 random products; collapse is a
  structured / low-rank input risk, not a generic one. See
  [`SCIRUST_HYPERMEMORY_F2_F6.md`](SCIRUST_HYPERMEMORY_F2_F6.md).*
* **F3 — latency without benefit.** If the sedenion product / relation
  evaluation adds latency (benchmarks in §11) without a corresponding
  improvement in retrieval or relation discrimination, the algebra is pure cost.
* **F4 — residual learning destabilizes old concepts.** (Deferred: learning is
  off in Phase 1. When enabled, if updating one concept's residual measurably
  perturbs the effective vectors or rankings of unrelated concepts beyond a
  stated tolerance, the residual mechanism is rejected.)
* **F5 — memory cost exceeds value.** If per-concept memory (payload + anchor +
  residual + cached effective + metadata + digest) is disproportionate to any
  demonstrated benefit, the representation is too heavy.
* **F6 — relation codes don't discriminate structure.** If distinct
  parenthesizations `(a·b)·c` vs `a·(b·c)` frequently evaluate to
  indistinguishable sedenion codes (within a stated tolerance) across a corpus,
  the *code* cannot stand in for the *structure* — in which case only the
  explicit stored tree (never the code) may be trusted, and any "the code
  encodes the tree" claim is falsified. Phase 1 already refuses to rely on the
  code for reconstruction, so it is robust to F6 by design; F6 would instead
  falsify any *future* code-only shortcut. *Result: **not triggered** for
  generic operands — 100% of random triples are discriminable (`min ρ ≈ 0.17`) —
  but it fires exactly, and only, inside associative subalgebras (complex /
  quaternion), where the associator vanishes. See
  [`SCIRUST_HYPERMEMORY_F2_F6.md`](SCIRUST_HYPERMEMORY_F2_F6.md).*

Passing Phase 1's tests does **not** refute F1–F6; it only establishes the exact
substrate on which they are measured. F1 is confirmed for retrieval (the
intended, conservative outcome); **F2 and F6 are now measured** and not
triggered for generic operands, with their failure regimes mapped precisely
(structured / low-rank inputs for F2; associative subalgebras for F6).

## 11. Benchmark protocol

Criterion benchmarks live behind an explicit `[[bench]]` target
(`benches/hypermemory_bench.rs`, `harness = false`). They measure:

* concept insertion (store);
* exact lookup by `ConceptId`;
* exhaustive top-k search at corpus sizes `1_000`, `10_000`, `100_000`;
* expression evaluation at several tree depths;
* a single sedenion product;
* `Real16` baseline search at the same sizes;
* (optional) `scirust_retrieval::DenseIndex` search at the same sizes.

Corpora are generated by a fixed-seed in-crate LCG (no RNG dependency), so a run
is reproducible. **No benchmark numbers are committed as fact.** If measurements
are run, they are reported only alongside the exact hardware, compiler version,
`cargo` invocation, and commit SHA. The benchmark *code* is the deliverable; the
*numbers* are provenance-stamped or absent.

## 12. Future phases (non-binding sketch)

* **Phase 2** — deterministic retention/eviction implementing `RetentionPolicy`
  over `S16Store`; bounded memory experiments compared head-to-head with
  `BoundedSemanticMemory`. *(Implemented: `S16BoundedMemory` — capacity-capped,
  lowest-retention eviction with ascending-`ConceptId` tie-break, recency bump
  on search, threshold `forget`; behavioural parity with
  `BoundedSemanticMemory` is tested in `tests/bounded_parity.rs`.)*
* **Phase 3** — residual *learning* (currently stored but frozen), with F4 as
  the gate.
* **Phase 4** — approximate indexes (HNSW / ANN) measured against this Phase 1
  exact oracle for recall.
* **Phase 5** — structured / relation-aware queries, measured against the
  `Real16` baseline for F1.

Each phase must beat the Phase 1 oracle/baseline on a stated, falsifiable
metric, or be rejected.

## 13. Limitations

* The effective representation is `f32`; cross-target bit-for-bit reproducibility
  holds where IEEE-754 `f32` `+`, `*`, and `sqrt` are correctly rounded and not
  reassociated (Phase 1 uses fixed index-order scalar reductions to maximize
  this). The crate builds with **no** `fast-math` and no implicit reassociation.
* `SedenionSimd::norm_sqr` uses the SIMD `reduce_sum` (tree order); this crate
  deliberately avoids it. Both the index (scoring) and the zero-divisor
  diagnostics compute every norm with the fixed index-order scalar reduction
  (`norm_sqr_ordered`), so all norm-based decisions share one auditable,
  reproducible reduction order. The library's `reduce_sum` is only reached
  transitively inside `SedenionSimd`'s own `Mul`/`norm` if a caller invokes them
  directly.
* Payloads are owned `Vec<u8>` in Phase 1 (documented); a deterministic payload
  *reference* / interning scheme is deferred.
* Relation evaluation resolves each atom occurrence independently; there is no
  common-subexpression caching in Phase 1 (correctness over speed).
* In the Phase 1 core (`S16Store`), the retention interface is defined but no
  automatic forgetting runs (Phase 2's opt-in `S16BoundedMemory` adds it); only
  explicit `remove` changes residency.
* `f32` similarity has limited dynamic range; extremely large anchors can
  overflow `norm_sqr` — this is detected (`NonFiniteRepresentation`), not
  silently normalized to zero.
