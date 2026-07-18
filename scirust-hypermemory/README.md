# scirust-hypermemory

> ```text
> RESEARCH-GRADE ASSOCIATIVE-MEMORY SUBSYSTEM — PHASE 1
>
> A deterministic associative-memory research substrate using 16-dimensional
> hypercomplex (sedenion) representations and explicitly parenthesized sedenion
> compositions, evaluated against ordinary real-vector baselines.
>
> This is NOT a database, vector database, RAG system, neural network, or model
> of biological memory, and it makes NO claim to be superior to storing 16 plain
> real numbers. Phase 1 exists to build the exact, auditable oracle on which any
> such claim must later be measured — and to already falsify the easy version of
> the claim for retrieval.
> ```

The authoritative design contract is
[`docs/research/SCIRUST_HYPERMEMORY_PHASE1.md`](../docs/research/SCIRUST_HYPERMEMORY_PHASE1.md).
Where this crate and that document disagree, the disagreement is a bug.

## Positioning (read before believing anything)

The one question Phase 1 makes answerable:

> Given the *same 16 scalar components*, does routing them through the sedenion
> algebra buy anything over treating them as 16 plain real numbers — for exact
> similarity retrieval, and for representing parenthesized relations?

The tested answer **for retrieval is: no.** [`S16ExactIndex`] and the
real-vector baseline [`Real16Index`] return **bit-identical** rankings and
scores over the same effective components (test
`sedenion_index_matches_real16_over_seeded_corpora`), and both agree with the
independent audited [`scirust_retrieval::DenseIndex`]
(`sedenion_index_agrees_with_dense_index`). Similarity search is a lane-wise
operation that never invokes sedenion multiplication, so the algebra is provably
inert there. That is the intended, conservative outcome (falsification criterion
F1), not a disappointment.

The only place the non-associative algebra does anything a real vector cannot is
**explicit relation composition** ([`S16Expr`]) — and Phase 1 only provides the
exact, auditable machinery there, never a claim that it is useful.

Every scientific claim in this crate is backed by a test, a benchmark, a
derivation, or is labelled a hypothesis.

## What it contains

- **generation-safe identifiers** ([`ConceptId`]) — stale ids never resolve to a
  reused slot;
- **exact payload separation** ([`ConceptRecord`]) — the sedenion is not the
  sole source of truth; the exact byte payload, content digest, immutable
  anchor, (frozen) residual, cached effective vector, metadata, and insertion
  sequence are all preserved;
- **a deterministic slot/generation store** ([`S16Store`]) — insert / get /
  guarded `get_mut` / remove / contains / len / is_empty / slot-ordered
  iteration, with generation-safe reuse and overflow retirement;
- **an exact exhaustive index** ([`S16ExactIndex`]) — the Phase 1 oracle;
  squared-Euclidean and cosine metrics, `f32::total_cmp` ordering, `ConceptId`
  tie-break, typed errors on degenerate input;
- **explicitly parenthesized relations** ([`S16Expr`] / [`S16Relation`]) —
  iterative (stack-safe) evaluation with depth/size limits and a stable,
  order-sensitive expression digest;
- **zero-divisor instrumentation** ([`ProductDiagnostics`]) — every relation
  product is measured for the near-zero-divisor condition; no silent NaN;
- **a retention interface** ([`RetentionPolicy`], [`NoForgetting`],
  [`LinearDecay`]) over a logical `u64` tick — no automatic eviction in the
  Phase 1 core (`S16Store` only grows via explicit `remove`);
- **a bounded, decay-aware memory (Phase 2)** ([`S16BoundedMemory`]) — a
  capacity-capped layer over the store + exact index: when full, it evicts the
  lowest-retention resident (deterministic ascending-`ConceptId` tie-break),
  `search` bumps recency at a caller-supplied tick, and `forget(now, threshold)`
  evicts everything below a retention threshold. Behavioural parity with the
  workspace's `scirust_retrieval::BoundedSemanticMemory` is tested head-to-head
  in `tests/bounded_parity.rs`;
- **a real-vector baseline** ([`Real16Index`]).

Properties: pure Rust, zero FFI, `#![forbid(unsafe_code)]`, deterministic
(fixed iteration and reduction order, `ConceptId` tie-break, no `HashMap` in any
observable output, no RNG, no wall-clock), non-panicking public API.

## Dependencies

- [`scirust-simd`] (feature `portable-simd`) — the reused `SedenionSimd`
  representation. This is a hard, always-on dependency; the crate is nothing
  without it.
- `sha2` — SHA-256 for content / expression **digests only** (identity,
  corruption detection, reproducibility). Not encryption, not a secrecy claim.
- (dev only) [`scirust-retrieval`] — the independent `DenseIndex` cross-check.

## Example

```rust
use scirust_hypermemory::{ConceptSpec, S16Store, S16ExactIndex, SimilarityMetric};
use scirust_simd::hypercomplex::SedenionSimd;

let mut store = S16Store::new();
let apple = store
    .insert(ConceptSpec::new(b"apple".to_vec(), SedenionSimd::unit(1), 1.0, 0))
    .unwrap();
let banana = store
    .insert(ConceptSpec::new(b"banana".to_vec(), SedenionSimd::unit(2), 1.0, 0))
    .unwrap();

let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
index.insert_concept(store.get(apple).unwrap());
index.insert_concept(store.get(banana).unwrap());

let hits = index.search(&SedenionSimd::unit(1), 1).unwrap();
assert_eq!(hits[0].id, apple);
```

## The canonical zero-divisor case

```rust
use scirust_hypermemory::{ConceptSpec, ExprLimits, S16Expr, S16Store};
use scirust_simd::hypercomplex::SedenionSimd;

let x = SedenionSimd::unit(1) + SedenionSimd::unit(10);
let y = SedenionSimd::unit(4) - SedenionSimd::unit(15);
let mut store = S16Store::new();
let cx = store.insert(ConceptSpec::new(b"x".to_vec(), x, 1.0, 0)).unwrap();
let cy = store.insert(ConceptSpec::new(b"y".to_vec(), y, 1.0, 0)).unwrap();

let expr = S16Expr::product(S16Expr::atom(cx), S16Expr::atom(cy));
let (code, diag) = expr
    .evaluate_with_diagnostics(&store, &ExprLimits::default(), 1e-12)
    .unwrap();

assert_eq!(code.to_array(), [0.0f32; 16]);          // exact zero product
assert!(diag.unwrap().near_zero_divisor());         // flagged, no panic
// The relation structure is still recoverable from `expr` — never by inverting
// the (zero) product.
```

## Nightly-only, separate workspace

This crate mandates `scirust-simd/portable-simd` (the nightly
`feature(portable_simd)`), so — like the repo's `fuzz` harness — it is a
**separate cargo workspace, excluded from the root workspace**, which keeps the
main "whole workspace builds on stable" gate intact. Build it on nightly from
its own directory (it keeps no committed lockfile). CI runs it in the dedicated
`hypermemory` job.

## Gates

```bash
cd scirust-hypermemory
cargo +nightly-2026-07-02 fmt    --all -- --check
cargo +nightly-2026-07-02 clippy --all-targets --all-features -- -D warnings
cargo +nightly-2026-07-02 test
cargo +nightly-2026-07-02 doc    --no-deps
```

## Benchmarks

```bash
cd scirust-hypermemory && cargo +nightly-2026-07-02 bench
```

Corpora are generated by a fixed-seed in-crate LCG (no RNG dependency).
**No benchmark numbers are committed as fact.** If you report timings, report the
exact hardware, compiler version, `cargo` invocation, and commit SHA alongside
them (research document §11).

## Falsification experiments (F2 / F6)

The `experiments` module and the `hypermemory-falsify` binary run **deterministic**
surveys of the two questions that decide whether relation composition is worth
anything (F1 already showed retrieval is not):

* **F2** — zero-divisor / norm-collapse frequency of `a·b`;
* **F6** — whether `(a·b)·c` and `a·(b·c)` produce distinguishable codes.

```bash
cd scirust-hypermemory
HYPERMEMORY_GIT_COMMIT=$(git rev-parse HEAD) \
  cargo +nightly-2026-07-02 run --release --bin hypermemory-falsify
```

Unlike the timing benchmarks, these results are **deterministic reproducible
facts** (pure `f32` algebra, fixed seed). Headline finding: neither F2 nor F6 is
triggered for generic operands, and both failure regimes are mapped precisely
(structured / low-rank inputs; associative subalgebras). Full results and
interpretation:
[`docs/research/SCIRUST_HYPERMEMORY_F2_F6.md`](../docs/research/SCIRUST_HYPERMEMORY_F2_F6.md).
This establishes *capacity*, **not** usefulness.

### Relational structure discrimination ("F1 for relations")

The `binding` module and the `hypermemory-relations` binary go one step further:
do the parenthesized products discriminate a triple's **structure** (order +
grouping) better than a real-vector encoding of the same 16 components — including
a **strong** structural baseline, HRR/VSA (circular convolution + role vectors)?

```bash
cd scirust-hypermemory
HYPERMEMORY_GIT_COMMIT=$(git rev-parse HEAD) \
  cargo +nightly-2026-07-02 run --release --bin hypermemory-relations
```

Finding (deterministic): vs **naive** real baselines the sedenion product wins
outright — it recovers structure from a noisy query at **~99.9%** (noise 0.1,
chance ≈ 8.3%) while `Sum`/`Hadamard` sit at chance and position-weighting caps
at ~50%. But vs **HRR** — a purpose-built structural encoding — it does **not**
win: HRR matches it at low noise and is **more robust under heavy noise**
(noise 0.5: **HRR 0.946 vs Sedenion 0.869**), without the zero-divisor collapse
risk (F2). So the algebra's structural capacity is real but **not superior** to
established methods. Full results:
[`docs/research/SCIRUST_HYPERMEMORY_RELATION_DISCRIMINATION.md`](../docs/research/SCIRUST_HYPERMEMORY_RELATION_DISCRIMINATION.md).

## Non-goals (Phase 1)

HNSW / ANN, dynamic PCA, KD-trees, distributed storage, lock-free concurrency,
GPU kernels, mmap persistence, autonomous agents, NL generation, continual
learning, residual *learning* (the residual is stored but frozen), production
cryptography, and reverse-mode differentiation over the store — all deferred.

[`ConceptId`]: crate::ConceptId
[`ConceptRecord`]: crate::ConceptRecord
[`S16Store`]: crate::S16Store
[`S16BoundedMemory`]: crate::S16BoundedMemory
[`S16ExactIndex`]: crate::S16ExactIndex
[`S16Expr`]: crate::S16Expr
[`S16Relation`]: crate::S16Relation
[`ProductDiagnostics`]: crate::ProductDiagnostics
[`RetentionPolicy`]: crate::RetentionPolicy
[`NoForgetting`]: crate::NoForgetting
[`LinearDecay`]: crate::LinearDecay
[`Real16Index`]: crate::Real16Index
[`scirust-simd`]: https://github.com/Memorithm/scirust
[`scirust-retrieval`]: https://github.com/Memorithm/scirust
[`scirust_retrieval::DenseIndex`]: https://github.com/Memorithm/scirust
