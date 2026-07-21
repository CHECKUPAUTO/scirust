# scirust-hypermemory — Conclusions of the falsification program

**Status: program complete and archived.** This page is the one-stop synthesis
so that nobody has to re-read eight pull requests to know what was claimed,
what was measured, and what survived. Full details: the design spec
([SCIRUST_HYPERMEMORY_PHASE1.md](SCIRUST_HYPERMEMORY_PHASE1.md)), the
experiment reports ([F2/F6](SCIRUST_HYPERMEMORY_F2_F6.md),
[relation discrimination](SCIRUST_HYPERMEMORY_RELATION_DISCRIMINATION.md)),
and the harnesses under `scirust-hypermemory/tests/`.

## The question

Does a 16-dimensional hypercomplex (sedenion) representation — with its
non-associative, zero-divisor-bearing product as a *binding operator* — buy an
associative memory anything that ordinary real vectors do not? The program was
built falsification-first: six pre-registered criteria (F1–F6), an exact
oracle before any approximation, deterministic corpora, and a real-vector
baseline (`Real16`) that every sedenion result had to be compared against.

## Verdicts, one line each

| Criterion | Question | Verdict |
|---|---|---|
| F1 | Does sedenion similarity retrieval beat a real 16-vector? | **Confirmed falsified** — retrieval is bit-identical to `Real16` and to `scirust-retrieval::DenseIndex`; the multiplicative structure plays no role in similarity search. |
| F2 | Do zero divisors corrupt stored/derived codes in practice? | **Not triggered generically** — exact zero products require exactly aligned subalgebra pairs (e.g. `(e₁+e₁₀)·(e₄−e₁₅) = 0`); the dangerous regimes (sparse, subalgebra-aligned operands) are mapped, instrumented by `ProductDiagnostics`, and avoidable. |
| F3 | Does non-associativity break explicitly parenthesized relations? | **Not triggered** — `S16Expr` keeps every parenthesization explicit; evaluation is deterministic and auditable, so non-associativity is a *representable fact*, not a silent corruption. |
| F4 | Does residual learning in one record contaminate others? | **Not triggered** — per-record isolation is bit-exact under interleaved learning (clamped residual updates, fixed-order reductions). |
| F5 | Does the system lose determinism anywhere? | **Not triggered** — every harness (117+ tests at archive time) is bit-reproducible: no RNG, no wall clock, no hash-order iteration, `total_cmp` + id tie-breaks everywhere. |
| F6 | Do norm blowup/collapse destabilize long derivation chains? | **Not triggered generically** — norm drift is measured and bounded in the surveyed regimes; the failure regimes (repeated near-zero-divisor products) coincide with F2's map. |

## The structure story (the honest headline)

- Sedenion binding **does** beat naive real encodings for relational
  structure: 99.9% discrimination of order/grouping vs ~50% (chance) for
  sum/Hadamard encodings.
- But against the **strong** baseline — Holographic Reduced Representations
  (circular convolution) at the *same* 16 dimensions — sedenion binding
  **loses**: structure-retrieval agreement 0.869 (sedenion) vs 0.946 (HRR) at
  noise 0.5.
- The gap is **capacity, not magic**: HRR scales with dimension while the
  sedenion product is pinned at 16. Measured after the stable port
  (`scirust-retrieval/tests/vsa_hrr.rs`): noiseless 3-pair recovery
  0.561 (dim 16) → 0.994 (dim 64) → 1.000 (dim 256).

**Conclusion: for VSA-style structure encoding, use HRR at a dimension sized
to the task. The sedenion's algebraic exotica (non-associativity, zero
divisors) provided auditability challenges worth instrumenting, not
representational advantages worth paying for.**

## What denoising added (and what it didn't)

- A **cleanup memory** (nearest-prototype snap, explicit acceptance
  threshold, idempotent by construction) recovers the structure pipeline from
  0.29 to 0.94 retrieval agreement at noise 0.5 — and rejects uncorrelated
  input (200/200) instead of hallucinating a match.
- **Observation fusion** on `scirust-signal`'s noise toolkit: repeated noisy
  observations of one code, denoised per lane. Broadband regime (dim 16):
  single-shot 0.187 → Kalman-RTS 1.000. Impulsive regime: naive mean 0.047 →
  Hampel+Kalman 0.807 (~17×). Honest notes kept: under pure zero-mean
  broadband noise the mean is already near-optimal; Hampel has a breakdown
  ceiling under spike clusters.
- Denoising **rescues robustness; it does not change the verdicts above.**

## What was extracted to stable (the lasting artifacts)

The winners need no nightly `portable_simd` — they are ordinary fixed-order
`f32` arithmetic — so they were ported into `scirust-retrieval`'s pure core at
**arbitrary dimension**:

- `vsa::{circular_convolution, circular_correlation, involution, superpose}` —
  HRR binding/unbinding;
- `vsa::CleanupMemory` — thresholded nearest-prototype cleanup (exact stored
  prototype, ascending-id tie-breaks, typed errors);
- `vsa::fuse_observations` + `FusionStrategy` (feature `fusion`) — the
  `scirust-signal` Kalman-RTS / Hampel observation-fusion front-end;
- `IvfIndex` — the deterministic IVF pattern generalized from Phase 4:
  RNG-free Lloyd (seeded from insertion order, fixed iterations, lowest-index
  tie-breaks), recall *measured* against the `DenseIndex` oracle, `nprobe =
  nlist` **bit-identical** to it, recall monotone in `nprobe` (measured at
  dim 32, 1000 docs, nlist 16: 0.250 / 0.408 / 0.600 / 0.833 / 1.000 for
  nprobe 1/2/4/8/16).

The `scirust-hypermemory` crate itself remains in the repository as the
research record — nightly-only, excluded from the main workspace, covered by
its dedicated CI job — but is **not** the recommended dependency for new work.
Use the `scirust-retrieval` ports.

## Methodology worth keeping

1. **Oracle first.** No approximate structure was trusted before an exact,
   audited reference existed to measure it against.
2. **Falsification pre-registered.** The failure criteria were written before
   the implementation, and several were *confirmed* — the program treated
   "the fancy representation does not help" as a publishable outcome, and
   that is what F1 returned.
3. **Determinism as a contract.** Bit-reproducibility everywhere made every
   measured number in this document checkable by `cargo test`.
4. **Honest bars.** Test assertions encode measured behaviour (including
   ceilings and degradations), never aspirations.

## Record

Eight merged pull requests: #640 (Phase 1: store, oracle index, expressions,
diagnostics), #645 (F2/F6 experiments), #648 (relational discrimination),
#654 (HRR strong baseline), #655 (Phase 2: bounded memory), #660 (Phase 3:
residual learning, F4), #661 (Phase 4: deterministic IVF), #667 (denoising:
cleanup memory + `scirust-signal` observation fusion) — plus the stable port
that closes the program.
