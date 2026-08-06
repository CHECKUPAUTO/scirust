# SciRust fusion — distilled linear algebra for linear ingestion + contradiction-aware retrieval

> Reproduce: `cargo run --release --example scirust_vs_rag_crux`

The goal of slice #14 was to couple CCOS's `learned-embed` layer with SciRust's linear algebra to get
**a net improvement in document ingestion** and a retrieval edge over classic RAG. After inspecting the
SciRust repo, the verdict was **distill, don't link** — and the measurements below show the distilled
fusion delivers on both axes, honestly.

## Why distill (the dep-vs-link decision)

SciRust (`CHECKUPAUTO/scirust`, a ~50-crate workspace) is, by its own description, a *"pure-Rust
**deterministic** deep-learning & scientific-computing platform"*. Its relevant pieces:

- `scirust-core/src/tn/ops/svd.rs` — a truncated SVD that is **a thin wrapper over `nalgebra::SVD`**
  (full SVD then truncate), with **no incremental update** (its own comment marks randomized/incremental
  as a future "Phase 2").
- `scirust-solvers/src/linalg/` — a clean, self-contained, deterministic `f64` `Matrix` with hand-rolled
  QR / LU / Cholesky / CG.

Linking it was rejected for two hard reasons:

1. **The thing we wanted doesn't exist there yet.** The *incremental rank-K update* is a SciRust TODO —
   we'd implement it regardless.
2. **Depending on the SVD means depending on `scirust-core`**, which pulls **`rayon` (default!)**,
   `nalgebra`, `ndarray`, `blas`, `matrixmultiply`, autodiff/simd/gpu — a heavy tree whose parallel
   float reductions are **non-deterministic**, which would break CCOS's sacred `replay == live`
   (bit-exact). CCOS's own `src/embeddings.rs` already warns against importing that stack.

So SciRust is used as an **algorithmic reference + correctness oracle**, never a runtime dependency. The
distilled implementation lives in `src/lsa.rs`, stays zero-extra-dependency and deterministic, and the
SciRust repo is never modified.

## The key insight: the Gram matrix is already incremental

CCOS's LSA factors the latent space through the Gram matrix `C = MᵀM` (`dim × dim`, fixed size): the
top-`rank` eigenvectors of `C` are the right singular vectors of the document–term matrix `M`. And `C`
is a **sum of per-document outer products**:

```
C = Σ_d (w_d · row_d) (w_d · row_d)ᵀ
```

Two consequences fall straight out, with no Brand-style incremental-SVD machinery:

- **Incremental ingestion.** A new batch just *adds* its outer products to the running `C` — O(batch),
  independent of the corpus already indexed. (`IncrementalLsa::update`.)
- **Causal weighting.** Scaling each document row by its **authority** `w_d` before the outer product
  shapes the latent space by *causal importance* (Q-Page belief × eigencentrality), not raw term
  frequency. (`weighted_lsa_projection`.)

Both are **deterministic and bit-exact** (the Gram is an order-fixed `f64` sum), so `replay == live`
holds: folding the same documents in the same order — one batch or many — yields the identical Gram and
projection. Proven by `lsa::tests::incremental_lsa_is_bit_exact_with_a_single_batch`.

## A. Ingestion — incremental fold vs full recompute

Both paths use the same Gram fold; the only difference is *incremental* keeps one running model and
folds each new batch, while *full recompute* rebuilds the Gram from every document seen so far on each
batch (what a naive "refit the LSA per batch" ingestion does):

```
  docs   incremental(ms)   full-recompute(ms)   speedup
   150            0.18               0.34      1.9x
   300            0.26               0.93      3.6x
   600            0.63               3.46      5.5x
```

Incremental is **~O(N)** (each batch folds only its own docs); full recompute is **~O(N²)** (each batch
re-folds the whole corpus). The on-demand projection (a constant Jacobi sweep on the fixed `128×128`
Gram) is identical for both, so the gap is pure ingestion cost — and the speedup **grows with N**, as
the O(N²)→O(N) shape predicts. This is the net ingestion improvement the fusion was for.

## B. Retrieval — contradiction-aware ranking (a "Conflict of Origins")

One authoritative source and one **refuted contradiction** make opposite claims about the same topic,
amid distractors that share vocabulary. Rank of each under three rankers (lower = better, #1 = top):

```
                              blind RAG      weighted-LSA      CCOS full (×belief)
  query                       auth contra    auth contra       auth contra
  q1 (…recommended timeout)   #2   #1        #2   #1           #1   #5
  q2 (…pool timeout setting)  #1   #4        #1   #4           #1   #7

  precision@1 (authoritative first): blind RAG 1/2   weighted-LSA 1/2   CCOS full 2/2
```

**Honest reading** (this is the measurement-first point):

- **Blind RAG** (raw TF-IDF cosine, every 512-token chunk equal) has **no belief axis**: the refuted
  contradiction shares the query's vocabulary, so it scores like — and on q1 *outranks* (#1) — the
  authoritative source. 1/2.
- **Weighting the matrix *before reduction* alone is necessary but NOT sufficient.** The weighted-LSA
  space ties RAG (1/2): cosine is a *direction*, and authority reshapes *variance*, not direction, so a
  lexically-aligned contradiction keeps a high cosine. (A real negative result — reported, not hidden.)
- **The contradiction-awareness comes from gating the score by belief at retrieval** — `CCOS full =
  latent cosine × authority`. The refuted origin (authority `0.12`) is crushed to the bottom (#5, #7)
  while the authoritative one (`0.95`) holds #1. **2/2**, and the contradiction is suppressed by a wide
  margin. A blind RAG structurally cannot do this — it has no notion of what the system *believes*.

So the fusion is two-sided: **SciRust-distilled latent algebra (semantic) × CCOS causal belief
(trust)**. The semantic latent space finds the topically-relevant documents; the causal belief axis
decides which *origin* to trust when they conflict.

## C. Live wiring (#14b) — the causally-weighted latent space in the engine

#14a proved the primitives; #14b threads them into the live `CcosMemory` semantic-recall path. Each
node's document row is now scaled by a **causal weight computed from the live graph**

```
w_node = (1 + λc · centrality_norm) · (1 + λa · authority)
```

*before* the LSA reduction, where `centrality_norm` is `spectral::eigenvector_centrality` (max-normalised
to `[0,1]`, so the weight is invariant to graph size) and `authority` is the node's Q-Page net belief
`qbeliefs()[node].belief.max(0)` — only genuine net support *amplifies* a document; a refuted node gets no
boost (the retrieval-time gate is what *suppresses* it). The recall path (`lsa_region_scores`) reads this
weighted projection, **version-cached**, instead of recomputing a uniform LSA on every query.

### The honest design choice: re-fold, not stream

There is a real tension here, and #14b resolves it on the side of the moat. A causally-weighted Gram with
**global** weights cannot be *both* ingest-order-incremental *and* bit-exact-rebuildable from a snapshot:
adding one document changes the eigencentrality — hence the weight, hence the outer product — of **every**
other document, and an `f64` sum is only bit-identical when re-added in the **same order**. A snapshot
stores the graph, not the ingest history, so the only order it can re-fold in is the canonical (id-sorted)
one.

So the live recall projection is a **pure function of the current graph**: rebuilt by re-folding all nodes
in id-sorted order with their *final* weights, cached by graph version. That buys three moat properties,
each a test in `external_memory::tests`:

- **`live == reload`** — the reloaded projection is bit-identical to the live one, so an audit replay's
  recall ranking can never diverge (`weighted_lsa_model_survives_a_reload`).
- **eager ≡ batch** — the projection is identical however the corpus was ingested
  (`weighted_lsa_model_is_order_independent`), preserving the B2 order-independence invariant.
- **causal monotonicity** — asserting evidence for a node raises its pull on the latent space
  (`causal_weights_are_deterministic_and_rise_with_evidence`).

The `O(batch)` **as-of-ingest** `IncrementalLsa` fold (Part A) keeps its place as the **append-only
streaming** primitive — correct when you accept that the latent space reflects each document *as it
arrived* (replay-exact by ingest order, but not snapshot-rebuildable). Live recall trades that streaming
speed for bit-exact reload determinism; both share the one Gram code path (`lsa::accumulate`), so they can
never silently drift apart.

## Verdict

- ✅ **Ingestion**: the distilled incremental Gram fold is linear and ~5.5× faster at 600 docs (growing)
  vs full recompute, **bit-exact** so `replay == live` holds (#14a).
- ✅ **Retrieval**: the full fusion is contradiction-aware (2/2 vs RAG 1/2) where blind RAG cannot be —
  with the honest caveat that the *retrieval-time* belief gate, not the pre-reduction weighting alone,
  is what does it.
- ✅ **Live wiring (#14b)**: the recall latent space is now causally weighted (centrality × belief) from
  the live graph, version-cached (an `O(1)` hit between mutations, replacing a full per-query recompute),
  and proven `live == reload` / eager ≡ batch / deterministic. The `learned-embed` rerank earns its keep
  on ranking (recall@k); the causal weighting shapes *which* documents that ranking trusts.

**Bottom line:** measure first. The Gram matrix was already incremental — distilling that (rather than
linking SciRust's heavyweight, rayon-parallel SVD) bought linear ingestion *and* kept the determinism
moat intact. #14b lands the causal weighting in the live engine on the moat-preserving side of the
incremental-vs-rebuildable tension, and the causal-belief gate is what actually beats RAG on
contradictions.
