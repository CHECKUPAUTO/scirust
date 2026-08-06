# Pure retrieval challenges RAG — distilled SciRust retrieval, measured

> Reproduce: `cargo run --release --example pure_retrieval_vs_rag`

The goal: take SciRust's *pure* semantic-retrieval platform (`scirust-retrieval`) and use it to
challenge CCOS's existing lexical RAG on relevance — deterministically, pure-Rust, zero FFI — by
implementing its `Encoder` over the embeddings CCOS already owns and **measuring**.

## Why distilled, not linked (the moat decision)

The mission's first instinct was to add `scirust-retrieval` as a `path` dependency. Inspecting the
crate stops that cold: **`scirust-retrieval` depends on `scirust-core`**, whose `Cargo.toml` has
`default = ["rayon"]` and unconditionally pulls `nalgebra` + `ndarray` + `matrixmultiply`. Linking it
would drag `rayon`'s **non-deterministic parallel `f32` reduction order** into CCOS's build — breaking
the sacred `replay == live` bit-exactness *and* the mission's own *"accumulation f32 à ordre fixe, bit
pour bit"* — and would forfeit CCOS's zero-extra-dependency, air-gappable identity. It is the exact
trap the #14 fusion avoided by **distilling** SciRust rather than linking it (and Cargo unifies
features additively, so a `default-features = false` on CCOS's side cannot remove rayon without
editing the scirust repo — which is forbidden).

The retrieval *algorithms*, however, are pure: `index.rs`, `hybrid.rs`, `metrics.rs`, `vector.rs`,
`rerank.rs`, `feedback.rs` reference **no** `scirust_core` and **no** `rayon`. So CCOS reimplements
them in `src/retrieval.rs` — every reduction left-to-right in a single `f32`, every ranking sorted by
score then by an ascending-id tie-break — over CCOS's `TfidfEmbedder`. The oracle tests carry the
**hand-derived** values from SciRust's own `scirust-retrieval` test vectors (e.g. BM25 `cat`/`cat cat`
→ `0.250692` / `0.182322`; nDCG `0.9433884`), so the port is verified, not merely mimicked.

## The components (`ccos::retrieval`, zero new deps)

- **`vector`** — `dot` / `norm` / `cosine` / `normalized`, fixed-order `f32`.
- **`DenseIndex`** — exact brute-force top-k cosine over L2-normalised vectors; deterministic.
- **`Bm25Index`** — classic BM25 (`k1=1.2`, `b=0.75`), IDF-weighted, length-normalised.
- **`reciprocal_rank_fusion`** — fuse incomparable scorers by rank, no calibration.
- **`SemanticRetriever`** (dense) and **`HybridRetriever`** (dense ⊕ BM25 via RRF).
- **`metrics`** — Recall@k, Precision@k, MRR, MAP, nDCG@k.
- **`CcosEncoder`** — the bridge: implements `Encoder` over CCOS's corpus-fitted TF-IDF embedder.

## The measurement (real output of the run)

Eval set = CCOS's own `src/*.rs` (the same corpus + ground truth as `rag_crux`): for each file `A`
with cross-file dependencies, the **query** is `A`'s text and the **relevant** docs are `A`'s true
dependency files (the file→file edges the AST resolved). Three retrievers, scored side by side:

```
files: 61   queries (files with deps): 39   relevant pairs: 141

  metric (%)              Recall@1/5/10 |  Prec@1/5/10  |  nDCG@1/5/10  |   MRR   MAP
  ----------------------------------------------------------------------------------
  ccos RAG (lexical)         24  52  66 |   49  25  18  |   49  50  56  | 0.626 0.491
  pure dense (distilled)     24  52  66 |   49  25  18  |   49  50  56  | 0.626 0.491
  pure hybrid (dense+BM25)   22  52  63 |   46  26  17  |   46  49  54  | 0.615 0.467
```

(Absolute figures are measured on *this revision's* `src/` tree, so they drift as the codebase grows —
re-run the example for the current numbers. What does **not** drift: pure dense equals ccos's RAG to
the digit, and every run is bit-for-bit identical.)

## The honest verdict

- **Pure dense reproduces ccos's RAG bit-for-bit.** It is an exact-cosine index over the *same* TF-IDF
  embedding ccos's lexical RAG uses, so every metric matches to the digit (24/52/66, MRR 0.626). That
  is the point of a faithful distillation: the distilled retriever *is* ccos's retriever — but now as a
  clean, serialisable, auditable `DenseIndex` rather than an ad-hoc cosine loop.
- **Hybrid trades slightly on this task.** Fusing BM25's exact-term/IDF signal *lowers* the scores
  here (Recall@1 24→22, MRR 0.626→0.615). Honest negative result: on *file-dependency* retrieval the
  TF-IDF cosine already captures the lexical overlap, and BM25's rare-term emphasis dilutes rather than
  sharpens it. Hybrid is the right tool for keyword-precise corpora; this structural task is not one.
- **The decisive, un-rowable win is determinism.** Every number above is reproducible **bit for bit**
  (fixed-order `f32`, id-tie-broken ranking, zero RNG, zero generative step). Re-run → identical; there
  is no hallucinating generator between query and result, and the index serialises and audits. A neural
  / generative RAG stage offers none of that.

## Beating RAG on its own turf — semantic recall (the LSA encoder)

The result above is honest but unflattering: pure dense over TF-IDF *equals* lexical RAG because it is
the same lexical signal. The win is the **encoder**. `LsaEncoder` projects the TF-IDF vector through
CCOS's deterministic **LSA latent space** (`crate::lsa`, a fixed-order Jacobi solve on the corpus Gram
matrix), so two documents that never share a word still encode to nearby vectors when their terms
co-occur with a common third — the **synonymy** a literal-term retriever structurally cannot represent.

`examples/semantic_retrieval_crux.rs` makes the gap concrete: a corpus where each query and its answer
share **zero vocabulary** (the query says "car vehicle drive", the answer "automobile motor sedan"),
linked only by *bridge* documents where both vocabularies co-occur. Real output of the run:

```
  retriever                Recall@1  @3    @5     MRR
  lexical RAG (TF-IDF)         0%      17%   50%    0.185
  LSA semantic (dense)        17%      83%  100%    0.458
  LSA semantic (hybrid)        0%      83%  100%    0.319
```

The lexical RAG, needing a literal match, **cannot retrieve the answer** (it ranks the bridge /
distractor docs instead); the LSA encoder learns "car ≈ automobile" from the bridges' co-occurrence and
recovers it — **Recall@3 17% → 83%, MRR 2.5×**. This is RAG's *own* turf — semantic recall — and a
deterministic, dependency-free LSA wins it, bit-for-bit reproducibly (a transformer embedder would too,
but could not be replayed bit-exact). (Honest sub-reading: the *hybrid* row trades Recall@1 because its
BM25 half is lexical and dead weight when query and answer share no term — fuse BM25 only when there is
a lexical signal to fuse.)

**So the full picture:** pure dense over TF-IDF **ties** ccos's lexical RAG; pure dense over LSA
**beats** it on semantic recall — same deterministic, zero-dependency machinery, the encoder chooses
the axis.

## Adaptive retrieval — the improvement loop (premium tier)

The retrieval **core** above (dense / BM25 / hybrid + metrics) is free and fully functional, exactly
like the rest of CCOS's core. The **premium** tier is a self-improving feedback loop
(`retrieval::feedback::ImprovementLoop`): record confirmed `(query, relevant-doc)` pairs, then learn a
linear projection of the embedding space by deterministic contrastive (InfoNCE) training so projected
retrieval improves. It distills `scirust-retrieval`'s `contrastive` + `feedback` modules — which use
`scirust-core`'s autodiff — reimplemented with a **seeded** RNG, **fixed-order `f32`**, and a
**hand-derived analytic gradient** (gradient-checked against finite differences, so the math is
verified, not trusted): no `scirust-core`, no rayon.

`examples/retrieval_improvement.rs` builds a deliberate **vocabulary gap** — `n` (query, doc) pairs with
*disjoint* terms, so a query shares zero vocabulary with its answer and base retrieval is at chance —
then watches Recall climb as feedback accumulates (real output of the run):

```
  cycle   Recall@1   Recall@3   (n=12, 2 epochs/cycle, seeded, deterministic)
    0         8%        17%      base (random projection)
    1        17%        42%
    2        33%        67%
    3        75%        83%
    4        83%        92%
    5        92%        92%
    7        92%       100%
    8       100%       100%
```

The loop learns the cross-vocabulary mapping purely from feedback (no shared terms to lean on), and —
seeded, fixed-order, hand-derived-gradient — the curve is **bit-for-bit identical** on every re-run.

**Premium gate.** `RetrievalAccess::unlock(&licensing, now)` gates the loop behind CCOS's *own* #29
ed25519 license (`Feature::AdaptiveRetrieval`): the community tier gets the standard no-silent-downgrade
refusal (the free core keeps working), a valid Pro license unlocks it. This reuses CCOS's offline,
deterministic, no-FFI license rather than linking `scirust-license` — one fewer dependency. (A
node-locked `$1/machine/month` model would come from the clean `scirust-license` crate, which *is*
safely linkable — `serde`/`sha2` only — if that commercial scheme is wanted.)

## The headline

**Pure retrieval ties ccos's RAG on lexical relevance, *beats* it on semantic recall (the `LsaEncoder`),
wins decisively on reproducibility, and — premium — improves itself from feedback** — all
deterministically, bit-for-bit replayable, with **zero new dependencies**, and SciRust never modified.
The encoder chooses the axis (TF-IDF for lexical, LSA for semantic); the index, fusion, metrics, and
improvement loop are the same auditable, dependency-free machinery underneath.
