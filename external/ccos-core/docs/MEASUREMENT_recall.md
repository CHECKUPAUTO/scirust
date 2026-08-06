# Measuring the new recall strategies (hybrid fusion, LSA) — honest findings

> Reproduce: `cargo run --release --example recall_eval` (default INT4 TF-IDF) and
> `cargo run --release --example recall_eval --features learned-embed` (LSA).

This is an **LLM-free** measurement: a synthetic corpus (~60 small files) with
**ground-truth** relevant files, three task types that each isolate where a signal
*should* help, and a hit-rate metric (did the target file land in the recalled
window at a tight 160-token budget?). The point is to check whether the strategies
added this session help **in measurement**, not just in micro unit-tests — and to
report it whether or not it flatters the features.

The three task types:
- **plain** — the query carries the target's own distinctive term (lexical should suffice).
- **decoy+fail** — the query is only *common* words a decoy file also has, so lexical
  points at a decoy; the target is the active **failing** file (isolates the causal vote).
- **synonym** — the query uses a synonym the target file never literally contains
  (isolates the latent/LSA benefit).

## Results

Default — **INT4 TF-IDF**:

| strategy    | plain | decoy+fail | synonym | overall |
|-------------|------:|-----------:|--------:|--------:|
| working_set |  50%  |    100%    |   50%   |   67%   |
| lexical     |  50%  |     0%     |    0%   |   17%   |
| semantic    |  50%  |     0%     |   12%   |   21%   |
| **hybrid**  | 100%  |    62%     |   12%   | **58%** |

Opt-in — **LSA (`learned-embed`)**:

| strategy    | plain | decoy+fail | synonym | overall |
|-------------|------:|-----------:|--------:|--------:|
| working_set |  50%  |    100%    |   50%   |   67%   |
| lexical     |  50%  |     0%     |    0%   |   17%   |
| semantic    |  62%  |     0%     |    0%   |   21%   |
| **hybrid**  | 100%  |    12%     |    0%   | **38%** |

## What this says (the robust signal, not the noise)

1. **Hybrid fusion (slice A) is measurably the best query strategy.** On the default
   embedder it leads every other query strategy overall (58% vs lexical 17%, semantic
   21%) and is the only one that recovers the target in the **decoy+fail** case (62%
   where lexical scores 0%) — the sparse causal vote pulling the active failing file
   into the window. This validates the slice-A design *in measurement*, not just in a
   unit test.

2. **LSA (slice B) does *not* help here — and appears to hurt.** Turning on the
   learned embedder drops hybrid overall from 58% to **38%** (decoy+fail 62%→12%,
   synonym 12%→0%). LSA works in the controlled micro-test (`lsa.rs`), but its
   classic strength is *dense ranking*; CCOS uses the semantic signal only to pick a
   single **entry** node before region expansion, and at corpus scale the rank-48
   truncation over ~60 docs adds enough noise to that one pick that it pollutes the
   RRF fusion. **Honest disposition:** LSA correctly stays **opt-in and off by
   default** — the data says "don't turn it on for this recall architecture yet."
   A genuine win would likely need LSA wired into a *ranking* stage (slice A's fusion
   over more candidates), not entry selection — a future experiment, not a current
   claim.

3. **Lexical is a strong, narrow baseline** — exactly the paper's honest stance. Its
   17% overall here is *by construction*: two of the three task types (decoy, synonym)
   are designed to defeat literal matching, so lexical can only win the plain third.
   This is not "lexical is bad," it's "this benchmark stresses the non-lexical signals."

## Honest limitations of this benchmark

- **Synthetic and small** (~60 files, 24 tasks); absolute numbers are
  benchmark-sensitive — read the *ordering* and *direction*, not the exact percentages.
- **`working_set` is a no-query control** — its `decoy+fail` 100% is just the failure
  cue making the target the single hottest node; its 50% elsewhere is id-ordering luck,
  not a meaningful signal.
- **`plain` lexical sits at 50%, not ~100%** — a region-expansion/budget artifact
  (the tight 160-token window plus `use`-chain region sometimes crowds the target out),
  which is why hybrid's anchor-proximity ranking does better there. It means the plain
  column understates lexical; it does not change the headline (hybrid best, LSA hurts).
- The **synonym** column never cleanly isolates LSA, because the glossary bridge doc
  literally contains the synonym and so intercepts the query as a better lexical/semantic
  match than the target — a real difficulty of testing latent links in an
  entry-selection (not ranking) recall. Treat the synonym column as inconclusive for LSA.

## Follow-up: is LSA a better *dense ranker*? (its home turf)

The entry-selection result above is not LSA's natural use — LSA's strength is ranking
*many* candidates, not picking one entry. `examples/embed_ranking.rs` isolates that:
fit TF-IDF and LSA on the same corpus, and for each query with a known target measure
**recall@k** (is the target in the top-k by cosine?). Both embedders in one run; no
feature flag.

**synonym queries** (the target file never contains the query term):

| embedder    | recall@1 | recall@3 | recall@5 | recall@10 |
|-------------|---------:|---------:|---------:|----------:|
| tfidf       |    0%    |    0%    |   10%    |    10%    |
| lsa-rank16  |    0%    |    0%    | **50%**  | **80%**   |
| lsa-rank48  |    0%    |    0%    |   10%    |    10%    |

(plain queries: all embedders ~90–100% at every k — they tie.)

Two clean conclusions:
1. **LSA *is* a better dense ranker for synonymy — but only at a low rank.** rank 16
   recovered synonyms (recall@10 80% vs TF-IDF 10%); **rank 48 gave no benefit** (a
   high rank means almost no truncation, so no latent smoothing). The default
   `build_embeddings` LSA rank was therefore changed **48 → 16**.
2. **LSA never helps the top-1** (recall@1 stays 0% even at rank 16). So it is useless
   for *entry selection* (which needs the single best node) and useful only for
   *ranking* a candidate set (recall@k≥5). This exactly explains why wiring it into
   entry selection hurt: doubly mis-applied (rank too high *and* wrong stage).

**Disposition:** LSA stays opt-in. The capability is now *validated for dense ranking
at low rank*; the honest next step to make it earn the default path is to wire it into
a **re-ranking** stage (rank the region/window candidates) rather than entry selection.
Until then it is correct but mis-placed, and off by default.

## LSA re-ranking, wired and measured (the follow-up)

The data-backed path above is now built: `set_lsa_rerank(Some(rank))` re-orders the
recalled **region** by rank-`rank` LSA similarity to the query — the *ranking* stage,
not entry selection. `examples/lsa_rerank.rs` measures it on a hub-and-spoke corpus
(every topic reachable in the region) for queries that resolve:

- **mean target-file rank: 11.8 → 2.1 (Δ −9.6)** with re-ranking on. The
  semantically-central target is pulled from the middle of the window to near the top —
  the recall@k≥5 win #39 predicted, now realised inside the pipeline.

Honest limiter, also measured: only **3/8 pure-synonym queries resolve to a region at
all** — entry selection (TF-IDF) scores a synonym ≈0, exactly #39's recall@1 finding.
Re-ranking sits at the region stage, so it re-orders what entry selection found and
**never repairs an empty region**. It is deterministic (id-sorted corpus, deterministic
eigensolve — replay stays exact) and opt-in.

## Bottom line

Hybrid fusion earns its place on the default path. The learned LSA embedder is now
*measured*, not assumed: useless for entry selection (recall@1=0), but a genuine
ranking win — wired as a re-ranking stage it pulls the target from rank ~12 to ~2.
It stays **opt-in** (its synonym benefit is gated by entry selection, and it carries a
per-recall fit cost), now with the measurement to back enabling it where that trade
pays.
