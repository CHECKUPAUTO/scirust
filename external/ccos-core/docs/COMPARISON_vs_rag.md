# CCOS vs RAG — a different category, measured

> All numbers below are reproduced by the runnable `examples/*_crux.rs` cited inline — measurement-first,
> not marketing.

Every flavour of RAG — naïve, hybrid, re-ranked, GraphRAG, agentic — is **stateless retrieval by
similarity**: chunk the corpus, embed it into a vector space, return the top-k nearest a query. CCOS is
a **deterministic, replayable, causal *working memory* that holds beliefs**. It is not "a better RAG";
it is the layer a RAG stack structurally lacks. This note compares them honestly, including where RAG
wins.

## The matrix

| Dimension | Naïve RAG (chunk+dense) | Hybrid RAG + reranker | GraphRAG (MS) | **CCOS** |
|---|---|---|---|---|
| Unit of retrieval | 512-token chunk | chunk + BM25 | chunk + entity graph | **typed causal node** |
| Similarity basis | neural dense | dense + lexical + cross-encoder | graph + LLM summaries | TF-IDF·INT4 + **incremental causal LSA** |
| Structure | ❌ flat | ❌ flat | entities / communities | **causal graph** (imports · calls · data-flow · Causes) |
| **Contradictions** | ❌ none | ❌ none | fuzzy aggregation | ✅ **Q-Page: support/contra → belief + tension, decay, propagation** |
| Provenance / audit | weak | weak | medium | ✅ **hash-chain + `replay == live` byte-exact** |
| Determinism | ❌ (model drift) | ❌ | ❌ (LLM) | ✅ **bit-exact, no RNG** |
| Time-travel / what-if | ❌ | ❌ | ❌ | ✅ **`replay_to`, `recall_what_if`, `retrieval_reward`** |
| Temporal dynamics | ❌ | ❌ | ❌ | ✅ **decay (half-life) + temporal tensor (the "fever curve")** |
| Ingestion cost | embed ~ms/chunk + ANN index | + reranker | **LLM summaries = expensive** | **O(N) graph (~2139 files/s) + O(batch) incremental LSA** |
| Dependencies / offline | model + vector DB | + reranker | + LLM | ✅ **zero-extra-dep, offline, air-gappable** |
| *Pure* semantic recall | ✅ strong (transformer) | ✅✅ strongest | ✅ strong | ⚠️ **medium (TF-IDF/LSA, no transformer)** |

## The measured evidence (this is the point)

- **Contradiction-awareness** — `examples/scirust_vs_rag_crux.rs`. On a *Conflict of Origins*, a blind
  512-chunk RAG ranks the **refuted** source **#1** (precision@1 = 1/2); CCOS, gating the latent score
  by causal belief, crushes the contradiction to #5/#7 and holds the authoritative source at #1
  (**2/2**). A RAG has **no belief axis** — it structurally cannot make this distinction.
- **Structure** — `examples/rag_crux.rs`. A lexical RAG recovers only **~50 %** of the real cross-file
  dependencies (recall@10); the causal graph recovers ~100 % by construction. Import edges cross
  vocabulary boundaries that similarity cannot see.
- **Ingestion** — `examples/ingest_profile.rs` + `examples/scirust_vs_rag_crux.rs`. Batch resolution is
  **~174×** faster (O(N²)→O(N)); the incremental LSA folds a batch in **O(batch)** (~**5.5×** at 600
  docs and growing) and is **bit-exact** so `replay == live` holds. Time-travel reconstruction is
  **47.5×** (`examples/replay_batch_crux.rs`).
- **Determinism / audit** — `tests/replay_equivalence_property.rs` proves `replay == live`
  **byte-for-byte** over random op streams; the event + dist logs are hash-chained and verified on
  reload; every checkpoint is fsync-durable.
- **Temporal dynamics** — `examples/temporal_tensor_crux.rs`. CCOS records `Θ[node, {Belief, Tension}, t]`
  — the belief/tension trajectory as a contradiction is injected, propagates, and decays (a system
  "fever curve"). No RAG has a time axis at all.

## Where RAG wins (no overselling)

On **pure semantic recall at web scale, a well-tuned neural dense RAG retrieves better** than CCOS's
TF-IDF/LSA embedder — exactly what `rag_crux` honestly showed (the lexical signal is real but
incomplete). CCOS makes a deliberate trade: a transformer embedder would break `replay == live` (weights
are not bit-stable across builds), so CCOS keeps a **deterministic, dependency-free** semantic floor and
invests the differentiation in *structure, belief, time, and auditability*. CCOS also targets
**agent working memory and code**, not general-purpose document QA at billions of chunks.

## The verdict

A RAG answers *"which documents **resemble** my query?"*. CCOS answers *"what should I **believe**, where
did it **come from**, and how has my understanding **changed over time?**"* — **replayably and
auditably**. For an agent that must not be deceived by a contradiction, must replay and audit its own
reasoning, and must stay deterministic and offline, **CCOS wins structurally**. For raw semantic
document QA, a tuned dense RAG retrieves more. The two are **complementary**: CCOS is the causal
working-memory layer that sits above — or in place of — flat retrieval.

**Bottom line:** CCOS is not competing on "nearest-neighbour recall." It is a deterministic, auditable,
belief-bearing causal memory — and on the axes that a RAG cannot even represent (contradiction, time,
provenance, replay), the comparison is not close.
