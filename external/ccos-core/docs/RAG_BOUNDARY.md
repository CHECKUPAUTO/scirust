# CCOS Core — RAG Boundary

RAG is supported as a **complementary provider**. It is not the centre of the
architecture and never the source of truth (§10).

## RetrievalProvider abstraction

```
RetrievalProvider
├── vector search      (embeddings / TF-IDF / LSA / octa_index)
├── keyword search     (hashing tokenizer, inverted indexes)
├── graph search       (MemoryGraph traversal — internal)
├── database query     (external systems)
└── external document systems (RAG pipelines, ccos-migrate ingestion)
```

Implementations in Core: `retrieval.rs`, `embeddings.rs`, `lsa.rs`,
`hashing_tokenizer.rs`, `neural_embed.rs` (quarantined, feature-gated),
`octa_index.rs` (optional, tag-pinned), `cold_index.rs`, `compressor.rs`.

## What retrieval may do

retrieve documents · search passages · provide external evidence · enrich
temporary context.

## What retrieval may never be

event memory · episodic memory · cognitive state · belief revision ·
contradiction resolution · temporal validity · consequence-based learning ·
invalidation.

## The structural difference

A retriever ranks by relatedness; relatedness has no polarity and no time.
CCOS keeps polarity (`Supports`/`Contradicts`), provenance, sequence and
derived belief as **structure** — so it can answer "which claim is currently
accepted and why", which no similarity ranking can (§12).
The working hypothesis (§6) is evaluated head-to-head in
`examples/*_crux.rs` and the cognitive benchmark (§32–§35).
