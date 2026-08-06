# Migrating from RAG to CCOS

A no-loss, automatable path from **any** retrieval-augmented-generation stack —
naïve dense, hybrid, re-ranked, GraphRAG, or agentic — into a CCOS causal
working-memory. You keep everything the RAG store held (text, metadata,
provenance, relationships, embeddings) and **gain** the structure a flat
retriever throws away: a causal graph, a hash-chained audit trail, deterministic
replay, and self-bounding recall.

> This is the *how*. For *why* — the axes a RAG cannot represent
> (contradiction, time, provenance, replay) — see
> [`COMPARISON_vs_rag.md`](COMPARISON_vs_rag.md).

- [The one-minute version](#the-one-minute-version)
- [The pipeline](#the-pipeline)
- [What "lossless" means](#what-lossless-means)
- [Every kind of RAG, and how it maps](#every-kind-of-rag-and-how-it-maps)
- [The bundle format (CMB v1)](#the-bundle-format-cmb-v1)
- [How structure is rebuilt](#how-structure-is-rebuilt)
- [`ccos-migrate` reference](#ccos-migrate-reference)
- [`rag2ccos` reference](#rag2ccos-reference)
- [Recall after migration](#recall-after-migration)
- [Safety, verification & rollback](#safety-verification--rollback)
- [Worked examples](#worked-examples)

## The one-minute version

Two steps. **Extract** your store into a canonical bundle, then **import** it:

```bash
# 1. normalise any RAG store → a CCOS Migration Bundle (*.cmb.jsonl)
python tools/rag2ccos/rag2ccos.py chroma \
    --in ./chroma_db --collection knowledge --out corpus.cmb.jsonl

# 2. import it into a causal workspace, losslessly and verified
ccos-migrate --bundle corpus.cmb.jsonl --path workspace.ccos --report report.json
```

That is the whole migration. `workspace.ccos` is now a live CCOS memory you can
drive over the [external-memory](MEMORY_INTERFACE.md) CLI or the **MCP** server
(`ccos mcp workspace.ccos`) — point Claude Code / Desktop at it and your agent
recalls from causal memory instead of a vector store.

## The pipeline

```
   your RAG store                canonical bundle              causal memory
 ┌───────────────┐   rag2ccos   ┌────────────────┐  ccos migrate  ┌───────────────┐
 │ Chroma /FAISS │ ───────────▶ │  corpus.cmb.   │ ────────────▶  │ workspace.ccos│
 │ Qdrant/pgvec  │  (extract)   │    jsonl       │   (import)     │  + sidecars   │
 │ GraphRAG /LC  │              └────────────────┘                └───────────────┘
 └───────────────┘                    ▲   the pivot: one lossless format,
                                      │   one importer, N thin adapters
```

The split is deliberate. RAG stores live in the Python ecosystem, so **extraction**
is a small Python tool (`rag2ccos`) with one adapter per store. **Import** is
native Rust inside CCOS (`ccos-migrate`), so it builds the real causal graph and
produces a first-class `workspace.ccos` — no bridge process, no new runtime
dependency. Anything that can write the bundle format can migrate; you can also
hand-author a bundle or emit it straight from your own ETL.

## What "lossless" means

A RAG item carries up to five things. The migration preserves **every one**:

| From the RAG store | Preserved as | Verified? |
|---|---|---|
| **text** (the chunk / document) | the node `content`, retained verbatim (after Unicode de-obfuscation) | ✅ SHA-256 re-checked per node |
| **metadata** (arbitrary key/values) | a line in the **manifest** sidecar `<workspace>.migration.jsonl`, keyed by node id | ✅ round-trips as JSON |
| **provenance** (source path / URI) | the node label + manifest | ✅ |
| **relationships** (GraphRAG edges, parent links, order) | typed causal **edges** in the graph | ✅ counted in the report |
| **embedding** (the dense vector) | a line in the **embeddings** sidecar `<workspace>.embeddings.jsonl` | ✅ dim recorded |

Nothing is dropped: the graph gets the text + structure, and the two sidecars
hold the exact remainder (metadata + vectors) that a causal node does not itself
store. The importer then **re-reads** every node and checks its content hash
against the source record; a single mismatch makes the run **not lossless** and
exits non-zero. The migration only ever *adds* — the source store is never
touched (see [Rollback](#safety-verification--rollback)).

## Every kind of RAG, and how it maps

CCOS treats "RAG" as a family. The bundle is expressive enough that each variant
maps cleanly; the only thing that changes is how much structure survives.

### Naïve / dense-vector RAG
*Chunks + embeddings in a vector DB (Chroma, FAISS, Qdrant, pgvector, Weaviate,
Pinecone, Milvus, LanceDB…).* Each chunk → a `doc:` content node. If the store
kept a `doc_id`/`source` and a `chunk_index`, those rebuild **containment**
(chunk → document) and **sequence** (chunk *i* → chunk *i+1*) edges — structure
the vector index never had. Embeddings are preserved for the semantic tier.

### Hybrid RAG (dense + BM25/sparse)
Same mapping. CCOS carries a **native BM25** lexical path
([`MEASUREMENT_beir.md`](MEASUREMENT_beir.md)), so the sparse half of your hybrid
retriever is a first-class recall signal after migration — you are not giving up
lexical matching to gain structure.

### RAG + reranker (cross-encoder)
The corpus imports exactly as dense RAG. The rerank *stage* maps to CCOS's
in-region exact re-ranking (and, on Pro, the OctaSoma semantic tier over the
preserved embeddings): rerank **within** the causal region rather than over a
flat top-k.

### GraphRAG (Microsoft)
The richest mapping — GraphRAG already built a graph, so the migration is nearly
1:1 and *this is where CCOS shines*:

| GraphRAG artifact | Becomes |
|---|---|
| `entities` | `entity/…` nodes |
| `relationships` | typed **edges** between entities (weighted) |
| `text_units` | `chunk` nodes, `mentions` edges to the entities they contain |
| `community_reports` | `community/…` summary nodes |

Your entity graph and community structure carry straight over — now with
determinism, replay, and a belief/tension axis on top.

### Agentic RAG (retrieval-as-a-tool)
Migrate the underlying corpus with the matching adapter, then **replace the
retrieval tool** with CCOS's MCP `recall` / `page_fault`. The agent stops calling
a similarity search and starts paging a causal region under a token budget — the
same interface, a different (self-bounding, auditable) backend.

### Long-context "stuff-it-all" (the non-RAG baseline)
Not RAG, but often what teams compare against. Ingest the corpus and let CCOS
page a bounded causal window instead of re-dumping everything each turn.

## The bundle format (CMB v1)

A **CCOS Migration Bundle** is JSON Lines (`*.cmb.jsonl`): an optional header
line, then one record per line. It is intentionally trivial to emit.

**Header** (optional — a headerless bundle of bare records is also accepted):

```json
{"cmb_version":"1.0","source":{"system":"chroma","collection":"kb"},
 "embedding":{"model":"text-embedding-3-small","dim":1536}}
```

**Record** — `id` and `content` are the only required fields:

```json
{
  "id":"chunk-0042",                    // stable, unique within the bundle
  "kind":"chunk",                       // document | chunk | entity | relationship | community | code_file
  "content":"Boost is regulated by the wastegate.",   // exact original text (the lossless anchor)
  "source_uri":"engine_manual.pdf",     // provenance (optional)
  "parent":"doc-engine_manual",         // containment target (optional)
  "ordinal":1,                          // position within parent → sequence edges (optional)
  "metadata":{"page":12,"section":"3.2"}, // preserved verbatim in the manifest (optional)
  "relations":[                          // explicit edges (optional; GraphRAG etc.)
    {"target":"entity/wastegate","kind":"mentions","weight":0.8}
  ],
  "embedding":[0.01, -0.04, ...]         // preserved in the embeddings sidecar (optional)
}
```

A `relations` entry may set `"from"` to link two *other* nodes (how a
`relationship` record models a GraphRAG edge); omitted, `from` is the record
itself. `content_sha256` is optional on input — the importer always recomputes
and verifies it.

## How structure is rebuilt

`ccos-migrate` reconstructs causal edges from what the bundle carries, then (for
code) derives the real ones from source:

| Bundle signal | CCOS edge | Kind |
|---|---|---|
| `parent` | parent → child | `Contains` |
| `ordinal` (adjacent within a parent) | chunk*ᵢ* → chunk*ᵢ₊₁* | `RelatedTo` (sequence) |
| `relations[]` (`mentions`,`cites`,`related`,`depends_on`,`supports`,`contradicts`,…) | from → target | mapped to the nearest [`EdgeType`](../src/memory.rs) |
| `kind:"code_file"` content | the file's real imports/calls/data-flow | `DependsOn`/`Calls`/`DataFlow` (parsed) |

Unknown relation kinds fall back to `RelatedTo`, and the original kind is kept in
the manifest — so the mapping never loses information. For a **code corpus** the
importer routes each record's source through the same parser the live kernel uses
(`ingest_deferred` + `resolve`), so you get the genuine dependency graph, not an
approximation.

## `ccos-migrate` reference

```
ccos-migrate --bundle <file.cmb.jsonl> [options]

  --bundle F, --in F      the CCOS Migration Bundle to import (required)
  --path W                target workspace (default: workspace.ccos; created or extended)
  --mode auto|code|docs   how to rebuild structure (default: auto — per-record by kind/extension)
  --report FILE           write the JSON migration report to FILE
  --dry-run               parse + import into an ephemeral memory + report; write nothing
  --no-verify             skip the per-node content-hash re-check (faster, unverified)
  --extend                merge into the existing workspace at --path (incremental
                          import — keeps its graph + hash-chained logs) instead of
                          assembling a fresh one
  --json                  emit the report as JSON on stdout
```

* **`--mode auto`** decides per record: a `code_file` kind (or a `source_uri`
  ending in a source extension) takes the code path; everything else is a
  document. Force it with `code` / `docs` for a homogeneous corpus.
* By default migration assembles a **fresh** workspace at `--path` (overwriting any
  file there). With **`--extend`** it instead *merges* the bundle into the existing
  workspace — keeping its causal graph, retained text and hash-chained logs — so you
  can fold many stores (or many simulated sessions) into one memory incrementally.
  The source RAG store is never touched, so a re-run is always safe.
* Exit code is **non-zero** if the lossless check fails, so it drops into CI.

**Outputs** (next to `--path`):

| File | Contents |
|---|---|
| `workspace.ccos` | the causal memory (graph + hash-chained logs + retained text) |
| `workspace.ccos.migration.jsonl` | the manifest — every original field, keyed by node id (lossless remainder) |
| `workspace.ccos.embeddings.jsonl` | preserved vectors (only if the bundle had embeddings) |

## `rag2ccos` reference

`tools/rag2ccos/rag2ccos.py` — the extractor. Standard-library core; live-store
adapters import their client lazily.

```
python tools/rag2ccos/rag2ccos.py <adapter> --in <source> --out <bundle> [opts]
python tools/rag2ccos/rag2ccos.py list      # show adapters
```

| Adapter | Source | Notes |
|---|---|---|
| `jsonl` | a JSON array or JSONL file | field names configurable; pure stdlib |
| `csv` | a CSV file | `--text-col` names the content column |
| `langchain` | LangChain / LlamaIndex JSON doc-store | preserves metadata + relationships |
| `graphrag` | a Microsoft GraphRAG output dir | parquet; needs `pyarrow` or `pandas` |
| `chroma` | a Chroma persist dir | needs `chromadb` |
| `faiss` | a FAISS index + JSON docstore | `--index` also carries the vectors |
| `qdrant` | a Qdrant URL + collection | needs `qdrant-client` |
| `pgvector` | a PostgreSQL + pgvector table | `--table`; needs `psycopg`/`psycopg2` |
| `weaviate` | a Weaviate class/collection | `--class`; needs `weaviate-client` |
| `pinecone` | a Pinecone index | text in vector metadata; needs `pinecone` |

Adding a store is a ~20-line adapter that yields `Record`s — the writer, hashing
and schema are shared. See the module docstring.

## Synthetic corpora — Qwen-AgentWorld (`worldsim`)

The bundle is not only for *existing* stores: anything that can write CMB can feed
CCOS. **[`tools/worldsim`](../tools/worldsim/README.md)** drives
[Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld) — a language *world
model* that simulates agentic environments (Terminal, SWE, OS, MCP) — as a
**synthetic-session generator**, emitting one bundle per simulated trajectory:

```bash
export WORLD_MODEL_URL=http://localhost:8000/v1     # vLLM/SGLang serving the model
export WORLD_MODEL_NAME=Qwen/Qwen-AgentWorld-35B-A3B
python tools/worldsim/worldsim.py --domain terminal \
    --task "find the largest file under /var/log" --actions pol.txt --out sess.cmb.jsonl
ccos-migrate --bundle sess.cmb.jsonl --path fleet.ccos --extend   # accumulate a fleet
```

Each session becomes a `document` task node with one `chunk` per turn
(`parent` = session, `ordinal` = turn) — so CCOS rebuilds the session's containment
and turn-sequence edges. Paired with **`--extend`**, thousands of simulated
sessions fold into one causal memory you can then stress with `ccos postmortem`,
warm up RL against, or fuzz the MCP surface with — no real environment required.
Everything is marked `simulated: true` (and `worldsim --offline` runs with no
endpoint at all, for testing the pipeline). Prefer the Terminal / SWE / OS / MCP
domains; the model card flags Search as weakest.

## Recall after migration

The workspace is ordinary CCOS memory. Which recall strategy fits depends on what
you migrated:

* **Document corpora** — use `working_set` (the globally hottest nodes, which is
  the natural "what's relevant now?" query for prose) and, on Pro, the
  **semantic tier** seeded from the preserved embeddings sidecar. This is the
  drop-in replacement for a vector-store `similarity_search`.
* **Code corpora** — use `around <anchor>` / `task <text>`: the causal-region
  strategies are tuned for the import/call/data-flow graph and self-bound at the
  region (no *k* to pick). This is the recall that beats flat top-k on cross-file
  structure ([`FIELD_CAMPAIGN_H.md`](FIELD_CAMPAIGN_H.md)).

Either way, the same memory also gives you `page_fault` (recall from a failing
test), `timeline`, and `recall_what_if` (time-travel) — capabilities a RAG stack
structurally lacks.

## Safety, verification & rollback

* **Non-destructive.** Migration only reads your RAG export and writes a new
  `workspace.ccos`. The source store is never modified. Rollback is `rm`.
* **Dry-run first.** `--dry-run` imports into memory and prints the full report
  (counts, edge reconstruction, lossless verdict) without writing anything.
* **Verified by default.** Every node's content hash is re-checked against the
  source record; the run reports `lossless: true/false` and exits non-zero on any
  mismatch — wire it into CI to gate a migration.
* **Deterministic.** Import is RNG-free: the same bundle produces the same
  workspace, and the assembled memory's hash chains verify on load (that check is
  what `--verify` runs), so a re-run is byte-identical.
* **Byte-exact.** Each document's text is stored exactly as the source held it —
  the lossless anchor the per-node content-hash check verifies, byte for byte.
  Code records additionally pass through the kernel's Unicode de-obfuscation and
  injection-scoring on the way in, the same as any `ingest`.

## Worked examples

### From Chroma
```bash
python tools/rag2ccos/rag2ccos.py chroma --in ./chroma_db --collection kb \
    --out kb.cmb.jsonl
ccos-migrate --bundle kb.cmb.jsonl --path kb.ccos --report kb.report.json
ccos mcp kb.ccos          # serve it to an MCP agent
```

### From Microsoft GraphRAG
```bash
python tools/rag2ccos/rag2ccos.py graphrag --in ./ragtest/output/artifacts \
    --out graph.cmb.jsonl
ccos-migrate --bundle graph.cmb.jsonl --path graph.ccos
# entities → nodes, relationships → edges, communities → summaries, all preserved
```

### From a plain JSONL export (any store you can dump)
```bash
python tools/rag2ccos/rag2ccos.py jsonl --in export.jsonl --out corpus.cmb.jsonl \
    --text-field text --id-field id --source-field source \
    --parent-field doc_id --ordinal-field chunk_index --embedding-field embedding
ccos-migrate --bundle corpus.cmb.jsonl --path corpus.ccos --dry-run   # preview
ccos-migrate --bundle corpus.cmb.jsonl --path corpus.ccos             # commit
```

### A RAG over a codebase (highest fidelity)
```bash
# emit code_file records (content = source), then let CCOS parse the real graph
python tools/rag2ccos/rag2ccos.py jsonl --in code_chunks.jsonl \
    --out code.cmb.jsonl --kind code_file --source-field path --text-field source
ccos-migrate --bundle code.cmb.jsonl --path repo.ccos --mode code
ccos memory --path repo.ccos <<< '{"op":"recall","strategy":"around","anchor":"file:src/db.rs","budget":2048}'
```
