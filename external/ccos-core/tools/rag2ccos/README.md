# rag2ccos

The **extraction** half of the [RAG → CCOS migration](../../docs/MIGRATION.md):
normalise any RAG store into a canonical **CCOS Migration Bundle**
(`*.cmb.jsonl`), which [`ccos-migrate`](../../docs/MIGRATION.md#ccos-migrate-reference)
then imports into a causal working-memory — losslessly.

```
  your RAG store  ──rag2ccos──▶  corpus.cmb.jsonl  ─ccos-migrate─▶  workspace.ccos
```

## Why a separate tool

RAG stores live in the Python ecosystem; CCOS is Rust. So extraction is a thin
Python tool (one adapter per store, yielding records into one shared,
loss-preserving format) and import is native to CCOS. The bundle is the contract
between them — anything that can write it can migrate, including your own ETL.

## Install

Nothing to install for the stdlib adapters (`jsonl`, `csv`, `langchain`,
`graphrag`-JSON). Live-store adapters import their client lazily; install only
what you use:

```bash
pip install chromadb          # chroma
pip install qdrant-client     # qdrant
pip install pyarrow           # graphrag (parquet)  — or: pip install pandas
pip install faiss-cpu numpy   # faiss (to also carry vectors)
```

Python ≥ 3.9. No dependency is required to run `--help` or the stdlib adapters.

## Usage

```bash
python rag2ccos.py list                      # show all adapters
python rag2ccos.py <adapter> --help          # per-adapter options
python rag2ccos.py <adapter> --in <src> --out corpus.cmb.jsonl [opts]
```

Every adapter shares `--in`, `--out`, and `--embedding-model` (records the model
name in the bundle header).

### Adapters

| Adapter | `--in` is | Key options | Needs |
|---|---|---|---|
| `jsonl` | a JSON array or JSONL file | `--text-field --id-field --source-field --parent-field --ordinal-field --embedding-field --kind` | stdlib |
| `csv` | a CSV file | `--text-col --id-col --source-col` | stdlib |
| `langchain` | LangChain/LlamaIndex JSON export | `--source-key` | stdlib |
| `graphrag` | a GraphRAG `output/artifacts` dir | `--artifacts` | `pyarrow` or `pandas` |
| `chroma` | a Chroma persist directory | `--collection --batch` | `chromadb` |
| `faiss` | a JSON docstore (`id → {text, metadata}`) | `--index` (optional, for vectors) | `faiss`+`numpy` if `--index` |
| `qdrant` | a Qdrant URL (`http://host:6333`) | `--collection --text-key --batch` | `qdrant-client` |

### Examples

```bash
# Chroma
python rag2ccos.py chroma --in ./chroma_db --collection knowledge --out kb.cmb.jsonl

# Microsoft GraphRAG (entities/relationships/text_units/community_reports)
python rag2ccos.py graphrag --in ./ragtest/output/artifacts --out graph.cmb.jsonl

# A generic JSONL dump, mapping your field names
python rag2ccos.py jsonl --in export.jsonl --out corpus.cmb.jsonl \
    --text-field text --id-field id --source-field source \
    --parent-field doc_id --ordinal-field chunk_index --embedding-field embedding
```

Then import (see the [migration guide](../../docs/MIGRATION.md)):

```bash
ccos-migrate --bundle corpus.cmb.jsonl --path workspace.ccos --verify
```

## The bundle it writes

One header line, then one record per line. `id` + `content` are the only required
fields; `source_uri`, `parent`, `ordinal`, `metadata`, `relations` and
`embedding` are preserved when present. Full schema:
[CMB v1](../../docs/MIGRATION.md#the-bundle-format-cmb-v1).

## Adding a store

An adapter is a small class that yields `Record`s; the writer, SHA-256 hashing
and schema are shared. Subclass `Adapter`, implement `add_args`, `header` and
`records`, and decorate with `@register`. See the module docstring in
`rag2ccos.py` and the existing adapters for the pattern (~20 lines each).

## Losslessness

The bundle is a **superset** of what the store held: exact text, all metadata,
provenance, relationships and embeddings travel through verbatim. `ccos-migrate`
recomputes each content hash on import and fails the run if any node's text does
not match — so a migration is verifiable end-to-end, and the source store is
never modified.
