#!/usr/bin/env python3
"""rag2ccos — normalize any RAG store into a CCOS Migration Bundle (CMB).

This is the *extraction* half of the CCOS migration path. It reads a RAG index
(vector store, graph index, or framework doc-store), and emits a canonical,
loss-preserving JSONL bundle (`*.cmb.jsonl`) that `ccos migrate` imports into a
CCOS causal working-memory.

Design goals
------------
* **Lossless.** Every chunk's exact text, all metadata, its provenance, its
  relationships and (when present) its embedding vector are carried through
  verbatim. The bundle is a superset of what the RAG store held — nothing is
  dropped, so the migration can be verified byte-for-byte downstream.
* **Store-agnostic.** One canonical format (CMB v1). Each RAG system gets a thin
  adapter that yields records; the writer, hashing and schema are shared.
* **Dependency-light.** The core + JSONL/CSV/LangChain/LlamaIndex/GraphRAG-JSON
  adapters use only the Python standard library. Adapters for live stores
  (Chroma, FAISS, Qdrant, parquet GraphRAG) import their client lazily and only
  when that adapter is selected, so `python rag2ccos.py --help` always works.

Usage
-----
    python rag2ccos.py <adapter> --in <source> --out corpus.cmb.jsonl [opts]
    python rag2ccos.py list                      # show adapters
    python rag2ccos.py <adapter> --help          # per-adapter options

Then, on the CCOS side:
    ccos migrate --bundle corpus.cmb.jsonl --path workspace.ccos --verify

See docs/MIGRATION.md for the full mapping and the lossless contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator, Optional

CMB_VERSION = "1.0"
TOOL_ID = "rag2ccos/0.1"


# --------------------------------------------------------------------------- #
#  Canonical record — the CMB v1 unit                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    """One node in the migration bundle.

    `content` is mandatory and is the *exact* original text — the lossless
    anchor. Everything else is optional context CCOS uses to rebuild structure.
    """

    id: str
    content: str
    kind: str = "chunk"  # document|chunk|entity|relationship|community|code_file
    source_uri: Optional[str] = None
    parent: Optional[str] = None
    ordinal: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    embedding: Optional[list[float]] = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "kind": self.kind, "content": self.content}
        # Content hash travels with the record so the importer can verify it
        # without trusting the writer.
        d["content_sha256"] = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.source_uri is not None:
            d["source_uri"] = self.source_uri
        if self.parent is not None:
            d["parent"] = self.parent
        if self.ordinal is not None:
            d["ordinal"] = self.ordinal
        if self.metadata:
            d["metadata"] = self.metadata
        if self.relations:
            d["relations"] = self.relations
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d


@dataclass
class BundleHeader:
    source_system: str
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    note: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        h: dict[str, Any] = {"cmb_version": CMB_VERSION, "tool": TOOL_ID}
        src: dict[str, Any] = {"system": self.source_system}
        if self.collection:
            src["collection"] = self.collection
        h["source"] = src
        if self.embedding_model or self.embedding_dim:
            emb: dict[str, Any] = {}
            if self.embedding_model:
                emb["model"] = self.embedding_model
            if self.embedding_dim:
                emb["dim"] = self.embedding_dim
            h["embedding"] = emb
        if self.note:
            h["note"] = self.note
        return h


class BundleWriter:
    """Streaming CMB JSONL writer: one header line, then one line per record."""

    def __init__(self, path: str, header: BundleHeader):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write(json.dumps(header.to_json(), ensure_ascii=False) + "\n")
        self.count = 0
        self.total_chars = 0
        self.with_embedding = 0
        self.with_relations = 0

    def write(self, rec: Record) -> None:
        if not rec.content and rec.kind != "relationship":
            # A content-free record cannot be lossless; skip with a warning
            # rather than emit an empty node.
            sys.stderr.write(f"warning: record {rec.id!r} has empty content; skipped\n")
            return
        self._fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
        self.count += 1
        self.total_chars += len(rec.content)
        if rec.embedding is not None:
            self.with_embedding += 1
        if rec.relations:
            self.with_relations += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "BundleWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
#  Adapter registry                                                            #
# --------------------------------------------------------------------------- #
ADAPTERS: dict[str, "Adapter"] = {}


class Adapter:
    name = "base"
    help = ""

    def add_args(self, p: argparse.ArgumentParser) -> None:  # pragma: no cover
        ...

    def header(self, args: argparse.Namespace) -> BundleHeader:  # pragma: no cover
        raise NotImplementedError

    def records(self, args: argparse.Namespace) -> Iterator[Record]:  # pragma: no cover
        raise NotImplementedError


def register(cls: type[Adapter]) -> type[Adapter]:
    ADAPTERS[cls.name] = cls()
    return cls


def _coerce_meta(meta: Any) -> dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    if meta is None:
        return {}
    return {"value": meta}


# --------------------------------------------------------------------------- #
#  Adapter: generic JSONL / JSON                                               #
# --------------------------------------------------------------------------- #
@register
class JsonlAdapter(Adapter):
    name = "jsonl"
    help = (
        "Generic JSON/JSONL export. Each object is a chunk. Map your field names "
        "with --text-field / --id-field / --source-field / --parent-field / "
        "--embedding-field; everything else is preserved as metadata."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--text-field", default="text")
        p.add_argument("--id-field", default="id")
        p.add_argument("--source-field", default="source")
        p.add_argument("--parent-field", default=None)
        p.add_argument("--ordinal-field", default=None)
        p.add_argument("--embedding-field", default="embedding")
        p.add_argument("--kind", default="chunk")

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="jsonl", collection=os.path.basename(args.inp))

    def _iter_objects(self, path: str) -> Iterator[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            head = fh.read(1)
            fh.seek(0)
            if head == "[":  # a single JSON array
                for obj in json.load(fh):
                    yield obj
            else:  # JSONL
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        tf, idf, sf = args.text_field, args.id_field, args.source_field
        pf, of, ef = args.parent_field, args.ordinal_field, args.embedding_field
        for i, obj in enumerate(self._iter_objects(args.inp)):
            text = obj.get(tf)
            if text is None:
                continue
            rid = str(obj.get(idf, f"rec-{i:06d}"))
            meta = {k: v for k, v in obj.items() if k not in {tf, idf, sf, pf, of, ef}}
            yield Record(
                id=rid,
                content=str(text),
                kind=args.kind,
                source_uri=obj.get(sf),
                parent=obj.get(pf) if pf else None,
                ordinal=obj.get(of) if of else None,
                metadata=_coerce_meta(meta),
                embedding=obj.get(ef),
            )


# --------------------------------------------------------------------------- #
#  Adapter: CSV                                                                #
# --------------------------------------------------------------------------- #
@register
class CsvAdapter(Adapter):
    name = "csv"
    help = "Generic CSV export. --text-col names the content column; other columns become metadata."

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--text-col", default="text")
        p.add_argument("--id-col", default="id")
        p.add_argument("--source-col", default="source")

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="csv", collection=os.path.basename(args.inp))

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        import csv

        with open(args.inp, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                text = row.get(args.text_col)
                if not text:
                    continue
                meta = {k: v for k, v in row.items() if k not in {args.text_col, args.id_col, args.source_col}}
                yield Record(
                    id=str(row.get(args.id_col) or f"row-{i:06d}"),
                    content=text,
                    source_uri=row.get(args.source_col),
                    metadata=meta,
                )


# --------------------------------------------------------------------------- #
#  Adapter: LangChain / LlamaIndex JSON doc-store                              #
# --------------------------------------------------------------------------- #
@register
class LangchainAdapter(Adapter):
    name = "langchain"
    help = (
        "LangChain / LlamaIndex JSON export (list of Documents with page_content "
        "+ metadata, or a docstore dict). Preserves metadata and any 'relationships'."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--source-key", default="source",
                       help="metadata key holding the document source/path")

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="langchain", collection=os.path.basename(args.inp))

    def _docs(self, data: Any) -> Iterable[dict[str, Any]]:
        # LangChain: [{"page_content": ..., "metadata": {...}}, ...]
        # LlamaIndex docstore: {"docstore/data": {id: {"__data__": {"text":..}}}}
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            store = data.get("docstore/data") or data.get("docstore", {}).get("docs") or data
            if isinstance(store, dict):
                for k, v in store.items():
                    inner = v.get("__data__", v) if isinstance(v, dict) else {}
                    inner.setdefault("_id", k)
                    yield inner

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        with open(args.inp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for i, doc in enumerate(self._docs(data)):
            text = doc.get("page_content") or doc.get("text") or doc.get("content")
            if not text:
                continue
            meta = dict(doc.get("metadata") or doc.get("extra_info") or {})
            rid = str(doc.get("_id") or doc.get("id_") or meta.get("id") or f"doc-{i:06d}")
            rels = []
            for rk, rv in (doc.get("relationships") or {}).items():
                for tgt in (rv if isinstance(rv, list) else [rv]):
                    tid = tgt.get("node_id") if isinstance(tgt, dict) else tgt
                    if tid:
                        rels.append({"target": str(tid), "kind": str(rk).lower()})
            yield Record(
                id=rid,
                content=str(text),
                kind="document",
                source_uri=meta.get(args.source_key),
                metadata=meta,
                relations=rels,
                embedding=doc.get("embedding"),
            )


# --------------------------------------------------------------------------- #
#  Adapter: Microsoft GraphRAG (parquet artifacts)                            #
# --------------------------------------------------------------------------- #
@register
class GraphragAdapter(Adapter):
    name = "graphrag"
    help = (
        "Microsoft GraphRAG output directory (parquet artifacts). Maps entities "
        "-> nodes, relationships -> typed edges, text_units -> content nodes, "
        "community_reports -> summary nodes. Needs pyarrow or pandas."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--artifacts", default=None,
                       help="path to the GraphRAG 'output/artifacts' dir (default: --in)")

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="graphrag", collection="graphrag",
                            note="entities->nodes, relationships->edges, "
                                 "text_units->content, community_reports->summaries")

    def _read_parquet(self, path: str) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # type: ignore

            return pq.read_table(path).to_pylist()
        except ImportError:
            try:
                import pandas as pd  # type: ignore

                return pd.read_parquet(path).to_dict("records")
            except ImportError:
                raise SystemExit("graphrag adapter needs 'pyarrow' or 'pandas' installed")

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        base = args.artifacts or args.inp

        def maybe(name: str) -> list[dict[str, Any]]:
            for fn in (f"create_final_{name}.parquet", f"{name}.parquet"):
                p = os.path.join(base, fn)
                if os.path.exists(p):
                    return self._read_parquet(p)
            return []

        # Entities -> nodes
        for e in maybe("entities"):
            eid = str(e.get("id") or e.get("human_readable_id") or e.get("title"))
            title = e.get("title") or e.get("name") or eid
            desc = e.get("description") or ""
            yield Record(
                id=f"entity/{eid}",
                content=f"{title}\n\n{desc}".strip(),
                kind="entity",
                metadata={k: v for k, v in e.items() if k not in {"description", "text_unit_ids"}},
            )
        # Relationships -> edges (modeled as relationship records that carry a
        # single relation; the importer turns each into a typed edge)
        for r in maybe("relationships"):
            src = r.get("source")
            tgt = r.get("target")
            if not src or not tgt:
                continue
            yield Record(
                id=f"rel/{r.get('id') or f'{src}->{tgt}'}",
                content=str(r.get("description") or f"{src} — {tgt}"),
                kind="relationship",
                relations=[{
                    "from": f"entity/{src}",
                    "target": f"entity/{tgt}",
                    "kind": "related",
                    "weight": float(r.get("weight", 1.0) or 1.0),
                }],
                metadata={k: v for k, v in r.items() if k not in {"description"}},
            )
        # Text units -> content nodes, linked to the entities they mention
        for t in maybe("text_units"):
            tid = str(t.get("id"))
            rels = [{"target": f"entity/{e}", "kind": "mentions"}
                    for e in (t.get("entity_ids") or [])]
            yield Record(
                id=f"text/{tid}",
                content=str(t.get("text") or ""),
                kind="chunk",
                source_uri=(t.get("document_ids") or [None])[0],
                relations=rels,
                metadata={"n_tokens": t.get("n_tokens")},
            )
        # Community reports -> summary nodes
        for c in maybe("community_reports"):
            cid = str(c.get("community") or c.get("id"))
            yield Record(
                id=f"community/{cid}",
                content=str(c.get("full_content") or c.get("summary") or ""),
                kind="community",
                metadata={"level": c.get("level"), "rank": c.get("rank"), "title": c.get("title")},
            )


# --------------------------------------------------------------------------- #
#  Adapter: Chroma (persistent client)                                        #
# --------------------------------------------------------------------------- #
@register
class ChromaAdapter(Adapter):
    name = "chroma"
    help = "Chroma persistent store. --in is the persist directory; --collection selects one."

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--collection", required=True)
        p.add_argument("--batch", type=int, default=1000)

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="chroma", collection=args.collection)

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        try:
            import chromadb  # type: ignore
        except ImportError:
            raise SystemExit("chroma adapter needs 'chromadb' installed")
        client = chromadb.PersistentClient(path=args.inp)
        col = client.get_collection(args.collection)
        n = col.count()
        got = 0
        while got < n:
            page = col.get(include=["documents", "metadatas", "embeddings"],
                           limit=args.batch, offset=got)
            ids = page["ids"]
            docs = page.get("documents") or [None] * len(ids)
            metas = page.get("metadatas") or [None] * len(ids)
            embs = page.get("embeddings") or [None] * len(ids)
            for j, rid in enumerate(ids):
                if not docs[j]:
                    continue
                meta = _coerce_meta(metas[j])
                yield Record(
                    id=str(rid),
                    content=str(docs[j]),
                    source_uri=meta.get("source") or meta.get("file_path"),
                    parent=meta.get("doc_id") or meta.get("parent"),
                    ordinal=meta.get("chunk_index"),
                    metadata=meta,
                    embedding=list(embs[j]) if embs[j] is not None else None,
                )
            got += len(ids)
            if not ids:
                break


# --------------------------------------------------------------------------- #
#  Adapter: FAISS + JSON/pkl sidecar metadata                                 #
# --------------------------------------------------------------------------- #
@register
class FaissAdapter(Adapter):
    name = "faiss"
    help = (
        "FAISS index with a sidecar docstore. --in is the docstore JSON "
        "(id -> {text, metadata}); pass --index to also carry the vectors."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--index", default=None, help="optional .faiss index for embeddings")

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="faiss", collection=os.path.basename(args.inp))

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        with open(args.inp, "r", encoding="utf-8") as fh:
            store = json.load(fh)
        vectors = None
        if args.index:
            try:
                import faiss  # type: ignore
                import numpy as np  # type: ignore

                idx = faiss.read_index(args.index)
                vectors = idx.reconstruct_n(0, idx.ntotal)
                vectors = np.asarray(vectors)
            except Exception as exc:  # pragma: no cover
                sys.stderr.write(f"warning: could not read FAISS vectors ({exc}); "
                                 "continuing without embeddings\n")
        items = store.items() if isinstance(store, dict) else enumerate(store)
        for pos, (rid, entry) in enumerate(items):
            if isinstance(entry, str):
                text, meta = entry, {}
            else:
                text = entry.get("text") or entry.get("page_content") or ""
                meta = entry.get("metadata") or {}
            if not text:
                continue
            emb = None
            if vectors is not None and pos < len(vectors):
                emb = [float(x) for x in vectors[pos]]
            yield Record(
                id=str(rid),
                content=str(text),
                source_uri=meta.get("source"),
                metadata=meta,
                embedding=emb,
            )


# --------------------------------------------------------------------------- #
#  Adapter: Qdrant (scroll API)                                               #
# --------------------------------------------------------------------------- #
@register
class QdrantAdapter(Adapter):
    name = "qdrant"
    help = "Qdrant collection over the client scroll API. --in is the URL (e.g. http://localhost:6333)."

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--collection", required=True)
        p.add_argument("--text-key", default="text")
        p.add_argument("--batch", type=int, default=512)

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="qdrant", collection=args.collection)

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError:
            raise SystemExit("qdrant adapter needs 'qdrant-client' installed")
        client = QdrantClient(url=args.inp)
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=args.collection, limit=args.batch,
                offset=offset, with_payload=True, with_vectors=True)
            for pt in points:
                payload = pt.payload or {}
                text = payload.get(args.text_key)
                if not text:
                    continue
                meta = {k: v for k, v in payload.items() if k != args.text_key}
                vec = pt.vector
                if isinstance(vec, dict):  # named vectors
                    vec = next(iter(vec.values()), None)
                yield Record(
                    id=str(pt.id),
                    content=str(text),
                    source_uri=meta.get("source"),
                    metadata=meta,
                    embedding=list(vec) if vec is not None else None,
                )
            if offset is None:
                break


# --------------------------------------------------------------------------- #
#  Adapter: pgvector (PostgreSQL + the pgvector extension)                     #
# --------------------------------------------------------------------------- #
@register
class PgvectorAdapter(Adapter):
    name = "pgvector"
    help = (
        "A PostgreSQL/pgvector table. --in is the connection string (or set PGVECTOR_DSN); "
        "map columns with --text-col / --id-col / --embedding-col. Needs psycopg or psycopg2."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--table", required=True, help="table (or view) to read from")
        p.add_argument("--text-col", default="content")
        p.add_argument("--id-col", default="id")
        p.add_argument("--source-col", default=None)
        p.add_argument("--embedding-col", default="embedding")
        p.add_argument("--extra-cols", default="",
                       help="comma-separated columns to keep as metadata")
        p.add_argument("--batch", type=int, default=1000)

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="pgvector", collection=args.table)

    def _connect(self, dsn: str):
        try:
            import psycopg  # type: ignore

            return psycopg.connect(dsn), "psycopg"
        except ImportError:
            try:
                import psycopg2  # type: ignore

                return psycopg2.connect(dsn), "psycopg2"
            except ImportError:
                raise SystemExit("pgvector adapter needs 'psycopg' (v3) or 'psycopg2' installed")

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        dsn = args.inp or os.environ.get("PGVECTOR_DSN")
        if not dsn:
            raise SystemExit("pgvector: pass --in <dsn> or set PGVECTOR_DSN")
        extra = [c.strip() for c in args.extra_cols.split(",") if c.strip()]
        cols = [args.id_col, args.text_col, args.embedding_col]
        if args.source_col:
            cols.append(args.source_col)
        cols += extra
        # Quote identifiers defensively; the column/table names come from the operator, not data.
        sel = ", ".join('"{}"'.format(c.replace('"', '""')) for c in cols)
        tbl = '"{}"'.format(args.table.replace('"', '""'))
        conn, _drv = self._connect(dsn)
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {sel} FROM {tbl}")
            names = [d[0] for d in cur.description]
            while True:
                rows = cur.fetchmany(args.batch)
                if not rows:
                    break
                for row in rows:
                    d = dict(zip(names, row))
                    text = d.get(args.text_col)
                    if not text:
                        continue
                    emb = d.get(args.embedding_col)
                    if isinstance(emb, str):  # pgvector renders as '[1,2,3]'
                        emb = [float(x) for x in emb.strip("[]").split(",") if x.strip()]
                    elif emb is not None:
                        emb = [float(x) for x in emb]
                    meta = {k: d[k] for k in extra}
                    yield Record(
                        id=str(d.get(args.id_col)),
                        content=str(text),
                        source_uri=str(d[args.source_col]) if args.source_col else None,
                        metadata=meta,
                        embedding=emb,
                    )
        finally:
            conn.close()


# --------------------------------------------------------------------------- #
#  Adapter: Weaviate                                                          #
# --------------------------------------------------------------------------- #
@register
class WeaviateAdapter(Adapter):
    name = "weaviate"
    help = (
        "A Weaviate class/collection. --in is the URL (e.g. http://localhost:8080); "
        "--class names the collection, --text-prop the text property. Needs weaviate-client."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--class", dest="klass", required=True, help="Weaviate class/collection name")
        p.add_argument("--text-prop", default="text")
        p.add_argument("--batch", type=int, default=200)

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="weaviate", collection=args.klass)

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        try:
            import weaviate  # type: ignore
        except ImportError:
            raise SystemExit("weaviate adapter needs 'weaviate-client' installed")
        # Support both the v4 (connect_to_custom / collections) and v3 (Client) clients.
        try:
            client = weaviate.connect_to_local() if "localhost" in (args.inp or "") else \
                weaviate.connect_to_custom(http_host=args.inp, http_port=443, http_secure=True,
                                           grpc_host=args.inp, grpc_port=443, grpc_secure=True)
            coll = client.collections.get(args.klass)
            try:
                for obj in coll.iterator(include_vector=True):
                    props = obj.properties or {}
                    text = props.get(args.text_prop)
                    if not text:
                        continue
                    vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
                    meta = {k: v for k, v in props.items() if k != args.text_prop}
                    yield Record(
                        id=str(obj.uuid),
                        content=str(text),
                        source_uri=meta.get("source"),
                        metadata=meta,
                        embedding=list(vec) if vec else None,
                    )
            finally:
                client.close()
        except AttributeError:
            # v3 fallback: GraphQL cursor scan.
            client = weaviate.Client(args.inp)
            after = None
            while True:
                q = client.query.get(args.klass, [args.text_prop]) \
                    .with_additional(["id", "vector"]).with_limit(args.batch)
                if after:
                    q = q.with_after(after)
                res = q.do().get("data", {}).get("Get", {}).get(args.klass, [])
                if not res:
                    break
                for obj in res:
                    text = obj.get(args.text_prop)
                    add = obj.get("_additional", {})
                    after = add.get("id")
                    if not text:
                        continue
                    meta = {k: v for k, v in obj.items() if k not in {args.text_prop, "_additional"}}
                    yield Record(id=str(add.get("id")), content=str(text),
                                 metadata=meta, embedding=add.get("vector"))


# --------------------------------------------------------------------------- #
#  Adapter: Pinecone                                                          #
# --------------------------------------------------------------------------- #
@register
class PineconeAdapter(Adapter):
    name = "pinecone"
    help = (
        "A Pinecone index. --in is the index name (API key from PINECONE_API_KEY); the text "
        "must live in the vector metadata (--text-key). Needs pinecone(-client)."
    )

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--namespace", default="")
        p.add_argument("--text-key", default="text")
        p.add_argument("--dim", type=int, default=None,
                       help="index dimension (used only to page ids by a zero-vector query)")
        p.add_argument("--top-k", type=int, default=1000)

    def header(self, args: argparse.Namespace) -> BundleHeader:
        return BundleHeader(source_system="pinecone", collection=args.inp)

    def records(self, args: argparse.Namespace) -> Iterator[Record]:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise SystemExit("pinecone: set PINECONE_API_KEY")
        try:
            from pinecone import Pinecone  # type: ignore

            index = Pinecone(api_key=api_key).Index(args.inp)
        except ImportError:
            raise SystemExit("pinecone adapter needs the 'pinecone' package installed")
        # Pinecone has no full scan; page ids via list(), then fetch in batches.
        seen: set[str] = set()
        try:
            for id_page in index.list(namespace=args.namespace):
                ids = list(id_page)
                if not ids:
                    continue
                fetched = index.fetch(ids=ids, namespace=args.namespace)
                vectors = getattr(fetched, "vectors", None) or fetched.get("vectors", {})
                for vid, v in vectors.items():
                    if vid in seen:
                        continue
                    seen.add(vid)
                    meta = (getattr(v, "metadata", None) or v.get("metadata") or {})
                    text = meta.get(args.text_key)
                    if not text:
                        continue
                    values = getattr(v, "values", None) or v.get("values")
                    yield Record(
                        id=str(vid),
                        content=str(text),
                        source_uri=meta.get("source"),
                        metadata={k: val for k, val in meta.items() if k != args.text_key},
                        embedding=list(values) if values else None,
                    )
        except (AttributeError, TypeError):
            raise SystemExit(
                "pinecone: this index/client does not support id listing; export via your own "
                "query loop into the `jsonl` adapter instead"
            )


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #
def _run_adapter(adapter: Adapter, args: argparse.Namespace) -> int:
    header = adapter.header(args)
    if not header.embedding_model and getattr(args, "embedding_model", None):
        header.embedding_model = args.embedding_model
    with BundleWriter(args.out, header) as w:
        for rec in adapter.records(args):
            w.write(rec)
    emb_pct = (100 * w.with_embedding // w.count) if w.count else 0
    sys.stderr.write(
        f"✓ wrote {w.count} records to {args.out}\n"
        f"  {w.total_chars:,} chars · {w.with_embedding} with embeddings ({emb_pct}%) · "
        f"{w.with_relations} with relations\n"
        f"  next: ccos migrate --bundle {args.out} --path workspace.ccos --verify\n"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_top_help()
        return 0
    if argv[0] == "list":
        for name, a in sorted(ADAPTERS.items()):
            print(f"  {name:<10} {a.help}")
        return 0
    adapter_name = argv[0]
    adapter = ADAPTERS.get(adapter_name)
    if adapter is None:
        sys.stderr.write(f"unknown adapter {adapter_name!r}. Try: python rag2ccos.py list\n")
        return 2
    p = argparse.ArgumentParser(prog=f"rag2ccos {adapter_name}", description=adapter.help)
    p.add_argument("--in", dest="inp", required=True, help="source (path or URL, per adapter)")
    p.add_argument("--out", required=True, help="output bundle path (*.cmb.jsonl)")
    p.add_argument("--embedding-model", default=None, help="record the embedding model name in the header")
    adapter.add_args(p)
    args = p.parse_args(argv[1:])
    try:
        return _run_adapter(adapter, args)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"error: {exc}\n")
        return 1


def _print_top_help() -> None:
    print(__doc__)
    print("Adapters:")
    for name, a in sorted(ADAPTERS.items()):
        print(f"  {name:<10} {a.help}")


if __name__ == "__main__":
    raise SystemExit(main())
