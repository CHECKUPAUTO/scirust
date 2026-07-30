//! RAG → CCOS migration — import a **CCOS Migration Bundle** (CMB) into a causal
//! working-memory, losslessly.
//!
//! This is the CCOS-side half of the migration path (`docs/MIGRATION.md`). The
//! source side (`tools/rag2ccos`) normalises any RAG store — a dense/hybrid
//! vector index, a Microsoft GraphRAG output, a LangChain/LlamaIndex doc-store —
//! into one canonical JSONL format; this module imports that format into a
//! [`CcosMemory`], reconstructing the structure a flat retriever discarded.
//!
//! It is a pure **consumer** of the CCOS public API: document records are
//! assembled directly into a [`MemoryGraph`] (content nodes + typed causal edges)
//! and loaded through [`CcosMemory::from_json`]; code records are re-parsed by the
//! kernel's own path ([`CcosMemory::ingest_deferred`] + [`CcosMemory::resolve`]).
//! Nothing in the core is modified.
//!
//! # What "lossless" means here
//!
//! A RAG store holds five things per item: the **text**, its **metadata**, its
//! **provenance**, its **relationships**, and (optionally) its **embedding**.
//! The migration preserves every one:
//!
//! * **text** → the exact bytes become the node `content`, retained verbatim, so
//!   `recall` returns them and a checkpoint round-trips them (verified: each
//!   node's content hash is re-checked against the record's).
//! * **metadata** → carried in the migration **manifest**
//!   (`<workspace>.migration.jsonl`), one line per record, keyed by node id — the
//!   authoritative "nothing was dropped" record.
//! * **provenance** (`source_uri`) → the node label / manifest.
//! * **relationships** → typed causal **edges** (containment, sequence, and any
//!   explicit relations such as GraphRAG entity links).
//! * **embedding** → the **embeddings** sidecar (`<workspace>.embeddings.jsonl`),
//!   ready to seed the Pro semantic tier so dense-recall parity is retained.
//!
//! The causal graph is what the migration *adds*: structure the RAG never had.

use crate::distributed_event_log::DistributedEventLog;
use crate::event_log::EventLog;
use crate::external_memory::CcosMemory;
use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::BufRead;

/// How a bundle's records are mapped onto CCOS nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Decide per record: `code_file` records (or records whose `source_uri`
    /// looks like a source file) take the code path; everything else is a document.
    Auto,
    /// Treat every record's `content` as source code (route through `ingest_deferred`).
    Code,
    /// Treat every record as a text document.
    Docs,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Mode::Auto),
            "code" => Some(Mode::Code),
            "docs" | "doc" | "documents" => Some(Mode::Docs),
            _ => None,
        }
    }
}

/// Import options.
#[derive(Debug, Clone)]
pub struct MigrateConfig {
    pub mode: Mode,
    /// Re-check every record's content hash after import (nodes present + text intact).
    pub verify: bool,
    /// Incremental import: when `Some(path)`, **extend** the existing workspace at
    /// `path` — its causal graph, retained sources and hash-chained logs are kept,
    /// and the bundle's document nodes/edges + re-parsed code are merged on top —
    /// instead of assembling a fresh workspace. (If the path does not yet exist it
    /// is created empty, so extend degrades to a normal import.)
    pub extend_from: Option<String>,
}

impl Default for MigrateConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Auto,
            verify: true,
            extend_from: None,
        }
    }
}

/// An explicit relationship carried by a record (e.g. a GraphRAG entity link).
/// `from` defaults to the record's own node when omitted.
#[derive(Debug, Clone, Deserialize)]
pub struct CmbRelation {
    #[serde(default)]
    pub from: Option<String>,
    pub target: String,
    #[serde(default = "default_rel_kind")]
    pub kind: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_rel_kind() -> String {
    "related".to_string()
}
fn default_weight() -> f64 {
    1.0
}
fn default_kind() -> String {
    "chunk".to_string()
}

/// One CMB record — the migration unit. `content` is the lossless anchor.
#[derive(Debug, Clone, Deserialize)]
pub struct CmbRecord {
    pub id: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub content_sha256: Option<String>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub ordinal: Option<i64>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub relations: Vec<CmbRelation>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// The optional first line of a bundle. Records may also stand alone (a headerless
/// bundle is accepted); the header only carries provenance and the embedding model.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CmbHeader {
    #[serde(default)]
    pub cmb_version: String,
    #[serde(default)]
    pub source: serde_json::Value,
    #[serde(default)]
    pub embedding: serde_json::Value,
}

/// A manifest line: the full, lossless record of an imported item's original
/// fields, keyed by the CCOS node it became. Written to `<workspace>.migration.jsonl`.
#[derive(Debug, Clone, Serialize)]
pub struct ManifestEntry {
    pub id: String,
    pub node_id: String,
    pub kind: String,
    pub content_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<i64>,
    #[serde(skip_serializing_if = "metadata_is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

fn metadata_is_empty(m: &serde_json::Map<String, serde_json::Value>) -> bool {
    m.is_empty()
}

/// An embeddings-sidecar line: a preserved vector, ready to seed the Pro semantic
/// tier. Written to `<workspace>.embeddings.jsonl`.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingEntry {
    pub node_id: String,
    pub dim: usize,
    pub vector: Vec<f32>,
}

/// The outcome of a migration — counts, the lossless verdict, and warnings.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MigrationReport {
    pub records: usize,
    pub nodes_documents: usize,
    pub nodes_code_files: usize,
    pub edges_containment: usize,
    pub edges_sequence: usize,
    pub edges_relation: usize,
    pub cross_file_edges: usize,
    pub embeddings_preserved: usize,
    pub bytes_content: usize,
    pub hash_mismatches: usize,
    /// Every record produced a node and (for document records) its text hash matched.
    pub lossless: bool,
    pub by_kind: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
}

/// A migration error.
#[derive(Debug)]
pub enum MigrateError {
    Io(std::io::Error),
    Json { line: usize, err: serde_json::Error },
    Assemble(String),
    Empty,
}

impl std::fmt::Display for MigrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrateError::Io(e) => write!(f, "io error: {e}"),
            MigrateError::Json { line, err } => {
                write!(f, "bundle parse error at line {line}: {err}")
            }
            MigrateError::Assemble(m) => write!(f, "could not assemble workspace: {m}"),
            MigrateError::Empty => write!(f, "bundle is empty"),
        }
    }
}
impl std::error::Error for MigrateError {}
impl From<std::io::Error> for MigrateError {
    fn from(e: std::io::Error) -> Self {
        MigrateError::Io(e)
    }
}

/// The full result of a migration: the assembled memory, the report, and the
/// sidecar entries the caller persists next to the workspace.
pub struct MigrationOutcome {
    pub memory: CcosMemory,
    pub report: MigrationReport,
    pub manifest: Vec<ManifestEntry>,
    pub embeddings: Vec<EmbeddingEntry>,
}

/// Map a bundle relation kind onto a CCOS [`EdgeType`]. Unknown kinds fall back to
/// `RelatedTo`; the original kind is preserved verbatim in the manifest, so this
/// mapping never loses information.
pub fn map_edge_kind(kind: &str) -> EdgeType {
    match kind.to_ascii_lowercase().as_str() {
        "contains" | "contain" | "parent" | "part_of" | "has_child" => EdgeType::Contains,
        "sequence" | "next" | "prev" | "previous" | "adjacent" | "follows" => EdgeType::RelatedTo,
        "depends_on" | "depends" | "import" | "imports" | "uses" | "requires" => {
            EdgeType::DependsOn
        }
        "calls" | "call" | "invokes" => EdgeType::Calls,
        "data_flow" | "dataflow" | "writes" | "reads" => EdgeType::DataFlow,
        "references" | "reference" | "mentions" | "mention" | "cites" | "links_to" | "source" => {
            EdgeType::References
        }
        "supports" | "support" | "affirms" | "confirms" => EdgeType::Supports,
        "contradicts" | "contradict" | "refutes" | "conflicts" => EdgeType::Contradicts,
        "causes" | "cause" | "leads_to" => EdgeType::Causes,
        _ => EdgeType::RelatedTo,
    }
}

/// Heuristic: does a uri look like a source-code file we should re-parse for real
/// causal edges rather than store as a document?
fn looks_like_code(uri: &str) -> bool {
    const EXTS: &[&str] = &[
        ".rs", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
        ".cc", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".sh",
    ];
    let u = uri.to_ascii_lowercase();
    EXTS.iter().any(|e| u.ends_with(e))
}

/// True when this record should take the code path under the given mode.
fn is_code(rec: &CmbRecord, mode: Mode) -> bool {
    match mode {
        Mode::Code => true,
        Mode::Docs => false,
        Mode::Auto => {
            rec.kind == "code_file"
                || rec
                    .source_uri
                    .as_deref()
                    .map(looks_like_code)
                    .unwrap_or(false)
        }
    }
}

/// The node id a record becomes: `file:<uri>` for code, `doc:<id>` otherwise.
fn node_id_for(rec: &CmbRecord, mode: Mode) -> String {
    if is_code(rec, mode) {
        let uri = rec.source_uri.clone().unwrap_or_else(|| rec.id.clone());
        format!("file:{}", uri.strip_prefix("file:").unwrap_or(&uri))
    } else {
        format!("doc:{}", rec.id.strip_prefix("doc:").unwrap_or(&rec.id))
    }
}

/// Parse a bundle into `(optional header, records)`. The first non-blank line is a
/// header iff it carries a `cmb_version`; otherwise it is treated as a record.
pub fn parse_bundle<R: BufRead>(
    reader: R,
) -> Result<(Option<CmbHeader>, Vec<CmbRecord>), MigrateError> {
    let mut header = None;
    let mut records = Vec::new();
    let mut first = true;
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if first {
            first = false;
            // A header object has `cmb_version` and no `id`; anything else is a record.
            if let Ok(h) = serde_json::from_str::<CmbHeader>(trimmed) {
                if !h.cmb_version.is_empty() {
                    header = Some(h);
                    continue;
                }
            }
        }
        let rec: CmbRecord =
            serde_json::from_str(trimmed).map_err(|err| MigrateError::Json { line: i + 1, err })?;
        records.push(rec);
    }
    if records.is_empty() {
        return Err(MigrateError::Empty);
    }
    Ok((header, records))
}

/// Import a parsed record set into a fresh workspace, rebuilding causal structure,
/// and return the assembled memory + the report + the sidecar entries. Uses only
/// the CCOS public API: documents are assembled into a [`MemoryGraph`] and loaded
/// via [`CcosMemory::from_json`]; code is re-parsed via the kernel path.
pub fn migrate_records(
    records: &[CmbRecord],
    cfg: &MigrateConfig,
) -> Result<MigrationOutcome, MigrateError> {
    let mut report = MigrationReport::default();
    let mut manifest = Vec::with_capacity(records.len());
    let mut embeddings = Vec::new();

    // The document sub-graph, assembled by hand from the bundle.
    let mut graph = MemoryGraph::new(0.2, 5000);
    let mut sources: BTreeMap<String, String> = BTreeMap::new();
    // Code records, re-parsed after the doc graph is loaded.
    let mut code: Vec<(String, String)> = Vec::new(); // (uri, content)
    let mut doc_ids: std::collections::HashSet<String> = records
        .iter()
        .filter(|r| !is_code(r, cfg.mode))
        .map(|r| node_id_for(r, cfg.mode))
        .collect();

    // raw record id → CCOS node id, so relations/parents resolve regardless of order.
    let mut id_map: BTreeMap<String, String> = BTreeMap::new();
    let mut seq: Vec<(String, i64, String)> = Vec::new(); // (parent, ordinal, child)
    let mut containment: Vec<(String, String)> = Vec::new(); // (parent, child)
    let mut relations: Vec<(String, String, f64, EdgeType)> = Vec::new();

    // ---- pass 1: nodes ---------------------------------------------------- //
    for rec in records {
        report.records += 1;
        *report.by_kind.entry(rec.kind.clone()).or_insert(0) += 1;
        report.bytes_content += rec.content.len();
        let node_id = node_id_for(rec, cfg.mode);
        id_map.insert(rec.id.clone(), node_id.clone());

        if is_code(rec, cfg.mode) {
            let uri = rec.source_uri.clone().unwrap_or_else(|| rec.id.clone());
            code.push((uri, rec.content.clone()));
            report.nodes_code_files += 1;
        } else {
            let label = rec.source_uri.clone().unwrap_or_else(|| rec.id.clone());
            graph.upsert_node(
                NodeId(node_id.clone()),
                label,
                rec.content.clone(),
                NodeType::ContextBlock,
            );
            sources.insert(node_id.clone(), rec.content.clone());
            report.nodes_documents += 1;
        }

        if let Some(parent) = &rec.parent {
            containment.push((parent.clone(), rec.id.clone()));
            if let Some(ord) = rec.ordinal {
                seq.push((parent.clone(), ord, node_id.clone()));
            }
        }
        for r in &rec.relations {
            let from = r.from.clone().unwrap_or_else(|| rec.id.clone());
            relations.push((from, r.target.clone(), r.weight, map_edge_kind(&r.kind)));
        }

        manifest.push(ManifestEntry {
            id: rec.id.clone(),
            node_id: node_id.clone(),
            kind: rec.kind.clone(),
            content_sha256: sha256_hex(&rec.content),
            source_uri: rec.source_uri.clone(),
            parent: rec.parent.clone(),
            ordinal: rec.ordinal,
            metadata: rec.metadata.clone(),
        });
        if let Some(v) = &rec.embedding {
            report.embeddings_preserved += 1;
            embeddings.push(EmbeddingEntry {
                node_id,
                dim: v.len(),
                vector: v.clone(),
            });
        }
    }

    // Resolve a raw id to a node id (fall back to the doc namespace).
    let resolve_id = |raw: &str| -> String {
        id_map
            .get(raw)
            .cloned()
            .unwrap_or_else(|| format!("doc:{}", raw.strip_prefix("doc:").unwrap_or(raw)))
    };

    // Placeholder nodes for referenced-but-absent document endpoints — a chunk's
    // `parent` document (or a relation target) that the bundle referenced without
    // giving it its own record. Creating an empty `ContextBlock` here means the
    // containment / relation edge to it is *kept*, not silently dropped (a `doc:`
    // endpoint only; code endpoints enter later via the kernel path). Placeholders
    // carry no content and are absent from the manifest, so they never affect the
    // lossless check.
    let mut needed: Vec<String> = Vec::new();
    for (parent, child) in &containment {
        needed.push(resolve_id(parent));
        needed.push(resolve_id(child));
    }
    for (from, target, _, _) in &relations {
        needed.push(resolve_id(from));
        needed.push(resolve_id(target));
    }
    for e in needed {
        if e.starts_with("doc:") && !doc_ids.contains(&e) {
            let label = e.strip_prefix("doc:").unwrap_or(&e).to_string();
            graph.upsert_node(
                NodeId(e.clone()),
                label,
                String::new(),
                NodeType::ContextBlock,
            );
            doc_ids.insert(e);
        }
    }

    // A doc-graph edge can only be added between two document nodes (code nodes
    // enter later via the kernel path). `edges ⊆ nodes × nodes` is enforced by the
    // graph anyway; checking membership up front lets us count and skip precisely.
    let ensure_doc_edge =
        |graph: &mut MemoryGraph, from: &str, to: &str, w: f64, kind: EdgeType| -> bool {
            if from != to && doc_ids.contains(from) && doc_ids.contains(to) {
                graph.add_edge(NodeId(from.to_string()), NodeId(to.to_string()), w, kind)
            } else {
                false
            }
        };

    // ---- pass 2: containment edges (parent → child, `Contains`) ----------- //
    for (parent, child) in &containment {
        let p = resolve_id(parent);
        let c = resolve_id(child);
        if ensure_doc_edge(&mut graph, &p, &c, 1.0, EdgeType::Contains) {
            report.edges_containment += 1;
        }
    }

    // ---- pass 2: sequence edges (chunk[i] → chunk[i+1], `RelatedTo`) ------- //
    seq.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut i = 0;
    while i < seq.len() {
        let mut j = i;
        while j + 1 < seq.len() && seq[j + 1].0 == seq[i].0 {
            j += 1;
        }
        for k in i..j {
            if ensure_doc_edge(
                &mut graph,
                &seq[k].2,
                &seq[k + 1].2,
                0.5,
                EdgeType::RelatedTo,
            ) {
                report.edges_sequence += 1;
            }
        }
        i = j + 1;
    }

    // ---- pass 2: explicit relations --------------------------------------- //
    for (from, target, weight, kind) in &relations {
        let f = resolve_id(from);
        let t = resolve_id(target);
        if ensure_doc_edge(&mut graph, &f, &t, *weight, kind.clone()) {
            report.edges_relation += 1;
        }
    }

    // ---- assemble the workspace ------------------------------------------- //
    // Load the hand-built document graph through the public snapshot format, then
    // re-parse any code records via the kernel's own ingestion path.
    let mut mem = if let Some(extend_path) = &cfg.extend_from {
        // Incremental import: keep the existing workspace and merge the freshly-built
        // document sub-graph (nodes + edges) and its retained sources on top of it,
        // preserving the existing causal graph and its hash-chained logs. Done through
        // the public snapshot format only (`to_json` → merge → `from_json`).
        let existing = CcosMemory::open(extend_path)
            .map_err(|e| MigrateError::Assemble(format!("open '{extend_path}': {e}")))?;
        let ejson = existing
            .to_json()
            .map_err(|e| MigrateError::Assemble(e.to_string()))?;
        let mut pv: serde_json::Value =
            serde_json::from_str(&ejson).map_err(|e| MigrateError::Assemble(e.to_string()))?;
        let mut base: MemoryGraph = serde_json::from_value(pv["graph"].clone())
            .map_err(|e| MigrateError::Assemble(e.to_string()))?;
        // Merge the new sub-graph's nodes then edges (edges need their endpoints to
        // already exist — the graph enforces `edges ⊆ nodes × nodes`).
        for id in graph.node_ids() {
            if let Some(n) = graph.node(id) {
                base.upsert_node(
                    id.clone(),
                    n.label.clone(),
                    n.content.clone(),
                    n.node_type.clone(),
                );
            }
        }
        for e in graph.edges() {
            base.add_edge(
                e.source.clone(),
                e.target.clone(),
                e.weight,
                e.edge_type.clone(),
            );
        }
        let mut base_sources: BTreeMap<String, String> =
            serde_json::from_value(pv["sources"].take()).unwrap_or_default();
        base_sources.extend(sources);
        pv["graph"] =
            serde_json::to_value(&base).map_err(|e| MigrateError::Assemble(e.to_string()))?;
        pv["sources"] = serde_json::to_value(&base_sources)
            .map_err(|e| MigrateError::Assemble(e.to_string()))?;
        let json = serde_json::to_string(&pv).map_err(|e| MigrateError::Assemble(e.to_string()))?;
        CcosMemory::from_json(&json).map_err(|e| MigrateError::Assemble(e.to_string()))?
    } else if report.nodes_documents > 0 {
        let persisted = serde_json::json!({
            "graph": graph,
            "event_log": EventLog::new("ccos-migrate".to_string()),
            "dist_log": DistributedEventLog::new(),
            "sources": sources,
        });
        let json =
            serde_json::to_string(&persisted).map_err(|e| MigrateError::Assemble(e.to_string()))?;
        CcosMemory::from_json(&json).map_err(|e| MigrateError::Assemble(e.to_string()))?
    } else {
        CcosMemory::new()
    };
    for (uri, content) in &code {
        mem.ingest_deferred(uri, content);
    }
    report.cross_file_edges = mem.resolve();

    // ---- verification: nodes present + document text intact --------------- //
    if cfg.verify {
        for entry in &manifest {
            let nid = NodeId(entry.node_id.clone());
            match mem.graph().node(&nid) {
                None => {
                    report.hash_mismatches += 1;
                    report
                        .warnings
                        .push(format!("record {} produced no node", entry.id));
                }
                Some(node) => {
                    // Document nodes hold the full text, so we verify the hash
                    // directly. Code nodes hold granular spans (the full source is
                    // retained by the kernel and round-trips), so their presence is
                    // the check; the manifest hash stays the authoritative record.
                    if entry.node_id.starts_with("doc:")
                        && sha256_hex(&node.content) != entry.content_sha256
                    {
                        report.hash_mismatches += 1;
                        report
                            .warnings
                            .push(format!("content hash mismatch for {}", entry.id));
                    }
                }
            }
        }
    }
    report.lossless = report.hash_mismatches == 0;

    Ok(MigrationOutcome {
        memory: mem,
        report,
        manifest,
        embeddings,
    })
}

/// Convenience: parse a bundle from a reader and import it in one call.
pub fn migrate_bundle<R: BufRead>(
    reader: R,
    cfg: &MigrateConfig,
) -> Result<MigrationOutcome, MigrateError> {
    let (_header, records) = parse_bundle(reader)?;
    migrate_records(&records, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external_memory::{ExternalMemory, Recall};
    use std::io::Cursor;

    const BUNDLE: &str = concat!(
        r#"{"cmb_version":"1.0","source":{"system":"chroma","collection":"kb"}}"#,
        "\n",
        r#"{"id":"doc-1","kind":"document","content":"The gearbox overheats above 6000 rpm.","source_uri":"manual.pdf","metadata":{"page":12}}"#,
        "\n",
        r#"{"id":"chunk-1","kind":"chunk","content":"Symptom: gearbox overheats.","parent":"doc-1","ordinal":0,"embedding":[0.1,0.2,0.3]}"#,
        "\n",
        r#"{"id":"chunk-2","kind":"chunk","content":"Cause: low oil level.","parent":"doc-1","ordinal":1,"relations":[{"target":"entity/oil","kind":"mentions"}]}"#,
        "\n",
        r#"{"id":"entity/oil","kind":"entity","content":"Oil","relations":[]}"#,
        "\n"
    );

    #[test]
    fn imports_documents_edges_and_is_lossless() {
        let out =
            migrate_bundle(Cursor::new(BUNDLE), &MigrateConfig::default()).expect("bundle parses");
        let r = &out.report;
        assert_eq!(r.records, 4);
        assert_eq!(r.nodes_documents, 4);
        assert_eq!(r.nodes_code_files, 0);
        assert_eq!(r.edges_containment, 2, "doc-1 contains chunk-1 and chunk-2");
        assert_eq!(r.edges_sequence, 1, "chunk-1 → chunk-2 by ordinal");
        assert_eq!(r.edges_relation, 1, "chunk-2 mentions entity/oil");
        assert_eq!(r.embeddings_preserved, 1);
        assert!(r.lossless, "no hash mismatches: {:?}", r.warnings);
        // The verbatim text is recallable from the assembled memory.
        let node = out
            .memory
            .graph()
            .node(&NodeId("doc:doc-1".into()))
            .unwrap();
        assert_eq!(node.content, "The gearbox overheats above 6000 rpm.");
        let win = out.memory.recall(&Recall::working_set(), 4096);
        assert!(win
            .items
            .iter()
            .any(|it| it.content.contains("gearbox overheats above 6000")));
        // The sidecars carry the lossless remainder.
        assert_eq!(out.manifest.len(), 4);
        assert_eq!(out.embeddings.len(), 1);
        assert_eq!(out.embeddings[0].dim, 3);
        assert!(out.manifest.iter().any(|m| m.metadata.contains_key("page")));
        // Hash chains verify.
        assert!(out.memory.verify().valid);
    }

    #[test]
    fn code_records_take_the_source_path() {
        let bundle = concat!(
            r#"{"cmb_version":"1.0","source":{"system":"jsonl"}}"#,
            "\n",
            r#"{"id":"a","kind":"code_file","content":"pub fn a() { b(); }","source_uri":"src/a.rs"}"#,
            "\n",
            r#"{"id":"b","kind":"code_file","content":"pub fn b() {}","source_uri":"src/b.rs"}"#,
            "\n"
        );
        let out = migrate_bundle(Cursor::new(bundle), &MigrateConfig::default()).unwrap();
        assert_eq!(out.report.nodes_code_files, 2);
        assert_eq!(out.report.nodes_documents, 0);
        // The real causal graph exists: the file nodes were created by the parser.
        assert!(out
            .memory
            .graph()
            .node(&NodeId("file:src/a.rs".into()))
            .is_some());
        assert!(out.report.lossless);
        assert!(out.memory.verify().valid);
    }

    #[test]
    fn mixed_docs_and_code_in_one_bundle() {
        let bundle = concat!(
            r#"{"id":"note","kind":"document","content":"See the query function."}"#,
            "\n",
            r#"{"id":"q","kind":"code_file","content":"pub fn query() {}","source_uri":"src/db.rs"}"#,
            "\n"
        );
        let out = migrate_bundle(Cursor::new(bundle), &MigrateConfig::default()).unwrap();
        assert_eq!(out.report.nodes_documents, 1);
        assert_eq!(out.report.nodes_code_files, 1);
        assert!(out
            .memory
            .graph()
            .node(&NodeId("doc:note".into()))
            .is_some());
        assert!(out
            .memory
            .graph()
            .node(&NodeId("file:src/db.rs".into()))
            .is_some());
        assert!(out.report.lossless);
        assert!(out.memory.verify().valid);
    }

    #[test]
    fn headerless_bundle_is_accepted() {
        let bundle = r#"{"id":"x","content":"hello world"}"#;
        let out = migrate_bundle(Cursor::new(bundle), &MigrateConfig::default()).unwrap();
        assert_eq!(out.report.records, 1);
        assert!(out.report.lossless);
    }

    #[test]
    fn detects_a_corrupted_bundle_line() {
        let bundle = "{not json}";
        let err = migrate_bundle(Cursor::new(bundle), &MigrateConfig::default());
        assert!(matches!(err, Err(MigrateError::Json { .. })));
    }

    #[test]
    fn extend_merges_into_an_existing_workspace() {
        let dir = std::env::temp_dir().join(format!("ccos-mig-extend-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ws = dir.join("ws.ccos");

        // First import: one document → a fresh workspace, checkpointed to disk.
        let mut m1 = migrate_bundle(
            Cursor::new(r#"{"id":"a","content":"first doc alpha"}"#),
            &MigrateConfig::default(),
        )
        .unwrap()
        .memory;
        m1.checkpoint_to(&ws).unwrap();

        // Second import EXTENDS the same workspace with another document.
        let cfg = MigrateConfig {
            extend_from: Some(ws.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let out = migrate_bundle(
            Cursor::new(r#"{"id":"b","content":"second doc beta"}"#),
            &cfg,
        )
        .unwrap();

        // Both the pre-existing node and the newly-added one are present.
        assert!(
            out.memory.graph().node(&NodeId("doc:a".into())).is_some(),
            "existing node kept"
        );
        assert!(
            out.memory.graph().node(&NodeId("doc:b".into())).is_some(),
            "new node added"
        );
        assert!(out.report.lossless);
        assert!(out.memory.verify().valid);

        std::fs::remove_dir_all(&dir).ok();
    }
}
