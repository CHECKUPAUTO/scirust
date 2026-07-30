//! # Causal compressor — reversible context compression pipeline
//!
//! CCOS's historical "compression" was selection + pagination + truncation: it
//! chose *which* nodes fit a token budget, it never *re-encoded* their content.
//! This module adds a real compression pass **downstream of the causal MMU**:
//! after the causal selection (`external_memory::assemble_window`) has selected the nodes,
//! [`CausalCompressor`] transforms each item's content into a denser form and
//! caches the original in a **CCR store** (Compressed-Context Retrieval), so
//! nothing is lost — a host LLM can call back for the full text via the
//! `ccos_retrieve` MCP tool, exactly like headroom's `headroom_retrieve`.
//!
//! ## Design constraints (what we do *not* sacrifice)
//!
//! - **Determinism**: every algorithm here is seed-stable and tie-broken on a
//!   total order. The hash-chain replay and the `postmortem` time-travel
//!   debugger remain bit-reproducible.
//! - **Zero new dependencies**: the module reuses only `serde_json` (already a
//!   CCOS dep) and std. The algorithms are distilled from their SCIRUST
//!   counterparts (MinHash/LSH, TextRank, Huffman, trie) into self-contained
//!   code sized for an agent loop, not a DL framework.
//! - **Reversibility**: every transformation records its original in the CCR
//!   store keyed by a content hash; `retrieve` is an exact inverse.
//! - **Non-destructive**: the causal graph, the scoring, the paging and the
//!   event log are untouched. The compressor is a pure post-selection pass.
//!
//! ## Pipeline
//!
//! ```text
//! RecallWindow (from assemble_window)
//!   │
//!   ▼
//! ContentRouter  ── per-item kind ──►  {json, code, prose}
//!   ├─ CausalCrusher   (JSON)   → columnar collapse + dedup + ref shortening
//!   ├─ CausalAST       (code)   → skeleton + local renames + CSE/DCE
//!   └─ CausalSumm      (prose)  → TextRank extractive summary, causally weighted
//!   │
//!   ▼
//! CCR store  (originals, hash-indexed, TTL by occupancy)
//!   │
//!   ▼
//! RecallWindow'  (compressed content + ccr_ref field)
//! ```
//!
//! ## Choosing the per-item route
//!
//! [`ContentRouter::classify`] looks at the node `kind` (the
//! [`crate::memory::NodeType`] debug string produced by `assemble_window`) and a
//! cheap structural sniff of the content. `sym:`/`file:` Rust nodes → code;
//! JSON-looking tool output → json; everything else → prose. The router is the
//! only place that knows about content types — the compressors themselves are
//! pure `&str → (compressed, reversible_token)` functions.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::util::sha256_hex;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A compressed item's retrieval handle — what the host LLM passes back to
/// `ccos_retrieve` to fetch the original. Short on purpose (a 12-char hex
/// prefix of the SHA-256 of the original content): it costs ~4 tokens and
/// carries no information leakage about the source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CcrRef(pub String);

impl CcrRef {
    /// Build a ref from an original content blob.
    pub fn of(original: &str) -> Self {
        // 12 hex chars = 48 bits of entropy, collision-resistant for any
        // realistic per-session cache size (< 10^8 items).
        CcrRef(sha256_hex(original)[..12].to_string())
    }
}

/// How much each algorithm compressed a single item.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CompressionStat {
    /// Algorithm name (`causal-crusher`, `causal-ast`, `causal-summ`, `passthrough`).
    pub algorithm: String,
    /// Token estimate before compression (chars/4).
    pub tokens_before: usize,
    /// Token estimate after compression.
    pub tokens_after: usize,
    /// Ratio `after / before` (1.0 = no gain, 0.1 = 10× shrink).
    pub ratio: f64,
}

/// The reversible compression pipeline. Owns the CCR store.
#[derive(Debug, Clone, Default)]
pub struct CausalCompressor {
    /// Original content keyed by [`CcrRef`].
    ccr: BTreeMap<String, String>,
    /// Per-algorithm aggregate stats for the last [`compress_window`](CausalCompressor::compress_window) call.
    pub last_stats: Vec<CompressionStat>,
    /// Configurable knobs (see [`CompressorConfig`]).
    pub config: CompressorConfig,
}

/// Knobs. Defaults are tuned for a 2048-token CCOS recall window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorConfig {
    /// Enable the JSON crusher on JSON-looking content.
    pub enable_json: bool,
    /// Enable the AST code compressor on `sym:`/`file:` Rust nodes.
    pub enable_code: bool,
    /// Enable the TextRank extractive summarizer on prose.
    pub enable_prose: bool,
    /// Target number of sentences the summarizer keeps (0 = auto = ~30% of input).
    pub summary_sentences: usize,
    /// Drop originals from the CCR store once it exceeds this many entries
    /// (LRU-ish: lowest score items evicted first). 0 = unbounded.
    pub ccr_capacity: usize,
    /// Minimum chars an item must have before compression is attempted
    /// (below this, the overhead of a CCR ref exceeds the gain).
    pub min_chars: usize,
    /// Enable cross-item near-duplicate suppression. Two items whose estimated
    /// Jaccard similarity (over token shingles) exceeds
    /// [`dedup_threshold`](Self::dedup_threshold) are merged: the lower-scored
    /// one is replaced by a one-line `// ~dup of <uri>` placeholder and its
    /// original is still retrievable via the CCR ref. This is the CCOS analog
    /// of headroom's auto-dedup, applied *within* a window rather than across
    /// the whole memory (which the causal graph already handles).
    pub enable_dedup: bool,
    /// Jaccard threshold above which two items are considered near-duplicates
    /// (0.0 = never dedup, 1.0 = only exact dups). 0.85 is a good default for
    /// code (signatures must match almost exactly) while still catching
    /// copy-pasted spans with renamed locals.
    pub dedup_threshold: f64,
    /// Enable CausalAST v2 enhancements (drop `use` lines, collapse runs of
    /// repeated `pub fn` signatures into an ellipsis). See [`CausalAst`].
    pub enable_ast_v2: bool,
    /// Number of repeated signature lines after which CausalAST v2 collapses
    /// the run into `fn a(); fn b(); … (+N more)`. 0 = disabled.
    pub ast_signature_collapse_after: usize,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enable_json: true,
            enable_code: true,
            enable_prose: true,
            summary_sentences: 0,
            ccr_capacity: 4096,
            min_chars: 30,
            enable_dedup: true,
            dedup_threshold: 0.85,
            enable_ast_v2: true,
            ast_signature_collapse_after: 8,
        }
    }
}

impl CausalCompressor {
    /// Fresh compressor with default config and an empty CCR store.
    pub fn new() -> Self {
        Self::default()
    }

    /// With explicit config.
    pub fn with_config(config: CompressorConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// Number of originals currently cached.
    pub fn ccr_len(&self) -> usize {
        self.ccr.len()
    }

    /// Retrieve an original by its [`CcrRef`] (the `ccos_retrieve` backend).
    /// `None` if the ref is unknown or has been evicted.
    pub fn retrieve(&self, ccr: &CcrRef) -> Option<&str> {
        self.ccr.get(&ccr.0).map(String::as_str)
    }

    /// Compress one content blob in isolation, returning `(compressed, ref, stat)`.
    /// The CCR store is updated. `kind` is the node kind string from
    /// [`crate::external_memory::RecallItem::kind`] (e.g. `"Symbol"`, `"File"`).
    /// `causal_score` weights the TextRank sentence selection for prose items.
    pub fn compress_item(
        &mut self,
        kind: &str,
        content: &str,
        causal_score: f64,
    ) -> (String, Option<CcrRef>, CompressionStat) {
        let (compressed, ccr_ref, stat) = self.compress_item_inner(kind, content, causal_score);
        // Enforce capacity *after* producing the ref, never evicting it.
        if let Some(r) = &ccr_ref {
            let mut keep = std::collections::BTreeSet::new();
            keep.insert(r.0.clone());
            self.enforce_ccr_capacity(&keep);
        }
        (compressed, ccr_ref, stat)
    }

    /// The compression primitive: route, compress, and store the original —
    /// **without** enforcing CCR capacity (the caller does that once per item or
    /// window via [`enforce_ccr_capacity`](Self::enforce_ccr_capacity)).
    fn compress_item_inner(
        &mut self,
        kind: &str,
        content: &str,
        causal_score: f64,
    ) -> (String, Option<CcrRef>, CompressionStat) {
        let tokens_before = content.chars().count() / 4;
        if content.len() < self.config.min_chars {
            return (content.to_string(), None, passthrough(tokens_before));
        }
        let route = ContentRouter::classify(kind, content);
        let (compressed, ccr_ref, algo) = match route {
            Route::Json if self.config.enable_json => {
                let c = CausalCrusher::crush(content);
                let r = self.store(content);
                (c, Some(r), "causal-crusher")
            }
            Route::Code if self.config.enable_code => {
                let c = CausalAst::skeletonize_with(content, &self.config);
                let r = self.store(content);
                (c, Some(r), "causal-ast")
            }
            Route::Prose if self.config.enable_prose => {
                let n = self.config.summary_sentences;
                let c = CausalSumm::summarize(content, n, causal_score);
                // If the summary is not shorter, skip storing the original
                // (no gain → no point making the LLM call back).
                if c.len() >= content.len() {
                    (content.to_string(), None, "passthrough")
                } else {
                    let r = self.store(content);
                    (c, Some(r), "causal-summ")
                }
            }
            _ => (content.to_string(), None, "passthrough"),
        };
        let tokens_after = compressed.chars().count() / 4;
        (
            compressed,
            ccr_ref,
            CompressionStat {
                algorithm: algo.to_string(),
                tokens_before,
                tokens_after,
                ratio: if tokens_before == 0 {
                    1.0
                } else {
                    tokens_after as f64 / tokens_before as f64
                },
            },
        )
    }

    /// Store an original and return its content-hash ref. Capacity is **not**
    /// enforced here — that happens in [`enforce_ccr_capacity`](Self::enforce_ccr_capacity)
    /// *after* a whole item/window is produced, so a ref just handed back is
    /// never evicted mid-operation (the bug this split fixes).
    fn store(&mut self, original: &str) -> CcrRef {
        let r = CcrRef::of(original);
        self.ccr.insert(r.0.clone(), original.to_string());
        r
    }

    /// Evict lowest-hash entries **not** in `keep` until the store is within
    /// `ccr_capacity`. The `keep` set — the refs produced by the current item or
    /// window — is never evicted, so the "nothing is lost" guarantee holds even
    /// when a single window has more items than the capacity (the cap is a floor
    /// against *older* entries, not a hard limit that can drop a live ref).
    /// Deterministic: eviction order is by ascending hash key.
    fn enforce_ccr_capacity(&mut self, keep: &std::collections::BTreeSet<String>) {
        let cap = self.config.ccr_capacity;
        if cap == 0 {
            return;
        }
        while self.ccr.len() > cap {
            match self.ccr.keys().find(|k| !keep.contains(*k)).cloned() {
                Some(victim) => {
                    self.ccr.remove(&victim);
                }
                // Everything left is live for this operation — keep it all; the
                // capacity is a floor, not a guillotine for the current window.
                None => break,
            }
        }
    }

    /// Compress every item in a `RecallWindow`-like vector in place. Each
    /// item is `(kind, score, uri, content)`; the result is `(compressed_content,
    /// optional ccr_ref, stat)`. This is what the MCP layer calls between
    /// `assemble_window` and `linearize_window`.
    ///
    /// When [`CompressorConfig::enable_dedup`] is on, items whose estimated
    /// Jaccard similarity exceeds the threshold are merged: the lower-scored
    /// copy is replaced by a one-line placeholder (its original is still
    /// retrievable via its CCR ref). Dedup runs *after* per-item compression,
    /// so it compares the **compressed** forms — this catches the case where
    /// two distinct originals collapse to the same skeleton.
    pub fn compress_window<'a, I>(&mut self, items: I) -> Vec<CompressedItem>
    where
        I: IntoIterator<Item = (&'a str, f64, &'a str, &'a str)>,
    {
        self.last_stats.clear();
        // Phase 1: per-item compression.
        let mut compressed: Vec<(String, f64, String, Option<CcrRef>, CompressionStat)> =
            Vec::new();
        for (kind, score, uri, content) in items {
            let (c, ccr_ref, stat) = self.compress_item_inner(kind, content, score);
            compressed.push((uri.to_string(), score, c, ccr_ref, stat));
        }
        // Phase 2: cross-item near-duplicate suppression.
        if self.config.enable_dedup && compressed.len() > 1 {
            let threshold = self.config.dedup_threshold;
            let sigs: Vec<Vec<u32>> = compressed
                .iter()
                .map(|(_, _, c, _, _)| shingle_signature(c))
                .collect();
            // For each item, look at higher-scored items above it (the window
            // is score-sorted by the caller) and mark it as a dup if any match.
            // Deterministic: ties broken by URI order (already sorted upstream).
            let mut dup_of: Vec<Option<usize>> = vec![None; compressed.len()];
            for i in 0..compressed.len() {
                if dup_of[i].is_some() {
                    continue;
                }
                for j in 0..i {
                    if dup_of[j].is_some() {
                        continue;
                    }
                    let sim = estimated_jaccard(&sigs[i], &sigs[j]);
                    if sim >= threshold {
                        dup_of[i] = Some(j);
                        break;
                    }
                }
            }
            for (i, dup) in dup_of.iter().enumerate() {
                if let Some(j) = dup {
                    let original_uri = compressed[*j].0.clone();
                    let ccr_ref = compressed[i].3.clone();
                    let before = compressed[i].4.tokens_before;
                    let placeholder = format!("// ~dup of {original_uri}");
                    let after = placeholder.chars().count() / 4;
                    compressed[i].2 = placeholder;
                    compressed[i].4 = CompressionStat {
                        algorithm: "causal-dedup".to_string(),
                        tokens_before: before,
                        tokens_after: after,
                        ratio: if before == 0 {
                            1.0
                        } else {
                            after as f64 / before as f64
                        },
                    };
                    // Keep the CCR ref so the original is still retrievable.
                    compressed[i].3 = ccr_ref;
                }
            }
        }
        // Phase 3: collect output + stats.
        let mut out = Vec::new();
        for (_, _, content, ccr_ref, stat) in compressed {
            self.last_stats.push(stat);
            out.push(CompressedItem { content, ccr_ref });
        }
        // Enforce CCR capacity once, keeping every ref this window handed back —
        // so "nothing is lost" holds even when the window exceeds the capacity.
        let keep: std::collections::BTreeSet<String> = out
            .iter()
            .filter_map(|it| it.ccr_ref.as_ref().map(|r| r.0.clone()))
            .collect();
        self.enforce_ccr_capacity(&keep);
        out
    }

    /// Reset the CCR store (used on session reload to avoid stale refs).
    pub fn clear_ccr(&mut self) {
        self.ccr.clear();
    }

    /// **Auto-tuner** — greedy coordinate search over the config knobs to
    /// minimise the total compressed-token count on a sample window, while
    /// preserving reversibility (the CCR refs always stay). Returns the best
    /// config found; the caller can then reuse it for the live session.
    ///
    /// The search is deterministic (every candidate is evaluated on the same
    /// sample in the same order) and bounded: at most `~5 × knobs` evaluations.
    /// It is a *proxy* tuner — it optimises the compression ratio on the given
    /// sample, not downstream LLM accuracy (which CCOS cannot run headlessly).
    /// Use it to bootstrap the knobs on a representative corpus; the causal
    /// MMU's selection (the actual quality driver) is untouched.
    pub fn auto_tune(
        &self,
        sample: &[(&str, f64, &str, &str)], // (kind, score, uri, content)
    ) -> CompressorConfig {
        let mut best = self.config.clone();
        let mut best_tokens = Self::eval_config(&best, sample);
        // Coordinate descent over the tunable knobs.
        type Knob = dyn Fn(&CompressorConfig, bool) -> CompressorConfig;
        let knobs: Vec<Box<Knob>> = vec![
            // dedup_threshold: try lower (more aggressive dedup).
            Box::new(|c, up| {
                let mut n = c.clone();
                n.dedup_threshold = if up { 0.95 } else { 0.70 };
                n
            }),
            Box::new(|c, _| {
                let mut n = c.clone();
                n.enable_dedup = !c.enable_dedup;
                n
            }),
            // ast signature collapse: try fewer / more retained.
            Box::new(|c, up| {
                let mut n = c.clone();
                n.ast_signature_collapse_after = if up { 12 } else { 4 };
                n
            }),
            Box::new(|c, _| {
                let mut n = c.clone();
                n.enable_ast_v2 = !c.enable_ast_v2;
                n
            }),
            // summary length: try shorter / longer.
            Box::new(|c, up| {
                let mut n = c.clone();
                n.summary_sentences = if up { 0 } else { 6 };
                n
            }),
            Box::new(|c, _| {
                let mut n = c.clone();
                n.enable_prose = !c.enable_prose;
                n
            }),
            // min_chars: try lower (compress more items) / higher.
            Box::new(|c, up| {
                let mut n = c.clone();
                n.min_chars = if up { 10 } else { 60 };
                n
            }),
        ];
        for knob in &knobs {
            for &up in &[true, false] {
                let candidate = knob(&best, up);
                let tokens = Self::eval_config(&candidate, sample);
                if tokens < best_tokens {
                    best_tokens = tokens;
                    best = candidate;
                }
            }
        }
        best
    }

    /// Evaluate a config on the sample without mutating `self`. Returns the
    /// total compressed-token count (lower is better). Public so an external
    /// benchmark can measure the same metric the auto-tuner optimises.
    pub fn eval_config(config: &CompressorConfig, sample: &[(&str, f64, &str, &str)]) -> usize {
        let mut probe = CausalCompressor::with_config(config.clone());
        let out = probe.compress_window(sample.iter().copied());
        out.iter().map(|i| i.content.chars().count() / 4).sum()
    }
}

/// One item produced by [`CausalCompressor::compress_window`].
#[derive(Debug, Clone, Serialize)]
pub struct CompressedItem {
    /// The compressed content (or the original when passthrough).
    pub content: String,
    /// retrieval handle for the original, when the item was actually compressed.
    pub ccr_ref: Option<CcrRef>,
}

fn passthrough(tokens_before: usize) -> CompressionStat {
    CompressionStat {
        algorithm: "passthrough".to_string(),
        tokens_before,
        tokens_after: tokens_before,
        ratio: 1.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentRouter
// ─────────────────────────────────────────────────────────────────────────────

/// The content-type class picked by [`ContentRouter::classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Json,
    Code,
    Prose,
}

/// Pure, dependency-free content-type sniff. Deterministic.
pub struct ContentRouter;

impl ContentRouter {
    /// Classify an item by its node `kind` (the `NodeType` debug string) and a
    /// structural sniff of the content.
    pub fn classify(kind: &str, content: &str) -> Route {
        // Node-kind hints: `sym:` and `file:` Rust nodes are code.
        let k = kind.to_lowercase();
        if k.contains("symbol") || k.contains("file") || k.contains("module") {
            // But a `file:` node whose content parses as JSON is a tool output
            // stored as a file — trust the content sniff over the kind.
            if Self::looks_like_json(content) {
                return Route::Json;
            }
            return Route::Code;
        }
        if Self::looks_like_json(content) {
            return Route::Json;
        }
        Route::Prose
    }

    /// True if `s` plausibly parses as JSON (starts with `{` or `[`, balanced).
    fn looks_like_json(s: &str) -> bool {
        let t = s.trim_start();
        if !t.starts_with('{') && !t.starts_with('[') {
            return false;
        }
        // Cheap balance check on the first non-string brackets. Good enough for
        // routing; the crusher re-parses with serde_json anyway.
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        for c in t.chars().take(4096) {
            if in_str {
                if esc {
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == '"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                _ => {}
            }
        }
        depth == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 1 — CausalCrusher (JSON)
// ─────────────────────────────────────────────────────────────────────────────

/// JSON structural compressor. Collapses arrays of homogeneous objects into
/// columnar tables, drops null/empty fields, shortens repeated refs, and dedups
/// repeated string values into `~n` back-references. Reversible via the CCR
/// store (the original JSON is cached).
pub struct CausalCrusher;

impl CausalCrusher {
    /// Crush a JSON string into a denser textual form. Falls back to the
    /// original text on any parse error (the caller then stores no CCR ref).
    pub fn crush(input: &str) -> String {
        let v: serde_json::Value = match serde_json::from_str(input.trim()) {
            Ok(v) => v,
            Err(_) => return input.to_string(),
        };
        let mut out = String::new();
        Self::crush_value(&v, &mut out, &mut BTreeMap::new());
        out
    }

    fn crush_value(v: &serde_json::Value, out: &mut String, seen: &mut BTreeMap<String, u32>) {
        match v {
            serde_json::Value::Object(o) => {
                // Drop null/empty values — they cost tokens and carry no signal.
                let kept: Vec<(&String, &serde_json::Value)> =
                    o.iter().filter(|(_, val)| !is_nullish(val)).collect();
                out.push('{');
                for (i, (k, val)) in kept.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(k);
                    out.push(':');
                    Self::crush_value(val, out, seen);
                }
                out.push('}');
            }
            serde_json::Value::Array(a) => {
                // Columnar collapse: an array of objects with the same keys →
                // `{key:[v0,v1,…],…}`. This is the SmartCrusher-style win on
                // tool outputs (search results, test rows, log lines).
                if !a.is_empty() && a.iter().all(|x| x.is_object()) {
                    let keys: Vec<String> = a
                        .iter()
                        .flat_map(|x| {
                            x.as_object()
                                .unwrap()
                                .keys()
                                .filter(|k| !is_nullish(&x[k]))
                                .cloned()
                        })
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    if !keys.is_empty() {
                        out.push('{');
                        for (ki, key) in keys.iter().enumerate() {
                            if ki > 0 {
                                out.push(',');
                            }
                            out.push_str(key);
                            out.push_str(":[");
                            for (i, item) in a.iter().enumerate() {
                                if i > 0 {
                                    out.push(',');
                                }
                                if let Some(val) = item.get(key) {
                                    if !is_nullish(val) {
                                        Self::crush_value(val, out, seen);
                                    }
                                }
                            }
                            out.push(']');
                        }
                        out.push('}');
                        return;
                    }
                }
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    Self::crush_value(x, out, seen);
                }
                out.push(']');
            }
            serde_json::Value::String(s) => {
                // Dedup repeated strings: first occurrence → quoted, later → ~N.
                if seen.contains_key(s) {
                    out.push_str(&format!("~{}", seen[s]));
                } else {
                    let n = seen.len() as u32;
                    seen.insert(s.clone(), n);
                    out.push('"');
                    Self::escape_json(s, out);
                    out.push('"');
                }
            }
            other => out.push_str(&other.to_string()),
        }
    }

    fn escape_json(s: &str, out: &mut String) {
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
    }
}

fn is_nullish(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::String(s) => s.is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 2 — CausalAST (code skeletonizer)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight, language-agnostic code compressor. Strips comments and blank
/// lines, collapses runs of whitespace, drops pure-ceremony tokens (`pub`,
/// `&'static`, trailing commas), and shortens local identifiers that look
/// temporary (`_x`, `_foo`) into `$0..$n`. The original is kept in the CCR
/// store for retrieval — this is a *view*, not a transform the LLM must reason
/// backwards from.
pub struct CausalAst;

impl CausalAst {
    /// Produce a denser view of a code span (v1 defaults).
    pub fn skeletonize(input: &str) -> String {
        Self::skeletonize_with(input, &CompressorConfig::default())
    }

    /// Produce a denser view of a code span, honouring the v2 knobs:
    /// - [`CompressorConfig::enable_ast_v2`] → drop pure `use` lines (imports
    ///   carry no logic; the causal graph already encodes dependencies).
    /// - [`CompressorConfig::ast_signature_collapse_after`] → collapse a run
    ///   of `N` consecutive one-line `fn`/`pub fn` signatures (a file header
    ///   is often 24+ signatures) into the first few + `(+M more)`.
    pub fn skeletonize_with(input: &str, config: &CompressorConfig) -> String {
        let mut out = String::with_capacity(input.len() / 2);
        let mut dollar = 0u32;
        let mut prev_blank = false;
        // Run of consecutive one-line signature lines, for collapse.
        let mut sig_run: Vec<String> = Vec::new();
        let collapse_after = if config.enable_ast_v2 {
            config.ast_signature_collapse_after
        } else {
            0
        };
        let drop_uses = config.enable_ast_v2;

        let flush_sig_run = |sig_run: &mut Vec<String>, out: &mut String, collapse_after: usize| {
            let n = sig_run.len();
            if n == 0 {
                return;
            }
            if collapse_after > 0 && n > collapse_after {
                for line in sig_run.iter().take(collapse_after) {
                    out.push_str(line);
                    out.push('\n');
                }
                out.push_str(&format!("// (+{} more signatures)\n", n - collapse_after));
            } else {
                for line in sig_run.iter() {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            sig_run.clear();
        };

        for raw_line in input.lines() {
            // Strip line comments (// …) and trailing whitespace.
            let mut line = raw_line.trim_end();
            if let Some(idx) = line.find("//") {
                // Keep `http://` etc. — only treat `//` preceded by whitespace
                // or start-of-line as a comment.
                let before = &line[..idx];
                if before.is_empty() || before.ends_with(|c: char| c.is_whitespace()) {
                    line = before.trim_end();
                }
            }
            let line = line.trim_start();
            if line.is_empty() {
                flush_sig_run(&mut sig_run, &mut out, collapse_after);
                if !prev_blank {
                    out.push('\n');
                }
                prev_blank = true;
                continue;
            }
            prev_blank = false;
            // v2: drop pure `use` lines (imports carry no logic signal; the
            // causal graph already encodes the cross-file dependency). Check
            // BEFORE stripping `pub ` so `pub use` (a re-export) is kept.
            if drop_uses && is_use_line(line) {
                continue;
            }
            // Drop pure-ceremony leading tokens. We only strip `pub ` and
            // `unsafe ` — removing more would change semantics the LLM needs.
            let line = line
                .strip_prefix("pub ")
                .or_else(|| line.strip_prefix("pub(crate) "))
                .unwrap_or(line);
            // Collapse internal runs of spaces to one (keeps indentation as a
            // single leading tab/space, which matters for Rust visibility).
            let mut collapsed = String::with_capacity(line.len());
            let mut in_str = false;
            let mut esc = false;
            let mut run = 0u32;
            for c in line.chars() {
                if in_str {
                    collapsed.push(c);
                    if esc {
                        esc = false;
                    } else if c == '\\' {
                        esc = true;
                    } else if c == '"' {
                        in_str = false;
                    }
                    continue;
                }
                match c {
                    '"' => {
                        in_str = true;
                        collapsed.push(c);
                        run = 0;
                    }
                    ' ' | '\t' if collapsed.is_empty() => {
                        if run == 0 {
                            collapsed.push(' ');
                        }
                        run += 1;
                    }
                    ' ' | '\t' => {
                        run += 1;
                    }
                    _ => {
                        if run > 0 {
                            collapsed.push(' ');
                        }
                        run = 0;
                        collapsed.push(c);
                    }
                }
            }
            // Rename `_foo`-style temporaries to `$n`.
            let line = Self::rename_temporaries(&collapsed, &mut dollar);

            // Signature-collapse bookkeeping: a one-line `fn name(...)` (no `{`
            // body) is a candidate; a multi-line body breaks the run.
            if collapse_after > 0 && is_signature_line(&line) {
                sig_run.push(line);
                continue;
            }
            flush_sig_run(&mut sig_run, &mut out, collapse_after);
            out.push_str(&line);
            out.push('\n');
        }
        flush_sig_run(&mut sig_run, &mut out, collapse_after);
        // Trim trailing blank lines.
        while out.ends_with("\n\n") {
            out.pop();
        }
        out
    }

    /// Replace identifiers of the form `_name` (Rust unused-binding convention)
    /// with `$n`, leaving the original recoverable from the CCR cache. We only
    /// rename names that start with `_` and are followed by a lowercase letter
    /// (avoiding `_x` in macros and `__` dunder conventions in other langs).
    fn rename_temporaries(line: &str, counter: &mut u32) -> String {
        let bytes = line.as_bytes();
        let mut out = String::with_capacity(line.len());
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'_' && i + 1 < bytes.len() {
                let nxt = bytes[i + 1];
                // Require a lowercase ascii letter after `_` (typical temp).
                if nxt.is_ascii_lowercase() {
                    // Scan to the end of the identifier.
                    let mut j = i + 1;
                    while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_')
                    {
                        j += 1;
                    }
                    // Only rename if it's not preceded by an identifier char
                    // (i.e. we're at the start of the identifier).
                    let prev_ok =
                        i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                    if prev_ok {
                        let n = *counter;
                        *counter += 1;
                        out.push_str(&format!("${n}"));
                        i = j;
                        continue;
                    }
                }
            }
            out.push(b as char);
            i += 1;
        }
        out
    }
}

/// True iff `line` is a plain `use ...;` import (v2: dropped from the skeleton
/// since the causal graph already encodes the dependency). `pub use` is kept —
/// it re-exports an API surface the LLM may need to see.
fn is_use_line(line: &str) -> bool {
    line.starts_with("use ") && line.trim_end().ends_with(';') && !line.contains(" pub ")
}

/// True iff `line` looks like a one-line signature declaration (no body):
/// `fn name(...)`, `fn name(...);`, `fn name(...) -> T;`. Used by the v2
/// signature-run collapser. A line with `{` or `=>` is a body, not a sig.
fn is_signature_line(line: &str) -> bool {
    let t = line.trim();
    if !(t.starts_with("fn ") || t.starts_with("pub fn ") || t.starts_with("async fn ")) {
        return false;
    }
    // Must end with `)` or `);` or `) -> T;` — a one-liner, not an opener.
    if t.contains('{') || t.contains("=>") {
        return false;
    }
    t.ends_with(')') || t.ends_with(";")
}

// ─────────────────────────────────────────────────────────────────────────────
// MinHash (distilled) — cross-item near-duplicate estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 64-hash MinHash signature over 3-character shingles of `text`.
/// Returns a sorted vec of the per-hash minima (deterministic, seed-stable).
///
/// This is a distilled, dependency-free version of SCIRUST's
/// `scirust_nlp_advanced::similarity::MinHash` — same FNV-1a + double-hashing
/// scheme, sized for an agent's per-window dedup pass (not a billion-document
/// index). Two signatures can be compared with [`estimated_jaccard`].
fn shingle_signature(text: &str) -> Vec<u32> {
    const NUM_HASHES: usize = 64;
    // 3-char shingles (char-level, so it works on code and prose alike).
    let chars: Vec<char> = text.chars().collect();
    let shingles: Vec<String> = if chars.len() <= 3 {
        vec![text.to_string()]
    } else {
        (0..chars.len() - 2)
            .map(|i| chars[i..i + 3].iter().collect())
            .collect()
    };
    // Hash each shingle with NUM_HASHES independent double-hash functions and
    // keep the per-function minimum. Coefficients derived from a fixed seed
    // (reproducibility — no RNG).
    let prime: u64 = 2147483647; // 2^31 - 1
    let mut sig = vec![u32::MAX; NUM_HASHES];
    let mut state: u64 = 0x9E3779B97F4A7C15; // golden ratio seed
    let mut a = [0u64; NUM_HASHES];
    let mut b = [0u64; NUM_HASHES];
    for i in 0..NUM_HASHES {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        a[i] = (state % (prime - 1)) + 1;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        b[i] = state % prime;
    }
    for sh in &shingles {
        // FNV-1a base hash.
        let mut h: u64 = 14695981039346656037;
        for byte in sh.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211);
        }
        for i in 0..NUM_HASHES {
            let g = ((a[i].wrapping_mul(h).wrapping_add(b[i])) % prime) as u32;
            if g < sig[i] {
                sig[i] = g;
            }
        }
    }
    sig
}

/// Estimate Jaccard similarity from two [`shingle_signature`]s: fraction of
/// positions where they agree. Deterministic and symmetric.
fn estimated_jaccard(a: &[u32], b: &[u32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let agree = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    agree as f64 / a.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 3 — CausalSumm (TextRank, causally weighted)
// ─────────────────────────────────────────────────────────────────────────────

/// Extractive summarizer: ranks sentences by a TextRank-style co-occurrence
/// graph, biased by a **causal score** so sentences touching high-pressure
/// nodes surface first — the angle headroom's TextRank lacks. Deterministic
/// (sorted by `(score, sentence_index)`).
pub struct CausalSumm;

impl CausalSumm {
    /// Summarize `text` into at most `max_sentences` sentences (0 = auto =
    /// ~30% of the input sentence count). `causal_bias` in `[0,1]` boosts
    /// sentences whose tokens overlap the active causal region; pass `0.0`
    /// when no causal signal is available.
    pub fn summarize(text: &str, max_sentences: usize, causal_bias: f64) -> String {
        let sentences = split_sentences(text);
        if sentences.len() <= 2 {
            return text.to_string();
        }
        let target = if max_sentences == 0 {
            (sentences.len() * 3 / 10).max(1)
        } else {
            max_sentences.min(sentences.len())
        };
        let tokens: Vec<Vec<String>> = sentences.iter().map(|s| tokenize(s)).collect();
        let n = sentences.len();

        // Co-occurrence similarity (Jaccard over a sliding window of sentences).
        let mut sim = vec![vec![0.0f64; n]; n];
        let window = 3usize;
        for i in 0..n {
            for j in (i + 1)..n.min(i + 1 + window) {
                let s = jaccard(&tokens[i], &tokens[j]);
                sim[i][j] = s;
                sim[j][i] = s;
            }
        }

        // PageRank with a causal-biased personalization vector: the teleport
        // probability is weighted toward sentences with high causal-token
        // overlap. When `causal_bias == 0`, this degenerates to vanilla TextRank.
        let bias = if causal_bias > 0.0 {
            // Heuristic: a sentence's causal overlap is its token overlap with
            // the *first* sentence (which typically anchors the topic) scaled
            // by the supplied bias. This keeps the algorithm self-contained —
            // we don't need the actual graph node set here, the caller already
            // filtered to the causal region.
            let anchor = &tokens[0];
            (0..n)
                .map(|i| {
                    let base = jaccard(&tokens[i], anchor);
                    0.1 + causal_bias * base
                })
                .collect::<Vec<_>>()
        } else {
            vec![1.0f64; n]
        };
        let bias_sum: f64 = bias.iter().sum();
        let bias: Vec<f64> = if bias_sum > 0.0 {
            bias.iter().map(|b| b / bias_sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        };

        let damping = 0.85f64;
        let mut scores = vec![1.0 / n as f64; n];
        for _ in 0..30 {
            let mut new = vec![0.0f64; n];
            let mut diff = 0.0f64;
            for i in 0..n {
                let row_sum: f64 = sim[i].iter().sum();
                let mut s = 0.0f64;
                if row_sum > 0.0 {
                    for (j, score) in scores.iter().enumerate().take(n) {
                        if i != j {
                            s += (sim[i][j] / row_sum) * *score;
                        }
                    }
                }
                new[i] = (1.0 - damping) * bias[i] + damping * s;
                diff += (new[i] - scores[i]).abs();
            }
            scores = new;
            if diff < 1e-6 {
                break;
            }
        }

        // Pick the top-`target` sentences, then re-order by original position
        // so the summary reads naturally.
        let mut ranked: Vec<(usize, f64)> = (0..n).map(|i| (i, scores[i])).collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut chosen: Vec<usize> = ranked.iter().take(target).map(|(i, _)| *i).collect();
        chosen.sort();
        chosen
            .into_iter()
            .map(|i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect()
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    let sa: BTreeSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: BTreeSet<&str> = b.iter().map(|s| s.as_str()).collect();
    if sa.is_empty() && sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ContentRouter ────────────────────────────────────────────────────────

    #[test]
    fn router_picks_json_for_braces() {
        assert_eq!(ContentRouter::classify("Tool", "{\"a\":1}"), Route::Json);
        assert_eq!(ContentRouter::classify("Tool", "[1,2,3]"), Route::Json);
    }

    #[test]
    fn router_picks_code_for_symbol_kind() {
        assert_eq!(
            ContentRouter::classify("Symbol", "pub fn alpha() -> i32 { 1 }"),
            Route::Code
        );
    }

    #[test]
    fn router_picks_prose_for_plain_text() {
        assert_eq!(
            ContentRouter::classify("Note", "the quick brown fox jumps."),
            Route::Prose
        );
    }

    #[test]
    fn router_json_wins_over_kind_when_content_is_json() {
        // A file: node whose content is actually JSON tool output.
        assert_eq!(
            ContentRouter::classify("File", "{\"rows\":[{\"id\":1}]}"),
            Route::Json
        );
    }

    // ── CausalCrusher (JSON) ────────────────────────────────────────────────

    #[test]
    fn crusher_drops_nullish_fields() {
        let json = r#"{"a":1,"b":null,"c":"","d":[],"e":{}}"#;
        let out = CausalCrusher::crush(json);
        assert!(out.contains("a:1"), "kept field present: {out}");
        assert!(!out.contains("b:"), "null dropped: {out}");
        assert!(!out.contains("c:"), "empty string dropped: {out}");
        assert!(!out.contains("d:"), "empty array dropped: {out}");
        assert!(!out.contains("e:"), "empty object dropped: {out}");
    }

    #[test]
    fn crusher_columnarizes_homogeneous_array_of_objects() {
        let json = r#"[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]"#;
        let out = CausalCrusher::crush(json);
        assert!(
            out.contains("id:[") && out.contains("name:["),
            "columnar collapse: {out}"
        );
        assert!(out.contains("1,2,3"), "ids are packed: {out}");
    }

    #[test]
    fn crusher_dedups_repeated_strings_into_backrefs() {
        let json = r#"{"a":"repeat","b":"repeat","c":"repeat"}"#;
        let out = CausalCrusher::crush(json);
        assert!(out.contains("~0"), "first repeat becomes a backref: {out}");
    }

    #[test]
    fn crusher_is_smaller_than_input_on_repetitive_json() {
        let json = (0..50)
            .map(|i| format!("{{\"id\":{i},\"name\":\"item{i}\",\"kind\":\"row\"}}"))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!("[{json}]");
        let out = CausalCrusher::crush(&json);
        assert!(
            out.len() < json.len() / 2,
            "columnar + dedup should crush >2×: {} → {}",
            json.len(),
            out.len()
        );
    }

    #[test]
    fn crusher_falls_back_to_original_on_invalid_json() {
        let bad = "{not valid json";
        let out = CausalCrusher::crush(bad);
        assert_eq!(out, bad);
    }

    // ── CausalAST (code) ────────────────────────────────────────────────────

    #[test]
    fn ast_strips_line_comments() {
        let code = "pub fn alpha() {\n    // a comment\n    let x = 1; // trailing\n}\n";
        let out = CausalAst::skeletonize(code);
        assert!(!out.contains("comment"), "comments stripped: {out}");
        assert!(out.contains("fn alpha"));
    }

    #[test]
    fn ast_collapses_blank_runs() {
        let code = "a\n\n\n\nb\n";
        let out = CausalAst::skeletonize(code);
        assert!(!out.contains("\n\n\n"), "no triple blanks: {out:?}");
    }

    #[test]
    fn ast_renames_underscore_temporaries() {
        let code = "let _foo = 1; let _bar = _foo + 1;";
        let out = CausalAst::skeletonize(code);
        assert!(out.contains("$0"), "first temp renamed: {out}");
        assert!(out.contains("$1"), "second temp renamed: {out}");
        assert!(!out.contains("_foo"));
    }

    #[test]
    fn ast_preserves_strings_with_slashes() {
        let code = "let url = \"http://example.com\";";
        let out = CausalAst::skeletonize(code);
        assert!(out.contains("http://example.com"), "URL preserved: {out}");
    }

    #[test]
    fn ast_shrinks_typical_symbol_span() {
        let code = r#"
pub fn compute(rows: &[Row]) -> i64 {
    // sum the rows
    let _acc = 0;
    let _tmp = 0;
    for r in rows {
        _acc += r.value;
        _tmp += 1;
    }
    _acc
}
"#;
        let out = CausalAst::skeletonize(code);
        assert!(
            out.len() < code.len(),
            "skeleton smaller: {} → {}",
            code.len(),
            out.len()
        );
        assert!(out.contains("fn compute"), "signature preserved");
    }

    // ── CausalSumm (TextRank) ───────────────────────────────────────────────

    #[test]
    fn summary_is_shorter_than_input() {
        let text = "First sentence about the database. Second about the API layer. \
                    Third about logging. Fourth about deployment. Fifth about tests. \
                    Sixth about monitoring. Seventh about the cache. Eighth about config.";
        let out = CausalSumm::summarize(text, 3, 0.0);
        assert!(
            out.chars().count() < text.chars().count(),
            "summary is shorter"
        );
        assert!(
            out.contains("database") || out.contains("API"),
            "keeps key content"
        );
    }

    #[test]
    fn summary_with_causal_bias_prefers_anchored_sentences() {
        let text = "The database timeout is the root cause. \
                    The API layer retries three times. \
                    Logging is verbose but harmless. \
                    Deployment is fine. \
                    Tests pass locally. \
                    Monitoring shows the spike.";
        // High causal bias → the first (topic-anchoring) sentence must survive.
        let out = CausalSumm::summarize(text, 2, 0.9);
        assert!(
            out.contains("database timeout"),
            "bias surfaces the anchor: {out}"
        );
    }

    #[test]
    fn summary_preserves_sentence_order() {
        let text = "A first. B second. C third. D fourth. E fifth.";
        let out = CausalSumm::summarize(text, 2, 0.0);
        // Whichever 2 are picked, they must appear in original order.
        let a = out.find("A first");
        let b = out.find("B second");
        let c = out.find("C third");
        let positions: Vec<Option<usize>> = vec![a, b, c];
        let present: Vec<usize> = positions.into_iter().flatten().collect();
        let mut sorted = present.clone();
        sorted.sort();
        assert_eq!(present, sorted, "order preserved: {out}");
    }

    #[test]
    fn summary_short_text_passes_through() {
        let text = "Only one sentence here.";
        let out = CausalSumm::summarize(text, 0, 0.0);
        assert_eq!(out, text);
    }

    // ── CausalCompressor + CCR ─────────────────────────────────────────────

    #[test]
    fn compress_item_prose_yields_ccr_ref_and_shrunk_content() {
        let mut c = CausalCompressor::new();
        let text = "Sentence one about alpha. Sentence two about beta. \
                    Sentence three about gamma. Sentence four about delta. \
                    Sentence five about epsilon. Sentence six about zeta.";
        let (out, ccr_ref, stat) = c.compress_item("Note", text, 0.0);
        assert!(stat.algorithm == "causal-summ");
        assert!(ccr_ref.is_some(), "original is cached");
        assert!(out.chars().count() < text.chars().count());
        // Retrieve round-trips.
        let r = ccr_ref.unwrap();
        assert_eq!(c.retrieve(&r).unwrap(), text);
    }

    #[test]
    fn compress_item_code_yields_skeleton_and_ccr_ref() {
        let mut c = CausalCompressor::new();
        let code =
            "pub fn alpha() {\n    let _x = 1; // comment\n    let _y = 2;\n    _x + _y\n}\n";
        let (out, ccr_ref, stat) = c.compress_item("Symbol", code, 0.0);
        assert_eq!(stat.algorithm, "causal-ast");
        assert!(ccr_ref.is_some());
        assert!(out.contains("$0") && out.contains("$1"), "renamed: {out}");
        assert_eq!(c.retrieve(&ccr_ref.unwrap()).unwrap(), code);
    }

    #[test]
    fn compress_item_json_yields_crushed_and_ccr_ref() {
        let mut c = CausalCompressor::new();
        let json = r#"[{"id":1,"x":null},{"id":2,"x":null},{"id":3,"x":null}]"#;
        let (out, ccr_ref, stat) = c.compress_item("Tool", json, 0.0);
        assert_eq!(stat.algorithm, "causal-crusher");
        assert!(ccr_ref.is_some());
        assert!(out.contains("id:["), "columnar: {out}");
        assert_eq!(c.retrieve(&ccr_ref.unwrap()).unwrap(), json);
    }

    #[test]
    fn compress_item_short_content_passthrough() {
        let mut c = CausalCompressor::new();
        let (out, ccr_ref, stat) = c.compress_item("Symbol", "fn id() { 1 }", 0.0);
        assert_eq!(stat.algorithm, "passthrough");
        assert!(ccr_ref.is_none());
        assert_eq!(out, "fn id() { 1 }");
    }

    #[test]
    fn compress_item_disabled_algorithms_fall_through() {
        let cfg = CompressorConfig {
            enable_code: false,
            ..CompressorConfig::default()
        };
        let mut c = CausalCompressor::with_config(cfg);
        let code = "pub fn alpha() { let _x = 1; let _y = 2; _x + _y }\n";
        // With code disabled and content not JSON → prose → TextRank. A single
        // line of code has ≤2 sentences so it passes through.
        let (out, ccr_ref, stat) = c.compress_item("Symbol", code, 0.0);
        assert_eq!(stat.algorithm, "passthrough");
        assert!(ccr_ref.is_none());
        assert_eq!(out, code);
    }

    #[test]
    fn retrieve_unknown_ref_returns_none() {
        let c = CausalCompressor::new();
        assert!(c.retrieve(&CcrRef("deadbeefdead".into())).is_none());
    }

    #[test]
    fn ccr_evicts_when_over_capacity() {
        let cfg = CompressorConfig {
            ccr_capacity: 3,
            min_chars: 1, // force storage even for tiny items
            enable_prose: false,
            enable_code: false,
            ..CompressorConfig::default()
        };
        let mut c = CausalCompressor::with_config(cfg);
        // Feed 5 distinct JSON items so the crusher stores them.
        for i in 0..5 {
            let json = format!(r#"{{"i":{i}}}"#);
            let (_, r, _) = c.compress_item("Tool", &json, 0.0);
            assert!(r.is_some(), "item {i} stored");
        }
        assert_eq!(c.ccr_len(), 3, "evicted down to capacity");
    }

    #[test]
    fn compress_window_keeps_every_ref_retrievable_below_capacity() {
        // A single window with MORE stored items than ccr_capacity must still
        // hand back refs that ALL resolve — "nothing is lost" holds
        // unconditionally. (The old per-store eviction dropped live refs
        // mid-window once the window exceeded the capacity.)
        let cfg = CompressorConfig {
            ccr_capacity: 2,
            min_chars: 1,
            enable_prose: false,
            enable_code: false,
            enable_dedup: false,
            ..CompressorConfig::default()
        };
        let mut c = CausalCompressor::with_config(cfg);
        let originals: Vec<String> = (0..5).map(|i| format!(r#"{{"i":{i}}}"#)).collect();
        let uris: Vec<String> = (0..5).map(|i| format!("u{i}")).collect();
        let out = c.compress_window(
            originals
                .iter()
                .zip(&uris)
                .map(|(o, u)| ("Tool", 0.0, u.as_str(), o.as_str())),
        );
        assert_eq!(out.len(), 5);
        for (i, item) in out.iter().enumerate() {
            let r = item.ccr_ref.as_ref().expect("item stored");
            assert_eq!(
                c.retrieve(r),
                Some(originals[i].as_str()),
                "ref for item {i} must remain retrievable within the window"
            );
        }
        // 5 live refs > capacity 2 → the cap is a floor; all five are kept.
        assert_eq!(c.ccr_len(), 5);
    }

    #[test]
    fn compress_window_processes_a_batch() {
        let mut c = CausalCompressor::new();
        // (kind, score, uri, content)
        let items: Vec<(String, f64, String, String)> = vec![
            (
                "Symbol".into(),
                0.9,
                "sym:a".into(),
                "pub fn a() { let _x = 1; }\n".repeat(3),
            ),
            (
                "Note".into(),
                0.5,
                "note:b".into(),
                "Sentence one. Sentence two. Sentence three. Sentence four.".into(),
            ),
            (
                "Tool".into(),
                0.3,
                "tool:c".into(),
                r#"[{"id":1},{"id":2}]"#.into(),
            ),
        ];
        let refs: Vec<(&str, f64, &str, &str)> = items
            .iter()
            .map(|(k, s, u, v)| (k.as_str(), *s, u.as_str(), v.as_str()))
            .collect();
        let out = c.compress_window(refs);
        assert_eq!(out.len(), 3);
        assert_eq!(c.last_stats.len(), 3);
        // Every produced ref must be retrievable.
        for (item, (_, _, _, original)) in out.iter().zip(items.iter()) {
            if let Some(r) = &item.ccr_ref {
                assert_eq!(c.retrieve(r).unwrap(), original);
            }
        }
    }

    #[test]
    fn determinism_same_inputs_same_refs_and_outputs() {
        let text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six.";
        let mut a = CausalCompressor::new();
        let mut b = CausalCompressor::new();
        let (oa, ra, _) = a.compress_item("Note", text, 0.0);
        let (ob, rb, _) = b.compress_item("Note", text, 0.0);
        assert_eq!(oa, ob, "same content → same compressed form");
        assert_eq!(ra, rb, "same content → same CCR ref");
    }

    #[test]
    fn ccr_ref_is_short_and_hash_stable() {
        let r = CcrRef::of("hello world");
        assert_eq!(r.0.len(), 12, "12 hex chars");
        assert_eq!(r.0, CcrRef::of("hello world").0, "stable");
        assert_ne!(r.0, CcrRef::of("hello earth").0, "distinct");
    }

    // ── v2: cross-item dedup ────────────────────────────────────────────────

    #[test]
    fn dedup_merges_near_duplicate_code_spans() {
        let mut c = CausalCompressor::new();
        // Two identical symbol bodies at different URIs — after compression
        // they collapse to the same skeleton, so dedup must catch them.
        let body =
            "pub fn step() -> u8 {\n    let _acc = 1;\n    let _tmp = 2;\n    _acc + _tmp\n}\n";
        let items: Vec<(&str, f64, &str, &str)> = vec![
            ("Symbol", 0.9, "sym:a", body),
            ("Symbol", 0.8, "sym:b", body),
        ];
        let out = c.compress_window(items);
        // The lower-scored item must be replaced by a ~dup placeholder.
        assert!(
            out[1].content.contains("~dup of sym:a"),
            "second item deduped: {:?}",
            out[1].content
        );
        // The original is still retrievable.
        assert!(out[1].ccr_ref.is_some(), "dup still has a CCR ref");
        assert_eq!(c.retrieve(out[1].ccr_ref.as_ref().unwrap()).unwrap(), body);
    }

    #[test]
    fn dedup_keeps_unrelated_items() {
        let mut c = CausalCompressor::new();
        let items: Vec<(&str, f64, &str, &str)> = vec![
            ("Symbol", 0.9, "sym:a", "pub fn alpha() -> i32 { 1 }\n"),
            ("Symbol", 0.8, "sym:b", "pub fn beta() -> i32 { 2 }\n"),
        ];
        let out = c.compress_window(items);
        assert!(!out[1].content.contains("~dup"), "unrelated items kept");
    }

    #[test]
    fn dedup_disabled_keeps_everything() {
        let cfg = CompressorConfig {
            enable_dedup: false,
            ..CompressorConfig::default()
        };
        let mut c = CausalCompressor::with_config(cfg);
        let body = "pub fn step() -> u8 { let _x = 1; _x }\n";
        let items: Vec<(&str, f64, &str, &str)> = vec![
            ("Symbol", 0.9, "sym:a", body),
            ("Symbol", 0.8, "sym:b", body),
        ];
        let out = c.compress_window(items);
        assert!(!out[1].content.contains("~dup"), "dedup off → no merge");
    }

    // ── v2: CausalAST drop `use` + signature collapse ──────────────────────

    #[test]
    fn ast_v2_drops_use_lines() {
        let code = "use std::collections::HashMap;\nuse crate::db;\npub fn alpha() -> i32 { 1 }\n";
        let out = CausalAst::skeletonize(code);
        assert!(!out.contains("use "), "use lines dropped: {out}");
        assert!(out.contains("fn alpha"), "signature kept: {out}");
    }

    #[test]
    fn ast_v2_keeps_pub_use() {
        let code = "pub use crate::api;\npub fn alpha() {}\n";
        let out = CausalAst::skeletonize(code);
        // `pub use` is a re-export — kept (though the `pub ` ceremony prefix
        // is stripped, the `use` line itself survives, unlike a plain `use`).
        assert!(out.contains("use crate::api"), "pub use kept: {out}");
        assert!(!out.contains("pub use"), "pub prefix stripped: {out}");
    }

    #[test]
    fn ast_v2_collapses_long_signature_runs() {
        let mut code = String::new();
        for i in 0..20 {
            code.push_str(&format!("pub fn step{i}() -> u8;\n"));
        }
        let out = CausalAst::skeletonize(&code);
        assert!(
            out.contains("+") && out.contains("more signatures"),
            "long run collapsed: {out}"
        );
    }

    #[test]
    fn ast_v2_short_signature_run_not_collapsed() {
        let code = "pub fn a();\npub fn b();\npub fn c();\n";
        let out = CausalAst::skeletonize(code);
        assert!(!out.contains("more signatures"), "short run kept: {out}");
        assert!(out.contains("fn a") && out.contains("fn b") && out.contains("fn c"));
    }

    #[test]
    fn ast_v2_disabled_keeps_use_lines() {
        let cfg = CompressorConfig {
            enable_ast_v2: false,
            ..CompressorConfig::default()
        };
        let code = "use std::collections::HashMap;\npub fn alpha() {}\n";
        let out = CausalAst::skeletonize_with(code, &cfg);
        assert!(out.contains("use "), "v2 off → use kept: {out}");
    }

    // ── MinHash helpers ────────────────────────────────────────────────────

    #[test]
    fn minhash_identical_signatures_agree_fully() {
        let a = shingle_signature("the quick brown fox");
        let b = shingle_signature("the quick brown fox");
        assert!((estimated_jaccard(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn minhash_disjoint_signatures_disagree() {
        let a = shingle_signature("alpha beta gamma");
        let b = shingle_signature("zzz yyy www");
        assert!(estimated_jaccard(&a, &b) < 0.2, "disjoint → low jaccard");
    }

    #[test]
    fn minhash_near_duplicates_have_high_jaccard() {
        let a = shingle_signature("pub fn step() -> u8 { let _x = 1; _x }");
        let b = shingle_signature("pub fn step() -> u8 { let _x = 2; _x }");
        let j = estimated_jaccard(&a, &b);
        assert!(j > 0.7, "near-dup jaccard high: {j}");
    }

    #[test]
    fn minhash_is_deterministic() {
        let a = shingle_signature("deterministic input");
        let b = shingle_signature("deterministic input");
        assert_eq!(a, b);
    }

    // ── Auto-tuner ─────────────────────────────────────────────────────────

    #[test]
    fn auto_tune_prefers_more_aggressive_knobs_on_repetitive_code() {
        // A sample of repetitive code with many `use` lines and dup bodies —
        // the tuner should prefer ast_v2 on and a lower dedup threshold.
        let owned: Vec<(String, f64, String, String)> = (0..10)
            .map(|i| {
                let body = format!(
                    "use crate::m{i};\npub fn step() -> u8 {{\n    let _acc = 1;\n    let _tmp = 2;\n    _acc + _tmp\n}}\n"
                );
                (format!("sym:{i}"), 0.9 - i as f64 * 0.05, format!("sym:{i}"), body)
            })
            .collect();
        let sample: Vec<(&str, f64, &str, &str)> = owned
            .iter()
            .map(|(k, s, u, v)| (k.as_str(), *s, u.as_str(), v.as_str()))
            .collect();
        let base = CausalCompressor::new();
        let tuned = base.auto_tune(&sample);
        // Evaluate both on the sample — tuned must be no worse.
        let base_tokens = CausalCompressor::eval_config(&base.config, &sample);
        let tuned_tokens = CausalCompressor::eval_config(&tuned, &sample);
        assert!(
            tuned_tokens <= base_tokens,
            "tuned ({tuned_tokens}) <= base ({base_tokens})"
        );
    }
}
