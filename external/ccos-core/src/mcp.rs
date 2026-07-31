//! # MCP server — expose CCOS memory as Model Context Protocol tools
//!
//! A dependency-free [Model Context Protocol](https://modelcontextprotocol.io)
//! server over **stdio JSON-RPC 2.0**, so any MCP-compatible agent (Claude, a
//! local agent on the Jetson, …) can use CCOS as its working memory natively. The
//! memory lives in an [`AgentSession`], so the whole interaction is event-sourced
//! and replayable.
//!
//! Fourteen tools: `ingest`, `recall`, `signal_failure`, `page_fault`, `stats`,
//! `verify`, the time-travel pair `timeline` / `recall_what_if`, `ccos_retrieve`
//! (fetch the original of a compressed item), the causal-intervention pair
//! `causal_intervene` (do(X): what a change forces) / `causal_blame` (candidate
//! root causes), `drift_cause` (which recorded op moved a node's score —
//! change-point attribution), `retrodict_belief` (the RTS-smoothed belief
//! trajectory: future evidence folded back into past steps), and `causal_flash`
//! (a bounded causal-cone context window rooted at the active frontier — a
//! high-density summary that scales without recomputing global centrality). It
//! also exposes two
//! read-only **resources** — `ccos://session/context` (the current
//! self-bounding working set, linearised for direct injection into a system
//! prompt) and `ccos://session/timeline` (the cognitive journal).
//!
//! Run with `ccos mcp [workspace.ccos]`. With a path, the session reloads that
//! checkpoint on start and re-checkpoints after every memory-changing call, so
//! the memory survives restarts; without one it stays purely in-process.
//! Point your MCP client's stdio transport at it.

use crate::agent_session::AgentSession;
use crate::compressor::CcrRef;
use crate::external_memory::{ExternalMemory, MemoryError, Recall, RecallWindow};
use serde_json::{json, Value};

/// MCP protocol revision we speak (echoed back to the client when offered).
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// The tool catalogue advertised by `tools/list`, with JSON-Schema inputs.
fn tool_specs() -> Value {
    // The Pro `octa-semantic` strategy is advertised only when it is compiled in
    // (the `octasoma` feature) — the catalogue never promises a strategy this
    // build cannot execute. Whether a *call* is allowed is then the runtime
    // license gate (see `octa_semantic_recall`).
    #[cfg(feature = "octasoma")]
    let recall_strategies = json!([
        "around",
        "task",
        "semantic",
        "hybrid",
        "working_set",
        "causal-flash",
        "octa-semantic"
    ]);
    #[cfg(not(feature = "octasoma"))]
    let recall_strategies = json!([
        "around",
        "task",
        "semantic",
        "hybrid",
        "working_set",
        "causal-flash"
    ]);
    let tools = json!([
        {
            "name": "ingest",
            "description": "Ingest (or update) a source file into the causal memory graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "uri": {"type": "string", "description": "file path, e.g. src/db.rs"},
                    "source": {"type": "string"}
                },
                "required": ["uri", "source"]
            }
        },
        {
            "name": "recall",
            "description": "Recall a bounded, causally-coherent context window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": recall_strategies},
                    "anchor": {"type": "string", "description": "node id / file uri for 'around'"},
                    "text": {"type": "string", "description": "free-text task for 'task' / 'semantic' (and the Pro 'octa-semantic')"},
                    "budget": {"type": "integer", "description": "token budget (default 2048)"},
                    "horizon": {"type": "integer", "description": "'causal-flash': max dependency depth (default 3)"},
                    "decay": {"type": "number", "description": "'causal-flash': per-hop relevance decay in (0,1] (default 0.5)"},
                    "include_callers": {"type": "boolean", "description": "'causal-flash': add the one-hop caller impact ring (default true)"},
                    "include_low_trust_seeds": {"type": "boolean", "description": "'causal-flash': also seed from low-trust nodes, not just Working (default false)"},
                    "trust_threshold": {"type": "number", "description": "'causal-flash': low-trust seeding threshold (default 0.5)"}
                }
            }
        },
        {
            "name": "signal_failure",
            "description": "Mark a node as failing and propagate the pressure across the graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "depth": {"type": "integer", "description": "propagation depth (default 3)"}
                },
                "required": ["node"]
            }
        },
        {
            "name": "page_fault",
            "description": "Feed cargo-test/compiler output back in: parse the faulting files, inject pressure, recall a refreshed window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output": {"type": "string", "description": "cargo test / panic / backtrace text"},
                    "budget": {"type": "integer"}
                },
                "required": ["output"]
            }
        },
        {
            "name": "stats",
            "description": "Memory counts (nodes/edges/events/files).",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "verify",
            "description": "Verify the tamper-evident hash chain.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "timeline",
            "description": "The event-sourced cognitive timeline: every recorded operation (ingest / signal_failure / recall / page_fault), in order.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "recall_what_if",
            "description": "Time-travel debugging: rewind to a past step and re-run a recall under (possibly) different parameters — a deterministic replay of what the agent's window would have been.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "timeline step to rewind to (0 = before any op)"},
                    "strategy": {"type": "string", "enum": ["around", "task", "working_set"]},
                    "anchor": {"type": "string"},
                    "text": {"type": "string"},
                    "budget": {"type": "integer"}
                },
                "required": ["step"]
            }
        },
        {
            "name": "ccos_retrieve",
            "description": "Retrieve the original (uncompressed) content of a previously-compressed context item. Pass the `ccr_ref` string returned alongside a compressed recall / context resource. Returns the full original text so the LLM can drill into a skeleton or summary CCOS emitted in its place.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ccr_ref": {"type": "string", "description": "the 12-char hex ref returned with a compressed item"}
                },
                "required": ["ccr_ref"]
            }
        },
        {
            "name": "causal_intervene",
            "description": "do(X): the interventional impact of changing a node — the nodes that (transitively) DEPEND on it, each with an attenuated impact weight. Read-only; a Pearl-style intervention over the resolved dependency graph, not a similarity query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string", "description": "node id / file path (bare paths get a file: prefix)"},
                    "magnitude": {"type": "number", "description": "intervention magnitude (default 1.0)"},
                    "damping": {"type": "number", "description": "per-hop attenuation (default 0.75)"},
                    "depth": {"type": "integer", "description": "max hops (default 4)"}
                },
                "required": ["node"]
            }
        },
        {
            "name": "causal_blame",
            "description": "The candidate root causes of a failure at a node — what it (transitively) DEPENDS ON, ranked by attenuated dependency weight. The dual of causal_intervene: the principled 'the culprit is upstream in a different file'. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string", "description": "node id / file path (bare paths get a file: prefix)"},
                    "damping": {"type": "number", "description": "per-hop attenuation (default 0.75)"},
                    "depth": {"type": "integer", "description": "max hops (default 4)"}
                },
                "required": ["node"]
            }
        },
        {
            "name": "drift_cause",
            "description": "Causal-of-drift attribution: reconstruct a node's score trajectory across the replayable history, locate the dominant level shift (CUSUM change-point), and name the recorded operation that caused it. Read-only but replays the whole timeline — an offline post-mortem query, not a hot-path call.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string", "description": "node id / file path (bare paths get a file: prefix)"}
                },
                "required": ["node"]
            }
        },
        {
            "name": "retrodict_belief",
            "description": "Retrodiction: a claim's belief/tension trajectory over the replayed timeline, plus the RTS-smoothed reconstruction that folds FUTURE evidence back into every PAST step (what the engine should have believed at t given everything since). Read-only; replays the timeline — offline analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "claim node id"},
                    "stride": {"type": "integer", "description": "sample every N steps (default 1)"},
                    "half_life": {"type": "number", "description": "knowledge half-life for decayed belief; <= 0 = undecayed (default 0)"},
                    "q": {"type": "number", "description": "smoother process variance (default 0.02)"},
                    "r": {"type": "number", "description": "smoother measurement variance (default 0.1)"}
                },
                "required": ["claim"]
            }
        },
        {
            "name": "causal_flash",
            "description": "Bounded causal-cone context for the active frontier: seed from Working (optionally low-trust) nodes, follow dependency (out-) edges to horizon n (or a fixpoint), add a one-hop caller ring for impact, and rank by decayed in-cone relevance. A high-density causal summary that fits a token budget WITHOUT recomputing global centrality — the scale lever for large graphs. Deterministic, read-only; reports a completeness flag (true iff the dependency closure was not cut).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "integer", "description": "max dependency depth n (default 3)"},
                    "decay": {"type": "number", "description": "per-hop relevance decay in (0,1] (default 0.5)"},
                    "include_callers": {"type": "boolean", "description": "add the one-hop in-edge impact ring (default true)"},
                    "include_low_trust_seeds": {"type": "boolean", "description": "also seed from low-trust nodes, not just Working (default false)"},
                    "trust_threshold": {"type": "number", "description": "'causal-flash': low-trust seeding threshold (default 0.5)"},
                    "max_nodes": {"type": "integer", "description": "token budget: cap node count, dropping callers first; dependencies are never dropped (default unbounded)"},
                    "query": {"type": "string", "description": "OpenClaw memory_search query; alias for `text` (and `anchor` under 'around'). When `strategy` is unset, defaults to 'semantic'."},
                    "limit": {"type": "integer", "description": "cap on the number of returned items (node-count cap, distinct from the `budget` token cap). Applied after recall."},
                    "minScore": {"type": "number", "description": "drop items whose `score` is below this threshold."},
                    "sessionKey": {"type": "string", "description": "optional session selector; CCOS resolves it to the active agent session when one is bound, ignored otherwise."}
                }
            }
        },
        {
            "name": "get",
            "description": "Read an ingested source file by path. Returns the whole-file text (the same source `ingest` stored), optionally windowed by `from`/`lines`. This is the file-read surface OpenClaw's memory_get maps to; it is distinct from `ccos_retrieve`, which decompresses a previously-compressed recall item by ccr_ref.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file path (e.g. src/db.rs); a `file:` prefix is accepted but not required"},
                    "from": {"type": "integer", "description": "1-indexed first line to return (default 1)"},
                    "lines": {"type": "integer", "description": "max number of lines to return (default unbounded)"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "sync",
            "description": "Boot/refresh ack: checkpoint the session so in-memory state is durable, and report the current timeline step. OpenClaw calls this at gateway boot and on explicit refresh. Read-only to the index (the causal graph is derived state) — it writes the snapshot, it does not change the graph. Fails visibly if the workspace could not be written; a no-op when no workspace is bound.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "force": {"type": "boolean", "description": "accepted for compatibility; the checkpoint is unconditional, so this changes nothing"},
                    "reason": {"type": "string", "description": "free-text reason for the sync (e.g. 'boot'); recorded for diagnostics only"}
                }
            }
        }
    ]);
    // The Pro octa-semantic feedback surface exists only in `octasoma` builds: the
    // `octa_feedback` tool, and the `alpha` gate parameter on `recall` — same
    // never-promise-what-this-build-cannot-execute rule as the strategy enum above.
    #[cfg(feature = "octasoma")]
    let tools = {
        let mut tools = tools;
        let list = tools.as_array_mut().expect("catalogue is an array");
        for t in list.iter_mut() {
            if t["name"] == "recall" {
                t["inputSchema"]["properties"]["alpha"] = json!({
                    "type": "number",
                    "description": "(Pro 'octa-semantic' only) miscoverage level in (0,1) for the feedback-calibrated anchor gate (default 0.1)"
                });
            }
        }
        list.push(json!({
            "name": "octa_feedback",
            "description": "Label the last octa-semantic recall (or an explicit query/uri/score triple) as relevant or not. The labels calibrate the conformal anchor gate future octa-semantic recalls run through — the explicit relevance-feedback channel of the Pro semantic tier. Stateful: served by the stdio loop; the stateless embedding API refuses it visibly.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "relevant": {"type": "boolean", "description": "was the resolved anchor actually useful for the query?"},
                    "query": {"type": "string", "description": "label an explicit observation instead of the last recall (requires uri and score too)"},
                    "uri": {"type": "string", "description": "anchor node uri of the explicit observation"},
                    "score": {"type": "number", "description": "similarity score in (0,1] the anchor was returned with"},
                    "alpha": {"type": "number", "description": "miscoverage level for the floor reported back (default 0.1)"}
                },
                "required": ["relevant"]
            }
        }));
        tools
    };
    tools
}

/// The read-only resources advertised by `resources/list`.
fn resource_specs() -> Value {
    json!([
        {
            "uri": "ccos://session/context",
            "name": "CCOS working-set context",
            "description": "The current causally-scored, token-bounded working set, linearised for direct injection into a system prompt. Reflects accumulated failure pressure and recency; self-bounds at the causal region (no K to tune). Budget via CCOS_MCP_CONTEXT_BUDGET (default 2048 tokens).",
            "mimeType": "text/plain"
        },
        {
            "uri": "ccos://session/timeline",
            "name": "CCOS cognitive timeline",
            "description": "The event-sourced journal of every memory operation this session (audit / replay).",
            "mimeType": "text/plain"
        },
        {
            "uri": "ccos://setup/report",
            "name": "CCOS setup verdict",
            "description": "The sealed installer report written by `ccos setup` (per-check pass/fail, wiring actions, host probe, content hash). Produced deterministically by the installer — relay its verdict to the user verbatim; the JSON is the source of truth, not this server's memory. Path: $CCOS_SETUP_REPORT or ./setup_report.json.",
            "mimeType": "application/json"
        }
    ])
}

/// Wrap a payload string as MCP tool-call content.
fn content(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// Read a string argument (empty when absent).
fn str_arg(args: &Value, k: &str) -> String {
    args.get(k)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// Read an f64 argument with a default.
fn f64_arg(args: &Value, k: &str, default: f64) -> f64 {
    args.get(k).and_then(Value::as_f64).unwrap_or(default)
}

/// Prefix a bare path with `file:`; leave known node-id prefixes untouched (the
/// same convenience the post-mortem REPL applies, so hosts can pass either form).
fn normalize_node(s: &str) -> String {
    const PREFIXES: [&str; 5] = ["file:", "sym:", "mod:", "use:", "dep:"];
    if PREFIXES.iter().any(|p| s.starts_with(p)) {
        s.to_string()
    } else {
        format!("file:{s}")
    }
}

/// Name of the argument a strategy needs, when it is missing or blank.
///
/// An anchored or free-text strategy without its query is not a recall that found
/// nothing — it is a request that was never asked. Answering it with a well-formed
/// empty window makes a caller's typo look like an empty memory: the window has
/// the right shape, the right strategy label, and zero items, so nothing
/// downstream has any reason to doubt it. Naming the missing field instead is the
/// difference between "you asked wrong" and "there is nothing there".
fn missing_recall_arg(args: &Value) -> Option<&'static str> {
    let blank = |k: &str| str_arg(args, k).trim().is_empty();
    match args
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or("working_set")
    {
        "around" if blank("anchor") => Some("anchor"),
        "task" | "semantic" | "hybrid" | "octa-semantic" | "octa_semantic" if blank("text") => {
            Some("text")
        }
        _ => None,
    }
}

/// The first argument a tool's own advertised schema marks `required` and the
/// call did not supply, if any.
///
/// The catalogue in [`tool_specs`] is a promise: `"required": ["output"]` tells a
/// client the call is invalid without it. Nothing enforced that promise, and three
/// tools quietly took the omission as a default — `page_fault` with no compiler
/// output recalled a plain working set and labelled it a page-fault result,
/// `recall_what_if` with no step replayed step 0 and returned an empty window,
/// `signal_failure` with no node reported "node not found: file:" as though the
/// caller had named a node that was missing.
///
/// Enforcing the declaration rather than hand-writing a check per tool means the
/// schema stays the single statement of what a call needs, and any tool added
/// later is covered the moment it declares.
///
/// Presence only — that is what `required` means in JSON Schema. Whether an
/// *empty* value is also meaningless is a per-tool question: `ingest` with
/// `source: ""` is a legitimately empty file, `page_fault` with `output: ""` is
/// not a page fault.
fn missing_required_arg(tool: &str, args: &Value) -> Option<String> {
    static SPECS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    let specs = SPECS.get_or_init(tool_specs);
    let required = specs
        .as_array()?
        .iter()
        .find(|t| t["name"] == tool)?
        .get("inputSchema")?
        .get("required")?
        .as_array()?;
    required
        .iter()
        .filter_map(Value::as_str)
        .find(|key| args.get(*key).is_none())
        .map(str::to_string)
}

/// Why `step` cannot be replayed faithfully, or `None` when it can.
///
/// `replay_to` clamps in both directions and cannot say that it did — it returns
/// a memory, not a `Result`. Past the end it hands back the present; below the
/// compaction floor it hands back the state *at* the floor. Either way the caller
/// gets a well-formed window for a moment that is not the one it named, and for
/// the tool whose entire job is answering "what did memory look like then" that is
/// the one failure mode that matters. Measured on a 3-op timeline: `step: 9999`
/// returned exactly what `step: 3` returned, with nothing marking the difference.
fn unreplayable_step(session: &AgentSession, step: usize) -> Option<String> {
    let (len, floor) = (session.len(), session.floor());
    if step > len {
        return Some(format!(
            "step {step} is past the end of the timeline ({len} op(s) recorded)"
        ));
    }
    // Step 0 is the empty baseline and always means "before anything happened".
    if step > 0 && step < floor {
        return Some(format!(
            "step {step} is below the compaction floor ({floor}): that history has \
             been folded into the baseline and cannot be replayed separately. The \
             earliest faithful step is {floor} (raise CCOS_OPLOG_MAX to keep more)"
        ));
    }
    None
}

/// The JSON-RPC refusal for a recall whose query argument is missing or blank.
/// Names both the strategy and the field, since the usual cause is a caller that
/// used the wrong key and has no way to see that from an empty window.
fn refuse_blank_recall(args: &Value, arg: &str) -> (i64, String) {
    let strategy = args
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or("working_set");
    (
        -32602,
        format!("recall strategy '{strategy}' requires a non-empty '{arg}'"),
    )
}

/// Build a [`Recall`] strategy from `{strategy, anchor, text}` arguments. Shared
/// by `recall` and the time-travel `recall_what_if`.
fn recall_from_args(args: &Value) -> Recall {
    match args
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or("working_set")
    {
        "around" => Recall::around(str_arg(args, "anchor")),
        "task" => Recall::task(str_arg(args, "text")),
        "semantic" => Recall::semantic(str_arg(args, "text")),
        "hybrid" => Recall::hybrid(str_arg(args, "text")),
        "causal-flash" | "causal_flash" => {
            Recall::causal_flash(crate::external_memory::CausalFlashRecall {
                horizon: args.get("horizon").and_then(Value::as_u64).unwrap_or(3) as usize,
                decay: f64_arg(args, "decay", 0.5),
                include_callers: args
                    .get("include_callers")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                include_low_trust_seeds: args
                    .get("include_low_trust_seeds")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                trust_threshold: f64_arg(args, "trust_threshold", 0.5),
            })
        }
        _ => Recall::working_set(),
    }
}

/// The Pro **`octa-semantic`** recall strategy (`octasoma` feature): OctaSoma resolves
/// the entry node semantically, then the recall goes through the **session** as
/// `Recall::Around(anchor)` — so the op recorded in the event-sourced timeline carries
/// the *resolved* anchor, and `replay == live` holds by construction even if a future
/// embedder is not replay-exact. The index is **derived state**, rebuilt
/// deterministically from the live graph on every call (octasoma's `HashEmbedder` is a
/// hash — microseconds per node; a cached, persistent index behind a real embedder is
/// the documented follow-up). On the community tier the refusal is a visible tool
/// result, never a silent downgrade — the free strategies keep working.
///
/// With a [`ServerState`] whose `octa_feedback` log supports the asked `alpha`
/// (default 0.1), the resolved anchor also runs through the **conformal gate**: score
/// at or above the certified floor → `"octa-semantic-certified"`; below → the anchor
/// is refused and the window comes from the lexical fallback,
/// `"octa-semantic-below-floor-fallback-task"`. The response carries the resolution
/// (`anchor`) and the gate's inputs (`calibration`) alongside the window, so the
/// client can label the anchor via `octa_feedback` and see the calibration progress.
#[cfg(feature = "octasoma")]
fn octa_semantic_recall(
    session: &mut AgentSession,
    state: Option<&mut ServerState>,
    args: &Value,
    budget: usize,
) -> Result<Value, (i64, String)> {
    use crate::octa_index::SemanticMemoryAccess;
    use octasoma::HashEmbedder;

    // Embedding width of the derived index (any fixed width works for the exact-text
    // `HashEmbedder`; matches the `octasoma_semantic` example).
    const DIM: usize = 128;

    let text = str_arg(args, "text");
    if text.is_empty() {
        return Err((-32602, "octa-semantic requires 'text'".into()));
    }
    let alpha = args.get("alpha").and_then(Value::as_f64).unwrap_or(0.1);
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err((-32602, "octa-semantic 'alpha' must be in (0,1)".into()));
    }
    let access = match SemanticMemoryAccess::unlock(session.licensing(), crate::license::now_unix())
    {
        Ok(a) => a,
        Err(e) => {
            return Ok(json!({
                "content": [{ "type": "text",
                    "text": format!("octa-semantic is a Pro strategy — {e}. The free \
                     strategies (around/task/semantic/hybrid/working_set) remain fully \
                     functional.") }],
                "isError": true
            }))
        }
    };
    let idx = access.sharded_index_from_graph(HashEmbedder::new(DIM), session.memory().graph());

    // Conformal anchor gate, calibrated on the server-held feedback log (see
    // `SemanticFeedback::certified_score_floor`). Stateless entry / empty log → no
    // floor → baseline behavior, and the response's `calibration` block says so.
    let floor = state
        .as_ref()
        .and_then(|st| st.octa.feedback.certified_score_floor(alpha));
    let labels = state.as_ref().map_or(0, |st| st.octa.feedback.len());

    let (window, anchor_json) = match idx.semantic_anchors(&text, 1).into_iter().next() {
        Some((anchor, score)) => {
            let trusted = match floor {
                Some(f) => score >= f,
                None => true,
            };
            let w = if trusted {
                let mut w = session.recall(Recall::around(anchor.clone()), budget);
                w.strategy = if floor.is_some() {
                    "octa-semantic-certified".to_string()
                } else {
                    "octa-semantic".to_string()
                };
                w
            } else {
                // The anchor exists but scores below the certified floor: trusting it
                // would be unwarranted, so the lexical fallback is taken *visibly*.
                let mut w = session.recall(Recall::task(text.clone()), budget);
                w.strategy = "octa-semantic-below-floor-fallback-task".to_string();
                w
            };
            if let Some(st) = state {
                st.octa.last = Some((text.clone(), anchor.clone(), score));
            }
            (w, json!({ "uri": anchor, "score": score }))
        }
        None => {
            let mut w = session.recall(Recall::task(text), budget);
            w.strategy = "octa-semantic-fallback-task".to_string();
            (w, Value::Null)
        }
    };
    let payload = json!({
        "window": window,
        "anchor": anchor_json,
        "calibration": { "alpha": alpha, "floor": floor, "labels": labels }
    });
    Ok(content(payload.to_string()))
}

/// The Pro **`octa_feedback`** tool — the explicit relevance channel for the
/// octa-semantic tier: the agent loop reports whether a resolved anchor was actually
/// useful, and the labels calibrate the conformal anchor gate the next recalls run
/// through (octasoma's design decision, CCOS side). Stateful by nature: the label log
/// lives in [`ServerState`] with the serve loop — the stateless [`handle`] refuses the
/// call visibly rather than dropping labels silently.
#[cfg(feature = "octasoma")]
fn octa_feedback_tool(
    session: &mut AgentSession,
    state: Option<&mut ServerState>,
    args: &Value,
) -> Result<Value, (i64, String)> {
    use crate::octa_index::SemanticMemoryAccess;

    // Same Pro gate as the recalls the labels calibrate.
    if let Err(e) = SemanticMemoryAccess::unlock(session.licensing(), crate::license::now_unix()) {
        return Ok(json!({
            "content": [{ "type": "text",
                "text": format!("octa_feedback is part of the Pro octa-semantic tier — {e}.") }],
            "isError": true
        }));
    }
    let Some(st) = state else {
        return Ok(json!({
            "content": [{ "type": "text",
                "text": "octa_feedback needs the stateful server loop (`serve`): this \
                 entry point is stateless, so the label would be dropped on return — \
                 refusing instead of forgetting silently." }],
            "isError": true
        }));
    };
    let Some(relevant) = args.get("relevant").and_then(Value::as_bool) else {
        return Err((-32602, "octa_feedback requires boolean 'relevant'".into()));
    };
    let alpha = args.get("alpha").and_then(Value::as_f64).unwrap_or(0.1);
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err((-32602, "octa_feedback 'alpha' must be in (0,1)".into()));
    }
    // Label an explicit `(query, uri, score)` triple, or — the common loop — the last
    // octa-semantic resolution this server performed.
    let explicit = match (
        args.get("query").and_then(Value::as_str),
        args.get("uri").and_then(Value::as_str),
        args.get("score").and_then(Value::as_f64),
    ) {
        (Some(q), Some(u), Some(s)) => Some((q.to_string(), u.to_string(), s)),
        (None, None, None) => None,
        _ => {
            return Err((
                -32602,
                "octa_feedback takes either all of 'query'/'uri'/'score' or none of \
                 them (none = label the last octa-semantic recall)"
                    .into(),
            ))
        }
    };
    let Some((query, uri, score)) = explicit.or_else(|| st.octa.last.clone()) else {
        return Ok(json!({
            "content": [{ "type": "text",
                "text": "no octa-semantic recall to label yet — call `recall` with \
                 strategy 'octa-semantic' first, or pass 'query'/'uri'/'score' \
                 explicitly." }],
            "isError": true
        }));
    };
    if !(score > 0.0 && score <= 1.0) {
        return Err((-32602, "octa_feedback 'score' must be in (0,1]".into()));
    }
    st.octa.feedback.record(&query, &uri, score, relevant);
    let payload = json!({
        "recorded": { "query": query, "uri": uri, "score": score, "relevant": relevant },
        "labels": st.octa.feedback.len(),
        "relevant_labels": st.octa.feedback.relevant_count(),
        "calibration": { "alpha": alpha, "floor": st.octa.feedback.certified_score_floor(alpha) }
    });
    Ok(content(payload.to_string()))
}

/// Execute a `tools/call`.
fn call_tool(
    session: &mut AgentSession,
    state: Option<&mut ServerState>,
    params: &Value,
) -> Result<Value, (i64, String)> {
    // The only stateful tools today are octasoma-gated.
    #[cfg(not(feature = "octasoma"))]
    let _ = state;
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let budget = args.get("budget").and_then(Value::as_u64).unwrap_or(2048) as usize;
    // Hold the catalogue to its own word before dispatching — see
    // `missing_required_arg`.
    if let Some(arg) = missing_required_arg(name, &args) {
        return Err((-32602, format!("{name} requires '{arg}'")));
    }

    let text = match name {
        "ingest" => {
            let uri = str_arg(&args, "uri");
            if uri.is_empty() {
                return Err((-32602, "ingest requires 'uri' and 'source'".into()));
            }
            // The message has always claimed both are required; only `uri` was
            // checked, so an `ingest` with no `source` succeeded and put an empty
            // node in the graph — a file that exists, in the memory, with no
            // content, which nothing downstream can tell from a real empty file.
            // `source: ""` stays legal; the absent key is what is refused.
            if args.get("source").is_none() {
                return Err((
                    -32602,
                    "ingest requires 'source' (pass \"\" for an empty file)".into(),
                ));
            }
            serde_json::to_string(&session.ingest(&uri, &str_arg(&args, "source")))
                .unwrap_or_default()
        }
        "signal_failure" => {
            let depth = args.get("depth").and_then(Value::as_u64).unwrap_or(3) as u32;
            match session.signal_failure(&str_arg(&args, "node"), depth) {
                Ok(n) => json!({ "affected": n }).to_string(),
                Err(e) => {
                    return Ok(json!({
                        "content": [{ "type": "text", "text": e.to_string() }],
                        "isError": true
                    }))
                }
            }
        }
        "recall" => {
            if let Some(arg) = missing_recall_arg(&args) {
                return Err(refuse_blank_recall(&args, arg));
            }
            #[cfg(feature = "octasoma")]
            if args.get("strategy").and_then(Value::as_str) == Some("octa-semantic") {
                return octa_semantic_recall(session, state, &args, budget);
            }
            serde_json::to_string(&session.recall(recall_from_args(&args), budget))
                .unwrap_or_default()
        }
        #[cfg(feature = "octasoma")]
        "octa_feedback" => return octa_feedback_tool(session, state, &args),
        "page_fault" => {
            // Presence is not enough here: an empty compiler output names no
            // faulting file, so the fault degrades to a plain working-set recall
            // that still comes back looking like a page-fault result.
            if str_arg(&args, "output").trim().is_empty() {
                return Err((-32602, "page_fault requires a non-empty 'output'".into()));
            }
            serde_json::to_string(&session.page_fault(&str_arg(&args, "output"), budget))
                .unwrap_or_default()
        }
        "stats" => serde_json::to_string(&session.memory().stats()).unwrap_or_default(),
        "verify" => serde_json::to_string(&session.memory().verify()).unwrap_or_default(),
        "timeline" => json!({ "timeline": session.timeline() }).to_string(),
        "recall_what_if" => {
            if let Some(arg) = missing_recall_arg(&args) {
                return Err(refuse_blank_recall(&args, arg));
            }
            // A `step` that is present but not a number would fall through
            // `as_u64` to 0 and silently replay the empty baseline.
            let Some(step) = args.get("step").and_then(Value::as_u64).map(|s| s as usize) else {
                return Err((-32602, "recall_what_if 'step' must be a number".into()));
            };
            if let Some(why) = unreplayable_step(session, step) {
                return Err((-32602, why));
            }
            let window = session.recall_what_if(step, &recall_from_args(&args), budget);
            serde_json::to_string(&window).unwrap_or_default()
        }
        "causal_intervene" => {
            let node = str_arg(&args, "node");
            if node.is_empty() {
                return Err((-32602, "causal_intervene requires 'node'".into()));
            }
            let id = crate::memory::NodeId(normalize_node(&node));
            let impact = session.memory().graph().intervene(
                &id,
                f64_arg(&args, "magnitude", 1.0),
                f64_arg(&args, "damping", 0.75),
                args.get("depth").and_then(Value::as_u64).unwrap_or(4) as usize,
            );
            let rows: Vec<Value> = impact
                .iter()
                .map(|(n, v)| json!({ "node": n.0, "impact": v }))
                .collect();
            json!({ "origin": id.0, "forced": rows }).to_string()
        }
        "causal_blame" => {
            let node = str_arg(&args, "node");
            if node.is_empty() {
                return Err((-32602, "causal_blame requires 'node'".into()));
            }
            let id = crate::memory::NodeId(normalize_node(&node));
            let causes = session.memory().graph().blame(
                &id,
                f64_arg(&args, "damping", 0.75),
                args.get("depth").and_then(Value::as_u64).unwrap_or(4) as usize,
            );
            let rows: Vec<Value> = causes
                .iter()
                .map(|(n, v)| json!({ "node": n.0, "weight": v }))
                .collect();
            json!({ "origin": id.0, "candidate_causes": rows }).to_string()
        }
        "drift_cause" => {
            let node = str_arg(&args, "node");
            if node.is_empty() {
                return Err((-32602, "drift_cause requires 'node'".into()));
            }
            match session.attribute_drift(&normalize_node(&node)) {
                Some(c) => json!({
                    "node": c.node,
                    "step": c.step,
                    "delta": c.delta,
                    "cusum": c.cusum,
                    "op": c.op,
                })
                .to_string(),
                None => json!({
                    "node": normalize_node(&node),
                    "cause": Value::Null,
                    "note": "no attributable drift (flat trajectory, or the break is below the compaction floor)",
                })
                .to_string(),
            }
        }
        "retrodict_belief" => {
            let claim = str_arg(&args, "claim");
            if claim.is_empty() {
                return Err((-32602, "retrodict_belief requires 'claim'".into()));
            }
            let id = crate::memory::NodeId(claim.clone());
            let stride = args.get("stride").and_then(Value::as_u64).unwrap_or(1) as usize;
            let profile = session.belief_tension_timeline(
                std::slice::from_ref(&id),
                stride,
                f64_arg(&args, "half_life", 0.0),
            );
            let (q, r) = (f64_arg(&args, "q", 0.02), f64_arg(&args, "r", 0.1));
            json!({
                "claim": claim,
                "stride": stride,
                "belief": profile.belief_series(&id),
                "belief_retrodicted": profile.retrodicted_belief(&id, q, r),
                "tension": profile.tension_series(&id),
                "tension_retrodicted": profile.retrodicted_tension(&id, q, r),
            })
            .to_string()
        }
        "causal_flash" => {
            let cfg = crate::causal_flash::CausalFlashConfig {
                horizon: args.get("horizon").and_then(Value::as_u64).unwrap_or(3) as usize,
                decay: f64_arg(&args, "decay", 0.5),
                include_callers: args
                    .get("include_callers")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                include_low_trust_seeds: args
                    .get("include_low_trust_seeds")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                trust_threshold: f64_arg(&args, "trust_threshold", 0.5),
                max_nodes: args
                    .get("max_nodes")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize),
            };
            let win = session.memory().graph().causal_flash_window(&cfg);
            let rows: Vec<Value> = win
                .nodes
                .iter()
                .map(|n| {
                    json!({
                        "node": n.id.0,
                        "role": match n.role {
                            crate::causal_flash::CausalRole::Seed => "seed",
                            crate::causal_flash::CausalRole::Dependency => "dependency",
                            crate::causal_flash::CausalRole::Caller => "caller",
                        },
                        "depth": n.depth,
                        "relevance": n.relevance,
                    })
                })
                .collect();
            json!({
                "seed_count": win.seed_count,
                "complete": win.complete,
                "omitted": win.omitted,
                "nodes": rows,
            })
            .to_string()
        }
        "ccos_retrieve" => {
            let key = str_arg(&args, "ccr_ref");
            if key.is_empty() {
                return Err((-32602, "ccos_retrieve requires 'ccr_ref'".into()));
            }
            match session.retrieve_original(&CcrRef(key.clone())) {
                Some(original) => {
                    return Ok(json!({
                        "content": [{ "type": "text", "text": original }],
                        "ccr_ref": key,
                        "bytes": original.len()
                    }))
                }
                None => {
                    return Ok(json!({
                        "content": [{ "type": "text",
                            "text": "ccr_ref not found (evicted or unknown)" }],
                        "isError": true
                    }))
                }
            }
        }
        "get" => {
            let path = str_arg(&args, "path");
            if path.is_empty() {
                return Err((-32602, "get requires 'path'".into()));
            }
            let source = session.memory().source_for(&path);
            match source {
                Some(text) => {
                    let from = args.get("from").and_then(Value::as_u64).unwrap_or(1) as usize;
                    let lines_specified = args.get("lines").and_then(Value::as_u64);
                    let text = if from > 1 || lines_specified.is_some() {
                        let v: Vec<&str> = text.lines().collect();
                        let from = (from.max(1) - 1).min(v.len());
                        let lines = lines_specified.map(|l| l as usize).unwrap_or(usize::MAX);
                        v[from..from + lines.min(v.len() - from)].join("\n")
                    } else {
                        text.to_string()
                    };
                    return Ok(json!({
                        "content": [{ "type": "text", "text": text }],
                        "lines": text.lines().count(),
                        "path": path
                    }));
                }
                None => {
                    return Ok(json!({
                        "content": [{ "type": "text",
                            "text": format!("file not found in ingested sources: {path}") }],
                        "isError": true
                    }))
                }
            }
        }
        "sync" => {
            // `sync` is the boot/refresh acknowledgement a gateway calls to confirm
            // the session is durable — making it durable is the whole job. Two
            // things were wrong with that.
            //
            // It discarded the checkpoint result (`let _ = session.checkpoint()`)
            // and answered `ok: true` regardless, so a gateway got a clean boot ack
            // for a session that had persisted nothing; the only trace was a stderr
            // line no MCP client reads.
            //
            // And it only checkpointed under `force`, while its description has
            // always said it checkpoints — so the default call, the one a gateway
            // actually makes at boot, persisted nothing at all. The description is
            // the contract worth keeping, so the checkpoint is now unconditional
            // and `force` stays accepted but inert rather than quietly deciding
            // whether the promise holds. `NoPath` is still success: a session with
            // no workspace bound was never asked to persist anything.
            if let Err(e) = persist(session) {
                return Err((
                    -32603,
                    format!("sync could not make the session durable: {e}"),
                ));
            }
            json!({
                "step": session.timeline().len(),
                "ok": true,
                "events": session.memory().verify().events,
            })
            .to_string()
        }
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };
    Ok(content(text))
}

/// Linearise a recalled window into a single text blob a host can drop straight
/// into a system prompt (the auto-calibrated context chain). When items carry a
/// [`CcrRef`] (produced by [`AgentSession::recall_compressed`]), the ref is
/// appended so the LLM knows it can call `ccos_retrieve` for the full original.
fn linearize_window(win: &RecallWindow, plain: bool) -> String {
    // Plain mode emits ordinary multi-file source (`// path` + code), dropping the
    // `[kind score]` annotations. A weak model (≤~3B) misreads a `// sym:…` header as code
    // and miscompiles (Campaign J2 finding); annotations help a strong model rank, so they
    // stay on by default. The caller decides via `CCOS_CONTEXT_PLAIN`.
    if plain {
        let mut out = String::new();
        for it in &win.items {
            let path = it.uri.split(':').nth(1).unwrap_or(&it.uri);
            out.push_str(&format!("// {path}\n{}\n\n", it.content));
            if let Some(r) = &it.ccr_ref {
                out.push_str(&format!(
                    "// ccr_ref={} (call ccos_retrieve for full)\n\n",
                    r.0
                ));
            }
        }
        return out;
    }
    let mut out = format!(
        "// CCOS context — {} ({} items, ~{} tokens)\n",
        win.strategy,
        win.items.len(),
        win.tokens
    );
    for it in &win.items {
        out.push_str(&format!(
            "\n// {} [{}] score={:.3}\n{}\n",
            it.uri, it.kind, it.score, it.content
        ));
        if let Some(r) = &it.ccr_ref {
            out.push_str(&format!(
                "// ccr_ref={} (call ccos_retrieve for full)\n",
                r.0
            ));
        }
    }
    out
}

/// Execute a `resources/read`.
fn read_resource(session: &mut AgentSession, params: &Value) -> Result<Value, (i64, String)> {
    let uri = params.get("uri").and_then(Value::as_str).unwrap_or("");
    let text = match uri {
        "ccos://session/context" => {
            // Budget tunable at launch without a flag.
            let budget = std::env::var("CCOS_MCP_CONTEXT_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048usize);
            // Compression is on by default; set CCOS_COMPRESS_CONTEXT=0 to get
            // the historical raw (uncompressed) context for A/B comparison.
            let compress = std::env::var("CCOS_COMPRESS_CONTEXT")
                .ok()
                .map(|v| v != "0" && v != "false")
                .unwrap_or(true);
            // Anchor on the workspace signal: if something is failing, inject the
            // causal *region* of that problem (far more useful on a real codebase
            // than the global working set, which a `use`-heavy repo fills with the
            // hottest file); otherwise fall back to the global working set.
            let mem = session.memory();
            let anchor = mem.hottest_failure_node();
            let recall = match &anchor {
                Some(a) => Recall::around(a.clone()),
                None => Recall::working_set(),
            };
            let window = if compress {
                session.recall_compressed(recall, budget)
            } else {
                session.recall(recall, budget)
            };
            let plain = std::env::var("CCOS_CONTEXT_PLAIN")
                .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
                .unwrap_or(false);
            linearize_window(&window, plain)
        }
        "ccos://session/timeline" => session.timeline().join("\n"),
        other => return Err((-32602, format!("unknown resource: {other}"))),
    };
    Ok(json!({ "contents": [{ "uri": uri, "mimeType": "text/plain", "text": text }] }))
}

/// Cross-call server state for **stateful** tools — today only the Pro octa-semantic
/// relevance-feedback log (`octasoma` feature); empty otherwise. Held by the serve loop
/// and deliberately NOT by [`AgentSession`]: feedback is calibration state describing the
/// *workload*, not causal history, so the event-sourced core (and `replay == live`) is
/// untouched — recalls still land in the timeline with their resolved anchor. It is also
/// not persisted with the workspace: stale labels silently void the guarantees they
/// exist to support (same stance as octasoma's `feedback` module).
#[derive(Default)]
pub struct ServerState {
    #[cfg(feature = "octasoma")]
    octa: OctaFeedbackState,
}

/// The octa-semantic feedback channel: the label log plus the last resolved anchor
/// (what a bare `octa_feedback {relevant}` refers to).
#[cfg(feature = "octasoma")]
#[derive(Default)]
struct OctaFeedbackState {
    feedback: crate::octa_index::SemanticFeedback,
    /// `(query, anchor_uri, score)` of the most recent octa-semantic resolution —
    /// recorded even when the anchor was refused by the floor, so a mistaken
    /// rejection can be labelled relevant and widen the gate back.
    last: Option<(String, String, f64)>,
}

/// Handle one JSON-RPC message **statelessly**. Returns `Some(response)` for a request,
/// `None` for a notification (which gets no reply). Stateful tools (the Pro
/// `octa_feedback`) are *refused visibly* here — labels this entry point accepted would
/// be dropped on return, and forgetting silently is exactly what the feedback channel
/// exists to avoid. Servers that keep state across calls use [`handle_with`], as the
/// stdio loop behind [`serve`]/[`serve_workspace`] does.
pub fn handle(session: &mut AgentSession, msg: &Value) -> Option<Value> {
    dispatch(session, None, msg)
}

/// [`handle`] with cross-call [`ServerState`] — the entry point the serve loop runs, and
/// the one that makes the stateful tools (octa-semantic feedback calibration) work.
pub fn handle_with(
    session: &mut AgentSession,
    state: &mut ServerState,
    msg: &Value,
) -> Option<Value> {
    dispatch(session, Some(state), msg)
}

fn dispatch(
    session: &mut AgentSession,
    state: Option<&mut ServerState>,
    msg: &Value,
) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    // Notifications carry no id and expect no response.
    id.as_ref()?;
    let id = id.unwrap();

    let result: Result<Value, (i64, String)> = match method {
        "initialize" => {
            let pv = msg
                .get("params")
                .and_then(|p| p.get("protocolVersion"))
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION)
                .to_string();
            Ok(json!({
                "protocolVersion": pv,
                "capabilities": { "tools": {}, "resources": {} },
                "serverInfo": { "name": "ccos-memory", "version": env!("CARGO_PKG_VERSION") }
            }))
        }
        "tools/list" => Ok(json!({ "tools": tool_specs() })),
        "tools/call" => call_tool(session, state, msg.get("params").unwrap_or(&Value::Null)),
        "resources/list" => Ok(json!({ "resources": resource_specs() })),
        "resources/read" => read_resource(session, msg.get("params").unwrap_or(&Value::Null)),
        "ping" => Ok(json!({})),
        _ => Err((-32601, format!("method not found: {method}"))),
    };

    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        }
    })
}

/// Run the stdio JSON-RPC loop on a fresh **in-memory** session (nothing is
/// persisted). See [`serve_workspace`] for the persistent variant.
pub fn serve() {
    serve_session(AgentSession::new());
}

/// Run the stdio loop, optionally persisting to (and reloading from) a workspace
/// checkpoint. With `Some(path)` the session loads that checkpoint on start and
/// re-checkpoints after every memory-changing call (and once more at EOF), so
/// the causal memory survives restarts; with `None` it behaves like [`serve`].
pub fn serve_workspace(
    workspace: Option<std::path::PathBuf>,
) -> Result<(), crate::external_memory::MemoryError> {
    let session = match workspace {
        Some(p) => AgentSession::open(p)?,
        None => AgentSession::new(),
    };
    serve_session(session);
    Ok(())
}

/// The shared stdio JSON-RPC loop until EOF. One JSON message per line; every
/// state-changing tool call is checkpointed before its reply goes out, and a
/// checkpoint that fails turns that reply into a visible error rather than a
/// silent loss (see [`step`] and [`mark_not_durable`]).
fn serve_session(mut session: AgentSession) {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    // Cross-call tool state (octa-semantic feedback log). Lives and dies with the
    // process, never with the workspace checkpoint — see `ServerState`.
    let mut state = ServerState::default();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if let Some(resp) = step(&mut session, &mut state, line.trim()) {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{resp}");
            let _ = out.flush();
        }
    }
    let _ = persist(&mut session); // final checkpoint at close; nobody left to tell
}

/// One request in, at most one reply out — the body of the stdio loop, lifted out
/// of it so the durability contract can be tested without a real stdin.
fn step(session: &mut AgentSession, state: &mut ServerState, line: &str) -> Option<Value> {
    if line.is_empty() {
        return None;
    }
    let Ok(msg) = serde_json::from_str::<Value>(line) else {
        return Some(json!({
            "jsonrpc": "2.0", "id": Value::Null,
            "error": { "code": -32700, "message": "parse error" }
        }));
    };
    let mutated = is_mutating_call(&msg);
    let resp = handle_with(session, state, &msg);
    // The checkpoint runs *before* the reply leaves, so a write that never reached
    // disk must not be reported as one — see `mark_not_durable`.
    match if mutated { persist(session) } else { Ok(()) } {
        Ok(()) => resp,
        Err(e) => resp.map(|r| mark_not_durable(r, &e)),
    }
}

/// Turn an already-successful tool reply into a visible failure when the
/// checkpoint that should have made it durable did not reach disk.
///
/// `serve_session` checkpoints *before* writing the reply, so without this the
/// server answered `{"result": …}` — an unqualified success — to an `ingest`
/// whose workspace write had just failed (read-only file, full disk, revoked
/// permission). The stderr line the failure did emit is invisible to an MCP
/// client: the agent was told its memory was stored, kept working on that
/// belief, and lost everything when the process exited. Reporting the loss in
/// the reply is the only channel the caller actually reads.
///
/// The original payload is kept: what happened *in memory* is still true for the
/// rest of this session, and the caller may want it. Only the durability claim
/// is retracted.
fn mark_not_durable(mut resp: Value, err: &MemoryError) -> Value {
    let note = format!(
        "WARNING — NOT DURABLE: the operation applied to the in-memory graph, but \
         the workspace checkpoint that should have persisted it failed ({err}). \
         Everything above is lost when this server exits. Treat it as unsaved: fix \
         the workspace path, permissions or free space, then replay the call."
    );
    let Some(result) = resp.get_mut("result") else {
        return resp;
    };
    match result.get_mut("content").and_then(Value::as_array_mut) {
        Some(items) => items.push(json!({ "type": "text", "text": note })),
        None => result["content"] = json!([{ "type": "text", "text": note }]),
    }
    result["isError"] = Value::Bool(true);
    resp
}

/// True iff the message is a `tools/call` to a state-changing tool.
fn is_mutating_call(msg: &Value) -> bool {
    if msg.get("method").and_then(Value::as_str) != Some("tools/call") {
        return false;
    }
    let name = msg
        .get("params")
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    matches!(name, "ingest" | "signal_failure" | "page_fault")
}

/// Checkpoint the session. Having no path bound is not a failure (that server
/// was never asked to persist anything); a real IO/serialisation error is, and
/// is both logged to stderr (stdout is reserved for JSON-RPC) and returned so
/// the caller's reply can carry the bad news.
fn persist(session: &mut AgentSession) -> Result<(), MemoryError> {
    match session.checkpoint() {
        Ok(()) | Err(MemoryError::NoPath) => Ok(()),
        Err(e) => {
            eprintln!("ccos mcp: checkpoint failed: {e}");
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: i64, method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    }

    #[test]
    fn initialize_echoes_protocol_and_names_the_server() {
        let mut s = AgentSession::new();
        let r = handle(
            &mut s,
            &req(1, "initialize", json!({ "protocolVersion": "2025-01-01" })),
        )
        .unwrap();
        assert_eq!(r["result"]["protocolVersion"], "2025-01-01");
        assert_eq!(r["result"]["serverInfo"]["name"], "ccos-memory");
        assert!(r["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_advertises_the_catalogue() {
        let mut s = AgentSession::new();
        let r = handle(&mut s, &req(2, "tools/list", Value::Null)).unwrap();
        let names: Vec<&str> = r["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for n in [
            "ingest",
            "recall",
            "signal_failure",
            "page_fault",
            "stats",
            "verify",
            "timeline",
            "recall_what_if",
            "ccos_retrieve",
            "causal_intervene",
            "causal_blame",
            "drift_cause",
            "retrodict_belief",
            "causal_flash",
        ] {
            assert!(names.contains(&n), "missing tool {n}");
        }
    }

    #[test]
    fn notification_gets_no_response() {
        let mut s = AgentSession::new();
        let n = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(handle(&mut s, &n).is_none());
    }

    #[test]
    fn ingest_then_recall_round_trips_through_tools() {
        let mut s = AgentSession::new();
        handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({
                    "name": "ingest",
                    "arguments": { "uri": "src/a.rs", "source": "pub fn a() {}\n" }
                }),
            ),
        )
        .unwrap();
        let r = handle(
            &mut s,
            &req(
                2,
                "tools/call",
                json!({
                    "name": "recall",
                    "arguments": { "strategy": "working_set", "budget": 1000 }
                }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/a.rs"),
            "recall returns the ingested file: {text}"
        );
    }

    #[test]
    fn unknown_method_is_a_jsonrpc_error() {
        let mut s = AgentSession::new();
        let r = handle(&mut s, &req(9, "frobnicate", Value::Null)).unwrap();
        assert_eq!(r["error"]["code"], -32601);
    }

    /// The Pro `octa-semantic` strategy: visible refusal on the community tier, and on
    /// the Pro tier an anchor-first window recalled *through the session* (the op lands
    /// in the timeline like every other recall).
    #[cfg(feature = "octasoma")]
    #[test]
    fn octa_semantic_is_pro_gated_and_anchors_the_window() {
        use crate::license::{License, Licensing};

        let mut s = AgentSession::new();
        s.ingest(
            "src/db.rs",
            "pub fn query() -> i64 { 1 }\npub fn pool() -> i64 { 2 }\n",
        );
        let call = |s: &mut AgentSession, id: i64| {
            handle(
                s,
                &req(
                    id,
                    "tools/call",
                    json!({
                        "name": "recall",
                        "arguments": { "strategy": "octa-semantic", "text": "pub fn query() -> i64 { 1 }", "budget": 512 }
                    }),
                ),
            )
            .unwrap()
        };

        // Community tier → a visible tool-level refusal (isError), not a protocol error,
        // and not a silent fallback.
        let refused = call(&mut s, 1);
        assert_eq!(refused["result"]["isError"], true);
        let msg = refused["result"]["content"][0]["text"].as_str().unwrap();
        assert!(msg.contains("Pro"), "the refusal explains the tier: {msg}");

        // Pro tier → the anchor-first window, strategy visible in the payload.
        s.set_licensing(Licensing::licensed(License {
            licensee: "acme".into(),
            expires_at: None,
            machine: None,
        }));
        let ok = call(&mut s, 2);
        let text = ok["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("\"strategy\":\"octa-semantic\""),
            "strategy is visible: {text}"
        );
        assert!(
            text.contains("db.rs"),
            "the anchor's region is recalled: {text}"
        );

        // The catalogue advertises the strategy in this build.
        let tools = handle(&mut s, &req(3, "tools/list", Value::Null)).unwrap();
        assert!(tools["result"]["tools"]
            .to_string()
            .contains("octa-semantic"));
    }

    /// The explicit feedback channel over MCP: labels accumulate in the server-held
    /// state, certify a conformal floor, and the floor gates the next octa-semantic
    /// anchors — certified when the anchor clears it, visible lexical fallback when
    /// it does not.
    #[cfg(feature = "octasoma")]
    #[test]
    fn octa_feedback_calibrates_the_conformal_anchor_gate() {
        use crate::license::{License, Licensing};

        let mut s = AgentSession::new();
        s.ingest(
            "src/db.rs",
            "pub fn query() -> i64 { 1 }\npub fn pool() -> i64 { 2 }\n",
        );
        s.set_licensing(Licensing::licensed(License {
            licensee: "acme".into(),
            expires_at: None,
            machine: None,
        }));
        let mut st = ServerState::default();
        let exact = "pub fn query() -> i64 { 1 }";

        let recall = |s: &mut AgentSession, st: &mut ServerState, id: i64, text: &str| {
            let r = handle_with(
                s,
                st,
                &req(
                    id,
                    "tools/call",
                    json!({ "name": "recall",
                        "arguments": { "strategy": "octa-semantic", "text": text,
                                       "budget": 512, "alpha": 0.25 } }),
                ),
            )
            .unwrap();
            let text = r["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_string();
            serde_json::from_str::<Value>(&text).expect("payload is JSON")
        };

        // Before any label: baseline strategy, and the calibration block says why the
        // gate is inactive (no floor, zero labels) — visible, not silent.
        let p = recall(&mut s, &mut st, 1, exact);
        assert_eq!(p["window"]["strategy"], "octa-semantic");
        assert_eq!(p["calibration"]["floor"], Value::Null);
        assert_eq!(p["calibration"]["labels"], 0);
        // The resolution is reported so the client can label it: an exact-content
        // query anchors at distance 0 → score 1.0.
        assert!(p["anchor"]["uri"].as_str().unwrap().contains("db.rs"));
        assert!((p["anchor"]["score"].as_f64().unwrap() - 1.0).abs() < 1e-12);

        // Three positive labels on the last resolution (score 1.0) → nonconformities
        // all 0 → the floor certifies at 1.0 for alpha = 0.25 (k = ⌈4·0.75⌉ = 3 ≤ n).
        for id in 2..5 {
            let r = handle_with(
                &mut s,
                &mut st,
                &req(
                    id,
                    "tools/call",
                    json!({ "name": "octa_feedback",
                        "arguments": { "relevant": true, "alpha": 0.25 } }),
                ),
            )
            .unwrap();
            let text = r["result"]["content"][0]["text"].as_str().unwrap();
            let p: Value = serde_json::from_str(text).unwrap();
            assert_eq!(p["labels"], id - 1);
        }

        // Anchor at the floor → certified.
        let p = recall(&mut s, &mut st, 5, exact);
        assert_eq!(p["window"]["strategy"], "octa-semantic-certified");
        assert!((p["calibration"]["floor"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(p["calibration"]["labels"], 3);

        // A non-matching query still resolves *some* nearest anchor, but below the
        // certified floor → the anchor is refused and the fallback is visible.
        let p = recall(&mut s, &mut st, 6, "unrelated gibberish");
        assert_eq!(
            p["window"]["strategy"],
            "octa-semantic-below-floor-fallback-task"
        );

        // The catalogue advertises the feedback surface in this build.
        let tools = handle(&mut s, &req(7, "tools/list", Value::Null)).unwrap();
        let ts = tools["result"]["tools"].to_string();
        assert!(ts.contains("octa_feedback") && ts.contains("alpha"));
    }

    /// `octa_feedback` never forgets silently and never downgrades silently: the
    /// stateless entry refuses it, the community tier gets the Pro refusal, and a
    /// label with nothing to refer to is an explicit error.
    #[cfg(feature = "octasoma")]
    #[test]
    fn octa_feedback_refuses_stateless_unlicensed_and_unanchored_calls() {
        use crate::license::{License, Licensing};

        let fb_req = |id: i64| {
            req(
                id,
                "tools/call",
                json!({ "name": "octa_feedback", "arguments": { "relevant": true } }),
            )
        };

        // Community tier → the Pro refusal (isError, tool-level), like octa-semantic.
        let mut s = AgentSession::new();
        let r = handle(&mut s, &fb_req(1)).unwrap();
        assert_eq!(r["result"]["isError"], true);
        assert!(r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Pro"));

        // Pro but stateless `handle` → the label would be dropped on return, so the
        // call is refused with the reason — never accepted-and-forgotten.
        s.set_licensing(Licensing::licensed(License {
            licensee: "acme".into(),
            expires_at: None,
            machine: None,
        }));
        let r = handle(&mut s, &fb_req(2)).unwrap();
        assert_eq!(r["result"]["isError"], true);
        assert!(r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("stateless"));

        // Stateful but nothing recalled yet → explicit error, not a fabricated label.
        let mut st = ServerState::default();
        let r = handle_with(&mut s, &mut st, &fb_req(3)).unwrap();
        assert_eq!(r["result"]["isError"], true);
        assert!(r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("no octa-semantic recall"));
    }

    /// A session with the import chain api → repo → db (each depends on the next).
    fn chain(s: &mut AgentSession) {
        ingest(s, 1, "src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
        ingest(
            s,
            2,
            "src/repo.rs",
            "use crate::db;\npub fn fetch() -> i64 { db::timeout() }\n",
        );
        ingest(
            s,
            3,
            "src/api.rs",
            "use crate::repo;\npub fn handle() -> i64 { repo::fetch() }\n",
        );
    }

    #[test]
    fn causal_intervene_and_blame_answer_over_mcp() {
        let mut s = AgentSession::new();
        chain(&mut s);
        // do(db): repo and api depend on it, so both are forced (bare path is normalized).
        let r = handle(
            &mut s,
            &req(
                4,
                "tools/call",
                json!({ "name": "causal_intervene", "arguments": { "node": "src/db.rs" } }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/repo.rs") && text.contains("file:src/api.rs"),
            "do(db) forces its dependents: {text}"
        );
        // blame(api): its dependencies are the candidate causes.
        let r = handle(
            &mut s,
            &req(
                5,
                "tools/call",
                json!({ "name": "causal_blame", "arguments": { "node": "src/api.rs" } }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/repo.rs") && text.contains("file:src/db.rs"),
            "blame(api) surfaces its dependencies: {text}"
        );
        // A missing 'node' argument is a JSON-RPC invalid-params error.
        let r = handle(
            &mut s,
            &req(6, "tools/call", json!({ "name": "causal_intervene" })),
        )
        .unwrap();
        assert_eq!(r["error"]["code"], -32602);
    }

    #[test]
    fn causal_flash_returns_a_bounded_cone_over_mcp() {
        let mut s = AgentSession::new();
        chain(&mut s); // api → repo → db, all Stable (no Working seed)

        // No Working nodes and default (no low-trust) seeding ⇒ an empty,
        // well-formed window. Verifies dispatch, arg defaults, and JSON shape.
        let r = handle(
            &mut s,
            &req(4, "tools/call", json!({ "name": "causal_flash" })),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(v["seed_count"], 0);
        assert_eq!(v["complete"], true);
        assert_eq!(v["nodes"].as_array().unwrap().len(), 0);

        // Force seeding without mutating node state: clean nodes have trust 1.0,
        // so a threshold above 1.0 makes every node a seed. The whole (closed)
        // dependency chain then reports complete with no omissions.
        let r = handle(
            &mut s,
            &req(
                5,
                "tools/call",
                json!({
                    "name": "causal_flash",
                    "arguments": {
                        "include_low_trust_seeds": true,
                        "trust_threshold": 1.5,
                        "horizon": 4
                    }
                }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        let v: Value = serde_json::from_str(text).unwrap();
        let nodes = v["nodes"].as_array().unwrap();
        // Ingest builds file + symbol + import nodes, so the graph has more than
        // three nodes; with the threshold above 1.0 every clean node seeds, so
        // seed_count equals the node count and all roles are "seed".
        assert!(
            v["seed_count"].as_u64().unwrap() >= 3,
            "chain seeded: {text}"
        );
        assert_eq!(v["seed_count"].as_u64().unwrap() as usize, nodes.len());
        assert!(nodes.iter().all(|n| n["role"] == "seed"));
        assert_eq!(v["complete"], true, "closed chain ⇒ complete");
        assert_eq!(v["omitted"], 0);
        let ids: Vec<&str> = nodes.iter().map(|n| n["node"].as_str().unwrap()).collect();
        assert!(
            ids.contains(&"file:src/db.rs")
                && ids.contains(&"file:src/repo.rs")
                && ids.contains(&"file:src/api.rs"),
            "the cone covers the whole chain: {text}"
        );
    }

    #[test]
    fn recall_causal_flash_strategy_selects_the_cone_over_mcp() {
        let mut s = AgentSession::new();
        chain(&mut s); // api → repo → db

        // The `recall` tool with the causal-flash strategy routes through
        // session.recall (so the op is journaled and replay-exact) and the
        // window assembler fits the token budget. trust_threshold > 1 seeds
        // every clean node without mutating state, so the cone spans the chain.
        let r = handle(
            &mut s,
            &req(
                7,
                "tools/call",
                json!({
                    "name": "recall",
                    "arguments": {
                        "strategy": "causal-flash",
                        "include_low_trust_seeds": true,
                        "trust_threshold": 1.5,
                        "budget": 4096
                    }
                }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(
            v["strategy"], "causal-flash",
            "window labels the strategy: {text}"
        );
        let uris: Vec<&str> = v["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|it| it["uri"].as_str().unwrap())
            .collect();
        assert!(
            uris.iter().any(|u| u.starts_with("file:src/")),
            "the cone recall selected chain nodes: {text}"
        );
    }

    #[test]
    fn drift_cause_names_the_culprit_op_over_mcp() {
        let mut s = AgentSession::new();
        chain(&mut s);
        handle(
            &mut s,
            &req(
                4,
                "tools/call",
                json!({ "name": "signal_failure", "arguments": { "node": "file:src/api.rs", "depth": 2 } }),
            ),
        )
        .unwrap();
        let r = handle(
            &mut s,
            &req(
                5,
                "tools/call",
                json!({ "name": "drift_cause", "arguments": { "node": "src/api.rs" } }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("\"op\"") && text.contains("\"step\""),
            "a drift attribution names the op and step: {text}"
        );
        // A node with no trajectory reports honestly instead of erroring.
        let r = handle(
            &mut s,
            &req(
                6,
                "tools/call",
                json!({ "name": "drift_cause", "arguments": { "node": "src/ghost.rs" } }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("no attributable drift"),
            "honest null: {text}"
        );
    }

    #[test]
    fn retrodict_belief_returns_raw_and_smoothed_series() {
        let mut s = AgentSession::new();
        // Build a claim whose belief grows over the timeline.
        for (i, ev) in ["e0", "e1", "e2"].iter().enumerate() {
            handle(
                &mut s,
                &req(
                    i as i64 + 1,
                    "tools/call",
                    json!({ "name": "ingest", "arguments": {
                        "uri": format!("src/{ev}.rs"), "source": "pub fn x() {}\n" } }),
                ),
            )
            .unwrap();
            s.assert_support(&format!("file:src/{ev}.rs"), "claim:db-is-slow", 1.0);
        }
        let r = handle(
            &mut s,
            &req(
                9,
                "tools/call",
                json!({ "name": "retrodict_belief", "arguments": { "claim": "claim:db-is-slow" } }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        let v: Value = serde_json::from_str(text).unwrap();
        let raw = v["belief"].as_array().unwrap();
        let smooth = v["belief_retrodicted"].as_array().unwrap();
        assert_eq!(
            raw.len(),
            smooth.len(),
            "same sampling for raw and smoothed"
        );
        assert!(!raw.is_empty());
        // The belief ends positive (three supports) in both views.
        assert!(raw.last().unwrap().as_f64().unwrap() > 0.0);
        assert!(smooth.last().unwrap().as_f64().unwrap() > 0.0);
    }

    fn ingest(s: &mut AgentSession, id: i64, uri: &str, src: &str) {
        handle(
            s,
            &req(
                id,
                "tools/call",
                json!({ "name": "ingest", "arguments": { "uri": uri, "source": src } }),
            ),
        )
        .unwrap();
    }

    #[test]
    fn time_travel_what_if_replays_a_past_step() {
        let mut s = AgentSession::new();
        ingest(&mut s, 1, "src/db.rs", "pub fn q() {}\n");
        ingest(
            &mut s,
            2,
            "src/api.rs",
            "use crate::db;\npub fn h() { db::q() }\n",
        );
        // Rewind to step 1 (only db.rs ingested): the window must predate api.rs.
        let r = handle(
            &mut s,
            &req(
                3,
                "tools/call",
                json!({
                    "name": "recall_what_if",
                    "arguments": { "step": 1, "strategy": "working_set", "budget": 4000 }
                }),
            ),
        )
        .unwrap();
        let text = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/db.rs"),
            "what-if sees db.rs: {text}"
        );
        assert!(
            !text.contains("file:src/api.rs"),
            "step-1 replay predates api.rs: {text}"
        );
    }

    #[test]
    fn initialize_advertises_resources() {
        let mut s = AgentSession::new();
        let r = handle(&mut s, &req(1, "initialize", json!({}))).unwrap();
        assert!(r["result"]["capabilities"]["resources"].is_object());
    }

    #[test]
    fn resources_list_and_read_the_context_window() {
        let mut s = AgentSession::new();
        ingest(&mut s, 1, "src/a.rs", "pub fn alpha() {}\n");

        let list = handle(&mut s, &req(2, "resources/list", Value::Null)).unwrap();
        let uris: Vec<&str> = list["result"]["resources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["uri"].as_str().unwrap())
            .collect();
        assert!(uris.contains(&"ccos://session/context"));

        let read = handle(
            &mut s,
            &req(
                3,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let text = read["result"]["contents"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/a.rs"),
            "context resource linearises the working set: {text}"
        );
    }

    #[test]
    fn context_resource_anchors_on_the_active_failure() {
        let mut s = AgentSession::new();
        ingest(&mut s, 1, "src/db.rs", "pub fn q() {}\n");
        ingest(
            &mut s,
            2,
            "src/api.rs",
            "use crate::db;\npub fn h() { db::q() }\n",
        );
        // A failure on db.rs → the injected context should be db.rs's causal region.
        handle(
            &mut s,
            &req(
                3,
                "tools/call",
                json!({ "name": "signal_failure", "arguments": { "node": "file:src/db.rs" } }),
            ),
        )
        .unwrap();
        let read = handle(
            &mut s,
            &req(
                4,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let text = read["result"]["contents"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("file:src/db.rs"),
            "context anchors on the failing file: {text}"
        );
    }

    #[test]
    fn unknown_resource_is_a_jsonrpc_error() {
        let mut s = AgentSession::new();
        let r = handle(
            &mut s,
            &req(1, "resources/read", json!({ "uri": "ccos://session/nope" })),
        )
        .unwrap();
        assert_eq!(r["error"]["code"], -32602);
    }

    /// A mutating call whose checkpoint failed must not come back as a success.
    ///
    /// The server checkpoints *before* it writes the reply, so it used to answer
    /// a plain `{"result": …}` to an `ingest` whose workspace write had just
    /// failed. The only sign was a stderr line no MCP client ever reads: the
    /// agent was told its memory was stored, kept reasoning on that belief, and
    /// lost the lot when the process exited.
    #[test]
    fn a_failed_checkpoint_is_reported_in_the_reply_not_only_on_stderr() {
        let root = std::env::temp_dir().join(format!("ccos-mcp-durability-{}", std::process::id()));
        let dir = root.join("ws");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = AgentSession::open(dir.join("workspace.ccos")).unwrap();
        let mut state = ServerState::default();

        let ingest = |n: i64, path: &str| {
            serde_json::to_string(&req(
                n,
                "tools/call",
                json!({ "name": "ingest", "arguments": {
                    "uri": path, "source": "pub fn f() {}\n" } }),
            ))
            .unwrap()
        };

        // While the workspace is writable the reply is an unqualified success.
        let ok = step(&mut session, &mut state, &ingest(1, "src/a.rs")).unwrap();
        assert!(ok["result"]["isError"].is_null(), "durable ingest: {ok}");

        // Now break the workspace for real. Deleting the directory is not enough —
        // `write_durable` recreates it on purpose — so put a *regular file* where
        // the directory was: `create_dir_all` then fails outright. (No chmod games:
        // CI may well run as root, which ignores permission bits.)
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::write(&dir, b"not a directory").unwrap();

        let lost = step(&mut session, &mut state, &ingest(2, "src/b.rs")).unwrap();
        assert_eq!(
            lost["result"]["isError"], true,
            "an ingest that never reached disk must be flagged: {lost}"
        );
        let texts: String = lost["result"]["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c["text"].as_str())
            .collect();
        assert!(
            texts.contains("NOT DURABLE"),
            "caller must be told: {texts}"
        );

        // A read-only call is not checkpointed, so a broken workspace does not
        // make `stats` lie about itself.
        let stats = step(
            &mut session,
            &mut state,
            &serde_json::to_string(&req(3, "tools/call", json!({ "name": "stats" }))).unwrap(),
        )
        .unwrap();
        assert!(
            stats["result"]["isError"].is_null(),
            "read-only calls stay unmarked: {stats}"
        );
    }

    /// A recall missing its query must be refused, not answered with an empty
    /// window.
    ///
    /// The window a blank query produced was perfectly well-formed — right
    /// strategy label, zero items, zero tokens — so a caller that used the wrong
    /// argument name could not tell its typo from an empty memory. Measured on a
    /// real 165-file workspace: `{"strategy":"task","task":"NdLinear"}` (the field
    /// is `text`) returned nothing, while the same query under `text` returned 15
    /// items and filled the whole 2048-token budget.
    #[test]
    fn a_recall_without_its_query_is_refused_not_answered_with_an_empty_window() {
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn query() {}\n");

        let call = |name: &str, args: Value| {
            req(1, "tools/call", json!({ "name": name, "arguments": args }))
        };

        for (strategy, present, absent) in [
            ("around", "anchor", "text"),
            ("task", "text", "anchor"),
            ("semantic", "text", "anchor"),
            ("hybrid", "text", "anchor"),
        ] {
            // The wrong key, which is what a caller actually gets wrong.
            let mut s2 = AgentSession::new();
            s2.ingest("src/db.rs", "pub fn query() {}\n");
            let wrong = handle(
                &mut s2,
                &call("recall", json!({ "strategy": strategy, absent: "query" })),
            )
            .unwrap();
            assert_eq!(
                wrong["error"]["code"], -32602,
                "{strategy} with only '{absent}' must be refused: {wrong}"
            );
            let msg = wrong["error"]["message"].as_str().unwrap_or_default();
            assert!(
                msg.contains(present) && msg.contains(strategy),
                "the refusal must name the strategy and the field it wants: {msg}"
            );

            // Blank is the same as absent — a caller passing "" asked nothing.
            let blank = handle(
                &mut s2,
                &call("recall", json!({ "strategy": strategy, present: "   " })),
            )
            .unwrap();
            assert_eq!(blank["error"]["code"], -32602, "{strategy} blank: {blank}");
        }

        // The strategy that genuinely needs no query is untouched.
        let ws = handle(
            &mut s,
            &call("recall", json!({ "strategy": "working_set" })),
        )
        .unwrap();
        assert!(
            ws["result"]["content"][0]["text"].is_string(),
            "working_set still answers without a query: {ws}"
        );

        // And `recall_what_if` guards the same way — it takes the same arguments.
        let what_if = handle(
            &mut s,
            &call("recall_what_if", json!({ "strategy": "around", "step": 0 })),
        )
        .unwrap();
        assert_eq!(what_if["error"]["code"], -32602, "{what_if}");
    }

    /// `ingest` must not accept a request with no `source`.
    ///
    /// Its own refusal message named both `uri` and `source`, but only `uri` was
    /// checked, so the call succeeded and put an empty node in the graph — a file
    /// that exists, in the memory, with no content, indistinguishable downstream
    /// from a genuinely empty file. An explicit `""` still means exactly that and
    /// stays legal.
    #[test]
    fn ingest_requires_the_source_its_error_message_has_always_claimed() {
        let mut s = AgentSession::new();
        let call = |args: Value| {
            req(
                1,
                "tools/call",
                json!({ "name": "ingest", "arguments": args }),
            )
        };

        let no_source = handle(&mut s, &call(json!({ "uri": "file:x.rs" }))).unwrap();
        assert_eq!(no_source["error"]["code"], -32602, "{no_source}");
        assert!(
            no_source["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("source"),
            "the refusal must name the missing field: {no_source}"
        );

        // An empty file is a real thing and must still ingest.
        let empty = handle(
            &mut s,
            &call(json!({ "uri": "file:empty.rs", "source": "" })),
        )
        .unwrap();
        assert!(
            empty["result"]["content"][0]["text"].is_string(),
            "an explicitly empty source is legal: {empty}"
        );
    }

    /// Time travel must refuse a moment it cannot reconstruct.
    ///
    /// `replay_to` clamps in both directions and cannot report that it did — it
    /// returns a memory, not a `Result`. Past the end it hands back the present;
    /// below the compaction floor it hands back the state *at* the floor. Either
    /// way `recall_what_if` answered with a well-formed window for a moment that
    /// is not the one it was asked about. Measured on a 3-op timeline before this
    /// guard: `step: 9999` returned byte-for-byte what `step: 3` returned. For the
    /// tool whose entire job is "what did memory look like then", that is the one
    /// failure mode that matters.
    #[test]
    fn time_travel_refuses_a_step_it_cannot_reconstruct() {
        let mut s = AgentSession::new();
        for i in 0..3 {
            s.ingest(&format!("src/f{i}.rs"), &format!("pub fn f{i}() {{}}\n"));
        }
        let what_if = |step: usize| {
            req(
                1,
                "tools/call",
                json!({ "name": "recall_what_if",
                        "arguments": { "strategy": "working_set", "step": step } }),
            )
        };

        // Every step that exists replays, including both ends. `recall_what_if` is
        // read-only — it replays into a throwaway memory — so one session serves.
        let last = s.len();
        for step in 0..=last {
            let r = handle(&mut s, &what_if(step)).unwrap();
            assert!(
                r["result"]["content"][0]["text"].is_string(),
                "step {step} is in range and must replay: {r}"
            );
        }

        // One past the end is already a moment that never happened.
        let past_end = handle(&mut s, &what_if(last + 1)).unwrap();
        assert_eq!(past_end["error"]["code"], -32602, "{past_end}");
        assert!(
            past_end["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("past the end"),
            "{past_end}"
        );

        // Below the compaction floor the answer would be the floor's state wearing
        // the requested step's name — the quieter and more misleading of the two.
        let mut compacted = AgentSession::new();
        for i in 0..40 {
            compacted.ingest(&format!("src/g{i}.rs"), &format!("pub fn g{i}() {{}}\n"));
        }
        compacted.compact(10);
        let floor = compacted.floor();
        assert!(floor > 1, "fixture must actually compact, floor={floor}");

        let below = handle(&mut compacted, &what_if(1)).unwrap();
        assert_eq!(below["error"]["code"], -32602, "{below}");
        let msg = below["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("compaction floor") && msg.contains(&floor.to_string()),
            "the refusal must name the floor so the caller can pick a real step: {msg}"
        );

        // Step 0 stays legal: "before anything happened" is always reconstructible,
        // and the floor never swallows it.
        let zero = handle(&mut compacted, &what_if(0)).unwrap();
        assert!(zero["result"]["content"][0]["text"].is_string(), "{zero}");
        // And the floor itself is exactly the baseline, so it replays.
        let at_floor = handle(&mut compacted, &what_if(floor)).unwrap();
        assert!(
            at_floor["result"]["content"][0]["text"].is_string(),
            "{at_floor}"
        );
    }

    /// `sync` must persist, and must not answer `ok: true` when it could not.
    ///
    /// `sync` is the boot/refresh acknowledgement a gateway calls to confirm the
    /// session is durable — making it durable is the whole job — and it failed at
    /// that job twice over. It discarded the checkpoint result
    /// (`let _ = session.checkpoint()`) and answered `{"ok":true,…}` regardless,
    /// so a gateway got a clean boot ack for a session that had persisted nothing;
    /// the only trace was a stderr line no MCP client reads. And it only
    /// checkpointed under `force`, while its description has always said it
    /// checkpoints — so the *default* call, the one a gateway actually makes at
    /// boot, wrote nothing at all.
    ///
    /// Same shape as the `ingest` case, but worse: for `ingest`, durability is a
    /// side effect; here it is the entire contract.
    #[test]
    fn sync_persists_and_reports_a_flush_that_failed() {
        let root = std::env::temp_dir().join(format!("ccos-sync-force-{}", std::process::id()));
        let dir = root.join("ws");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = AgentSession::open(dir.join("workspace.ccos")).unwrap();
        session.ingest("src/a.rs", "pub fn a() {}\n");

        let sync = req(
            1,
            "tools/call",
            json!({ "name": "sync", "arguments": { "force": true } }),
        );
        // The default call — no `force` — is the one a gateway makes at boot, and
        // it is the one that used to write nothing.
        let plain_sync = req(1, "tools/call", json!({ "name": "sync" }));

        let acked = handle(&mut session, &plain_sync).unwrap();
        assert!(
            acked["result"]["content"][0]["text"].is_string(),
            "a plain sync acks: {acked}"
        );
        assert!(
            dir.join("workspace.ccos").is_file(),
            "…and it must actually have written the workspace"
        );

        let healthy = handle(&mut session, &sync).unwrap();
        assert!(
            healthy["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .contains("\"ok\":true"),
            "a healthy flush still acks: {healthy}"
        );

        // Break the workspace the same way as elsewhere in this file: a regular
        // file where the directory was, so `create_dir_all` fails outright and no
        // chmod is involved (CI may run as root).
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::write(&dir, b"not a directory").unwrap();

        for (label, call) in [("force", &sync), ("plain", &plain_sync)] {
            let refused = handle(&mut session, call).unwrap();
            assert_eq!(
                refused["error"]["code"], -32603,
                "{label} sync must not ack a failed flush: {refused}"
            );
        }
        let broken = handle(&mut session, &sync).unwrap();
        assert_eq!(
            broken["error"]["code"], -32603,
            "a failed flush must not ack: {broken}"
        );
        assert!(
            broken["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("durable"),
            "{broken}"
        );

        // A session with nothing bound is not a failure: `force` is documented as
        // a no-op there, and it must stay one.
        let mut unbound = AgentSession::new();
        unbound.ingest("src/a.rs", "pub fn a() {}\n");
        let no_path = handle(&mut unbound, &sync).unwrap();
        assert!(
            no_path["result"]["content"][0]["text"].is_string(),
            "an unbound session still acks: {no_path}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Every argument the catalogue marks `required` must actually be required.
    ///
    /// `"required": ["output"]` in an advertised schema tells a client the call is
    /// invalid without it. Nothing enforced that, and three tools took the
    /// omission as a default instead: `page_fault` with no compiler output
    /// recalled a plain working set and returned it looking like a page-fault
    /// result, `recall_what_if` with no step replayed step 0 and returned an empty
    /// window, `signal_failure` with no node answered "node not found: file:" as
    /// though a node had been named and was missing.
    ///
    /// Driven off the catalogue rather than a hand-written list, so a tool added
    /// later cannot quietly opt out of its own declaration.
    #[test]
    fn every_declared_required_argument_is_enforced() {
        let specs = tool_specs();
        let mut checked = 0;
        for spec in specs.as_array().expect("catalogue is an array") {
            let name = spec["name"].as_str().expect("tool has a name");
            let Some(required) = spec["inputSchema"]["required"].as_array() else {
                continue;
            };
            for key in required.iter().filter_map(Value::as_str) {
                // Supply every *other* required key, so the refusal can only be
                // about the one left out.
                let mut args = serde_json::Map::new();
                for other in required.iter().filter_map(Value::as_str) {
                    if other != key {
                        args.insert(other.to_string(), json!("x"));
                    }
                }
                let mut s = AgentSession::new();
                s.ingest("src/a.rs", "pub fn a() {}\n");
                let r = handle(
                    &mut s,
                    &req(
                        1,
                        "tools/call",
                        json!({ "name": name, "arguments": Value::Object(args) }),
                    ),
                )
                .unwrap();
                assert_eq!(
                    r["error"]["code"], -32602,
                    "{name} declares '{key}' required but accepted the call: {r}"
                );
                assert!(
                    r["error"]["message"]
                        .as_str()
                        .unwrap_or_default()
                        .contains(key),
                    "the refusal must name the missing argument: {r}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 9, "expected the whole catalogue, saw {checked}");
    }

    /// Presence is not always enough, and emptiness does not always mean absent.
    #[test]
    fn empty_values_are_judged_per_tool_not_uniformly() {
        let mut s = AgentSession::new();

        // An empty compiler output names no faulting file, so the "page fault"
        // would be an ordinary working-set recall wearing the wrong label.
        let blank_fault = handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({ "name": "page_fault", "arguments": { "output": "   " } }),
            ),
        )
        .unwrap();
        assert_eq!(blank_fault["error"]["code"], -32602, "{blank_fault}");

        // An empty *file* is a real thing and must still ingest.
        let empty_file = handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({ "name": "ingest",
                        "arguments": { "uri": "file:empty.rs", "source": "" } }),
            ),
        )
        .unwrap();
        assert!(
            empty_file["result"]["content"][0]["text"].is_string(),
            "{empty_file}"
        );

        // A `step` that is present but not a number fell through `as_u64` to 0 and
        // silently replayed the empty baseline.
        let bad_step = handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({ "name": "recall_what_if",
                        "arguments": { "strategy": "working_set", "step": "2" } }),
            ),
        )
        .unwrap();
        assert_eq!(bad_step["error"]["code"], -32602, "{bad_step}");
    }

    #[test]
    fn only_state_changing_tools_trigger_a_checkpoint() {
        let mutating = |name: &str| {
            is_mutating_call(&json!({
                "method": "tools/call", "params": { "name": name }
            }))
        };
        assert!(mutating("ingest"));
        assert!(mutating("signal_failure"));
        assert!(mutating("page_fault"));
        assert!(!mutating("recall"));
        assert!(!mutating("stats"));
        assert!(!mutating("recall_what_if"));
        assert!(!mutating("ccos_retrieve"));
        // The causal/temporal analysis tools are read-only: no checkpoint after them.
        assert!(!mutating("causal_intervene"));
        assert!(!mutating("causal_blame"));
        assert!(!mutating("drift_cause"));
        assert!(!mutating("retrodict_belief"));
        assert!(!mutating("causal_flash"));
        // Non-tools/call messages never checkpoint.
        assert!(!is_mutating_call(&json!({ "method": "resources/read" })));
    }

    #[test]
    fn linearize_plain_drops_annotations() {
        let win = crate::external_memory::RecallWindow {
            strategy: "region".to_string(),
            items: vec![crate::external_memory::RecallItem {
                uri: "sym:src/config.rs:HEADER_SIZE".to_string(),
                score: 0.87,
                kind: "Symbol".to_string(),
                content: "pub const HEADER_SIZE: usize = 24;".to_string(),
                ccr_ref: None,
            }],
            tokens: 10,
        };
        let annotated = linearize_window(&win, false);
        assert!(annotated.contains("[Symbol]") && annotated.contains("score="));
        let plain = linearize_window(&win, true);
        assert!(
            plain.contains("// src/config.rs"),
            "plain uses the file path: {plain}"
        );
        assert!(
            !plain.contains("sym:") && !plain.contains("score="),
            "plain drops the annotations a weak model misreads: {plain}"
        );
        assert!(plain.contains("pub const HEADER_SIZE"));
    }

    // ── Compression: ccos_retrieve + compressed context resource ───────────

    use std::sync::Mutex;
    // The compression tests toggle a process-global env var, so they must not
    // run in parallel with each other (or with any other test reading that var).
    static COMPRESS_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Helper that ingests a Rust source file large enough to exercise the
    /// CausalAST compressor (the route a real `sym:`/`file:` node takes).
    fn ingest_code(s: &mut AgentSession, id: i64, uri: &str, code: &str) {
        handle(
            s,
            &req(
                id,
                "tools/call",
                json!({ "name": "ingest", "arguments": { "uri": uri, "source": code } }),
            ),
        )
        .unwrap();
    }

    /// A Rust source fixture with one large function (comments, blank lines,
    /// `_`-temporaries) — the structure CausalAST compresses best. Small
    /// one-liners don't amortize the CCR ref overhead.
    fn code_fixture() -> String {
        let mut s = String::from("pub fn big_calc() -> u64 {\n");
        for i in 0..60 {
            s.push_str(&format!(
                "    // phase {i} — accumulate intermediate\n    let _acc{i} = {i} * 2;\n    let _tmp{i} = _acc{i} + 1;\n"
            ));
        }
        s.push_str("    _tmp59\n}\n");
        s
    }

    #[test]
    fn ccos_retrieve_returns_the_original_for_a_known_ref() {
        let _guard = COMPRESS_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = AgentSession::new();
        let code = code_fixture();
        ingest_code(&mut s, 1, "src/calc.rs", &code);

        // The context resource uses recall_compressed by default
        // (CCOS_COMPRESS_CONTEXT != "0").
        std::env::set_var("CCOS_COMPRESS_CONTEXT", "1");
        let read = handle(
            &mut s,
            &req(
                2,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let text = read["result"]["contents"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        std::env::remove_var("CCOS_COMPRESS_CONTEXT");

        // The compressed context must carry at least one ccr_ref.
        let ref_str = text
            .lines()
            .find_map(|l| l.strip_prefix("// ccr_ref="))
            .map(|r| r.split_whitespace().next().unwrap_or(r).to_string());
        assert!(
            ref_str.is_some(),
            "context resource emitted a ccr_ref: {text}"
        );
        let ref_str = ref_str.unwrap();

        // Retrieve the original through the MCP tool. The "original" here is
        // the node content CCOS selected (a file header of signatures, not the
        // whole source — see docs/DESIGN_symbol_granularity.md); it must still
        // be the *uncompressed* form, distinct from the skeletonized version
        // the compressed resource showed.
        let r = handle(
            &mut s,
            &req(
                3,
                "tools/call",
                json!({ "name": "ccos_retrieve", "arguments": { "ccr_ref": ref_str } }),
            ),
        )
        .unwrap();
        assert!(
            !r["result"]["isError"].as_bool().unwrap_or(false),
            "retrieve succeeded: {r}"
        );
        let original = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            original.contains("big_calc"),
            "retrieved the original node content: {original}"
        );
    }

    #[test]
    fn ccos_retrieve_unknown_ref_is_an_error_response() {
        let mut s = AgentSession::new();
        let r = handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({ "name": "ccos_retrieve", "arguments": { "ccr_ref": "deadbeefdead" } }),
            ),
        )
        .unwrap();
        assert!(r["result"]["isError"] == true, "unknown ref → isError: {r}");
    }

    #[test]
    fn ccos_retrieve_requires_the_ref_argument() {
        let mut s = AgentSession::new();
        let r = handle(
            &mut s,
            &req(
                1,
                "tools/call",
                json!({ "name": "ccos_retrieve", "arguments": {} }),
            ),
        )
        .unwrap();
        assert_eq!(
            r["error"]["code"], -32602,
            "missing arg → JSON-RPC error: {r}"
        );
    }

    #[test]
    fn compressed_context_resource_is_smaller_than_raw() {
        let _guard = COMPRESS_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = AgentSession::new();
        let code = code_fixture();
        ingest_code(&mut s, 1, "src/calc.rs", &code);

        std::env::set_var("CCOS_COMPRESS_CONTEXT", "0");
        let raw = handle(
            &mut s,
            &req(
                2,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let raw_text = raw["result"]["contents"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        std::env::remove_var("CCOS_COMPRESS_CONTEXT");

        std::env::set_var("CCOS_COMPRESS_CONTEXT", "1");
        let compressed = handle(
            &mut s,
            &req(
                3,
                "resources/read",
                json!({ "uri": "ccos://session/context" }),
            ),
        )
        .unwrap();
        let comp_text = compressed["result"]["contents"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        std::env::remove_var("CCOS_COMPRESS_CONTEXT");

        assert!(
            comp_text.chars().count() < raw_text.chars().count(),
            "compressed context ({}) must be smaller than raw ({}):\nRAW={raw_text}\nCOMP={comp_text}",
            comp_text.chars().count(),
            raw_text.chars().count()
        );
    }
}
