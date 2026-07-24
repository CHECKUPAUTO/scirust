//! The SOS tool set: each already-landed engine syscall exposed as an MCP
//! tool.
//!
//! Store-backed **read** tools (`sos_log`, `sos_why`, `sos_verify`,
//! `sos_diff`, `sos_know`, `sos_ask`) wrap `sos-cli`'s own command functions
//! directly — the exact same code path `sos log`/`sos why`/… run, so there is
//! no second implementation to drift from the first. Pure-computation tools
//! (`sos_plan`, `sos_publish`, `sos_plugins`) call the engines directly with
//! arguments taken **inline** from the tool call rather than a file path —
//! more natural for an MCP client, which passes structured JSON, not
//! filesystem paths. `sos_propose` is the one genuinely new surface: the
//! untrusted-proposer entry point (Invariant IX).

use std::collections::BTreeMap;

use serde_json::{Value, json};
use sos_cli::args::{Args, parse_object_id};
use sos_core::{Author, ObjectId};
use sos_planner::{Candidate, GreedyPlanner, Planner, UtilityPolicy};
use sos_publication::Publication;
use sos_registry::{PluginDescriptor, Registry, Role};

use crate::registry::McpTool;

/// Read a required string field from a tool's arguments.
fn required_str(args: &Value, field: &str) -> Result<String, String> {
    args.get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing required argument `{field}`"))
}

/// Read an optional string field.
fn optional_str(args: &Value, field: &str) -> Option<String> {
    args.get(field).and_then(Value::as_str).map(str::to_owned)
}

/// Read an optional integer field.
fn optional_i64(args: &Value, field: &str) -> Option<i64> {
    args.get(field).and_then(Value::as_i64)
}

/// Parse a required object-id field.
fn required_id(args: &Value, field: &str) -> Result<ObjectId, String> {
    parse_object_id(&required_str(args, field)?).map_err(|e| e.to_string())
}

/// `sos_log` — list every object in a store.
#[must_use]
pub fn log_tool() -> McpTool {
    McpTool {
        name: "sos_log".to_owned(),
        description:
            "List every object in an SOS store, with kind, determinism level, and parent count."
                .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": { "store": { "type": "string", "description": "Path to the store directory." } },
            "required": ["store"],
        }),
        handler: Box::new(|args| {
            let store = required_str(&args, "store")?;
            sos_cli::log::run(Some(&store))
                .map(Value::String)
                .map_err(|e| e.to_string())
        }),
    }
}

/// `sos_why` — the provenance ancestor chain behind an object.
#[must_use]
pub fn why_tool() -> McpTool {
    McpTool {
        name: "sos_why".to_owned(),
        description: "Print every provenance ancestor of an object — why it is justified."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "store": { "type": "string" },
                "object": { "type": "string", "description": "The object id (sos1:... hex)." },
            },
            "required": ["store", "object"],
        }),
        handler: Box::new(|args| {
            let store = required_str(&args, "store")?;
            let id = required_id(&args, "object")?;
            sos_cli::why::run(Some(&store), id)
                .map(Value::String)
                .map_err(|e| e.to_string())
        }),
    }
}

/// `sos_verify` — structural + (where recognized) content-hash check.
#[must_use]
pub fn verify_tool() -> McpTool {
    McpTool {
        name: "sos_verify".to_owned(),
        description: "Check an object's structural identity, and recompute its content hash for every kind this server recognizes.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": { "store": { "type": "string" }, "object": { "type": "string" } },
            "required": ["store", "object"],
        }),
        handler: Box::new(|args| {
            let store = required_str(&args, "store")?;
            let id = required_id(&args, "object")?;
            sos_cli::verify::run(Some(&store), id).map(Value::String).map_err(|e| e.to_string())
        }),
    }
}

/// `sos_diff` — compare two studies' ancestor sets.
#[must_use]
pub fn diff_tool() -> McpTool {
    McpTool {
        name: "sos_diff".to_owned(),
        description: "Compare two studies by their provenance-ancestor sets: what's only in each, what's shared.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "store": { "type": "string" },
                "root_a": { "type": "string" },
                "root_b": { "type": "string" },
            },
            "required": ["store", "root_a", "root_b"],
        }),
        handler: Box::new(|args| {
            let store = required_str(&args, "store")?;
            let root_a = required_id(&args, "root_a")?;
            let root_b = required_id(&args, "root_b")?;
            sos_cli::diff::run(Some(&store), root_a, root_b).map(Value::String).map_err(|e| e.to_string())
        }),
    }
}

/// `sos_know` — knowledge-graph queries (neighbors / in-neighbors / related / path).
#[must_use]
pub fn know_tool() -> McpTool {
    McpTool {
        name: "sos_know".to_owned(),
        description: "Query the knowledge graph: `neighbors`/`in-neighbors <id> <relation>`, `related <a> <b>`, `path <a> <b> [relation]`.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "store": { "type": "string" },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "[subcommand, ...positional args], e.g. [\"neighbors\", \"<id>\", \"contradicts\"].",
                },
            },
            "required": ["store", "args"],
        }),
        handler: Box::new(|args| {
            let store = required_str(&args, "store")?;
            let positional = string_array(&args, "args")?;
            let parsed = Args {
                positional,
                flags: BTreeMap::new(),
            };
            sos_cli::know::run(Some(&store), &parsed).map(Value::String).map_err(|e| e.to_string())
        }),
    }
}

/// `sos_ask` — a curiosity sweep over the knowledge graph.
#[must_use]
pub fn ask_tool() -> McpTool {
    McpTool {
        name: "sos_ask".to_owned(),
        description: "Run a curiosity sweep: the ranked agenda of open questions found in the knowledge graph.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "store": { "type": "string" },
                "limit": { "type": "integer", "description": "Max questions returned (default 20)." },
            },
            "required": ["store"],
        }),
        handler: Box::new(|args| {
            let store = required_str(&args, "store")?;
            let mut flags = BTreeMap::new();
            if let Some(limit) = optional_i64(&args, "limit")
            {
                flags.insert("limit".to_owned(), limit.to_string());
            }
            let parsed = Args {
                positional: Vec::new(),
                flags,
            };
            sos_cli::ask::run(Some(&store), &parsed).map(Value::String).map_err(|e| e.to_string())
        }),
    }
}

/// Parse a required array-of-strings field.
fn string_array(args: &Value, field: &str) -> Result<Vec<String>, String> {
    args.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing required array argument `{field}`"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("`{field}` must be an array of strings"))
        })
        .collect()
}

/// `sos_plan` — recommend the next experiment from inline candidate estimates.
#[must_use]
pub fn plan_tool() -> McpTool {
    McpTool {
        name: "sos_plan".to_owned(),
        description: "Rank candidate experiment designs by expected-information-gain utility and recommend the next one (or report information exhausted).".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "candidates": { "type": "array", "description": "Array of Candidate {experiment, eig, cost}." },
                "floor": { "type": "integer", "description": "Significance floor in millibits (default 0)." },
                "budget": { "type": "integer", "description": "If set, ranks by EigBudgeted instead of EigPerCost." },
            },
            "required": ["candidates"],
        }),
        handler: Box::new(|args| {
            let candidates: Vec<Candidate> =
                serde_json::from_value(args.get("candidates").cloned().unwrap_or(Value::Null))
                    .map_err(|e| format!("invalid `candidates`: {e}"))?;
            let floor = optional_i64(&args, "floor").unwrap_or(0);
            let policy = match optional_i64(&args, "budget")
            {
                Some(budget) => UtilityPolicy::EigBudgeted { budget },
                None => UtilityPolicy::EigPerCost,
            };
            let plan = GreedyPlanner::new().recommend(&candidates, policy, floor).map_err(|e| e.to_string())?;
            serde_json::to_value(&plan).map_err(|e| e.to_string())
        }),
    }
}

/// `sos_publish` — seal (and optionally render) an inline publication.
#[must_use]
pub fn publish_tool() -> McpTool {
    McpTool {
        name: "sos_publish".to_owned(),
        description: "Seal a Publication (content-addressed) and, if `format` is given, render it to markdown/html/json.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "publication": { "type": "object", "description": "A Publication value." },
                "author": { "type": "string", "description": "Sealing principal (default \"sos-mcp\")." },
                "format": { "type": "string", "enum": ["md", "html", "json"] },
            },
            "required": ["publication"],
        }),
        handler: Box::new(|args| {
            let publication: Publication =
                serde_json::from_value(args.get("publication").cloned().unwrap_or(Value::Null))
                    .map_err(|e| format!("invalid `publication`: {e}"))?;
            let author = Author::human(optional_str(&args, "author").as_deref().unwrap_or("sos-mcp"));
            let sealed = sos_publication::seal_publication(publication.clone(), author);
            let mut out = json!({ "id": sealed.id.to_string(), "title": publication.meta.title });
            if let Some(fmt) = optional_str(&args, "format")
            {
                let format = match fmt.as_str()
                {
                    "md" | "markdown" => sos_publication::Format::Markdown,
                    "html" => sos_publication::Format::Html,
                    "json" => sos_publication::Format::Json,
                    other => return Err(format!("unknown format `{other}`")),
                };
                let artifact = sos_publication::render(&publication, format).map_err(|e| e.to_string())?;
                out["rendered"] = Value::String(artifact.content);
            }
            Ok(out)
        }),
    }
}

/// `sos_plugins` — list or find plugins by role/domain from inline descriptors.
#[must_use]
pub fn plugins_tool() -> McpTool {
    McpTool {
        name: "sos_plugins".to_owned(),
        description: "List, or find by role/domain, plugin descriptors.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "descriptors": { "type": "array", "description": "Array of PluginDescriptor." },
                "role": { "type": "string" },
                "domain": { "type": "string" },
            },
            "required": ["descriptors"],
        }),
        handler: Box::new(|args| {
            let descriptors: Vec<PluginDescriptor> =
                serde_json::from_value(args.get("descriptors").cloned().unwrap_or(Value::Null))
                    .map_err(|e| format!("invalid `descriptors`: {e}"))?;
            let mut registry = Registry::new();
            for d in descriptors
            {
                registry.register(d);
            }
            let matches = match optional_str(&args, "role")
            {
                Some(role) =>
                {
                    let role = parse_role(&role);
                    let domain = optional_str(&args, "domain").map(sos_registry::DomainTag);
                    registry.find(&role, domain.as_ref())
                },
                None => registry.all(),
            };
            serde_json::to_value(&matches).map_err(|e| e.to_string())
        }),
    }
}

fn parse_role(s: &str) -> Role {
    match s
    {
        "knowledge" => Role::Knowledge,
        "reasoning" => Role::Reasoning,
        "curiosity" => Role::Curiosity,
        "simulation" => Role::Simulation,
        "planning" => Role::Planning,
        "publication" => Role::Publication,
        "memory" => Role::Memory,
        "hypothesis-generator" => Role::HypothesisGenerator,
        "predictor" => Role::Predictor,
        "experiment-designer" => Role::ExperimentDesigner,
        "executor" => Role::Executor,
        "evidence-extractor" => Role::EvidenceExtractor,
        "statistical-evaluator" => Role::StatisticalEvaluator,
        "hypothesis-ranker" => Role::HypothesisRanker,
        "theory-reviser" => Role::TheoryReviser,
        other => Role::Custom(other.to_owned()),
    }
}

/// `sos_propose` — the untrusted-proposer entry point (Invariant IX).
///
/// The submitted [`sos_ccos::Proposal`] is sealed, stored, and returned as an
/// **untrusted** object id — nothing about calling this tool makes an agent's
/// suggestion true. It becomes part of the trusted graph only if a
/// deterministic engine later disposes of it
/// ([`sos_ccos::dispose`]/[`sos_ccos::Admission`]), which this tool
/// deliberately does not do on the caller's behalf.
#[must_use]
pub fn propose_tool() -> McpTool {
    McpTool {
        name: "sos_propose".to_owned(),
        description: "Submit an UNTRUSTED proposal (question/hypothesis/analogy/conjecture), grounded in real objects. Stored, attested, but not trusted until a deterministic engine disposes of it.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "store": { "type": "string" },
                "kind": { "type": "string", "enum": ["question", "hypothesis", "analogy", "conjecture"] },
                "statement": { "type": "string" },
                "concerns": { "type": "array", "items": { "type": "string" }, "description": "Object ids this proposal is grounded in (non-empty)." },
                "rationale": { "type": "string" },
            },
            "required": ["store", "kind", "statement", "concerns"],
        }),
        handler: Box::new(|args| {
            let store_path = required_str(&args, "store")?;
            let kind = match required_str(&args, "kind")?.as_str()
            {
                "question" => sos_ccos::ProposalKind::Question,
                "hypothesis" => sos_ccos::ProposalKind::Hypothesis,
                "analogy" => sos_ccos::ProposalKind::Analogy,
                "conjecture" => sos_ccos::ProposalKind::Conjecture,
                other => return Err(format!("unknown proposal kind `{other}`")),
            };
            let statement = required_str(&args, "statement")?;
            let concerns: Vec<ObjectId> = string_array(&args, "concerns")?
                .iter()
                .map(|s| parse_object_id(s).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?;
            let rationale = optional_str(&args, "rationale").unwrap_or_default();

            let proposal = sos_ccos::Proposal::new(kind, statement, concerns, rationale).map_err(|e| e.to_string())?;
            let sealed = sos_ccos::seal_proposal(proposal, Author::agent("mcp-client"));

            let root = sos_cli::store::resolve_root(Some(&store_path)).map_err(|e| e.to_string())?;
            let mut s = sos_cli::store::open(&root).map_err(|e| e.to_string())?;
            use sos_store::TypedStore;
            s.put_object(&sealed).map_err(|e| e.to_string())?;

            Ok(json!({
                "id": sealed.id.to_string(),
                "trusted": false,
                "note": "Untrusted: awaits deterministic disposition before it can be trusted.",
            }))
        }),
    }
}
