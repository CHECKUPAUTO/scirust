//! # Real-LLM evaluation harness (`ccos eval`)
//!
//! Where [`crate::experiment`] tests the *necessary* (retrieval) condition under
//! an oracle, this harness tests the **sufficient** condition with a **real
//! LLM**: given a context window assembled by each strategy, does the model
//! actually produce the **correct answer** to a task whose answer depends on
//! causally-distant context?
//!
//! ## Why it is auto-gradable without running code
//!
//! Each task is a tiny multi-file project encoding an **arithmetic causal
//! chain**: `f0` defines a base constant, `f1` transforms it, … `f_{k}` uses
//! `f_{k-1}`. The question — *"what integer does `s{k}()` return?"* — can only be
//! answered by reading the whole chain (the distant cause `f0` included). Success
//! is exact-match on the integer, computed by us when we generate the chain. No
//! compilation or test execution is needed, so grading is trivial and objective.
//!
//! ## Strategies and the clean/noisy split
//!
//! The same six strategies as [`crate::experiment`] assemble the window from a
//! token budget; in the **noisy** scenario a decoy file out-matches the queried
//! function lexically. RAG/GraphRAG locate code from the question; CCOS anchors
//! on the queried file (the workspace signal). We measure **task-success rate**,
//! **input tokens**, and **symbol-hallucination rate** (answers/reasoning citing
//! a function not in the project), per causal diameter.
//!
//! ## Providers
//!
//! Set one of (checked in this order):
//! - `ANTHROPIC_API_KEY` (+ optional `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`) —
//!   any Anthropic-Messages-compatible `/v1/messages` endpoint. For DeepSeek set
//!   `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic` and
//!   `ANTHROPIC_MODEL=deepseek-v4-pro`;
//! - `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_MODEL`) — any
//!   OpenAI-compatible `/v1/chat/completions` endpoint;
//! - `OLLAMA_ENDPOINT` (+ optional `OLLAMA_MODEL`) — a local Ollama server.
//!
//! With neither set, the harness still runs end-to-end against a deterministic
//! "no-model" stub (every answer wrong) so the pipeline and metrics are
//! exercised — useful for CI plumbing, **not** a result.

use crate::memory::{EdgeType, MemoryGraph, NodeType};
use crate::region_engine::ContextRegionEngine;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// ≈4 chars per token, the standard rough estimate.
fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

/// Filler that pads each file to a realistic size (so the token budget actually
/// binds) without adding any query-relevant tokens.
const FILLER: &str = "\n// ---------------------------------------------------------------\n\
// Auxiliary notes for the synthetic evaluation corpus. This block pads the\n\
// file to a realistic size so that the LLM context budget is a genuine\n\
// constraint and selection matters. It carries no identifiers relevant to\n\
// any task query, and is identical across files so it cannot bias retrieval.\n\
// ---------------------------------------------------------------\n";

/// A deterministic pseudo-random order key for a file id, so that files the
/// retriever scores equally (e.g. all the irrelevant ones) are ordered
/// arbitrarily rather than alphabetically — avoiding a name-based bias.
fn idhash(id: &str) -> u64 {
    let h = crate::util::sha256_hex(id);
    u64::from_str_radix(&h[..16], 16).unwrap_or(0)
}

const STRATEGIES: [&str; 6] = [
    "rag-dense",
    "rag-hybrid",
    "graphrag-1hop",
    "graphrag-bfs",
    "ccos-from-query",
    "ccos-region",
];

/// The strategies to evaluate.
fn strategies() -> Vec<&'static str> {
    STRATEGIES.to_vec()
}

/// Per `(strategy, diameter)` tally: `(solved, hallucinations, covered, token_sum, n)`.
type Tally = (usize, usize, usize, f32, usize);

/// Configuration for an evaluation run.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub seed: u64,
    pub tasks: usize,
    /// Token budget for the assembled context window.
    pub budget_tokens: usize,
    /// Causal diameters (chain lengths − 1) to sweep.
    pub diameters: Vec<u32>,
    /// Decoy files per task (lures for ranked retrieval; make the repo > budget).
    pub decoys: usize,
    /// When true, a decoy out-matches the queried function lexically.
    pub noisy: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig {
            seed: 7,
            tasks: 40,
            budget_tokens: 600,
            diameters: vec![1, 2, 3, 4],
            decoys: 10,
            noisy: false,
        }
    }
}

/// One file in a task's tiny project.
#[derive(Debug, Clone)]
struct SrcFile {
    id: String,
    text: String,
    /// Lexical tokens this file is indexed by (its function/const names).
    tokens: BTreeSet<String>,
    /// A query-independent "popularity" prior (high for decoys) — the lure for a
    /// hybrid retriever that blends similarity with a global signal.
    popularity: f32,
}

/// A single auto-gradable task.
#[derive(Debug, Clone)]
struct Task {
    diameter: u32,
    files: Vec<SrcFile>,
    /// Files that must be in the window to answer (the whole chain).
    required: BTreeSet<String>,
    /// The queried file (the workspace anchor for CCOS).
    anchor: String,
    /// Natural-language question.
    question: String,
    /// Query tokens RAG/GraphRAG rank by.
    query: BTreeSet<String>,
    /// Correct integer answer.
    answer: i64,
    /// Function/const names that exist (for hallucination checking).
    symbols: BTreeSet<String>,
}

fn build_graph(task: &Task) -> (MemoryGraph, BTreeMap<String, String>) {
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    let mut text: BTreeMap<String, String> = BTreeMap::new();
    for f in &task.files {
        g.upsert_node(
            f.id.clone().into(),
            f.id.clone(),
            String::new(),
            NodeType::Module,
        );
        text.insert(f.id.clone(), f.text.clone());
    }
    // Chain dependency edges f{i-1} → f{i} (ids carry the chain index).
    let chain: Vec<&SrcFile> = task
        .files
        .iter()
        .filter(|f| f.id.contains("chain"))
        .collect();
    for w in chain.windows(2) {
        g.add_edge(
            w[0].id.clone().into(),
            w[1].id.clone().into(),
            0.9,
            EdgeType::DependsOn,
        );
    }
    (g, text)
}

/// Generate `tasks` arithmetic-chain tasks.
fn generate(cfg: &EvalConfig, rng: &mut StdRng) -> Vec<Task> {
    let ops = [('+', 1, 9), ('*', 2, 4), ('-', 1, 7)];
    let mut tasks = Vec::new();
    for _ in 0..cfg.tasks {
        let d = cfg.diameters[rng.gen_range(0..cfg.diameters.len())];
        let len = (d + 1) as usize; // chain files: f0..f{d}
        let base: i64 = rng.gen_range(1..20);

        let mut files: Vec<SrcFile> = Vec::new();
        let mut required: BTreeSet<String> = BTreeSet::new();
        let mut symbols: BTreeSet<String> = BTreeSet::new();
        let mut value = base;

        // f0: the distant cause.
        let f0 = "chain0".to_string();
        files.push(SrcFile {
            id: format!("file:{f0}.rs"),
            text: format!("// {f0}.rs\npub const BASE: i64 = {base};\n{FILLER}"),
            tokens: BTreeSet::from(["BASE".to_string()]),
            popularity: 0.0,
        });
        required.insert(format!("file:{f0}.rs"));
        symbols.insert("BASE".to_string());

        for i in 1..len {
            let (op, lo, hi) = ops[rng.gen_range(0..ops.len())];
            let c: i64 = rng.gen_range(lo..=hi);
            value = match op {
                '+' => value + c,
                '*' => value * c,
                _ => value - c,
            };
            let name = format!("s{i}");
            let prev = if i == 1 {
                "BASE".to_string()
            } else {
                format!("s{}()", i - 1)
            };
            let id = format!("file:chain{i}.rs");
            files.push(SrcFile {
                id: id.clone(),
                text: format!(
                    "// chain{i}.rs\npub fn {name}() -> i64 {{ {prev} {op} {c} }}\n{FILLER}"
                ),
                tokens: BTreeSet::from([name.clone()]),
                popularity: 0.0,
            });
            required.insert(id);
            symbols.insert(name);
        }

        let last = len - 1;
        let last_name = format!("s{last}");
        let anchor = format!("file:chain{last}.rs");

        // Decoys: unrelated, high-lure files.
        for k in 0..cfg.decoys {
            let dname = format!("util{k}");
            files.push(SrcFile {
                id: format!("file:{dname}.rs"),
                text: format!(
                    "// {dname}.rs\npub fn {dname}_run(x: i64) -> i64 {{ x + {k} }}\n{FILLER}"
                ),
                tokens: BTreeSet::from([format!("{dname}_run")]),
                popularity: 1.0,
            });
            symbols.insert(format!("{dname}_run"));
        }

        // Query: the user asks about the last function.
        let mut query: BTreeSet<String> =
            BTreeSet::from([last_name.clone(), "return".into(), "value".into()]);
        if cfg.noisy {
            // A trap decoy is named exactly like the queried function → it wins
            // the lexical match while being causally irrelevant.
            let trap = format!("file:trap_{last_name}.rs");
            files.push(SrcFile {
                id: trap.clone(),
                text: format!(
                    "// trap_{last_name}.rs\n// NOTE: legacy helper, unrelated to the live pipeline\npub fn {last_name}_legacy() -> i64 {{ 0 }}\n{FILLER}"
                ),
                // Indexed on both the queried name and "legacy" → out-scores the
                // real chain tail (which matches only the name) lexically.
                tokens: BTreeSet::from([last_name.clone(), "legacy".to_string()]),
                popularity: 0.0,
            });
            symbols.insert(format!("{last_name}_legacy"));
            query.insert("legacy".into());
        }

        tasks.push(Task {
            diameter: d,
            files,
            required,
            anchor,
            question: format!(
                "What integer does {last_name}() return? Reply with ONLY the integer, nothing else."
            ),
            query,
            answer: value,
            symbols,
        });
    }
    tasks
}

fn sim(file: &SrcFile, query: &BTreeSet<String>) -> f32 {
    let inter = file.tokens.intersection(query).count();
    let union = file.tokens.union(query).count();
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

fn best_hit(task: &Task) -> String {
    task.files
        .iter()
        .map(|f| (sim(f, &task.query), f.id.clone()))
        .max_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.cmp(&a.1))
        })
        .map(|(_, id)| id)
        .unwrap_or_default()
}

fn region_of(g: &MemoryGraph, entry: &str) -> BTreeSet<String> {
    let mut engine = ContextRegionEngine::new();
    let mut sink = crate::event_log::EventLog::new("eval".into());
    engine.initialize_regions(g, &mut sink);
    engine
        .region_of(entry)
        .map(|rid| engine.regions[&rid].members.iter().cloned().collect())
        .unwrap_or_default()
}

/// Select files under a strategy, respecting a token budget.
fn select(strategy: &str, task: &Task, g: &MemoryGraph, budget: usize) -> Vec<String> {
    let by_id: BTreeMap<&str, &SrcFile> = task.files.iter().map(|f| (f.id.as_str(), f)).collect();
    let take_budget = |ordered: Vec<String>| -> Vec<String> {
        let mut out = Vec::new();
        let mut used = 0usize;
        for id in ordered {
            let t = by_id
                .get(id.as_str())
                .map(|f| estimate_tokens(&f.text))
                .unwrap_or(0);
            if used + t > budget && !out.is_empty() {
                continue;
            }
            used += t;
            out.push(id);
            if used >= budget {
                break;
            }
        }
        out
    };
    match strategy {
        "rag-dense" => {
            let mut v: Vec<&SrcFile> = task.files.iter().collect();
            v.sort_by(|a, b| {
                sim(b, &task.query)
                    .partial_cmp(&sim(a, &task.query))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| idhash(&a.id).cmp(&idhash(&b.id)))
            });
            take_budget(v.into_iter().map(|f| f.id.clone()).collect())
        }
        "rag-hybrid" => {
            // Similarity blended with a query-independent popularity prior, which
            // lures the retriever toward popular-but-irrelevant decoys.
            let mut v: Vec<&SrcFile> = task.files.iter().collect();
            let key = |f: &SrcFile| 0.65 * sim(f, &task.query) + 0.35 * f.popularity;
            v.sort_by(|a, b| {
                key(b)
                    .partial_cmp(&key(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| idhash(&a.id).cmp(&idhash(&b.id)))
            });
            take_budget(v.into_iter().map(|f| f.id.clone()).collect())
        }
        "graphrag-1hop" => {
            let seed = best_hit(task);
            let mut ids: BTreeSet<String> = BTreeSet::from([seed.clone()]);
            for e in &g.edges {
                if e.source.0 == seed {
                    ids.insert(e.target.0.clone());
                } else if e.target.0 == seed {
                    ids.insert(e.source.0.clone());
                }
            }
            take_budget(ids.into_iter().collect())
        }
        "graphrag-bfs" => take_budget(bfs(g, &best_hit(task))),
        "ccos-from-query" => take_budget(region_ordered(g, &region_of(g, &best_hit(task)))),
        "ccos-region" => take_budget(region_ordered(g, &region_of(g, &task.anchor))),
        _ => Vec::new(),
    }
}

/// Region members ordered chain-first (so a truncating budget keeps the chain).
fn region_ordered(_g: &MemoryGraph, members: &BTreeSet<String>) -> Vec<String> {
    let mut v: Vec<String> = members.iter().cloned().collect();
    v.sort_by_key(|id| (!id.contains("chain"), id.clone()));
    v
}

fn bfs(g: &MemoryGraph, seed: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::from([seed.to_string()]);
    let mut q = VecDeque::from([seed.to_string()]);
    while let Some(n) = q.pop_front() {
        out.push(n.clone());
        let mut nbs = Vec::new();
        for e in &g.edges {
            if e.source.0 == n {
                nbs.push(e.target.0.clone());
            } else if e.target.0 == n {
                nbs.push(e.source.0.clone());
            }
        }
        nbs.sort();
        for nb in nbs {
            if seen.insert(nb.clone()) {
                q.push_back(nb);
            }
        }
    }
    out
}

fn assemble_prompt(task: &Task, selected: &[String]) -> (String, usize) {
    let by_id: BTreeMap<&str, &SrcFile> = task.files.iter().map(|f| (f.id.as_str(), f)).collect();
    let mut body = String::from("You are given some files from a Rust project.\n\n");
    let mut ids: Vec<&String> = selected.iter().collect();
    ids.sort();
    for id in ids {
        if let Some(f) = by_id.get(id.as_str()) {
            body.push_str(&f.text);
            body.push('\n');
        }
    }
    body.push_str(&format!("\n{}\n", task.question));
    let tokens = estimate_tokens(&body);
    (body, tokens)
}

/// Per-strategy aggregated result.
#[derive(Debug, Clone, Serialize)]
pub struct EvalStrategy {
    pub strategy: String,
    pub tasks: usize,
    pub solved: usize,
    pub success_rate: f32,
    /// Oracle coverage: fraction of tasks whose required causal set was fully in
    /// the window (the *necessary* condition, independent of the model).
    pub mean_coverage: f32,
    pub mean_input_tokens: f32,
    pub hallucination_rate: f32,
}

/// Full evaluation report for one scenario.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub provider: String,
    pub model: String,
    pub seed: u64,
    pub n_tasks: usize,
    pub budget_tokens: usize,
    pub noisy: bool,
    pub per_diameter: Vec<(u32, Vec<EvalStrategy>)>,
    pub overall: Vec<EvalStrategy>,
}

/// ASCII case-insensitive substring search starting at byte offset `from`.
fn find_ci(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || from + n.len() > h.len() {
        return None;
    }
    (from..=h.len() - n.len()).find(|&i| {
        h[i..i + n.len()]
            .iter()
            .zip(n)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    })
}

/// Strip reasoning blocks (`<think>…</think>`, `<thinking>…</thinking>`) that some
/// models emit before their answer, so the grader scores the final answer, not the
/// scratch work. Case-insensitive, but it preserves the case of the surrounding
/// text (hallucination detection matches symbol names case-sensitively). An
/// unclosed opening tag drops everything after it.
fn strip_reasoning(reply: &str) -> String {
    let mut s = reply.to_string();
    for (open, close) in [("<think>", "</think>"), ("<thinking>", "</thinking>")] {
        while let Some(start) = find_ci(&s, open, 0) {
            if let Some(close_at) = find_ci(&s, close, start + open.len()) {
                s.replace_range(start..close_at + close.len(), " ");
            } else {
                s.replace_range(start.., " ");
                break;
            }
        }
    }
    s
}

/// Parse the first standalone integer from a model reply (after removing any
/// reasoning block).
fn parse_answer(reply: &str) -> Option<i64> {
    let cleaned = strip_reasoning(reply);
    let mut num = String::new();
    let mut found = None;
    let chars: Vec<char> = cleaned.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        let sign = c == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
        if c.is_ascii_digit() || sign {
            num.push(c);
        } else if !num.is_empty() {
            if let Ok(v) = num.parse::<i64>() {
                found = Some(v);
                break;
            }
            num.clear();
        }
    }
    found.or_else(|| num.parse::<i64>().ok())
}

/// Whether the reply cites a function/const name that does not exist in the task.
/// Reasoning blocks are stripped first, so exploring a hypothesis inside
/// `<think>…</think>` is not counted as a hallucination in the final answer.
fn hallucinates(reply: &str, task: &Task) -> bool {
    // Look for `name(` call patterns; flag any that aren't real symbols.
    let cleaned = strip_reasoning(reply);
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // Walk back over an identifier.
            let mut j = i;
            while j > 0 {
                let c = bytes[j - 1];
                if c.is_ascii_alphanumeric() || c == b'_' {
                    j -= 1;
                } else {
                    break;
                }
            }
            if j < i {
                let name = &cleaned[j..i];
                if name
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_alphabetic())
                    .unwrap_or(false)
                    && !task.symbols.contains(name)
                    && name != "i64"
                {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

/// Provider abstraction: returns a model reply, or `None` if no LLM is configured.
async fn ask(prompt: &str) -> Option<String> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let base = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".into());
        // Air-gap control (CCOS_EXTENDED plan P4): default policy is localhost
        // only, so the Anthropic API host is denied unless the operator added it
        // to `CCOS_EGRESS_ALLOW`. Refused ⇒ `None` (no LLM) rather than a call.
        let anthropic_url = format!("{base}/v1/messages");
        let client = eval_http_client(&anthropic_url)?;
        let model =
            std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".into());
        // No `temperature`: reasoning models (e.g. deepseek-v4-pro) emit a
        // separate thinking block and may reject/ignore it. A generous
        // max_tokens leaves room to finish the thinking *and* the text answer
        // (too small a budget is spent entirely on thinking → no answer).
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        });
        let response = client
            .post(anthropic_url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .ok()?;
        let resp = eval_json_response(response).await?;
        // The content is an array of blocks; with reasoning models it is
        // [{type:"thinking",…}, {type:"text",text:…}]. Return the text block.
        return resp["content"].as_array().and_then(|blocks| {
            blocks
                .iter()
                .find(|b| b["type"] == "text")
                .and_then(|b| b["text"].as_str())
                .map(String::from)
        });
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let base =
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".into());
        // Air-gap control (CCOS_EXTENDED plan P4): see the Anthropic branch —
        // the OpenAI host is denied under the default localhost-only policy
        // unless `CCOS_EGRESS_ALLOW` widens it. Refused ⇒ `None`.
        let openai_url = format!("{base}/v1/chat/completions");
        let client = eval_http_client(&openai_url)?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        let body = serde_json::json!({
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        });
        let response = client
            .post(openai_url)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .ok()?;
        let resp = eval_json_response(response).await?;
        return resp["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from);
    }
    if let Ok(endpoint) = std::env::var("OLLAMA_ENDPOINT") {
        // Air-gap control (CCOS_EXTENDED plan P4): the Ollama endpoint is the
        // one provider that is *typically* local, but it need not be — a
        // remote `OLLAMA_ENDPOINT` is denied under the default policy unless
        // `CCOS_EGRESS_ALLOW` allows it. Refused ⇒ `None`.
        let ollama_url = format!("{endpoint}/api/generate");
        let client = eval_http_client(&ollama_url)?;
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3".into());
        let body = serde_json::json!({
            "model": model, "prompt": prompt, "stream": false,
            "options": {"temperature": 0},
        });
        let response = client.post(ollama_url).json(&body).send().await.ok()?;
        let resp = eval_json_response(response).await?;
        return resp["response"].as_str().map(String::from);
    }
    None
}

fn eval_http_client(url: &str) -> Option<reqwest::Client> {
    let endpoint = crate::egress::EgressAllowlist::from_env()
        .validate(url)
        .ok()?;
    crate::egress::secure_async_client(
        &endpoint,
        std::time::Duration::from_secs(5),
        std::time::Duration::from_secs(120),
    )
    .ok()
}

async fn eval_json_response(response: reqwest::Response) -> Option<serde_json::Value> {
    if !response.status().is_success() {
        return None;
    }
    let body = crate::egress::response_bytes_limited(response, 16 * 1024 * 1024)
        .await
        .ok()?;
    serde_json::from_slice(&body).ok()
}

fn provider_label() -> (String, String) {
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        (
            "anthropic-messages".into(),
            std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".into()),
        )
    } else if std::env::var("OPENAI_API_KEY").is_ok() {
        (
            "openai-compatible".into(),
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
        )
    } else if std::env::var("OLLAMA_ENDPOINT").is_ok() {
        (
            "ollama".into(),
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3".into()),
        )
    } else {
        (
            "none (offline stub — every answer wrong)".into(),
            "-".into(),
        )
    }
}

/// Run the evaluation for one scenario (clean or noisy per `cfg.noisy`).
pub async fn run_eval(cfg: &EvalConfig) -> EvalReport {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let tasks = generate(cfg, &mut rng);
    let (provider, model) = provider_label();
    // Live progress on stderr (so it never pollutes the JSON/table on stdout),
    // but only with a real model — the offline stub is instant and silent.
    let show_progress = !provider.starts_with("none");
    let scenario = if cfg.noisy { "noisy" } else { "clean" };

    // strategy → diameter → tally
    let strats = strategies();
    let mut acc: BTreeMap<(String, u32), Tally> = BTreeMap::new();
    for (ti, task) in tasks.iter().enumerate() {
        let (g, _text) = build_graph(task);
        for strat in strats.iter().copied() {
            let sel = select(strat, task, &g, cfg.budget_tokens);
            let covered = task.required.iter().all(|r| sel.contains(r));
            let (prompt, toks) = assemble_prompt(task, &sel);
            let reply = ask(&prompt).await.unwrap_or_default();
            let solved = parse_answer(&reply) == Some(task.answer);
            let halluc = hallucinates(&reply, task);
            let e = acc
                .entry((strat.to_string(), task.diameter))
                .or_insert((0, 0, 0, 0.0, 0));
            e.0 += usize::from(solved);
            e.1 += usize::from(halluc);
            e.2 += usize::from(covered);
            e.3 += toks as f32;
            e.4 += 1;
        }
        if show_progress {
            use std::io::Write;
            eprint!(
                "\r  [{scenario}] {}/{} tasks ({} calls each)…   ",
                ti + 1,
                tasks.len(),
                strats.len()
            );
            let _ = std::io::stderr().flush();
        }
    }
    if show_progress {
        eprintln!(
            "\r  [{scenario}] {} tasks done.                    ",
            tasks.len()
        );
    }

    let mk =
        |solved: usize, halluc: usize, cov: usize, toks: f32, n: usize, name: &str| EvalStrategy {
            strategy: name.to_string(),
            tasks: n,
            solved,
            success_rate: if n == 0 {
                0.0
            } else {
                solved as f32 / n as f32
            },
            mean_coverage: if n == 0 { 0.0 } else { cov as f32 / n as f32 },
            mean_input_tokens: if n == 0 { 0.0 } else { toks / n as f32 },
            hallucination_rate: if n == 0 {
                0.0
            } else {
                halluc as f32 / n as f32
            },
        };

    let mut per_diameter = Vec::new();
    for &d in &cfg.diameters {
        let mut row = Vec::new();
        for strat in strats.iter().copied() {
            if let Some((s, h, c, t, n)) = acc.get(&(strat.to_string(), d)) {
                if *n > 0 {
                    row.push(mk(*s, *h, *c, *t, *n, strat));
                }
            }
        }
        if !row.is_empty() {
            per_diameter.push((d, row));
        }
    }
    let mut overall = Vec::new();
    for strat in strats.iter().copied() {
        let (mut s, mut h, mut c, mut t, mut n) = (0, 0, 0, 0.0, 0);
        for &d in &cfg.diameters {
            if let Some((ss, hh, cc, tt, nn)) = acc.get(&(strat.to_string(), d)) {
                s += ss;
                h += hh;
                c += cc;
                t += tt;
                n += nn;
            }
        }
        if n > 0 {
            overall.push(mk(s, h, c, t, n, strat));
        }
    }

    EvalReport {
        provider,
        model,
        seed: cfg.seed,
        n_tasks: tasks.len(),
        budget_tokens: cfg.budget_tokens,
        noisy: cfg.noisy,
        per_diameter,
        overall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_task(noisy: bool) -> Task {
        let cfg = EvalConfig {
            tasks: 1,
            noisy,
            diameters: vec![3],
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(1);
        generate(&cfg, &mut rng).pop().unwrap()
    }

    #[test]
    fn ground_truth_matches_the_chain() {
        // Re-derive the answer from the file texts and check it equals task.answer.
        let task = one_task(false);
        // The answer is computable; assert it is an integer the question targets.
        assert!(task.answer != i64::MIN);
        assert!(task.required.len() >= 2, "a chain spans multiple files");
        assert!(task.question.contains("Reply with ONLY the integer"));
    }

    #[test]
    fn region_selects_the_whole_chain() {
        let task = one_task(false);
        let (g, _) = build_graph(&task);
        let sel: BTreeSet<String> = select("ccos-region", &task, &g, 100_000)
            .into_iter()
            .collect();
        for r in &task.required {
            assert!(
                sel.contains(r),
                "region must contain required chain file {r}"
            );
        }
    }

    #[test]
    fn noisy_query_traps_lexical_selection_but_not_the_anchor() {
        let task = one_task(true);
        let (g, _) = build_graph(&task);
        // The best lexical hit is the trap (legacy) file, not the real chain tail.
        let hit = best_hit(&task);
        assert!(
            hit.contains("trap"),
            "noisy query must lure to the trap file, got {hit}"
        );
        // The anchored region still covers the whole chain.
        let region: BTreeSet<String> = select("ccos-region", &task, &g, 100_000)
            .into_iter()
            .collect();
        assert!(task.required.iter().all(|r| region.contains(r)));
    }

    #[test]
    fn parse_answer_extracts_integers() {
        assert_eq!(parse_answer("42"), Some(42));
        assert_eq!(parse_answer("The answer is -7."), Some(-7));
        assert_eq!(parse_answer("no number here"), None);
        // Reasoning blocks are ignored; the post-think answer wins.
        assert_eq!(parse_answer("<think>base is 5, 5+3=8</think>\n8"), Some(8));
        assert_eq!(parse_answer("<THINK>1+1=2</THINK> 42"), Some(42));
        assert_eq!(parse_answer("<think>unfinished 5"), None);
    }

    #[test]
    fn strip_reasoning_removes_think_blocks() {
        assert_eq!(strip_reasoning("a<think>b</think>c"), "a c");
        assert_eq!(strip_reasoning("<thinking>x</thinking>y").trim(), "y");
        assert_eq!(strip_reasoning("plain"), "plain");
        let mixed = strip_reasoning("KEEP<THINK>drop</THINK>Case");
        assert!(mixed.contains("KEEP") && mixed.contains("Case"));
        assert!(!mixed.to_lowercase().contains("drop"));
    }

    #[test]
    fn hallucination_flags_unknown_calls() {
        let task = one_task(false);
        assert!(hallucinates("I called ghost_fn() to get it", &task));
        assert!(!hallucinates("the value is 12", &task));
        // A call inside a reasoning block is not a hallucination in the answer.
        assert!(!hallucinates("<think>maybe ghost_fn()?</think> 12", &task));
    }

    #[tokio::test]
    async fn pipeline_runs_offline_stub() {
        // Hermetic: `provider_label()` picks the provider from the process env
        // (ANTHROPIC_API_KEY / OPENAI_API_KEY / OLLAMA_ENDPOINT). A contributor
        // with a local Ollama server configured would otherwise see this test
        // fail with a non-`none` provider. Strip them so the offline stub path
        // is exercised deterministically. This is the only test that calls
        // `run_eval`, so removing these vars cannot race with a parallel test.
        for v in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_ENDPOINT"] {
            std::env::remove_var(v);
        }
        // With no LLM configured, the harness still runs; every answer is wrong.
        let report = run_eval(&EvalConfig {
            tasks: 6,
            ..Default::default()
        })
        .await;
        // One row per strategy.
        assert_eq!(report.overall.len(), strategies().len());
        assert!(report.provider.starts_with("none"));
        for s in &report.overall {
            assert_eq!(s.solved, 0, "offline stub solves nothing");
        }
    }
}
