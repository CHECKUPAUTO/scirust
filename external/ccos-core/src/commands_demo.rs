//! The `ccos demo` command — an end-to-end walkthrough of every kernel
//! subsystem on a small synthetic workspace. Extracted from `main` to keep the
//! CLI entry point small and focused (one responsibility per module).

use ccos_core::consensus::ConsensusEngine;
use ccos_core::distributed_event_log::DistributedEventLog;
use ccos_core::event_log::{EventLog, EventPayload, EventType};
use ccos_core::guard::{GuardConfig, GuardLayer};
use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::llm::{LlmClient, LlmConfig};
use ccos_core::memory::{MemoryGraph, NodeId};
use ccos_core::parser::ASTParser;
use ccos_core::util::sha256_hex as compute_file_hash;
use std::collections::HashMap;
use uuid::Uuid;

pub(crate) async fn run_demo() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS — Causal Context Operating System     ║");
    println!(
        "║  Kernel v{} | Rust 2021                   ║",
        env!("CARGO_PKG_VERSION")
    );
    println!("╚══════════════════════════════════════════════╝\n");

    // ── Initialization ─────────────────────────────────────────────
    let session_id = Uuid::new_v4().to_string();
    println!("[INIT] Session ID: {}", session_id);

    let mut event_log = EventLog::new(session_id.clone());
    let mut memory_graph = MemoryGraph::new(MemoryGraph::paging_threshold_from_env(0.2), 80);
    let mut incremental_engine = IncrementalGraphEngine::new();
    let _guard = GuardLayer::new(GuardConfig::from_env());

    let llm_config = LlmConfig {
        endpoint: std::env::var("OLLAMA_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:11434".into()),
        model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "codellama".into()),
        ..Default::default()
    };
    let llm_client = LlmClient::new(llm_config.clone());
    let mut dist_log = DistributedEventLog::new();
    dist_log.append("kernel_initialized".into(), "kernel".into());

    // ── Event Sourcing: Log initialization ────────────────────────
    event_log.append(
        EventType::CycleStart,
        EventPayload::CycleEvent {
            cycle_number: 0,
            action: "kernel_initialized".into(),
        },
    );

    // ── Workspace Simulation ──────────────────────────────────────
    println!("\n─── PHASE 1: Workspace Ingestion & AST Parsing ───\n");

    let mut workspace: HashMap<String, String> = HashMap::new();
    workspace.insert(
        "src/lib.rs".into(),
        r#"mod auth;
mod database;
mod api;

use std::collections::HashMap;
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};

pub struct AppState {
    pub db: Database,
    pub auth: AuthService,
    pub config: AppConfig,
}

pub struct AppConfig {
    pub port: u16,
    pub host: String,
}

pub fn init_app(config: AppConfig) -> AppState {
    let db = Database::connect(&config);
    let auth = AuthService::new();
    AppState { db, auth, config }
}"#
        .into(),
    );

    workspace.insert(
        "src/auth.rs".into(),
        r#"use sha2::{Sha256, Digest};
use std::collections::HashMap;

pub struct AuthService {
    sessions: HashMap<String, Session>,
}

pub struct Session {
    pub user_id: String,
    pub token: String,
    pub expires_at: u64,
}

impl AuthService {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    pub fn authenticate(&self, token: &str) -> Option<&Session> {
        self.sessions.get(token)
    }

    pub fn create_session(&mut self, user_id: &str) -> Session {
        let token = Self::generate_token(user_id);
        let session = Session {
            user_id: user_id.to_string(),
            token: token.clone(),
            expires_at: 0,
        };
        self.sessions.insert(token.clone(), session.clone());
        session
    }

    fn generate_token(user_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(user_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}"#
        .into(),
    );

    workspace.insert(
        "src/database.rs".into(),
        r#"use std::sync::Mutex;
use std::collections::HashMap;

pub struct Database {
    store: Mutex<HashMap<String, Vec<u8>>>,
    connected: bool,
}

impl Database {
    pub fn connect(config: &crate::AppConfig) -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            connected: true,
        }
    }

    pub fn insert(&self, key: &str, value: Vec<u8>) {
        if let Ok(mut store) = self.store.lock() {
            store.insert(key.to_string(), value);
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.lock().ok()?.get(key).cloned()
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }
}"#
        .into(),
    );

    workspace.insert(
        "src/api.rs".into(),
        r#"use crate::AppState;
use serde_json::Value;

pub async fn handle_request(state: &AppState, path: &str, body: &str) -> Result<Value, String> {
    match path {
        "/health" => Ok(serde_json::json!({"status": "ok"})),
        "/auth/login" => {
            let session = state.auth.create_session("user_1");
            Ok(serde_json::json!({"token": session.token}))
        }
        "/data" => {
            let data = state.db.get("default");
            Ok(serde_json::json!({"data": data}))
        }
        _ => Err("not found".into()),
    }
}"#
        .into(),
    );

    // ── Phase 1: Initial parsing ──────────────────────────────────
    for (file_path, source_code) in &workspace {
        event_log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: file_path.clone(),
                file_hash: compute_file_hash(source_code),
                modules_found: 0,
                uses_found: 0,
                symbols_found: 0,
            },
        );

        let parse_result = incremental_engine.register_file(file_path, source_code);
        let parser = ASTParser::new();
        parser.update_memory_graph(&parse_result, source_code, &mut memory_graph);

        let _event_id = event_log.append(
            EventType::GraphMutation,
            EventPayload::GraphMutation {
                node_id: format!("file:{}", file_path),
                operation: "ingest".into(),
                nodes_before: memory_graph
                    .node_count()
                    .saturating_sub(parse_result.generated_nodes),
                nodes_after: memory_graph.node_count(),
                edges_before: memory_graph
                    .edge_count()
                    .saturating_sub(parse_result.generated_edges),
                edges_after: memory_graph.edge_count(),
            },
        );

        println!(
            "  [PARSE] {} → {} modules, {} uses, {} symbols (nodes: {}, edges: {})",
            file_path,
            parse_result.modules.len(),
            parse_result.use_statements.len(),
            parse_result.symbols.len(),
            memory_graph.node_count(),
            memory_graph.edge_count(),
        );
    }

    // ── Phase 2: LLM Interaction ──────────────────────────────────
    println!("\n─── PHASE 2: LLM Integration & Guard Layer ───\n");

    let prompt = format!(
        r#"Analyze the following Rust codebase graph:
Nodes: {}
Edges: {}
Top modules: Analyze the dependency structure and identify potential issues."#,
        memory_graph.node_count(),
        memory_graph.edge_count()
    );

    event_log.append(
        EventType::LlmCall,
        EventPayload::LlmCallRequest {
            model: llm_config.model.clone(),
            prompt_hash: compute_file_hash(&prompt),
            input_tokens: prompt.len(),
        },
    );

    let validated = llm_client
        .query(
            &prompt,
            Some("You are a code analysis assistant. Respond with valid JSON only."),
        )
        .await;

    let _guard_event = event_log.append(
        EventType::GuardCheck,
        EventPayload::GuardCheck {
            input_hash: validated.response_hash.clone(),
            passed: validated.guard_passed,
            score: validated.reliability_score,
            warnings: validated.guard_warnings.clone(),
        },
    );

    event_log.append(
        EventType::LlmResponse,
        EventPayload::LlmCallResponse {
            model: validated.model.clone(),
            response_hash: validated.response_hash.clone(),
            output_tokens: validated.sanitized_output.len(),
            latency_ms: validated.latency_ms,
            guard_passed: validated.guard_passed,
            reliability_score: validated.reliability_score,
        },
    );

    let guard_status = if validated.guard_passed {
        "PASSED"
    } else {
        "BLOCKED"
    };
    println!(
        "  [LLM] Query → {} | Guard: {} (score: {:.2}) | Latency: {}ms | Fallback: {}",
        validated.model,
        guard_status,
        validated.reliability_score,
        validated.latency_ms,
        validated.is_fallback
    );
    println!("  [LLM] Output: {:.120}...", validated.sanitized_output);
    dist_log.append(
        format!("llm_guard:{}", validated.guard_passed),
        "llm".into(),
    );

    // Multi-model consensus (illustrative — offline models all fall back, so
    // they agree on the deterministic fallback output).
    let consensus_models = ["codellama", "mistral"];
    let votes = llm_client
        .query_models(
            &prompt,
            Some("Respond with valid JSON only."),
            &consensus_models,
        )
        .await;
    let consensus = ConsensusEngine::with_threshold(0.5).resolve_weighted(&votes);
    println!(
        "  [CONSENSUS] {} models, agreement {:.2} → reached: {} (models: {:?})",
        consensus.total_votes,
        consensus.agreement_ratio,
        consensus.consensus_reached,
        consensus.models_in_agreement,
    );
    dist_log.append(
        format!("consensus_reached:{}", consensus.consensus_reached),
        "consensus".into(),
    );

    // ── Phase 3: Incremental Update Simulation ────────────────────
    println!("\n─── PHASE 3: Incremental Graph Update (O(Δ)) ───\n");
    memory_graph.tick(); // advance the recency clock between cycles

    // Simulate a code change: modify api.rs
    let modified_api = r#"use crate::AppState;
use serde_json::Value;

pub async fn handle_request(state: &AppState, path: &str, body: &str) -> Result<Value, String> {
    match path {
        "/health" => Ok(serde_json::json!({"status": "ok", "version": "1.0"})),
        "/auth/login" => {
            let session = state.auth.create_session("user_1");
            Ok(serde_json::json!({"token": session.token, "expires": 3600}))
        }
        "/data" => {
            let data = state.db.get("default");
            Ok(serde_json::json!({"data": data}))
        }
        "/metrics" => {
            Ok(serde_json::json!({"cpu": 0.42, "mem": 0.73}))
        }
        _ => Err("not found".into()),
    }
}"#;

    let old_source = workspace.get("src/api.rs").map(|s| s.as_str());
    let delta =
        incremental_engine.process_delta("src/api.rs", old_source, modified_api, &mut memory_graph);

    workspace.insert("src/api.rs".into(), modified_api.into());

    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "file:src/api.rs".into(),
            operation: format!("{:?}", delta.operation).to_lowercase(),
            nodes_before: memory_graph.node_count().saturating_sub(delta.nodes_added),
            nodes_after: memory_graph.node_count(),
            edges_before: memory_graph.edge_count().saturating_sub(delta.edges_added),
            edges_after: memory_graph.edge_count(),
        },
    );

    println!(
        "  [DELTA] api.rs → {:?} | Δnodes: +{}/-{} | Δedges: +{}/-{}",
        delta.operation,
        delta.nodes_added,
        delta.nodes_removed,
        delta.edges_added,
        delta.edges_removed,
    );

    // ── Phase 4: Failure Propagation ──────────────────────────────
    println!("\n─── PHASE 4: Failure Propagation ───\n");

    let failure_node = NodeId::from("file:src/database.rs");
    memory_graph.set_failure_relevance(&failure_node, 0.95);

    event_log.append(
        EventType::FailureDetection,
        EventPayload::FailureDetection {
            node_id: failure_node.to_string(),
            failure_type: "connection_lost".into(),
            severity: 0.95,
        },
    );

    let affected: Vec<String> = memory_graph
        .edges()
        .iter()
        .filter(|e| e.source == failure_node)
        .map(|e| e.target.to_string())
        .collect();

    memory_graph.propagate_failure(&failure_node, 0, 3);

    event_log.append(
        EventType::FailurePropagation,
        EventPayload::FailurePropagation {
            origin_node_id: failure_node.to_string(),
            affected_nodes: affected.clone(),
            depth: 3,
        },
    );

    println!(
        "  [FAILURE] Origin: {} | Severity: 0.95 | Affected: {:?}",
        failure_node, affected
    );

    // Display updated scores
    let scores = memory_graph.get_node_scores();
    println!("\n  Memory Graph Scores (top 8):");
    for (id, score) in scores.iter().take(8) {
        let node = memory_graph.node(id);
        let fr = node.map(|n| n.failure_relevance).unwrap_or(0.0);
        let rec = node.map(|n| n.recency).unwrap_or(0.0);
        println!(
            "    {} → score: {:.4} (failure: {:.2}, recency: {:.2})",
            id, score, fr, rec
        );
    }

    // ── Phase 5: Context Window Selection ─────────────────────────
    println!("\n─── PHASE 5: VRAM-like Context Selection ───\n");

    let context = memory_graph.select_context_window(2048);
    println!(
        "  [VRAM] Selected {} nodes for context window (2048 token budget):",
        context.len()
    );
    for node in &context {
        let score = memory_graph.compute_node_score(node);
        println!(
            "    {} ({:?}) — score: {:.3} | recency: {:.2}",
            node.label, node.node_type, score, node.recency
        );
    }

    // ── Phase 6: Event Sourcing Replay ────────────────────────────
    println!("\n─── PHASE 6: Deterministic Replay ───\n");

    let replay_event_id = event_log.append(
        EventType::ReplayStart,
        EventPayload::ReplayEvent {
            original_event_id: "none".into(),
            replayed_at: 0,
        },
    );

    let mut replayer = ccos_core::event_log::EventReplayer::new();
    match event_log.replay_deterministic(&mut replayer) {
        Ok(count) => {
            println!("  [REPLAY] Replayed {} events successfully", count);
            println!(
                "  [REPLAY] Stats: {} llm, {} parse, {} graph, {} failures",
                replayer.statistics.llm_calls,
                replayer.statistics.parsing_events,
                replayer.statistics.graph_mutations,
                replayer.statistics.failures
            );
        }
        Err(e) => {
            println!("  [REPLAY] Error: {}", e);
        }
    }

    event_log.append(
        EventType::ReplayEnd,
        EventPayload::ReplayEvent {
            original_event_id: replay_event_id,
            replayed_at: 1,
        },
    );

    // ── Phase 7: Paging Enforcement ───────────────────────────────
    println!("\n─── PHASE 7: Paging Threshold Enforcement ───\n");
    let before_paging = memory_graph.node_count();
    memory_graph.max_in_memory_nodes = 15;
    memory_graph.enforce_paging();
    println!(
        "  [PAGING] Nodes: {} → {} (threshold: {})",
        before_paging,
        memory_graph.node_count(),
        memory_graph.paging_threshold
    );

    // ── Final Snapshot ────────────────────────────────────────────
    event_log.append(
        EventType::Snapshot,
        EventPayload::Snapshot {
            nodes_count: memory_graph.node_count(),
            edges_count: memory_graph.edge_count(),
            total_events: event_log.event_count(),
        },
    );

    event_log.append(
        EventType::CycleEnd,
        EventPayload::CycleEvent {
            cycle_number: 1,
            action: "cycle_complete".into(),
        },
    );

    // ── Phase 8: Tamper-evident Distributed Log ───────────────────
    println!("\n─── PHASE 8: Hash-Chained Log Integrity ───\n");
    dist_log.append("cycle_complete".into(), "kernel".into());
    let integrity = dist_log.verify_integrity();
    println!(
        "  [CHAIN] {} links | valid: {} | errors: {}",
        integrity.verified_events,
        integrity.valid,
        integrity.errors.len()
    );

    // ── Summary ──────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  CCOS CYCLE COMPLETE                         ║");
    println!("╠══════════════════════════════════════════════╣");
    println!(
        "║  Session:      {:<30}║",
        &session_id[..30.min(session_id.len())]
    );
    println!("║  Total Events: {:<30}║", event_log.event_count());
    println!("║  Graph Nodes:  {:<30}║", memory_graph.node_count());
    println!("║  Graph Edges:  {:<30}║", memory_graph.edge_count());
    println!(
        "║  Mutations:    {:<30}║",
        incremental_engine.total_mutations()
    );
    println!("║  Guard Status: {:<30}║", guard_status);
    println!("╚══════════════════════════════════════════════╝");
}
