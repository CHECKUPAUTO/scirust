//! `ccos setup` — the one-command installer/verifier behind `scripts/install.sh`.
//!
//! Turns a freshly built binary into a **wired, self-tested** deployment in one
//! deterministic pass: probe the host (read-only), register the MCP server with
//! the agent host in the project's `.mcp.json` (idempotent merge, consent-gated),
//! run a first-run **self-test battery** against the real kernel, and seal the
//! verdict into `setup_report.json`.
//!
//! Two design rules, both inherited from the rest of CCOS:
//!
//! 1. **The verdict is produced by code, not by a model.** An LLM agent cannot
//!    be *obliged* to report success truthfully; the installer can. The sealed
//!    report (content-hashed, per-check pass/fail) is the source of truth; the
//!    agent is only the *messenger* — it reads the report through the
//!    `ccos://setup/report` MCP resource (see [`crate::mcp`]) and relays it.
//! 2. **Fail closed, announce every refusal.** Nothing outside the project
//!    directory is ever written; the project `.mcp.json` is only written with
//!    explicit consent (`--yes` or an interactive confirmation); an unparseable
//!    `.mcp.json` is never overwritten; a skipped step appears in the report as
//!    a visible `skipped_*` status, never as silence.
//!
//! See `docs/SETUP.md` for the operator guide and the single-writer rule
//! (Mode A tools vs Mode B hook — `docs/SELF_ANALYSIS.md`).

use std::io;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::egress::EgressAllowlist;
use crate::external_memory::{CcosMemory, ExternalMemory, Recall};

/// Environment variable that points the MCP resource reader (and anything else)
/// at a non-default report location.
pub const REPORT_ENV: &str = "CCOS_SETUP_REPORT";
/// Default report filename, resolved relative to the process working directory
/// (the project root — the same directory `.mcp.json` launches the server from).
pub const REPORT_DEFAULT: &str = "setup_report.json";
/// The report schema tag, bumped on any breaking shape change.
pub const REPORT_SCHEMA: &str = "ccos.setup.report/v1";

/// Resolve the report path: `$CCOS_SETUP_REPORT` when set, else
/// [`REPORT_DEFAULT`] in the current directory. Shared with the
/// `ccos://setup/report` MCP resource so the CLI writer and the agent-facing
/// reader can never drift apart.
pub fn report_path() -> PathBuf {
    std::env::var(REPORT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(REPORT_DEFAULT))
}

// ── Phase 1: probe (read-only) ──────────────────────────────────────

/// The wiring state of the project's `.mcp.json`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum McpWiring {
    /// No `.mcp.json` in the project directory.
    Missing,
    /// `.mcp.json` exists and parses, but has no `ccos` server entry.
    PresentUnwired,
    /// A `ccos` server entry exists. `command_exists` says whether its command
    /// path resolves to a file on this host (a stale entry is a warning).
    Wired {
        command: String,
        command_exists: bool,
    },
    /// `.mcp.json` exists but is not valid JSON — setup refuses to touch it.
    Invalid { error: String },
}

/// Reachability of the optional local LLM endpoint (`$OLLAMA_ENDPOINT`).
/// Purely informational: CCOS needs no LLM to install or self-test.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum OllamaProbe {
    /// TCP connect succeeded within the probe timeout.
    Reachable { endpoint: String },
    /// Nothing answered (not installed / not running) — not an error.
    Unreachable { endpoint: String },
    /// The egress allowlist refused the endpoint (air-gap posture, fail-closed);
    /// the probe never opened a socket.
    RefusedByEgress { endpoint: String, reason: String },
}

/// What [`probe`] found on the host. Strictly read-only.
#[derive(Debug, Clone, Serialize)]
pub struct HostProbe {
    pub os: &'static str,
    pub arch: &'static str,
    pub release_build: bool,
    /// Active license tier (`pro` / `community`) — mirrors `ccos doctor`.
    pub license_tier: &'static str,
    /// Absolute path of the running binary, when the OS reports one.
    pub binary_path: Option<String>,
    /// SHA-256 of the running binary — pins the report to the exact build it
    /// certifies (the same role a chain link plays in the event log).
    pub binary_sha256: Option<String>,
    pub mcp_json: McpWiring,
    /// Whether the project's `.claude/settings.json` already wires the Mode B
    /// automatic-feed hook (`ccos_self_feed.py`) — see the single-writer rule.
    pub hook_wired: bool,
    pub ollama: OllamaProbe,
    /// Whether a `workspace.ccos` already exists in the project directory.
    pub workspace_present: bool,
}

/// Probe the host and the project directory. Read-only: opens no socket beyond
/// the egress-checked localhost LLM probe, writes nothing.
pub fn probe(project_dir: &Path) -> HostProbe {
    let now = crate::license::now_unix();
    let licensing = crate::license::Licensing::detect(now);
    let pro = matches!(licensing.tier(now), crate::license::Tier::Pro);

    let binary_path = std::env::current_exe()
        .ok()
        .map(|p| p.display().to_string());
    let binary_sha256 = binary_path
        .as_deref()
        .and_then(|p| std::fs::read(p).ok())
        .map(|bytes| crate::util::hex32(&sha256_of_bytes(&bytes)));

    HostProbe {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        release_build: !cfg!(debug_assertions),
        license_tier: if pro { "pro" } else { "community" },
        binary_path,
        binary_sha256,
        mcp_json: probe_mcp_json(&project_dir.join(".mcp.json")),
        hook_wired: probe_hook(&project_dir.join(".claude").join("settings.json")),
        ollama: probe_ollama(),
        workspace_present: project_dir.join("workspace.ccos").exists(),
    }
}

/// Raw 32-byte SHA-256 of a byte slice (the binary is not UTF-8, so
/// [`crate::util::sha256_hex`]'s `&str` surface does not apply).
fn sha256_of_bytes(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn probe_mcp_json(path: &Path) -> McpWiring {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return McpWiring::Missing,
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return McpWiring::Invalid {
                error: e.to_string(),
            }
        }
    };
    match parsed.get("mcpServers").and_then(|s| s.get("ccos")) {
        Some(entry) => {
            let command = entry
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            // A relative command resolves against the project dir the host
            // launches from — the .mcp.json's own directory.
            let base = path.parent().unwrap_or(Path::new("."));
            let command_exists = !command.is_empty()
                && (Path::new(&command).is_absolute() && Path::new(&command).exists()
                    || base.join(&command).exists());
            McpWiring::Wired {
                command,
                command_exists,
            }
        }
        None => McpWiring::PresentUnwired,
    }
}

/// Detect the Mode B hook by content: any mention of the shipped feed script in
/// the project's Claude settings counts as wired (we only need the signal for
/// the single-writer warning, not a full hook parse).
fn probe_hook(settings_path: &Path) -> bool {
    std::fs::read_to_string(settings_path)
        .map(|t| t.contains("ccos_self_feed"))
        .unwrap_or(false)
}

fn probe_ollama() -> OllamaProbe {
    let endpoint =
        std::env::var("OLLAMA_ENDPOINT").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Same air-gap gate as every network call site: the probe itself must not
    // become an egress bypass.
    let validated = match EgressAllowlist::from_env().validate(&endpoint) {
        Ok(validated) => validated,
        Err(e) => {
            return OllamaProbe::RefusedByEgress {
                endpoint,
                reason: e.to_string(),
            }
        }
    };
    let reachable = validated
        .addresses()
        .iter()
        .any(|address| TcpStream::connect_timeout(address, Duration::from_millis(300)).is_ok());
    if reachable {
        OllamaProbe::Reachable { endpoint }
    } else {
        OllamaProbe::Unreachable { endpoint }
    }
}

/// `scheme://host[:port][/…]` → `(host, port)`, defaulting the port from the
/// scheme. `None` when there is no authority to probe.
#[cfg(test)]
fn host_port(url: &str) -> Option<(String, u16)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.is_empty() {
        return None;
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() => {
            (h.to_string(), p.parse().ok()?)
        }
        _ => (
            authority.to_string(),
            if scheme == "https" { 443 } else { 80 },
        ),
    };
    Some((host, port))
}

// ── Phase 2: wiring (consent-gated, idempotent) ─────────────────────

/// One planned (or performed) write, with its outcome. Every skipped write is
/// recorded — an announced refusal, never silence.
#[derive(Debug, Clone, Serialize)]
pub struct WiringAction {
    /// The file the action targets.
    pub path: String,
    pub status: ActionStatus,
    pub summary: String,
}

/// Outcome of a [`WiringAction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    /// The write was performed.
    Applied,
    /// Nothing to do — the wiring was already in place (idempotent re-run).
    AlreadyWired,
    /// `--dry-run`: shown, not written.
    SkippedDryRun,
    /// No consent (`--yes` absent and no interactive confirmation).
    SkippedNoConsent,
    /// The write was attempted and failed, or is refused (unparseable target).
    Failed,
}

/// Merge a `ccos` server entry into `.mcp.json` at `path` (creating the file if
/// absent), preserving every other key. Refuses an unparseable file and an
/// existing `ccos` entry (the caller treats that as [`ActionStatus::AlreadyWired`]).
/// Returns a one-line summary of what was written.
pub fn wire_mcp_json(path: &Path, command: &str, workspace: &str) -> io::Result<String> {
    let mut root: Value = match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("refusing to rewrite unparseable {}: {e}", path.display()),
            )
        })?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(e),
    };
    if !root.is_object() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "refusing to rewrite {}: top level is not a JSON object",
                path.display()
            ),
        ));
    }
    let servers = root
        .as_object_mut()
        .expect("checked object above")
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    let Some(servers) = servers.as_object_mut() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "refusing to rewrite {}: \"mcpServers\" is not an object",
                path.display()
            ),
        ));
    };
    if servers.contains_key("ccos") {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "a ccos server entry already exists",
        ));
    }
    servers.insert(
        "ccos".to_string(),
        json!({ "command": command, "args": ["mcp", workspace] }),
    );
    let pretty = serde_json::to_string_pretty(&root).expect("serializable value");
    crate::util::write_durable(path, format!("{pretty}\n").as_bytes())?;
    Ok(format!(
        "registered MCP server `ccos mcp {workspace}` (command {command})"
    ))
}

/// The Mode B (automatic feed) hook block for a Claude Code `settings.json`,
/// printed for the operator to wire by hand. Setup never writes it: a hook
/// executes commands on the agent host's behalf, so its installation belongs
/// to the operator — and the single-writer rule means choosing it usually
/// means *not* letting the MCP tools write the same workspace.
pub fn hook_snippet() -> &'static str {
    r#"{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Read|Edit|Write|Bash",
        "hooks": [
          { "type": "command", "command": "python3 scripts/ccos_self_feed.py" }
        ]
      }
    ]
  }
}"#
}

// ── Phase 3: the first-run self-test battery ────────────────────────

/// One deterministic check against the real kernel.
#[derive(Debug, Clone, Serialize)]
pub struct SelfTest {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

impl SelfTest {
    fn new(name: &'static str, passed: bool, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed,
            detail: detail.into(),
        }
    }
}

/// Run the first-run battery: ingest → causal recall → failure propagation →
/// hash-chain integrity → checkpoint round-trip determinism → MCP handshake.
/// Everything runs in-process against a throwaway kernel (plus one temp file
/// for the round-trip); the project is never touched. Deterministic by
/// construction — the same battery on the same build yields the same verdict.
pub fn run_self_test() -> Vec<SelfTest> {
    let mut results = Vec::new();
    let mut mem = CcosMemory::new();

    // 1 — ingest a two-file witness with a cross-file dependency.
    let db = mem.ingest_source("src/db.rs", "pub fn connect() -> i32 { 1 }\n");
    let service = mem.ingest_source(
        "src/service.rs",
        "use crate::db;\npub fn load() -> i32 { db::connect() }\n",
    );
    let stats = mem.stats();
    results.push(SelfTest::new(
        "ingest",
        db.nodes_added > 0 && service.nodes_added > 0,
        format!(
            "2 witness files -> {} nodes, {} edges",
            stats.nodes, stats.edges
        ),
    ));

    // 2 — the core promise: the cross-file dependency is paged into a bounded
    // window anchored on the dependent file (no K to tune).
    let window = mem.recall(&Recall::around("file:src/service.rs"), 2048);
    let dep_recalled = window.items.iter().any(|i| i.uri == "file:src/db.rs");
    results.push(SelfTest::new(
        "causal recall",
        dep_recalled && window.tokens <= 2048,
        format!(
            "cross-file dependency {} a 2048-token window ({} items, {} tokens)",
            if dep_recalled {
                "paged into"
            } else {
                "MISSING from"
            },
            window.items.len(),
            window.tokens
        ),
    ));

    // 3 — failure pressure propagates from the failing file to its causes.
    let affected = mem.signal_failure("file:src/db.rs", 3);
    results.push(SelfTest::new(
        "failure propagation",
        matches!(affected, Ok(n) if n > 0),
        match &affected {
            Ok(n) => format!("{n} node(s) under pressure from the failing file"),
            Err(e) => format!("signal_failure failed: {e}"),
        },
    ));

    // 4 — the tamper-evident chain over everything the battery just did.
    let integrity = mem.verify();
    results.push(SelfTest::new(
        "hash-chain integrity",
        integrity.valid && integrity.events > 0,
        if integrity.valid {
            format!("{} event(s) verified, chain intact", integrity.events)
        } else {
            format!("chain BROKEN: {:?}", integrity.errors)
        },
    ));

    // 5 — checkpoint → reload is bit-identical (the replay guarantee, measured
    // on this host's filesystem rather than assumed).
    results.push(checkpoint_round_trip(&mut mem));

    // 6 — an MCP host can complete the handshake and sees the tool catalogue
    // plus the setup-report resource the agent relays the verdict from.
    results.push(mcp_handshake());

    results
}

fn checkpoint_round_trip(mem: &mut CcosMemory) -> SelfTest {
    let dir = std::env::temp_dir().join(format!("ccos-setup-selftest-{}", std::process::id()));
    let ws = dir.join("selftest.ccos");
    let outcome = (|| -> Result<(String, String), String> {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        mem.checkpoint_to(&ws).map_err(|e| e.to_string())?;
        let live = mem.state_fingerprint().map_err(|e| e.to_string())?;
        let reloaded = CcosMemory::open(&ws)
            .and_then(|m| m.state_fingerprint())
            .map_err(|e| e.to_string())?;
        Ok((live, reloaded))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    match outcome {
        Ok((live, reloaded)) if live == reloaded => SelfTest::new(
            "checkpoint determinism",
            true,
            format!(
                "state fingerprint identical after reload ({}…)",
                &live[..12]
            ),
        ),
        Ok((live, reloaded)) => SelfTest::new(
            "checkpoint determinism",
            false,
            format!("fingerprint DIVERGED: live {live} vs reloaded {reloaded}"),
        ),
        Err(e) => SelfTest::new("checkpoint determinism", false, e),
    }
}

fn mcp_handshake() -> SelfTest {
    let mut session = crate::agent_session::AgentSession::new();
    let call = |session: &mut crate::agent_session::AgentSession, id: u64, method: &str| {
        crate::mcp::handle(
            session,
            &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": {} }),
        )
    };
    let init_ok = call(&mut session, 1, "initialize")
        .and_then(|r| r.get("result")?.get("capabilities").cloned())
        .map(|caps| caps.get("tools").is_some() && caps.get("resources").is_some())
        .unwrap_or(false);
    let tools = call(&mut session, 2, "tools/list")
        .and_then(|r| Some(r.get("result")?.get("tools")?.as_array()?.len()))
        .unwrap_or(0);
    let report_resource = call(&mut session, 3, "resources/list")
        .and_then(|r| {
            Some(
                r.get("result")?
                    .get("resources")?
                    .as_array()?
                    .iter()
                    .any(|s| s.get("uri").and_then(Value::as_str) == Some("ccos://setup/report")),
            )
        })
        .unwrap_or(false);
    SelfTest::new(
        "mcp handshake",
        init_ok && tools > 0 && report_resource,
        format!(
            "initialize ok={init_ok}, {tools} tool(s), setup-report resource advertised={report_resource}"
        ),
    )
}

// ── Phase 4: the sealed report ──────────────────────────────────────

/// The deterministic verdict of a setup run — the artifact the operator reads
/// and the agent relays (via the `ccos://setup/report` MCP resource). Sealed:
/// `report_sha256` commits to every other field.
#[derive(Debug, Clone, Serialize)]
pub struct SetupReport {
    pub schema: &'static str,
    pub version: &'static str,
    /// Unix time the report was generated (informational; excluded from
    /// nothing — the hash commits to it like every other field).
    pub generated_unix: u64,
    pub probe: HostProbe,
    pub actions: Vec<WiringAction>,
    pub tests: Vec<SelfTest>,
    pub tests_passed: usize,
    pub tests_total: usize,
    /// The verdict: every check in the battery passed.
    pub ok: bool,
    /// SHA-256 over the canonical JSON of this report with this field empty —
    /// the same content-hash sealing used across CCOS.
    pub report_sha256: String,
}

impl SetupReport {
    /// Assemble and seal a report.
    pub fn seal(probe: HostProbe, actions: Vec<WiringAction>, tests: Vec<SelfTest>) -> Self {
        let tests_passed = tests.iter().filter(|t| t.passed).count();
        let tests_total = tests.len();
        let mut report = SetupReport {
            schema: REPORT_SCHEMA,
            version: env!("CARGO_PKG_VERSION"),
            generated_unix: crate::license::now_unix(),
            probe,
            actions,
            tests,
            tests_passed,
            tests_total,
            ok: tests_passed == tests_total && tests_total > 0,
            report_sha256: String::new(),
        };
        report.report_sha256 = report.content_hash();
        report
    }

    /// The content hash: canonical JSON with `report_sha256` blanked.
    pub fn content_hash(&self) -> String {
        let mut unsealed = serde_json::to_value(self).expect("report serializes");
        unsealed["report_sha256"] = json!("");
        crate::util::sha256_hex(&unsealed.to_string())
    }

    /// Write the sealed report durably (atomic rename + fsync) to `path`.
    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let pretty = serde_json::to_string_pretty(self).expect("report serializes");
        crate::util::write_durable(path, format!("{pretty}\n").as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ccos-setup-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn self_test_battery_passes_on_this_build() {
        // The battery is the product: if any check fails on a healthy build, the
        // installer would certify a broken deployment (or refuse a good one).
        let results = run_self_test();
        assert_eq!(results.len(), 6, "the battery runs all six checks");
        for t in &results {
            assert!(t.passed, "self-test '{}' failed: {}", t.name, t.detail);
        }
    }

    #[test]
    fn wire_mcp_json_creates_and_preserves() {
        let dir = tmp_dir("wire");
        let path = dir.join(".mcp.json");

        // Creates the file when missing.
        wire_mcp_json(&path, "/usr/local/bin/ccos", "workspace.ccos").unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["mcpServers"]["ccos"]["command"], "/usr/local/bin/ccos");
        assert_eq!(v["mcpServers"]["ccos"]["args"][1], "workspace.ccos");

        // Refuses a second wiring (idempotence is the caller's AlreadyWired path).
        let err = wire_mcp_json(&path, "/elsewhere/ccos", "workspace.ccos").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);

        // Merging preserves foreign servers and top-level keys.
        std::fs::write(
            &path,
            r#"{ "mcpServers": { "other": { "command": "x" } }, "keep": 1 }"#,
        )
        .unwrap();
        wire_mcp_json(&path, "/usr/local/bin/ccos", "workspace.ccos").unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["mcpServers"]["other"]["command"], "x");
        assert_eq!(v["keep"], 1);
        assert!(v["mcpServers"]["ccos"].is_object());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wire_mcp_json_refuses_an_unparseable_file() {
        // Fail closed: never rewrite a file we cannot faithfully merge.
        let dir = tmp_dir("invalid");
        let path = dir.join(".mcp.json");
        std::fs::write(&path, "{ not json").unwrap();
        let err = wire_mcp_json(&path, "/usr/local/bin/ccos", "workspace.ccos").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{ not json",
            "the unparseable file is left byte-identical"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_reports_mcp_json_states() {
        let dir = tmp_dir("probe");
        assert!(matches!(
            probe_mcp_json(&dir.join(".mcp.json")),
            McpWiring::Missing
        ));

        std::fs::write(dir.join(".mcp.json"), r#"{ "mcpServers": {} }"#).unwrap();
        assert!(matches!(
            probe_mcp_json(&dir.join(".mcp.json")),
            McpWiring::PresentUnwired
        ));

        std::fs::write(
            dir.join(".mcp.json"),
            r#"{ "mcpServers": { "ccos": { "command": "./missing" } } }"#,
        )
        .unwrap();
        match probe_mcp_json(&dir.join(".mcp.json")) {
            McpWiring::Wired {
                command,
                command_exists,
            } => {
                assert_eq!(command, "./missing");
                assert!(!command_exists, "a stale command path is flagged");
            }
            other => panic!("expected Wired, got {other:?}"),
        }

        std::fs::write(dir.join(".mcp.json"), "nope").unwrap();
        assert!(matches!(
            probe_mcp_json(&dir.join(".mcp.json")),
            McpWiring::Invalid { .. }
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_is_sealed_and_reproducible() {
        let tests = vec![
            SelfTest::new("a", true, "ok"),
            SelfTest::new("b", false, "boom"),
        ];
        let dir = tmp_dir("report");
        let report = SetupReport::seal(probe(&dir), Vec::new(), tests);
        assert_eq!(report.tests_passed, 1);
        assert_eq!(report.tests_total, 2);
        assert!(!report.ok, "one failing check fails the verdict");
        // The seal commits to the content: recomputing over the sealed report
        // (with the hash field blanked) reproduces it exactly.
        assert_eq!(report.report_sha256, report.content_hash());
        assert_eq!(report.report_sha256.len(), 64);

        // And it round-trips to disk as valid JSON with the same hash.
        let path = dir.join("setup_report.json");
        report.write_to(&path).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["report_sha256"], report.report_sha256.as_str());
        assert_eq!(v["schema"], REPORT_SCHEMA);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_battery_never_certifies() {
        // ok requires at least one passing check — an installer that ran no
        // checks must not claim success.
        let dir = tmp_dir("empty");
        let report = SetupReport::seal(probe(&dir), Vec::new(), Vec::new());
        assert!(!report.ok);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_port_parses_common_endpoint_shapes() {
        assert_eq!(
            host_port("http://localhost:11434"),
            Some(("localhost".into(), 11434))
        );
        assert_eq!(
            host_port("http://127.0.0.1:11434/api/generate"),
            Some(("127.0.0.1".into(), 11434))
        );
        assert_eq!(
            host_port("http://localhost"),
            Some(("localhost".into(), 80))
        );
        assert_eq!(host_port("https://host"), Some(("host".into(), 443)));
        assert_eq!(host_port("not a url"), None);
    }
}
