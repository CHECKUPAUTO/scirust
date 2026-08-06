mod commands_demo;
mod commands_runtime;

// Optional drop-in allocator for bare-metal benchmarking (off by default; build
// with `--features mimalloc`). CCOS is not allocation-bound at its scale, so this
// is a knob to *measure*, not a default win — see docs/PERFORMANCE.md.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use ccos_core::adversarial::{AdversarialEngine, AdversarialMode};
use ccos_core::agent_session::AgentSession;
use ccos_core::context_policy::ContextPolicy;
use ccos_core::context_region::file_of;
use ccos_core::distributed_event_log::DistributedEventLog;
use ccos_core::eval::{run_eval, EvalConfig};
use ccos_core::event_log::{EventLog, EventPayload, EventReplayer, EventType, GraphReconstructor};
use ccos_core::experiment::{run_experiment, ExperimentConfig};
use ccos_core::external_memory::{CcosMemory, ExternalMemory, Recall, RecallWindow};
use ccos_core::guard::{GuardConfig, GuardLayer};
use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::{MemoryGraph, NodeId, ScoringWeights};
use ccos_core::persist::KernelSnapshot;
use ccos_core::query;
use ccos_core::region_engine::{ContextRegionEngine, RegionQuery};
use ccos_core::region_metrics;
use ccos_core::trace::parse_cargo_test_output;
use ccos_core::trace::ExecutionTrace;
use ccos_core::util::sha256_hex;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// A CLI failure: an exit `code`, plus an optional stderr `message`. `message: None` is a non-zero
/// *status* whose report the handler already printed (not an error to announce again).
struct CliError {
    code: i32,
    message: Option<String>,
}
impl CliError {
    fn usage(m: impl Into<String>) -> Self {
        Self {
            code: 2,
            message: Some(m.into()),
        }
    }
    fn fail(m: impl Into<String>) -> Self {
        Self {
            code: 1,
            message: Some(m.into()),
        }
    }
    fn status(code: i32) -> Self {
        Self {
            code,
            message: None,
        }
    }
}
type CliResult = Result<(), CliError>;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("demo");
    let rest = &args[args.len().min(2)..];

    // Map an `i32` exit code from a handler that has already printed its own
    // report (the v0.3 runtime handlers in `commands_runtime`) into a CliResult,
    // preserving the exact code without re-announcing any message.
    let from_code = |code: i32| -> CliResult {
        if code == 0 {
            Ok(())
        } else {
            Err(CliError::status(code))
        }
    };

    let result: CliResult = match command {
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        "-V" | "--version" | "version" => {
            println!("ccos {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "demo" => {
            commands_demo::run_demo().await;
            Ok(())
        }
        "analyze" => run_analyze(&AnalyzeOpts::parse(rest)),
        "verify" => run_verify(rest.first().map(String::as_str)),
        "replay" => run_replay(rest.first().map(String::as_str)),
        "diff" => run_diff(
            rest.first().map(String::as_str),
            rest.get(1).map(String::as_str),
        ),
        "failure" => run_failure(&FailureOpts::parse(rest)),
        "focus" => run_focus(&FocusOpts::parse(rest)),
        "chaos" => run_chaos(&ChaosOpts::parse(rest)),
        "top" => run_top(&TopOpts::parse(rest)),
        "blame" => run_blame(&BlameOpts::parse(rest)),
        "export" => run_export(&ExportOpts::parse(rest)),
        "regions" => run_regions(&RegionsOpts::parse(rest)),
        "experiment" => run_experiment_cmd(rest),
        "eval" => run_eval_cmd(rest).await,
        "memory" => run_memory_cmd(rest),
        "stdin" => run_stdin_cmd(rest),
        "doctor" => run_doctor(&DoctorOpts::parse(rest)),
        "setup" => run_setup(&SetupOpts::parse(rest)),
        "license" => run_license(rest),
        "tensions" => run_tensions(&TensionsOpts::parse(rest)),
        "audit" => run_audit(&AuditOpts::parse(rest)),
        "trace" => run_trace_cmd(),
        "mcp" => {
            // Optional positional workspace path (else $CCOS_MCP_WORKSPACE, else
            // a purely in-memory session).
            let workspace = rest
                .first()
                .filter(|a| !a.starts_with("--"))
                .map(PathBuf::from)
                .or_else(|| std::env::var("CCOS_MCP_WORKSPACE").ok().map(PathBuf::from));
            match ccos_core::mcp::serve_workspace(workspace) {
                Ok(()) => Ok(()),
                Err(e) => Err(CliError::fail(format!("ccos mcp: {e}"))),
            }
        }
        "postmortem" => run_postmortem(rest),
        "sanitize" => run_sanitize(rest),
        "sync" => run_sync(rest),
        // ── CCOS v0.3 — Autonomous Context Runtime ──────────────────
        "scan" => from_code(commands_runtime::run_scan(rest).await),
        "agents" => from_code(commands_runtime::run_agents(rest).await),
        "benchmark" => from_code(commands_runtime::run_benchmark(rest)),
        "runtime" => from_code(commands_runtime::run_runtime(rest).await),

        other => {
            eprintln!("ccos: unknown command '{other}'\n");
            print_help();
            Err(CliError::status(2))
        }
    };
    let code = match result {
        Ok(()) => 0,
        Err(e) => {
            if let Some(m) = e.message {
                eprintln!("{m}");
            }
            e.code
        }
    };
    std::process::exit(code);
}

/// Options for `ccos doctor`.
struct DoctorOpts {
    json: bool,
}

impl DoctorOpts {
    fn parse(args: &[String]) -> Self {
        Self {
            json: args.iter().any(|a| a == "--json"),
        }
    }
}

/// `ccos doctor [--json]` — a **deployment self-check**: version, build profile, target, compiled
/// features, active parser, license tier / vendor-key status, MCP readiness, and actionable warnings
/// (debug build, missing feature, placeholder key, unverified token). Pure and read-only — the first
/// thing to run after installing on a server. See `docs/DEPLOYMENT.md`.
fn run_doctor(opts: &DoctorOpts) -> CliResult {
    let now = ccos_core::license::now_unix();
    let licensing = ccos_core::license::Licensing::detect(now);
    let pro = matches!(licensing.tier(now), ccos_core::license::Tier::Pro);
    let key_set = ccos_core::license::embedded_key_is_set();
    let token_present = ccos_core::license::load_license_blob().is_some();

    let release = !cfg!(debug_assertions);
    let f_llm = cfg!(feature = "llm");
    let f_license = cfg!(feature = "license");
    let f_license_pq = cfg!(feature = "license-pq");
    let f_syn = cfg!(feature = "syn-parser");
    let f_learned = cfg!(feature = "learned-embed");
    let f_mimalloc = cfg!(feature = "mimalloc");
    let f_slhav2 = cfg!(feature = "slhav2");
    let f_octasoma = cfg!(feature = "octasoma");
    let pq_key_set = ccos_core::license::embedded_slh_dsa_key_is_set();
    let any_verifier = f_license || f_license_pq;
    let any_key_set = key_set || pq_key_set;

    // Actionable deployment warnings.
    let mut warnings: Vec<String> = Vec::new();
    if !release {
        warnings.push(
            "debug build — rebuild with --release for production (faster; debug asserts off)"
                .into(),
        );
    }
    if !f_syn {
        warnings.push(
            "syn AST parser not compiled — using the line heuristic (~36% less accurate); rebuild \
             with default features"
                .into(),
        );
    }
    if !any_verifier {
        warnings.push(
            "no license verifier compiled — Pro features unavailable; rebuild with \
             --features license (ed25519) and/or --features license-pq (post-quantum SLH-DSA)"
                .into(),
        );
    } else if !any_key_set {
        warnings.push(
            "no vendor public key is set (both embedded keys are the all-zero placeholder) — Pro is \
             fail-closed until a vendor key is set (see docs/DEPLOYMENT.md)"
                .into(),
        );
    } else if f_license_pq && !pq_key_set {
        warnings.push(
            "the post-quantum (SLH-DSA) public key is the all-zero placeholder — slhdsa. tokens \
             cannot unlock Pro until a vendor key is set (see docs/DEPLOYMENT.md §4b)"
                .into(),
        );
    }
    if any_verifier && any_key_set && !pro && token_present {
        warnings.push("a license token was found but did not verify or is expired".into());
    }

    if opts.json {
        let report = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "build": if release { "release" } else { "debug" },
            "target": { "arch": std::env::consts::ARCH, "os": std::env::consts::OS },
            "features": {
                "llm": f_llm, "license": f_license, "license-pq": f_license_pq,
                "syn-parser": f_syn, "learned-embed": f_learned, "mimalloc": f_mimalloc,
                "slhav2": f_slhav2,
                "octasoma": f_octasoma,
            },
            "parser": if f_syn { "syn-ast" } else { "line-heuristic" },
            "license": {
                "verifier": ccos_core::license::compiled_verifier_scheme(),
                "ed25519_key_set": key_set,
                "slh_dsa_key_set": pq_key_set,
                "tier": if pro { "pro" } else { "community" },
                "licensee": licensing.licensee(),
                "token_present": token_present,
            },
            "mcp_ready": f_llm,
            "warnings": warnings,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    let yn = |b: bool| if b { "yes" } else { "no" };
    println!("ccos doctor — deployment self-check\n");
    println!("  version      {}", env!("CARGO_PKG_VERSION"));
    println!(
        "  build        {}",
        if release { "release" } else { "debug  ⚠" }
    );
    println!(
        "  target       {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "  parser       {}",
        if f_syn {
            "syn AST (accurate)"
        } else {
            "line heuristic"
        }
    );
    println!(
        "  features     llm={} license={} license-pq={} syn-parser={} learned-embed={} mimalloc={}",
        yn(f_llm),
        yn(f_license),
        yn(f_license_pq),
        yn(f_syn),
        yn(f_learned),
        yn(f_mimalloc)
    );
    println!(
        "  mcp          {}",
        if f_llm {
            "ready  (ccos mcp <workspace>)"
        } else {
            "unavailable (needs --features llm)"
        }
    );
    println!("\n  license");
    let verifier_label = match ccos_core::license::compiled_verifier_scheme() {
        "slh-dsa+ed25519" => "slh-dsa (SLH-DSA-128s) + ed25519 (both compiled in)",
        "slh-dsa" => "slh-dsa (SLH-DSA-128s, post-quantum, compiled in)",
        "ed25519" => "ed25519 (compiled in)",
        _ => "none (community only)",
    };
    println!("    verifier   {verifier_label}");
    // Show each compiled-in verifier's vendor-key status on its own line.
    if f_license {
        println!(
            "    ed25519 key {}",
            if key_set {
                "set"
            } else {
                "placeholder (fail-closed)"
            }
        );
    }
    if f_license_pq {
        println!(
            "    slh-dsa key {}",
            if pq_key_set {
                "set"
            } else {
                "placeholder (fail-closed)"
            }
        );
    }
    println!("    tier       {}", if pro { "PRO" } else { "community" });
    if let Some(who) = licensing.licensee() {
        println!("    licensee   {who}");
    }
    println!(
        "    token      {}",
        if token_present { "present" } else { "none" }
    );

    if warnings.is_empty() {
        println!("\n  ✓ no warnings — looks deployment-ready");
    } else {
        println!("\n  ⚠ {} warning(s):", warnings.len());
        for w in &warnings {
            println!("    - {w}");
        }
        println!(
            "\n  see docs/DEPLOYMENT.md for the recommended build \
             (`--release --features llm,license,license-pq`)."
        );
    }
    Ok(())
}

/// Options for `ccos setup`.
struct SetupOpts {
    /// Project directory whose `.mcp.json` (and default report) setup targets.
    dir: PathBuf,
    /// Apply the wiring without an interactive confirmation.
    yes: bool,
    /// Show the plan and run the battery, but write nothing.
    dry_run: bool,
    /// Emit the sealed report JSON on stdout instead of the human summary.
    json: bool,
    /// Also print the Mode B (PostToolUse hook) snippet for manual wiring.
    hook: bool,
    /// Report destination (default `<dir>/setup_report.json`).
    report: Option<PathBuf>,
    /// Workspace path the registered MCP server will persist to.
    workspace: String,
}

impl SetupOpts {
    fn parse(args: &[String]) -> Self {
        let mut o = Self {
            dir: PathBuf::from("."),
            yes: false,
            dry_run: false,
            json: false,
            hook: false,
            report: None,
            workspace: "workspace.ccos".to_string(),
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--yes" | "-y" => o.yes = true,
                "--dry-run" => o.dry_run = true,
                "--json" => o.json = true,
                "--hook" => o.hook = true,
                "--dir" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        o.dir = PathBuf::from(v);
                    }
                }
                "--report" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        o.report = Some(PathBuf::from(v));
                    }
                }
                "--workspace" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        o.workspace = v.clone();
                    }
                }
                _ => {}
            }
            i += 1;
        }
        o
    }
}

/// Ask for an interactive yes/no. Fail-closed: a non-interactive stdin, a read
/// error, or anything but an explicit `y`/`yes` all mean **no**.
fn confirm(prompt: &str) -> bool {
    use std::io::{IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return false;
    }
    print!("{prompt} [y/N] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

/// `ccos setup [--yes] [--dry-run] [--json] [--hook] [--dir D] [--report F] [--workspace W]`
/// — the one-command install-and-verify pass (see `docs/SETUP.md` and `ccos_core::setup`):
/// probe the host, register the MCP server in the project's `.mcp.json`
/// (idempotent, consent-gated, fail-closed on anything unparseable), run the
/// deterministic first-run self-test battery, and seal the verdict into
/// `setup_report.json` — the artifact an MCP agent relays via `ccos://setup/report`.
/// Exits 0 only when every check passed.
fn run_setup(opts: &SetupOpts) -> CliResult {
    use ccos_core::setup::{self, ActionStatus, McpWiring, OllamaProbe, SetupReport, WiringAction};

    let probe = setup::probe(&opts.dir);
    let quiet = opts.json;
    let yn = |b: bool| if b { "yes" } else { "no" };

    if !quiet {
        println!("ccos setup — install, wire, self-test\n");
        println!("  host");
        println!(
            "    target       {}-{} ({} build)",
            probe.arch,
            probe.os,
            if probe.release_build {
                "release"
            } else {
                "debug ⚠"
            }
        );
        println!("    license      {}", probe.license_tier);
        if let (Some(path), Some(sha)) = (&probe.binary_path, &probe.binary_sha256) {
            println!("    binary       {path} (sha256 {}…)", &sha[..12]);
        }
        println!("  agent host");
        match &probe.mcp_json {
            McpWiring::Wired {
                command,
                command_exists,
            } => println!(
                "    .mcp.json    wired → {command}{}",
                if *command_exists {
                    ""
                } else {
                    "  ⚠ command not found on this host"
                }
            ),
            McpWiring::PresentUnwired => println!("    .mcp.json    present, no ccos entry"),
            McpWiring::Missing => println!("    .mcp.json    missing"),
            McpWiring::Invalid { error } => {
                println!("    .mcp.json    ⚠ unparseable ({error}) — setup will not touch it")
            }
        }
        println!("    hook (B)     wired={}", yn(probe.hook_wired));
        match &probe.ollama {
            OllamaProbe::Reachable { endpoint } => {
                println!("    ollama       reachable ({endpoint})")
            }
            OllamaProbe::Unreachable { endpoint } => {
                println!(
                    "    ollama       not running ({endpoint}) — optional, nothing requires it"
                )
            }
            OllamaProbe::RefusedByEgress { endpoint, reason } => {
                println!("    ollama       probe refused by egress policy ({endpoint}): {reason}")
            }
        }
        println!(
            "    workspace    {}",
            if probe.workspace_present {
                "workspace.ccos present"
            } else {
                "none yet (created on first MCP write)"
            }
        );
    }

    // ── Wiring (consent-gated, idempotent, fail-closed) ─────────────
    let mcp_path = opts.dir.join(".mcp.json");
    let command = probe
        .binary_path
        .clone()
        .unwrap_or_else(|| "ccos".to_string());
    let mut actions: Vec<WiringAction> = Vec::new();
    match &probe.mcp_json {
        McpWiring::Wired { .. } => actions.push(WiringAction {
            path: mcp_path.display().to_string(),
            status: ActionStatus::AlreadyWired,
            summary: "ccos MCP server already registered — nothing to do".into(),
        }),
        McpWiring::Invalid { error } => actions.push(WiringAction {
            path: mcp_path.display().to_string(),
            status: ActionStatus::Failed,
            summary: format!("refused: existing .mcp.json is unparseable ({error})"),
        }),
        McpWiring::Missing | McpWiring::PresentUnwired => {
            let plan = format!(
                "register MCP server `ccos mcp {}` (command {command}) in {}",
                opts.workspace,
                mcp_path.display()
            );
            if opts.dry_run {
                if !quiet {
                    println!("\n  plan (dry-run, not written)\n    - {plan}");
                }
                actions.push(WiringAction {
                    path: mcp_path.display().to_string(),
                    status: ActionStatus::SkippedDryRun,
                    summary: plan,
                });
            } else {
                let consent = opts.yes || {
                    if !quiet {
                        println!("\n  plan\n    - {plan}");
                    }
                    confirm("  apply?")
                };
                if consent {
                    match setup::wire_mcp_json(&mcp_path, &command, &opts.workspace) {
                        Ok(summary) => actions.push(WiringAction {
                            path: mcp_path.display().to_string(),
                            status: ActionStatus::Applied,
                            summary,
                        }),
                        Err(e) => actions.push(WiringAction {
                            path: mcp_path.display().to_string(),
                            status: ActionStatus::Failed,
                            summary: format!("write failed: {e}"),
                        }),
                    }
                } else {
                    actions.push(WiringAction {
                        path: mcp_path.display().to_string(),
                        status: ActionStatus::SkippedNoConsent,
                        summary: format!("{plan} — skipped (no consent; re-run with --yes)"),
                    });
                }
            }
        }
    }
    if !quiet {
        println!("\n  wiring");
        for a in &actions {
            let mark = match a.status {
                ActionStatus::Applied | ActionStatus::AlreadyWired => "✓",
                ActionStatus::SkippedDryRun | ActionStatus::SkippedNoConsent => "•",
                ActionStatus::Failed => "✗",
            };
            println!("    {mark} {}", a.summary);
        }
        // One writer per workspace (docs/SELF_ANALYSIS.md): the Mode B hook and
        // the live MCP server must not both write the same workspace.ccos.
        if probe.hook_wired {
            println!(
                "    ⚠ the Mode B feed hook is already wired — keep ONE writer per \
                 workspace: don't let agents call mutating ccos tools on the same \
                 workspace the hook feeds (docs/SELF_ANALYSIS.md)"
            );
        }
    }
    if opts.hook && !quiet {
        println!(
            "\n  Mode B hook (manual wiring — setup never edits agent settings itself):\n\
             \x20   add to .claude/settings.json, then pick ONE writer per workspace:\n{}",
            ccos_core::setup::hook_snippet()
        );
    }

    // ── The deterministic first-run battery ─────────────────────────
    if !quiet {
        println!("\n  self-test (deterministic first-run battery)");
    }
    let tests = setup::run_self_test();
    if !quiet {
        for t in &tests {
            println!(
                "    {} {:<24} {}",
                if t.passed { "✓" } else { "✗" },
                t.name,
                t.detail
            );
        }
    }

    // ── Seal and write the verdict ──────────────────────────────────
    let report = SetupReport::seal(probe, actions, tests);
    let report_file = opts
        .report
        .clone()
        .unwrap_or_else(|| opts.dir.join(setup::REPORT_DEFAULT));
    if !opts.dry_run {
        report
            .write_to(&report_file)
            .map_err(|e| CliError::fail(format!("ccos setup: writing the report: {e}")))?;
    }
    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        if !opts.dry_run {
            println!(
                "\n  report → {} (sha256 {}…)",
                report_file.display(),
                &report.report_sha256[..12]
            );
        } else {
            println!("\n  report not written (dry-run)");
        }
        if report.ok {
            println!(
                "\n  ✓ setup complete — {}/{} checks passed. An MCP agent reads this \
                 verdict at ccos://setup/report and relays it; the report file is \
                 the source of truth.",
                report.tests_passed, report.tests_total
            );
        } else {
            println!(
                "\n  ✗ setup NOT certified — {}/{} checks passed; see the report above.",
                report.tests_passed, report.tests_total
            );
        }
    }
    if report.ok {
        Ok(())
    } else {
        Err(CliError::status(1))
    }
}

/// `ccos license` — report the active licensing tier (community / Pro), the licensee and expiry, and
/// the Pro feature set. Read-only and **offline**: it loads any local token (`$CCOS_LICENSE` or the
/// license file) and verifies it against the baked-in public key. Without the `license` feature there
/// is no embedded verifier, so it always reports community — the core is never gated or degraded.
fn run_license(_args: &[String]) -> CliResult {
    use ccos_core::license::{Feature, Tier};
    let now = ccos_core::license::now_unix();
    let blob = ccos_core::license::load_license_blob();
    let licensing = ccos_core::license::Licensing::detect(now);

    match licensing.tier(now) {
        Tier::Pro => {
            println!("ccos license: PRO");
            if let Some(who) = licensing.licensee() {
                println!("  licensee: {who}");
            }
            println!("  unlocked Pro features:");
            for f in Feature::ALL {
                println!("    - {f}");
            }
        }
        Tier::Community => {
            println!("ccos license: COMMUNITY (full core, no Pro features)");
            if blob.is_some() {
                println!("  note: a license token was found but is invalid or expired.");
            }
            println!(
                "  the core (ingestion, causal graph, Q-Page belief/decay/propagation) is fully \
                 functional."
            );
            println!("  Pro features (locally-verified annual license, nothing leaves your host):");
            for f in Feature::ALL {
                println!("    - {f}");
            }
            #[cfg(not(any(feature = "license", feature = "license-pq")))]
            println!(
                "  (this build has no license verifier compiled in — rebuild with \
                 `--features license` (ed25519) and/or `--features license-pq` \
                 (post-quantum SLH-DSA) to enable Pro)"
            );
        }
    }
    Ok(())
}

/// Load the host licensing and gate `feature`. Returns the licensing when **unlocked**; on the
/// community tier `Licensing::require` has already logged the announced refusal, and this returns
/// `None` — the caller prints a one-line note and exits 0 (announced, never an error, never a degraded
/// core). The shared front door for the Pro CLI commands.
fn gate_pro(feature: ccos_core::license::Feature) -> Option<ccos_core::license::Licensing> {
    let now = ccos_core::license::now_unix();
    let licensing = ccos_core::license::Licensing::detect(now);
    licensing.require(feature, now).is_ok().then_some(licensing)
}

/// Options for `ccos tensions`.
struct TensionsOpts {
    snapshot: Option<String>,
    min: f64,
    limit: usize,
}

impl TensionsOpts {
    fn parse(args: &[String]) -> Self {
        let mut o = Self {
            snapshot: None,
            min: 0.15,
            limit: 20,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--min" => {
                    i += 1;
                    if let Some(v) = args.get(i).and_then(|v| v.parse().ok()) {
                        o.min = v;
                    }
                }
                "--limit" => {
                    i += 1;
                    if let Some(v) = args.get(i).and_then(|v| v.parse().ok()) {
                        o.limit = v;
                    }
                }
                s if !s.starts_with("--") => o.snapshot = Some(s.to_string()),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        o
    }
}

/// `ccos tensions <snapshot.json> [--min N] [--limit N]` — **Pro**: render the contested Q-Page
/// claims (those whose `conflict ≥ --min`) ranked by tension, as a compact bar. Locked in the
/// community tier (the announced refusal is logged; the core is untouched).
fn run_tensions(opts: &TensionsOpts) -> CliResult {
    let Some(file) = opts.snapshot.as_deref() else {
        return Err(CliError::usage(
            "usage: ccos tensions <snapshot.json> [--min N] [--limit N]",
        ));
    };
    if gate_pro(ccos_core::license::Feature::TensionVisualization).is_none() {
        println!(
            "ccos tensions: locked — run `ccos license` to see how to unlock (core unaffected)."
        );
        return Ok(());
    }
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => return Err(CliError::fail(format!("ccos: cannot load '{file}': {e}"))),
    };
    let claims: Vec<_> = snapshot
        .graph
        .claim_beliefs()
        .into_iter()
        .filter(|(_, q)| q.conflict >= opts.min)
        .collect();

    println!(
        "─── Cognitive tensions — contested claims (conflict ≥ {:.2}) ───\n",
        opts.min
    );
    if claims.is_empty() {
        println!("  none (no contested Q-Page claims in this snapshot)");
        return Ok(());
    }
    for (id, q) in claims.iter().take(opts.limit) {
        println!(
            "  {}  {}",
            ccos_core::memory::render_tension_bar(q),
            truncate(&id.0, 48)
        );
    }
    if claims.len() > opts.limit {
        println!("  … (+{} more)", claims.len() - opts.limit);
    }
    Ok(())
}

/// Options for `ccos audit`.
struct AuditOpts {
    snapshot: Option<String>,
    json: bool,
    min: f64,
}

impl AuditOpts {
    fn parse(args: &[String]) -> Self {
        let mut o = Self {
            snapshot: None,
            json: false,
            min: 0.0,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => o.json = true,
                "--min" => {
                    i += 1;
                    if let Some(v) = args.get(i).and_then(|v| v.parse().ok()) {
                        o.min = v;
                    }
                }
                s if !s.starts_with("--") => o.snapshot = Some(s.to_string()),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        o
    }
}

/// `ccos audit <snapshot.json> [--json] [--min N]` — **Pro**: a belief / conflict / provenance report
/// over every asserted Q-Page claim (belief, conflict, supporting & contradicting evidence) plus the
/// hash-chain integrity. Locked in the community tier (announced refusal; core untouched).
fn run_audit(opts: &AuditOpts) -> CliResult {
    use ccos_core::memory::EdgeType;
    let Some(file) = opts.snapshot.as_deref() else {
        return Err(CliError::usage(
            "usage: ccos audit <snapshot.json> [--json] [--min N]",
        ));
    };
    let Some(licensing) = gate_pro(ccos_core::license::Feature::AuditReports) else {
        println!("ccos audit: locked — run `ccos license` to see how to unlock (core unaffected).");
        return Ok(());
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => return Err(CliError::fail(format!("ccos: cannot load '{file}': {e}"))),
    };
    let graph = &snapshot.graph;
    let claims: Vec<_> = graph
        .claim_beliefs()
        .into_iter()
        .filter(|(_, q)| q.conflict >= opts.min)
        .collect();
    let el = snapshot.event_log.verify_integrity();
    let dl = snapshot.dist_log.verify_integrity();

    if opts.json {
        let rows: Vec<_> = claims
            .iter()
            .map(|(id, q)| {
                let supports: Vec<String> = graph
                    .evidence_of(id, EdgeType::Supports)
                    .iter()
                    .map(|n| n.0.clone())
                    .collect();
                let contradicts: Vec<String> = graph
                    .evidence_of(id, EdgeType::Contradicts)
                    .iter()
                    .map(|n| n.0.clone())
                    .collect();
                serde_json::json!({
                    "claim": id.0,
                    "belief": q.belief,
                    "conflict": q.conflict,
                    "support_sum": q.support,
                    "contradiction_sum": q.contradiction,
                    "supports": supports,
                    "contradicts": contradicts,
                })
            })
            .collect();
        let report = serde_json::json!({
            "licensee": licensing.licensee(),
            "claims": rows,
            "integrity": { "event_log_valid": el.valid, "dist_log_valid": dl.valid },
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS audit — belief / conflict / provenance ║");
    println!("╚══════════════════════════════════════════════╝\n");
    if let Some(who) = licensing.licensee() {
        println!("  licensee: {who}");
    }
    println!("  claims with assertions: {}\n", claims.len());
    for (id, q) in &claims {
        let marker = if q.belief > 0.5 {
            '✓'
        } else if q.belief < -0.5 {
            '✗'
        } else {
            '?'
        };
        println!("  {marker} {}", truncate(&id.0, 50));
        println!(
            "      belief {:+.3}   conflict {:.3}   (support {:.2} / contra {:.2})",
            q.belief, q.conflict, q.support, q.contradiction
        );
        for ev in graph.evidence_of(id, EdgeType::Supports) {
            println!("      + {}", truncate(&ev.0, 46));
        }
        for ev in graph.evidence_of(id, EdgeType::Contradicts) {
            println!("      - {}", truncate(&ev.0, 46));
        }
    }
    println!("\n  ── provenance / integrity ──");
    println!(
        "  event-log chain: {} events, valid: {}",
        snapshot.event_log.event_count(),
        el.valid
    );
    println!(
        "  dist-log chain:  {} links, valid: {}",
        dl.verified_events, dl.valid
    );
    Ok(())
}

/// Options for `ccos analyze`.
struct AnalyzeOpts {
    path: String,
    json: bool,
    cycles: bool,
    out: Option<String>,
    dot: Option<String>,
    max_nodes: usize,
    budget: usize,
}

impl AnalyzeOpts {
    fn parse(args: &[String]) -> Self {
        let mut opts = Self {
            path: ".".to_string(),
            json: false,
            cycles: false,
            out: None,
            dot: None,
            max_nodes: 5000,
            budget: 2048,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => opts.json = true,
                "--cycles" => opts.cycles = true,
                "--out" => {
                    i += 1;
                    opts.out = args.get(i).cloned();
                }
                "--dot" => {
                    i += 1;
                    opts.dot = args.get(i).cloned();
                }
                "--max-nodes" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.max_nodes = n;
                    }
                }
                "--budget" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.budget = n;
                    }
                }
                s if !s.starts_with("--") => opts.path = s.to_string(),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        opts
    }
}

/// `ccos analyze <path> [--json] [--cycles] [--out FILE]` — ingest every `.rs`
/// file under `path` into the causal memory graph and print (or export) a
/// structural report. Returns a process exit code (0 on success).
fn run_analyze(opts: &AnalyzeOpts) -> CliResult {
    let root = Path::new(&opts.path);
    if !root.exists() {
        return Err(CliError::fail(format!(
            "ccos: path '{}' does not exist",
            opts.path
        )));
    }

    let human = !opts.json;
    if human {
        println!("╔══════════════════════════════════════════════╗");
        println!("║  CCOS analyze — {:<29}║", truncate(&opts.path, 29));
        println!("╚══════════════════════════════════════════════╝\n");
    }

    let mut files: Vec<PathBuf> = Vec::new();
    if root.is_dir() {
        collect_rs_files(root, &mut files);
    } else if root.extension().and_then(|e| e.to_str()) == Some("rs") {
        files.push(root.to_path_buf());
    }
    files.sort();

    if files.is_empty() {
        return Err(CliError::fail(format!(
            "ccos: no .rs files found under '{}'",
            opts.path
        )));
    }

    let mut graph = MemoryGraph::new(MemoryGraph::paging_threshold_from_env(0.2), opts.max_nodes);
    // Honour scoring-weight (CCOS_W_*) and paging (CCOS_PAGING_THRESHOLD) overrides so the
    // validation harness can re-score the ingest under a trial's hyperparameters without recompiling.
    graph.set_scoring_weights(ScoringWeights::from_env());
    let mut engine = IncrementalGraphEngine::new();
    let mut event_log = EventLog::new(Uuid::new_v4().to_string());
    let mut dist_log = DistributedEventLog::new();

    let mut read_errors = 0usize;
    for file in &files {
        let source = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  [SKIP] {}: {}", file.display(), e);
                read_errors += 1;
                continue;
            }
        };
        let path_str = file.to_string_lossy().to_string();
        let file_hash = sha256_hex(&source);
        let delta = engine.process_delta(&path_str, None, &source, &mut graph);
        let (m, u, s) = engine
            .get_state(&path_str)
            .map(|st| (st.modules_count, st.uses_count, st.symbols_count))
            .unwrap_or((0, 0, 0));
        event_log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: path_str.clone(),
                file_hash: file_hash.clone(),
                modules_found: m,
                uses_found: u,
                symbols_found: s,
            },
        );
        dist_log.append(file_hash, "parser".into());
        if human {
            println!(
                "  [PARSE] {:<40} Δnodes:+{:<4} Δedges:+{}",
                truncate(&path_str, 40),
                delta.nodes_added,
                delta.edges_added
            );
        }
    }

    // Resolve intra-crate imports into file→file edges so failure propagation,
    // regions and the working set see the real cross-file causal structure.
    let cross_edges = graph.link_module_imports();

    // Integrity: the graph must never hold edges to evicted/absent nodes.
    let dangling = graph.prune_dangling_edges();
    let cycles = if opts.cycles || opts.json {
        graph.find_cycles()
    } else {
        Vec::new()
    };
    let orphans = graph.orphan_nodes().len();
    // Dead-symbol *candidates*: symbols nothing references (heuristic — pub API, `main`, and
    // trait-impl methods reached from outside the analyzed set are false positives).
    let dead = graph.dead_symbols();

    if let Some(dot_path) = &opts.dot {
        match std::fs::write(dot_path, graph.to_dot()) {
            Ok(()) => eprintln!("[DOT] graph written to {dot_path}"),
            Err(e) => eprintln!("ccos: failed to write DOT to {dot_path}: {e}"),
        }
    }

    if opts.json {
        let top: Vec<_> = graph
            .get_node_scores()
            .iter()
            .take(15)
            .map(|(id, s)| serde_json::json!({ "id": id.0, "score": s }))
            .collect();
        let types: Vec<_> = graph
            .node_type_counts()
            .into_iter()
            .map(|(t, c)| serde_json::json!({ "type": t, "count": c }))
            .collect();
        let report = serde_json::json!({
            "path": opts.path,
            "files_ingested": files.len() - read_errors,
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "cross_file_edges": cross_edges,
            "dangling_edges": dangling,
            "orphan_nodes": orphans,
            "dead_symbol_candidates": dead.iter().map(|id| &id.0).collect::<Vec<_>>(),
            "dependency_cycles": cycles.len(),
            "node_types": types,
            "top_nodes": top,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("\n─── Graph Summary ───");
        println!("  Files ingested:  {}", files.len() - read_errors);
        println!("  Graph nodes:     {}", graph.node_count());
        println!("  Graph edges:     {}", graph.edge_count());
        println!("  Cross-file edges:{cross_edges} (resolved imports)");
        println!("  Mutations:       {}", engine.total_mutations());
        println!("  Events logged:   {}", event_log.event_count());
        println!("  Dangling edges:  {dangling} (must be 0)");
        println!("  Orphan nodes:    {orphans}");
        println!(
            "  Dead symbols:    {} (unreferenced; heuristic)",
            dead.len()
        );

        if !dead.is_empty() {
            println!(
                "\n─── Potentially unreferenced symbols ({}) ───",
                dead.len()
            );
            for id in dead.iter().take(10) {
                println!("    {}", truncate(&id.0, 46));
            }
            if dead.len() > 10 {
                println!("    … (+{} more)", dead.len() - 10);
            }
        }

        println!("\n─── Node types ───");
        for (ty, count) in graph.node_type_counts() {
            println!("    {ty:<16} {count}");
        }

        if opts.cycles {
            println!("\n─── Dependency cycles ({}) ───", cycles.len());
            for cycle in cycles.iter().take(5) {
                let path: Vec<&str> = cycle.iter().map(|n| n.0.as_str()).collect();
                println!("    {} → {}", path.join(" → "), path.first().unwrap_or(&""));
            }
        }

        println!("\n─── Top 10 nodes by causal score ───");
        for (id, score) in graph.get_node_scores().iter().take(10) {
            println!("    {:<46} {:.4}", truncate(&id.0, 46), score);
        }

        let context = graph.select_context_window(opts.budget);
        println!(
            "\n─── Context window ({} tokens → {} nodes) ───",
            opts.budget,
            context.len()
        );
        for node in context.iter().take(10) {
            println!(
                "    {:<40} ({:?})",
                truncate(&node.label, 40),
                node.node_type
            );
        }
    }

    if let Some(out) = &opts.out {
        // Record the graph as replayable events so `ccos replay` can rebuild it
        // from the log alone (event-sourcing round-trip), then snapshot.
        event_log.record_graph(&graph);
        let snapshot = KernelSnapshot::new(graph, event_log, dist_log);
        match snapshot.save(out) {
            Ok(()) => eprintln!("\n[SAVE] snapshot written to {out}"),
            Err(e) => {
                return Err(CliError::fail(format!(
                    "ccos: failed to save snapshot to {out}: {e}"
                )));
            }
        }
    }

    if dangling != 0 {
        return Err(CliError::status(1));
    }
    Ok(())
}

/// `ccos sync <export|import|status|materialize>` — the file-based multi-agent
/// store: exchange chain-verified timeline bundles between agent workspaces and
/// materialize the convergent merged view. No network, no new dependency — a
/// bundle is a plain JSON file, so any transport (including sneakernet) works
/// and the air-gappable posture survives federation.
fn run_sync(rest: &[String]) -> CliResult {
    let usage = "usage: ccos sync export <workspace> --agent <id> --out <bundle.json> [--since N]\n       ccos sync import <workspace> <bundle.json>\n       ccos sync status <workspace>\n       ccos sync materialize <workspace> --out <merged.ccos>\n       ccos sync keygen <workspace>   (signed-sync builds: create the signing identity)";
    let sub = rest.first().map(String::as_str);
    let arg = |flag: &str| -> Option<String> {
        rest.iter()
            .position(|a| a == flag)
            .and_then(|i| rest.get(i + 1))
            .cloned()
    };
    let open = |ws: &str| {
        AgentSession::open(ws)
            .map_err(|e| CliError::fail(format!("ccos sync: cannot open workspace '{ws}': {e}")))
    };
    match sub {
        Some("export") => {
            let Some(ws) = rest.get(1).filter(|a| !a.starts_with("--")) else {
                return Err(CliError::usage(usage));
            };
            let Some(out) = arg("--out") else {
                return Err(CliError::usage(usage));
            };
            let since = arg("--since")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let mut s = open(ws)?;
            if let Some(id) = arg("--agent") {
                s.set_agent(id);
                s.checkpoint()
                    .map_err(|e| CliError::fail(format!("ccos sync: {e}")))?;
            }
            let bundle = s
                .export_bundle(since)
                .map_err(|e| CliError::fail(format!("ccos sync export: {e}")))?;
            let json = bundle
                .to_json()
                .map_err(|e| CliError::fail(format!("ccos sync export: {e}")))?;
            std::fs::write(&out, &json)
                .map_err(|e| CliError::fail(format!("ccos sync export: write '{out}': {e}")))?;
            println!(
                "✓ exported {} op(s) of agent '{}' (from step {}, {}) → {}",
                bundle.len(),
                bundle.agent,
                bundle.first_seq,
                if bundle.sig.is_some() {
                    "signed"
                } else {
                    "unsigned — run `ccos sync keygen` on a signed-sync build to sign"
                },
                out
            );
            Ok(())
        }
        Some("import") => {
            let (Some(ws), Some(file)) = (rest.get(1), rest.get(2)) else {
                return Err(CliError::usage(usage));
            };
            let raw = std::fs::read_to_string(file)
                .map_err(|e| CliError::fail(format!("ccos sync import: read '{file}': {e}")))?;
            let bundle = ccos_core::agent_session::SyncBundle::from_json(&raw)
                .map_err(|e| CliError::fail(format!("ccos sync import: parse '{file}': {e}")))?;
            let mut s = open(ws)?;
            let added = s
                .import_bundle(&bundle)
                .map_err(|e| CliError::fail(format!("ccos sync import: {e}")))?;
            s.checkpoint()
                .map_err(|e| CliError::fail(format!("ccos sync import: {e}")))?;
            println!(
                "✓ imported {added} new op(s) from agent '{}' (verified chain)",
                bundle.agent
            );
            Ok(())
        }
        Some("status") => {
            let Some(ws) = rest.get(1) else {
                return Err(CliError::usage(usage));
            };
            let s = open(ws)?;
            let own = if s.agent().is_empty() {
                "(unset — pass --agent on export)".to_string()
            } else {
                s.agent().to_string()
            };
            println!("agent:   {own}");
            println!(
                "own ops: {} (chain head {})",
                s.len(),
                truncate(&s.timeline_head(), 16)
            );
            if let Some(pk) = s.signing_pubkey() {
                println!("signing: enabled (pubkey {})", truncate(&pk, 16));
            }
            for (id, ops, head) in s.foreign_agents() {
                let pinned = match s.pinned_keys().get(&id) {
                    Some(k) => format!(", key pinned {}", truncate(k, 12)),
                    None => String::new(),
                };
                println!(
                    "foreign: {id} — {ops} op(s), head {}{pinned}",
                    truncate(&head, 16)
                );
            }
            let v = s.merged_view();
            let st = v.stats();
            println!(
                "merged view: {} nodes / {} edges / {} files",
                st.nodes, st.edges, st.files
            );
            Ok(())
        }
        Some("keygen") => {
            let Some(ws) = rest.get(1) else {
                return Err(CliError::usage(usage));
            };
            #[cfg(feature = "signed-sync")]
            {
                let pk = ccos_core::agent_session::generate_workspace_key(ws)
                    .map_err(|e| CliError::fail(format!("ccos sync keygen: {e}")))?;
                println!("✓ signing identity created for {ws}");
                println!("  public key: {pk}");
                println!("  (the seed lives in <workspace>.ccos.key — back it up, never share it)");
                Ok(())
            }
            #[cfg(not(feature = "signed-sync"))]
            {
                let _ = ws;
                Err(CliError::fail(
                    "ccos sync keygen: this build has no signature support — rebuild with --features signed-sync",
                ))
            }
        }
        Some("materialize") => {
            let Some(ws) = rest.get(1).filter(|a| !a.starts_with("--")) else {
                return Err(CliError::usage(usage));
            };
            let Some(out) = arg("--out") else {
                return Err(CliError::usage(usage));
            };
            let s = open(ws)?;
            let mut v = s.merged_view();
            v.checkpoint_to(&out)
                .map_err(|e| CliError::fail(format!("ccos sync materialize: {e}")))?;
            let st = v.stats();
            println!(
                "✓ merged view materialized → {out} ({} nodes / {} edges / {} files)",
                st.nodes, st.edges, st.files
            );
            Ok(())
        }
        _ => Err(CliError::usage(usage)),
    }
}

/// `ccos verify <snapshot.json | workspace>` — re-check a saved snapshot's
/// integrity (both hash chains must validate, no dangling edges), or — when the
/// argument is an agent workspace with a `.oplog` sidecar — audit the timeline's
/// tamper-evident chain without opening (or healing) the session.
fn run_verify(file: Option<&str>) -> CliResult {
    let Some(file) = file else {
        return Err(CliError::usage(
            "usage: ccos verify <snapshot.json | workspace>",
        ));
    };
    // Workspace mode: the argument (a `workspace.ccos` file or its directory)
    // has a timeline sidecar next to it.
    if let Some(audit) = ccos_core::agent_session::audit_workspace(Path::new(file)) {
        println!("╔══════════════════════════════════════════════╗");
        println!("║  CCOS verify — {:<30}║", truncate(file, 30));
        println!("╚══════════════════════════════════════════════╝\n");
        println!(
            "  Timeline ops:      {} live tail + {} compacted",
            audit.ops, audit.folded
        );
        if audit.legacy {
            println!("  Timeline chain:    pre-chain sidecar (established on next checkpoint)");
        } else {
            println!(
                "  Timeline chain:    {} link(s) verified | valid: {}",
                audit.integrity.verified_events, audit.integrity.valid
            );
            println!("  Chain head:        {}", truncate(&audit.head, 24));
        }
        for err in audit.integrity.errors.iter().take(10) {
            println!("    ! {err}");
        }
        return if audit.integrity.valid {
            println!("\n  ✓ timeline verified");
            Ok(())
        } else {
            println!("\n  ✗ verification FAILED (sidecar left untouched for forensics)");
            Err(CliError::status(1))
        };
    }
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        // Not a kernel snapshot. Before reporting a parse error, check whether it
        // is the *other* thing `verify` is legitimately pointed at: a workspace
        // written by `ccos memory`, which carries no timeline sidecar. Those files
        // are perfectly valid and fully verifiable — they just have no recorded
        // history — and reporting `missing field 'version'` at them said nothing a
        // caller could act on.
        Err(e) => return verify_memory_workspace(file, &e.to_string()),
    };

    let integrity = snapshot.dist_log.verify_integrity();
    let log_integrity = snapshot.event_log.verify_integrity();
    let mut graph = snapshot.graph.clone();
    let dangling = graph.prune_dangling_edges();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS verify — {:<30}║", truncate(file, 30));
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  Snapshot version:  {}", snapshot.version);
    println!(
        "  Graph nodes/edges: {}/{}",
        snapshot.graph.node_count(),
        snapshot.graph.edge_count()
    );
    println!("  Dangling edges:    {dangling} (must be 0)");
    println!("  Event-log events:  {}", snapshot.event_log.event_count());
    println!(
        "  Dist-log chain:    {} links | valid: {}",
        integrity.verified_events, integrity.valid
    );
    for err in integrity.errors.iter().take(10) {
        println!("    ! {err}");
    }
    println!(
        "  Event-log chain:   {} verified | valid: {}",
        log_integrity.verified_events, log_integrity.valid
    );
    for err in log_integrity.errors.iter().take(10) {
        println!("    ! {err}");
    }

    if integrity.valid && log_integrity.valid && dangling == 0 {
        println!("\n  ✓ snapshot verified");
        Ok(())
    } else {
        println!("\n  ✗ verification FAILED");
        Err(CliError::status(1))
    }
}

/// Verify a workspace that has no timeline sidecar — one written by
/// `ccos memory`, whose ops go straight to the snapshot.
///
/// `verify` knew two shapes, a workspace *with* a sidecar and a kernel snapshot,
/// and a `ccos memory` workspace is neither. It fell through to the snapshot
/// loader and surfaced its serde error verbatim — `cannot load '…': missing field
/// 'version'` — which reads as corruption for a file that is intact, fully
/// verifiable, and produced by this same binary two commands earlier.
///
/// `snapshot_err` is what the kernel-snapshot loader said, kept for the case where
/// the file really is neither thing.
fn verify_memory_workspace(file: &str, snapshot_err: &str) -> CliResult {
    use ccos_core::external_memory::{CcosMemory, ExternalMemory};

    let Ok(mem) = CcosMemory::open(file) else {
        return Err(CliError::fail(format!(
            "ccos: cannot load '{file}': {snapshot_err}"
        )));
    };
    let integrity = mem.verify();
    let stats = mem.stats();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS verify — {:<30}║", truncate(file, 30));
    println!("╚══════════════════════════════════════════════╝\n");
    println!(
        "  Graph nodes/edges: {}/{} ({} file(s), {} cold)",
        stats.nodes, stats.edges, stats.files, stats.cold
    );
    println!(
        "  Event-log chain:   {} verified | valid: {}",
        integrity.events, integrity.valid
    );
    for err in integrity.errors.iter().take(10) {
        println!("    ! {err}");
    }
    println!("  Timeline sidecar:  none");
    println!(
        "\n  This workspace was written through `ccos memory`, which persists the\n  \
         graph but records no op-log. Time-travel (`ccos postmortem`, `recall_what_if`)\n  \
         needs the sidecar, which `ccos mcp` and `ccos setup` create and maintain."
    );

    if integrity.valid {
        println!("\n  ✓ workspace verified (graph + event-log chain)");
        Ok(())
    } else {
        println!("\n  ✗ verification FAILED");
        Err(CliError::status(1))
    }
}

/// `ccos replay <snapshot.json>` — deterministically replay a saved event log
/// and print the reconstructed statistics, then re-verify the hash chain.
fn run_replay(file: Option<&str>) -> CliResult {
    let Some(file) = file else {
        return Err(CliError::usage("usage: ccos replay <snapshot.json>"));
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            return Err(CliError::fail(format!("ccos: cannot load '{file}': {e}")));
        }
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS replay — {:<30}║", truncate(file, 30));
    println!("╚══════════════════════════════════════════════╝\n");

    let mut replayer = EventReplayer::new();
    match snapshot.event_log.replay_deterministic(&mut replayer) {
        Ok(n) => {
            let s = &replayer.statistics;
            println!("  Replayed {n} events");
            println!(
                "  Stats: {} llm · {} parse · {} graph · {} guard · {} failures · {} cycles",
                s.llm_calls,
                s.parsing_events,
                s.graph_mutations,
                s.guard_checks,
                s.failures,
                s.cycles
            );
        }
        Err(e) => {
            return Err(CliError::fail(format!("ccos: replay error: {e}")));
        }
    }

    // Rebuild the graph purely from the log and check it matches the snapshot.
    let mut recon = GraphReconstructor::new();
    let _ = snapshot.event_log.replay_deterministic(&mut recon);
    if recon.nodes_built > 0 {
        let matches = recon.graph.node_count() == snapshot.graph.node_count()
            && recon.graph.edge_count() == snapshot.graph.edge_count();
        println!(
            "  Reconstructed graph: {} nodes / {} edges (matches snapshot: {})",
            recon.graph.node_count(),
            recon.graph.edge_count(),
            matches
        );
    }

    let integrity = snapshot.dist_log.verify_integrity();
    let log_integrity = snapshot.event_log.verify_integrity();
    println!(
        "  Hash-chain valid: {} (dist-log) · {} (event-log, {} links)",
        integrity.valid, log_integrity.valid, log_integrity.verified_events
    );
    if integrity.valid && log_integrity.valid {
        Ok(())
    } else {
        Err(CliError::status(1))
    }
}

/// `ccos diff <a.json> <b.json>` — structural difference between two saved
/// snapshots: nodes/edges added & removed, plus the biggest causal-score movers.
fn run_diff(a: Option<&str>, b: Option<&str>) -> CliResult {
    let (Some(a_path), Some(b_path)) = (a, b) else {
        return Err(CliError::usage(
            "usage: ccos diff <old-snapshot.json> <new-snapshot.json>",
        ));
    };
    let load = |p: &str| KernelSnapshot::load(p).map_err(|e| format!("cannot load '{p}': {e}"));
    let (snap_a, snap_b) = match (load(a_path), load(b_path)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            return Err(CliError::fail(format!("ccos: {e}")));
        }
    };

    let d = snap_a.graph.diff(&snap_b.graph);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS diff                                   ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  {}  →  {}", truncate(a_path, 18), truncate(b_path, 18));
    println!(
        "  Nodes:  +{} / -{}  ({} common)",
        d.nodes_added.len(),
        d.nodes_removed.len(),
        d.common_nodes
    );
    println!("  Edges:  +{} / -{}", d.edges_added, d.edges_removed);

    if !d.nodes_added.is_empty() {
        println!("\n  Added nodes:");
        for id in d.nodes_added.iter().take(10) {
            println!("    + {}", truncate(&id.0, 50));
        }
    }
    if !d.nodes_removed.is_empty() {
        println!("\n  Removed nodes:");
        for id in d.nodes_removed.iter().take(10) {
            println!("    - {}", truncate(&id.0, 50));
        }
    }

    // Causal-score drift among nodes present in both snapshots.
    let mut movers: Vec<(String, f64)> = Vec::new();
    for (id, node_b) in snap_b.graph.node_entries() {
        if let Some(node_a) = snap_a.graph.node(id) {
            let drift =
                snap_b.graph.compute_node_score(node_b) - snap_a.graph.compute_node_score(node_a);
            if drift.abs() > 1e-9 {
                movers.push((id.0.clone(), drift));
            }
        }
    }
    movers.sort_by(|x, y| {
        y.1.abs()
            .partial_cmp(&x.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(&y.0))
    });
    if !movers.is_empty() {
        println!("\n  Top causal-score movers:");
        for (id, drift) in movers.iter().take(10) {
            println!("    {:+.4}  {}", drift, truncate(id, 44));
        }
    }
    Ok(())
}

/// Options for `ccos failure`.
struct FailureOpts {
    snapshot: Option<String>,
    node: Option<String>,
    depth: u32,
    /// Re-page the graph to this node budget K after injection, exposing the
    /// surviving WorkingSet_K (the proxy-coverage measurement of the harness).
    max_nodes: Option<usize>,
    /// Emit a machine-readable working set instead of the human report.
    json: bool,
    /// Propagate failure pressure in both edge directions (reach upstream causes
    /// as well as downstream dependencies).
    bidirectional: bool,
}

impl FailureOpts {
    fn parse(args: &[String]) -> Self {
        let (mut snapshot, mut node, mut depth) = (None, None, 3u32);
        let mut max_nodes = None;
        let mut json = false;
        let mut bidirectional = false;
        let mut positional = 0;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--depth" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        depth = n;
                    }
                }
                "--max-nodes" => {
                    i += 1;
                    max_nodes = args.get(i).and_then(|v| v.parse().ok());
                }
                "--json" => json = true,
                "--bidirectional" => bidirectional = true,
                s if !s.starts_with("--") => {
                    if positional == 0 {
                        snapshot = Some(s.to_string());
                    } else {
                        node = Some(s.to_string());
                    }
                    positional += 1;
                }
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self {
            snapshot,
            node,
            depth,
            max_nodes,
            json,
            bidirectional,
        }
    }
}

/// `ccos failure <snapshot.json> <node-id> [--depth N] [--max-nodes K]
/// [--bidirectional] [--json]` — inject a fault at a node and propagate it across
/// the causal graph, reporting the affected neighborhood ranked by failure
/// relevance. `--bidirectional` also reaches upstream causes (callers/importers).
///
/// With `--max-nodes K` the graph is re-paged to the budget *after* injection,
/// so the survivors are the bounded **WorkingSet_K**; with `--json` that working
/// set is emitted as a machine-readable object. Together they are the Phase-1/2
/// hook the causal-validation harness drives: inject a mined fault, then measure
/// `R_cov = |F_target ∩ WorkingSet_K| / |F_target|`. Honours `CCOS_W_*` /
/// `CCOS_FAILURE_DECAY` so a hyperparameter trial re-scores without recompiling.
fn run_failure(opts: &FailureOpts) -> CliResult {
    let (Some(file), Some(node_id)) = (opts.snapshot.as_deref(), opts.node.as_deref()) else {
        return Err(CliError::usage(
            "usage: ccos failure <snapshot.json> <node-id> [--depth N] [--max-nodes K] [--bidirectional] [--json]",
        ));
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            return Err(CliError::fail(format!("ccos: cannot load '{file}': {e}")));
        }
    };
    let mut graph = snapshot.graph;
    // Re-score under any trial weights before injection/eviction.
    graph.set_scoring_weights(ScoringWeights::from_env());
    let origin = NodeId(node_id.to_string());
    if !graph.contains_node(&origin) {
        return Err(CliError::fail(format!(
            "ccos: node '{node_id}' not found ({} nodes). List ids with `ccos analyze <path> --json`.",
            graph.node_count()
        )));
    }

    let nodes_before = graph.node_count();
    graph.set_failure_relevance(&origin, 0.95);
    if opts.bidirectional {
        graph.propagate_failure_bidirectional(&origin, 0, opts.depth);
    } else {
        graph.propagate_failure(&origin, 0, opts.depth);
    }

    // Optionally constrain the working set to the top-K by score (failure
    // pressure has just lifted the causally-relevant subgraph, so eviction keeps
    // it preferentially). This is the WorkingSet_K the proxy metric scores.
    if let Some(k) = opts.max_nodes {
        graph.max_in_memory_nodes = k;
        graph.enforce_paging();
    }

    let mut affected: Vec<(String, f64)> = graph
        .node_entries()
        .filter(|(id, n)| **id != origin && n.failure_relevance > 0.0)
        .map(|(id, n)| (id.0.clone(), n.failure_relevance))
        .collect();
    affected.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    if opts.json {
        let mut working_set: Vec<&NodeId> = graph.node_ids().collect();
        working_set.sort();
        let w = graph.scoring_weights;
        let report = serde_json::json!({
            "origin": node_id,
            "severity": 0.95,
            "depth": opts.depth,
            "max_nodes": opts.max_nodes,
            "nodes_before": nodes_before,
            "working_set_size": working_set.len(),
            "working_set": working_set.iter().map(|id| &id.0).collect::<Vec<_>>(),
            "affected": affected
                .iter()
                .map(|(id, fr)| serde_json::json!({ "id": id, "failure_relevance": fr }))
                .collect::<Vec<_>>(),
            "weights": {
                "w_base": w.w_base,
                "w_failure": w.w_failure,
                "w_recency": w.w_recency,
                "w_access": w.w_access,
                "failure_decay": w.failure_decay,
            },
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS failure propagation                    ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  Origin:   {}", truncate(node_id, 40));
    println!("  Severity: 0.95   depth: {}", opts.depth);
    if let Some(k) = opts.max_nodes {
        println!(
            "  WorkingSet_K: {} survivors of {nodes_before} (K={k})",
            graph.node_count()
        );
    }
    println!("  Affected: {} nodes", affected.len());
    if !affected.is_empty() {
        println!("\n  Causal neighborhood (by failure relevance):");
        for (id, fr) in affected.iter().take(15) {
            println!("    {:.3}  {}", fr, truncate(id, 46));
        }
    }
    Ok(())
}

/// Options for `ccos chaos`.
struct ChaosOpts {
    iters: usize,
}

impl ChaosOpts {
    fn parse(args: &[String]) -> Self {
        let mut iters = 1000usize;
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--iters" {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    iters = n;
                }
            }
            i += 1;
        }
        Self { iters }
    }
}

/// `ccos chaos [--iters N]` — drive adversarial payloads (JSON corruption,
/// hallucination, prompt injection, timeouts) through the guard and assert its
/// core invariant: the guard must *never* emit non-JSON output.
fn run_chaos(opts: &ChaosOpts) -> CliResult {
    println!("╔══════════════════════════════════════════════╗");
    println!(
        "║  CCOS chaos — {:>5} iterations                ║",
        opts.iters
    );
    println!("╚══════════════════════════════════════════════╝\n");

    let guard = GuardLayer::new(GuardConfig::default());
    let modes = [
        AdversarialMode::JsonCorruption,
        AdversarialMode::Hallucination,
        AdversarialMode::PromptInjection,
        AdversarialMode::TimeoutSimulation,
    ];

    let (mut passed, mut blocked, mut invalid_outputs) = (0u64, 0u64, 0u64);
    for i in 0..opts.iters {
        let mut engine =
            AdversarialEngine::with_corruption_rate(modes[i % modes.len()].clone(), 0.9);
        let corrupted = engine.corrupt("{\"action\": \"analyze\", \"ok\": true}");
        let result = guard.validate_and_sanitize(&corrupted);
        if result.passed {
            passed += 1;
        } else {
            blocked += 1;
        }
        if serde_json::from_str::<serde_json::Value>(&result.sanitized_output).is_err() {
            invalid_outputs += 1;
        }
    }

    println!("  Iterations:            {}", opts.iters);
    println!("  Guard passed:          {passed}");
    println!("  Guard blocked:         {blocked}");
    println!("  Invalid guard outputs: {invalid_outputs} (must be 0)");

    if invalid_outputs == 0 {
        println!("\n  ✓ guard never emitted invalid JSON under chaos");
        Ok(())
    } else {
        println!("\n  ✗ guard emitted invalid JSON — safety invariant violated");
        Err(CliError::status(1))
    }
}

/// Ingest every `.rs` file under `path` into a fresh memory graph (the same way
/// `analyze` does, minus the event log and reporting). Shared by `top`.
fn build_graph_from_path(path: &str, max_nodes: usize) -> Result<MemoryGraph, String> {
    let root = Path::new(path);
    if !root.exists() {
        return Err(format!("path '{path}' does not exist"));
    }
    let mut files: Vec<PathBuf> = Vec::new();
    if root.is_dir() {
        collect_rs_files(root, &mut files);
    } else if root.extension().and_then(|e| e.to_str()) == Some("rs") {
        files.push(root.to_path_buf());
    }
    files.sort();
    if files.is_empty() {
        return Err(format!("no .rs files found under '{path}'"));
    }

    let mut graph = MemoryGraph::new(MemoryGraph::paging_threshold_from_env(0.2), max_nodes);
    let mut engine = IncrementalGraphEngine::new();
    for file in &files {
        if let Ok(source) = std::fs::read_to_string(file) {
            let path_str = file.to_string_lossy().to_string();
            engine.process_delta(&path_str, None, &source, &mut graph);
        }
    }
    graph.prune_dangling_edges();
    Ok(graph)
}

/// Options for `ccos top`.
struct TopOpts {
    path: String,
    limit: usize,
    json: bool,
    max_nodes: usize,
}

impl TopOpts {
    fn parse(args: &[String]) -> Self {
        let mut opts = Self {
            path: ".".to_string(),
            limit: 20,
            json: false,
            max_nodes: 5000,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => opts.json = true,
                "--limit" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.limit = n;
                    }
                }
                "--max-nodes" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.max_nodes = n;
                    }
                }
                s if !s.starts_with("--") => opts.path = s.to_string(),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        opts
    }
}

/// `ccos top <path> [--limit N] [--json]` — ingest `path` and print the hottest
/// nodes by causal score: the working set the kernel would page in first.
fn run_top(opts: &TopOpts) -> CliResult {
    let graph = match build_graph_from_path(&opts.path, opts.max_nodes) {
        Ok(g) => g,
        Err(e) => {
            return Err(CliError::fail(format!("ccos: {e}")));
        }
    };
    let hot = query::hot_set(&graph, opts.limit);

    if opts.json {
        let rows: Vec<_> = hot
            .iter()
            .map(|(id, s)| serde_json::json!({ "id": id.0, "score": s }))
            .collect();
        let report = serde_json::json!({
            "path": opts.path,
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "top": rows,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS top — {:<33}║", truncate(&opts.path, 33));
    println!("╚══════════════════════════════════════════════╝\n");
    println!(
        "  {} nodes / {} edges — top {}:\n",
        graph.node_count(),
        graph.edge_count(),
        hot.len()
    );
    println!("    {:>7}  {:<8}  NODE", "SCORE", "TYPE");
    for (id, score) in &hot {
        let ty = graph
            .node(id)
            .map(|n| format!("{:?}", n.node_type))
            .unwrap_or_else(|| "?".into());
        println!(
            "    {:>7.4}  {:<8}  {}",
            score,
            truncate(&ty, 8),
            truncate(&id.0, 44)
        );
    }
    Ok(())
}

/// Options for `ccos blame`.
struct BlameOpts {
    snapshot: Option<String>,
    node: Option<String>,
    depth: u32,
    json: bool,
}

impl BlameOpts {
    fn parse(args: &[String]) -> Self {
        let (mut snapshot, mut node, mut depth, mut json) = (None, None, 3u32, false);
        let mut positional = 0;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => json = true,
                "--depth" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        depth = n;
                    }
                }
                s if !s.starts_with("--") => {
                    if positional == 0 {
                        snapshot = Some(s.to_string());
                    } else {
                        node = Some(s.to_string());
                    }
                    positional += 1;
                }
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self {
            snapshot,
            node,
            depth,
            json,
        }
    }
}

/// `ccos blame <snapshot.json> <node-id> [--depth N]` — show a node's upstream
/// causes (what it rests on) and downstream blast radius (what breaks with it).
fn run_blame(opts: &BlameOpts) -> CliResult {
    let (Some(file), Some(node_id)) = (opts.snapshot.as_deref(), opts.node.as_deref()) else {
        return Err(CliError::usage(
            "usage: ccos blame <snapshot.json> <node-id> [--depth N]",
        ));
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            return Err(CliError::fail(format!("ccos: cannot load '{file}': {e}")));
        }
    };
    let graph = snapshot.graph;
    let origin = NodeId(node_id.to_string());
    if !graph.contains_node(&origin) {
        return Err(CliError::fail(format!(
            "ccos: node '{node_id}' not found ({} nodes). List ids with `ccos analyze <path> --json`.",
            graph.node_count()
        )));
    }

    let causes = query::source_set(&graph, &origin, opts.depth);
    let impact = query::impact_set(&graph, &origin, opts.depth);

    if opts.json {
        let to_rows = |v: &[query::Reached]| {
            v.iter()
                .map(|r| {
                    serde_json::json!({ "id": r.id.0, "distance": r.distance, "score": r.score })
                })
                .collect::<Vec<_>>()
        };
        let report = serde_json::json!({
            "node": node_id,
            "depth": opts.depth,
            "causes": to_rows(&causes),
            "impact": to_rows(&impact),
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS blame                                  ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  Node:  {}", truncate(node_id, 40));
    println!("  Depth: {}\n", opts.depth);

    println!(
        "  ── Causes (upstream — what it rests on): {} ──",
        causes.len()
    );
    for r in causes.iter().take(15) {
        println!(
            "    d{}  {:.4}  {}",
            r.distance,
            r.score,
            truncate(&r.id.0, 42)
        );
    }
    println!(
        "\n  ── Blast radius (downstream — what breaks with it): {} ──",
        impact.len()
    );
    for r in impact.iter().take(15) {
        println!(
            "    d{}  {:.4}  {}",
            r.distance,
            r.score,
            truncate(&r.id.0, 42)
        );
    }
    Ok(())
}

/// Options for `ccos export`.
struct ExportOpts {
    snapshot: Option<String>,
    out: String,
}

impl ExportOpts {
    fn parse(args: &[String]) -> Self {
        let (mut snapshot, mut out) = (None, "ccos.graphml".to_string());
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--out" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        out = v.clone();
                    }
                }
                // `--format graphml` is accepted for forward-compatibility; GraphML
                // is currently the only target.
                "--format" => {
                    i += 1;
                    if let Some(fmt) = args.get(i) {
                        if fmt != "graphml" {
                            eprintln!("ccos: unknown export format '{fmt}', using graphml");
                        }
                    }
                }
                s if !s.starts_with("--") => snapshot = Some(s.to_string()),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self { snapshot, out }
    }
}

/// `ccos export <snapshot.json> [--out FILE]` — export the snapshot's causal
/// graph as GraphML for Gephi / yEd / Cytoscape / networkx.
fn run_export(opts: &ExportOpts) -> CliResult {
    let Some(file) = opts.snapshot.as_deref() else {
        return Err(CliError::usage(
            "usage: ccos export <snapshot.json> [--out FILE]",
        ));
    };
    let snapshot = match KernelSnapshot::load(file) {
        Ok(s) => s,
        Err(e) => {
            return Err(CliError::fail(format!("ccos: cannot load '{file}': {e}")));
        }
    };
    let graphml = query::to_graphml(&snapshot.graph);
    match std::fs::write(&opts.out, graphml) {
        Ok(()) => {
            println!(
                "[EXPORT] {} nodes / {} edges → {} (GraphML)",
                snapshot.graph.node_count(),
                snapshot.graph.edge_count(),
                opts.out
            );
            Ok(())
        }
        Err(e) => Err(CliError::fail(format!(
            "ccos: failed to write '{}': {e}",
            opts.out
        ))),
    }
}

/// Options for `ccos regions`.
struct RegionsOpts {
    path: String,
    json: bool,
    activate: Option<String>,
    metrics: Option<String>,
    radius: u32,
    max_nodes: usize,
}

impl RegionsOpts {
    fn parse(args: &[String]) -> Self {
        let mut opts = Self {
            path: ".".to_string(),
            json: false,
            activate: None,
            metrics: None,
            radius: 2,
            max_nodes: 5000,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => opts.json = true,
                "--activate" => {
                    i += 1;
                    opts.activate = args.get(i).cloned();
                }
                "--metrics" => {
                    i += 1;
                    opts.metrics = args.get(i).cloned();
                }
                "--radius" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.radius = n;
                    }
                }
                "--max-nodes" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        opts.max_nodes = n;
                    }
                }
                s if !s.starts_with("--") => opts.path = s.to_string(),
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        opts
    }
}

/// `ccos regions <path> [--activate ID] [--metrics ID] [--radius N] [--json]` —
/// cluster the causal graph into spatial regions; optionally activate one
/// (hydrate a context window) or print the flat-vs-region locality comparison.
fn run_regions(opts: &RegionsOpts) -> CliResult {
    let graph = match build_graph_from_path(&opts.path, opts.max_nodes) {
        Ok(g) => g,
        Err(e) => {
            return Err(CliError::fail(format!("ccos: {e}")));
        }
    };
    let mut engine = ContextRegionEngine::new();
    let mut log = EventLog::new(Uuid::new_v4().to_string());
    engine.initialize_regions(&graph, &mut log);

    // ── metrics mode: flat vs region locality for a target node ──
    if let Some(target) = &opts.metrics {
        let Some(report) = region_metrics::locality_report(&graph, target, opts.radius) else {
            return Err(CliError::fail(format!(
                "ccos: node '{target}' not found in graph"
            )));
        };
        if opts.json {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return Ok(());
        }
        println!("╔══════════════════════════════════════════════╗");
        println!("║  CCOS regions — locality metrics             ║");
        println!("╚══════════════════════════════════════════════╝\n");
        println!("  Target:            {}", truncate(target, 40));
        println!(
            "  Causal nbhd (r={}): {} nodes",
            report.radius, report.neighborhood_size
        );
        println!(
            "  flat   : precision {:.2}  recall {:.2}  ({} nodes)",
            report.flat.causal_precision, report.flat.causal_recall, report.flat.nodes_selected
        );
        println!(
            "  region : precision {:.2}  recall {:.2}  ({} nodes)",
            report.region.causal_precision,
            report.region.causal_recall,
            report.region.nodes_selected
        );
        println!("  Precision gain:    {:+.2}", report.precision_gain);
        println!(
            "  Tokens to cover Nk: flat {} vs region {}  (saving {:+.0}%)",
            report.flat_tokens_to_cover,
            report.region_tokens_to_cover,
            report.token_saving_ratio * 100.0
        );
        return Ok(());
    }

    // ── activate mode: hydrate a context window from a region ──
    if let Some(target) = &opts.activate {
        let policy = ContextPolicy::default();
        let Some(win) = engine.activate_region(
            &graph,
            &RegionQuery::Node(target.clone()),
            &policy,
            &mut log,
        ) else {
            return Err(CliError::fail(format!(
                "ccos: node '{target}' not found in any region"
            )));
        };
        if opts.json {
            let report = serde_json::json!({
                "region": win.region,
                "files": win.files,
                "tokens_estimated": win.tokens_estimated,
                "region_score": win.region_score,
                "reason": win.reason,
            });
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return Ok(());
        }
        println!("╔══════════════════════════════════════════════╗");
        println!("║  CCOS regions — context window               ║");
        println!("╚══════════════════════════════════════════════╝\n");
        println!("  Region:  {}", truncate(&win.region, 38));
        println!("  Score:   {:.3}", win.region_score);
        println!("  Tokens:  ~{}", win.tokens_estimated);
        println!("  Reason:  {}", win.reason);
        println!("\n  Files ({}):", win.files.len());
        for f in win.files.iter().take(20) {
            println!("    • {}", truncate(f, 44));
        }
        return Ok(());
    }

    // ── default: region map summary ──
    let mut regions: Vec<_> = engine.regions.values().collect();
    regions.sort_by(|a, b| {
        b.temperature
            .partial_cmp(&a.temperature)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    if opts.json {
        let rows: Vec<_> = regions
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "center": r.center,
                    "members": r.member_count(),
                    "temperature": r.temperature,
                    "causal_density": r.causal_density,
                })
            })
            .collect();
        let report = serde_json::json!({
            "path": opts.path,
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "regions": engine.region_count(),
            "map": rows,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS regions — {:<29}║", truncate(&opts.path, 29));
    println!("╚══════════════════════════════════════════════╝\n");
    println!(
        "  {} nodes / {} edges → {} regions\n",
        graph.node_count(),
        graph.edge_count(),
        engine.region_count()
    );
    println!("    {:>5}  {:>7}  {:>7}  REGION", "MEMB", "TEMP", "DENS");
    for r in regions.iter().take(20) {
        println!(
            "    {:>5}  {:>7.4}  {:>7.3}  {}",
            r.member_count(),
            r.temperature,
            r.causal_density,
            truncate(&r.id, 40)
        );
    }
    Ok(())
}

/// `ccos experiment [--tasks N] [--seed S] [--budget B] [--json]` — run the
/// LLM-free hypothesis simulation: regional causal memory vs. RAG / GraphRAG
/// baselines on synthetic multi-file causal tasks of growing diameter.
fn run_experiment_cmd(args: &[String]) -> CliResult {
    let mut cfg = ExperimentConfig::default();
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json = true,
            "--tasks" => {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    cfg.tasks = n;
                }
            }
            "--seed" => {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    cfg.seed = n;
                }
            }
            "--budget" => {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    cfg.budget = n;
                }
            }
            other => eprintln!("ccos: ignoring unknown flag '{other}'"),
        }
        i += 1;
    }

    // Run both scenarios: clean (query points at the target) and noisy (a trap
    // decoy out-scores the target lexically).
    let clean = run_experiment(&ExperimentConfig {
        noisy: false,
        ..cfg.clone()
    });
    let noisy = run_experiment(&ExperimentConfig {
        noisy: true,
        ..cfg.clone()
    });

    if json {
        let out = serde_json::json!({ "clean": clean, "noisy": noisy });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    let strategies = [
        "rag-dense",
        "rag-hybrid",
        "graphrag-1hop",
        "graphrag-bfs",
        "ccos-from-query",
        "ccos-region",
    ];
    let print_table = |report: &ccos_core::experiment::ExperimentReport, title: &str| {
        println!("  ── {title} ──");
        println!(
            "    {:<16} {:>6} {:>6} {:>6} {:>6}",
            "strategy", "d=1", "d=2", "d=3", "d=4"
        );
        for strat in strategies {
            let cell = |d: u32| -> String {
                report
                    .per_diameter
                    .iter()
                    .find(|(dd, _)| *dd == d)
                    .and_then(|(_, row)| row.iter().find(|r| r.strategy == strat))
                    .map(|r| format!("{:.2}", r.success_rate))
                    .unwrap_or_else(|| "  – ".into())
            };
            println!(
                "    {:<16} {:>6} {:>6} {:>6} {:>6}",
                strat,
                cell(1),
                cell(2),
                cell(3),
                cell(4)
            );
        }
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS experiment — regional memory vs RAG    ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!(
        "  seed={}  tasks={}  budget={} tokens   (success = required causal set ⊆ window)\n",
        clean.seed, clean.n_tasks, clean.budget_tokens
    );
    print_table(&clean, "CLEAN query (points at the target)");
    println!();
    print_table(
        &noisy,
        "NOISY query (a decoy out-scores the target lexically)",
    );
    println!(
        "\n  Reading: lexical RAG fails on cross-file tasks; structure-aware methods\n  \
         (graph-BFS, CCOS) tie when the query is clean — but under a misleading query\n  \
         only `ccos-region`, which anchors on the workspace signal (not the query),\n  \
         survives. The differentiator is the anchor, not the region machinery."
    );
    Ok(())
}

/// `ccos trace` — read `cargo test` / panic / backtrace text on **stdin** and
/// emit the project source locations the crash touched as JSON (`message`,
/// `files`, `hits`). The seed set for a trace-driven context page fault.
fn run_trace_cmd() -> CliResult {
    use std::io::Read;
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        return Err(CliError::fail("ccos: failed to read stdin"));
    }
    let trace = parse_cargo_test_output(&input);
    let hits: Vec<_> = trace
        .hits
        .iter()
        .map(
            |h| serde_json::json!({ "file": h.file, "line": h.line, "frame_depth": h.frame_depth }),
        )
        .collect();
    let report = serde_json::json!({
        "message": trace.message,
        "files": trace.files(),
        "hits": hits,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}

/// Options for `ccos focus` — the human "attentional shield".
struct FocusOpts {
    path: String,
    budget: usize,
    json: bool,
    input: Option<String>,
    /// Reuse/persist a workspace checkpoint so only *changed* files are re-parsed
    /// (O(Δ) freshness for an editor calling `focus` on every run). `--workspace`
    /// with no path defaults to `workspace.ccos`.
    workspace: Option<String>,
}

impl FocusOpts {
    fn parse(args: &[String]) -> Self {
        let mut path = None;
        let mut budget = 2048usize;
        let mut json = false;
        let mut input = None;
        let mut workspace = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--budget" => {
                    i += 1;
                    if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                        budget = n;
                    }
                }
                "--input" => {
                    i += 1;
                    input = args.get(i).cloned();
                }
                "--workspace" => {
                    // Optional path; default when the next token is another flag/absent.
                    let p = match args.get(i + 1) {
                        Some(v) if !v.starts_with("--") => {
                            i += 1;
                            v.clone()
                        }
                        _ => "workspace.ccos".to_string(),
                    };
                    workspace = Some(p);
                }
                "--json" => json = true,
                s if !s.starts_with("--") => {
                    if path.is_none() {
                        path = Some(s.to_string());
                    }
                }
                other => eprintln!("ccos: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        Self {
            path: path.unwrap_or_else(|| "src".to_string()),
            budget,
            json,
            input,
            workspace,
        }
    }
}

/// A file's role in the focused view, relative to the failing trace.
#[derive(Debug, PartialEq, Eq)]
enum FocusRole {
    /// A file the trace itself names — where the failure *manifests*.
    Symptom,
    /// The top file pulled in *causally* (not in the trace) — the likely root.
    Cause,
    /// Another file in the causal region.
    Context,
}

/// One file in the focused view: the highest-scored window node from that file.
struct FocusEntry {
    file: String,
    content: String,
    score: f64,
    role: FocusRole,
}

/// Reduce a recall window to one entry per file (highest score first), tagging the
/// trace's own files as the *symptom* and the top causally-pulled file as the likely
/// *cause* — the "skip to the root" signal a raw backtrace buries. Pure + testable.
fn focus_view(window: &RecallWindow, trace: &ExecutionTrace) -> Vec<FocusEntry> {
    let symptom_files: std::collections::BTreeSet<String> = trace.files().into_iter().collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut out: Vec<FocusEntry> = Vec::new();
    let mut cause_assigned = false;
    for it in &window.items {
        let file = file_of(&it.uri).to_string();
        if file.is_empty() || !seen.insert(file.clone()) {
            continue;
        }
        let role = if symptom_files.contains(&file) {
            FocusRole::Symptom
        } else if !cause_assigned {
            cause_assigned = true;
            FocusRole::Cause
        } else {
            FocusRole::Context
        };
        out.push(FocusEntry {
            file,
            content: it.content.clone(),
            score: it.score,
            role,
        });
    }
    out
}

/// Crate-relative path in the form `cargo` reports (`src/…`): the tail from the last
/// `src/` segment, so an absolute ingest path still matches a crate-relative trace path.
fn crate_relative(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    match s.rfind("src/") {
        Some(i) => s[i..].to_string(),
        None => s,
    }
}

/// `ccos focus [src] [--budget N] [--input FILE] [--json]` — the attentional shield.
/// Pipe `cargo test` / panic output in; CCOS ingests the tree, page-faults on the
/// trace, and shows the **causal region** (the likely root cause + its direct
/// dependencies), hiding the backtrace noise and the unrelated files. The host can
/// be a human (terminal) or an editor (`--json`).
fn run_focus(opts: &FocusOpts) -> CliResult {
    let root = Path::new(&opts.path);
    if !root.exists() {
        return Err(CliError::fail(format!(
            "ccos: path '{}' does not exist",
            opts.path
        )));
    }
    let mut files: Vec<PathBuf> = Vec::new();
    if root.is_dir() {
        collect_rs_files(root, &mut files);
    } else if root.extension().and_then(|e| e.to_str()) == Some("rs") {
        files.push(root.to_path_buf());
    }
    files.sort();
    if files.is_empty() {
        return Err(CliError::fail(format!(
            "ccos: no .rs files under '{}'",
            opts.path
        )));
    }

    // Ingest under crate-relative URIs (`src/…`), matching how `cargo` reports paths
    // in the trace — so a fault on `src/writer.rs` anchors regardless of whether the
    // user passed `src` or an absolute path. With `--workspace`, reuse the persisted
    // checkpoint and `sync` (re-parse only changed files — O(Δ) for an editor); without
    // it, a fresh ephemeral session ingesting the whole tree.
    let mut session = match &opts.workspace {
        Some(ws) => match AgentSession::open(ws) {
            Ok(s) => s,
            Err(e) => {
                return Err(CliError::fail(format!(
                    "ccos: cannot open workspace '{ws}': {e}"
                )));
            }
        },
        None => AgentSession::new(),
    };
    let mut reparsed = 0usize;
    for f in &files {
        if let Ok(src) = std::fs::read_to_string(f) {
            let uri = crate_relative(f);
            if opts.workspace.is_some() {
                if session.sync(&uri, &src) {
                    reparsed += 1;
                }
            } else {
                session.ingest(&uri, &src);
            }
        }
    }

    let output = match &opts.input {
        Some(p) => std::fs::read_to_string(p).unwrap_or_default(),
        None => {
            use std::io::Read;
            let mut s = String::new();
            let _ = std::io::stdin().read_to_string(&mut s);
            s
        }
    };

    let trace = parse_cargo_test_output(&output);
    let window = session.page_fault(&output, opts.budget);
    let view = focus_view(&window, &trace);

    // Persist the synced graph + this page-fault so the next `--workspace` run is O(Δ).
    if opts.workspace.is_some() {
        if let Err(e) = session.checkpoint() {
            eprintln!("ccos focus: checkpoint failed: {e}");
        }
    }

    if opts.json {
        let entries: Vec<_> = view
            .iter()
            .map(|e| {
                serde_json::json!({
                    "file": e.file,
                    "role": format!("{:?}", e.role).to_lowercase(),
                    "score": e.score,
                    "content": e.content,
                })
            })
            .collect();
        let report = serde_json::json!({
            "message": trace.message,
            "symptom_files": trace.files(),
            "workspace_files": files.len(),
            "reparsed_files": reparsed,
            "tokens": window.tokens,
            "entries": entries,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(());
    }

    render_focus_human(
        &trace,
        &view,
        files.len(),
        window.tokens,
        opts.workspace.as_deref(),
        reparsed,
    );
    Ok(())
}

/// Render the focused view for a human terminal — the cause first, noise hidden.
fn render_focus_human(
    trace: &ExecutionTrace,
    view: &[FocusEntry],
    total_files: usize,
    tokens: usize,
    workspace: Option<&str>,
    reparsed: usize,
) {
    let delta = match workspace {
        Some(_) => format!(", {reparsed} re-parsed (Δ)"),
        None => String::new(),
    };
    println!(
        "⚡ CCOS focus — {} files in workspace → {} in view (~{} tokens{})\n",
        total_files,
        view.len(),
        tokens,
        delta
    );
    if !trace.message.is_empty() {
        println!("  panicked: {}", truncate(trace.message.trim(), 76));
    }
    if let Some(h) = trace.hits.first() {
        println!("  symptom:  {}:{}", h.file, h.line);
    }
    println!();

    for e in view {
        let tag = match e.role {
            FocusRole::Cause => "◀ likely cause (pulled in causally)",
            FocusRole::Symptom => "· symptom site",
            FocusRole::Context => "· related",
        };
        println!("  ▸ {}   {tag}   [{:.2}]", e.file, e.score);
        for line in e.content.lines().take(6) {
            println!("      {line}");
        }
        if e.content.lines().count() > 6 {
            println!("      …");
        }
        println!();
    }

    let hidden = total_files.saturating_sub(view.len());
    if hidden > 0 {
        println!("  hidden: {hidden} unrelated file(s) + the rest of the backtrace");
    }
}

/// `ccos memory [--path FILE]` — drive the [`CcosMemory`] external-memory façade
/// over **stdin JSON Lines**: one request object per line, one JSON response per
/// line. Loads `FILE` (default `workspace.ccos`), applies each request, and
/// checkpoints back if any mutation occurred — scriptable from any language.
///
/// Requests: `{"op":"ingest","uri":..,"source":..}`,
/// `{"op":"failure","node":..,"depth":N}`,
/// `{"op":"recall","strategy":"around|task|working_set",..,"budget":N}`,
/// `{"op":"impact|causes","node":..,"depth":N}`, `{"op":"verify"}`,
/// `{"op":"stats"}`.
fn run_memory_cmd(args: &[String]) -> CliResult {
    let mut path = "workspace.ccos".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    path = p.clone();
                }
            }
            other => eprintln!("ccos: ignoring unknown flag '{other}'"),
        }
        i += 1;
    }

    let mut mem = match CcosMemory::open(&path) {
        Ok(m) => m,
        Err(e) => {
            return Err(CliError::fail(format!(
                "ccos: cannot open memory '{path}': {e}"
            )));
        }
    };

    // Process the op-stream against the persisted workspace, then checkpoint if it mutated.
    let (dirty, had_error) = run_op_stream(&mut mem);
    if dirty {
        if let Err(e) = mem.checkpoint() {
            return Err(CliError::fail(format!("ccos: checkpoint failed: {e}")));
        }
    }
    if had_error {
        Err(CliError::status(1))
    } else {
        Ok(())
    }
}

/// `ccos stdin` — read the same newline-delimited JSON op-stream as `ccos memory`, but process it in an
/// **ephemeral in-memory graph** (no workspace file, no checkpoint): the pipe-friendly, persistence-free
/// sibling. Prints one JSON response per op and exits non-zero if any op errored.
fn run_stdin_cmd(_args: &[String]) -> CliResult {
    let mut mem = CcosMemory::new();
    let (_dirty, had_error) = run_op_stream(&mut mem);
    if had_error {
        Err(CliError::status(1))
    } else {
        Ok(())
    }
}

/// Read a newline-delimited JSON op-stream from stdin and apply it to `mem`, printing one JSON response
/// per op (`ingest` / `failure` / `recall` / `impact` / `causes` / `verify` / `stats`). Returns
/// `(dirty, had_error)` — whether any op mutated the memory, and whether any op failed. Shared by
/// [`run_memory_cmd`] (persistent workspace) and [`run_stdin_cmd`] (in-memory).
fn run_op_stream(mem: &mut CcosMemory) -> (bool, bool) {
    use std::io::BufRead;
    let err = |msg: String| serde_json::json!({ "error": msg });
    let mut dirty = false;
    let mut had_error = false;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                println!("{}", err(format!("invalid JSON: {e}")));
                had_error = true;
                continue;
            }
        };
        let s = |k: &str| req[k].as_str().unwrap_or("").to_string();
        let op = req["op"].as_str().unwrap_or("").to_string();
        let resp: serde_json::Value = match op.as_str() {
            "ingest" => {
                let (uri, src) = (s("uri"), s("source"));
                // The message has always claimed both are required; only `uri` was
                // checked, so `{"op":"ingest","uri":"file:x.rs"}` succeeded and put
                // an empty node in the graph — a file that exists, in the memory,
                // with no content, which nothing downstream can tell from a real
                // empty file. `source: ""` stays legal: an empty file is a real
                // thing. What is refused is the key not being there at all.
                if uri.is_empty() {
                    had_error = true;
                    err("ingest requires 'uri' and 'source'".into())
                } else if req.get("source").is_none() {
                    had_error = true;
                    err("ingest requires 'source' (pass \"\" for an empty file)".into())
                } else {
                    dirty = true;
                    serde_json::to_value(mem.ingest_source(&uri, &src)).unwrap()
                }
            }
            "failure" => {
                let depth = req["depth"].as_u64().unwrap_or(3) as u32;
                match mem.signal_failure(&s("node"), depth) {
                    Ok(n) => {
                        dirty = true;
                        serde_json::json!({ "affected": n })
                    }
                    Err(e) => {
                        had_error = true;
                        err(e.to_string())
                    }
                }
            }
            "recall" => {
                let budget = req["budget"].as_u64().unwrap_or(2048) as usize;
                let strategy = req["strategy"].as_str().unwrap_or("working_set");
                // An anchored or free-text strategy without its query is a malformed
                // request, not a memory that holds nothing. Answering it with a
                // well-formed empty window made a caller's typo indistinguishable
                // from an empty workspace: `{"strategy":"task","task":"NdLinear"}`
                // — the field is `text` — returned `{"items":[],"tokens":0}` with no
                // hint, on a workspace where the same query under the right key
                // returns 15 items.
                let needed = match strategy {
                    "around" if s("anchor").trim().is_empty() => Some("anchor"),
                    "task" | "semantic" | "hybrid" if s("text").trim().is_empty() => Some("text"),
                    _ => None,
                };
                if let Some(field) = needed {
                    had_error = true;
                    println!(
                        "{}",
                        err(format!(
                            "recall strategy '{strategy}' requires a non-empty '{field}'"
                        ))
                    );
                    continue;
                }
                let recall = match strategy {
                    "around" => Recall::around(s("anchor")),
                    "task" => Recall::task(s("text")),
                    "semantic" => Recall::semantic(s("text")),
                    "hybrid" => Recall::hybrid(s("text")),
                    _ => Recall::working_set(),
                };
                // Page fault on the read path, exactly as `AgentSession::recall` does:
                // an `around` recall whose anchor was demoted to the COLD tier pages it
                // (and its cold neighbours) back first. Without this the façade returns
                // an empty window for any anchor past the resident cap, while the same
                // anchor recalls fine over MCP — the cold tier must be transparent on
                // both paths. `ensure_resident` adds no event, so it never sets `dirty`.
                if let Recall::Around(uri) = &recall {
                    mem.ensure_resident(uri);
                }
                serde_json::to_value(mem.recall(&recall, budget)).unwrap()
            }
            "impact" | "causes" => {
                let depth = req["depth"].as_u64().unwrap_or(2) as u32;
                // Without a node these answered `{"reached":[]}` — a well-formed
                // "nothing depends on it" for a question nobody asked. `failure`,
                // which takes the same argument, already refuses; these two were
                // the odd ones out.
                if s("node").trim().is_empty() {
                    had_error = true;
                    println!("{}", err(format!("{op} requires 'node'")));
                    continue;
                }
                let reached = if op == "impact" {
                    mem.impact(&s("node"), depth)
                } else {
                    mem.causes(&s("node"), depth)
                };
                let arr: Vec<_> = reached
                    .iter()
                    .map(|r| {
                        serde_json::json!({ "id": r.id.0, "distance": r.distance, "score": r.score })
                    })
                    .collect();
                serde_json::json!({ "reached": arr })
            }
            "verify" => serde_json::to_value(mem.verify()).unwrap(),
            "stats" => serde_json::to_value(mem.stats()).unwrap(),
            "" => {
                had_error = true;
                err("missing 'op'".into())
            }
            other => {
                had_error = true;
                err(format!("unknown op '{other}'"))
            }
        };
        println!("{}", serde_json::to_string(&resp).unwrap());
    }
    (dirty, had_error)
}

/// `ccos postmortem [workspace.ccos] [--json]` — open the interactive **time-travel
/// debugger** over an agent session's recorded timeline. With a workspace path it
/// loads the persisted op-log (`<workspace>.oplog` written by `ccos mcp`);
/// with none it walks a built-in session that drifts. Reads REPL commands on
/// stdin (`timeline`, `goto N`, `recall`, `diff A B`, `help`, `quit`). With
/// `--json` it dumps the field record (stats / integrity / timeline / working set)
/// as JSON and exits — for archiving / fleet collection (see `scripts/fleet_collect.sh`).
fn run_postmortem(args: &[String]) -> CliResult {
    let as_json = args.iter().any(|a| a == "--json");
    let path = args.iter().find(|a| !a.starts_with("--"));
    let session = match path {
        Some(p) => match ccos_core::agent_session::AgentSession::open(p) {
            Ok(s) => s,
            Err(e) => {
                return Err(CliError::fail(format!(
                    "ccos: cannot open session '{p}': {e}"
                )));
            }
        },
        None => ccos_core::postmortem::demo_session(),
    };
    if as_json {
        let ws = path.map(String::as_str).unwrap_or("(built-in demo)");
        let record = ccos_core::postmortem::export(&session, ws, 4096);
        println!("{}", serde_json::to_string_pretty(&record).unwrap());
        return Ok(());
    }
    // A workspace written through `ccos memory` persists its graph but records no
    // op-log, so there is nothing to step through. The debugger used to open on it
    // anyway and report `0 steps`, which reads as "your history is gone" rather
    // than "this workspace never kept one" — the two need very different reactions
    // from whoever is staring at it.
    if path.is_some() && session.is_empty() {
        eprintln!(
            "ccos postmortem: this workspace has no recorded timeline, so there is\n\
             nothing to replay. Time-travel needs the `.oplog` sidecar, which\n\
             `ccos mcp` and `ccos setup` maintain; `ccos memory` writes the graph\n\
             only. `ccos verify <workspace>` reports what is there."
        );
    }
    ccos_core::postmortem::serve(session);
    Ok(())
}

/// `ccos eval [--tasks N] [--seed S] [--budget T] [--model M] [--json]` — the
/// **real-LLM** evaluation (clean + noisy). Configure a model with
/// `ANTHROPIC_API_KEY` (+`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`), `OPENAI_API_KEY`
/// (+`OPENAI_BASE_URL`, `OPENAI_MODEL`) or `OLLAMA_ENDPOINT`; with none set it
/// runs an offline stub (every answer wrong) to exercise the pipeline. `--model`
/// overrides the active provider's model (defaulting to a local Ollama server if
/// no provider env is set).
async fn run_eval_cmd(args: &[String]) -> CliResult {
    let mut cfg = EvalConfig::default();
    let mut json = false;
    let mut model: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json = true,
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--tasks" => {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    cfg.tasks = n;
                }
            }
            "--seed" => {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    cfg.seed = n;
                }
            }
            "--budget" => {
                i += 1;
                if let Some(n) = args.get(i).and_then(|v| v.parse().ok()) {
                    cfg.budget_tokens = n;
                }
            }
            other => eprintln!("ccos: ignoring unknown flag '{other}'"),
        }
        i += 1;
    }

    // `--model M` overrides the model for whichever provider is active; with no
    // provider env set, default to a local Ollama server (the common case).
    if let Some(m) = model {
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            std::env::set_var("ANTHROPIC_MODEL", &m);
        } else if std::env::var("OPENAI_API_KEY").is_ok() {
            std::env::set_var("OPENAI_MODEL", &m);
        } else {
            if std::env::var("OLLAMA_ENDPOINT").is_err() {
                std::env::set_var("OLLAMA_ENDPOINT", "http://localhost:11434");
            }
            std::env::set_var("OLLAMA_MODEL", &m);
        }
    }

    let clean = run_eval(&EvalConfig {
        noisy: false,
        ..cfg.clone()
    })
    .await;
    let noisy = run_eval(&EvalConfig {
        noisy: true,
        ..cfg.clone()
    })
    .await;

    if json {
        let out = serde_json::json!({ "clean": clean, "noisy": noisy });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }

    let strategies = [
        "rag-dense",
        "rag-hybrid",
        "graphrag-1hop",
        "graphrag-bfs",
        "ccos-from-query",
        "ccos-region",
    ];
    let table = |report: &ccos_core::eval::EvalReport, title: &str| {
        println!("  ── {title} ──");
        println!(
            "    {:<16} {:>6} {:>6} {:>6} {:>6}  {:>6} {:>7} {:>7}",
            "strategy (success →)", "d=1", "d=2", "d=3", "d=4", "cover", "halluc", "tokens"
        );
        for strat in strategies {
            let cell = |d: u32| -> String {
                report
                    .per_diameter
                    .iter()
                    .find(|(dd, _)| *dd == d)
                    .and_then(|(_, row)| row.iter().find(|r| r.strategy == strat))
                    .map(|r| format!("{:.2}", r.success_rate))
                    .unwrap_or_else(|| "  – ".into())
            };
            let ov = report.overall.iter().find(|r| r.strategy == strat);
            let (cov, h, t) = ov
                .map(|r| (r.mean_coverage, r.hallucination_rate, r.mean_input_tokens))
                .unwrap_or((0.0, 0.0, 0.0));
            println!(
                "    {:<16} {:>6} {:>6} {:>6} {:>6}  {:>5.0}% {:>6.0}% {:>7.0}",
                strat,
                cell(1),
                cell(2),
                cell(3),
                cell(4),
                cov * 100.0,
                h * 100.0,
                t
            );
        }
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS eval — real-LLM task success vs RAG    ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  provider: {} · model: {}", clean.provider, clean.model);
    println!(
        "  seed={}  tasks={}  budget={} tokens   (success = correct integer answer)\n",
        clean.seed, clean.n_tasks, clean.budget_tokens
    );
    if clean.provider.starts_with("none") {
        println!(
            "  ⚠  No LLM configured — set ANTHROPIC_API_KEY (+ ANTHROPIC_BASE_URL,\n     \
             ANTHROPIC_MODEL), OPENAI_API_KEY, or OLLAMA_ENDPOINT, and allowlist the host.\n     \
             Running the offline stub: every answer is wrong (pipeline check, NOT a result).\n"
        );
    }
    table(&clean, "CLEAN query (names the target function)");
    println!();
    table(
        &noisy,
        "NOISY query (a decoy out-matches the target lexically)",
    );
    Ok(())
}

/// Recursively collect `.rs` files, skipping `target/`, VCS and hidden dirs.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name == ".git" || name.starts_with('.') {
                continue;
            }
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    // Keep the last `max-1` *characters* (room for the leading ellipsis) and cut
    // on a char boundary — a byte slice would panic on multi-byte UTF-8 (e.g. a
    // non-ASCII identifier `fn café()` or an accented panic message).
    let keep = max.saturating_sub(1);
    let start = s
        .char_indices()
        .nth(n - keep)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    format!("…{}", &s[start..])
}

/// `ccos sanitize [path] [--json] [--strict]` — de-obfuscate hidden Unicode in a
/// file (or stdin), surfacing Trojan-Source bidi overrides, zero-width formatting
/// and Unicode-Tags ASCII smuggling as explicit literals, and score the cleaned
/// residue for injection with a per-feature forensic decomposition. `--strict`
/// exits non-zero when a high-severity anomaly or a flagged injection is found
/// (handy as a pre-commit / CI gate).
fn run_sanitize(args: &[String]) -> CliResult {
    use ccos_core::injection_classifier::InjectionDetector;
    use ccos_core::sanitizer::{self, Severity};

    let mut path: Option<String> = None;
    let mut as_json = false;
    let mut strict = false;
    for a in args {
        match a.as_str() {
            "--json" => as_json = true,
            "--strict" => strict = true,
            s if !s.starts_with("--") => path = Some(s.to_string()),
            other => {
                return Err(CliError::usage(format!(
                    "ccos sanitize: unknown flag '{other}'"
                )));
            }
        }
    }

    let input = match path.as_deref() {
        None | Some("-") => {
            use std::io::Read;
            let mut s = String::new();
            if std::io::stdin().read_to_string(&mut s).is_err() {
                return Err(CliError::fail("ccos sanitize: failed to read stdin"));
            }
            s
        }
        Some(p) => match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                return Err(CliError::fail(format!("ccos sanitize: {p}: {e}")));
            }
        },
    };

    let (clean, report) = sanitizer::defang(&input);
    let det = InjectionDetector::default();
    let p_inj = det.injection_probability(&clean);
    let ex = det.explain(&clean);
    let flagged = p_inj >= 0.5;
    let dangerous = report.highest_severity() == Some(Severity::High) || flagged;

    if as_json {
        let findings: Vec<serde_json::Value> = report
            .findings
            .iter()
            .map(|f| {
                serde_json::json!({
                    "byte_offset": f.byte_offset,
                    "char_index": f.char_index,
                    "codepoint": format!("U+{:04X}", f.codepoint),
                    "kind": f.kind.as_str(),
                    "label": f.label,
                    "literal": f.literal(),
                })
            })
            .collect();
        let top: Vec<serde_json::Value> = ex
            .top_terms
            .iter()
            .take(6)
            .map(|t| serde_json::json!({"feature": t.feature, "contribution": t.contribution}))
            .collect();
        let out = serde_json::json!({
            "anomalies": findings,
            "anomaly_summary": report.summary(),
            "injection_probability": p_inj,
            "injection_flagged": flagged,
            "injection_margin": ex.margin,
            "top_terms": top,
            "dangerous": dangerous,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        println!("hidden characters : {}", report.summary());
        for f in &report.findings {
            println!(
                "  @byte {:>5} char {:>4}  {:<13} {}",
                f.byte_offset,
                f.char_index,
                f.kind.as_str(),
                f.literal()
            );
        }
        println!(
            "injection signal  : p={p_inj:.3}{}",
            if flagged { "  [FLAGGED]" } else { "" }
        );
        if p_inj >= 0.2 && !ex.top_terms.is_empty() {
            print!("  top terms       :");
            for t in ex.top_terms.iter().take(5) {
                print!(" {}({:+.2})", t.feature, t.contribution);
            }
            println!();
        }
        if report.is_clean() && !flagged {
            println!("verdict           : clean");
        } else if dangerous {
            println!("verdict           : DANGEROUS");
        }
    }

    if strict && dangerous {
        Err(CliError::status(1))
    } else {
        Ok(())
    }
}

fn print_help() {
    println!(
        "CCOS — Causal Context Operating System (v{})\n\n\
USAGE:\n\
    ccos [COMMAND]\n\n\
COMMANDS:\n\
    demo                       Run the built-in end-to-end kernel demo (default)\n\
    analyze <path> [flags]     Ingest all .rs files under <path> and report\n\
        --json                 Emit the report as JSON instead of text\n\
        --cycles               Detect and list dependency cycles\n\
        --dot <file>           Export the causal graph as Graphviz DOT\n\
        --out <file>           Save a full kernel snapshot (graph + logs) to <file>\n\
        --max-nodes <N>        Paging cap (default 5000)\n\
        --budget <N>           Context-window token budget (default 2048)\n\
    verify <snap | workspace>  Re-check a snapshot's hash chains & integrity, or\n\
    \x20                          audit a workspace op-log's tamper-evident chain\n\
    replay <snapshot.json>     Deterministically replay a saved event log\n\
    sync <sub> <workspace>     Multi-agent store: export/import chain-verified\n\
    \x20                          timeline bundles, status, materialize merged view\n\
    diff <a.json> <b.json>     Structural diff between two snapshots (+ score drift)\n\
    failure <snap> <node-id>   Inject a fault at a node and propagate it (--depth N,\n\
    \x20                          --max-nodes K, --bidirectional, --json)\n\
    focus [src]                Pipe `cargo test` output in → show the causal region\n\
    \x20                          (likely root cause + deps), hiding the noise (--budget,\n\
    \x20                          --input FILE, --json, --workspace [ws] for O(Δ) reuse)\n\
    chaos [--iters N]          Fuzz the guard with adversarial payloads\n\
\n\
  Inspection & export:\n\
    top <path> [--limit N]     Show the hottest nodes by causal score (--json)\n\
    blame <snap> <node-id>     Causes (upstream) + blast radius (downstream) (--depth N, --json)\n\
    export <snap> [--out F]    Export the causal graph as GraphML (default ccos.graphml)\n\
\n\
  Context Region Engine (spatial memory):\n\
    regions <path>             Cluster the causal graph into context regions (--json)\n\
        --activate <node-id>   Hydrate the context window for a node's region\n\
        --metrics <node-id>    Flat-vs-region locality comparison (--radius N)\n\
    experiment [--tasks N]     Hypothesis test: regional memory vs RAG/GraphRAG (--json)\n\
    eval [--tasks N] [--model M]  Real-LLM eval (ANTHROPIC/OPENAI_API_KEY or OLLAMA_ENDPOINT)\n\
    memory [--path FILE]       External-memory façade over stdin JSON Lines (ingest/recall/verify)\n\
    trace                      Parse cargo-test/panic/backtrace (stdin) into the crash's source files\n\
    mcp [workspace.ccos]       Serve memory as MCP tools + resources over stdio JSON-RPC\n\
    \x20                          (persistent if a workspace path is given; for MCP-compatible agents)\n\
    postmortem [workspace] [--json]  Time-travel debugger over a session timeline; --json\n\
    \x20                          dumps the field record (stats/timeline/integrity) and exits\n\
\n\
  Input hardening (de-obfuscation + injection signal):\n\
    sanitize [path] [--json]   De-obfuscate hidden Unicode (Trojan-Source bidi,\n\
    \x20                          zero-width, Tags ASCII-smuggling) into visible\n\
    \x20                          literals + a forensic injection score; reads\n\
    \x20                          stdin when no path. --strict exits non-zero on danger\n\
\n\
  Deployment:\n\
    doctor [--json]            Deployment self-check: build profile, features,\n\
    \x20                          parser, license status, actionable warnings\n\
    setup [flags]              One-command install pass: probe the host, wire the\n\
    \x20                          MCP server into .mcp.json (consent-gated), run the\n\
    \x20                          first-run self-test battery, seal setup_report.json\n\
        --yes                  Apply the wiring without an interactive prompt\n\
        --dry-run              Show the plan + run the battery; write nothing\n\
        --json                 Emit the sealed report JSON instead of the summary\n\
        --hook                 Also print the Mode B feed-hook snippet (manual wiring)\n\
        --dir <D>              Project directory (default .)\n\
        --report <F>           Report path (default <dir>/setup_report.json)\n\
        --workspace <W>        Workspace the MCP server persists (default workspace.ccos)\n\
    license                    Show the active tier (community / PRO) and the Pro feature set\n\
\n\
  CCOS v0.3 — Autonomous Context Runtime:\n\
    scan <path>                Scan a real workspace and ingest the delta\n\
    agents <path>              Run Coder/Reviewer/Security agents over a workspace\n\
    benchmark [--cycles N]     Run the cycle benchmark → benchmark_report.json\n\
                               (also: --cap N, --out FILE)\n\
    runtime <path> [--state D] Scan → schedule → agents → persist (capstone)\n\
\n\
    help, --help               Show this help\n\
    version, --version         Show the version\n\n\
ENVIRONMENT (demo only):\n\
    OLLAMA_ENDPOINT            LLM endpoint (default http://localhost:11434)\n\
    OLLAMA_MODEL               Model name (default codellama)\n\n\
EXAMPLES:\n\
    ccos analyze src --cycles\n\
    ccos analyze src --out run.json && ccos verify run.json && ccos replay run.json\n\
    ccos top src --limit 15\n\
    ccos blame run.json file:src/memory.rs --depth 4\n\
    ccos export run.json --out graph.graphml\n\
    ccos runtime src --state data\n\
    ccos benchmark --cycles 100000\n",
        env!("CARGO_PKG_VERSION")
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccos_core::external_memory::RecallItem;
    use ccos_core::trace::TraceHit;

    /// Build an owned `Vec<String>` arg list from string literals.
    fn argv(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// `verify` must accept the workspace shape `ccos memory` writes.
    ///
    /// It knew a workspace-with-sidecar and a kernel snapshot; a `ccos memory`
    /// workspace is neither, so it fell through to the snapshot loader and printed
    /// that loader's serde error — `cannot load '…': missing field 'version'` — at
    /// a file that is intact, fully verifiable, and was produced by this same
    /// binary. A file that really is neither must still be reported as unloadable.
    #[test]
    fn verify_accepts_a_workspace_with_no_timeline_sidecar() {
        use ccos_core::external_memory::{CcosMemory, ExternalMemory};
        let dir =
            std::env::temp_dir().join(format!("ccos-verify-nosidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Exactly what `ccos memory` leaves behind: a snapshot, no `.oplog`.
        let ws = dir.join("ws.ccos");
        let mut mem = CcosMemory::open(&ws).unwrap();
        mem.ingest_source("src/a.rs", "pub fn a() {}\n");
        mem.checkpoint().unwrap();
        assert!(
            !dir.join("ws.ccos.oplog").exists(),
            "the fixture must have no sidecar, or it tests the other branch"
        );

        assert!(
            run_verify(Some(ws.to_str().unwrap())).is_ok(),
            "a sidecar-less workspace still verifies"
        );

        // Something that is genuinely neither shape is still an error, and the
        // message is the loader's, not a misleading success.
        let bad = dir.join("bad.ccos");
        std::fs::write(&bad, b"not json at all").unwrap();
        assert!(
            run_verify(Some(bad.to_str().unwrap())).is_err(),
            "an unloadable file must still fail"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn setup_opts_parse_defaults_and_flags() {
        let d = SetupOpts::parse(&[]);
        assert_eq!(d.dir, PathBuf::from("."));
        assert!(!d.yes && !d.dry_run && !d.json && !d.hook);
        assert!(d.report.is_none());
        assert_eq!(d.workspace, "workspace.ccos");

        let o = SetupOpts::parse(&argv(&[
            "--yes",
            "--dry-run",
            "--json",
            "--hook",
            "--dir",
            "proj",
            "--report",
            "out.json",
            "--workspace",
            "ws.ccos",
        ]));
        assert!(o.yes && o.dry_run && o.json && o.hook);
        assert_eq!(o.dir, PathBuf::from("proj"));
        assert_eq!(o.report, Some(PathBuf::from("out.json")));
        assert_eq!(o.workspace, "ws.ccos");
    }

    #[test]
    fn analyze_opts_parse_defaults_flags_and_positional() {
        let d = AnalyzeOpts::parse(&[]);
        assert_eq!(d.path, ".");
        assert!(!d.json && !d.cycles);
        assert_eq!(d.max_nodes, 5000);
        assert_eq!(d.budget, 2048);
        assert!(d.out.is_none() && d.dot.is_none());

        let o = AnalyzeOpts::parse(&argv(&[
            "src",
            "--json",
            "--cycles",
            "--out",
            "r.json",
            "--max-nodes",
            "10",
            "--budget",
            "512",
        ]));
        assert_eq!(o.path, "src");
        assert!(o.json && o.cycles);
        assert_eq!(o.out.as_deref(), Some("r.json"));
        assert_eq!(o.max_nodes, 10);
        assert_eq!(o.budget, 512);
        // A non-numeric value leaves the default rather than panicking.
        assert_eq!(AnalyzeOpts::parse(&argv(&["--budget", "abc"])).budget, 2048);
    }

    #[test]
    fn top_opts_parse() {
        let o = TopOpts::parse(&argv(&["src", "--limit", "7", "--json"]));
        assert_eq!(o.path, "src");
        assert_eq!(o.limit, 7);
        assert!(o.json);
        assert_eq!(TopOpts::parse(&[]).limit, 20);
    }

    #[test]
    fn chaos_opts_parse() {
        assert_eq!(ChaosOpts::parse(&[]).iters, 1000);
        assert_eq!(ChaosOpts::parse(&argv(&["--iters", "42"])).iters, 42);
    }

    #[test]
    fn blame_opts_two_positionals_then_flag() {
        let o = BlameOpts::parse(&argv(&["snap.json", "file:src/a.rs", "--depth", "5"]));
        assert_eq!(o.snapshot.as_deref(), Some("snap.json"));
        assert_eq!(o.node.as_deref(), Some("file:src/a.rs"));
        assert_eq!(o.depth, 5);
        // Missing positionals → None (run_blame then prints usage and exits 2).
        let d = BlameOpts::parse(&[]);
        assert!(d.snapshot.is_none() && d.node.is_none());
        assert_eq!(d.depth, 3);
    }

    #[test]
    fn focus_opts_workspace_optional_arg() {
        // --workspace with an explicit path.
        let a = FocusOpts::parse(&argv(&["src", "--workspace", "ws.ccos", "--budget", "999"]));
        assert_eq!(a.path, "src");
        assert_eq!(a.workspace.as_deref(), Some("ws.ccos"));
        assert_eq!(a.budget, 999);
        // --workspace with the next token a flag → defaults to workspace.ccos.
        let b = FocusOpts::parse(&argv(&["--workspace", "--json"]));
        assert_eq!(b.workspace.as_deref(), Some("workspace.ccos"));
        assert!(b.json);
        // --workspace as the final token → default.
        assert_eq!(
            FocusOpts::parse(&argv(&["--workspace"]))
                .workspace
                .as_deref(),
            Some("workspace.ccos")
        );
        // Default path is `src` when no positional is given.
        assert_eq!(FocusOpts::parse(&[]).path, "src");
    }

    #[test]
    fn truncate_cuts_on_char_boundaries_without_panicking() {
        assert_eq!(truncate("hello", 10), "hello"); // under the cap: unchanged
                                                    // Multi-byte input longer than the cap must not panic (the old byte-slice
                                                    // bug panicked inside a multi-byte char).
        let many = "é".repeat(20);
        let out = truncate(&many, 10);
        assert!(out.starts_with('…'));
        assert!(out.chars().count() <= 10);
        // max == 0 must not underflow `max - 1`.
        assert_eq!(truncate("anything", 0), "…");
        // A realistic non-ASCII node id past the cap.
        let id = "file:src/café_handler_extra_long_name.rs";
        assert!(truncate(id, 12).chars().count() <= 12);
    }

    #[test]
    fn focus_view_tags_symptom_and_likely_cause() {
        // The trace blames writer.rs (the symptom). The window (around writer.rs) holds
        // writer.rs and config.rs; config.rs is not in the trace → the causally-pulled
        // likely cause. One entry per file, symptom first, cause next.
        let trace = ExecutionTrace {
            message: "index out of bounds".to_string(),
            hits: vec![TraceHit {
                file: "src/writer.rs".to_string(),
                line: 3,
                frame_depth: 0,
            }],
        };
        let item = |uri: &str, score: f64, content: &str| RecallItem {
            uri: uri.to_string(),
            score,
            kind: "Module".to_string(),
            content: content.to_string(),
            ccr_ref: None,
        };
        let window = RecallWindow {
            strategy: "region".to_string(),
            items: vec![
                item("file:src/writer.rs", 0.90, "pub fn render() {}"),
                item(
                    "sym:src/config.rs:buffer_size",
                    0.60,
                    "pub fn buffer_size() -> usize { 0 }",
                ),
                item("file:src/config.rs", 0.55, "// header"),
            ],
            tokens: 30,
        };
        let view = focus_view(&window, &trace);
        assert_eq!(view.len(), 2, "one entry per distinct file");
        assert_eq!(view[0].file, "src/writer.rs");
        assert_eq!(view[0].role, FocusRole::Symptom);
        assert_eq!(view[1].file, "src/config.rs");
        assert_eq!(
            view[1].role,
            FocusRole::Cause,
            "the top file not named by the trace is the likely cause"
        );
    }
}
