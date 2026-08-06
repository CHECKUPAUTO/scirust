//! `ccos-migrate` — import a **CCOS Migration Bundle** (produced from any RAG
//! store by `tools/rag2ccos`) into a causal working-memory, **losslessly**.
//!
//! A self-contained companion to the main `ccos` CLI: it uses only the sync core,
//! so it builds in the default feature set (no async, no `llm`). See
//! `docs/MIGRATION.md` for the full mapping and the lossless contract.
//!
//! ```text
//! ccos-migrate --bundle <file.cmb.jsonl> [--path W] [--mode auto|code|docs]
//!              [--report FILE] [--dry-run] [--no-verify] [--json]
//! ```
//!
//! Document/chunk/entity records become content nodes (text retained verbatim,
//! hash-verified); `parent`/`ordinal`/`relations` become causal edges;
//! `code_file` records are re-parsed into the real dependency graph. Metadata and
//! embeddings are preserved in sidecars next to the workspace. Exits non-zero if
//! the lossless content-hash check fails.

use ccos_core::migrate::{self, MigrateConfig, MigrationOutcome, Mode};
use std::process::ExitCode;

struct Opts {
    bundle: Option<String>,
    path: String,
    mode: Mode,
    report: Option<String>,
    dry_run: bool,
    verify: bool,
    extend: bool,
    json: bool,
}

impl Opts {
    fn parse(args: &[String]) -> Self {
        let mut o = Opts {
            bundle: None,
            path: "workspace.ccos".to_string(),
            mode: Mode::Auto,
            report: None,
            dry_run: false,
            verify: true,
            extend: false,
            json: false,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--bundle" | "--in" => {
                    i += 1;
                    o.bundle = args.get(i).cloned();
                }
                "--path" | "--workspace" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        o.path = v.clone();
                    }
                }
                "--mode" => {
                    i += 1;
                    if let Some(m) = args.get(i).and_then(|v| Mode::parse(v)) {
                        o.mode = m;
                    }
                }
                "--report" => {
                    i += 1;
                    o.report = args.get(i).cloned();
                }
                "--dry-run" => o.dry_run = true,
                "--no-verify" => o.verify = false,
                "--extend" => o.extend = true,
                "--json" => o.json = true,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                s if !s.starts_with("--") => o.bundle = Some(s.to_string()),
                other => eprintln!("ccos-migrate: ignoring unknown flag '{other}'"),
            }
            i += 1;
        }
        o
    }
}

fn print_help() {
    println!(
        "ccos-migrate — import a RAG corpus into a CCOS causal workspace, losslessly\n\n\
USAGE:\n\
    ccos-migrate --bundle <file.cmb.jsonl> [options]\n\n\
OPTIONS:\n\
    --bundle F, --in F      the CCOS Migration Bundle to import (required)\n\
    --path W                target workspace (default: workspace.ccos; created or extended)\n\
    --mode auto|code|docs   how to rebuild structure (default: auto)\n\
    --report FILE           write the JSON migration report to FILE\n\
    --dry-run               parse + import in memory + report; write nothing\n\
    --no-verify             skip the per-node content-hash re-check\n\
    --extend                merge into the existing workspace at --path instead of\n\
    \x20                         overwriting it (incremental import; keeps its graph + logs)\n\
    --json                  emit the report as JSON on stdout\n\
    -h, --help              show this help\n\n\
Produce a bundle from any RAG store with tools/rag2ccos. See docs/MIGRATION.md."
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = Opts::parse(&args);

    let Some(bundle) = opts.bundle.as_deref() else {
        eprintln!(
            "usage: ccos-migrate --bundle <file.cmb.jsonl> [--path workspace.ccos] \
             [--mode auto|code|docs] [--report FILE] [--dry-run] [--no-verify] [--json]"
        );
        return ExitCode::from(2);
    };

    let file = match std::fs::File::open(bundle) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ccos-migrate: cannot open bundle '{bundle}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let reader = std::io::BufReader::new(file);

    let cfg = MigrateConfig {
        mode: opts.mode,
        verify: opts.verify,
        // `--extend` merges into the workspace at --path; otherwise a fresh one is built.
        extend_from: if opts.extend {
            Some(opts.path.clone())
        } else {
            None
        },
    };
    let outcome = match migrate::migrate_bundle(reader, &cfg) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ccos-migrate: {e}");
            return ExitCode::FAILURE;
        }
    };
    let MigrationOutcome {
        mut memory,
        report,
        manifest,
        embeddings,
    } = outcome;

    if !opts.dry_run {
        // The migration assembled a fresh workspace; persist it (overwriting any
        // file at `--path`). Rollback is `rm`; the source RAG store is untouched.
        if let Err(e) = memory.checkpoint_to(&opts.path) {
            eprintln!("ccos-migrate: checkpoint '{}': {e}", opts.path);
            return ExitCode::FAILURE;
        }
        // Manifest sidecar — the authoritative, lossless record of every original
        // field (esp. metadata the graph node cannot hold).
        let manifest_path = format!("{}.migration.jsonl", opts.path);
        let mut mbuf = String::new();
        for m in &manifest {
            mbuf.push_str(&serde_json::to_string(m).unwrap_or_default());
            mbuf.push('\n');
        }
        if let Err(e) = std::fs::write(&manifest_path, mbuf) {
            eprintln!("ccos-migrate: write '{manifest_path}': {e}");
            return ExitCode::FAILURE;
        }
        // Embeddings sidecar (only if any were preserved) — seeds the Pro semantic tier.
        if !embeddings.is_empty() {
            let emb_path = format!("{}.embeddings.jsonl", opts.path);
            let mut ebuf = String::new();
            for e in &embeddings {
                ebuf.push_str(&serde_json::to_string(e).unwrap_or_default());
                ebuf.push('\n');
            }
            if let Err(e) = std::fs::write(&emb_path, ebuf) {
                eprintln!("ccos-migrate: write '{emb_path}': {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let report_json = serde_json::to_string_pretty(&report).unwrap_or_default();
    if let Some(rp) = opts.report.as_deref() {
        if let Err(e) = std::fs::write(rp, &report_json) {
            eprintln!("ccos-migrate: write report '{rp}': {e}");
            return ExitCode::FAILURE;
        }
    }

    if opts.json {
        println!("{report_json}");
    } else {
        let edges = report.edges_containment + report.edges_sequence + report.edges_relation;
        println!("─── CCOS migration ───");
        println!(
            "  {} records → {} document + {} code nodes",
            report.records, report.nodes_documents, report.nodes_code_files
        );
        println!(
            "  edges: {edges} rebuilt ({} containment · {} sequence · {} relation) + {} cross-file",
            report.edges_containment,
            report.edges_sequence,
            report.edges_relation,
            report.cross_file_edges
        );
        println!(
            "  {} embeddings preserved · {} bytes of content",
            report.embeddings_preserved, report.bytes_content
        );
        let mut kinds: Vec<_> = report.by_kind.iter().collect();
        kinds.sort();
        let kinds_str: Vec<String> = kinds.iter().map(|(k, n)| format!("{k}:{n}")).collect();
        if !kinds_str.is_empty() {
            println!("  by kind: {}", kinds_str.join(" · "));
        }
        if opts.dry_run {
            println!("  (dry run — nothing written)");
        } else {
            println!("  → {} (+ .migration.jsonl manifest)", opts.path);
        }
        if report.lossless {
            println!("  ✓ lossless: every record verified");
        } else {
            println!(
                "  ✗ {} content-hash mismatch(es) — NOT lossless",
                report.hash_mismatches
            );
            for w in report.warnings.iter().take(10) {
                println!("      {w}");
            }
        }
    }

    if report.lossless {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
