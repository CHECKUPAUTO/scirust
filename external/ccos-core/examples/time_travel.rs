//! Time-travel debugging demo: an agent session whose context **drifts**, then is
//! debugged by rewinding and replaying under a larger budget.
//!
//! Run with: `cargo run --example time_travel`
//!
//! This is the capability a probabilistic RAG/framework stack cannot offer: the
//! session is event-sourced, so the exact context an agent saw at any step is
//! reconstructible, and a recall can be replayed counterfactually.

use ccos_core::agent_session::AgentSession;
use ccos_core::external_memory::{ExternalMemory, Recall, RecallWindow};

/// A causal chain api → repo → db (the bug lives in db), plus unrelated noise.
const WORKSPACE: &[(&str, &str)] = &[
    (
        "src/db.rs",
        "// connection pool\npub fn timeout_ms() -> i64 { 30 } // BUG: far too low\n",
    ),
    (
        "src/repo.rs",
        "use crate::db;\npub fn fetch() -> i64 { db::timeout_ms() * 2 }\n",
    ),
    (
        "src/api.rs",
        "use crate::repo;\npub fn handle() -> i64 { repo::fetch() + 1 }\n",
    ),
    (
        "src/util.rs",
        "pub fn format_date() -> String { String::new() }\n",
    ),
    ("src/log.rs", "pub fn info(_m: &str) {}\n"),
];

const FAULT: &str = "file:src/api.rs";
const CAUSE: &str = "file:src/db.rs";

fn files_in(w: &RecallWindow) -> Vec<&str> {
    w.items
        .iter()
        .map(|i| i.uri.as_str())
        .filter(|u| u.starts_with("file:"))
        .collect()
}

fn show(window: &RecallWindow) {
    for f in ["file:src/api.rs", "file:src/repo.rs", "file:src/db.rs"] {
        let present = files_in(window).contains(&f);
        let mark = if present { "✓" } else { "✗ MISSING" };
        let tag = if f == CAUSE {
            "   ← the root cause (timeout_ms)"
        } else {
            ""
        };
        println!("     {:<20} {}{}", f, mark, tag);
    }
    println!(
        "     ({} files, ~{} tokens)",
        files_in(window).len(),
        window.tokens
    );
}

fn main() {
    let mut s = AgentSession::new();
    for (uri, src) in WORKSPACE {
        s.ingest(uri, src);
    }
    // A test fails on api.rs — the agent signals it.
    s.signal_failure(FAULT, 3).ok();

    // The agent recalls context with a TIGHT token budget → it drifts: the window
    // fills with the symptom but the *cause* (db.rs, two hops away) is evicted.
    let tight_budget = 18;
    let drifted = s.recall(Recall::around(FAULT), tight_budget);
    let recall_step = s.len() - 1;

    println!("=== Cognitive timeline ===");
    for line in s.timeline() {
        println!("  {line}");
    }

    println!(
        "\n=== t={} — the agent's window (TIGHT budget {}) ===",
        recall_step + 1,
        tight_budget
    );
    show(&drifted);
    let drifted_has_cause = files_in(&drifted).contains(&CAUSE);
    println!(
        "  → cause visible? {}. The agent patches api.rs blindly; the db.rs bug survives.",
        drifted_has_cause
    );

    // --- TIME-TRAVEL DEBUG: rewind to the recall and replay with a larger budget.
    println!(
        "\n=== Time-travel debug: rewind to t={}, replay with a larger budget ===",
        recall_step + 1
    );
    let roomy_budget = 4000;
    let fixed = s.recall_what_if(recall_step, &Recall::around(FAULT), roomy_budget);
    show(&fixed);
    let fixed_has_cause = files_in(&fixed).contains(&CAUSE);
    println!(
        "  → cause visible? {}. With more budget the agent would have seen db.rs::timeout_ms.",
        fixed_has_cause
    );

    // The whole thing is deterministic and auditable.
    let replayed = s.replay_to(s.len());
    println!("\n=== Auditability ===");
    println!(
        "  replay_to({}) reconstructs the exact state: {} nodes (live: {}).",
        s.len(),
        replayed.stats().nodes,
        s.memory().stats().nodes
    );
    assert_eq!(replayed.stats().nodes, s.memory().stats().nodes);

    println!(
        "\nVerdict: the drift ({}→{} on the cause) was diagnosed by rewinding the\n\
         exact context and replaying a single parameter — time-travel debugging a\n\
         RAG stack cannot do.",
        drifted_has_cause, fixed_has_cause
    );
}
