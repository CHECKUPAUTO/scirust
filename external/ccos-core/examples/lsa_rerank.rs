//! Does wiring LSA as a **re-ranking** stage in `recall` help? This is the follow-up
//! #39 called for: #39 showed LSA-rank16 is a better *dense ranker* than TF-IDF for
//! synonymy (recall@k≥5) but useless for entry selection (recall@1=0). So it belongs
//! at the ranking stage, not entry selection — which is exactly where
//! `set_lsa_rerank(Some(16))` puts it (it re-orders the recalled region, never the
//! entry).
//!
//! Setup: an `app` hub `use`s many topic modules (so a resolving query's region spans
//! them), with context files co-occurring each topic term with a synonym. We measure
//! the **mean rank** of each topic's target file in the window for a query that
//! resolves — re-ranking off vs on (lower is better) — and how many *pure synonym*
//! queries resolve at all, which is the honest limiter.
//!
//! Run: `cargo run --release --example lsa_rerank`

use ccos_core::external_memory::{CcosMemory, ExternalMemory, Recall};

const TOPICS: &[(&str, &str)] = &[
    ("payment", "billing"),
    ("authentication", "login"),
    ("inventory", "stock"),
    ("shipping", "delivery"),
    ("catalog", "listing"),
    ("ledger", "accounts"),
    ("checkout", "purchase"),
    ("refund", "chargeback"),
];

fn build() -> CcosMemory {
    let mut mem = CcosMemory::new();
    let mut hub = String::new();
    for (t, _) in TOPICS {
        hub.push_str(&format!("use crate::{t};\nuse crate::ctx_{t};\n"));
    }
    hub.push_str("pub fn run() {}\n");
    mem.ingest_source("src/app.rs", &hub);

    for (t, s) in TOPICS {
        mem.ingest_source(
            &format!("src/{t}.rs"),
            &format!("// {t} module: {t} domain logic.\npub fn {t}_run() {{}}\n"),
        );
        let mut ctx =
            format!("// ctx_{t}: the {t} step (also known as {s}); {s} and {t} share state.\n");
        for j in 0..3 {
            ctx.push_str(&format!(
                "// note {j}: {s} feeds {t}; the {s}/{t} stage persists.\n"
            ));
        }
        ctx.push_str(&format!("use crate::{t};\npub fn ctx_{t}_run() {{}}\n"));
        mem.ingest_source(&format!("src/ctx_{t}.rs"), &ctx);
    }
    mem
}

/// Mean 1-based rank of each topic's target file in the window for a query that
/// resolves (the topic term); `999` if absent. Lower is better.
fn mean_target_rank(mem: &CcosMemory) -> f64 {
    let mut total = 0usize;
    for (t, _) in TOPICS {
        let win = mem.recall(&Recall::semantic(format!("{t} domain logic stage")), 4096);
        let target = format!("file:src/{t}.rs");
        let rank = win
            .items
            .iter()
            .position(|i| i.uri == target)
            .map_or(999, |p| p + 1);
        total += rank;
    }
    total as f64 / TOPICS.len() as f64
}

/// How many pure-synonym queries resolve to a non-empty region at all.
fn synonym_resolution(mem: &CcosMemory) -> usize {
    TOPICS
        .iter()
        .filter(|(_, s)| {
            !mem.recall(
                &Recall::semantic(format!("{s} stage state persistence")),
                4096,
            )
            .items
            .is_empty()
        })
        .count()
}

fn main() {
    println!("# Recall LSA re-ranking — effect on a resolving query, and the synonym limiter\n");
    let mut mem = build();

    mem.set_lsa_rerank(None);
    let off = mean_target_rank(&mem);
    mem.set_lsa_rerank(Some(16));
    let on = mean_target_rank(&mem);
    println!(
        "mean target rank ({} resolving topic queries, lower=better):  off {off:.1}   on {on:.1}   \u{0394} {:+.1}",
        TOPICS.len(),
        on - off,
    );

    mem.set_lsa_rerank(None);
    let resolved = synonym_resolution(&mem);
    println!(
        "\npure-synonym queries that resolve to a region: {resolved}/{} \u{2014} the honest limiter is\n\
         ENTRY selection (TF-IDF scores a synonym ~0), exactly #39's recall@1 finding. LSA\n\
         re-ranking sits at the *region* stage, so it can only re-order what entry selection\n\
         already found; it never repairs an empty region. It is deterministic and opt-in\n\
         (`set_lsa_rerank`); on this corpus it earns the default only where \u{0394} < 0.",
        TOPICS.len(),
    );
}
