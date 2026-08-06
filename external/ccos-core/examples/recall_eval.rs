//! Honest, LLM-free measurement of the recall strategies added this session
//! (hybrid fusion, and — under `--features learned-embed` — the LSA embedder)
//! against the lexical/semantic/working-set baselines, on a synthetic corpus
//! with **ground-truth** relevant files.
//!
//! Run:
//!   cargo run --release --example recall_eval                       # TF-IDF default
//!   cargo run --release --example recall_eval --features learned-embed   # LSA
//!
//! The point is to let the data speak: each task type isolates where a signal
//! *should* help, and the table reports hit-rate (did the ground-truth file land
//! in the recalled window?) per strategy. Where a strategy doesn't beat lexical,
//! the table says so — no cherry-picking.

use ccos_core::external_memory::{CcosMemory, ExternalMemory, Recall};

/// One benchmark task: a free-text query and the file the recall *should* surface.
struct Task {
    kind: Kind,
    query: String,
    target: String,          // file uri that must appear in the window
    fail_on: Option<String>, // a failure signal to inject before recall (causal cue)
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Plain,   // query contains the target's own distinctive term (lexical suffices)
    Decoy,   // a decoy out-matches lexically; the target is the *failing* file
    Synonym, // query term only *co-occurs* with the target's term (needs LSA)
}

const DOMAINS: [&str; 8] = [
    "payment",
    "inventory",
    "shipping",
    "auth",
    "billing",
    "catalog",
    "ledger",
    "checkout",
];
// A synonym for each domain that never appears in the domain's own file, but
// co-occurs with it in a "glossary" bridge doc (so LSA can link them).
const SYNONYMS: [&str; 8] = [
    "remittance",
    "stock",
    "dispatch",
    "login",
    "invoicing",
    "listing",
    "accounts",
    "purchase",
];

fn build_corpus() -> (CcosMemory, Vec<Task>) {
    let mut mem = CcosMemory::new();

    // Each domain is a module with its distinctive term + shared boilerplate, and
    // a causal `use` edge to the previous module (a realistic dependency chain).
    for (i, d) in DOMAINS.iter().enumerate() {
        let prev = if i == 0 {
            String::new()
        } else {
            format!("use crate::{};\n", DOMAINS[i - 1])
        };
        // Distinctive content only — NO shared boilerplate words — so a query of
        // the common words (the "decoy" task) does not match the target at all,
        // isolating the non-lexical (causal/failure) signal.
        let src = format!(
            "{prev}// {d} module: {d} domain logic.\n\
             pub fn {d}_run() -> u32 {{ 0 }}\n\
             pub fn {d}_apply() -> u32 {{ 1 }}\n"
        );
        mem.ingest_source(&format!("src/{d}.rs"), &src);
    }
    // Glossary bridge docs: co-occur each domain term with its synonym so the
    // latent space (LSA) links them. Raw TF-IDF gets no help from these for a
    // synonym query, because the *target file* still never contains the synonym.
    for (d, s) in DOMAINS.iter().zip(SYNONYMS.iter()) {
        mem.ingest_source(
            &format!("src/glossary_{d}.rs"),
            &format!("// glossary: {d} is also called {s}; {s} means {d}.\n"),
        );
    }
    // Lexical decoys: files stuffed with the *common* words so they out-match a
    // query lexically without being the real target.
    for i in 0..4 {
        mem.ingest_source(
            &format!("src/decoy_{i}.rs"),
            "// notes: service handler manages process update flow service handler.\n\
             pub fn notes_service_handler_process_update() -> u32 { 0 }\n",
        );
    }
    // Unrelated filler modules so the corpus is large enough that a tight window
    // is genuinely selective (working-set can't just hold everything).
    for i in 0..40 {
        mem.ingest_source(
            &format!("src/filler_{i}.rs"),
            &format!(
                "// module {i}: assorted helper utilities.\npub fn helper_{i}() -> u32 {{ {i} }}\n"
            ),
        );
    }

    let mut tasks = Vec::new();
    for (d, s) in DOMAINS.iter().zip(SYNONYMS.iter()) {
        let target = format!("file:src/{d}.rs");
        // Plain: the domain term is in the query and the target — lexical wins.
        tasks.push(Task {
            kind: Kind::Plain,
            query: format!("{d} domain logic"),
            target: target.clone(),
            fail_on: None,
        });
        // Decoy + failure: the query is all common words a decoy also has, so
        // lexical points at a decoy; the target is the active *failing* file.
        tasks.push(Task {
            kind: Kind::Decoy,
            query: "service handler process update".to_string(),
            target: target.clone(),
            fail_on: Some(target.clone()),
        });
        // Synonym: the query uses the synonym, which the target file never says.
        tasks.push(Task {
            kind: Kind::Synonym,
            query: format!("{s} flow"),
            target: target.clone(),
            fail_on: None,
        });
    }
    (mem, tasks)
}

/// Does the recalled window hold the target file (at file granularity)?
fn hit(mem: &mut CcosMemory, task: &Task, strat: &str, budget: usize) -> bool {
    if let Some(uri) = &task.fail_on {
        let _ = mem.signal_failure(uri, 0);
    }
    let recall = match strat {
        "working_set" => Recall::working_set(),
        "lexical" => Recall::task(&task.query),
        "semantic" => Recall::semantic(&task.query),
        "hybrid" => Recall::hybrid(&task.query),
        _ => unreachable!(),
    };
    let win = mem.recall(&recall, budget);
    let want = task.target.trim_start_matches("file:");
    win.items.iter().any(|it| it.uri.contains(want))
}

fn main() {
    let strategies = ["working_set", "lexical", "semantic", "hybrid"];
    let kinds = [
        ("plain", Kind::Plain),
        ("decoy+fail", Kind::Decoy),
        ("synonym", Kind::Synonym),
    ];
    let budget = 160; // tight: fits only a few files out of ~60, so selection matters

    let embedder = if cfg!(feature = "learned-embed") {
        "LSA (learned-embed)"
    } else {
        "INT4 TF-IDF (default)"
    };
    println!("# Recall strategy hit-rate — semantic embedder: {embedder}");
    println!("# (fresh memory per task so failure cues don't leak; budget={budget} tokens)\n");
    print!("{:<14}", "strategy");
    for (label, _) in &kinds {
        print!("{label:>12}");
    }
    println!("{:>12}", "overall");

    for strat in strategies {
        print!("{strat:<14}");
        let (mut tot_hit, mut tot_n) = (0usize, 0usize);
        for (_, kind) in &kinds {
            let (mut hits, mut n) = (0usize, 0usize);
            let (_, tasks) = build_corpus();
            for task in tasks.iter().filter(|t| t.kind == *kind) {
                // A fresh corpus per task so a prior failure cue never biases the
                // next task's causal scores.
                let (mut mem, _) = build_corpus();
                n += 1;
                if hit(&mut mem, task, strat, budget) {
                    hits += 1;
                }
            }
            tot_hit += hits;
            tot_n += n;
            let pct = if n == 0 {
                0.0
            } else {
                100.0 * hits as f64 / n as f64
            };
            print!("{:>11.0}%", pct);
        }
        let pct = if tot_n == 0 {
            0.0
        } else {
            100.0 * tot_hit as f64 / tot_n as f64
        };
        println!("{:>11.0}%", pct);
    }
}
