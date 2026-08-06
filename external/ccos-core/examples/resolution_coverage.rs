//! # Resolution coverage — CCOS's call / data-flow resolver on real Rust path shapes, measured.
//!
//! CCOS builds a *causal* graph a RAG stack has no representation of: `Calls` (fn → fn) and
//! `DataFlow` (fn → `static`/`const`) edges, resolved from source with a strict **precision-first**
//! discipline — every edge is **resolve-uniquely-or-skip**, so a wrong edge is never invented. This
//! example enumerates the Rust path shapes the resolver handles (each tagged with the slice that
//! added it), confirms the shapes it *deliberately skips* to preserve precision, and reports the
//! structural yield on CCOS's own `src/`. Every row is the REAL output of the run; two runs are
//! identical, bit for bit (the determinism/auditability a generative stage cannot offer).
//!
//! Run: `cargo run --release --example resolution_coverage`

use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::memory::EdgeType;
use ccos_core::parser::ASTParser;
use std::fs;
use std::path::{Path, PathBuf};

fn edges(m: &CcosMemory, kind: EdgeType) -> usize {
    m.graph()
        .edges()
        .iter()
        .filter(|e| e.edge_type == kind)
        .count()
}

/// Ingest a tiny crafted workspace and report how many `Calls` + `DataFlow` edges the resolver
/// produced. Each fixture is minimal — it exercises exactly one path shape — so a non-zero count
/// means that shape resolved, and zero means it was (correctly) skipped.
fn resolved(files: &[(&str, &str)]) -> usize {
    let mut m = CcosMemory::new();
    for (uri, src) in files {
        m.ingest_source(uri, src);
    }
    edges(&m, EdgeType::Calls) + edges(&m, EdgeType::DataFlow)
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// One fixture: the files of a tiny crafted workspace, `(uri, source)` pairs.
type Fixture = &'static [(&'static str, &'static str)];

fn main() {
    println!("# Resolution coverage — CCOS's call/data-flow resolver, measured\n");

    // ── Shapes the resolver RESOLVES (precision-first) ──────────────────────────────────────────
    // (label, tag, fixture files). The last file's body holds the reference under test.
    let resolves: &[(&str, &str, Fixture)] = &[
        (
            "crate-rooted   crate::m::f()",
            "",
            &[
                ("src/m.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/lib.rs", "mod m;\npub fn r() -> i64 { crate::m::f() }\n"),
            ],
        ),
        (
            "use fn         use m::f; f()",
            "",
            &[
                ("src/m.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/lib.rs", "use m::f;\npub fn r() -> i64 { f() }\n"),
            ],
        ),
        (
            "use module     use crate::m; m::f()",
            "",
            &[
                ("src/m.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/lib.rs", "use crate::m;\npub fn r() -> i64 { m::f() }\n"),
            ],
        ),
        (
            "local submod   mod m; m::f()",
            "#122",
            &[
                ("src/m.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/lib.rs", "mod m;\npub fn r() -> i64 { m::f() }\n"),
            ],
        ),
        (
            "nested submod  a::b::f()",
            "#122",
            &[
                ("src/a/b.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/lib.rs", "pub mod a;\npub fn r() -> i64 { a::b::f() }\n"),
            ],
        ),
        (
            "type method    x.bar()  (x: &T)",
            "#23",
            &[(
                "src/lib.rs",
                "pub struct T;\nimpl T { pub fn bar(&self) -> i64 { 0 } }\npub fn r(x: &T) -> i64 { x.bar() }\n",
            )],
        ),
        (
            "Self method    self.helper()",
            "#20",
            &[(
                "src/lib.rs",
                "pub struct T;\nimpl T { fn helper(&self) -> i64 { 0 }\n  pub fn r(&self) -> i64 { self.helper() } }\n",
            )],
        ),
        (
            "bare const     FOO (globally unique)",
            "",
            &[(
                "src/lib.rs",
                "pub const FOO: i64 = 1;\npub fn r() -> i64 { FOO }\n",
            )],
        ),
        (
            "import const   use m::MAX; MAX",
            "#113",
            &[
                ("src/m.rs", "pub const MAX: i64 = 1;\n"),
                (
                    "src/lib.rs",
                    "use crate::m::MAX;\npub fn r() -> i64 { MAX }\n",
                ),
            ],
        ),
        (
            "renamed const  use m::MAX as L; L",
            "#124",
            &[
                ("src/m.rs", "pub const MAX: i64 = 1;\n"),
                (
                    "src/lib.rs",
                    "use crate::m::MAX as L;\npub fn r() -> i64 { L }\n",
                ),
            ],
        ),
        (
            "field receiver self.db.q()",
            "S4",
            &[(
                "src/lib.rs",
                "pub struct Db;\nimpl Db { pub fn q(&self) -> i64 { 0 } }\npub struct S { db: Db }\nimpl S { pub fn r(&self) -> i64 { self.db.q() } }\n",
            )],
        ),
        (
            "fn-return      make().q()",
            "S4",
            &[(
                "src/lib.rs",
                "pub struct B;\nimpl B { pub fn q(&self) -> i64 { 0 } }\npub fn make() -> B { B }\npub fn r() -> i64 { make().q() }\n",
            )],
        ),
        (
            "method chain   x.b().q()",
            "S4",
            &[(
                "src/lib.rs",
                "pub struct A;\npub struct B;\nimpl A { pub fn b(&self) -> B { B } }\nimpl B { pub fn q(&self) -> i64 { 0 } }\npub fn r(x: A) -> i64 { x.b().q() }\n",
            )],
        ),
        (
            "assoc -> Self  A::make().t()",
            "S4",
            &[(
                "src/lib.rs",
                "pub struct A;\nimpl A { pub fn make() -> Self { A }\n  pub fn t(&self) -> i64 { 0 } }\npub fn r() -> i64 { A::make().t() }\n",
            )],
        ),
    ];

    // ── Shapes the resolver correctly SKIPS (precision-first: never a guessed edge) ──────────────
    let skips: &[(&str, Fixture)] = &[
        (
            "bare extern    othercrate::f() (no use)",
            &[
                ("othercrate/src/lib.rs", "pub fn f() -> i64 { 0 }\n"),
                (
                    "mycrate/src/api.rs",
                    "pub fn r() -> i64 { othercrate::f() }\n",
                ),
            ],
        ),
        (
            "ambiguous      f() (two defs, no import)",
            &[
                ("src/a.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/b.rs", "pub fn f() -> i64 { 0 }\n"),
                ("src/api.rs", "pub fn r() -> i64 { f() }\n"),
            ],
        ),
        (
            "unknown module nope::f()",
            &[("src/api.rs", "pub fn r() -> i64 { nope::f() }\n")],
        ),
        (
            "wrapper field  self.v.push() (v: Vec<_>)",
            // A field whose declared type is a std wrapper is never a concrete receiver — the
            // method dispatches to the wrapper, so minting an edge could only be wrong (S4).
            &[(
                "src/api.rs",
                "pub struct S { v: Vec<i64> }\nimpl S { pub fn r(&mut self) { self.v.push(1) } }\n",
            )],
        ),
    ];

    println!("RESOLVES — each a distinct path shape, resolve-uniquely-or-skip:");
    println!("  {:<40}{:>7}   resolved?", "shape", "slice");
    println!("  {}", "-".repeat(64));
    let mut ok = 0;
    for (label, tag, files) in resolves {
        let r = resolved(files) > 0;
        ok += usize::from(r);
        println!(
            "  {:<40}{:>7}   {}",
            label,
            tag,
            if r { "✓ edge" } else { "✗ MISSED" }
        );
    }
    println!("\nCORRECTLY SKIPPED — precision-first (a wrong edge is worse than none):");
    println!("  {}", "-".repeat(64));
    let mut skipped = 0;
    for (label, files) in skips {
        let r = resolved(files) > 0;
        skipped += usize::from(!r);
        println!(
            "  {:<48}   {}",
            label,
            if r { "✗ FALSE EDGE" } else { "✓ skipped" }
        );
    }
    println!(
        "\n  → {}/{} idioms resolve; {}/{} precision-skips hold.",
        ok,
        resolves.len(),
        skipped,
        skips.len()
    );

    // ── Structural yield on CCOS's own src/ (real code) ─────────────────────────────────────────
    let files = rust_files(Path::new("src"));
    let parser = ASTParser::new();
    let (mut call_sites, mut data_refs) = (0usize, 0usize);
    let mut mem = CcosMemory::new();
    for f in &files {
        let uri = f.to_string_lossy().to_string();
        let Ok(src) = fs::read_to_string(f) else {
            continue;
        };
        let pr = parser.parse_source(&uri, &src);
        call_sites += pr.call_sites.len();
        data_refs += pr.data_refs.len();
        mem.ingest_source(&uri, &src);
    }
    let call_edges = edges(&mem, EdgeType::Calls);
    let df_edges = edges(&mem, EdgeType::DataFlow);
    println!("\n── At scale: CCOS's own src/ ({} files) ──", files.len());
    println!(
        "  {call_sites} call references parsed  →  {call_edges} fn→fn `Calls` edges resolved (deduped)",
    );
    println!(
        "  {data_refs} const/static references parsed  →  {df_edges} `DataFlow` edges resolved (deduped)",
    );
    println!(
        "\n→ The resolved edges are the intra-crate causal structure — the fn→fn and fn→const graph a\n\
         vector RAG index has no representation of. The parsed-but-unresolved remainder is dominated by\n\
         calls into `std`/external crates, method chains on non-inferable receivers, and macro paths —\n\
         all **correctly** left unresolved: resolve-uniquely-or-skip means CCOS never asserts a causal\n\
         edge it cannot prove. Deterministic and replay-exact; see docs/MEASUREMENT_resolution_coverage.md."
    );
}
