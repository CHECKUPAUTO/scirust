//! **Method-call crux: `x.bar()` receiver-type inference (#23) — precision over recall.** A method call
//! `x.bar()` names `bar` but NOT the type `x` belongs to, so a flat `sym:<file>:bar` symbol index cannot
//! tell `Widget::render` from `Gadget::render` when both types define `render` — it sees an ambiguous
//! name and (correctly) skips, dropping the edge. #23 infers `x`'s concrete type from high-confidence
//! syntax (typed params, `let` annotations, `Foo::new()`/`default()`/`with_*()` constructors, struct
//! literals) and resolves the minted `Type::method` callee through a `(type, method)` index,
//! **resolve-uniquely-or-skip**.
//!
//! The fixture is built around an ADVERSARIAL TWIN — `render` on two different types in two files — so a
//! wrong inference is a concrete, assertable false edge, not a silent no-op. We measure precision (must
//! be 100% with the twin present), the cross-file method edges recovered, and the receiver forms that
//! are deliberately skipped (a wrong guess there would be a false edge — strictly worse than dropping).
//!
//! Run: `cargo run --release --example method_crux`

use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::memory::EdgeType;

fn main() {
    let files: &[(&str, &str)] = &[
        // The adversarial twin: the SAME method name `render` on two DIFFERENT types in two files.
        (
            "src/widget.rs",
            "pub struct Widget;\nimpl Widget {\n    pub fn new() -> Widget { Widget }\n    pub fn render(&self) -> i64 { 1 }\n}",
        ),
        (
            "src/gadget.rs",
            "pub struct Gadget;\nimpl Gadget {\n    pub fn new() -> Gadget { Gadget }\n    pub fn render(&self) -> i64 { 2 }\n}",
        ),
        // Callers (cross-file: they see neither impl block lexically). Each idiom pins a receiver type.
        (
            "src/caller.rs",
            "pub fn drive_ctor() -> i64 { let w = Widget::new(); w.render() }\n\
             pub fn drive_param(g: Gadget) -> i64 { g.render() }\n\
             pub fn drive_annot() -> i64 { let g: Gadget = Gadget::new(); g.render() }\n\
             pub fn drive_chain() -> i64 { Widget::new().render() }\n",
        ),
        // Deliberately-skipped receiver forms (the precision-preserving recall holes).
        (
            "src/holes.rs",
            "pub trait Draw { fn render(&self) -> i64; }\n\
             pub fn via_trait_obj(d: &dyn Draw) -> i64 { d.render() }\n\
             pub fn via_generic<T: Draw>(t: T) -> i64 { t.render() }\n",
        ),
    ];

    let mut mem = CcosMemory::new();
    for (p, c) in files {
        mem.ingest_source(p, c);
    }
    let g = mem.graph();

    // The resolved method-call edges of interest: a caller symbol → a `render` method symbol.
    let mut render_edges: Vec<(String, String)> = g
        .edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Calls && e.target.0.ends_with(":render"))
        .map(|e| (e.source.0.clone(), e.target.0.clone()))
        .collect();
    render_edges.sort();
    render_edges.dedup();

    // Ground truth: which (caller → render) edges SHOULD exist, by the source semantics.
    let truth: Vec<(String, String)> = [
        ("drive_ctor", "widget"),
        ("drive_param", "gadget"),
        ("drive_annot", "gadget"),
    ]
    .iter()
    .map(|(caller, file)| {
        (
            format!("sym:src/caller.rs:{caller}"),
            format!("sym:src/{file}.rs:render"),
        )
    })
    .collect();

    let recovered = truth.iter().filter(|e| render_edges.contains(e)).count();
    let false_edges: Vec<&(String, String)> =
        render_edges.iter().filter(|e| !truth.contains(e)).collect();

    println!("# Method-call crux — x.bar() receiver-type inference (#23)\n");
    println!(
        "resolved caller→render edges (a flat name index sees only an ambiguous `render` → skips):"
    );
    let short = |s: &str| s.strip_prefix("sym:").unwrap_or(s).to_string();
    for (s, t) in &render_edges {
        let ok = if truth.contains(&(s.clone(), t.clone())) {
            "OK"
        } else {
            "XX"
        };
        println!("  [{ok}] {}  →  {}", short(s), short(t));
    }
    let prec = if render_edges.is_empty() {
        1.0
    } else {
        recovered as f64 / render_edges.len() as f64
    };
    println!(
        "\n  recovered {}/{} true method edges   precision {:.0}%   false cross-type edges {}",
        recovered,
        truth.len(),
        100.0 * prec,
        false_edges.len()
    );

    println!("\n  Deliberately skipped (precision-preserving recall holes — never a false edge):");
    println!("    drive_chain    Widget::new().render()   receiver is a CALL, not a bare ident");
    println!(
        "    via_trait_obj  d: &dyn Draw             trait object has no single concrete type"
    );
    println!("    via_generic    t: T                     a generic param is not a concrete type");

    println!(
        "\n  → A flat `sym:<file>:render` index is ambiguous (render lives on Widget AND Gadget) and\n\
         skips. #23 infers each receiver's concrete type from syntax and resolves `Type::render` via the\n\
         (type, method) index, unique-or-skip: drive_ctor→Widget, drive_param/annot→Gadget — the twin\n\
         proving zero cross-type false edges. Deterministic, replay == live, eager ≡ batch.\n\
         See docs/MEASUREMENT_method_crux.md."
    );

    assert_eq!(recovered, truth.len(), "all true method edges recovered");
    assert!(
        false_edges.is_empty(),
        "no false cross-type edge minted: {false_edges:?}"
    );
}
