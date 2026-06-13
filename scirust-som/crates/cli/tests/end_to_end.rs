//! End-to-end: real Rust source → syn frontend → ownership oracle.
//!
//! These pin that the full real-Rust path reports the faults a Rust
//! programmer expects, on non-`Copy` types where the oracle's uniform-move
//! model matches rustc.

use scirust_som_frontend::lower_str;
use scirust_som_symbolic::{FaultKind, OwnershipOracle};

fn faults(src: &str) -> Vec<FaultKind> {
    let lowered = lower_str(src).expect("valid rust");
    OwnershipOracle::new()
        .analyze(&lowered.ast)
        .diagnostics
        .into_iter()
        .map(|d| d.kind)
        .collect()
}

#[test]
fn use_after_move_on_string_is_flagged() {
    let src = r#"
        fn process(input: String) {
            let owned = input;
            let moved = owned;
            let oops = owned;
            drop(oops);
            drop(moved);
        }
    "#;
    let f = faults(src);
    assert_eq!(
        f.iter().filter(|k| **k == FaultKind::UseAfterMove).count(),
        1,
        "expected exactly one use-after-move, got {f:?}"
    );
}

#[test]
fn clean_program_has_no_faults() {
    let src = r#"
        fn ok(a: String) {
            let b = a;
            drop(b);
        }
    "#;
    assert!(faults(src).is_empty());
}

#[test]
fn mutable_borrow_while_shared_is_flagged() {
    // shared borrow is later used (`.len()`), so this is a genuine E0502
    // even under NLL.
    let src = r#"
        fn conflict(data: Vec<u8>) {
            let shared = &data;
            let exclusive = &mut data;
            let n = shared.len();
            drop(exclusive);
            drop(n);
        }
    "#;
    assert!(faults(src).contains(&FaultKind::BorrowConflict));
}

#[test]
fn end_to_end_is_deterministic() {
    let src = "fn h() { let a = String::new(); let b = a; let c = a; }";
    assert_eq!(faults(src), faults(src));
}

#[test]
fn copy_types_double_use_is_legal() {
    // i32 is Copy: rustc accepts this, and so does the type-aware oracle.
    let src = r#"
        fn calc(a: i32) {
            let b: i32 = a;
            let c: i32 = a;
            let d: i32 = b + c;
            drop(d);
        }
    "#;
    assert!(
        faults(src).is_empty(),
        "Copy double-use must not fault: {:?}",
        faults(src)
    );
}

#[test]
fn copy_inference_through_unannotated_let() {
    // `let b = a;` with a: i32 inherits Copy-ness without an annotation.
    let src = r#"
        fn calc(a: i32) {
            let b = a;
            let c = a;
            drop(b);
            drop(c);
        }
    "#;
    assert!(faults(src).is_empty(), "got {:?}", faults(src));
}

#[test]
fn copy_read_under_mut_borrow_is_flagged() {
    // E0503: cannot use `a` because it was mutably borrowed.
    let src = r#"
        fn calc(a: i32) {
            let m = &mut a;
            let b = a;
            drop(m);
            drop(b);
        }
    "#;
    assert!(faults(src).contains(&FaultKind::UseWhileMutBorrowed));
}

#[test]
fn sarif_output_is_valid_and_lists_faults() {
    use scirust_som_symbolic::OwnershipOracle;
    let src = r#"
        fn process(input: String) {
            let owned = input;
            let moved = owned;
            let oops = owned;
            drop(oops);
            drop(moved);
        }
    "#;
    let lowered = scirust_som_frontend::lower_str(src).unwrap();
    let analysis = OwnershipOracle::new().analyze(&lowered.ast);
    let sarif = scirust_som_cli::render_sarif("src/process.rs", &analysis);

    // Structurally valid JSON with the SARIF 2.1.0 envelope and our fault.
    let parsed: serde_json::Value = serde_json::from_str(&sarif).expect("valid JSON");
    assert_eq!(parsed["version"], "2.1.0");
    assert_eq!(parsed["runs"][0]["tool"]["driver"]["name"], "scirust-som");
    let results = parsed["runs"][0]["results"].as_array().unwrap();
    assert_eq!(results.len(), analysis.diagnostics.len());
    assert!(results.iter().any(|r| r["ruleId"] == "UseAfterMove"));
    assert_eq!(
        results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
        "src/process.rs"
    );
}

#[test]
fn sarif_clean_file_has_empty_results() {
    use scirust_som_symbolic::OwnershipOracle;
    let lowered = scirust_som_frontend::lower_str("fn ok(a: String) { drop(a); }").unwrap();
    let analysis = OwnershipOracle::new().analyze(&lowered.ast);
    let sarif = scirust_som_cli::render_sarif("ok.rs", &analysis);
    let parsed: serde_json::Value = serde_json::from_str(&sarif).unwrap();
    assert_eq!(parsed["runs"][0]["results"].as_array().unwrap().len(), 0);
}
