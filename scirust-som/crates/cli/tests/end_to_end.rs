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
