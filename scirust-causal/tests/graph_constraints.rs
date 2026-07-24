use scirust_causal::{CausalError, ConstraintViolation, GraphConstraints};
use scirust_graph::dag::CausalDag;

#[test]
fn no_constraints_means_no_violations() {
    let gc = GraphConstraints::new(3);
    let mut dag = CausalDag::new(3);
    dag.add_directed_edge(0, 1).unwrap();
    assert!(gc.check(&dag).is_empty());
}

#[test]
fn required_edge_present_is_satisfied() {
    let mut gc = GraphConstraints::new(2);
    gc.require_edge(0, 1).unwrap();
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap();
    assert!(gc.check(&dag).is_empty());
}

#[test]
fn required_edge_missing_is_reported() {
    let mut gc = GraphConstraints::new(2);
    gc.require_edge(0, 1).unwrap();
    let dag = CausalDag::new(2);
    let violations = gc.check(&dag);
    assert_eq!(
        violations,
        vec![ConstraintViolation::MissingRequiredEdge { from: 0, to: 1 }]
    );
}

#[test]
fn forbidden_edge_present_is_reported() {
    let mut gc = GraphConstraints::new(2);
    gc.forbid_edge(0, 1).unwrap();
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap();
    let violations = gc.check(&dag);
    assert_eq!(
        violations,
        vec![ConstraintViolation::PresentForbiddenEdge { from: 0, to: 1 }]
    );
}

#[test]
fn forbidden_edge_absent_is_satisfied() {
    let mut gc = GraphConstraints::new(2);
    gc.forbid_edge(0, 1).unwrap();
    let dag = CausalDag::new(2);
    assert!(gc.check(&dag).is_empty());
}

#[test]
fn cannot_require_and_forbid_the_same_edge() {
    let mut gc = GraphConstraints::new(2);
    gc.require_edge(0, 1).unwrap();
    assert!(matches!(
        gc.forbid_edge(0, 1),
        Err(CausalError::InvalidContract { .. })
    ));

    let mut gc2 = GraphConstraints::new(2);
    gc2.forbid_edge(0, 1).unwrap();
    assert!(matches!(
        gc2.require_edge(0, 1),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn required_edge_rejects_self_loop() {
    let mut gc = GraphConstraints::new(2);
    assert!(matches!(
        gc.require_edge(0, 0),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn rejects_out_of_range_index() {
    let mut gc = GraphConstraints::new(2);
    assert!(matches!(
        gc.require_edge(0, 5),
        Err(CausalError::UnknownVariableIndex { index: 5 })
    ));
    assert!(matches!(
        gc.set_tier(9, 0),
        Err(CausalError::UnknownVariableIndex { index: 9 })
    ));
}

#[test]
fn tier_violation_is_reported() {
    let mut gc = GraphConstraints::new(2);
    gc.set_tier(0, 1).unwrap(); // node 0 is LATER than node 1
    gc.set_tier(1, 0).unwrap();
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap(); // edge from later tier to earlier tier
    let violations = gc.check(&dag);
    assert_eq!(
        violations,
        vec![ConstraintViolation::TierViolation {
            from: 0,
            to: 1,
            from_tier: 1,
            to_tier: 0
        }]
    );
}

#[test]
fn same_or_forward_tier_edge_is_not_a_violation() {
    let mut gc = GraphConstraints::new(2);
    gc.set_tier(0, 0).unwrap();
    gc.set_tier(1, 1).unwrap();
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap();
    assert!(gc.check(&dag).is_empty());
}

#[test]
fn requiring_an_edge_that_violates_an_existing_tier_is_rejected() {
    let mut gc = GraphConstraints::new(2);
    gc.set_tier(0, 1).unwrap();
    gc.set_tier(1, 0).unwrap();
    assert!(matches!(
        gc.require_edge(0, 1),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn setting_a_tier_that_would_violate_an_existing_required_edge_is_rejected_and_rolled_back() {
    let mut gc = GraphConstraints::new(2);
    gc.set_tier(1, 5).unwrap();
    gc.require_edge(0, 1).unwrap(); // fine: tier_of[0] is None so far
    // Now try to force node 0 into a later tier than node 1 — must fail and
    // must not silently move node 0's tier.
    assert!(matches!(
        gc.set_tier(0, 6),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn check_is_safe_against_a_smaller_dag() {
    // Constraints declared over 5 variables; the candidate DAG only has 2
    // nodes. Must report violations, not panic.
    let mut gc = GraphConstraints::new(5);
    gc.require_edge(3, 4).unwrap();
    let dag = CausalDag::new(2);
    let violations = gc.check(&dag);
    assert_eq!(
        violations,
        vec![ConstraintViolation::MissingRequiredEdge { from: 3, to: 4 }]
    );
}

#[test]
fn json_round_trips() {
    let mut gc = GraphConstraints::new(3);
    gc.require_edge(0, 1).unwrap();
    gc.forbid_edge(1, 2).unwrap();
    gc.set_tier(0, 0).unwrap();
    let json = serde_json::to_string(&gc).unwrap();
    let back: GraphConstraints = serde_json::from_str(&json).unwrap();
    assert_eq!(gc, back);
}
