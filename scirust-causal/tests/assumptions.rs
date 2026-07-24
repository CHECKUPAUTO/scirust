use scirust_causal::{AssumptionBasis, AssumptionRegistry, CausalAssumption, CausalError};

#[test]
fn new_registry_is_empty() {
    let reg = AssumptionRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
}

#[test]
fn assert_then_get_round_trips() {
    let mut reg = AssumptionRegistry::new();
    reg.assert(
        CausalAssumption::Acyclicity,
        AssumptionBasis::AssertedByAnalyst,
        None,
    )
    .unwrap();
    let record = reg.get(&CausalAssumption::Acyclicity).unwrap();
    assert_eq!(record.basis, AssumptionBasis::AssertedByAnalyst);
    assert_eq!(reg.len(), 1);
}

#[test]
fn asserting_twice_is_rejected() {
    let mut reg = AssumptionRegistry::new();
    reg.assert(CausalAssumption::Sutva, AssumptionBasis::Unverified, None)
        .unwrap();
    assert!(matches!(
        reg.assert(
            CausalAssumption::Sutva,
            AssumptionBasis::AssertedByAnalyst,
            None
        ),
        Err(CausalError::InvalidContract { .. })
    ));
    // The original entry survives the rejected second assert.
    assert_eq!(
        reg.get(&CausalAssumption::Sutva).unwrap().basis,
        AssumptionBasis::Unverified
    );
}

#[test]
fn overwrite_replaces_silently_on_purpose() {
    let mut reg = AssumptionRegistry::new();
    reg.assert(
        CausalAssumption::Positivity,
        AssumptionBasis::Unverified,
        None,
    )
    .unwrap();
    reg.overwrite(
        CausalAssumption::Positivity,
        AssumptionBasis::GuaranteedByDesign {
            mechanism: "randomized assignment".to_string(),
        },
        Some("checked post-hoc".to_string()),
    );
    assert!(matches!(
        reg.get(&CausalAssumption::Positivity).unwrap().basis,
        AssumptionBasis::GuaranteedByDesign { .. }
    ));
}

#[test]
fn is_supported_is_false_for_unverified_and_absent() {
    let mut reg = AssumptionRegistry::new();
    assert!(!reg.is_supported(&CausalAssumption::Faithfulness));
    reg.assert(
        CausalAssumption::Faithfulness,
        AssumptionBasis::Unverified,
        None,
    )
    .unwrap();
    assert!(!reg.is_supported(&CausalAssumption::Faithfulness));
}

#[test]
fn is_supported_is_true_for_any_non_unverified_basis() {
    let mut reg = AssumptionRegistry::new();
    reg.assert(
        CausalAssumption::CausalSufficiency,
        AssumptionBasis::DomainKnowledge {
            citation: "Pearl (2009)".to_string(),
        },
        None,
    )
    .unwrap();
    assert!(reg.is_supported(&CausalAssumption::CausalSufficiency));
}

#[test]
fn iteration_order_is_deterministic_regardless_of_insertion_order() {
    let mut a = AssumptionRegistry::new();
    a.assert(CausalAssumption::Sutva, AssumptionBasis::Unverified, None)
        .unwrap();
    a.assert(
        CausalAssumption::Acyclicity,
        AssumptionBasis::Unverified,
        None,
    )
    .unwrap();
    a.assert(
        CausalAssumption::Faithfulness,
        AssumptionBasis::Unverified,
        None,
    )
    .unwrap();

    let mut b = AssumptionRegistry::new();
    b.assert(
        CausalAssumption::Faithfulness,
        AssumptionBasis::Unverified,
        None,
    )
    .unwrap();
    b.assert(
        CausalAssumption::Acyclicity,
        AssumptionBasis::Unverified,
        None,
    )
    .unwrap();
    b.assert(CausalAssumption::Sutva, AssumptionBasis::Unverified, None)
        .unwrap();

    let a_keys: Vec<_> = a.iter().map(|(k, _)| k.clone()).collect();
    let b_keys: Vec<_> = b.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(
        a_keys, b_keys,
        "insertion order must not leak into iteration order"
    );
}

#[test]
fn other_variant_supports_freeform_assumptions() {
    let mut reg = AssumptionRegistry::new();
    reg.assert(
        CausalAssumption::Other("no measurement error".to_string()),
        AssumptionBasis::AssertedByAnalyst,
        None,
    )
    .unwrap();
    assert!(reg.is_supported(&CausalAssumption::Other("no measurement error".to_string())));
    assert!(!reg.is_supported(&CausalAssumption::Other("a different one".to_string())));
}

#[test]
fn json_round_trips() {
    let mut reg = AssumptionRegistry::new();
    reg.assert(
        CausalAssumption::InvarianceAcrossEnvironments,
        AssumptionBasis::TestedStatistically {
            test_name: "ICP".to_string(),
            p_value: Some(0.03),
        },
        Some("borderline".to_string()),
    )
    .unwrap();
    let json = serde_json::to_string(&reg).unwrap();
    let back: AssumptionRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(reg, back);
}
