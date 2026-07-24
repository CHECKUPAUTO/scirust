use scirust_causal::{CausalAssumption, CausalCertificate, CausalError, IdentifiabilityStatus};

#[test]
fn identifiable_certificate_can_carry_an_estimate() {
    let cert = CausalCertificate::builder(
        "ATE of X0 on X2",
        IdentifiabilityStatus::Identifiable,
        vec![
            CausalAssumption::CausalSufficiency,
            CausalAssumption::Acyclicity,
        ],
        "20 samples, single observational environment",
    )
    .with_estimate("backdoor adjustment", 1.23, 0.05)
    .finalize()
    .unwrap();

    assert_eq!(cert.status(), IdentifiabilityStatus::Identifiable);
    assert_eq!(cert.estimate(), Some(1.23));
    assert_eq!(cert.uncertainty(), Some(0.05));
    assert_eq!(cert.method(), Some("backdoor adjustment"));
}

#[test]
fn not_identifiable_certificate_cannot_carry_an_estimate() {
    let result = CausalCertificate::builder(
        "ATE of X0 on X2",
        IdentifiabilityStatus::NotIdentifiable,
        vec![],
        "unblocked backdoor path, no valid adjustment set",
    )
    .with_estimate("backdoor adjustment", 1.23, 0.05)
    .finalize();

    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}

#[test]
fn equivalence_class_only_cannot_carry_an_estimate() {
    let result = CausalCertificate::builder(
        "direction of X0 -> X1",
        IdentifiabilityStatus::EquivalenceClassOnly,
        vec![CausalAssumption::Faithfulness],
        "observational data, CPDAG only",
    )
    .with_estimate("score", 0.7, 0.1)
    .finalize();

    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}

#[test]
fn inconclusive_cannot_carry_an_estimate() {
    let result = CausalCertificate::builder(
        "ATE of X0 on X2",
        IdentifiabilityStatus::Inconclusive,
        vec![],
        "identifiability check not run",
    )
    .with_estimate("naive", 0.0, 0.0)
    .finalize();

    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}

#[test]
fn negative_result_without_an_estimate_is_fine() {
    let cert = CausalCertificate::builder(
        "ATE of X0 on X2",
        IdentifiabilityStatus::NotIdentifiable,
        vec![CausalAssumption::CausalSufficiency],
        "unblocked backdoor path",
    )
    .with_unresolved_alternative("latent confounder Z")
    .finalize()
    .unwrap();

    assert_eq!(cert.estimate(), None);
    assert_eq!(cert.method(), None);
    assert_eq!(
        cert.unresolved_alternatives(),
        &["latent confounder Z".to_string()]
    );
}

#[test]
fn rejects_empty_query() {
    let result = CausalCertificate::builder(
        "   ",
        IdentifiabilityStatus::Inconclusive,
        vec![],
        "evidence",
    )
    .finalize();
    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}

#[test]
fn rejects_non_finite_estimate() {
    let result = CausalCertificate::builder("q", IdentifiabilityStatus::Identifiable, vec![], "e")
        .with_estimate("m", f64::NAN, 0.1)
        .finalize();
    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}

#[test]
fn rejects_negative_uncertainty() {
    let result = CausalCertificate::builder("q", IdentifiabilityStatus::Identifiable, vec![], "e")
        .with_estimate("m", 1.0, -0.1)
        .finalize();
    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}

#[test]
fn assumptions_used_are_sorted_and_deduped_independent_of_input_order() {
    let a = CausalCertificate::builder(
        "q",
        IdentifiabilityStatus::Inconclusive,
        vec![
            CausalAssumption::Faithfulness,
            CausalAssumption::Acyclicity,
            CausalAssumption::Faithfulness,
        ],
        "e",
    )
    .finalize()
    .unwrap();

    let b = CausalCertificate::builder(
        "q",
        IdentifiabilityStatus::Inconclusive,
        vec![CausalAssumption::Acyclicity, CausalAssumption::Faithfulness],
        "e",
    )
    .finalize()
    .unwrap();

    assert_eq!(a.assumptions_used(), b.assumptions_used());
    assert_eq!(a.assumptions_used().len(), 2);
}

#[test]
fn fingerprint_is_deterministic_and_content_addressed() {
    let build = || {
        CausalCertificate::builder(
            "ATE of X0 on X2",
            IdentifiabilityStatus::Identifiable,
            vec![CausalAssumption::Acyclicity],
            "evidence",
        )
        .with_estimate("m", 1.0, 0.1)
        .finalize()
        .unwrap()
    };
    let cert1 = build();
    let cert2 = build();
    assert_eq!(cert1.fingerprint(), cert2.fingerprint());
    assert!(!cert1.fingerprint().is_empty());

    // Changing any semantic content changes the fingerprint.
    let cert3 = CausalCertificate::builder(
        "ATE of X0 on X2",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Acyclicity],
        "evidence",
    )
    .with_estimate("m", 1.0000001, 0.1)
    .finalize()
    .unwrap();
    assert_ne!(cert1.fingerprint(), cert3.fingerprint());
}

#[test]
fn fingerprint_is_order_independent_over_assumptions_used() {
    let cert_ordered = CausalCertificate::builder(
        "q",
        IdentifiabilityStatus::Inconclusive,
        vec![CausalAssumption::Acyclicity, CausalAssumption::Faithfulness],
        "e",
    )
    .finalize()
    .unwrap();
    let cert_reordered = CausalCertificate::builder(
        "q",
        IdentifiabilityStatus::Inconclusive,
        vec![CausalAssumption::Faithfulness, CausalAssumption::Acyclicity],
        "e",
    )
    .finalize()
    .unwrap();
    assert_eq!(cert_ordered.fingerprint(), cert_reordered.fingerprint());
}

#[test]
fn json_round_trips() {
    let cert = CausalCertificate::builder(
        "ATE of X0 on X2",
        IdentifiabilityStatus::Identifiable,
        vec![
            CausalAssumption::Acyclicity,
            CausalAssumption::Other("no measurement error".to_string()),
        ],
        "evidence",
    )
    .with_estimate("m", 1.0, 0.1)
    .with_sensitivity("E-value 2.1")
    .with_unresolved_alternative("reverse causation")
    .finalize()
    .unwrap();

    let json = serde_json::to_string(&cert).unwrap();
    let back: CausalCertificate = serde_json::from_str(&json).unwrap();
    assert_eq!(cert, back);
    assert_eq!(cert.fingerprint(), back.fingerprint());
}
