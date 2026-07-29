//! Auditing the crate against its own contract.
//!
//! Two of these tests are about holes that were open until this phase, and
//! were found by probing rather than by reasoning.
//!
//! [`CausalCertificateBuilder::finalize`] structurally forbids attaching a
//! numeric estimate to any status other than `Identifiable`. That rule is the
//! reason phase 5C.1's certificate layer exists, and every phase since has
//! leaned on it. But `CausalCertificate` used to *derive* `Deserialize`, and
//! serde populates private fields directly — so any JSON could produce a
//! certificate the builder would have rejected, and every phase of this crate
//! serializes certificates. Separately, the fingerprint was a stored string
//! nothing ever recomputed, so editing an estimate left it intact and
//! accepted.
//!
//! `certificate_deserialization_cannot_bypass_the_coherence_rule` and
//! `an_edited_estimate_no_longer_passes_unnoticed` are those two probes,
//! turned into regression tests.
//!
//! The third headline is cheaper and, in its way, more pointed: the crate's own
//! adversarial fixture — the estimate wrong by tens of standard errors that
//! 5C.5 quantified, 5C.6 detected and 5C.9 withdrew — is flagged here from its
//! provenance metadata alone, before any test is run.

use scirust_causal::{
    AdjustmentStrategy, AssumptionBasis, AssumptionRegistry, CausalAssumption, CausalCertificate,
    CausalDataset, CausalError, CausalVariable, ClaimAuditConfig, ClaimAuditVerdict, ClaimFinding,
    EffectEstimationConfig, Environment, FindingSeverity, IdentifiabilityStatus, MethodRequirement,
    SampleBlock, VariableKind, VariableRole, audit_claim_set, estimate_effect_from_dag,
};
use scirust_graph::dag::CausalDag;
use scirust_stats::SplitMix64;

// ─── Fixtures ────────────────────────────────────────────────────────────

fn variables(d: usize) -> Vec<CausalVariable> {
    (0..d)
        .map(|i| {
            CausalVariable::new(
                i,
                format!("v{i}"),
                VariableRole::Unspecified,
                VariableKind::Continuous,
            )
            .unwrap()
        })
        .collect()
}

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

/// The 5C.4 adversarial fixture: `U` confounds `X` and `Y` and is never
/// measured, so the true effect `0.7` is estimated as roughly `1.5`.
fn hidden_confounder(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let mut data = vec![0.0; n * 2];
    for row in 0..n
    {
        let u = noise(&mut rng);
        let x = 0.9 * u + 0.3 * noise(&mut rng);
        data[row * 2] = x;
        data[row * 2 + 1] = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
    }
    CausalDataset::new(
        variables(2),
        vec![
            SampleBlock::new(Environment::observational("baseline").unwrap(), n, 2, data).unwrap(),
        ],
        "test fixture",
    )
    .unwrap()
}

fn backdoor_claim() -> CausalCertificate {
    let dataset = hidden_confounder(81, 1500);
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap();
    estimate_effect_from_dag(
        &dataset,
        &dag,
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap()
    .certificate
}

/// Every assumption the effect estimator declares, each merely asserted.
fn asserted_registry() -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    for assumption in [
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::Positivity,
        CausalAssumption::AdequateSampleSize,
    ]
    {
        registry
            .assert(assumption, AssumptionBasis::AssertedByAnalyst, None)
            .unwrap();
    }
    registry
}

// ─── The two holes, now closed ───────────────────────────────────────────

#[test]
fn certificate_deserialization_cannot_bypass_the_coherence_rule() {
    let claim = backdoor_claim();
    assert_eq!(claim.status(), IdentifiabilityStatus::Identifiable);
    assert!(claim.estimate().is_some());

    // Flip the status while keeping the estimate. The builder makes this
    // impossible; a derived Deserialize would have waved it through.
    let json = serde_json::to_string(&claim).unwrap();
    let flipped = json.replace(r#""status":"Identifiable""#, r#""status":"Inconclusive""#);
    assert_ne!(flipped, json, "the fixture must actually have been edited");

    let parsed = serde_json::from_str::<CausalCertificate>(&flipped);
    assert!(
        parsed.is_err(),
        "deserialization must enforce the same rule as the builder"
    );
    assert!(
        parsed.unwrap_err().to_string().contains("Identifiable"),
        "the error should name the rule it is protecting"
    );

    // A legitimate round trip is unaffected.
    assert_eq!(
        serde_json::from_str::<CausalCertificate>(&json).unwrap(),
        claim
    );
}

#[test]
fn an_edited_estimate_no_longer_passes_unnoticed() {
    let claim = backdoor_claim();
    let original = claim.estimate().unwrap();
    let json = serde_json::to_string(&claim).unwrap();

    // Editing the estimate keeps the coherence rule satisfied -- status is
    // still Identifiable -- so only the fingerprint can catch it.
    let tampered = json.replace(&format!(r#""estimate":{original}"#), r#""estimate":99.0"#);
    assert_ne!(tampered, json, "the fixture must actually have been edited");

    let altered: CausalCertificate = serde_json::from_str(&tampered).unwrap();
    assert_eq!(altered.estimate(), Some(99.0));
    assert_eq!(
        altered.fingerprint(),
        claim.fingerprint(),
        "the stored fingerprint is carried along unchanged -- that is the point"
    );
    assert!(
        !altered.verify_fingerprint(),
        "recomputing it must expose the edit"
    );
    assert!(
        claim.verify_fingerprint(),
        "and an untouched certificate must still verify"
    );

    let audit = audit_claim_set(
        &[altered],
        &asserted_registry(),
        &ClaimAuditConfig::default(),
    )
    .unwrap();
    assert!(
        audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::FingerprintMismatch { .. }))
    );
    assert!(matches!(
        audit.verdict,
        ClaimAuditVerdict::Violations { .. }
    ));
}

// ─── The provenance signal ───────────────────────────────────────────────

#[test]
fn the_wrong_estimate_is_flaggable_from_provenance_before_any_test_runs() {
    // This is the same certificate 5C.5 quantified, 5C.6 detected and 5C.9
    // withdrew -- each by measuring something. Here it is flagged with no data
    // read at all: it quotes a number, and every assumption beneath it is the
    // analyst's word.
    let audit = audit_claim_set(
        &[backdoor_claim()],
        &asserted_registry(),
        &ClaimAuditConfig::default(),
    )
    .unwrap();

    let flagged: Vec<_> = audit
        .findings
        .iter()
        .filter_map(|f| match f
        {
            ClaimFinding::EstimateOnUnbackedProvenance { unbacked, .. } => Some(unbacked.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(flagged.len(), 1, "findings were {:?}", audit.findings);
    assert!(flagged[0].contains(&CausalAssumption::CausalSufficiency));

    // A warning, not a violation. Asserting an assumption is allowed; the
    // finding is about what is behind the number, not about a broken rule.
    assert_eq!(
        audit
            .findings
            .iter()
            .filter(|f| f.severity() == FindingSeverity::Violation)
            .count(),
        0
    );
    assert!(matches!(
        audit.verdict,
        ClaimAuditVerdict::WarningsOnly { .. }
    ));
}

#[test]
fn the_same_claim_clears_provenance_once_one_assumption_is_actually_backed() {
    // The signal is deliberately weak: it fires only when a claim rests on
    // nothing at all. Backing a single assumption clears it, which is why this
    // is a first look and not a verdict.
    let mut registry = asserted_registry();
    registry.overwrite(
        CausalAssumption::Acyclicity,
        AssumptionBasis::DomainKnowledge {
            citation: "the process is a one-way pipeline".to_string(),
        },
        None,
    );
    let audit =
        audit_claim_set(&[backdoor_claim()], &registry, &ClaimAuditConfig::default()).unwrap();
    assert!(
        !audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::EstimateOnUnbackedProvenance { .. }))
    );
}

// ─── Contract compliance of the crate's own output ───────────────────────

#[test]
fn the_crates_own_effect_certificate_declares_what_its_method_requires() {
    // The default requirement table says a backdoor method must declare
    // acyclicity, causal sufficiency, correct functional form and positivity.
    // The estimator's own certificate is checked against that here, so the
    // table and the implementation cannot drift apart silently.
    let mut registry = AssumptionRegistry::new();
    for assumption in [
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::Positivity,
        CausalAssumption::AdequateSampleSize,
    ]
    {
        registry
            .assert(
                assumption,
                AssumptionBasis::GuaranteedByDesign {
                    mechanism: "randomized assignment".to_string(),
                },
                None,
            )
            .unwrap();
    }
    let audit =
        audit_claim_set(&[backdoor_claim()], &registry, &ClaimAuditConfig::default()).unwrap();
    assert_eq!(
        audit.verdict,
        ClaimAuditVerdict::Compliant,
        "the crate's own certificate should pass its own audit: {:?}",
        audit.findings
    );
}

#[test]
fn a_compliant_verdict_is_carefully_worded() {
    let mut registry = AssumptionRegistry::new();
    for assumption in [
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::Positivity,
        CausalAssumption::AdequateSampleSize,
    ]
    {
        registry
            .assert(
                assumption,
                AssumptionBasis::GuaranteedByDesign {
                    mechanism: "randomized assignment".to_string(),
                },
                None,
            )
            .unwrap();
    }
    let audit =
        audit_claim_set(&[backdoor_claim()], &registry, &ClaimAuditConfig::default()).unwrap();
    let note = audit.certificate.sensitivity_note().unwrap();
    assert!(note.contains("reads no data"));
    assert!(note.contains("can be uniformly wrong"));
    assert_eq!(audit.certificate.estimate(), None);
}

#[test]
fn a_caller_method_outside_the_table_is_reported_not_assumed_compliant() {
    let bespoke = CausalCertificate::builder(
        "effect of A on B",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Acyclicity],
        "a procedure this crate knows nothing about",
    )
    .with_estimate("in-house instrumental variable estimator", 0.4, 0.05)
    .finalize()
    .unwrap();

    let mut registry = AssumptionRegistry::new();
    registry
        .assert(
            CausalAssumption::Acyclicity,
            AssumptionBasis::DomainKnowledge {
                citation: "x".to_string(),
            },
            None,
        )
        .unwrap();

    let audit = audit_claim_set(
        std::slice::from_ref(&bespoke),
        &registry,
        &ClaimAuditConfig::default(),
    )
    .unwrap();
    assert!(
        audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::UnrecognizedMethod { .. })),
        "an unknown method must not read as a compliant one"
    );

    // Registering it removes the finding and starts enforcing its
    // requirements instead.
    let with_table = ClaimAuditConfig {
        method_requirements: vec![MethodRequirement {
            method_key: "instrumental variable".to_string(),
            required: vec![
                CausalAssumption::Acyclicity,
                CausalAssumption::Exchangeability,
            ],
        }],
    };
    let audit = audit_claim_set(&[bespoke], &registry, &with_table).unwrap();
    assert!(
        !audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::UnrecognizedMethod { .. }))
    );
    assert!(audit.findings.iter().any(|f| matches!(
        f,
        ClaimFinding::UndeclaredAssumption {
            missing: CausalAssumption::Exchangeability,
            ..
        }
    )));
}

#[test]
fn two_estimates_of_the_same_effect_that_disagree_are_a_violation() {
    // The confounded estimate and a corrected one, both claiming to answer the
    // same question. Whichever is right, presenting both is incoherent.
    let confounded = backdoor_claim();
    let query = confounded.query().to_string();
    let corrected = CausalCertificate::builder(
        query,
        IdentifiabilityStatus::Identifiable,
        confounded.assumptions_used().to_vec(),
        "adjusted for the confounder once it was measured",
    )
    .with_estimate("linear backdoor adjustment", 0.7, 0.01)
    .finalize()
    .unwrap();

    let audit = audit_claim_set(
        &[confounded, corrected],
        &asserted_registry(),
        &ClaimAuditConfig::default(),
    )
    .unwrap();
    assert!(
        audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::ConflictingClaims { .. }))
    );
    assert!(matches!(
        audit.verdict,
        ClaimAuditVerdict::Violations { .. }
    ));
}

#[test]
fn results_are_deterministic_and_json_round_trip() {
    let claims = [backdoor_claim()];
    let registry = asserted_registry();
    let config = ClaimAuditConfig::default();
    let first = audit_claim_set(&claims, &registry, &config).unwrap();
    let second = audit_claim_set(&claims, &registry, &config).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.certificate.fingerprint(),
        second.certificate.fingerprint()
    );
    let encoded = serde_json::to_string(&first).unwrap();
    assert_eq!(
        serde_json::from_str::<scirust_causal::ClaimAudit>(&encoded).unwrap(),
        first
    );
}

#[test]
fn a_malformed_requirement_table_is_a_typed_error() {
    assert!(matches!(
        audit_claim_set(
            &[],
            &asserted_registry(),
            &ClaimAuditConfig {
                method_requirements: vec![MethodRequirement {
                    method_key: String::new(),
                    required: vec![],
                }],
            }
        ),
        Err(CausalError::InvalidContract { .. })
    ));
}
