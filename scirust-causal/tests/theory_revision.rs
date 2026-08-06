//! Withdrawing a claim when the ground moves.
//!
//! The headline test is the program's own loose end, tied off. Phase 5C.4's
//! adversarial fixture produces a certificate that says `Identifiable` and
//! carries an estimate wrong by more than ten standard errors, because a
//! latent confounder violates the causal sufficiency it assumed. Phase 5C.6
//! showed Invariant Causal Prediction *detecting* that on the same data.
//! Nothing connected them: the detection landed in one object and the bad
//! certificate sat in another, unchanged and still quotable.
//!
//! Here the detection is fed back as evidence, and the certificate stops
//! standing.
//!
//! The second thing these tests pin down is *how far* that goes. ICP's own
//! documentation says it detects that something is wrong and cannot say what.
//! So the certificate must come back `InDoubt`, not `Retracted` — retracting
//! it would claim an attribution the test never made. Getting that boundary
//! right is the point of the phase.

use scirust_causal::{
    AdjustmentStrategy, AssumptionBasis, AssumptionEvidence, AssumptionRegistry, CausalAssumption,
    CausalCertificate, CausalDataset, CausalError, CausalVariable, CertificateStanding,
    EffectEstimationConfig, Environment, EvidenceDirection, IdentifiabilityStatus, Intervention,
    InterventionKind, InvarianceConfig, InvariantPredictionOutcome, RevisionVerdict, SampleBlock,
    VariableKind, VariableRole, audit_certificate, estimate_effect_from_dag,
    evidence_from_invariance, invariant_causal_prediction, revise_assumptions,
};
use scirust_graph::dag::CausalDag;
use scirust_stats::SplitMix64;

// ─── The 5C.4 / 5C.6 fixture, unchanged ──────────────────────────────────

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

fn block(environment: Environment, columns: &[Vec<f64>]) -> SampleBlock {
    let n = columns[0].len();
    let d = columns.len();
    let mut data = vec![0.0; n * d];
    for row in 0..n
    {
        for col in 0..d
        {
            data[row * d + col] = columns[col][row];
        }
    }
    SampleBlock::new(environment, n, d, data).unwrap()
}

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

fn dag_from_edges(n: usize, edges: &[(usize, usize)]) -> CausalDag {
    let mut dag = CausalDag::new(n);
    for &(u, v) in edges
    {
        dag.add_directed_edge(u, v).unwrap();
    }
    dag
}

/// `U` confounds `X` and `Y`; `U` is never measured. The true effect of `X`
/// on `Y` is `0.7`. The second environment sets `X` independently of `U`.
fn hidden_confounder_with_intervention(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let (mut x_obs, mut y_obs) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 0.9 * u + 0.3 * noise(&mut rng);
        let y = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
        x_obs.push(x);
        y_obs.push(y);
    }
    let (mut x_int, mut y_int) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 2.0 * noise(&mut rng);
        let y = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
        x_int.push(x);
        y_int.push(y);
    }
    CausalDataset::new(
        variables(2),
        vec![
            block(
                Environment::observational("baseline").unwrap(),
                &[x_obs, y_obs],
            ),
            block(
                Environment::new(
                    "x_intervened",
                    vec![Intervention::new(0, InterventionKind::Unspecified).unwrap()],
                )
                .unwrap(),
                &[x_int, y_int],
            ),
        ],
        "test fixture",
    )
    .unwrap()
}

/// The registry an analyst running the 5C.4 estimate would have written: the
/// key assumption asserted, not checked.
fn analyst_registry() -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    registry
        .assert(
            CausalAssumption::CausalSufficiency,
            AssumptionBasis::AssertedByAnalyst,
            Some("no unmeasured common cause of X and Y was thought to exist".to_string()),
        )
        .unwrap();
    registry
        .assert(
            CausalAssumption::CorrectFunctionalForm,
            AssumptionBasis::AssertedByAnalyst,
            None,
        )
        .unwrap();
    registry
        .assert(
            CausalAssumption::InvarianceAcrossEnvironments,
            AssumptionBasis::Unverified,
            None,
        )
        .unwrap();
    registry
        .assert(
            CausalAssumption::Acyclicity,
            AssumptionBasis::DomainKnowledge {
                citation: "the process is a one-way pipeline".to_string(),
            },
            None,
        )
        .unwrap();
    registry
}

// ─── The headline ────────────────────────────────────────────────────────

#[test]
fn the_confidently_wrong_estimate_stops_standing_once_icp_reports_in() {
    let dataset = hidden_confounder_with_intervention(81, 1500);
    let true_effect = 0.7;

    // 1. The claim, made in good faith under stated assumptions, and wrong.
    let backdoor = estimate_effect_from_dag(
        &dataset,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    assert_eq!(
        backdoor.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    let biased = backdoor.estimate.unwrap();
    assert!(
        (biased - true_effect).abs() > 10.0 * backdoor.standard_error.unwrap(),
        "precondition: the estimate should be wrong by many standard errors, got {biased}"
    );

    // 2. Evidence arrives. ICP, on the same data with no graph, finds nothing
    //    invariant.
    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    assert_eq!(icp.outcome, InvariantPredictionOutcome::AssumptionsViolated);

    let evidence = evidence_from_invariance(&icp);
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].direction, EvidenceDirection::Contradicts);
    assert_eq!(
        evidence[0].assumptions.len(),
        3,
        "ICP falsifies a conjunction; recording it against one assumption would invent an \
         attribution it never made"
    );

    // 3. The registry is revised. No single assumption is blamed.
    let revision = revise_assumptions(&analyst_registry(), &evidence).unwrap();
    assert!(
        revision.contradicted().is_empty(),
        "nothing is directly attributable here"
    );
    assert_eq!(revision.jointly_contradicted().len(), 3);
    // Acyclicity was not part of what ICP tested, and is left alone.
    let acyclicity = revision
        .outcomes
        .iter()
        .find(|o| o.assumption == CausalAssumption::Acyclicity)
        .unwrap();
    assert_eq!(acyclicity.verdict, RevisionVerdict::Untouched);
    assert_eq!(acyclicity.prior_basis, acyclicity.revised_basis);

    // 4. And the claim stops standing.
    let audit = audit_certificate(&backdoor.certificate, &revision).unwrap();
    assert!(
        matches!(audit.standing, CertificateStanding::InDoubt { .. }),
        "expected InDoubt, got {:?}",
        audit.standing
    );
    assert!(
        !audit.standing.estimate_is_usable(),
        "the number must not survive its premises"
    );
    assert_eq!(audit.original_estimate, Some(biased));
    assert_eq!(
        audit.certificate.status(),
        IdentifiabilityStatus::Inconclusive
    );
    assert!(
        audit
            .warnings
            .iter()
            .any(|w| w.contains("must not be quoted"))
    );
}

#[test]
fn in_doubt_is_not_retracted_because_icp_cannot_attribute() {
    // The distinction the phase exists to hold. A test that *can* attribute
    // produces a retraction on the same certificate; ICP's cannot, so it
    // produces a doubt. Same claim, same registry, different epistemic force.
    let dataset = hidden_confounder_with_intervention(81, 1500);
    let backdoor = estimate_effect_from_dag(
        &dataset,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    let from_icp =
        revise_assumptions(&analyst_registry(), &evidence_from_invariance(&icp)).unwrap();
    let doubted = audit_certificate(&backdoor.certificate, &from_icp).unwrap();

    // Now suppose the confounder is actually found and measured. That is
    // attributable, and it retracts.
    let attributable = revise_assumptions(
        &analyst_registry(),
        &[AssumptionEvidence::attributed(
            CausalAssumption::CausalSufficiency,
            EvidenceDirection::Contradicts,
            "follow-up measurement",
            "the omitted common cause U was located and measured; it predicts both X and Y",
        )],
    )
    .unwrap();
    let retracted = audit_certificate(&backdoor.certificate, &attributable).unwrap();

    assert!(matches!(
        doubted.standing,
        CertificateStanding::InDoubt { .. }
    ));
    assert!(matches!(
        retracted.standing,
        CertificateStanding::Retracted { .. }
    ));
    // Both block the number; they differ in what they let you say about why.
    assert!(!doubted.standing.estimate_is_usable());
    assert!(!retracted.standing.estimate_is_usable());
}

// ─── Honesty properties ──────────────────────────────────────────────────

#[test]
fn a_successful_icp_run_corroborates_without_proving() {
    // Data where the invariance premise actually holds: no confounder.
    let mut rng = SplitMix64::new(404);
    let n = 1200;
    let build = |rng: &mut SplitMix64, scale: f64| {
        let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n
        {
            let xi = scale * noise(rng);
            y.push(0.7 * xi + 0.3 * noise(rng));
            x.push(xi);
        }
        [x, y]
    };
    let first = build(&mut rng, 1.0);
    let second = build(&mut rng, 3.0);
    let dataset = CausalDataset::new(
        variables(2),
        vec![
            block(Environment::observational("a").unwrap(), &first),
            block(
                Environment::new(
                    "b",
                    vec![Intervention::new(0, InterventionKind::Unspecified).unwrap()],
                )
                .unwrap(),
                &second,
            ),
        ],
        "test fixture",
    )
    .unwrap();

    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    assert_ne!(
        icp.outcome,
        InvariantPredictionOutcome::AssumptionsViolated,
        "precondition: this fixture should not trip the misspecification flag"
    );

    let revision =
        revise_assumptions(&analyst_registry(), &evidence_from_invariance(&icp)).unwrap();
    let sufficiency = revision
        .outcomes
        .iter()
        .find(|o| o.assumption == CausalAssumption::CausalSufficiency)
        .unwrap();
    assert!(
        !sufficiency.verdict.undermines(),
        "a clean ICP run must not undermine anything, got {:?}",
        sufficiency.verdict
    );
    // Whatever it did, it did not turn an assumption into a fact: the basis
    // is at best "tested", never a guarantee.
    assert!(
        !matches!(
            sufficiency.revised_basis,
            AssumptionBasis::GuaranteedByDesign { .. }
        ),
        "corroboration must never manufacture a design guarantee"
    );
}

#[test]
fn a_retraction_is_itself_a_citable_certificate() {
    let claim = CausalCertificate::builder(
        "effect of treatment on outcome",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::CausalSufficiency],
        "backdoor adjustment on 1500 rows",
    )
    .with_estimate("linear adjustment", 1.4949, 0.05)
    .finalize()
    .unwrap();

    let revision = revise_assumptions(
        &analyst_registry(),
        &[AssumptionEvidence::attributed(
            CausalAssumption::CausalSufficiency,
            EvidenceDirection::Contradicts,
            "follow-up measurement",
            "the confounder was found",
        )],
    )
    .unwrap();
    let audit = audit_certificate(&claim, &revision).unwrap();

    // The audit's own certificate carries the same discipline as the claim it
    // withdraws: a fingerprint, the assumptions at stake, and no estimate.
    assert!(!audit.certificate.fingerprint().is_empty());
    assert_eq!(audit.certificate.estimate(), None);
    assert_eq!(
        audit.certificate.status(),
        IdentifiabilityStatus::NotIdentifiable
    );
    assert!(
        audit
            .certificate
            .unresolved_alternatives()
            .iter()
            .any(|a| a.contains("no longer supports this claim"))
    );
    // The original number is preserved for the record, not erased.
    assert_eq!(audit.original_estimate, Some(1.4949));
    assert_eq!(audit.original_status, IdentifiabilityStatus::Identifiable);
}

#[test]
fn a_claim_resting_on_untouched_assumptions_still_stands() {
    let claim = CausalCertificate::builder(
        "a claim about acyclicity only",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Acyclicity],
        "domain knowledge",
    )
    .finalize()
    .unwrap();

    let dataset = hidden_confounder_with_intervention(81, 1500);
    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    let revision =
        revise_assumptions(&analyst_registry(), &evidence_from_invariance(&icp)).unwrap();

    let audit = audit_certificate(&claim, &revision).unwrap();
    assert_eq!(audit.standing, CertificateStanding::Stands);
    assert!(audit.standing.estimate_is_usable());
    // But "stands" is carefully worded.
    assert!(
        audit
            .certificate
            .sensitivity_note()
            .unwrap()
            .contains("not that the original claim was ever correct")
    );
}

#[test]
fn a_claim_citing_an_unregistered_assumption_is_unauditable_not_fine() {
    let claim = CausalCertificate::builder(
        "a claim resting on positivity",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Positivity],
        "somewhere else",
    )
    .with_estimate("m", 1.0, 0.1)
    .finalize()
    .unwrap();

    let revision = revise_assumptions(&analyst_registry(), &[]).unwrap();
    let audit = audit_certificate(&claim, &revision).unwrap();
    assert_eq!(
        audit.standing,
        CertificateStanding::Unauditable {
            unknown: vec![CausalAssumption::Positivity],
        }
    );
    assert!(
        !audit.standing.estimate_is_usable(),
        "an undetermined standing is not permission"
    );
}

#[test]
fn the_revised_registry_keeps_absence_of_evidence_apart_from_evidence_of_absence() {
    let dataset = hidden_confounder_with_intervention(81, 1500);
    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    let revision =
        revise_assumptions(&analyst_registry(), &evidence_from_invariance(&icp)).unwrap();

    // Touched by the falsified conjunction.
    assert!(matches!(
        revision
            .revised_registry
            .get(&CausalAssumption::CausalSufficiency)
            .unwrap()
            .basis,
        AssumptionBasis::ContradictedByEvidence { .. }
    ));
    assert!(
        !revision
            .revised_registry
            .is_supported(&CausalAssumption::CausalSufficiency)
    );

    // Never examined: still domain knowledge, still supported.
    assert!(
        revision
            .revised_registry
            .is_supported(&CausalAssumption::Acyclicity)
    );
}

#[test]
fn results_are_deterministic_and_json_round_trip() {
    let dataset = hidden_confounder_with_intervention(81, 800);
    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    let evidence = evidence_from_invariance(&icp);

    let first = revise_assumptions(&analyst_registry(), &evidence).unwrap();
    let second = revise_assumptions(&analyst_registry(), &evidence).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.certificate.fingerprint(),
        second.certificate.fingerprint()
    );

    let encoded = serde_json::to_string(&first).unwrap();
    assert_eq!(
        serde_json::from_str::<scirust_causal::AssumptionRevision>(&encoded).unwrap(),
        first
    );

    let claim = CausalCertificate::builder(
        "q",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::CausalSufficiency],
        "e",
    )
    .finalize()
    .unwrap();
    let audit = audit_certificate(&claim, &first).unwrap();
    let encoded = serde_json::to_string(&audit).unwrap();
    assert_eq!(
        serde_json::from_str::<scirust_causal::CertificateAudit>(&encoded).unwrap(),
        audit
    );
}

#[test]
fn malformed_evidence_is_a_typed_error() {
    assert!(matches!(
        revise_assumptions(
            &analyst_registry(),
            &[AssumptionEvidence::joint(
                vec![],
                EvidenceDirection::Contradicts,
                "s",
                "d",
            )]
        ),
        Err(CausalError::InvalidContract { .. })
    ));
    assert!(matches!(
        revise_assumptions(
            &analyst_registry(),
            &[AssumptionEvidence::attributed(
                CausalAssumption::Acyclicity,
                EvidenceDirection::Supports,
                "",
                "d",
            )]
        ),
        Err(CausalError::InvalidContract { .. })
    ));
}
