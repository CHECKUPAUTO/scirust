//! Phase 5C.10 deterministic benchmark: auditing a claim set against the
//! crate's own contract.
//!
//! # What this demonstrates
//!
//! Three things, in increasing order of how much work they save.
//!
//! The `coherence_rule` and `tampered_estimate` rows exercise two holes this
//! phase closed. `CausalCertificateBuilder::finalize` forbids attaching an
//! estimate to a non-`Identifiable` status — the rule phase 5C.1's certificate
//! layer exists to enforce — but a derived `Deserialize` let any JSON produce
//! a certificate the builder would have rejected, and the fingerprint was a
//! stored string nothing recomputed. Both are now checked.
//!
//! The `asserted_provenance` row is the cheap one. The crate's own adversarial
//! certificate — the estimate 5C.5 quantified, 5C.6 detected and 5C.9 withdrew,
//! each by measuring something — is flagged here from its metadata alone, with
//! no data read. It is the weakest signal in the program and the earliest.
//!
//! # Reproducibility contract
//!
//! Fixed [`SplitMix64`] seed, no wall-clock/hostname/thread-count input. All
//! scientific content goes to **stdout** in a fixed field order; nothing else
//! does. Two runs hashed with SHA-256 must be byte-identical.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    AdjustmentStrategy, AssumptionBasis, AssumptionRegistry, CausalAssumption, CausalCertificate,
    CausalDataset, CausalVariable, ClaimAudit, ClaimAuditConfig, ClaimAuditVerdict, ClaimFinding,
    EffectEstimationConfig, Environment, FindingSeverity, IdentifiabilityStatus, MethodRequirement,
    SampleBlock, VariableKind, VariableRole, audit_claim_set, estimate_effect_from_dag,
};
use scirust_graph::dag::CausalDag;
use scirust_stats::SplitMix64;

fn expect(condition: bool, description: String) {
    if !condition
    {
        eprintln!("ORACLE FAILURE: {description}");
        std::process::exit(1);
    }
}

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

/// The 5C.4 adversarial fixture: `U` confounds `X` and `Y`, true effect `0.7`.
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
    let variables: Vec<CausalVariable> = (0..2)
        .map(|i| {
            CausalVariable::new(
                i,
                format!("v{i}"),
                VariableRole::Unspecified,
                VariableKind::Continuous,
            )
            .unwrap()
        })
        .collect();
    CausalDataset::new(
        variables,
        vec![
            SampleBlock::new(Environment::observational("baseline").unwrap(), n, 2, data).unwrap(),
        ],
        "5C.10 benchmark fixture",
    )
    .unwrap()
}

fn effect_assumptions() -> [CausalAssumption; 5] {
    [
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::Positivity,
        CausalAssumption::AdequateSampleSize,
    ]
}

fn registry_with(basis: &AssumptionBasis) -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    for assumption in effect_assumptions()
    {
        registry.assert(assumption, basis.clone(), None).unwrap();
    }
    registry
}

fn describe_verdict(verdict: &ClaimAuditVerdict) -> String {
    match verdict
    {
        ClaimAuditVerdict::Compliant => "Compliant".to_string(),
        ClaimAuditVerdict::Violations { count } => format!("Violations({count})"),
        ClaimAuditVerdict::WarningsOnly { count } => format!("WarningsOnly({count})"),
    }
}

/// A stable, order-independent tag for a finding's kind.
fn kind(finding: &ClaimFinding) -> &'static str {
    match finding
    {
        ClaimFinding::FingerprintMismatch { .. } => "FingerprintMismatch",
        ClaimFinding::ContradictedAssumptionCited { .. } => "ContradictedAssumptionCited",
        ClaimFinding::ConflictingClaims { .. } => "ConflictingClaims",
        ClaimFinding::UndeclaredAssumption { .. } => "UndeclaredAssumption",
        ClaimFinding::UnrecognizedMethod { .. } => "UnrecognizedMethod",
        ClaimFinding::UnregisteredAssumption { .. } => "UnregisteredAssumption",
        ClaimFinding::EstimateOnUnbackedProvenance { .. } => "EstimateOnUnbackedProvenance",
    }
}

fn report(scenario: &str, audit: &ClaimAudit) {
    let violations = audit
        .findings
        .iter()
        .filter(|f| f.severity() == FindingSeverity::Violation)
        .count();
    println!(
        "scenario={scenario} claims={} verdict={} violations={} warnings={} kinds={:?} \
         audit_status={:?}",
        audit.claims_examined,
        describe_verdict(&audit.verdict),
        violations,
        audit.findings.len() - violations,
        audit.findings.iter().map(kind).collect::<Vec<_>>(),
        audit.certificate.status(),
    );
}

fn main() {
    println!("# Phase 5C.10 deterministic claim-audit benchmark");
    println!("# fields: scenario claims verdict violations warnings kinds audit_status");
    println!(
        "# scope: this audits contract compliance and internal coherence; it reads no data and \
         evaluates no method, so a Compliant verdict is about the claims' form and provenance, \
         never about their correctness"
    );

    let default = ClaimAuditConfig::default();
    let dataset = hidden_confounder(81, 1500);
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap();
    let estimate = estimate_effect_from_dag(
        &dataset,
        &dag,
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    let claim = estimate.certificate.clone();
    let biased = estimate.estimate.unwrap();
    let standard_error = estimate.standard_error.unwrap();

    // ── The crate's own certificate, against a backed registry ───────────
    let backed = registry_with(&AssumptionBasis::GuaranteedByDesign {
        mechanism: "randomized assignment".to_string(),
    });
    let audit = audit_claim_set(std::slice::from_ref(&claim), &backed, &default).unwrap();
    report("crate_own_certificate", &audit);
    expect(
        audit.verdict == ClaimAuditVerdict::Compliant,
        format!(
            "crate_own_certificate: the crate's own output should pass its own audit, got {:?}",
            audit.findings
        ),
    );

    // ── The same certificate, resting on nothing but assertion ───────────
    let asserted = registry_with(&AssumptionBasis::AssertedByAnalyst);
    let audit = audit_claim_set(std::slice::from_ref(&claim), &asserted, &default).unwrap();
    report("asserted_provenance", &audit);
    println!(
        "# provenance_signal: this is the certificate that estimates {biased:.6e} against a truth \
         of 0.7 -- a {:.1}-standard-error miss. 5C.5 quantified it, 5C.6 detected it and 5C.9 \
         withdrew it, each by measuring something; this flags it from metadata alone, reading no \
         data. Weakest signal in the program, and the earliest.",
        (biased - 0.7).abs() / standard_error,
    );
    expect(
        matches!(audit.verdict, ClaimAuditVerdict::WarningsOnly { .. })
            && audit
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::EstimateOnUnbackedProvenance { .. })),
        format!(
            "asserted_provenance: expected a provenance warning, got {:?}",
            audit.findings
        ),
    );

    // ── Backing one assumption clears it: a first look, not a verdict ────
    let mut partly_backed = registry_with(&AssumptionBasis::AssertedByAnalyst);
    partly_backed.overwrite(
        CausalAssumption::Acyclicity,
        AssumptionBasis::DomainKnowledge {
            citation: "one-way pipeline".to_string(),
        },
        None,
    );
    let audit = audit_claim_set(std::slice::from_ref(&claim), &partly_backed, &default).unwrap();
    report("one_assumption_backed", &audit);
    expect(
        !audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::EstimateOnUnbackedProvenance { .. })),
        "one_assumption_backed: the check is for claims resting on nothing at all".to_string(),
    );

    // ── Hole 1: deserialization can no longer bypass the coherence rule ──
    let json = serde_json::to_string(&claim).unwrap();
    let flipped = json.replace(r#""status":"Identifiable""#, r#""status":"Inconclusive""#);
    let parsed = serde_json::from_str::<CausalCertificate>(&flipped);
    println!(
        "scenario=coherence_rule edited=status_flipped_estimate_kept rejected={} \
         legitimate_round_trip_ok={}",
        parsed.is_err(),
        serde_json::from_str::<CausalCertificate>(&json).unwrap() == claim,
    );
    expect(
        parsed.is_err(),
        "coherence_rule: deserialization must enforce the same rule as the builder".to_string(),
    );

    // ── Hole 2: an edited estimate no longer passes unnoticed ────────────
    let tampered_json = json.replace(&format!(r#""estimate":{biased}"#), r#""estimate":99.0"#);
    expect(
        tampered_json != json,
        "tampered_estimate: the fixture must actually have been edited".to_string(),
    );
    let tampered: CausalCertificate = serde_json::from_str(&tampered_json).unwrap();
    println!(
        "scenario=tampered_estimate original={biased:.12e} altered={:?} \
         fingerprint_carried_unchanged={} verify_fingerprint={} untouched_verifies={}",
        tampered.estimate(),
        tampered.fingerprint() == claim.fingerprint(),
        tampered.verify_fingerprint(),
        claim.verify_fingerprint(),
    );
    let audit = audit_claim_set(&[tampered], &backed, &default).unwrap();
    report("tampered_estimate_audit", &audit);
    println!(
        "# integrity_holes_closed: the coherence rule now holds for parsed certificates as well \
         as built ones, and the fingerprint is checkable rather than merely stored -- both were \
         open until this phase, and both were found by probing rather than by reasoning"
    );
    expect(
        audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::FingerprintMismatch { .. })),
        "tampered_estimate_audit: the edit must surface as a fingerprint mismatch".to_string(),
    );

    // ── A contradicted assumption still cited ────────────────────────────
    let mut contradicted = backed.clone();
    contradicted.overwrite(
        CausalAssumption::CausalSufficiency,
        AssumptionBasis::ContradictedByEvidence {
            source: "follow_up_measurement".to_string(),
            jointly_with: vec![],
        },
        None,
    );
    let audit = audit_claim_set(std::slice::from_ref(&claim), &contradicted, &default).unwrap();
    report("contradicted_assumption_cited", &audit);
    expect(
        matches!(audit.verdict, ClaimAuditVerdict::Violations { .. }),
        "contradicted_assumption_cited: expected a violation".to_string(),
    );

    // ── Two answers to one question ──────────────────────────────────────
    let corrected = CausalCertificate::builder(
        claim.query().to_string(),
        IdentifiabilityStatus::Identifiable,
        claim.assumptions_used().to_vec(),
        "adjusted once the confounder was measured",
    )
    .with_estimate("linear backdoor adjustment", 0.7, 0.01)
    .finalize()
    .unwrap();
    let audit = audit_claim_set(&[claim.clone(), corrected], &backed, &default).unwrap();
    report("conflicting_claims", &audit);
    expect(
        audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::ConflictingClaims { .. })),
        "conflicting_claims: two answers to one question must be flagged".to_string(),
    );

    // ── A method with no requirement table ───────────────────────────────
    let bespoke = CausalCertificate::builder(
        "effect of A on B",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Acyclicity],
        "a procedure this crate knows nothing about",
    )
    .with_estimate("in-house instrumental variable estimator", 0.4, 0.05)
    .finalize()
    .unwrap();
    let audit = audit_claim_set(std::slice::from_ref(&bespoke), &backed, &default).unwrap();
    report("unrecognized_method", &audit);
    let registered = ClaimAuditConfig {
        method_requirements: vec![MethodRequirement {
            method_key: "instrumental variable".to_string(),
            required: vec![
                CausalAssumption::Acyclicity,
                CausalAssumption::Exchangeability,
            ],
        }],
    };
    let audit = audit_claim_set(&[bespoke], &backed, &registered).unwrap();
    report("method_registered_then_enforced", &audit);
    println!(
        "# silence_is_not_compliance: an unknown method is reported rather than passed over; \
         registering it removes that finding and starts enforcing its requirements instead"
    );
    expect(
        audit
            .findings
            .iter()
            .any(|f| matches!(f, ClaimFinding::UndeclaredAssumption { .. })),
        "method_registered_then_enforced: the registered requirement must now bite".to_string(),
    );

    // ── An empty set is compliant, and says how many it saw ──────────────
    let audit = audit_claim_set(&[], &backed, &default).unwrap();
    report("empty_claim_set", &audit);
    println!(
        "# compliant_means: every Compliant above says the claims obey this crate's rules and \
         nothing more -- the audit reads no data, so a uniformly wrong claim set can pass"
    );
    expect(
        audit.verdict == ClaimAuditVerdict::Compliant && audit.claims_examined == 0,
        "empty_claim_set: expected a compliant empty audit".to_string(),
    );

    println!("# all oracle checks passed");
}
