//! Phase 5C.9 deterministic benchmark: what evidence does to stated
//! assumptions, and to the claims that rested on them.
//!
//! # What this demonstrates
//!
//! The first three rows tie off the program's own loose end. Phase 5C.4's
//! adversarial fixture produces a certificate that says `Identifiable` and
//! carries an estimate wrong by tens of standard errors. Phase 5C.6 showed
//! Invariant Causal Prediction detecting that on the same data. Nothing
//! connected them, so the bad certificate stayed quotable. Here the detection
//! is fed back as evidence and the claim stops standing.
//!
//! The `audit_*` rows mark the boundary that makes this honest rather than
//! merely dramatic. ICP falsifies a *conjunction* and its own documentation
//! says it cannot name the member that broke — so the claim comes back
//! `InDoubt`, not `Retracted`. Feeding in a finding that genuinely *is*
//! attributable retracts the same claim. Same certificate, same registry,
//! different epistemic force.
//!
//! # Reproducibility contract
//!
//! Fixed [`SplitMix64`] seeds, no wall-clock/hostname/thread-count input. All
//! scientific content goes to **stdout** in a fixed field order; nothing else
//! does. Two runs hashed with SHA-256 must be byte-identical.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    AdjustmentStrategy, AssumptionBasis, AssumptionEvidence, AssumptionRegistry,
    AssumptionRevision, CausalAssumption, CausalCertificate, CausalDataset, CausalVariable,
    CertificateAudit, CertificateStanding, EffectEstimationConfig, Environment, EvidenceDirection,
    IdentifiabilityStatus, Intervention, InterventionKind, InvarianceConfig,
    InvariantPredictionOutcome, RevisionVerdict, SampleBlock, VariableKind, VariableRole,
    audit_certificate, estimate_effect_from_dag, evidence_from_invariance,
    invariant_causal_prediction, revise_assumptions,
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

fn intervened(name: &str) -> Environment {
    Environment::new(
        name,
        vec![Intervention::new(0, InterventionKind::Unspecified).unwrap()],
    )
    .unwrap()
}

/// `U` confounds `X` and `Y` and is never measured. True effect `0.7`.
fn hidden_confounder(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let (mut x_obs, mut y_obs) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 0.9 * u + 0.3 * noise(&mut rng);
        y_obs.push(0.7 * x + 0.8 * u + 0.3 * noise(&mut rng));
        x_obs.push(x);
    }
    let (mut x_int, mut y_int) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 2.0 * noise(&mut rng);
        y_int.push(0.7 * x + 0.8 * u + 0.3 * noise(&mut rng));
        x_int.push(x);
    }
    CausalDataset::new(
        variables(2),
        vec![
            block(
                Environment::observational("baseline").unwrap(),
                &[x_obs, y_obs],
            ),
            block(intervened("x_intervened"), &[x_int, y_int]),
        ],
        "5C.9 benchmark fixture",
    )
    .unwrap()
}

/// No confounder: the invariance premise actually holds.
fn clean(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
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
    let a = build(&mut rng, 1.0);
    let b = build(&mut rng, 3.0);
    CausalDataset::new(
        variables(2),
        vec![
            block(Environment::observational("a").unwrap(), &a),
            block(intervened("b"), &b),
        ],
        "5C.9 benchmark fixture",
    )
    .unwrap()
}

/// The registry an analyst running the 5C.4 estimate would have written.
fn analyst_registry() -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    for (assumption, basis) in [
        (
            CausalAssumption::CausalSufficiency,
            AssumptionBasis::AssertedByAnalyst,
        ),
        (
            CausalAssumption::CorrectFunctionalForm,
            AssumptionBasis::AssertedByAnalyst,
        ),
        (
            CausalAssumption::InvarianceAcrossEnvironments,
            AssumptionBasis::Unverified,
        ),
        (
            CausalAssumption::Acyclicity,
            AssumptionBasis::DomainKnowledge {
                citation: "one-way pipeline".to_string(),
            },
        ),
    ]
    {
        registry.assert(assumption, basis, None).unwrap();
    }
    registry
}

fn describe_standing(standing: &CertificateStanding) -> String {
    match standing
    {
        CertificateStanding::Stands => "Stands".to_string(),
        CertificateStanding::Retracted { contradicted } =>
        {
            format!("Retracted({})", contradicted.len())
        },
        CertificateStanding::InDoubt {
            jointly_contradicted,
        } => format!("InDoubt({})", jointly_contradicted.len()),
        CertificateStanding::Unauditable { unknown } => format!("Unauditable({})", unknown.len()),
    }
}

fn count(revision: &AssumptionRevision, verdict: RevisionVerdict) -> usize {
    revision
        .outcomes
        .iter()
        .filter(|o| o.verdict == verdict)
        .count()
}

fn report_revision(scenario: &str, revision: &AssumptionRevision) {
    println!(
        "scenario={scenario} assumptions={} contradicted={} jointly_contradicted={} \
         corroborated={} inconclusive={} untouched={} status={:?} warnings={}",
        revision.outcomes.len(),
        count(revision, RevisionVerdict::Contradicted),
        count(revision, RevisionVerdict::JointlyContradicted),
        count(revision, RevisionVerdict::Corroborated),
        count(revision, RevisionVerdict::Inconclusive),
        count(revision, RevisionVerdict::Untouched),
        revision.certificate.status(),
        revision.warnings.len(),
    );
}

fn report_audit(scenario: &str, audit: &CertificateAudit) {
    println!(
        "scenario={scenario} original_status={:?} original_estimate={} standing={} \
         estimate_usable={} audit_status={:?} warnings={}",
        audit.original_status,
        audit
            .original_estimate
            .map_or("none".to_string(), |v| format!("{v:.12e}")),
        describe_standing(&audit.standing),
        audit.standing.estimate_is_usable(),
        audit.certificate.status(),
        audit.warnings.len(),
    );
}

fn main() {
    println!("# Phase 5C.9 deterministic theory-revision benchmark");
    println!(
        "# revision fields: scenario assumptions contradicted jointly_contradicted corroborated \
         inconclusive untouched status warnings"
    );
    println!(
        "# audit fields: scenario original_status original_estimate standing estimate_usable \
         audit_status warnings"
    );
    println!(
        "# scope: this records what evidence did to STATED assumptions and which claims that \
         undermines; it re-derives no estimate and establishes no assumption as true"
    );

    // ── The claim, made in good faith and wrong ──────────────────────────
    let dataset = hidden_confounder(81, 1500);
    let mut dag = CausalDag::new(2);
    dag.add_directed_edge(0, 1).unwrap();
    let backdoor = estimate_effect_from_dag(
        &dataset,
        &dag,
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    let biased = backdoor.estimate.unwrap();
    let standard_error = backdoor.standard_error.unwrap();
    println!(
        "scenario=backdoor_claim status={:?} estimate={biased:.12e} standard_error={:.12e} \
         truth=7.000000000000e-1 error_in_standard_errors={:.6e}",
        backdoor.certificate.status(),
        standard_error,
        (biased - 0.7).abs() / standard_error,
    );
    expect(
        backdoor.certificate.status() == IdentifiabilityStatus::Identifiable,
        "backdoor_claim: the fixture must certify Identifiable".to_string(),
    );
    expect(
        (biased - 0.7).abs() > 10.0 * standard_error,
        format!("backdoor_claim: expected a large bias, got {biased}"),
    );

    // ── Evidence arrives, and cannot attribute ───────────────────────────
    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();
    let evidence = evidence_from_invariance(&icp);
    println!(
        "scenario=icp_evidence outcome={:?} direction={:?} assumptions_named={} source={}",
        icp.outcome,
        evidence[0].direction,
        evidence[0].assumptions.len(),
        evidence[0].source,
    );
    expect(
        icp.outcome == InvariantPredictionOutcome::AssumptionsViolated,
        format!(
            "icp_evidence: expected AssumptionsViolated, got {:?}",
            icp.outcome
        ),
    );
    expect(
        evidence[0].direction == EvidenceDirection::Contradicts
            && evidence[0].assumptions.len() == 3,
        "icp_evidence: ICP falsifies a conjunction of three, attributable to none".to_string(),
    );

    let revision = revise_assumptions(&analyst_registry(), &evidence).unwrap();
    report_revision("revision_from_icp", &revision);
    expect(
        revision.contradicted().is_empty() && revision.jointly_contradicted().len() == 3,
        format!(
            "revision_from_icp: expected 0 attributed and 3 joint, got {} and {}",
            revision.contradicted().len(),
            revision.jointly_contradicted().len()
        ),
    );
    expect(
        count(&revision, RevisionVerdict::Untouched) == 1,
        "revision_from_icp: acyclicity was not tested and must be left alone".to_string(),
    );

    // ── The claim stops standing ─────────────────────────────────────────
    let doubted = audit_certificate(&backdoor.certificate, &revision).unwrap();
    report_audit("audit_from_icp", &doubted);
    println!(
        "# the_loose_end: 5C.4 certified {biased:.6e} against a truth of 0.7 as Identifiable and \
         5C.6 detected the misspecification, but nothing connected them; that certificate now \
         audits as {} and its estimate is no longer usable",
        describe_standing(&doubted.standing),
    );
    expect(
        matches!(doubted.standing, CertificateStanding::InDoubt { .. })
            && !doubted.standing.estimate_is_usable(),
        format!(
            "audit_from_icp: expected an unusable InDoubt, got {:?}",
            doubted.standing
        ),
    );

    // ── The attribution boundary ─────────────────────────────────────────
    let attributable = revise_assumptions(
        &analyst_registry(),
        &[AssumptionEvidence::attributed(
            CausalAssumption::CausalSufficiency,
            EvidenceDirection::Contradicts,
            "follow_up_measurement",
            "the omitted common cause was located and measured",
        )],
    )
    .unwrap();
    report_revision("revision_from_measurement", &attributable);
    let retracted = audit_certificate(&backdoor.certificate, &attributable).unwrap();
    report_audit("audit_from_measurement", &retracted);
    println!(
        "# attribution_boundary: the same claim audits as {} from ICP and {} from a measurement \
         that names the assumption; both block the number, and only the second says which \
         premise failed",
        describe_standing(&doubted.standing),
        describe_standing(&retracted.standing),
    );
    expect(
        matches!(retracted.standing, CertificateStanding::Retracted { .. }),
        format!(
            "audit_from_measurement: expected Retracted, got {:?}",
            retracted.standing
        ),
    );

    // ── A claim on untouched ground still stands ─────────────────────────
    let unrelated = CausalCertificate::builder(
        "a claim resting only on acyclicity",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Acyclicity],
        "domain knowledge",
    )
    .finalize()
    .unwrap();
    let stands = audit_certificate(&unrelated, &revision).unwrap();
    report_audit("audit_untouched_ground", &stands);
    expect(
        stands.standing == CertificateStanding::Stands,
        format!(
            "audit_untouched_ground: expected Stands, got {:?}",
            stands.standing
        ),
    );

    // ── An unregistered assumption is a gap, not a pass ──────────────────
    let foreign = CausalCertificate::builder(
        "a claim resting on positivity",
        IdentifiabilityStatus::Identifiable,
        vec![CausalAssumption::Positivity],
        "elsewhere",
    )
    .with_estimate("m", 1.0, 0.1)
    .finalize()
    .unwrap();
    let unauditable = audit_certificate(&foreign, &revision).unwrap();
    report_audit("audit_unregistered_assumption", &unauditable);
    expect(
        matches!(
            unauditable.standing,
            CertificateStanding::Unauditable { .. }
        ) && !unauditable.standing.estimate_is_usable(),
        format!(
            "audit_unregistered_assumption: an undetermined standing is not permission, got {:?}",
            unauditable.standing
        ),
    );

    // ── Corroboration, and its ceiling ───────────────────────────────────
    let clean_icp = invariant_causal_prediction(
        &clean(404, 1200),
        1,
        &[0],
        &InvarianceConfig::new(0.05).unwrap(),
    )
    .unwrap();
    let corroboration =
        revise_assumptions(&analyst_registry(), &evidence_from_invariance(&clean_icp)).unwrap();
    report_revision("revision_from_clean_run", &corroboration);
    println!(
        "# corroboration_ceiling: a clean run leaves 0 undermined, and lifts an unbacked belief \
         only to \"tested and not contradicted\" -- outcome={:?}, no basis here becomes a \
         guarantee and no assumption becomes true",
        clean_icp.outcome,
    );
    expect(
        count(&corroboration, RevisionVerdict::Contradicted) == 0
            && count(&corroboration, RevisionVerdict::JointlyContradicted) == 0,
        "revision_from_clean_run: a clean run must undermine nothing".to_string(),
    );
    expect(
        corroboration.outcomes.iter().all(|o| {
            !matches!(o.revised_basis, AssumptionBasis::GuaranteedByDesign { .. })
                || matches!(o.prior_basis, AssumptionBasis::GuaranteedByDesign { .. })
        }),
        "revision_from_clean_run: corroboration must never manufacture a design guarantee"
            .to_string(),
    );

    // ── Contradiction is not outvoted ────────────────────────────────────
    let outvoted = revise_assumptions(
        &analyst_registry(),
        &[
            AssumptionEvidence::attributed(
                CausalAssumption::CausalSufficiency,
                EvidenceDirection::Supports,
                "check_a",
                "consistent",
            ),
            AssumptionEvidence::attributed(
                CausalAssumption::CausalSufficiency,
                EvidenceDirection::Supports,
                "check_b",
                "consistent",
            ),
            AssumptionEvidence::attributed(
                CausalAssumption::CausalSufficiency,
                EvidenceDirection::Contradicts,
                "check_c",
                "failed",
            ),
        ],
    )
    .unwrap();
    report_revision("revision_two_for_one_against", &outvoted);
    println!(
        "# asymmetry: 2 supporting findings and 1 contradicting leave the assumption \
         contradicted, because corroboration cannot cancel a falsification; the counts are \
         reported and a warning notes that a test at level alpha fires spuriously with \
         probability alpha"
    );
    expect(
        count(&outvoted, RevisionVerdict::Contradicted) == 1 && !outvoted.warnings.is_empty(),
        "revision_two_for_one_against: the contradiction must stand, with the counts disclosed"
            .to_string(),
    );

    println!("# all oracle checks passed");
}
