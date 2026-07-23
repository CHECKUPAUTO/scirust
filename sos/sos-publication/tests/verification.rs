//! The verification matrix: every [`ClaimStatus`] branch, structural findings,
//! scope, and the reproducibility summary — driven through the real [`verify`]
//! pipeline over a deterministic [`MapSource`].

use sos_core::{DeterminismLevel, HashAlgo, Kind, ObjectId};
use sos_publication::{
    BindingRole, Claim, ClaimBinding, ClaimStatus, MapSource, ObjectFacts, Publication,
    ReproRequirement, ReproVerdict, StandardPolicy, StructuralIssue, verify,
};

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"pub-verify", tag)
}

fn facts(id: ObjectId, parents: Vec<ObjectId>, level: DeterminismLevel) -> ObjectFacts {
    ObjectFacts::new(id, Kind::new("Derivation", 1), parents, level)
}

/// Root `R` derived from evidence `E`; both in the graph. Closure of `R` is
/// `{R, E}`, so `E` is in scope.
fn graph_with_evidence(evidence_level: DeterminismLevel) -> (ObjectId, ObjectId, MapSource) {
    let e = oid(b"evidence");
    let r = oid(b"root");
    let mut source = MapSource::new();
    source.insert(facts(e, Vec::new(), evidence_level));
    source.insert(facts(r, vec![e], DeterminismLevel::L3));
    (r, e, source)
}

fn one_claim(root: ObjectId, claim: Claim) -> Publication {
    Publication::builder("Study")
        .declared_root(root)
        .claim(claim)
        .build()
}

#[test]
fn a_directly_supported_in_scope_claim_is_supported() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Holds.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::Supported);
    assert_eq!(report.claims[0].supporting, vec![e]);
    assert!(report.is_publishable());
    assert!(report.is_structurally_complete());
    assert_eq!(report.closure_size, 2);
}

#[test]
fn only_indirect_support_is_partial() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Corroborated.",
            vec![ClaimBinding::new(BindingRole::SuppliesData, e)],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::PartiallySupported);
    // Partial support still clears the publishable bar.
    assert!(report.is_publishable());
}

#[test]
fn a_resolved_contradiction_dominates_and_is_reported() {
    let (r, e, mut source) = graph_with_evidence(DeterminismLevel::L3);
    let other = oid(b"other-support");
    source.insert(facts(other, vec![r], DeterminismLevel::L3));
    // Both a support edge and a contradiction edge (to different objects).
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Disputed.",
            vec![
                ClaimBinding::new(BindingRole::DirectlySupports, other),
                ClaimBinding::new(BindingRole::Contradicts, e),
            ],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::Contradicted);
    assert_eq!(report.claims[0].contradicting, vec![e]);
    assert_eq!(report.contradicted().len(), 1);
    assert!(!report.is_publishable());
}

#[test]
fn incoherent_wiring_is_policy_rejected() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    // The same object bound as both support and contradiction.
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Self-contradictory wiring.",
            vec![
                ClaimBinding::new(BindingRole::DirectlySupports, e),
                ClaimBinding::new(BindingRole::Contradicts, e),
            ],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::PolicyRejected);
}

#[test]
fn a_missing_dependency_is_caught_not_hidden() {
    let (r, _e, source) = graph_with_evidence(DeterminismLevel::L3);
    let ghost = oid(b"ghost");
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Leans on nothing.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, ghost)],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::DependencyMissing);
    assert_eq!(report.claims[0].missing, vec![ghost]);
    assert_eq!(report.missing_objects, vec![ghost]);
    assert!(!report.is_publishable());
}

#[test]
fn support_outside_the_declared_scope_is_unresolved() {
    let (r, _e, mut source) = graph_with_evidence(DeterminismLevel::L3);
    // Present in the graph, but not reachable from the declared root.
    let outsider = oid(b"outsider");
    source.insert(facts(outsider, Vec::new(), DeterminismLevel::L3));
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Out of scope.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, outsider)],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::Unresolved);
    assert_eq!(report.claims[0].out_of_scope, vec![outsider]);
    assert!(report.claims[0].supporting.is_empty());
}

#[test]
fn context_only_bindings_are_unresolved() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "Only a dependency.",
            vec![ClaimBinding::new(BindingRole::DependsOn, e)],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::Unresolved);
}

#[test]
fn a_claim_with_no_bindings_is_unverifiable() {
    let (r, _e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = one_claim(r, Claim::new("C1", "A bare sentence.", Vec::new()));
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::Unverifiable);
    // ... and it is flagged structurally as a claim without bindings.
    assert!(
        report
            .structural
            .iter()
            .any(|i| matches!(i, StructuralIssue::ClaimWithoutBindings(k) if k.as_str() == "C1"))
    );
}

#[test]
fn a_duplicated_claim_key_is_structurally_invalid() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = Publication::builder("Study")
        .declared_root(r)
        .claim(Claim::new(
            "C1",
            "First.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .claim(Claim::new(
            "C1",
            "Second, same key.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .build();
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert!(
        report
            .claims
            .iter()
            .all(|c| c.status == ClaimStatus::StructurallyInvalid)
    );
    assert!(
        report
            .structural
            .iter()
            .any(|i| matches!(i, StructuralIssue::DuplicateClaimKey(k) if k.as_str() == "C1"))
    );
}

#[test]
fn reproducibility_bar_below_requirement_fails() {
    // Evidence is only statistically reproducible (L1) but the paper demands L3.
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L1);
    let paper = Publication::builder("Study")
        .declared_root(r)
        .reproducibility(ReproRequirement::MinimumLevel(DeterminismLevel::L3))
        .claim(Claim::new(
            "C1",
            "Needs bit-repro.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .build();
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::ReproducibilityFailed);
    assert!(matches!(
        report.reproducibility,
        ReproVerdict::Insufficient {
            required: DeterminismLevel::L3,
            realized: DeterminismLevel::L1
        }
    ));
    assert!(!report.is_publishable());
}

#[test]
fn reproducibility_bar_met_is_satisfied_and_supported() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = Publication::builder("Study")
        .declared_root(r)
        .reproducibility(ReproRequirement::MinimumLevel(DeterminismLevel::L2))
        .claim(Claim::new(
            "C1",
            "Meets the bar.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .build();
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::Supported);
    assert!(matches!(
        report.reproducibility,
        ReproVerdict::Satisfied { .. }
    ));
    assert!(report.reproducibility.is_met());
}

#[test]
fn empty_declared_roots_is_flagged() {
    let e = oid(b"e");
    let mut source = MapSource::new();
    source.insert(facts(e, Vec::new(), DeterminismLevel::L3));
    let paper = Publication::builder("Study")
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .build();
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert!(
        report
            .structural
            .iter()
            .any(|i| matches!(i, StructuralIssue::EmptyDeclaredRoots))
    );
    // With no declared scope, nothing is in scope, so the claim is unresolved.
    assert_eq!(report.claims[0].status, ClaimStatus::Unresolved);
}

#[test]
fn a_publication_with_no_claims_is_flagged() {
    let r = oid(b"root");
    let mut source = MapSource::new();
    source.insert(facts(r, Vec::new(), DeterminismLevel::L3));
    let paper = Publication::builder("Empty").declared_root(r).build();
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert!(
        report
            .structural
            .iter()
            .any(|i| matches!(i, StructuralIssue::NoClaims))
    );
    assert!(!report.is_publishable());
}

#[test]
fn the_policy_id_is_recorded_in_the_report() {
    let (r, e, source) = graph_with_evidence(DeterminismLevel::L3);
    let paper = one_claim(
        r,
        Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ),
    );
    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.policy.name, "standard");
    assert_eq!(report.policy.version, 1);
}
