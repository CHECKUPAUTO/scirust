//! The trust boundary: grounding, the disposition gate, tamper/ungrounded
//! downgrades, the `Trusted` guard, and content-addressing of proposals.

use sos_ccos::{
    Admission, CcosError, Disposition, Proposal, ProposalKind, Ruling, dispose, seal_admission,
    seal_proposal,
};
use sos_core::{Author, HashAlgo, ObjectId};

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"ccos-trust", tag)
}

fn agent() -> Author {
    Author::agent("ccos")
}

fn engine() -> Author {
    Author::engine("sos-reasoning")
}

#[test]
fn a_proposal_must_ground_in_at_least_one_object() {
    let err = Proposal::new(
        ProposalKind::Question,
        "About nothing?",
        Vec::new(),
        "hunch",
    )
    .unwrap_err();
    assert_eq!(err, CcosError::Ungrounded);

    // Grounded is fine.
    assert!(Proposal::new(ProposalKind::Question, "About X?", vec![oid(b"x")], "").is_ok());
}

#[test]
fn a_proposals_content_address_is_independent_of_concern_order() {
    let a = Proposal::new(ProposalKind::Analogy, "q", vec![oid(b"1"), oid(b"2")], "r").unwrap();
    let b = Proposal::new(
        ProposalKind::Analogy,
        "q",
        vec![oid(b"2"), oid(b"1"), oid(b"1")],
        "r",
    )
    .unwrap();
    assert_eq!(a, b); // sorted + deduplicated
    assert_eq!(seal_proposal(a, agent()).id, seal_proposal(b, agent()).id);
}

#[test]
fn grounding_objects_become_provenance_parents() {
    let c1 = oid(b"c1");
    let c2 = oid(b"c2");
    let obj = seal_proposal(
        Proposal::new(ProposalKind::Hypothesis, "H", vec![c1, c2], "r").unwrap(),
        agent(),
    );
    assert!(obj.verify_id());
    assert_eq!(obj.kind.name, "Proposal");
    // The grounding concerns are exactly the provenance parents (in canonical order).
    assert_eq!(obj.parents, obj.body.concerns);
    assert!(obj.parents.contains(&c1) && obj.parents.contains(&c2));
}

#[test]
fn determinism_admits_a_verified_proposal_into_the_trusted_graph() {
    let obj = seal_proposal(
        Proposal::new(ProposalKind::Hypothesis, "H", vec![oid(b"c")], "r").unwrap(),
        agent(),
    );
    let verdict = oid(b"derivation");
    let admission = seal_admission(
        Admission::new(obj.id, dispose(&obj, Ruling::Admit { verdict })),
        engine(),
    );
    assert!(admission.verify_id());
    assert_eq!(admission.kind.name, "Admission");
    assert!(admission.body.is_admitted());
    // The admission is authored by the engine and parents both proposal + verdict.
    assert_eq!(admission.parents, vec![obj.id, verdict]);

    // The only path to a trusted reference — and it carries the verdict.
    let trusted = admission.body.into_trusted().unwrap();
    assert_eq!(trusted.proposal(), obj.id);
    assert_eq!(trusted.verdict(), verdict);
}

#[test]
fn a_rejected_proposal_yields_no_trusted_reference() {
    let obj = seal_proposal(
        Proposal::new(ProposalKind::Conjecture, "maybe", vec![oid(b"c")], "").unwrap(),
        agent(),
    );
    let disposition = dispose(
        &obj,
        Ruling::Reject {
            reason: "no supporting derivation".to_owned(),
        },
    );
    let admission = Admission::new(obj.id, disposition);
    assert!(!admission.is_admitted());
    assert!(admission.into_trusted().is_none());
}

#[test]
fn a_tampered_proposal_is_rejected_even_under_an_admit_ruling() {
    let obj = seal_proposal(
        Proposal::new(ProposalKind::Hypothesis, "H", vec![oid(b"c")], "r").unwrap(),
        agent(),
    );
    // Mutate the content without resealing — the id no longer verifies.
    let mut tampered = obj.clone();
    tampered.body.statement = "a different claim".to_owned();
    assert!(!tampered.verify_id());

    let disposition = dispose(&tampered, Ruling::Admit { verdict: oid(b"v") });
    assert!(matches!(disposition, Disposition::Rejected { .. }));
    assert!(disposition.verdict().is_none());
}

#[test]
fn an_ungrounded_proposal_is_rejected_even_under_an_admit_ruling() {
    // Build an ungrounded proposal directly (bypassing the checked constructor),
    // as an adversary deserializing hostile input might.
    let ungrounded = Proposal {
        kind: ProposalKind::Question,
        statement: "About nothing".to_owned(),
        concerns: Vec::new(),
        rationale: String::new(),
    };
    let obj = seal_proposal(ungrounded, agent());
    assert!(obj.verify_id()); // it is a valid object...
    // ...but the gate refuses it regardless of the ruling.
    let disposition = dispose(&obj, Ruling::Admit { verdict: oid(b"v") });
    assert!(matches!(disposition, Disposition::Rejected { .. }));
}
