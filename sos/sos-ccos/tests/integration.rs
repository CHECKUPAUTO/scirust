//! Real integration: a cognitive proposal is admitted by a genuine
//! `sos-reasoning` `Derivation`, and the whole untrusted → verdict → admission
//! chain is stored in a content-addressed `sos-store`.

use sos_core::{Author, HashAlgo, Object, ObjectId};
use sos_store::{MemoryStore, TypedStore};

use sos_ccos::{
    Admission, Cognition, LocalMemory, Proposal, ProposalKind, Ruling, propose_capability,
};
use sos_reasoning::{Derivation, DerivationStep, Soundness};
use sos_registry::Grant;

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"ccos-integration", tag)
}

#[test]
fn cognition_proposes_and_reasoning_disposes_end_to_end() {
    // A real observation the cognitive backend will reason about.
    let observation = oid(b"observed-relaxation-to-equilibrium");

    // The cognitive backend proposes an analogy — grounded, untrusted.
    let grant = Grant::new().allow(propose_capability());
    let mut ccos = Cognition::new(LocalMemory::new(), grant);
    let proposal = Proposal::new(
        ProposalKind::Analogy,
        "A damped oscillator is analogous to an Ornstein–Uhlenbeck process.",
        vec![observation],
        "Both relax to equilibrium at a rate set by one time constant.",
    )
    .unwrap();
    let untrusted = ccos.propose(proposal, Author::agent("ccos")).unwrap();

    // A genuine deterministic derivation verifies the analogy (proof-grade).
    let derivation = Object::builder(Derivation::new(
        "damped oscillator and OU process share the same relaxation ODE",
        vec![DerivationStep::new(
            "identify-generator",
            vec![observation],
            "dx = -theta x dt (+ noise)",
        )],
        vec![observation],
        Soundness::Proof,
    ))
    .author(Author::engine("sos-reasoning"))
    .seal();
    assert!(derivation.verify_id());

    // Determinism disposes using that derivation as the verdict.
    let admission = ccos.dispose(
        &untrusted,
        Ruling::Admit {
            verdict: derivation.id,
        },
        Author::engine("sos-reasoning"),
    );
    assert!(admission.body.is_admitted());
    let trusted = admission.body.into_trusted().unwrap();
    assert_eq!(trusted.proposal(), untrusted.id);
    assert_eq!(trusted.verdict(), derivation.id);

    // The whole chain is content-addressed and round-trips through the store.
    let mut store = MemoryStore::new();
    store.put_object(&derivation).unwrap();
    store.put_object(&untrusted).unwrap();
    store.put_object(&admission).unwrap();

    let back: Object<Admission> = store.get_object(admission.id).unwrap().unwrap();
    assert!(back.verify_id());
    assert_eq!(back.body.proposal, untrusted.id);
    assert_eq!(back.body.disposition.verdict(), Some(derivation.id));

    // The untrusted proposal is retrievable, and still typed as a Proposal —
    // it lives in the graph, but only the admission makes it trusted.
    let stored_proposal: Object<Proposal> = store.get_object(untrusted.id).unwrap().unwrap();
    assert_eq!(stored_proposal.kind.name, "Proposal");
    assert_eq!(stored_proposal.parents, vec![observation]);
}

#[test]
fn a_refuted_proposal_is_recorded_but_never_trusted() {
    let observation = oid(b"anomalous-reading");
    let grant = Grant::new().allow(propose_capability());
    let mut ccos = Cognition::new(LocalMemory::new(), grant);
    let untrusted = ccos
        .propose(
            Proposal::new(
                ProposalKind::Conjecture,
                "The anomaly is caused by a new particle.",
                vec![observation],
                "It is unexplained.",
            )
            .unwrap(),
            Author::agent("ccos"),
        )
        .unwrap();

    // Reasoning refutes it; the rejection is a first-class, stored record.
    let admission = ccos.dispose(
        &untrusted,
        Ruling::Reject {
            reason: "explained by a calibration error; no new physics".to_owned(),
        },
        Author::engine("sos-reasoning"),
    );
    assert!(!admission.body.is_admitted());
    assert!(admission.body.into_trusted().is_none());

    let mut store = MemoryStore::new();
    store.put_object(&admission).unwrap();
    let back: Object<Admission> = store.get_object(admission.id).unwrap().unwrap();
    assert!(matches!(
        back.body.disposition,
        sos_ccos::Disposition::Rejected { .. }
    ));
}
