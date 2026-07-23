//! Deterministic memory recall/context and the capability-scoped session.

use sos_ccos::{
    CcosError, Cognition, LocalMemory, Proposal, ProposalKind, Recall, Remember, Ruling,
    TokenBudget, propose_capability, recall_capability,
};
use sos_core::{Author, HashAlgo, ObjectId};
use sos_registry::Grant;

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"ccos-mem", tag)
}

fn agent() -> Author {
    Author::agent("ccos")
}

fn proposal(kind: ProposalKind, statement: &str, concerns: Vec<ObjectId>) -> Proposal {
    Proposal::new(kind, statement, concerns, "rationale").unwrap()
}

#[test]
fn recall_ranks_by_structural_overlap() {
    let (c1, c2, c3) = (oid(b"1"), oid(b"2"), oid(b"3"));
    let a = sos_ccos::seal_proposal(proposal(ProposalKind::Question, "A", vec![c1, c2]), agent());
    let b = sos_ccos::seal_proposal(proposal(ProposalKind::Question, "B", vec![c2, c3]), agent());
    let c = sos_ccos::seal_proposal(proposal(ProposalKind::Hypothesis, "C", vec![c3]), agent());

    let mut memory = LocalMemory::new();
    memory.store(&a);
    memory.store(&b);
    memory.store(&c);
    assert_eq!(memory.len(), 3);

    // Recall about {c2, c3}: B overlaps both (2), A and C overlap one (1) each.
    let recalled = memory.recall(&Recall::about(vec![c2, c3]), 10);
    assert_eq!(recalled.len(), 3);
    assert_eq!(
        recalled[0], b.id,
        "the best-overlapping proposal comes first"
    );

    // Recall is deterministic and replay-exact.
    assert_eq!(recalled, memory.recall(&Recall::about(vec![c2, c3]), 10));
}

#[test]
fn recall_can_filter_by_kind() {
    let c3 = oid(b"3");
    let b = sos_ccos::seal_proposal(proposal(ProposalKind::Question, "B", vec![c3]), agent());
    let c = sos_ccos::seal_proposal(proposal(ProposalKind::Hypothesis, "C", vec![c3]), agent());
    let mut memory = LocalMemory::new();
    memory.store(&b);
    memory.store(&c);

    let only_hypotheses = memory.recall(
        &Recall::about(vec![c3]).of_kind(ProposalKind::Hypothesis),
        10,
    );
    assert_eq!(only_hypotheses, vec![c.id]);
}

#[test]
fn context_pages_are_bounded_and_deterministic() {
    let (c1, c2, c3) = (oid(b"1"), oid(b"2"), oid(b"3"));
    let a = sos_ccos::seal_proposal(proposal(ProposalKind::Question, "A", vec![c1, c2]), agent());
    let b = sos_ccos::seal_proposal(proposal(ProposalKind::Question, "B", vec![c2, c3]), agent());
    let mut memory = LocalMemory::new();
    memory.store(&a);
    memory.store(&b);

    // A generous budget yields the whole neighborhood of c2.
    let page = memory.context(c2, TokenBudget::new(100));
    assert!(page.objects.contains(&c2));
    assert!(page.objects.contains(&a.id));
    assert!(page.objects.contains(&b.id));
    assert!(!page.truncated);

    // A tight budget truncates deterministically.
    let small = memory.context(c2, TokenBudget::new(2));
    assert_eq!(small.objects.len(), 2);
    assert!(small.truncated);
    assert_eq!(
        small.objects,
        memory.context(c2, TokenBudget::new(2)).objects
    );
}

#[test]
fn cognitive_acts_are_refused_by_default() {
    // An empty grant permits no cognitive act (least privilege).
    let mut ccos = Cognition::new(LocalMemory::new(), Grant::new());
    let err = ccos
        .propose(
            proposal(ProposalKind::Question, "?", vec![oid(b"x")]),
            agent(),
        )
        .unwrap_err();
    assert!(matches!(err, CcosError::Denied { .. }));
    assert!(ccos.recall(&Recall::about(vec![oid(b"x")]), 1).is_err());
}

#[test]
fn a_granted_session_proposes_recalls_attests_and_disposes() {
    let grant = Grant::new()
        .allow(propose_capability())
        .allow(recall_capability());
    let mut ccos = Cognition::new(LocalMemory::new(), grant);

    let concern = oid(b"c");
    let untrusted = ccos
        .propose(
            proposal(ProposalKind::Hypothesis, "H", vec![concern]),
            agent(),
        )
        .unwrap();

    // Proposing stored the object and attested the act.
    assert_eq!(ccos.memory().len(), 1);
    assert_eq!(ccos.memory().chain().len(), 1);

    // Recall finds it by its grounding.
    let recalled = ccos.recall(&Recall::about(vec![concern]), 5).unwrap();
    assert_eq!(recalled, vec![untrusted.id]);

    // Determinism disposes; the disposition is attested too.
    let admission = ccos.dispose(
        &untrusted,
        Ruling::Admit {
            verdict: oid(b"verdict"),
        },
        Author::engine("sos-reasoning"),
    );
    assert!(admission.body.is_admitted());
    assert_eq!(ccos.memory().chain().len(), 2);
    ccos.memory().chain().verify().unwrap();
}
