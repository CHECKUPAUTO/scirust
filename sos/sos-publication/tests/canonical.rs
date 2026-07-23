//! Determinism and content-addressing properties: canonical encoding is stable
//! and collision-free where it must be, claim identity is order-independent, and
//! the interchange form round-trips.

use sos_core::canonical::Canonical;
use sos_core::{HashAlgo, ObjectId};
use sos_publication::{
    BindingRole, Claim, ClaimBinding, MapSource, ObjectFacts, Publication, PublicationObjectSource,
};

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"pub-canonical", tag)
}

#[test]
fn a_claims_content_id_is_independent_of_binding_order() {
    let a = oid(b"a");
    let b = oid(b"b");
    let first = Claim::new(
        "C1",
        "statement",
        vec![
            ClaimBinding::new(BindingRole::DirectlySupports, a),
            ClaimBinding::new(BindingRole::SuppliesData, b),
        ],
    );
    let second = Claim::new(
        "C1",
        "statement",
        vec![
            ClaimBinding::new(BindingRole::SuppliesData, b),
            ClaimBinding::new(BindingRole::DirectlySupports, a),
        ],
    );
    assert_eq!(first.content_id(), second.content_id());
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
}

#[test]
fn duplicate_bindings_are_normalized_away() {
    let a = oid(b"a");
    let claim = Claim::new(
        "C1",
        "statement",
        vec![
            ClaimBinding::new(BindingRole::DirectlySupports, a),
            ClaimBinding::new(BindingRole::DirectlySupports, a),
        ],
    );
    assert_eq!(claim.bindings.len(), 1);
}

#[test]
fn distinct_claims_have_distinct_content_ids() {
    let a = oid(b"a");
    let restated = Claim::new(
        "C1",
        "one thing",
        vec![ClaimBinding::new(BindingRole::DirectlySupports, a)],
    );
    let other = Claim::new(
        "C1",
        "another thing",
        vec![ClaimBinding::new(BindingRole::DirectlySupports, a)],
    );
    assert_ne!(restated.content_id(), other.content_id());

    // Different binding role ⇒ different claim.
    let role_changed = Claim::new(
        "C1",
        "one thing",
        vec![ClaimBinding::new(BindingRole::Contradicts, a)],
    );
    assert_ne!(restated.content_id(), role_changed.content_id());
}

#[test]
fn equal_publications_encode_identically() {
    let build = || {
        Publication::builder("Paper")
            .declared_root(oid(b"root"))
            .claim(Claim::new(
                "C1",
                "x",
                vec![ClaimBinding::new(BindingRole::DirectlySupports, oid(b"e"))],
            ))
            .build()
    };
    assert_eq!(build().canonical_bytes(), build().canonical_bytes());
}

#[test]
fn declared_roots_are_order_independent_after_build() {
    let r1 = oid(b"r1");
    let r2 = oid(b"r2");
    let a = Publication::builder("P")
        .declared_root(r1)
        .declared_root(r2)
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, r1)],
        ))
        .build();
    let b = Publication::builder("P")
        .declared_root(r2)
        .declared_root(r1)
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, r1)],
        ))
        .build();
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    assert_eq!(a, b);
}

#[test]
fn a_publication_round_trips_through_json() {
    let paper = Publication::builder("Paper")
        .declared_root(oid(b"root"))
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, oid(b"e"))],
        ))
        .build();
    let json = serde_json::to_string(&paper).unwrap();
    let back: Publication = serde_json::from_str(&json).unwrap();
    assert_eq!(paper, back);
    assert_eq!(paper.canonical_bytes(), back.canonical_bytes());
}

#[test]
fn the_dependency_closure_is_deterministic() {
    // root -> a -> c ; root -> b ; verify the closure is exactly {root,a,b,c}.
    let root = oid(b"root");
    let a = oid(b"a");
    let b = oid(b"b");
    let c = oid(b"c");
    let kind = sos_core::Kind::new("Node", 1);
    let level = sos_core::DeterminismLevel::L3;
    let mut source = MapSource::new();
    source.insert(ObjectFacts::new(root, kind.clone(), vec![a, b], level));
    source.insert(ObjectFacts::new(a, kind.clone(), vec![c], level));
    source.insert(ObjectFacts::new(b, kind.clone(), Vec::new(), level));
    source.insert(ObjectFacts::new(c, kind, Vec::new(), level));

    let closure = sos_publication::dependency_closure(&source, &[root]).unwrap();
    assert_eq!(closure.len(), 4);
    for id in [root, a, b, c]
    {
        assert!(closure.contains(&id));
    }
    // The trait object path resolves too.
    let dyn_source: &dyn PublicationObjectSource = &source;
    assert!(dyn_source.facts(root).unwrap().is_some());
}
