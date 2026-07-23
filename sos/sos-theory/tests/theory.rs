//! End-to-end Theory Engine: revision lineage with retained anomalies, and
//! evidential comparison of rival theories over a shared domain of validity.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};
use sos_store::{MemoryStore, TypedStore};
use sos_theory::{Scope, Theories, Theory, TheoryEngine, seal_theory};

/// A stand-in knowledge node (an equation, a datum, …) so tests reference real,
/// stored object ids rather than fabricated ones.
#[derive(Clone, Serialize, Deserialize)]
struct Node {
    name: String,
}

impl Canonical for Node {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
    }
}

impl Body for Node {
    const KIND: &'static str = "Node";
    const SCHEMA_VERSION: u32 = 1;
}

struct World {
    store: MemoryStore,
}

impl World {
    fn new() -> Self {
        Self {
            store: MemoryStore::new(),
        }
    }

    fn node(&mut self, name: &str) -> ObjectId {
        let obj = Object::builder(Node { name: name.into() })
            .author(Author::human("curator"))
            .seal();
        let id = obj.id;
        self.store.put_object(&obj).unwrap();
        id
    }

    fn put_theory(&mut self, theory: Theory) -> ObjectId {
        let obj = seal_theory(theory, Author::engine("sos-theory-test"));
        let id = obj.id;
        self.store.put_object(&obj).unwrap();
        id
    }
}

#[test]
fn revision_chain_walks_the_full_lineage() {
    let mut w = World::new();
    let eq = w.node("inverse-square-law");
    let anomaly = w.node("mercury-perihelion");

    // v1 → v2 (forced by the anomaly) → v3.
    let v1 = w.put_theory(
        Theory::builder(Scope::universal())
            .equations(vec![eq])
            .build(),
    );
    let v2_body = {
        let engine = Theories::new(&w.store);
        engine.revise(v1, &[anomaly]).unwrap()
    };
    let v2 = w.put_theory(v2_body);
    let v3_body = {
        let engine = Theories::new(&w.store);
        engine.revise(v2, &[]).unwrap()
    };
    let v3 = w.put_theory(v3_body);

    let engine = Theories::new(&w.store);
    // Newest-first lineage back to the root.
    assert_eq!(engine.revision_chain(v3).unwrap(), vec![v3, v2, v1]);
    assert_eq!(engine.revision_chain(v1).unwrap(), vec![v1]);

    // The anomaly that forced v2 is retained by v2 and inherited by v3 — never
    // hidden.
    assert!(engine.get(v2).unwrap().contradicting.contains(&anomaly));
    assert!(engine.get(v3).unwrap().contradicting.contains(&anomaly));
    // Revision inherits the parent's equations.
    assert!(engine.get(v3).unwrap().equations.contains(&eq));
}

#[test]
fn compare_ranks_rivals_by_evidential_balance_over_shared_scope() {
    let mut w = World::new();
    let (e1, e2, e3) = (w.node("ev1"), w.node("ev2"), w.node("ev3"));
    let anomaly = w.node("anomaly");

    // Two rivals valid in the same regime; `strong` has more net support.
    let strong = w.put_theory(
        Theory::builder(Scope::from_predicates(["regime-A"]))
            .supporting(vec![e1, e2, e3])
            .contradicting(vec![anomaly])
            .build(),
    );
    let weak = w.put_theory(
        Theory::builder(Scope::from_predicates(["regime-A"]))
            .supporting(vec![e1])
            .contradicting(vec![anomaly])
            .build(),
    );
    // A third rival whose domain does NOT cover the queried scope.
    let off_domain = w.put_theory(
        Theory::builder(Scope::from_predicates(["regime-A", "regime-B"]))
            .supporting(vec![e1, e2, e3])
            .build(),
    );

    let engine = Theories::new(&w.store);
    let scope = Scope::from_predicates(["regime-A"]);
    let ranking = engine.compare(&[weak, strong, off_domain], &scope).unwrap();

    // off_domain is excluded (its domain does not contain the queried scope).
    assert_eq!(ranking.ranked.len(), 2);
    // strong ranks first (net 2 vs net 0).
    assert_eq!(ranking.ranked[0].theory, strong);
    assert_eq!(ranking.ranked[0].net, 2);
    assert_eq!(ranking.ranked[1].theory, weak);
    assert_eq!(ranking.ranked[1].net, 0);
}

#[test]
fn compare_is_deterministic() {
    let mut w = World::new();
    let e = w.node("ev");
    let a = w.put_theory(
        Theory::builder(Scope::universal())
            .supporting(vec![e])
            .build(),
    );
    let b = w.put_theory(Theory::builder(Scope::universal()).build());

    let engine = Theories::new(&w.store);
    let scope = Scope::universal();
    let r1 = engine.compare(&[a, b], &scope).unwrap();
    let r2 = engine.compare(&[b, a], &scope).unwrap();
    // Ranking is independent of input order.
    assert_eq!(r1, r2);
}

#[test]
fn unknown_theory_is_a_clean_error() {
    let w = World::new();
    let missing = {
        // An id that was never stored.
        let mut w2 = World::new();
        w2.node("nowhere")
    };
    let engine = Theories::new(&w.store);
    assert!(engine.get(missing).is_err());
    assert!(engine.revision_chain(missing).is_err());
}

#[test]
fn a_stored_theory_round_trips_through_the_engine() {
    let mut w = World::new();
    let eq = w.node("field-equations");
    let id = w.put_theory(
        Theory::builder(Scope::from_predicates(["strong-field"]))
            .equations(vec![eq])
            .build(),
    );
    let engine = Theories::new(&w.store);
    let loaded = engine.get(id).unwrap();
    assert_eq!(loaded.equations, vec![eq]);
    assert!(loaded.domain_of_validity.contains(&Scope::from_predicates([
        "strong-field",
        "spherical-symmetry"
    ])));
}
