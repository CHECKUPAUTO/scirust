//! End-to-end reasoning over a real [`KnowledgeGraph`] built from a store:
//! transitive entailment producing an explanatory [`Derivation`], the honest
//! `Undetermined`/`Check` outcome, and contradiction detection.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};
use sos_knowledge::{KnowledgeGraph, Relation, seal_edge};
use sos_reasoning::{Reason, Reasoner, Soundness, Verdict};
use sos_store::{MemoryStore, TypedStore};

/// A minimal knowledge node used to populate the store.
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

/// A fresh world: seal a node object per name and remember its id.
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

    fn edge(&mut self, from: ObjectId, relation: Relation, to: ObjectId) {
        self.store
            .put_object(&seal_edge(
                from,
                to,
                relation,
                Author::engine("sos-reasoning-test"),
            ))
            .unwrap();
    }

    fn graph(&self) -> KnowledgeGraph {
        KnowledgeGraph::build(&self.store).unwrap()
    }
}

#[test]
fn transitive_chain_yields_a_multi_step_proof() {
    // electron ⊂ lepton ⊂ fermion ⊂ particle : three specialization hops.
    let mut w = World::new();
    let electron = w.node("electron");
    let lepton = w.node("lepton");
    let fermion = w.node("fermion");
    let particle = w.node("particle");
    w.edge(electron, Relation::Specializes, lepton);
    w.edge(lepton, Relation::Specializes, fermion);
    w.edge(fermion, Relation::Specializes, particle);

    let kg = w.graph();
    let r = Reasoner::new(&kg);
    let c = r.entails(electron, &Relation::Specializes, particle);

    assert_eq!(c.verdict, Verdict::Proven);
    assert_eq!(c.derivation.soundness, Soundness::Proof); // transitivity is sound
    assert_eq!(c.derivation.steps.len(), 3); // three edges chained
    assert_eq!(c.derivation.premises.len(), 3); // three edges cited
    // Every step is a transitivity step over `specializes`.
    for step in &c.derivation.steps
    {
        assert_eq!(step.rule, "transitivity(specializes)");
    }
}

#[test]
fn direct_edge_is_a_one_step_proof() {
    let mut w = World::new();
    let a = w.node("a");
    let b = w.node("b");
    w.edge(a, Relation::Implies, b);

    let kg = w.graph();
    let r = Reasoner::new(&kg);
    let c = r.entails(a, &Relation::Implies, b);

    assert_eq!(c.verdict, Verdict::Proven);
    assert_eq!(c.derivation.soundness, Soundness::Proof);
    assert_eq!(c.derivation.steps.len(), 1);
    assert_eq!(c.derivation.steps[0].rule, "direct-edge");
}

#[test]
fn missing_link_is_undetermined_not_refuted() {
    // a specializes b, but nothing connects b to an unrelated c.
    let mut w = World::new();
    let a = w.node("a");
    let b = w.node("b");
    let c = w.node("c");
    w.edge(a, Relation::Specializes, b);

    let kg = w.graph();
    let r = Reasoner::new(&kg);
    let concl = r.entails(a, &Relation::Specializes, c);

    // Honest: not found ≠ disproved.
    assert_eq!(concl.verdict, Verdict::Undetermined);
    assert_eq!(concl.derivation.soundness, Soundness::Check);
    assert!(concl.derivation.steps.is_empty());
}

#[test]
fn non_transitive_relation_gets_no_closure() {
    // a cites b, b cites c — but `cites` is not transitive, so a⇒c is NOT derived
    // (only a directly-asserted `cites` edge would count).
    let mut w = World::new();
    let a = w.node("a");
    let b = w.node("b");
    let c = w.node("c");
    w.edge(a, Relation::Cites, b);
    w.edge(b, Relation::Cites, c);

    let kg = w.graph();
    let r = Reasoner::new(&kg);

    // The chained query is undetermined...
    assert_eq!(
        r.entails(a, &Relation::Cites, c).verdict,
        Verdict::Undetermined
    );
    // ...but each directly-asserted hop is a one-step proof.
    assert_eq!(r.entails(a, &Relation::Cites, b).verdict, Verdict::Proven);
    assert_eq!(r.entails(b, &Relation::Cites, c).verdict, Verdict::Proven);
}

#[test]
fn derivation_seals_to_a_verifiable_object() {
    let mut w = World::new();
    let a = w.node("a");
    let b = w.node("b");
    let c = w.node("c");
    w.edge(a, Relation::DerivesFrom, b);
    w.edge(b, Relation::DerivesFrom, c);

    let kg = w.graph();
    let r = Reasoner::new(&kg);
    let concl = r.entails(a, &Relation::DerivesFrom, c);
    assert_eq!(concl.verdict, Verdict::Proven);

    // The explanation is itself a content-addressed, independently verifiable
    // object.
    let obj = Object::builder(concl.derivation.clone())
        .author(Author::engine("sos-reasoning"))
        .seal();
    assert!(obj.verify_id());
    assert_eq!(obj.kind.name, "Derivation");
    // Content-addressed: re-sealing the same derivation with the same provenance
    // reproduces the same id.
    let obj2 = Object::builder(concl.derivation)
        .author(Author::engine("sos-reasoning"))
        .seal();
    assert_eq!(obj.id, obj2.id);
}

#[test]
fn detects_asserted_and_mutual_contradictions() {
    let mut w = World::new();
    let p = w.node("phlogiston-theory");
    let q = w.node("oxygen-theory");
    let m = w.node("model-a");
    let n = w.node("model-b");
    // An asserted contradiction.
    w.edge(p, Relation::Contradicts, q);
    // A mutual-supersession cycle (each claims to replace the other).
    w.edge(m, Relation::Supersedes, n);
    w.edge(n, Relation::Supersedes, m);

    let kg = w.graph();
    let r = Reasoner::new(&kg);
    let contradictions = r.contradictions();

    // Two distinct incompatibilities, deduplicated.
    assert_eq!(contradictions.len(), 2);
    let reasons: Vec<&str> = contradictions.iter().map(|c| c.reason.as_str()).collect();
    assert!(reasons.contains(&"asserted-contradiction"));
    assert!(reasons.contains(&"mutual-supersession"));

    // Deterministic: same graph ⇒ same contradiction list, byte-for-byte.
    assert_eq!(r.contradictions(), contradictions);
    for c in &contradictions
    {
        assert!(c.left <= c.right); // endpoints normalised
    }
}

#[test]
fn no_contradictions_in_a_coherent_graph() {
    let mut w = World::new();
    let a = w.node("a");
    let b = w.node("b");
    let c = w.node("c");
    w.edge(a, Relation::Specializes, b);
    w.edge(b, Relation::Supersedes, c); // one-directional: not a contradiction

    let kg = w.graph();
    let r = Reasoner::new(&kg);
    assert!(r.contradictions().is_empty());
}
