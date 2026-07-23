//! Integration test: a small scientific knowledge graph over real objects.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};
use sos_knowledge::{Knowledge, KnowledgeGraph, Relation, seal_edge};
use sos_store::{MemoryStore, TypedStore};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Law {
    name: String,
}
impl Canonical for Law {
    fn encode(&self, e: &mut CanonicalEncoder) {
        e.str(&self.name);
    }
}
impl Body for Law {
    const KIND: &'static str = "Law";
    const SCHEMA_VERSION: u32 = 1;
}

fn law(store: &mut MemoryStore, name: &str) -> ObjectId {
    let obj = Object::builder(Law { name: name.into() })
        .author(Author::human("researcher"))
        .seal();
    store.put_object(&obj).unwrap()
}

#[test]
fn a_small_scientific_knowledge_graph() {
    let mut s = MemoryStore::new();
    let a = Author::engine("sos-knowledge");

    // Nodes: laws across two domains.
    let kepler = law(&mut s, "kepler-3");
    let newton = law(&mut s, "newton-gravity");
    let relativity = law(&mut s, "general-relativity");
    let oscillator = law(&mut s, "damped-oscillator"); // physics
    let mean_reversion = law(&mut s, "ornstein-uhlenbeck"); // finance

    // Edges: the scientific relationships.
    for e in [
        seal_edge(kepler, newton, Relation::Specializes, a.clone()),
        seal_edge(relativity, newton, Relation::Supersedes, a.clone()),
        seal_edge(newton, relativity, Relation::LimitOf, a.clone()),
        // A cross-domain structural analogy (the standout Curiosity capability).
        seal_edge(oscillator, mean_reversion, Relation::AnalogousTo, a.clone()),
    ]
    {
        s.put_object(&e).unwrap();
    }

    let kg = KnowledgeGraph::build(&s).unwrap();
    assert_eq!(kg.edge_count(), 4);
    assert_eq!(kg.len(), 5); // five distinct endpoints

    // "What does Kepler's third law specialize?" -> Newton's gravity.
    assert_eq!(kg.neighbors(kepler, &Relation::Specializes), vec![newton]);

    // "What supersedes Newton?" -> general relativity (an in-edge).
    assert_eq!(
        kg.in_neighbors(newton, &Relation::Supersedes),
        vec![relativity]
    );

    // Newton is the low-velocity limit of relativity, which in turn supersedes
    // it — a small cycle across two relations; a relation-agnostic path exists.
    let p = kg.path(kepler, relativity, None).unwrap();
    assert_eq!(p.first(), Some(&kepler));
    assert_eq!(p.last(), Some(&relativity));

    // The cross-domain analogy connects the two otherwise-disconnected domains.
    assert_eq!(
        kg.related(oscillator, mean_reversion),
        vec![Relation::AnalogousTo]
    );
    // But there is no path from a physics law to the finance law except through
    // that analogy edge — following only `specializes`, they are disconnected.
    assert_eq!(
        kg.path(kepler, mean_reversion, Some(&Relation::Specializes)),
        None
    );

    // Building the graph twice yields identical, deterministic results.
    let kg2 = KnowledgeGraph::build(&s).unwrap();
    assert_eq!(kg.nodes(), kg2.nodes());
    assert_eq!(kg.edges(), kg2.edges());
}
