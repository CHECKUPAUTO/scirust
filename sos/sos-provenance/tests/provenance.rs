//! Integration tests: provenance queries over a real object lineage, and
//! environment capture feeding a reproducibility key.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, HashAlgo, Object, ObjectId, ReproMeta, RngId};
use sos_provenance::{EnvCapture, ProvenanceGraph, ancestors, descendants};
use sos_store::{MemoryStore, TypedStore};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Step {
    label: String,
}
impl Canonical for Step {
    fn encode(&self, e: &mut CanonicalEncoder) {
        e.str(&self.label);
    }
}
impl Body for Step {
    const KIND: &'static str = "Step";
    const SCHEMA_VERSION: u32 = 1;
}

fn step(label: &str, parents: Vec<ObjectId>) -> Object<Step> {
    Object::builder(Step {
        label: label.into(),
    })
    .author(Author::human("t"))
    .parents(parents)
    .seal()
}

#[test]
fn provenance_over_a_real_discovery_lineage() {
    // q -> h -> p -> e   (a discovery chain), plus q -> h2 (a rival hypothesis).
    let mut s = MemoryStore::new();
    let q = step("question", vec![]);
    let h = step("hypothesis", vec![q.id]);
    let p = step("prediction", vec![h.id]);
    let e = step("evidence", vec![p.id]);
    let h2 = step("rival-hypothesis", vec![q.id]);
    for o in [&q, &h, &p, &e, &h2]
    {
        s.put_object(o).unwrap();
    }

    let g = ProvenanceGraph::build(&s).unwrap();
    assert_eq!(g.len(), 5);

    // "Why do we believe the evidence?" — its whole derivation chain.
    let mut anc = g.ancestors(e.id);
    let mut want = vec![q.id, h.id, p.id];
    anc.sort();
    want.sort();
    assert_eq!(anc, want);

    // "What depends on the question?" — everything in the study.
    let mut desc = g.descendants(q.id);
    let mut want_desc = vec![h.id, p.id, e.id, h2.id];
    desc.sort();
    want_desc.sort();
    assert_eq!(desc, want_desc);

    // The root is the question; the tips are the evidence and the untested rival.
    assert_eq!(g.roots(), vec![q.id]);
    let mut tips = g.tips();
    let mut want_tips = vec![e.id, h2.id];
    tips.sort();
    want_tips.sort();
    assert_eq!(tips, want_tips);

    // Free-function convenience matches the graph.
    assert_eq!(ancestors(&s, e.id).unwrap(), g.ancestors(e.id));
    assert_eq!(descendants(&s, q.id).unwrap(), g.descendants(q.id));

    // Direct edges.
    assert_eq!(g.direct_parents(e.id), &[p.id]);
    assert_eq!(g.direct_children(q.id), {
        let mut c = vec![h.id, h2.id];
        c.sort();
        c
    });
}

#[test]
fn empty_store_has_empty_graph() {
    let s = MemoryStore::new();
    let g = ProvenanceGraph::build(&s).unwrap();
    assert!(g.is_empty());
    assert!(g.roots().is_empty());
    assert!(g.tips().is_empty());
}

#[test]
fn captured_env_feeds_a_reproducible_object() {
    // Build an env, put its digest in a ReproMeta, seal an object with it, and
    // confirm it round-trips through the store with a matching reproducibility
    // key.
    let env = EnvCapture::new("1.89.0-stable").build();
    let repro = ReproMeta::new(7, RngId::new("SplitMix64"), env.digest(HashAlgo::default()));

    let obj = Object::builder(Step {
        label: "computed".into(),
    })
    .author(Author::engine("sos-reasoning"))
    .repro(repro.clone())
    .seal();

    let mut s = MemoryStore::new();
    let id = s.put_object(&obj).unwrap();
    let back: Object<Step> = s.get_object(id).unwrap().unwrap();
    assert_eq!(back.repro.env_digest, env.digest(HashAlgo::default()));
    assert_eq!(back.repro.seed, 7);
    assert!(back.verify_id());
}
