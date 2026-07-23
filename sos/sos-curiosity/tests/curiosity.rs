//! End-to-end curiosity sweep over a real [`KnowledgeGraph`]: the three
//! deterministic scanners, ranked and explained, plus determinism, grounding,
//! and budget bounds.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};
use sos_curiosity::{BeCurious, Budget, Curiosity, CuriosityPolicy, Strategy, seal_question};
use sos_knowledge::{KnowledgeGraph, Relation, seal_edge};
use sos_store::{MemoryStore, TypedStore};

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

    fn edge(&mut self, from: ObjectId, relation: Relation, to: ObjectId) {
        self.store
            .put_object(&seal_edge(
                from,
                to,
                relation,
                Author::engine("sos-curiosity-test"),
            ))
            .unwrap();
    }

    fn graph(&self) -> KnowledgeGraph {
        KnowledgeGraph::build(&self.store).unwrap()
    }
}

/// Build a small scientific world exercising all three lenses:
/// - `phlogiston` **contradicts** `oxygen` (contradiction lens),
/// - `combustion` is a pendant (degree-1) node while `oxygen` is well-connected
///   (under-connected lens),
/// - `caloric` is **refuted-by** `experiment` with no support (weakly-supported lens).
///
/// Note: a node only exists in a [`KnowledgeGraph`] if some edge references it,
/// so every node here is deliberately given at least one edge.
fn scientific_world() -> (World, [ObjectId; 5]) {
    let mut w = World::new();
    let phlogiston = w.node("phlogiston-theory");
    let oxygen = w.node("oxygen-theory");
    let combustion = w.node("combustion");
    let caloric = w.node("caloric-theory");
    let experiment = w.node("rumford-cannon-experiment");

    w.edge(phlogiston, Relation::Contradicts, oxygen);
    w.edge(oxygen, Relation::Specializes, combustion);
    w.edge(oxygen, Relation::SupportedBy, experiment);
    w.edge(caloric, Relation::RefutedBy, experiment);

    (w, [phlogiston, oxygen, combustion, caloric, experiment])
}

#[test]
fn sweep_finds_all_three_lenses_each_explained() {
    let (w, [phlogiston, oxygen, combustion, caloric, _experiment]) = scientific_world();
    let kg = w.graph();
    let questions = Curiosity::new(&kg).sweep(&CuriosityPolicy::default(), &Budget::new(50));

    // Every question is grounded, explained, and positively scored.
    assert!(!questions.is_empty());
    for q in &questions
    {
        assert!(q.question.is_grounded());
        assert!(!q.derivation.steps.is_empty());
        assert!(q.priority.total > 0);
    }

    let subjects_for = |s: Strategy| -> Vec<Vec<ObjectId>> {
        questions
            .iter()
            .filter(|q| q.question.strategy == s)
            .map(|q| q.question.subject.clone())
            .collect()
    };

    // Contradiction lens: exactly the phlogiston/oxygen conflict.
    let contradictions = subjects_for(Strategy::ContradictionHunt);
    assert_eq!(contradictions.len(), 1);
    assert_eq!(contradictions[0], {
        let mut v = vec![phlogiston, oxygen];
        v.sort_unstable();
        v
    });

    // Under-connected lens: the pendant `combustion` node is flagged; the
    // well-connected oxygen node (degree 3) is not.
    let under = subjects_for(Strategy::UnderConnected);
    assert!(under.iter().any(|subj| subj == &vec![combustion]));
    assert!(under.iter().all(|subj| subj != &vec![oxygen]));

    // Weakly-supported lens: caloric is refuted with no support.
    let weak = subjects_for(Strategy::WeaklySupported);
    assert_eq!(weak.len(), 1);
    assert_eq!(weak[0], vec![caloric]);
}

#[test]
fn contradiction_is_ranked_first_under_default_policy() {
    let (w, _ids) = scientific_world();
    let kg = w.graph();
    let questions = Curiosity::new(&kg).sweep(&CuriosityPolicy::default(), &Budget::new(50));
    // The contradiction term gives it the highest total.
    assert_eq!(questions[0].question.strategy, Strategy::ContradictionHunt);
}

#[test]
fn sweep_is_deterministic() {
    let (w, _ids) = scientific_world();
    let kg = w.graph();
    let engine = Curiosity::new(&kg);
    let a = engine.sweep(&CuriosityPolicy::default(), &Budget::new(50));
    let b = engine.sweep(&CuriosityPolicy::default(), &Budget::new(50));
    assert_eq!(a, b);
    // Rebuilding the graph from the same store yields the same agenda.
    let kg2 = w.graph();
    let c = Curiosity::new(&kg2).sweep(&CuriosityPolicy::default(), &Budget::new(50));
    assert_eq!(a, c);
}

#[test]
fn budget_bounds_the_sweep_keeping_the_top_ranked() {
    let (w, _ids) = scientific_world();
    let kg = w.graph();
    let engine = Curiosity::new(&kg);
    let full = engine.sweep(&CuriosityPolicy::default(), &Budget::new(50));
    let capped = engine.sweep(&CuriosityPolicy::default(), &Budget::new(1));
    assert_eq!(capped.len(), 1);
    assert_eq!(capped[0], full[0]); // the cap keeps the highest-priority question
}

#[test]
fn policy_weights_change_the_ranking() {
    // With contradiction weight zeroed, a contradiction no longer dominates.
    let (w, _ids) = scientific_world();
    let kg = w.graph();
    let engine = Curiosity::new(&kg);

    let policy = CuriosityPolicy {
        w_contradiction: 0,
        w_novelty: 100,
        ..CuriosityPolicy::default()
    };
    let ranked = engine.sweep(&policy, &Budget::new(50));
    // Now the most novel (most isolated) question ranks first — an isolated node
    // has degree 0, maximal novelty.
    assert_ne!(ranked[0].question.strategy, Strategy::ContradictionHunt);
}

#[test]
fn empty_graph_yields_no_questions() {
    let w = World::new();
    let kg = w.graph();
    let questions = Curiosity::new(&kg).sweep(&CuriosityPolicy::default(), &Budget::default());
    assert!(questions.is_empty());
}

#[test]
fn a_question_seals_to_a_verifiable_object() {
    let (w, _ids) = scientific_world();
    let kg = w.graph();
    let questions = Curiosity::new(&kg).sweep(&CuriosityPolicy::default(), &Budget::new(50));
    let obj = seal_question(
        questions[0].question.clone(),
        Author::engine("sos-curiosity"),
    );
    assert!(obj.verify_id());
    assert_eq!(obj.kind.name, "ScientificQuestion");
}
