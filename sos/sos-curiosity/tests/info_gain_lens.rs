//! End-to-end tests for the maximal-information-gain lens: the sanctioned
//! `sos-curiosity` → `sos-planner` composition edge, exercised through the
//! public API only — a caller with planner-supplied designs and a real graph.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, DeterminismLevel, Object, ObjectId};
use sos_curiosity::{
    BeCurious, Budget, Curiosity, CuriosityPolicy, DEFAULT_EIG_SATURATION_MILLI, Strategy,
};
use sos_knowledge::{KnowledgeGraph, Relation, seal_edge};
use sos_planner::{Candidate, Cost, Estimate};
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
            .put_object(&seal_edge(from, to, relation, Author::engine("test")))
            .unwrap();
    }

    fn graph(&self) -> KnowledgeGraph {
        KnowledgeGraph::build(&self.store).unwrap()
    }
}

/// A graph with one contradiction, so the information lens can be compared
/// against a structural question on the same agenda.
fn world_with_a_contradiction() -> (World, ObjectId, ObjectId) {
    let mut w = World::new();
    let (a, b) = (w.node("phlogiston"), w.node("oxygen"));
    w.edge(a, Relation::Contradicts, b);
    (w, a, b)
}

#[test]
fn a_real_design_becomes_a_ranked_explained_question() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();

    let design = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"calorimetry");
    // 1.5 bits, measured exactly, at unit cost.
    let designs = [Candidate::new(
        design,
        Estimate::exact(1_500),
        Cost::new(1, 0, 0, 0),
    )];

    let questions = Curiosity::new(&graph)
        .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
        .sweep(&CuriosityPolicy::default(), &Budget::new(10));

    let info = questions
        .iter()
        .find(|q| q.question.strategy == Strategy::MaxInfoGain)
        .expect("the information lens should raise a question");

    // Grounded in the real design node, explained, and scored on info-gain.
    assert!(info.question.is_grounded());
    assert_eq!(info.question.subject, vec![design]);
    assert_eq!(info.derivation.steps.len(), 1);
    assert!(info.priority.info_gain > 0);
    assert_eq!(info.priority.contradiction, 0);
}

#[test]
fn without_designs_the_sweep_is_unchanged() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();
    let policy = CuriosityPolicy::default();
    let budget = Budget::new(10);

    let plain = Curiosity::new(&graph).sweep(&policy, &budget);
    let empty_designs: [Candidate; 0] = [];
    let with_none = Curiosity::new(&graph)
        .with_designs(&empty_designs, DEFAULT_EIG_SATURATION_MILLI)
        .sweep(&policy, &budget);

    assert_eq!(plain, with_none);
    assert!(
        plain
            .iter()
            .all(|q| q.question.strategy != Strategy::MaxInfoGain)
    );
}

#[test]
fn an_insignificant_estimate_raises_no_question_at_all() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();

    let design = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"noise");
    // 0.02 ± 0.03 bits — the estimate does not exceed its own noise, so the
    // premise "this would teach us something" is not supported.
    let designs = [Candidate::new(
        design,
        Estimate::new(20, 30, DeterminismLevel::L1),
        Cost::new(1, 0, 0, 0),
    )];

    let questions = Curiosity::new(&graph)
        .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
        .sweep(&CuriosityPolicy::default(), &Budget::new(10));

    assert!(
        questions
            .iter()
            .all(|q| q.question.strategy != Strategy::MaxInfoGain),
        "an insignificant EIG must not produce a question"
    );
}

#[test]
fn a_noisy_estimate_ranks_below_an_exact_one_of_the_same_point_value() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();

    let exact_id = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"exact");
    let noisy_id = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"noisy");
    // Identical point estimates and costs; only the uncertainty differs.
    let designs = [
        Candidate::new(exact_id, Estimate::exact(1_800), Cost::new(1, 0, 0, 0)),
        Candidate::new(
            noisy_id,
            Estimate::new(1_800, 1_500, DeterminismLevel::L1),
            Cost::new(1, 0, 0, 0),
        ),
    ];

    let questions = Curiosity::new(&graph)
        .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
        .sweep(&CuriosityPolicy::default(), &Budget::new(10));

    let score_of = |id: ObjectId| {
        questions
            .iter()
            .find(|q| q.question.subject == vec![id])
            .map(|q| q.priority.info_gain)
            .expect("both designs are significant, so both are asked about")
    };
    assert!(
        score_of(exact_id) > score_of(noisy_id),
        "the same claim, measured less certainly, must not score the same"
    );
}

#[test]
fn a_cheaper_design_outranks_an_identical_expensive_one() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();

    let cheap = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"cheap");
    let dear = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"dear");
    let designs = [
        Candidate::new(cheap, Estimate::exact(1_500), Cost::new(1, 0, 0, 0)),
        Candidate::new(dear, Estimate::exact(1_500), Cost::new(50, 0, 0, 0)),
    ];

    let questions = Curiosity::new(&graph)
        .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
        .sweep(&CuriosityPolicy::default(), &Budget::new(10));

    let rank_of = |id: ObjectId| {
        questions
            .iter()
            .position(|q| q.question.subject == vec![id])
            .expect("both designs are asked about")
    };
    assert!(rank_of(cheap) < rank_of(dear));
}

#[test]
fn design_and_structural_questions_share_one_ranked_agenda() {
    let (w, a, b) = world_with_a_contradiction();
    let graph = w.graph();

    let design = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"decisive");
    // A maximally informative, free experiment — the best case the lens can make.
    let designs = [Candidate::new(
        design,
        Estimate::exact(DEFAULT_EIG_SATURATION_MILLI),
        Cost::default(),
    )];

    let questions = Curiosity::new(&graph)
        .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
        .sweep(&CuriosityPolicy::default(), &Budget::new(10));

    // Both lenses are represented, ranked against each other by one policy.
    let strategies: Vec<Strategy> = questions.iter().map(|q| q.question.strategy).collect();
    assert!(strategies.contains(&Strategy::MaxInfoGain));
    assert!(strategies.contains(&Strategy::ContradictionHunt));

    // The default policy deliberately spaces weights so a contradiction
    // (severity 4·SCALE) outranks any non-contradiction total (≤ 3·SCALE) —
    // that ordering must still hold now that info-gain can be non-zero.
    let top = &questions[0];
    assert_eq!(top.question.strategy, Strategy::ContradictionHunt);
    assert_eq!(top.question.subject, {
        let mut v = vec![a, b];
        v.sort_unstable();
        v
    });
}

#[test]
fn the_sweep_stays_deterministic_with_designs() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();

    let designs = [
        Candidate::new(
            ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"one"),
            Estimate::exact(1_200),
            Cost::new(2, 0, 0, 0),
        ),
        Candidate::new(
            ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"two"),
            Estimate::exact(1_200),
            Cost::new(2, 0, 0, 0),
        ),
    ];

    let sweep = || {
        Curiosity::new(&graph)
            .with_designs(&designs, DEFAULT_EIG_SATURATION_MILLI)
            .sweep(&CuriosityPolicy::default(), &Budget::new(10))
    };
    // Even with tied scores, ranking is stable (broken on canonical bytes).
    assert_eq!(sweep(), sweep());
}

#[test]
fn the_saturation_reference_changes_the_score_not_the_question() {
    let (w, _, _) = world_with_a_contradiction();
    let graph = w.graph();

    let design = ObjectId::compute(sos_core::HashAlgo::default(), b"design", b"calibrate");
    let designs = [Candidate::new(
        design,
        Estimate::exact(1_000),
        Cost::new(1, 0, 0, 0),
    )];

    let ask = |saturation: i64| {
        Curiosity::new(&graph)
            .with_designs(&designs, saturation)
            .sweep(&CuriosityPolicy::default(), &Budget::new(10))
            .into_iter()
            .find(|q| q.question.strategy == Strategy::MaxInfoGain)
            .expect("significant design is asked about")
    };

    // 1 bit against a 1-bit reference saturates; against 4 bits it is a quarter.
    let generous = ask(1_000);
    let strict = ask(4_000);
    assert!(generous.priority.info_gain > strict.priority.info_gain);
    // The *question* is the same object either way — only its priority moves.
    assert_eq!(generous.question, strict.question);
}
