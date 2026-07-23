//! Real cross-engine integration. Seal genuine `sos-reasoning`, `sos-theory`,
//! `sos-workflow`, `sos-repro`, and `sos-planner` objects into a content-addressed
//! [`MemoryStore`], then verify a publication whose claims are typed-bound to
//! them — through the production [`StoreSource`] path (type-erased headers
//! decoded straight out of the store). This is the engine consuming the other
//! engines' objects, never recomputing them.

use sos_core::{Author, Body, DeterminismLevel, EnvRecord, HashAlgo, Object, ObjectId};
use sos_store::{MemoryStore, TypedStore};

use sos_publication::{
    BindingRole, Claim, ClaimBinding, ClaimStatus, Publication, ReproRequirement, ReproVerdict,
    StandardPolicy, StoreSource, verify,
};

use sos_planner::{Candidate, Cost, Estimate, GreedyPlanner, Planner, UtilityPolicy};
use sos_reasoning::{Contradiction, Derivation, DerivationStep, Soundness};
use sos_repro::EnvLock;
use sos_theory::{Scope, Theory};
use sos_workflow::RunLedger;

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"pub-integration", tag)
}

fn seal<B: Body>(body: B, level: DeterminismLevel) -> Object<B> {
    Object::builder(body)
        .author(Author::engine("test"))
        .level(level)
        .seal()
}

/// A real graph: a proof derivation, a theory, a workflow run ledger, an
/// environment lock, and a planner design — all sealed into the store — plus a
/// study root that provenance-links them (so they lie in its dependency closure).
struct Graph {
    store: MemoryStore,
    root: ObjectId,
    derivation: ObjectId,
    theory: ObjectId,
    ledger: ObjectId,
    envlock: ObjectId,
    plan: ObjectId,
}

fn build_graph(evidence_level: DeterminismLevel) -> Graph {
    let mut store = MemoryStore::new();

    // sos-reasoning: a proof-grade derivation.
    let derivation = seal(
        Derivation::new(
            "Period squared scales with semi-major axis cubed",
            vec![DerivationStep::new("algebra", Vec::new(), "P^2 = k a^3")],
            Vec::new(),
            Soundness::Proof,
        ),
        evidence_level,
    );
    store.put_object(&derivation).unwrap();

    // sos-theory: the governing theory.
    let theory = seal(Theory::builder(Scope::universal()).build(), evidence_level);
    store.put_object(&theory).unwrap();

    // sos-workflow: the experiment/simulation execution record.
    let digest = HashAlgo::default().hash(b"pub-integration", b"plan-digest");
    let ledger = seal(
        RunLedger {
            plan_digest: digest,
            env_digest: digest,
            steps: Vec::new(),
        },
        evidence_level,
    );
    store.put_object(&ledger).unwrap();

    // sos-repro: the pinned environment the result was produced in.
    let envlock = seal(
        EnvLock::pin(EnvRecord::new(
            "1.89.0-stable",
            Vec::new(),
            "x86_64/avx2",
            "linux",
        )),
        evidence_level,
    );
    store.put_object(&envlock).unwrap();

    // sos-planner: the design that was executed.
    let candidate = Candidate::new(
        oid(b"experiment-design"),
        Estimate::exact(2000),
        Cost::new(1, 1, 1, 0),
    );
    let plan_body = GreedyPlanner::new()
        .recommend(&[candidate], UtilityPolicy::EigPerCost, 100)
        .unwrap();
    let plan = seal(plan_body, evidence_level);
    store.put_object(&plan).unwrap();

    // The study root, provenance-linked to every piece so they are in scope.
    let root = Object::builder(Derivation::undetermined("Kepler study"))
        .parents(vec![
            derivation.id,
            theory.id,
            ledger.id,
            envlock.id,
            plan.id,
        ])
        .author(Author::human("ada"))
        .level(evidence_level)
        .seal();
    store.put_object(&root).unwrap();

    Graph {
        store,
        root: root.id,
        derivation: derivation.id,
        theory: theory.id,
        ledger: ledger.id,
        envlock: envlock.id,
        plan: plan.id,
    }
}

#[test]
fn a_claim_bound_across_five_engines_verifies_as_supported() {
    let g = build_graph(DeterminismLevel::L3);
    let source = StoreSource::new(&g.store);

    let paper = Publication::builder("Kepler's Third Law, Rederived")
        .author(Author::human("ada"))
        .declared_root(g.root)
        .reproducibility(ReproRequirement::MinimumLevel(DeterminismLevel::L3))
        .claim(Claim::new(
            "C1",
            "Period squared scales with semi-major axis cubed.",
            vec![
                ClaimBinding::new(BindingRole::DirectlySupports, g.derivation),
                ClaimBinding::new(BindingRole::IndirectlySupports, g.theory),
                ClaimBinding::new(BindingRole::SuppliesData, g.ledger),
                ClaimBinding::new(BindingRole::SuppliesMethod, g.plan),
                ClaimBinding::new(BindingRole::Reproduces, g.envlock),
            ],
        ))
        .build();

    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    let claim = &report.claims[0];
    assert_eq!(claim.status, ClaimStatus::Supported);
    // Every engine object resolved, in scope, as supporting evidence.
    assert_eq!(claim.supporting.len(), 5);
    assert!(claim.supporting.contains(&g.derivation));
    assert!(claim.supporting.contains(&g.envlock));
    assert!(claim.missing.is_empty());
    assert!(claim.out_of_scope.is_empty());
    // Closure = root + five engine objects.
    assert_eq!(report.closure_size, 6);
    assert!(matches!(
        report.reproducibility,
        ReproVerdict::Satisfied { .. }
    ));
    assert!(report.is_publishable());
}

#[test]
fn a_weak_environment_fails_the_reproducibility_bar() {
    // The same graph, but every object realized only statistical reproducibility.
    let g = build_graph(DeterminismLevel::L1);
    let source = StoreSource::new(&g.store);

    let paper = Publication::builder("Overclaimed")
        .declared_root(g.root)
        .reproducibility(ReproRequirement::MinimumLevel(DeterminismLevel::L3))
        .claim(Claim::new(
            "C1",
            "Claims bit-reproducibility it does not have.",
            vec![ClaimBinding::new(
                BindingRole::DirectlySupports,
                g.derivation,
            )],
        ))
        .build();

    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::ReproducibilityFailed);
    assert!(!report.is_publishable());
}

#[test]
fn a_contradiction_object_is_surfaced_from_the_graph() {
    let mut g = build_graph(DeterminismLevel::L3);
    // Seal a real sos-reasoning Contradiction and store it.
    let contradiction = seal(
        Contradiction::new(
            g.derivation,
            g.theory,
            "the derivation and the theory disagree",
        ),
        DeterminismLevel::L3,
    );
    g.store.put_object(&contradiction).unwrap();
    let contra_id = contradiction.id;
    let source = StoreSource::new(&g.store);

    let paper = Publication::builder("Disputed")
        .declared_root(g.root)
        .claim(Claim::new(
            "C1",
            "A disputed statement.",
            vec![
                ClaimBinding::new(BindingRole::DirectlySupports, g.derivation),
                ClaimBinding::new(BindingRole::Contradicts, contra_id),
            ],
        ))
        .build();

    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    // Contradiction is never hidden and dominates the support.
    assert_eq!(report.claims[0].status, ClaimStatus::Contradicted);
    assert_eq!(report.claims[0].contradicting, vec![contra_id]);
    assert!(!report.is_publishable());
}

#[test]
fn a_claim_pointing_outside_the_stored_graph_is_missing() {
    let g = build_graph(DeterminismLevel::L3);
    let source = StoreSource::new(&g.store);

    let paper = Publication::builder("Dangling")
        .declared_root(g.root)
        .claim(Claim::new(
            "C1",
            "Rests on an object that was never stored.",
            vec![ClaimBinding::new(
                BindingRole::DirectlySupports,
                oid(b"never-stored"),
            )],
        ))
        .build();

    let report = verify(&paper, &source, &StandardPolicy::new()).unwrap();
    assert_eq!(report.claims[0].status, ClaimStatus::DependencyMissing);
    assert_eq!(report.missing_objects, vec![oid(b"never-stored")]);
}

#[test]
fn reproducibility_evidence_is_grounded_in_the_repro_engine() {
    // The `Reproduces` binding in a publication points at repro-engine evidence;
    // here we exercise that engine's own verdict for the same kind of node.
    use sos_repro::{NodeClaim, Reproduced, verify_reproduction};

    let node = oid(b"repro-node");
    let claims = [NodeClaim::new(node, DeterminismLevel::L3)];
    let reproduced = [Reproduced::Id(node)];
    let report = verify_reproduction(&claims, &reproduced).unwrap();
    assert!(report.reproduced());
    assert_eq!(report.level, DeterminismLevel::L3);
}

#[test]
fn a_simulation_observation_is_deterministic_evidence() {
    // `sos-simulation` observations are the reproducible data a workflow ledger
    // records; their content address is stable, which is what makes them citable.
    use sos_simulation::Observation;

    let a = Observation::new(42_i64, DeterminismLevel::L3, 7);
    let b = Observation::new(42_i64, DeterminismLevel::L3, 7);
    assert_eq!(a.digest(), b.digest());
    assert_eq!(a.level(), DeterminismLevel::L3);
    // A different datum has a different address.
    let c = Observation::new(43_i64, DeterminismLevel::L3, 7);
    assert_ne!(a.digest(), c.digest());
}
