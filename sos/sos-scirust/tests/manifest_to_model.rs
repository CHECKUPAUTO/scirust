//! A study written as text runs a **named scientific model**, end to end.
//!
//! `tests/manifest_to_result.rs` already shows a TOML manifest reaching real
//! numerics, but through [`OdeStageHandler`](sos_scirust::stage::OdeStageHandler),
//! whose model is a closure the handler closes over. That path computes
//! something real; what it cannot do is *say what*. The manifest names a
//! plugin and a config digest, and the physics comes from whichever right-hand
//! side happened to be compiled in — so the same study text, run against a
//! differently-built handler, is a different experiment.
//!
//! With the model inside the config's content address that gap closes, and
//! this file is where the claim becomes checkable rather than argued:
//!
//! * the study's `config` digests are the real addresses of two [`ModelRun`]s,
//!   written out as an author would have to write them;
//! * before anything runs, the plan can be **audited** — asked which model each
//!   stage will integrate;
//! * after it runs, each stored object still names the model that produced it;
//! * and the numbers are checked against `scirust-sim`'s own analytic oracle,
//!   so "it ran" means "it integrated the epidemic it said it would".
//!
//! Scope, stated as narrowly as it deserves: this is one wired plugin and a
//! hand-assembled `Dispatch`. `sos run` reaches the same handler from the
//! command line, but through files rather than this Rust API.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{HashAlgo, Object, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry, Role};
use sos_scirust::model::{ModelKind, ModelRun, ModelSpec, ModeledTrajectoryBody};
use sos_scirust::stage::{CatalogStageHandler, model_config_address, sim_backend};
use sos_store::{MemoryStore, ObjectStore, TypedStore};
use sos_workflow::{Dispatch, MemoTable, resolve_manifest, run_plan};

const PLUGIN: &str = "sim-catalog";

fn plugin_hash() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"plugin", b"sim-catalog@1.0.0")
}

fn env() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"env", b"test")
}

/// The same epidemic at two transmission rates — `R₀ = β/γ` of 3 and of 1.2,
/// which is the difference between an outbreak and a fizzle. A study a
/// scientist might actually write.
fn epidemic(beta: f64) -> ModelRun {
    ModelRun::new(
        ModelSpec::new(ModelKind::Sir, [beta, 0.2]),
        [0.99, 0.01, 0.0],
        0.0,
        120.0,
        0.01,
    )
}

fn virulent() -> ModelRun {
    epidemic(0.6)
}
fn mild() -> ModelRun {
    epidemic(0.24)
}

fn study_text() -> String {
    format!(
        r#"
[study]
name = "sir-transmission-sweep"
seed = 7

[[stage]]
id      = "virulent"
plugin  = "{PLUGIN}"
version = "1.0.0"
pin     = "{pin}"
config  = "{c_virulent}"

[[stage]]
id      = "mild"
plugin  = "{PLUGIN}"
version = "1.0.0"
pin     = "{pin}"
config  = "{c_mild}"
needs   = ["virulent"]
"#,
        pin = plugin_hash().to_hex(),
        c_virulent = model_config_address(&virulent()).to_hex(),
        c_mild = model_config_address(&mild()).to_hex(),
    )
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register(
        PluginDescriptor::new(
            PLUGIN,
            SemVer::new(1, 0, 0),
            plugin_hash(),
            Role::Simulation,
        )
        .needs(Capability::Effectful),
    );
    registry
}

/// The handler, with both runs on offer, plus the shared store.
fn wired() -> (CatalogStageHandler<MemoryStore>, Rc<RefCell<MemoryStore>>) {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut handler = CatalogStageHandler::new(
        Rc::clone(&store),
        sim_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"backend", b"scirust-sim"),
        ),
    );
    handler.offer(virulent());
    handler.offer(mild());
    (handler, store)
}

#[test]
fn a_study_written_as_text_runs_the_model_it_names() {
    let (handler, store) = wired();
    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(handler));

    let plan = resolve_manifest(&study_text()).expect("the study must resolve");
    let ledger = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap();

    assert_eq!(ledger.ran_count(), 2);
    assert_eq!(ledger.steps[0].stage.0, "virulent");
    assert_eq!(ledger.steps[1].stage.0, "mild");

    // Each stored result names its own model, recovered from the object.
    let outputs: Vec<Object<ModeledTrajectoryBody>> = ledger
        .steps
        .iter()
        .map(|step| {
            store
                .borrow()
                .get_object(step.outputs[0])
                .unwrap()
                .expect("every stage's output must be in the store")
        })
        .collect();
    for out in &outputs
    {
        assert_eq!(out.body.model.kind, ModelKind::Sir);
        // The study's seed reached the run.
        assert_eq!(out.body.seed, 7);
    }
    assert_eq!(outputs[0].body.model.params, vec![0.6, 0.2]);
    assert_eq!(outputs[1].body.model.params, vec![0.24, 0.2]);

    // And real epidemiology happened: R0 = 3 burns through the population,
    // R0 = 1.2 barely moves it. `scirust-sim`'s SIR is validated against its
    // own final-size oracle, so these are its physics, not this test's.
    let susceptible_left =
        |o: &Object<ModeledTrajectoryBody>| o.body.trajectory.last().unwrap().1[0];
    assert!(
        susceptible_left(&outputs[0]) < 0.1,
        "R0 = 3 should infect nearly everyone, got {}",
        susceptible_left(&outputs[0])
    );
    assert!(
        susceptible_left(&outputs[1]) > 0.6,
        "R0 = 1.2 should leave most of the population, got {}",
        susceptible_left(&outputs[1])
    );
}

#[test]
fn the_plan_can_be_audited_before_a_single_step_runs() {
    // The reviewable property the closure-based path cannot offer: read the
    // study, resolve it, and ask what science each stage will do — without
    // running anything and without reading the handler's source.
    let (handler, _store) = wired();
    let plan = resolve_manifest(&study_text()).unwrap();

    let mut seen = Vec::new();
    for id in plan.schedule()
    {
        let stage = plan
            .stages()
            .iter()
            .find(|s| s.id == id)
            .expect("scheduled stages are in the plan");
        let model = handler
            .model_at(&stage.config_hash)
            .expect("every stage's model is resolvable from its config address");
        seen.push((id.0.clone(), model.kind, model.params.clone()));
    }

    assert_eq!(
        seen,
        vec![
            ("virulent".to_owned(), ModelKind::Sir, vec![0.6, 0.2]),
            ("mild".to_owned(), ModelKind::Sir, vec![0.24, 0.2]),
        ]
    );
}

#[test]
fn a_study_naming_a_config_nobody_offers_fails_rather_than_substituting() {
    // A manifest pinned to a model run this deployment does not have must not
    // quietly run a different one.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let handler = CatalogStageHandler::new(
        Rc::clone(&store),
        sim_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"backend", b"scirust-sim"),
        ),
    ); // nothing offered

    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(handler));

    let plan = resolve_manifest(&study_text()).unwrap();
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(
        err.to_string().contains("no model run is on offer"),
        "{err}"
    );
    assert!(
        store.borrow().object_ids().is_empty(),
        "a refused study stores nothing"
    );
}

#[test]
fn rerunning_the_same_study_memoizes_instead_of_recomputing() {
    let (handler, _store) = wired();
    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(handler));

    let plan = resolve_manifest(&study_text()).unwrap();
    let mut memo = MemoTable::new();
    let first = run_plan(&plan, &env(), &mut memo, &mut dispatch).unwrap();
    let second = run_plan(&plan, &env(), &mut memo, &mut dispatch).unwrap();

    assert_eq!(first.ran_count(), 2);
    assert_eq!(second.ran_count(), 0, "the second run is entirely cached");
    for (a, b) in first.steps.iter().zip(&second.steps)
    {
        assert_eq!(a.outputs, b.outputs);
    }
}
