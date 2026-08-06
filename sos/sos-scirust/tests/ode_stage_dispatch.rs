//! The first end-to-end SOS execution path: a [`Plan`] whose stage names an
//! ODE plugin, run through [`run_plan`] over a [`Dispatch`] that binds it to
//! [`OdeStageHandler`], which integrates it with `scirust-solvers`' RK4 and
//! seals the trajectory into an object store.
//!
//! Every layer here is the real one — real scheduler, real registry pinning,
//! real capability authorization, real solver, real content-addressed
//! storage. Nothing is stubbed, and nothing is asserted that this one path
//! does not actually demonstrate: it is a single `L3` ODE backend behind a
//! hand-constructed plan, not a general "SOS runs studies" claim. Manifest
//! resolution (TOML study → `Plan`) remains deferred, and no other backend or
//! engine has a handler.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{HashAlgo, Object, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry, Role};
use sos_scirust::ode::{OdeConfig, Rk4OdeSimulator};
use sos_scirust::stage::{OdeStageHandler, TrajectoryBody, solvers_backend};
use sos_simulation::SimDescriptor;
use sos_store::{MemoryStore, ObjectStore, TypedStore};
use sos_workflow::{
    Dispatch, MemoTable, Plan, Stage, StageDescriptor, StageId, WorkflowError, run_plan,
};

const PLUGIN: &str = "ode-rk4";

/// Exponential decay `dy/dt = -y`, whose exact solution `y(t) = y0 · e^(-t)`
/// makes "did a real integration happen?" a checkable question.
fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}

type Handler = OdeStageHandler<fn(f64, &[f64], &mut [f64]), MemoryStore>;

fn handler(store: &Rc<RefCell<MemoryStore>>) -> Handler {
    OdeStageHandler::new(
        Rk4OdeSimulator::new(
            SimDescriptor::new("scirust-solvers/ode-rk4", SemVer::new(1, 0, 0)),
            decay as fn(f64, &[f64], &mut [f64]),
        ),
        Rc::clone(store),
        solvers_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"backend", b"scirust-solvers"),
        ),
    )
}

/// The plugin's content hash — what the registry holds and the stage pins.
fn plugin_hash() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"plugin", b"ode-rk4@1.0.0")
}

/// A registry holding the ODE plugin, declaring the one capability a solver
/// that produces recorded results needs.
fn registry_with_plugin(hash: sos_core::Digest) -> Registry {
    let mut registry = Registry::new();
    registry.register(
        PluginDescriptor::new(PLUGIN, SemVer::new(1, 0, 0), hash, Role::Simulation)
            .needs(Capability::Effectful),
    );
    registry
}

fn stage(id: &str, config_hash: sos_core::Digest, deps: Vec<StageId>) -> Stage {
    Stage::new(
        StageId::new(id),
        StageDescriptor::new(PLUGIN, SemVer::new(1, 0, 0), plugin_hash()),
        vec![],
        config_hash,
        0,
        deps,
    )
}

fn env() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"env", b"test")
}

#[test]
fn a_plan_runs_a_real_ode_end_to_end_and_records_the_result() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    // y(1) = e^-1 for y0 = 1.
    let address = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 1.0 / 1024.0));

    let registry = registry_with_plugin(plugin_hash());
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = Plan::new(vec![stage("integrate", address, vec![])]).unwrap();
    let ledger = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap();

    assert_eq!(ledger.ran_count(), 1);
    let outputs = &ledger.steps[0].outputs;
    assert_eq!(outputs.len(), 1);

    let stored: Object<TrajectoryBody> = store
        .borrow()
        .get_object(outputs[0])
        .unwrap()
        .expect("the ledger's recorded id must resolve in the store");
    let (t_end, y_end) = stored.body.trajectory.last().unwrap().clone();
    assert!((t_end - 1.0).abs() < 1e-9);
    assert!(
        (y_end[0] - std::f64::consts::E.recip()).abs() < 1e-9,
        "y(1) should be 1/e, got {}",
        y_end[0]
    );
}

#[test]
fn a_warm_rerun_is_all_cache_hits_and_never_touches_the_solver() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    let address = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 0.01));

    let registry = registry_with_plugin(plugin_hash());
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = Plan::new(vec![stage("integrate", address, vec![])]).unwrap();
    let mut memo = MemoTable::new();

    let cold = run_plan(&plan, &env(), &mut memo, &mut dispatch).unwrap();
    assert_eq!(cold.ran_count(), 1);
    let stored_after_cold = store.borrow().object_ids().len();

    let warm = run_plan(&plan, &env(), &mut memo, &mut dispatch).unwrap();
    assert_eq!(warm.cache_hit_count(), 1);
    assert_eq!(warm.ran_count(), 0);
    assert_eq!(cold.steps[0].outputs, warm.steps[0].outputs);
    assert_eq!(
        store.borrow().object_ids().len(),
        stored_after_cold,
        "a cache hit must not re-run the solver, so it must store nothing new"
    );
}

#[test]
fn two_stages_run_in_dependency_order_and_produce_distinct_results() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    let coarse = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 0.1));
    let fine = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 0.001));

    let registry = registry_with_plugin(plugin_hash());
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = Plan::new(vec![
        stage("coarse", coarse, vec![]),
        stage("fine", fine, vec![StageId::new("coarse")]),
    ])
    .unwrap();
    let ledger = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap();

    assert_eq!(ledger.ran_count(), 2);
    assert_eq!(ledger.steps[0].stage, StageId::new("coarse"));
    assert_eq!(ledger.steps[1].stage, StageId::new("fine"));
    assert_ne!(ledger.steps[0].outputs, ledger.steps[1].outputs);

    // The finer step size really is more accurate — the two stages ran
    // different configurations, not the same one twice.
    let exact = std::f64::consts::E.recip();
    let err = |i: usize| {
        let o: Object<TrajectoryBody> = store
            .borrow()
            .get_object(ledger.steps[i].outputs[0])
            .unwrap()
            .unwrap();
        (o.body.trajectory.last().unwrap().1[0] - exact).abs()
    };
    assert!(
        err(1) < err(0),
        "the finer integration must be more accurate"
    );
}

#[test]
fn an_ungranted_capability_refuses_the_stage_before_the_solver_runs() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    let address = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 0.01));

    let registry = registry_with_plugin(plugin_hash());
    // Grant::new() grants nothing — least privilege, refusing by default.
    let mut dispatch = Dispatch::new(&registry, Grant::new());
    dispatch.register(PLUGIN, Box::new(h));

    let plan = Plan::new(vec![stage("integrate", address, vec![])]).unwrap();
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(matches!(err, WorkflowError::Unbound { .. }), "{err:?}");
    assert!(
        store.borrow().object_ids().is_empty(),
        "a refused stage must never reach the solver, let alone store a result"
    );
}

#[test]
fn a_drifted_plugin_fails_the_run_rather_than_computing_something_else() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    let address = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 0.01));

    // The registry now holds a *different* build under the same name and
    // version than the stage pinned.
    let registry = registry_with_plugin(HashAlgo::Sha256.hash(b"plugin", b"ode-rk4@1.0.0-rebuilt"));
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = Plan::new(vec![stage("integrate", address, vec![])]).unwrap();
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(matches!(err, WorkflowError::Unbound { .. }), "{err:?}");
    assert!(store.borrow().object_ids().is_empty());
}

#[test]
fn a_stage_whose_config_was_never_offered_fails_the_run() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let h = handler(&store); // nothing offered
    let orphan = sos_scirust::stage::config_address(&OdeConfig::new(0.0, 1.0, vec![1.0], 0.01));

    let registry = registry_with_plugin(plugin_hash());
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = Plan::new(vec![stage("integrate", orphan, vec![])]).unwrap();
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(matches!(err, WorkflowError::StageFailed { .. }), "{err:?}");
}

#[test]
fn a_plugin_with_no_registered_handler_is_an_error_not_an_empty_result() {
    let registry = registry_with_plugin(plugin_hash());
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    // Deliberately no `dispatch.register(...)`.

    let address = HashAlgo::Sha256.hash(b"cfg", b"whatever");
    let plan = Plan::new(vec![stage("integrate", address, vec![])]).unwrap();
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(matches!(err, WorkflowError::NoHandler { .. }), "{err:?}");
}

#[test]
fn the_recorded_result_is_reproducible_across_independent_runs() {
    // Two runs, two fresh stores and memos, same plan: an `L3` backend must
    // produce the same content address both times. This is the property the
    // whole ledger rests on — an id that varied per run would make the
    // record meaningless.
    let ids: Vec<_> = (0..2)
        .map(|_| {
            let store = Rc::new(RefCell::new(MemoryStore::new()));
            let mut h = handler(&store);
            let address = h.offer(OdeConfig::new(0.0, 1.0, vec![1.0], 0.01));
            let registry = registry_with_plugin(plugin_hash());
            let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
            dispatch.register(PLUGIN, Box::new(h));
            let plan = Plan::new(vec![stage("integrate", address, vec![])]).unwrap();
            run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch)
                .unwrap()
                .steps[0]
                .outputs
                .clone()
        })
        .collect();
    assert_eq!(ids[0], ids[1]);
}
