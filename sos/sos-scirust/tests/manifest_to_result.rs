//! The full vertical, from authorable text to a recorded scientific result:
//! a TOML study manifest → [`resolve_manifest`] → [`Plan`] → [`run_plan`] over
//! a [`Dispatch`] → [`OdeStageHandler`] → `scirust-solvers`' RK4 → an
//! [`Observation`](sos_simulation::Observation) sealed as a content-addressed
//! object.
//!
//! `sos-workflow`'s own tests cover manifest resolution against a `Plan`, but
//! that crate cannot depend on a backend (Invariant VIII), so nothing there
//! can show a manifest actually *computing* anything. This file can: it is the
//! first place a study written as text produces a real number.
//!
//! Scope is still one path, and these tests are written not to overstate it —
//! the only wired plugin is the RK4 ODE backend, and the manifest names it by
//! the same content hash the registry holds.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{HashAlgo, Object, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry, Role};
use sos_scirust::ode::{OdeConfig, Rk4OdeSimulator};
use sos_scirust::stage::{OdeStageHandler, TrajectoryBody, config_address, solvers_backend};
use sos_simulation::SimDescriptor;
use sos_store::{MemoryStore, ObjectStore, TypedStore};
use sos_workflow::{Dispatch, MemoTable, WorkflowError, resolve_manifest, run_plan};

const PLUGIN: &str = "ode-rk4";

/// Exponential decay `dy/dt = -y`, exact solution `y(t) = y0 · e^(-t)`.
fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}

type Rhs = fn(f64, &[f64], &mut [f64]);

fn plugin_hash() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"plugin", b"ode-rk4@1.0.0")
}

fn handler(store: &Rc<RefCell<MemoryStore>>) -> OdeStageHandler<Rhs, MemoryStore> {
    OdeStageHandler::new(
        Rk4OdeSimulator::new(
            SimDescriptor::new("scirust-solvers/ode-rk4", SemVer::new(1, 0, 0)),
            decay as Rhs,
        ),
        Rc::clone(store),
        solvers_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"backend", b"scirust-solvers"),
        ),
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

/// The two configurations this study integrates: `y(1)` for `y0 = 1`, at a
/// coarse and a fine step.
fn coarse() -> OdeConfig {
    OdeConfig::new(0.0, 1.0, vec![1.0], 0.1)
}
fn fine() -> OdeConfig {
    OdeConfig::new(0.0, 1.0, vec![1.0], 1.0 / 1024.0)
}

/// A study whose stage `pin`/`config` fields are the *real* content addresses
/// — written out exactly as an author would have to write them, because that
/// is the point of pinning.
fn study_text() -> String {
    format!(
        r#"
[study]
name = "decay-step-refinement"
seed = 42

[[stage]]
id      = "coarse"
plugin  = "{PLUGIN}"
version = "1.0.0"
pin     = "{pin}"
config  = "{c_coarse}"

[[stage]]
id      = "fine"
plugin  = "{PLUGIN}"
version = "1.0.0"
pin     = "{pin}"
config  = "{c_fine}"
needs   = ["coarse"]
"#,
        pin = plugin_hash().to_hex(),
        c_coarse = config_address(&coarse()).to_hex(),
        c_fine = config_address(&fine()).to_hex(),
    )
}

fn env() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"env", b"test")
}

#[test]
fn a_study_written_as_text_produces_a_real_recorded_result() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    h.offer(coarse());
    h.offer(fine());

    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = resolve_manifest(&study_text()).expect("the study must resolve");
    let ledger = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap();

    assert_eq!(ledger.ran_count(), 2);
    assert_eq!(ledger.steps[0].stage.0, "coarse");
    assert_eq!(ledger.steps[1].stage.0, "fine");

    // Both stages really integrated, and the finer step is closer to 1/e —
    // the study's whole point, recovered from the object store by the ids the
    // ledger recorded.
    let exact = std::f64::consts::E.recip();
    let error_at = |step: usize| {
        let stored: Object<TrajectoryBody> = store
            .borrow()
            .get_object(ledger.steps[step].outputs[0])
            .unwrap()
            .expect("every recorded output id must resolve");
        (stored.body.trajectory.last().unwrap().1[0] - exact).abs()
    };
    assert!(error_at(0) < 1e-4, "even the coarse step must integrate");
    assert!(
        error_at(1) < error_at(0),
        "refining the step must reduce the error"
    );
}

#[test]
fn the_studys_seed_reaches_the_recorded_observation() {
    // `[study] seed = 42` is not decoration: it has to survive resolution,
    // scheduling, dispatch and the handler, and end up stamped on the object.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    h.offer(coarse());
    h.offer(fine());

    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = resolve_manifest(&study_text()).unwrap();
    let ledger = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap();

    let stored: Object<TrajectoryBody> = store
        .borrow()
        .get_object(ledger.steps[0].outputs[0])
        .unwrap()
        .unwrap();
    assert_eq!(stored.body.seed, 42);
    assert_eq!(stored.repro.seed, 42);
}

#[test]
fn rerunning_an_unchanged_study_is_free() {
    // The build-system-for-knowledge claim, end to end from the text: the
    // same study file against a warm memo runs nothing.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    h.offer(coarse());
    h.offer(fine());

    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = resolve_manifest(&study_text()).unwrap();
    let mut memo = MemoTable::new();

    let cold = run_plan(&plan, &env(), &mut memo, &mut dispatch).unwrap();
    assert_eq!(cold.ran_count(), 2);
    let objects_after_cold = store.borrow().object_ids().len();

    let warm = run_plan(&plan, &env(), &mut memo, &mut dispatch).unwrap();
    assert_eq!(warm.cache_hit_count(), 2);
    assert_eq!(warm.ran_count(), 0);
    assert_eq!(cold.steps[1].outputs, warm.steps[1].outputs);
    assert_eq!(store.borrow().object_ids().len(), objects_after_cold);
}

#[test]
fn editing_only_the_studys_seed_invalidates_the_cache() {
    // The other half of memoization: a study that changed must *not* reuse
    // the old result. Only the seed differs here — no configuration, no
    // plugin, no dependency — and that alone must force a re-run.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    h.offer(coarse());
    h.offer(fine());

    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let mut memo = MemoTable::new();
    let first = run_plan(
        &resolve_manifest(&study_text()).unwrap(),
        &env(),
        &mut memo,
        &mut dispatch,
    )
    .unwrap();
    assert_eq!(first.ran_count(), 2);

    let reseeded = study_text().replace("seed = 42", "seed = 43");
    let second = run_plan(
        &resolve_manifest(&reseeded).unwrap(),
        &env(),
        &mut memo,
        &mut dispatch,
    )
    .unwrap();
    assert_eq!(
        second.ran_count(),
        2,
        "a changed seed must re-run the study"
    );
    assert_ne!(first.steps[0].outputs, second.steps[0].outputs);
}

#[test]
fn a_study_pinning_a_plugin_build_the_registry_no_longer_has_is_refused() {
    // The reason a manifest writes content hashes out longhand. The study is
    // valid TOML and resolves to a valid plan; it is the *bind* that refuses,
    // because the registry now holds a different build under that name and
    // version.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    h.offer(coarse());
    h.offer(fine());

    let mut drifted = Registry::new();
    drifted.register(
        PluginDescriptor::new(
            PLUGIN,
            SemVer::new(1, 0, 0),
            HashAlgo::Sha256.hash(b"plugin", b"ode-rk4@1.0.0-rebuilt"),
            Role::Simulation,
        )
        .needs(Capability::Effectful),
    );
    let mut dispatch = Dispatch::new(&drifted, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = resolve_manifest(&study_text()).expect("resolution still succeeds");
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(matches!(err, WorkflowError::Unbound { .. }), "{err:?}");
    assert!(
        store.borrow().object_ids().is_empty(),
        "a refused study must produce no results at all"
    );
}

#[test]
fn a_study_naming_a_config_nobody_offered_fails_at_the_stage_that_names_it() {
    // Resolution cannot know which configurations a handler holds — it is
    // pure text. The failure therefore lands at run time, and must name the
    // stage rather than failing anonymously.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut h = handler(&store);
    h.offer(coarse()); // `fine` is deliberately never offered

    let registry = registry();
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(h));

    let plan = resolve_manifest(&study_text()).unwrap();
    let err = run_plan(&plan, &env(), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    match err
    {
        WorkflowError::StageFailed { stage, reason } =>
        {
            assert_eq!(stage.0, "fine");
            assert!(reason.contains("no ODE configuration is on offer"));
        },
        other => panic!("expected StageFailed, got {other:?}"),
    }
    // The stage that *could* run still did, and its result is kept — a failed
    // study does not roll back the work that succeeded, which is what makes a
    // corrected re-run resumable rather than a restart.
    assert_eq!(store.borrow().object_ids().len(), 1);
}
