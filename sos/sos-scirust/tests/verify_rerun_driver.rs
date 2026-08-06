//! The reproduction loop closed end to end: run a study, re-run it, and let
//! `sos-repro`'s driver decide whether it reproduced — with this crate
//! supplying the `L2`/`L1` verdicts the driver refuses to invent.
//!
//! `sos-repro`'s own tests exercise the driver against hand-built ledgers and
//! a stub certifier, because that crate cannot depend on a backend
//! (Invariant VIII). This file supplies the real one: a [`Certifier`] backed
//! by [`sos_scirust::verdict`], over ledgers produced by really running a
//! plan through `Dispatch` and the RK4 backend.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use sos_core::{DeterminismLevel, HashAlgo, Object, ObjectId, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry, Role};
use sos_repro::{Certifier, NoCertifier, ReproError, verify_rerun};
use sos_scirust::ode::{OdeConfig, Rk4OdeSimulator};
use sos_scirust::stage::{OdeStageHandler, TrajectoryBody, solvers_backend};
use sos_simulation::SimDescriptor;
use sos_store::{MemoryStore, TypedStore};
use sos_workflow::{
    Dispatch, MemoTable, Plan, RunLedger, Stage, StageDescriptor, StageId, resolve_manifest,
    run_plan,
};

const PLUGIN: &str = "ode-rk4";

fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}

type Rhs = fn(f64, &[f64], &mut [f64]);

fn plugin_hash() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"plugin", b"ode-rk4@1.0.0")
}

fn config() -> OdeConfig {
    OdeConfig::new(0.0, 1.0, vec![1.0], 1.0 / 256.0)
}

fn env() -> sos_core::Digest {
    HashAlgo::Sha256.hash(b"env", b"test")
}

/// Run the one-stage plan, returning its ledger. The store is shared, so
/// repeated runs accumulate into the same object graph — which is what the
/// driver later reads declared levels from.
fn run_once(store: &Rc<RefCell<MemoryStore>>, memo: &mut MemoTable) -> (RunLedger, Plan) {
    let mut handler = OdeStageHandler::new(
        Rk4OdeSimulator::new(
            SimDescriptor::new("scirust-solvers/ode-rk4", SemVer::new(1, 0, 0)),
            decay as Rhs,
        ),
        Rc::clone(store),
        solvers_backend(
            SemVer::new(0, 1, 0),
            HashAlgo::Sha256.hash(b"backend", b"scirust-solvers"),
        ),
    );
    let address = handler.offer(config());

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
    let mut dispatch = Dispatch::new(&registry, Grant::new().allow(Capability::Effectful));
    dispatch.register(PLUGIN, Box::new(handler));

    let text = format!(
        r#"
[study]
name = "decay"
seed = 7

[[stage]]
id      = "integrate"
plugin  = "{PLUGIN}"
version = "1.0.0"
pin     = "{pin}"
config  = "{cfg}"
"#,
        pin = plugin_hash().to_hex(),
        cfg = address.to_hex(),
    );
    let plan = resolve_manifest(&text).expect("the study must resolve");
    let ledger = run_plan(&plan, &env(), memo, &mut dispatch).expect("the run must succeed");
    (ledger, plan)
}

/// A [`Certifier`] backed by this crate's `verdict` module, comparing the two
/// stored trajectories at their endpoints.
///
/// The RK4 backend is `L3`, so in practice its nodes are decided by id and
/// never reach here; this exists to show the seam works with a real numeric
/// backend behind it, and is exercised below by relabelling a node `L2`.
struct TrajectoryCertifier {
    store: Rc<RefCell<MemoryStore>>,
    /// Endpoint values seen, for the assertions below.
    compared: BTreeMap<StageId, (f64, f64)>,
}

impl Certifier for TrajectoryCertifier {
    fn certify(
        &mut self,
        stage: &StageId,
        _level: DeterminismLevel,
        original: ObjectId,
        rerun: ObjectId,
    ) -> Result<bool, String> {
        let endpoint = |id: ObjectId| -> Result<f64, String> {
            let stored: Object<TrajectoryBody> = self
                .store
                .borrow()
                .get_object(id)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("{id} is not in the store"))?;
            stored
                .body
                .trajectory
                .last()
                .map(|(_, y)| y[0])
                .ok_or_else(|| "empty trajectory".to_string())
        };
        let (a, b) = (endpoint(original)?, endpoint(rerun)?);
        self.compared.insert(stage.clone(), (a, b));
        // A real numeric decision: agree within a declared absolute band.
        Ok((a - b).abs() <= 1e-9)
    }
}

#[test]
fn an_unchanged_study_reproduces_through_the_whole_loop() {
    // Study → plan → run → re-run → driver → VerifyReport. Every layer real.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let mut memo = MemoTable::new();
    let (original, _) = run_once(&store, &mut memo);
    let (rerun, _) = run_once(&store, &mut memo);

    // The re-run is all cache hits, so it is genuinely the same objects.
    assert_eq!(rerun.cache_hit_count(), 1);

    let report = verify_rerun(&original, &rerun, &*store.borrow(), &mut NoCertifier).unwrap();
    assert!(report.reproduced());
    assert_eq!(report.level, DeterminismLevel::L3, "RK4 is L3");
    assert_eq!(report.nodes.len(), 1);
}

#[test]
fn a_study_run_from_a_cold_store_still_reproduces() {
    // Not a cache tautology: the second run executes the solver again from
    // an empty memo, and the L3 backend must land on the same object id.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let (original, _) = run_once(&store, &mut MemoTable::new());
    let (rerun, _) = run_once(&store, &mut MemoTable::new());
    assert_eq!(rerun.ran_count(), 1, "the solver really ran again");

    let report = verify_rerun(&original, &rerun, &*store.borrow(), &mut NoCertifier).unwrap();
    assert!(report.reproduced());
}

#[test]
fn a_changed_result_is_caught_and_localized() {
    // Substitute a genuinely different trajectory for the re-run's output —
    // the same shape a real divergence would take.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let (original, _) = run_once(&store, &mut MemoTable::new());

    let mut other_store = MemoryStore::new();
    let other_id = {
        let obj = Object::builder(TrajectoryBody {
            trajectory: vec![(1.0, vec![0.5])],
            level: DeterminismLevel::L3,
            seed: 7,
        })
        .level(DeterminismLevel::L3)
        .seal();
        other_store.put_object(&obj).unwrap();
        store.borrow_mut().put_object(&obj).unwrap()
    };

    let mut diverged = original.clone();
    diverged.steps[0].outputs = vec![other_id];

    let report = verify_rerun(&original, &diverged, &*store.borrow(), &mut NoCertifier).unwrap();
    assert!(!report.reproduced());
    let deviation = report.first_deviation().unwrap();
    assert_eq!(
        deviation.node, original.steps[0].outputs[0],
        "the report names the original node, not the substitute"
    );
}

#[test]
fn an_l2_node_is_decided_by_this_crates_numeric_backend() {
    // The seam `sos-repro` declined to fill. The RK4 handler is L3, so to
    // exercise the L2 path we store the same trajectory *declared* L2 — the
    // driver must then route it to the certifier rather than to id equality.
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let (original, _) = run_once(&store, &mut MemoTable::new());
    let produced: Object<TrajectoryBody> = store
        .borrow()
        .get_object(original.steps[0].outputs[0])
        .unwrap()
        .unwrap();

    // Two objects with the same trajectory but declared L2 — distinct ids,
    // so id equality would call this diverged; the numeric verdict must not.
    let as_l2 = |tag: u64| {
        let obj = Object::builder(produced.body.clone())
            .level(DeterminismLevel::L2)
            .repro(produced.repro.clone())
            .version(SemVer::new(1, 0, u32::try_from(tag).unwrap()))
            .seal();
        store.borrow_mut().put_object(&obj).unwrap()
    };
    let (a, b) = (as_l2(0), as_l2(1));
    assert_ne!(a, b, "distinct objects, identical science");

    let mut original_l2 = original.clone();
    original_l2.steps[0].outputs = vec![a];
    let mut rerun_l2 = original.clone();
    rerun_l2.steps[0].outputs = vec![b];

    let mut certifier = TrajectoryCertifier {
        store: Rc::clone(&store),
        compared: BTreeMap::new(),
    };
    let report = verify_rerun(&original_l2, &rerun_l2, &*store.borrow(), &mut certifier).unwrap();

    assert!(
        report.reproduced(),
        "identical trajectories must certify even with different ids"
    );
    assert_eq!(report.level, DeterminismLevel::L2);
    let (x, y) = certifier.compared[&StageId::new("integrate")];
    assert_eq!(x, y, "the backend compared the real endpoint values");
    // And it is a real e^-1, not an empty result the certifier waved through.
    assert!((x - std::f64::consts::E.recip()).abs() < 1e-6);
}

#[test]
fn an_l2_node_the_backend_rejects_diverges() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let (original, _) = run_once(&store, &mut MemoTable::new());
    let produced: Object<TrajectoryBody> = store
        .borrow()
        .get_object(original.steps[0].outputs[0])
        .unwrap()
        .unwrap();

    let store_l2 = |traj, tag: u32| {
        let obj = Object::builder(TrajectoryBody {
            trajectory: traj,
            level: DeterminismLevel::L2,
            seed: 7,
        })
        .level(DeterminismLevel::L2)
        .version(SemVer::new(1, 0, tag))
        .seal();
        store.borrow_mut().put_object(&obj).unwrap()
    };
    let good = store_l2(produced.body.trajectory.clone(), 0);
    // A trajectory ending far from e^-1 — a genuine numeric divergence.
    let bad = store_l2(vec![(1.0, vec![0.9])], 1);

    let mut original_l2 = original.clone();
    original_l2.steps[0].outputs = vec![good];
    let mut rerun_l2 = original.clone();
    rerun_l2.steps[0].outputs = vec![bad];

    let mut certifier = TrajectoryCertifier {
        store: Rc::clone(&store),
        compared: BTreeMap::new(),
    };
    let report = verify_rerun(&original_l2, &rerun_l2, &*store.borrow(), &mut certifier).unwrap();
    assert!(!report.reproduced());
}

#[test]
fn verifying_a_different_study_is_refused_rather_than_reported_on() {
    let store = Rc::new(RefCell::new(MemoryStore::new()));
    let (original, plan) = run_once(&store, &mut MemoTable::new());

    // A different plan: same stage, different seed, so a different digest.
    let other = Plan::new(vec![Stage::new(
        StageId::new("integrate"),
        StageDescriptor::new(PLUGIN, SemVer::new(1, 0, 0), plugin_hash()),
        vec![],
        HashAlgo::Sha256.hash(b"cfg", b"other"),
        99,
        vec![],
    )])
    .unwrap();
    assert_ne!(plan.digest(), other.digest());

    let mut foreign = original.clone();
    foreign.plan_digest = other.digest();

    assert!(matches!(
        verify_rerun(&original, &foreign, &*store.borrow(), &mut NoCertifier),
        Err(ReproError::PlanMismatch { .. })
    ));
}
