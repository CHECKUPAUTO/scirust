//! End-to-end: a real [`Plan`] scheduled by [`run_plan`] and executed through
//! [`Dispatch`], exercised through the public API only. This is the seam that
//! was missing — the scheduler could always memoize and order stages, but
//! nothing turned a stage's plugin *name* into running code.

use sos_core::{Digest, HashAlgo, ObjectId, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry, RegistryError, Role};
use sos_workflow::{
    Dispatch, MemoTable, Plan, Stage, StageDescriptor, StageExecutor, StageId, StepOutcome,
    WorkflowError, run_plan,
};

fn hash(tag: &[u8]) -> Digest {
    HashAlgo::default().hash(b"test", tag)
}

/// A handler that records every stage it ran, in order.
struct Recording {
    ran: Vec<String>,
}

impl StageExecutor for Recording {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        self.ran.push(stage.id.0.clone());
        Ok(vec![ObjectId::compute(
            HashAlgo::default(),
            b"out",
            stage.id.0.as_bytes(),
        )])
    }
}

/// A two-plugin registry: a pure `analyze` and an effectful `measure`.
fn registry() -> (Registry, Digest, Digest) {
    let (analyze, measure) = (hash(b"analyze-v1"), hash(b"measure-v1"));
    let mut r = Registry::new();
    r.register(PluginDescriptor::new(
        "analyze",
        SemVer::new(1, 0, 0),
        analyze,
        Role::Reasoning,
    ));
    r.register(
        PluginDescriptor::new("measure", SemVer::new(1, 0, 0), measure, Role::Simulation)
            .needs(Capability::Effectful),
    );
    (r, analyze, measure)
}

fn stage(id: &str, plugin: &str, content: Digest, deps: &[&str]) -> Stage {
    Stage::new(
        StageId::new(id),
        StageDescriptor::new(plugin, SemVer::new(1, 0, 0), content),
        vec![],
        hash(b"cfg"),
        0,
        deps.iter().map(|d| StageId::new(*d)).collect(),
    )
}

#[test]
fn a_plan_runs_end_to_end_through_the_registry() {
    let (reg, analyze, measure) = registry();
    // measure -> analyze: the effectful stage feeds the pure one.
    let plan = Plan::new(vec![
        stage("measure", "measure", measure, &[]),
        stage("analyze", "analyze", analyze, &["measure"]),
    ])
    .unwrap();

    let mut dispatch = Dispatch::new(&reg, Grant::new().allow(Capability::Effectful));
    dispatch.register("measure", Box::new(Recording { ran: vec![] }));
    dispatch.register("analyze", Box::new(Recording { ran: vec![] }));

    let env = hash(b"env");
    let mut memo = MemoTable::new();
    let ledger = run_plan(&plan, &env, &mut memo, &mut dispatch).unwrap();

    assert_eq!(ledger.ran_count(), 2);
    // Dependency order is preserved through dispatch.
    assert_eq!(
        ledger.steps.iter().map(|s| &s.stage).collect::<Vec<_>>(),
        vec![&StageId::new("measure"), &StageId::new("analyze")]
    );
}

#[test]
fn a_warm_rerun_never_binds_at_all() {
    // Cache hits do not call the executor, so a stage that would now fail to
    // bind still replays fine — memoization sits *above* dispatch.
    let (reg, analyze, _) = registry();
    let plan = Plan::new(vec![stage("analyze", "analyze", analyze, &[])]).unwrap();
    let env = hash(b"env");
    let mut memo = MemoTable::new();

    let mut dispatch = Dispatch::new(&reg, Grant::new());
    dispatch.register("analyze", Box::new(Recording { ran: vec![] }));
    let first = run_plan(&plan, &env, &mut memo, &mut dispatch).unwrap();
    assert_eq!(first.ran_count(), 1);

    // A dispatcher with *no* handlers at all: binding would fail outright.
    let mut broken = Dispatch::new(&reg, Grant::new());
    let second = run_plan(&plan, &env, &mut memo, &mut broken).unwrap();
    assert_eq!(second.cache_hit_count(), 1);
    assert_eq!(first.steps[0].outputs, second.steps[0].outputs);
}

#[test]
fn an_ungranted_effect_stops_the_run() {
    let (reg, _, measure) = registry();
    let plan = Plan::new(vec![stage("measure", "measure", measure, &[])]).unwrap();

    // The study grants nothing; `measure` needs the effect capability.
    let mut dispatch = Dispatch::new(&reg, Grant::new());
    dispatch.register("measure", Box::new(Recording { ran: vec![] }));

    let err = run_plan(&plan, &hash(b"env"), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    assert!(matches!(
        err,
        WorkflowError::Unbound {
            source: RegistryError::Denied { .. },
            ..
        }
    ));
}

#[test]
fn a_drifted_plugin_fails_the_run_rather_than_computing_something_else() {
    // The plan was written against one build of `analyze`; the registry now
    // holds a different one under the same name and version.
    let mut reg = Registry::new();
    reg.register(PluginDescriptor::new(
        "analyze",
        SemVer::new(1, 0, 0),
        hash(b"analyze-rebuilt"),
        Role::Reasoning,
    ));
    let plan = Plan::new(vec![stage(
        "analyze",
        "analyze",
        hash(b"analyze-as-written"),
        &[],
    )])
    .unwrap();

    let mut dispatch = Dispatch::new(&reg, Grant::new());
    dispatch.register("analyze", Box::new(Recording { ran: vec![] }));

    let err = run_plan(&plan, &hash(b"env"), &mut MemoTable::new(), &mut dispatch).unwrap_err();
    match err
    {
        WorkflowError::Unbound {
            stage,
            source: RegistryError::Drift { name, .. },
        } =>
        {
            assert_eq!(stage, StageId::new("analyze"));
            assert_eq!(name, "analyze");
        },
        other => panic!("expected a drift refusal, got {other:?}"),
    }
}

#[test]
fn a_failure_partway_through_does_not_corrupt_the_memo() {
    // Two independent stages; the second has no handler. The first still
    // memoizes, so a corrected rerun resumes rather than redoing it.
    let (reg, analyze, measure) = registry();
    let plan = Plan::new(vec![
        stage("analyze", "analyze", analyze, &[]),
        stage("measure", "measure", measure, &[]),
    ])
    .unwrap();

    let env = hash(b"env");
    let mut memo = MemoTable::new();
    let mut partial = Dispatch::new(&reg, Grant::new().allow(Capability::Effectful));
    partial.register("analyze", Box::new(Recording { ran: vec![] }));
    // `measure` deliberately left unregistered.

    let err = run_plan(&plan, &env, &mut memo, &mut partial).unwrap_err();
    assert!(matches!(err, WorkflowError::NoHandler { .. }));
    // `analyze` sorts first and completed before the failure, so it is memoized.
    assert_eq!(memo.len(), 1);

    // Register the missing handler and rerun: `analyze` replays, `measure` runs.
    let mut complete = Dispatch::new(&reg, Grant::new().allow(Capability::Effectful));
    complete.register("analyze", Box::new(Recording { ran: vec![] }));
    complete.register("measure", Box::new(Recording { ran: vec![] }));
    let ledger = run_plan(&plan, &env, &mut memo, &mut complete).unwrap();

    assert_eq!(ledger.cache_hit_count(), 1);
    assert_eq!(ledger.ran_count(), 1);
    let measured = ledger
        .steps
        .iter()
        .find(|s| s.stage == StageId::new("measure"))
        .unwrap();
    assert_eq!(measured.outcome, StepOutcome::Ran);
}
