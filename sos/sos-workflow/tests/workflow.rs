//! End-to-end workflow scheduling: a memoized diamond DAG driven by a real
//! object-storing executor, proving deterministic scheduling, free re-runs,
//! selective recomputation, and a content-addressed RunLedger.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Digest, HashAlgo, Object, ObjectId, SemVer};
use sos_store::{MemoryStore, TypedStore};
use sos_workflow::{
    MemoTable, Plan, Stage, StageDescriptor, StageExecutor, StageId, StepOutcome, WorkflowError,
    run_plan,
};

fn digest(tag: &[u8]) -> Digest {
    HashAlgo::default().hash(b"wf-test", tag)
}

/// The object a stage produces.
#[derive(Clone, Serialize, Deserialize)]
struct Product {
    stage: String,
    /// The config the stage ran under, so a different configuration really
    /// produces a different object — without this the executor's output
    /// depends only on the stage id and "the upstream changed" is
    /// unobservable downstream.
    config: String,
}

impl Canonical for Product {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.stage);
        enc.str(&self.config);
    }
}

impl Body for Product {
    const KIND: &'static str = "Product";
    const SCHEMA_VERSION: u32 = 1;
}

/// A real executor: it seals a `Product` object per stage into a store and
/// returns its id, recording the order of stages it actually ran.
struct StoringExecutor<'s> {
    store: &'s mut MemoryStore,
    ran: Vec<StageId>,
    /// The inputs each stage was actually handed — the only way to observe
    /// that dataflow resolution happened at all.
    saw: Vec<(StageId, Vec<ObjectId>)>,
}

impl StageExecutor for StoringExecutor<'_> {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        let obj = Object::builder(Product {
            stage: stage.id.0.clone(),
            config: stage.config_hash.to_hex(),
        })
        .author(Author::engine("stage-runner"))
        .seal();
        let id = obj.id;
        self.store
            .put_object(&obj)
            .map_err(|e| WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: e.to_string(),
            })?;
        self.ran.push(stage.id.clone());
        self.saw.push((stage.id.clone(), stage.inputs.clone()));
        Ok(vec![id])
    }
}

fn stage(id: &str, config: &[u8], deps: &[&str]) -> Stage {
    Stage::new(
        StageId::new(id),
        StageDescriptor::new(id, SemVer::new(1, 0, 0), digest(id.as_bytes())),
        vec![],
        digest(config),
        0,
        deps.iter().map(|d| StageId::new(*d)).collect(),
    )
}

/// The diamond `a -> {b, c} -> d`.
fn diamond() -> Plan {
    Plan::new(vec![
        stage("d", b"cfg-d", &["b", "c"]),
        stage("b", b"cfg-b", &["a"]),
        stage("c", b"cfg-c", &["a"]),
        stage("a", b"cfg-a", &[]),
    ])
    .unwrap()
}

#[test]
fn cold_run_executes_every_stage_in_topological_order() {
    let plan = diamond();
    let env = digest(b"env");
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: vec![],
    };

    let ledger = run_plan(&plan, &env, &mut memo, &mut exec).unwrap();

    assert_eq!(ledger.ran_count(), 4);
    assert_eq!(ledger.cache_hit_count(), 0);
    // Deterministic schedule: a before b,c before d; ties by id (b before c).
    assert_eq!(exec.ran, ["a", "b", "c", "d"].map(StageId::new).to_vec());
    // Every step recorded exactly one produced object.
    assert!(ledger.steps.iter().all(|s| s.outputs.len() == 1));
}

#[test]
fn warm_rerun_is_all_cache_hits_and_identical() {
    let plan = diamond();
    let env = digest(b"env");
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();

    let first = {
        let mut exec = StoringExecutor {
            saw: Vec::new(),
            store: &mut store,
            ran: vec![],
        };
        run_plan(&plan, &env, &mut memo, &mut exec).unwrap()
    };
    // Re-run against the warm memo: nothing executes.
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: vec![],
    };
    let second = run_plan(&plan, &env, &mut memo, &mut exec).unwrap();

    assert!(exec.ran.is_empty()); // executor never called
    assert_eq!(second.cache_hit_count(), 4);
    assert!(first.steps.iter().all(|s| s.outcome == StepOutcome::Ran));
    assert!(
        second
            .steps
            .iter()
            .all(|s| s.outcome == StepOutcome::CacheHit)
    );
    // Same schedule, cache keys, and outputs — only the outcome differs.
    let first_io: Vec<_> = first
        .steps
        .iter()
        .map(|s| (&s.stage, s.cache_key, &s.outputs))
        .collect();
    let second_io: Vec<_> = second
        .steps
        .iter()
        .map(|s| (&s.stage, s.cache_key, &s.outputs))
        .collect();
    assert_eq!(first_io, second_io);
    assert_eq!(first.plan_digest, second.plan_digest);
}

#[test]
fn changing_one_stage_config_reruns_only_that_stage() {
    let env = digest(b"env");
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();

    // Warm the cache with the original diamond.
    {
        let mut exec = StoringExecutor {
            saw: Vec::new(),
            store: &mut store,
            ran: vec![],
        };
        run_plan(&diamond(), &env, &mut memo, &mut exec).unwrap();
    }

    // A new plan identical except stage `b`'s config changed.
    let changed = Plan::new(vec![
        stage("d", b"cfg-d", &["b", "c"]),
        stage("b", b"cfg-b-v2", &["a"]),
        stage("c", b"cfg-c", &["a"]),
        stage("a", b"cfg-a", &[]),
    ])
    .unwrap();

    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: vec![],
    };
    let ledger = run_plan(&changed, &env, &mut memo, &mut exec).unwrap();

    // Only `b` has a new cache key, so only `b` re-runs; a, c, d are cache hits.
    assert_eq!(exec.ran, vec![StageId::new("b")]);
    let outcome = |id: &str| {
        ledger
            .steps
            .iter()
            .find(|s| s.stage == StageId::new(id))
            .unwrap()
            .outcome
    };
    assert_eq!(outcome("b"), StepOutcome::Ran);
    assert_eq!(outcome("a"), StepOutcome::CacheHit);
    assert_eq!(outcome("c"), StepOutcome::CacheHit);
    assert_eq!(outcome("d"), StepOutcome::CacheHit);
}

#[test]
fn a_different_env_busts_the_whole_cache() {
    let plan = diamond();
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();

    {
        let mut exec = StoringExecutor {
            saw: Vec::new(),
            store: &mut store,
            ran: vec![],
        };
        run_plan(&plan, &digest(b"linux"), &mut memo, &mut exec).unwrap();
    }
    // A different environment digest ⇒ different keys ⇒ everything re-runs.
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: vec![],
    };
    let ledger = run_plan(&plan, &digest(b"macos"), &mut memo, &mut exec).unwrap();
    assert_eq!(ledger.ran_count(), 4);
    assert_eq!(exec.ran.len(), 4);
}

#[test]
fn the_ledger_seals_to_a_verifiable_object() {
    let plan = diamond();
    let env = digest(b"env");
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: vec![],
    };
    let ledger = run_plan(&plan, &env, &mut memo, &mut exec).unwrap();

    let obj = Object::builder(ledger)
        .author(Author::engine("sos-workflow"))
        .seal();
    assert!(obj.verify_id());
    assert_eq!(obj.kind.name, "RunLedger");
}

// ---- dataflow: `consumes`, and why it is not `needs` ----------------------

/// A stage that reads `consumed`'s outputs.
fn consumer(id: &str, config: &[u8], consumed: &[&str]) -> Stage {
    Stage::consuming(
        StageId::new(id),
        StageDescriptor::new(id, SemVer::new(1, 0, 0), digest(id.as_bytes())),
        vec![],
        digest(config),
        0,
        vec![],
        consumed.iter().map(|d| StageId::new(*d)).collect(),
    )
}

#[test]
fn a_consuming_stage_is_handed_its_upstream_outputs() {
    // The gap this closes: before `consumes`, no stage could read another's
    // output at all. `inputs` takes literal object ids, and an upstream
    // stage's ids do not exist until it has run, so a plan could sequence its
    // stages but never feed one into the next.
    let plan = Plan::new(vec![
        stage("source", b"cfg-source", &[]),
        consumer("sink", b"cfg-sink", &["source"]),
    ])
    .unwrap();
    let env = digest(b"env");
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: Vec::new(),
    };

    let ledger = run_plan(&plan, &env, &mut memo, &mut exec).unwrap();
    let source_out = ledger.steps[0].outputs.clone();
    assert_eq!(ledger.steps[0].stage, StageId::new("source"));

    let sink_inputs = exec
        .saw
        .iter()
        .find(|(id, _)| *id == StageId::new("sink"))
        .map(|(_, inputs)| inputs.clone())
        .expect("the sink ran");
    assert_eq!(
        sink_inputs, source_out,
        "the sink reads what the source made"
    );
}

#[test]
fn consuming_implies_ordering_without_repeating_it() {
    // An author writes `consumes` once; the dependency edge follows, so the
    // two can never be stated inconsistently.
    let sink = consumer("sink", b"cfg", &["source"]);
    assert_eq!(sink.deps, vec![StageId::new("source")]);
    assert_eq!(sink.consumes, vec![StageId::new("source")]);

    // And a consumed stage that does not exist is the same dangling-dependency
    // error a `needs` typo already produced.
    let orphan = Plan::new(vec![consumer("sink", b"cfg", &["nowhere"])]);
    assert!(matches!(
        orphan,
        Err(WorkflowError::MissingDependency { .. })
    ));
}

#[test]
fn needs_still_means_ordering_only() {
    // The reason this is a separate field rather than a new meaning for
    // `needs`: a study that genuinely only wants sequencing must not acquire
    // provenance parents it never read.
    let plan = Plan::new(vec![
        stage("first", b"cfg-first", &[]),
        stage("second", b"cfg-second", &["first"]),
    ])
    .unwrap();
    let env = digest(b"env");
    let mut store = MemoryStore::new();
    let mut memo = MemoTable::new();
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: Vec::new(),
    };

    run_plan(&plan, &env, &mut memo, &mut exec).unwrap();
    let second_inputs = exec
        .saw
        .iter()
        .find(|(id, _)| *id == StageId::new("second"))
        .map(|(_, i)| i.clone())
        .unwrap();
    assert!(
        second_inputs.is_empty(),
        "ordering alone must not invent inputs, got {second_inputs:?}"
    );
}

#[test]
fn a_downstream_stage_misses_the_cache_when_its_upstream_output_changes() {
    // The correctness property this feature lives or dies on. Resolution
    // happens *before* the cache key is computed, so a changed upstream
    // output changes the downstream key. Resolving afterwards would let the
    // sink cache-hit on a source that produced something different — the one
    // way this could quietly corrupt a result.
    let env = digest(b"env");
    let mut memo = MemoTable::new();
    let mut store = MemoryStore::new();

    let build = |cfg: &[u8]| {
        Plan::new(vec![
            stage("source", cfg, &[]),
            consumer("sink", b"cfg-sink", &["source"]),
        ])
        .unwrap()
    };

    let first = build(b"cfg-v1");
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: Vec::new(),
    };
    let a = run_plan(&first, &env, &mut memo, &mut exec).unwrap();
    assert_eq!(exec.ran.len(), 2);

    // Re-running the identical plan is free, sink included.
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: Vec::new(),
    };
    let cached = run_plan(&first, &env, &mut memo, &mut exec).unwrap();
    assert!(exec.ran.is_empty(), "an unchanged plan re-runs nothing");
    assert_eq!(cached.steps[1].outcome, StepOutcome::CacheHit);

    // Change only the source's config. The sink's own fields are untouched,
    // so if resolution came after the key it would still cache-hit.
    let second = build(b"cfg-v2");
    let mut exec = StoringExecutor {
        saw: Vec::new(),
        store: &mut store,
        ran: Vec::new(),
    };
    let b = run_plan(&second, &env, &mut memo, &mut exec).unwrap();
    assert!(
        exec.ran.contains(&StageId::new("sink")),
        "the sink must re-run when its upstream output changed, ran: {:?}",
        exec.ran
    );
    assert_ne!(
        a.steps[1].cache_key, b.steps[1].cache_key,
        "the downstream cache key must cover the resolved inputs"
    );
}

/// An executor whose output ignores the stage configuration, so a
/// reconfigured stage produces a byte-identical object.
struct FixedOutputExecutor<'s> {
    store: &'s mut MemoryStore,
    ran: Vec<StageId>,
}

impl StageExecutor for FixedOutputExecutor<'_> {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        let obj = Object::builder(Product {
            stage: stage.id.0.clone(),
            config: "fixed".to_owned(),
        })
        .author(Author::engine("stage-runner"))
        .seal();
        let id = obj.id;
        self.store
            .put_object(&obj)
            .map_err(|e| WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: e.to_string(),
            })?;
        self.ran.push(stage.id.clone());
        Ok(vec![id])
    }
}

#[test]
fn a_downstream_stage_still_cache_hits_when_its_upstream_output_did_not_change() {
    // Early cutoff, and the distinction this test exists to draw: the
    // downstream key covers the upstream's *outputs*, not its configuration.
    // A source reconfigured into producing the identical object gives the
    // sink nothing new to do, and re-running it would be waste rather than
    // caution. (This test exists because the one above was originally written
    // to change the source's *config* and expect a downstream miss — it did
    // not, correctly, and the difference is worth pinning.)
    let env = digest(b"env");
    let mut memo = MemoTable::new();
    let mut store = MemoryStore::new();

    let build = |cfg: &[u8]| {
        Plan::new(vec![
            stage("source", cfg, &[]),
            consumer("sink", b"cfg-sink", &["source"]),
        ])
        .unwrap()
    };

    let mut exec = FixedOutputExecutor {
        store: &mut store,
        ran: Vec::new(),
    };
    run_plan(&build(b"cfg-v1"), &env, &mut memo, &mut exec).unwrap();
    assert_eq!(exec.ran.len(), 2);

    let mut exec = FixedOutputExecutor {
        store: &mut store,
        ran: Vec::new(),
    };
    let second = run_plan(&build(b"cfg-v2"), &env, &mut memo, &mut exec).unwrap();
    assert_eq!(
        exec.ran,
        vec![StageId::new("source")],
        "only the source re-runs; the sink's inputs are unchanged"
    );
    assert_eq!(second.steps[1].outcome, StepOutcome::CacheHit);
}

#[test]
fn a_plan_from_before_consumes_existed_still_loads() {
    // The field is defaulted on deserialization, so no stored plan is
    // invalidated by its arrival — an absent `consumes` means "consumed
    // nothing", which is exactly what such a plan did.
    let json = serde_json::to_string(&diamond()).unwrap();
    let stripped = json.replace(r#","consumes":[]"#, "");
    assert!(!stripped.contains("consumes"), "the field really is gone");
    let reloaded: Plan = serde_json::from_str(&stripped).expect("an older plan must still load");
    assert_eq!(reloaded.stages().len(), 4);
    assert!(reloaded.stages().iter().all(|s| s.consumes.is_empty()));
}
