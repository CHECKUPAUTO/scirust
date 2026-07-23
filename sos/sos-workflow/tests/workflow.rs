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
}

impl Canonical for Product {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.stage);
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
}

impl StageExecutor for StoringExecutor<'_> {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        let obj = Object::builder(Product {
            stage: stage.id.0.clone(),
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
            store: &mut store,
            ran: vec![],
        };
        run_plan(&plan, &env, &mut memo, &mut exec).unwrap()
    };
    // Re-run against the warm memo: nothing executes.
    let mut exec = StoringExecutor {
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
            store: &mut store,
            ran: vec![],
        };
        run_plan(&plan, &digest(b"linux"), &mut memo, &mut exec).unwrap();
    }
    // A different environment digest ⇒ different keys ⇒ everything re-runs.
    let mut exec = StoringExecutor {
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
