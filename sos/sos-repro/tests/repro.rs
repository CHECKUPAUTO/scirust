//! End-to-end reproducibility: an `EnvLock` pins the workflow cache, so a lock
//! that binds the original reproduces every stage from cache while a drifted lock
//! re-executes — and the contract verifier localizes a divergence.

use sos_core::{BackendVersion, DeterminismLevel, EnvRecord, HashAlgo, ObjectId, SemVer};
use sos_repro::{EnvLock, NodeClaim, NodeVerdict, Reproduced, rerun, verify_reproduction};
use sos_workflow::{
    MemoTable, Plan, Stage, StageDescriptor, StageExecutor, StageId, WorkflowError,
};

fn digest(tag: &[u8]) -> sos_core::Digest {
    HashAlgo::default().hash(b"repro-test", tag)
}

/// A deterministic executor that produces one output id per stage and counts how
/// many stages actually ran.
struct Counting {
    ran: usize,
}

impl StageExecutor for Counting {
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
        self.ran += 1;
        Ok(vec![ObjectId::compute(
            HashAlgo::default(),
            b"out",
            stage.id.0.as_bytes(),
        )])
    }
}

fn plan() -> Plan {
    let mk = |id: &str, deps: &[&str]| {
        Stage::new(
            StageId::new(id),
            StageDescriptor::new(id, SemVer::new(1, 0, 0), digest(id.as_bytes())),
            vec![],
            digest(b"cfg"),
            0,
            deps.iter().map(|d| StageId::new(*d)).collect(),
        )
    };
    Plan::new(vec![mk("a", &[]), mk("b", &["a"]), mk("c", &["a", "b"])]).unwrap()
}

fn env(hardware: &str) -> EnvRecord {
    EnvRecord::new(
        "1.89.0-stable",
        vec![BackendVersion::new(
            "scirust-solvers",
            SemVer::new(0, 1, 0),
            digest(b"solvers"),
        )],
        hardware,
        "linux",
    )
}

#[test]
fn a_binding_lock_reproduces_from_cache_a_drifted_lock_reexecutes() {
    let plan = plan();
    let pinned = EnvLock::pin(env("x86_64/avx2"));
    let drifted = EnvLock::pin(env("aarch64/neon"));
    let mut memo = MemoTable::new();

    // Cold run under the pinned lock: all three stages execute.
    let mut exec = Counting { ran: 0 };
    let cold = rerun(&plan, &pinned, &mut memo, &mut exec).unwrap();
    assert_eq!(cold.ran_count(), 3);
    assert_eq!(exec.ran, 3);

    // Re-run under a lock that BINDS the original: everything reproduces from
    // cache, nothing runs.
    let same = EnvLock::pin(env("x86_64/avx2"));
    assert!(pinned.binds(&same));
    let mut exec2 = Counting { ran: 0 };
    let warm = rerun(&plan, &same, &mut memo, &mut exec2).unwrap();
    assert_eq!(warm.cache_hit_count(), 3);
    assert_eq!(exec2.ran, 0);

    // Re-run under a DRIFTED lock: the environment digest changed, so every stage
    // re-executes — the drift is realized as recomputation.
    assert!(!pinned.binds(&drifted));
    assert_ne!(pinned.env_digest(), drifted.env_digest());
    let mut exec3 = Counting { ran: 0 };
    let redo = rerun(&plan, &drifted, &mut memo, &mut exec3).unwrap();
    assert_eq!(redo.ran_count(), 3);
    assert_eq!(exec3.ran, 3);
}

#[test]
fn the_env_digest_reproduces_across_rebuilt_locks() {
    // The reproducibility key is a pure function of the pinned environment.
    let a = EnvLock::pin(env("x86_64/avx2"));
    let b = EnvLock::pin(env("x86_64/avx2"));
    assert_eq!(a.env_digest(), b.env_digest());
    assert!(a.binds(&b) && b.binds(&a));
}

#[test]
fn the_contract_verifies_an_l3_subdag_and_localizes_divergence() {
    let oid = |t: &[u8]| ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", t);

    let claims = [
        NodeClaim::new(oid(b"n1"), DeterminismLevel::L3),
        NodeClaim::new(oid(b"n2"), DeterminismLevel::L3),
        NodeClaim::new(oid(b"n3"), DeterminismLevel::L2),
    ];

    // A faithful reproduction: L3 ids match, the L2 node is certified in-tolerance.
    let faithful = [
        Reproduced::Id(oid(b"n1")),
        Reproduced::Id(oid(b"n2")),
        Reproduced::Certified(true),
    ];
    let ok = verify_reproduction(&claims, &faithful).unwrap();
    assert!(ok.reproduced());
    assert!(ok.first_deviation().is_none());
    assert_eq!(ok.level, DeterminismLevel::L2); // weakest over the sub-DAG

    // A divergence at the second node is localized.
    let broken = [
        Reproduced::Id(oid(b"n1")),
        Reproduced::Id(oid(b"WRONG")),
        Reproduced::Certified(true),
    ];
    let bad = verify_reproduction(&claims, &broken).unwrap();
    assert!(!bad.reproduced());
    let dev = bad.first_deviation().unwrap();
    assert_eq!(dev.node, oid(b"n2"));
    assert_eq!(dev.verdict, NodeVerdict::Diverged);
}
