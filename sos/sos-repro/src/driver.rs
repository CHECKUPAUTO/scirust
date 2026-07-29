//! [`verify_rerun`] — the object-graph re-execution driver.
//!
//! [`rerun`](crate::rerun) re-realizes a plan and [`verify_reproduction`]
//! decides a sub-DAG's verdicts, but nothing joined them: a caller had to
//! hand-build the parallel `claims` and `reproduced` arrays, which meant
//! knowing every original object's declared level, pairing it with the right
//! re-run output, and routing `L2`/`L1` nodes to a numeric backend. That
//! assembly is this module, and it was the last piece the crate's own docs
//! named as missing.
//!
//! ```text
//! original RunLedger ─┐
//!                     ├─► verify_rerun ─► VerifyReport
//! re-run   RunLedger ─┤        │
//! object store ───────┘        └─► Certifier (L2/L1 only)
//! ```
//!
//! ## Why the two ledgers, rather than a root object id
//!
//! "Walk the object's sub-DAG" is the intuitive framing, but the ledger is
//! the better source: it already *is* the record of which stage produced
//! which objects, in a deterministic order, and RFC-0002 keeps it precisely
//! because control flow is data too. Pairing by stage is therefore exact,
//! where pairing by provenance walk would have to guess which of a re-run's
//! objects corresponds to which of the original's.
//!
//! ## What it refuses to do
//!
//! * **Verify across different plans.** If the two ledgers disagree on
//!   [`plan_digest`](RunLedger::plan_digest), this errors rather than
//!   reporting on whatever the stages happen to have in common. Declaring
//!   that a run "reproduced" when it executed a different plan would be a
//!   false claim about the science, not a lenient one.
//! * **Invent `L2`/`L1` verdicts.** Those nodes go to a [`Certifier`] — in
//!   practice `sos-scirust`'s `verdict` module. This crate still never
//!   fabricates a numeric verdict; it just knows who to ask.
//! * **Guess a level.** An object the store does not hold, or whose bytes do
//!   not parse, is not silently assumed `L3`; it is a
//!   [`ReproError::UnreadableClaim`].
//!
//! ## Arity mismatches are an error, not a divergence
//!
//! If a stage produced two objects originally and one on re-run, this errors
//! with [`ReproError::OutputArityMismatch`] rather than reporting the missing
//! node as diverged. How many objects a stage emits is a property of the
//! *plugin*, not of the environment — and both ledgers already agree on
//! `plan_digest`, which pins each stage's plugin to a content hash that
//! [`Dispatch`](sos_workflow::Dispatch) refuses to bind if it drifted. So a
//! differing arity means something structurally impossible happened
//! (a hand-edited ledger, a corrupted store), which a numeric verdict would
//! misdescribe as "the science changed".

use serde::Deserialize;
use sos_core::{DeterminismLevel, ObjectId};
use sos_store::ObjectStore;
use sos_workflow::{RunLedger, StageId};

use crate::contract::{MatchRule, NodeClaim, Reproduced, VerifyReport, verify_reproduction};
use crate::error::{ReproError, Result};

/// Supplies the verdict for nodes the contract cannot decide from ids alone —
/// `L2` (within certificate) and `L1` (in distribution).
///
/// The driver calls this only for those levels; `L3`/`L0` nodes never reach
/// it. Implementations live in the numeric backend (`sos-scirust`), which is
/// where the certificates and standard errors are understood.
pub trait Certifier {
    /// Whether `rerun` reproduces `original` at `level`, for the node the
    /// stage `stage` produced.
    ///
    /// # Errors
    /// Implementation-defined; surfaced as
    /// [`ReproError::Certifier`].
    fn certify(
        &mut self,
        stage: &StageId,
        level: DeterminismLevel,
        original: ObjectId,
        rerun: ObjectId,
    ) -> core::result::Result<bool, String>;
}

/// A [`Certifier`] that refuses every `L2`/`L1` node.
///
/// Useful when a caller wants to verify a study believed to be entirely
/// `L3`/`L0` and wants to be *told* if it is not, rather than quietly
/// certifying nodes no backend examined. Every call it receives is a node
/// that needed real numerics.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCertifier;

impl Certifier for NoCertifier {
    fn certify(
        &mut self,
        stage: &StageId,
        level: DeterminismLevel,
        _original: ObjectId,
        _rerun: ObjectId,
    ) -> core::result::Result<bool, String> {
        Err(format!(
            "stage {stage} produced a {level:?} node, which needs a numeric \
             backend to certify; none was supplied"
        ))
    }
}

/// The declared level of a stored object, read **without knowing its body
/// type** — the same trick [`sos_store::ObjectHeader`] uses for provenance
/// walking: serde ignores every other field.
#[derive(Debug, Deserialize)]
struct LevelHeader {
    level: DeterminismLevel,
}

/// Verify that `rerun` reproduces `original`, node by node.
///
/// Pairs the two ledgers stage by stage, reads each original output's declared
/// [`DeterminismLevel`] from `store`, decides `L3`/`L0` nodes by id equality,
/// and delegates `L2`/`L1` nodes to `certifier`. The result is the same
/// [`VerifyReport`] [`verify_reproduction`] produces — this driver assembles
/// its inputs rather than replacing it.
///
/// # Errors
/// [`ReproError::PlanMismatch`] if the ledgers ran different plans;
/// [`ReproError::ScheduleMismatch`] if their stage sequences disagree;
/// [`ReproError::OutputArityMismatch`] if a stage emitted a different number
/// of objects; [`ReproError::UnreadableClaim`] if an original output is
/// absent from the store or its level cannot be read;
/// [`ReproError::Certifier`] if the backend fails.
pub fn verify_rerun<S: ObjectStore, C: Certifier>(
    original: &RunLedger,
    rerun: &RunLedger,
    store: &S,
    certifier: &mut C,
) -> Result<VerifyReport> {
    if original.plan_digest != rerun.plan_digest
    {
        return Err(ReproError::PlanMismatch {
            original: original.plan_digest,
            rerun: rerun.plan_digest,
        });
    }
    if original.steps.len() != rerun.steps.len()
    {
        return Err(ReproError::ScheduleMismatch {
            detail: format!(
                "{} steps originally, {} on re-run",
                original.steps.len(),
                rerun.steps.len()
            ),
        });
    }

    let mut claims = Vec::new();
    let mut reproduced = Vec::new();

    for (a, b) in original.steps.iter().zip(&rerun.steps)
    {
        if a.stage != b.stage
        {
            return Err(ReproError::ScheduleMismatch {
                detail: format!(
                    "step order diverged: `{}` originally, `{}` on re-run",
                    a.stage, b.stage
                ),
            });
        }
        if a.outputs.len() != b.outputs.len()
        {
            return Err(ReproError::OutputArityMismatch {
                stage: a.stage.clone(),
                original: a.outputs.len(),
                rerun: b.outputs.len(),
            });
        }

        for (&was, &now) in a.outputs.iter().zip(&b.outputs)
        {
            let level = declared_level(store, was)?;
            claims.push(NodeClaim::new(was, level));
            reproduced.push(
                if MatchRule::of(level).is_id_decidable()
                {
                    Reproduced::Id(now)
                }
                else
                {
                    Reproduced::Certified(certifier.certify(&a.stage, level, was, now).map_err(
                        |detail| ReproError::Certifier {
                            stage: a.stage.clone(),
                            detail,
                        },
                    )?)
                },
            );
        }
    }

    verify_reproduction(&claims, &reproduced)
}

/// Read an object's declared determinism level from the store.
fn declared_level<S: ObjectStore>(store: &S, id: ObjectId) -> Result<DeterminismLevel> {
    let record = store
        .get_raw(id)
        .ok_or(ReproError::UnreadableClaim { id, detail: None })?;
    serde_json::from_slice::<LevelHeader>(&record.bytes)
        .map(|h| h.level)
        .map_err(|e| ReproError::UnreadableClaim {
            id,
            detail: Some(e.to_string()),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use sos_core::SemVer;
    use sos_core::canonical::{Canonical, CanonicalEncoder};
    use sos_core::{Body, HashAlgo, Object};
    use sos_store::{MemoryStore, TypedStore};
    use sos_workflow::{CacheKey, LedgerStep, StageDescriptor, StepOutcome};

    use serde::Serialize;

    /// A minimal stored body, so tests can put real objects in a real store.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Sample {
        tag: String,
    }

    impl Canonical for Sample {
        fn encode(&self, enc: &mut CanonicalEncoder) {
            enc.str(&self.tag);
        }
    }

    impl Body for Sample {
        const KIND: &'static str = "ReproDriverSample";
        const SCHEMA_VERSION: u32 = 1;
    }

    fn digest(tag: &[u8]) -> sos_core::Digest {
        HashAlgo::Sha256.hash(b"test", tag)
    }

    /// Seal an object at `level` into `store`, returning its id.
    fn put(store: &mut MemoryStore, tag: &str, level: DeterminismLevel) -> ObjectId {
        let obj = Object::builder(Sample {
            tag: tag.to_string(),
        })
        .level(level)
        .seal();
        store.put_object(&obj).unwrap()
    }

    fn step(stage: &str, outputs: Vec<ObjectId>) -> LedgerStep {
        LedgerStep {
            stage: StageId::new(stage),
            cache_key: CacheKey::compute(
                &StageDescriptor::new(stage, SemVer::new(1, 0, 0), digest(b"plugin")),
                &[],
                &digest(stage.as_bytes()),
                0,
                &digest(b"env"),
            ),
            outcome: StepOutcome::Ran,
            outputs,
        }
    }

    fn ledger(plan: &[u8], steps: Vec<LedgerStep>) -> RunLedger {
        RunLedger {
            plan_digest: digest(plan),
            env_digest: digest(b"env"),
            steps,
        }
    }

    /// A certifier that answers with a fixed verdict and records what it saw.
    struct Fixed {
        verdict: bool,
        seen: Vec<(StageId, DeterminismLevel)>,
    }

    impl Fixed {
        fn new(verdict: bool) -> Self {
            Self {
                verdict,
                seen: Vec::new(),
            }
        }
    }

    impl Certifier for Fixed {
        fn certify(
            &mut self,
            stage: &StageId,
            level: DeterminismLevel,
            _o: ObjectId,
            _r: ObjectId,
        ) -> core::result::Result<bool, String> {
            self.seen.push((stage.clone(), level));
            Ok(self.verdict)
        }
    }

    #[test]
    fn an_identical_l3_rerun_reproduces() {
        let mut store = MemoryStore::new();
        let a = put(&mut store, "a", DeterminismLevel::L3);
        let b = put(&mut store, "b", DeterminismLevel::L3);
        let original = ledger(b"p", vec![step("one", vec![a]), step("two", vec![b])]);

        let report = verify_rerun(&original, &original, &store, &mut NoCertifier).unwrap();
        assert!(report.reproduced());
        assert_eq!(report.nodes.len(), 2);
        assert_eq!(report.level, DeterminismLevel::L3);
    }

    #[test]
    fn a_changed_l3_output_is_localized_to_its_node() {
        let mut store = MemoryStore::new();
        let a = put(&mut store, "a", DeterminismLevel::L3);
        let b = put(&mut store, "b", DeterminismLevel::L3);
        let other = put(&mut store, "b-prime", DeterminismLevel::L3);

        let original = ledger(b"p", vec![step("one", vec![a]), step("two", vec![b])]);
        let rerun = ledger(b"p", vec![step("one", vec![a]), step("two", vec![other])]);

        let report = verify_rerun(&original, &rerun, &store, &mut NoCertifier).unwrap();
        assert!(!report.reproduced());
        let deviation = report.first_deviation().unwrap();
        assert_eq!(deviation.node, b, "the deviation names the original node");
        assert_eq!(deviation.verdict, crate::NodeVerdict::Diverged);
    }

    #[test]
    fn l2_and_l1_nodes_go_to_the_certifier_and_l3_ones_do_not() {
        let mut store = MemoryStore::new();
        let exact = put(&mut store, "exact", DeterminismLevel::L3);
        let tol = put(&mut store, "tol", DeterminismLevel::L2);
        let stoch = put(&mut store, "stoch", DeterminismLevel::L1);

        let original = ledger(
            b"p",
            vec![
                step("exact", vec![exact]),
                step("tol", vec![tol]),
                step("stoch", vec![stoch]),
            ],
        );
        // The re-run produced *different* ids for every node. The L3 node
        // must therefore diverge, while the other two are the backend's call.
        let other_exact = put(&mut store, "exact-2", DeterminismLevel::L3);
        let other_tol = put(&mut store, "tol-2", DeterminismLevel::L2);
        let other_stoch = put(&mut store, "stoch-2", DeterminismLevel::L1);
        let rerun = ledger(
            b"p",
            vec![
                step("exact", vec![other_exact]),
                step("tol", vec![other_tol]),
                step("stoch", vec![other_stoch]),
            ],
        );

        let mut certifier = Fixed::new(true);
        let report = verify_rerun(&original, &rerun, &store, &mut certifier).unwrap();

        assert_eq!(
            certifier.seen,
            vec![
                (StageId::new("tol"), DeterminismLevel::L2),
                (StageId::new("stoch"), DeterminismLevel::L1),
            ],
            "only the non-id-decidable nodes reach the backend"
        );
        // The L2/L1 nodes were certified; the L3 node still diverged on id.
        assert!(!report.reproduced());
        assert_eq!(report.first_deviation().unwrap().node, exact);
        // The sub-DAG's level is the weakest node's.
        assert_eq!(report.level, DeterminismLevel::L1);
    }

    #[test]
    fn a_refusing_backend_diverges_rather_than_erroring() {
        let mut store = MemoryStore::new();
        let tol = put(&mut store, "tol", DeterminismLevel::L2);
        let original = ledger(b"p", vec![step("tol", vec![tol])]);

        let report = verify_rerun(&original, &original, &store, &mut Fixed::new(false)).unwrap();
        assert!(!report.reproduced());
        assert_eq!(
            report.first_deviation().unwrap().verdict,
            crate::NodeVerdict::Diverged
        );
    }

    #[test]
    fn no_certifier_reports_which_stage_needed_numerics() {
        // The point of `NoCertifier`: be told that a study is not as
        // id-decidable as assumed, rather than certifying nodes nobody looked
        // at.
        let mut store = MemoryStore::new();
        let tol = put(&mut store, "tol", DeterminismLevel::L2);
        let original = ledger(b"p", vec![step("integrate", vec![tol])]);

        let err = verify_rerun(&original, &original, &store, &mut NoCertifier).unwrap_err();
        assert!(matches!(err, ReproError::Certifier { .. }), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("integrate"), "{text}");
    }

    #[test]
    fn verifying_across_different_plans_is_refused() {
        let mut store = MemoryStore::new();
        let a = put(&mut store, "a", DeterminismLevel::L3);
        let original = ledger(b"plan-a", vec![step("one", vec![a])]);
        let other_plan = ledger(b"plan-b", vec![step("one", vec![a])]);

        assert!(matches!(
            verify_rerun(&original, &other_plan, &store, &mut NoCertifier),
            Err(ReproError::PlanMismatch { .. })
        ));
    }

    #[test]
    fn a_reordered_or_shorter_schedule_is_refused() {
        let mut store = MemoryStore::new();
        let a = put(&mut store, "a", DeterminismLevel::L3);
        let b = put(&mut store, "b", DeterminismLevel::L3);

        let original = ledger(b"p", vec![step("one", vec![a]), step("two", vec![b])]);
        let reordered = ledger(b"p", vec![step("two", vec![b]), step("one", vec![a])]);
        assert!(matches!(
            verify_rerun(&original, &reordered, &store, &mut NoCertifier),
            Err(ReproError::ScheduleMismatch { .. })
        ));

        let shorter = ledger(b"p", vec![step("one", vec![a])]);
        assert!(matches!(
            verify_rerun(&original, &shorter, &store, &mut NoCertifier),
            Err(ReproError::ScheduleMismatch { .. })
        ));
    }

    #[test]
    fn an_arity_mismatch_is_an_error_naming_the_stage() {
        // See the module docs: a stage's output count is the plugin's, and
        // the plans already agree, so this is structurally impossible rather
        // than a numeric divergence.
        let mut store = MemoryStore::new();
        let a = put(&mut store, "a", DeterminismLevel::L3);
        let b = put(&mut store, "b", DeterminismLevel::L3);

        let original = ledger(b"p", vec![step("pair", vec![a, b])]);
        let single = ledger(b"p", vec![step("pair", vec![a])]);

        let err = verify_rerun(&original, &single, &store, &mut NoCertifier).unwrap_err();
        match err
        {
            ReproError::OutputArityMismatch {
                stage,
                original,
                rerun,
            } =>
            {
                assert_eq!(stage, StageId::new("pair"));
                assert_eq!((original, rerun), (2, 1));
            },
            other => panic!("expected OutputArityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn an_object_missing_from_the_store_is_not_assumed_reproducible() {
        // Guessing `L3` for an unreadable claim would silently decide it by
        // id equality — turning "we cannot check this" into "it reproduced".
        let store = MemoryStore::new(); // deliberately empty
        let ghost = ObjectId::compute(HashAlgo::Sha256, b"t", b"never-stored");
        let original = ledger(b"p", vec![step("one", vec![ghost])]);

        assert!(matches!(
            verify_rerun(&original, &original, &store, &mut NoCertifier),
            Err(ReproError::UnreadableClaim { id, detail: None }) if id == ghost
        ));
    }

    #[test]
    fn an_empty_ledger_verifies_vacuously() {
        let store = MemoryStore::new();
        let empty = ledger(b"p", vec![]);
        let report = verify_rerun(&empty, &empty, &store, &mut NoCertifier).unwrap();
        assert!(report.reproduced(), "nothing to contradict");
        assert!(report.nodes.is_empty());
    }

    #[test]
    fn a_multi_output_stage_pairs_positionally() {
        let mut store = MemoryStore::new();
        let a = put(&mut store, "a", DeterminismLevel::L3);
        let b = put(&mut store, "b", DeterminismLevel::L3);
        let original = ledger(b"p", vec![step("pair", vec![a, b])]);
        // Swapping the two outputs must be caught: position is the pairing.
        let swapped = ledger(b"p", vec![step("pair", vec![b, a])]);

        let report = verify_rerun(&original, &swapped, &store, &mut NoCertifier).unwrap();
        assert!(!report.reproduced());
        assert_eq!(report.nodes.len(), 2);
    }
}
