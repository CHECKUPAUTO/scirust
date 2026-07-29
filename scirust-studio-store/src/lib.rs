//! `scirust-studio-store` — immutable, content-hashed storage for SciRust
//! Studio runs.
//!
//! This is Phase 2B of the SciRust Studio effort (see
//! `docs/studio/adr/0004-immutable-run-storage.md` and
//! `docs/studio/STORAGE_LAYOUT.md`). One run is one directory holding three
//! files:
//!
//! ```text
//! <root>/runs/<run_id>/
//!     manifest.json           provenance, hashes, status, environment
//!     scenario.scirust.toml   the exact input, byte for byte
//!     result.json             the RunResult (completed runs only)
//! ```
//!
//! Three properties hold, and each is tested rather than asserted:
//!
//! 1. **Immutable.** The API offers no way to modify a finalized run — no
//!    update, no overwrite, no re-save. [`RunStore::verify`] recomputes the
//!    stored hashes so a run modified *outside* this API is detectable too.
//! 2. **Atomic.** A run is assembled under a `.partial-` directory and
//!    published by a single [`std::fs::rename`], so a reader never sees a
//!    half-written run.
//! 3. **Interruption is evidence, not corruption.** A writer that dies
//!    leaves its `.partial-` directory behind, where
//!    [`RunStore::interrupted`] reports it — including the scenario that
//!    was being attempted. Nothing is silently cleaned up.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod manifest;
mod store;

pub use error::StoreError;
pub use manifest::{
    EnvironmentFingerprint, MANIFEST_SCHEMA_VERSION, RunManifest, RunStatus, manifests_to_json,
};
pub use store::{InterruptedRun, LoadedRunResult, PendingRun, RunStore};

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_studio_runtime::{
        Axis, AxisMonotonicity, DeterminismClass, Metric, MetricValue, RESULT_SCHEMA_VERSION,
        RunProvenance, RunResult, RunSummary, Series,
    };

    const SCENARIO: &str = "schema_version = 1\n[capability]\nid = \"demo\"\n";

    fn sample_result() -> RunResult {
        RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: "demo".to_string(),
            summary: RunSummary {
                capability_display_name: "Demo".to_string(),
                scenario_name: "demo".to_string(),
                steps: 2,
                t_start: 0.0,
                t_end: 1.0,
            },
            axes: vec![Axis {
                id: "t".to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
                monotonicity: AxisMonotonicity::StrictlyIncreasing,
                values: vec![0.0, 0.5, 1.0],
            }],
            series: vec![Series {
                id: "x".to_string(),
                display_name: "X".to_string(),
                unit: "m".to_string(),
                axis_id: "t".to_string(),
                values: vec![1.0, 0.5, 0.25],
            }],
            metrics: vec![Metric {
                id: "final".to_string(),
                display_name: "Final".to_string(),
                value: MetricValue::Scalar(0.25),
                unit: Some("m".to_string()),
            }],
            warnings: vec![],
            verifications: vec![],
            provenance: RunProvenance {
                capability_id: "demo".to_string(),
                determinism: DeterminismClass::StrictSameBinarySameTarget,
                adapter_crate: "test".to_string(),
                adapter_version: "0.1.0".to_string(),
                started_at_rfc3339: "2026-01-01T00:00:00Z".to_string(),
                completed_at_rfc3339: "2026-01-01T00:00:01Z".to_string(),
                elapsed_seconds: 1.0,
            },
        }
    }

    fn temp_store() -> (tempfile::TempDir, RunStore) {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = RunStore::open(dir.path()).expect("open store");
        (dir, store)
    }

    #[test]
    fn a_completed_run_round_trips_scenario_result_and_manifest() {
        let (_dir, store) = temp_store();
        let result = sample_result();

        let pending = store.begin("demo", SCENARIO).unwrap();
        let run_id = pending.complete(&result).unwrap();

        assert_eq!(store.load_scenario(&run_id).unwrap(), SCENARIO);
        assert_eq!(store.load_result(&run_id).unwrap().as_v2(), Some(&result));

        let manifest = store.load_manifest(&run_id).unwrap();
        assert_eq!(manifest.run_id, run_id);
        assert_eq!(manifest.capability_id, "demo");
        assert_eq!(manifest.status, RunStatus::Completed);
        assert!(manifest.result_sha256.is_some());
        assert_eq!(manifest.manifest_schema_version, MANIFEST_SCHEMA_VERSION);
        assert!(
            manifest
                .environment
                .same_target_as(&EnvironmentFingerprint::current())
        );
    }

    /// Storage must be bit-exact, not merely close. Values chosen so that a
    /// shortest-representation round-trip that is off by one ULP fails here
    /// — this is what pins `serde_json`'s `float_roundtrip` feature in place.
    #[test]
    fn a_stored_result_loads_back_bit_for_bit() {
        let (_dir, store) = temp_store();
        let mut result = sample_result();
        result.axes[0].values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        result.series[0].values = vec![
            0.998969729210804,
            0.1 + 0.2,
            f64::MIN_POSITIVE,
            f64::MAX,
            -1.0e-300,
            std::f64::consts::PI,
        ];
        result.summary.t_start = 0.0;
        result.summary.t_end = 5.0;
        result.metrics[0].value = MetricValue::Scalar(0.9988752100232527);

        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&result)
            .unwrap();
        let loaded = store.load_result(&run_id).unwrap();
        let loaded = loaded.as_v2().expect("a run written now is v2").clone();

        for (a, b) in result.series[0]
            .values
            .iter()
            .zip(loaded.series[0].values.iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{a:?} came back as {b:?} — float parsing is not exact"
            );
        }
        assert_eq!(loaded, result);
    }

    #[test]
    fn a_stored_run_verifies_against_its_own_hashes() {
        let (_dir, store) = temp_store();
        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        store
            .verify(&run_id)
            .expect("a freshly written run verifies");
    }

    /// The point of storing hashes: editing a run behind the store's back is
    /// detected, and the error says which file changed.
    #[test]
    fn tampering_with_a_stored_result_is_detected() {
        let (_dir, store) = temp_store();
        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();

        let result_path = store.run_dir(&run_id).join("result.json");
        let mut text = std::fs::read_to_string(&result_path).unwrap();
        text = text.replace("0.25", "0.26");
        std::fs::write(&result_path, text).unwrap();

        let err = store.verify(&run_id).unwrap_err();
        match err
        {
            StoreError::HashMismatch {
                file, run_id: id, ..
            } =>
            {
                assert_eq!(file, "result.json");
                assert_eq!(id, run_id);
            },
            other => panic!("expected a hash mismatch, got {other:?}"),
        }
    }

    #[test]
    fn tampering_with_a_stored_scenario_is_detected() {
        let (_dir, store) = temp_store();
        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();

        let path = store.run_dir(&run_id).join("scenario.scirust.toml");
        std::fs::write(&path, "schema_version = 1\n# edited\n").unwrap();

        let err = store.verify(&run_id).unwrap_err();
        assert!(
            matches!(err, StoreError::HashMismatch { ref file, .. } if file == "scenario.scirust.toml"),
            "{err:?}"
        );
    }

    #[test]
    fn a_cancelled_run_is_recorded_with_its_scenario_but_no_result() {
        let (_dir, store) = temp_store();
        let run_id = store.begin("demo", SCENARIO).unwrap().cancel().unwrap();

        let manifest = store.load_manifest(&run_id).unwrap();
        assert_eq!(manifest.status, RunStatus::Cancelled);
        assert!(manifest.result_sha256.is_none());
        assert_eq!(store.load_scenario(&run_id).unwrap(), SCENARIO);

        let err = store.load_result(&run_id).unwrap_err();
        assert!(matches!(err, StoreError::NoResult { .. }), "{err:?}");
        store
            .verify(&run_id)
            .expect("a cancelled run still verifies");
    }

    #[test]
    fn a_failed_run_records_why() {
        let (_dir, store) = temp_store();
        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .fail("integration blew up at t = 3")
            .unwrap();

        let manifest = store.load_manifest(&run_id).unwrap();
        match manifest.status
        {
            RunStatus::Failed { message } => assert!(message.contains("blew up"), "{message}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// A run in progress must be invisible: `list` must not show it, and the
    /// finalized directory must not exist yet.
    #[test]
    fn a_run_is_invisible_until_it_is_finalized() {
        let (_dir, store) = temp_store();
        let pending = store.begin("demo", SCENARIO).unwrap();
        let run_id = pending.run_id().to_string();

        assert!(store.list().unwrap().is_empty());
        assert!(!store.run_dir(&run_id).exists());
        assert!(pending.pending_dir().exists());

        pending.complete(&sample_result()).unwrap();
        assert_eq!(store.list().unwrap().len(), 1);
        assert!(store.run_dir(&run_id).exists());
    }

    /// Dropping a `PendingRun` without finalizing it is exactly what a crash
    /// looks like on disk, so it is how interruption is tested.
    #[test]
    fn an_abandoned_run_is_reported_as_interrupted_with_its_scenario() {
        let (_dir, store) = temp_store();
        let pending = store.begin("demo", SCENARIO).unwrap();
        let run_id = pending.run_id().to_string();
        drop(pending);

        assert!(
            store.list().unwrap().is_empty(),
            "an interrupted run must not appear as a real run"
        );
        let interrupted = store.interrupted().unwrap();
        assert_eq!(interrupted.len(), 1);
        assert_eq!(interrupted[0].run_id, run_id);
        assert_eq!(interrupted[0].scenario_toml.as_deref(), Some(SCENARIO));
    }

    #[test]
    fn finalizing_clears_the_interrupted_marker() {
        let (_dir, store) = temp_store();
        let pending = store.begin("demo", SCENARIO).unwrap();
        pending.complete(&sample_result()).unwrap();
        assert!(store.interrupted().unwrap().is_empty());
    }

    #[test]
    fn discarding_an_interrupted_run_removes_only_that_run() {
        let (_dir, store) = temp_store();
        let kept = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        let abandoned = store
            .begin("demo", "schema_version = 1\n# other\n")
            .unwrap();
        let abandoned_id = abandoned.run_id().to_string();
        drop(abandoned);

        store.discard_interrupted(&abandoned_id).unwrap();
        assert!(store.interrupted().unwrap().is_empty());
        assert!(store.load_manifest(&kept).is_ok(), "the real run survived");
    }

    #[test]
    fn discarding_refuses_an_id_that_is_not_interrupted() {
        let (_dir, store) = temp_store();
        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        // A finalized run is not a `.partial-` directory, so this must not
        // delete it.
        let err = store.discard_interrupted(&run_id).unwrap_err();
        assert!(matches!(err, StoreError::NoSuchRun { .. }), "{err:?}");
        assert!(store.load_manifest(&run_id).is_ok(), "still there");
    }

    #[test]
    fn identical_scenarios_run_twice_get_distinct_ids() {
        let (_dir, store) = temp_store();
        let first = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        let second = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        assert_ne!(first, second);
        assert_eq!(store.list().unwrap().len(), 2);
    }

    #[test]
    fn identical_scenarios_share_a_scenario_hash() {
        let (_dir, store) = temp_store();
        let a = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        let b = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        assert_eq!(
            store.load_manifest(&a).unwrap().scenario_sha256,
            store.load_manifest(&b).unwrap().scenario_sha256,
            "the same input must hash the same"
        );
    }

    #[test]
    fn different_scenarios_hash_differently() {
        let (_dir, store) = temp_store();
        let a = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        let b = store
            .begin("demo", "schema_version = 1\n# different\n")
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        assert_ne!(
            store.load_manifest(&a).unwrap().scenario_sha256,
            store.load_manifest(&b).unwrap().scenario_sha256
        );
    }

    #[test]
    fn listing_is_chronological() {
        let (_dir, store) = temp_store();
        let mut ids = Vec::new();
        for i in 0..3
        {
            ids.push(
                store
                    .begin("demo", &format!("schema_version = 1\n# {i}\n"))
                    .unwrap()
                    .complete(&sample_result())
                    .unwrap(),
            );
        }
        let listed: Vec<String> = store
            .list()
            .unwrap()
            .into_iter()
            .map(|m| m.run_id)
            .collect();
        let mut sorted = listed.clone();
        sorted.sort();
        assert_eq!(listed, sorted, "list() must come back in run-id order");
        assert_eq!(listed.len(), ids.len());
    }

    #[test]
    fn an_unknown_run_id_is_reported_as_such() {
        let (_dir, store) = temp_store();
        let err = store.load_manifest("no-such-run").unwrap_err();
        assert!(matches!(err, StoreError::NoSuchRun { .. }), "{err:?}");
    }

    #[test]
    fn a_corrupt_manifest_is_reported_as_corrupt_not_missing() {
        let (_dir, store) = temp_store();
        let run_id = store
            .begin("demo", SCENARIO)
            .unwrap()
            .complete(&sample_result())
            .unwrap();
        std::fs::write(store.run_dir(&run_id).join("manifest.json"), "{ not json").unwrap();

        let err = store.load_manifest(&run_id).unwrap_err();
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err:?}");
    }

    #[test]
    fn reopening_an_existing_store_sees_its_runs() {
        let dir = tempfile::tempdir().unwrap();
        let run_id = {
            let store = RunStore::open(dir.path()).unwrap();
            store
                .begin("demo", SCENARIO)
                .unwrap()
                .complete(&sample_result())
                .unwrap()
        };
        let reopened = RunStore::open(dir.path()).unwrap();
        assert_eq!(reopened.list().unwrap().len(), 1);
        reopened.verify(&run_id).unwrap();
    }

    #[test]
    fn opening_creates_the_root_when_it_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("store");
        let store = RunStore::open(&nested).unwrap();
        assert!(nested.join("runs").is_dir());
        assert!(store.list().unwrap().is_empty());
    }
}
