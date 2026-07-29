//! Reading runs stored before result schema v2 existed.
//!
//! The fixtures under `tests/fixtures/` are **real files**, produced by the
//! shipped Phase 2B code and copied out of a real store before schema v2 was
//! written. They are not hand-authored approximations of what v1 looked
//! like, which is the only way this test can prove anything about existing
//! user data.
//!
//! The behaviour being pinned down is deliberately conservative: an old run
//! stays listable, verifiable and readable, but is never dressed up as
//! something it is not. In particular a v1 result has no axis coordinates,
//! and none are invented for it.

use scirust_studio_store::{LoadedRunResult, RunStore, StoreError};

const SM_RESULT: &str = include_str!("fixtures/v1_spring_mass_damper_result.json");
const SM_MANIFEST: &str = include_str!("fixtures/v1_spring_mass_damper_manifest.json");
const SM_SCENARIO: &str = include_str!("fixtures/v1_spring_mass_damper_scenario.scirust.toml");

const RB_RESULT: &str = include_str!("fixtures/v1_robertson_result.json");
const RB_MANIFEST: &str = include_str!("fixtures/v1_robertson_manifest.json");
const RB_SCENARIO: &str = include_str!("fixtures/v1_robertson_scenario.scirust.toml");

/// Plant a genuine v1 run into a fresh store, exactly as it sat on disk.
///
/// Written directly rather than through `RunStore::begin`/`complete`,
/// because those produce v2 — the whole point is to reconstruct a store as
/// it existed before this change.
fn store_with_v1_run(
    manifest: &str,
    result: &str,
    scenario: &str,
) -> (tempfile::TempDir, RunStore, String) {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = RunStore::open(dir.path()).expect("open");

    let run_id: String = serde_json::from_str::<serde_json::Value>(manifest)
        .expect("manifest parses")["run_id"]
        .as_str()
        .expect("run_id")
        .to_string();

    let run_dir = store.run_dir(&run_id);
    std::fs::create_dir_all(&run_dir).expect("mkdir");
    std::fs::write(run_dir.join("manifest.json"), manifest).expect("manifest");
    std::fs::write(run_dir.join("result.json"), result).expect("result");
    std::fs::write(run_dir.join("scenario.scirust.toml"), scenario).expect("scenario");

    (dir, store, run_id)
}

fn spring_mass_store() -> (tempfile::TempDir, RunStore, String) {
    store_with_v1_run(SM_MANIFEST, SM_RESULT, SM_SCENARIO)
}

fn robertson_store() -> (tempfile::TempDir, RunStore, String) {
    store_with_v1_run(RB_MANIFEST, RB_RESULT, RB_SCENARIO)
}

#[test]
fn a_v1_run_still_appears_in_the_listing() {
    let (_dir, store, run_id) = spring_mass_store();
    let runs = store.list().expect("lists");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].run_id, run_id);
    assert_eq!(runs[0].capability_id, "sim.mechanics.spring_mass_damper");
    assert_eq!(
        runs[0].result_schema_version, 1,
        "the manifest records the version the result was written under"
    );
}

/// The property that matters most for existing stores: integrity checking
/// still works. Hashes are over the stored bytes, so they are inherently
/// version-agnostic — this test makes that a guarantee rather than a
/// coincidence.
#[test]
fn a_v1_run_still_verifies() {
    let (_dir, store, run_id) = spring_mass_store();
    store.verify(&run_id).expect("a v1 run must still verify");
}

#[test]
fn tampering_with_a_v1_run_is_still_detected() {
    let (_dir, store, run_id) = spring_mass_store();
    let path = store.run_dir(&run_id).join("result.json");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replacen("1.0", "1.5", 1)).unwrap();

    let err = store.verify(&run_id).unwrap_err();
    assert!(
        matches!(err, StoreError::HashMismatch { ref file, .. } if file == "result.json"),
        "{err:?}"
    );
}

#[test]
fn a_v1_result_loads_as_v1_and_not_as_v2() {
    let (_dir, store, run_id) = spring_mass_store();
    let loaded = store.load_result(&run_id).expect("loads");

    assert_eq!(loaded.schema_version(), 1);
    assert!(loaded.as_v1().is_some(), "must decode as v1");
    assert!(
        loaded.as_v2().is_none(),
        "a v1 file must never be presented as a v2 result"
    );
    assert!(matches!(loaded, LoadedRunResult::V1(_)));
}

/// Everything a v1 result genuinely does contain stays available.
#[test]
fn v1_metrics_verifications_and_provenance_remain_readable() {
    let (_dir, store, run_id) = spring_mass_store();
    let loaded = store.load_result(&run_id).expect("loads");

    assert!(!loaded.metrics().is_empty(), "metrics must survive");
    assert!(
        !loaded.verifications().is_empty(),
        "scientific checks must survive"
    );
    assert_eq!(loaded.capability_id(), "sim.mechanics.spring_mass_damper");
    assert!(!loaded.provenance().adapter_version.is_empty());
    assert!(loaded.series_len() > 0, "series values are still there");
    assert!(loaded.summary().steps > 0);
}

/// The load-bearing honesty check.
#[test]
fn a_v1_result_never_claims_physical_coordinates() {
    use scirust_studio_runtime::XAxisMeaning;

    let (_dir, store, run_id) = spring_mass_store();
    let loaded = store.load_result(&run_id).expect("loads");
    assert_eq!(
        loaded.x_axis_meaning(),
        XAxisMeaning::SampleIndexOnly,
        "a consumer must be told it only has sample indices"
    );
}

/// Robertson is the case where treating a v1 result as evenly spaced would
/// be actively wrong, so it gets its own test rather than relying on the
/// general one.
#[test]
fn a_v1_adaptive_result_is_flagged_and_gets_no_time_axis() {
    use scirust_studio_runtime::XAxisMeaning;

    let (_dir, store, run_id) = robertson_store();
    let loaded = store.load_result(&run_id).expect("loads");

    let v1 = loaded.as_v1().expect("v1");
    assert!(
        v1.has_adaptive_solver(),
        "Robertson's stored samples are not uniformly spaced"
    );
    assert_eq!(loaded.x_axis_meaning(), XAxisMeaning::SampleIndexOnly);
    store.verify(&run_id).expect("still verifies");
}

/// A caller that requires exact coordinates is refused rather than served a
/// reconstruction.
#[test]
fn requiring_v2_refuses_a_v1_run_with_a_clear_error() {
    let (_dir, store, run_id) = spring_mass_store();
    let err = store.load_result_v2(&run_id).unwrap_err();
    match err
    {
        StoreError::UnsupportedResultSchema { found, .. } => assert_eq!(found, 1),
        other => panic!("expected UnsupportedResultSchema, got {other:?}"),
    }
    let text = store.load_result_v2(&run_id).unwrap_err().to_string();
    assert!(text.contains("v1"), "{text}");
}

/// Reading must not rewrite. An immutable store that silently upgraded its
/// own files would invalidate every hash it had recorded.
#[test]
fn loading_a_v1_run_leaves_the_stored_bytes_untouched() {
    let (_dir, store, run_id) = spring_mass_store();
    let path = store.run_dir(&run_id).join("result.json");
    let before = std::fs::read(&path).unwrap();

    let _ = store.load_result(&run_id).expect("loads");
    let _ = store.load_result_v2(&run_id);
    let _ = store.list();
    store.verify(&run_id).expect("verifies");

    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after, "reading must never rewrite a stored run");
}

/// v1 and v2 runs coexist: an existing store that gains new runs stays
/// entirely readable.
#[test]
fn a_store_can_hold_both_versions_at_once() {
    use scirust_studio_runtime::{
        Axis, AxisMonotonicity, DeterminismClass, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
        RunSummary, Series, SeriesRole,
    };

    let (_dir, store, v1_id) = spring_mass_store();

    let v2 = RunResult {
        schema_version: RESULT_SCHEMA_VERSION,
        capability_id: "demo".to_string(),
        summary: RunSummary {
            capability_display_name: "Demo".to_string(),
            scenario_name: "demo".to_string(),
            axis_id: "t".to_string(),
            steps: 1,
            t_start: 0.0,
            t_end: 1.0,
        },
        axes: vec![Axis {
            id: "t".to_string(),
            display_name: "time".to_string(),
            unit: "s".to_string(),
            monotonicity: AxisMonotonicity::StrictlyIncreasing,
            values: vec![0.0, 1.0],
        }],
        series: vec![Series {
            id: "x".to_string(),
            display_name: "X".to_string(),
            unit: "m".to_string(),
            axis_id: "t".to_string(),
            role: SeriesRole::Trajectory,
            values: vec![1.0, 0.5],
        }],
        fields: vec![],
        distributions: vec![],
        metrics: vec![],
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
            seed: None,
        },
    };
    let v2_id = store
        .begin("demo", "schema_version = 1\n")
        .unwrap()
        .complete(&v2)
        .unwrap();

    assert_eq!(store.list().unwrap().len(), 2);
    assert_eq!(store.load_result(&v1_id).unwrap().schema_version(), 1);
    assert_eq!(store.load_result(&v2_id).unwrap().schema_version(), 2);
    store.verify(&v1_id).unwrap();
    store.verify(&v2_id).unwrap();
    assert_eq!(store.load_result_v2(&v2_id).unwrap(), v2);
}

/// A version this build has never heard of is refused, not guessed at.
#[test]
fn an_unknown_future_schema_version_is_refused() {
    let (_dir, store, run_id) = spring_mass_store();
    let path = store.run_dir(&run_id).join("result.json");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replacen("\"schema_version\": 1", "\"schema_version\": 99", 1),
    )
    .unwrap();

    let err = store.load_result(&run_id).unwrap_err();
    match err
    {
        StoreError::UnsupportedResultSchema { found, .. } => assert_eq!(found, 99),
        other => panic!("expected UnsupportedResultSchema, got {other:?}"),
    }
}
