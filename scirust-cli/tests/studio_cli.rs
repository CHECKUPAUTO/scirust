//! End-to-end tests against the real `scirust` binary.
//!
//! These check the things only a real process can check: what actually lands
//! on stdout versus stderr, and whether `--format json` output can be piped
//! into a machine.

use std::process::Command;

const ROBERTSON: &str = "docs/studio/tutorials/robertson_stiff.scirust.toml";
const SPRING_MASS: &str = "docs/studio/tutorials/spring_mass_damper.scirust.toml";

/// The tutorials are addressed relative to the repository root, which is one
/// level above this crate.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives inside the repository")
        .to_path_buf()
}

fn scirust() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_scirust"));
    cmd.current_dir(repo_root());
    cmd
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("scirust-cli-it-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&path).expect("mkdir");
    path
}

/// Regression: recording a run printed its confirmation to **stdout**, so
/// `scirust run --format json --store …` emitted a non-JSON first line and
/// could not be piped into anything. Status notices belong on stderr;
/// stdout carries the result.
#[test]
fn run_with_a_store_still_emits_parseable_json_on_stdout() {
    let store = temp_dir("json-store");
    let out = scirust()
        .args([
            "run",
            SPRING_MASS,
            "--store",
            store.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("runs");

    assert!(out.status.success(), "exit status {:?}", out.status);

    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout is not valid JSON ({e}):\n{stdout}"));
    assert_eq!(parsed["schema_version"], 2);

    // The notice is still shown to a human, just on the other stream.
    let stderr = String::from_utf8(out.stderr).expect("utf-8");
    assert!(
        stderr.contains("recorded as run"),
        "the run id must still be reported: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&store);
}

/// The adaptive capability's coordinates must reach the CLI's JSON output
/// unchanged — this is the user-visible half of result schema v2.
#[test]
fn json_output_carries_exact_adaptive_coordinates() {
    let out = scirust()
        .args(["run", ROBERTSON, "--format", "json"])
        .output()
        .expect("runs");
    assert!(out.status.success());

    let parsed: serde_json::Value =
        serde_json::from_str(&String::from_utf8(out.stdout).unwrap()).expect("valid JSON");

    assert_eq!(parsed["schema_version"], 2);
    let axis = &parsed["axes"][0];
    assert_eq!(axis["id"], "t");
    assert_eq!(axis["monotonicity"], "strictly_increasing");

    let values: Vec<f64> = axis["values"]
        .as_array()
        .expect("axis carries its coordinates")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect();
    assert!(values.len() > 10);

    let gaps: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
    let smallest = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
    let largest = gaps.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        largest > smallest * 10.0,
        "the stiff solver's own step sizes should span orders of magnitude, \
         got {smallest:e}..{largest:e}"
    );

    // Every series says which axis it belongs to.
    for series in parsed["series"].as_array().expect("series")
    {
        assert_eq!(series["axis_id"], "t");
        assert_eq!(
            series["values"].as_array().unwrap().len(),
            values.len(),
            "series and axis must align"
        );
    }
}

#[test]
fn catalog_json_is_parseable() {
    let out = scirust()
        .args(["catalog", "--format", "json"])
        .output()
        .expect("runs");
    assert!(out.status.success());
    let parsed: serde_json::Value =
        serde_json::from_str(&String::from_utf8(out.stdout).unwrap()).expect("valid JSON");
    assert_eq!(
        parsed.as_array().expect("an array").len(),
        5,
        "this build exposes five executable capabilities"
    );
}

#[test]
fn runs_list_json_is_parseable_after_a_recorded_run() {
    let store = temp_dir("list-store");
    let store_arg = store.to_str().unwrap().to_string();

    assert!(
        scirust()
            .args(["run", SPRING_MASS, "--store", &store_arg])
            .output()
            .expect("runs")
            .status
            .success()
    );

    let out = scirust()
        .args(["runs", "list", "--store", &store_arg, "--format", "json"])
        .output()
        .expect("runs");
    assert!(out.status.success());

    let parsed: serde_json::Value =
        serde_json::from_str(&String::from_utf8(out.stdout).unwrap()).expect("valid JSON");
    let entries = parsed.as_array().expect("an array");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["result_schema_version"], 2);

    let _ = std::fs::remove_dir_all(&store);
}
