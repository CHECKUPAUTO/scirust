//! `sos run` end to end: a study and its runs written to disk as **files**,
//! executed by the command, checked in the store afterwards.
//!
//! Everything here goes through the filesystem and the argument parser, not
//! through a Rust API — because that is the claim. `sos-scirust`'s own tests
//! already show a manifest reaching real numerics from Rust; what was missing
//! was that a person with a text editor and a shell could do it. These tests
//! write the two files a user would write and run the command a user would
//! run.

use std::fs;
use std::path::{Path, PathBuf};

use sos_cli::args::Args;
use sos_cli::run::{CATALOG_PLUGIN, StageConfig, catalog_plugin_hash};
use sos_core::Object;
use sos_scirust::model::{ModelKind, ModelRun, ModelSpec, ModeledTrajectoryBody};
use sos_scirust::spectrum::{SpectrumBody, SpectrumConfig, WindowKind};
use sos_store::{FileStore, ObjectStore, TypedStore};

fn temp_root(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("sos-run-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A logistic-growth study: two populations at different carrying capacities.
fn growth(capacity: f64) -> StageConfig {
    StageConfig::Model(ModelRun::new(
        ModelSpec::new(ModelKind::LogisticGrowth, [0.8, capacity]),
        [5.0],
        0.0,
        6.0,
        0.001,
    ))
}

/// Write `study.toml` and `runs.json` into `dir`, exactly as an author would.
fn write_study(dir: &Path, runs: &[StageConfig]) -> (PathBuf, PathBuf) {
    let stages: String = runs
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "\n[[stage]]\nid      = \"stage-{i}\"\nplugin  = \"{plugin}\"\n\
                 version = \"1.0.0\"\npin     = \"{pin}\"\nconfig  = \"{cfg}\"\n",
                plugin = r.plugin(),
                pin = sos_cli::run::plugin_hash(r.plugin()).to_hex(),
                cfg = r.address().to_hex(),
            )
        })
        .collect();
    let manifest = dir.join("study.toml");
    fs::write(
        &manifest,
        format!("[study]\nname = \"growth-sweep\"\nseed = 3\n{stages}"),
    )
    .unwrap();

    let runs_path = dir.join("runs.json");
    fs::write(&runs_path, serde_json::to_string_pretty(runs).unwrap()).unwrap();
    (manifest, runs_path)
}

/// Invoke `sos run` the way the binary does: through `Args::parse`.
fn sos_run(
    manifest: &Path,
    runs: &Path,
    store: &Path,
    extra: &[&str],
) -> sos_cli::error::Result<String> {
    let mut argv = vec![
        manifest.to_str().unwrap().to_owned(),
        runs.to_str().unwrap().to_owned(),
        "--store".to_owned(),
        store.to_str().unwrap().to_owned(),
    ];
    argv.extend(extra.iter().map(|s| (*s).to_owned()));
    let args = Args::parse(&argv)?;
    sos_cli::run::run(&args)
}

#[test]
fn a_study_on_disk_runs_and_stores_its_results() {
    let dir = temp_root("basic");
    let store_path = dir.join("store");
    let runs = [growth(100.0), growth(50.0)];
    let (manifest, runs_file) = write_study(&dir, &runs);

    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 2 of 2 stage(s)"), "{out}");
    assert!(out.contains("stage-0") && out.contains("stage-1"), "{out}");

    // The results are really in the store, and still name their model.
    let store = FileStore::open(&store_path).unwrap();
    let ids = store.object_ids();
    assert_eq!(ids.len(), 2, "one object per stage");
    let mut capacities: Vec<f64> = Vec::new();
    for id in ids
    {
        let obj: Object<ModeledTrajectoryBody> = store.get_object(id).unwrap().unwrap();
        assert_eq!(obj.body.model.kind, ModelKind::LogisticGrowth);
        assert_eq!(obj.body.seed, 3, "the study's seed reached the run");
        capacities.push(obj.body.model.params[1]);
        // Real integration: logistic growth approaches its own capacity.
        let x_end = obj.body.trajectory.last().unwrap().1[0];
        let k = obj.body.model.params[1];
        assert!(x_end > 0.8 * k && x_end < k, "{x_end} vs capacity {k}");
    }
    capacities.sort_by(f64::total_cmp);
    assert_eq!(capacities, vec![50.0, 100.0]);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_runs_file_that_does_not_contain_the_pinned_config_is_refused() {
    // The property pinning exists for: pointing the command at a *different*
    // experiment's runs file must fail, not quietly compute something else.
    let dir = temp_root("mismatch");
    let store_path = dir.join("store");
    let (manifest, _) = write_study(&dir, &[growth(100.0)]);

    let other = dir.join("other-runs.json");
    fs::write(&other, serde_json::to_string(&[growth(999.0)]).unwrap()).unwrap();

    let err = sos_run(&manifest, &other, &store_path, &["--allow", "effectful"]).unwrap_err();
    assert!(
        err.to_string().contains("no model run is on offer"),
        "{err}"
    );
    let store = FileStore::open(&store_path).unwrap();
    assert!(
        store.object_ids().is_empty(),
        "a refused study stores nothing"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn capabilities_are_denied_by_default() {
    // No `--allow`, so the effectful plugin is refused. Least privilege has to
    // be the default or the gate is decorative.
    let dir = temp_root("caps");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);

    let err = sos_run(&manifest, &runs_file, &store_path, &[]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("denied: missing capabilities [Effectful]"),
        "expected a capability refusal, got: {msg}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_misspelled_capability_is_an_error_not_a_silent_no_op() {
    let dir = temp_root("typo");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);

    let err = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectfull"],
    )
    .unwrap_err();
    assert!(err.to_string().contains("unknown capability"), "{err}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_study_pinned_to_a_different_plugin_build_is_refused() {
    // The drift check `Dispatch` exists for, reached from the command line: a
    // manifest written against another build must not run against this one.
    let dir = temp_root("drift");
    let store_path = dir.join("store");
    let runs = [growth(100.0)];
    let (_, runs_file) = write_study(&dir, &runs);

    let manifest = dir.join("drifted.toml");
    fs::write(
        &manifest,
        format!(
            "[study]\nname = \"drifted\"\nseed = 3\n\n[[stage]]\nid      = \"s\"\n\
             plugin  = \"{CATALOG_PLUGIN}\"\nversion = \"1.0.0\"\n\
             pin     = \"{wrong}\"\nconfig  = \"{cfg}\"\n",
            wrong = sos_core::HashAlgo::Sha256
                .hash(b"some-other", b"build")
                .to_hex(),
            cfg = runs[0].address().to_hex(),
        ),
    )
    .unwrap();

    let err = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("drifted: expected content"),
        "expected a pin-drift refusal, got: {msg}"
    );
    assert!(
        msg.contains(&catalog_plugin_hash().to_hex()),
        "the refusal must name the hash actually resolved: {msg}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn rerunning_the_same_study_is_served_from_the_persisted_memo() {
    // The point of persisting: the second invocation is a different process
    // (a different `FileMemo::open`), and it still does no work.
    let dir = temp_root("rerun");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0), growth(50.0)]);

    let first = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(first.contains("ran 2 of 2 stage(s)"), "{first}");
    assert!(
        store_path.join(sos_cli::memo::MEMO_FILE).is_file(),
        "the memo must be written beside the store"
    );

    let ids_after_first = FileStore::open(&store_path).unwrap().object_ids();
    let second = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(
        second.contains("ran 0 of 2 stage(s)"),
        "the second run must be entirely cached, got: {second}"
    );

    // Same reported outputs, and the store is unchanged.
    assert_eq!(
        first.lines().skip(1).collect::<Vec<_>>(),
        second.lines().skip(1).collect::<Vec<_>>()
    );
    assert_eq!(
        ids_after_first,
        FileStore::open(&store_path).unwrap().object_ids()
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_corrupt_memo_stops_the_run_rather_than_silently_recomputing() {
    let dir = temp_root("corrupt-memo");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);
    sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();

    fs::write(store_path.join(sos_cli::memo::MEMO_FILE), b"{not json").unwrap();
    assert!(
        sos_run(
            &manifest,
            &runs_file,
            &store_path,
            &["--allow", "effectful"]
        )
        .is_err(),
        "an unreadable memo must be surfaced, not reset"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_different_environment_label_is_a_different_run() {
    // `--env` is folded into every cache key, so binding results to a host is
    // possible even though the default is deliberately host-independent.
    let dir = temp_root("env");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);

    sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful", "--env", "lab-workstation-3"],
    )
    .unwrap();
    // The object is content-addressed and host-independent by construction, so
    // the store still holds one — what changes is the cache key, so the run
    // happens again rather than being served from the memo the first
    // invocation just persisted.
    assert!(out.contains("ran 1 of 1 stage(s)"), "{out}");
    assert_eq!(FileStore::open(&store_path).unwrap().object_ids().len(), 1);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn sos_address_emits_the_fields_a_study_must_pin() {
    // Without this the command is unusable from a shell: a manifest pins its
    // stages by content address, and there was no way to obtain those hex
    // strings short of writing Rust.
    let dir = temp_root("address");
    let runs = [growth(100.0), growth(50.0)];
    let runs_path = dir.join("runs.json");
    fs::write(&runs_path, serde_json::to_string(&runs).unwrap()).unwrap();

    let argv = vec![runs_path.to_str().unwrap().to_owned()];
    let out = sos_cli::run::address(&Args::parse(&argv).unwrap()).unwrap();

    // Paste-ready, and the addresses are the ones `sos run` will resolve.
    for r in &runs
    {
        assert!(out.contains(&r.address().to_hex()), "{out}");
    }
    assert!(out.contains(&catalog_plugin_hash().to_hex()), "{out}");
    assert!(out.contains("ecology/logistic-growth"), "{out}");
    assert_eq!(out.matches("[[stage]]").count(), 2, "{out}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn what_sos_address_prints_is_what_sos_run_accepts() {
    // The two commands agree, checked by feeding one into the other exactly as
    // a user would — the whole workflow, with no Rust in between.
    let dir = temp_root("address-run");
    let store_path = dir.join("store");
    let runs = [growth(100.0)];
    let runs_path = dir.join("runs.json");
    fs::write(&runs_path, serde_json::to_string(&runs).unwrap()).unwrap();

    let stages =
        sos_cli::run::address(&Args::parse(&[runs_path.to_str().unwrap().to_owned()]).unwrap())
            .unwrap();
    let body: String = stages
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let manifest = dir.join("study.toml");
    fs::write(
        &manifest,
        format!("[study]\nname = \"pasted\"\nseed = 1\n{body}"),
    )
    .unwrap();

    let out = sos_run(
        &manifest,
        &runs_path,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 1 of 1 stage(s)"), "{out}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_runs_file_naming_an_unknown_model_is_refused_by_sos_address() {
    let dir = temp_root("unknown-model");
    let runs_path = dir.join("runs.json");
    fs::write(
        &runs_path,
        r#"[{"kind":"model","model":{"model":"ecology/logistic-decay","params":["0.5"]},
             "y0":["1"],"t0":"0","t_end":"1","step":"0.1"}]"#,
    )
    .unwrap();

    let argv = vec![runs_path.to_str().unwrap().to_owned()];
    let err = sos_cli::run::address(&Args::parse(&argv).unwrap()).unwrap_err();
    assert!(err.to_string().contains("ecology/logistic-decay"), "{err}");
    fs::remove_dir_all(&dir).ok();
}

/// A 64 Hz tone sampled at 1024 Hz over 1024 samples — bin 64 exactly.
fn tone() -> StageConfig {
    StageConfig::Spectrum(SpectrumConfig::new(
        (0..1024)
            .map(|i| (std::f64::consts::TAU * 64.0 * f64::from(i) / 1024.0).sin())
            .collect(),
        1024.0,
        WindowKind::Hann,
    ))
}

#[test]
fn one_study_can_mix_both_backends() {
    // The point of tagging the runs file: a study is not restricted to one
    // kind of science. Here it simulates a population and measures a signal,
    // and each stage is routed to the plugin its configuration names.
    let dir = temp_root("mixed");
    let store_path = dir.join("store");
    let configs = [growth(100.0), tone()];
    let (manifest, runs_file) = write_study(&dir, &configs);

    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 2 of 2 stage(s)"), "{out}");

    // Two objects of two different kinds, each real.
    let store = FileStore::open(&store_path).unwrap();
    let ids = store.object_ids();
    assert_eq!(ids.len(), 2);
    let mut saw_trajectory = false;
    let mut saw_spectrum = false;
    for id in ids
    {
        if let Ok(Some(obj)) = store.get_object::<ModeledTrajectoryBody>(id)
        {
            assert_eq!(obj.body.model.kind, ModelKind::LogisticGrowth);
            saw_trajectory = true;
        }
        else if let Ok(Some(obj)) = store.get_object::<SpectrumBody>(id)
        {
            // Real spectral analysis: the tone is in its own bin.
            let peak = obj
                .body
                .spectrum
                .psd
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(peak, 64, "the tone must land in bin 64");
            assert_eq!(obj.body.spectrum.window, WindowKind::Hann);
            saw_spectrum = true;
        }
    }
    assert!(saw_trajectory && saw_spectrum, "both kinds must be stored");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn sos_address_routes_each_config_to_its_own_plugin() {
    let dir = temp_root("routing");
    let runs_path = dir.join("runs.json");
    fs::write(
        &runs_path,
        serde_json::to_string(&[growth(100.0), tone()]).unwrap(),
    )
    .unwrap();

    let out =
        sos_cli::run::address(&Args::parse(&[runs_path.to_str().unwrap().to_owned()]).unwrap())
            .unwrap();
    assert!(out.contains("sim-catalog"), "{out}");
    assert!(out.contains("signal-periodogram"), "{out}");
    assert!(out.contains("hann window over 1024 samples"), "{out}");
    // Different plugins, so different pins.
    assert!(out.contains(&sos_cli::run::catalog_plugin_hash().to_hex()));
    assert!(out.contains(&sos_cli::run::spectrum_plugin_hash().to_hex()));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_untagged_configuration_is_refused() {
    // The tag is what routes a config to a backend; guessing from shape would
    // be exactly the silent substitution the rest of this system refuses.
    let dir = temp_root("untagged");
    let runs_path = dir.join("runs.json");
    fs::write(
        &runs_path,
        r#"[{"model":{"model":"epidemiology/sir","params":["0.6","0.2"]},
             "y0":["0.99","0.01","0"],"t0":"0","t_end":"1","step":"0.1"}]"#,
    )
    .unwrap();

    let argv = vec![runs_path.to_str().unwrap().to_owned()];
    let err = sos_cli::run::address(&Args::parse(&argv).unwrap()).unwrap_err();
    assert!(err.to_string().contains("kind"), "{err}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_command_is_reachable_through_the_top_level_dispatcher() {
    // `sos run ...` as the binary sees it, including the help text listing it.
    let dir = temp_root("dispatch");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);

    let argv: Vec<String> = vec![
        "run".to_owned(),
        manifest.to_str().unwrap().to_owned(),
        runs_file.to_str().unwrap().to_owned(),
        "--store".to_owned(),
        store_path.to_str().unwrap().to_owned(),
        "--allow".to_owned(),
        "effectful".to_owned(),
    ];
    assert_eq!(sos_cli::run(&argv), 0, "the command must succeed");

    let help = sos_cli::run(&["help".to_owned()]);
    assert_eq!(help, 0);
    fs::remove_dir_all(&dir).ok();
}
