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
use sos_scirust::pipeline::{TrajectorySpectrumBody, TrajectorySpectrumConfig};
use sos_scirust::spectrum::{
    AveragedSpectrumBody, SpectrumBody, SpectrumConfig, WelchConfig, WindowKind,
};
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

/// The stored ids whose kind is `kind` — the store now holds the run's own
/// record (a `Plan` and a `RunLedger`) beside the stage outputs, so a test
/// about results has to say which it means.
fn ids_of_kind(store: &FileStore, kind: &str) -> Vec<sos_core::ObjectId> {
    store
        .object_ids()
        .into_iter()
        .filter(|id| store.get_raw(*id).is_some_and(|r| r.kind.name == kind))
        .collect()
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
    let ids = ids_of_kind(&store, "ModeledTrajectory");
    assert_eq!(ids.len(), 2, "one result per stage");
    assert_eq!(store.object_ids().len(), 4, "results plus plan and ledger");
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

    // The *results* are identical — but the ledgers are not, and that is
    // correct: a `RunLedger` records the schedule actually taken, including
    // whether each stage ran or was reused. The second run legitimately did
    // something different (it reused), so it honestly records a different
    // history for the same outputs.
    let results = |out: &str| -> Vec<String> {
        out.lines()
            .filter(|l| l.trim_start().starts_with("stage-"))
            .map(str::to_owned)
            .collect()
    };
    assert_eq!(results(&first), results(&second));
    let ledger_line = |out: &str| parse_reported(out, "ledger");
    assert_ne!(
        ledger_line(&first),
        ledger_line(&second),
        "a run that reused rather than recomputed must not claim the same history"
    );
    // The plan is the same object, though — it is what was asked for, not
    // what happened.
    assert_eq!(
        parse_reported(&first, "plan"),
        parse_reported(&second, "plan")
    );
    let ids_after_second = FileStore::open(&store_path).unwrap().object_ids();
    assert_eq!(
        ids_after_second.len(),
        ids_after_first.len() + 1,
        "only the second run's own ledger is new"
    );
    assert!(
        ids_after_first
            .iter()
            .all(|id| ids_after_second.contains(id))
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
    let store = FileStore::open(&store_path).unwrap();
    assert_eq!(ids_of_kind(&store, "ModeledTrajectory").len(), 1);
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

/// The same tone, measured by averaging 7 overlapping segments.
fn averaged_tone() -> StageConfig {
    StageConfig::Welch(WelchConfig::new(
        (0..4096)
            .map(|i| (std::f64::consts::TAU * 64.0 * f64::from(i) / 1024.0).sin())
            .collect(),
        1024.0,
        WindowKind::Hann,
        1024,
        512,
    ))
}

#[test]
fn a_welch_stage_stores_its_segment_count_beside_the_numbers() {
    // The descriptor that says how much the estimate is worth has to survive
    // into the store, or a reader cannot tell an average of 7 from an average
    // of 1.
    let dir = temp_root("welch");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[averaged_tone()]);

    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 1 of 1 stage(s)"), "{out}");

    let store = FileStore::open(&store_path).unwrap();
    let id = ids_of_kind(&store, "AveragedSpectrum")[0];
    let obj: Object<AveragedSpectrumBody> = store.get_object(id).unwrap().unwrap();
    assert_eq!(obj.body.spectrum.segments, 7);
    assert_eq!(obj.body.spectrum.dropped_samples, 0);
    assert_eq!(obj.body.spectrum.window, WindowKind::Hann);
    // And it really measured the tone.
    let peak = obj
        .body
        .spectrum
        .psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(peak, 64);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn all_three_bindable_backends_run_in_one_study() {
    // The handler table is a table, not a special case: simulate a
    // population, measure a signal, and average a signal, in one study.
    let dir = temp_root("all-three");
    let store_path = dir.join("store");
    let configs = [growth(100.0), tone(), averaged_tone()];
    let (manifest, runs_file) = write_study(&dir, &configs);

    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 3 of 3 stage(s)"), "{out}");
    let store = FileStore::open(&store_path).unwrap();
    assert_eq!(
        store.object_ids().len(),
        5,
        "3 results plus plan and ledger"
    );

    // Three distinct plugins, three distinct pins.
    let stages =
        sos_cli::run::address(&Args::parse(&[runs_file.to_str().unwrap().to_owned()]).unwrap())
            .unwrap();
    for plugin in ["sim-catalog", "signal-periodogram", "signal-welch"]
    {
        assert!(stages.contains(plugin), "{stages}");
        assert!(
            stages.contains(&sos_cli::run::plugin_hash(plugin).to_hex()),
            "{stages}"
        );
    }
    fs::remove_dir_all(&dir).ok();
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
    assert_eq!(ids.len(), 4, "2 results plus plan and ledger");
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
fn a_run_records_its_plan_and_ledger_beside_the_results() {
    // Without these a result exists but the record of how it came to exist
    // does not — and `sos-repro`'s verify_object, which finds the producing
    // run by scanning stored ledgers, could never find anything.
    let dir = temp_root("ledger");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0), growth(50.0)]);

    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("plan "), "{out}");
    assert!(out.contains("ledger "), "{out}");

    // Two results plus the plan and the ledger.
    let store = FileStore::open(&store_path).unwrap();
    assert_eq!(store.object_ids().len(), 4);

    // The ledger really describes this run.
    let ledger_id = parse_reported(&out, "ledger");
    let ledger: Object<sos_workflow::RunLedger> = store.get_object(ledger_id).unwrap().unwrap();
    assert_eq!(ledger.body.steps.len(), 2);
    assert_eq!(ledger.body.ran_count(), 2);
    assert_eq!(ledger.body.steps[0].stage.0, "stage-0");

    // And the plan it names is the one that was stored.
    let plan_id = parse_reported(&out, "plan");
    let plan: Object<sos_workflow::Plan> = store.get_object(plan_id).unwrap().unwrap();
    assert_eq!(plan.body.digest(), ledger.body.plan_digest);
    fs::remove_dir_all(&dir).ok();
}

/// Pull the object id `sos run` reported on the line labelled `label`.
fn parse_reported(out: &str, label: &str) -> sos_core::ObjectId {
    let line = out
        .lines()
        .find(|l| l.trim_start().starts_with(label))
        .unwrap_or_else(|| panic!("no `{label}` line in:\n{out}"));
    let hex = line.split_whitespace().nth(1).expect("id after the label");
    sos_cli::args::parse_object_id(hex).expect("a reported id must parse")
}

#[test]
fn sos_verify_can_check_the_objects_sos_run_produced() {
    // It could not before: the four kinds `sos run` writes were not in
    // `verify`'s table, so the same binary reported "unrecognized kind" for
    // its own output.
    let dir = temp_root("verify");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0), tone(), averaged_tone()]);
    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();

    let store = FileStore::open(&store_path).unwrap();
    let mut checked = 0;
    for id in store.object_ids()
    {
        let report = sos_cli::verify::run(store_path.to_str(), id).unwrap();
        assert!(
            !report.contains("unrecognized kind"),
            "every kind this binary writes must be verifiable:\n{report}"
        );
        assert!(
            report.contains("OK (recomputed address matches)"),
            "{report}"
        );
        checked += 1;
    }
    // Three results, plus the plan and the ledger.
    assert_eq!(checked, 5, "{out}");
    fs::remove_dir_all(&dir).ok();
}

/// Invoke `sos verify --rerun` the way the binary does.
fn sos_verify_rerun(
    store: &Path,
    id: sos_core::ObjectId,
    runs: &Path,
) -> sos_cli::error::Result<String> {
    let argv = vec![
        id.to_string(),
        "--store".to_owned(),
        store.to_str().unwrap().to_owned(),
        "--rerun".to_owned(),
        runs.to_str().unwrap().to_owned(),
        "--allow".to_owned(),
        "effectful".to_owned(),
    ];
    let a = Args::parse(&argv)?;
    sos_cli::verify::rerun(a.flag("store"), id, &a)
}

#[test]
fn sos_verify_rerun_re_executes_and_reports_reproduction() {
    // The check RFC-0002 §10.4 describes, as opposed to the identity check:
    // find the run that made this object, run its plan again, and compare.
    let dir = temp_root("rerun-verify");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0), tone()]);
    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();

    let target = parse_reported(&out, "stage-0");
    let report = sos_verify_rerun(&store_path, target, &runs_file).unwrap();
    assert!(report.contains("REPRODUCED"), "{report}");
    assert!(report.contains("re-executed: 2 node(s)"), "{report}");
    assert!(report.contains("L3"), "{report}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_rerun_verification_does_not_read_the_persisted_memo() {
    // The failure mode that would make this whole command a lie: reusing
    // `sos run`'s memo would replay the recorded outputs and "verify"
    // nothing. Checked by making the memo claim an absurd answer — if the
    // verification consulted it, the re-run would return that instead of
    // recomputing, and the report would not say REPRODUCED.
    let dir = temp_root("rerun-memo");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);
    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();

    // Poison the persisted memo: every entry now points at nothing real.
    let memo_path = store_path.join(sos_cli::memo::MEMO_FILE);
    let poisoned = fs::read_to_string(&memo_path)
        .unwrap()
        .replace("sos1:", "sos1:ff");
    fs::write(&memo_path, poisoned).unwrap();

    // The verification still reproduces, because it never looks there.
    let target = parse_reported(&out, "stage-0");
    let report = sos_verify_rerun(&store_path, target, &runs_file).unwrap();
    assert!(report.contains("REPRODUCED"), "{report}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn verifying_an_object_no_stored_run_produced_is_an_error() {
    // An object with no producing ledger cannot be re-executed, and saying so
    // is the honest answer rather than reporting a vacuous success.
    let dir = temp_root("rerun-orphan");
    let store_path = dir.join("store");
    let (manifest, runs_file) = write_study(&dir, &[growth(100.0)]);
    let out = sos_run(
        &manifest,
        &runs_file,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();

    // The plan object is stored, but no run *produced* it as a stage output.
    let plan_id = parse_reported(&out, "plan");
    let err = sos_verify_rerun(&store_path, plan_id, &runs_file).unwrap_err();
    assert!(
        err.to_string().contains("no") || err.to_string().contains("ledger"),
        "{err}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_consuming_stage_records_its_upstream_as_a_provenance_parent() {
    // The payoff, seen from the outside: before `consumes`, every object a
    // study produced had zero parents, so a workflow engine emitted a
    // provenance graph with no edges between its own nodes. A study written
    // in TOML can now say a stage reads another's output, and the stored
    // object records it — with no change to any stage handler, because they
    // already copy `stage.inputs` into parents.
    let dir = temp_root("consumes");
    let store_path = dir.join("store");
    let configs = [growth(100.0), growth(50.0)];
    let runs_path = dir.join("runs.json");
    fs::write(&runs_path, serde_json::to_string(&configs).unwrap()).unwrap();

    let manifest = dir.join("study.toml");
    fs::write(
        &manifest,
        format!(
            "[study]\nname = \"chained\"\nseed = 3\n\n\
             [[stage]]\nid       = \"first\"\nplugin   = \"{p0}\"\nversion  = \"1.0.0\"\n\
             pin      = \"{pin0}\"\nconfig   = \"{c0}\"\n\n\
             [[stage]]\nid       = \"second\"\nplugin   = \"{p1}\"\nversion  = \"1.0.0\"\n\
             pin      = \"{pin1}\"\nconfig   = \"{c1}\"\nconsumes = [\"first\"]\n",
            p0 = configs[0].plugin(),
            pin0 = sos_cli::run::plugin_hash(configs[0].plugin()).to_hex(),
            c0 = configs[0].address().to_hex(),
            p1 = configs[1].plugin(),
            pin1 = sos_cli::run::plugin_hash(configs[1].plugin()).to_hex(),
            c1 = configs[1].address().to_hex(),
        ),
    )
    .unwrap();

    let out = sos_run(
        &manifest,
        &runs_path,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 2 of 2 stage(s)"), "{out}");

    let first = parse_reported(&out, "first");
    let second = parse_reported(&out, "second");
    let store = FileStore::open(&store_path).unwrap();
    let downstream: Object<ModeledTrajectoryBody> = store.get_object(second).unwrap().unwrap();
    assert_eq!(
        downstream.parents,
        vec![first],
        "the consuming stage's output must name what it read"
    );
    assert_eq!(downstream.repro.inputs, vec![first]);

    // And the upstream, which consumed nothing, still has none.
    let upstream: Object<ModeledTrajectoryBody> = store.get_object(first).unwrap().unwrap();
    assert!(upstream.parents.is_empty());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_study_simulates_a_system_and_then_analyses_what_it_produced() {
    // The arc this closes. Before `consumes` a study could simulate *or*
    // analyse; before the trajectory backend it could pass an object to a
    // stage that had no idea what to do with it. Now one study does both, and
    // the signal being analysed was computed rather than typed in.
    let dir = temp_root("pipeline");
    let store_path = dir.join("store");

    // A Lotka-Volterra system sampled every 0.01 s, then the spectrum of its
    // predator component.
    let configs = [
        StageConfig::Model(ModelRun::new(
            ModelSpec::new(ModelKind::LotkaVolterra, [1.1, 0.4, 0.4, 0.1]),
            [10.0, 5.0],
            0.0,
            40.96,
            0.01,
        )),
        StageConfig::Trajectory(TrajectorySpectrumConfig::new(
            1,
            WindowKind::Hann,
            1024,
            512,
        )),
    ];
    let runs_path = dir.join("runs.json");
    fs::write(&runs_path, serde_json::to_string(&configs).unwrap()).unwrap();

    let manifest = dir.join("study.toml");
    fs::write(
        &manifest,
        format!(
            "[study]\nname = \"predator-prey-spectrum\"\nseed = 11\n\n\
             [[stage]]\nid       = \"simulate\"\nplugin   = \"{p0}\"\nversion  = \"1.0.0\"\n\
             pin      = \"{pin0}\"\nconfig   = \"{c0}\"\n\n\
             [[stage]]\nid       = \"analyse\"\nplugin   = \"{p1}\"\nversion  = \"1.0.0\"\n\
             pin      = \"{pin1}\"\nconfig   = \"{c1}\"\nconsumes = [\"simulate\"]\n",
            p0 = configs[0].plugin(),
            pin0 = sos_cli::run::plugin_hash(configs[0].plugin()).to_hex(),
            c0 = configs[0].address().to_hex(),
            p1 = configs[1].plugin(),
            pin1 = sos_cli::run::plugin_hash(configs[1].plugin()).to_hex(),
            c1 = configs[1].address().to_hex(),
        ),
    )
    .unwrap();

    let out = sos_run(
        &manifest,
        &runs_path,
        &store_path,
        &["--allow", "effectful"],
    )
    .unwrap();
    assert!(out.contains("ran 2 of 2 stage(s)"), "{out}");

    let simulated = parse_reported(&out, "simulate");
    let analysed = parse_reported(&out, "analyse");
    let store = FileStore::open(&store_path).unwrap();

    let spectrum: Object<TrajectorySpectrumBody> = store.get_object(analysed).unwrap().unwrap();
    // Nobody wrote 100 Hz anywhere: it is 1/0.01, read off the trajectory.
    assert!(
        (spectrum.body.measured.sample_rate_hz - 100.0).abs() < 1e-6,
        "derived rate was {}",
        spectrum.body.measured.sample_rate_hz
    );
    assert_eq!(spectrum.body.measured.component, 1);
    assert!(spectrum.body.measured.spectrum.segments >= 2);
    // And it names the trajectory it measured.
    assert_eq!(spectrum.parents, vec![simulated]);

    // The whole pipeline verifies by re-execution, not just by identity.
    let report = sos_verify_rerun(&store_path, analysed, &runs_path).unwrap();
    assert!(report.contains("REPRODUCED"), "{report}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_two_body_kinds_collide() {
    // The guard for the defect `sos verify` surfaced: sos-planner's `Plan` and
    // sos-workflow's `Plan` both declared kind "Plan". Two structurally
    // different types under one kind defeats the store's own KindMismatch
    // check — the very thing that stops a record being decoded as the wrong
    // type — and it fails loudly only when the layouts happen to be
    // incompatible. Here it produced `missing field 'policy'`; with
    // overlapping shapes it could have silently "verified" the wrong type.
    use sos_core::Body;
    let kinds: Vec<&'static str> = vec![
        <sos_reasoning::Derivation as Body>::KIND,
        <sos_reasoning::Contradiction as Body>::KIND,
        <sos_theory::Theory as Body>::KIND,
        <sos_workflow::RunLedger as Body>::KIND,
        <sos_workflow::Plan as Body>::KIND,
        <sos_repro::EnvLock as Body>::KIND,
        <sos_planner::Plan as Body>::KIND,
        <sos_publication::Publication as Body>::KIND,
        <sos_publication::ReleaseManifest as Body>::KIND,
        <sos_ccos::Proposal as Body>::KIND,
        <sos_ccos::Admission as Body>::KIND,
        <sos_knowledge::Edge as Body>::KIND,
        <sos_curiosity::ScientificQuestion as Body>::KIND,
        <sos_curiosity::CuriosityPolicy as Body>::KIND,
        <sos_scirust::stage::TrajectoryBody as Body>::KIND,
        <sos_scirust::model::ModeledTrajectoryBody as Body>::KIND,
        <sos_scirust::spectrum::SpectrumBody as Body>::KIND,
        <sos_scirust::spectrum::AveragedSpectrumBody as Body>::KIND,
    ];
    let mut seen = std::collections::BTreeSet::new();
    for kind in &kinds
    {
        assert!(
            seen.insert(*kind),
            "two Body types both declare kind `{kind}`"
        );
    }
    assert_eq!(seen.len(), kinds.len());
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
