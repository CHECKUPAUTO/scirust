//! `scirust runs` — inspect the immutable run store.
//!
//! Reading only. Nothing here can modify a recorded run; the one destructive
//! operation, `runs discard`, removes a *never-finalized* leftover and is
//! refused for anything else (see `scirust-studio-store`).

use scirust_studio_store::{RunStore, StoreError};

use crate::studio::{resolve_store, take_option};
use crate::ux;

/// Exit code for a usage problem.
const USAGE: u8 = 2;
/// Exit code for a storage problem.
const STORE_FAILURE: u8 = 7;

fn open_store(store_arg: Option<String>) -> Result<RunStore, u8> {
    let Some(root) = resolve_store(store_arg)
    else
    {
        eprintln!(
            "{} no run store selected: pass `--store <dir>` or set {}",
            ux::error_prefix(),
            crate::studio::STORE_ENV
        );
        return Err(USAGE);
    };
    RunStore::open(&root).map_err(|e| {
        eprintln!("{} cannot open run store `{root}`: {e}", ux::error_prefix());
        STORE_FAILURE
    })
}

fn report(e: StoreError) -> u8 {
    eprintln!("{} {e}", ux::error_prefix());
    STORE_FAILURE
}

/// `scirust runs <list|show|verify|discard> [...]`
pub fn run(args: &[String]) -> u8 {
    let (store_arg, rest) = take_option(args, "store");
    let (format, rest) = take_option(&rest, "format");
    let format = format.unwrap_or_else(|| "text".to_string());
    if format != "text" && format != "json"
    {
        eprintln!(
            "{} unknown --format `{format}` (use `text` or `json`)",
            ux::error_prefix()
        );
        return USAGE;
    }

    match rest.first().map(String::as_str)
    {
        Some("list") => list(store_arg, &format),
        Some("show") => show(store_arg, &format, rest.get(1)),
        Some("verify") => verify(store_arg, rest.get(1)),
        Some("discard") => discard(store_arg, rest.get(1)),
        Some(other) =>
        {
            eprintln!("{} unknown subcommand `{other}`", ux::error_prefix());
            usage();
            USAGE
        },
        None =>
        {
            usage();
            USAGE
        },
    }
}

fn usage() {
    eprintln!("usage: scirust runs <subcommand> [--store <dir>] [--format text|json]");
    eprintln!("  list                 every recorded run, oldest first");
    eprintln!("  show <run-id>        one run's manifest and result summary");
    eprintln!("  verify [<run-id>]    recompute stored hashes (all runs if no id)");
    eprintln!("  discard <run-id>     delete one interrupted run's leftover directory");
}

fn list(store_arg: Option<String>, format: &str) -> u8 {
    let store = match open_store(store_arg)
    {
        Ok(s) => s,
        Err(code) => return code,
    };
    let runs = match store.list()
    {
        Ok(r) => r,
        Err(e) => return report(e),
    };
    let interrupted = match store.interrupted()
    {
        Ok(r) => r,
        Err(e) => return report(e),
    };

    if format == "json"
    {
        match scirust_studio_store::manifests_to_json(&runs)
        {
            Ok(json) => println!("{json}"),
            Err(e) =>
            {
                eprintln!("{} failed to serialize runs: {e}", ux::error_prefix());
                return STORE_FAILURE;
            },
        }
        return 0;
    }

    println!("{}", ux::heading("RUNS"));
    if runs.is_empty()
    {
        println!("  {}", ux::dim("no runs recorded yet"));
    }
    for m in &runs
    {
        let status = match m.status.label()
        {
            "completed" => ux::green("completed"),
            "cancelled" => ux::yellow("cancelled"),
            other => ux::red(other),
        };
        println!(
            "  {}  {:<32} {status}",
            ux::green(&m.run_id),
            m.capability_id
        );
    }

    // Interrupted runs are surfaced here rather than hidden behind a
    // separate command: a user listing their runs is exactly who needs to
    // know that one of them never finished.
    if !interrupted.is_empty()
    {
        println!();
        println!("{}", ux::heading("INTERRUPTED"));
        for r in &interrupted
        {
            println!(
                "  {}  {}",
                ux::yellow(&r.run_id),
                ux::dim("never finished — the process died before the run was recorded")
            );
        }
    }
    0
}

fn show(store_arg: Option<String>, format: &str, run_id: Option<&String>) -> u8 {
    let Some(run_id) = run_id
    else
    {
        eprintln!("usage: scirust runs show <run-id> [--store <dir>]");
        return USAGE;
    };
    let store = match open_store(store_arg)
    {
        Ok(s) => s,
        Err(code) => return code,
    };
    let manifest = match store.load_manifest(run_id)
    {
        Ok(m) => m,
        Err(e) => return report(e),
    };

    if format == "json"
    {
        match manifest.to_json_pretty()
        {
            Ok(json) => println!("{json}"),
            Err(e) =>
            {
                eprintln!("{} failed to serialize manifest: {e}", ux::error_prefix());
                return STORE_FAILURE;
            },
        }
        return 0;
    }

    println!("{}", ux::heading(&manifest.run_id));
    println!("  capability     {}", manifest.capability_id);
    println!("  status         {}", manifest.status.label());
    if let scirust_studio_store::RunStatus::Failed { message } = &manifest.status
    {
        println!("  failure        {message}");
    }
    println!("  started        {}", manifest.started_at_rfc3339);
    println!("  finished       {}", manifest.finished_at_rfc3339);
    println!("  scenario hash  {}", manifest.scenario_sha256);
    match &manifest.result_sha256
    {
        Some(h) => println!("  result hash    {h}"),
        None => println!("  result hash    {}", ux::dim("(no result)")),
    }
    println!(
        "  target         {}-{} ({}-bit)",
        manifest.environment.os, manifest.environment.arch, manifest.environment.pointer_width
    );

    if manifest.status.has_result()
    {
        match store.load_result(run_id)
        {
            Ok(result) =>
            {
                println!();
                println!("{}", ux::heading("SERIES"));
                for s in &result.series
                {
                    println!("  {:<18} {} points, unit {}", s.id, s.values.len(), s.unit);
                }
            },
            Err(e) => return report(e),
        }
    }
    0
}

fn verify(store_arg: Option<String>, run_id: Option<&String>) -> u8 {
    let store = match open_store(store_arg)
    {
        Ok(s) => s,
        Err(code) => return code,
    };

    let ids: Vec<String> = match run_id
    {
        Some(id) => vec![id.clone()],
        None => match store.list()
        {
            Ok(runs) => runs.into_iter().map(|m| m.run_id).collect(),
            Err(e) => return report(e),
        },
    };

    if ids.is_empty()
    {
        println!("{}", ux::dim("no runs to verify"));
        return 0;
    }

    let mut failures = 0;
    for id in &ids
    {
        match store.verify(id)
        {
            Ok(()) => println!("  [{}] {id}", ux::green("OK")),
            Err(e) =>
            {
                failures += 1;
                println!("  [{}] {id}: {e}", ux::red("FAILED"));
            },
        }
    }
    if failures > 0
    {
        eprintln!(
            "{} {failures} of {} runs no longer match their recorded hashes",
            ux::error_prefix(),
            ids.len()
        );
        return STORE_FAILURE;
    }
    println!("{}", ux::dim(&format!("{} runs verified", ids.len())));
    0
}

fn discard(store_arg: Option<String>, run_id: Option<&String>) -> u8 {
    let Some(run_id) = run_id
    else
    {
        eprintln!("usage: scirust runs discard <run-id> [--store <dir>]");
        return USAGE;
    };
    let store = match open_store(store_arg)
    {
        Ok(s) => s,
        Err(code) => return code,
    };
    match store.discard_interrupted(run_id)
    {
        Ok(()) =>
        {
            println!("discarded interrupted run {run_id}");
            0
        },
        Err(e) => report(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TUTORIAL: &str =
        include_str!("../../docs/studio/tutorials/spring_mass_damper.scirust.toml");

    fn s(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    /// A temp directory that cleans itself up, without adding a dev
    /// dependency to this crate just for tests.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "scirust-runs-test-{tag}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Run the tutorial into a fresh store and return (dir, scenario path).
    fn recorded_store(tag: &str) -> (TempDir, TempDir) {
        let store = TempDir::new(&format!("{tag}-store"));
        let work = TempDir::new(&format!("{tag}-work"));
        let scenario = work.0.join("scenario.scirust.toml");
        std::fs::write(&scenario, TUTORIAL).unwrap();

        let code = crate::studio::run_scenario(&[
            scenario.to_string_lossy().into_owned(),
            "--store".to_string(),
            store.path().to_string(),
        ]);
        assert_eq!(code, 0, "the tutorial should run and be recorded");
        (store, work)
    }

    #[test]
    fn listing_an_empty_store_succeeds() {
        let dir = TempDir::new("empty");
        assert_eq!(run(&s(&["list", "--store", dir.path()])), 0);
    }

    #[test]
    fn a_run_is_recorded_and_then_listed_shown_and_verified() {
        let (store, _work) = recorded_store("roundtrip");

        assert_eq!(run(&s(&["list", "--store", store.path()])), 0);
        assert_eq!(run(&s(&["verify", "--store", store.path()])), 0);

        // Find the id the run was given, then show it.
        let opened = RunStore::open(store.path()).unwrap();
        let runs = opened.list().unwrap();
        assert_eq!(runs.len(), 1, "exactly one run was recorded");
        assert_eq!(runs[0].capability_id, "sim.mechanics.spring_mass_damper");
        assert_eq!(runs[0].status, scirust_studio_store::RunStatus::Completed);

        assert_eq!(
            run(&s(&["show", &runs[0].run_id, "--store", store.path()])),
            0
        );
        assert_eq!(
            run(&s(&[
                "show",
                &runs[0].run_id,
                "--store",
                store.path(),
                "--format",
                "json"
            ])),
            0
        );
    }

    #[test]
    fn verify_fails_when_a_stored_run_is_modified() {
        let (store, _work) = recorded_store("tamper");
        let opened = RunStore::open(store.path()).unwrap();
        let run_id = opened.list().unwrap()[0].run_id.clone();

        let result_path = opened.run_dir(&run_id).join("result.json");
        let text = std::fs::read_to_string(&result_path).unwrap();
        std::fs::write(&result_path, text.replacen("0.", "1.", 1)).unwrap();

        assert_eq!(
            run(&s(&["verify", "--store", store.path()])),
            STORE_FAILURE,
            "a modified run must not verify"
        );
    }

    #[test]
    fn a_failed_run_is_still_recorded() {
        let store = TempDir::new("failed-store");
        let work = TempDir::new("failed-work");
        let scenario = work.0.join("bad.scirust.toml");
        // Stiffness has no upper bound, so this passes both validation
        // stages and then blows RK4 up: h*sqrt(k/m) is astronomically past
        // the method's stability limit, so the state goes non-finite within
        // a few steps. That makes it a genuine *execution* failure, which
        // is the case this test is about — a validation failure never
        // reaches the store at all.
        std::fs::write(
            &scenario,
            TUTORIAL.replace(
                "stiffness = { value = 4.0, unit = \"kg/s^2\" }",
                "stiffness = { value = 1e30, unit = \"kg/s^2\" }",
            ),
        )
        .unwrap();

        let code = crate::studio::run_scenario(&[
            scenario.to_string_lossy().into_owned(),
            "--store".to_string(),
            store.path().to_string(),
        ]);
        assert_eq!(code, 5, "an integrator blow-up is a numerical failure");

        let opened = RunStore::open(store.path()).unwrap();
        let runs = opened.list().unwrap();
        assert_eq!(runs.len(), 1, "a failed run is still a run worth recording");
        match &runs[0].status
        {
            scirust_studio_store::RunStatus::Failed { message } =>
            {
                assert!(!message.is_empty(), "the failure reason must be recorded")
            },
            other => panic!("expected Failed, got {other:?}"),
        }
        assert!(
            runs[0].result_sha256.is_none(),
            "a failed run has no result to hash"
        );
        // The attempt is still fully recoverable: the scenario that failed
        // is stored, and the record verifies.
        opened.verify(&runs[0].run_id).unwrap();
    }

    /// Whether the ambient environment already selects a store.
    ///
    /// The two tests below are about the *absence* of a store, and
    /// `cargo test` runs tests as threads in one process — so clearing the
    /// variable here would race every other test that reads it. Reading it
    /// and skipping is the only safe option; mutating shared process state
    /// to make a test pass is not.
    fn environment_selects_a_store() -> bool {
        std::env::var(crate::studio::STORE_ENV)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    }

    #[test]
    fn without_a_store_nothing_is_recorded_and_the_run_still_succeeds() {
        if environment_selects_a_store()
        {
            return;
        }
        let work = TempDir::new("nostore");
        let scenario = work.0.join("scenario.scirust.toml");
        std::fs::write(&scenario, TUTORIAL).unwrap();
        let code = crate::studio::run_scenario(&[scenario.to_string_lossy().into_owned()]);
        assert_eq!(code, 0, "a run without a store still succeeds");
    }

    #[test]
    fn runs_without_a_store_is_a_usage_error() {
        if environment_selects_a_store()
        {
            return;
        }
        assert_eq!(run(&s(&["list"])), USAGE);
    }

    /// The precedence rule itself, tested as the pure function it is —
    /// no environment mutation required.
    #[test]
    fn an_explicit_store_argument_wins_over_the_environment() {
        assert_eq!(
            crate::studio::resolve_store(Some("/explicit/path".to_string())).as_deref(),
            Some("/explicit/path")
        );
    }

    #[test]
    fn an_unknown_subcommand_is_a_usage_error() {
        let dir = TempDir::new("unknown");
        assert_eq!(run(&s(&["frobnicate", "--store", dir.path()])), USAGE);
        assert_eq!(run(&s(&["--store", dir.path()])), USAGE);
    }

    #[test]
    fn an_unknown_format_is_a_usage_error() {
        let dir = TempDir::new("badformat");
        assert_eq!(
            run(&s(&["list", "--store", dir.path(), "--format", "yaml"])),
            USAGE
        );
    }

    #[test]
    fn showing_an_unknown_run_is_a_store_failure() {
        let dir = TempDir::new("missing");
        assert_eq!(
            run(&s(&["show", "no-such-run", "--store", dir.path()])),
            STORE_FAILURE
        );
        assert_eq!(run(&s(&["show", "--store", dir.path()])), USAGE);
    }

    #[test]
    fn discarding_a_finalized_run_is_refused() {
        let (store, _work) = recorded_store("discard");
        let opened = RunStore::open(store.path()).unwrap();
        let run_id = opened.list().unwrap()[0].run_id.clone();

        assert_eq!(
            run(&s(&["discard", &run_id, "--store", store.path()])),
            STORE_FAILURE,
            "a finalized run is not an interrupted one"
        );
        assert!(
            opened.load_manifest(&run_id).is_ok(),
            "the run must survive a refused discard"
        );
    }
}
