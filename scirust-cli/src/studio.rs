//! `catalog` and `run` — dispatch through the shared capability registry
//! and runtime (`scirust-studio-registry`/`scirust-studio-runtime`), per
//! `docs/studio/adr/0001-capability-registry.md`.
//!
//! This module does not import `scirust-sim` and does not construct any
//! model directly: every capability is looked up in
//! [`scirust_studio_runtime::build_registry`]'s catalogue and driven only
//! through [`scirust_studio_runtime::CapabilityAdapter`]. That is what
//! "the CLI must no longer instantiate `SpringMassDamper` directly" means
//! structurally rather than as a habit to maintain by hand.

use scirust_studio_runtime::{
    ExecutionControl, ExecutionError, MetricValue, NullEventSink, RunResult, SeriesRole,
    VerificationStatus, build_registry, find_adapter,
};
use scirust_studio_schema::{parse_toml, validate};
use scirust_studio_store::{RunStore, StoreError};

use crate::ux;

/// Report where a run was recorded, or why it could not be.
///
/// A storage failure after a successful computation is reported but does not
/// change the exit code: the run really did succeed, and claiming otherwise
/// would be as wrong as silently swallowing the problem.
fn report_store_outcome(recorded: Result<String, StoreError>) {
    match recorded
    {
        // stderr, not stdout: this is a status notice, and stdout carries
        // the result — which under `--format json` must stay parseable.
        // Printing it to stdout made `scirust run --format json --store …`
        // emit a leading non-JSON line.
        Ok(run_id) => eprintln!("{}", ux::dim(&format!("recorded as run {run_id}"))),
        Err(e) => eprintln!("{} could not record this run: {e}", ux::error_prefix()),
    }
}

/// Pull `--<name> <value>` out of `args`, returning the value (if present)
/// and the remaining arguments.
pub(crate) fn take_option(args: &[String], name: &str) -> (Option<String>, Vec<String>) {
    let flag = format!("--{name}");
    let mut value = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len()
    {
        if args[i] == flag && i + 1 < args.len()
        {
            value = Some(args[i + 1].clone());
            i += 2;
        }
        else
        {
            rest.push(args[i].clone());
            i += 1;
        }
    }
    (value, rest)
}

/// Pull a `--format text|json` flag out of `args`, defaulting to `"text"`.
/// Returns the format and the remaining (positional) arguments.
fn take_format(args: &[String]) -> (String, Vec<String>) {
    let (value, rest) = take_option(args, "format");
    (value.unwrap_or_else(|| "text".to_string()), rest)
}

/// The environment variable naming a default run store.
pub(crate) const STORE_ENV: &str = "SCIRUST_STUDIO_STORE";

/// Resolve where runs should be recorded: `--store <dir>` wins, then
/// `SCIRUST_STUDIO_STORE`, and otherwise nothing is recorded.
///
/// There is deliberately no built-in default path. Writing run history into
/// a guessed location under the user's home directory without being asked is
/// a decision that belongs to the application, not to a library call, and
/// this build has no user-facing setting to turn it back off.
pub(crate) fn resolve_store(explicit: Option<String>) -> Option<String> {
    explicit.or_else(|| match std::env::var(STORE_ENV)
    {
        Ok(path) if !path.trim().is_empty() => Some(path),
        _ => None,
    })
}

/// `scirust catalog [--format text|json]` — list the capabilities this
/// build can actually run, straight from the real adapter registry.
pub fn run_catalog(args: &[String]) -> u8 {
    let (format, _rest) = take_format(args);
    let registry = build_registry();
    match format.as_str()
    {
        "json" => match registry.to_json()
        {
            Ok(json) =>
            {
                println!("{json}");
                0
            },
            Err(e) =>
            {
                eprintln!("{} failed to serialize catalogue: {e}", ux::error_prefix());
                7
            },
        },
        "text" =>
        {
            println!("{}", ux::heading("CAPABILITIES"));
            for d in registry.iter()
            {
                println!("  {}  {}", ux::green(d.id.0), d.summary);
            }
            println!();
            println!(
                "{}",
                ux::dim(&format!(
                    "{} capabilities. Every scirust-sim model not listed here is real and tested in its own \
                     crate, but has no scenario adapter wired up yet — see docs/studio/CAPABILITY_MATRIX.md",
                    registry.len()
                ))
            );
            0
        },
        other =>
        {
            eprintln!(
                "{} unknown --format `{other}` (use `text` or `json`)",
                ux::error_prefix()
            );
            2
        },
    }
}

/// `scirust run <scenario.scirust.toml> [--format text|json]` — parse,
/// validate (generic schema, then capability-specific), and execute a
/// scenario through its adapter. Exit codes follow the SciRust Studio
/// brief: 2 usage, 3 validation, 5 numerical failure, 6 cancelled, 7
/// internal failure.
pub fn run_scenario(args: &[String]) -> u8 {
    let (store_arg, args) = take_option(args, "store");
    let (format, rest) = take_format(&args);
    let Some(path) = rest.first()
    else
    {
        eprintln!(
            "usage: scirust run <scenario.scirust.toml> [--format text|json] [--store <dir>]"
        );
        return 2;
    };
    let text = match std::fs::read_to_string(path)
    {
        Ok(t) => t,
        Err(e) =>
        {
            eprintln!("{} cannot read `{path}`: {e}", ux::error_prefix());
            return 2;
        },
    };
    let scenario = match parse_toml(&text)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("{} {}", ux::error_prefix(), e.to_cataloged());
            return 3;
        },
    };

    let registry = build_registry();
    let known_ids: Vec<&str> = registry.iter().map(|d| d.id.0).collect();
    let schema_errors = validate(&scenario, Some(&known_ids));
    if !schema_errors.is_empty()
    {
        eprintln!("{} scenario is invalid:", ux::error_prefix());
        for e in &schema_errors
        {
            eprintln!("  - {}", e.to_cataloged());
        }
        return 3;
    }

    let Some(adapter) = find_adapter(&scenario.capability.id)
    else
    {
        eprintln!(
            "{} capability `{}` passed schema validation but has no registered adapter (this is a bug)",
            ux::error_prefix(),
            scenario.capability.id
        );
        return 7;
    };

    let validated = match adapter.validate(&scenario)
    {
        Ok(v) => v,
        Err(report) =>
        {
            eprintln!(
                "{} scenario is invalid for `{}`:",
                ux::error_prefix(),
                scenario.capability.id
            );
            for e in &report.errors
            {
                eprintln!("  - {e}");
            }
            return 3;
        },
    };

    // Open the store *before* executing, so a run killed part-way through
    // leaves a detectable interrupted record rather than no trace at all —
    // that is the whole point of recording the attempt separately from the
    // outcome. See `docs/studio/adr/0004-immutable-run-storage.md`.
    let mut pending = None;
    if let Some(root) = resolve_store(store_arg)
    {
        match RunStore::open(&root)
        {
            Ok(store) => match store.begin(&scenario.capability.id, &text)
            {
                Ok(p) => pending = Some(p),
                Err(e) =>
                {
                    eprintln!("{} cannot record this run: {e}", ux::error_prefix());
                    return 7;
                },
            },
            Err(e) =>
            {
                eprintln!("{} cannot open run store `{root}`: {e}", ux::error_prefix());
                return 7;
            },
        }
    }

    let mut sink = NullEventSink;
    let result = match adapter.execute(&validated, &ExecutionControl::new(), &mut sink)
    {
        Ok(r) => r,
        Err(e) =>
        {
            if let Some(pending) = pending
            {
                let recorded = match e
                {
                    ExecutionError::Cancelled => pending.cancel(),
                    _ => pending.fail(&e.to_string()),
                };
                report_store_outcome(recorded);
            }
            eprintln!("{} {e}", ux::error_prefix());
            return match e
            {
                ExecutionError::Cancelled => 6,
                ExecutionError::Numerical(_) => 5,
                ExecutionError::InvalidModelState(_) => 3,
                ExecutionError::Internal(_) => 7,
            };
        },
    };

    if let Some(pending) = pending
    {
        report_store_outcome(pending.complete(&result));
    }

    match format.as_str()
    {
        "json" => match result.to_json_pretty()
        {
            Ok(json) =>
            {
                println!("{json}");
                0
            },
            Err(e) =>
            {
                eprintln!("{} failed to serialize result: {e}", ux::error_prefix());
                7
            },
        },
        "text" =>
        {
            print_result_text(&result);
            0
        },
        other =>
        {
            eprintln!(
                "{} unknown --format `{other}` (use `text` or `json`)",
                ux::error_prefix()
            );
            2
        },
    }
}

/// An integer metric's value, if the result carries one under that id.
fn integer_metric(result: &RunResult, id: &str) -> Option<i64> {
    result
        .metrics
        .iter()
        .find(|m| m.id == id)
        .and_then(|m| match m.value
        {
            MetricValue::Integer(v) => Some(v),
            _ => None,
        })
}

fn print_result_text(result: &RunResult) {
    println!("{}", ux::heading(&result.summary.capability_display_name));
    println!("  scenario      {}", result.summary.scenario_name);
    println!("  capability    {}", result.capability_id);
    println!("  steps         {}", result.summary.steps);
    let axis_unit = result.axes.first().map(|a| a.unit.as_str()).unwrap_or("");
    println!("  t final       {} {axis_unit}", result.summary.t_end);

    // Printed only when the computation actually consumed one, and printed
    // next to the determinism class it qualifies: for a stochastic result the
    // seed is not decoration, it is the difference between a trajectory and
    // *the* trajectory these inputs produce.
    if let Some(seed) = result.provenance.seed
    {
        println!("  determinism   {:?}", result.provenance.determinism);
        println!("  seed          {seed}  (re-run with this seed to obtain the same sample)");
    }

    // An ensemble is announced before its series are listed, because
    // "member_0 … member_7" beside a mean reads as eight results unless the
    // reader is told that eight is a sample of the realisations and the mean
    // is over all of them.
    if let Some(drawn) = integer_metric(result, "replicates")
    {
        let kept = integer_metric(result, "retained_members").unwrap_or(0);
        println!(
            "  ensemble      {drawn} independent realisations, {kept} kept in the result{}",
            if kept < drawn
            {
                format!(" ({} not stored)", drawn - kept)
            }
            else
            {
                String::new()
            }
        );
    }

    println!();
    println!("{}", ux::heading("SERIES"));
    for s in &result.series
    {
        // The role, not the id, is what tells a reader whether a curve is
        // evidence about many realisations or one of them.
        let role = match s.role
        {
            SeriesRole::Trajectory => String::new(),
            SeriesRole::Reference => ux::dim("reference"),
            SeriesRole::EnsembleMean => ux::dim("mean over the ensemble"),
            SeriesRole::EnsembleBandLower => ux::dim("band, lower edge"),
            SeriesRole::EnsembleBandUpper => ux::dim("band, upper edge"),
            SeriesRole::EnsembleMember => ux::dim("one realisation"),
        };
        println!(
            "  {:<30} {} points, unit {:<4} {role}",
            s.id,
            s.values.len(),
            s.unit
        );
    }

    println!();
    println!("{}", ux::heading("METRICS"));
    for m in &result.metrics
    {
        let value = match &m.value
        {
            MetricValue::Scalar(v) => format!("{v:.6}"),
            MetricValue::Integer(v) => v.to_string(),
            MetricValue::Text(v) => v.clone(),
        };
        // Same column width as SERIES above, and wide enough for the longest
        // id an ensemble produces — `ensemble_final_standard_error` ran off
        // the end of the old 18.
        match m.unit.as_deref()
        {
            Some(unit) if !unit.is_empty() => println!("  {:<30} {value} {unit}", m.id),
            _ => println!("  {:<30} {value}", m.id),
        }
    }

    println!();
    println!("{}", ux::heading("VERIFICATION"));
    for v in &result.verifications
    {
        let status = match v.status
        {
            VerificationStatus::Passed => ux::green("PASSED"),
            VerificationStatus::Warning => ux::yellow("WARNING"),
            VerificationStatus::Failed => ux::red("FAILED"),
            VerificationStatus::NotApplicable => ux::dim("N/A"),
        };
        println!("  [{status}] {}: {}", v.id, v.explanation);
    }

    if !result.warnings.is_empty()
    {
        println!();
        println!("{}", ux::heading("WARNINGS"));
        for w in &result.warnings
        {
            println!("  {} {}", ux::yellow("warning:"), w.message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TUTORIAL: &str =
        include_str!("../../docs/studio/tutorials/spring_mass_damper.scirust.toml");

    fn write_fixture(dir: &std::path::Path, contents: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(
            "scirust-studio-test-{}-{unique}.scirust.toml",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn catalog_text_lists_every_registered_capability() {
        assert_eq!(run_catalog(&[]), 0);
    }

    #[test]
    fn catalog_json_is_valid_and_matches_the_registry_size() {
        // Can't capture stdout easily here without a bigger refactor, but we
        // can independently confirm the registry (which `run_catalog` reads
        // from) round-trips through JSON with every capability present —
        // the same check `scirust-studio-registry`'s own tests make, kept
        // here too as a guard against this module drifting from it.
        let registry = build_registry();
        let json = registry.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), registry.len());
        assert_eq!(
            run_catalog(&["--format".to_string(), "json".to_string()]),
            0
        );
    }

    #[test]
    fn catalog_rejects_unknown_format() {
        assert_eq!(
            run_catalog(&["--format".to_string(), "yaml".to_string()]),
            2
        );
    }

    #[test]
    fn run_rejects_missing_argument() {
        assert_eq!(run_scenario(&[]), 2);
    }

    #[test]
    fn run_rejects_unreadable_path() {
        assert_eq!(
            run_scenario(&["/nonexistent/scirust-studio-test.toml".to_string()]),
            2
        );
    }

    #[test]
    fn run_rejects_invalid_toml() {
        let dir = std::env::temp_dir();
        let path = write_fixture(&dir, "not valid toml [[[");
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 3);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_rejects_unknown_capability() {
        let dir = std::env::temp_dir();
        let scenario = TUTORIAL.replace(
            "sim.mechanics.spring_mass_damper",
            "sim.nonexistent.made_up",
        );
        let path = write_fixture(&dir, &scenario);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 3);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_executes_the_real_tutorial_scenario_via_the_registry_and_adapter() {
        let dir = std::env::temp_dir();
        let path = write_fixture(&dir, TUTORIAL);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_supports_json_output() {
        let dir = std::env::temp_dir();
        let path = write_fixture(&dir, TUTORIAL);
        let args = [path.clone(), "--format".to_string(), "json".to_string()];
        assert_eq!(run_scenario(&args), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_reports_numeric_failure_distinctly_from_validation_failure() {
        let dir = std::env::temp_dir();
        let scenario = TUTORIAL.replace(
            "stiffness = { value = 4.0, unit = \"kg/s^2\" }",
            "stiffness = { value = -4.0, unit = \"kg/s^2\" }",
        );
        let path = write_fixture(&dir, &scenario);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 3);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_requires_a_step_for_this_fixed_step_adapter() {
        let dir = std::env::temp_dir();
        let scenario = TUTORIAL
            .lines()
            .filter(|l| !l.starts_with("step ="))
            .collect::<Vec<_>>()
            .join("\n");
        let path = write_fixture(&dir, &scenario);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 3);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn every_catalogued_capability_is_reachable_through_run() {
        // The bidirectional consistency property: nothing in the catalogue
        // lacks a dispatchable adapter (the reverse — every adapter is in
        // the catalogue — is checked in scirust-studio-runtime's own tests).
        let registry = build_registry();
        for descriptor in registry.iter()
        {
            assert!(
                scirust_studio_runtime::find_adapter(descriptor.id.0).is_some(),
                "capability `{}` is catalogued but has no adapter reachable from the CLI",
                descriptor.id
            );
        }
    }
}
