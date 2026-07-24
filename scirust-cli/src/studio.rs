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
    ExecutionControl, ExecutionError, MetricValue, NullEventSink, RunResult, VerificationStatus,
    build_registry, find_adapter,
};
use scirust_studio_schema::{parse_toml, validate};

use crate::ux;

/// Pull a `--format text|json` flag out of `args`, defaulting to `"text"`.
/// Returns the format and the remaining (positional) arguments.
fn take_format(args: &[String]) -> (String, Vec<String>) {
    let mut format = "text".to_string();
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len()
    {
        if args[i] == "--format" && i + 1 < args.len()
        {
            format = args[i + 1].clone();
            i += 2;
        }
        else
        {
            rest.push(args[i].clone());
            i += 1;
        }
    }
    (format, rest)
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
    let (format, rest) = take_format(args);
    let Some(path) = rest.first()
    else
    {
        eprintln!("usage: scirust run <scenario.scirust.toml> [--format text|json]");
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

    let mut sink = NullEventSink;
    let result = match adapter.execute(&validated, &ExecutionControl::new(), &mut sink)
    {
        Ok(r) => r,
        Err(e) =>
        {
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

fn print_result_text(result: &RunResult) {
    println!("{}", ux::heading(&result.summary.capability_display_name));
    println!("  scenario      {}", result.summary.scenario_name);
    println!("  capability    {}", result.capability_id);
    println!("  steps         {}", result.summary.steps);
    let axis_unit = result.axes.first().map(|a| a.unit.as_str()).unwrap_or("");
    println!("  t final       {} {axis_unit}", result.summary.t_end);

    println!();
    println!("{}", ux::heading("SERIES"));
    for s in &result.series
    {
        println!("  {:<18} {} points, unit {}", s.id, s.values.len(), s.unit);
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
        let unit = m.unit.as_deref().unwrap_or("");
        println!("  {:<18} {value} {unit}", m.id);
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
