//! `catalog` and `run` — the first two SciRust Studio commands, backed by
//! `scirust-studio-schema`/`scirust-studio-command` and a real `scirust-sim`
//! model, per `docs/studio/adr/0000-scope-and-sequencing.md`.
//!
//! Phase 1 of the SciRust Studio effort adds exactly one real, tested,
//! end-to-end capability adapter (`sim.mechanics.spring_mass_damper`) rather
//! than a shallow one for many — see `docs/studio/REPOSITORY_AUDIT.md`.
//! Every other `scirust-sim` model is real and oracle-tested in its own
//! crate but is not yet wired to a scenario here; adding one is Phase 3 work,
//! not a rename of this module.

use scirust_sim::mechanics::SpringMassDamper;
use scirust_sim::simulate;
use scirust_studio_schema::{Scenario, ValueWithUnit, parse_toml, validate};

use crate::ux;

/// Capability ids this build actually knows how to run. `catalog` lists
/// exactly this set; `run` rejects anything else during validation rather
/// than falling through to a confusing "no adapter" error later.
const KNOWN_CAPABILITY_IDS: &[&str] = &["sim.mechanics.spring_mass_damper"];

/// `scirust catalog` — list the capabilities this build can actually run.
pub fn run_catalog(_args: &[String]) -> u8 {
    println!("{}", ux::heading("CAPABILITIES"));
    println!(
        "  {}  mass on a linear spring with viscous damping (scirust_sim::mechanics::SpringMassDamper)",
        ux::green("sim.mechanics.spring_mass_damper")
    );
    println!();
    println!(
        "{}",
        ux::dim(
            "every other scirust-sim model is real and tested in its own crate, but has no \
             scenario adapter wired up yet — see docs/studio/REPOSITORY_AUDIT.md"
        )
    );
    0
}

/// `scirust run <scenario.scirust.toml>` — parse, validate, and execute a
/// scenario. Exit codes follow the SciRust Studio brief: 2 usage, 3
/// validation, 5 numerical failure.
pub fn run_scenario(args: &[String]) -> u8 {
    let Some(path) = args.first()
    else
    {
        eprintln!("usage: scirust run <scenario.scirust.toml>");
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
    let errors = validate(&scenario, Some(KNOWN_CAPABILITY_IDS));
    if !errors.is_empty()
    {
        eprintln!("{} scenario is invalid:", ux::error_prefix());
        for e in &errors
        {
            eprintln!("  - {}", e.to_cataloged());
        }
        return 3;
    }
    match scenario.capability.id.as_str()
    {
        "sim.mechanics.spring_mass_damper" => run_spring_mass_damper(&scenario),
        other =>
        {
            eprintln!(
                "{} capability `{other}` passed validation but has no adapter registered (this is a bug)",
                ux::error_prefix()
            );
            7
        },
    }
}

/// Read a named `model.*` parameter as an SI-coherent scalar.
fn model_scalar(scenario: &Scenario, name: &str) -> Result<f64, u8> {
    let value = scenario.model.get(name).ok_or_else(|| {
        eprintln!(
            "{} missing required model parameter `model.{name}`",
            ux::error_prefix()
        );
        3
    })?;
    quantity_value(value, &format!("model.{name}"))
}

/// Read a named `initial_state.*` component's single scalar (this adapter
/// only accepts one-dimensional state components).
fn state_scalar(scenario: &Scenario, name: &str) -> Result<f64, u8> {
    let components = scenario.initial_state.get(name).ok_or_else(|| {
        eprintln!(
            "{} missing required initial_state component `initial_state.{name}`",
            ux::error_prefix()
        );
        3
    })?;
    let [value]: &[ValueWithUnit; 1] = components.as_slice().try_into().map_err(|_| {
        eprintln!(
            "{} initial_state.{name} must have exactly one component for this capability, got {}",
            ux::error_prefix(),
            components.len()
        );
        3
    })?;
    quantity_value(value, &format!("initial_state.{name}[0]"))
}

fn quantity_value(value: &ValueWithUnit, field: &str) -> Result<f64, u8> {
    value.to_quantity(field).map(|q| q.value).map_err(|e| {
        eprintln!("{} {}", ux::error_prefix(), e.to_cataloged());
        3
    })
}

fn run_spring_mass_damper(scenario: &Scenario) -> u8 {
    match run_spring_mass_damper_checked(scenario)
    {
        Ok(()) => 0,
        Err(code) => code,
    }
}

fn run_spring_mass_damper_checked(scenario: &Scenario) -> Result<(), u8> {
    let mass = model_scalar(scenario, "mass")?;
    let damping = model_scalar(scenario, "damping")?;
    let stiffness = model_scalar(scenario, "stiffness")?;
    let x0 = state_scalar(scenario, "position")?;
    let v0 = state_scalar(scenario, "velocity")?;
    let model = SpringMassDamper::new(mass, damping, stiffness).map_err(|e| {
        eprintln!("{} {e}", ux::error_prefix());
        3
    })?;

    let t0 = quantity_value(&scenario.solver.start, "solver.start")?;
    let t1 = quantity_value(&scenario.solver.end, "solver.end")?;
    let Some(step) = &scenario.solver.step
    else
    {
        eprintln!(
            "{} this build's spring_mass_damper adapter requires solver.step (fixed-step RK4 only; \
             adaptive stepping is future work)",
            ux::error_prefix()
        );
        return Err(3);
    };
    let step = quantity_value(step, "solver.step")?;

    let energy0 = model.energy(&[x0, v0]);
    let traj = simulate(&model, &[x0, v0], t0, t1, step).map_err(|e| {
        eprintln!("{} simulation failed: {e}", ux::error_prefix());
        5
    })?;
    let last = traj
        .last_state()
        .expect("simulate always returns at least the initial state");
    let energy1 = model.energy(last);

    println!("{}", ux::heading(&scenario.experiment.name));
    println!("  capability    {}", scenario.capability.id);
    println!("  steps         {}", traj.len() - 1);
    println!("  t final       {} s", traj.last_time().unwrap_or(t1));
    println!("  position      {:.6} m", last[0]);
    println!("  velocity      {:.6} m/s", last[1]);
    if let (Some(e0), Some(e1)) = (energy0, energy1)
    {
        println!("  energy(t0)    {e0:.6} J");
        println!("  energy(tend)  {e1:.6} J");
        if damping == 0.0
        {
            let drift = (e1 - e0).abs() / e0.abs().max(1e-300);
            println!(
                "  {}",
                ux::dim(&format!(
                    "undamped: relative energy drift {drift:.3e} (RK4, not exactly conserved)"
                ))
            );
        }
        else
        {
            println!(
                "  {}",
                ux::dim("damping > 0: energy is expected to decay, not conserve")
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The actual shipped tutorial scenario (`docs/studio/tutorials/`),
    /// executed here rather than duplicated, so the file a user is told to
    /// run is the file that is tested.
    const SCENARIO: &str =
        include_str!("../../docs/studio/tutorials/spring_mass_damper.scirust.toml");

    /// Write `contents` to a scratch file with a name unique to this call, so
    /// concurrently-running tests (the default `cargo test` behaviour) never
    /// share a path and race on each other's writes/deletes.
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
    fn catalog_lists_the_known_capability() {
        assert_eq!(run_catalog(&[]), 0);
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
        let scenario = SCENARIO.replace(
            "sim.mechanics.spring_mass_damper",
            "sim.nonexistent.made_up",
        );
        let path = write_fixture(&dir, &scenario);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 3);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_executes_the_real_spring_mass_damper_model() {
        let dir = std::env::temp_dir();
        let path = write_fixture(&dir, SCENARIO);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn run_reports_numeric_failure_distinctly_from_validation_failure() {
        // A negative stiffness passes schema validation (units resolve fine)
        // but SpringMassDamper::new rejects it, and the adapter must map
        // that to exit code 3 (bad model parameter), not crash.
        let dir = std::env::temp_dir();
        let scenario = SCENARIO.replace(
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
        let scenario = SCENARIO
            .lines()
            .filter(|l| !l.starts_with("step ="))
            .collect::<Vec<_>>()
            .join("\n");
        let path = write_fixture(&dir, &scenario);
        assert_eq!(run_scenario(std::slice::from_ref(&path)), 3);
        std::fs::remove_file(path).unwrap();
    }
}
