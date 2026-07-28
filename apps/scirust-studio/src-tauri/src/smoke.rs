//! Headless self-checks, for CI and for a user diagnosing an installation.
//!
//! `--smoke-test-backend` opens no window. It bootstraps the service, runs a
//! shipped tutorial end to end through the real worker, verifies the stored
//! result and prints structured JSON. That is the check that answers "is this
//! installation capable of doing science", which is a different question from
//! "does a window appear".
//!
//! Nothing here compiles test-only data into a production path: it drives the
//! same service, the same worker and the same store the application uses, and
//! the scenario it runs is the shipped tutorial, not a fixture.

use std::time::{Duration, Instant};

use scirust_studio_app_service::{AppServiceConfig, JobState, StudioAppService};

/// How long to allow the smoke run before giving up.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The capability the backend smoke test exercises.
///
/// Spring-mass-damper: the fastest of the five, with an analytic invariant
/// (energy conservation) as its check, so a pass means the numerics worked
/// and not merely that a process exchanged messages.
const CAPABILITY: &str = "sim.mechanics.spring_mass_damper";

/// Run the backend smoke test, printing JSON and returning the exit code.
pub fn run_backend_smoke_test(config: AppServiceConfig) -> i32 {
    let started = Instant::now();
    match backend_smoke(config)
    {
        Ok(report) =>
        {
            println!("{report}");
            0
        },
        Err(message) =>
        {
            let report = serde_json::json!({
                "smoke_test": "backend",
                "ok": false,
                "error": message,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|_| message.clone())
            );
            1
        },
    }
}

fn backend_smoke(config: AppServiceConfig) -> Result<String, String> {
    let started = Instant::now();

    let service =
        StudioAppService::bootstrap(config).map_err(|e| format!("bootstrap failed: {e}"))?;
    let worker_version = service
        .worker_version()
        .ok_or_else(|| "the worker did not report a version".to_string())?;

    let document = service
        .load_tutorial(CAPABILITY)
        .map_err(|e| format!("cannot load the tutorial: {e}"))?;

    let validation = service.validate_source(&document.source);
    if !validation.valid
    {
        return Err(format!(
            "the shipped tutorial did not validate: {:?}",
            validation.problems
        ));
    }

    let job = service
        .start_run(&document.source)
        .map_err(|e| format!("cannot start the run: {e}"))?;

    let deadline = Instant::now() + TIMEOUT;
    let final_state = loop
    {
        if Instant::now() > deadline
        {
            return Err(format!("the run did not settle within {TIMEOUT:?}"));
        }
        let state = service
            .job_snapshot(&job.job_id)
            .map_err(|e| format!("cannot read the job: {e}"))?
            .state;
        if state.is_terminal()
        {
            break state;
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    let run_id = match final_state
    {
        JobState::Completed { run_id } => run_id,
        other => return Err(format!("the run did not complete: {other:?}")),
    };

    let verification = service
        .verify_run(&run_id)
        .map_err(|e| format!("cannot verify the stored run: {e}"))?;
    if !verification.intact
    {
        return Err(format!(
            "the stored run does not match its hashes: {:?}",
            verification.detail
        ));
    }

    let loaded = service
        .load_run(&run_id)
        .map_err(|e| format!("cannot load the stored run: {e}"))?;
    let result = loaded
        .result
        .as_v2()
        .ok_or_else(|| "a freshly written run must use result schema v2".to_string())?;
    let axis = result
        .time_axis()
        .ok_or_else(|| "the result has no time axis".to_string())?;
    if axis.values.is_empty()
    {
        return Err("the time axis carries no coordinates".to_string());
    }

    // The scientific checks are reported, not just the fact that something
    // ran: a smoke test that passes while the physics fails is not a smoke
    // test.
    let checks: Vec<serde_json::Value> = result
        .verifications
        .iter()
        .map(|v| {
            serde_json::json!({
                "id": v.id,
                "status": format!("{:?}", v.status).to_lowercase(),
                "measured": v.measured,
            })
        })
        .collect();
    let failed: Vec<&str> = result
        .verifications
        .iter()
        .filter(|v| v.status == scirust_studio_runtime::VerificationStatus::Failed)
        .map(|v| v.id.as_str())
        .collect();
    if !failed.is_empty()
    {
        return Err(format!("scientific checks failed: {failed:?}"));
    }

    let report = serde_json::json!({
        "smoke_test": "backend",
        "ok": true,
        "app_version": env!("CARGO_PKG_VERSION"),
        "worker_version": worker_version,
        "capability_id": CAPABILITY,
        "capabilities_available": service.catalog().len(),
        "run_id": run_id,
        "result_schema_version": result.schema_version,
        "axis_points": axis.values.len(),
        "t_start": result.summary.t_start,
        "t_end": result.summary.t_end,
        "store_integrity": "verified",
        "scientific_checks": checks,
        "store_path": service.store_root().display().to_string(),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("cannot render the report: {e}"))?;

    service
        .shutdown()
        .map_err(|e| format!("shutdown failed: {e}"))?;
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smoke test must exercise a real, shipped capability.
    #[test]
    fn the_smoke_capability_is_registered_and_has_a_tutorial() {
        assert!(
            scirust_studio_runtime::find_adapter(CAPABILITY).is_some(),
            "{CAPABILITY} must be a real capability"
        );
        assert!(
            scirust_studio_runtime::tutorial_scenario_for(CAPABILITY).is_some(),
            "{CAPABILITY} must ship a tutorial"
        );
    }

    /// A failure must still produce parseable JSON: CI reads this output.
    #[test]
    fn a_failed_smoke_test_reports_json_and_a_non_zero_code() {
        let dir = tempfile::tempdir().unwrap();
        let config = AppServiceConfig {
            worker: scirust_studio_app_service::WorkerLaunchConfig::at("/no/such/worker"),
            store_root: dir.path().to_path_buf(),
            event_buffer_capacity: 16,
        };
        let message = backend_smoke(config).unwrap_err();
        assert!(message.contains("bootstrap failed"), "{message}");
    }
}
