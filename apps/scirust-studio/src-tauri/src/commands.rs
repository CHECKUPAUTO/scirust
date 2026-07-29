//! The Tauri commands the frontend may call.
//!
//! Every command is narrow and typed. There is deliberately **no**
//! `read_file(path)`, `write_file(path, bytes)`, `spawn(program)`,
//! `shell(command)`, `http(url)` or `env(name)` — a webview that can ask for
//! any of those is a webview that can do anything the user can, and the
//! frontend needs none of them.
//!
//! **The file dialogs are native and run here.** `studio_open_scenario` and
//! `studio_save_scenario` show a real picker on this thread and exchange the
//! file's *contents* with the frontend — never a path. The frontend supplies
//! no destination and receives no location it could name again later, so it
//! cannot re-read a file the user picked once, and cannot reach a file the
//! user never picked at all. `dialog:*` and `fs:*` stay ungranted in the
//! capability file for exactly this reason: they would let the webview go
//! around these two commands. `tests/security_audit.rs` asserts both.
//!
//! The distinction is the whole argument. A command taking a path would be
//! `read_file(path)` with a friendlier name, and "the file the user just
//! picked" is not a smaller permission than "any file" once the frontend
//! holds the path.
//!
//! Each command delegates to `scirust-studio-app-service` and returns a view
//! type. None of them contains scientific logic.

use tauri::State;

use scirust_studio_app_service::StudioAppService;

use crate::state::AppState;
use crate::views::{
    BootstrapView, CapabilityView, ErrorView, JobView, RunView, ScenarioFileView, ScenarioView,
    StoredRunView, VerificationReportView, run_view,
};

/// The result every command returns: a view, or a renderable error.
///
/// The error is boxed. An `ErrorView` carries a full validation outcome when
/// it is a validation failure, which makes it several times larger than most
/// of the success values here — and an unboxed error would make every
/// command's `Result` that size whether or not anything failed.
type CommandResult<T> = Result<T, Box<ErrorView>>;

/// Wrap a service error for a command's return type.
fn error(e: scirust_studio_app_service::AppServiceError) -> Box<ErrorView> {
    Box::new(ErrorView::from(e))
}

/// Reject an identifier that is not one this application issued.
///
/// Ids come back from the frontend, which is the least trusted input this
/// process handles. Bounding the length and the alphabet costs nothing and
/// removes a whole class of "what if it contains a path separator" question
/// from every downstream use.
fn check_id(kind: &str, value: &str) -> Result<(), Box<ErrorView>> {
    const MAX: usize = 128;
    let ok = !value.is_empty()
        && value.len() <= MAX
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if ok
    {
        Ok(())
    }
    else
    {
        Err(Box::new(ErrorView {
            code: "INVALID_IDENTIFIER".to_string(),
            title: format!("Invalid {kind}"),
            explanation: format!(
                "`{}` is not a well-formed {kind}.",
                value.chars().take(64).collect::<String>()
            ),
            recoverable: true,
            suggested_action: None,
            validation: None,
        }))
    }
}

/// The largest scenario source this shell will accept, from the frontend or
/// from a file the user picks. A scenario is a short configuration file; the
/// bound exists so neither route can hand this process an arbitrarily large
/// allocation.
const MAX_SOURCE_BYTES: usize = 1024 * 1024;

/// Reject scenario source that is implausibly large.
///
/// The bound is generous — a scenario is a short configuration file — and
/// exists so a runaway frontend cannot hand this process an arbitrarily
/// large allocation.
fn check_source(source: &str) -> Result<(), Box<ErrorView>> {
    if source.len() <= MAX_SOURCE_BYTES
    {
        Ok(())
    }
    else
    {
        Err(Box::new(ErrorView {
            code: "SCENARIO_TOO_LARGE".to_string(),
            title: "Scenario too large".to_string(),
            explanation: format!(
                "A scenario file may be at most {MAX_SOURCE_BYTES} bytes; this one is {}.",
                source.len()
            ),
            recoverable: true,
            suggested_action: Some("Open a smaller scenario file.".to_string()),
            validation: None,
        }))
    }
}

fn service(state: &AppState) -> Result<&StudioAppService, Box<ErrorView>> {
    state.service().map_err(error)
}

/// What the interface needs at start-up.
#[tauri::command]
pub fn studio_bootstrap(state: State<'_, AppState>) -> CommandResult<BootstrapView> {
    Ok(state.bootstrap_view())
}

/// The capability catalogue, from the real registry.
#[tauri::command]
pub fn studio_catalog(state: State<'_, AppState>) -> CommandResult<Vec<CapabilityView>> {
    let service = service(&state)?;
    Ok(service
        .catalog()
        .into_iter()
        .map(CapabilityView::from)
        .collect())
}

/// Every shipped tutorial.
#[tauri::command]
pub fn studio_tutorials(state: State<'_, AppState>) -> CommandResult<Vec<CapabilityView>> {
    let service = service(&state)?;
    Ok(service
        .catalog()
        .into_iter()
        .filter(|d| scirust_studio_runtime::tutorial_scenario_for(d.id.0).is_some())
        .map(CapabilityView::from)
        .collect())
}

/// Load a capability's tested tutorial scenario.
#[tauri::command]
pub fn studio_load_tutorial(
    state: State<'_, AppState>,
    capability_id: String,
) -> CommandResult<ScenarioView> {
    check_id("capability id", &capability_id)?;
    let service = service(&state)?;
    let doc = service.load_tutorial(&capability_id).map_err(error)?;
    Ok(ScenarioView {
        source: doc.source,
        path: doc.path,
        capability_id: doc.capability_id,
    })
}

/// Validate scenario source without running it.
#[tauri::command]
pub fn studio_validate_scenario(
    state: State<'_, AppState>,
    source: String,
) -> CommandResult<scirust_studio_app_service::ValidationOutcome> {
    check_source(&source)?;
    let service = service(&state)?;
    Ok(service.validate_source(&source))
}

/// Validate and start a run.
#[tauri::command]
pub fn studio_start_run(state: State<'_, AppState>, source: String) -> CommandResult<JobView> {
    check_source(&source)?;
    let service = service(&state)?;
    service.start_run(&source).map_err(error)
}

/// Cancel a run.
#[tauri::command]
pub fn studio_cancel_run(state: State<'_, AppState>, job_id: String) -> CommandResult<()> {
    check_id("job id", &job_id)?;
    let service = service(&state)?;
    service.cancel_run(&job_id).map_err(error)
}

/// The current state of a run.
#[tauri::command]
pub fn studio_job_snapshot(state: State<'_, AppState>, job_id: String) -> CommandResult<JobView> {
    check_id("job id", &job_id)?;
    let service = service(&state)?;
    service.job_snapshot(&job_id).map_err(error)
}

/// Everything recorded in the run store.
#[tauri::command]
pub fn studio_list_runs(state: State<'_, AppState>) -> CommandResult<Vec<StoredRunView>> {
    let service = service(&state)?;
    Ok(service
        .list_runs()
        .map_err(error)?
        .into_iter()
        .map(StoredRunView::from)
        .collect())
}

/// Load one stored run, including its integrity check.
///
/// Integrity is computed here and returned alongside the scientific checks,
/// as a **separate** field: "the bytes are unchanged" and "the physics
/// checks passed" are different claims and the interface must not merge
/// them.
#[tauri::command]
pub fn studio_load_run(state: State<'_, AppState>, run_id: String) -> CommandResult<RunView> {
    check_id("run id", &run_id)?;
    let service = service(&state)?;
    let loaded = service.load_run(&run_id).map_err(error)?;
    let integrity: VerificationReportView = service.verify_run(&run_id).map_err(error)?.into();
    Ok(run_view(
        &run_id,
        &loaded.manifest,
        &loaded.result,
        loaded.scenario_source,
        integrity,
    ))
}

/// Re-check a stored run's hashes.
#[tauri::command]
pub fn studio_verify_run(
    state: State<'_, AppState>,
    run_id: String,
) -> CommandResult<VerificationReportView> {
    check_id("run id", &run_id)?;
    let service = service(&state)?;
    Ok(service.verify_run(&run_id).map_err(error)?.into())
}

/// Drain buffered application events.
///
/// Polled by the frontend rather than pushed, so a slow interface cannot
/// build an unbounded backlog inside this process — the buffer is bounded
/// and reports what it discarded.
#[tauri::command]
pub fn studio_poll_events(
    state: State<'_, AppState>,
) -> CommandResult<scirust_studio_app_service::EventBatch> {
    let service = service(&state)?;
    Ok(service.drain_events())
}

/// Where runs are stored, so the interface can show and reveal it.
#[tauri::command]
pub fn studio_store_path(state: State<'_, AppState>) -> CommandResult<String> {
    let service = service(&state)?;
    Ok(service.store_root().display().to_string())
}

/// Start a worker after one has exited.
#[tauri::command]
pub fn studio_restart_worker(state: State<'_, AppState>) -> CommandResult<BootstrapView> {
    let service = service(&state)?;
    service.start_worker().map_err(error)?;
    Ok(state.bootstrap_view())
}

/// Bounded worker stderr, for the diagnostics section only.
#[tauri::command]
pub fn studio_worker_diagnostics(state: State<'_, AppState>) -> CommandResult<Vec<String>> {
    let service = service(&state)?;
    Ok(service.worker_diagnostics())
}

/// Whether a run is active, and which — so the window can ask before closing.
#[tauri::command]
pub fn studio_active_job(state: State<'_, AppState>) -> CommandResult<Option<JobView>> {
    let service = service(&state)?;
    Ok(service.active_job())
}

/// Called by the frontend once it has rendered.
///
/// Used by `--smoke-test-window` to know the WebView really loaded the
/// bundled assets, rather than assuming it because a window appeared.
#[tauri::command]
pub fn frontend_ready(state: State<'_, AppState>) -> CommandResult<BootstrapView> {
    state.mark_frontend_ready();
    Ok(state.bootstrap_view())
}

/// Ask the user for a scenario file, and return **its contents**.
///
/// The dialog is native and runs here, in the shell. The frontend supplies no
/// path, receives no path, and cannot ask for this file again later: what
/// comes back is the text and a bare file name for the title bar. That
/// distinction is the whole security argument. A command that took a path
/// would be `read_file(path)` with a friendlier name, and the `dialog:*` and
/// `fs:*` permissions stay ungranted to the webview precisely so the webview
/// cannot reach around this one.
///
/// `Ok(None)` means the user cancelled, which is not an error.
#[tauri::command]
pub fn studio_open_scenario(app: tauri::AppHandle) -> CommandResult<Option<ScenarioFileView>> {
    use tauri_plugin_dialog::DialogExt;

    // Blocking is correct here: Tauri runs synchronous commands off the main
    // thread, and the alternative — resolving a channel inside an async
    // command — buys nothing but a chance to get the main-thread rule wrong.
    let Some(picked) = app
        .dialog()
        .file()
        .add_filter("SciRust scenario", &["toml"])
        .blocking_pick_file()
    else
    {
        return Ok(None);
    };

    let path = picked.into_path().map_err(|e| {
        Box::new(ErrorView {
            code: "FILE_NOT_READABLE".to_string(),
            title: "Cannot open that selection".to_string(),
            explanation: format!("The chosen item is not a file this application can read: {e}"),
            recoverable: true,
            suggested_action: Some("Choose a `.toml` scenario file.".to_string()),
            validation: None,
        })
    })?;

    // Read the length before the contents. A scenario is a short
    // configuration file; pointing the picker at a multi-gigabyte file should
    // not make this process try to hold it.
    if let Ok(meta) = std::fs::metadata(&path)
    {
        if meta.len() > MAX_SOURCE_BYTES as u64
        {
            return Err(too_large(meta.len()));
        }
    }

    let source = std::fs::read_to_string(&path).map_err(|e| {
        Box::new(ErrorView {
            code: "FILE_NOT_READABLE".to_string(),
            title: "Cannot read that file".to_string(),
            // The path is in the message because the *user* chose it and is
            // looking at it; it is not being handed back to the frontend as
            // data it could reuse.
            explanation: format!("{}: {e}", path.display()),
            recoverable: true,
            suggested_action: Some("Check the file is readable text.".to_string()),
            validation: None,
        })
    })?;
    check_source(&source)?;

    Ok(Some(ScenarioFileView {
        file_name: file_name_of(&path),
        source,
    }))
}

/// Ask the user where to save the scenario currently being edited.
///
/// Same shape as opening, in reverse: the frontend hands over *text*, never a
/// destination. `Ok(None)` means the user cancelled.
#[tauri::command]
pub fn studio_save_scenario(
    app: tauri::AppHandle,
    source: String,
) -> CommandResult<Option<String>> {
    use tauri_plugin_dialog::DialogExt;

    check_source(&source)?;

    let Some(picked) = app
        .dialog()
        .file()
        .add_filter("SciRust scenario", &["toml"])
        .set_file_name("scenario.scirust.toml")
        .blocking_save_file()
    else
    {
        return Ok(None);
    };

    let path = picked.into_path().map_err(|e| {
        Box::new(ErrorView {
            code: "FILE_NOT_WRITABLE".to_string(),
            title: "Cannot save there".to_string(),
            explanation: format!("The chosen destination is not a file path: {e}"),
            recoverable: true,
            suggested_action: None,
            validation: None,
        })
    })?;

    std::fs::write(&path, source.as_bytes()).map_err(|e| {
        Box::new(ErrorView {
            code: "FILE_NOT_WRITABLE".to_string(),
            title: "Cannot write that file".to_string(),
            explanation: format!("{}: {e}", path.display()),
            recoverable: true,
            suggested_action: Some("Choose a location you can write to.".to_string()),
            validation: None,
        })
    })?;

    Ok(Some(file_name_of(&path)))
}

/// The last component of a path, for display only.
///
/// Never the path itself. The frontend shows this in a title and must not be
/// able to reconstruct a location from it.
fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "scenario.toml".to_string())
}

fn too_large(actual: u64) -> Box<ErrorView> {
    Box::new(ErrorView {
        code: "SCENARIO_TOO_LARGE".to_string(),
        title: "Scenario too large".to_string(),
        explanation: format!(
            "A scenario file may be at most {MAX_SOURCE_BYTES} bytes; that one is {actual}."
        ),
        recoverable: true,
        suggested_action: Some("Choose a scenario file.".to_string()),
        validation: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_formed_identifiers_are_accepted() {
        for id in [
            "job-1",
            "20260728T215109Z-ef820514594aa333",
            "sim.epidemiology.sir",
            "a_b.c-1",
        ]
        {
            assert!(check_id("id", id).is_ok(), "{id}");
        }
    }

    /// The ids reach this process from the webview; anything that could be
    /// mistaken for a path is refused before it is used.
    #[test]
    fn identifiers_that_could_traverse_a_path_are_refused() {
        for id in [
            "",
            "../etc/passwd",
            "a/b",
            "a\\b",
            "a b",
            "a\0b",
            "run;rm -rf /",
        ]
        {
            assert!(check_id("id", id).is_err(), "{id:?} must be refused");
        }
    }

    #[test]
    fn an_over_long_identifier_is_refused() {
        assert!(check_id("id", &"a".repeat(129)).is_err());
        assert!(check_id("id", &"a".repeat(128)).is_ok());
    }

    #[test]
    fn a_refused_identifier_produces_a_renderable_error() {
        let err = check_id("run id", "../secret").unwrap_err();
        assert_eq!(err.code, "INVALID_IDENTIFIER");
        assert!(err.title.contains("run id"));
        assert!(err.recoverable);
    }

    #[test]
    fn scenario_source_is_bounded() {
        assert!(check_source("schema_version = 1").is_ok());
        assert!(check_source(&"x".repeat(1024 * 1024)).is_ok());
        assert!(check_source(&"x".repeat(1024 * 1024 + 1)).is_err());
    }
}
