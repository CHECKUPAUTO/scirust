//! The application state Tauri hands to every command.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use scirust_studio_app_service::{
    AppServiceConfig, AppServiceError, StudioAppService, WorkerLaunchConfig,
};

use crate::views::{BootstrapView, ErrorView};

/// Shared state: the app service, plus the small amount the shell itself
/// tracks.
pub struct AppState {
    service: Option<StudioAppService>,
    /// Why bootstrap failed, when it did. The application still opens: the
    /// catalogue and the help must remain usable so someone can read *why*
    /// their engine is missing.
    startup_error: Option<AppServiceError>,
    store_path: String,
    interrupted: Mutex<Vec<String>>,
    frontend_ready: Arc<AtomicBool>,
}

impl AppState {
    /// Bootstrap the service, keeping the failure rather than aborting.
    pub fn new(config: AppServiceConfig) -> Self {
        let store_path = config.store_root.display().to_string();
        match StudioAppService::bootstrap(config)
        {
            Ok(service) =>
            {
                let interrupted = service.interrupted_runs().unwrap_or_default();
                AppState {
                    service: Some(service),
                    startup_error: None,
                    store_path,
                    interrupted: Mutex::new(interrupted),
                    frontend_ready: Arc::new(AtomicBool::new(false)),
                }
            },
            Err(e) => AppState {
                service: None,
                startup_error: Some(e),
                store_path,
                interrupted: Mutex::new(Vec::new()),
                frontend_ready: Arc::new(AtomicBool::new(false)),
            },
        }
    }

    /// The service, or the reason there is not one.
    pub fn service(&self) -> Result<&StudioAppService, AppServiceError> {
        match (&self.service, &self.startup_error)
        {
            (Some(service), _) => Ok(service),
            (None, Some(e)) => Err(e.clone()),
            (None, None) => Err(AppServiceError::WorkerUnavailable {
                reason: "the application service did not start".to_string(),
            }),
        }
    }

    /// Whether the frontend has reported itself ready.
    pub fn frontend_is_ready(&self) -> bool {
        self.frontend_ready.load(Ordering::SeqCst)
    }

    /// A handle a smoke-test loop can poll without holding the state.
    pub fn frontend_ready_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.frontend_ready)
    }

    /// Record that the frontend rendered.
    pub fn mark_frontend_ready(&self) {
        self.frontend_ready.store(true, Ordering::SeqCst);
    }

    /// Everything the interface needs at start-up, including the reason the
    /// worker is missing when it is.
    pub fn bootstrap_view(&self) -> BootstrapView {
        let (worker_version, worker_running, capability_count, interrupted) = match &self.service
        {
            Some(service) => (
                service.worker_version(),
                service.worker_is_running(),
                service.catalog().len(),
                self.interrupted
                    .lock()
                    .map(|i| i.clone())
                    .unwrap_or_default(),
            ),
            // The catalogue is static data compiled into this binary, so it
            // is still accurate with no worker — which is what lets the
            // application stay useful while reporting the problem.
            None => (
                None,
                false,
                scirust_studio_runtime::build_registry().len(),
                Vec::new(),
            ),
        };

        BootstrapView {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            worker_version,
            worker_running,
            store_path: self.store_path.clone(),
            capability_count,
            interrupted_runs: interrupted,
            worker_problem: self.startup_error.clone().map(ErrorView::from),
        }
    }
}

/// Build the service configuration, preferring a worker beside this
/// executable and falling back to the development target directory.
///
/// The fallback exists so `cargo run` works from a checkout without a
/// bundling step. It is a *list of candidates*, never a `PATH` search: an
/// unrelated executable of the same name must not be launchable as this
/// application's engine.
pub fn default_config() -> Result<AppServiceConfig, AppServiceError> {
    let mut config = AppServiceConfig::platform_default()?;

    let mut candidates = config.worker.candidates.clone();
    if let Ok(exe) = std::env::current_exe()
    {
        // A development build: apps/scirust-studio/src-tauri/target/<profile>/
        // sits beside the root workspace's target directory.
        if let Some(dir) = exe.parent()
        {
            for up in [3usize, 4, 5]
            {
                if let Some(root) = dir.ancestors().nth(up)
                {
                    for profile in ["debug", "release"]
                    {
                        candidates.push(
                            root.join("target")
                                .join(profile)
                                .join(WorkerLaunchConfig::executable_name()),
                        );
                    }
                }
            }
        }
    }
    candidates.dedup();
    config.worker.candidates = candidates;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_bootstrap_still_yields_a_usable_view() {
        let dir = tempfile::tempdir().expect("temp");
        let state = AppState::new(AppServiceConfig {
            worker: WorkerLaunchConfig::at("/definitely/not/a/worker"),
            store_root: dir.path().to_path_buf(),
            event_buffer_capacity: 16,
        });

        let view = state.bootstrap_view();
        assert!(!view.worker_running);
        assert!(
            view.worker_problem.is_some(),
            "the reason must be reported, not hidden"
        );
        assert_eq!(
            view.capability_count, 5,
            "the catalogue is compiled in and stays accurate without a worker"
        );
        assert!(!view.app_version.is_empty());
        assert!(state.service().is_err());
    }

    #[test]
    fn the_frontend_ready_flag_starts_false_and_latches() {
        let dir = tempfile::tempdir().expect("temp");
        let state = AppState::new(AppServiceConfig {
            worker: WorkerLaunchConfig::at("/definitely/not/a/worker"),
            store_root: dir.path().to_path_buf(),
            event_buffer_capacity: 16,
        });
        assert!(!state.frontend_is_ready());
        state.mark_frontend_ready();
        assert!(state.frontend_is_ready());
        assert!(state.frontend_ready_flag().load(Ordering::SeqCst));
    }

    /// The worker is found by an explicit candidate list, never by searching
    /// `PATH` — an unrelated program of the same name must not be runnable
    /// as this application's engine.
    #[test]
    fn worker_candidates_are_explicit_paths() {
        let Ok(config) = default_config()
        else
        {
            return; // no home directory in this environment
        };
        assert!(!config.worker.candidates.is_empty());
        for candidate in &config.worker.candidates
        {
            assert!(
                candidate.is_absolute(),
                "{} must be absolute",
                candidate.display()
            );
            assert!(
                candidate.file_name().and_then(|n| n.to_str())
                    == Some(WorkerLaunchConfig::executable_name()),
                "{} is not the worker",
                candidate.display()
            );
        }
    }
}
