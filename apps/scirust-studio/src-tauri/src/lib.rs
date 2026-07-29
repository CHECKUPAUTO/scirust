//! SciRust Studio — the Tauri 2 native shell.
//!
//! This crate owns the window, the sidecar worker's lifetime and a narrow set
//! of typed commands. It contains **no scientific code**: every capability,
//! validation rule, integrator and storage guarantee lives in the root
//! workspace's Studio crates and is reached through
//! `scirust-studio-app-service`.
//!
//! See `docs/studio/adr/0005-tauri-dioxus-desktop-architecture.md` for why
//! Tauri hosts a Dioxus **Web** frontend rather than Dioxus Desktop, and
//! `docs/studio/DESKTOP_SECURITY.md` for the permission and CSP decisions.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod commands;
pub mod smoke;
pub mod state;
pub mod views;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use state::{AppState, default_config};

/// How the application was asked to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Open the window normally.
    Application,
    /// Run the headless backend self-check and exit.
    BackendSmokeTest,
    /// Open the window, wait for the frontend to report itself ready, and
    /// exit.
    WindowSmokeTest,
    /// Print the version and exit.
    Version,
}

/// Parse the command line.
///
/// Deliberately tiny and explicit rather than a dependency: the application
/// accepts four things, and an argument parser would be more code than the
/// thing it parses.
pub fn parse_mode<I, S>(args: I) -> Mode
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for arg in args
    {
        match arg.as_ref()
        {
            "--smoke-test-backend" => return Mode::BackendSmokeTest,
            "--smoke-test-window" => return Mode::WindowSmokeTest,
            "--version" | "-V" => return Mode::Version,
            _ =>
            {},
        }
    }
    Mode::Application
}

/// How long `--smoke-test-window` waits for the frontend before failing.
const WINDOW_SMOKE_TIMEOUT: Duration = Duration::from_secs(60);

/// Entry point shared by the desktop binary.
pub fn run() -> i32 {
    let mode = parse_mode(std::env::args().skip(1));

    if mode == Mode::Version
    {
        println!("SciRust Studio {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }

    let config = match default_config()
    {
        Ok(config) => config,
        Err(e) =>
        {
            eprintln!("SciRust Studio: cannot start: {e}");
            return 1;
        },
    };

    if mode == Mode::BackendSmokeTest
    {
        return smoke::run_backend_smoke_test(config);
    }

    let state = AppState::new(config);
    let ready = state.frontend_ready_flag();

    let builder = tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::studio_bootstrap,
            commands::studio_catalog,
            commands::studio_tutorials,
            commands::studio_load_tutorial,
            commands::studio_validate_scenario,
            commands::studio_start_run,
            commands::studio_cancel_run,
            commands::studio_job_snapshot,
            commands::studio_list_runs,
            commands::studio_load_run,
            commands::studio_verify_run,
            commands::studio_poll_events,
            commands::studio_store_path,
            commands::studio_restart_worker,
            commands::studio_worker_diagnostics,
            commands::studio_active_job,
            commands::frontend_ready,
        ]);

    if mode == Mode::WindowSmokeTest
    {
        // Watch for the frontend's `frontend_ready` call from a side thread
        // and end the process when it arrives. This is what distinguishes
        // "the WebView actually loaded the bundled assets and ran our code"
        // from "a window appeared".
        std::thread::spawn(move || {
            let deadline = Instant::now() + WINDOW_SMOKE_TIMEOUT;
            while Instant::now() < deadline
            {
                if ready.load(Ordering::SeqCst)
                {
                    println!(
                        "{}",
                        serde_json::json!({
                            "smoke_test": "window",
                            "ok": true,
                            "app_version": env!("CARGO_PKG_VERSION"),
                        })
                    );
                    std::process::exit(0);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "smoke_test": "window",
                    "ok": false,
                    "error": "the frontend did not report itself ready",
                })
            );
            std::process::exit(1);
        });
    }

    match builder.run(tauri::generate_context!())
    {
        Ok(()) => 0,
        Err(e) =>
        {
            eprintln!("SciRust Studio: {e}");
            1
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_mode_opens_the_application() {
        assert_eq!(parse_mode(Vec::<String>::new()), Mode::Application);
        assert_eq!(parse_mode(["--unrecognised"]), Mode::Application);
    }

    #[test]
    fn each_mode_has_a_flag() {
        assert_eq!(parse_mode(["--smoke-test-backend"]), Mode::BackendSmokeTest);
        assert_eq!(parse_mode(["--smoke-test-window"]), Mode::WindowSmokeTest);
        assert_eq!(parse_mode(["--version"]), Mode::Version);
        assert_eq!(parse_mode(["-V"]), Mode::Version);
    }

    #[test]
    fn a_flag_is_recognised_among_other_arguments() {
        assert_eq!(
            parse_mode(["--other", "--smoke-test-backend", "--more"]),
            Mode::BackendSmokeTest
        );
    }
}
