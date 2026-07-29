//! The Dioxus components.
//!
//! Compiled for `wasm32` only. Everything these components decide is decided
//! in [`crate::model`], [`crate::chart`] or [`crate::actions`] — this module
//! draws the result and performs the I/O, and holds no rules of its own. The
//! practical test of that split: the run lifecycle, the action availability
//! rules and the chart reduction are all covered by host tests, and none of
//! those tests need a browser.

mod chart_view;
mod parts;
mod views;

use std::collections::BTreeSet;
use std::rc::Rc;

use dioxus::prelude::*;
use js_sys::Promise;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen_futures::JsFuture;

use crate::actions::{Action, Shortcut, resolve};
use crate::backend::{
    ActiveBackend, CapabilityWire, FrontendError, RunWire, StoredRunWire, StudioBackend,
    USES_MOCK_DATA, active_backend,
};
use crate::i18n::{Language, t};
use crate::model::{
    ActivityLevel, AppModel, ErrorLine, Msg, Outcome, ProblemLine, RunDisplay, View, update,
};

/// How often the interface asks the shell for news.
///
/// A poll rather than a push because the shell's event buffer is bounded and
/// drains on request: a stalled interface cannot then build an unbounded
/// backlog inside the native process. 120 ms is below the threshold at which
/// a progress bar looks stuttery and far above the cost of one IPC round
/// trip.
const POLL_INTERVAL_MS: i32 = 120;

/// Start the interface.
pub fn launch() {
    dioxus::launch(App);
}

/// Everything a component needs, passed through context.
///
/// `Signal` is `Copy`, so this is cheap to clone into an event handler or an
/// async task.
#[derive(Clone)]
pub struct Ui {
    /// The application state.
    pub model: Signal<AppModel>,
    /// The capability catalogue, once loaded.
    pub catalog: Signal<Vec<CapabilityWire>>,
    /// The stored runs.
    pub runs: Signal<Vec<StoredRunWire>>,
    /// The run currently displayed.
    pub run: Signal<Option<RunWire>>,
    /// Series the user has switched off in the legend.
    pub hidden_series: Signal<BTreeSet<String>>,
    /// Worker stderr, when the diagnostics panel has asked for it.
    pub diagnostics: Signal<Vec<String>>,
    /// The catalogue's filter box.
    pub catalogue_query: Signal<String>,
    /// An action requested by a keyboard shortcut, awaiting handling.
    pub pending: Signal<Option<Action>>,
    /// The backend. `Rc` because the mock owns state; the real bridge is
    /// zero-sized.
    pub backend: Rc<ActiveBackend>,
}

impl Ui {
    /// The current language.
    pub fn language(&self) -> Language {
        self.model.read().language
    }

    /// A translated string.
    pub fn t(&self, key: &str) -> &'static str {
        t(self.model.read().language, key)
    }

    /// Apply a message to the model.
    pub fn send(&self, message: Msg) {
        let mut model = self.model;
        let next = update(model.read().clone(), message);
        model.set(next);
    }

    /// Show a backend failure as an error card and an activity line.
    pub fn fail(&self, error: FrontendError) {
        self.send(Msg::ShowError(ErrorLine {
            code: error.code.clone(),
            title: error.title.clone(),
            explanation: error.explanation.clone(),
            recoverable: error.recoverable,
            suggested_action: error.suggested_action.clone(),
        }));
    }

    /// Load the catalogue and the run history.
    pub async fn refresh(&self) {
        match self.backend.catalog().await
        {
            Ok(items) =>
            {
                let mut catalog = self.catalog;
                catalog.set(items);
            },
            Err(e) => self.fail(e),
        }
        self.refresh_runs().await;
    }

    /// Reload the stored-run list.
    pub async fn refresh_runs(&self) {
        match self.backend.list_runs().await
        {
            Ok(items) =>
            {
                let mut runs = self.runs;
                runs.set(items);
            },
            Err(e) => self.fail(e),
        }
    }

    /// Put a capability's tutorial into the editor.
    pub async fn open_tutorial(&self, capability_id: String) {
        match self.backend.load_tutorial(&capability_id).await
        {
            Ok(scenario) =>
            {
                self.send(Msg::ScenarioLoaded {
                    source: scenario.source,
                    path: scenario.path,
                });
                self.send(Msg::Navigate(View::Experiment));
            },
            Err(e) => self.fail(e),
        }
    }

    /// Validate the editor's contents without running them.
    pub async fn validate(&self) {
        let source = self.model.read().source.clone();
        match self.backend.validate_scenario(&source).await
        {
            Ok(outcome) => self.send(Msg::Validated {
                valid: outcome.valid,
                problems: outcome.problems.iter().map(problem_line).collect(),
            }),
            Err(e) => self.fail(e),
        }
    }

    /// Validate and start a run.
    pub async fn start_run(&self) {
        let source = self.model.read().source.clone();
        match self.backend.start_run(&source).await
        {
            Ok(job) => self.send(Msg::RunStarted {
                job_id: job.job_id,
                supports_progress: job.supports_progress,
            }),
            Err(error) =>
            {
                // A rejected scenario belongs in the editor's problem list,
                // not only in a modal a user has to dismiss to act on it.
                if let Some(validation) = &error.validation
                {
                    self.send(Msg::Validated {
                        valid: false,
                        problems: validation.problems.iter().map(problem_line).collect(),
                    });
                    self.send(Msg::Navigate(View::Experiment));
                }
                self.fail(error);
            },
        }
    }

    /// Ask the active run to stop.
    pub async fn cancel_run(&self) {
        let Some(job_id) = self.model.read().job_id.clone()
        else
        {
            return;
        };
        self.send(Msg::CancelRequested);
        if let Err(e) = self.backend.cancel_run(&job_id).await
        {
            self.fail(e);
        }
    }

    /// Open a stored run and chart it.
    pub async fn open_run(&self, run_id: String) {
        match self.backend.load_run(&run_id).await
        {
            Ok(view) =>
            {
                let mut hidden = self.hidden_series;
                hidden.set(BTreeSet::new());
                let mut run = self.run;
                run.set(Some(view));
                self.send(Msg::RunLoaded {
                    run_id: run_id.clone(),
                });
                self.send(Msg::Navigate(View::Experiment));
            },
            Err(e) => self.fail(e),
        }
    }

    /// Re-check the displayed run's hashes.
    pub async fn verify_displayed_run(&self) {
        let Some(run_id) = self.model.read().loaded_run_id.clone()
        else
        {
            return;
        };
        match self.backend.verify_run(&run_id).await
        {
            Ok(report) =>
            {
                let mut run = self.run;
                let updated = run.read().clone().map(|mut r| {
                    r.integrity = report.clone();
                    r
                });
                if updated.is_some()
                {
                    run.set(updated);
                }
                let (level, message) = if report.intact
                {
                    (
                        ActivityLevel::Pass,
                        format!("{run_id}: stored bytes still match their hashes"),
                    )
                }
                else
                {
                    (
                        ActivityLevel::Error,
                        format!(
                            "{run_id}: integrity check FAILED — {}",
                            report.detail.unwrap_or_default()
                        ),
                    )
                };
                self.send(Msg::Log { level, message });
            },
            Err(e) => self.fail(e),
        }
    }

    /// Start a worker after one has exited.
    pub async fn restart_worker(&self) {
        match self.backend.restart_worker().await
        {
            Ok(boot) => self.send(Msg::WorkerState {
                running: boot.worker_running,
                version: boot.worker_version,
            }),
            Err(e) => self.fail(e),
        }
    }

    /// Fetch the worker's captured stderr.
    pub async fn load_diagnostics(&self) {
        match self.backend.worker_diagnostics().await
        {
            Ok(lines) =>
            {
                let mut diagnostics = self.diagnostics;
                diagnostics.set(lines);
            },
            Err(e) => self.fail(e),
        }
    }

    /// Perform an action, whatever asked for it.
    ///
    /// One place, so the palette, the keyboard and the buttons cannot drift
    /// into doing different things for the same action.
    pub fn perform(&self, action: Action) {
        let context = self.model.read().action_context();
        if let Err(reason) = resolve(action, context)
        {
            self.send(Msg::Log {
                level: ActivityLevel::Warning,
                message: format!("{}: {}", self.t(action.label_key()), reason.reason()),
            });
            return;
        }

        if let Some(view) = crate::model::navigation_target(action)
        {
            if action == Action::NewExperiment
            {
                self.send(Msg::ScenarioLoaded {
                    source: String::new(),
                    path: None,
                });
            }
            self.send(Msg::Navigate(view));
            return;
        }

        match action
        {
            Action::ToggleActivityPanel => self.send(Msg::ToggleActivity),
            Action::ToggleInspector => self.send(Msg::ToggleInspector),
            Action::ChangeLanguage =>
            {
                let next = match self.language()
                {
                    Language::English => Language::French,
                    Language::French => Language::English,
                };
                self.send(Msg::SetLanguage(next));
            },
            Action::ChangeTheme =>
            {
                let themes = crate::model::Theme::all();
                let current = self.model.read().theme;
                let index = themes.iter().position(|t| *t == current).unwrap_or(0);
                self.send(Msg::SetTheme(themes[(index + 1) % themes.len()]));
            },
            _ =>
            {
                let ui = self.clone();
                spawn(async move {
                    match action
                    {
                        Action::ValidateScenario => ui.validate().await,
                        Action::RunExperiment => ui.start_run().await,
                        Action::CancelRun => ui.cancel_run().await,
                        Action::VerifyCurrentRun => ui.verify_displayed_run().await,
                        Action::RestartWorker => ui.restart_worker().await,
                        Action::ShowDiagnostics => ui.load_diagnostics().await,
                        _ =>
                        {},
                    }
                });
            },
        }
    }
}

/// Flatten a wire problem into the display form the reducers use.
fn problem_line(problem: &crate::backend::ProblemWire) -> ProblemLine {
    ProblemLine {
        code: problem.code.clone(),
        title: problem.title.clone(),
        explanation: problem.explanation.clone(),
        suggested_action: problem.suggested_action.clone(),
        field: problem.field.clone(),
        line: problem.location.map(|l| l.line),
        stage: problem.stage.clone(),
    }
}

/// Await a timer without pulling in an async runtime.
async fn sleep(ms: i32) {
    let promise = Promise::new(&mut |resolve, _reject| {
        if let Some(window) = web_sys::window()
        {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = JsFuture::from(promise).await;
}

/// The whole application.
#[component]
fn App() -> Element {
    let ui = use_hook(|| Ui {
        model: Signal::new(AppModel::default()),
        catalog: Signal::new(Vec::new()),
        runs: Signal::new(Vec::new()),
        run: Signal::new(None),
        hidden_series: Signal::new(BTreeSet::new()),
        diagnostics: Signal::new(Vec::new()),
        catalogue_query: Signal::new(String::new()),
        pending: Signal::new(None),
        backend: Rc::new(active_backend()),
    });
    use_context_provider(|| ui.clone());

    // Start-up: pick up the operating system's language, tell the shell the
    // interface rendered, then load the catalogue.
    {
        let ui = ui.clone();
        use_future(move || {
            let ui = ui.clone();
            async move {
                if let Some(locale) = browser_language()
                {
                    ui.send(Msg::SetLanguage(Language::from_locale(&locale)));
                }
                match ui.backend.frontend_ready().await
                {
                    Ok(boot) => ui.send(Msg::Bootstrapped {
                        worker_running: boot.worker_running,
                        worker_version: boot.worker_version,
                        store_path: boot.store_path,
                        capability_count: boot.capability_count,
                        interrupted_runs: boot.interrupted_runs,
                        problem: boot.worker_problem.map(|p| ErrorLine {
                            code: p.code,
                            title: p.title,
                            explanation: p.explanation,
                            recoverable: p.recoverable,
                            suggested_action: p.suggested_action,
                        }),
                    }),
                    Err(e) => ui.fail(e),
                }
                ui.refresh().await;
            }
        });
    }

    // The one polling loop. It drains the shell's event buffer and, while a
    // run is active, follows the job.
    {
        let ui = ui.clone();
        use_future(move || {
            let ui = ui.clone();
            async move {
                loop
                {
                    sleep(POLL_INTERVAL_MS).await;
                    drain_events(&ui).await;
                    follow_active_job(&ui).await;
                }
            }
        });
    }

    // Keyboard shortcuts, registered on the document so they work wherever
    // focus happens to be.
    use_keyboard(ui.pending);
    {
        let ui = ui.clone();
        use_effect(move || {
            let mut pending = ui.pending;
            // Read into a local so the borrow guard is released before the
            // write below; `Signal` would otherwise be borrowed twice.
            let requested = *pending.read();
            if let Some(action) = requested
            {
                pending.set(None);
                ui.perform(action);
            }
        });
    }

    let model = ui.model.read().clone();
    let theme = model.theme.attribute();
    let language = model.language.tag();

    rsx! {
        div {
            class: "app",
            "data-theme": theme,
            lang: language,

            if USES_MOCK_DATA {
                div { class: "mock-banner", role: "alert",
                    strong { "{ui.t(\"status.mock\")}" }
                    " "
                    "{ui.t(\"mock.banner\")}"
                }
            }

            parts::TitleBar {}

            main { class: "main",
                div { class: "content",
                    match model.view {
                        View::Home => rsx! { views::Home {} },
                        View::Catalogue => rsx! { views::Catalogue {} },
                        View::Experiment => rsx! { views::Experiment {} },
                        View::Runs => rsx! { views::Runs {} },
                        View::Help => rsx! { views::Help {} },
                    }
                }
                if model.inspector_open {
                    parts::Inspector {}
                }
            }

            if model.activity_open {
                parts::ActivityPanel {}
            }

            parts::StatusBar {}

            if model.palette_open {
                parts::CommandPalette {}
            }

            if let Some(error) = model.error.clone() {
                parts::ErrorCard { error }
            }
        }
    }
}

/// The browser's language tag, when there is one.
fn browser_language() -> Option<String> {
    let navigator = web_sys::window()?.navigator();
    navigator.language()
}

/// Drain the shell's event buffer into the activity log.
async fn drain_events(ui: &Ui) {
    let Ok(batch) = ui.backend.poll_events().await
    else
    {
        return;
    };
    if batch.dropped > 0
    {
        ui.send(Msg::Log {
            level: ActivityLevel::Warning,
            message: format!(
                "{} activity messages were dropped: the interface fell behind",
                batch.dropped
            ),
        });
    }
    for event in batch.events
    {
        use crate::backend::EventWire;
        match event
        {
            EventWire::WorkerStarted { worker_version, .. } =>
            {
                ui.send(Msg::WorkerState {
                    running: true,
                    version: Some(worker_version.clone()),
                });
                ui.send(Msg::Log {
                    level: ActivityLevel::Info,
                    message: format!("calculation engine {worker_version} started"),
                });
            },
            EventWire::WorkerExited { class } =>
            {
                ui.send(Msg::WorkerState {
                    running: false,
                    version: None,
                });
                ui.send(Msg::Log {
                    level: ActivityLevel::Warning,
                    message: format!("calculation engine stopped: {class:?}"),
                });
            },
            EventWire::JobWarning { warning, .. } => ui.send(Msg::Log {
                level: ActivityLevel::Warning,
                message: format!("{}: {}", warning.category, warning.message),
            }),
            EventWire::Diagnostic { level, message } => ui.send(Msg::Log {
                level: match level.as_str()
                {
                    "pass" => ActivityLevel::Pass,
                    "warning" => ActivityLevel::Warning,
                    "error" => ActivityLevel::Error,
                    _ => ActivityLevel::Info,
                },
                message,
            }),
            // Job state is followed through `job_snapshot`, which is the
            // authoritative answer; logging it from here as well would
            // double every line.
            EventWire::JobStarted { .. } | EventWire::JobState { .. } =>
            {},
        }
    }
}

/// Follow the active job to its settled state.
async fn follow_active_job(ui: &Ui) {
    let (job_id, active) = {
        let model = ui.model.read();
        (model.job_id.clone(), model.run.is_active())
    };
    let (Some(job_id), true) = (job_id, active)
    else
    {
        return;
    };

    let snapshot = match ui.backend.job_snapshot(&job_id).await
    {
        Ok(s) => s,
        Err(e) =>
        {
            ui.fail(e);
            return;
        },
    };

    use crate::backend::JobStateWire;
    match snapshot.state
    {
        JobStateWire::Running { fraction, t } => ui.send(Msg::RunProgress { fraction, t }),
        JobStateWire::Completed { run_id } =>
        {
            ui.send(Msg::RunSettled(Outcome::Completed {
                run_id: run_id.clone(),
            }));
            ui.refresh_runs().await;
            ui.open_run(run_id).await;
        },
        JobStateWire::Cancelled => ui.send(Msg::RunSettled(Outcome::Cancelled)),
        JobStateWire::FailedNumerical { message } => ui.send(Msg::RunSettled(Outcome::Failed {
            kind: crate::model::FailureKind::Numerical,
            message,
        })),
        JobStateWire::FailedValidation { message } => ui.send(Msg::RunSettled(Outcome::Failed {
            kind: crate::model::FailureKind::Validation,
            message,
        })),
        JobStateWire::FailedInternal { message } => ui.send(Msg::RunSettled(Outcome::Failed {
            kind: crate::model::FailureKind::Internal,
            message,
        })),
        JobStateWire::Interrupted { detail } =>
        {
            ui.send(Msg::RunSettled(Outcome::Interrupted { detail }));
            ui.refresh_runs().await;
        },
        // Queued, indeterminate and cancelling need no model change: the
        // reducers already put the interface in the right state when the
        // run started or when cancellation was requested.
        JobStateWire::Queued | JobStateWire::RunningIndeterminate | JobStateWire::Cancelling =>
        {},
    }
}

/// Register the document-level keyboard handler.
///
/// On the document rather than on a component, so a shortcut works whether
/// focus is in the editor, on a button or nowhere at all — and so the
/// handler is registered exactly once instead of once per render.
fn use_keyboard(mut pending: Signal<Option<Action>>) {
    use_hook(move || {
        let closure = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(
            move |event: web_sys::KeyboardEvent| {
                let pressed = Shortcut {
                    ctrl: event.ctrl_key() || event.meta_key(),
                    shift: event.shift_key(),
                    key: "",
                };
                let key = event.key();
                for action in Action::all()
                {
                    let Some(shortcut) = action.shortcut()
                    else
                    {
                        continue;
                    };
                    if shortcut.ctrl == pressed.ctrl
                        && shortcut.shift == pressed.shift
                        && shortcut.key.eq_ignore_ascii_case(&key)
                    {
                        event.prevent_default();
                        pending.set(Some(*action));
                        return;
                    }
                }
            },
        );
        if let Some(document) = web_sys::window().and_then(|w| w.document())
        {
            let _ = document
                .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
        }
        // Kept alive for the life of the application: dropping it would
        // silently unregister every shortcut. `Rc` because a hook's value
        // must be `Clone` and a `Closure` is not.
        Rc::new(closure)
    });
}

/// A short, readable rendering of a run's progress for the status bar.
pub fn progress_label(ui: &Ui, run: &RunDisplay) -> String {
    let label = ui.t(run.label_key());
    match run.percentage()
    {
        Some(percent) => format!("{label} {percent:.0}%"),
        None => label.to_string(),
    }
}
