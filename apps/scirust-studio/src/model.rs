//! Application state and the pure reducers that move it.
//!
//! Every transition is a plain function from `(state, message) -> state`
//! with no Dioxus, no DOM and no I/O, so the lifecycle a user actually
//! experiences — bootstrap, edit, validate, run, cancel, interrupt — is unit
//! tested on the host instead of only being clicked through in a window.

use serde::{Deserialize, Serialize};

use crate::actions::{Action, ActionContext};
use crate::i18n::Language;

/// Which view is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum View {
    /// Landing page.
    Home,
    /// The capability catalogue.
    Catalogue,
    /// The editor and the run.
    Experiment,
    /// Stored runs.
    Runs,
    /// Help for the shipped features.
    Help,
}

/// The visual theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Theme {
    /// Follow the operating system.
    System,
    /// Light.
    Light,
    /// Dark.
    Dark,
    /// High contrast.
    HighContrast,
}

impl Theme {
    /// The `data-theme` attribute value.
    pub fn attribute(self) -> &'static str {
        match self
        {
            Theme::System => "system",
            Theme::Light => "light",
            Theme::Dark => "dark",
            Theme::HighContrast => "high-contrast",
        }
    }

    /// Every theme, in picker order.
    pub fn all() -> [Theme; 4] {
        [
            Theme::System,
            Theme::Light,
            Theme::Dark,
            Theme::HighContrast,
        ]
    }
}

/// How the run being watched is progressing, as the UI needs to draw it.
///
/// The distinction between [`RunDisplay::Determinate`] and
/// [`RunDisplay::Indeterminate`] is the whole reason this type exists: a
/// solver that cannot report progress must produce a spinner, never a
/// percentage that would be invented.
#[derive(Debug, Clone, PartialEq)]
pub enum RunDisplay {
    /// Nothing is running.
    Idle,
    /// Accepted, not started.
    Queued,
    /// Running with a genuine fraction.
    Determinate {
        /// `0.0..=1.0`.
        fraction: f64,
        /// Simulation time reached.
        t: f64,
    },
    /// Running with no fraction available.
    Indeterminate,
    /// Cancellation requested.
    Cancelling,
    /// Settled.
    Settled(Outcome),
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Finished and stored.
    Completed {
        /// The stored run's id.
        run_id: String,
    },
    /// Stopped because the user asked.
    Cancelled,
    /// Integration failed.
    Failed {
        /// Which kind, for the label.
        kind: FailureKind,
        /// The explanation.
        message: String,
    },
    /// The engine died mid-run.
    Interrupted {
        /// What is known.
        detail: String,
    },
}

/// The kinds of failure the UI distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// The scenario was rejected.
    Validation,
    /// Integration failed numerically.
    Numerical,
    /// A bug in the engine.
    Internal,
}

impl RunDisplay {
    /// The localization key for the status chip.
    pub fn label_key(&self) -> &'static str {
        match self
        {
            RunDisplay::Idle => "run.state.queued",
            RunDisplay::Queued => "run.state.queued",
            RunDisplay::Determinate { .. } | RunDisplay::Indeterminate => "run.state.running",
            RunDisplay::Cancelling => "run.state.cancelling",
            RunDisplay::Settled(Outcome::Completed { .. }) => "run.state.completed",
            RunDisplay::Settled(Outcome::Cancelled) => "run.state.cancelled",
            RunDisplay::Settled(Outcome::Failed { .. }) => "run.state.failed",
            RunDisplay::Settled(Outcome::Interrupted { .. }) => "run.state.interrupted",
        }
    }

    /// Whether a run is occupying the single active slot.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            RunDisplay::Queued
                | RunDisplay::Determinate { .. }
                | RunDisplay::Indeterminate
                | RunDisplay::Cancelling
        )
    }

    /// The percentage to draw, when one genuinely exists.
    ///
    /// `None` means "draw indeterminate activity". A caller that defaulted
    /// this to zero would show a bar stuck at 0% for the whole of a
    /// Robertson run.
    pub fn percentage(&self) -> Option<f64> {
        match self
        {
            RunDisplay::Determinate { fraction, .. } => Some(fraction * 100.0),
            RunDisplay::Settled(Outcome::Completed { .. }) => Some(100.0),
            _ => None,
        }
    }
}

/// One line in the activity panel.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivityLine {
    /// Severity tag, rendered as `[INFO]`, `[PASS]`, …
    pub level: ActivityLevel,
    /// The text.
    pub message: String,
}

/// How prominent an activity line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityLevel {
    /// Routine.
    Info,
    /// A check that passed.
    Pass,
    /// A run's progress.
    Run,
    /// Something to notice.
    Warning,
    /// Something that failed.
    Error,
}

impl ActivityLevel {
    /// The bracketed tag shown at the start of the line.
    ///
    /// Plain text, never an ANSI escape sequence: this is a structured
    /// panel, not a terminal emulator.
    pub fn tag(self) -> &'static str {
        match self
        {
            ActivityLevel::Info => "INFO",
            ActivityLevel::Pass => "PASS",
            ActivityLevel::Run => "RUN",
            ActivityLevel::Warning => "WARN",
            ActivityLevel::Error => "FAIL",
        }
    }
}

/// The application's whole state.
#[derive(Debug, Clone, PartialEq)]
pub struct AppModel {
    /// Which view is showing.
    pub view: View,
    /// Interface language.
    pub language: Language,
    /// Visual theme.
    pub theme: Theme,
    /// Whether a worker is up.
    pub worker_running: bool,
    /// The worker's version.
    pub worker_version: Option<String>,
    /// Where runs are stored.
    pub store_path: String,
    /// How many capabilities can be run.
    pub capability_count: usize,
    /// The editor's contents.
    pub source: String,
    /// The last saved contents, for the dirty flag.
    pub saved_source: String,
    /// Where the scenario came from.
    pub source_path: Option<String>,
    /// The last validation outcome.
    pub problems: Vec<ProblemLine>,
    /// Whether the last validation passed.
    pub validated_ok: bool,
    /// The active or last job.
    pub job_id: Option<String>,
    /// How to draw it.
    pub run: RunDisplay,
    /// The run currently displayed, if one has been loaded.
    pub loaded_run_id: Option<String>,
    /// Activity lines, newest last.
    pub activity: Vec<ActivityLine>,
    /// Whether the activity panel is open.
    pub activity_open: bool,
    /// Whether the inspector is open.
    pub inspector_open: bool,
    /// Whether the command palette is open.
    pub palette_open: bool,
    /// The palette's query.
    pub palette_query: String,
    /// Runs whose writer never finished.
    pub interrupted_runs: Vec<String>,
    /// A blocking error to show, when one is outstanding.
    pub error: Option<ErrorLine>,
}

/// A validation problem, flattened for display.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemLine {
    /// Stable code, when the error carries one.
    pub code: Option<String>,
    /// Heading.
    pub title: String,
    /// Explanation.
    pub explanation: String,
    /// Suggested fix.
    pub suggested_action: Option<String>,
    /// The field, when identified.
    pub field: Option<String>,
    /// Line to jump to, when known.
    pub line: Option<usize>,
    /// Which stage rejected it.
    pub stage: String,
}

/// A structured error card.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorLine {
    /// Stable code.
    pub code: String,
    /// Heading.
    pub title: String,
    /// Explanation.
    pub explanation: String,
    /// Whether the user can carry on.
    pub recoverable: bool,
    /// What to try.
    pub suggested_action: Option<String>,
}

impl Default for AppModel {
    fn default() -> Self {
        AppModel {
            view: View::Home,
            language: Language::English,
            theme: Theme::System,
            worker_running: false,
            worker_version: None,
            store_path: String::new(),
            capability_count: 0,
            source: String::new(),
            saved_source: String::new(),
            source_path: None,
            problems: Vec::new(),
            validated_ok: false,
            job_id: None,
            run: RunDisplay::Idle,
            loaded_run_id: None,
            activity: Vec::new(),
            activity_open: true,
            inspector_open: true,
            palette_open: false,
            palette_query: String::new(),
            interrupted_runs: Vec::new(),
            error: None,
        }
    }
}

impl AppModel {
    /// Whether the editor has unsaved changes.
    pub fn dirty(&self) -> bool {
        self.source != self.saved_source
    }

    /// The context the action registry needs.
    pub fn action_context(&self) -> ActionContext {
        ActionContext {
            worker_running: self.worker_running,
            run_active: self.run.is_active(),
            has_scenario: !self.source.trim().is_empty(),
            dirty: self.dirty(),
            has_loaded_run: self.loaded_run_id.is_some(),
        }
    }

    fn log(&mut self, level: ActivityLevel, message: impl Into<String>) {
        // Bounded: an application left open for a week must not accumulate
        // an unbounded log in memory.
        const MAX_LINES: usize = 500;
        self.activity.push(ActivityLine {
            level,
            message: message.into(),
        });
        if self.activity.len() > MAX_LINES
        {
            let excess = self.activity.len() - MAX_LINES;
            self.activity.drain(0..excess);
        }
    }
}

/// Everything that can change the model.
#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    /// Bootstrap completed.
    Bootstrapped {
        /// Whether a worker is up.
        worker_running: bool,
        /// Its version.
        worker_version: Option<String>,
        /// Where runs are stored.
        store_path: String,
        /// How many capabilities.
        capability_count: usize,
        /// Interrupted runs found at start-up.
        interrupted_runs: Vec<String>,
        /// Why the worker is missing, when it is.
        problem: Option<ErrorLine>,
    },
    /// Show a different view.
    Navigate(View),
    /// Switch language.
    SetLanguage(Language),
    /// Switch theme.
    SetTheme(Theme),
    /// The editor changed.
    EditSource(String),
    /// A scenario was loaded.
    ScenarioLoaded {
        /// The text.
        source: String,
        /// Where from.
        path: Option<String>,
    },
    /// The scenario was saved.
    Saved {
        /// Where to.
        path: Option<String>,
    },
    /// Validation finished.
    Validated {
        /// Whether it passed.
        valid: bool,
        /// What was found.
        problems: Vec<ProblemLine>,
    },
    /// A run was accepted.
    RunStarted {
        /// The job.
        job_id: String,
        /// Whether it can report progress.
        supports_progress: bool,
    },
    /// Progress arrived.
    RunProgress {
        /// `0.0..=1.0`.
        fraction: f64,
        /// Simulation time.
        t: f64,
    },
    /// Cancellation was requested.
    CancelRequested,
    /// The run settled.
    RunSettled(Outcome),
    /// The worker's state changed.
    WorkerState {
        /// Whether it is up.
        running: bool,
        /// Its version.
        version: Option<String>,
    },
    /// A stored run was opened.
    RunLoaded {
        /// Its id.
        run_id: String,
    },
    /// Show or hide the activity panel.
    ToggleActivity,
    /// Show or hide the inspector.
    ToggleInspector,
    /// Show or hide the palette.
    TogglePalette,
    /// The palette's query changed.
    PaletteQuery(String),
    /// Show an error card.
    ShowError(ErrorLine),
    /// Dismiss the error card.
    DismissError,
    /// Clear the activity panel.
    ClearActivity,
    /// Record an activity line.
    ///
    /// The route by which the shell's own events — a worker starting, a
    /// dropped event batch, an integrity check — reach the log the user
    /// reads, without the interface keeping a second log of its own.
    Log {
        /// Severity.
        level: ActivityLevel,
        /// The text.
        message: String,
    },
}

/// Apply a message. Pure: no I/O, no globals, no clock.
pub fn update(mut model: AppModel, message: Msg) -> AppModel {
    match message
    {
        Msg::Bootstrapped {
            worker_running,
            worker_version,
            store_path,
            capability_count,
            interrupted_runs,
            problem,
        } =>
        {
            model.worker_running = worker_running;
            model.worker_version = worker_version;
            model.store_path = store_path;
            model.capability_count = capability_count;
            model.interrupted_runs = interrupted_runs.clone();
            if worker_running
            {
                model.log(ActivityLevel::Info, "Calculation engine started");
            }
            else
            {
                model.log(ActivityLevel::Error, "Calculation engine is not running");
            }
            if !interrupted_runs.is_empty()
            {
                model.log(
                    ActivityLevel::Warning,
                    format!(
                        "{} interrupted run(s) found from a previous session",
                        interrupted_runs.len()
                    ),
                );
            }
            model.error = problem;
        },
        Msg::Navigate(view) => model.view = view,
        Msg::SetLanguage(language) => model.language = language,
        Msg::SetTheme(theme) => model.theme = theme,
        Msg::EditSource(source) =>
        {
            model.source = source;
            // Editing invalidates the previous verdict: showing a stale
            // "valid" tick against changed text would be a lie.
            model.validated_ok = false;
        },
        Msg::ScenarioLoaded { source, path } =>
        {
            model.saved_source = source.clone();
            model.source = source;
            model.source_path = path;
            model.problems.clear();
            model.validated_ok = false;
            model.view = View::Experiment;
            model.log(ActivityLevel::Info, "Scenario opened");
        },
        Msg::Saved { path } =>
        {
            model.saved_source = model.source.clone();
            if path.is_some()
            {
                model.source_path = path;
            }
            model.log(ActivityLevel::Info, "Scenario saved");
        },
        Msg::Validated { valid, problems } =>
        {
            model.validated_ok = valid;
            model.problems = problems;
            if valid
            {
                model.log(ActivityLevel::Pass, "Validation passed");
            }
            else
            {
                model.log(
                    ActivityLevel::Error,
                    format!("Validation found {} problem(s)", model.problems.len()),
                );
            }
        },
        Msg::RunStarted {
            job_id,
            supports_progress,
        } =>
        {
            model.job_id = Some(job_id);
            model.run = if supports_progress
            {
                RunDisplay::Queued
            }
            else
            {
                RunDisplay::Indeterminate
            };
            model.view = View::Experiment;
            model.error = None;
            model.log(ActivityLevel::Info, "Run started");
            if !supports_progress
            {
                model.log(
                    ActivityLevel::Info,
                    "This solver cannot report progress; showing activity only",
                );
            }
        },
        Msg::RunProgress { fraction, t } =>
        {
            // A late progress message must not resurrect a settled run.
            if model.run.is_active()
            {
                // ...and must not undo a cancellation the user has asked for.
                if !matches!(model.run, RunDisplay::Cancelling)
                {
                    model.run = RunDisplay::Determinate { fraction, t };
                    model.log(
                        ActivityLevel::Run,
                        format!("Integrating: {:.0}%", fraction * 100.0),
                    );
                }
            }
        },
        Msg::CancelRequested =>
        {
            if model.run.is_active()
            {
                model.run = RunDisplay::Cancelling;
                model.log(ActivityLevel::Info, "Cancellation requested");
            }
        },
        Msg::RunSettled(outcome) =>
        {
            match &outcome
            {
                Outcome::Completed { run_id } =>
                {
                    model.loaded_run_id = Some(run_id.clone());
                    model.log(ActivityLevel::Pass, format!("Run committed as {run_id}"));
                },
                Outcome::Cancelled => model.log(ActivityLevel::Info, "Run cancelled"),
                Outcome::Failed { message, .. } =>
                {
                    model.log(ActivityLevel::Error, format!("Run failed: {message}"))
                },
                Outcome::Interrupted { detail } =>
                {
                    model.log(ActivityLevel::Error, format!("Run interrupted: {detail}"));
                    // The engine died with it.
                    model.worker_running = false;
                },
            }
            model.run = RunDisplay::Settled(outcome);
        },
        Msg::WorkerState { running, version } =>
        {
            model.worker_running = running;
            model.worker_version = version;
            model.log(
                if running
                {
                    ActivityLevel::Info
                }
                else
                {
                    ActivityLevel::Error
                },
                if running
                {
                    "Calculation engine started"
                }
                else
                {
                    "Calculation engine stopped"
                },
            );
        },
        Msg::RunLoaded { run_id } =>
        {
            model.loaded_run_id = Some(run_id);
            model.view = View::Experiment;
        },
        Msg::ToggleActivity => model.activity_open = !model.activity_open,
        Msg::ToggleInspector => model.inspector_open = !model.inspector_open,
        Msg::TogglePalette =>
        {
            model.palette_open = !model.palette_open;
            if !model.palette_open
            {
                model.palette_query.clear();
            }
        },
        Msg::PaletteQuery(query) => model.palette_query = query,
        Msg::ShowError(error) => model.error = Some(error),
        Msg::DismissError => model.error = None,
        Msg::ClearActivity => model.activity.clear(),
        Msg::Log { level, message } => model.log(level, message),
    }
    model
}

/// The view an action navigates to, when it is a navigation.
pub fn navigation_target(action: Action) -> Option<View> {
    match action
    {
        Action::OpenCatalogue => Some(View::Catalogue),
        Action::OpenRunHistory => Some(View::Runs),
        Action::OpenHelp => Some(View::Help),
        Action::NewExperiment => Some(View::Experiment),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bootstrapped() -> AppModel {
        update(
            AppModel::default(),
            Msg::Bootstrapped {
                worker_running: true,
                worker_version: Some("0.1.0".into()),
                store_path: "/store".into(),
                capability_count: 5,
                interrupted_runs: Vec::new(),
                problem: None,
            },
        )
    }

    #[test]
    fn bootstrap_records_the_engine_and_the_catalogue() {
        let model = bootstrapped();
        assert!(model.worker_running);
        assert_eq!(model.capability_count, 5);
        assert_eq!(model.store_path, "/store");
        assert!(model.error.is_none());
        assert!(!model.activity.is_empty());
    }

    #[test]
    fn a_failed_bootstrap_records_the_reason_and_leaves_navigation_working() {
        let model = update(
            AppModel::default(),
            Msg::Bootstrapped {
                worker_running: false,
                worker_version: None,
                store_path: "/store".into(),
                capability_count: 5,
                interrupted_runs: Vec::new(),
                problem: Some(ErrorLine {
                    code: "WorkerNotFound".into(),
                    title: "The calculation engine is missing".into(),
                    explanation: "…".into(),
                    recoverable: false,
                    suggested_action: Some("Reinstall".into()),
                }),
            },
        );
        assert!(!model.worker_running);
        assert!(model.error.is_some());
        assert_eq!(
            model.capability_count, 5,
            "the catalogue stays visible without an engine"
        );
        assert_eq!(
            crate::actions::availability(Action::OpenHelp, model.action_context()),
            Ok(())
        );
    }

    #[test]
    fn interrupted_runs_are_reported_at_start_up() {
        let model = update(
            AppModel::default(),
            Msg::Bootstrapped {
                worker_running: true,
                worker_version: None,
                store_path: "/s".into(),
                capability_count: 5,
                interrupted_runs: vec!["run-a".into()],
                problem: None,
            },
        );
        assert_eq!(model.interrupted_runs.len(), 1);
        assert!(
            model
                .activity
                .iter()
                .any(|l| l.level == ActivityLevel::Warning)
        );
    }

    #[test]
    fn editing_marks_the_document_dirty() {
        let mut model = bootstrapped();
        assert!(!model.dirty());
        model = update(model, Msg::EditSource("schema_version = 1".into()));
        assert!(model.dirty());
        model = update(model, Msg::Saved { path: None });
        assert!(!model.dirty());
    }

    /// A stale "valid" tick against changed text would be a lie.
    #[test]
    fn editing_invalidates_the_previous_verdict() {
        let mut model = bootstrapped();
        model = update(
            model,
            Msg::Validated {
                valid: true,
                problems: Vec::new(),
            },
        );
        assert!(model.validated_ok);
        model = update(model, Msg::EditSource("changed".into()));
        assert!(!model.validated_ok);
    }

    #[test]
    fn opening_a_scenario_starts_clean_and_shows_the_editor() {
        let model = update(
            bootstrapped(),
            Msg::ScenarioLoaded {
                source: "schema_version = 1".into(),
                path: Some("/tmp/a.toml".into()),
            },
        );
        assert!(!model.dirty());
        assert_eq!(model.view, View::Experiment);
        assert_eq!(model.source_path.as_deref(), Some("/tmp/a.toml"));
    }

    #[test]
    fn a_fixed_step_run_starts_queued_then_reports_a_percentage() {
        let mut model = bootstrapped();
        model = update(
            model,
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: true,
            },
        );
        assert_eq!(model.run, RunDisplay::Queued);
        assert_eq!(model.run.percentage(), None);

        model = update(
            model,
            Msg::RunProgress {
                fraction: 0.25,
                t: 5.0,
            },
        );
        assert_eq!(model.run.percentage(), Some(25.0));
        assert!(model.run.is_active());
    }

    /// The rule that keeps the UI honest about the stiff solver.
    #[test]
    fn an_indeterminate_run_never_offers_a_percentage() {
        let model = update(
            bootstrapped(),
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: false,
            },
        );
        assert_eq!(model.run, RunDisplay::Indeterminate);
        assert_eq!(
            model.run.percentage(),
            None,
            "a solver that cannot report progress must not appear to"
        );
        assert!(model.run.is_active());
    }

    #[test]
    fn cancelling_is_a_distinct_state_before_the_run_settles() {
        let mut model = bootstrapped();
        model = update(
            model,
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: true,
            },
        );
        model = update(model, Msg::CancelRequested);
        assert_eq!(model.run, RunDisplay::Cancelling);
        assert!(model.run.is_active());

        model = update(model, Msg::RunSettled(Outcome::Cancelled));
        assert_eq!(model.run, RunDisplay::Settled(Outcome::Cancelled));
        assert!(!model.run.is_active());
    }

    /// A progress message that arrives after the user asked to cancel must
    /// not put the run back into "running".
    #[test]
    fn late_progress_does_not_undo_a_cancellation() {
        let mut model = bootstrapped();
        model = update(
            model,
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: true,
            },
        );
        model = update(model, Msg::CancelRequested);
        model = update(
            model,
            Msg::RunProgress {
                fraction: 0.9,
                t: 9.0,
            },
        );
        assert_eq!(model.run, RunDisplay::Cancelling, "cancellation must stick");
    }

    #[test]
    fn late_progress_does_not_resurrect_a_settled_run() {
        let mut model = bootstrapped();
        model = update(
            model,
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: true,
            },
        );
        model = update(
            model,
            Msg::RunSettled(Outcome::Completed {
                run_id: "run-1".into(),
            }),
        );
        model = update(
            model,
            Msg::RunProgress {
                fraction: 0.5,
                t: 5.0,
            },
        );
        assert!(matches!(model.run, RunDisplay::Settled(_)));
    }

    #[test]
    fn a_completed_run_becomes_the_displayed_run() {
        let mut model = bootstrapped();
        model = update(
            model,
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: true,
            },
        );
        model = update(
            model,
            Msg::RunSettled(Outcome::Completed {
                run_id: "run-1".into(),
            }),
        );
        assert_eq!(model.loaded_run_id.as_deref(), Some("run-1"));
        assert_eq!(model.run.percentage(), Some(100.0));
        assert_eq!(
            crate::actions::availability(Action::VerifyCurrentRun, model.action_context()),
            Ok(())
        );
    }

    /// Interruption means the engine died: the UI must stop offering Run.
    #[test]
    fn an_interrupted_run_also_marks_the_engine_stopped() {
        let mut model = bootstrapped();
        // A scenario has to be loaded, or Run would be unavailable for that
        // reason instead and the assertion below would prove nothing about
        // the engine.
        model = update(model, Msg::EditSource("schema_version = 1".into()));
        model = update(
            model,
            Msg::RunStarted {
                job_id: "job-1".into(),
                supports_progress: true,
            },
        );
        model = update(
            model,
            Msg::RunSettled(Outcome::Interrupted {
                detail: "worker exited".into(),
            }),
        );
        assert!(!model.worker_running);
        assert_eq!(
            crate::actions::availability(Action::RunExperiment, model.action_context()),
            Err(crate::actions::Unavailable::NoWorker)
        );
        assert_eq!(
            crate::actions::availability(Action::RestartWorker, model.action_context()),
            Ok(())
        );
    }

    /// Cancelled and interrupted must not render the same way.
    #[test]
    fn cancelled_and_interrupted_have_different_labels() {
        let cancelled = RunDisplay::Settled(Outcome::Cancelled);
        let interrupted = RunDisplay::Settled(Outcome::Interrupted { detail: "x".into() });
        assert_ne!(cancelled.label_key(), interrupted.label_key());
    }

    #[test]
    fn each_failure_kind_is_carried_through_for_display() {
        for kind in [
            FailureKind::Validation,
            FailureKind::Numerical,
            FailureKind::Internal,
        ]
        {
            let model = update(
                bootstrapped(),
                Msg::RunSettled(Outcome::Failed {
                    kind,
                    message: "boom".into(),
                }),
            );
            match model.run
            {
                RunDisplay::Settled(Outcome::Failed { kind: k, .. }) => assert_eq!(k, kind),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn language_and_theme_switch() {
        let mut model = bootstrapped();
        model = update(model, Msg::SetLanguage(Language::French));
        assert_eq!(model.language, Language::French);
        model = update(model, Msg::SetTheme(Theme::HighContrast));
        assert_eq!(model.theme.attribute(), "high-contrast");
    }

    #[test]
    fn panels_and_the_palette_toggle() {
        let mut model = bootstrapped();
        let activity = model.activity_open;
        model = update(model, Msg::ToggleActivity);
        assert_ne!(model.activity_open, activity);

        model = update(model, Msg::TogglePalette);
        assert!(model.palette_open);
        model = update(model, Msg::PaletteQuery("run".into()));
        assert_eq!(model.palette_query, "run");

        model = update(model, Msg::TogglePalette);
        assert!(!model.palette_open);
        assert!(
            model.palette_query.is_empty(),
            "closing the palette clears its query"
        );
    }

    #[test]
    fn the_activity_log_is_bounded() {
        let mut model = bootstrapped();
        for i in 0..2000
        {
            model = update(model, Msg::EditSource(format!("x{i}")));
            model = update(
                model,
                Msg::Validated {
                    valid: true,
                    problems: Vec::new(),
                },
            );
        }
        assert!(model.activity.len() <= 500, "{}", model.activity.len());
    }

    #[test]
    fn clearing_the_activity_panel_empties_it() {
        let mut model = bootstrapped();
        assert!(!model.activity.is_empty());
        model = update(model, Msg::ClearActivity);
        assert!(model.activity.is_empty());
    }

    /// The shell's events land in the same bounded log as everything else.
    #[test]
    fn a_logged_line_is_recorded_and_still_bounded() {
        let mut model = bootstrapped();
        model.activity.clear();
        model = update(
            model,
            Msg::Log {
                level: ActivityLevel::Pass,
                message: "stored bytes still match their hashes".into(),
            },
        );
        assert_eq!(model.activity.len(), 1);
        assert_eq!(model.activity[0].level, ActivityLevel::Pass);

        for i in 0..1000
        {
            model = update(
                model,
                Msg::Log {
                    level: ActivityLevel::Info,
                    message: format!("line {i}"),
                },
            );
        }
        assert_eq!(model.activity.len(), 500);
        assert_eq!(model.activity[499].message, "line 999");
    }

    #[test]
    fn errors_are_shown_and_dismissed() {
        let error = ErrorLine {
            code: "Busy".into(),
            title: "A run is already in progress".into(),
            explanation: "…".into(),
            recoverable: true,
            suggested_action: None,
        };
        let mut model = update(bootstrapped(), Msg::ShowError(error.clone()));
        assert_eq!(model.error, Some(error));
        model = update(model, Msg::DismissError);
        assert!(model.error.is_none());
    }

    #[test]
    fn navigation_actions_map_to_views() {
        assert_eq!(
            navigation_target(Action::OpenCatalogue),
            Some(View::Catalogue)
        );
        assert_eq!(navigation_target(Action::OpenRunHistory), Some(View::Runs));
        assert_eq!(navigation_target(Action::OpenHelp), Some(View::Help));
        assert_eq!(navigation_target(Action::CancelRun), None);
    }

    #[test]
    fn every_run_display_has_a_translated_label() {
        use crate::i18n::has_key;
        for display in [
            RunDisplay::Idle,
            RunDisplay::Queued,
            RunDisplay::Determinate {
                fraction: 0.5,
                t: 1.0,
            },
            RunDisplay::Indeterminate,
            RunDisplay::Cancelling,
            RunDisplay::Settled(Outcome::Completed { run_id: "r".into() }),
            RunDisplay::Settled(Outcome::Cancelled),
            RunDisplay::Settled(Outcome::Failed {
                kind: FailureKind::Numerical,
                message: "x".into(),
            }),
            RunDisplay::Settled(Outcome::Interrupted { detail: "x".into() }),
        ]
        {
            assert!(has_key(display.label_key()), "{display:?}");
        }
    }

    /// The panel is a structured log, not a terminal.
    #[test]
    fn activity_tags_are_plain_text() {
        for level in [
            ActivityLevel::Info,
            ActivityLevel::Pass,
            ActivityLevel::Run,
            ActivityLevel::Warning,
            ActivityLevel::Error,
        ]
        {
            let tag = level.tag();
            assert!(!tag.is_empty());
            assert!(!tag.contains('\u{1b}'), "no ANSI escapes: {tag:?}");
        }
    }
}
