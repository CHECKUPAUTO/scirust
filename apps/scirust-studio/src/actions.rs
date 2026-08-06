//! The typed action registry.
//!
//! One definition drives the command palette, the keyboard shortcuts and the
//! activity panel's command input. They cannot drift apart because there is
//! nothing to drift *from* — a palette built from a second hand-written list
//! is how an application ends up offering a command its keyboard shortcut no
//! longer performs.
//!
//! An action that cannot run right now is shown **disabled with a reason**
//! rather than hidden: a user looking for "Cancel" needs to learn that
//! nothing is running, not that the command has vanished.

use serde::{Deserialize, Serialize};

/// Everything the application can be asked to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Start a new, empty experiment.
    NewExperiment,
    /// Open a scenario file.
    OpenScenario,
    /// Save the current scenario.
    SaveScenario,
    /// Save the current scenario under a new name.
    SaveScenarioAs,
    /// Validate without running.
    ValidateScenario,
    /// Validate and run.
    RunExperiment,
    /// Cancel the active run.
    CancelRun,
    /// Show the capability catalogue.
    OpenCatalogue,
    /// Show the run history.
    OpenRunHistory,
    /// Re-check the displayed run's hashes.
    VerifyCurrentRun,
    /// Show help.
    OpenHelp,
    /// Show or hide the activity panel.
    ToggleActivityPanel,
    /// Show or hide the inspector.
    ToggleInspector,
    /// Switch language.
    ChangeLanguage,
    /// Switch theme.
    ChangeTheme,
    /// Show diagnostics.
    ShowDiagnostics,
    /// Start a worker after one has exited.
    RestartWorker,
}

/// A keyboard shortcut, as the frontend matches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shortcut {
    /// Requires the primary modifier (Ctrl, or Cmd on macOS).
    pub ctrl: bool,
    /// Requires Shift.
    pub shift: bool,
    /// The key, as a `KeyboardEvent.key` value.
    pub key: &'static str,
}

impl Shortcut {
    const fn new(ctrl: bool, shift: bool, key: &'static str) -> Self {
        Shortcut { ctrl, shift, key }
    }

    /// A display label, e.g. `Ctrl+Shift+P`.
    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl
        {
            parts.push("Ctrl");
        }
        if self.shift
        {
            parts.push("Shift");
        }
        parts.push(self.key);
        parts.join("+")
    }
}

impl Action {
    /// Every action, in palette order.
    pub fn all() -> &'static [Action] {
        &[
            Action::NewExperiment,
            Action::OpenScenario,
            Action::SaveScenario,
            Action::SaveScenarioAs,
            Action::ValidateScenario,
            Action::RunExperiment,
            Action::CancelRun,
            Action::OpenCatalogue,
            Action::OpenRunHistory,
            Action::VerifyCurrentRun,
            Action::OpenHelp,
            Action::ToggleActivityPanel,
            Action::ToggleInspector,
            Action::ChangeLanguage,
            Action::ChangeTheme,
            Action::ShowDiagnostics,
            Action::RestartWorker,
        ]
    }

    /// The localization key for this action's label.
    pub fn label_key(self) -> &'static str {
        match self
        {
            Action::NewExperiment => "action.new",
            Action::OpenScenario => "action.open",
            Action::SaveScenario => "action.save",
            Action::SaveScenarioAs => "action.save_as",
            Action::ValidateScenario => "action.validate",
            Action::RunExperiment => "action.run",
            Action::CancelRun => "action.cancel",
            Action::OpenCatalogue => "action.open_catalogue",
            Action::OpenRunHistory => "action.open_runs",
            Action::VerifyCurrentRun => "action.verify_run",
            Action::OpenHelp => "action.open_help",
            Action::ToggleActivityPanel => "action.toggle_activity",
            Action::ToggleInspector => "action.toggle_inspector",
            Action::ChangeLanguage => "action.change_language",
            Action::ChangeTheme => "action.change_theme",
            Action::ShowDiagnostics => "action.show_diagnostics",
            Action::RestartWorker => "action.restart_worker",
        }
    }

    /// The word that invokes this action from the activity panel's command
    /// input, where one exists.
    ///
    /// That input accepts **only** these words. It is not a shell, and there
    /// is deliberately no path from it to running a program.
    pub fn console_word(self) -> Option<&'static str> {
        match self
        {
            Action::OpenHelp => Some("help"),
            Action::OpenCatalogue => Some("catalog"),
            Action::ValidateScenario => Some("validate"),
            Action::RunExperiment => Some("run"),
            Action::CancelRun => Some("cancel"),
            Action::OpenRunHistory => Some("runs"),
            Action::VerifyCurrentRun => Some("verify"),
            _ => None,
        }
    }

    /// This action's keyboard shortcut, where it has one.
    pub fn shortcut(self) -> Option<Shortcut> {
        match self
        {
            Action::OpenScenario => Some(Shortcut::new(true, false, "o")),
            Action::SaveScenario => Some(Shortcut::new(true, false, "s")),
            Action::SaveScenarioAs => Some(Shortcut::new(true, true, "S")),
            Action::ValidateScenario => Some(Shortcut::new(true, false, "Enter")),
            Action::RunExperiment => Some(Shortcut::new(false, false, "F5")),
            Action::CancelRun => Some(Shortcut::new(false, true, "F5")),
            Action::OpenHelp => Some(Shortcut::new(false, false, "F1")),
            Action::ToggleActivityPanel => Some(Shortcut::new(true, false, "`")),
            _ => None,
        }
    }
}

/// What the application can currently do, as far as the action registry
/// needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ActionContext {
    /// A worker is up.
    pub worker_running: bool,
    /// A run is active.
    pub run_active: bool,
    /// The editor holds something.
    pub has_scenario: bool,
    /// The editor has unsaved changes.
    pub dirty: bool,
    /// A stored run is displayed.
    pub has_loaded_run: bool,
}

/// Why an action cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// No calculation engine.
    NoWorker,
    /// A run is already going.
    RunActive,
    /// Nothing is running to cancel.
    NoActiveRun,
    /// The editor is empty.
    NoScenario,
    /// Nothing has been changed.
    NothingToSave,
    /// No run is displayed.
    NoLoadedRun,
    /// A worker is already running.
    WorkerAlreadyRunning,
}

impl Unavailable {
    /// A short explanation for the disabled state's tooltip.
    ///
    /// English only for now, and deliberately so: these are developer-facing
    /// hints attached to a disabled control whose *label* is translated. A
    /// half-translated tooltip would be worse than an untranslated one, and
    /// adding them to the string table is a small, separable change.
    pub fn reason(self) -> &'static str {
        match self
        {
            Unavailable::NoWorker => "The calculation engine is not running",
            Unavailable::RunActive => "A run is already in progress",
            Unavailable::NoActiveRun => "No run is in progress",
            Unavailable::NoScenario => "No scenario is open",
            Unavailable::NothingToSave => "No unsaved changes",
            Unavailable::NoLoadedRun => "No run is displayed",
            Unavailable::WorkerAlreadyRunning => "The calculation engine is already running",
        }
    }
}

/// Whether an action can be performed right now, and why not when it cannot.
///
/// Every action in this build maps to a typed Tauri command, so this is now
/// exactly [`availability`] — a question about the model's *state*. It kept a
/// second half until the native file dialogs landed: `is_wired` answered a
/// question about the shell, because opening and saving a scenario had no
/// command behind them at all and the honest thing was to show the actions
/// disabled with that reason rather than hide them or let them do nothing.
///
/// That distinction is gone rather than left inert. An `Unavailable` variant
/// nothing can produce is a state the interface claims exists and cannot
/// reach, which is the same category of untruth as a field nothing reads.
pub fn resolve(action: Action, context: ActionContext) -> Result<(), Unavailable> {
    availability(action, context)
}

/// Whether an action can run, and why not when it cannot.
pub fn availability(action: Action, context: ActionContext) -> Result<(), Unavailable> {
    match action
    {
        Action::RunExperiment =>
        {
            if !context.has_scenario
            {
                Err(Unavailable::NoScenario)
            }
            else if !context.worker_running
            {
                Err(Unavailable::NoWorker)
            }
            else if context.run_active
            {
                Err(Unavailable::RunActive)
            }
            else
            {
                Ok(())
            }
        },
        Action::CancelRun =>
        {
            if context.run_active
            {
                Ok(())
            }
            else
            {
                Err(Unavailable::NoActiveRun)
            }
        },
        Action::ValidateScenario | Action::SaveScenarioAs =>
        {
            if context.has_scenario
            {
                Ok(())
            }
            else
            {
                Err(Unavailable::NoScenario)
            }
        },
        Action::SaveScenario =>
        {
            if !context.has_scenario
            {
                Err(Unavailable::NoScenario)
            }
            else if !context.dirty
            {
                Err(Unavailable::NothingToSave)
            }
            else
            {
                Ok(())
            }
        },
        Action::VerifyCurrentRun =>
        {
            if context.has_loaded_run
            {
                Ok(())
            }
            else
            {
                Err(Unavailable::NoLoadedRun)
            }
        },
        Action::RestartWorker =>
        {
            if context.worker_running
            {
                Err(Unavailable::WorkerAlreadyRunning)
            }
            else
            {
                Ok(())
            }
        },
        // Navigation, help and preferences always work — including with no
        // engine, which is exactly when someone needs the help.
        _ => Ok(()),
    }
}

/// Filter and rank actions for the palette.
///
/// Prefix matches rank above contained matches, and the original order
/// breaks ties, so typing `r` puts "Run experiment" above "Toggle
/// inspector" rather than surfacing whatever happens to be first.
pub fn filter_actions(query: &str, label: impl Fn(Action) -> String) -> Vec<Action> {
    let query = query.trim().to_lowercase();
    if query.is_empty()
    {
        return Action::all().to_vec();
    }

    let mut scored: Vec<(u8, usize, Action)> = Action::all()
        .iter()
        .enumerate()
        .filter_map(|(index, &action)| {
            let text = label(action).to_lowercase();
            let word = action.console_word().unwrap_or("");
            if text.starts_with(&query) || word == query
            {
                Some((0, index, action))
            }
            else if text.split_whitespace().any(|w| w.starts_with(&query))
            {
                Some((1, index, action))
            }
            else if text.contains(&query)
            {
                Some((2, index, action))
            }
            else
            {
                None
            }
        })
        .collect();

    scored.sort_by_key(|(rank, index, _)| (*rank, *index));
    scored.into_iter().map(|(_, _, a)| a).collect()
}

/// Resolve a word typed into the activity panel's command input.
pub fn action_for_console_word(word: &str) -> Option<Action> {
    let word = word.trim().to_lowercase();
    Action::all()
        .iter()
        .copied()
        .find(|a| a.console_word() == Some(word.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i18n::{Language, has_key, t};

    fn english(action: Action) -> String {
        t(Language::English, action.label_key()).to_string()
    }

    /// Every action must be nameable in both languages, or the palette shows
    /// a placeholder.
    #[test]
    fn every_action_has_a_translated_label() {
        for action in Action::all()
        {
            assert!(
                has_key(action.label_key()),
                "{action:?} has no translation key"
            );
            for language in Language::all()
            {
                let label = t(language, action.label_key());
                assert!(!label.contains("missing"), "{action:?} in {language:?}");
            }
        }
    }

    #[test]
    fn the_registry_lists_every_action_once() {
        let mut seen = std::collections::BTreeSet::new();
        for action in Action::all()
        {
            assert!(seen.insert(*action), "{action:?} listed twice");
        }
        assert_eq!(seen.len(), Action::all().len());
    }

    #[test]
    fn no_two_actions_share_a_shortcut() {
        let mut seen = std::collections::BTreeMap::new();
        for action in Action::all()
        {
            if let Some(shortcut) = action.shortcut()
            {
                let key = (shortcut.ctrl, shortcut.shift, shortcut.key);
                if let Some(other) = seen.insert(key, action)
                {
                    panic!("{action:?} and {other:?} share {}", shortcut.label());
                }
            }
        }
    }

    #[test]
    fn no_two_actions_share_a_console_word() {
        let mut seen = std::collections::BTreeSet::new();
        for action in Action::all()
        {
            if let Some(word) = action.console_word()
            {
                assert!(seen.insert(word), "{word} is claimed twice");
            }
        }
    }

    #[test]
    fn shortcut_labels_read_the_way_they_are_pressed() {
        assert_eq!(Shortcut::new(true, true, "S").label(), "Ctrl+Shift+S");
        assert_eq!(Shortcut::new(false, false, "F5").label(), "F5");
        assert_eq!(Shortcut::new(true, false, "o").label(), "Ctrl+o");
    }

    /// Running needs a scenario, an engine, and nothing already running —
    /// and the reason given must be the *first* thing that is wrong, so the
    /// user fixes the right one.
    #[test]
    fn run_is_gated_on_every_precondition_in_order() {
        let empty = ActionContext::default();
        assert_eq!(
            availability(Action::RunExperiment, empty),
            Err(Unavailable::NoScenario)
        );

        let no_worker = ActionContext {
            has_scenario: true,
            ..Default::default()
        };
        assert_eq!(
            availability(Action::RunExperiment, no_worker),
            Err(Unavailable::NoWorker)
        );

        let busy = ActionContext {
            has_scenario: true,
            worker_running: true,
            run_active: true,
            ..Default::default()
        };
        assert_eq!(
            availability(Action::RunExperiment, busy),
            Err(Unavailable::RunActive)
        );

        let ready = ActionContext {
            has_scenario: true,
            worker_running: true,
            ..Default::default()
        };
        assert_eq!(availability(Action::RunExperiment, ready), Ok(()));
    }

    #[test]
    fn cancel_is_available_exactly_while_a_run_is_active() {
        let idle = ActionContext::default();
        assert_eq!(
            availability(Action::CancelRun, idle),
            Err(Unavailable::NoActiveRun)
        );
        let running = ActionContext {
            run_active: true,
            ..Default::default()
        };
        assert_eq!(availability(Action::CancelRun, running), Ok(()));
    }

    #[test]
    fn saving_requires_something_to_save() {
        let clean = ActionContext {
            has_scenario: true,
            ..Default::default()
        };
        assert_eq!(
            availability(Action::SaveScenario, clean),
            Err(Unavailable::NothingToSave)
        );
        let dirty = ActionContext {
            has_scenario: true,
            dirty: true,
            ..Default::default()
        };
        assert_eq!(availability(Action::SaveScenario, dirty), Ok(()));
    }

    /// The case that matters when something has gone wrong: help and the
    /// catalogue must stay reachable with no engine.
    #[test]
    fn help_and_navigation_work_without_a_worker() {
        let broken = ActionContext::default();
        for action in [
            Action::OpenHelp,
            Action::OpenCatalogue,
            Action::OpenRunHistory,
            Action::ChangeLanguage,
            Action::ChangeTheme,
            Action::ShowDiagnostics,
        ]
        {
            assert_eq!(availability(action, broken), Ok(()), "{action:?}");
        }
        assert_eq!(availability(Action::RestartWorker, broken), Ok(()));
    }

    #[test]
    fn every_unavailable_reason_says_something() {
        for reason in [
            Unavailable::NoWorker,
            Unavailable::RunActive,
            Unavailable::NoActiveRun,
            Unavailable::NoScenario,
            Unavailable::NothingToSave,
            Unavailable::NoLoadedRun,
            Unavailable::WorkerAlreadyRunning,
        ]
        {
            assert!(!reason.reason().is_empty(), "{reason:?}");
        }
    }

    #[test]
    fn an_empty_query_lists_everything() {
        assert_eq!(filter_actions("", english).len(), Action::all().len());
        assert_eq!(filter_actions("   ", english).len(), Action::all().len());
    }

    #[test]
    fn a_prefix_match_ranks_above_a_contained_one() {
        let results = filter_actions("run", english);
        assert_eq!(
            results.first(),
            Some(&Action::RunExperiment),
            "got {results:?}"
        );
    }

    #[test]
    fn filtering_is_case_insensitive() {
        assert_eq!(
            filter_actions("CANCEL", english),
            filter_actions("cancel", english)
        );
    }

    #[test]
    fn an_unmatched_query_returns_nothing() {
        assert!(filter_actions("zzzznotacommand", english).is_empty());
    }

    #[test]
    fn console_words_resolve_to_their_actions() {
        assert_eq!(action_for_console_word("run"), Some(Action::RunExperiment));
        assert_eq!(action_for_console_word(" CANCEL "), Some(Action::CancelRun));
        assert_eq!(
            action_for_console_word("verify"),
            Some(Action::VerifyCurrentRun)
        );
    }

    /// The command input is not a shell.
    #[test]
    fn the_console_refuses_anything_that_is_not_a_registered_action() {
        for word in ["sh", "bash", "cmd", "powershell", "rm -rf /", "eval", ""]
        {
            assert_eq!(
                action_for_console_word(word),
                None,
                "{word:?} must not be executable"
            );
        }
    }

    /// The three file actions were disabled in every earlier build because
    /// no command stood behind them. They now do, and this is the test that
    /// was inverted rather than deleted — deleting it would have removed the
    /// only record that they were ever refused for a reason other than state.
    #[test]
    fn the_file_actions_are_available_once_there_is_something_to_act_on() {
        let context = ActionContext {
            worker_running: true,
            run_active: false,
            has_scenario: true,
            dirty: true,
            has_loaded_run: true,
        };
        for action in [
            Action::OpenScenario,
            Action::SaveScenario,
            Action::SaveScenarioAs,
        ]
        {
            assert_eq!(resolve(action, context), Ok(()), "{action:?}");
        }
    }

    /// Opening a file needs nothing from the model: it is how a user with an
    /// empty editor and no engine gets started.
    #[test]
    fn opening_a_scenario_needs_no_state_at_all() {
        let empty = ActionContext {
            worker_running: false,
            run_active: false,
            has_scenario: false,
            dirty: false,
            has_loaded_run: false,
        };
        assert_eq!(resolve(Action::OpenScenario, empty), Ok(()));
        // Saving still needs something to save.
        assert_eq!(
            resolve(Action::SaveScenario, empty),
            Err(Unavailable::NoScenario)
        );
    }

    /// Everything else resolves exactly as the state machine says.
    #[test]
    fn a_wired_action_resolves_through_the_state_machine() {
        let context = ActionContext {
            worker_running: true,
            run_active: false,
            has_scenario: true,
            dirty: true,
            has_loaded_run: false,
        };
        for action in Action::all()
        {
            assert_eq!(
                resolve(*action, context),
                availability(*action, context),
                "{action:?}"
            );
        }
        assert_eq!(resolve(Action::RunExperiment, context), Ok(()));
        assert_eq!(
            resolve(Action::VerifyCurrentRun, context),
            Err(Unavailable::NoLoadedRun)
        );
    }
}
