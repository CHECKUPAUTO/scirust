//! The frame around the views: title bar, status bar, activity console,
//! inspector, command palette and the error card.

use dioxus::prelude::*;

use crate::actions::{Action, Unavailable, filter_actions, resolve};
use crate::i18n::Language;
use crate::model::{ActivityLevel, ErrorLine, Msg, Theme, View};
use crate::ui::{Ui, progress_label};

/// The application's header: identity, navigation and preferences.
#[component]
pub fn TitleBar() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();

    let tabs = [
        (View::Home, "nav.home"),
        (View::Catalogue, "nav.catalogue"),
        (View::Experiment, "nav.experiment"),
        (View::Runs, "nav.runs"),
        (View::Help, "nav.help"),
    ];

    rsx! {
        header { class: "titlebar",
            div { class: "brand",
                span { class: "brand-mark", "◈" }
                span { class: "brand-name", "{ui.t(\"app.name\")}" }
            }

            nav { class: "tabs", "aria-label": "{ui.t(\"app.name\")}",
                for (view, key) in tabs {
                    button {
                        key: "{key}",
                        class: if model.view == view { "tab tab-active" } else { "tab" },
                        "aria-current": if model.view == view { "page" } else { "false" },
                        onclick: {
                            let ui = ui.clone();
                            move |_| ui.send(Msg::Navigate(view))
                        },
                        "{ui.t(key)}"
                    }
                }
            }

            div { class: "titlebar-tools",
                button {
                    class: "tool",
                    title: "{ui.t(\"palette.title\")} (Ctrl+Shift+P)",
                    onclick: {
                        let ui = ui.clone();
                        move |_| ui.send(Msg::TogglePalette)
                    },
                    "⌘ {ui.t(\"palette.title\")}"
                }

                select {
                    class: "tool",
                    "aria-label": "{ui.t(\"action.change_language\")}",
                    value: "{model.language.tag()}",
                    onchange: {
                        let ui = ui.clone();
                        move |event: FormEvent| {
                            ui.send(Msg::SetLanguage(Language::from_locale(&event.value())));
                        }
                    },
                    for language in Language::all() {
                        option { key: "{language.tag()}", value: "{language.tag()}",
                            "{language.endonym()}"
                        }
                    }
                }

                select {
                    class: "tool",
                    "aria-label": "{ui.t(\"action.change_theme\")}",
                    value: "{model.theme.attribute()}",
                    onchange: {
                        let ui = ui.clone();
                        move |event: FormEvent| {
                            let value = event.value();
                            if let Some(theme) = Theme::all()
                                .into_iter()
                                .find(|t| t.attribute() == value)
                            {
                                ui.send(Msg::SetTheme(theme));
                            }
                        }
                    },
                    for theme in Theme::all() {
                        option { key: "{theme.attribute()}", value: "{theme.attribute()}",
                            "{ui.t(theme_key(theme))}"
                        }
                    }
                }
            }
        }
    }
}

/// The localization key for a theme's name.
fn theme_key(theme: Theme) -> &'static str {
    match theme
    {
        Theme::System => "theme.system",
        Theme::Light => "theme.light",
        Theme::Dark => "theme.dark",
        Theme::HighContrast => "theme.high_contrast",
    }
}

/// The bottom bar: engine, precision, run and store.
///
/// Every claim here is one the application can substantiate. In particular
/// the precision indicator reports the numeric type the engine actually
/// used, not a marketing word.
#[component]
pub fn StatusBar() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();

    let worker = if model.worker_running
    {
        ui.t("status.worker.running")
    }
    else
    {
        ui.t("status.worker.stopped")
    };
    let version = model.worker_version.clone().unwrap_or_default();

    rsx! {
        footer { class: "statusbar",
            span {
                class: if model.worker_running { "status ok" } else { "status bad" },
                "{ui.t(\"status.worker\")}: {worker} {version}"
            }
            span { class: "status", "{ui.t(\"status.backend\")}: Rust / Tauri" }
            span { class: "status", "{ui.t(\"status.precision\")}: f64" }
            span { class: "status", "{ui.t(\"status.job\")}: {progress_label(&ui, &model.run)}" }
            span { class: "status status-path", title: "{model.store_path}",
                "{ui.t(\"status.store\")}: {model.store_path}"
            }
        }
    }
}

/// The activity console, with its typed command input.
///
/// The input accepts the registry's console words and nothing else. It is
/// not a shell: there is no path by which a typed string reaches a process,
/// and [`crate::actions::action_for_console_word`] is the whole vocabulary.
#[component]
pub fn ActivityPanel() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();
    let mut input = use_signal(String::new);

    rsx! {
        section { class: "activity", "aria-label": "{ui.t(\"panel.activity\")}",
            div { class: "activity-head",
                strong { "{ui.t(\"panel.activity\")}" }
                button {
                    class: "link",
                    onclick: {
                        let ui = ui.clone();
                        move |_| ui.send(Msg::ClearActivity)
                    },
                    "{ui.t(\"panel.clear\")}"
                }
            }
            div { class: "activity-lines", role: "log", "aria-live": "polite",
                for (index, line) in model.activity.iter().enumerate() {
                    div { key: "{index}", class: "line line-{level_class(line.level)}",
                        span { class: "line-tag", "{line.level.tag()}" }
                        span { class: "line-text", "{line.message}" }
                    }
                }
            }
            form {
                class: "activity-input",
                onsubmit: {
                    let ui = ui.clone();
                    move |event: FormEvent| {
                        event.prevent_default();
                        let word = input.read().clone();
                        match crate::actions::action_for_console_word(&word) {
                            Some(action) => ui.perform(action),
                            None => ui.send(Msg::Log {
                                level: ActivityLevel::Error,
                                message: format!(
                                    "`{}` is not a command. This console runs named actions only.",
                                    word.trim()
                                ),
                            }),
                        }
                        input.set(String::new());
                    }
                },
                input {
                    r#type: "text",
                    "aria-label": "{ui.t(\"panel.activity\")}",
                    placeholder: "run · validate · cancel · catalogue · runs · help",
                    value: "{input}",
                    oninput: move |event| input.set(event.value()),
                }
            }
        }
    }
}

/// A CSS class fragment for an activity level.
fn level_class(level: ActivityLevel) -> &'static str {
    match level
    {
        ActivityLevel::Info => "info",
        ActivityLevel::Pass => "pass",
        ActivityLevel::Run => "run",
        ActivityLevel::Warning => "warn",
        ActivityLevel::Error => "error",
    }
}

/// The right-hand inspector: the displayed run's metrics, checks and
/// provenance.
#[component]
pub fn Inspector() -> Element {
    let ui = use_context::<Ui>();
    let run = ui.run.read().clone();

    rsx! {
        aside { class: "inspector", "aria-label": "{ui.t(\"panel.metrics\")}",
            match run {
                None => rsx! { p { class: "muted", "{ui.t(\"run.none\")}" } },
                Some(run) => rsx! {
                    h3 { "{run.scenario_name}" }
                    p { class: "muted mono", "{run.run_id}" }

                    h4 { "{ui.t(\"run.integrity\")}" }
                    p {
                        class: if run.integrity.intact { "status ok" } else { "status bad" },
                        if run.integrity.intact {
                            "{ui.t(\"run.integrity.ok\")}"
                        } else {
                            "{ui.t(\"run.integrity.failed\")}: {run.integrity.detail.clone().unwrap_or_default()}"
                        }
                    }

                    h4 { "{ui.t(\"run.metrics\")}" }
                    dl { class: "metrics",
                        for metric in run.metrics.iter() {
                            Fragment { key: "{metric.id}",
                                dt { "{metric.display_name}" }
                                dd { class: "mono",
                                    "{metric.value}"
                                    if let Some(unit) = metric.unit.clone() {
                                        " {unit}"
                                    }
                                }
                            }
                        }
                    }

                    h4 { "{ui.t(\"run.verification\")}" }
                    ul { class: "checks",
                        for check in run.verifications.iter() {
                            li { key: "{check.id}", class: "check check-{check.status}",
                                span { class: "check-status", "{ui.t(status_key(&check.status))}" }
                                span { class: "check-id mono", "{check.id}" }
                                if let Some(measured) = check.measured {
                                    span { class: "mono", " {measured:.3e}" }
                                }
                                p { class: "muted", "{check.explanation}" }
                            }
                        }
                    }

                    if !run.warnings.is_empty() {
                        h4 { "{ui.t(\"run.warnings\")}" }
                        ul {
                            for (index, warning) in run.warnings.iter().enumerate() {
                                li { key: "{index}", "[{warning.category}] {warning.message}" }
                            }
                        }
                    }

                    h4 { "{ui.t(\"run.provenance\")}" }
                    dl { class: "metrics",
                        dt { "Capability" }
                        dd { class: "mono", "{run.provenance.capability_id}" }
                        dt { "Determinism" }
                        dd { "{run.provenance.determinism}" }
                        dt { "Adapter" }
                        dd { class: "mono",
                            "{run.provenance.adapter_crate} {run.provenance.adapter_version}"
                        }
                        dt { "Result schema" }
                        dd { "v{run.provenance.result_schema_version}" }
                        dt { "Target" }
                        dd { class: "mono", "{run.provenance.target}" }
                        dt { "Started" }
                        dd { class: "mono", "{run.provenance.started_at}" }
                        dt { "Elapsed" }
                        dd { class: "mono", "{run.provenance.elapsed_seconds:.3} s" }
                        // Only when one was consumed: a seed shown against a
                        // result that does not depend on it would be a lie
                        // told in a provenance panel.
                        if let Some(seed) = run.provenance.seed {
                            dt { "Seed" }
                            dd { class: "mono", "{seed}" }
                        }
                    }
                },
            }
        }
    }
}

/// The localization key for a verification status.
pub fn status_key(status: &str) -> &'static str {
    match status
    {
        "passed" => "run.status.passed",
        "warning" => "run.status.warning",
        "failed" => "run.status.failed",
        _ => "run.status.not_applicable",
    }
}

/// The command palette.
///
/// Built from the action registry, so it can only offer what the application
/// can actually do, and it shows unavailable actions disabled **with their
/// reason** rather than hiding them.
#[component]
pub fn CommandPalette() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();
    let language = model.language;
    let context = model.action_context();

    let matches = filter_actions(&model.palette_query, move |action| {
        crate::i18n::t(language, action.label_key()).to_string()
    });

    rsx! {
        div { class: "palette-backdrop",
            onclick: {
                let ui = ui.clone();
                move |_| ui.send(Msg::TogglePalette)
            },
            div { class: "palette",
                role: "dialog",
                "aria-modal": "true",
                "aria-label": "{ui.t(\"palette.title\")}",
                onclick: move |event| event.stop_propagation(),

                input {
                    class: "palette-query",
                    r#type: "text",
                    autofocus: true,
                    placeholder: "{ui.t(\"palette.placeholder\")}",
                    value: "{model.palette_query}",
                    oninput: {
                        let ui = ui.clone();
                        move |event: FormEvent| ui.send(Msg::PaletteQuery(event.value()))
                    },
                }

                if matches.is_empty() {
                    p { class: "muted", "{ui.t(\"palette.no_match\")}" }
                }

                ul { class: "palette-list", role: "listbox",
                    for action in matches {
                        PaletteRow { key: "{action:?}", action, available: resolve(action, context) }
                    }
                }
            }
        }
    }
}

/// One row of the palette.
#[component]
fn PaletteRow(action: Action, available: Result<(), Unavailable>) -> Element {
    let ui = use_context::<Ui>();
    let shortcut = action.shortcut().map(|s| s.label()).unwrap_or_default();

    rsx! {
        li { role: "option", "aria-selected": "false",
            button {
                class: if available.is_ok() { "palette-row" } else { "palette-row disabled" },
                disabled: available.is_err(),
                title: match available {
                    Ok(()) => String::new(),
                    Err(reason) => reason.reason().to_string(),
                },
                onclick: {
                    let ui = ui.clone();
                    move |_| {
                        ui.send(Msg::TogglePalette);
                        ui.perform(action);
                    }
                },
                span { class: "palette-label", "{ui.t(action.label_key())}" }
                if let Err(reason) = available {
                    span { class: "palette-reason",
                        "{ui.t(\"palette.unavailable\")}: {reason.reason()}"
                    }
                }
                span { class: "palette-shortcut mono", {shortcut} }
            }
        }
    }
}

/// A blocking error, rendered as a real card.
#[component]
pub fn ErrorCard(error: ErrorLine) -> Element {
    let ui = use_context::<Ui>();
    let diagnostics = format!(
        "code: {}\ntitle: {}\nexplanation: {}\nrecoverable: {}\nsuggested action: {}",
        error.code,
        error.title,
        error.explanation,
        error.recoverable,
        error.suggested_action.clone().unwrap_or_default()
    );

    rsx! {
        div { class: "error-backdrop",
            div { class: "error-card", role: "alertdialog", "aria-label": "{error.title}",
                h2 { "{error.title}" }
                p { class: "mono muted", "{error.code}" }
                p { "{error.explanation}" }
                if let Some(action) = error.suggested_action.clone() {
                    p { class: "suggestion", "→ {action}" }
                }
                div { class: "error-actions",
                    button {
                        class: "primary",
                        onclick: {
                            let ui = ui.clone();
                            move |_| ui.send(Msg::DismissError)
                        },
                        "{ui.t(\"error.dismiss\")}"
                    }
                    button {
                        class: "secondary",
                        onclick: {
                            let ui = ui.clone();
                            let diagnostics = diagnostics.clone();
                            move |_| {
                                // Copying is the only outward path offered
                                // here: the interface has no way to send a
                                // report anywhere, and inventing one would
                                // mean shipping a network capability the
                                // application deliberately does not have.
                                ui.send(Msg::Log {
                                    level: ActivityLevel::Info,
                                    message: diagnostics.clone(),
                                });
                            }
                        },
                        "{ui.t(\"error.copy_diagnostics\")}"
                    }
                }
            }
        }
    }
}
