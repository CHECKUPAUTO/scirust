//! The five views.

use dioxus::prelude::*;

use crate::actions::{Action, resolve};
use crate::model::{Msg, RunDisplay, View};
use crate::ui::chart_view::{Chart, FieldMaps};
use crate::ui::parts::status_key;
use crate::ui::{Ui, progress_label};

/// The landing page: what this application is, and the two ways in.
#[component]
pub fn Home() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();
    let catalog = ui.catalog.read().clone();
    let runs = ui.runs.read().clone();

    rsx! {
        section { class: "view view-home",
            h1 { "{ui.t(\"home.welcome\")}" }
            p { class: "tagline", "{ui.t(\"app.tagline\")}" }

            if !model.interrupted_runs.is_empty() {
                div { class: "notice notice-warn", role: "alert",
                    strong { "{ui.t(\"runs.interrupted\")}" }
                    ul {
                        for run_id in model.interrupted_runs.iter() {
                            li { key: "{run_id}", class: "mono", {run_id.clone()} }
                        }
                    }
                }
            }

            div { class: "home-columns",
                div { class: "card",
                    h2 { "{ui.t(\"home.tutorials\")}" }
                    p { class: "muted", "{model.capability_count} {ui.t(\"catalogue.executable\")}" }
                    ul { class: "tutorial-list",
                        for capability in catalog.iter().filter(|c| c.has_tutorial) {
                            li { key: "{capability.id}",
                                div { class: "tutorial-row",
                                    div {
                                        strong { "{capability.display_name}" }
                                        p { class: "muted", "{capability.summary}" }
                                    }
                                    button {
                                        class: "primary",
                                        onclick: {
                                            let ui = ui.clone();
                                            let id = capability.id.clone();
                                            move |_| {
                                                let ui = ui.clone();
                                                let id = id.clone();
                                                spawn(async move { ui.open_tutorial(id).await });
                                            }
                                        },
                                        "{ui.t(\"home.open_tutorial\")}"
                                    }
                                }
                            }
                        }
                    }
                }

                div { class: "card",
                    h2 { "{ui.t(\"home.recent_runs\")}" }
                    if runs.is_empty() {
                        p { class: "muted", "{ui.t(\"home.no_runs\")}" }
                    }
                    ul { class: "run-list",
                        for run in runs.iter().rev().take(6) {
                            li { key: "{run.run_id}",
                                button {
                                    class: "link mono",
                                    onclick: {
                                        let ui = ui.clone();
                                        let id = run.run_id.clone();
                                        move |_| {
                                            let ui = ui.clone();
                                            let id = id.clone();
                                            spawn(async move { ui.open_run(id).await });
                                        }
                                    },
                                    "{run.run_id}"
                                }
                                span { class: "muted", " {run.capability_id} · {run.status}" }
                            }
                        }
                    }
                    p { class: "muted mono", title: "{model.store_path}",
                        "{ui.t(\"home.reveal_store\")}: {model.store_path}"
                    }
                }
            }
        }
    }
}

/// The capability catalogue, from the real registry.
#[component]
pub fn Catalogue() -> Element {
    let ui = use_context::<Ui>();
    let catalog = ui.catalog.read().clone();
    let query = ui.catalogue_query.read().clone();
    let needle = query.trim().to_lowercase();

    let shown: Vec<_> = catalog
        .iter()
        .filter(|c| {
            needle.is_empty()
                || c.display_name.to_lowercase().contains(&needle)
                || c.id.to_lowercase().contains(&needle)
                || c.category.to_lowercase().contains(&needle)
                || c.summary.to_lowercase().contains(&needle)
        })
        .cloned()
        .collect();

    rsx! {
        section { class: "view view-catalogue",
            header { class: "view-head",
                h1 { "{ui.t(\"nav.catalogue\")}" }
                input {
                    r#type: "search",
                    class: "search",
                    placeholder: "{ui.t(\"catalogue.search\")}",
                    value: query.clone(),
                    oninput: {
                        let ui = ui.clone();
                        move |event: FormEvent| {
                            let mut q = ui.catalogue_query;
                            q.set(event.value());
                        }
                    },
                }
            }

            if shown.is_empty() {
                p { class: "muted", "{ui.t(\"catalogue.no_match\")}" }
            }

            div { class: "capability-grid",
                for capability in shown {
                    article { key: "{capability.id}", class: "card capability",
                        header {
                            h2 { "{capability.display_name}" }
                            p { class: "mono muted", "{capability.id}" }
                        }
                        p { "{capability.summary}" }
                        ul { class: "chips",
                            li { class: "chip", "{capability.category}" }
                            li { class: "chip", "{capability.maturity}" }
                            li { class: "chip", "{capability.determinism}" }
                            li { class: "chip mono", "{capability.source_crate}" }
                            if !capability.supports_progress {
                                li { class: "chip chip-warn", "indeterminate progress" }
                            }
                        }

                        details {
                            summary { "{ui.t(\"catalogue.parameters\")}" }
                            ul {
                                for field in capability.parameters.iter() {
                                    li { key: "{field.name}",
                                        span { class: "mono", "{field.name}" }
                                        " — {field.description} "
                                        span { class: "muted", "[{field.units.join(\", \")}]" }
                                    }
                                }
                            }
                        }
                        details {
                            summary { "{ui.t(\"catalogue.initial_state\")}" }
                            ul {
                                for field in capability.initial_state.iter() {
                                    li { key: "{field.name}",
                                        span { class: "mono", "{field.name}" }
                                        " — {field.description}"
                                    }
                                }
                            }
                        }
                        details {
                            summary { "{ui.t(\"catalogue.outputs\")}" }
                            ul {
                                for output in capability.outputs.iter() {
                                    li { key: "{output.id}",
                                        span { class: "mono", "{output.id}" }
                                        " — {output.display_name} ({output.unit})"
                                    }
                                }
                            }
                        }
                        details {
                            summary { "{ui.t(\"catalogue.checks\")}" }
                            ul {
                                for check in capability.checks.iter() {
                                    li { key: "{check.id}",
                                        span { class: "mono", "{check.id}" }
                                        " — {check.description}"
                                    }
                                }
                            }
                        }

                        footer { class: "capability-foot",
                            span { class: "muted",
                                "{ui.t(\"catalogue.solvers\")}: {capability.solvers.join(\", \")}"
                            }
                            if capability.has_tutorial {
                                button {
                                    class: "primary",
                                    onclick: {
                                        let ui = ui.clone();
                                        let id = capability.id.clone();
                                        move |_| {
                                            let ui = ui.clone();
                                            let id = id.clone();
                                            spawn(async move { ui.open_tutorial(id).await });
                                        }
                                    },
                                    "{ui.t(\"catalogue.open_example\")}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The editor, the run controls and the chart.
#[component]
pub fn Experiment() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();
    let context = model.action_context();
    let can_run = resolve(Action::RunExperiment, context);
    let can_cancel = resolve(Action::CancelRun, context);
    let can_validate = resolve(Action::ValidateScenario, context);

    rsx! {
        section { class: "view view-experiment",
            header { class: "view-head",
                h1 {
                    "{ui.t(\"editor.title\")}"
                    if model.dirty() {
                        span { class: "dirty", " · {ui.t(\"editor.unsaved\")}" }
                    }
                }
                div { class: "run-controls",
                    button {
                        class: "secondary",
                        disabled: can_validate.is_err(),
                        title: can_validate.err().map(|r| r.reason()).unwrap_or_default(),
                        onclick: {
                            let ui = ui.clone();
                            move |_| ui.perform(Action::ValidateScenario)
                        },
                        "{ui.t(\"action.validate\")}"
                    }
                    button {
                        class: "primary",
                        disabled: can_run.is_err(),
                        title: can_run.err().map(|r| r.reason()).unwrap_or_default(),
                        onclick: {
                            let ui = ui.clone();
                            move |_| ui.perform(Action::RunExperiment)
                        },
                        "{ui.t(\"action.run\")}"
                    }
                    button {
                        class: "danger",
                        disabled: can_cancel.is_err(),
                        title: can_cancel.err().map(|r| r.reason()).unwrap_or_default(),
                        onclick: {
                            let ui = ui.clone();
                            move |_| ui.perform(Action::CancelRun)
                        },
                        "{ui.t(\"action.cancel\")}"
                    }
                }
            }

            RunProgress {}

            div { class: "editor-columns",
                div { class: "editor-pane",
                    textarea {
                        class: "editor mono",
                        spellcheck: false,
                        "aria-label": "{ui.t(\"editor.title\")}",
                        value: "{model.source}",
                        oninput: {
                            let ui = ui.clone();
                            move |event: FormEvent| ui.send(Msg::EditSource(event.value()))
                        },
                    }
                }

                div { class: "problems-pane",
                    h3 { "{ui.t(\"editor.problems\")}" }
                    if model.problems.is_empty() {
                        p {
                            class: if model.validated_ok { "status ok" } else { "muted" },
                            if model.validated_ok {
                                "{ui.t(\"editor.validated\")}"
                            } else {
                                "{ui.t(\"editor.no_problems\")}"
                            }
                        }
                    }
                    ul { class: "problems",
                        for (index, problem) in model.problems.iter().enumerate() {
                            li { key: "{index}", class: "problem",
                                header {
                                    strong { "{problem.title}" }
                                    if let Some(code) = problem.code.clone() {
                                        span { class: "mono muted", " {code}" }
                                    }
                                }
                                p { "{problem.explanation}" }
                                if let Some(fix) = problem.suggested_action.clone() {
                                    p { class: "suggestion", "→ {fix}" }
                                }
                                p { class: "muted mono",
                                    "{problem.stage}"
                                    if let Some(field) = problem.field.clone() {
                                        " · {field}"
                                    }
                                    if let Some(line) = problem.line {
                                        " · {ui.t(\"editor.line\")} {line}"
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Chart {}
            // Only renders for a run that has fields; for the heat rod the
            // field is the result and the curves beside it are summaries.
            FieldMaps {}
        }
    }
}

/// The progress control for the active run.
///
/// A capability that cannot report progress gets indeterminate activity, not
/// a percentage — the whole reason [`RunDisplay`] distinguishes the two.
#[component]
fn RunProgress() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();

    if matches!(model.run, RunDisplay::Idle)
    {
        return rsx! {};
    }

    rsx! {
        div { class: "run-progress", role: "status", "aria-live": "polite",
            span { class: "run-state", "{progress_label(&ui, &model.run)}" }
            match model.run {
                RunDisplay::Determinate { fraction, t } => rsx! {
                    progress { max: "1", value: "{fraction}" }
                    span { class: "mono muted", " t = {t:.6}" }
                },
                RunDisplay::Indeterminate | RunDisplay::Queued | RunDisplay::Cancelling => rsx! {
                    progress {}
                },
                _ => rsx! {},
            }
            if let RunDisplay::Settled(outcome) = &model.run {
                span { class: "run-outcome", "{outcome_text(&ui, outcome)}" }
            }
        }
    }
}

/// A one-line rendering of how a run ended.
fn outcome_text(ui: &Ui, outcome: &crate::model::Outcome) -> String {
    use crate::model::Outcome;
    match outcome
    {
        Outcome::Completed { run_id } => format!("{} · {run_id}", ui.t("run.state.completed")),
        Outcome::Cancelled => ui.t("run.state.cancelled").to_string(),
        Outcome::Failed { message, .. } => format!("{}: {message}", ui.t("run.state.failed")),
        Outcome::Interrupted { detail } =>
        {
            format!("{}: {detail}", ui.t("run.state.interrupted"))
        },
    }
}

/// The stored-run history.
#[component]
pub fn Runs() -> Element {
    let ui = use_context::<Ui>();
    let runs = ui.runs.read().clone();
    let model = ui.model.read().clone();

    rsx! {
        section { class: "view view-runs",
            header { class: "view-head",
                h1 { "{ui.t(\"runs.title\")}" }
                span { class: "muted mono", "{model.store_path}" }
            }

            if runs.is_empty() {
                p { class: "muted", "{ui.t(\"runs.none\")}" }
            }

            table { class: "runs-table",
                thead {
                    tr {
                        th { "{ui.t(\"runs.title\")}" }
                        th { "{ui.t(\"nav.catalogue\")}" }
                        th { "{ui.t(\"status.job\")}" }
                        th { "{ui.t(\"runs.schema\")}" }
                        th { }
                    }
                }
                tbody {
                    for run in runs.iter().rev() {
                        tr { key: "{run.run_id}",
                            td { class: "mono", "{run.run_id}" }
                            td { class: "mono", "{run.capability_id}" }
                            td { "{run.status}" }
                            td {
                                "v{run.result_schema_version}"
                                if run.result_schema_version < 2 {
                                    span { class: "chip chip-warn", " {ui.t(\"runs.legacy\")}" }
                                }
                            }
                            td { class: "row-actions",
                                button {
                                    class: "secondary",
                                    onclick: {
                                        let ui = ui.clone();
                                        let id = run.run_id.clone();
                                        move |_| {
                                            let ui = ui.clone();
                                            let id = id.clone();
                                            spawn(async move { ui.open_run(id).await });
                                        }
                                    },
                                    "{ui.t(\"runs.open\")}"
                                }
                                button {
                                    class: "secondary",
                                    onclick: {
                                        let ui = ui.clone();
                                        let id = run.run_id.clone();
                                        move |_| {
                                            let ui = ui.clone();
                                            let id = id.clone();
                                            spawn(async move {
                                                ui.open_run(id).await;
                                                ui.verify_displayed_run().await;
                                            });
                                        }
                                    },
                                    "{ui.t(\"runs.verify\")}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Help for what this preview actually does.
///
/// Deliberately scoped: it documents the features that exist, and says so,
/// rather than describing an application that has not been built yet.
#[component]
pub fn Help() -> Element {
    let ui = use_context::<Ui>();
    let model = ui.model.read().clone();
    let context = model.action_context();
    let run = ui.run.read().clone();

    rsx! {
        section { class: "view view-help",
            h1 { "{ui.t(\"help.title\")}" }
            p { class: "notice", "{ui.t(\"help.preview_notice\")}" }

            h2 { "{ui.t(\"help.workflow\")}" }
            ol { class: "workflow",
                li { "{ui.t(\"home.tutorials\")} → {ui.t(\"home.open_tutorial\")}" }
                li { "{ui.t(\"action.validate\")}" }
                li { "{ui.t(\"action.run\")} → {ui.t(\"action.cancel\")}" }
                li { "{ui.t(\"nav.runs\")} → {ui.t(\"runs.open\")} → {ui.t(\"runs.verify\")}" }
                li { "{ui.t(\"chart.title\")} → {ui.t(\"chart.table_view\")}" }
            }

            h2 { "{ui.t(\"help.shortcuts\")}" }
            table { class: "shortcuts",
                tbody {
                    for action in Action::all() {
                        if let Some(shortcut) = action.shortcut() {
                            tr { key: "{action:?}",
                                td { class: "mono", "{shortcut.label()}" }
                                td { "{ui.t(action.label_key())}" }
                                td { class: "muted",
                                    match resolve(*action, context) {
                                        Ok(()) => String::new(),
                                        Err(reason) => reason.reason().to_string(),
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(run) = run {
                h2 { "{ui.t(\"run.verification\")}" }
                ul {
                    for check in run.verifications.iter() {
                        li { key: "{check.id}",
                            span { class: "mono", "{check.id}" }
                            " — {ui.t(status_key(&check.status))}"
                        }
                    }
                }
            }

            h2 { "{ui.t(\"panel.activity\")}" }
            p {
                "The activity console runs named actions only — "
                span { class: "mono", "run" }
                ", "
                span { class: "mono", "validate" }
                ", "
                span { class: "mono", "cancel" }
                ". It is not a shell, and there is no path from it to a process."
            }

            h2 { "{ui.t(\"nav.experiment\")}" }
            p {
                "A result stored before coordinates were recorded (schema v1) is plotted "
                "against sample ordinals and labelled as such. The application will not "
                "invent an axis for it."
            }

            p { class: "muted mono",
                "interface {crate::VERSION} · view {model.view:?} · {View::Help:?}"
            }
        }
    }
}
