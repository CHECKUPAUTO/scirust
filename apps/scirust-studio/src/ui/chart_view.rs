//! The chart, drawn as SVG from [`crate::chart`]'s geometry.
//!
//! Two things this component is careful about, both of which are about not
//! lying:
//!
//! * A legacy (v1) result has no stored coordinates. It is plotted against
//!   sample ordinals and the axis says so, prominently — not in a tooltip.
//! * A long series is reduced for display by min/max bucketing, and the
//!   caption states how many points of how many are drawn, so nobody reads
//!   a smoothed line as the data.
//!
//! A text alternative is always rendered alongside: a chart that exists only
//! as a picture cannot be read by someone using a screen reader, and cannot
//! be checked by a reviewer at all.

use std::collections::BTreeSet;

use dioxus::prelude::*;

use crate::backend::SeriesRoleWire;
use crate::chart::{
    ChartModel, PlotRole, PlotSeries, XAxisKind, accessible_summary, build_chart, polyline_points,
    stroke_for,
};
use crate::ui::Ui;

/// The SVG viewbox. Fixed, with the element scaled by CSS, so the geometry
/// is resolution-independent and needs no layout measurement.
const WIDTH: f64 = 960.0;
/// The viewbox height.
const HEIGHT: f64 = 420.0;
/// Room for the axes and their labels.
const PADDING: f64 = 48.0;

/// Colours chosen to stay distinguishable in the high-contrast theme and
/// under the most common forms of colour-vision deficiency. Series are also
/// distinguished by their legend order and by the text alternative, so
/// colour is never the only carrier.
const SERIES_COLOURS: [&str; 6] = [
    "#2f6fed", "#d1610b", "#0f8f6a", "#9b30d9", "#c02b52", "#5a6b7a",
];

/// The one colour every ensemble member and band edge shares.
///
/// Deliberately not from [`SERIES_COLOURS`]. Eight realisations in eight
/// different colours read as eight different quantities; in one muted grey
/// they read as what they are — eight draws of the same one.
const ENSEMBLE_GREY: &str = "#8b98a5";

/// The colour a curve is drawn in.
fn stroke_colour(role: PlotRole, index: usize) -> &'static str {
    if stroke_for(role).own_colour
    {
        SERIES_COLOURS[index % SERIES_COLOURS.len()]
    }
    else
    {
        ENSEMBLE_GREY
    }
}

/// The chart for the displayed run.
#[component]
pub fn Chart() -> Element {
    let ui = use_context::<Ui>();
    let run = ui.run.read().clone();
    let hidden = ui.hidden_series.read().clone();

    let Some(run) = run
    else
    {
        return rsx! { p { class: "muted", "{ui.t(\"run.none\")}" } };
    };

    let series: Vec<PlotSeries> = run
        .series
        .iter()
        .map(|s| PlotSeries {
            id: s.id.clone(),
            display_name: s.display_name.clone(),
            unit: s.unit.clone(),
            role: match s.role
            {
                SeriesRoleWire::Trajectory => PlotRole::Trajectory,
                SeriesRoleWire::Reference => PlotRole::Reference,
                SeriesRoleWire::EnsembleMember => PlotRole::EnsembleMember,
                SeriesRoleWire::EnsembleMean => PlotRole::EnsembleMean,
                // The two edges are drawn the same; which side an edge is on
                // is already visible from where it is.
                SeriesRoleWire::EnsembleBandLower | SeriesRoleWire::EnsembleBandUpper =>
                {
                    PlotRole::EnsembleBandEdge
                },
            },
            values: s.values.clone(),
            visible: !hidden.contains(&s.id),
        })
        .collect();

    let kind = match run.x_axis_kind
    {
        crate::backend::XAxisKindWire::PhysicalCoordinates => XAxisKind::PhysicalCoordinates,
        crate::backend::XAxisKindWire::SampleIndex => XAxisKind::SampleIndex,
    };

    let chart = build_chart(
        &run.x_values,
        &series,
        kind,
        &run.x_axis_label,
        &run.x_axis_unit,
    );
    let summary = accessible_summary(&chart);

    rsx! {
        section { class: "chart", "aria-label": "{ui.t(\"chart.title\")}",
            header { class: "chart-head",
                h3 { "{ui.t(\"chart.title\")}" }
                span { class: "chart-caption",
                    "{ui.t(\"chart.showing\")} {chart.drawn_points} {ui.t(\"chart.of\")} "
                    "{chart.source_points} {ui.t(\"chart.points\")}"
                }
            }

            if kind == XAxisKind::SampleIndex {
                p { class: "legacy-notice", role: "note", "{ui.t(\"chart.axis_legacy\")}" }
            }

            match chart.empty_reason {
                Some(reason) => rsx! {
                    p { class: "muted chart-empty", "{ui.t(reason.message_key())}" }
                },
                None => rsx! {
                    svg {
                        class: "plot",
                        view_box: "0 0 {WIDTH} {HEIGHT}",
                        role: "img",
                        "aria-label": summary.clone(),
                        preserve_aspect_ratio: "none",

                        // Frame.
                        rect {
                            x: "{PADDING}",
                            y: "{PADDING}",
                            width: "{WIDTH - PADDING * 2.0}",
                            height: "{HEIGHT - PADDING * 2.0}",
                            fill: "none",
                            stroke: "currentColor",
                            "stroke-width": "1",
                            opacity: "0.35",
                        }

                        for (index, tick) in ticks(&chart).into_iter().enumerate() {
                            g { key: "x{index}",
                                line {
                                    x1: "{tick.x}",
                                    y1: "{PADDING}",
                                    x2: "{tick.x}",
                                    y2: "{HEIGHT - PADDING}",
                                    stroke: "currentColor",
                                    "stroke-width": "0.5",
                                    opacity: "0.15",
                                }
                                text {
                                    x: "{tick.x}",
                                    y: "{HEIGHT - PADDING + 18.0}",
                                    "text-anchor": "middle",
                                    class: "tick",
                                    "{tick.label}"
                                }
                            }
                        }

                        for (index, tick) in value_ticks(&chart).into_iter().enumerate() {
                            g { key: "y{index}",
                                line {
                                    x1: "{PADDING}",
                                    y1: "{tick.y}",
                                    x2: "{WIDTH - PADDING}",
                                    y2: "{tick.y}",
                                    stroke: "currentColor",
                                    "stroke-width": "0.5",
                                    opacity: "0.15",
                                }
                                text {
                                    x: "{PADDING - 8.0}",
                                    y: "{tick.y + 4.0}",
                                    "text-anchor": "end",
                                    class: "tick",
                                    "{tick.label}"
                                }
                            }
                        }

                        for (index, plot) in chart.series.iter().enumerate() {
                            polyline {
                                key: "{plot.id}",
                                points: "{polyline_points(&chart, plot, WIDTH, HEIGHT, PADDING)}",
                                fill: "none",
                                stroke: "{stroke_colour(plot.role, index)}",
                                "stroke-width": "{stroke_for(plot.role).width}",
                                "stroke-opacity": "{stroke_for(plot.role).opacity}",
                                "stroke-dasharray": if stroke_for(plot.role).dashed { "5 3" } else { "none" },
                                "stroke-linejoin": "round",
                            }
                        }

                        text {
                            x: "{WIDTH / 2.0}",
                            y: "{HEIGHT - 8.0}",
                            "text-anchor": "middle",
                            class: "axis-label",
                            "{axis_label(&chart)}"
                        }
                    }
                },
            }

            Legend { chart_series: chart.series.clone() }

            details { class: "chart-text",
                summary { "{ui.t(\"chart.table_view\")}" }
                p { {summary.clone()} }
                TextTable { chart: chart.clone() }
            }
        }
    }
}

/// The legend, which is also the visibility control.
#[component]
fn Legend(chart_series: Vec<PlotSeries>) -> Element {
    let ui = use_context::<Ui>();
    let hidden = ui.hidden_series.read().clone();
    let run = ui.run.read().clone();
    let all: Vec<(String, String, String)> = run
        .map(|r| {
            r.series
                .iter()
                .map(|s| (s.id.clone(), s.display_name.clone(), s.unit.clone()))
                .collect()
        })
        .unwrap_or_default();

    rsx! {
        ul { class: "legend", "aria-label": "{ui.t(\"chart.legend\")}",
            for (index, (id, name, unit)) in all.into_iter().enumerate() {
                li { key: "{id}",
                    label { class: "legend-item",
                        input {
                            r#type: "checkbox",
                            checked: !hidden.contains(&id),
                            onchange: {
                                let ui = ui.clone();
                                let id = id.clone();
                                move |_| {
                                    let mut hidden_series = ui.hidden_series;
                                    let mut set: BTreeSet<String> = hidden_series.read().clone();
                                    if !set.remove(&id)
                                    {
                                        set.insert(id.clone());
                                    }
                                    hidden_series.set(set);
                                }
                            },
                        }
                        span {
                            class: "swatch",
                            style: "background: {SERIES_COLOURS[index % SERIES_COLOURS.len()]}",
                        }
                        span { {name} }
                        if !unit.is_empty() {
                            span { class: "muted", " ({unit})" }
                        }
                    }
                }
            }
            if chart_series.is_empty() {
                li { class: "muted", "{ui.t(\"chart.empty.all_hidden\")}" }
            }
        }
    }
}

/// The first rows of the plotted data, as a table.
///
/// Bounded on purpose: this is a readable alternative to the picture, not an
/// export. It shows exactly the coordinates the chart drew, so a reader can
/// check the picture against numbers.
#[component]
fn TextTable(chart: ChartModel) -> Element {
    const ROWS: usize = 12;
    let step = (chart.x.len() / ROWS).max(1);

    rsx! {
        table { class: "chart-table",
            thead {
                tr {
                    th { "{chart.x_label}" }
                    for series in chart.series.iter() {
                        th { key: "{series.id}", "{series.display_name}" }
                    }
                }
            }
            tbody {
                for (index, x) in chart.x.iter().enumerate().step_by(step).take(ROWS) {
                    tr { key: "{index}",
                        td { class: "mono", "{x:.6}" }
                        for series in chart.series.iter() {
                            td { key: "{series.id}", class: "mono",
                                match series.values.get(index) {
                                    Some(v) => format!("{v:.6}"),
                                    None => "—".to_string(),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One axis tick.
struct Tick {
    x: f64,
    y: f64,
    label: String,
}

/// Five ticks across the horizontal axis, labelled with real coordinates.
fn ticks(chart: &ChartModel) -> Vec<Tick> {
    let plot_w = WIDTH - PADDING * 2.0;
    (0..=4)
        .map(|i| {
            let fraction = f64::from(i) / 4.0;
            let value = chart.x_range.min + (chart.x_range.max - chart.x_range.min) * fraction;
            Tick {
                x: PADDING + fraction * plot_w,
                y: 0.0,
                label: format_tick(value),
            }
        })
        .collect()
}

/// Five ticks up the vertical axis.
fn value_ticks(chart: &ChartModel) -> Vec<Tick> {
    let plot_h = HEIGHT - PADDING * 2.0;
    (0..=4)
        .map(|i| {
            let fraction = f64::from(i) / 4.0;
            let value = chart.y_range.min + (chart.y_range.max - chart.y_range.min) * fraction;
            Tick {
                x: 0.0,
                y: PADDING + (1.0 - fraction) * plot_h,
                label: format_tick(value),
            }
        })
        .collect()
}

/// A tick label that stays readable across the ranges a physical result
/// covers, from `1e-12` to `1e12`.
fn format_tick(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-3..1e5).contains(&magnitude)
    {
        format!("{value:.2e}")
    }
    else if magnitude >= 100.0
    {
        format!("{value:.0}")
    }
    else
    {
        format!("{value:.3}")
    }
}

/// The horizontal axis label, which never says "time" for a legacy result.
fn axis_label(chart: &ChartModel) -> String {
    match chart.x_axis_kind
    {
        XAxisKind::PhysicalCoordinates if !chart.x_unit.is_empty() =>
        {
            format!("{} ({})", chart.x_label, chart.x_unit)
        },
        _ => chart.x_label.clone(),
    }
}
