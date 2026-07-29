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
    ChartModel, FieldModel, PlotRole, PlotSeries, XAxisKind, accessible_field_summary,
    accessible_summary, build_chart, build_field, polyline_points, stroke_for,
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

/// The heat map for a run's fields, when it has any.
///
/// Rendered beside the chart rather than instead of it: for the heat rod the
/// field is the result and the three curves are summaries of it, and a reader
/// wants both.
#[component]
pub fn FieldMaps() -> Element {
    let ui = use_context::<Ui>();
    let Some(run) = ui.run.read().clone()
    else
    {
        return rsx! {};
    };
    if run.fields.is_empty()
    {
        return rsx! {};
    }

    rsx! {
        for grid in run.fields.iter() {
            match build_field(&grid.values, grid.columns) {
                None => rsx! {
                    section { key: "{grid.id}", class: "chart",
                        p { class: "muted chart-empty", "{ui.t(\"field.unreadable\")}" }
                    }
                },
                Some(model) => rsx! {
                    FieldMap {
                        key: "{grid.id}",
                        model: model,
                        name: grid.display_name.clone(),
                        unit: grid.unit.clone(),
                        row_label: field_axis_label(&run, &grid.row_axis_id),
                        column_label: field_axis_label(&run, &grid.column_axis_id),
                    }
                },
            }
        }
    }
}

/// The histograms for a run's distributions, when it has any.
///
/// Horizontal bars rather than the field renderer's shaded cells: a
/// distribution is one-dimensional, so a *length* is available and exact
/// where a shade would make the reader estimate a count.
///
/// The bin's interval is labelled, not its centre. A histogram's whole
/// content is "this much fell between these two values", and printing a
/// centre makes the reader infer the width — the same off-by-one that made
/// `Distribution` its own type rather than a one-row field.
#[component]
pub fn Histograms() -> Element {
    let ui = use_context::<Ui>();
    let Some(run) = ui.run.read().clone()
    else
    {
        return rsx! {};
    };
    if run.distributions.is_empty()
    {
        return rsx! {};
    }

    rsx! {
        for distribution in run.distributions.iter() {
            {
                let peak = distribution.counts.iter().copied().max().unwrap_or(0);
                let total: u64 = distribution.counts.iter().sum::<u64>()
                    + distribution.underflow
                    + distribution.overflow;
                let outside = distribution.underflow + distribution.overflow;
                rsx! {
                    section { key: "{distribution.id}", class: "chart histogram",
                        h3 { class: "chart-title",
                            "{distribution.display_name}"
                            span { class: "muted", " ({total} samples, {distribution.unit})" }
                        }
                        // A count of zero everywhere means every sample fell
                        // outside the range; drawing empty bars would look
                        // like an empty run rather than a mis-chosen range.
                        if peak == 0 {
                            p { class: "muted chart-empty", "{ui.t(\"distribution.all-outside\")}" }
                        } else {
                            ul { class: "histogram-bins",
                                for (index, count) in distribution.counts.iter().enumerate() {
                                    li { key: "{index}", class: "histogram-bin",
                                        span { class: "histogram-range",
                                            "{format_edge(distribution.edges[index])}..{format_edge(distribution.edges[index + 1])}"
                                        }
                                        span { class: "histogram-bar",
                                            style: "width: {(*count as f64 / peak as f64) * 100.0}%",
                                            "aria-hidden": "true",
                                        }
                                        span { class: "histogram-count", "{count}" }
                                    }
                                }
                            }
                        }
                        // Never silently: a sample outside the range is a
                        // statement about the range, and hiding it turns a
                        // badly chosen histogram into a plausible one.
                        if outside > 0 {
                            p { class: "warn histogram-outside",
                                "{distribution.underflow} below and {distribution.overflow} above the binned range"
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Render a bin edge compactly enough to sit in a narrow column.
fn format_edge(value: f64) -> String {
    let magnitude = value.abs();
    if value == 0.0
    {
        "0".to_string()
    }
    else if (1e-3..1e6).contains(&magnitude)
    {
        format!("{value:.4}")
    }
    else
    {
        format!("{value:.2e}")
    }
}

/// An axis's label and unit, for a field's edge captions.
fn field_axis_label(run: &crate::backend::RunWire, axis_id: &str) -> String {
    // The interface holds the primary axis flattened onto the run, so only
    // its id can be matched; any other axis is named by its id, which is what
    // the shell would have shown anyway.
    if axis_id == "t" && !run.x_axis_label.is_empty()
    {
        if run.x_axis_unit.is_empty()
        {
            run.x_axis_label.clone()
        }
        else
        {
            format!("{} ({})", run.x_axis_label, run.x_axis_unit)
        }
    }
    else
    {
        axis_id.to_string()
    }
}

/// One field, drawn as a grid of rectangles.
#[component]
fn FieldMap(
    model: FieldModel,
    name: String,
    unit: String,
    row_label: String,
    column_label: String,
) -> Element {
    let ui = use_context::<Ui>();
    let summary = accessible_field_summary(&model, &name, &unit);

    // A fixed viewbox scaled by CSS, like the line chart's: the geometry is
    // resolution-independent and needs no layout measurement.
    let cell_w = 100.0 / model.columns as f64;
    let cell_h = 100.0 / model.rows as f64;

    rsx! {
        section { class: "chart", "aria-label": name.clone(),
            header { class: "chart-head",
                h3 { {name.clone()} }
                span { class: "chart-caption",
                    "{model.source_rows} x {model.source_columns} {ui.t(\"field.reduced_to\")} "
                    "{model.rows} x {model.columns}"
                }
            }
            svg {
                class: "field-map",
                view_box: "0 0 100 100",
                preserve_aspect_ratio: "none",
                role: "img",
                "aria-label": summary.clone(),

                for r in 0..model.rows {
                    for c in 0..model.columns {
                        rect {
                            key: "{r}-{c}",
                            x: "{c as f64 * cell_w}",
                            y: "{r as f64 * cell_h}",
                            width: "{cell_w * 1.02}",
                            height: "{cell_h * 1.02}",
                            fill: "{heat_colour(model.level(model.cells[r * model.columns + c]))}",
                        }
                    }
                }
            }
            p { class: "chart-caption",
                "{column_label} \u{2192}  |  {row_label} \u{2193}  |  "
                "{model.min:.4} \u{2013} {model.max:.4} {unit}"
            }
            p { class: "visually-hidden", {summary.clone()} }
        }
    }
}

/// The colour for a level in `0.0..=1.0`.
///
/// A blue-to-red ramp through a light middle. Not a rainbow: a rainbow has no
/// perceptual ordering, so a reader cannot tell which of two colours is the
/// larger value without consulting the legend — which is the one thing a heat
/// map is supposed to make unnecessary.
fn heat_colour(level: f64) -> String {
    let level = level.clamp(0.0, 1.0);
    let (r, g, b) = if level < 0.5
    {
        let k = level / 0.5;
        (
            (40.0 + k * 215.0) as u8,
            (90.0 + k * 145.0) as u8,
            (200.0 + k * 35.0) as u8,
        )
    }
    else
    {
        let k = (level - 0.5) / 0.5;
        (
            (255.0 - k * 15.0) as u8,
            (235.0 - k * 190.0) as u8,
            (235.0 - k * 195.0) as u8,
        )
    };
    format!("rgb({r},{g},{b})")
}
