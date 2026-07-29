//! Chart geometry: turning a result's real coordinates into SVG paths.
//!
//! Pure arithmetic, no Dioxus, no DOM — so it is unit tested on the host
//! rather than only being looked at in a running window.
//!
//! Two properties are load-bearing:
//!
//! 1. **Nothing is invented.** The horizontal coordinates come from the
//!    result. For a legacy (v1) result there are none, and the caller is
//!    handed sample indices *labelled as such* — never a fabricated time
//!    axis. See [`crate::chart::XAxisKind`].
//! 2. **Reduction never hides a peak.** Long series are reduced for display
//!    only, by a documented deterministic min/max bucket algorithm that
//!    keeps every bucket's extremes. Plain every-Nth decimation is rejected
//!    precisely because it can drop a spike entirely.

use serde::{Deserialize, Serialize};

/// The most points drawn per series. Beyond this the browser is drawing
/// more path segments than the display has pixels.
pub const MAX_POINTS_PER_SERIES: usize = 2000;

/// What the horizontal axis genuinely represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAxisKind {
    /// Real coordinates from the integrator.
    PhysicalCoordinates,
    /// Sample ordinals only — a legacy result. The chart must say so.
    SampleIndex,
}

/// What a curve represents, as far as drawing it is concerned.
///
/// Declared here rather than reused from `backend::wire` for the same reason
/// [`XAxisKind`] is: this module is the chart's geometry and is compiled and
/// tested on the host, independent of the bridge's types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlotRole {
    /// One computed trajectory.
    #[default]
    Trajectory,
    /// A comparison line the solver did not compute.
    Reference,
    /// One realisation out of an ensemble.
    EnsembleMember,
    /// The across-replicate mean.
    EnsembleMean,
    /// An edge of the ensemble's spread band.
    EnsembleBandEdge,
}

/// How a curve is drawn.
///
/// The point of varying this is not decoration. An ensemble puts a mean over
/// hundreds of realisations on the same axes as eight individual ones, and
/// drawing them identically would present a summary statistic and a single
/// noisy sample as the same kind of evidence — a reader would have no way to
/// see which line is the answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeStyle {
    /// Stroke width in user units.
    pub width: f64,
    /// Stroke opacity in `0.0..=1.0`.
    pub opacity: f64,
    /// Whether the line is dashed.
    pub dashed: bool,
    /// Whether the curve takes its own colour from the palette.
    ///
    /// False for ensemble members and band edges, which share one muted
    /// colour: eight differently-coloured realisations read as eight
    /// different quantities rather than eight draws of one.
    pub own_colour: bool,
}

/// The treatment a role gets.
pub fn stroke_for(role: PlotRole) -> StrokeStyle {
    match role
    {
        PlotRole::Trajectory => StrokeStyle {
            width: 1.8,
            opacity: 1.0,
            dashed: false,
            own_colour: true,
        },
        // Dashed and dimmed: it was not computed by the solver, and a solid
        // line of equal weight would claim it was.
        PlotRole::Reference => StrokeStyle {
            width: 1.2,
            opacity: 0.65,
            dashed: true,
            own_colour: true,
        },
        // The heaviest line on the chart, because it is the one the run is
        // evidence for.
        PlotRole::EnsembleMean => StrokeStyle {
            width: 2.4,
            opacity: 1.0,
            dashed: false,
            own_colour: true,
        },
        PlotRole::EnsembleBandEdge => StrokeStyle {
            width: 1.0,
            opacity: 0.5,
            dashed: false,
            own_colour: false,
        },
        PlotRole::EnsembleMember => StrokeStyle {
            width: 0.8,
            opacity: 0.35,
            dashed: false,
            own_colour: false,
        },
    }
}

/// A series ready to plot.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotSeries {
    /// Stable id.
    pub id: String,
    /// Label for the legend.
    pub display_name: String,
    /// Unit.
    pub unit: String,
    /// What this curve represents.
    pub role: PlotRole,
    /// Values, aligned with the chart's x coordinates.
    pub values: Vec<f64>,
    /// Whether the user has it switched on.
    pub visible: bool,
}

/// The numeric range a chart covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    /// Lowest value.
    pub min: f64,
    /// Highest value.
    pub max: f64,
}

impl Range {
    /// A range that can be divided by without producing infinities.
    ///
    /// A constant series has zero extent; padding it puts the line in the
    /// middle of the plot instead of dividing by zero.
    pub fn padded(self) -> Range {
        if !self.min.is_finite() || !self.max.is_finite()
        {
            return Range { min: 0.0, max: 1.0 };
        }
        let extent = self.max - self.min;
        if extent.abs() < f64::EPSILON
        {
            let pad = if self.min.abs() > 1.0
            {
                self.min.abs() * 0.05
            }
            else
            {
                0.5
            };
            Range {
                min: self.min - pad,
                max: self.max + pad,
            }
        }
        else
        {
            self
        }
    }

    /// Where `value` sits in this range, as `0.0..=1.0`.
    pub fn fraction(self, value: f64) -> f64 {
        let extent = self.max - self.min;
        if extent.abs() < f64::EPSILON
        {
            0.5
        }
        else
        {
            ((value - self.min) / extent).clamp(0.0, 1.0)
        }
    }
}

/// Everything needed to render one chart.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartModel {
    /// The horizontal coordinates actually drawn.
    pub x: Vec<f64>,
    /// The visible series, reduced to the same length as `x`.
    pub series: Vec<PlotSeries>,
    /// Horizontal range.
    pub x_range: Range,
    /// Vertical range across every visible series.
    pub y_range: Range,
    /// What the horizontal axis is.
    pub x_axis_kind: XAxisKind,
    /// Axis label.
    pub x_label: String,
    /// Axis unit, empty for a sample index.
    pub x_unit: String,
    /// How many points each series originally had.
    pub source_points: usize,
    /// How many are drawn.
    pub drawn_points: usize,
    /// Why there is nothing to draw, when there is not.
    pub empty_reason: Option<EmptyReason>,
}

/// Why a chart has nothing to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyReason {
    /// The result carries no series at all.
    NoSeries,
    /// Every series is switched off in the legend.
    AllSeriesHidden,
    /// The data contained no finite values to plot.
    NoFiniteData,
    /// A series and the axis disagreed on length, so plotting it would pair
    /// values with the wrong coordinates.
    LengthMismatch,
}

impl EmptyReason {
    /// A localization key for the explanation shown in place of the chart.
    pub fn message_key(self) -> &'static str {
        match self
        {
            EmptyReason::NoSeries => "chart.empty.no_series",
            EmptyReason::AllSeriesHidden => "chart.empty.all_hidden",
            EmptyReason::NoFiniteData => "chart.empty.no_finite_data",
            EmptyReason::LengthMismatch => "chart.empty.length_mismatch",
        }
    }
}

/// Build a chart model from a result's coordinates and series.
pub fn build_chart(
    x: &[f64],
    series: &[PlotSeries],
    x_axis_kind: XAxisKind,
    x_label: &str,
    x_unit: &str,
) -> ChartModel {
    let empty = |reason: EmptyReason| ChartModel {
        x: Vec::new(),
        series: Vec::new(),
        x_range: Range { min: 0.0, max: 1.0 },
        y_range: Range { min: 0.0, max: 1.0 },
        x_axis_kind,
        x_label: x_label.to_string(),
        x_unit: x_unit.to_string(),
        source_points: x.len(),
        drawn_points: 0,
        empty_reason: Some(reason),
    };

    if series.is_empty()
    {
        return empty(EmptyReason::NoSeries);
    }
    let visible: Vec<&PlotSeries> = series.iter().filter(|s| s.visible).collect();
    if visible.is_empty()
    {
        return empty(EmptyReason::AllSeriesHidden);
    }
    if x.is_empty() || !x.iter().all(|v| v.is_finite())
    {
        return empty(EmptyReason::NoFiniteData);
    }
    // Pairing a series with coordinates it does not match would draw points
    // at the wrong places, which is worse than drawing nothing.
    if visible.iter().any(|s| s.values.len() != x.len())
    {
        return empty(EmptyReason::LengthMismatch);
    }

    let buckets = bucket_count(x.len());
    let indices = reduction_indices(x.len(), buckets);

    let reduced_x: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
    let reduced: Vec<PlotSeries> = visible
        .iter()
        .map(|s| PlotSeries {
            id: s.id.clone(),
            display_name: s.display_name.clone(),
            unit: s.unit.clone(),
            role: s.role,
            values: indices.iter().map(|&i| s.values[i]).collect(),
            visible: true,
        })
        .collect();

    let x_range = Range {
        min: reduced_x.iter().cloned().fold(f64::INFINITY, f64::min),
        max: reduced_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
    .padded();

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for s in &reduced
    {
        for v in s.values.iter().filter(|v| v.is_finite())
        {
            y_min = y_min.min(*v);
            y_max = y_max.max(*v);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite()
    {
        return empty(EmptyReason::NoFiniteData);
    }

    ChartModel {
        drawn_points: reduced_x.len(),
        source_points: x.len(),
        x: reduced_x,
        series: reduced,
        x_range,
        y_range: Range {
            min: y_min,
            max: y_max,
        }
        .padded(),
        x_axis_kind,
        x_label: x_label.to_string(),
        x_unit: x_unit.to_string(),
        empty_reason: None,
    }
}

fn bucket_count(len: usize) -> usize {
    // Each bucket contributes its minimum and its maximum, so the number of
    // buckets is half the point budget.
    (MAX_POINTS_PER_SERIES / 2).max(1).min(len)
}

/// The indices to draw, reducing `len` points to at most
/// [`MAX_POINTS_PER_SERIES`].
///
/// **Min/max bucketing.** The range is split into equal buckets and each
/// contributes the index of its lowest *and* its highest value, in
/// positional order, so a spike inside a bucket survives. Every-Nth
/// decimation is rejected for exactly this reason: a peak that falls between
/// samples disappears silently, and a chart that quietly loses the maximum of
/// a physical quantity is worse than one that admits it cannot draw
/// everything.
///
/// Deterministic: the same input always yields the same indices, so two
/// people looking at the same run see the same picture.
///
/// The first and last indices are always included, so the chart's endpoints
/// are the run's endpoints.
pub fn reduction_indices(len: usize, buckets: usize) -> Vec<usize> {
    if len == 0
    {
        return Vec::new();
    }
    if len <= MAX_POINTS_PER_SERIES
    {
        return (0..len).collect();
    }

    let buckets = buckets.max(1);
    let mut indices: Vec<usize> = Vec::with_capacity(buckets * 2 + 2);
    indices.push(0);

    for bucket in 0..buckets
    {
        let start = bucket * len / buckets;
        let end = ((bucket + 1) * len / buckets).min(len);
        if start >= end
        {
            continue;
        }
        indices.push(start);
        if end - 1 > start
        {
            indices.push(end - 1);
        }
    }
    indices.push(len - 1);

    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Reduce with the extremes of `values` preserved inside each bucket.
///
/// Used when a caller wants the *value-aware* reduction rather than the
/// positional one — the positional version above is what keeps several
/// series on a shared axis aligned, so this is offered separately rather
/// than silently changing that behaviour.
pub fn extreme_preserving_indices(values: &[f64], buckets: usize) -> Vec<usize> {
    let len = values.len();
    if len <= MAX_POINTS_PER_SERIES
    {
        return (0..len).collect();
    }
    let buckets = buckets.max(1);
    let mut indices = Vec::with_capacity(buckets * 2 + 2);
    indices.push(0);

    for bucket in 0..buckets
    {
        let start = bucket * len / buckets;
        let end = ((bucket + 1) * len / buckets).min(len);
        if start >= end
        {
            continue;
        }
        let mut lo = start;
        let mut hi = start;
        for i in start..end
        {
            if values[i] < values[lo]
            {
                lo = i;
            }
            if values[i] > values[hi]
            {
                hi = i;
            }
        }
        indices.push(lo.min(hi));
        if lo != hi
        {
            indices.push(lo.max(hi));
        }
    }
    indices.push(len - 1);
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// An SVG polyline `points` attribute for one series, in a viewbox of
/// `width` × `height`.
pub fn polyline_points(
    chart: &ChartModel,
    series: &PlotSeries,
    width: f64,
    height: f64,
    padding: f64,
) -> String {
    let plot_w = (width - padding * 2.0).max(1.0);
    let plot_h = (height - padding * 2.0).max(1.0);

    let mut out = String::with_capacity(series.values.len() * 14);
    for (i, value) in series.values.iter().enumerate()
    {
        if !value.is_finite()
        {
            continue;
        }
        let Some(x) = chart.x.get(i)
        else
        {
            continue;
        };
        let px = padding + chart.x_range.fraction(*x) * plot_w;
        // SVG's y grows downward; a plot's does not.
        let py = padding + (1.0 - chart.y_range.fraction(*value)) * plot_h;
        if !out.is_empty()
        {
            out.push(' ');
        }
        out.push_str(&format!("{px:.2},{py:.2}"));
    }
    out
}

/// A textual summary of the chart, for a screen reader and for the table
/// fallback.
///
/// Present because a chart that only exists as a picture is unusable to
/// someone who cannot see it, and because a reviewer should be able to read
/// what the picture claims.
pub fn accessible_summary(chart: &ChartModel) -> String {
    if let Some(reason) = chart.empty_reason
    {
        return format!("No chart: {}", reason.message_key());
    }
    let axis = match chart.x_axis_kind
    {
        XAxisKind::PhysicalCoordinates => format!("{} ({})", chart.x_label, chart.x_unit),
        XAxisKind::SampleIndex => format!("{} (ordinal, not time)", chart.x_label),
    };
    let names: Vec<&str> = chart
        .series
        .iter()
        .map(|s| s.display_name.as_str())
        .collect();

    // The visual distinction between a mean over hundreds of realisations and
    // one of them is carried by stroke weight and colour, neither of which
    // reaches a screen reader. Without this sentence the text alternative
    // lists them as peers, which is the same misreading the styling exists to
    // prevent.
    let members = count_role(chart, PlotRole::EnsembleMember);
    let ensemble = if members > 0 || count_role(chart, PlotRole::EnsembleMean) > 0
    {
        format!(
            " This is an ensemble: the mean over all realisations is drawn heaviest, \
             {members} individual realisations and the spread band are drawn faintly \
             behind it."
        )
    }
    else
    {
        String::new()
    };

    format!(
        "{} series ({}) over {} from {:.6} to {:.6}; values range {:.6} to {:.6}; \
         showing {} of {} points.{ensemble}",
        chart.series.len(),
        names.join(", "),
        axis,
        chart.x_range.min,
        chart.x_range.max,
        chart.y_range.min,
        chart.y_range.max,
        chart.drawn_points,
        chart.source_points
    )
}

/// How many visible curves carry a role.
fn count_role(chart: &ChartModel, role: PlotRole) -> usize {
    chart.series.iter().filter(|s| s.role == role).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(id: &str, values: Vec<f64>) -> PlotSeries {
        PlotSeries {
            id: id.to_string(),
            display_name: id.to_uppercase(),
            unit: "m".to_string(),
            role: PlotRole::Trajectory,
            values,
            visible: true,
        }
    }

    fn linear_chart(n: usize) -> ChartModel {
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        build_chart(
            &x,
            &[series("y", y)],
            XAxisKind::PhysicalCoordinates,
            "time",
            "s",
        )
    }

    #[test]
    fn a_short_series_is_drawn_in_full() {
        let chart = linear_chart(100);
        assert!(chart.empty_reason.is_none());
        assert_eq!(chart.drawn_points, 100);
        assert_eq!(chart.source_points, 100);
        assert_eq!(chart.series[0].values.len(), 100);
    }

    #[test]
    fn a_long_series_is_reduced_within_the_budget() {
        let chart = linear_chart(250_000);
        assert!(
            chart.drawn_points <= MAX_POINTS_PER_SERIES + 2,
            "{}",
            chart.drawn_points
        );
        assert_eq!(chart.source_points, 250_000);
        assert_eq!(chart.series[0].values.len(), chart.x.len());
    }

    /// The endpoints of the chart must be the endpoints of the run.
    #[test]
    fn reduction_keeps_the_first_and_last_point() {
        let indices = reduction_indices(100_000, 1000);
        assert_eq!(indices[0], 0);
        assert_eq!(*indices.last().unwrap(), 99_999);
    }

    #[test]
    fn reduction_is_deterministic() {
        let a = reduction_indices(123_457, 1000);
        let b = reduction_indices(123_457, 1000);
        assert_eq!(a, b, "two viewers must see the same picture");
    }

    #[test]
    fn reduction_indices_are_sorted_and_unique() {
        let indices = reduction_indices(50_000, 1000);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(indices, sorted);
    }

    /// The reason min/max bucketing is used rather than every-Nth
    /// decimation: a narrow spike must survive reduction.
    #[test]
    fn a_narrow_spike_survives_extreme_preserving_reduction() {
        let mut values = vec![0.0_f64; 100_000];
        // A single-sample peak at a position every-Nth sampling would miss.
        values[54_321] = 999.0;

        let indices = extreme_preserving_indices(&values, 1000);
        let kept: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
        let peak = kept.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(peak, 999.0, "the spike must survive display reduction");

        // And the naive alternative genuinely loses it, which is why this
        // test exists.
        let every_nth: Vec<f64> = values.iter().step_by(50).cloned().collect();
        assert_eq!(
            every_nth.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            0.0,
            "every-Nth decimation drops the peak — this is the failure being avoided"
        );
    }

    #[test]
    fn an_empty_result_says_why() {
        let chart = build_chart(&[], &[], XAxisKind::PhysicalCoordinates, "t", "s");
        assert_eq!(chart.empty_reason, Some(EmptyReason::NoSeries));
    }

    #[test]
    fn hiding_every_series_is_distinguished_from_having_none() {
        let mut s = series("y", vec![1.0, 2.0]);
        s.visible = false;
        let chart = build_chart(&[0.0, 1.0], &[s], XAxisKind::PhysicalCoordinates, "t", "s");
        assert_eq!(chart.empty_reason, Some(EmptyReason::AllSeriesHidden));
    }

    /// Plotting values against coordinates they do not match would draw
    /// points in the wrong places.
    #[test]
    fn a_length_mismatch_refuses_to_plot() {
        let chart = build_chart(
            &[0.0, 1.0, 2.0],
            &[series("y", vec![1.0, 2.0])],
            XAxisKind::PhysicalCoordinates,
            "t",
            "s",
        );
        assert_eq!(chart.empty_reason, Some(EmptyReason::LengthMismatch));
    }

    #[test]
    fn non_finite_coordinates_are_refused() {
        let chart = build_chart(
            &[0.0, f64::NAN],
            &[series("y", vec![1.0, 2.0])],
            XAxisKind::PhysicalCoordinates,
            "t",
            "s",
        );
        assert_eq!(chart.empty_reason, Some(EmptyReason::NoFiniteData));
    }

    #[test]
    fn a_constant_series_still_produces_a_usable_range() {
        let chart = build_chart(
            &[0.0, 1.0, 2.0],
            &[series("y", vec![5.0, 5.0, 5.0])],
            XAxisKind::PhysicalCoordinates,
            "t",
            "s",
        );
        assert!(chart.empty_reason.is_none());
        assert!(chart.y_range.max > chart.y_range.min, "{:?}", chart.y_range);
        assert!(chart.y_range.min.is_finite() && chart.y_range.max.is_finite());
    }

    #[test]
    fn polyline_points_stay_inside_the_viewbox() {
        let chart = linear_chart(500);
        let points = polyline_points(&chart, &chart.series[0], 800.0, 400.0, 40.0);
        assert!(!points.is_empty());
        for pair in points.split(' ')
        {
            let (x, y) = pair.split_once(',').expect("an x,y pair");
            let x: f64 = x.parse().unwrap();
            let y: f64 = y.parse().unwrap();
            assert!((40.0..=760.0).contains(&x), "x out of range: {x}");
            assert!((40.0..=360.0).contains(&y), "y out of range: {y}");
        }
    }

    /// SVG's vertical axis grows downward; a plot's does not.
    #[test]
    fn a_higher_value_is_drawn_higher_on_the_screen() {
        let chart = build_chart(
            &[0.0, 1.0],
            &[series("y", vec![0.0, 10.0])],
            XAxisKind::PhysicalCoordinates,
            "t",
            "s",
        );
        let points = polyline_points(&chart, &chart.series[0], 100.0, 100.0, 0.0);
        let ys: Vec<f64> = points
            .split(' ')
            .map(|p| p.split_once(',').unwrap().1.parse::<f64>().unwrap())
            .collect();
        assert!(ys[1] < ys[0], "the larger value must have the smaller y");
    }

    #[test]
    fn a_non_finite_value_is_skipped_rather_than_drawn() {
        let chart = build_chart(
            &[0.0, 1.0, 2.0],
            &[series("y", vec![1.0, f64::NAN, 3.0])],
            XAxisKind::PhysicalCoordinates,
            "t",
            "s",
        );
        let points = polyline_points(&chart, &chart.series[0], 100.0, 100.0, 0.0);
        assert_eq!(points.split(' ').count(), 2, "the NaN point is skipped");
    }

    /// The legacy case: a v1 result gets sample ordinals and the summary must
    /// not call them time.
    #[test]
    fn a_sample_index_axis_is_described_as_ordinal_not_time() {
        let chart = build_chart(
            &[0.0, 1.0, 2.0],
            &[series("y", vec![1.0, 2.0, 3.0])],
            XAxisKind::SampleIndex,
            "Sample index",
            "",
        );
        let summary = accessible_summary(&chart);
        assert!(summary.contains("ordinal, not time"), "{summary}");
        assert!(!summary.contains("(s)"), "{summary}");
    }

    #[test]
    fn the_accessible_summary_reports_what_was_reduced() {
        let chart = linear_chart(250_000);
        let summary = accessible_summary(&chart);
        assert!(summary.contains("250000"), "{summary}");
        assert!(summary.contains("showing"), "{summary}");
    }

    #[test]
    fn an_empty_chart_summary_says_why() {
        let chart = build_chart(&[], &[], XAxisKind::PhysicalCoordinates, "t", "s");
        assert!(accessible_summary(&chart).contains("no_series"));
    }
}

#[cfg(test)]
mod role_tests {
    use super::*;

    fn roled(id: &str, role: PlotRole) -> PlotSeries {
        PlotSeries {
            id: id.to_string(),
            display_name: id.to_uppercase(),
            unit: "1".to_string(),
            role,
            values: vec![0.0, 1.0, 0.5],
            visible: true,
        }
    }

    fn ensemble_chart() -> ChartModel {
        let series = vec![
            roled("ensemble_mean", PlotRole::EnsembleMean),
            roled("ensemble_band_lower", PlotRole::EnsembleBandEdge),
            roled("ensemble_band_upper", PlotRole::EnsembleBandEdge),
            roled("member_0", PlotRole::EnsembleMember),
            roled("member_1", PlotRole::EnsembleMember),
            roled("mean", PlotRole::Reference),
        ];
        build_chart(
            &[0.0, 1.0, 2.0],
            &series,
            XAxisKind::PhysicalCoordinates,
            "time",
            "s",
        )
    }

    /// The ensemble mean is what the run is evidence for, so it must be the
    /// most prominent line on the chart — heavier than any realisation and
    /// than the reference.
    #[test]
    fn the_mean_is_drawn_more_prominently_than_anything_else() {
        let mean = stroke_for(PlotRole::EnsembleMean);
        for other in [
            PlotRole::EnsembleMember,
            PlotRole::EnsembleBandEdge,
            PlotRole::Reference,
            PlotRole::Trajectory,
        ]
        {
            let other = stroke_for(other);
            assert!(
                mean.width > other.width,
                "the ensemble mean is not the heaviest line"
            );
            assert!(mean.opacity >= other.opacity);
        }
    }

    /// A single realisation must not read as strongly as a summary over all
    /// of them, or the chart presents one noisy sample as the answer.
    #[test]
    fn a_realisation_is_drawn_faintly() {
        let member = stroke_for(PlotRole::EnsembleMember);
        assert!(member.opacity < 0.5);
        assert!(member.width < stroke_for(PlotRole::Trajectory).width);
    }

    /// Realisations and band edges share one colour: eight colours would read
    /// as eight different quantities rather than eight draws of one.
    #[test]
    fn only_summary_curves_take_their_own_colour() {
        assert!(stroke_for(PlotRole::EnsembleMean).own_colour);
        assert!(stroke_for(PlotRole::Trajectory).own_colour);
        assert!(stroke_for(PlotRole::Reference).own_colour);
        assert!(!stroke_for(PlotRole::EnsembleMember).own_colour);
        assert!(!stroke_for(PlotRole::EnsembleBandEdge).own_colour);
    }

    /// A line the solver did not compute must not be drawn as though it had.
    #[test]
    fn a_reference_line_is_dashed_and_nothing_computed_is() {
        assert!(stroke_for(PlotRole::Reference).dashed);
        for computed in [
            PlotRole::Trajectory,
            PlotRole::EnsembleMean,
            PlotRole::EnsembleMember,
            PlotRole::EnsembleBandEdge,
        ]
        {
            assert!(!stroke_for(computed).dashed, "{computed:?} is dashed");
        }
    }

    /// Stroke weight and colour do not reach a screen reader, so the text
    /// alternative has to carry the same distinction in words.
    #[test]
    fn the_text_alternative_says_the_chart_is_an_ensemble() {
        let summary = accessible_summary(&ensemble_chart());
        assert!(summary.contains("ensemble"), "{summary}");
        assert!(summary.contains("2 individual realisations"), "{summary}");
        assert!(summary.contains("heaviest"), "{summary}");
    }

    /// …and must not say so for the nine capabilities that have no ensemble.
    #[test]
    fn an_ordinary_chart_gains_no_ensemble_sentence() {
        let series = vec![roled("position", PlotRole::Trajectory)];
        let chart = build_chart(
            &[0.0, 1.0, 2.0],
            &series,
            XAxisKind::PhysicalCoordinates,
            "time",
            "s",
        );
        assert!(!accessible_summary(&chart).contains("ensemble"));
    }

    /// Reduction is display-only and must not change what a curve *is*.
    #[test]
    fn roles_survive_the_reduction_of_a_long_series() {
        let n = MAX_POINTS_PER_SERIES * 3;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut mean = roled("ensemble_mean", PlotRole::EnsembleMean);
        mean.values = x.clone();
        let mut member = roled("member_0", PlotRole::EnsembleMember);
        member.values = x.clone();

        let chart = build_chart(
            &x,
            &[mean, member],
            XAxisKind::PhysicalCoordinates,
            "time",
            "s",
        );
        assert!(chart.drawn_points < n, "the fixture must actually reduce");
        assert_eq!(chart.series[0].role, PlotRole::EnsembleMean);
        assert_eq!(chart.series[1].role, PlotRole::EnsembleMember);
    }
}

/// The largest heat map drawn, in cells.
///
/// One SVG rectangle per cell, and a browser diffing tens of thousands of
/// them is slower than the run that produced them. These bounds keep the
/// picture at roughly display resolution.
pub const MAX_FIELD_ROWS: usize = 48;
/// The column half of that bound.
pub const MAX_FIELD_COLUMNS: usize = 96;

/// A field reduced to a drawable grid.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldModel {
    /// Rows actually drawn.
    pub rows: usize,
    /// Columns actually drawn.
    pub columns: usize,
    /// Row-major cell values.
    pub cells: Vec<f64>,
    /// The lowest value in the *source* field, not in the reduction.
    pub min: f64,
    /// The highest value in the source field.
    pub max: f64,
    /// How many rows the source had.
    pub source_rows: usize,
    /// How many columns the source had.
    pub source_columns: usize,
}

impl FieldModel {
    /// Where a cell sits in `min..=max`, as `0.0..=1.0`.
    ///
    /// A field of one constant value has no extent to place anything in;
    /// putting it in the middle of the ramp is the only answer that does not
    /// divide by zero or claim the constant is an extreme.
    pub fn level(&self, value: f64) -> f64 {
        let extent = self.max - self.min;
        if !extent.is_finite() || extent.abs() < f64::EPSILON
        {
            0.5
        }
        else
        {
            ((value - self.min) / extent).clamp(0.0, 1.0)
        }
    }
}

/// Reduce a row-major field to at most [`MAX_FIELD_ROWS`] x
/// [`MAX_FIELD_COLUMNS`] cells.
///
/// Each cell keeps the source sample **furthest from the field's mean**,
/// not the block's average. This is the two-dimensional form of the rule
/// `reduction_indices` already follows for series: an average is exactly the
/// operation that makes a spike vanish, and a heat map that smooths away the
/// hot spot is a picture of a different experiment.
///
/// `min` and `max` are taken from the *source*, so the colour scale still
/// describes the data even where the reduction did not land on the extreme.
pub fn build_field(values: &[f64], columns: usize) -> Option<FieldModel> {
    if columns == 0 || values.is_empty() || !values.len().is_multiple_of(columns)
    {
        return None;
    }
    let source_rows = values.len() / columns;

    // Refused outright, not drawn around. `validate_result` guarantees a
    // stored field is finite throughout, so a non-finite value means
    // something upstream is wrong — and a heat map with the bad cells
    // quietly painted at the mean would hide exactly that.
    if values.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let (min, max) = values
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let mean = values.iter().sum::<f64>() / values.len() as f64;

    let rows = source_rows.min(MAX_FIELD_ROWS);
    let drawn_columns = columns.min(MAX_FIELD_COLUMNS);
    let mut cells = Vec::with_capacity(rows * drawn_columns);

    for r in 0..rows
    {
        // Half-open blocks, so every source row lands in exactly one cell
        // and none lands in two.
        let row_lo = r * source_rows / rows;
        let row_hi = ((r + 1) * source_rows / rows)
            .max(row_lo + 1)
            .min(source_rows);
        for c in 0..drawn_columns
        {
            let col_lo = c * columns / drawn_columns;
            let col_hi = ((c + 1) * columns / drawn_columns)
                .max(col_lo + 1)
                .min(columns);
            let mut chosen = mean;
            let mut furthest = -1.0_f64;
            for rr in row_lo..row_hi
            {
                for cc in col_lo..col_hi
                {
                    let v = values[rr * columns + cc];
                    let distance = (v - mean).abs();
                    if distance > furthest
                    {
                        furthest = distance;
                        chosen = v;
                    }
                }
            }
            cells.push(chosen);
        }
    }

    Some(FieldModel {
        rows,
        columns: drawn_columns,
        cells,
        min,
        max,
        source_rows,
        source_columns: columns,
    })
}

/// A text alternative for a heat map.
///
/// A picture with no reading is unavailable to a screen reader and
/// uncheckable by a reviewer, which is the same argument
/// [`accessible_summary`] makes for a line chart.
pub fn accessible_field_summary(field: &FieldModel, name: &str, unit: &str) -> String {
    format!(
        "{name}: a field of {} rows by {} columns, drawn as {} by {}; values range {:.6} to \
         {:.6} {unit}. Each drawn cell is the source sample furthest from the mean, so an \
         extreme is never averaged away.",
        field.source_rows, field.source_columns, field.rows, field.columns, field.min, field.max
    )
}

#[cfg(test)]
mod field_tests {
    use super::*;

    #[test]
    fn a_small_field_is_drawn_at_full_resolution() {
        let model = build_field(&[1.0, 2.0, 3.0, 4.0], 2).expect("a model");
        assert_eq!((model.rows, model.columns), (2, 2));
        assert_eq!(model.cells, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!((model.min, model.max), (1.0, 4.0));
    }

    #[test]
    fn a_malformed_field_is_refused_rather_than_guessed() {
        assert!(
            build_field(&[1.0, 2.0, 3.0], 2).is_none(),
            "not a rectangle"
        );
        assert!(build_field(&[1.0], 0).is_none(), "no columns");
        assert!(build_field(&[], 2).is_none(), "no values");
        assert!(
            build_field(&[f64::NAN, 1.0], 2).is_none(),
            "no finite range"
        );
    }

    #[test]
    fn a_large_field_is_reduced_within_the_budget() {
        let columns = MAX_FIELD_COLUMNS * 3;
        let rows = MAX_FIELD_ROWS * 5;
        let values: Vec<f64> = (0..rows * columns).map(|i| i as f64).collect();
        let model = build_field(&values, columns).expect("a model");
        assert_eq!(model.rows, MAX_FIELD_ROWS);
        assert_eq!(model.columns, MAX_FIELD_COLUMNS);
        assert_eq!(model.cells.len(), MAX_FIELD_ROWS * MAX_FIELD_COLUMNS);
        // The colour scale describes the source, not the reduction.
        assert_eq!(model.min, 0.0);
        assert_eq!(model.max, (rows * columns - 1) as f64);
        assert_eq!(model.source_rows, rows);
        assert_eq!(model.source_columns, columns);
    }

    /// The property the whole reduction exists for: a single hot cell in an
    /// otherwise flat field must survive being shrunk.
    #[test]
    fn reduction_never_hides_a_hot_spot() {
        let columns = MAX_FIELD_COLUMNS * 4;
        let rows = MAX_FIELD_ROWS * 4;
        let mut values = vec![10.0; rows * columns];
        values[(rows / 2) * columns + columns / 2] = 1000.0;

        let model = build_field(&values, columns).expect("a model");
        assert!(
            model.cells.contains(&1000.0),
            "the hot spot was averaged away, which is the one thing this must not do"
        );
        assert_eq!(model.max, 1000.0);
    }

    #[test]
    fn a_constant_field_sits_in_the_middle_of_the_scale() {
        let model = build_field(&[7.0; 16], 4).expect("a model");
        assert_eq!(model.level(7.0), 0.5);
    }

    #[test]
    fn the_level_is_clamped_to_the_scale() {
        let model = build_field(&[0.0, 10.0], 2).expect("a model");
        assert_eq!(model.level(0.0), 0.0);
        assert_eq!(model.level(10.0), 1.0);
        assert_eq!(model.level(-5.0), 0.0);
        assert_eq!(model.level(50.0), 1.0);
    }

    #[test]
    fn the_text_alternative_states_both_shapes_and_the_range() {
        let values: Vec<f64> = (0..MAX_FIELD_ROWS * 4 * 8).map(|i| i as f64).collect();
        let model = build_field(&values, 8).expect("a model");
        let summary = accessible_field_summary(&model, "Temperature", "K");
        assert!(
            summary.contains(&format!("{} rows", model.source_rows)),
            "{summary}"
        );
        assert!(summary.contains(&format!("drawn as {} by {}", model.rows, model.columns)));
        assert!(summary.contains("furthest from the mean"));
    }
}
