//! Result schema **version 1**, kept for reading runs stored before schema
//! version 2 existed.
//!
//! These types are read-only by design. Nothing in this workspace produces a
//! v1 result any more, and nothing rewrites a stored one — a run store is
//! immutable, so silently upgrading a v1 file in place would break the hash
//! recorded for it and destroy the very property the store exists to
//! provide.
//!
//! ## What v1 cannot tell you
//!
//! A v1 result describes its axes but does **not** store their coordinates.
//! There is therefore no way to recover when each sample was taken. For the
//! fixed-step capabilities the spacing happened to be uniform, but that is a
//! fact about those runs, not about the format, and for
//! `sim.chemistry.robertson` — whose adaptive Rosenbrock-W solver chooses
//! its own steps across several decades of time — it is simply false.
//!
//! So a v1 result is never given a reconstructed time axis. A consumer may
//! plot its series against the sample index, and must say that is what it is
//! doing: see [`RunResultV1::x_axis_meaning`].

use serde::{Deserialize, Serialize};

use crate::result::{Metric, RunProvenance, RunWarning, VerificationResult};

/// The schema version these types describe.
pub const RESULT_SCHEMA_VERSION_V1: u32 = 1;

/// A v1 axis: a label with no coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisDescriptorV1 {
    /// Stable id, e.g. `"t"`.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// Unit symbol.
    pub unit: String,
}

/// A v1 series: values with no declared axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesV1 {
    /// Stable id.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// Unit symbol of the values.
    pub unit: String,
    /// The values. Which coordinates they correspond to is not recorded.
    pub values: Vec<f64>,
}

/// A v1 run summary.
///
/// This deliberately does **not** reuse [`crate::RunSummary`], although the
/// two were identical when v1 was frozen and are still nearly so.
///
/// They were shared once, and that coupling meant every field added to the v2
/// summary silently appeared in re-serialized v1 JSON — which is exactly what
/// `v1_round_trips_without_gaining_fields` exists to forbid, and exactly what
/// it caught when `RunSummary` gained `axis_id` for parameter sweeps (ADR
/// 0010). A frozen format must not track a living one's evolution, so the
/// duplication is the point rather than an oversight: v1 has these four
/// fields because v1 has always had these four fields, and nothing that
/// happens to v2 can change that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummaryV1 {
    /// The capability's display name at the time of the run.
    pub capability_display_name: String,
    /// The scenario's `experiment.name`.
    pub scenario_name: String,
    /// Number of integration steps taken.
    pub steps: usize,
    /// Start time.
    pub t_start: f64,
    /// End time reached.
    pub t_end: f64,
}

impl From<&RunSummaryV1> for crate::result::RunSummary {
    /// Widen a v1 summary to the current one.
    ///
    /// `axis_id` becomes `t` because a v1 result is always a time run: v1
    /// predates parameter sweeps entirely, and predates axis coordinates at
    /// all. Nothing is guessed — it is the only thing v1 could ever have
    /// meant.
    fn from(v1: &RunSummaryV1) -> Self {
        crate::result::RunSummary {
            capability_display_name: v1.capability_display_name.clone(),
            scenario_name: v1.scenario_name.clone(),
            axis_id: crate::result::TIME_AXIS_ID.to_string(),
            steps: v1.steps,
            t_start: v1.t_start,
            t_end: v1.t_end,
        }
    }
}

/// A result stored under schema version 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResultV1 {
    /// Always `1` for a genuine v1 result.
    pub schema_version: u32,
    /// The capability that produced this result.
    pub capability_id: String,
    /// Human-facing summary.
    pub summary: RunSummaryV1,
    /// Axis labels, without coordinates.
    pub axes: Vec<AxisDescriptorV1>,
    /// Series values, without a declared axis.
    pub series: Vec<SeriesV1>,
    /// Named scalar/derived metrics — fully readable.
    pub metrics: Vec<Metric>,
    /// Non-fatal warnings — fully readable.
    pub warnings: Vec<RunWarning>,
    /// Verification checks — fully readable.
    pub verifications: Vec<VerificationResult>,
    /// What produced this result and when — fully readable.
    pub provenance: RunProvenance,
}

/// What a consumer is entitled to say the horizontal axis represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XAxisMeaning {
    /// The stored coordinates are real physical values from the integrator.
    PhysicalCoordinates,
    /// Only the ordinal position of each sample is known. A chart must say
    /// so rather than implying time.
    SampleIndexOnly,
}

impl RunResultV1 {
    /// Always [`XAxisMeaning::SampleIndexOnly`].
    ///
    /// This is a method rather than a constant so a consumer handling both
    /// versions asks the same question of either and gets a truthful answer,
    /// instead of remembering to special-case v1 at every call site.
    pub fn x_axis_meaning(&self) -> XAxisMeaning {
        XAxisMeaning::SampleIndexOnly
    }

    /// Whether this result came from a capability whose solver chooses its
    /// own step sizes, making even an index-based plot's spacing misleading.
    ///
    /// Named from the capability id rather than guessed from the data: the
    /// data is exactly what cannot answer this question.
    pub fn has_adaptive_solver(&self) -> bool {
        self.capability_id == "sim.chemistry.robertson"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPRING_MASS: &str =
        include_str!("../../scirust-studio-store/tests/fixtures/v1_spring_mass_damper_result.json");
    const ROBERTSON: &str =
        include_str!("../../scirust-studio-store/tests/fixtures/v1_robertson_result.json");

    /// The fixtures are real files produced by the shipped v1 code before
    /// schema v2 existed, not hand-written approximations of it.
    #[test]
    fn a_real_v1_fixture_parses() {
        let r: RunResultV1 = serde_json::from_str(SPRING_MASS).expect("parses");
        assert_eq!(r.schema_version, RESULT_SCHEMA_VERSION_V1);
        assert_eq!(r.capability_id, "sim.mechanics.spring_mass_damper");
        assert!(!r.series.is_empty());
        assert!(!r.axes.is_empty());
    }

    #[test]
    fn v1_metrics_warnings_and_verifications_stay_readable() {
        let r: RunResultV1 = serde_json::from_str(SPRING_MASS).expect("parses");
        assert!(!r.metrics.is_empty(), "metrics must survive");
        assert!(
            !r.verifications.is_empty(),
            "scientific checks must survive"
        );
        assert!(!r.provenance.capability_id.is_empty());
    }

    #[test]
    fn v1_never_claims_physical_coordinates() {
        let r: RunResultV1 = serde_json::from_str(SPRING_MASS).expect("parses");
        assert_eq!(r.x_axis_meaning(), XAxisMeaning::SampleIndexOnly);
    }

    #[test]
    fn a_v1_adaptive_result_is_identified_as_such() {
        let r: RunResultV1 = serde_json::from_str(ROBERTSON).expect("parses");
        assert!(
            r.has_adaptive_solver(),
            "Robertson's stored samples are not uniformly spaced and nothing \
             in a v1 file records where they were"
        );
        assert_eq!(r.x_axis_meaning(), XAxisMeaning::SampleIndexOnly);
    }

    #[test]
    fn a_v1_fixed_step_result_is_not_flagged_adaptive() {
        let r: RunResultV1 = serde_json::from_str(SPRING_MASS).expect("parses");
        assert!(!r.has_adaptive_solver());
    }

    /// v1 axes genuinely lack coordinates — this pins the difference that
    /// motivated schema v2 rather than leaving it as prose.
    #[test]
    fn the_v1_fixture_has_no_axis_coordinates_at_all() {
        let raw: serde_json::Value = serde_json::from_str(SPRING_MASS).unwrap();
        let axis = &raw["axes"][0];
        assert!(
            axis.get("values").is_none(),
            "a v1 axis must not have coordinates: {axis}"
        );
        let series = &raw["series"][0];
        assert!(
            series.get("axis_id").is_none(),
            "a v1 series must not name an axis: {series}"
        );
    }

    #[test]
    fn v1_round_trips_without_gaining_fields() {
        let r: RunResultV1 = serde_json::from_str(SPRING_MASS).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let again: RunResultV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(r, again);
        assert!(
            !json.contains("axis_id"),
            "re-serializing v1 must not invent v2 fields"
        );
    }
}
