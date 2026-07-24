//! The structured, versioned result model every [`crate::CapabilityAdapter`]
//! returns from `execute()`.
//!
//! Terminal formatting (colour, column widths, layout) lives outside this
//! module — in `scirust-cli`'s formatter — so the same [`RunResult`] can be
//! rendered as text today and fed to a chart or a desktop UI later without
//! re-running anything. JSON serialization lives here (via
//! [`RunResult::to_json_pretty`]), the same way
//! `scirust_studio_registry::CapabilityRegistry::to_json` keeps its
//! serialization in the crate that owns the type, so callers such as
//! `scirust-cli` do not need a direct `serde_json` dependency just to expose
//! `--format json`.

use serde::{Deserialize, Serialize};

use scirust_studio_registry::DeterminismClass;

/// The schema version of [`RunResult`] itself, independent of the scenario
/// schema version (`scirust_studio_schema::CURRENT_SCHEMA_VERSION`) and of
/// any individual capability's own versioning.
pub const RESULT_SCHEMA_VERSION: u32 = 1;

/// The full, versioned result of one capability execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    /// [`RESULT_SCHEMA_VERSION`] at the time this result was produced.
    pub schema_version: u32,
    /// The capability that produced this result.
    pub capability_id: String,
    /// Human-facing summary.
    pub summary: RunSummary,
    /// Shared axes the series are plotted against (almost always exactly
    /// one: time).
    pub axes: Vec<AxisDescriptor>,
    /// Named output series, each a flat time-course.
    pub series: Vec<Series>,
    /// Named scalar/derived metrics.
    pub metrics: Vec<Metric>,
    /// Non-fatal warnings raised during validation or execution.
    pub warnings: Vec<RunWarning>,
    /// Scientific verification checks and their outcomes.
    pub verifications: Vec<VerificationResult>,
    /// What produced this result and when.
    pub provenance: RunProvenance,
}

impl RunResult {
    /// Serialize to pretty-printed, stable-field-order JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Human-facing summary of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// The capability's display name at the time of the run.
    pub capability_display_name: String,
    /// The scenario's `experiment.name`.
    pub scenario_name: String,
    /// Number of integration steps taken (excluding the initial condition).
    pub steps: usize,
    /// Start time, in the axis's unit.
    pub t_start: f64,
    /// End time actually reached, in the axis's unit.
    pub t_end: f64,
}

/// One shared axis (almost always time) that series are plotted against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisDescriptor {
    /// Stable id, e.g. `"t"`.
    pub id: String,
    /// Human-facing label, e.g. `"time"`.
    pub display_name: String,
    /// Unit symbol, e.g. `"s"`.
    pub unit: String,
}

/// One named output time-course. Every value in `values` must be finite —
/// see [`assert_finite`], which every adapter calls before returning a
/// [`RunResult`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Series {
    /// Stable id, e.g. `"position"`.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// Unit symbol of the values.
    pub unit: String,
    /// The time-course itself, aligned with the run's single time axis.
    pub values: Vec<f64>,
}

/// A metric's value. Three kinds cover every metric an adapter in this
/// crate currently produces: a plain number, an integer count, and a
/// text classification (e.g. an RLC damping regime).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricValue {
    /// A finite floating-point value. Never `NaN`/infinite — see
    /// [`assert_finite`].
    Scalar(f64),
    /// An integer count.
    Integer(i64),
    /// A text classification.
    Text(String),
}

/// One named scalar or derived metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    /// Stable id, e.g. `"peak_infected"`.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// The value.
    pub value: MetricValue,
    /// Unit symbol, when the metric has one (a classification like a
    /// damping regime does not).
    pub unit: Option<String>,
}

/// What kind of thing a warning is about — the brief's rule that "a warning
/// is not an error unless the operation cannot produce a meaningful result"
/// only works if warnings are categorised, not one undifferentiated bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningCategory {
    /// About the scenario's input, short of a hard validation failure.
    Input,
    /// About the numerics of the run itself.
    Numerical,
    /// About solver convergence.
    Convergence,
    /// About a resource limit approached or exceeded.
    Resource,
    /// About the execution backend.
    Backend,
    /// Raised by a verification check that did not fully pass.
    Verification,
}

/// A non-fatal warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunWarning {
    /// The warning's category.
    pub category: WarningCategory,
    /// Human-facing message.
    pub message: String,
}

/// The outcome of one verification check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    /// The check passed within its threshold.
    Passed,
    /// The check did not fully pass but the result is still usable.
    Warning,
    /// The check failed.
    Failed,
    /// The check does not apply to this run's configuration (e.g. an
    /// energy-conservation check when the model configuration is dissipative
    /// by design).
    NotApplicable,
}

/// The result of one scientific verification check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Stable id, matching a
    /// `scirust_studio_registry::VerificationCheckDescriptor::id`.
    pub id: String,
    /// The outcome.
    pub status: VerificationStatus,
    /// The measured quantity the check was based on, if numeric.
    pub measured: Option<f64>,
    /// The threshold the measured quantity was compared against, if any.
    pub threshold: Option<f64>,
    /// Human-facing explanation of the check and its outcome.
    pub explanation: String,
}

/// What produced a [`RunResult`] and when.
///
/// This is deliberately minimal: a content-addressed run manifest, a
/// scenario hash, a result hash, and a full hardware/OS fingerprint belong
/// to the run-storage system (Phase 2B), which does not exist yet. Nothing
/// here is fabricated to look more complete than that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    /// The capability that produced this result.
    pub capability_id: String,
    /// The capability's determinism classification.
    pub determinism: DeterminismClass,
    /// The crate that implemented the adapter.
    pub adapter_crate: String,
    /// That crate's version at build time (`env!("CARGO_PKG_VERSION")` of
    /// `scirust-studio-runtime` itself — the one version this crate can
    /// stamp robustly; a per-dependency version for `scirust-sim` would
    /// need a build script to stay honest as that crate's version moves, so
    /// it is not included here — see Phase 2B's manifest/provenance work).
    pub adapter_version: String,
    /// Wall-clock start time, RFC 3339 UTC.
    pub started_at_rfc3339: String,
    /// Wall-clock completion time, RFC 3339 UTC.
    pub completed_at_rfc3339: String,
    /// Monotonic elapsed wall-clock duration, in seconds.
    pub elapsed_seconds: f64,
}

/// Every value in `result`'s series and every `Scalar` metric must be
/// finite. Called by every adapter immediately before returning a
/// successful [`RunResult`], so a `NaN`/infinite value from a derived
/// computation (a division the model's own domain doesn't protect against)
/// can never reach a "successful" result — it becomes an
/// [`crate::ExecutionError`] instead.
///
/// `scirust_sim`'s own blow-ups are already caught earlier, by
/// `SimError::NonFinite` from the integrator itself; this guard is for
/// values *derived* from an otherwise-finite trajectory (a ratio, a log)
/// that could still individually divide by zero or take a negative log.
pub fn assert_finite(result: &RunResult) -> Result<(), String> {
    for series in &result.series
    {
        if let Some((i, v)) = series
            .values
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(format!(
                "series `{}`[{i}] is {v} (non-finite), not a valid result value",
                series.id
            ));
        }
    }
    for metric in &result.metrics
    {
        if let MetricValue::Scalar(v) = &metric.value
        {
            if !v.is_finite()
            {
                return Err(format!(
                    "metric `{}` is {v} (non-finite), not a valid result value",
                    metric.id
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> RunResult {
        RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            summary: RunSummary {
                capability_display_name: "Spring-mass-damper".to_string(),
                scenario_name: "test".to_string(),
                steps: 10,
                t_start: 0.0,
                t_end: 1.0,
            },
            axes: vec![AxisDescriptor {
                id: "t".to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
            }],
            series: vec![Series {
                id: "position".to_string(),
                display_name: "Position".to_string(),
                unit: "m".to_string(),
                values: vec![1.0, 0.9, 0.8],
            }],
            metrics: vec![Metric {
                id: "final_position".to_string(),
                display_name: "Final position".to_string(),
                value: MetricValue::Scalar(0.8),
                unit: Some("m".to_string()),
            }],
            warnings: vec![],
            verifications: vec![VerificationResult {
                id: "energy_drift".to_string(),
                status: VerificationStatus::Passed,
                measured: Some(1e-14),
                threshold: Some(1e-6),
                explanation: "relative energy drift within tolerance".to_string(),
            }],
            provenance: RunProvenance {
                capability_id: "sim.mechanics.spring_mass_damper".to_string(),
                determinism: DeterminismClass::StrictSameBinarySameTarget,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: "0.1.0".to_string(),
                started_at_rfc3339: "2026-01-01T00:00:00Z".to_string(),
                completed_at_rfc3339: "2026-01-01T00:00:01Z".to_string(),
                elapsed_seconds: 1.0,
            },
        }
    }

    #[test]
    fn assert_finite_accepts_a_well_formed_result() {
        assert!(assert_finite(&sample_result()).is_ok());
    }

    #[test]
    fn assert_finite_rejects_a_nan_series_value() {
        let mut result = sample_result();
        result.series[0].values[1] = f64::NAN;
        let err = assert_finite(&result).unwrap_err();
        assert!(err.contains("position"), "{err}");
    }

    #[test]
    fn assert_finite_rejects_an_infinite_metric() {
        let mut result = sample_result();
        result.metrics[0].value = MetricValue::Scalar(f64::INFINITY);
        let err = assert_finite(&result).unwrap_err();
        assert!(err.contains("final_position"), "{err}");
    }

    #[test]
    fn assert_finite_ignores_non_scalar_metrics() {
        let mut result = sample_result();
        result.metrics[0].value = MetricValue::Text("underdamped".to_string());
        assert!(assert_finite(&result).is_ok());
    }

    #[test]
    fn round_trips_through_json() {
        let result = sample_result();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: RunResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, parsed);
    }

    #[test]
    fn json_uses_stable_snake_case_variant_names() {
        let result = sample_result();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"passed\""), "{json}");
        assert!(
            json.contains("\"scalar\":0.8") || json.contains("\"scalar\": 0.8"),
            "{json}"
        );
    }
}
