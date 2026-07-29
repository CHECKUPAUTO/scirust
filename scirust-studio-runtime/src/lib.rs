//! `scirust-studio-runtime` — the [`CapabilityAdapter`] execution contract,
//! the structured [`RunResult`] model, and the real `scirust-sim` adapters
//! that implement it.
//!
//! This is Phase 2A of the SciRust Studio effort (see
//! `docs/studio/adr/0001-capability-registry.md` and
//! `docs/studio/adr/0002-structured-run-results.md`). `scirust-cli` drives
//! every capability through [`CapabilityAdapter`] — it does not import
//! `scirust-sim` types directly, and neither will the future worker process
//! or desktop application.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod adapter;
mod adapters;
mod control;
mod ensemble;
mod execute_support;
mod measure;
mod result;
mod result_v1;
mod sink;
mod validate_support;

pub use adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
pub use control::ExecutionControl;

pub use result::{
    Axis, AxisMonotonicity, Field, Metric, MetricValue, RESULT_SCHEMA_VERSION, ResultDefect,
    RunProvenance, RunResult, RunSummary, RunWarning, Series, SeriesRole, TIME_AXIS_ID,
    VerificationResult, VerificationStatus, WarningCategory, describe_defects, validate_result,
};
pub use result_v1::{
    AxisDescriptorV1, RESULT_SCHEMA_VERSION_V1, RunResultV1, RunSummaryV1, SeriesV1, XAxisMeaning,
};
/// Re-exported because [`RunProvenance::determinism`] is a public field of a
/// public type: a caller cannot name the type of something it can read
/// without this.
pub use scirust_studio_registry::DeterminismClass;
pub use sink::{CollectingEventSink, EventSink, NullEventSink, RunEvent};

pub use adapters::{
    all_adapters, build_registry, find_adapter, tutorial_file_name_for, tutorial_scenario_for,
};
