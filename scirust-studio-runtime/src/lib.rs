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
mod result;
mod sink;
mod validate_support;

pub use adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
pub use control::ExecutionControl;
pub use result::{
    AxisDescriptor, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, RunWarning, Series, VerificationResult, VerificationStatus, WarningCategory,
    assert_finite,
};
pub use sink::{CollectingEventSink, EventSink, NullEventSink, RunEvent};

pub use adapters::{all_adapters, build_registry, find_adapter};
