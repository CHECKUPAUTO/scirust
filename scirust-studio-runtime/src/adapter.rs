//! The [`CapabilityAdapter`] execution boundary.

use scirust_studio_command::CatalogedError;
use scirust_studio_registry::CapabilityDescriptor;
use scirust_studio_schema::Scenario;

use crate::control::ExecutionControl;
use crate::result::RunResult;
use crate::sink::EventSink;

/// A scenario that has passed both generic schema validation
/// (`scirust_studio_schema::validate`) and capability-specific validation
/// (`CapabilityAdapter::validate`).
///
/// The only way to construct one is a successful call to a
/// [`CapabilityAdapter::validate`] implementation in this crate — the
/// constructor is `pub(crate)`, so "validated" is a fact the type system
/// carries, not a convention a caller has to remember to uphold.
#[derive(Debug, Clone)]
pub struct ValidatedScenario {
    scenario: Scenario,
}

impl ValidatedScenario {
    pub(crate) fn new(scenario: Scenario) -> Self {
        ValidatedScenario { scenario }
    }

    /// The validated scenario.
    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }
}

/// Capability-specific validation failed. Carries every violation found,
/// not just the first — matching `scirust_studio_schema::validate`'s own
/// "report everything at once" behaviour.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValidationReport {
    /// Every validation failure found.
    pub errors: Vec<CatalogedError>,
}

impl ValidationReport {
    /// A report with exactly one error.
    pub fn single(error: CatalogedError) -> Self {
        ValidationReport {
            errors: vec![error],
        }
    }

    /// Whether this report carries no errors (a report is only ever
    /// returned as the `Err` variant of a `Result`, so an empty report
    /// should not normally be constructed — this exists for callers
    /// building one up incrementally).
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, e) in self.errors.iter().enumerate()
        {
            if i > 0
            {
                writeln!(f)?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationReport {}

/// Execution failed after validation succeeded.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionError {
    /// The underlying model rejected a parameter combination that passed
    /// validation. Should be rare — validation exists to catch this first —
    /// but the model's own constructor remains the final authority.
    InvalidModelState(String),
    /// A numerical failure during integration: either `scirust_sim`'s own
    /// `SimError`, or a derived computation (a ratio, a classification)
    /// that would otherwise have produced a non-finite result.
    Numerical(String),
    /// Execution was cancelled before or during the run.
    Cancelled,
    /// An internal error indicating a bug in the adapter, not the input.
    Internal(String),
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            ExecutionError::InvalidModelState(msg) => write!(f, "invalid model state: {msg}"),
            ExecutionError::Numerical(msg) => write!(f, "numerical failure: {msg}"),
            ExecutionError::Cancelled => write!(f, "execution was cancelled"),
            ExecutionError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for ExecutionError {}

/// The execution boundary every capability implements.
///
/// `scirust-cli`, and later the worker process and the desktop application,
/// drive capabilities *only* through this trait — never by importing a
/// `scirust-sim` type directly. That is what makes "the CLI must no longer
/// instantiate `SpringMassDamper` directly" (and the same for every other
/// model) a structural property rather than a habit.
pub trait CapabilityAdapter: Send + Sync {
    /// This adapter's static descriptor.
    fn descriptor(&self) -> &'static CapabilityDescriptor;

    /// Capability-specific validation, run after generic schema validation
    /// has already passed. Must not execute anything.
    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport>;

    /// Run the capability. `scenario` is guaranteed to have passed both
    /// validation stages. `control` may carry a cancellation request;
    /// `sink` receives lifecycle events.
    fn execute(
        &self,
        scenario: &ValidatedScenario,
        control: &ExecutionControl,
        sink: &mut dyn EventSink,
    ) -> Result<RunResult, ExecutionError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_studio_command::{ErrorCode, ErrorFamily};

    #[test]
    fn validation_report_display_joins_multiple_errors_with_newlines() {
        let report = ValidationReport {
            errors: vec![
                CatalogedError {
                    code: ErrorCode::new(ErrorFamily::Validation, 1),
                    title: "First".to_string(),
                    explanation: "first problem".to_string(),
                    recoverable: true,
                    suggested_action: None,
                },
                CatalogedError {
                    code: ErrorCode::new(ErrorFamily::Validation, 2),
                    title: "Second".to_string(),
                    explanation: "second problem".to_string(),
                    recoverable: true,
                    suggested_action: None,
                },
            ],
        };
        let text = report.to_string();
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
        assert!(text.find("First").unwrap() < text.find("Second").unwrap());
    }

    #[test]
    fn validation_report_single_wraps_exactly_one_error() {
        let report = ValidationReport::single(CatalogedError {
            code: ErrorCode::new(ErrorFamily::Validation, 1),
            title: "T".to_string(),
            explanation: "E".to_string(),
            recoverable: true,
            suggested_action: None,
        });
        assert_eq!(report.errors.len(), 1);
        assert!(!report.is_empty());
    }
}
