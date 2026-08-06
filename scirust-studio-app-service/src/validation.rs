//! Scenario validation, reported in a shape an editor can act on.

use serde::{Deserialize, Serialize};

/// Which validation stage rejected something.
///
/// The distinction is useful to a user: a schema problem is about the file's
/// shape, a capability problem is about what this particular model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemStage {
    /// The file did not parse as TOML.
    Parse,
    /// The generic scenario schema rejected it.
    Schema,
    /// The capability's own validation rejected it.
    Capability,
}

/// How severe a problem is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemSeverity {
    /// Prevents the scenario from running.
    Error,
    /// Does not prevent running.
    Warning,
}

/// A position in the source, when one can be determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    /// 1-based line.
    pub line: usize,
    /// 1-based column.
    pub column: usize,
}

/// One problem found in a scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Problem {
    /// Stable `SRST-…` code, when the underlying error carries one.
    pub code: Option<String>,
    /// Short heading.
    pub title: String,
    /// Full explanation.
    pub explanation: String,
    /// What to do about it, when known.
    pub suggested_action: Option<String>,
    /// The field this is about, when identifiable.
    pub field: Option<String>,
    /// Where in the source, when determinable.
    pub location: Option<SourceLocation>,
    /// Which stage rejected it.
    pub stage: ProblemStage,
    /// How severe.
    pub severity: ProblemSeverity,
}

/// The outcome of validating a scenario source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationOutcome {
    /// Whether the scenario can be run.
    pub valid: bool,
    /// The capability the scenario targets, when it could be determined.
    pub capability_id: Option<String>,
    /// The experiment name, when parsing got that far.
    pub scenario_name: Option<String>,
    /// Everything found, errors and warnings alike.
    pub problems: Vec<Problem>,
}

impl ValidationOutcome {
    /// An outcome carrying exactly one parse-stage error.
    pub(crate) fn parse_failure(explanation: String, location: Option<SourceLocation>) -> Self {
        ValidationOutcome {
            valid: false,
            capability_id: None,
            scenario_name: None,
            problems: vec![Problem {
                code: None,
                title: "The file is not valid TOML".to_string(),
                explanation,
                suggested_action: Some(
                    "Check for an unclosed quote, bracket or table header.".to_string(),
                ),
                field: None,
                location,
                stage: ProblemStage::Parse,
                severity: ProblemSeverity::Error,
            }],
        }
    }

    /// Whether anything at all was reported.
    pub fn is_clean(&self) -> bool {
        self.problems.is_empty()
    }

    /// Only the blocking problems.
    pub fn errors(&self) -> impl Iterator<Item = &Problem> {
        self.problems
            .iter()
            .filter(|p| p.severity == ProblemSeverity::Error)
    }
}

/// Best-effort extraction of a line/column from a `toml` parse error's own
/// message.
///
/// `toml`'s error carries a span, but the crate's public error type does not
/// expose it in a stable way across versions, so this reads the rendered
/// message instead. It returns `None` rather than guessing when the shape is
/// unfamiliar — an editor that jumps to the wrong line is worse than one
/// that does not jump at all.
pub(crate) fn location_from_toml_message(message: &str) -> Option<SourceLocation> {
    // Messages look like "... at line 12 column 5 ...".
    let after_line = message.split("line ").nth(1)?;
    let mut parts = after_line.split_whitespace();
    let line: usize = parts.next()?.trim_end_matches(',').parse().ok()?;
    let column = match parts.next()
    {
        Some("column") => parts
            .next()
            .and_then(|c| c.trim_end_matches(',').parse().ok())
            .unwrap_or(1),
        _ => 1,
    };
    Some(SourceLocation { line, column })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_parse_failure_reports_one_blocking_problem() {
        let outcome = ValidationOutcome::parse_failure("bad".to_string(), None);
        assert!(!outcome.valid);
        assert_eq!(outcome.problems.len(), 1);
        assert_eq!(outcome.problems[0].stage, ProblemStage::Parse);
        assert_eq!(outcome.errors().count(), 1);
    }

    #[test]
    fn a_location_is_read_out_of_a_toml_message() {
        let loc = location_from_toml_message("expected `=` at line 12 column 5").unwrap();
        assert_eq!(loc.line, 12);
        assert_eq!(loc.column, 5);
    }

    #[test]
    fn a_line_without_a_column_still_yields_a_location() {
        let loc = location_from_toml_message("something failed at line 7").unwrap();
        assert_eq!(loc.line, 7);
        assert_eq!(loc.column, 1);
    }

    /// Guessing is worse than declining: an editor that jumps to a
    /// fabricated line actively misleads.
    #[test]
    fn an_unfamiliar_message_yields_no_location() {
        assert!(location_from_toml_message("something went wrong").is_none());
        assert!(location_from_toml_message("at line banana").is_none());
    }

    #[test]
    fn the_outcome_round_trips_through_json() {
        let outcome = ValidationOutcome {
            valid: false,
            capability_id: Some("sim.epidemiology.sir".to_string()),
            scenario_name: Some("demo".to_string()),
            problems: vec![Problem {
                code: Some("SRST-VAL-0110".to_string()),
                title: "Missing field".to_string(),
                explanation: "beta is required".to_string(),
                suggested_action: Some("Add beta".to_string()),
                field: Some("beta".to_string()),
                location: Some(SourceLocation { line: 3, column: 1 }),
                stage: ProblemStage::Capability,
                severity: ProblemSeverity::Error,
            }],
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let back: ValidationOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, outcome);
        assert!(json.contains("\"capability\""), "{json}");
    }
}
