//! Turning the existing validation stages into an editor-shaped report.
//!
//! This module adds no validation rules of its own. It calls
//! `scirust_studio_schema::validate` and the capability's own
//! `CapabilityAdapter::validate`, then re-shapes what they already found —
//! duplicating a rule here would create a second, drifting definition of
//! "valid".

use scirust_studio_runtime::{build_registry, find_adapter};

use crate::validation::{
    Problem, ProblemSeverity, ProblemStage, ValidationOutcome, location_from_toml_message,
};

/// Validate scenario source through every stage, collecting all problems.
pub fn validate_source(source: &str) -> ValidationOutcome {
    let scenario = match scirust_studio_schema::parse_toml(source)
    {
        Ok(s) => s,
        Err(e) =>
        {
            let message = e.to_string();
            let location = location_from_toml_message(&message);
            return ValidationOutcome::parse_failure(message, location);
        },
    };

    let capability_id = scenario.capability.id.clone();
    let scenario_name = Some(scenario.experiment.name.clone());

    let registry = build_registry();
    let known: Vec<&str> = registry.iter().map(|d| d.id.0).collect();
    let schema_errors = scirust_studio_schema::validate(&scenario, Some(&known));

    let mut problems: Vec<Problem> = schema_errors
        .iter()
        .map(|e| {
            let cataloged = e.to_cataloged();
            Problem {
                code: Some(cataloged.code.to_string()),
                title: cataloged.title.clone(),
                explanation: cataloged.explanation.clone(),
                suggested_action: cataloged.suggested_action.clone(),
                field: None,
                location: locate_field_in_source(source, &cataloged.explanation),
                stage: ProblemStage::Schema,
                severity: ProblemSeverity::Error,
            }
        })
        .collect();

    // Capability validation only runs once the generic stage is happy:
    // running it against a scenario whose units or ranges are already known
    // to be wrong produces a second wave of errors about the same mistake.
    if problems.is_empty()
    {
        match find_adapter(&capability_id)
        {
            Some(adapter) =>
            {
                if let Err(report) = adapter.validate(&scenario)
                {
                    problems.extend(report.errors.iter().map(|e| {
                        let field = field_from_explanation(&e.explanation);
                        Problem {
                            code: Some(e.code.to_string()),
                            title: e.title.clone(),
                            explanation: e.explanation.clone(),
                            suggested_action: e.suggested_action.clone(),
                            location: field.as_deref().and_then(|f| locate_named_field(source, f)),
                            field,
                            stage: ProblemStage::Capability,
                            severity: ProblemSeverity::Error,
                        }
                    }));
                }
            },
            None =>
            {
                // Generic validation already checks the id against the
                // registry, so this means the registry and the adapter table
                // disagree — a bug in this build, not in the scenario.
                problems.push(Problem {
                    code: None,
                    title: "Capability not executable".to_string(),
                    explanation: format!(
                        "`{capability_id}` is listed in the catalogue but has no adapter in this build"
                    ),
                    suggested_action: Some("Reinstall the application.".to_string()),
                    field: None,
                    location: None,
                    stage: ProblemStage::Capability,
                    severity: ProblemSeverity::Error,
                });
            },
        }
    }

    ValidationOutcome {
        valid: problems
            .iter()
            .all(|p| p.severity != ProblemSeverity::Error),
        capability_id: Some(capability_id),
        scenario_name,
        problems,
    }
}

/// Pull a field path out of an explanation that names one in backticks.
///
/// The catalogued errors phrase themselves as ``field `beta` is required``
/// and ``` `initial_state.s[0]` = 5 is outside … ```, so the path is
/// recoverable without the error type carrying a separate field. Any array
/// index is dropped — `initial_state.s[0]` identifies the field `s`, and the
/// element number is not something the source has a separate line for.
///
/// Returns `None` when nothing looks like a field path, rather than
/// guessing: a wrong field name would put the editor's cursor on unrelated
/// code.
fn field_from_explanation(explanation: &str) -> Option<String> {
    let mut parts = explanation.split('`');
    let _before = parts.next()?;
    let candidate = parts.next()?;

    // Drop a trailing `[…]` index, which names an element rather than a
    // field.
    let candidate = match candidate.find('[')
    {
        Some(open) if candidate.ends_with(']') => &candidate[..open],
        Some(_) => return None,
        None => candidate,
    };

    if candidate.is_empty()
        || candidate.len() > 64
        || !candidate
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }
    Some(candidate.to_string())
}

/// Find where a named field is assigned in the source.
///
/// A dotted path is matched on its final segment: the source writes
/// `s = [...]` under a `[initial_state]` header, not the full path.
fn locate_named_field(source: &str, field: &str) -> Option<crate::validation::SourceLocation> {
    let leaf = field.rsplit('.').next().unwrap_or(field);
    if leaf.is_empty()
    {
        return None;
    }
    for (index, line) in source.lines().enumerate()
    {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(leaf)
        {
            if rest.trim_start().starts_with('=')
            {
                let column = line.len() - trimmed.len() + 1;
                return Some(crate::validation::SourceLocation {
                    line: index + 1,
                    column,
                });
            }
        }
    }
    None
}

/// Best-effort location for a schema error, via any field name it mentions.
fn locate_field_in_source(
    source: &str,
    explanation: &str,
) -> Option<crate::validation::SourceLocation> {
    field_from_explanation(explanation).and_then(|f| locate_named_field(source, &f))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIR: &str = include_str!("../../docs/studio/tutorials/sir_epidemic.scirust.toml");

    #[test]
    fn a_shipped_tutorial_validates_cleanly() {
        let outcome = validate_source(SIR);
        assert!(outcome.valid, "{:?}", outcome.problems);
        assert!(outcome.is_clean());
        assert_eq!(
            outcome.capability_id.as_deref(),
            Some("sim.epidemiology.sir")
        );
        assert!(outcome.scenario_name.is_some());
    }

    #[test]
    fn unparseable_toml_reports_a_parse_problem() {
        let outcome = validate_source("this is not = = toml");
        assert!(!outcome.valid);
        assert_eq!(outcome.problems[0].stage, ProblemStage::Parse);
    }

    #[test]
    fn a_missing_required_field_is_a_capability_problem() {
        let source = SIR.replace(r#"beta = { value = 0.6, unit = "1/s" }"#, "");
        let outcome = validate_source(&source);
        assert!(!outcome.valid);
        assert!(
            outcome
                .problems
                .iter()
                .any(|p| p.stage == ProblemStage::Capability && p.explanation.contains("beta")),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn an_unknown_capability_is_a_schema_problem() {
        let source = SIR.replace("sim.epidemiology.sir", "sim.nope.nothing");
        let outcome = validate_source(&source);
        assert!(!outcome.valid);
        assert!(
            outcome
                .problems
                .iter()
                .any(|p| p.stage == ProblemStage::Schema),
            "{:?}",
            outcome.problems
        );
    }

    #[test]
    fn every_problem_carries_a_title_and_explanation() {
        let source = SIR.replace(r#"beta = { value = 0.6, unit = "1/s" }"#, "");
        for problem in validate_source(&source).problems
        {
            assert!(!problem.title.is_empty());
            assert!(!problem.explanation.is_empty());
        }
    }

    #[test]
    fn a_field_name_is_extracted_from_a_backticked_explanation() {
        assert_eq!(
            field_from_explanation("field `beta` is required").as_deref(),
            Some("beta")
        );
        assert_eq!(field_from_explanation("no field named here"), None);
        assert_eq!(
            field_from_explanation("a `phrase with spaces` is not a field"),
            None
        );
    }

    /// The real phrasing the capability validators use.
    #[test]
    fn an_indexed_field_path_resolves_to_its_field() {
        assert_eq!(
            field_from_explanation("`initial_state.s[0]` = 5 is outside the accepted range")
                .as_deref(),
            Some("initial_state.s")
        );
        assert_eq!(
            field_from_explanation("`model.beta` is required").as_deref(),
            Some("model.beta")
        );
        // A malformed index is not silently accepted.
        assert_eq!(field_from_explanation("`weird[unclosed` thing"), None);
    }

    #[test]
    fn a_dotted_path_is_located_by_its_final_segment() {
        let source = "[initial_state]\ns = [{ value = 0.9 }]\ni = [{ value = 0.1 }]\n";
        let loc = locate_named_field(source, "initial_state.s").unwrap();
        assert_eq!(loc.line, 2);
    }

    #[test]
    fn a_field_is_located_at_its_assignment_line() {
        let source = "[model]\nbeta = 1\ngamma = 2\n";
        let loc = locate_named_field(source, "gamma").unwrap();
        assert_eq!(loc.line, 3);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn an_absent_field_has_no_location() {
        assert!(locate_named_field("[model]\nbeta = 1\n", "delta").is_none());
    }

    /// A field mentioned only inside a comment must not be treated as the
    /// assignment — jumping there would send the user to the wrong line.
    #[test]
    fn a_mention_that_is_not_an_assignment_is_not_located() {
        let source = "# beta matters a lot\n[model]\ngamma = 2\n";
        assert!(locate_named_field(source, "beta").is_none());
    }

    #[test]
    fn a_capability_problem_points_at_the_offending_line_when_it_can() {
        // An out-of-range value: the field exists in the source, so the
        // problem should carry its location.
        let source = SIR.replace(
            r#"s = [{ value = 0.999, unit = "1" }]"#,
            r#"s = [{ value = 5.0, unit = "1" }]"#,
        );
        let outcome = validate_source(&source);
        assert!(!outcome.valid);
        let located = outcome.problems.iter().any(|p| p.location.is_some());
        assert!(located, "{:?}", outcome.problems);
    }
}
