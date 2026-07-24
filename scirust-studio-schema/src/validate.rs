//! Content validation for a parsed [`crate::Scenario`].

use crate::error::SchemaError;
use crate::scenario::{CURRENT_SCHEMA_VERSION, Scenario};

/// Maximum accepted length for a free-text field.
const MAX_STRING_LEN: usize = 4096;
/// Maximum number of series `outputs.record` may list.
const MAX_OUTPUT_RECORD_COUNT: usize = 64;

/// Validate a parsed scenario's *content* (parsing already checked its
/// shape). Every violation found is returned — not just the first — so a
/// caller can show them all at once instead of a fix-one-rerun loop.
///
/// `known_capability_ids`, when `Some`, is checked against
/// `scenario.capability.id`. Pass `None` when the caller has no capability
/// registry available; this crate deliberately does not depend on one.
///
/// An empty return value means the scenario is valid.
pub fn validate(scenario: &Scenario, known_capability_ids: Option<&[&str]>) -> Vec<SchemaError> {
    let mut errors = Vec::new();

    if scenario.schema_version != CURRENT_SCHEMA_VERSION
    {
        errors.push(SchemaError::UnsupportedSchemaVersion {
            found: scenario.schema_version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }

    if let Some(known) = known_capability_ids
    {
        if !known.contains(&scenario.capability.id.as_str())
        {
            errors.push(SchemaError::UnknownCapability {
                id: scenario.capability.id.clone(),
            });
        }
    }

    check_string_len(&mut errors, "experiment.name", &scenario.experiment.name);
    if let Some(desc) = &scenario.experiment.description
    {
        check_string_len(&mut errors, "experiment.description", desc);
    }

    match scenario.backend.precision.as_str()
    {
        "f32" | "f64" =>
        {},
        other => errors.push(SchemaError::UnsupportedPrecision {
            precision: other.to_string(),
        }),
    }
    match scenario.backend.kind.as_str()
    {
        "cpu" =>
        {},
        other => errors.push(SchemaError::UnsupportedBackendKind {
            kind: other.to_string(),
        }),
    }

    for (name, qty) in &scenario.model
    {
        if let Err(e) = qty.to_quantity(&format!("model.{name}"))
        {
            errors.push(e);
        }
    }
    for (name, components) in &scenario.initial_state
    {
        for (i, qty) in components.iter().enumerate()
        {
            if let Err(e) = qty.to_quantity(&format!("initial_state.{name}[{i}]"))
            {
                errors.push(e);
            }
        }
    }

    let start = scenario.solver.start.to_quantity("solver.start");
    let end = scenario.solver.end.to_quantity("solver.end");
    if let (Ok(s), Ok(e)) = (&start, &end)
    {
        if e.value <= s.value
        {
            errors.push(SchemaError::EndNotAfterStart {
                start: s.value,
                end: e.value,
            });
        }
    }
    if let Err(e) = start
    {
        errors.push(e);
    }
    if let Err(e) = end
    {
        errors.push(e);
    }

    if let Some(step) = &scenario.solver.step
    {
        match step.to_quantity("solver.step")
        {
            Ok(q) if q.value <= 0.0 =>
            {
                errors.push(SchemaError::NonPositiveStep { step: q.value });
            },
            Ok(_) =>
            {},
            Err(e) => errors.push(e),
        }
    }

    if scenario.outputs.record.len() > MAX_OUTPUT_RECORD_COUNT
    {
        errors.push(SchemaError::TooManyOutputs {
            count: scenario.outputs.record.len(),
            max: MAX_OUTPUT_RECORD_COUNT,
        });
    }
    if scenario.outputs.sample_every == Some(0)
    {
        errors.push(SchemaError::ZeroSampleEvery);
    }

    errors
}

fn check_string_len(errors: &mut Vec<SchemaError>, field: &str, text: &str) {
    if text.len() > MAX_STRING_LEN
    {
        errors.push(SchemaError::StringTooLong {
            field: field.to_string(),
            length: text.len(),
            max: MAX_STRING_LEN,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{BackendConfig, parse_toml};

    fn valid_scenario() -> Scenario {
        parse_toml(
            r#"
schema_version = 1
[experiment]
name = "valid"
[capability]
id = "sim.mechanics.spring_mass_damper"
[model]
mass = { value = 1.0, unit = "kg" }
[initial_state]
position = [{ value = 1.0, unit = "m" }]
velocity = [{ value = 0.0, unit = "m/s" }]
[solver]
id = "rk4"
start = { value = 0.0, unit = "s" }
end = { value = 10.0, unit = "s" }
step = { value = 0.01, unit = "s" }
[outputs]
record = ["position"]
"#,
        )
        .unwrap()
    }

    #[test]
    fn a_well_formed_scenario_has_no_errors() {
        let scenario = valid_scenario();
        assert_eq!(validate(&scenario, None), Vec::new());
    }

    #[test]
    fn a_well_formed_scenario_passes_a_matching_known_capability_list() {
        let scenario = valid_scenario();
        let known = ["sim.mechanics.spring_mass_damper", "sim.epidemiology.sir"];
        assert_eq!(validate(&scenario, Some(&known)), Vec::new());
    }

    #[test]
    fn rejects_unknown_capability_id() {
        let scenario = valid_scenario();
        let known = ["sim.epidemiology.sir"];
        let errors = validate(&scenario, Some(&known));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::UnknownCapability { .. }))
        );
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let mut scenario = valid_scenario();
        scenario.schema_version = 99;
        let errors = validate(&scenario, None);
        assert!(errors.iter().any(|e| matches!(
            e,
            SchemaError::UnsupportedSchemaVersion {
                found: 99,
                supported: 1
            }
        )));
    }

    #[test]
    fn rejects_end_before_or_equal_to_start() {
        let mut scenario = valid_scenario();
        scenario.solver.end = scenario.solver.start.clone();
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::EndNotAfterStart { .. }))
        );
    }

    #[test]
    fn rejects_non_positive_step() {
        let mut scenario = valid_scenario();
        scenario.solver.step = Some(crate::scenario::ValueWithUnit {
            value: 0.0,
            unit: "s".to_string(),
        });
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::NonPositiveStep { .. }))
        );
    }

    #[test]
    fn rejects_non_finite_model_value() {
        let mut scenario = valid_scenario();
        scenario.model.insert(
            "mass".to_string(),
            crate::scenario::ValueWithUnit {
                value: f64::INFINITY,
                unit: "kg".to_string(),
            },
        );
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::NonFinite { .. }))
        );
    }

    #[test]
    fn rejects_unknown_unit_in_initial_state() {
        let mut scenario = valid_scenario();
        scenario.initial_state.insert(
            "position".to_string(),
            vec![crate::scenario::ValueWithUnit {
                value: 1.0,
                unit: "furlong".to_string(),
            }],
        );
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::UnknownUnit { .. }))
        );
    }

    #[test]
    fn rejects_unsupported_precision_and_backend_kind() {
        let mut scenario = valid_scenario();
        scenario.backend = BackendConfig {
            kind: "gpu".to_string(),
            precision: "f16".to_string(),
            threads: None,
            deterministic: true,
        };
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::UnsupportedPrecision { .. }))
        );
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::UnsupportedBackendKind { .. }))
        );
    }

    #[test]
    fn rejects_oversized_strings() {
        let mut scenario = valid_scenario();
        scenario.experiment.name = "x".repeat(MAX_STRING_LEN + 1);
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::StringTooLong { .. }))
        );
    }

    #[test]
    fn rejects_too_many_output_series() {
        let mut scenario = valid_scenario();
        scenario.outputs.record = (0..MAX_OUTPUT_RECORD_COUNT + 1)
            .map(|i| format!("s{i}"))
            .collect();
        let errors = validate(&scenario, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, SchemaError::TooManyOutputs { .. }))
        );
    }

    #[test]
    fn rejects_zero_sample_every() {
        let mut scenario = valid_scenario();
        scenario.outputs.sample_every = Some(0);
        let errors = validate(&scenario, None);
        assert!(errors.contains(&SchemaError::ZeroSampleEvery));
    }

    #[test]
    fn reports_every_violation_at_once_not_just_the_first() {
        let mut scenario = valid_scenario();
        scenario.schema_version = 99;
        scenario.outputs.sample_every = Some(0);
        scenario.experiment.name = "x".repeat(MAX_STRING_LEN + 1);
        let errors = validate(&scenario, None);
        assert!(
            errors.len() >= 3,
            "expected at least 3 errors, got {errors:?}"
        );
    }
}
