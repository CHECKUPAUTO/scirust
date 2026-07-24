//! Shared capability-specific validation helpers.
//!
//! Every adapter in this crate calls into these functions with its *own*
//! [`FieldDescriptor`] table rather than re-implementing the same
//! "is the field present, does its unit resolve to the right dimension, is
//! it in range" logic five times. What is genuinely capability-specific —
//! which fields exist, their dimensions, their ranges, which solvers are
//! supported — still lives entirely in each adapter module.

use scirust_studio_command::{CatalogedError, ErrorCode, ErrorFamily};
use scirust_studio_registry::{Cardinality, FieldDescriptor, SolverDescriptor};
use scirust_studio_schema::Scenario;

/// Generic (not field-specific) capability-validation error codes. Field
/// -specific codes live on each adapter's own `FieldDescriptor`s.
pub const CODE_UNKNOWN_FIELD: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 90);
pub const CODE_UNSUPPORTED_SOLVER: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 91);
pub const CODE_MISSING_STEP: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 92);
pub const CODE_MISSING_TOLERANCE: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 93);
pub const CODE_SUM_CONSTRAINT: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 94);

fn field_error(
    field: &FieldDescriptor,
    explanation: String,
    suggested_action: Option<String>,
) -> CatalogedError {
    CatalogedError {
        code: field.error_code,
        title: format!("Invalid `{}`", field.canonical_name),
        explanation,
        recoverable: true,
        suggested_action,
    }
}

fn in_range(field: &FieldDescriptor, value: f64) -> bool {
    let above_min = match (field.min, field.min_inclusive)
    {
        (Some(min), true) => value >= min,
        (Some(min), false) => value > min,
        (None, _) => true,
    };
    let below_max = match (field.max, field.max_inclusive)
    {
        (Some(max), true) => value <= max,
        (Some(max), false) => value < max,
        (None, _) => true,
    };
    above_min && below_max
}

fn range_description(field: &FieldDescriptor) -> String {
    let lo = match (field.min, field.min_inclusive)
    {
        (Some(min), true) => format!("[{min}"),
        (Some(min), false) => format!("({min}"),
        (None, _) => "(-inf".to_string(),
    };
    let hi = match (field.max, field.max_inclusive)
    {
        (Some(max), true) => format!("{max}]"),
        (Some(max), false) => format!("{max})"),
        (None, _) => "inf)".to_string(),
    };
    format!("{lo}, {hi}")
}

/// Resolve one required-or-optional scalar `model.*` parameter: checks
/// presence, unit resolution, dimension, and range. Returns the
/// SI-coherent value, or `field.default` when the field is absent and
/// optional.
pub fn resolve_model_scalar(
    scenario: &Scenario,
    field: &FieldDescriptor,
) -> Result<f64, CatalogedError> {
    let Some(raw) = scenario.model.get(field.canonical_name)
    else
    {
        return match (field.required, field.default)
        {
            (false, Some(default)) => Ok(default),
            _ => Err(field_error(
                field,
                format!(
                    "`model.{}` is required but was not provided",
                    field.canonical_name
                ),
                Some(format!(
                    "add `{} = {{ value = ..., unit = ... }}` under [model]",
                    field.canonical_name
                )),
            )),
        };
    };
    let quantity = raw.to_quantity(field.canonical_name).map_err(|e| {
        field_error(
            field,
            e.to_string(),
            Some("check the value and unit".to_string()),
        )
    })?;
    if quantity.dim != field.dimension
    {
        return Err(field_error(
            field,
            format!(
                "`model.{}` has dimension {} but this field requires {}",
                field.canonical_name, quantity.dim, field.dimension
            ),
            Some(format!("use one of: {}", field.accepted_units.join(", "))),
        ));
    }
    if !in_range(field, quantity.value)
    {
        return Err(field_error(
            field,
            format!(
                "`model.{}` = {} is outside the accepted range {}",
                field.canonical_name,
                quantity.value,
                range_description(field)
            ),
            None,
        ));
    }
    Ok(quantity.value)
}

/// Resolve one `initial_state.*` component: checks presence, cardinality
/// (component count), unit resolution, dimension, and range for each
/// component. Returns the SI-coherent values in order.
pub fn resolve_state_vector(
    scenario: &Scenario,
    field: &FieldDescriptor,
) -> Result<Vec<f64>, CatalogedError> {
    let expected_len = match field.cardinality
    {
        Cardinality::Scalar => 1,
        Cardinality::Vector(n) => n,
    };
    let Some(components) = scenario.initial_state.get(field.canonical_name)
    else
    {
        return Err(field_error(
            field,
            format!(
                "`initial_state.{}` is required but was not provided",
                field.canonical_name
            ),
            Some(format!(
                "add `{} = [{{ value = ..., unit = ... }}{}]` under [initial_state]",
                field.canonical_name,
                ", ...".repeat(expected_len.saturating_sub(1))
            )),
        ));
    };
    if components.len() != expected_len
    {
        return Err(field_error(
            field,
            format!(
                "`initial_state.{}` has {} component(s), expected exactly {expected_len}",
                field.canonical_name,
                components.len()
            ),
            None,
        ));
    }
    let mut values = Vec::with_capacity(expected_len);
    for (i, raw) in components.iter().enumerate()
    {
        let component_field = format!("{}[{i}]", field.canonical_name);
        let quantity = raw.to_quantity(&component_field).map_err(|e| {
            field_error(
                field,
                e.to_string(),
                Some("check the value and unit".to_string()),
            )
        })?;
        if quantity.dim != field.dimension
        {
            return Err(field_error(
                field,
                format!(
                    "`initial_state.{component_field}` has dimension {} but this field requires {}",
                    quantity.dim, field.dimension
                ),
                Some(format!("use one of: {}", field.accepted_units.join(", "))),
            ));
        }
        if !in_range(field, quantity.value)
        {
            return Err(field_error(
                field,
                format!(
                    "`initial_state.{component_field}` = {} is outside the accepted range {}",
                    quantity.value,
                    range_description(field)
                ),
                None,
            ));
        }
        values.push(quantity.value);
    }
    Ok(values)
}

/// Reject any `model.*` key not named in `known` — an unrecognised
/// parameter is a mistake to surface, not to silently ignore.
pub fn check_unknown_model_fields(scenario: &Scenario, known: &[&str]) -> Vec<CatalogedError> {
    scenario
        .model
        .keys()
        .filter(|k| !known.contains(&k.as_str()))
        .map(|k| CatalogedError {
            code: CODE_UNKNOWN_FIELD,
            title: "Unknown model parameter".to_string(),
            explanation: format!("`model.{k}` is not a parameter this capability accepts"),
            recoverable: true,
            suggested_action: Some(format!(
                "remove `model.{k}`, or check for a typo (known: {})",
                known.join(", ")
            )),
        })
        .collect()
}

/// Reject any `initial_state.*` key not named in `known`.
pub fn check_unknown_state_fields(scenario: &Scenario, known: &[&str]) -> Vec<CatalogedError> {
    scenario
        .initial_state
        .keys()
        .filter(|k| !known.contains(&k.as_str()))
        .map(|k| CatalogedError {
            code: CODE_UNKNOWN_FIELD,
            title: "Unknown initial-state component".to_string(),
            explanation: format!("`initial_state.{k}` is not a component this capability accepts"),
            recoverable: true,
            suggested_action: Some(format!(
                "remove `initial_state.{k}`, or check for a typo (known: {})",
                known.join(", ")
            )),
        })
        .collect()
}

/// Resolve `scenario.solver` against a capability's supported solver list:
/// checks the solver id is supported, and that a fixed step or adaptive
/// tolerances are present exactly when that solver needs them.
pub fn resolve_solver<'a>(
    scenario: &Scenario,
    supported: &'a [SolverDescriptor],
) -> Result<&'a SolverDescriptor, CatalogedError> {
    let Some(solver) = supported.iter().find(|s| s.id == scenario.solver.id)
    else
    {
        let ids: Vec<&str> = supported.iter().map(|s| s.id).collect();
        return Err(CatalogedError {
            code: CODE_UNSUPPORTED_SOLVER,
            title: "Unsupported solver".to_string(),
            explanation: format!(
                "`solver.id = \"{}\"` is not supported by this capability",
                scenario.solver.id
            ),
            recoverable: true,
            suggested_action: Some(format!("use one of: {}", ids.join(", "))),
        });
    };
    if solver.fixed_step && scenario.solver.step.is_none()
    {
        return Err(CatalogedError {
            code: CODE_MISSING_STEP,
            title: "Missing step".to_string(),
            explanation: format!(
                "solver `{}` is fixed-step and requires `solver.step`",
                solver.id
            ),
            recoverable: true,
            suggested_action: Some(
                "add `step = { value = ..., unit = \"s\" }` under [solver]".to_string(),
            ),
        });
    }
    if solver.adaptive_tolerance
        && (scenario.solver.rtol.is_none() || scenario.solver.atol.is_none())
    {
        return Err(CatalogedError {
            code: CODE_MISSING_TOLERANCE,
            title: "Missing tolerance".to_string(),
            explanation: format!(
                "solver `{}` is adaptive and requires both `solver.rtol` and `solver.atol`",
                solver.id
            ),
            recoverable: true,
            suggested_action: Some("add `rtol = ...` and `atol = ...` under [solver]".to_string()),
        });
    }
    Ok(solver)
}

/// Check that a set of SI-coherent values sums to `expected` within
/// `tolerance` (e.g. Robertson's `a0 + b0 + c0 ≈ 1`).
pub fn check_sum_constraint(
    values: &[f64],
    expected: f64,
    tolerance: f64,
    description: &str,
) -> Result<(), CatalogedError> {
    let sum: f64 = values.iter().sum();
    if (sum - expected).abs() > tolerance
    {
        return Err(CatalogedError {
            code: CODE_SUM_CONSTRAINT,
            title: "Sum constraint violated".to_string(),
            explanation: format!(
                "{description}: sum = {sum}, expected {expected} (tolerance {tolerance})"
            ),
            recoverable: true,
            suggested_action: Some(format!("adjust the initial state so it sums to {expected}")),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_studio_schema::parse_toml;

    const MASS: FieldDescriptor = FieldDescriptor {
        canonical_name: "mass",
        display_name: "Mass",
        required: true,
        dimension: scirust_units::Dimension::MASS,
        accepted_units: &["kg"],
        min: Some(0.0),
        min_inclusive: false,
        max: None,
        max_inclusive: false,
        default: None,
        cardinality: Cardinality::Scalar,
        description: "test mass field",
        error_code: ErrorCode::new(ErrorFamily::Validation, 900),
    };

    const POSITION: FieldDescriptor = FieldDescriptor {
        canonical_name: "position",
        display_name: "Position",
        required: true,
        dimension: scirust_units::Dimension::LENGTH,
        accepted_units: &["m"],
        min: None,
        min_inclusive: false,
        max: None,
        max_inclusive: false,
        default: None,
        cardinality: Cardinality::Vector(2),
        description: "test position field",
        error_code: ErrorCode::new(ErrorFamily::Validation, 901),
    };

    fn scenario_with(model: &str, initial_state: &str, solver: &str) -> Scenario {
        let text = format!(
            "schema_version = 1\n[experiment]\nname = \"t\"\n[capability]\nid = \"test.capability\"\n[model]\n{model}\n[initial_state]\n{initial_state}\n[solver]\n{solver}\n"
        );
        parse_toml(&text).expect("valid TOML shape")
    }

    #[test]
    fn resolve_model_scalar_accepts_a_valid_field() {
        let scenario = scenario_with(
            "mass = { value = 2.0, unit = \"kg\" }",
            "",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        assert_eq!(resolve_model_scalar(&scenario, &MASS).unwrap(), 2.0);
    }

    #[test]
    fn resolve_model_scalar_rejects_missing_required_field() {
        let scenario = scenario_with(
            "",
            "",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let err = resolve_model_scalar(&scenario, &MASS).unwrap_err();
        assert_eq!(err.code, MASS.error_code);
    }

    #[test]
    fn resolve_model_scalar_rejects_wrong_dimension() {
        let scenario = scenario_with(
            "mass = { value = 2.0, unit = \"m\" }",
            "",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let err = resolve_model_scalar(&scenario, &MASS).unwrap_err();
        assert!(err.explanation.contains("dimension"), "{}", err.explanation);
    }

    #[test]
    fn resolve_model_scalar_rejects_out_of_range() {
        let scenario = scenario_with(
            "mass = { value = -1.0, unit = \"kg\" }",
            "",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let err = resolve_model_scalar(&scenario, &MASS).unwrap_err();
        assert!(err.explanation.contains("range"), "{}", err.explanation);
    }

    #[test]
    fn resolve_model_scalar_uses_default_when_optional_and_absent() {
        let optional = FieldDescriptor {
            required: false,
            default: Some(9.0),
            ..MASS
        };
        let scenario = scenario_with(
            "",
            "",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        assert_eq!(resolve_model_scalar(&scenario, &optional).unwrap(), 9.0);
    }

    #[test]
    fn resolve_state_vector_checks_cardinality() {
        let scenario = scenario_with(
            "",
            "position = [{ value = 1.0, unit = \"m\" }]",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let err = resolve_state_vector(&scenario, &POSITION).unwrap_err();
        assert!(err.explanation.contains("component"), "{}", err.explanation);
    }

    #[test]
    fn resolve_state_vector_accepts_matching_cardinality() {
        let scenario = scenario_with(
            "",
            "position = [{ value = 1.0, unit = \"m\" }, { value = 2.0, unit = \"m\" }]",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        assert_eq!(
            resolve_state_vector(&scenario, &POSITION).unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn check_unknown_model_fields_flags_unrecognised_keys() {
        let scenario = scenario_with(
            "mass = { value = 1.0, unit = \"kg\" }\ntypo_field = { value = 1.0, unit = \"kg\" }",
            "",
            "id = \"x\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let errors = check_unknown_model_fields(&scenario, &["mass"]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].explanation.contains("typo_field"));
    }

    #[test]
    fn resolve_solver_rejects_unsupported_id() {
        let scenario = scenario_with(
            "",
            "",
            "id = \"unsupported\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let rk4 = SolverDescriptor {
            id: "rk4",
            summary: "s",
            fixed_step: true,
            adaptive_tolerance: false,
        };
        let err = resolve_solver(&scenario, std::slice::from_ref(&rk4)).unwrap_err();
        assert_eq!(err.code, CODE_UNSUPPORTED_SOLVER);
    }

    #[test]
    fn resolve_solver_requires_step_for_fixed_step_solvers() {
        let scenario = scenario_with(
            "",
            "",
            "id = \"rk4\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let rk4 = SolverDescriptor {
            id: "rk4",
            summary: "s",
            fixed_step: true,
            adaptive_tolerance: false,
        };
        let err = resolve_solver(&scenario, std::slice::from_ref(&rk4)).unwrap_err();
        assert_eq!(err.code, CODE_MISSING_STEP);
    }

    #[test]
    fn resolve_solver_requires_tolerances_for_adaptive_solvers() {
        let scenario = scenario_with(
            "",
            "",
            "id = \"stiff\"\nstart = { value = 0.0, unit = \"s\" }\nend = { value = 1.0, unit = \"s\" }",
        );
        let stiff = SolverDescriptor {
            id: "stiff",
            summary: "s",
            fixed_step: false,
            adaptive_tolerance: true,
        };
        let err = resolve_solver(&scenario, std::slice::from_ref(&stiff)).unwrap_err();
        assert_eq!(err.code, CODE_MISSING_TOLERANCE);
    }

    #[test]
    fn check_sum_constraint_rejects_violation() {
        let err = check_sum_constraint(&[0.5, 0.2], 1.0, 1e-9, "test").unwrap_err();
        assert_eq!(err.code, CODE_SUM_CONSTRAINT);
    }

    #[test]
    fn check_sum_constraint_accepts_within_tolerance() {
        assert!(check_sum_constraint(&[0.6, 0.4], 1.0, 1e-9, "test").is_ok());
    }
}
