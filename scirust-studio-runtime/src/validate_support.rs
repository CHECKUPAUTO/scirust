//! Shared capability-specific validation helpers.
//!
//! Every adapter in this crate calls into these functions with its *own*
//! [`FieldDescriptor`] table rather than re-implementing the same
//! "is the field present, does its unit resolve to the right dimension, is
//! it in range" logic five times. What is genuinely capability-specific —
//! which fields exist, their dimensions, their ranges, which solvers are
//! supported — still lives entirely in each adapter module.

use scirust_studio_command::{CatalogedError, ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityDescriptor, Cardinality, DeterminismClass, FieldDescriptor,
    PrecisionKind, SolverDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::ensemble::MAX_REPLICATES;

/// Generic (not field-specific) capability-validation error codes. Field
/// -specific codes live on each adapter's own `FieldDescriptor`s.
pub const CODE_UNKNOWN_FIELD: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 90);
pub const CODE_UNSUPPORTED_SOLVER: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 91);
pub const CODE_MISSING_STEP: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 92);
pub const CODE_MISSING_TOLERANCE: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 93);
pub const CODE_SUM_CONSTRAINT: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 94);
/// A stochastic capability was given no `experiment.seed`.
pub const CODE_MISSING_SEED: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 95);
/// `experiment.replicates` was set above 1 on a capability that draws no
/// sample, so the realisations would all be identical.
pub const CODE_REPLICATES_UNSUPPORTED: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 96);
/// `experiment.replicates` was zero, or above [`MAX_REPLICATES`].
pub const CODE_REPLICATES_OUT_OF_RANGE: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 97);
/// `backend.precision` names a precision this capability does not compute in.
pub const CODE_UNSUPPORTED_PRECISION: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 98);
/// `backend.kind` names a backend this capability does not run on.
pub const CODE_UNSUPPORTED_BACKEND: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 99);

/// The fixed-step RK4 descriptor most capabilities declare.
///
/// Eight adapters had been repeating the same four fields with the same
/// summary. A capability whose RK4 needs saying something *particular* — the
/// heat rod's stability bound, the two-body's symplectic alternative — still
/// declares its own; this is only for the ones where the honest summary is
/// the generic one.
pub const RK4_SOLVER: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Classical fixed-step Runge-Kutta 4. Requires `solver.step`.",
    fixed_step: true,
    adaptive_tolerance: false,
};

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

/// Resolve the seed a stochastic capability must run with.
///
/// A deterministic capability never calls this: its result does not depend on
/// a seed, and demanding one would be theatre.
///
/// A stochastic capability calls it and is **refused without a seed**. That is
/// a deliberate constraint rather than a convenience default. A result is only
/// evidence if someone else can obtain it again, and for a single sample from
/// a distribution the seed is the entire difference between "here is a
/// trajectory" and "here is *the* trajectory these inputs produce". Picking
/// one silently — from the clock, from the OS, from a hard-coded constant —
/// would either make the run unreproducible or make every run identical while
/// looking as though it had been sampled; both are worse than asking.
pub fn resolve_seed(scenario: &Scenario) -> Result<u64, CatalogedError> {
    scenario.experiment.seed.ok_or_else(|| CatalogedError {
        code: CODE_MISSING_SEED,
        title: "Missing seed".to_string(),
        explanation: "this capability is stochastic: its result is one sample from a \
                      distribution, and without `experiment.seed` that sample cannot be \
                      reproduced"
            .to_string(),
        recoverable: true,
        suggested_action: Some(
            "add `seed = 42` (or any integer you choose) under [experiment]".to_string(),
        ),
    })
}

/// Resolve `experiment.replicates` for a capability of the given determinism
/// class.
///
/// Absent means one realisation, which is what every capability produced
/// before ensembles existed and what every deterministic one still produces.
///
/// **Every** adapter calls this, not only the stochastic ones, and that is the
/// point: a scenario asking a spring-mass-damper for 500 replicates is asking
/// for the same trajectory 500 times. Ignoring the field would waste the time
/// silently; honouring it would present 500 identical curves as a
/// distribution. It is refused instead, and the message says which of the two
/// the user probably meant.
///
/// `docs/studio/adr/0008-ensembles.md` records why the class — rather than a
/// per-adapter flag — decides this.
pub fn resolve_replicates(
    scenario: &Scenario,
    determinism: DeterminismClass,
) -> Result<usize, CatalogedError> {
    let Some(requested) = scenario.experiment.replicates
    else
    {
        return Ok(1);
    };

    if requested == 0
    {
        return Err(CatalogedError {
            code: CODE_REPLICATES_OUT_OF_RANGE,
            title: "Zero replicates".to_string(),
            explanation: "`experiment.replicates = 0` asks for a run with no realisations in \
                          it, which has no result to report"
                .to_string(),
            recoverable: true,
            suggested_action: Some(
                "use 1 for a single realisation, or remove the field — they mean the same thing"
                    .to_string(),
            ),
        });
    }

    // One replicate is the single-realisation case and is always allowed,
    // including for deterministic capabilities: it is what every scenario
    // without the field already asks for, so rejecting it would make an
    // explicit `replicates = 1` mean something different from omitting it.
    if requested == 1
    {
        return Ok(1);
    }

    if !determinism.draws_a_sample()
    {
        return Err(CatalogedError {
            code: CODE_REPLICATES_UNSUPPORTED,
            title: "This capability has nothing to draw".to_string(),
            explanation: format!(
                "`experiment.replicates = {requested}` asks for {requested} independent \
                 realisations, but this capability's determinism class is {determinism:?} — its \
                 result is a function of its parameters, so every realisation would be the same \
                 curve. An ensemble of identical curves is not a distribution"
            ),
            recoverable: true,
            suggested_action: Some(
                "remove `replicates` to run once; to vary the outcome, vary a parameter instead"
                    .to_string(),
            ),
        });
    }

    if requested > MAX_REPLICATES
    {
        return Err(CatalogedError {
            code: CODE_REPLICATES_OUT_OF_RANGE,
            title: "Too many replicates".to_string(),
            explanation: format!(
                "`experiment.replicates = {requested}` exceeds the limit of {MAX_REPLICATES}. \
                 The limit guards against a mistyped number; it is not a statement that \
                 {MAX_REPLICATES} is affordable, because the cost of an ensemble is replicates \
                 times steps and this bounds only the first"
            ),
            recoverable: true,
            suggested_action: Some(format!(
                "use at most {MAX_REPLICATES}; the standard error falls as 1/sqrt(n), so \
                 quadrupling the replicates only halves it"
            )),
        });
    }

    Ok(requested as usize)
}

/// Check `backend.precision` against the precisions the capability declares.
///
/// The schema already restricts the field to `"f32"` and `"f64"`, which is
/// what made this gap easy to miss: a scenario asking for `f32` passed
/// validation, and then every adapter computed in `f64` regardless, because
/// nothing compared the scenario's request against
/// [`CapabilityDescriptor::supported_precisions`]. The result was correct
/// arithmetic carrying a stated precision it did not have.
///
/// That is the same shape of defect as `experiment.seed` before Phase 3B-2 —
/// a field the schema accepts and nothing reads — and it is worse here,
/// because a silently-upgraded precision looks like the user got what they
/// asked for. `f32` is not a smaller `f64`; someone selecting it is usually
/// asking a question about conditioning or about matching another
/// implementation's arithmetic, and answering in `f64` answers a different
/// question.
///
/// Refusing costs the user one line. Every current capability declares `f64`
/// only, so today this always refuses `f32` — and when a capability grows an
/// `f32` path, this starts accepting it with no change here.
pub fn resolve_precision(
    scenario: &Scenario,
    descriptor: &CapabilityDescriptor,
) -> Result<PrecisionKind, CatalogedError> {
    let requested = match scenario.backend.precision.as_str()
    {
        "f64" => PrecisionKind::F64,
        "f32" => PrecisionKind::F32,
        // Unreachable through `scirust_studio_schema::validate`, which rejects
        // anything else first. Reported rather than asserted, because an
        // adapter can be driven directly.
        other =>
        {
            return Err(CatalogedError {
                code: CODE_UNSUPPORTED_PRECISION,
                title: "Unknown precision".to_string(),
                explanation: format!("`backend.precision = \"{other}\"` is not a precision"),
                recoverable: true,
                suggested_action: Some("use \"f64\", or \"f32\" where supported".to_string()),
            });
        },
    };

    if descriptor.supported_precisions.contains(&requested)
    {
        return Ok(requested);
    }

    let available: Vec<&str> = descriptor
        .supported_precisions
        .iter()
        .map(|p| match p
        {
            PrecisionKind::F64 => "f64",
            PrecisionKind::F32 => "f32",
        })
        .collect();
    Err(CatalogedError {
        code: CODE_UNSUPPORTED_PRECISION,
        title: "Unsupported precision".to_string(),
        explanation: format!(
            "`backend.precision = \"{}\"` was requested, but `{}` computes in {} only. \
             Running it anyway would record a result whose stated precision is not the \
             precision it was computed at",
            scenario.backend.precision,
            descriptor.id.0,
            available.join(" or ")
        ),
        recoverable: true,
        suggested_action: Some(format!(
            "set `precision = \"{}\"` under [backend], or remove the field to take the default",
            available.first().copied().unwrap_or("f64")
        )),
    })
}

/// Check `backend.kind` against the backends the capability declares.
///
/// The same gap as [`resolve_precision`], and closed at the same time for the
/// same reason rather than waiting for it to matter. The schema restricts the
/// field to `"cpu"` and every capability declares `Cpu`, so this refuses
/// nothing today; it is what makes the first GPU-only — or CPU-only, once a
/// GPU worker exists — capability fail loudly instead of running somewhere it
/// never claimed to.
pub fn resolve_backend_kind(
    scenario: &Scenario,
    descriptor: &CapabilityDescriptor,
) -> Result<BackendKind, CatalogedError> {
    let requested = match scenario.backend.kind.as_str()
    {
        "cpu" => BackendKind::Cpu,
        other =>
        {
            return Err(CatalogedError {
                code: CODE_UNSUPPORTED_BACKEND,
                title: "Unknown backend".to_string(),
                explanation: format!(
                    "`backend.kind = \"{other}\"` is not a backend this build has"
                ),
                recoverable: true,
                suggested_action: Some("use \"cpu\"".to_string()),
            });
        },
    };

    if descriptor.supported_backends.contains(&requested)
    {
        return Ok(requested);
    }
    Err(CatalogedError {
        code: CODE_UNSUPPORTED_BACKEND,
        title: "Unsupported backend".to_string(),
        explanation: format!(
            "`backend.kind = \"{}\"` was requested, but `{}` does not declare it",
            scenario.backend.kind, descriptor.id.0
        ),
        recoverable: true,
        suggested_action: Some("remove the field to take the default".to_string()),
    })
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
