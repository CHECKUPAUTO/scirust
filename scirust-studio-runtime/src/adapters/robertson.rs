//! `sim.chemistry.robertson` — the canonical stiff-ODE benchmark
//! (`scirust_sim::chemistry::Robertson`), integrated through
//! `scirust_sim::stiff_bridge::simulate_rosenbrock` (the `stiff` feature).
//!
//! Chosen as a representative adapter specifically because it *cannot* use
//! the fixed-step explicit engine every other adapter in this crate uses —
//! `scirust-sim`'s own tests demonstrate RK4 blowing up on this system — so
//! it forces the architecture to support a genuinely different solver
//! family (adaptive, linearly-implicit) and to *reject* a request for an
//! unsuitable solver explicitly rather than silently falling back to one.

use scirust_sim::chemistry::Robertson;
use scirust_sim::stiff_bridge::simulate_rosenbrock;
use scirust_studio_command::{CatalogedError, ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::result::{
    AxisDescriptor, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, Series, VerificationResult, VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    CODE_MISSING_STEP, check_sum_constraint, check_unknown_model_fields,
    check_unknown_state_fields, resolve_model_scalar, resolve_solver, resolve_state_vector,
};

const K1: FieldDescriptor = FieldDescriptor {
    canonical_name: "k1",
    display_name: "Rate constant k1 (A -> B)",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate constant for A -> B. In this dimensionless-fraction formulation all three rate constants share the dimension 1/time.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 140),
};

const K2: FieldDescriptor = FieldDescriptor {
    canonical_name: "k2",
    display_name: "Rate constant k2 (B+B -> C+B)",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate constant for the autocatalytic B + B -> C + B step.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 141),
};

const K3: FieldDescriptor = FieldDescriptor {
    canonical_name: "k3",
    display_name: "Rate constant k3 (B+C -> A+C)",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate constant for B + C -> A + C.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 142),
};

const SPECIES_A: FieldDescriptor = FieldDescriptor {
    canonical_name: "a",
    display_name: "Species A fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial mass fraction of species A.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 143),
};

const SPECIES_B: FieldDescriptor = FieldDescriptor {
    canonical_name: "b",
    display_name: "Species B fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial mass fraction of species B.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 144),
};

const SPECIES_C: FieldDescriptor = FieldDescriptor {
    canonical_name: "c",
    display_name: "Species C fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial mass fraction of species C.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 145),
};

const STIFF_ROSENBROCK: SolverDescriptor = SolverDescriptor {
    id: "stiff_rosenbrock_w",
    summary: "Adaptive, linearly-implicit Rosenbrock-W(2,3) from scirust-stiff — the recommended stiff integrator; `solver.step` is used as its initial step guess h0.",
    fixed_step: false,
    adaptive_tolerance: true,
};

const MASS_CONSERVATION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "mass_conservation",
    description: "a + b + c is a linear invariant (the three reaction rates sum to zero at any state); conserved regardless of how stiff k1/k2/k3 make the system.",
};

const NON_NEGATIVE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "non_negative_concentrations",
    description: "Every species fraction must stay within a small numerical tolerance of non-negative.",
};

const SOLVER_COMPLETION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "solver_completion",
    description: "The adaptive stiff solver must reach solver.end without a step-size underflow or non-finite state.",
};

/// The capability descriptor for `sim.chemistry.robertson`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.chemistry.robertson"),
    display_name: "Robertson stiff kinetics",
    category: CapabilityCategory::Chemistry,
    source_crate: "scirust-sim",
    summary: "The canonical Robertson autocatalytic stiff-ODE benchmark, integrated with the adaptive Rosenbrock-W stiff solver.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[STIFF_ROSENBROCK],
    parameters: &[K1, K2, K3],
    initial_state: &[SPECIES_A, SPECIES_B, SPECIES_C],
    outputs: &[
        OutputDescriptor {
            id: "a",
            display_name: "Species A",
            unit: "1",
            description: "Species A fraction over time.",
        },
        OutputDescriptor {
            id: "b",
            display_name: "Species B",
            unit: "1",
            description: "Species B fraction over time.",
        },
        OutputDescriptor {
            id: "c",
            display_name: "Species C",
            unit: "1",
            description: "Species C fraction over time.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[
            MASS_CONSERVATION_CHECK,
            NON_NEGATIVE_CHECK,
            SOLVER_COMPLETION_CHECK,
        ],
    },
};

/// The `sim.chemistry.robertson` adapter.
#[derive(Debug, Default)]
pub struct RobertsonAdapter;

impl CapabilityAdapter for RobertsonAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["k1", "k2", "k3"]));
        errors.extend(check_unknown_state_fields(scenario, &["a", "b", "c"]));
        let mut resolved_state = Vec::new();
        for e in [
            resolve_model_scalar(scenario, &K1).err(),
            resolve_model_scalar(scenario, &K2).err(),
            resolve_model_scalar(scenario, &K3).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for r in [
            resolve_state_vector(scenario, &SPECIES_A),
            resolve_state_vector(scenario, &SPECIES_B),
            resolve_state_vector(scenario, &SPECIES_C),
        ]
        {
            match r
            {
                Ok(v) => resolved_state.push(v[0]),
                Err(e) => errors.push(e),
            }
        }
        if resolved_state.len() == 3
        {
            if let Err(e) =
                check_sum_constraint(&resolved_state, 1.0, 1e-6, "initial_state a + b + c")
            {
                errors.push(e);
            }
        }
        // resolve_solver checks the id against supported_solvers and (since
        // STIFF_ROSENBROCK.adaptive_tolerance = true) that rtol/atol are
        // present; it does NOT check `solver.step`, because this solver is
        // genuinely not fixed-step. `solver.step` is still required here —
        // it is the initial step guess h0 `simulate_rosenbrock` takes as an
        // explicit argument — so that requirement is enforced directly.
        match resolve_solver(scenario, DESCRIPTOR.supported_solvers)
        {
            Ok(_) =>
            {
                if scenario.solver.step.is_none()
                {
                    errors.push(CatalogedError {
                        code: CODE_MISSING_STEP,
                        title: "Missing initial step".to_string(),
                        explanation: "solver `stiff_rosenbrock_w` requires `solver.step` as its initial step guess h0".to_string(),
                        recoverable: true,
                        suggested_action: Some("add `step = { value = ..., unit = \"s\" }` under [solver]".to_string()),
                    });
                }
            },
            Err(e) => errors.push(e),
        }
        if !errors.is_empty()
        {
            return Err(ValidationReport { errors });
        }
        Ok(ValidatedScenario::new(scenario.clone()))
    }

    fn execute(
        &self,
        scenario: &ValidatedScenario,
        control: &ExecutionControl,
        sink: &mut dyn EventSink,
    ) -> Result<RunResult, ExecutionError> {
        sink.emit(RunEvent::Started);
        if control.is_cancelled()
        {
            sink.emit(RunEvent::Cancelled);
            return Err(ExecutionError::Cancelled);
        }
        let scn = scenario.scenario();
        let k1 = resolve_model_scalar(scn, &K1).expect("validated");
        let k2 = resolve_model_scalar(scn, &K2).expect("validated");
        let k3 = resolve_model_scalar(scn, &K3).expect("validated");
        let a0 = resolve_state_vector(scn, &SPECIES_A).expect("validated")[0];
        let b0 = resolve_state_vector(scn, &SPECIES_B).expect("validated")[0];
        let c0 = resolve_state_vector(scn, &SPECIES_C).expect("validated")[0];

        let model = Robertson::new(k1, k2, k3)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;

        let t0 = scn
            .solver
            .start
            .to_quantity("solver.start")
            .expect("validated")
            .value;
        let t1 = scn
            .solver
            .end
            .to_quantity("solver.end")
            .expect("validated")
            .value;
        let h0 = scn
            .solver
            .step
            .as_ref()
            .expect("validated: stiff_rosenbrock_w requires solver.step as h0")
            .to_quantity("solver.step")
            .expect("validated")
            .value;
        let rtol = scn
            .solver
            .rtol
            .expect("validated: adaptive solver requires solver.rtol");
        let atol = scn
            .solver
            .atol
            .expect("validated: adaptive solver requires solver.atol");

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let traj =
            simulate_rosenbrock(&model, &[a0, b0, c0], t0, t1, rtol, atol, h0).map_err(|e| {
                sink.emit(RunEvent::Failed(e.to_string()));
                ExecutionError::Numerical(e.to_string())
            })?;

        let a_series = traj.column(0).expect("dim 0 exists");
        let b_series = traj.column(1).expect("dim 1 exists");
        let c_series = traj.column(2).expect("dim 2 exists");

        let mass_drift = traj
            .y
            .iter()
            .map(|row| (row[0] + row[1] + row[2] - 1.0).abs())
            .fold(0.0_f64, f64::max);
        let mass_threshold = 1e-6;

        let min_component = traj
            .y
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min);
        let non_negative_threshold = -1e-6;

        let last = traj
            .last_state()
            .expect("simulate_rosenbrock always returns at least the initial state");
        let accepted_steps = traj.len() - 1;

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: scn.experiment.name.clone(),
                steps: accepted_steps,
                t_start: t0,
                t_end: traj.last_time().unwrap_or(t1),
            },
            axes: vec![AxisDescriptor {
                id: "t".to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
            }],
            series: vec![
                Series {
                    id: "a".to_string(),
                    display_name: "Species A".to_string(),
                    unit: "1".to_string(),
                    values: a_series,
                },
                Series {
                    id: "b".to_string(),
                    display_name: "Species B".to_string(),
                    unit: "1".to_string(),
                    values: b_series,
                },
                Series {
                    id: "c".to_string(),
                    display_name: "Species C".to_string(),
                    unit: "1".to_string(),
                    values: c_series,
                },
            ],
            metrics: vec![
                Metric {
                    id: "final_a".to_string(),
                    display_name: "Final species A".to_string(),
                    value: MetricValue::Scalar(last[0]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_b".to_string(),
                    display_name: "Final species B".to_string(),
                    value: MetricValue::Scalar(last[1]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_c".to_string(),
                    display_name: "Final species C".to_string(),
                    value: MetricValue::Scalar(last[2]),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![
                VerificationResult {
                    id: "mass_conservation".to_string(),
                    status: if mass_drift <= mass_threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(mass_drift),
                    threshold: Some(mass_threshold),
                    explanation: format!("max|a+b+c-1| over the trajectory = {mass_drift:.3e}"),
                },
                VerificationResult {
                    id: "non_negative_concentrations".to_string(),
                    status: if min_component >= non_negative_threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(min_component),
                    threshold: Some(non_negative_threshold),
                    explanation: format!(
                        "minimum species fraction over the trajectory = {min_component:.3e}"
                    ),
                },
                VerificationResult {
                    id: "solver_completion".to_string(),
                    status: VerificationStatus::Passed,
                    measured: Some(accepted_steps as f64),
                    threshold: None,
                    explanation: format!(
                        "stiff_rosenbrock_w completed {accepted_steps} accepted steps and reached t = {:.6}",
                        traj.last_time().unwrap_or(t1)
                    ),
                },
            ],
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
            },
        };
        crate::result::assert_finite(&result).map_err(ExecutionError::Internal)?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    /// The actual shipped tutorial scenario (`docs/studio/tutorials/`),
    /// executed here rather than duplicated, so the file a user is told to
    /// run is the file that is tested.
    const SCENARIO: &str =
        include_str!("../../../docs/studio/tutorials/robertson_stiff.scirust.toml");

    fn adapter_and_scenario() -> (RobertsonAdapter, scirust_studio_schema::Scenario) {
        (RobertsonAdapter, parse_toml(SCENARIO).unwrap())
    }

    #[test]
    fn matches_the_published_reference_solution_at_t_0_4() {
        let (adapter, scenario) = adapter_and_scenario();
        let validated = adapter.validate(&scenario).expect("valid");
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes");
        let final_a = result.metrics.iter().find(|m| m.id == "final_a").unwrap();
        let MetricValue::Scalar(a) = final_a.value
        else
        {
            panic!("scalar")
        };
        // Hairer & Wanner's reference point: a(0.4) ~ 0.9851.
        assert!((a - 0.9851).abs() < 2e-3, "a(0.4) = {a}");
        let mass = result
            .verifications
            .iter()
            .find(|v| v.id == "mass_conservation")
            .unwrap();
        assert_eq!(mass.status, VerificationStatus::Passed);
        let non_neg = result
            .verifications
            .iter()
            .find(|v| v.id == "non_negative_concentrations")
            .unwrap();
        assert_eq!(non_neg.status, VerificationStatus::Passed);
        let completion = result
            .verifications
            .iter()
            .find(|v| v.id == "solver_completion")
            .unwrap();
        assert_eq!(completion.status, VerificationStatus::Passed);
    }

    #[test]
    fn explicitly_rejects_rk4_instead_of_silently_falling_back() {
        let (adapter, mut scenario) = adapter_and_scenario();
        scenario.solver.id = "rk4".to_string();
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("rk4") && e.explanation.contains("not supported"))
        );
    }

    #[test]
    fn rejects_initial_state_not_summing_to_one() {
        let (adapter, mut scenario) = adapter_and_scenario();
        scenario.initial_state.insert(
            "b".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 0.5,
                unit: "1".to_string(),
            }],
        );
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.explanation.contains("sum")));
    }

    #[test]
    fn rejects_missing_initial_step_h0() {
        let (adapter, mut scenario) = adapter_and_scenario();
        scenario.solver.step = None;
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == CODE_MISSING_STEP));
    }

    #[test]
    fn rejects_missing_tolerances() {
        let (adapter, mut scenario) = adapter_and_scenario();
        scenario.solver.rtol = None;
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(!report.errors.is_empty());
    }
}
