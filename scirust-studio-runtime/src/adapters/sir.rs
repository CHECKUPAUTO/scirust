//! `sim.epidemiology.sir` — the Kermack-McKendrick SIR compartmental model
//! (`scirust_sim::epidemiology::Sir`).
//!
//! Chosen as a representative adapter (per
//! `docs/studio/adr/0000-scope-and-sequencing.md`'s Phase 2A plan) because
//! its state is three coupled dimensionless fractions rather than a single
//! mechanical degree of freedom, and its scientific invariant (population
//! conservation) is a *sum* across components rather than an energy
//! expression.

use scirust_sim::epidemiology::Sir;
use scirust_sim::simulate;
use scirust_studio_command::{ErrorCode, ErrorFamily};
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
    check_unknown_model_fields, check_unknown_state_fields, resolve_model_scalar, resolve_solver,
    resolve_state_vector,
};

const BETA: FieldDescriptor = FieldDescriptor {
    canonical_name: "beta",
    display_name: "Transmission rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Transmission rate beta in s' = -beta*s*i.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 110),
};

const GAMMA: FieldDescriptor = FieldDescriptor {
    canonical_name: "gamma",
    display_name: "Recovery rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Recovery rate gamma in i' = beta*s*i - gamma*i.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 111),
};

const SUSCEPTIBLE: FieldDescriptor = FieldDescriptor {
    canonical_name: "s",
    display_name: "Susceptible fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial susceptible population fraction.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 112),
};

const INFECTED: FieldDescriptor = FieldDescriptor {
    canonical_name: "i",
    display_name: "Infected fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial infected population fraction.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 113),
};

const RECOVERED: FieldDescriptor = FieldDescriptor {
    canonical_name: "r",
    display_name: "Recovered fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial recovered population fraction.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 114),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const POPULATION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "population_conservation",
    description: "s + i + r is a linear invariant of the model; RK4 preserves it to round-off.",
};

const NON_NEGATIVE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "non_negative_compartments",
    description: "Every compartment must stay within a small numerical tolerance of non-negative.",
};

/// The capability descriptor for `sim.epidemiology.sir`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.epidemiology.sir"),
    display_name: "SIR epidemic model",
    category: CapabilityCategory::Epidemiology,
    source_crate: "scirust-sim",
    summary: "The Kermack-McKendrick SIR compartmental epidemic model, integrated with fixed-step RK4.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[BETA, GAMMA],
    initial_state: &[SUSCEPTIBLE, INFECTED, RECOVERED],
    outputs: &[
        OutputDescriptor {
            id: "s",
            display_name: "Susceptible",
            unit: "1",
            description: "Susceptible fraction over time.",
        },
        OutputDescriptor {
            id: "i",
            display_name: "Infected",
            unit: "1",
            description: "Infected fraction over time.",
        },
        OutputDescriptor {
            id: "r",
            display_name: "Recovered",
            unit: "1",
            description: "Recovered fraction over time.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[POPULATION_CHECK, NON_NEGATIVE_CHECK],
    },
};

/// The `sim.epidemiology.sir` adapter.
#[derive(Debug, Default)]
pub struct SirAdapter;

impl CapabilityAdapter for SirAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["beta", "gamma"]));
        errors.extend(check_unknown_state_fields(scenario, &["s", "i", "r"]));
        for e in [
            resolve_model_scalar(scenario, &BETA).err(),
            resolve_model_scalar(scenario, &GAMMA).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for e in [
            resolve_state_vector(scenario, &SUSCEPTIBLE).err(),
            resolve_state_vector(scenario, &INFECTED).err(),
            resolve_state_vector(scenario, &RECOVERED).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        if let Err(e) = resolve_solver(scenario, DESCRIPTOR.supported_solvers)
        {
            errors.push(e);
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
        let beta = resolve_model_scalar(scn, &BETA).expect("validated");
        let gamma = resolve_model_scalar(scn, &GAMMA).expect("validated");
        let s0 = resolve_state_vector(scn, &SUSCEPTIBLE).expect("validated")[0];
        let i0 = resolve_state_vector(scn, &INFECTED).expect("validated")[0];
        let r0 = resolve_state_vector(scn, &RECOVERED).expect("validated")[0];

        let model =
            Sir::new(beta, gamma).map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;

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
        let step = scn
            .solver
            .step
            .as_ref()
            .expect("validated: rk4 requires solver.step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let traj = simulate(&model, &[s0, i0, r0], t0, t1, step).map_err(|e| {
            sink.emit(RunEvent::Failed(e.to_string()));
            ExecutionError::Numerical(e.to_string())
        })?;

        let s_series = traj.column(0).expect("dim 0 exists");
        let i_series = traj.column(1).expect("dim 1 exists");
        let r_series = traj.column(2).expect("dim 2 exists");

        let population_drift = traj
            .y
            .iter()
            .map(|row| (row[0] + row[1] + row[2] - 1.0).abs())
            .fold(0.0_f64, f64::max);
        let population_threshold = 1e-9;

        let min_component = traj
            .y
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min);
        let non_negative_threshold = -1e-9;

        let (peak_i, peak_time) = traj.t.iter().zip(i_series.iter()).fold(
            (f64::MIN, t0),
            |(best_i, best_t), (&t, &i)| {
                if i > best_i { (i, t) } else { (best_i, best_t) }
            },
        );

        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: scn.experiment.name.clone(),
                steps: traj.len() - 1,
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
                    id: "s".to_string(),
                    display_name: "Susceptible".to_string(),
                    unit: "1".to_string(),
                    values: s_series,
                },
                Series {
                    id: "i".to_string(),
                    display_name: "Infected".to_string(),
                    unit: "1".to_string(),
                    values: i_series,
                },
                Series {
                    id: "r".to_string(),
                    display_name: "Recovered".to_string(),
                    unit: "1".to_string(),
                    values: r_series,
                },
            ],
            metrics: vec![
                Metric {
                    id: "peak_infected".to_string(),
                    display_name: "Peak infected fraction".to_string(),
                    value: MetricValue::Scalar(peak_i),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "peak_time".to_string(),
                    display_name: "Time of peak infection".to_string(),
                    value: MetricValue::Scalar(peak_time),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "final_s".to_string(),
                    display_name: "Final susceptible fraction".to_string(),
                    value: MetricValue::Scalar(last[0]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_i".to_string(),
                    display_name: "Final infected fraction".to_string(),
                    value: MetricValue::Scalar(last[1]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_r".to_string(),
                    display_name: "Final recovered fraction".to_string(),
                    value: MetricValue::Scalar(last[2]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "r0".to_string(),
                    display_name: "Basic reproduction number".to_string(),
                    value: MetricValue::Scalar(model.r0()),
                    unit: None,
                },
            ],
            warnings: vec![],
            verifications: vec![
                VerificationResult {
                    id: "population_conservation".to_string(),
                    status: if population_drift <= population_threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(population_drift),
                    threshold: Some(population_threshold),
                    explanation: format!(
                        "max|s+i+r-1| over the trajectory = {population_drift:.3e}"
                    ),
                },
                VerificationResult {
                    id: "non_negative_compartments".to_string(),
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
                        "minimum compartment value over the trajectory = {min_component:.3e}"
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
    const SCENARIO: &str = include_str!("../../../docs/studio/tutorials/sir_epidemic.scirust.toml");

    fn adapter_and_scenario() -> (SirAdapter, scirust_studio_schema::Scenario) {
        (SirAdapter, parse_toml(SCENARIO).unwrap())
    }

    #[test]
    fn executes_and_conserves_population() {
        let (adapter, scenario) = adapter_and_scenario();
        let validated = adapter.validate(&scenario).expect("valid");
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes");
        let pop = result
            .verifications
            .iter()
            .find(|v| v.id == "population_conservation")
            .unwrap();
        assert_eq!(pop.status, VerificationStatus::Passed);
        let non_neg = result
            .verifications
            .iter()
            .find(|v| v.id == "non_negative_compartments")
            .unwrap();
        assert_eq!(non_neg.status, VerificationStatus::Passed);
    }

    #[test]
    fn r0_above_one_produces_a_real_epidemic_peak() {
        let (adapter, scenario) = adapter_and_scenario();
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();
        let peak = result
            .metrics
            .iter()
            .find(|m| m.id == "peak_infected")
            .unwrap();
        let MetricValue::Scalar(peak_value) = peak.value
        else
        {
            panic!("expected scalar")
        };
        // beta/gamma = 3: a real outbreak must rise well above i0 = 0.001.
        assert!(peak_value > 0.2, "peak {peak_value}");
    }

    #[test]
    fn rejects_state_fraction_outside_zero_one() {
        let (adapter, mut scenario) = adapter_and_scenario();
        scenario.initial_state.insert(
            "s".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 1.5,
                unit: "1".to_string(),
            }],
        );
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == SUSCEPTIBLE.error_code)
        );
    }

    #[test]
    fn rejects_unknown_initial_state_field() {
        let (adapter, mut scenario) = adapter_and_scenario();
        scenario.initial_state.insert(
            "e".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "1".to_string(),
            }],
        );
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.explanation.contains('e')));
    }

    #[test]
    fn descriptor_declares_both_verification_checks() {
        assert_eq!(DESCRIPTOR.verification.checks.len(), 2);
    }
}
