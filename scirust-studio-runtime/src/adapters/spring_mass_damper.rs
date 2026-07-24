//! `sim.mechanics.spring_mass_damper` — a mass on a linear spring with
//! viscous damping (`scirust_sim::mechanics::SpringMassDamper`).
//!
//! Moved here from `scirust-cli` verbatim in spirit (same model, same
//! tutorial scenario, same energy-drift check); the CLI no longer
//! constructs `SpringMassDamper` itself.

use scirust_sim::mechanics::SpringMassDamper;
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
    description: "The mass on the spring.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 100),
};

const DAMPING: FieldDescriptor = FieldDescriptor {
    canonical_name: "damping",
    display_name: "Damping coefficient",
    required: true,
    dimension: scirust_units::Dimension::MASS.div(scirust_units::Dimension::TIME),
    accepted_units: &["kg/s"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Viscous damping coefficient c in m*x'' + c*x' + k*x = 0. Zero means undamped.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 101),
};

const STIFFNESS: FieldDescriptor = FieldDescriptor {
    canonical_name: "stiffness",
    display_name: "Spring stiffness",
    required: true,
    dimension: scirust_units::Dimension::MASS.div(scirust_units::Dimension::TIME.powi(2)),
    accepted_units: &["kg/s^2"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Spring constant k in m*x'' + c*x' + k*x = 0.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 102),
};

const POSITION: FieldDescriptor = FieldDescriptor {
    canonical_name: "position",
    display_name: "Initial position",
    required: true,
    dimension: scirust_units::Dimension::LENGTH,
    accepted_units: &["m"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial displacement from equilibrium.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 103),
};

const VELOCITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "velocity",
    display_name: "Initial velocity",
    required: true,
    dimension: scirust_units::Dimension::VELOCITY,
    accepted_units: &["m/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial velocity.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 104),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const ENERGY_DRIFT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_drift",
    description: "When damping = 0, mechanical energy m*v^2/2 + k*x^2/2 is conserved by the \
                   real physics; the check measures RK4's relative drift from that invariant.",
};

/// The capability descriptor for `sim.mechanics.spring_mass_damper`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.mechanics.spring_mass_damper"),
    display_name: "Spring-mass-damper",
    category: CapabilityCategory::Mechanics,
    source_crate: "scirust-sim",
    summary: "A mass on a linear spring with viscous damping, integrated with fixed-step RK4.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[MASS, DAMPING, STIFFNESS],
    initial_state: &[POSITION, VELOCITY],
    outputs: &[
        OutputDescriptor {
            id: "position",
            display_name: "Position",
            unit: "m",
            description: "Displacement over time.",
        },
        OutputDescriptor {
            id: "velocity",
            display_name: "Velocity",
            unit: "m/s",
            description: "Velocity over time.",
        },
        OutputDescriptor {
            id: "energy",
            display_name: "Energy",
            unit: "J",
            description: "Mechanical energy over time.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ENERGY_DRIFT_CHECK],
    },
};

/// The `sim.mechanics.spring_mass_damper` adapter.
#[derive(Debug, Default)]
pub struct SpringMassDamperAdapter;

impl CapabilityAdapter for SpringMassDamperAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["mass", "damping", "stiffness"],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["position", "velocity"],
        ));
        for e in [
            resolve_model_scalar(scenario, &MASS).err(),
            resolve_model_scalar(scenario, &DAMPING).err(),
            resolve_model_scalar(scenario, &STIFFNESS).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for e in [
            resolve_state_vector(scenario, &POSITION).err(),
            resolve_state_vector(scenario, &VELOCITY).err(),
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
        let s = scenario.scenario();
        let mass = resolve_model_scalar(s, &MASS).expect("validated");
        let damping = resolve_model_scalar(s, &DAMPING).expect("validated");
        let stiffness = resolve_model_scalar(s, &STIFFNESS).expect("validated");
        let position = resolve_state_vector(s, &POSITION).expect("validated");
        let velocity = resolve_state_vector(s, &VELOCITY).expect("validated");
        let (x0, v0) = (position[0], velocity[0]);

        let model = SpringMassDamper::new(mass, damping, stiffness)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;

        let t0 = s
            .solver
            .start
            .to_quantity("solver.start")
            .expect("validated")
            .value;
        let t1 = s
            .solver
            .end
            .to_quantity("solver.end")
            .expect("validated")
            .value;
        let step = s
            .solver
            .step
            .as_ref()
            .expect("validated: rk4 requires solver.step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let energy0 = model.energy(&[x0, v0]);
        let traj = simulate(&model, &[x0, v0], t0, t1, step).map_err(|e| {
            sink.emit(RunEvent::Failed(e.to_string()));
            ExecutionError::Numerical(e.to_string())
        })?;
        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");
        let energy1 = model.energy(last);

        let position_series = traj.column(0).expect("dim 0 exists");
        let velocity_series = traj.column(1).expect("dim 1 exists");
        let energy_series: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.energy(row).expect("state rows always have length 2"))
            .collect();

        let verification = match (damping == 0.0, energy0, energy1)
        {
            (true, Some(e0), Some(e1)) =>
            {
                let drift = (e1 - e0).abs() / e0.abs().max(1e-300);
                let threshold = 1e-6;
                VerificationResult {
                    id: "energy_drift".to_string(),
                    status: if drift <= threshold { VerificationStatus::Passed } else { VerificationStatus::Failed },
                    measured: Some(drift),
                    threshold: Some(threshold),
                    explanation: format!(
                        "undamped spring-mass-damper: relative energy drift {drift:.3e} (RK4 approximately conserves energy)"
                    ),
                }
            },
            _ =>
            {
                VerificationResult {
                    id: "energy_drift".to_string(),
                    status: VerificationStatus::NotApplicable,
                    measured: None,
                    threshold: None,
                    explanation: "damping > 0: energy is expected to decay, not conserve; this check only applies when damping = 0".to_string(),
                }
            },
        };

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
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
                    id: "position".to_string(),
                    display_name: "Position".to_string(),
                    unit: "m".to_string(),
                    values: position_series,
                },
                Series {
                    id: "velocity".to_string(),
                    display_name: "Velocity".to_string(),
                    unit: "m/s".to_string(),
                    values: velocity_series,
                },
                Series {
                    id: "energy".to_string(),
                    display_name: "Energy".to_string(),
                    unit: "J".to_string(),
                    values: energy_series,
                },
            ],
            metrics: vec![
                Metric {
                    id: "final_position".to_string(),
                    display_name: "Final position".to_string(),
                    value: MetricValue::Scalar(last[0]),
                    unit: Some("m".to_string()),
                },
                Metric {
                    id: "final_velocity".to_string(),
                    display_name: "Final velocity".to_string(),
                    value: MetricValue::Scalar(last[1]),
                    unit: Some("m/s".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![verification],
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

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/spring_mass_damper.scirust.toml");

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        assert_eq!(scenario.capability.id, DESCRIPTOR.id.0);
    }

    #[test]
    fn validates_and_executes_the_real_tutorial_scenario() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = SpringMassDamperAdapter;
        let validated = adapter
            .validate(&scenario)
            .expect("tutorial scenario is valid");
        let mut sink = NullEventSink;
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .expect("executes");
        assert_eq!(result.capability_id, "sim.mechanics.spring_mass_damper");
        assert_eq!(result.series.len(), 3);
        let energy_check = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_drift")
            .unwrap();
        assert_eq!(energy_check.status, VerificationStatus::Passed);
        assert!(
            energy_check.measured.unwrap() < 1e-9,
            "drift {:?}",
            energy_check.measured
        );
    }

    #[test]
    fn rejects_missing_required_model_field() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let mut scenario = scenario;
        scenario.model.remove("mass");
        let report = SpringMassDamperAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == MASS.error_code));
    }

    #[test]
    fn rejects_unknown_model_field() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "gravity".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 9.8,
                unit: "m/s^2".to_string(),
            },
        );
        let report = SpringMassDamperAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("gravity"))
        );
    }

    #[test]
    fn rejects_wrong_dimension_on_mass() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "mass".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 1.0,
                unit: "s".to_string(),
            },
        );
        let report = SpringMassDamperAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == MASS.error_code));
    }

    #[test]
    fn rejects_unsupported_solver() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.id = "dopri5".to_string();
        let report = SpringMassDamperAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("dopri5"))
        );
    }

    #[test]
    fn rejects_negative_stiffness_at_execution_as_invalid_model_state() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "stiffness".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: -4.0,
                unit: "kg/s^2".to_string(),
            },
        );
        // Negative stiffness is in-range per STIFFNESS's descriptor bounds
        // being [0, inf) is violated too, so this is actually already caught
        // by validate() — proving validation, not just the model's own
        // constructor, is the first line of defense.
        let report = SpringMassDamperAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == STIFFNESS.error_code));
    }

    #[test]
    fn emits_started_and_completed_events() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = SpringMassDamperAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        assert_eq!(sink.events().first(), Some(&RunEvent::Started));
        assert_eq!(sink.events().last(), Some(&RunEvent::Completed));
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = SpringMassDamperAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let mut sink = NullEventSink;
        let err = adapter
            .execute(&validated, &control, &mut sink)
            .unwrap_err();
        assert_eq!(err, ExecutionError::Cancelled);
    }
}
