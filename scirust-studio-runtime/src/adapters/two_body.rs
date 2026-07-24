//! `sim.orbital.two_body` — the planar two-body (Kepler) problem
//! (`scirust_sim::orbital::TwoBody`).
//!
//! Chosen as a representative adapter because its state is genuinely
//! vector-valued (a 2-D position and velocity, not a single scalar or a set
//! of independent scalars) and because the real model supports two
//! qualitatively different solvers with different long-horizon behaviour
//! (bounded vs. drifting energy), which exercises `supported_solvers`
//! having more than one entry and the "unsupported solver" validation path
//! meaningfully.

use scirust_sim::engine::FirstOrderForm;
use scirust_sim::orbital::TwoBody;
use scirust_sim::{simulate, simulate_second_order};
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

const MU: FieldDescriptor = FieldDescriptor {
    canonical_name: "mu",
    display_name: "Gravitational parameter",
    required: true,
    dimension: scirust_units::Dimension::LENGTH
        .powi(3)
        .div(scirust_units::Dimension::TIME.powi(2)),
    accepted_units: &["m^3/s^2"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Standard gravitational parameter mu = G*M of the fixed primary.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 120),
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
    cardinality: Cardinality::Vector(2),
    description: "Initial planar position [x, y].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 121),
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
    cardinality: Cardinality::Vector(2),
    description: "Initial planar velocity [vx, vy].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 122),
};

const SYMPLECTIC_EULER: SolverDescriptor = SolverDescriptor {
    id: "symplectic_euler",
    summary: "Fixed-step semi-implicit Euler; energy error stays bounded over long horizons.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta; more accurate short-horizon, but energy drifts secularly over many orbits.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const ENERGY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_drift",
    description: "Specific orbital energy v^2/2 - mu/r is conserved by the real physics; measures the integrator's relative drift from its initial value.",
};

const ANGULAR_MOMENTUM_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "angular_momentum_drift",
    description: "Specific angular momentum x*vy - y*vx is conserved by the real physics (central force); measures the integrator's relative drift.",
};

const FINITE_TRAJECTORY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "finite_trajectory",
    description: "Every recorded state must be finite; scirust_sim itself rejects a blown-up integration before this check ever sees it, so this records that guarantee explicitly.",
};

/// The capability descriptor for `sim.orbital.two_body`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.orbital.two_body"),
    display_name: "Two-body orbit",
    category: CapabilityCategory::Orbital,
    source_crate: "scirust-sim",
    summary: "The planar two-body (Kepler) problem around a fixed primary, integrated with symplectic Euler or RK4.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[SYMPLECTIC_EULER, RK4],
    parameters: &[MU],
    initial_state: &[POSITION, VELOCITY],
    outputs: &[
        OutputDescriptor {
            id: "position_x",
            display_name: "Position x",
            unit: "m",
            description: "x-coordinate over time.",
        },
        OutputDescriptor {
            id: "position_y",
            display_name: "Position y",
            unit: "m",
            description: "y-coordinate over time.",
        },
        OutputDescriptor {
            id: "velocity_x",
            display_name: "Velocity x",
            unit: "m/s",
            description: "x-velocity over time.",
        },
        OutputDescriptor {
            id: "velocity_y",
            display_name: "Velocity y",
            unit: "m/s",
            description: "y-velocity over time.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[
            ENERGY_CHECK,
            ANGULAR_MOMENTUM_CHECK,
            FINITE_TRAJECTORY_CHECK,
        ],
    },
};

/// The `sim.orbital.two_body` adapter.
#[derive(Debug, Default)]
pub struct TwoBodyAdapter;

impl CapabilityAdapter for TwoBodyAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["mu"]));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["position", "velocity"],
        ));
        if let Err(e) = resolve_model_scalar(scenario, &MU)
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
        let scn = scenario.scenario();
        let mu = resolve_model_scalar(scn, &MU).expect("validated");
        let position = resolve_state_vector(scn, &POSITION).expect("validated");
        let velocity = resolve_state_vector(scn, &VELOCITY).expect("validated");
        let q0 = [position[0], position[1]];
        let v0 = [velocity[0], velocity[1]];

        let model =
            TwoBody::new(mu).map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;

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
            .expect("validated: both solvers are fixed-step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        // Both solvers produce Trajectory rows shaped [x, y, vx, vy]:
        // simulate_second_order concatenates (q, v) directly, and
        // FirstOrderForm's dim = 2*dof with the same [q, v] split.
        let traj = match scn.solver.id.as_str()
        {
            "symplectic_euler" => simulate_second_order(&model, &q0, &v0, t0, t1, step),
            "rk4" =>
            {
                let y0 = [q0[0], q0[1], v0[0], v0[1]];
                simulate(&FirstOrderForm(&model), &y0, t0, t1, step)
            },
            other =>
            {
                return Err(ExecutionError::Internal(format!(
                    "validated but unhandled solver id `{other}`"
                )));
            },
        }
        .map_err(|e| {
            sink.emit(RunEvent::Failed(e.to_string()));
            ExecutionError::Numerical(e.to_string())
        })?;

        let position_x = traj.column(0).expect("dim 0 exists");
        let position_y = traj.column(1).expect("dim 1 exists");
        let velocity_x = traj.column(2).expect("dim 2 exists");
        let velocity_y = traj.column(3).expect("dim 3 exists");

        let energy0 = model
            .energy(&q0, &v0)
            .expect("q0/v0 have length 2 and r != 0 (validated: mu > 0 with a finite position)");
        let angmom0 = model
            .angular_momentum(&q0, &v0)
            .expect("q0/v0 have length 2");

        let mut max_energy_drift = 0.0_f64;
        let mut max_angmom_drift = 0.0_f64;
        let mut all_finite = true;
        for row in &traj.y
        {
            if row.iter().any(|c| !c.is_finite())
            {
                all_finite = false;
                continue;
            }
            let (q, v) = ([row[0], row[1]], [row[2], row[3]]);
            if let Some(e) = model.energy(&q, &v)
            {
                max_energy_drift =
                    max_energy_drift.max((e - energy0).abs() / energy0.abs().max(1e-300));
            }
            if let Some(h) = model.angular_momentum(&q, &v)
            {
                max_angmom_drift =
                    max_angmom_drift.max((h - angmom0).abs() / angmom0.abs().max(1e-300));
            }
        }

        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");
        let energy_threshold = 1e-3;
        let angmom_threshold = 1e-3;

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
                    id: "position_x".to_string(),
                    display_name: "Position x".to_string(),
                    unit: "m".to_string(),
                    values: position_x,
                },
                Series {
                    id: "position_y".to_string(),
                    display_name: "Position y".to_string(),
                    unit: "m".to_string(),
                    values: position_y,
                },
                Series {
                    id: "velocity_x".to_string(),
                    display_name: "Velocity x".to_string(),
                    unit: "m/s".to_string(),
                    values: velocity_x,
                },
                Series {
                    id: "velocity_y".to_string(),
                    display_name: "Velocity y".to_string(),
                    unit: "m/s".to_string(),
                    values: velocity_y,
                },
            ],
            metrics: vec![
                Metric {
                    id: "final_x".to_string(),
                    display_name: "Final x".to_string(),
                    value: MetricValue::Scalar(last[0]),
                    unit: Some("m".to_string()),
                },
                Metric {
                    id: "final_y".to_string(),
                    display_name: "Final y".to_string(),
                    value: MetricValue::Scalar(last[1]),
                    unit: Some("m".to_string()),
                },
                Metric {
                    id: "final_vx".to_string(),
                    display_name: "Final vx".to_string(),
                    value: MetricValue::Scalar(last[2]),
                    unit: Some("m/s".to_string()),
                },
                Metric {
                    id: "final_vy".to_string(),
                    display_name: "Final vy".to_string(),
                    value: MetricValue::Scalar(last[3]),
                    unit: Some("m/s".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![
                VerificationResult {
                    id: "energy_drift".to_string(),
                    status: if max_energy_drift <= energy_threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(max_energy_drift),
                    threshold: Some(energy_threshold),
                    explanation: format!(
                        "max relative specific-energy drift over the trajectory = {max_energy_drift:.3e}"
                    ),
                },
                VerificationResult {
                    id: "angular_momentum_drift".to_string(),
                    status: if max_angmom_drift <= angmom_threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(max_angmom_drift),
                    threshold: Some(angmom_threshold),
                    explanation: format!(
                        "max relative specific-angular-momentum drift over the trajectory = {max_angmom_drift:.3e}"
                    ),
                },
                VerificationResult {
                    id: "finite_trajectory".to_string(),
                    status: if all_finite
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: None,
                    threshold: None,
                    explanation: if all_finite
                    {
                        "every recorded state was finite".to_string()
                    }
                    else
                    {
                        "a non-finite state was recorded (this should not happen: scirust_sim's own integrator rejects blow-up before returning)".to_string()
                    },
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
    /// varied by solver id — so the file a user is told to run is the file
    /// that is tested. It ships with `id = "symplectic_euler"`.
    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/two_body_orbit.scirust.toml");

    fn scenario_with_solver(solver_id: &str) -> scirust_studio_schema::Scenario {
        let text = TUTORIAL.replace(
            "id = \"symplectic_euler\"",
            &format!("id = \"{solver_id}\""),
        );
        parse_toml(&text).unwrap()
    }

    #[test]
    fn symplectic_euler_keeps_energy_and_angular_momentum_bounded() {
        let adapter = TwoBodyAdapter;
        let scenario = scenario_with_solver("symplectic_euler");
        let validated = adapter.validate(&scenario).expect("valid");
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes");
        let energy = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_drift")
            .unwrap();
        assert_eq!(
            energy.status,
            VerificationStatus::Passed,
            "{:?}",
            energy.measured
        );
        let angmom = result
            .verifications
            .iter()
            .find(|v| v.id == "angular_momentum_drift")
            .unwrap();
        assert_eq!(
            angmom.status,
            VerificationStatus::Passed,
            "{:?}",
            angmom.measured
        );
    }

    #[test]
    fn rk4_also_conserves_over_one_orbit() {
        let adapter = TwoBodyAdapter;
        let scenario = scenario_with_solver("rk4");
        let validated = adapter.validate(&scenario).expect("valid");
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes");
        let energy = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_drift")
            .unwrap();
        assert_eq!(
            energy.status,
            VerificationStatus::Passed,
            "{:?}",
            energy.measured
        );
    }

    #[test]
    fn produces_four_split_series_for_the_vector_state() {
        let adapter = TwoBodyAdapter;
        let scenario = scenario_with_solver("symplectic_euler");
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();
        let ids: Vec<&str> = result.series.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["position_x", "position_y", "velocity_x", "velocity_y"]
        );
    }

    #[test]
    fn rejects_unsupported_solver() {
        let adapter = TwoBodyAdapter;
        let scenario = scenario_with_solver("dopri5");
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("dopri5"))
        );
    }

    #[test]
    fn rejects_position_with_wrong_cardinality() {
        let adapter = TwoBodyAdapter;
        let mut scenario = scenario_with_solver("rk4");
        scenario.initial_state.insert(
            "position".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 1.0,
                unit: "m".to_string(),
            }],
        );
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == POSITION.error_code));
    }

    #[test]
    fn rejects_non_positive_mu() {
        let adapter = TwoBodyAdapter;
        let mut scenario = scenario_with_solver("rk4");
        scenario.model.insert(
            "mu".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "m^3/s^2".to_string(),
            },
        );
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == MU.error_code));
    }
}
