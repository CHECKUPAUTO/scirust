//! `sim.mechanics.rigid_body` — free rotation of a torque-free rigid body
//! (`scirust_sim::rigid_body::RigidBody`).
//!
//! Euler's equations for a body spun about its principal axes, with no
//! external torque. State `y = [w1, w2, w3]`, the body-frame angular velocity.
//!
//! # Two exact invariants, and why both are needed
//!
//! With no torque the rotational kinetic energy and the *magnitude* of the
//! angular momentum are both conserved exactly. Checking only one would miss
//! a whole class of error: the trajectory is confined to the intersection of
//! an energy ellipsoid and a momentum sphere, and a wrong moment of inertia
//! can leave a body on the right sphere while drifting off the ellipsoid.
//! Together they pin the motion; separately neither does.
//!
//! # The tutorial spins about the intermediate axis
//!
//! Deliberately. A body spun about its largest or smallest principal axis is
//! stable and its invariants are easy to conserve; the *intermediate* axis is
//! unstable — the tennis-racket or Dzhanibekov effect — and the motion tumbles
//! through large excursions. That is where a solver's conservation is actually
//! tested, so that is what ships.
use scirust_sim::rigid_body::RigidBody;
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind,
    VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::execute_support::{TimeSpan, simulate_cancellable};
use crate::result::{
    Axis, AxisMonotonicity, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, Series, SeriesRole, TIME_AXIS_ID, VerificationResult, VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    RK4_SOLVER, check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// Relative drift allowed in each invariant.
///
/// Both are conserved *exactly* by the continuous equations, so anything here
/// is RK4's own error. At the tutorial's step that error is around 1e-11; a
/// threshold of 1e-6 leaves five orders of margin for a coarser step while
/// still catching the factor-level error a wrong moment of inertia produces.
const DRIFT_TOLERANCE: f64 = 1e-6;

const I1: FieldDescriptor = FieldDescriptor {
    canonical_name: "i1",
    display_name: "Principal moment I1",
    required: true,
    dimension: scirust_units::Dimension::MASS.mul(scirust_units::Dimension::LENGTH.powi(2)),
    accepted_units: &["kg*m^2"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Moment of inertia about principal axis 1.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 230),
};

const I2: FieldDescriptor = FieldDescriptor {
    canonical_name: "i2",
    display_name: "Principal moment I2",
    required: true,
    dimension: scirust_units::Dimension::MASS.mul(scirust_units::Dimension::LENGTH.powi(2)),
    accepted_units: &["kg*m^2"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Moment of inertia about principal axis 2.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 231),
};

const I3: FieldDescriptor = FieldDescriptor {
    canonical_name: "i3",
    display_name: "Principal moment I3",
    required: true,
    dimension: scirust_units::Dimension::MASS.mul(scirust_units::Dimension::LENGTH.powi(2)),
    accepted_units: &["kg*m^2"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Moment of inertia about principal axis 3.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 232),
};

const OMEGA: FieldDescriptor = FieldDescriptor {
    canonical_name: "omega",
    display_name: "Initial angular velocity",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["rad/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Vector(3),
    description: "Body-frame angular velocity at t = start, as [w1, w2, w3].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 233),
};

const ENERGY_DRIFT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_drift",
    description: "Rotational kinetic energy is conserved exactly by torque-free Euler equations, so any drift is the integrator's own error. Checked together with the momentum invariant because the trajectory lies on the intersection of an energy ellipsoid and a momentum sphere: a wrong moment of inertia can hold one and not the other.",
};

const ANGULAR_MOMENTUM_DRIFT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "angular_momentum_drift",
    description: "The squared angular-momentum magnitude is the second exact invariant. Independent of the energy check in exactly the way that matters — together they pin the motion, separately neither does.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.mechanics.rigid_body"),
    display_name: "Free rigid-body rotation",
    category: CapabilityCategory::Mechanics,
    source_crate: "scirust-sim",
    summary: "Torque-free rotation of a rigid body about its principal axes, with two exact invariants and the intermediate-axis instability.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[I1, I2, I3],
    initial_state: &[OMEGA],
    outputs: &[
        OutputDescriptor {
            id: "omega1",
            display_name: "Angular velocity 1",
            unit: "rad/s",
            description: "Body-frame angular velocity about axis 1.",
        },
        OutputDescriptor {
            id: "omega2",
            display_name: "Angular velocity 2",
            unit: "rad/s",
            description: "About axis 2.",
        },
        OutputDescriptor {
            id: "omega3",
            display_name: "Angular velocity 3",
            unit: "rad/s",
            description: "About axis 3.",
        },
        OutputDescriptor {
            id: "energy",
            display_name: "Kinetic energy",
            unit: "J",
            description: "Rotational kinetic energy, which must not drift.",
        },
        OutputDescriptor {
            id: "angular_momentum",
            display_name: "Angular momentum",
            unit: "1",
            description: "The magnitude of the angular momentum, which must not drift.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ENERGY_DRIFT_CHECK, ANGULAR_MOMENTUM_DRIFT_CHECK],
    },
};

/// The `sim.mechanics.rigid_body` adapter.
#[derive(Debug, Default)]
pub struct RigidBodyAdapter;

impl CapabilityAdapter for RigidBodyAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["i1", "i2", "i3"]));
        errors.extend(check_unknown_state_fields(scenario, &["omega"]));
        for e in [
            resolve_model_scalar(scenario, &I1).err(),
            resolve_model_scalar(scenario, &I2).err(),
            resolve_model_scalar(scenario, &I3).err(),
            resolve_state_vector(scenario, &OMEGA).err(),
            resolve_solver(scenario, DESCRIPTOR.supported_solvers).err(),
            resolve_replicates(scenario, DESCRIPTOR.determinism).err(),
            resolve_backend_kind(scenario, &DESCRIPTOR).err(),
            resolve_precision(scenario, &DESCRIPTOR).err(),
        ]
        .into_iter()
        .flatten()
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
        let s = scenario.scenario();
        let i1 = resolve_model_scalar(s, &I1).expect("validated");
        let i2 = resolve_model_scalar(s, &I2).expect("validated");
        let i3 = resolve_model_scalar(s, &I3).expect("validated");
        let w0 = resolve_state_vector(s, &OMEGA).expect("validated");

        let model = RigidBody::new(i1, i2, i3)
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

        let traj = simulate_cancellable(
            &model,
            &w0,
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let w1 = traj.column(0).expect("dim 0 exists");
        let w2 = traj.column(1).expect("dim 1 exists");
        let w3 = traj.column(2).expect("dim 2 exists");

        let energy: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.kinetic_energy(row).expect("dim 3"))
            .collect();
        // The magnitude rather than its square, so the series reads as a
        // physical quantity and the drift is measured on the same scale the
        // energy's is.
        let momentum: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.angular_momentum_squared(row).expect("dim 3").sqrt())
            .collect();

        let energy_drift = drift_check("energy_drift", "kinetic energy", &energy);
        let momentum_drift = drift_check(
            "angular_momentum_drift",
            "angular-momentum magnitude",
            &momentum,
        );

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                steps: traj.t.len() - 1,
                t_start: traj.t[0],
                t_end: *traj.t.last().expect("at least one sample"),
            },
            axes: vec![Axis {
                id: TIME_AXIS_ID.to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
                monotonicity: AxisMonotonicity::StrictlyIncreasing,
                values: traj.t.clone(),
            }],
            series: vec![
                Series {
                    id: "omega1".to_string(),
                    display_name: "Angular velocity 1".to_string(),
                    unit: "rad/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: w1.clone(),
                },
                Series {
                    id: "omega2".to_string(),
                    display_name: "Angular velocity 2".to_string(),
                    unit: "rad/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: w2.clone(),
                },
                Series {
                    id: "omega3".to_string(),
                    display_name: "Angular velocity 3".to_string(),
                    unit: "rad/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: w3.clone(),
                },
                Series {
                    id: "energy".to_string(),
                    display_name: "Kinetic energy".to_string(),
                    unit: "J".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: energy.clone(),
                },
                Series {
                    id: "angular_momentum".to_string(),
                    display_name: "Angular momentum".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: momentum.clone(),
                },
            ],
            fields: vec![],
            metrics: vec![
                Metric {
                    id: "initial_energy".to_string(),
                    display_name: "Initial kinetic energy".to_string(),
                    value: MetricValue::Scalar(energy[0]),
                    unit: Some("J".to_string()),
                },
                Metric {
                    id: "initial_momentum".to_string(),
                    display_name: "Initial angular-momentum magnitude".to_string(),
                    value: MetricValue::Scalar(momentum[0]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "intermediate_axis".to_string(),
                    display_name: "Spun about the intermediate axis".to_string(),
                    value: MetricValue::Text(intermediate_axis_label(i1, i2, i3, &w0)),
                    unit: None,
                },
            ],
            warnings: vec![],
            verifications: vec![energy_drift, momentum_drift],
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
                seed: None,
            },
        };
        crate::result::validate_result(&result)
            .map_err(|d| ExecutionError::Internal(crate::result::describe_defects(&d)))?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

/// Relative drift of a quantity that should not change at all.
fn drift_check(id: &str, name: &str, values: &[f64]) -> VerificationResult {
    let first = values.first().copied().unwrap_or(0.0);
    let scale = first.abs().max(f64::MIN_POSITIVE);
    let worst = values
        .iter()
        .map(|v| (v - first).abs() / scale)
        .fold(0.0_f64, f64::max);

    VerificationResult {
        id: id.to_string(),
        status: if worst <= DRIFT_TOLERANCE
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(worst),
        threshold: Some(DRIFT_TOLERANCE),
        explanation: format!(
            "the {name} started at {first:.9e} and drifted by at most {worst:.3e} of that \
             over the run; torque-free rotation conserves it exactly, so this is the \
             integrator's error and nothing else"
        ),
    }
}

/// Whether this run is the interesting case: spun near the intermediate axis.
///
/// Reported rather than checked. It is not a correctness property — a body
/// spun about any axis is a valid run — but it is what tells a reader whether
/// the invariants were tested against a tumbling motion or a placid one.
fn intermediate_axis_label(i1: f64, i2: f64, i3: f64, w0: &[f64]) -> String {
    let mut sorted = [(i1, 0usize), (i2, 1), (i3, 2)];
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted[1].1;
    let dominant = w0
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.abs()
                .partial_cmp(&b.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    if dominant == middle
    {
        "yes — the unstable case, where conservation is actually tested".to_string()
    }
    else
    {
        "no — spun about a stable axis".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/rigid_body.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = RigidBodyAdapter;
        let validated = adapter.validate(&scenario).expect("valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    fn check<'a>(result: &'a RunResult, id: &str) -> &'a VerificationResult {
        result
            .verifications
            .iter()
            .find(|v| v.id == id)
            .expect("check")
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        assert_eq!(parse_toml(TUTORIAL).unwrap().capability.id, DESCRIPTOR.id.0);
    }

    #[test]
    fn the_same_scenario_run_twice_is_bit_identical() {
        let (a, b) = (run(), run());
        assert!(
            a.series[0]
                .values
                .iter()
                .zip(b.series[0].values.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let validated = RigidBodyAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            RigidBodyAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn both_invariants_hold() {
        let r = run();
        for id in ["energy_drift", "angular_momentum_drift"]
        {
            let v = check(&r, id);
            assert_eq!(
                v.status,
                VerificationStatus::Passed,
                "{}: {}",
                id,
                v.explanation
            );
        }
    }

    /// The tutorial must actually exercise the unstable case, or the
    /// invariants are being checked against a motion that barely moves.
    #[test]
    fn the_tutorial_spins_about_the_intermediate_axis() {
        let r = run();
        let label = r
            .metrics
            .iter()
            .find(|m| m.id == "intermediate_axis")
            .expect("the metric");
        assert_eq!(
            label.value,
            MetricValue::Text(
                "yes — the unstable case, where conservation is actually tested".to_string()
            )
        );
    }

    /// …and the motion must genuinely tumble: the dominant axis has to hand
    /// its rotation to the others and take it back.
    #[test]
    fn the_motion_actually_tumbles() {
        let r = run();
        let w1 = &r
            .series
            .iter()
            .find(|s| s.id == "omega1")
            .expect("omega1")
            .values;
        let swing = w1.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - w1.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            swing > 1.0,
            "omega1 swung only {swing}, so this run does not test conservation against \
             a large excursion"
        );
    }

    #[test]
    fn the_drift_check_catches_a_quantity_that_moves() {
        let verdict = drift_check("energy_drift", "kinetic energy", &[1.0, 1.0, 1.5]);
        assert_eq!(verdict.status, VerificationStatus::Failed);
        assert!(verdict.measured.unwrap() > 0.4);
    }

    #[test]
    fn the_axis_label_reports_the_stable_case_too() {
        // I2 is the intermediate axis, but the body is spun about axis 3.
        assert!(intermediate_axis_label(1.0, 2.0, 3.0, &[0.0, 0.0, 5.0]).starts_with("no"));
        assert!(intermediate_axis_label(1.0, 2.0, 3.0, &[0.0, 5.0, 0.0]).starts_with("yes"));
    }
}
