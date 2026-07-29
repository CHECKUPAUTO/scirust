//! `sim.mechanics.pendulum` — a rigid pendulum under gravity
//! (`scirust_sim::mechanics::Pendulum`).
//!
//! The full nonlinear equation `θ'' = −(g/L)·sin θ − c·θ'`, not the
//! small-angle linearisation. That distinction is the point of the
//! capability: the linearised period `2π·√(L/g)` is *wrong* for a
//! large-amplitude swing, and the run reports both the small-angle period and
//! the period the trajectory actually shows, so the discrepancy is visible
//! rather than assumed away.
//!
//! Undamped, mechanical energy is conserved by the true dynamics, and that is
//! the verification check.

use scirust_sim::mechanics::Pendulum;

use crate::execute_support::{TimeSpan, simulate_cancellable};
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
    Axis, AxisMonotonicity, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, Series, SeriesRole, TIME_AXIS_ID, VerificationResult, VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

const LENGTH: FieldDescriptor = FieldDescriptor {
    canonical_name: "length",
    display_name: "Rod length",
    required: true,
    dimension: scirust_units::Dimension::LENGTH,
    accepted_units: &["m"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Length L of the rigid rod, in theta'' = -(g/L)*sin(theta) - c*theta'.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 170),
};

const GRAVITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "gravity",
    display_name: "Gravitational acceleration",
    required: true,
    dimension: scirust_units::Dimension::ACCELERATION,
    accepted_units: &["m/s^2"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Gravitational acceleration g.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 171),
};

const DAMPING: FieldDescriptor = FieldDescriptor {
    canonical_name: "damping",
    display_name: "Viscous damping",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Viscous damping c. Zero means undamped, which is when energy is conserved.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 172),
};

const ANGLE: FieldDescriptor = FieldDescriptor {
    canonical_name: "angle",
    display_name: "Initial angle",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["rad"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial angle from the downward vertical, in radians.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 173),
};

const ANGULAR_VELOCITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "angular_velocity",
    display_name: "Initial angular velocity",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["rad/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial angular velocity, in radians per second.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 174),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const ENERGY_DRIFT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_drift",
    description: "With damping = 0, the energy per unit mass \
                  L^2*omega^2/2 + g*L*(1 - cos theta) is conserved by the real physics; the \
                  check measures RK4's relative drift from it.",
};

const AMPLITUDE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "amplitude_bounded",
    description: "An undamped pendulum released from rest can never swing past its release \
                  angle: energy conservation forbids it. Exceeding it means the integration \
                  gained energy.",
};

/// The capability descriptor for `sim.mechanics.pendulum`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.mechanics.pendulum"),
    display_name: "Pendulum (nonlinear)",
    category: CapabilityCategory::Mechanics,
    source_crate: "scirust-sim",
    summary: "A rigid pendulum under gravity, solved without the small-angle approximation, \
              integrated with fixed-step RK4.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[LENGTH, GRAVITY, DAMPING],
    initial_state: &[ANGLE, ANGULAR_VELOCITY],
    outputs: &[
        OutputDescriptor {
            id: "angle",
            display_name: "Angle",
            unit: "rad",
            description: "Angle from the downward vertical over time.",
        },
        OutputDescriptor {
            id: "angular_velocity",
            display_name: "Angular velocity",
            unit: "rad/s",
            description: "Angular velocity over time.",
        },
        OutputDescriptor {
            id: "energy",
            display_name: "Energy per unit mass",
            unit: "J",
            description: "Mechanical energy per unit mass over time.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ENERGY_DRIFT_CHECK, AMPLITUDE_CHECK],
    },
};

/// The `sim.mechanics.pendulum` adapter.
#[derive(Debug, Default)]
pub struct PendulumAdapter;

impl CapabilityAdapter for PendulumAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["length", "gravity", "damping"],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["angle", "angular_velocity"],
        ));
        for e in [
            resolve_model_scalar(scenario, &LENGTH).err(),
            resolve_model_scalar(scenario, &GRAVITY).err(),
            resolve_model_scalar(scenario, &DAMPING).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for e in [
            resolve_state_vector(scenario, &ANGLE).err(),
            resolve_state_vector(scenario, &ANGULAR_VELOCITY).err(),
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
        // Every adapter checks this, including the deterministic ones — see
        // `resolve_replicates`.
        if let Err(e) = resolve_replicates(scenario, DESCRIPTOR.determinism)
        {
            errors.push(e);
        }
        // The scenario's declared backend and precision must be ones this
        // capability actually has — see `resolve_precision`.
        if let Err(e) = resolve_backend_kind(scenario, &DESCRIPTOR)
        {
            errors.push(e);
        }
        if let Err(e) = resolve_precision(scenario, &DESCRIPTOR)
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
        let length = resolve_model_scalar(s, &LENGTH).expect("validated");
        let gravity = resolve_model_scalar(s, &GRAVITY).expect("validated");
        let damping = resolve_model_scalar(s, &DAMPING).expect("validated");
        let theta0 = resolve_state_vector(s, &ANGLE).expect("validated")[0];
        let omega0 = resolve_state_vector(s, &ANGULAR_VELOCITY).expect("validated")[0];

        let model = Pendulum::new(length, gravity, damping)
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

        let energy0 = model.energy(&[theta0, omega0]);
        let traj = simulate_cancellable(
            &model,
            &[theta0, omega0],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;
        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");
        let energy1 = model.energy(last);

        let angle_series = traj.column(0).expect("dim 0 exists");
        let velocity_series = traj.column(1).expect("dim 1 exists");
        let energy_series: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.energy(row).expect("state rows always have length 2"))
            .collect();

        let max_angle = angle_series
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()));

        let energy_check = match (damping == 0.0, energy0, energy1)
        {
            (true, Some(e0), Some(e1)) =>
            {
                let drift = (e1 - e0).abs() / e0.abs().max(1e-300);
                let threshold = 1e-6;
                VerificationResult {
                    id: "energy_drift".to_string(),
                    status: if drift <= threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(drift),
                    threshold: Some(threshold),
                    explanation: format!("undamped pendulum: relative energy drift {drift:.3e}"),
                }
            },
            _ => VerificationResult {
                id: "energy_drift".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: "damping > 0: energy is expected to decay, so there is no invariant \
                              to check"
                    .to_string(),
            },
        };

        // Released from rest, the release angle is the turning point: the
        // pendulum has no energy to go further. Swinging past it means the
        // integrator manufactured energy.
        let released_from_rest = omega0 == 0.0 && damping == 0.0;
        let amplitude_check = if released_from_rest
        {
            let excess = max_angle - theta0.abs();
            let threshold = 1e-6;
            VerificationResult {
                id: "amplitude_bounded".to_string(),
                status: if excess <= threshold
                {
                    VerificationStatus::Passed
                }
                else
                {
                    VerificationStatus::Failed
                },
                measured: Some(excess),
                threshold: Some(threshold),
                explanation: format!(
                    "largest swing beyond the release angle: {excess:.3e} rad \
                     (energy conservation forbids any)"
                ),
            }
        }
        else
        {
            VerificationResult {
                id: "amplitude_bounded".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: "this check applies only to an undamped pendulum released from \
                              rest, where the release angle is the turning point"
                    .to_string(),
            }
        };

        // The small-angle period is reported next to the amplitude so a
        // reader can see when it stops being a good approximation: the true
        // period grows with amplitude, by about 18% at 90 degrees.
        let mut metrics = vec![
            Metric {
                id: "max_angle".to_string(),
                display_name: "Largest angle reached".to_string(),
                value: MetricValue::Scalar(max_angle),
                unit: Some("rad".to_string()),
            },
            Metric {
                id: "small_angle_period".to_string(),
                display_name: "Small-angle period 2*pi*sqrt(L/g)".to_string(),
                value: MetricValue::Scalar(model.small_angle_period()),
                unit: Some("s".to_string()),
            },
            Metric {
                id: "final_angle".to_string(),
                display_name: "Final angle".to_string(),
                value: MetricValue::Scalar(last[0]),
                unit: Some("rad".to_string()),
            },
        ];
        if let Some(period) = measured_period(&traj.t, &angle_series)
        {
            metrics.push(Metric {
                id: "measured_period".to_string(),
                display_name: "Period measured from the trajectory".to_string(),
                value: MetricValue::Scalar(period),
                unit: Some("s".to_string()),
            });
        }

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                steps: traj.len() - 1,
                t_start: traj.t.first().copied().unwrap_or(t0),
                t_end: traj.last_time().unwrap_or(t1),
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
                    id: "angle".to_string(),
                    display_name: "Angle".to_string(),
                    unit: "rad".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: angle_series,
                },
                Series {
                    id: "angular_velocity".to_string(),
                    display_name: "Angular velocity".to_string(),
                    unit: "rad/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: velocity_series,
                },
                Series {
                    id: "energy".to_string(),
                    display_name: "Energy per unit mass".to_string(),
                    unit: "J".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: energy_series,
                },
            ],
            metrics,
            warnings: vec![],
            verifications: vec![energy_check, amplitude_check],
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
                // This capability's result does not depend on a seed, so
                // recording one would imply it did.
                seed: None,
            },
        };
        crate::result::validate_result(&result)
            .map_err(|d| ExecutionError::Internal(crate::result::describe_defects(&d)))?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

/// The period, measured from the trajectory's own upward zero crossings.
///
/// Returned as `None` rather than guessed when there are fewer than two
/// crossings — a run too short to show a full swing has no period to report,
/// and inventing one from the small-angle formula would defeat the purpose of
/// measuring it.
///
/// The crossing time is linearly interpolated between the bracketing samples,
/// so the answer is not quantised to the step size.
fn measured_period(t: &[f64], angle: &[f64]) -> Option<f64> {
    let mut crossings = Vec::new();
    for i in 1..angle.len()
    {
        let (a, b) = (angle[i - 1], angle[i]);
        if a < 0.0 && b >= 0.0
        {
            let fraction = -a / (b - a);
            crossings.push(t[i - 1] + fraction * (t[i] - t[i - 1]));
        }
    }
    if crossings.len() < 2
    {
        return None;
    }
    let span = crossings.last()? - crossings.first()?;
    Some(span / (crossings.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/pendulum.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = PendulumAdapter;
        let validated = adapter.validate(&scenario).expect("tutorial is valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    fn metric(result: &RunResult, id: &str) -> Option<f64> {
        match result.metrics.iter().find(|m| m.id == id).map(|m| &m.value)
        {
            Some(MetricValue::Scalar(v)) => Some(*v),
            _ => None,
        }
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        assert_eq!(scenario.capability.id, DESCRIPTOR.id.0);
    }

    #[test]
    fn the_undamped_pendulum_conserves_energy() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_drift")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed, "{check:?}");
        assert!(check.measured.unwrap() < 1e-9, "{:?}", check.measured);
    }

    #[test]
    fn the_pendulum_never_swings_past_its_release_angle() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "amplitude_bounded")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed, "{check:?}");
    }

    /// The reason this capability is not just "spring-mass-damper with
    /// different names": at a large amplitude the true period is
    /// measurably longer than the small-angle formula predicts. The tutorial
    /// releases from 90 degrees, where the excess is about 18%.
    #[test]
    fn a_large_swing_has_a_longer_period_than_the_small_angle_formula() {
        let result = run();
        let small = metric(&result, "small_angle_period").expect("small-angle period");
        let measured = metric(&result, "measured_period").expect("a measured period");

        let ratio = measured / small;
        assert!(
            ratio > 1.15 && ratio < 1.22,
            "at a 90-degree release the true period should exceed the small-angle period by \
             roughly 18%; measured {measured:.6} vs {small:.6} (ratio {ratio:.4})"
        );
    }

    /// The period is measured from the trajectory, so a run too short to
    /// complete a swing must report nothing rather than a guess.
    #[test]
    fn a_run_too_short_to_show_a_swing_reports_no_period() {
        assert_eq!(measured_period(&[0.0, 1.0], &[-1.0, 1.0]), None);
        assert_eq!(measured_period(&[], &[]), None);
    }

    /// The crossing times are interpolated, so the answer is not quantised to
    /// the step size: a period of exactly 2 must come back as exactly 2 even
    /// when the samples straddle the crossing.
    #[test]
    fn the_period_is_interpolated_between_samples() {
        // A triangle wave crossing zero upward at t = 0.5, 2.5, 4.5.
        let t: Vec<f64> = (0..=50).map(|i| i as f64 * 0.1).collect();
        let angle: Vec<f64> = t
            .iter()
            .map(|t| (std::f64::consts::PI * (t - 0.5)).sin())
            .collect();
        let period = measured_period(&t, &angle).expect("a period");
        assert!((period - 2.0).abs() < 1e-6, "{period}");
    }

    #[test]
    fn a_damped_pendulum_reports_the_energy_check_as_not_applicable() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "damping".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.5,
                unit: "1/s".to_string(),
            },
        );
        let adapter = PendulumAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();

        for id in ["energy_drift", "amplitude_bounded"]
        {
            let check = result.verifications.iter().find(|v| v.id == id).unwrap();
            assert_eq!(
                check.status,
                VerificationStatus::NotApplicable,
                "{id} must not claim to have checked a damped run"
            );
        }
        // …and the run must still be a valid, usable result.
        let energy = &result.series[2].values;
        assert!(
            energy.last().unwrap() < &energy[0],
            "a damped pendulum must lose energy"
        );
    }

    #[test]
    fn rejects_a_non_positive_length() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "length".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "m".to_string(),
            },
        );
        let report = PendulumAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == LENGTH.error_code));
    }

    #[test]
    fn rejects_an_angle_given_in_the_wrong_dimension() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.insert(
            "angle".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 1.0,
                unit: "m".to_string(),
            }],
        );
        let report = PendulumAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == ANGLE.error_code));
    }

    #[test]
    fn rejects_unsupported_solver() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.id = "euler".to_string();
        let report = PendulumAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("euler"))
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = PendulumAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let err = adapter
            .execute(&validated, &control, &mut NullEventSink)
            .unwrap_err();
        assert_eq!(err, ExecutionError::Cancelled);
    }

    #[test]
    fn emits_started_and_completed_events() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = PendulumAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        assert_eq!(sink.events().first(), Some(&RunEvent::Started));
        assert_eq!(sink.events().last(), Some(&RunEvent::Completed));
    }
}
