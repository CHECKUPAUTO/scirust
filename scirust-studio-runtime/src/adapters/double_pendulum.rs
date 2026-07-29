//! `sim.mechanics.double_pendulum` — a planar double pendulum
//! (`scirust_sim::mechanics::DoublePendulum`).
//!
//! This capability exists to be honest about something the rest of the
//! catalogue lets you take for granted.
//!
//! Every other model here is *predictable*: run it twice from almost the same
//! start and you get almost the same answer, so a trajectory is a statement
//! about the system. The double pendulum is **deterministic but not
//! predictable**. Its motion is fixed entirely by the initial conditions —
//! this crate's determinism guarantee holds exactly, bit for bit — and yet
//! two starts differing in the twelfth decimal place diverge to completely
//! different motions within a few seconds. That is deterministic chaos, and
//! it is a property of the physics, not a defect of the solver.
//!
//! So the run does two things no other adapter does:
//!
//! * It **measures the divergence** by integrating a second trajectory from a
//!   deliberately perturbed start, and reports how far apart they end up.
//!   That number is evidence, not a caveat in a document.
//! * It **raises a warning** whenever the trajectory has entered the regime
//!   where late values cannot be read as predictions.
//!
//! Energy is still exactly conserved by the true dynamics, so the drift check
//! remains the measure of integration quality — and it is the *only* thing
//! about a long run that stays meaningful.

use scirust_sim::mechanics::DoublePendulum;

use crate::execute_support::{TimeSpan, simulate_cancellable};
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind, RunDomain,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::result::{
    Axis, AxisMonotonicity, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, RunWarning, Series, SeriesRole, TIME_AXIS_ID, VerificationResult,
    VerificationStatus, WarningCategory,
};
use crate::sink::{EventSink, RunEvent, SubRangeSink};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// How far the shadow trajectory's first angle is displaced, in radians.
///
/// Small enough to be physically indistinguishable — far below any angle
/// anyone could set a real pendulum to — and large enough to stay well clear
/// of `f64` rounding, so the divergence it reveals is the physics and not the
/// arithmetic.
const PERTURBATION: f64 = 1e-9;

/// Angular separation beyond which the two trajectories have stopped
/// describing the same motion. Half a radian is roughly 30 degrees: at that
/// point the bobs are visibly elsewhere.
const DECORRELATED_RADIANS: f64 = 0.5;

const M1: FieldDescriptor = FieldDescriptor {
    canonical_name: "m1",
    display_name: "Upper bob mass",
    required: true,
    dimension: scirust_units::Dimension::MASS,
    accepted_units: &["kg"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Mass of the bob on the upper rod.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 180),
};

const M2: FieldDescriptor = FieldDescriptor {
    canonical_name: "m2",
    display_name: "Lower bob mass",
    required: true,
    dimension: scirust_units::Dimension::MASS,
    accepted_units: &["kg"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Mass of the bob on the lower rod.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 181),
};

const L1: FieldDescriptor = FieldDescriptor {
    canonical_name: "l1",
    display_name: "Upper rod length",
    required: true,
    dimension: scirust_units::Dimension::LENGTH,
    accepted_units: &["m"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Length of the rod from the pivot to the upper bob.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 182),
};

const L2: FieldDescriptor = FieldDescriptor {
    canonical_name: "l2",
    display_name: "Lower rod length",
    required: true,
    dimension: scirust_units::Dimension::LENGTH,
    accepted_units: &["m"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Length of the rod from the upper bob to the lower bob.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 183),
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
    error_code: ErrorCode::new(ErrorFamily::Validation, 184),
};

const THETA1: FieldDescriptor = FieldDescriptor {
    canonical_name: "theta1",
    display_name: "Initial upper angle",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["rad"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Upper rod's angle from the downward vertical, in radians.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 185),
};

const OMEGA1: FieldDescriptor = FieldDescriptor {
    canonical_name: "omega1",
    display_name: "Initial upper angular velocity",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["rad/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Upper rod's angular velocity, in radians per second.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 186),
};

const THETA2: FieldDescriptor = FieldDescriptor {
    canonical_name: "theta2",
    display_name: "Initial lower angle",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["rad"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Lower rod's angle from the downward vertical, in radians.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 187),
};

const OMEGA2: FieldDescriptor = FieldDescriptor {
    canonical_name: "omega2",
    display_name: "Initial lower angular velocity",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["rad/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Lower rod's angular velocity, in radians per second.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 188),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
    reports_progress: true,
};

const ENERGY_DRIFT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_drift",
    description: "Total mechanical energy is exactly conserved by the undamped double-pendulum \
                  dynamics; the check measures RK4's relative drift. For a chaotic system this \
                  is the one property of a long run that stays meaningful — the individual \
                  angles do not.",
};

const SENSITIVITY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "sensitive_dependence",
    description: "A second trajectory is integrated from a start perturbed by 1e-9 rad. The \
                  check reports how far the two have separated by the end of the run: a large \
                  separation is the expected, correct behaviour of a chaotic system, and is \
                  reported so nobody reads a late-time angle as a prediction.",
};

/// The capability descriptor for `sim.mechanics.double_pendulum`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.mechanics.double_pendulum"),
    display_name: "Double pendulum (chaotic)",
    category: CapabilityCategory::Mechanics,
    source_crate: "scirust-sim",
    summary: "A planar double pendulum: deterministic, energy-conserving, and chaotic. The run \
              measures its own sensitivity to initial conditions.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[M1, M2, L1, L2, GRAVITY],
    initial_state: &[THETA1, OMEGA1, THETA2, OMEGA2],
    outputs: &[
        OutputDescriptor {
            id: "theta1",
            display_name: "Upper angle",
            unit: "rad",
            description: "Upper rod's angle over time.",
        },
        OutputDescriptor {
            id: "theta2",
            display_name: "Lower angle",
            unit: "rad",
            description: "Lower rod's angle over time.",
        },
        OutputDescriptor {
            id: "energy",
            display_name: "Energy",
            unit: "J",
            description: "Total mechanical energy over time.",
        },
        OutputDescriptor {
            id: "separation",
            display_name: "Separation from the perturbed twin",
            unit: "rad",
            description: "Distance in angle space between this trajectory and one started \
                          1e-9 rad away. Watching it grow is watching chaos happen.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ENERGY_DRIFT_CHECK, SENSITIVITY_CHECK],
    },
};

/// The `sim.mechanics.double_pendulum` adapter.
#[derive(Debug, Default)]
pub struct DoublePendulumAdapter;

impl CapabilityAdapter for DoublePendulumAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["m1", "m2", "l1", "l2", "gravity"],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["theta1", "omega1", "theta2", "omega2"],
        ));
        for e in [
            resolve_model_scalar(scenario, &M1).err(),
            resolve_model_scalar(scenario, &M2).err(),
            resolve_model_scalar(scenario, &L1).err(),
            resolve_model_scalar(scenario, &L2).err(),
            resolve_model_scalar(scenario, &GRAVITY).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for e in [
            resolve_state_vector(scenario, &THETA1).err(),
            resolve_state_vector(scenario, &OMEGA1).err(),
            resolve_state_vector(scenario, &THETA2).err(),
            resolve_state_vector(scenario, &OMEGA2).err(),
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
        let m1 = resolve_model_scalar(s, &M1).expect("validated");
        let m2 = resolve_model_scalar(s, &M2).expect("validated");
        let l1 = resolve_model_scalar(s, &L1).expect("validated");
        let l2 = resolve_model_scalar(s, &L2).expect("validated");
        let gravity = resolve_model_scalar(s, &GRAVITY).expect("validated");
        let y0 = [
            resolve_state_vector(s, &THETA1).expect("validated")[0],
            resolve_state_vector(s, &OMEGA1).expect("validated")[0],
            resolve_state_vector(s, &THETA2).expect("validated")[0],
            resolve_state_vector(s, &OMEGA2).expect("validated")[0],
        ];

        let model = DoublePendulum::new(m1, m2, l1, l2, gravity)
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
        let span = TimeSpan {
            t0,
            t_end: t1,
            h: step,
        };

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let energy0 = model.energy(&y0);
        // Half the progress window each: the shadow is the same length as
        // the trajectory, so a bar driven by the first alone would reach one
        // hundred percent halfway through the work. See `SubRangeSink`.
        let traj = simulate_cancellable(
            &model,
            &y0,
            span,
            control,
            &mut SubRangeSink::new(sink, 0.0, 0.5),
        )?;

        // The shadow run. Deliberately *not* reported through the sink: the
        // user asked for one run and should see one run's progress, and this
        // is measurement of that run rather than a second experiment.
        let mut perturbed_start = y0;
        perturbed_start[0] += PERTURBATION;
        let shadow = simulate_cancellable(
            &model,
            &perturbed_start,
            span,
            control,
            &mut SubRangeSink::new(sink, 0.5, 1.0),
        )?;

        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");
        let energy1 = model.energy(last);

        let theta1_series = traj.column(0).expect("dim 0 exists");
        let theta2_series = traj.column(2).expect("dim 2 exists");
        let energy_series: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.energy(row).expect("state rows always have length 4"))
            .collect();

        // The two runs use the same fixed step over the same span, so they
        // have the same coordinates; zip defensively anyway rather than index
        // one by the other's length.
        let separation_series: Vec<f64> = traj
            .y
            .iter()
            .zip(shadow.y.iter())
            .map(|(a, b)| angular_separation(a, b))
            .collect();
        let final_separation = separation_series.last().copied().unwrap_or(0.0);
        let decorrelation_time = separation_series
            .iter()
            .position(|d| *d >= DECORRELATED_RADIANS)
            .and_then(|i| traj.t.get(i).copied());

        let energy_check = match (energy0, energy1)
        {
            (Some(e0), Some(e1)) =>
            {
                // Normalised by the system's gravitational energy scale, not
                // by the initial energy. That distinction is not pedantry:
                // this model measures potential energy from the pivot, so a
                // pendulum released from horizontal starts at *exactly zero*
                // energy, and dividing by it turns a drift of 2e-12 J into a
                // reported "relative drift" of 1.8e3. The scale below is the
                // potential energy of the fully raised configuration — a real
                // quantity of the system that never vanishes.
                let drift = (e1 - e0).abs() / energy_scale(m1, m2, l1, l2, gravity);
                // Looser than the single pendulum's 1e-6: this is a stiffer,
                // faster-moving nonlinear system and the tutorial runs it for
                // long enough to be interesting.
                let threshold = 1e-5;
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
                    explanation: format!(
                        "energy drift {drift:.3e}, relative to the system's gravitational \
                         energy scale (m1+m2)*g*l1 + m2*g*l2. For a chaotic system this is the \
                         property of a long run that stays meaningful; the individual angles \
                         do not."
                    ),
                }
            },
            _ => VerificationResult {
                id: "energy_drift".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: "the energy could not be evaluated for this state".to_string(),
            },
        };

        // Deliberately never `Failed`. Divergence is the correct behaviour of
        // the real physics, so reporting it as a failure would be teaching
        // the user something untrue; it is measured and stated instead.
        let sensitivity_check = VerificationResult {
            id: "sensitive_dependence".to_string(),
            status: VerificationStatus::Passed,
            measured: Some(final_separation),
            threshold: None,
            explanation: match decorrelation_time
            {
                Some(t) => format!(
                    "a start perturbed by {PERTURBATION:.0e} rad separates from this one by \
                     {DECORRELATED_RADIANS} rad at t = {t:.3} s and ends {final_separation:.3} \
                     rad away. Beyond that time the angles are not predictions of anything."
                ),
                None => format!(
                    "a start perturbed by {PERTURBATION:.0e} rad ends {final_separation:.3e} rad \
                     away; over this span the trajectories have not yet separated \
                     appreciably."
                ),
            },
        };

        let mut warnings = Vec::new();
        if let Some(t) = decorrelation_time
        {
            warnings.push(RunWarning {
                category: WarningCategory::Numerical,
                message: format!(
                    "This system is chaotic. After t = {t:.3} s the trajectory is no longer a \
                     prediction: an initial angle different by {PERTURBATION:.0e} rad — far \
                     below anything measurable — produces a completely different motion. The \
                     result is exactly reproducible from these inputs, but it is not the only \
                     motion consistent with the experiment it describes."
                ),
            });
        }

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: TIME_AXIS_ID.to_string(),
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
                    id: "theta1".to_string(),
                    display_name: "Upper angle".to_string(),
                    unit: "rad".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: theta1_series,
                },
                Series {
                    id: "theta2".to_string(),
                    display_name: "Lower angle".to_string(),
                    unit: "rad".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: theta2_series,
                },
                Series {
                    id: "energy".to_string(),
                    display_name: "Energy".to_string(),
                    unit: "J".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: energy_series,
                },
                Series {
                    id: "separation".to_string(),
                    display_name: "Separation from the perturbed twin".to_string(),
                    unit: "rad".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: separation_series,
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "final_separation".to_string(),
                    display_name: "Final separation from the perturbed twin".to_string(),
                    value: MetricValue::Scalar(final_separation),
                    unit: Some("rad".to_string()),
                },
                Metric {
                    id: "decorrelation_time".to_string(),
                    display_name: "Time to separate by half a radian".to_string(),
                    value: match decorrelation_time
                    {
                        Some(t) => MetricValue::Scalar(t),
                        None => MetricValue::Text("not reached in this run".to_string()),
                    },
                    unit: decorrelation_time.map(|_| "s".to_string()),
                },
                Metric {
                    id: "perturbation".to_string(),
                    display_name: "Perturbation applied to the twin".to_string(),
                    value: MetricValue::Scalar(PERTURBATION),
                    unit: Some("rad".to_string()),
                },
            ],
            warnings,
            verifications: vec![energy_check, sensitivity_check],
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

/// The system's gravitational energy scale: the potential energy of the
/// fully raised configuration, `(m1 + m2)·g·l1 + m2·g·l2`.
///
/// Used to normalise the energy drift. Strictly positive for any valid model
/// — every factor is validated as positive — so unlike the initial energy it
/// can never be zero, and a drift reported against it means the same thing
/// whatever configuration the run started from.
fn energy_scale(m1: f64, m2: f64, l1: f64, l2: f64, gravity: f64) -> f64 {
    (m1 + m2) * gravity * l1 + m2 * gravity * l2
}

/// Distance between two double-pendulum states in angle space.
///
/// Only the angles: the angular velocities are unbounded and would dominate
/// the number without saying anything more about whether the two pendulums
/// are in the same place.
fn angular_separation(a: &[f64], b: &[f64]) -> f64 {
    let d1 = a[0] - b[0];
    let d2 = a[2] - b[2];
    (d1 * d1 + d2 * d2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/double_pendulum.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = DoublePendulumAdapter;
        let validated = adapter.validate(&scenario).expect("tutorial is valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        assert_eq!(scenario.capability.id, DESCRIPTOR.id.0);
    }

    #[test]
    fn energy_is_conserved_despite_the_chaos() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_drift")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed, "{check:?}");
    }

    /// The claim the capability is built around: a perturbation nine orders
    /// of magnitude below anything measurable grows into a completely
    /// different motion.
    #[test]
    fn a_perturbation_of_one_nanoradian_grows_into_a_different_motion() {
        let result = run();
        let separation = &result.series[3].values;

        assert!(
            separation[0] <= PERTURBATION * 1.001,
            "the two trajectories must start together, not {}",
            separation[0]
        );
        let final_separation = *separation.last().unwrap();
        assert!(
            final_separation > DECORRELATED_RADIANS,
            "the tutorial must run long enough to show the divergence; \
             it ended only {final_separation:.3e} rad apart"
        );
        // Growth of at least eight orders of magnitude from 1e-9.
        assert!(
            final_separation / separation[0] > 1e8,
            "separation grew only {:.3e}x",
            final_separation / separation[0]
        );
    }

    /// Divergence is the physics behaving correctly, so it must never be
    /// reported as a failed check — that would teach the user something
    /// untrue about their own experiment.
    #[test]
    fn divergence_is_reported_but_never_as_a_failure() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "sensitive_dependence")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed);
        assert!(check.measured.unwrap() > DECORRELATED_RADIANS);
        assert!(
            check.explanation.contains("not predictions"),
            "{}",
            check.explanation
        );
    }

    /// And the user is told, in the result itself, that late values are not
    /// predictions.
    #[test]
    fn a_chaotic_run_carries_a_warning_naming_the_time_it_stopped_predicting() {
        let result = run();
        assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
        let warning = &result.warnings[0];
        assert_eq!(warning.category, WarningCategory::Numerical);
        assert!(warning.message.contains("chaotic"), "{}", warning.message);
        assert!(
            warning.message.contains("reproducible"),
            "the warning must not leave the reader thinking the run is random: {}",
            warning.message
        );
    }

    /// Chaotic is not the same as random. The same inputs must produce the
    /// same output, bit for bit — that is what this crate's determinism class
    /// claims, and it is exactly the claim a chaotic system makes tempting to
    /// doubt.
    #[test]
    fn the_same_scenario_run_twice_is_bit_identical() {
        let first = run();
        let second = run();
        for (a, b) in first.series.iter().zip(second.series.iter())
        {
            assert_eq!(a.id, b.id);
            for (x, y) in a.values.iter().zip(b.values.iter())
            {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{}: a chaotic system is still deterministic",
                    a.id
                );
            }
        }
    }

    /// A short run has not decorrelated yet, and must say so rather than
    /// warning about something that has not happened.
    #[test]
    fn a_run_too_short_to_decorrelate_carries_no_warning() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.end = scirust_studio_schema::ValueWithUnit {
            value: 1.0,
            unit: "s".to_string(),
        };
        let adapter = DoublePendulumAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();

        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        let decorrelation = result
            .metrics
            .iter()
            .find(|m| m.id == "decorrelation_time")
            .expect("the metric is always present");
        assert!(
            matches!(&decorrelation.value, MetricValue::Text(t) if t.contains("not reached")),
            "{:?}",
            decorrelation.value
        );
        assert_eq!(decorrelation.unit, None);
    }

    #[test]
    fn the_separation_series_is_aligned_with_the_time_axis() {
        let result = run();
        let axis = &result.axes[0];
        for series in &result.series
        {
            assert_eq!(series.values.len(), axis.values.len(), "{}", series.id);
        }
    }

    #[test]
    fn rejects_a_non_positive_rod_length() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "l2".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "m".to_string(),
            },
        );
        let report = DoublePendulumAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == L2.error_code));
    }

    #[test]
    fn rejects_a_missing_state_component() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.remove("omega2");
        let report = DoublePendulumAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == OMEGA2.error_code));
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = DoublePendulumAdapter;
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
        let adapter = DoublePendulumAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        assert_eq!(sink.events().first(), Some(&RunEvent::Started));
        assert_eq!(sink.events().last(), Some(&RunEvent::Completed));
    }

    /// The shadow run must not appear in the event stream: the user asked for
    /// one run and a doubled progress sequence would look like a bug.
    #[test]
    fn the_shadow_run_does_not_reach_the_event_sink() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = DoublePendulumAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();

        assert_eq!(
            sink.events()
                .iter()
                .filter(|e| matches!(e, RunEvent::Started))
                .count(),
            1
        );
        assert_eq!(
            sink.events()
                .iter()
                .filter(|e| matches!(e, RunEvent::Completed))
                .count(),
            1
        );
    }

    /// The reason the drift is not measured against the initial energy: this
    /// model measures potential energy from the pivot, so the tutorial's own
    /// starting configuration has energy zero to within rounding. A relative
    /// drift computed against that is meaningless, and was — it reported
    /// 1.8e3 for a run whose absolute drift was two picojoules.
    #[test]
    fn the_energy_scale_is_positive_where_the_initial_energy_is_not() {
        let model = DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 9.81).unwrap();
        let horizontal = std::f64::consts::FRAC_PI_2;
        let e0 = model
            .energy(&[horizontal, 0.0, horizontal, 0.0])
            .expect("an energy");
        assert!(
            e0.abs() < 1e-14,
            "the tutorial's start really is at zero energy: {e0}"
        );

        let scale = energy_scale(1.0, 1.0, 1.0, 1.0, 9.81);
        assert!(scale > 0.0, "{scale}");
        assert_eq!(scale, 2.0 * 9.81 + 9.81);
    }

    #[test]
    fn separation_uses_the_angles_only() {
        // Identical angles, wildly different velocities: same place.
        assert_eq!(
            angular_separation(&[0.0, 100.0, 0.0, -100.0], &[0.0, 0.0, 0.0, 0.0]),
            0.0
        );
        // A 3-4-5 triangle in angle space.
        assert_eq!(
            angular_separation(&[3.0, 0.0, 4.0, 0.0], &[0.0, 0.0, 0.0, 0.0]),
            5.0
        );
    }
}
