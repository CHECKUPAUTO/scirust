//! `sim.mechanics.projectile` — ballistic flight with linear drag
//! (`scirust_sim::mechanics::Projectile`).
//!
//! Four states, `[x, y, vx, vy]`, under gravity and a drag force proportional
//! to velocity. The linear (Stokes) drag law rather than the quadratic one,
//! which is what makes the problem exactly solvable — and what makes both of
//! this capability's checks exact statements rather than approximations.
//!
//! # Two facts read straight off the equations
//!
//! The horizontal equation `vx' = -drag * vx` is **decoupled**: it involves
//! nothing but `vx`. So `vx(t) = vx0 * exp(-drag * t)` exactly, regardless of
//! gravity, of the vertical motion, or of where the projectile is. That is a
//! reference the integrator cannot influence, and the threshold against it is
//! derivable in closed form — see [`rk4_decay_error`].
//!
//! The vertical equation has a fixed point: `vy' = 0` when `vy = -gravity /
//! drag`. That is the terminal velocity, and it is read off the model's own
//! stated derivative rather than looked up. The approach to it is the same
//! exponential, so its threshold is the same residual every first-order
//! capability here uses.
//!
//! The two fail independently: a wrong drag moves both, a wrong gravity moves
//! only the terminal velocity, and an integrator that lost the decoupling
//! would move only the horizontal check.
use scirust_sim::mechanics::Projectile;
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind, RunDomain,
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

/// Slack over RK4's own truncation error on the horizontal decay.
const DECAY_MARGIN: f64 = 8.0;

/// Slack over the residual the run's own length leaves on the approach to
/// terminal velocity.
const TERMINAL_MARGIN: f64 = 5.0;

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
    description: "Downward acceleration. With the drag it sets the terminal velocity, and it does not touch the horizontal motion at all.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 307),
};

const DRAG: FieldDescriptor = FieldDescriptor {
    canonical_name: "drag",
    display_name: "Linear drag coefficient",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Drag per unit mass, so a rate. Stokes drag rather than quadratic: that is what makes the problem exactly solvable.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 308),
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
    description: "Launch point [x, y].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 309),
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
    description: "Launch velocity [vx, vy].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 310),
};

const HORIZONTAL_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "horizontal_velocity_decays_exactly",
    description: "The horizontal equation vx' = -drag*vx is decoupled from everything else, so vx(t) = vx0*exp(-drag*t) exactly, whatever gravity and the vertical motion do. The threshold is RK4's own truncation error for exponential decay — |R(z) - exp(z)| per step with z = -drag*h — accumulated over the run, not a chosen number.",
};

const TERMINAL_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "approaches_terminal_velocity",
    description: "The vertical velocity approaches the fixed point of the model's own stated derivative, vy = -gravity/drag. The threshold is the residual exp(-drag*span) that the run's own length leaves. A wrong gravity moves this and leaves the horizontal decay untouched.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.mechanics.projectile"),
    display_name: "Projectile with linear drag",
    category: CapabilityCategory::Mechanics,
    source_crate: "scirust-sim",
    summary: "Ballistic flight under gravity and Stokes drag: a decoupled horizontal decay and a vertical terminal velocity, both exact.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[GRAVITY, DRAG],
    initial_state: &[POSITION, VELOCITY],
    outputs: &[
        OutputDescriptor {
            id: "position_x",
            display_name: "Horizontal position",
            unit: "m",
            description: "Downrange distance, approaching vx0/drag asymptotically.",
        },
        OutputDescriptor {
            id: "position_y",
            display_name: "Height",
            unit: "m",
            description: "Vertical position.",
        },
        OutputDescriptor {
            id: "velocity_x",
            display_name: "Horizontal velocity",
            unit: "m/s",
            description: "Decaying exactly as exp(-drag*t).",
        },
        OutputDescriptor {
            id: "velocity_y",
            display_name: "Vertical velocity",
            unit: "m/s",
            description: "Approaching the terminal -gravity/drag.",
        },
        OutputDescriptor {
            id: "exact_velocity_x",
            display_name: "Exact horizontal velocity",
            unit: "m/s",
            description: "vx0*exp(-drag*t), for comparison.",
        },
        OutputDescriptor {
            id: "terminal_velocity",
            display_name: "Terminal velocity",
            unit: "m/s",
            description: "The level the vertical velocity approaches.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[HORIZONTAL_CHECK, TERMINAL_CHECK],
    },
};

/// The `sim.mechanics.projectile` adapter.
#[derive(Debug, Default)]
pub struct ProjectileAdapter;

impl CapabilityAdapter for ProjectileAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["gravity", "drag"]));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["position", "velocity"],
        ));
        for e in [
            resolve_model_scalar(scenario, &GRAVITY).err(),
            resolve_model_scalar(scenario, &DRAG).err(),
            resolve_state_vector(scenario, &POSITION).err(),
            resolve_state_vector(scenario, &VELOCITY).err(),
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
        let gravity = resolve_model_scalar(s, &GRAVITY).expect("validated");
        let drag = resolve_model_scalar(s, &DRAG).expect("validated");
        let model = Projectile::new(gravity, drag)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
        let p0 = resolve_state_vector(s, &POSITION).expect("validated");
        let v0 = resolve_state_vector(s, &VELOCITY).expect("validated");

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
            &[p0[0], p0[1], v0[0], v0[1]],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let position_x = traj.column(0).expect("dim 0");
        let position_y = traj.column(1).expect("dim 1");
        let velocity_x = traj.column(2).expect("dim 2");
        let velocity_y = traj.column(3).expect("dim 3");
        let exact_vx: Vec<f64> = traj
            .t
            .iter()
            .map(|t| v0[0] * (-drag * (t - t0)).exp())
            .collect();

        let terminal = -gravity / drag;
        let span = *traj.t.last().expect("at least one sample") - traj.t[0];
        let horizontal = horizontal_check(&velocity_x, &exact_vx, v0[0], drag, step, span);
        let terminal_check = terminal_check(
            velocity_y.last().copied().unwrap_or(f64::NAN),
            terminal,
            (v0[1] - terminal).abs(),
            drag,
            span,
        );

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: TIME_AXIS_ID.to_string(),
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
                    id: "position_x".to_string(),
                    display_name: "Horizontal position".to_string(),
                    unit: "m".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: position_x.clone(),
                },
                Series {
                    id: "position_y".to_string(),
                    display_name: "Height".to_string(),
                    unit: "m".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: position_y.clone(),
                },
                Series {
                    id: "velocity_x".to_string(),
                    display_name: "Horizontal velocity".to_string(),
                    unit: "m/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: velocity_x.clone(),
                },
                Series {
                    id: "velocity_y".to_string(),
                    display_name: "Vertical velocity".to_string(),
                    unit: "m/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: velocity_y.clone(),
                },
                Series {
                    id: "exact_velocity_x".to_string(),
                    display_name: "Exact horizontal velocity".to_string(),
                    unit: "m/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact_vx.clone(),
                },
                Series {
                    id: "terminal_velocity".to_string(),
                    display_name: "Terminal velocity".to_string(),
                    unit: "m/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![terminal; traj.t.len()],
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "terminal_velocity".to_string(),
                    display_name: "Terminal velocity -g/drag".to_string(),
                    value: MetricValue::Scalar(terminal),
                    unit: Some("m/s".to_string()),
                },
                Metric {
                    id: "drag_time_constant".to_string(),
                    display_name: "Drag time constant 1/drag".to_string(),
                    value: MetricValue::Scalar(1.0 / drag),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "horizontal_range_limit".to_string(),
                    display_name: "Asymptotic downrange distance".to_string(),
                    value: MetricValue::Scalar(p0[0] + v0[0] / drag),
                    unit: Some("m".to_string()),
                },
                Metric {
                    id: "final_height".to_string(),
                    display_name: "Height at the end".to_string(),
                    value: MetricValue::Scalar(position_y.last().copied().unwrap_or(f64::NAN)),
                    unit: Some("m".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![horizontal, terminal_check],
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

/// RK4's relative error per step on `y' = -k*y`, accumulated over `n` steps.
///
/// The method's amplification factor for a linear problem is the degree-four
/// Taylor polynomial `R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24` with `z = -k*h`,
/// while the exact factor is `exp(z)`. Their difference is the whole of the
/// error, it is a closed form, and multiplying it by the step count bounds
/// what the run can accumulate. Nothing here is fitted.
fn rk4_decay_error(k: f64, h: f64, steps: f64) -> f64 {
    let z = -k * h;
    let r = 1.0 + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0;
    (r - z.exp()).abs() * steps
}

/// The horizontal velocity against its exact decoupled decay.
fn horizontal_check(
    velocity_x: &[f64],
    exact: &[f64],
    vx0: f64,
    drag: f64,
    step: f64,
    span: f64,
) -> VerificationResult {
    let scale = vx0.abs().max(f64::MIN_POSITIVE);
    let worst = velocity_x
        .iter()
        .zip(exact.iter())
        .map(|(g, x)| (g - x).abs() / scale)
        .fold(0.0_f64, f64::max);
    let steps = if step > 0.0 { span / step } else { 0.0 };
    let threshold = (rk4_decay_error(drag, step, steps) * DECAY_MARGIN).max(f64::EPSILON * steps);
    VerificationResult {
        id: HORIZONTAL_CHECK.id.to_string(),
        status: if worst <= threshold
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(worst),
        threshold: Some(threshold),
        explanation: format!(
            "the horizontal velocity tracked vx0*exp(-drag*t) to within {worst:.3e} of its \
             launch value across all {} samples, against the {threshold:.3e} that RK4's own \
             amplification factor differs from exp(-drag*h) over {steps:.0} steps",
            velocity_x.len()
        ),
    }
}

/// The vertical velocity against the fixed point of its own equation.
fn terminal_check(
    final_vy: f64,
    terminal: f64,
    initial_gap: f64,
    drag: f64,
    span: f64,
) -> VerificationResult {
    let residual = if drag > 0.0 && span.is_finite()
    {
        (-drag * span).exp()
    }
    else
    {
        1.0
    };
    let scale = initial_gap.max(f64::MIN_POSITIVE);
    let error = (final_vy - terminal).abs() / scale;
    let threshold = residual * TERMINAL_MARGIN;
    VerificationResult {
        id: TERMINAL_CHECK.id.to_string(),
        status: if error <= threshold
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(error),
        threshold: Some(threshold),
        explanation: format!(
            "the vertical velocity ended {error:.3e} of its launch gap away from the \
             {terminal:.6} m/s terminal velocity that vy' = 0 implies, against the \
             {residual:.3e} the run's own length of {:.2} drag time constants still leaves",
            span * drag
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/projectile.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = ProjectileAdapter;
        let validated = adapter.validate(scenario).expect("valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    fn run() -> RunResult {
        run_scenario(&parse_toml(TUTORIAL).unwrap())
    }

    #[allow(dead_code)]
    fn check<'a>(result: &'a RunResult, id: &str) -> &'a VerificationResult {
        result
            .verifications
            .iter()
            .find(|v| v.id == id)
            .unwrap_or_else(|| panic!("no check {id}"))
    }

    #[allow(dead_code)]
    fn metric(result: &RunResult, id: &str) -> f64 {
        match result
            .metrics
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("no metric {id}"))
            .value
        {
            MetricValue::Scalar(v) => v,
            ref other => panic!("{id} is not a scalar: {other:?}"),
        }
    }

    #[allow(dead_code)]
    fn series<'a>(result: &'a RunResult, id: &str) -> &'a [f64] {
        &result
            .series
            .iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("no series {id}"))
            .values
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
        let validated = ProjectileAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            ProjectileAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn the_horizontal_velocity_decays_exactly() {
        let r = run();
        let v = check(&r, HORIZONTAL_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn it_approaches_terminal_velocity() {
        let r = run();
        let v = check(&r, TERMINAL_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The decoupling is the reason the horizontal check is exact, so it is
    /// worth asserting rather than only describing: changing gravity must
    /// leave the horizontal trajectory bit-for-bit identical.
    #[test]
    fn gravity_does_not_touch_the_horizontal_motion() {
        let base = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.get_mut("gravity").expect("gravity").value *= 2.0;
        let heavier = run_scenario(&scenario);
        assert!(
            series(&base, "velocity_x")
                .iter()
                .zip(series(&heavier, "velocity_x"))
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "doubling gravity moved the horizontal velocity, so the equations are coupled"
        );
    }

    /// The projectile must actually reach terminal velocity, or the second
    /// check is measuring a transient rather than an asymptote.
    #[test]
    fn the_run_is_long_enough_to_reach_terminal_velocity() {
        let r = run();
        let terminal = metric(&r, "terminal_velocity");
        let final_vy = *series(&r, "velocity_y").last().unwrap();
        assert!(
            (final_vy - terminal).abs() < 0.01 * terminal.abs(),
            "ended at {final_vy} against a terminal {terminal}"
        );
    }

    /// The derived threshold must describe the method rather than happen to
    /// be a small number, so this pins its scaling.
    ///
    /// RK4's *local* error on a linear decay is `O(z^5)`, so halving the step
    /// divides it by 32; twice as many steps then bring the accumulated total
    /// back up by two, for a net factor of 16. That 16 — not 32, and not the
    /// 8 a global-fourth-order argument would suggest — is the signature of
    /// this particular formula, and it is what tells you `rk4_decay_error` is
    /// computing what its name says.
    #[test]
    fn the_rk4_decay_error_scales_as_the_method_does() {
        let coarse = rk4_decay_error(2.0, 0.1, 100.0);
        let fine = rk4_decay_error(2.0, 0.05, 200.0);
        assert!(coarse > 0.0 && fine > 0.0, "the error must not be zero");
        let ratio = coarse / fine;
        assert!(
            (ratio - 16.0).abs() < 1.0,
            "halving the step changed the accumulated error by {ratio}, expected about 16 \
             (a factor of 32 per step against twice as many steps)"
        );
    }
}
