//! `sim.thermal.heat_rod_1d` — heat diffusing along a rod
//! (`scirust_sim::thermal::HeatRod1d`).
//!
//! The first capability whose result is a **field** rather than a set of
//! curves. `∂u/∂t = α·∂²u/∂x²` on a rod with both ends held at fixed
//! temperatures, discretised into `n` interior nodes and integrated with RK4.
//!
//! # Why this needed [`crate::result::Field`]
//!
//! Every earlier capability produced quantities that vary along one axis.
//! This one produces `u(x, t)`, which varies along two. A [`Series`] is
//! aligned one-to-one with a single axis, so the most it can hold is a slice:
//! one node's history, or one instant's profile. `REPOSITORY_AUDIT.md` said
//! for two phases that schema v2 could already express a discretised field;
//! reading it again while writing this showed it could express `n` slices and
//! lose the fact they are one quantity. See
//! `docs/studio/adr/0009-fields.md`.
//!
//! # What is verified, and why these three
//!
//! The semi-discrete system is *exactly* right about two things, and
//! `scirust-sim`'s own tests already exploit both. This adapter turns them
//! into checks on the run:
//!
//! * **The steady state is the linear profile** between the boundary
//!   temperatures. Not approximately — exactly, for the discretisation.
//! * **The slowest mode decays at `λ₁ = (2α/dx²)(1 − cos(π/(n+1)))`.** After
//!   the transient any initial condition is dominated by that mode, so the
//!   deviation from steady state decays at a rate the model can state in
//!   closed form and the run can be measured against.
//!
//! The third is the discrete maximum principle: with no source term, no
//! interior node may leave the range spanned by the initial profile and the
//! two boundaries. It catches a class of bug — an index slip, a sign error in
//! the stencil — that the first two can miss, because a wrong stencil can
//! still relax to something.
//!
//! # The stability limit is enforced, not discovered
//!
//! Explicit RK4 on this stencil is unstable above `h ≈ 0.7·dx²/α`, and the
//! model exposes that bound as `stable_step()`. A scenario above it does not
//! produce a poor answer; it produces `NaN` after a few hundred steps. So it
//! is a **validation error**, with the limit in the message. Letting it run
//! and reporting "non-finite value" would be telling the user what happened
//! instead of what to do.

use scirust_sim::thermal::HeatRod1d;

use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind, RunDomain,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::execute_support::{TimeSpan, simulate_cancellable};
use crate::result::{
    Axis, AxisMonotonicity, Field, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance,
    RunResult, RunSummary, Series, SeriesRole, TIME_AXIS_ID, VerificationResult,
    VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// The id of the spatial axis, alongside [`TIME_AXIS_ID`].
///
/// The first capability to need a second axis at all. `validate_result` finds
/// the time axis by id, so this must not be `"t"`.
pub const SPACE_AXIS_ID: &str = "x";

/// How much of the run's tail the decay-rate check measures over.
///
/// The slowest mode only dominates once the faster ones have died, so the
/// measurement starts halfway through rather than at the beginning. Fitting
/// from `t = 0` would measure a mixture and then blame the model for not
/// being a single exponential.
const DECAY_FIT_TAIL_FRACTION: f64 = 0.5;

/// How far the measured decay rate may sit from the analytic one.
///
/// Twenty percent, which is loose for a quantity known in closed form — and
/// deliberately so. The measurement is a two-point fit over a tail that still
/// holds a little of the second mode, and the second mode's rate is roughly
/// four times the first, so a residue of it biases the estimate upward. A
/// band tight enough to catch that residue would fail on correct runs.
/// Catching a *wrong stencil* — which changes the rate by a factor, not a
/// percent — is what this check is for.
const DECAY_RATE_TOLERANCE: f64 = 0.2;

/// How far the final profile may sit from the exact steady state, relative to
/// the temperature range the rod spans.
const STEADY_STATE_TOLERANCE: f64 = 0.02;

const DIFFUSIVITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "diffusivity",
    display_name: "Thermal diffusivity",
    required: true,
    dimension: scirust_units::Dimension::LENGTH
        .powi(2)
        .div(scirust_units::Dimension::TIME),
    accepted_units: &["m^2/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Thermal diffusivity alpha = k/(rho*c_p). Copper is about 1.1e-4 m^2/s, \
                  stainless steel about 4e-6.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 200),
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
    description: "The rod's length. The node spacing is length/(nodes + 1), because the two \
                  boundaries sit at the ends and the nodes are the interior.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 201),
};

const NODES: FieldDescriptor = FieldDescriptor {
    canonical_name: "nodes",
    display_name: "Interior nodes",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(1.0),
    min_inclusive: true,
    max: Some(MAX_NODES as f64),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "How many interior points the rod is discretised into. More nodes resolve the \
                  profile better and force a quadratically smaller stable step.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 202),
};

const T_LEFT: FieldDescriptor = FieldDescriptor {
    canonical_name: "t_left",
    display_name: "Left boundary temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The temperature the left end is held at.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 203),
};

const T_RIGHT: FieldDescriptor = FieldDescriptor {
    canonical_name: "t_right",
    display_name: "Right boundary temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The temperature the right end is held at.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 204),
};

const TEMPERATURE: FieldDescriptor = FieldDescriptor {
    canonical_name: "temperature",
    display_name: "Initial temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The uniform temperature the whole interior starts at. A single number rather \
                  than a per-node profile: `initial_state` has a fixed shape per capability, and \
                  the node count is a parameter of this one.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 205),
};

/// The largest discretisation this adapter accepts.
///
/// The field it produces is `rows x nodes`, so the node count multiplies the
/// whole result. This bounds one factor; the stable-step rule bounds the
/// other far more aggressively, since halving `dx` quarters the step and so
/// quadruples the rows.
const MAX_NODES: usize = 512;

/// A stable step is a property of the discretisation, not a preference.
const CODE_UNSTABLE_STEP: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 206);

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Classical fixed-step Runge-Kutta 4 over the semi-discrete system. Explicit, so \
              `solver.step` is bounded above by the diffusion stability limit 0.7*dx^2/alpha, \
              which this capability checks.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const STEADY_STATE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "approaches_steady_state",
    description: "With both ends held fixed and no source, the semi-discrete system's steady \
                  state is exactly the linear profile between the boundary temperatures. The \
                  final profile is compared against it, relative to the temperature range the \
                  rod spans.",
};

const DECAY_RATE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "slowest_mode_decay_rate",
    description: "After the transient, the deviation from steady state is dominated by the \
                  slowest discrete sine mode, which decays at exactly \
                  lambda_1 = (2*alpha/dx^2)*(1 - cos(pi/(n+1))). The measured decay of the \
                  deviation's L2 norm over the run's tail is compared against that closed form.",
};

const MAXIMUM_PRINCIPLE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "maximum_principle",
    description: "Heat diffusion with no source creates no new extremes: every interior \
                  temperature must stay within the range spanned by the initial profile and the \
                  two boundary temperatures, at every time. Catches stencil and index errors \
                  that a run still relaxing to a plausible profile would hide.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.thermal.heat_rod_1d"),
    display_name: "Heat diffusion along a rod",
    category: CapabilityCategory::Thermal,
    source_crate: "scirust-sim",
    summary: "One-dimensional heat diffusion with both ends held at fixed temperatures, \
              discretised into interior nodes and integrated with fixed-step RK4. Produces a \
              field over space and time rather than a set of curves.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[DIFFUSIVITY, LENGTH, NODES, T_LEFT, T_RIGHT],
    initial_state: &[TEMPERATURE],
    outputs: &[
        OutputDescriptor {
            id: "temperature",
            display_name: "Temperature",
            unit: "K",
            description: "The temperature field over space and time.",
        },
        OutputDescriptor {
            id: "deviation_norm",
            display_name: "Deviation from steady state",
            unit: "K",
            description: "The L2 norm of the profile's deviation from the steady state, over \
                          time. This is the quantity whose decay rate is verified.",
        },
        OutputDescriptor {
            id: "final_profile",
            display_name: "Final profile",
            unit: "K",
            description: "The temperature along the rod at the end of the run.",
        },
        OutputDescriptor {
            id: "steady_state",
            display_name: "Steady state",
            unit: "K",
            description: "The exact linear profile the rod relaxes to, for comparison.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[
            STEADY_STATE_CHECK,
            DECAY_RATE_CHECK,
            MAXIMUM_PRINCIPLE_CHECK,
        ],
    },
};

/// The `sim.thermal.heat_rod_1d` adapter.
#[derive(Debug, Default)]
pub struct HeatRodAdapter;

/// The resolved geometry of a scenario, shared by validation and execution.
struct Geometry {
    diffusivity: f64,
    nodes: usize,
    dx: f64,
}

impl Geometry {
    fn resolve(scenario: &Scenario) -> Option<Geometry> {
        let diffusivity = resolve_model_scalar(scenario, &DIFFUSIVITY).ok()?;
        let length = resolve_model_scalar(scenario, &LENGTH).ok()?;
        let nodes = resolve_model_scalar(scenario, &NODES).ok()?;
        // `nodes` is a count carried as an f64. A non-integer is a typo, not
        // a discretisation, and rounding it silently would run something the
        // scenario does not say.
        if nodes.fract() != 0.0
        {
            return None;
        }
        let nodes = nodes as usize;
        Some(Geometry {
            diffusivity,
            nodes,
            dx: length / (nodes + 1) as f64,
        })
    }

    /// The explicit diffusion stability limit, `0.7·dx²/α`.
    fn stable_step(&self) -> f64 {
        0.7 * self.dx * self.dx / self.diffusivity
    }
}

impl CapabilityAdapter for HeatRodAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["diffusivity", "length", "nodes", "t_left", "t_right"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["temperature"]));
        for e in [
            resolve_model_scalar(scenario, &DIFFUSIVITY).err(),
            resolve_model_scalar(scenario, &LENGTH).err(),
            resolve_model_scalar(scenario, &NODES).err(),
            resolve_model_scalar(scenario, &T_LEFT).err(),
            resolve_model_scalar(scenario, &T_RIGHT).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        if let Err(e) = resolve_state_vector(scenario, &TEMPERATURE)
        {
            errors.push(e);
        }
        if let Err(e) = resolve_solver(scenario, DESCRIPTOR.supported_solvers)
        {
            errors.push(e);
        }
        // Every adapter checks these, including the deterministic ones — see
        // `resolve_replicates`.
        if let Err(e) = resolve_replicates(scenario, DESCRIPTOR.determinism)
        {
            errors.push(e);
        }
        if let Err(e) = resolve_backend_kind(scenario, &DESCRIPTOR)
        {
            errors.push(e);
        }
        if let Err(e) = resolve_precision(scenario, &DESCRIPTOR)
        {
            errors.push(e);
        }

        // The node count is a count.
        if let Ok(nodes) = resolve_model_scalar(scenario, &NODES)
        {
            if nodes.fract() != 0.0
            {
                errors.push(scirust_studio_command::CatalogedError {
                    code: NODES.error_code,
                    title: "Invalid `nodes`".to_string(),
                    explanation: format!("`model.nodes` = {nodes} is not a whole number of nodes"),
                    recoverable: true,
                    suggested_action: Some("use an integer, e.g. 40".to_string()),
                });
            }
        }

        // The stability limit. Checked here rather than discovered at run
        // time, because above it the run does not degrade — it diverges.
        if errors.is_empty()
        {
            if let (Some(geometry), Some(step)) = (
                Geometry::resolve(scenario),
                scenario
                    .solver
                    .step
                    .as_ref()
                    .and_then(|s| s.to_quantity("solver.step").ok())
                    .map(|q| q.value),
            )
            {
                let limit = geometry.stable_step();
                if step > limit
                {
                    errors.push(scirust_studio_command::CatalogedError {
                        code: CODE_UNSTABLE_STEP,
                        title: "Step above the diffusion stability limit".to_string(),
                        explanation: format!(
                            "`solver.step` = {step:.6e} s exceeds {limit:.6e} s, the explicit \
                             stability limit 0.7*dx^2/alpha for this discretisation \
                             (dx = {:.6e} m, alpha = {:.6e} m^2/s). Above it RK4 on this \
                             stencil does not lose accuracy, it diverges to non-finite values",
                            geometry.dx, geometry.diffusivity
                        ),
                        recoverable: true,
                        suggested_action: Some(format!(
                            "use `step = {{ value = {:.3e}, unit = \"s\" }}` or smaller, or \
                             reduce `nodes` — the limit grows as dx^2, so halving the node \
                             count roughly quadruples it",
                            limit * 0.9
                        )),
                    });
                }
            }
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

        let geometry = Geometry::resolve(s).expect("validated");
        let t_left = resolve_model_scalar(s, &T_LEFT).expect("validated");
        let t_right = resolve_model_scalar(s, &T_RIGHT).expect("validated");
        let u0 = resolve_state_vector(s, &TEMPERATURE).expect("validated")[0];

        let model = HeatRod1d::new(
            geometry.diffusivity,
            geometry.dx,
            geometry.nodes,
            t_left,
            t_right,
        )
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

        let initial = vec![u0; geometry.nodes];
        let traj = simulate_cancellable(
            &model,
            &initial,
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let rows = traj.t.len();
        let columns = geometry.nodes;

        // The exact steady state, straight from the model rather than
        // re-derived here: two implementations of the same formula is one
        // more than the number that can be right.
        let steady: Vec<f64> = (0..columns)
            .map(|i| model.steady_state(i).expect("i < nodes"))
            .collect();

        let positions: Vec<f64> = (0..columns).map(|i| (i + 1) as f64 * geometry.dx).collect();

        let mut values = Vec::with_capacity(rows * columns);
        for row in &traj.y
        {
            values.extend_from_slice(row);
        }

        let deviation_norm: Vec<f64> = traj
            .y
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&steady)
                    .map(|(u, s)| (u - s).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let final_profile = traj
            .y
            .last()
            .cloned()
            .expect("a trajectory has at least the initial state");

        let steady_state = steady_state_check(&final_profile, &steady, t_left, t_right);
        let decay = decay_rate_check(&traj.t, &deviation_norm, &model, columns);
        let maximum = maximum_principle_check(&traj.y, u0, t_left, t_right);

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: TIME_AXIS_ID.to_string(),
                steps: rows - 1,
                t_start: traj.t[0],
                t_end: *traj.t.last().expect("at least one sample"),
            },
            axes: vec![
                Axis {
                    id: TIME_AXIS_ID.to_string(),
                    display_name: "time".to_string(),
                    unit: "s".to_string(),
                    monotonicity: AxisMonotonicity::StrictlyIncreasing,
                    values: traj.t.clone(),
                },
                Axis {
                    id: SPACE_AXIS_ID.to_string(),
                    display_name: "position".to_string(),
                    unit: "m".to_string(),
                    monotonicity: AxisMonotonicity::StrictlyIncreasing,
                    values: positions,
                },
            ],
            series: vec![
                Series {
                    id: "deviation_norm".to_string(),
                    display_name: "Deviation from steady state".to_string(),
                    unit: "K".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: deviation_norm.clone(),
                },
                // Two profiles against the *space* axis — the first series in
                // this catalogue that are not plotted against time.
                Series {
                    id: "final_profile".to_string(),
                    display_name: "Final profile".to_string(),
                    unit: "K".to_string(),
                    axis_id: SPACE_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: final_profile.clone(),
                },
                Series {
                    id: "steady_state".to_string(),
                    display_name: "Steady state".to_string(),
                    unit: "K".to_string(),
                    axis_id: SPACE_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: steady.clone(),
                },
            ],
            fields: vec![Field {
                id: "temperature".to_string(),
                display_name: "Temperature".to_string(),
                unit: "K".to_string(),
                row_axis_id: TIME_AXIS_ID.to_string(),
                column_axis_id: SPACE_AXIS_ID.to_string(),
                columns,
                values,
            }],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "nodes".to_string(),
                    display_name: "Interior nodes".to_string(),
                    value: MetricValue::Integer(columns as i64),
                    unit: None,
                },
                Metric {
                    id: "node_spacing".to_string(),
                    display_name: "Node spacing dx".to_string(),
                    value: MetricValue::Scalar(geometry.dx),
                    unit: Some("m".to_string()),
                },
                Metric {
                    id: "stable_step_limit".to_string(),
                    display_name: "Stability limit 0.7*dx^2/alpha".to_string(),
                    value: MetricValue::Scalar(geometry.stable_step()),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "slowest_mode_rate".to_string(),
                    display_name: "Slowest mode decay rate lambda_1".to_string(),
                    value: MetricValue::Scalar(model.mode_decay_rate(1).unwrap_or(0.0)),
                    unit: Some("1/s".to_string()),
                },
                Metric {
                    id: "final_deviation".to_string(),
                    display_name: "Final deviation from steady state".to_string(),
                    value: MetricValue::Scalar(
                        *deviation_norm.last().expect("at least one sample"),
                    ),
                    unit: Some("K".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![steady_state, decay, maximum],
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

/// Compare the final profile against the exact linear steady state.
fn steady_state_check(
    final_profile: &[f64],
    steady: &[f64],
    t_left: f64,
    t_right: f64,
) -> VerificationResult {
    let worst = final_profile
        .iter()
        .zip(steady)
        .map(|(u, s)| (u - s).abs())
        .fold(0.0_f64, f64::max);

    // Relative to the temperature range the rod spans, so the tolerance means
    // the same thing for a 1 K difference and a 500 K one. A rod whose ends
    // are at the same temperature spans nothing, and the absolute error is
    // then the only meaningful measure.
    let span = (t_right - t_left).abs();
    let (measured, threshold, scale) = if span > 0.0
    {
        (worst / span, STEADY_STATE_TOLERANCE, "of the boundary span")
    }
    else
    {
        (worst, STEADY_STATE_TOLERANCE, "K (absolute)")
    };

    VerificationResult {
        id: "approaches_steady_state".to_string(),
        status: if measured <= threshold
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(measured),
        threshold: Some(threshold),
        explanation: format!(
            "the final profile differs from the exact linear steady state by at most \
             {worst:.6} K, i.e. {measured:.2e} {scale}; the run may simply be too short if \
             this is the only check that failed"
        ),
    }
}

/// Measure the decay of the deviation over the run's tail and compare it
/// against the model's closed-form slowest-mode rate.
fn decay_rate_check(
    t: &[f64],
    deviation: &[f64],
    model: &HeatRod1d,
    nodes: usize,
) -> VerificationResult {
    let id = "slowest_mode_decay_rate".to_string();
    let not_applicable = |explanation: String| VerificationResult {
        id: id.clone(),
        status: VerificationStatus::NotApplicable,
        measured: None,
        threshold: None,
        explanation,
    };

    let Some(expected) = model.mode_decay_rate(1)
    else
    {
        return not_applicable("the rod has no modes to decay".to_string());
    };
    let _ = nodes;

    let start = (t.len() as f64 * DECAY_FIT_TAIL_FRACTION) as usize;
    let (Some(&t0), Some(&t1)) = (t.get(start), t.last())
    else
    {
        return not_applicable("the run is too short to measure a decay rate".to_string());
    };
    let (Some(&d0), Some(&d1)) = (deviation.get(start), deviation.last())
    else
    {
        return not_applicable("the run is too short to measure a decay rate".to_string());
    };

    // A deviation that has already reached the floor carries no rate: the
    // logarithm of a number at the edge of f64 resolution is noise, and
    // reporting a failure for a run that converged *better* than the check
    // can measure would be exactly backwards.
    if d0 <= 0.0 || d1 <= 0.0 || d1 / d0 < 1e-12
    {
        return not_applicable(format!(
            "the deviation fell from {d0:.3e} to {d1:.3e} K, which is at or below the \
             resolution this measurement has; the profile has converged"
        ));
    }
    if t1 <= t0
    {
        return not_applicable("the tail spans no time".to_string());
    }

    let measured = -(d1 / d0).ln() / (t1 - t0);
    let ratio = measured / expected;
    let passed = (ratio - 1.0).abs() <= DECAY_RATE_TOLERANCE;

    VerificationResult {
        id,
        status: if passed
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(ratio),
        threshold: Some(1.0 + DECAY_RATE_TOLERANCE),
        explanation: format!(
            "the deviation from steady state decayed at {measured:.6e} 1/s over the run's \
             tail, against the closed-form slowest-mode rate \
             lambda_1 = (2*alpha/dx^2)*(1 - cos(pi/(n+1))) = {expected:.6e} 1/s \
             (ratio {ratio:.3})"
        ),
    }
}

/// The discrete maximum principle: diffusion with no source creates no new
/// extremes.
fn maximum_principle_check(
    rows: &[Vec<f64>],
    initial: f64,
    t_left: f64,
    t_right: f64,
) -> VerificationResult {
    let lower = initial.min(t_left).min(t_right);
    let upper = initial.max(t_left).max(t_right);

    // A tolerance is needed because RK4 is not the exact semi-discrete flow:
    // it overshoots by O(h^5) per step. Scaled to the range so it means the
    // same thing whatever the temperatures are.
    let tolerance = ((upper - lower).abs() * 1e-9).max(1e-9);

    let mut worst = 0.0_f64;
    for row in rows
    {
        for &u in row
        {
            worst = worst.max((lower - u).max(u - upper).max(0.0));
        }
    }

    VerificationResult {
        id: "maximum_principle".to_string(),
        status: if worst <= tolerance
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(worst),
        threshold: Some(tolerance),
        explanation: format!(
            "every interior temperature stayed within [{lower}, {upper}] K — the range spanned \
             by the initial profile and the two boundaries — to within {worst:.3e} K; a \
             violation would mean the stencil created heat that was not there"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/heat_rod.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = HeatRodAdapter;
        let validated = adapter.validate(scenario).expect("valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    fn run() -> RunResult {
        run_scenario(&parse_toml(TUTORIAL).unwrap())
    }

    fn check<'a>(result: &'a RunResult, id: &str) -> &'a VerificationResult {
        result
            .verifications
            .iter()
            .find(|v| v.id == id)
            .unwrap_or_else(|| panic!("no check {id}"))
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        assert_eq!(scenario.capability.id, DESCRIPTOR.id.0);
    }

    /// The point of the whole phase: a result that is a field, shaped by two
    /// axes, and consistent with both.
    #[test]
    fn the_result_carries_a_field_shaped_by_both_axes() {
        let result = run();
        assert_eq!(result.fields.len(), 1);
        let field = &result.fields[0];

        let time = result
            .axes
            .iter()
            .find(|a| a.id == TIME_AXIS_ID)
            .expect("a time axis");
        let space = result
            .axes
            .iter()
            .find(|a| a.id == SPACE_AXIS_ID)
            .expect("a space axis");

        assert_eq!(field.columns, space.values.len());
        assert_eq!(field.rows(), time.values.len());
        assert_eq!(field.values.len(), field.rows() * field.columns);
        assert_eq!(field.row_axis_id, TIME_AXIS_ID);
        assert_eq!(field.column_axis_id, SPACE_AXIS_ID);
    }

    /// The field's first row must be the initial condition and its last the
    /// final profile — otherwise the row-major packing is transposed, which
    /// produces a plausible picture of nothing.
    #[test]
    fn the_fields_rows_are_instants_and_its_columns_are_positions() {
        let result = run();
        let field = &result.fields[0];
        let final_profile = &result
            .series
            .iter()
            .find(|s| s.id == "final_profile")
            .expect("a final profile")
            .values;

        let first = field.row(0).expect("a first row");
        assert_eq!(first.len(), field.columns);
        // The rod starts uniform, so every value in row 0 is the same.
        assert!(first.windows(2).all(|w| w[0] == w[1]), "{first:?}");

        let last = field.row(field.rows() - 1).expect("a last row");
        assert_eq!(last, final_profile.as_slice());
        assert_eq!(field.at(field.rows() - 1, 0), Some(final_profile[0]));
    }

    #[test]
    fn the_rod_relaxes_to_the_exact_linear_steady_state() {
        let result = run();
        let verdict = check(&result, "approaches_steady_state");
        assert_eq!(
            verdict.status,
            VerificationStatus::Passed,
            "{}",
            verdict.explanation
        );
    }

    #[test]
    fn the_decay_rate_matches_the_closed_form() {
        let result = run();
        let verdict = check(&result, "slowest_mode_decay_rate");
        assert_eq!(
            verdict.status,
            VerificationStatus::Passed,
            "{}",
            verdict.explanation
        );
        assert!(verdict.explanation.contains("lambda_1"));
    }

    #[test]
    fn no_interior_node_leaves_the_range_it_started_in() {
        let result = run();
        let verdict = check(&result, "maximum_principle");
        assert_eq!(
            verdict.status,
            VerificationStatus::Passed,
            "{}",
            verdict.explanation
        );
    }

    /// The steady state is the linear profile, so the two boundary-adjacent
    /// nodes must bracket everything and the middle must be the mean. Checked
    /// against the model's own formula rather than re-derived.
    #[test]
    fn the_steady_state_series_is_the_linear_profile() {
        let result = run();
        let steady = &result
            .series
            .iter()
            .find(|s| s.id == "steady_state")
            .expect("a steady state")
            .values;
        assert_eq!(steady.len(), result.fields[0].columns);
        // Strictly monotone between two different boundary temperatures.
        assert!(
            steady.windows(2).all(|w| w[1] > w[0]) || steady.windows(2).all(|w| w[1] < w[0]),
            "{steady:?}"
        );
        // Evenly spaced, because it is a straight line on an even grid.
        let first_gap = steady[1] - steady[0];
        for w in steady.windows(2)
        {
            assert!((w[1] - w[0] - first_gap).abs() < 1e-9);
        }
    }

    /// A check that cannot fail is decoration. A wrong steady state — here,
    /// one that is simply the wrong profile — must be caught.
    #[test]
    fn the_steady_state_check_catches_a_profile_that_is_wrong() {
        let steady = vec![300.0, 310.0, 320.0];
        let wrong = vec![300.0, 380.0, 320.0];
        let verdict = steady_state_check(&wrong, &steady, 290.0, 330.0);
        assert_eq!(verdict.status, VerificationStatus::Failed);
    }

    /// The maximum principle must fire on a value outside the range, not just
    /// report a number.
    #[test]
    fn the_maximum_principle_catches_heat_from_nowhere() {
        let rows = vec![vec![300.0, 300.0], vec![300.0, 900.0]];
        let verdict = maximum_principle_check(&rows, 300.0, 300.0, 400.0);
        assert_eq!(verdict.status, VerificationStatus::Failed);
        assert!(verdict.measured.unwrap() > 400.0);
    }

    /// A step above the stability limit does not give a worse answer, it
    /// gives `NaN`. Refusing is the only honest option, and the message has
    /// to carry the limit.
    #[test]
    fn a_step_above_the_stability_limit_is_refused_with_the_limit() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        let stable = Geometry::resolve(&scenario)
            .expect("resolves")
            .stable_step();
        scenario.solver.step = Some(scirust_studio_schema::ValueWithUnit {
            value: stable * 4.0,
            unit: "s".to_string(),
        });
        let report = HeatRodAdapter.validate(&scenario).unwrap_err();
        let error = report
            .errors
            .iter()
            .find(|e| e.code == CODE_UNSTABLE_STEP)
            .expect("the stability error");
        assert!(error.explanation.contains("stability limit"));
        assert!(error.suggested_action.is_some());
    }

    /// …and the tutorial itself must be comfortably inside it, or the shipped
    /// example is one edit away from diverging.
    #[test]
    fn the_tutorial_sits_inside_the_stability_limit() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let geometry = Geometry::resolve(&scenario).expect("resolves");
        let step = scenario
            .solver
            .step
            .as_ref()
            .expect("a step")
            .to_quantity("solver.step")
            .expect("resolves")
            .value;
        assert!(
            step <= geometry.stable_step(),
            "the tutorial's step {step} exceeds the limit {}",
            geometry.stable_step()
        );
    }

    #[test]
    fn a_fractional_node_count_is_refused() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "nodes".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 12.5,
                unit: "1".to_string(),
            },
        );
        let report = HeatRodAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("whole number"))
        );
    }

    #[test]
    fn the_same_scenario_run_twice_is_bit_identical() {
        let first = run();
        let second = run();
        assert!(
            first.fields[0]
                .values
                .iter()
                .zip(second.fields[0].values.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = HeatRodAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            adapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }
}
