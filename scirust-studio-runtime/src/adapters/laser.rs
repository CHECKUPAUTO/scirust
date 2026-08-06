//! `sim.optoelectronics.semiconductor_laser` — the single-mode rate equations
//! of a laser diode (`scirust_sim::laser::SemiconductorLaser`).
//!
//! Two coupled states, the carrier density `n` and the photon density `s`,
//! driven by a constant pump. This is the model behind a laser diode's
//! threshold, its linear light–current curve, and the relaxation-oscillation
//! ringing you see on a fast turn-on.
//!
//! # Two checks that come from the same equations by different routes
//!
//! The first check is about *where* the run ends up. Above threshold the
//! modal gain clamps the carrier density at `n_th = n_t + 1/(Γ·g₀·τ_p)` no
//! matter how hard the diode is pumped, and every additional carrier goes
//! into light: `s_ss = Γ·τ_p·(J − J_th)`. Both are closed forms the model
//! itself states, so the check compares against them.
//!
//! The second is about the *shape of the approach*. Linearising the rate
//! equations about that operating point gives the Jacobian
//!
//! ```text
//!     [ -(1/τ_n + g₀·s_ss)    -1/(Γ·τ_p) ]
//!     [    Γ·g₀·s_ss               0     ]
//! ```
//!
//! — the lower-right entry is exactly zero because gain equals loss at the
//! clamp, which is what makes the operating point a clamp in the first place.
//! Its determinant is `ω_n² = g₀·s_ss/τ_p`, the relaxation-oscillation
//! frequency the model reports, and its trace is `−2γ`, so the ringing decays
//! at `γ = (1/τ_n + g₀·s_ss)/2` and oscillates at the *damped* frequency
//! `ω_d = √(ω_n² − γ²)`.
//!
//! So the run's own ringing period is compared against `2π/ω_d`. The two
//! sides of that comparison come from the same rate equations by entirely
//! different routes — one by numerically integrating them, the other by
//! taking eigenvalues of their Jacobian — which is what makes it a check
//! rather than a restatement.
//!
//! The damping correction is not cosmetic. For the shipped tutorial the
//! damped period is 0.57 % longer than the undamped one, and the tolerance is
//! 0.1 %, so a version of this check that compared against `1/f_r` would
//! fail. There is a test that asserts exactly that, so the correction cannot
//! be simplified away without something going red.
//!
//! # β must be zero for any of that to be true
//!
//! Every closed form above is the `β → 0` limit. With spontaneous emission
//! coupled into the mode there is no threshold in the strict sense — the
//! diode is never quite dark, the carrier density is never quite clamped —
//! and the model does not state the (quadratic) `β > 0` operating point.
//!
//! Rather than re-derive it here, which would put a second copy of the
//! physics in the adapter and leave the "verification" checking the adapter
//! against itself, a scenario with `β > 0` runs and reports both checks as
//! [`VerificationStatus::NotApplicable`], with a warning saying why. The
//! simulation is still correct; it is the *oracle* that does not apply.

use scirust_sim::laser::{LaserParams, SemiconductorLaser};
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
use crate::measure::period_from_second_half;
use crate::result::{
    Axis, AxisMonotonicity, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, RunWarning, Series, SeriesRole, TIME_AXIS_ID, VerificationResult,
    VerificationStatus, WarningCategory,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    RK4_SOLVER, check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// Slack over the residual the run's own damping implies.
const SETTLE_MARGIN: f64 = 5.0;

/// The fraction of the run used to measure how big the ringing started out.
const AMPLITUDE_WINDOW: f64 = 0.1;

/// Relative agreement demanded between the measured ringing period and
/// `2π/ω_d`.
///
/// This one is *chosen*, not derived, and the two error sources it has to
/// cover are worth naming because both are far below it:
///
///   * locating a zero crossing by linear interpolation. The signal's
///     curvature-to-slope ratio at a crossing is exactly `2γ`, so the error
///     per crossing is bounded by `γ·(h/2)²` — under 2e-7 s for the shipped
///     tutorial, against a measured span of some 3.6 s;
///   * the nonlinearity the linearised Jacobian ignores, which is second
///     order in the ringing amplitude. The period is measured over the
///     *second half* of the run precisely so that amplitude is small: by then
///     it is under 1 % of the steady state, so the correction is of order
///     1e-4 relative.
///
/// 1e-3 is comfortably above both and comfortably below the 5.7e-3 that
/// separates the damped period from the undamped one — which is the gap the
/// check has to be able to see.
const PERIOD_TOLERANCE: f64 = 1e-3;

/// Below this the round-off floor, not the physics, sets the answer.
const ROUNDOFF_FLOOR: f64 = 1e-12;

const G0: FieldDescriptor = FieldDescriptor {
    canonical_name: "g0",
    display_name: "Differential gain",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Gain per unit carrier density above transparency. Densities are normalised (see the tutorial), so this is a rate.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 262),
};

const N_T: FieldDescriptor = FieldDescriptor {
    canonical_name: "n_t",
    display_name: "Transparency carrier density",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Carrier density at which the medium stops absorbing and starts amplifying.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 263),
};

const TAU_N: FieldDescriptor = FieldDescriptor {
    canonical_name: "tau_n",
    display_name: "Carrier lifetime",
    required: true,
    dimension: scirust_units::Dimension::TIME,
    accepted_units: &["s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Spontaneous recombination lifetime of a carrier.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 264),
};

const TAU_P: FieldDescriptor = FieldDescriptor {
    canonical_name: "tau_p",
    display_name: "Photon lifetime",
    required: true,
    dimension: scirust_units::Dimension::TIME,
    accepted_units: &["s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "How long a photon survives in the cavity before it is lost. Sets the threshold.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 265),
};

const GAMMA: FieldDescriptor = FieldDescriptor {
    canonical_name: "gamma",
    display_name: "Optical confinement factor",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction of the optical mode that overlaps the active region, in (0, 1].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 266),
};

const BETA: FieldDescriptor = FieldDescriptor {
    canonical_name: "beta",
    display_name: "Spontaneous emission coupling",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction of spontaneous emission that lands in the lasing mode. Zero is the limit in which the closed forms this capability verifies against hold.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 267),
};

const PUMP: FieldDescriptor = FieldDescriptor {
    canonical_name: "pump",
    display_name: "Pump rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Carriers injected per unit time. Above J_th the output rises linearly with it.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 268),
};

const CARRIER_DENSITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "carrier_density",
    display_name: "Initial carrier density",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Carrier density at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 269),
};

const PHOTON_DENSITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "photon_density",
    display_name: "Initial photon density",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Photon density at t = start. With beta = 0 an exactly dark cavity stays dark, so a turn-on run needs a seed.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 270),
};

const OPERATING_POINT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "settles_at_the_clamped_operating_point",
    description: "Above threshold the modal gain clamps the carrier density at n_t + 1/(Gamma*g0*tau_p) however hard the diode is pumped, and the surplus goes into light as Gamma*tau_p*(J - J_th). Both states are checked against those closed forms, with a threshold derived from how much of its own ringing the run's length damps out. Reported as not applicable when beta > 0, where the closed forms do not hold.",
};

const RINGING_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "rings_at_the_relaxation_frequency",
    description: "The run's own ringing period is measured from the zero crossings of the photon density about its steady state and compared with 2*pi/omega_d, where omega_d comes from the eigenvalues of the linearised rate equations. Numerical integration and linear stability analysis are different routes to the same answer, so a model whose fixed point is right but whose curvature there is wrong passes the operating-point check and fails this one.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.optoelectronics.semiconductor_laser"),
    display_name: "Semiconductor laser rate equations",
    category: CapabilityCategory::Optoelectronics,
    source_crate: "scirust-sim",
    summary: "A single-mode laser diode: carrier and photon densities under a constant pump, clamping at threshold and ringing on the way there.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[G0, N_T, TAU_N, TAU_P, GAMMA, BETA, PUMP],
    initial_state: &[CARRIER_DENSITY, PHOTON_DENSITY],
    outputs: &[
        OutputDescriptor {
            id: "carrier_density",
            display_name: "Carrier density",
            unit: "1",
            description: "Carriers available to be stimulated into the mode.",
        },
        OutputDescriptor {
            id: "photon_density",
            display_name: "Photon density",
            unit: "1",
            description: "The light in the cavity — the diode's output, up to a constant.",
        },
        OutputDescriptor {
            id: "net_gain",
            display_name: "Net modal gain",
            unit: "1/s",
            description: "Modal gain minus cavity loss. Exactly zero at the clamped operating point, which is what clamping means.",
        },
        OutputDescriptor {
            id: "steady_carrier_density",
            display_name: "Threshold carrier density",
            unit: "1",
            description: "The closed-form level the carrier density clamps at.",
        },
        OutputDescriptor {
            id: "steady_photon_density",
            display_name: "Steady photon density",
            unit: "1",
            description: "The closed-form light-current level.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[OPERATING_POINT_CHECK, RINGING_CHECK],
    },
};

/// The `sim.optoelectronics.semiconductor_laser` adapter.
#[derive(Debug, Default)]
pub struct SemiconductorLaserAdapter;

impl CapabilityAdapter for SemiconductorLaserAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["g0", "n_t", "tau_n", "tau_p", "gamma", "beta", "pump"],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["carrier_density", "photon_density"],
        ));
        for e in [
            resolve_model_scalar(scenario, &G0).err(),
            resolve_model_scalar(scenario, &N_T).err(),
            resolve_model_scalar(scenario, &TAU_N).err(),
            resolve_model_scalar(scenario, &TAU_P).err(),
            resolve_model_scalar(scenario, &GAMMA).err(),
            resolve_model_scalar(scenario, &BETA).err(),
            resolve_model_scalar(scenario, &PUMP).err(),
            resolve_state_vector(scenario, &CARRIER_DENSITY).err(),
            resolve_state_vector(scenario, &PHOTON_DENSITY).err(),
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
        let g0 = resolve_model_scalar(s, &G0).expect("validated");
        let n_t = resolve_model_scalar(s, &N_T).expect("validated");
        let tau_n = resolve_model_scalar(s, &TAU_N).expect("validated");
        let tau_p = resolve_model_scalar(s, &TAU_P).expect("validated");
        let gamma = resolve_model_scalar(s, &GAMMA).expect("validated");
        let beta = resolve_model_scalar(s, &BETA).expect("validated");
        let pump = resolve_model_scalar(s, &PUMP).expect("validated");
        let n0 = resolve_state_vector(s, &CARRIER_DENSITY).expect("validated")[0];
        let s0 = resolve_state_vector(s, &PHOTON_DENSITY).expect("validated")[0];

        let model = SemiconductorLaser::new(LaserParams {
            g0,
            n_t,
            tau_n,
            tau_p,
            gamma,
            beta,
            pump,
        })
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

        let y0 = model.initial_state(n0, s0);
        let traj = simulate_cancellable(
            &model,
            &y0,
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let carrier = traj.column(0).expect("dim 0 exists");
        let photon = traj.column(1).expect("dim 1 exists");
        // Modal gain minus cavity loss. Zero exactly where the clamp is.
        let net_gain: Vec<f64> = carrier
            .iter()
            .map(|n| gamma * g0 * (n - n_t) - 1.0 / tau_p)
            .collect();

        let n_ss = model.steady_state_carrier_density();
        let s_ss = model.steady_state_photon_density();
        let decay = decay_rate(g0, tau_n, s_ss);

        let lasing = s_ss > 0.0;
        let oracle_applies = beta == 0.0 && lasing;
        let mut warnings = Vec::new();
        if beta != 0.0
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: format!(
                    "beta = {beta} is not zero, so the threshold and light-current closed forms \
                     this capability verifies against do not describe this run. The integration \
                     is unaffected; both checks report as not applicable."
                ),
            });
        }
        else if !lasing
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: format!(
                    "the pump is at or below threshold ({pump:.6} against J_th = {:.6}), so the \
                     diode stays dark: there is no lasing operating point to settle at and no \
                     relaxation oscillation to measure. Both checks report as not applicable.",
                    model.threshold_pump()
                ),
            });
        }

        let operating_point = operating_point_check(
            &traj.t,
            &carrier,
            &photon,
            n_ss,
            s_ss,
            decay,
            oracle_applies,
        );
        let measured_period = period_from_second_half(&traj.t, &photon, s_ss);
        let predicted_period = damped_period(model.relaxation_frequency(), decay);
        let ringing = ringing_check(measured_period, predicted_period, oracle_applies);

        // A period that does not exist is reported by its absence, not by a
        // NaN: a metric a reader can see is a number they will use.
        let period_metrics = [
            ("damped_period", "Predicted damped period", predicted_period),
            (
                "measured_period",
                "Measured ringing period",
                measured_period,
            ),
        ]
        .into_iter()
        .filter_map(|(id, display_name, value)| {
            value.map(|v| Metric {
                id: id.to_string(),
                display_name: display_name.to_string(),
                value: MetricValue::Scalar(v),
                unit: Some("s".to_string()),
            })
        });

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
                    id: "carrier_density".to_string(),
                    display_name: "Carrier density".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: carrier.clone(),
                },
                Series {
                    id: "photon_density".to_string(),
                    display_name: "Photon density".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: photon.clone(),
                },
                Series {
                    id: "net_gain".to_string(),
                    display_name: "Net modal gain".to_string(),
                    unit: "1/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: net_gain,
                },
                Series {
                    id: "steady_carrier_density".to_string(),
                    display_name: "Threshold carrier density".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![n_ss; traj.t.len()],
                },
                Series {
                    id: "steady_photon_density".to_string(),
                    display_name: "Steady photon density".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![s_ss; traj.t.len()],
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "threshold_density".to_string(),
                    display_name: "Threshold carrier density n_th".to_string(),
                    value: MetricValue::Scalar(model.threshold_density()),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "threshold_pump".to_string(),
                    display_name: "Threshold pump J_th".to_string(),
                    value: MetricValue::Scalar(model.threshold_pump()),
                    unit: Some("1/s".to_string()),
                },
                Metric {
                    id: "steady_photon_density".to_string(),
                    display_name: "Steady photon density".to_string(),
                    value: MetricValue::Scalar(s_ss),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "relaxation_frequency".to_string(),
                    display_name: "Undamped relaxation frequency f_r".to_string(),
                    value: MetricValue::Scalar(model.relaxation_frequency()),
                    unit: Some("Hz".to_string()),
                },
                Metric {
                    id: "ringing_decay_rate".to_string(),
                    display_name: "Ringing decay rate".to_string(),
                    value: MetricValue::Scalar(decay),
                    unit: Some("1/s".to_string()),
                },
            ]
            .into_iter()
            .chain(period_metrics)
            .collect(),
            warnings,
            verifications: vec![operating_point, ringing],
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

/// The rate at which relaxation oscillations decay: half the negated trace of
/// the Jacobian at the clamped operating point, `(1/tau_n + g0*s_ss)/2`.
fn decay_rate(g0: f64, tau_n: f64, s_ss: f64) -> f64 {
    (1.0 / tau_n + g0 * s_ss) / 2.0
}

/// The period the ringing actually has, `2*pi/sqrt(omega_n^2 - gamma^2)`.
///
/// `None` when there is no ringing to have a period: below threshold
/// (`omega_n = 0`) or, in principle, if the damping ever overwhelmed the
/// oscillation.
fn damped_period(relaxation_frequency: f64, decay: f64) -> Option<f64> {
    let omega_n = 2.0 * std::f64::consts::PI * relaxation_frequency;
    let omega_d_squared = omega_n * omega_n - decay * decay;
    if omega_d_squared > 0.0
    {
        Some(2.0 * std::f64::consts::PI / omega_d_squared.sqrt())
    }
    else
    {
        None
    }
}

/// How far a series ends from a level, relative to how far it started —
/// paired with the largest excursion seen early on, which is the amplitude
/// the decay is measured against.
fn deviation(values: &[f64], level: f64, window: usize, scale: f64) -> (f64, f64) {
    let amplitude = values[..window.max(1).min(values.len())]
        .iter()
        .map(|v| (v - level).abs())
        .fold(0.0_f64, f64::max);
    let final_error = (values.last().copied().unwrap_or(f64::NAN) - level).abs();
    let scale = scale.abs().max(f64::MIN_POSITIVE);
    (amplitude / scale, final_error / scale)
}

/// Both states against their closed-form operating point.
fn operating_point_check(
    t: &[f64],
    carrier: &[f64],
    photon: &[f64],
    n_ss: f64,
    s_ss: f64,
    decay: f64,
    oracle_applies: bool,
) -> VerificationResult {
    if !oracle_applies
    {
        return VerificationResult {
            id: OPERATING_POINT_CHECK.id.to_string(),
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: "the closed-form threshold and light-current levels are the beta -> 0, \
                          above-threshold limit, which this scenario is not in"
                .to_string(),
        };
    }

    let span = t.last().copied().unwrap_or(0.0) - t[0];
    let residual = (-decay * span).exp();
    let window = ((t.len() as f64) * AMPLITUDE_WINDOW).ceil() as usize;

    // The amplitude is taken over the *start* of the run and the decay over
    // the *whole* run, so the product is an upper bound on the envelope at
    // the end rather than an estimate of it.
    let (n_amplitude, n_error) = deviation(carrier, n_ss, window, n_ss);
    let (s_amplitude, s_error) = deviation(photon, s_ss, window, s_ss);
    let worst = n_error.max(s_error);
    let threshold = (n_amplitude.max(s_amplitude) * residual * SETTLE_MARGIN).max(ROUNDOFF_FLOOR);

    VerificationResult {
        id: OPERATING_POINT_CHECK.id.to_string(),
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
            "the carrier density ended {n_error:.3e} from its clamp at {n_ss:.6} and the photon \
             density {s_error:.3e} from {s_ss:.6}, both relative; the run started ringing at \
             {:.3e} and its own damping rate of {decay:.6} 1/s leaves {residual:.3e} of that \
             after {span:.4} s",
            n_amplitude.max(s_amplitude)
        ),
    }
}

/// The measured ringing period against the linear-stability prediction.
fn ringing_check(
    measured: Option<f64>,
    predicted: Option<f64>,
    oracle_applies: bool,
) -> VerificationResult {
    let id = RINGING_CHECK.id.to_string();
    let not_applicable = |why: &str| VerificationResult {
        id: id.clone(),
        status: VerificationStatus::NotApplicable,
        measured: None,
        threshold: None,
        explanation: why.to_string(),
    };
    if !oracle_applies
    {
        return not_applicable(
            "the linearised operating point this frequency is derived about is the beta -> 0, \
             above-threshold one, which this scenario is not in",
        );
    }
    let Some(predicted) = predicted
    else
    {
        return not_applicable("the operating point is not oscillatory, so it has no period");
    };
    let Some(measured) = measured
    else
    {
        return not_applicable(
            "the run's second half contains fewer than three crossings of the steady level, so \
             there is no period in it to measure",
        );
    };

    let error = (measured - predicted).abs() / predicted;
    VerificationResult {
        id,
        status: if error <= PERIOD_TOLERANCE
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(error),
        threshold: Some(PERIOD_TOLERANCE),
        explanation: format!(
            "the run rang with a period of {measured:.9} s against the {predicted:.9} s the \
             eigenvalues of the linearised rate equations predict, a relative difference of \
             {error:.3e}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::crossing_times;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/semiconductor_laser.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = SemiconductorLaserAdapter;
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

    fn metric(result: &RunResult, id: &str) -> f64 {
        match result
            .metrics
            .iter()
            .find(|m| m.id == id)
            .expect("metric")
            .value
        {
            MetricValue::Scalar(v) => v,
            ref other => panic!("{id} is not a scalar: {other:?}"),
        }
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        assert_eq!(parse_toml(TUTORIAL).unwrap().capability.id, DESCRIPTOR.id.0);
    }

    #[test]
    fn the_same_scenario_run_twice_is_bit_identical() {
        let (a, b) = (run(), run());
        assert!(
            a.series[1]
                .values
                .iter()
                .zip(b.series[1].values.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let validated = SemiconductorLaserAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            SemiconductorLaserAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn it_settles_at_the_clamped_operating_point() {
        let r = run();
        let v = check(&r, OPERATING_POINT_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn it_rings_at_the_relaxation_frequency() {
        let r = run();
        let v = check(&r, RINGING_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The run must actually ring, or the period check is measuring noise.
    #[test]
    fn the_tutorial_genuinely_oscillates() {
        let r = run();
        let s_ss = metric(&r, "steady_photon_density");
        let photon = &r
            .series
            .iter()
            .find(|s| s.id == "photon_density")
            .expect("photon_density")
            .values;
        let t = &r.time_axis().expect("time axis").values;
        let crossings = crossing_times(t, photon, s_ss);
        assert!(
            crossings.len() > 10,
            "only {} crossings — this is not a ringing run",
            crossings.len()
        );
    }

    /// The whole reason the check compares against `2*pi/omega_d` rather than
    /// `1/f_r`: the damping correction is larger than the tolerance, so a
    /// simplification back to the undamped frequency fails.
    ///
    /// If this ever stops holding — because the tutorial's parameters moved
    /// to a more weakly damped regime — the correction has become invisible
    /// and the check has stopped testing it, which is worth knowing.
    #[test]
    fn the_period_check_distinguishes_the_damped_frequency_from_the_undamped_one() {
        let r = run();
        let damped = metric(&r, "damped_period");
        let undamped = 1.0 / metric(&r, "relaxation_frequency");
        let gap = (damped - undamped).abs() / damped;
        assert!(
            gap > PERIOD_TOLERANCE,
            "the damped and undamped periods differ by only {gap:.3e}, which is inside the \
             {PERIOD_TOLERANCE:.0e} tolerance — the check can no longer tell them apart"
        );

        // And the measured period really is the damped one.
        let measured = metric(&r, "measured_period");
        assert!(
            (measured - damped).abs() < (measured - undamped).abs(),
            "measured {measured:.9} is closer to the undamped {undamped:.9} than to the damped \
             {damped:.9}"
        );
    }

    /// Below threshold there is no lasing operating point and no oscillation.
    /// The run is still valid; the oracle is what does not apply, and saying
    /// so is different from silently passing.
    #[test]
    fn a_dark_diode_reports_both_checks_as_not_applicable() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        // J_th is 101 for these parameters.
        scenario.model.get_mut("pump").expect("pump").value = 50.0;
        let adapter = SemiconductorLaserAdapter;
        let validated = adapter.validate(&scenario).expect("still valid");
        let r = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("still runs");
        for id in [OPERATING_POINT_CHECK.id, RINGING_CHECK.id]
        {
            assert_eq!(
                check(&r, id).status,
                VerificationStatus::NotApplicable,
                "{id}"
            );
        }
        assert!(
            !r.warnings.is_empty(),
            "a run whose checks all abstain must say why"
        );
    }

    /// The same for spontaneous emission: `beta > 0` runs, and abstains.
    #[test]
    fn spontaneous_emission_makes_the_closed_forms_inapplicable() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.get_mut("beta").expect("beta").value = 1.0e-4;
        let adapter = SemiconductorLaserAdapter;
        let validated = adapter.validate(&scenario).expect("still valid");
        let r = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("still runs");
        for id in [OPERATING_POINT_CHECK.id, RINGING_CHECK.id]
        {
            assert_eq!(
                check(&r, id).status,
                VerificationStatus::NotApplicable,
                "{id}"
            );
        }
        assert!(r.warnings.iter().any(|w| w.message.contains("beta")));
    }

    /// The clamp, stated as a test rather than a comment: at the operating
    /// point the modal gain equals the cavity loss exactly, so the net gain
    /// series must end at zero.
    #[test]
    fn the_gain_clamps_against_the_cavity_loss() {
        let r = run();
        let net = &r
            .series
            .iter()
            .find(|s| s.id == "net_gain")
            .expect("net_gain")
            .values;
        let end = net.last().copied().unwrap();
        // Relative to the cavity loss the gain is being compared against.
        let loss = 1.0 / parse_toml(TUTORIAL).unwrap().model["tau_p"].value;
        assert!(
            end.abs() / loss < 1e-4,
            "net gain ended at {end}, which is {} of the cavity loss",
            end.abs() / loss
        );
    }
}
