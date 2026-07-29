//! `sim.power.swing_equation` — the rotor angle of a synchronous machine tied
//! to an infinite bus (`scirust_sim::grid::SwingEquation`).
//!
//! One second-order state, the rotor angle `δ`, obeying
//!
//! ```text
//!     δ'' = (ω_s/2H)·(P_m − P_max·sin δ) − (D/2H)·δ'
//! ```
//!
//! This is the electromechanical mode behind transient stability: the ~1 Hz
//! swing a generator makes after a disturbance, and whether it comes back.
//!
//! # An integrator whose errors are the right shape
//!
//! With `D = 0` the swing equation is a one-degree-of-freedom Hamiltonian
//! system, and this capability integrates it with symplectic Euler for a
//! reason that both of its checks depend on.
//!
//! Symplectic Euler conserves a *modified* Hamiltonian `H̃ = H − (h/2)·p·V'(q)`
//! exactly. The consequence is that the true energy does not drift: it
//! oscillates within a band of half-width `(h/2)·max|δ'·δ''|` and stays
//! there for as long as you care to integrate. For a small swing that band,
//! measured against the oscillation's own energy `½·(a·ω_n)²`, works out to
//! `h·ω_n/2` — the amplitude cancels out entirely, which is why the energy
//! check's threshold is `h·ω_n` and has no fitted constant in it.
//!
//! The second consequence is subtler and is what makes the period check
//! possible at all. Symplectic Euler is only first-order accurate in the
//! trajectory, but its *frequency* error is second order: for a harmonic
//! oscillator the discrete frequency satisfies `cos(ω_h·h) = 1 − ω_n²h²/2`,
//! giving `ω_h = ω_n·(1 + ω_n²h²/24)`. For the shipped tutorial that is a
//! relative error of 9e-8, four orders of magnitude below the physical effect
//! the check is looking for.
//!
//! # The tolerance is the nonlinearity, computed
//!
//! The measured swing period is compared against the small-signal `2π/ω_n`
//! the model states — but a finite swing is slower than an infinitesimal one,
//! and by a knowable amount. Writing `x = δ − δ*` and expanding gives
//!
//! ```text
//!     x'' + ω_n²x − (k·P_max·sin δ*/2)·x² − (ω_n²/6)·x³ = 0
//! ```
//!
//! and the standard Lindstedt result for `x'' + ω₀²x + α₂x² + α₃x³ = 0`,
//! `ω = ω₀·[1 + (3α₃/8ω₀² − 5α₂²/12ω₀⁴)·a²]`, collapses to
//!
//! ```text
//!     ΔT/T = a²·(1/16 + 5·tan²δ*/48)
//! ```
//!
//! (At `δ* = 0` that is the textbook pendulum's `a²/16`, which is the check
//! on the algebra.)
//!
//! That number is used to *size the tolerance*, not to shift the prediction.
//! Correcting the prediction would make the check depend on a
//! perturbation-theory coefficient being exactly right; sizing the tolerance
//! with it means the check still fails whenever the frequency is off by more
//! than the nonlinearity could possibly explain, even if the coefficient were
//! out by a factor of two. A separate test asserts the measured period comes
//! out *longer* than the small-signal one, which is the direction the
//! nonlinearity has to push it and which no tolerance can fake.

use scirust_sim::grid::SwingEquation;
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind, RunDomain,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::execute_support::{TimeSpan, simulate_second_order_cancellable};
use crate::measure::period_from_second_half;
use crate::result::{
    Axis, AxisMonotonicity, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, RunWarning, Series, SeriesRole, TIME_AXIS_ID, VerificationResult,
    VerificationStatus, WarningCategory,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// Slack over the `h*omega_n` energy band symplectic Euler's modified
/// Hamiltonian implies.
const ENERGY_MARGIN: f64 = 4.0;

/// Slack over the computed nonlinear period correction.
const PERIOD_MARGIN: f64 = 4.0;

/// How much of the run each half-mean is taken over, when asking whether the
/// energy band is drifting rather than merely wide.
const HALF: f64 = 0.5;

const SYMPLECTIC_EULER: SolverDescriptor = SolverDescriptor {
    id: "symplectic_euler",
    summary: "Fixed-step semi-implicit Euler. First order in the trajectory, second order in the oscillation frequency, and — for the undamped machine — energy-bounded rather than energy-drifting. Requires `solver.step`.",
    fixed_step: true,
    adaptive_tolerance: false,
    reports_progress: true,
};

const FREQUENCY_HZ: FieldDescriptor = FieldDescriptor {
    canonical_name: "frequency_hz",
    display_name: "System frequency",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["Hz"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Nominal system frequency; the synchronous speed is 2*pi times it.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 271),
};

const INERTIA_H: FieldDescriptor = FieldDescriptor {
    canonical_name: "inertia_h",
    display_name: "Inertia constant H",
    required: true,
    dimension: scirust_units::Dimension::TIME,
    accepted_units: &["s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Stored kinetic energy at rated speed divided by rated power — seconds of full output the rotor could supply from its own spin.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 272),
};

const DAMPING: FieldDescriptor = FieldDescriptor {
    canonical_name: "damping",
    display_name: "Damping coefficient",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["pu"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Per-unit damping torque. Zero is the conservative case in which the transient energy is an exact invariant.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 273),
};

const P_MECH: FieldDescriptor = FieldDescriptor {
    canonical_name: "p_mech",
    display_name: "Mechanical power",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["pu"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Per-unit power the turbine puts in. Above p_max there is no synchronous operating point at all.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 274),
};

const P_MAX: FieldDescriptor = FieldDescriptor {
    canonical_name: "p_max",
    display_name: "Peak transfer power",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["pu"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Per-unit peak of the power-angle curve, E*V/X. Losing a line lowers it, which is what a fault does to this model.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 275),
};

const ROTOR_ANGLE: FieldDescriptor = FieldDescriptor {
    canonical_name: "rotor_angle",
    display_name: "Initial rotor angle",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["rad"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rotor angle relative to the infinite bus at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 276),
};

const ROTOR_SPEED: FieldDescriptor = FieldDescriptor {
    canonical_name: "rotor_speed",
    display_name: "Initial rotor speed deviation",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["rad/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "How fast the rotor angle is moving at t = start; zero is a machine displaced but not yet accelerating.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 277),
};

const ENERGY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "transient_energy_stays_bounded",
    description: "With no damping the transient energy is an exact invariant, and symplectic Euler conserves a modified Hamiltonian that keeps the true energy inside a band of relative half-width h*omega_n/2 rather than letting it drift. Both halves of that claim are checked: the band's width against h*omega_n, and its centre in the run's second half against its centre in the first. A drifting integrator can pass the first and not the second. Reported as not applicable when damping is nonzero, where the energy is genuinely dissipated.",
};

const PERIOD_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "swings_at_the_small_signal_frequency",
    description: "The run's own swing period, measured from the crossings of the equilibrium angle, against the 2*pi/omega_n the model states. The tolerance is the leading nonlinear correction a^2*(1/16 + 5*tan(delta*)^2/48) computed from the run's own amplitude, so a swing large enough to be measurably slower does not need the tolerance loosened by hand. Reported as not applicable when there is no equilibrium to swing about.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.power.swing_equation"),
    display_name: "Synchronous machine swing equation",
    category: CapabilityCategory::Power,
    source_crate: "scirust-sim",
    summary: "A generator tied to an infinite bus: the rotor-angle transient behind power-system stability, integrated symplectically.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[SYMPLECTIC_EULER],
    parameters: &[FREQUENCY_HZ, INERTIA_H, DAMPING, P_MECH, P_MAX],
    initial_state: &[ROTOR_ANGLE, ROTOR_SPEED],
    outputs: &[
        OutputDescriptor {
            id: "rotor_angle",
            display_name: "Rotor angle",
            unit: "rad",
            description: "Angle of the rotor relative to the infinite bus.",
        },
        OutputDescriptor {
            id: "rotor_speed",
            display_name: "Rotor speed deviation",
            unit: "rad/s",
            description: "Rate of change of the rotor angle — the machine running fast or slow.",
        },
        OutputDescriptor {
            id: "electrical_power",
            display_name: "Electrical power out",
            unit: "pu",
            description: "P_max*sin(delta): what the machine is actually delivering. Its gap from P_mech is what accelerates the rotor.",
        },
        OutputDescriptor {
            id: "transient_energy",
            display_name: "Transient energy",
            unit: "1",
            description: "The conserved quantity of the undamped machine, tracked so its band is visible rather than merely asserted.",
        },
        OutputDescriptor {
            id: "equilibrium_angle",
            display_name: "Equilibrium angle",
            unit: "rad",
            description: "asin(P_mech/P_max), the angle the machine settles at.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ENERGY_CHECK, PERIOD_CHECK],
    },
};

/// The `sim.power.swing_equation` adapter.
#[derive(Debug, Default)]
pub struct SwingEquationAdapter;

impl CapabilityAdapter for SwingEquationAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["frequency_hz", "inertia_h", "damping", "p_mech", "p_max"],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["rotor_angle", "rotor_speed"],
        ));
        for e in [
            resolve_model_scalar(scenario, &FREQUENCY_HZ).err(),
            resolve_model_scalar(scenario, &INERTIA_H).err(),
            resolve_model_scalar(scenario, &DAMPING).err(),
            resolve_model_scalar(scenario, &P_MECH).err(),
            resolve_model_scalar(scenario, &P_MAX).err(),
            resolve_state_vector(scenario, &ROTOR_ANGLE).err(),
            resolve_state_vector(scenario, &ROTOR_SPEED).err(),
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
        let damping = resolve_model_scalar(s, &DAMPING).expect("validated");
        let p_mech = resolve_model_scalar(s, &P_MECH).expect("validated");
        let p_max = resolve_model_scalar(s, &P_MAX).expect("validated");
        let delta0 = resolve_state_vector(s, &ROTOR_ANGLE).expect("validated")[0];
        let omega0 = resolve_state_vector(s, &ROTOR_SPEED).expect("validated")[0];

        let model = SwingEquation::new(
            resolve_model_scalar(s, &FREQUENCY_HZ).expect("validated"),
            resolve_model_scalar(s, &INERTIA_H).expect("validated"),
            damping,
            p_mech,
            p_max,
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
            .expect("validated: symplectic_euler requires solver.step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let traj = simulate_second_order_cancellable(
            &model,
            &[delta0],
            &[omega0],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        // `simulate_second_order` concatenates (q, v), so with one degree of
        // freedom the rows are [delta, delta'].
        let angle = traj.column(0).expect("dof 0 exists");
        let speed = traj.column(1).expect("dof 0's velocity exists");
        let electrical: Vec<f64> = angle.iter().map(|d| p_max * d.sin()).collect();
        let energy: Vec<f64> = angle
            .iter()
            .zip(speed.iter())
            .map(|(d, w)| model.energy(&[*d], &[*w]).expect("length 1"))
            .collect();

        let equilibrium = model.equilibrium_angle();
        let omega_n = model.small_signal_frequency();

        let mut warnings = Vec::new();
        if equilibrium.is_none()
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: format!(
                    "p_mech = {p_mech} exceeds p_max = {p_max}, so the power-angle curve never \
                     meets the mechanical input and the machine has no synchronous operating \
                     point: this run is a loss of synchronism, not a swing about an equilibrium. \
                     The integration is valid; both checks report as not applicable because \
                     neither has an equilibrium to be about."
                ),
            });
        }
        if damping != 0.0
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: format!(
                    "damping = {damping} is nonzero, so the transient energy is dissipated rather \
                     than conserved and the energy check reports as not applicable."
                ),
            });
        }

        let energy_result = energy_check(&energy, &speed, damping, omega_n, step);
        let measured_period =
            equilibrium.and_then(|eq| period_from_second_half(&traj.t, &angle, eq));
        let period_result = period_check(
            measured_period,
            omega_n,
            equilibrium,
            &angle,
            p_mech / p_max,
        );

        let mut metrics = vec![
            Metric {
                id: "peak_angle".to_string(),
                display_name: "Largest rotor angle reached".to_string(),
                value: MetricValue::Scalar(angle.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
                unit: Some("rad".to_string()),
            },
            Metric {
                id: "peak_speed".to_string(),
                display_name: "Largest speed deviation reached".to_string(),
                value: MetricValue::Scalar(speed.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)),
                unit: Some("rad/s".to_string()),
            },
        ];
        if let Some(eq) = equilibrium
        {
            metrics.push(Metric {
                id: "equilibrium_angle".to_string(),
                display_name: "Equilibrium rotor angle".to_string(),
                value: MetricValue::Scalar(eq),
                unit: Some("rad".to_string()),
            });
        }
        if let Some(w) = omega_n
        {
            metrics.push(Metric {
                id: "small_signal_frequency".to_string(),
                display_name: "Small-signal frequency omega_n".to_string(),
                value: MetricValue::Scalar(w),
                unit: Some("rad/s".to_string()),
            });
            metrics.push(Metric {
                id: "small_signal_period".to_string(),
                display_name: "Small-signal period".to_string(),
                value: MetricValue::Scalar(2.0 * std::f64::consts::PI / w),
                unit: Some("s".to_string()),
            });
        }
        if let Some(p) = measured_period
        {
            metrics.push(Metric {
                id: "measured_period".to_string(),
                display_name: "Measured swing period".to_string(),
                value: MetricValue::Scalar(p),
                unit: Some("s".to_string()),
            });
        }

        let mut series = vec![
            Series {
                id: "rotor_angle".to_string(),
                display_name: "Rotor angle".to_string(),
                unit: "rad".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Trajectory,
                values: angle.clone(),
            },
            Series {
                id: "rotor_speed".to_string(),
                display_name: "Rotor speed deviation".to_string(),
                unit: "rad/s".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Trajectory,
                values: speed.clone(),
            },
            Series {
                id: "electrical_power".to_string(),
                display_name: "Electrical power out".to_string(),
                unit: "pu".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Trajectory,
                values: electrical,
            },
            Series {
                id: "transient_energy".to_string(),
                display_name: "Transient energy".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Trajectory,
                values: energy.clone(),
            },
        ];
        if let Some(eq) = equilibrium
        {
            series.push(Series {
                id: "equilibrium_angle".to_string(),
                display_name: "Equilibrium angle".to_string(),
                unit: "rad".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Reference,
                values: vec![eq; traj.t.len()],
            });
        }

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
            series,
            fields: vec![],
            distributions: vec![],
            metrics,
            warnings,
            verifications: vec![energy_result, period_result],
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

/// The mean of a slice, or `NaN` for an empty one.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// The transient energy's band width, and whether the band is drifting.
///
/// The width is the easy half. The drift is the half that says something
/// about the *integrator*: a symplectic method conserves a nearby Hamiltonian
/// exactly, so the band's centre stays put however long the run, while a
/// method that merely has small errors accumulates them. Comparing the mean
/// energy over the run's two halves is the cheapest statement of that
/// difference, and the worse of the two numbers is what the check reports.
fn energy_check(
    energy: &[f64],
    speed: &[f64],
    damping: f64,
    omega_n: Option<f64>,
    step: f64,
) -> VerificationResult {
    let id = ENERGY_CHECK.id.to_string();
    let (Some(omega_n), true) = (omega_n, damping == 0.0)
    else
    {
        return VerificationResult {
            id,
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: if damping != 0.0
            {
                "the transient energy is an invariant only of the undamped machine, and this one \
                 has damping"
                    .to_string()
            }
            else
            {
                "there is no synchronous operating point, so there is no small-signal frequency \
                 to size the integrator's energy band against"
                    .to_string()
            },
        };
    };

    // The oscillation's own energy scale: peak kinetic energy. Normalising by
    // the total energy instead would divide by the large constant potential
    // offset and make any band look tiny.
    let scale = speed
        .iter()
        .map(|v| 0.5 * v * v)
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);

    let lo = energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let width = (hi - lo) / scale;

    let split = ((energy.len() as f64) * HALF) as usize;
    let drift = (mean(&energy[split..]) - mean(&energy[..split])).abs() / scale;

    // Half-width h*omega_n/2 gives a full width of h*omega_n.
    let threshold = step * omega_n * ENERGY_MARGIN;
    let worst = width.max(drift);

    VerificationResult {
        id,
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
            "the transient energy stayed inside a band {width:.3e} wide relative to the swing's \
             own peak kinetic energy, and the band's centre moved {drift:.3e} between the run's \
             two halves, against the {threshold:.3e} that a step of {step:e} at omega_n = \
             {omega_n:.6} rad/s allows"
        ),
    }
}

/// The leading nonlinear lengthening of the swing period, as a fraction.
///
/// `a` is the swing amplitude in radians and `sin_delta_eq` is `P_m/P_max`.
/// See this module's header for where the coefficient comes from; at
/// `sin_delta_eq = 0` it reduces to the textbook pendulum's `a²/16`.
fn nonlinear_period_correction(amplitude: f64, sin_delta_eq: f64) -> f64 {
    let cos_squared = (1.0 - sin_delta_eq * sin_delta_eq).max(f64::MIN_POSITIVE);
    let tan_squared = sin_delta_eq * sin_delta_eq / cos_squared;
    amplitude * amplitude * (1.0 / 16.0 + 5.0 * tan_squared / 48.0)
}

/// The measured swing period against the small-signal one.
fn period_check(
    measured: Option<f64>,
    omega_n: Option<f64>,
    equilibrium: Option<f64>,
    angle: &[f64],
    sin_delta_eq: f64,
) -> VerificationResult {
    let id = PERIOD_CHECK.id.to_string();
    let not_applicable = |why: &str| VerificationResult {
        id: id.clone(),
        status: VerificationStatus::NotApplicable,
        measured: None,
        threshold: None,
        explanation: why.to_string(),
    };
    let (Some(measured), Some(omega_n), Some(equilibrium)) = (measured, omega_n, equilibrium)
    else
    {
        return not_applicable(
            "this run has no equilibrium angle to swing about, or too few crossings of it in the \
             measurement window to give a period",
        );
    };

    let amplitude = angle
        .iter()
        .map(|d| (d - equilibrium).abs())
        .fold(0.0_f64, f64::max);
    let predicted = 2.0 * std::f64::consts::PI / omega_n;
    let error = (measured - predicted).abs() / predicted;
    let correction = nonlinear_period_correction(amplitude, sin_delta_eq);
    let threshold = correction * PERIOD_MARGIN;

    VerificationResult {
        id,
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
            "the rotor swung with a period of {measured:.9} s against the small-signal \
             {predicted:.9} s, a relative difference of {error:.3e}; a swing of {amplitude:.4} rad \
             about an equilibrium at {equilibrium:.4} rad is expected to be {correction:.3e} \
             slower than an infinitesimal one, which is what sets the threshold"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/swing_equation.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = SwingEquationAdapter;
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
            .expect("check")
    }

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
        let validated = SwingEquationAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            SwingEquationAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn the_transient_energy_stays_bounded() {
        let r = run();
        let v = check(&r, ENERGY_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn it_swings_at_the_small_signal_frequency() {
        let r = run();
        let v = check(&r, PERIOD_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The direction no tolerance can fake: a finite swing is *slower* than an
    /// infinitesimal one. If the measured period ever came out shorter than
    /// the small-signal prediction, the tolerance would still be satisfied
    /// and the physics would still be wrong.
    #[test]
    fn a_finite_swing_is_slower_than_an_infinitesimal_one() {
        let r = run();
        let measured = metric(&r, "measured_period");
        let small_signal = metric(&r, "small_signal_period");
        assert!(
            measured > small_signal,
            "measured {measured:.9} s is not longer than the small-signal {small_signal:.9} s"
        );
        // And by roughly the predicted amount, not by an order of magnitude.
        let observed = (measured - small_signal) / small_signal;
        let amplitude = metric(&r, "peak_angle") - metric(&r, "equilibrium_angle");
        let predicted = nonlinear_period_correction(amplitude, 0.5);
        assert!(
            (observed / predicted - 1.0).abs() < 0.5,
            "lengthening was {observed:.3e} against a predicted {predicted:.3e}"
        );
    }

    /// The energy band must be *measurable*, or the check is asserting that
    /// a number nobody computed is small. Symplectic Euler is first order, so
    /// at this step size the band is real and visible.
    #[test]
    fn the_energy_band_is_not_trivially_zero() {
        let r = run();
        let v = check(&r, ENERGY_CHECK.id);
        let measured = v.measured.expect("a measured band");
        assert!(
            measured > 1e-9,
            "the band measured {measured:e}, which is round-off rather than the integrator's \
             O(h) energy error — the threshold is no longer testing anything"
        );
    }

    /// A damped machine dissipates its transient energy, so the invariant is
    /// not one. The run is still valid and still settles; the energy check
    /// says it does not apply rather than failing or being quietly widened.
    #[test]
    fn a_damped_machine_abstains_from_the_energy_check_and_settles() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        // A realistic 2 pu of damping is light: it takes about a hundred
        // seconds to settle, not ten. Coarsening the step to match keeps this
        // test's cost where the tutorial's is.
        scenario.model.get_mut("damping").expect("damping").value = 2.0;
        scenario.solver.end.value = 100.0;
        scenario.solver.step.as_mut().expect("step").value = 0.002;
        let r = run_scenario(&scenario);
        assert_eq!(
            check(&r, ENERGY_CHECK.id).status,
            VerificationStatus::NotApplicable
        );
        assert!(r.warnings.iter().any(|w| w.message.contains("damping")));

        // It really did settle: the last angle is at the equilibrium.
        let angle = &r
            .series
            .iter()
            .find(|s| s.id == "rotor_angle")
            .expect("rotor_angle")
            .values;
        let equilibrium = metric(&r, "equilibrium_angle");
        assert!(
            (angle.last().unwrap() - equilibrium).abs() < 1e-3,
            "settled at {} rather than {equilibrium}",
            angle.last().unwrap()
        );
    }

    /// More mechanical power than the line can transfer: the power-angle
    /// curve never meets the input, the rotor accelerates away, and there is
    /// no equilibrium for either check to be about. This is loss of
    /// synchronism, the failure mode the whole model exists to predict, and
    /// the adapter must report it as such rather than compare against an
    /// equilibrium that does not exist.
    #[test]
    fn losing_synchronism_is_reported_rather_than_checked_against_nothing() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.get_mut("p_mech").expect("p_mech").value = 3.0; // p_max is 2.0
        let r = run_scenario(&scenario);
        for id in [ENERGY_CHECK.id, PERIOD_CHECK.id]
        {
            assert_eq!(
                check(&r, id).status,
                VerificationStatus::NotApplicable,
                "{id}"
            );
        }
        assert!(
            r.warnings
                .iter()
                .any(|w| w.message.contains("synchronous operating point"))
        );

        // And the rotor really did run away rather than oscillate.
        let angle = &r
            .series
            .iter()
            .find(|s| s.id == "rotor_angle")
            .expect("rotor_angle")
            .values;
        assert!(
            angle.last().unwrap() - angle[0] > 10.0,
            "the rotor advanced only {} rad",
            angle.last().unwrap() - angle[0]
        );
    }

    /// At zero equilibrium angle the correction must be the pendulum's
    /// `a^2/16` exactly — the check on the algebra in the module header.
    #[test]
    fn the_nonlinear_correction_reduces_to_the_pendulum_at_zero_load() {
        let a = 0.3;
        let got = nonlinear_period_correction(a, 0.0);
        assert!((got - a * a / 16.0).abs() < 1e-15, "{got}");
        // And a loaded machine is more nonlinear than an unloaded one, since
        // the potential is no longer symmetric about the equilibrium.
        assert!(nonlinear_period_correction(a, 0.5) > got);
    }

    #[test]
    fn an_undamped_run_declares_no_damping_warning() {
        assert!(!run().warnings.iter().any(|w| w.message.contains("damping")));
    }
}
