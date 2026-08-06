//! `sim.optoelectronics.avalanche_photodiode` — the gain–SNR trade-off of an
//! APD receiver (`scirust_sim::apd::Apd`).
//!
//! The first capability with no time in it. An avalanche photodiode's
//! receiver analysis is algebraic: given a gain, closed forms give the signal
//! current, the multiplied shot noise, the thermal floor and the
//! signal-to-noise ratio. Nothing is integrated. So this capability sweeps
//! the *gain* instead of stepping time, and its results carry a `gain` axis
//! and no `t` — which is what
//! [`RunDomain::ParameterSweep`](scirust_studio_registry::RunDomain) exists
//! to declare.
//!
//! # The question the sweep is asking
//!
//! Raising the avalanche gain `M` multiplies the signal by `M`, so the signal
//! *power* grows as `M²`. It also multiplies the shot noise — but by more
//! than `M²`, because the cascade is itself random: the McIntyre excess-noise
//! factor `F(M) = k·M + (1−k)·(2 − 1/M)` grows with `M` too. Against a fixed
//! thermal floor that produces a genuine optimum. Turn the gain up from unity
//! and the SNR improves, because you are lifting the signal off the thermal
//! noise; keep going and it collapses, because you are now amplifying the
//! shot noise faster than the signal.
//!
//! # Two checks, and neither of them has a tolerance
//!
//! Writing `SNR = A·M²/(C·M²·F(M) + T)` with `A = (ℛ·P)²`, `C = 2q·I_p·B` and
//! `T` the thermal variance, maximising the SNR means minimising
//! `C·F(M) + T/M²`, whose stationary condition is
//!
//! ```text
//!     C·k·M³ + C·(1−k)·M = 2T
//! ```
//!
//! The left side is strictly increasing in `M` for every `k` in `[0, 1]`, so
//! that cubic has exactly one positive root — the optimal gain — and
//! bisection finds it without a derivative or an initial guess.
//!
//! That gives both checks their sharp form:
//!
//!   * **the optimum is where the closed form says.** The SNR evaluated at
//!     the analytic root must be at least as large as the SNR at *every*
//!     swept gain. That is not an approximation to compare loosely against;
//!     it is what "this is the maximum" means, and the only slack allowed is
//!     a few ulps of round-off.
//!   * **the SNR is unimodal in gain.** Because the stationary condition has
//!     exactly one root, `C·F + T/M²` falls then rises and the SNR rises then
//!     falls — one sign change in the difference sequence, no more. A model
//!     with the right optimum but a spurious second bump passes the first
//!     check and fails this one.
//!
//! Both are theorems about the model that a sweep can falsify, rather than
//! numbers to agree with to within something.

use scirust_sim::apd::{Apd, ApdParams, excess_noise_factor};
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
    RunSummary, RunWarning, Series, SeriesRole, VerificationResult, VerificationStatus,
    WarningCategory,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
};

/// The id of the axis this capability sweeps. Not `t`: there is no time here.
const GAIN_AXIS_ID: &str = "gain";

/// Elementary charge, in coulombs. Needed to state the shot-noise
/// coefficient `C = 2q*I_p*B` the stationary condition is written in.
/// `scirust-sim` holds its own private copy; this one is compared against it
/// by a test rather than trusted to have been typed identically.
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

/// How much larger than the largest swept SNR the analytic optimum's SNR may
/// fail to be. Pure round-off: the analytic root is the true maximum of a
/// smooth function, so in exact arithmetic the ratio is at most one.
const OPTIMALITY_ROUNDOFF: f64 = 1.0 + 1e-12;

/// Differences in SNR below this fraction of the peak are treated as flat
/// rather than as a turn.
///
/// Right at the optimum the curve is quadratic, so consecutive samples differ
/// by only a few ulps and their *sign* is round-off. Without this the
/// unimodality count would report a handful of spurious turns for a perfectly
/// smooth curve — a measurement artefact, not a property of the model.
const FLATNESS_EPSILON: f64 = 1e-12;

/// Bisection iterations for the stationary condition. Sixty halvings take any
/// bracket to the last bit of an `f64`, so this is "converged", not "close".
const BISECTION_STEPS: usize = 60;

const SWEEP_SOLVER: SolverDescriptor = SolverDescriptor {
    id: "gain_sweep",
    summary: "Evaluates the receiver's closed forms at each of a uniform range of avalanche gains. Nothing is integrated; `solver.start`, `solver.end` and `solver.step` are gains, not times.",
    fixed_step: true,
    adaptive_tolerance: false,
    reports_progress: true,
};

const RESPONSIVITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "responsivity",
    display_name: "Unity-gain responsivity",
    required: true,
    dimension: scirust_units::Dimension::CURRENT.div(scirust_units::Dimension::POWER),
    accepted_units: &["A/W"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Primary photocurrent per watt of incident light, before any avalanche multiplication.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 278),
};

const IONIZATION_RATIO: FieldDescriptor = FieldDescriptor {
    canonical_name: "ionization_ratio",
    display_name: "Ionization ratio k",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Ratio of hole to electron ionization rates. Lower is quieter: k = 0 caps the excess noise near 2, k = 1 makes it grow as the gain itself.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 279),
};

const DARK_CURRENT: FieldDescriptor = FieldDescriptor {
    canonical_name: "dark_current",
    display_name: "Primary dark current",
    required: true,
    dimension: scirust_units::Dimension::CURRENT,
    accepted_units: &["A"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Current flowing with no light on the detector, before multiplication. It is multiplied and shot-noise-generating exactly as the signal is.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 280),
};

const R_LOAD: FieldDescriptor = FieldDescriptor {
    canonical_name: "r_load",
    display_name: "Load resistance",
    required: true,
    dimension: scirust_units::Dimension::RESISTANCE,
    accepted_units: &["Ohm"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Transimpedance load. It sets the thermal noise floor the avalanche gain exists to escape.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 281),
};

const TEMPERATURE: FieldDescriptor = FieldDescriptor {
    canonical_name: "temperature",
    display_name: "Receiver temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Temperature of the load, which sets the Johnson noise.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 282),
};

const BANDWIDTH: FieldDescriptor = FieldDescriptor {
    canonical_name: "bandwidth",
    display_name: "Noise-equivalent bandwidth",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["Hz"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Bandwidth both noise variances are integrated over. It scales them equally, so it moves the SNR without moving the optimal gain.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 283),
};

const OPTICAL_POWER: FieldDescriptor = FieldDescriptor {
    canonical_name: "optical_power",
    display_name: "Incident optical power",
    required: true,
    dimension: scirust_units::Dimension::POWER,
    accepted_units: &["W"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Light on the detector. A weaker signal sits further under the thermal floor, so it wants more gain.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 284),
};

const OPTIMUM_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "optimum_is_where_the_closed_form_says",
    description: "Maximising SNR = A*M^2/(C*M^2*F(M) + T) means minimising C*F(M) + T/M^2, whose stationary condition C*k*M^3 + C*(1-k)*M = 2T has exactly one positive root because its left side is strictly increasing. The SNR at that root must be at least the SNR at every gain the run swept. That is what a maximum is, so the only slack allowed is round-off. Reported as not applicable when the swept range does not contain the root, since then the sweep has not looked where the answer is.",
};

const UNIMODAL_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "snr_is_unimodal_in_gain",
    description: "One root of the stationary condition means the SNR rises then falls and never does anything else, so its difference sequence must change sign exactly once. Differences below a round-off floor are treated as flat, because at the peak itself consecutive samples differ by a few ulps whose sign carries no information. A model with the right optimum but a spurious second bump passes the optimum check and fails this one.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.optoelectronics.avalanche_photodiode"),
    display_name: "Avalanche photodiode gain sweep",
    category: CapabilityCategory::Optoelectronics,
    source_crate: "scirust-sim",
    summary: "An APD receiver's signal, noise and SNR swept across avalanche gain, locating the optimum where excess noise overtakes the thermal floor.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::ParameterSweep,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[SWEEP_SOLVER],
    parameters: &[
        RESPONSIVITY,
        IONIZATION_RATIO,
        DARK_CURRENT,
        R_LOAD,
        TEMPERATURE,
        BANDWIDTH,
        OPTICAL_POWER,
    ],
    // Deliberately empty. There is no state to start from: every point of the
    // sweep is an independent evaluation of closed forms, and a scenario that
    // offered an `[initial_state]` would be describing something this
    // capability does not have.
    initial_state: &[],
    outputs: &[
        OutputDescriptor {
            id: "snr",
            display_name: "Signal-to-noise ratio",
            unit: "1",
            description: "Electrical SNR: multiplied signal power over total noise variance.",
        },
        OutputDescriptor {
            id: "snr_db",
            display_name: "Signal-to-noise ratio in dB",
            unit: "1",
            description: "The same curve in decibels, where the optimum is easiest to see.",
        },
        OutputDescriptor {
            id: "signal_current",
            display_name: "Multiplied signal current",
            unit: "A",
            description: "M*R*P_opt. Grows linearly with gain, forever.",
        },
        OutputDescriptor {
            id: "shot_noise_variance",
            display_name: "Shot noise variance",
            unit: "1",
            description: "2*q*I_primary*M^2*F(M)*B. Grows faster than the signal power, which is what ends the bargain.",
        },
        OutputDescriptor {
            id: "thermal_noise_variance",
            display_name: "Thermal noise variance",
            unit: "1",
            description: "4*kB*T*B/R_L. Flat in gain — the floor the whole exercise is about escaping.",
        },
        OutputDescriptor {
            id: "excess_noise_factor",
            display_name: "Excess noise factor F(M)",
            unit: "1",
            description: "The McIntyre factor: how much noisier than an ideal multiplier the avalanche is.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[OPTIMUM_CHECK, UNIMODAL_CHECK],
    },
};

/// The `sim.optoelectronics.avalanche_photodiode` adapter.
#[derive(Debug, Default)]
pub struct AvalanchePhotodiodeAdapter;

impl CapabilityAdapter for AvalanchePhotodiodeAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &[
                "responsivity",
                "ionization_ratio",
                "dark_current",
                "r_load",
                "temperature",
                "bandwidth",
                "optical_power",
            ],
        ));
        // An empty allow-list, so any `[initial_state]` at all is rejected
        // with the standard "unknown field" error rather than ignored.
        errors.extend(check_unknown_state_fields(scenario, &[]));
        for e in [
            resolve_model_scalar(scenario, &RESPONSIVITY).err(),
            resolve_model_scalar(scenario, &IONIZATION_RATIO).err(),
            resolve_model_scalar(scenario, &DARK_CURRENT).err(),
            resolve_model_scalar(scenario, &R_LOAD).err(),
            resolve_model_scalar(scenario, &TEMPERATURE).err(),
            resolve_model_scalar(scenario, &BANDWIDTH).err(),
            resolve_model_scalar(scenario, &OPTICAL_POWER).err(),
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
        // A gain below one is not a gain, and the model refuses it. Catching
        // it here means the scenario is rejected before anything runs, with
        // the field that is wrong named.
        if let Ok(q) = scenario.solver.start.to_quantity("solver.start")
        {
            if q.value < 1.0
            {
                errors.push(scirust_studio_command::CatalogedError {
                    code: ErrorCode::new(ErrorFamily::Validation, 285),
                    title: "Invalid `solver.start`".to_string(),
                    explanation: format!(
                        "the sweep starts at a gain of {}, but avalanche gain is at least 1 — a \
                         gain below unity would be attenuation, which an APD does not do",
                        q.value
                    ),
                    recoverable: true,
                    suggested_action: Some(
                        "set `solver.start` to 1.0 to begin at the plain-photodiode limit"
                            .to_string(),
                    ),
                });
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
        if control.is_cancelled()
        {
            sink.emit(RunEvent::Cancelled);
            return Err(ExecutionError::Cancelled);
        }
        let s = scenario.scenario();
        let responsivity = resolve_model_scalar(s, &RESPONSIVITY).expect("validated");
        let ionization_ratio = resolve_model_scalar(s, &IONIZATION_RATIO).expect("validated");
        let dark_current = resolve_model_scalar(s, &DARK_CURRENT).expect("validated");
        let r_load = resolve_model_scalar(s, &R_LOAD).expect("validated");
        let temperature = resolve_model_scalar(s, &TEMPERATURE).expect("validated");
        let bandwidth = resolve_model_scalar(s, &BANDWIDTH).expect("validated");
        let optical_power = resolve_model_scalar(s, &OPTICAL_POWER).expect("validated");

        let gain_lo = s
            .solver
            .start
            .to_quantity("solver.start")
            .expect("validated")
            .value;
        let gain_hi = s
            .solver
            .end
            .to_quantity("solver.end")
            .expect("validated")
            .value;
        let gain_step = s
            .solver
            .step
            .as_ref()
            .expect("validated: gain_sweep requires solver.step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let at = |gain: f64| -> Result<Apd, ExecutionError> {
            Apd::new(ApdParams {
                responsivity,
                gain,
                ionization_ratio,
                dark_current,
                r_load,
                temperature,
                bandwidth,
                optical_power,
            })
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))
        };

        // The sweep's coordinates, generated once and then carried through —
        // never regenerated downstream, for the same reason a solver's
        // timestamps are not.
        let points = ((gain_hi - gain_lo) / gain_step).floor() as usize + 1;
        let gains: Vec<f64> = (0..points)
            .map(|i| gain_lo + i as f64 * gain_step)
            .collect();

        let mut snr = Vec::with_capacity(gains.len());
        let mut snr_db = Vec::with_capacity(gains.len());
        let mut signal = Vec::with_capacity(gains.len());
        let mut shot = Vec::with_capacity(gains.len());
        let mut thermal = Vec::with_capacity(gains.len());
        let mut excess = Vec::with_capacity(gains.len());
        for (i, gain) in gains.iter().enumerate()
        {
            if control.is_cancelled()
            {
                sink.emit(RunEvent::Cancelled);
                return Err(ExecutionError::Cancelled);
            }
            let apd = at(*gain)?;
            let value = apd.snr();
            snr.push(value);
            snr_db.push(10.0 * value.max(f64::MIN_POSITIVE).log10());
            signal.push(apd.signal_current());
            shot.push(apd.shot_noise_variance());
            thermal.push(apd.thermal_noise_variance());
            excess.push(apd.excess_noise());
            emit_progress(i, gains.len(), *gain, sink);
        }

        // The stationary condition's coefficients, from the model's own
        // closed forms at unity gain: the primary photocurrent and the
        // thermal variance are both gain-independent.
        let unity = at(1.0)?;
        let shot_coefficient = 2.0 * ELEMENTARY_CHARGE * unity.primary_photocurrent() * bandwidth;
        let thermal_variance = unity.thermal_noise_variance();
        let optimum = stationary_gain(
            shot_coefficient,
            ionization_ratio,
            thermal_variance,
            gain_lo,
            gain_hi,
        );

        let mut warnings = Vec::new();
        if optimum.is_none()
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: format!(
                    "the optimal gain does not lie between {gain_lo} and {gain_hi}, so this sweep \
                     does not contain the answer it was asked for. The curve it did compute is \
                     correct; widen `solver.start`/`solver.end` to bracket the optimum."
                ),
            });
        }

        let optimum_snr = match optimum
        {
            Some(m) => Some(at(m)?.snr()),
            None => None,
        };
        let checks = vec![
            optimum_check(&snr, optimum, optimum_snr),
            unimodality_check(&snr),
        ];

        let mut metrics = vec![Metric {
            id: "thermal_noise_variance".to_string(),
            display_name: "Thermal noise variance".to_string(),
            value: MetricValue::Scalar(thermal_variance),
            unit: Some("1".to_string()),
        }];
        if let (Some(m), Some(value)) = (optimum, optimum_snr)
        {
            metrics.push(Metric {
                id: "optimal_gain".to_string(),
                display_name: "Optimal avalanche gain".to_string(),
                value: MetricValue::Scalar(m),
                unit: Some("1".to_string()),
            });
            metrics.push(Metric {
                id: "peak_snr_db".to_string(),
                display_name: "SNR at the optimal gain".to_string(),
                value: MetricValue::Scalar(10.0 * value.max(f64::MIN_POSITIVE).log10()),
                unit: Some("1".to_string()),
            });
            // What the gain bought: the same receiver with no multiplication.
            let unity_snr = unity.snr();
            metrics.push(Metric {
                id: "gain_over_unity_db".to_string(),
                display_name: "Improvement over unity gain".to_string(),
                value: MetricValue::Scalar(
                    10.0 * (value / unity_snr.max(f64::MIN_POSITIVE))
                        .max(f64::MIN_POSITIVE)
                        .log10(),
                ),
                unit: Some("1".to_string()),
            });
            metrics.push(Metric {
                id: "excess_noise_at_optimum".to_string(),
                display_name: "Excess noise factor at the optimum".to_string(),
                value: MetricValue::Scalar(excess_noise_factor(m, ionization_ratio)),
                unit: Some("1".to_string()),
            });
        }

        let series = |id: &str, display_name: &str, unit: &str, values: Vec<f64>| Series {
            id: id.to_string(),
            display_name: display_name.to_string(),
            unit: unit.to_string(),
            axis_id: GAIN_AXIS_ID.to_string(),
            role: SeriesRole::Trajectory,
            values,
        };

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: GAIN_AXIS_ID.to_string(),
                steps: gains.len() - 1,
                t_start: gains[0],
                t_end: *gains.last().expect("at least one gain"),
            },
            axes: vec![Axis {
                id: GAIN_AXIS_ID.to_string(),
                display_name: "avalanche gain".to_string(),
                unit: "1".to_string(),
                monotonicity: AxisMonotonicity::StrictlyIncreasing,
                values: gains.clone(),
            }],
            series: vec![
                series("snr", "Signal-to-noise ratio", "1", snr.clone()),
                series("snr_db", "Signal-to-noise ratio in dB", "1", snr_db),
                series("signal_current", "Multiplied signal current", "A", signal),
                series("shot_noise_variance", "Shot noise variance", "1", shot),
                series(
                    "thermal_noise_variance",
                    "Thermal noise variance",
                    "1",
                    thermal,
                ),
                series(
                    "excess_noise_factor",
                    "Excess noise factor F(M)",
                    "1",
                    excess,
                ),
            ],
            fields: vec![],
            distributions: vec![],
            metrics,
            warnings,
            verifications: checks,
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

/// Report the sweep's progress on the same schedule an integration would.
fn emit_progress(index: usize, total: usize, gain: f64, sink: &mut dyn EventSink) {
    if total < 2
    {
        return;
    }
    let every = total.div_ceil(crate::execute_support::MAX_PROGRESS_EVENTS);
    if index.is_multiple_of(every) || index + 1 == total
    {
        sink.emit(RunEvent::Progress {
            fraction: (index + 1) as f64 / total as f64,
            t: gain,
        });
    }
}

/// The single positive root of `C*k*M^3 + C*(1-k)*M = 2T`, by bisection,
/// or `None` when it does not lie in `[lo, hi]`.
///
/// The left side is strictly increasing in `M` for every `k` in `[0, 1]` — at
/// `k = 0` it is the line `C*M`, at `k = 1` the cubic `C*M^3`, and in between
/// a positive combination of both — so a sign change across the bracket is
/// proof the root is inside it, and bisection needs neither a derivative nor
/// a starting guess to find it.
fn stationary_gain(
    shot_coefficient: f64,
    ionization_ratio: f64,
    thermal_variance: f64,
    lo: f64,
    hi: f64,
) -> Option<f64> {
    let g = |m: f64| {
        shot_coefficient * ionization_ratio * m * m * m
            + shot_coefficient * (1.0 - ionization_ratio) * m
            - 2.0 * thermal_variance
    };
    let (mut a, mut b) = (lo, hi);
    if !(g(a) < 0.0 && g(b) > 0.0)
    {
        return None;
    }
    for _ in 0..BISECTION_STEPS
    {
        let mid = 0.5 * (a + b);
        if g(mid) < 0.0
        {
            a = mid;
        }
        else
        {
            b = mid;
        }
    }
    Some(0.5 * (a + b))
}

/// The analytic optimum's SNR against every swept gain's.
fn optimum_check(
    snr: &[f64],
    optimum: Option<f64>,
    optimum_snr: Option<f64>,
) -> VerificationResult {
    let id = OPTIMUM_CHECK.id.to_string();
    let (Some(optimum), Some(optimum_snr)) = (optimum, optimum_snr)
    else
    {
        return VerificationResult {
            id,
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: "the stationary condition's root does not lie in the swept range, so \
                          this sweep does not contain the optimum to check against"
                .to_string(),
        };
    };

    let best = snr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ratio = best / optimum_snr.max(f64::MIN_POSITIVE);

    VerificationResult {
        id,
        status: if ratio <= OPTIMALITY_ROUNDOFF
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(ratio),
        threshold: Some(OPTIMALITY_ROUNDOFF),
        explanation: format!(
            "the best SNR anywhere in the sweep was {ratio:.15} times the SNR at the gain of \
             {optimum:.6} that the stationary condition C*k*M^3 + C*(1-k)*M = 2T picks out; \
             anything above one would mean the closed form is not describing the maximum"
        ),
    }
}

/// One turn in the SNR curve, and no more.
fn unimodality_check(snr: &[f64]) -> VerificationResult {
    let id = UNIMODAL_CHECK.id.to_string();
    let peak = snr.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let flat = peak * FLATNESS_EPSILON;

    let mut turns = 0usize;
    let mut previous: Option<bool> = None;
    for pair in snr.windows(2)
    {
        let delta = pair[1] - pair[0];
        if delta.abs() <= flat
        {
            continue;
        }
        let rising = delta > 0.0;
        if previous.is_some_and(|was| was != rising)
        {
            turns += 1;
        }
        previous = Some(rising);
    }

    VerificationResult {
        id,
        status: if turns == 1
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(turns as f64),
        threshold: Some(1.0),
        explanation: format!(
            "the SNR curve changed direction {turns} time(s) across {} swept gains; the \
             stationary condition has exactly one root, so it must rise and then fall and do \
             nothing else",
            snr.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{CollectingEventSink, NullEventSink};
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/avalanche_photodiode.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = AvalanchePhotodiodeAdapter;
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
        let validated = AvalanchePhotodiodeAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let mut sink = CollectingEventSink::new();
        assert!(matches!(
            AvalanchePhotodiodeAdapter.execute(&validated, &control, &mut sink),
            Err(ExecutionError::Cancelled)
        ));
        assert!(sink.events().contains(&RunEvent::Cancelled));
    }

    /// This capability has no time axis, and that is the point of it.
    #[test]
    fn the_run_is_swept_over_gain_and_has_no_time_axis() {
        let r = run();
        assert_eq!(DESCRIPTOR.domain, RunDomain::ParameterSweep);
        assert!(
            r.time_axis().is_none(),
            "a receiver analysis has no time in it, so it must not claim a `t` axis"
        );
        let axis = r.summary_axis().expect("the summary names a real axis");
        assert_eq!(axis.id, GAIN_AXIS_ID);
        assert_eq!(r.summary.t_start, axis.values[0]);
        assert_eq!(r.summary.t_end, *axis.values.last().unwrap());
        for s in &r.series
        {
            assert_eq!(s.axis_id, GAIN_AXIS_ID, "{}", s.id);
            assert_eq!(s.values.len(), axis.values.len(), "{}", s.id);
        }
    }

    #[test]
    fn the_optimum_beats_every_swept_gain() {
        let r = run();
        let v = check(&r, OPTIMUM_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn the_snr_curve_turns_exactly_once() {
        let r = run();
        let v = check(&r, UNIMODAL_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// A sweep whose optimum sits at an endpoint would satisfy both checks
    /// trivially while showing nothing. The tutorial must bracket it properly.
    #[test]
    fn the_tutorial_brackets_its_optimum_with_room_on_both_sides() {
        let r = run();
        let optimum = metric(&r, "optimal_gain");
        let axis = r.summary_axis().expect("gain axis");
        let (lo, hi) = (axis.values[0], *axis.values.last().unwrap());
        let margin = 0.1 * (hi - lo);
        assert!(
            optimum > lo + margin && optimum < hi - margin,
            "the optimum at {optimum} sits within 10 % of an end of the {lo}..{hi} sweep"
        );
    }

    /// The gain has to be worth something, or the curve has no story.
    #[test]
    fn the_avalanche_gain_actually_buys_signal_to_noise() {
        let r = run();
        let improvement = metric(&r, "gain_over_unity_db");
        assert!(
            improvement > 10.0,
            "the optimal gain beat unity by only {improvement:.2} dB, so this receiver is not \
             thermal-limited enough for the trade-off to be visible"
        );
    }

    /// The limit that ties this capability to the plain photodiode: at unity
    /// gain there is no multiplication, so the excess noise factor is exactly
    /// one and the shot noise is the unmultiplied 2*q*I_p*B.
    #[test]
    fn unity_gain_reduces_to_the_plain_photodiode() {
        let r = run();
        let axis = r.summary_axis().expect("gain axis");
        assert_eq!(axis.values[0], 1.0, "the sweep must start at unity gain");

        assert!(
            (series(&r, "excess_noise_factor")[0] - 1.0).abs() < 1e-15,
            "F(1) must be exactly 1 for any k"
        );

        let scenario = parse_toml(TUTORIAL).unwrap();
        let primary = scenario.model["responsivity"].value * scenario.model["optical_power"].value
            + scenario.model["dark_current"].value;
        let expected = 2.0 * ELEMENTARY_CHARGE * primary * scenario.model["bandwidth"].value;
        let got = series(&r, "shot_noise_variance")[0];
        assert!(
            (got - expected).abs() <= expected * 1e-12,
            "unmultiplied shot noise was {got:e}, expected {expected:e}"
        );
    }

    /// The elementary charge this adapter uses to write the stationary
    /// condition must be the one `scirust-sim` computes its shot noise with.
    /// Two constants typed in two places is exactly how a check ends up
    /// agreeing with a model it was supposed to be independent of — or worse,
    /// disagreeing by a digit nobody notices.
    #[test]
    fn the_elementary_charge_matches_the_models_own() {
        let apd = Apd::new(ApdParams {
            responsivity: 1.0,
            gain: 1.0,
            ionization_ratio: 0.0,
            dark_current: 0.0,
            r_load: 1.0,
            temperature: 300.0,
            bandwidth: 1.0,
            optical_power: 1.0,
        })
        .unwrap();
        // At M = 1, k = 0: F = 1, so the variance is 2*q*I_p*B with
        // I_p = responsivity * optical_power = 1 and B = 1.
        let implied = apd.shot_noise_variance() / 2.0;
        assert!(
            (implied - ELEMENTARY_CHARGE).abs() <= ELEMENTARY_CHARGE * 1e-15,
            "the model implies q = {implied:e}, this adapter uses {ELEMENTARY_CHARGE:e}"
        );
    }

    /// A sweep that stops short of the optimum has not answered the question.
    /// Saying so is different from reporting the endpoint as the answer,
    /// which is what an argmax with no analytic check would have done.
    #[test]
    fn a_sweep_that_misses_the_optimum_says_so() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.end.value = 3.0;
        let r = run_scenario(&scenario);
        assert_eq!(
            check(&r, OPTIMUM_CHECK.id).status,
            VerificationStatus::NotApplicable
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.message.contains("optimal gain"))
        );
        // The curve it did compute is still monotone rising, so unimodality
        // has no turn to find and correctly fails rather than passing on a
        // technicality.
        assert_eq!(
            check(&r, UNIMODAL_CHECK.id).status,
            VerificationStatus::Failed
        );
    }

    /// A noisier avalanche (larger k) should want less gain. This is the
    /// design rule the whole model exists to state, and it follows from the
    /// stationary condition without needing a second sweep to confirm.
    #[test]
    fn a_noisier_avalanche_wants_less_gain() {
        let quiet = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario
            .model
            .get_mut("ionization_ratio")
            .expect("ionization_ratio")
            .value = 0.5;
        let noisy = run_scenario(&scenario);
        assert!(
            metric(&noisy, "optimal_gain") < metric(&quiet, "optimal_gain"),
            "k = 0.5 wanted {} against k = 0.02's {}",
            metric(&noisy, "optimal_gain"),
            metric(&quiet, "optimal_gain")
        );
    }

    /// There is no state to start from, so a scenario that supplies one is
    /// describing something else.
    #[test]
    fn an_initial_state_is_rejected_rather_than_ignored() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.insert(
            "photocurrent".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "A".to_string(),
            }],
        );
        assert!(AvalanchePhotodiodeAdapter.validate(&scenario).is_err());
    }

    #[test]
    fn a_sweep_starting_below_unity_gain_is_rejected() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.start.value = 0.5;
        let report = AvalanchePhotodiodeAdapter
            .validate(&scenario)
            .expect_err("gain below 1 is not a gain");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == ErrorCode::new(ErrorFamily::Validation, 285)),
            "{:?}",
            report.errors.iter().map(|e| e.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn bisection_finds_the_root_of_a_hand_checkable_case() {
        // k = 1 reduces the condition to C*M^3 = 2T; with C = 1 and T = 4 the
        // root is 2 exactly.
        let root = stationary_gain(1.0, 1.0, 4.0, 1.0, 10.0).expect("bracketed");
        assert!((root - 2.0).abs() < 1e-12, "{root}");
        // k = 0 reduces it to C*M = 2T; with C = 1 and T = 1.5 the root is 3.
        let root = stationary_gain(1.0, 0.0, 1.5, 1.0, 10.0).expect("bracketed");
        assert!((root - 3.0).abs() < 1e-12, "{root}");
        // Outside the bracket, in either direction, there is no answer.
        assert_eq!(stationary_gain(1.0, 1.0, 4.0, 1.0, 1.5), None);
        assert_eq!(stationary_gain(1.0, 1.0, 4.0, 5.0, 10.0), None);
    }

    #[test]
    fn a_curve_with_two_bumps_fails_unimodality() {
        let verdict = unimodality_check(&[1.0, 2.0, 1.0, 2.0, 1.0]);
        assert_eq!(verdict.status, VerificationStatus::Failed);
        assert_eq!(verdict.measured, Some(3.0));
    }

    /// Consecutive samples at the peak of a smooth curve differ by a few
    /// ulps whose sign is round-off. Those must not be counted as turns, or
    /// every correct sweep would look multimodal.
    #[test]
    fn round_off_wobble_at_the_peak_is_not_a_turn() {
        let mut curve = vec![1.0, 2.0, 3.0];
        // A flat top whose steps are far below the round-off floor.
        for i in 0..8
        {
            curve.push(3.0 + if i % 2 == 0 { 1e-15 } else { -1e-15 });
        }
        curve.extend([2.0, 1.0]);
        let verdict = unimodality_check(&curve);
        assert_eq!(
            verdict.status,
            VerificationStatus::Passed,
            "{}",
            verdict.explanation
        );
    }
}
