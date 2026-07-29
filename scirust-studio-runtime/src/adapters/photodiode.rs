//! `sim.optoelectronics.photodiode` — a photodiode receiver
//! (`scirust_sim::photodiode::Photodiode`).
//!
//! Incident optical power becomes a photocurrent through the diode\'s spectral
//! responsivity, and that current charges the junction capacitance through the
//! load resistance: a first-order RC low-pass.
//!
//! # Why this one is worth verifying
//!
//! The model is linear and first-order, so its solution is a closed form:
//! `v(t) = v_ss + (v_0 - v_ss)*exp(-t/tau)` with `v_ss = I_ph*R_L` and
//! `tau = R_L*C_j`. Both the level and the rate are stated by the model, so a
//! run can be checked against *both* rather than against a plausible-looking
//! curve. The two are independent: a wrong responsivity moves the level and
//! leaves the rate, a wrong capacitance does the reverse.

use scirust_sim::photodiode::{Photodiode, PhotodiodeParams};

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
use crate::validate_support::RK4_SOLVER;
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// How much slack the settled-level check allows on top of the residual the
/// run's own length implies.
///
/// The threshold itself is *derived*, not chosen: a first-order approach
/// leaves exactly `exp(-span/tau)` of its initial deviation after a span, so
/// a run of five time constants cannot land closer than 6.7e-3 however
/// correct it is. Comparing against a fixed number would either fail correct
/// short runs or pass wrong long ones. This factor covers RK4's own error on
/// top of that exact residual.
const LEVEL_MARGIN: f64 = 4.0;

/// Relative tolerance on the measured time constant against `R_L*C_j`.
///
/// Looser than the level check because it is a fit from two samples rather
/// than a reading of the final value, and because it is measured over a
/// window where the exponential has not fully decayed.
const TAU_TOLERANCE: f64 = 0.05;

const RESPONSIVITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "responsivity",
    display_name: "Responsivity",
    required: true,
    dimension: scirust_units::Dimension::CURRENT.div(scirust_units::Dimension::POWER),
    accepted_units: &["A/W"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Photocurrent per watt of incident optical power. A silicon diode at 850 nm is \
                  around 0.5 A/W.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 210),
};

const DARK_CURRENT: FieldDescriptor = FieldDescriptor {
    canonical_name: "dark_current",
    display_name: "Dark current",
    required: true,
    dimension: scirust_units::Dimension::CURRENT,
    accepted_units: &["A"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Current that flows with no light on the diode.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 211),
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
    description: "The transimpedance load. Raising it raises the signal and lowers the \
                  bandwidth, in exact proportion.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 212),
};

const C_JUNCTION: FieldDescriptor = FieldDescriptor {
    canonical_name: "c_junction",
    display_name: "Junction capacitance",
    required: true,
    dimension: scirust_units::Dimension::CHARGE.div(scirust_units::Dimension::VOLTAGE),
    accepted_units: &["F"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Depletion capacitance of the junction.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 213),
};

const OPTICAL_POWER: FieldDescriptor = FieldDescriptor {
    canonical_name: "optical_power",
    display_name: "Optical power",
    required: true,
    dimension: scirust_units::Dimension::POWER,
    accepted_units: &["W"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Incident optical power, held constant for the run.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 214),
};

const VOLTAGE: FieldDescriptor = FieldDescriptor {
    canonical_name: "voltage",
    display_name: "Initial load voltage",
    required: true,
    dimension: scirust_units::Dimension::VOLTAGE,
    accepted_units: &["V"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The load voltage at t = start, usually zero.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 215),
};

const LEVEL_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "settles_at_the_closed_form_level",
    description: "The settled load voltage must equal I_ph*R_L, where the photocurrent is \
                  responsivity*P_opt + I_dark. A level error means the conversion is wrong; it \
                  is independent of the rate check below.",
};

const TAU_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "rc_time_constant",
    description: "The approach to that level is a single exponential with time constant \
                  tau = R_L*C_j. The run measures its own decay and compares against that \
                  closed form, which is what distinguishes a correct RC from any curve that \
                  happens to end in the right place.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.optoelectronics.photodiode"),
    display_name: "Photodiode receiver",
    category: CapabilityCategory::Optoelectronics,
    source_crate: "scirust-sim",
    summary: "A photodiode converting optical power to a load voltage through its junction \
              capacitance — a first-order RC whose level and time constant are both closed form.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[
        RESPONSIVITY,
        DARK_CURRENT,
        R_LOAD,
        C_JUNCTION,
        OPTICAL_POWER,
    ],
    initial_state: &[VOLTAGE],
    outputs: &[
        OutputDescriptor {
            id: "voltage",
            display_name: "Load voltage",
            unit: "V",
            description: "The voltage across the load, over time.",
        },
        OutputDescriptor {
            id: "steady_state",
            display_name: "Settled voltage",
            unit: "V",
            description: "The closed-form level I_ph*R_L, for comparison.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[LEVEL_CHECK, TAU_CHECK],
    },
};

/// The `sim.optoelectronics.photodiode` adapter.
#[derive(Debug, Default)]
pub struct PhotodiodeAdapter;

fn build(scenario: &Scenario) -> Result<Photodiode, String> {
    Photodiode::new(PhotodiodeParams {
        responsivity: resolve_model_scalar(scenario, &RESPONSIVITY).map_err(|e| e.to_string())?,
        dark_current: resolve_model_scalar(scenario, &DARK_CURRENT).map_err(|e| e.to_string())?,
        r_load: resolve_model_scalar(scenario, &R_LOAD).map_err(|e| e.to_string())?,
        c_junction: resolve_model_scalar(scenario, &C_JUNCTION).map_err(|e| e.to_string())?,
        optical_power: resolve_model_scalar(scenario, &OPTICAL_POWER).map_err(|e| e.to_string())?,
    })
    .map_err(|e| e.to_string())
}

impl CapabilityAdapter for PhotodiodeAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &[
                "responsivity",
                "dark_current",
                "r_load",
                "c_junction",
                "optical_power",
            ],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["voltage"]));
        for e in [
            resolve_model_scalar(scenario, &RESPONSIVITY).err(),
            resolve_model_scalar(scenario, &DARK_CURRENT).err(),
            resolve_model_scalar(scenario, &R_LOAD).err(),
            resolve_model_scalar(scenario, &C_JUNCTION).err(),
            resolve_model_scalar(scenario, &OPTICAL_POWER).err(),
            resolve_state_vector(scenario, &VOLTAGE).err(),
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
        let model = build(s).map_err(ExecutionError::InvalidModelState)?;
        let v0 = resolve_state_vector(s, &VOLTAGE).expect("validated")[0];

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
            &[v0],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let voltage = traj.column(0).expect("dim 0 exists");
        let settled = model.steady_state_voltage();
        let tau = model.time_constant();

        let span = *traj.t.last().expect("at least one sample") - traj.t[0];
        let level = level_check(&voltage, settled, v0, span, tau);
        let rate = tau_check(&traj.t, &voltage, settled, v0, tau);

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
                    id: "voltage".to_string(),
                    display_name: "Load voltage".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: voltage.clone(),
                },
                Series {
                    id: "steady_state".to_string(),
                    display_name: "Settled voltage".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![settled; traj.t.len()],
                },
            ],
            fields: vec![],
            metrics: vec![
                Metric {
                    id: "photocurrent".to_string(),
                    display_name: "Photocurrent".to_string(),
                    value: MetricValue::Scalar(model.photocurrent()),
                    unit: Some("A".to_string()),
                },
                Metric {
                    id: "settled_voltage".to_string(),
                    display_name: "Settled voltage I_ph*R_L".to_string(),
                    value: MetricValue::Scalar(settled),
                    unit: Some("V".to_string()),
                },
                Metric {
                    id: "time_constant".to_string(),
                    display_name: "Time constant R_L*C_j".to_string(),
                    value: MetricValue::Scalar(tau),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "bandwidth".to_string(),
                    display_name: "Bandwidth 1/(2*pi*tau)".to_string(),
                    value: MetricValue::Scalar(model.bandwidth()),
                    unit: Some("Hz".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![level, rate],
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

/// Did the run settle where the closed form says, to the accuracy its own
/// length allows?
///
/// The threshold is derived from the span: after `span` the exact solution
/// still holds `exp(-span/tau)` of its initial deviation, so that residual is
/// the floor no correct run can beat. Measuring against a fixed number would
/// be asking a five-time-constant run to be something it cannot be.
fn level_check(voltage: &[f64], settled: f64, v0: f64, span: f64, tau: f64) -> VerificationResult {
    let final_v = *voltage.last().unwrap_or(&0.0);
    let scale = (v0 - settled)
        .abs()
        .max(settled.abs())
        .max(f64::MIN_POSITIVE);
    let relative = (final_v - settled).abs() / scale;

    let residual = if tau > 0.0 && span.is_finite()
    {
        (-span / tau).exp()
    }
    else
    {
        1.0
    };
    let threshold = residual * LEVEL_MARGIN;

    VerificationResult {
        id: "settles_at_the_closed_form_level".to_string(),
        status: if relative <= threshold
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(relative),
        threshold: Some(threshold),
        explanation: format!(
            "the load voltage ended at {final_v:.9} V against the closed-form \
             I_ph*R_L = {settled:.9} V — {relative:.3e} of the initial deviation, against the \
             {residual:.3e} the run's own length of {:.2} time constants still leaves",
            span / tau
        ),
    }
}

/// Does it approach that level at the rate the closed form says?
///
/// Measured from the *deviation* rather than the voltage, because the
/// deviation is the pure exponential: `v - v_ss = (v_0 - v_ss)*exp(-t/tau)`.
fn tau_check(t: &[f64], voltage: &[f64], settled: f64, v0: f64, tau: f64) -> VerificationResult {
    let id = "rc_time_constant".to_string();
    let not_applicable = |explanation: String| VerificationResult {
        id: id.clone(),
        status: VerificationStatus::NotApplicable,
        measured: None,
        threshold: None,
        explanation,
    };

    if (v0 - settled).abs() == 0.0
    {
        return not_applicable(
            "the run started already settled, so there is no approach to measure".to_string(),
        );
    }

    // One time constant in, the deviation has fallen to 1/e. Measuring there
    // rather than at the end keeps the sample well above the floor where the
    // logarithm is noise.
    let target = t[0] + tau;
    let Some(index) = t.iter().position(|&x| x >= target)
    else
    {
        return not_applicable(format!(
            "the run is shorter than one time constant ({tau:.3e} s), so the decay rate \
             cannot be measured"
        ));
    };

    let d0 = (voltage[0] - settled).abs();
    let d1 = (voltage[index] - settled).abs();
    if d1 <= 0.0 || d1 / d0 < 1e-12
    {
        return not_applicable(
            "the deviation reached the floor of what this measurement can resolve".to_string(),
        );
    }

    let measured = (t[index] - t[0]) / (d0 / d1).ln();
    let ratio = measured / tau;
    VerificationResult {
        id,
        status: if (ratio - 1.0).abs() <= TAU_TOLERANCE
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(ratio),
        threshold: Some(1.0 + TAU_TOLERANCE),
        explanation: format!(
            "the deviation from the settled level decayed with a measured time constant of \
             {measured:.6e} s against the closed-form R_L*C_j = {tau:.6e} s (ratio {ratio:.4})"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/photodiode.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = PhotodiodeAdapter;
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
    fn it_settles_at_the_closed_form_level() {
        let r = run();
        let v = check(&r, "settles_at_the_closed_form_level");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn it_approaches_at_the_closed_form_rate() {
        let r = run();
        let v = check(&r, "rc_time_constant");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The two checks are independent, and this is what makes the pair worth
    /// having: a level that is right says nothing about the rate.
    #[test]
    fn the_rate_check_catches_a_curve_that_ends_in_the_right_place() {
        let t: Vec<f64> = (0..101).map(|i| i as f64 * 1e-9).collect();
        // Settles at the right level, but four times too fast.
        let v: Vec<f64> = t.iter().map(|x| 1.0 - (-x / 25e-9).exp()).collect();
        let verdict = tau_check(&t, &v, 1.0, 0.0, 100e-9);
        assert_eq!(
            verdict.status,
            VerificationStatus::Failed,
            "{}",
            verdict.explanation
        );
    }

    /// A level that is simply wrong must fail even on a long run, where the
    /// derived threshold is at its tightest.
    #[test]
    fn the_level_check_catches_a_wrong_conversion() {
        // Ten time constants: the exact residual is 4.5e-5, so ending at half
        // the right level is four orders of magnitude out.
        let verdict = level_check(&[0.0, 0.5], 1.0, 0.0, 10.0, 1.0);
        assert_eq!(
            verdict.status,
            VerificationStatus::Failed,
            "{}",
            verdict.explanation
        );
    }

    /// …and a correct run must pass at *any* length, which a fixed tolerance
    /// could not deliver. Two time constants leaves 13.5% of the deviation;
    /// that is the right answer, not a failure.
    #[test]
    fn the_level_threshold_follows_the_runs_own_length() {
        let short = level_check(&[0.0, 1.0 - (-2.0f64).exp()], 1.0, 0.0, 2.0, 1.0);
        assert_eq!(
            short.status,
            VerificationStatus::Passed,
            "{}",
            short.explanation
        );
        let long = level_check(&[0.0, 1.0 - (-9.0f64).exp()], 1.0, 0.0, 9.0, 1.0);
        assert_eq!(
            long.status,
            VerificationStatus::Passed,
            "{}",
            long.explanation
        );
        // The long run is held to a far tighter standard than the short one.
        assert!(long.threshold.unwrap() < short.threshold.unwrap() / 100.0);
    }

    #[test]
    fn a_run_that_starts_settled_declines_the_rate_check() {
        let verdict = tau_check(&[0.0, 1.0], &[1.0, 1.0], 1.0, 1.0, 1.0);
        assert_eq!(verdict.status, VerificationStatus::NotApplicable);
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
        let adapter = PhotodiodeAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            adapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }
}
