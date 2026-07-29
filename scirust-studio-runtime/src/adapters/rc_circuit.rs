//! `sim.electrical.rc` — a capacitor charging through a resistor
//! (`scirust_sim::electrical::RcCircuit`).
//!
//! The first-order system. One state, one time constant, one exponential.
//! It is here because it is the simplest thing in the catalogue that has an
//! exact solution *and* an exact rate, and those two facts fail
//! independently:
//!
//!   * `settles_at_the_source_voltage` — the capacitor ends at `v_source`,
//!     whatever it started from. A wrong source moves this and leaves the
//!     rate alone.
//!   * `matches_the_closed_form` — the approach is `v_source + (v0 -
//!     v_source)*exp(-t/RC)`, checked at every recorded sample. A wrong `R`
//!     or `C` moves the rate and leaves the destination alone.
//!
//! A capability that only compared endpoints would pass the first and never
//! notice the second, which is the mistake `sim.optoelectronics.photodiode`
//! exists to make visible on a physically richer model. This one is the
//! textbook version of the same lesson, and it is cheap.
use scirust_sim::electrical::RcCircuit;
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

/// Relative agreement demanded with the closed form.
///
/// RK4's truncation error and nothing else: the reference is analytic and
/// evaluated at the integrator's own timestamps.
const EXACT_TOLERANCE: f64 = 1e-9;

/// Slack over the residual the run's own length in time constants leaves.
const SETTLE_MARGIN: f64 = 5.0;

const RESISTANCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "resistance",
    display_name: "Resistance",
    required: true,
    dimension: scirust_units::Dimension::RESISTANCE,
    accepted_units: &["Ohm"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Series resistance. With the capacitance it sets the time constant, and nothing else.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 303),
};

const CAPACITANCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "capacitance",
    display_name: "Capacitance",
    required: true,
    dimension: scirust_units::Dimension::CHARGE.div(scirust_units::Dimension::VOLTAGE),
    accepted_units: &["F"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Capacitance being charged.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 304),
};

const V_SOURCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "v_source",
    display_name: "Source voltage",
    required: true,
    dimension: scirust_units::Dimension::VOLTAGE,
    accepted_units: &["V"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The level the capacitor charges toward. It sets the destination and not the rate.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 305),
};

const VOLTAGE: FieldDescriptor = FieldDescriptor {
    canonical_name: "voltage",
    display_name: "Initial capacitor voltage",
    required: true,
    dimension: scirust_units::Dimension::VOLTAGE,
    accepted_units: &["V"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Capacitor voltage at t = start. May be above the source, in which case it discharges.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 306),
};

const EXACT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "matches_the_closed_form",
    description: "The capacitor voltage against v_source + (v0 - v_source)*exp(-t/RC) at every recorded sample. The reference is exact, so the measurement is RK4's truncation error alone. A wrong R or C moves this and leaves the settled level untouched.",
};

const SETTLE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "settles_at_the_source_voltage",
    description: "The capacitor must end at the source voltage whatever it started from. The threshold is derived from the run's own length in time constants: after a span the exact solution still holds exp(-span/RC) of its initial deviation, and no correct run can beat that. A wrong source voltage moves this and leaves the rate untouched.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.electrical.rc"),
    display_name: "RC charging circuit",
    category: CapabilityCategory::Electrical,
    source_crate: "scirust-sim",
    summary: "A capacitor charging through a resistor: one state, one time constant, and a level and a rate that fail independently.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[RESISTANCE, CAPACITANCE, V_SOURCE],
    initial_state: &[VOLTAGE],
    outputs: &[
        OutputDescriptor {
            id: "voltage",
            display_name: "Capacitor voltage",
            unit: "V",
            description: "What a voltmeter across the capacitor reads.",
        },
        OutputDescriptor {
            id: "exact_voltage",
            display_name: "Exact capacitor voltage",
            unit: "V",
            description: "The closed form, for comparison.",
        },
        OutputDescriptor {
            id: "source_voltage",
            display_name: "Source voltage",
            unit: "V",
            description: "The level being charged toward.",
        },
        OutputDescriptor {
            id: "current",
            display_name: "Charging current",
            unit: "A",
            description: "(v_source - v)/R, largest at the start and decaying with the voltage gap.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[EXACT_CHECK, SETTLE_CHECK],
    },
};

/// The `sim.electrical.rc` adapter.
#[derive(Debug, Default)]
pub struct RcCircuitAdapter;

impl CapabilityAdapter for RcCircuitAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["resistance", "capacitance", "v_source"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["voltage"]));
        for e in [
            resolve_model_scalar(scenario, &RESISTANCE).err(),
            resolve_model_scalar(scenario, &CAPACITANCE).err(),
            resolve_model_scalar(scenario, &V_SOURCE).err(),
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
        let resistance = resolve_model_scalar(s, &RESISTANCE).expect("validated");
        let v_source = resolve_model_scalar(s, &V_SOURCE).expect("validated");
        let model = RcCircuit::new(
            resistance,
            resolve_model_scalar(s, &CAPACITANCE).expect("validated"),
            v_source,
        )
        .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
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

        let voltage = traj.column(0).expect("dim 0");
        let exact: Vec<f64> = traj
            .t
            .iter()
            .map(|t| model.exact(v0, *t).expect("validated: v0 is finite"))
            .collect();
        let current: Vec<f64> = voltage
            .iter()
            .map(|v| (v_source - v) / resistance)
            .collect();

        let tau = model.time_constant();
        let span = *traj.t.last().expect("at least one sample") - traj.t[0];
        let exact_check = closed_form_check(&voltage, &exact, (v0 - v_source).abs());
        let settle = settle_check(
            voltage.last().copied().unwrap_or(f64::NAN),
            v_source,
            (v0 - v_source).abs(),
            tau,
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
                    id: "voltage".to_string(),
                    display_name: "Capacitor voltage".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: voltage.clone(),
                },
                Series {
                    id: "exact_voltage".to_string(),
                    display_name: "Exact capacitor voltage".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact.clone(),
                },
                Series {
                    id: "source_voltage".to_string(),
                    display_name: "Source voltage".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![v_source; traj.t.len()],
                },
                Series {
                    id: "current".to_string(),
                    display_name: "Charging current".to_string(),
                    unit: "A".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: current,
                },
            ],
            fields: vec![],
            metrics: vec![
                Metric {
                    id: "time_constant".to_string(),
                    display_name: "Time constant R*C".to_string(),
                    value: MetricValue::Scalar(tau),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "span_in_time_constants".to_string(),
                    display_name: "Run length in time constants".to_string(),
                    value: MetricValue::Scalar(span / tau),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_voltage".to_string(),
                    display_name: "Final capacitor voltage".to_string(),
                    value: MetricValue::Scalar(voltage.last().copied().unwrap_or(f64::NAN)),
                    unit: Some("V".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![exact_check, settle],
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

/// The trajectory against the closed form at every sample.
fn closed_form_check(voltage: &[f64], exact: &[f64], scale: f64) -> VerificationResult {
    let scale = scale.abs().max(f64::MIN_POSITIVE);
    let worst = voltage
        .iter()
        .zip(exact.iter())
        .map(|(g, x)| (g - x).abs() / scale)
        .fold(0.0_f64, f64::max);
    VerificationResult {
        id: EXACT_CHECK.id.to_string(),
        status: if worst <= EXACT_TOLERANCE
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(worst),
        threshold: Some(EXACT_TOLERANCE),
        explanation: format!(
            "the integrated voltage tracked v_source + (v0 - v_source)*exp(-t/RC) to within \
             {worst:.3e} of the initial gap, across all {} samples",
            voltage.len()
        ),
    }
}

/// The settled level against the source voltage.
fn settle_check(
    final_voltage: f64,
    v_source: f64,
    initial_gap: f64,
    tau: f64,
    span: f64,
) -> VerificationResult {
    let residual = if tau > 0.0 && span.is_finite()
    {
        (-span / tau).exp()
    }
    else
    {
        1.0
    };
    let scale = initial_gap.max(f64::MIN_POSITIVE);
    let error = (final_voltage - v_source).abs() / scale;
    let threshold = residual * SETTLE_MARGIN;
    VerificationResult {
        id: SETTLE_CHECK.id.to_string(),
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
            "the capacitor ended {error:.3e} of its initial gap away from the {v_source:.6} V \
             source, against the {residual:.3e} that the run's own length of {:.2} time \
             constants still leaves",
            span / tau
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/rc_circuit.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = RcCircuitAdapter;
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
        let validated = RcCircuitAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            RcCircuitAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn it_matches_the_closed_form() {
        let r = run();
        let v = check(&r, EXACT_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn it_settles_at_the_source_voltage() {
        let r = run();
        let v = check(&r, SETTLE_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The two checks must be able to move independently, which is the whole
    /// argument for having both. Halving the capacitance halves the time
    /// constant; the destination does not move at all.
    ///
    /// "Does not move" is stated carefully. The two runs do *not* end on the
    /// same number, because the same wall-clock span is eight time constants
    /// for one and sixteen for the other, so each is a different distance
    /// short of the same 5 V. What is asserted is that each lands within what
    /// its own length allows, and that the one with more time constants lands
    /// closer. A source voltage that had actually moved would break both.
    #[test]
    fn the_rate_and_the_level_are_independent() {
        let base = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario
            .model
            .get_mut("capacitance")
            .expect("capacitance")
            .value /= 2.0;
        let faster = run_scenario(&scenario);

        let (tau, tau_fast) = (
            metric(&base, "time_constant"),
            metric(&faster, "time_constant"),
        );
        assert!(
            (tau_fast / tau - 0.5).abs() < 1e-12,
            "the time constant went {tau} -> {tau_fast}, expected a halving"
        );

        let tutorial = parse_toml(TUTORIAL).unwrap();
        let source = tutorial.model["v_source"].value;
        let gap = (tutorial.initial_state["voltage"][0].value - source).abs();
        for (label, r) in [("base", &base), ("faster", &faster)]
        {
            let residual = (-metric(r, "span_in_time_constants")).exp();
            let miss = (metric(r, "final_voltage") - source).abs();
            assert!(
                miss <= gap * residual * 2.0,
                "{label} ended {miss} V from the source, more than its own {residual:e} \
                 residual on a {gap} V gap allows"
            );
        }

        assert!(
            (metric(&faster, "final_voltage") - source).abs()
                < (metric(&base, "final_voltage") - source).abs(),
            "the run with twice as many time constants did not land closer to the same source"
        );
    }
}
