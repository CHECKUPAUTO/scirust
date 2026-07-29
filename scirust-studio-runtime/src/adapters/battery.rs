//! `sim.energy.battery_thevenin` — a lithium-ion cell under constant current
//! (`scirust_sim::battery::TheveninBattery`).
//!
//! The Thevenin equivalent circuit — an open-circuit voltage source, a series
//! resistance and one RC polarisation branch — coupled to a lumped thermal
//! state that heats up on its own losses.
//!
//! # Three states, three different kinds of check
//!
//! The state of charge is **exact**: coulomb counting under constant current
//! is a straight line, `soc(t) = soc0 - I*t/(capacity*3600)`, and the model
//! states it in closed form. So that one is checked against the analytic
//! answer, not an asymptote.
//!
//! The polarisation overpotential and the temperature are both **first-order
//! approaches** to closed-form levels, `I*R1` and the self-heating
//! equilibrium. Their thresholds are derived from the run's own length in
//! time constants, the same way the photodiode's is.
//!
//! Checking all three matters because they fail independently: a wrong
//! capacity moves the SoC line and leaves the others, a wrong `R1` moves the
//! overpotential, and a wrong thermal resistance moves only the temperature.
//!
//! # Capacity is in coulombs, not amp-hours
//!
//! The scenario unit table holds only symbols whose SI conversion factor is
//! exactly 1, deliberately — an entry carrying a hidden rescale is the kind
//! of thing that produces an answer wrong by 3600 and no error message. So
//! this capability takes capacity in coulombs and converts, in the open, to
//! the amp-hours the model wants. A 2.5 A*h cell is 9000 C.

use scirust_sim::battery::{BatteryParams, TheveninBattery};
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

/// Relative error allowed on the exact coulomb-counting line.
const SOC_TOLERANCE: f64 = 1e-9;

/// Slack on the two first-order approaches, over the residual the run's own
/// length implies.
const SETTLE_MARGIN: f64 = 5.0;

/// Seconds in an hour, for the coulombs-to-amp-hours conversion.
const SECONDS_PER_HOUR: f64 = 3600.0;

const CAPACITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "capacity",
    display_name: "Cell capacity",
    required: true,
    dimension: scirust_units::Dimension::CHARGE,
    accepted_units: &["C"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Charge the cell holds, in coulombs. A 2.5 A*h cell is 9000 C — the unit table carries no hidden conversions, so the amp-hour is spelled out.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 250),
};

const R0: FieldDescriptor = FieldDescriptor {
    canonical_name: "r0",
    display_name: "Series resistance",
    required: true,
    dimension: scirust_units::Dimension::RESISTANCE,
    accepted_units: &["Ohm"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Ohmic resistance: the instantaneous voltage step on load.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 251),
};

const R1: FieldDescriptor = FieldDescriptor {
    canonical_name: "r1",
    display_name: "Polarisation resistance",
    required: true,
    dimension: scirust_units::Dimension::RESISTANCE,
    accepted_units: &["Ohm"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Resistance of the RC branch. The overpotential relaxes to I*R1.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 252),
};

const C1: FieldDescriptor = FieldDescriptor {
    canonical_name: "c1",
    display_name: "Polarisation capacitance",
    required: true,
    dimension: scirust_units::Dimension::CHARGE.div(scirust_units::Dimension::VOLTAGE),
    accepted_units: &["F"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Capacitance of the RC branch. R1*C1 is the polarisation time constant.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 253),
};

const CURRENT: FieldDescriptor = FieldDescriptor {
    canonical_name: "current",
    display_name: "Discharge current",
    required: true,
    dimension: scirust_units::Dimension::CURRENT,
    accepted_units: &["A"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Constant current, positive on discharge.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 254),
};

const OCV_MIN: FieldDescriptor = FieldDescriptor {
    canonical_name: "ocv_min",
    display_name: "Open-circuit voltage at empty",
    required: true,
    dimension: scirust_units::Dimension::VOLTAGE,
    accepted_units: &["V"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Open-circuit voltage at SoC = 0.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 255),
};

const OCV_MAX: FieldDescriptor = FieldDescriptor {
    canonical_name: "ocv_max",
    display_name: "Open-circuit voltage at full",
    required: true,
    dimension: scirust_units::Dimension::VOLTAGE,
    accepted_units: &["V"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Open-circuit voltage at SoC = 1.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 256),
};

const R_TH: FieldDescriptor = FieldDescriptor {
    canonical_name: "r_th",
    display_name: "Thermal resistance",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE.div(scirust_units::Dimension::POWER),
    accepted_units: &["K/W"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Cell-to-ambient thermal resistance.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 257),
};

const C_TH: FieldDescriptor = FieldDescriptor {
    canonical_name: "c_th",
    display_name: "Thermal capacitance",
    required: true,
    dimension: scirust_units::Dimension::ENERGY.div(scirust_units::Dimension::TEMPERATURE),
    accepted_units: &["J/K"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Lumped heat capacity of the cell.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 258),
};

const T_AMBIENT: FieldDescriptor = FieldDescriptor {
    canonical_name: "t_ambient",
    display_name: "Ambient temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Surroundings the cell rejects heat to.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 259),
};

const SOC: FieldDescriptor = FieldDescriptor {
    canonical_name: "soc",
    display_name: "Initial state of charge",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "State of charge at t = start, in [0, 1].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 260),
};

const TEMPERATURE: FieldDescriptor = FieldDescriptor {
    canonical_name: "temperature",
    display_name: "Initial cell temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Cell temperature at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 261),
};

const COULOMB_COUNTING_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "coulomb_counting",
    description: "Under constant current the state of charge is a straight line the model states in closed form, so it is checked against the exact answer rather than an asymptote. A wrong capacity moves this line and leaves the other two states alone.",
};

const RELAXES_TO_CLOSED_FORM_LEVELS_CHECK: VerificationCheckDescriptor =
    VerificationCheckDescriptor {
        id: "relaxes_to_closed_form_levels",
        description: "The polarisation overpotential approaches I*R1 and the cell temperature approaches its self-heating equilibrium, both first-order. The threshold is derived from the run's own length in time constants. The two fail independently of each other and of the SoC line, which is why all three are checked.",
    };

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.energy.battery_thevenin"),
    display_name: "Lithium-ion cell under constant current",
    category: CapabilityCategory::Energy,
    source_crate: "scirust-sim",
    summary: "A Thevenin equivalent-circuit cell with one RC polarisation branch and lumped self-heating, discharged at constant current.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[
        CAPACITY, R0, R1, C1, CURRENT, OCV_MIN, OCV_MAX, R_TH, C_TH, T_AMBIENT,
    ],
    initial_state: &[SOC, TEMPERATURE],
    outputs: &[
        OutputDescriptor {
            id: "soc",
            display_name: "State of charge",
            unit: "1",
            description: "Charge remaining, as a fraction.",
        },
        OutputDescriptor {
            id: "overpotential",
            display_name: "Polarisation overpotential",
            unit: "V",
            description: "Voltage across the RC branch.",
        },
        OutputDescriptor {
            id: "temperature",
            display_name: "Cell temperature",
            unit: "K",
            description: "Cell temperature under self-heating.",
        },
        OutputDescriptor {
            id: "terminal_voltage",
            display_name: "Terminal voltage",
            unit: "V",
            description: "What a voltmeter across the terminals would read.",
        },
        OutputDescriptor {
            id: "exact_soc",
            display_name: "Exact state of charge",
            unit: "1",
            description: "The closed-form coulomb-counting line, for comparison.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[COULOMB_COUNTING_CHECK, RELAXES_TO_CLOSED_FORM_LEVELS_CHECK],
    },
};

/// The `sim.energy.battery_thevenin` adapter.
#[derive(Debug, Default)]
pub struct BatteryAdapter;

impl CapabilityAdapter for BatteryAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &[
                "capacity",
                "r0",
                "r1",
                "c1",
                "current",
                "ocv_min",
                "ocv_max",
                "r_th",
                "c_th",
                "t_ambient",
            ],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["soc", "temperature"],
        ));
        for e in [
            resolve_model_scalar(scenario, &CAPACITY).err(),
            resolve_model_scalar(scenario, &R0).err(),
            resolve_model_scalar(scenario, &R1).err(),
            resolve_model_scalar(scenario, &C1).err(),
            resolve_model_scalar(scenario, &CURRENT).err(),
            resolve_model_scalar(scenario, &OCV_MIN).err(),
            resolve_model_scalar(scenario, &OCV_MAX).err(),
            resolve_model_scalar(scenario, &R_TH).err(),
            resolve_model_scalar(scenario, &C_TH).err(),
            resolve_model_scalar(scenario, &T_AMBIENT).err(),
            resolve_state_vector(scenario, &SOC).err(),
            resolve_state_vector(scenario, &TEMPERATURE).err(),
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
        let coulombs = resolve_model_scalar(s, &CAPACITY).expect("validated");
        let r1 = resolve_model_scalar(s, &R1).expect("validated");
        let c1 = resolve_model_scalar(s, &C1).expect("validated");
        let r_th = resolve_model_scalar(s, &R_TH).expect("validated");
        let c_th = resolve_model_scalar(s, &C_TH).expect("validated");
        let soc0 = resolve_state_vector(s, &SOC).expect("validated")[0];
        let temp0 = resolve_state_vector(s, &TEMPERATURE).expect("validated")[0];

        let model = TheveninBattery::new(BatteryParams {
            // In the open: the scenario says coulombs, the model wants
            // amp-hours. See this module's header for why the unit table
            // does not carry the factor.
            capacity_ah: coulombs / SECONDS_PER_HOUR,
            r0: resolve_model_scalar(s, &R0).expect("validated"),
            r1,
            c1,
            current: resolve_model_scalar(s, &CURRENT).expect("validated"),
            ocv_min: resolve_model_scalar(s, &OCV_MIN).expect("validated"),
            ocv_max: resolve_model_scalar(s, &OCV_MAX).expect("validated"),
            r_th,
            c_th,
            t_ambient: resolve_model_scalar(s, &T_AMBIENT).expect("validated"),
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

        let y0 = model.initial_state(soc0, temp0);
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

        let soc = traj.column(0).expect("dim 0 exists");
        let v_rc = traj.column(1).expect("dim 1 exists");
        let temperature = traj.column(2).expect("dim 2 exists");
        let terminal: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.terminal_voltage(row).expect("dim 3"))
            .collect();
        let exact_soc: Vec<f64> = traj.t.iter().map(|t| model.soc_at(soc0, *t)).collect();

        let span = *traj.t.last().expect("at least one sample") - traj.t[0];
        let counting = coulomb_check(&soc, &exact_soc);
        let settle = settle_check(
            &v_rc,
            &temperature,
            model.steady_state_overpotential(),
            model.steady_state_temperature(),
            temp0,
            span,
            model.polarization_time_constant(),
            r_th * c_th,
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
                    id: "soc".to_string(),
                    display_name: "State of charge".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: soc.clone(),
                },
                Series {
                    id: "overpotential".to_string(),
                    display_name: "Polarisation overpotential".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: v_rc.clone(),
                },
                Series {
                    id: "temperature".to_string(),
                    display_name: "Cell temperature".to_string(),
                    unit: "K".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: temperature.clone(),
                },
                Series {
                    id: "terminal_voltage".to_string(),
                    display_name: "Terminal voltage".to_string(),
                    unit: "V".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: terminal.clone(),
                },
                Series {
                    id: "exact_soc".to_string(),
                    display_name: "Exact state of charge".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact_soc.clone(),
                },
            ],
            fields: vec![],
            metrics: vec![
                Metric {
                    id: "polarization_time_constant".to_string(),
                    display_name: "Polarisation time constant R1*C1".to_string(),
                    value: MetricValue::Scalar(model.polarization_time_constant()),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "thermal_time_constant".to_string(),
                    display_name: "Thermal time constant R_th*C_th".to_string(),
                    value: MetricValue::Scalar(r_th * c_th),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "steady_overpotential".to_string(),
                    display_name: "Settled overpotential I*R1".to_string(),
                    value: MetricValue::Scalar(model.steady_state_overpotential()),
                    unit: Some("V".to_string()),
                },
                Metric {
                    id: "steady_temperature".to_string(),
                    display_name: "Settled temperature".to_string(),
                    value: MetricValue::Scalar(model.steady_state_temperature()),
                    unit: Some("K".to_string()),
                },
                Metric {
                    id: "final_terminal_voltage".to_string(),
                    display_name: "Terminal voltage at the end".to_string(),
                    value: MetricValue::Scalar(terminal.last().copied().unwrap_or(f64::NAN)),
                    unit: Some("V".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![counting, settle],
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

/// The state of charge against its exact line.
fn coulomb_check(soc: &[f64], exact: &[f64]) -> VerificationResult {
    let worst = soc
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    VerificationResult {
        id: "coulomb_counting".to_string(),
        status: if worst <= SOC_TOLERANCE
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(worst),
        threshold: Some(SOC_TOLERANCE),
        explanation: format!(
            "the integrated state of charge tracks the exact coulomb-counting line \
             soc0 - I*t/(capacity*3600) to within {worst:.3e} across all {} samples",
            soc.len()
        ),
    }
}

/// The two first-order states against their closed-form levels.
#[allow(clippy::too_many_arguments)]
fn settle_check(
    v_rc: &[f64],
    temperature: &[f64],
    v_target: f64,
    t_target: f64,
    temp0: f64,
    span: f64,
    tau_rc: f64,
    tau_th: f64,
) -> VerificationResult {
    let residual = |tau: f64| {
        if tau > 0.0 && span.is_finite()
        {
            (-span / tau).exp()
        }
        else
        {
            1.0
        }
    };
    // The slower of the two governs: a run long enough for the polarisation
    // may be nowhere near settled thermally.
    let threshold = residual(tau_rc).max(residual(tau_th)) * SETTLE_MARGIN;

    let v_error = (v_rc.last().copied().unwrap_or(0.0) - v_target).abs()
        / v_target.abs().max(f64::MIN_POSITIVE);
    let t_scale = (temp0 - t_target).abs().max(1.0);
    let t_error = (temperature.last().copied().unwrap_or(0.0) - t_target).abs() / t_scale;
    let worst = v_error.max(t_error);

    VerificationResult {
        id: "relaxes_to_closed_form_levels".to_string(),
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
            "the overpotential ended {v_error:.3e} from I*R1 = {v_target:.6} V and the \
             temperature {t_error:.3e} from its {t_target:.4} K equilibrium, against the \
             {threshold:.3e} the run's length allows given time constants of {tau_rc:.3e} s \
             and {tau_th:.3e} s"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/battery_discharge.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = BatteryAdapter;
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
        let validated = BatteryAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            BatteryAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn the_state_of_charge_follows_the_exact_line() {
        let r = run();
        let v = check(&r, "coulomb_counting");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn both_first_order_states_settle() {
        let r = run();
        let v = check(&r, "relaxes_to_closed_form_levels");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The cell must genuinely warm up, or the thermal half of the check is
    /// verifying a state that never moved.
    #[test]
    fn the_cell_actually_self_heats() {
        let r = run();
        let temp = &r
            .series
            .iter()
            .find(|s| s.id == "temperature")
            .expect("temperature")
            .values;
        let rise = temp.last().unwrap() - temp[0];
        assert!(
            rise > 1.0,
            "the cell warmed only {rise} K, so the thermal check is idle"
        );
    }

    #[test]
    fn the_coulomb_check_catches_a_wrong_capacity() {
        let verdict = coulomb_check(&[1.0, 0.9], &[1.0, 0.8]);
        assert_eq!(verdict.status, VerificationStatus::Failed);
    }

    /// The threshold must follow the SLOWER of the two time constants: a run
    /// long enough for the polarisation can be nowhere near settled
    /// thermally, and holding it to the fast one would fail correct runs.
    #[test]
    fn the_settle_threshold_follows_the_slower_state() {
        // Ten polarisation time constants but only one thermal one.
        let verdict = settle_check(
            &[0.0, 1.0],
            &[300.0, 310.0],
            1.0,
            313.0,
            300.0,
            10.0,
            1.0,
            10.0,
        );
        assert_eq!(
            verdict.status,
            VerificationStatus::Passed,
            "{}",
            verdict.explanation
        );
    }
}
