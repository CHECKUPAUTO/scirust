//! `sim.thermal.hvac_zone` — a single-zone building thermal model
//! (`scirust_sim::hvac::ZoneThermal2R2C`).
//!
//! Two thermal masses — the air and the wall — coupled by `R_aw`, with the
//! wall tied to the outside through `R_wo`. Fix the outside temperature and
//! the HVAC heat input, and the zone relaxes.
//!
//! # What makes it verifiable
//!
//! The model states its **exact steady state** in closed form, and it is not
//! the obvious one: at equilibrium all the HVAC heat must leave through the
//! series path, so the air sits `Q*(R_aw + R_wo)` above outside and the wall
//! only `Q*R_wo`. A sign error in either resistance moves one node and not
//! the other, which is why both are checked separately rather than as a pair.
//!
//! The second check is the direction of heat flow: with the HVAC heating and
//! the outside colder, the air must be the warmest node and the outside the
//! coldest, at every sample. That ordering is a consequence of the physics
//! rather than of the parameters, so it catches a coupling written backwards
//! even when the steady state happens to come out right.
use scirust_sim::hvac::ZoneThermal2R2C;
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

/// Slack on the steady-state check, on top of the residual the run's own
/// length implies.
///
/// The zone relaxes with two time constants; the slower one sets the floor no
/// correct run can beat, exactly as for a first-order system. The threshold
/// is derived from it rather than chosen.
const SETTLE_MARGIN: f64 = 6.0;

const C_AIR: FieldDescriptor = FieldDescriptor {
    canonical_name: "c_air",
    display_name: "Air heat capacity",
    required: true,
    dimension: scirust_units::Dimension::ENERGY.div(scirust_units::Dimension::TEMPERATURE),
    accepted_units: &["J/K"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Thermal capacitance of the zone air. Small, so the air responds in minutes.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 220),
};

const C_WALL: FieldDescriptor = FieldDescriptor {
    canonical_name: "c_wall",
    display_name: "Wall heat capacity",
    required: true,
    dimension: scirust_units::Dimension::ENERGY.div(scirust_units::Dimension::TEMPERATURE),
    accepted_units: &["J/K"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Thermal capacitance of the wall mass. Large, so the wall responds in hours.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 221),
};

const R_AW: FieldDescriptor = FieldDescriptor {
    canonical_name: "r_aw",
    display_name: "Air-to-wall resistance",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE.div(scirust_units::Dimension::POWER),
    accepted_units: &["K/W"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Thermal resistance coupling the air node to the wall mass.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 222),
};

const R_WO: FieldDescriptor = FieldDescriptor {
    canonical_name: "r_wo",
    display_name: "Wall-to-outside resistance",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE.div(scirust_units::Dimension::POWER),
    accepted_units: &["K/W"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Thermal resistance from the wall mass to outside — the building envelope.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 223),
};

const T_OUTSIDE: FieldDescriptor = FieldDescriptor {
    canonical_name: "t_outside",
    display_name: "Outside temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Outside air temperature, held constant for the run.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 224),
};

const Q_HVAC: FieldDescriptor = FieldDescriptor {
    canonical_name: "q_hvac",
    display_name: "HVAC heat input",
    required: true,
    dimension: scirust_units::Dimension::POWER,
    accepted_units: &["W"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Heat delivered to the air node. Negative for cooling.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 225),
};

const T_AIR: FieldDescriptor = FieldDescriptor {
    canonical_name: "t_air",
    display_name: "Initial air temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Zone air temperature at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 226),
};

const T_WALL: FieldDescriptor = FieldDescriptor {
    canonical_name: "t_wall",
    display_name: "Initial wall temperature",
    required: true,
    dimension: scirust_units::Dimension::TEMPERATURE,
    accepted_units: &["K"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Wall mass temperature at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 227),
};

const STEADY_STATE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "steady_state",
    description: "Both nodes must approach the model's closed-form equilibrium: the air Q*(R_aw + R_wo) above outside, the wall only Q*R_wo. The two are checked separately because a sign error in one resistance moves one node and not the other. The tolerance is derived from the run's own length in slow time constants, not chosen.",
};

const HEAT_FLOWS_DOWNHILL_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "heat_flows_downhill",
    description: "With the HVAC heating and the outside colder, the air must be the warmest node and the outside the coldest at every sample. That ordering follows from the physics rather than from the parameters, so it catches a coupling written backwards even when the steady state comes out right.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.thermal.hvac_zone"),
    display_name: "Building zone thermal response",
    category: CapabilityCategory::Thermal,
    source_crate: "scirust-sim",
    summary: "A single-zone 2R2C building model: air and wall thermal masses relaxing toward a closed-form equilibrium under a fixed outside temperature and HVAC input.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[C_AIR, C_WALL, R_AW, R_WO, T_OUTSIDE, Q_HVAC],
    initial_state: &[T_AIR, T_WALL],
    outputs: &[
        OutputDescriptor {
            id: "t_air",
            display_name: "Air temperature",
            unit: "K",
            description: "Zone air temperature over time.",
        },
        OutputDescriptor {
            id: "t_wall",
            display_name: "Wall temperature",
            unit: "K",
            description: "Wall mass temperature over time.",
        },
        OutputDescriptor {
            id: "air_steady_state",
            display_name: "Air equilibrium",
            unit: "K",
            description: "The closed-form air equilibrium, for comparison.",
        },
        OutputDescriptor {
            id: "wall_steady_state",
            display_name: "Wall equilibrium",
            unit: "K",
            description: "The closed-form wall equilibrium, for comparison.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[STEADY_STATE_CHECK, HEAT_FLOWS_DOWNHILL_CHECK],
    },
};

/// The `sim.thermal.hvac_zone` adapter.
#[derive(Debug, Default)]
pub struct HvacZoneAdapter;

impl CapabilityAdapter for HvacZoneAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["c_air", "c_wall", "r_aw", "r_wo", "t_outside", "q_hvac"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["t_air", "t_wall"]));
        for e in [
            resolve_model_scalar(scenario, &C_AIR).err(),
            resolve_model_scalar(scenario, &C_WALL).err(),
            resolve_model_scalar(scenario, &R_AW).err(),
            resolve_model_scalar(scenario, &R_WO).err(),
            resolve_model_scalar(scenario, &T_OUTSIDE).err(),
            resolve_model_scalar(scenario, &Q_HVAC).err(),
            resolve_state_vector(scenario, &T_AIR).err(),
            resolve_state_vector(scenario, &T_WALL).err(),
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
        let r_aw = resolve_model_scalar(s, &R_AW).expect("validated");
        let r_wo = resolve_model_scalar(s, &R_WO).expect("validated");
        let t_outside = resolve_model_scalar(s, &T_OUTSIDE).expect("validated");
        let q = resolve_model_scalar(s, &Q_HVAC).expect("validated");
        let c_air = resolve_model_scalar(s, &C_AIR).expect("validated");
        let c_wall = resolve_model_scalar(s, &C_WALL).expect("validated");
        let air0 = resolve_state_vector(s, &T_AIR).expect("validated")[0];
        let wall0 = resolve_state_vector(s, &T_WALL).expect("validated")[0];

        let model = ZoneThermal2R2C::new(c_air, c_wall, r_aw, r_wo, t_outside, q)
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

        let traj = simulate_cancellable(
            &model,
            &[air0, wall0],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let air = traj.column(0).expect("dim 0 exists");
        let wall = traj.column(1).expect("dim 1 exists");
        let (air_ss, wall_ss) = model.steady_state();

        // The slow time constant, exactly. A 2R2C network is a 2x2 linear
        // system, so its eigenvalues are the roots of a quadratic and there
        // is no reason to settle for a bound: `(C_air + C_wall)*(R_aw + R_wo)`
        // over-estimates it by a factor of about four here, which would make
        // the settle threshold four times looser than the model allows and
        // the check correspondingly weaker.
        let slow_tau = slow_time_constant(c_air, c_wall, r_aw, r_wo);
        let span = *traj.t.last().expect("at least one sample") - traj.t[0];

        let settle = steady_state_check(&air, &wall, air_ss, wall_ss, air0, wall0, span, slow_tau);
        let ordering = ordering_check(&air, &wall, t_outside, q);

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
                    id: "t_air".to_string(),
                    display_name: "Air temperature".to_string(),
                    unit: "K".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: air.clone(),
                },
                Series {
                    id: "t_wall".to_string(),
                    display_name: "Wall temperature".to_string(),
                    unit: "K".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: wall.clone(),
                },
                Series {
                    id: "air_steady_state".to_string(),
                    display_name: "Air equilibrium".to_string(),
                    unit: "K".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![air_ss; traj.t.len()],
                },
                Series {
                    id: "wall_steady_state".to_string(),
                    display_name: "Wall equilibrium".to_string(),
                    unit: "K".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![wall_ss; traj.t.len()],
                },
            ],
            fields: vec![],
            metrics: vec![
                Metric {
                    id: "air_equilibrium".to_string(),
                    display_name: "Air equilibrium t_out + Q*(R_aw + R_wo)".to_string(),
                    value: MetricValue::Scalar(air_ss),
                    unit: Some("K".to_string()),
                },
                Metric {
                    id: "wall_equilibrium".to_string(),
                    display_name: "Wall equilibrium t_out + Q*R_wo".to_string(),
                    value: MetricValue::Scalar(wall_ss),
                    unit: Some("K".to_string()),
                },
                Metric {
                    id: "slow_time_constant".to_string(),
                    display_name: "Slow time constant".to_string(),
                    value: MetricValue::Scalar(slow_tau),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "envelope_conductance".to_string(),
                    display_name: "Envelope conductance".to_string(),
                    value: MetricValue::Scalar(model.conductance()),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![settle, ordering],
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

/// The slower of the 2R2C network's two time constants.
///
/// The state matrix is
/// `[[-a, a], [b, -(b + c)]]` with `a = 1/(C_air*R_aw)`,
/// `b = 1/(C_wall*R_aw)` and `c = 1/(C_wall*R_wo)`, whose eigenvalues are the
/// roots of `l^2 - tr*l + det`. Both are real and negative — a passive
/// thermal network cannot oscillate — so the slow one is the root nearer
/// zero, and its reciprocal is the time constant the zone actually settles
/// at.
fn slow_time_constant(c_air: f64, c_wall: f64, r_aw: f64, r_wo: f64) -> f64 {
    let a = 1.0 / (c_air * r_aw);
    let b = 1.0 / (c_wall * r_aw);
    let c = 1.0 / (c_wall * r_wo);
    let trace = -(a + b + c);
    let det = a * c;
    let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();
    // Both roots are negative; the one nearer zero governs the tail.
    let slow = (trace + discriminant) / 2.0;
    if slow < 0.0
    {
        -1.0 / slow
    }
    else
    {
        f64::INFINITY
    }
}

/// Both nodes against their closed-form equilibria, to the accuracy the run's
/// own length allows.
#[allow(clippy::too_many_arguments)]
fn steady_state_check(
    air: &[f64],
    wall: &[f64],
    air_ss: f64,
    wall_ss: f64,
    air0: f64,
    wall0: f64,
    span: f64,
    slow_tau: f64,
) -> VerificationResult {
    let residual = if slow_tau > 0.0 && span.is_finite()
    {
        (-span / slow_tau).exp()
    }
    else
    {
        1.0
    };
    let threshold = residual * SETTLE_MARGIN;

    let relative = |final_v: f64, target: f64, start: f64| {
        let scale = (start - target).abs().max(1.0);
        (final_v - target).abs() / scale
    };
    let air_error = relative(*air.last().unwrap_or(&0.0), air_ss, air0);
    let wall_error = relative(*wall.last().unwrap_or(&0.0), wall_ss, wall0);
    let worst = air_error.max(wall_error);

    VerificationResult {
        id: "steady_state".to_string(),
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
            "the air ended {air_error:.3e} and the wall {wall_error:.3e} of their initial \
             deviations from the closed-form equilibria ({air_ss:.4} K and {wall_ss:.4} K), \
             against the {residual:.3e} that {:.2} slow time constants still leave",
            span / slow_tau
        ),
    }
}

/// Heat must flow from the hot node to the cold one.
fn ordering_check(air: &[f64], wall: &[f64], t_outside: f64, q: f64) -> VerificationResult {
    let id = "heat_flows_downhill".to_string();
    if q == 0.0
    {
        return VerificationResult {
            id,
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: "with no HVAC input the zone relaxes to the outside temperature and \
                          there is no ordering to hold"
                .to_string(),
        };
    }

    // Heating puts the air above the wall above outside; cooling reverses
    // every inequality. Checking the signed gaps handles both without a
    // second code path.
    let sign = if q > 0.0 { 1.0 } else { -1.0 };
    let mut worst = 0.0_f64;
    for (a, w) in air.iter().zip(wall.iter())
    {
        // A negative gap is a violation; its magnitude is how bad.
        worst = worst.max(-sign * (a - w)).max(-sign * (w - t_outside));
    }
    // The transient can briefly invert the ordering when the run starts far
    // from equilibrium, so the tolerance is scaled to the driving gap.
    let tolerance = (q.abs() * 1e-9).max(1e-9);

    VerificationResult {
        id,
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
            "with Q = {q} W the nodes stayed ordered air-wall-outside throughout, to within \
             {worst:.3e} K; an inversion would mean heat flowing up a temperature gradient"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/hvac_zone.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = HvacZoneAdapter;
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
        let validated = HvacZoneAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            HvacZoneAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn both_nodes_reach_the_closed_form_equilibrium() {
        let r = run();
        let v = check(&r, "steady_state");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn the_nodes_stay_ordered() {
        let r = run();
        let v = check(&r, "heat_flows_downhill");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The steady-state check must catch a node that settles in the wrong
    /// place — which is what a sign error in one resistance produces.
    #[test]
    fn the_steady_state_check_catches_a_node_in_the_wrong_place() {
        let verdict = steady_state_check(
            &[290.0, 295.0],
            &[290.0, 292.0],
            300.0,
            292.0,
            290.0,
            290.0,
            10.0,
            1.0,
        );
        assert_eq!(
            verdict.status,
            VerificationStatus::Failed,
            "{}",
            verdict.explanation
        );
    }

    /// …and the ordering check must catch heat flowing uphill, which a
    /// steady state landing in the right place would not reveal.
    #[test]
    fn the_ordering_check_catches_heat_flowing_uphill() {
        let verdict = ordering_check(&[290.0], &[300.0], 280.0, 500.0);
        assert_eq!(
            verdict.status,
            VerificationStatus::Failed,
            "{}",
            verdict.explanation
        );
    }

    #[test]
    fn a_zone_with_no_hvac_input_declines_the_ordering_check() {
        let verdict = ordering_check(&[290.0], &[290.0], 290.0, 0.0);
        assert_eq!(verdict.status, VerificationStatus::NotApplicable);
    }
}
