//! `sim.electrical.van_der_pol` — the self-sustaining oscillator
//! (`scirust_sim::electrical::VanDerPol`).
//!
//! `x'' - mu*(1 - x^2)*x' + x = 0`. Negative damping inside the strip
//! `|x| < 1` and positive damping outside, so energy flows *in* near the
//! origin and *out* far from it. The result is a **limit cycle**: an orbit the
//! system settles onto from anywhere, unlike a conservative oscillator whose
//! amplitude is whatever you gave it.
//!
//! # The two checks ask genuinely different questions
//!
//! **`energy_balance`** is a differential identity. The model states its own
//! energy rate, `dE/dt = mu*(1 - x^2)*v^2`, so integrating that rate along
//! the run must reproduce the change in `E` computed from the endpoints. Two
//! quantities from the same trajectory by different routes — one differenced,
//! one integrated — which is what makes it a check rather than a restatement.
//!
//! Its threshold is the composite trapezoid rule's own leading error, which
//! Euler-Maclaurin gives in closed form as `(h^2/12)*(f'(end) - f'(start))`
//! for the integrand `f = mu*(1 - x^2)*v^2`. Both endpoint derivatives are
//! available from the trajectory, so the threshold is computed rather than
//! chosen.
//!
//! **`limit_cycle_is_independent_of_the_start`** integrates a *second*
//! trajectory from a deliberately different initial condition and compares
//! the two late amplitudes. This is the defining property of a limit cycle
//! and the one thing an energy identity cannot see: a conservative oscillator
//! satisfies its own energy balance perfectly while having a different orbit
//! for every start.
//!
//! That second tolerance is **chosen**, not derived, and the reason is worth
//! naming: the convergence rate onto the cycle is set by the orbit's Floquet
//! multiplier, which the model does not state and which has no elementary
//! closed form. A test asserts a longer run agrees better, which is the
//! evidence available in the absence of the exact rate.
use scirust_sim::electrical::VanDerPol;
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
use crate::sink::{EventSink, RunEvent, SubRangeSink};
use crate::validate_support::{
    RK4_SOLVER, check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

/// Slack over the trapezoid rule's computed leading error.
const ENERGY_MARGIN: f64 = 8.0;

/// Relative agreement demanded between two limit-cycle amplitudes reached
/// from different starting points.
///
/// Chosen, not derived — see the module header on the Floquet multiplier.
/// Sized so the tutorial's ten cycles of transient are comfortably enough and
/// a run that had not converged would fail.
const AMPLITUDE_TOLERANCE: f64 = 1e-3;

/// Where the second trajectory starts, as a multiple of the first's radius.
///
/// Far enough out that it approaches the cycle from *outside* while the
/// tutorial's start is inside it: the two therefore converge from opposite
/// directions, and agreeing anyway is a much stronger statement than two
/// nearby starts agreeing.
const PROBE_RADIUS: f64 = 3.0;

const MU: FieldDescriptor = FieldDescriptor {
    canonical_name: "mu",
    display_name: "Nonlinearity parameter",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Strength of the nonlinear damping. Zero makes it a plain harmonic oscillator with no limit cycle at all; large values make the cycle strongly relaxational.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 311),
};

const POSITION: FieldDescriptor = FieldDescriptor {
    canonical_name: "position",
    display_name: "Initial position",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "x at t = start. The limit cycle is reached from anywhere but the origin, which is an unstable fixed point.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 312),
};

const VELOCITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "velocity",
    display_name: "Initial velocity",
    required: true,
    // The position is dimensionless, so its rate is a frequency. Writing
    // `unit = "1/s"` in a scenario and declaring DIMENSIONLESS here is the
    // mismatch this catches.
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "x' at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 313),
};

const ENERGY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_balance",
    description: "The model states its own energy rate dE/dt = mu*(1-x^2)*v^2. Integrating that rate along the run must reproduce the change in E computed from the endpoints — the same trajectory by two different routes, one differenced and one integrated. The threshold is the composite trapezoid rule's leading Euler-Maclaurin error, (h^2/12)*(f'(end) - f'(start)), computed from the run rather than chosen.",
};

const LIMIT_CYCLE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "limit_cycle_is_independent_of_the_start",
    description: "A second trajectory is integrated from a start outside the cycle while the scenario's own start is inside it, and the two late amplitudes are compared. This is what makes it a limit cycle rather than an oscillator: a conservative system satisfies its energy balance perfectly and still has a different orbit for every initial condition, so the energy check alone cannot see this.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.electrical.van_der_pol"),
    display_name: "Van der Pol oscillator",
    category: CapabilityCategory::Electrical,
    source_crate: "scirust-sim",
    summary: "A self-sustaining oscillator that settles onto the same orbit from any start, verified against its own stated energy rate and against a second trajectory.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[MU],
    initial_state: &[POSITION, VELOCITY],
    outputs: &[
        OutputDescriptor {
            id: "position",
            display_name: "Position x",
            unit: "1",
            description: "The oscillating state.",
        },
        OutputDescriptor {
            id: "velocity",
            display_name: "Velocity x'",
            unit: "1/s",
            description: "Its rate.",
        },
        OutputDescriptor {
            id: "energy",
            display_name: "Energy",
            unit: "1",
            description: "(x^2 + v^2)/2, which grows inside |x| < 1 and shrinks outside.",
        },
        OutputDescriptor {
            id: "energy_rate",
            display_name: "Energy rate",
            unit: "1/s",
            description: "mu*(1-x^2)*v^2, the model's own stated derivative of the energy.",
        },
        OutputDescriptor {
            id: "probe_position",
            display_name: "Probe position",
            unit: "1",
            description: "A second trajectory from outside the cycle, shown converging onto the same orbit.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ENERGY_CHECK, LIMIT_CYCLE_CHECK],
    },
};

/// The `sim.electrical.van_der_pol` adapter.
#[derive(Debug, Default)]
pub struct VanDerPolAdapter;

impl CapabilityAdapter for VanDerPolAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["mu"]));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["position", "velocity"],
        ));
        for e in [
            resolve_model_scalar(scenario, &MU).err(),
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
        let mu = resolve_model_scalar(s, &MU).expect("validated");
        let model =
            VanDerPol::new(mu).map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
        let x0 = resolve_state_vector(s, &POSITION).expect("validated")[0];
        let v0 = resolve_state_vector(s, &VELOCITY).expect("validated")[0];

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

        let span = TimeSpan {
            t0,
            t_end: t1,
            h: step,
        };
        // Two trajectories, so each gets half the progress window rather
        // than reporting its own zero-to-one. See `SubRangeSink`.
        let traj = simulate_cancellable(
            &model,
            &[x0, v0],
            span,
            control,
            &mut SubRangeSink::new(sink, 0.0, 0.5),
        )?;

        // A second trajectory from outside the cycle. The first starts inside
        // it, so the two approach from opposite directions and agreeing is a
        // stronger statement than two nearby starts agreeing.
        let probe_start = (x0 * x0 + v0 * v0).sqrt().max(1.0) * PROBE_RADIUS;
        let probe = simulate_cancellable(
            &model,
            &[probe_start, 0.0],
            span,
            control,
            &mut SubRangeSink::new(sink, 0.5, 1.0),
        )?;

        let position = traj.column(0).expect("dim 0");
        let velocity = traj.column(1).expect("dim 1");
        let probe_position = probe.column(0).expect("dim 0");
        let energy: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.energy(row).expect("dim 2"))
            .collect();
        let energy_rate: Vec<f64> = traj
            .y
            .iter()
            .map(|row| mu * (1.0 - row[0] * row[0]) * row[1] * row[1])
            .collect();

        let balance = energy_check(&traj.t, &energy, &energy_rate, &traj.y, mu, step);
        let cycle = limit_cycle_check(&position, &probe_position);

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
                    id: "position".to_string(),
                    display_name: "Position x".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: position.clone(),
                },
                Series {
                    id: "velocity".to_string(),
                    display_name: "Velocity x'".to_string(),
                    unit: "1/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: velocity.clone(),
                },
                Series {
                    id: "energy".to_string(),
                    display_name: "Energy".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: energy.clone(),
                },
                Series {
                    id: "energy_rate".to_string(),
                    display_name: "Energy rate".to_string(),
                    unit: "1/s".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: energy_rate.clone(),
                },
                Series {
                    id: "probe_position".to_string(),
                    display_name: "Probe position".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: probe_position.clone(),
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "mu".to_string(),
                    display_name: "Nonlinearity parameter".to_string(),
                    value: MetricValue::Scalar(model.mu()),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "limit_cycle_amplitude".to_string(),
                    display_name: "Limit-cycle amplitude".to_string(),
                    value: MetricValue::Scalar(late_amplitude(&position)),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "probe_amplitude".to_string(),
                    display_name: "Probe amplitude".to_string(),
                    value: MetricValue::Scalar(late_amplitude(&probe_position)),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "energy_change".to_string(),
                    display_name: "Net energy change over the run".to_string(),
                    value: MetricValue::Scalar(
                        energy.last().copied().unwrap_or(f64::NAN)
                            - energy.first().copied().unwrap_or(f64::NAN),
                    ),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![balance, cycle],
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

/// The largest excursion in the run's final quarter — the amplitude once the
/// transient has gone, rather than the largest the run ever saw.
fn late_amplitude(values: &[f64]) -> f64 {
    let from = values.len() * 3 / 4;
    values[from..]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
}

/// The model's own stated energy rate, integrated, against the change in its
/// energy, differenced.
fn energy_check(
    t: &[f64],
    energy: &[f64],
    rate: &[f64],
    rows: &[Vec<f64>],
    mu: f64,
    h: f64,
) -> VerificationResult {
    // Composite trapezoid on the integrator's own grid.
    let integrated: f64 = rate
        .windows(2)
        .zip(t.windows(2))
        .map(|(r, t)| 0.5 * (r[0] + r[1]) * (t[1] - t[0]))
        .sum();
    let differenced = energy.last().copied().unwrap_or(f64::NAN) - energy[0];
    let discrepancy = (integrated - differenced).abs();

    // Euler-Maclaurin: the composite trapezoid's leading error is
    // (h^2/12)*(f'(b) - f'(a)) for the integrand f. Here
    // f = mu*(1 - x^2)*v^2, so f' = mu*(-2*x*v^3 + 2*(1 - x^2)*v*a) with
    // a = mu*(1 - x^2)*v - x, the model's own acceleration.
    let df = |row: &[f64]| {
        let (x, v) = (row[0], row[1]);
        let a = mu * (1.0 - x * x) * v - x;
        mu * (-2.0 * x * v * v * v + 2.0 * (1.0 - x * x) * v * a)
    };
    let leading =
        (h * h / 12.0) * (df(rows.last().expect("at least one sample")) - df(&rows[0])).abs();
    let scale = energy.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let threshold = (leading * ENERGY_MARGIN).max(scale * f64::EPSILON * rate.len() as f64);

    VerificationResult {
        id: ENERGY_CHECK.id.to_string(),
        status: if discrepancy <= threshold
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(discrepancy),
        threshold: Some(threshold),
        explanation: format!(
            "integrating the model's stated rate mu*(1-x^2)*v^2 along the run gave \
             {integrated:.9}, against the {differenced:.9} the energy actually changed by — a \
             discrepancy of {discrepancy:.3e}, against the {threshold:.3e} the trapezoid rule's \
             own leading error over {} samples allows",
            rate.len()
        ),
    }
}

/// Two trajectories from opposite sides of the cycle, compared late.
fn limit_cycle_check(position: &[f64], probe: &[f64]) -> VerificationResult {
    let a = late_amplitude(position);
    let b = late_amplitude(probe);
    let error = (a - b).abs() / a.abs().max(f64::MIN_POSITIVE);
    VerificationResult {
        id: LIMIT_CYCLE_CHECK.id.to_string(),
        status: if error <= AMPLITUDE_TOLERANCE
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(error),
        threshold: Some(AMPLITUDE_TOLERANCE),
        explanation: format!(
            "the scenario's trajectory settled at an amplitude of {a:.6} and a second one \
             started well outside the cycle settled at {b:.6}, a relative difference of \
             {error:.3e}; a conservative oscillator would have kept two different orbits"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/van_der_pol.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = VanDerPolAdapter;
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
        let validated = VanDerPolAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            VanDerPolAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn the_energy_balance_holds() {
        let r = run();
        let v = check(&r, ENERGY_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn the_limit_cycle_does_not_depend_on_the_start() {
        let r = run();
        let v = check(&r, LIMIT_CYCLE_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// For small mu the cycle is near-circular with amplitude 2 — the
    /// classical Lienard result. This is an oracle the model does not state,
    /// so it lives in a test rather than in a shipped check, but it is worth
    /// having: it says the orbit is the right *size*, which neither shipped
    /// check does.
    #[test]
    fn a_weakly_nonlinear_cycle_has_amplitude_two() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.get_mut("mu").expect("mu").value = 0.05;
        // A weak nonlinearity is also a *slow* one: the approach to the cycle
        // decays at rate mu, so mu = 0.05 needs some five hundred time units
        // where the tutorial's mu = 1 needs ten. Starting nearer the cycle and
        // stepping more coarsely — the dynamics are gentle at this mu, and the
        // period is still about 2*pi — keeps the test cheap while giving it
        // the twenty-five e-foldings it actually needs.
        scenario
            .initial_state
            .get_mut("position")
            .expect("position")[0]
            .value = 1.5;
        scenario.solver.end.value = 500.0;
        scenario.solver.step.as_mut().expect("step").value = 0.01;
        let r = run_scenario(&scenario);
        let amplitude = metric(&r, "limit_cycle_amplitude");
        assert!(
            (amplitude - 2.0).abs() < 0.05,
            "a weakly nonlinear cycle settled at {amplitude}, expected about 2"
        );
    }

    /// The evidence available in place of a Floquet multiplier: a longer run
    /// converges better. If this ever stopped holding, the amplitude
    /// tolerance would be hiding a failure to converge rather than bounding
    /// a residual transient.
    #[test]
    fn a_longer_run_agrees_more_closely() {
        let short = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.end.value *= 3.0;
        let long = run_scenario(&scenario);
        let err = |r: &RunResult| check(r, LIMIT_CYCLE_CHECK.id).measured.expect("measured");
        assert!(
            err(&long) < err(&short),
            "tripling the run made the two amplitudes agree less well: {} then {}",
            err(&short),
            err(&long)
        );
    }

    /// The probe must genuinely approach from the other side, or the check is
    /// comparing two trajectories that were already together.
    #[test]
    fn the_probe_starts_outside_the_cycle() {
        let r = run();
        let probe = series(&r, "probe_position");
        let amplitude = metric(&r, "limit_cycle_amplitude");
        assert!(
            probe[0] > amplitude * 1.2,
            "the probe started at {} against a cycle amplitude of {amplitude}",
            probe[0]
        );
    }
}
