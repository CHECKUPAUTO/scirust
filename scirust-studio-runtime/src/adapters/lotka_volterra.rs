//! `sim.ecology.lotka_volterra` — predator–prey dynamics
//! (`scirust_sim::ecology::LotkaVolterra`).
//!
//! The scientifically interesting property, and the one this adapter
//! verifies, is that the model has an **exact first integral**:
//!
//! ```text
//! V = δ·x − γ·ln x + β·y − α·ln y
//! ```
//!
//! is constant along every true trajectory. That makes it a far sharper
//! oracle than "the curve looks periodic": RK4 does not conserve `V`
//! exactly, so its drift is a direct, quantitative measure of how much the
//! integration has wandered off the true closed orbit — reported here as a
//! number, not as an impression.

use scirust_sim::ecology::LotkaVolterra;

use crate::execute_support::{TimeSpan, simulate_cancellable};
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
    RunSummary, Series, SeriesRole, TIME_AXIS_ID, VerificationResult, VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_solver,
    resolve_state_vector,
};

const ALPHA: FieldDescriptor = FieldDescriptor {
    canonical_name: "alpha",
    display_name: "Prey growth rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Per-capita prey birth rate alpha in x' = alpha*x - beta*x*y.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 150),
};

const BETA: FieldDescriptor = FieldDescriptor {
    canonical_name: "beta",
    display_name: "Predation rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate at which encounters remove prey, beta in x' = alpha*x - beta*x*y.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 151),
};

const DELTA: FieldDescriptor = FieldDescriptor {
    canonical_name: "delta",
    display_name: "Predator conversion rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate at which consumed prey become predators, delta in y' = delta*x*y - gamma*y.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 152),
};

const GAMMA: FieldDescriptor = FieldDescriptor {
    canonical_name: "gamma",
    display_name: "Predator death rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Per-capita predator death rate gamma in y' = delta*x*y - gamma*y.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 153),
};

const PREY: FieldDescriptor = FieldDescriptor {
    canonical_name: "prey",
    display_name: "Initial prey population",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    // Strictly positive: the first integral contains ln x, and a population
    // that starts at zero stays at zero — a fixed point, not an orbit.
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Prey population at t = start. Must be strictly positive.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 154),
};

const PREDATOR: FieldDescriptor = FieldDescriptor {
    canonical_name: "predator",
    display_name: "Initial predator population",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Predator population at t = start. Must be strictly positive.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 155),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const FIRST_INTEGRAL_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "first_integral_drift",
    description: "V = delta*x - gamma*ln x + beta*y - alpha*ln y is exactly conserved along a \
                  true Lotka-Volterra trajectory; the check measures RK4's relative drift from \
                  it, which bounds how far the computed orbit has left the real one.",
};

const POSITIVITY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "populations_stay_positive",
    description: "Both populations remain strictly positive: the true dynamics can never reach \
                  zero from a positive start, so a non-positive value means the integration \
                  left the physically meaningful region.",
};

/// The capability descriptor for `sim.ecology.lotka_volterra`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.ecology.lotka_volterra"),
    display_name: "Lotka-Volterra predator-prey",
    category: CapabilityCategory::Ecology,
    source_crate: "scirust-sim",
    summary: "Predator-prey population dynamics with an exactly conserved first integral, \
              integrated with fixed-step RK4.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[ALPHA, BETA, DELTA, GAMMA],
    initial_state: &[PREY, PREDATOR],
    outputs: &[
        OutputDescriptor {
            id: "prey",
            display_name: "Prey",
            unit: "1",
            description: "Prey population over time.",
        },
        OutputDescriptor {
            id: "predator",
            display_name: "Predator",
            unit: "1",
            description: "Predator population over time.",
        },
        OutputDescriptor {
            id: "first_integral",
            display_name: "First integral V",
            unit: "1",
            description: "The conserved quantity, plotted so its drift is visible rather than \
                          only summarised.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[FIRST_INTEGRAL_CHECK, POSITIVITY_CHECK],
    },
};

/// The `sim.ecology.lotka_volterra` adapter.
#[derive(Debug, Default)]
pub struct LotkaVolterraAdapter;

impl CapabilityAdapter for LotkaVolterraAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["alpha", "beta", "delta", "gamma"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["prey", "predator"]));
        for e in [
            resolve_model_scalar(scenario, &ALPHA).err(),
            resolve_model_scalar(scenario, &BETA).err(),
            resolve_model_scalar(scenario, &DELTA).err(),
            resolve_model_scalar(scenario, &GAMMA).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for e in [
            resolve_state_vector(scenario, &PREY).err(),
            resolve_state_vector(scenario, &PREDATOR).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        if let Err(e) = resolve_solver(scenario, DESCRIPTOR.supported_solvers)
        {
            errors.push(e);
        }
        // Every adapter checks this, including the deterministic ones — see
        // `resolve_replicates`.
        if let Err(e) = resolve_replicates(scenario, DESCRIPTOR.determinism)
        {
            errors.push(e);
        }
        // The scenario's declared backend and precision must be ones this
        // capability actually has — see `resolve_precision`.
        if let Err(e) = resolve_backend_kind(scenario, &DESCRIPTOR)
        {
            errors.push(e);
        }
        if let Err(e) = resolve_precision(scenario, &DESCRIPTOR)
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
        if control.is_cancelled()
        {
            sink.emit(RunEvent::Cancelled);
            return Err(ExecutionError::Cancelled);
        }
        let s = scenario.scenario();
        let alpha = resolve_model_scalar(s, &ALPHA).expect("validated");
        let beta = resolve_model_scalar(s, &BETA).expect("validated");
        let delta = resolve_model_scalar(s, &DELTA).expect("validated");
        let gamma = resolve_model_scalar(s, &GAMMA).expect("validated");
        let prey0 = resolve_state_vector(s, &PREY).expect("validated")[0];
        let predator0 = resolve_state_vector(s, &PREDATOR).expect("validated")[0];

        let model = LotkaVolterra::new(alpha, beta, delta, gamma)
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

        let v0 = model.first_integral(&[prey0, predator0]);
        let traj = simulate_cancellable(
            &model,
            &[prey0, predator0],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;
        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");

        let prey_series = traj.column(0).expect("dim 0 exists");
        let predator_series = traj.column(1).expect("dim 1 exists");

        // The first integral is plotted, not just summarised, so a reader can
        // see *where* the drift happened rather than only how large it got.
        // A state that has left the positive quadrant has no defined V; the
        // series carries the last defined value forward rather than a NaN,
        // because a non-finite value in a stored result is refused outright
        // by `validate_result` and the positivity check below is the honest
        // place to report that failure.
        let mut first_integral_series = Vec::with_capacity(traj.len());
        let mut last_defined = v0.unwrap_or(0.0);
        for row in traj.y.iter()
        {
            if let Some(v) = model.first_integral(row)
            {
                last_defined = v;
            }
            first_integral_series.push(last_defined);
        }

        let mut min_population = f64::INFINITY;
        for row in traj.y.iter()
        {
            min_population = min_population.min(row[0]).min(row[1]);
        }

        let v1 = model.first_integral(last);
        let integral_check = match (v0, v1)
        {
            (Some(a), Some(b)) =>
            {
                let drift = (b - a).abs() / a.abs().max(1e-300);
                // Looser than the spring's 1e-6: this is a nonlinear
                // oscillator whose orbit RK4 tracks less exactly, and a
                // threshold nobody can meet is a check nobody believes.
                let threshold = 1e-4;
                VerificationResult {
                    id: "first_integral_drift".to_string(),
                    status: if drift <= threshold
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: Some(drift),
                    threshold: Some(threshold),
                    explanation: format!(
                        "relative drift {drift:.3e} in the conserved quantity \
                         V = delta*x - gamma*ln x + beta*y - alpha*ln y"
                    ),
                }
            },
            _ => VerificationResult {
                id: "first_integral_drift".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: "V is defined only where both populations are strictly positive; \
                              the trajectory left that region, so there is no invariant to \
                              compare against"
                    .to_string(),
            },
        };

        let positivity_check = VerificationResult {
            id: "populations_stay_positive".to_string(),
            status: if min_population > 0.0
            {
                VerificationStatus::Passed
            }
            else
            {
                VerificationStatus::Failed
            },
            measured: Some(min_population),
            threshold: Some(0.0),
            explanation: format!(
                "smallest population reached over the run: {min_population:.6e} \
                 (the true dynamics never reach zero from a positive start)"
            ),
        };

        let (equilibrium_prey, equilibrium_predator) = model.equilibrium();

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: TIME_AXIS_ID.to_string(),
                steps: traj.len() - 1,
                t_start: traj.t.first().copied().unwrap_or(t0),
                t_end: traj.last_time().unwrap_or(t1),
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
                    id: "prey".to_string(),
                    display_name: "Prey".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: prey_series,
                },
                Series {
                    id: "predator".to_string(),
                    display_name: "Predator".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: predator_series,
                },
                Series {
                    id: "first_integral".to_string(),
                    display_name: "First integral V".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: first_integral_series,
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "final_prey".to_string(),
                    display_name: "Final prey".to_string(),
                    value: MetricValue::Scalar(last[0]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_predator".to_string(),
                    display_name: "Final predator".to_string(),
                    value: MetricValue::Scalar(last[1]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "equilibrium_prey".to_string(),
                    display_name: "Coexistence equilibrium (prey)".to_string(),
                    value: MetricValue::Scalar(equilibrium_prey),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "equilibrium_predator".to_string(),
                    display_name: "Coexistence equilibrium (predator)".to_string(),
                    value: MetricValue::Scalar(equilibrium_predator),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![integral_check, positivity_check],
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
                // This capability's result does not depend on a seed, so
                // recording one would imply it did.
                seed: None,
            },
        };
        crate::result::validate_result(&result)
            .map_err(|d| ExecutionError::Internal(crate::result::describe_defects(&d)))?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/lotka_volterra.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = LotkaVolterraAdapter;
        let validated = adapter.validate(&scenario).expect("tutorial is valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        assert_eq!(scenario.capability.id, DESCRIPTOR.id.0);
    }

    /// The point of the capability: RK4 tracks the true closed orbit closely
    /// enough that its conserved quantity barely moves.
    #[test]
    fn the_first_integral_is_conserved_to_within_its_threshold() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "first_integral_drift")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed, "{check:?}");
        assert!(
            check.measured.unwrap() < 1e-6,
            "drift {:?} is far worse than RK4 at this step should manage",
            check.measured
        );
    }

    #[test]
    fn both_populations_stay_positive() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "populations_stay_positive")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed);
        assert!(check.measured.unwrap() > 0.0);
    }

    /// A predator–prey orbit is periodic; a run over several periods must
    /// come back near where it started. This is a property the first
    /// integral alone does not pin down — a trajectory could conserve V and
    /// still crawl along the orbit at the wrong speed.
    #[test]
    fn the_orbit_is_periodic_and_returns_near_its_start() {
        let result = run();
        let prey = &result.series[0].values;
        let predator = &result.series[1].values;

        let (prey0, predator0) = (prey[0], predator[0]);
        let mut closest = f64::INFINITY;
        // Skip the first tenth so "closest" cannot be the start itself.
        for i in prey.len() / 10..prey.len()
        {
            let d = ((prey[i] - prey0).powi(2) + (predator[i] - predator0).powi(2)).sqrt();
            closest = closest.min(d);
        }
        assert!(
            closest < 1e-2,
            "the trajectory never returned near its start (closest {closest:.3e}); \
             that is not a closed orbit"
        );
    }

    /// Both populations must actually oscillate. A scenario that happened to
    /// start at the equilibrium would pass every conservation check while
    /// showing nothing, so the tutorial is asserted to be a real orbit.
    #[test]
    fn the_tutorial_scenario_actually_oscillates() {
        let result = run();
        for series in result.series.iter().take(2)
        {
            let min = series.values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = series
                .values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                max / min > 1.5,
                "{} barely moves (min {min}, max {max}); the tutorial would show a flat line",
                series.id
            );
        }
    }

    #[test]
    fn the_equilibrium_metrics_match_the_model() {
        let result = run();
        let scenario = parse_toml(TUTORIAL).unwrap();
        let alpha = resolve_model_scalar(&scenario, &ALPHA).unwrap();
        let beta = resolve_model_scalar(&scenario, &BETA).unwrap();
        let delta = resolve_model_scalar(&scenario, &DELTA).unwrap();
        let gamma = resolve_model_scalar(&scenario, &GAMMA).unwrap();

        let metric = |id: &str| match result.metrics.iter().find(|m| m.id == id).map(|m| &m.value)
        {
            Some(MetricValue::Scalar(v)) => *v,
            other => panic!("{id}: {other:?}"),
        };
        assert_eq!(metric("equilibrium_prey"), gamma / delta);
        assert_eq!(metric("equilibrium_predator"), alpha / beta);
    }

    #[test]
    fn the_axis_carries_the_integrator_s_own_coordinates() {
        let result = run();
        let axis = &result.axes[0];
        assert_eq!(axis.id, TIME_AXIS_ID);
        assert_eq!(axis.values.len(), result.series[0].values.len());
        assert_eq!(axis.values[0], result.summary.t_start);
        assert_eq!(*axis.values.last().unwrap(), result.summary.t_end);
    }

    #[test]
    fn rejects_a_non_positive_initial_population() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.insert(
            "prey".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "1".to_string(),
            }],
        );
        let report = LotkaVolterraAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == PREY.error_code));
    }

    #[test]
    fn rejects_a_non_positive_rate() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "alpha".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "1/s".to_string(),
            },
        );
        let report = LotkaVolterraAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == ALPHA.error_code));
    }

    #[test]
    fn rejects_unknown_model_field() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "carrying_capacity".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 100.0,
                unit: "1".to_string(),
            },
        );
        let report = LotkaVolterraAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("carrying_capacity"))
        );
    }

    #[test]
    fn rejects_unsupported_solver() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.id = "rosenbrock".to_string();
        let report = LotkaVolterraAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("rosenbrock"))
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = LotkaVolterraAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let err = adapter
            .execute(&validated, &control, &mut NullEventSink)
            .unwrap_err();
        assert_eq!(err, ExecutionError::Cancelled);
    }

    #[test]
    fn emits_started_and_completed_events() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = LotkaVolterraAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        assert_eq!(sink.events().first(), Some(&RunEvent::Started));
        assert_eq!(sink.events().last(), Some(&RunEvent::Completed));
    }
}
