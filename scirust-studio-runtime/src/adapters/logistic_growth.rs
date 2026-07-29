//! `sim.ecology.logistic_growth` — bounded population growth
//! (`scirust_sim::ecology::LogisticGrowth`).
//!
//! Two things make this capability worth having beyond the model itself.
//!
//! It is the first capability with a **one-component state**. Every other
//! adapter carries at least a position and a velocity, so nothing until now
//! exercised the narrowest case the runtime, the result schema and the chart
//! all have to handle.
//!
//! And it has a **closed-form solution**, `x(t) = K / (1 + (K/x₀ − 1)·e^{−rt})`,
//! so the verification here is not a conservation law that a wrong-but-smooth
//! trajectory could satisfy: it is the maximum absolute error against the
//! exact answer at every recorded point. That is the strongest oracle in the
//! catalogue.

use scirust_sim::ecology::LogisticGrowth;

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

const RATE: FieldDescriptor = FieldDescriptor {
    canonical_name: "rate",
    display_name: "Intrinsic growth rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Per-capita growth rate r at low density, in x' = r*x*(1 - x/K).",
    error_code: ErrorCode::new(ErrorFamily::Validation, 160),
};

const CAPACITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "capacity",
    display_name: "Carrying capacity",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The population K the environment supports, in x' = r*x*(1 - x/K).",
    error_code: ErrorCode::new(ErrorFamily::Validation, 161),
};

const POPULATION: FieldDescriptor = FieldDescriptor {
    canonical_name: "population",
    display_name: "Initial population",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    // Strictly positive: zero is a fixed point, and the closed form divides
    // by x0.
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Population at t = start. Must be strictly positive.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 162),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
    reports_progress: true,
};

const ANALYTIC_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "analytic_error",
    description: "Logistic growth has the closed-form solution \
                  x(t) = K / (1 + (K/x0 - 1)*exp(-r*t)); the check reports the largest absolute \
                  difference between the integrated trajectory and that exact answer, over \
                  every recorded point.",
};

const CAPACITY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "stays_below_capacity",
    description: "A population starting below the carrying capacity approaches it from below \
                  and never exceeds it; overshoot means the step size is too coarse for the \
                  growth rate.",
};

/// The capability descriptor for `sim.ecology.logistic_growth`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.ecology.logistic_growth"),
    display_name: "Logistic growth",
    category: CapabilityCategory::Ecology,
    source_crate: "scirust-sim",
    summary: "Bounded population growth toward a carrying capacity, verified against its \
              closed-form solution.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[RATE, CAPACITY],
    initial_state: &[POPULATION],
    outputs: &[
        OutputDescriptor {
            id: "population",
            display_name: "Population",
            unit: "1",
            description: "Integrated population over time.",
        },
        OutputDescriptor {
            id: "exact",
            display_name: "Exact solution",
            unit: "1",
            description: "The closed-form solution at the same coordinates, so the two curves \
                          can be compared directly rather than through a single error number.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[ANALYTIC_CHECK, CAPACITY_CHECK],
    },
};

/// The `sim.ecology.logistic_growth` adapter.
#[derive(Debug, Default)]
pub struct LogisticGrowthAdapter;

impl CapabilityAdapter for LogisticGrowthAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["rate", "capacity"]));
        errors.extend(check_unknown_state_fields(scenario, &["population"]));
        for e in [
            resolve_model_scalar(scenario, &RATE).err(),
            resolve_model_scalar(scenario, &CAPACITY).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        if let Err(e) = resolve_state_vector(scenario, &POPULATION)
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
        let rate = resolve_model_scalar(s, &RATE).expect("validated");
        let capacity = resolve_model_scalar(s, &CAPACITY).expect("validated");
        let x0 = resolve_state_vector(s, &POPULATION).expect("validated")[0];

        let model = LogisticGrowth::new(rate, capacity)
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
            &[x0],
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

        let population_series = traj.column(0).expect("dim 0 exists");

        // The exact solution is evaluated at the integrator's own
        // coordinates, not at a regenerated linear grid: comparing against
        // the wrong times would measure the grid, not the solver.
        let mut exact_series = Vec::with_capacity(traj.len());
        let mut worst_error = 0.0_f64;
        let mut exact_defined = true;
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            match model.exact(x0, t - t0)
            {
                Some(exact) =>
                {
                    exact_series.push(exact);
                    worst_error = worst_error.max((row[0] - exact).abs());
                },
                None =>
                {
                    // Cannot happen for a validated x0 > 0, but the result
                    // must stay finite regardless: a NaN here would be
                    // refused by `validate_result`, which is a worse error
                    // message than the honest not-applicable below.
                    exact_defined = false;
                    exact_series.push(row[0]);
                },
            }
        }

        let overshoot = population_series
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - capacity;

        let analytic_check = if exact_defined
        {
            // Scaled to the carrying capacity: an absolute error of 1e-9 means
            // something very different for K = 1 and for K = 10^6.
            let relative = worst_error / capacity;
            let threshold = 1e-6;
            VerificationResult {
                id: "analytic_error".to_string(),
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
                    "largest deviation from the closed-form solution over the whole run: \
                     {worst_error:.3e} absolute, {relative:.3e} relative to the carrying capacity"
                ),
            }
        }
        else
        {
            VerificationResult {
                id: "analytic_error".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: "the closed form is undefined for this initial population".to_string(),
            }
        };

        let capacity_check = if x0 < capacity
        {
            VerificationResult {
                id: "stays_below_capacity".to_string(),
                status: if overshoot <= 0.0
                {
                    VerificationStatus::Passed
                }
                else
                {
                    VerificationStatus::Failed
                },
                measured: Some(overshoot),
                threshold: Some(0.0),
                explanation: format!(
                    "largest excess over the carrying capacity {capacity}: {overshoot:.3e} \
                     (a population starting below K approaches it from below)"
                ),
            }
        }
        else
        {
            VerificationResult {
                id: "stays_below_capacity".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: format!(
                    "this run starts at or above the carrying capacity ({x0} >= {capacity}), \
                     so it approaches K from above and the check does not apply"
                ),
            }
        };

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
                    id: "population".to_string(),
                    display_name: "Population".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: population_series,
                },
                Series {
                    id: "exact".to_string(),
                    display_name: "Exact solution".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact_series,
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "final_population".to_string(),
                    display_name: "Final population".to_string(),
                    value: MetricValue::Scalar(last[0]),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "fraction_of_capacity".to_string(),
                    display_name: "Fraction of carrying capacity reached".to_string(),
                    value: MetricValue::Scalar(last[0] / capacity),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![analytic_check, capacity_check],
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
        include_str!("../../../docs/studio/tutorials/logistic_growth.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = LogisticGrowthAdapter;
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

    /// The strongest claim in the catalogue: the integrated curve matches the
    /// exact solution, everywhere, not just at the end.
    #[test]
    fn the_trajectory_matches_the_closed_form_everywhere() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "analytic_error")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed, "{check:?}");
        assert!(
            check.measured.unwrap() < 1e-9,
            "relative error {:?} is far worse than RK4 at this step should manage",
            check.measured
        );
    }

    /// A single-component state is the narrowest case the schema handles, and
    /// nothing before this capability exercised it.
    #[test]
    fn a_one_component_state_produces_a_valid_result() {
        let result = run();
        assert_eq!(DESCRIPTOR.initial_state.len(), 1);
        assert_eq!(result.axes.len(), 1);
        assert_eq!(result.series.len(), 2);
        for series in &result.series
        {
            assert_eq!(series.values.len(), result.axes[0].values.len());
            assert_eq!(series.axis_id, TIME_AXIS_ID);
        }
        crate::result::validate_result(&result).expect("a valid result");
    }

    #[test]
    fn the_population_stays_below_the_carrying_capacity() {
        let result = run();
        let check = result
            .verifications
            .iter()
            .find(|v| v.id == "stays_below_capacity")
            .expect("the check ran");
        assert_eq!(check.status, VerificationStatus::Passed, "{check:?}");
    }

    /// The tutorial must show the sigmoid, not a flat line: it has to start
    /// well below the capacity and get most of the way there.
    #[test]
    fn the_tutorial_scenario_shows_a_real_sigmoid() {
        let result = run();
        let population = &result.series[0].values;
        let first = population[0];
        let last = *population.last().unwrap();
        assert!(
            last / first > 10.0,
            "the population barely grew ({first} -> {last})"
        );
        let fraction = match result
            .metrics
            .iter()
            .find(|m| m.id == "fraction_of_capacity")
            .map(|m| &m.value)
        {
            Some(MetricValue::Scalar(v)) => *v,
            other => panic!("{other:?}"),
        };
        assert!(
            fraction > 0.95,
            "the run stops before the curve flattens (reached {fraction} of capacity)"
        );
    }

    /// The exact series must be evaluated at the integrator's coordinates,
    /// so the two curves are comparable point by point.
    #[test]
    fn the_exact_series_is_sampled_at_the_integrator_s_own_coordinates() {
        let result = run();
        let axis = &result.axes[0];
        let exact = &result.series[1].values;
        assert_eq!(exact.len(), axis.values.len());

        let scenario = parse_toml(TUTORIAL).unwrap();
        let rate = resolve_model_scalar(&scenario, &RATE).unwrap();
        let capacity = resolve_model_scalar(&scenario, &CAPACITY).unwrap();
        let x0 = resolve_state_vector(&scenario, &POPULATION).unwrap()[0];
        let model = LogisticGrowth::new(rate, capacity).unwrap();

        for (i, t) in axis
            .values
            .iter()
            .enumerate()
            .step_by(axis.values.len() / 8)
        {
            let expected = model.exact(x0, t - axis.values[0]).unwrap();
            assert_eq!(
                exact[i], expected,
                "the exact series was not evaluated at t = {t}"
            );
        }
    }

    #[test]
    fn rejects_a_non_positive_initial_population() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.insert(
            "population".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: -1.0,
                unit: "1".to_string(),
            }],
        );
        let report = LogisticGrowthAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == POPULATION.error_code)
        );
    }

    #[test]
    fn rejects_a_non_positive_capacity() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "capacity".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "1".to_string(),
            },
        );
        let report = LogisticGrowthAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == CAPACITY.error_code));
    }

    #[test]
    fn rejects_unknown_state_field() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.insert(
            "predator".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 1.0,
                unit: "1".to_string(),
            }],
        );
        let report = LogisticGrowthAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("predator"))
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = LogisticGrowthAdapter;
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
        let adapter = LogisticGrowthAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        assert_eq!(sink.events().first(), Some(&RunEvent::Started));
        assert_eq!(sink.events().last(), Some(&RunEvent::Completed));
    }
}
