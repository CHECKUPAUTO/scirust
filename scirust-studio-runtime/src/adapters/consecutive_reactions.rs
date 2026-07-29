//! `sim.chemistry.consecutive_reactions` — first-order `A -> B -> C`
//! (`scirust_sim::chemistry::ConsecutiveReactions`).
//!
//! The simplest reaction scheme with an *intermediate*: A decays into B, B
//! decays into C, and B therefore rises and then falls. Every
//! radioactive-decay chain, every two-step synthesis, and the absorption half
//! of a pharmacokinetic model has this shape.
//!
//! # Verified against the complete closed form, not an asymptote
//!
//! The model states the Bateman solution for all three species, so the check
//! compares against the analytic answer at every recorded sample rather than
//! against where the curve ends up. The only thing that can move the
//! measurement is RK4's own truncation error.
//!
//! The second check is the mass balance `a + b + c = a0`, exact because the
//! three derivatives sum to zero identically. It is not redundant with the
//! first: a solution that drifted off the closed form would usually still
//! conserve mass, and a stencil that lost mass could still track two of the
//! three species.
//!
//! # `k1 == k2` is refused, not evaluated
//!
//! The Bateman form divides by `k2 - k1`. At equal rates the limit exists —
//! `b(t) = a0*k*t*exp(-k*t)` — but the model does not state it and returns
//! `None`, so this capability refuses the scenario with its own error code
//! rather than integrating something it cannot then verify. The same decision
//! `sim.pharmacology.oral_one_compartment` makes about `k_a == k_e`.
use scirust_sim::chemistry::ConsecutiveReactions;
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

/// Relative agreement demanded with the Bateman closed form.
///
/// This is RK4's truncation error and nothing else: the reference is the
/// analytic solution, evaluated at the integrator's own timestamps, so there
/// is no discretisation on the reference side to hide behind.
const EXACT_TOLERANCE: f64 = 1e-9;

/// Slack over the accumulated round-off of the three-species mass sum.
const CONSERVATION_MARGIN: f64 = 16.0;

/// Refuses a scenario whose two rate constants are equal.
const CODE_DEGENERATE_RATES: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 298);

const K1: FieldDescriptor = FieldDescriptor {
    canonical_name: "k1",
    display_name: "First rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate at which A becomes B.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 293),
};

const K2: FieldDescriptor = FieldDescriptor {
    canonical_name: "k2",
    display_name: "Second rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate at which B becomes C. Must differ from k1: the Bateman form divides by k2 - k1.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 294),
};

const A: FieldDescriptor = FieldDescriptor {
    canonical_name: "a",
    display_name: "Initial A",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Concentration of A at t = start. B and C start at zero, which is what the closed form assumes.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 295),
};

const EXACT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "matches_the_bateman_closed_form",
    description: "All three species are compared against the analytic Bateman solution at every recorded sample, not just at the end. The reference is exact, so the measurement is RK4's truncation error alone.",
};

const MASS_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "mass_is_conserved",
    description: "a + b + c is a linear invariant because the three derivatives sum to zero identically. The threshold is the run's own accumulated round-off. Not redundant with the closed-form check: a trajectory can drift off the analytic curve while still conserving mass, and can lose mass while still tracking two species.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.chemistry.consecutive_reactions"),
    display_name: "Consecutive first-order reactions",
    category: CapabilityCategory::Chemistry,
    source_crate: "scirust-sim",
    summary: "A -> B -> C with first-order kinetics: the canonical rising-then-falling intermediate, verified against its complete Bateman solution.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[K1, K2],
    initial_state: &[A],
    outputs: &[
        OutputDescriptor {
            id: "a",
            display_name: "Species A",
            unit: "1",
            description: "The starting material, decaying exponentially.",
        },
        OutputDescriptor {
            id: "b",
            display_name: "Species B",
            unit: "1",
            description: "The intermediate: rises, peaks, falls.",
        },
        OutputDescriptor {
            id: "c",
            display_name: "Species C",
            unit: "1",
            description: "The product, approaching a0.",
        },
        OutputDescriptor {
            id: "exact_b",
            display_name: "Exact species B",
            unit: "1",
            description: "The Bateman closed form for the intermediate, for comparison.",
        },
        OutputDescriptor {
            id: "mass",
            display_name: "Total mass",
            unit: "1",
            description: "The conserved sum, plotted so the invariant is visible.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[EXACT_CHECK, MASS_CHECK],
    },
};

/// The `sim.chemistry.consecutive_reactions` adapter.
#[derive(Debug, Default)]
pub struct ConsecutiveReactionsAdapter;

impl CapabilityAdapter for ConsecutiveReactionsAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["k1", "k2"]));
        errors.extend(check_unknown_state_fields(scenario, &["a"]));
        for e in [
            resolve_model_scalar(scenario, &K1).err(),
            resolve_model_scalar(scenario, &K2).err(),
            resolve_state_vector(scenario, &A).err(),
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
        if let (Ok(k1), Ok(k2)) = (
            resolve_model_scalar(scenario, &K1),
            resolve_model_scalar(scenario, &K2),
        )
        {
            if k1 == k2
            {
                errors.push(scirust_studio_command::CatalogedError {
                    code: CODE_DEGENERATE_RATES,
                    title: "Equal rate constants".to_string(),
                    explanation: format!(
                        "k1 and k2 are both {k1}, and the Bateman closed form divides by \
                         k2 - k1. The limit exists — b(t) = a0*k*t*exp(-k*t) — but the model \
                         does not state it, so this capability would be integrating something \
                         it could not then verify against anything"
                    ),
                    recoverable: true,
                    suggested_action: Some(
                        "separate the two rate constants, however slightly".to_string(),
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
        let s = scenario.scenario();
        let model = ConsecutiveReactions::new(
            resolve_model_scalar(s, &K1).expect("validated"),
            resolve_model_scalar(s, &K2).expect("validated"),
        )
        .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
        let a0 = resolve_state_vector(s, &A).expect("validated")[0];

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
            &[a0, 0.0, 0.0],
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let a = traj.column(0).expect("dim 0");
        let b = traj.column(1).expect("dim 1");
        let c = traj.column(2).expect("dim 2");
        let mass: Vec<f64> = traj.y.iter().map(|row| row.iter().sum()).collect();
        let exact: Vec<[f64; 3]> = traj
            .t
            .iter()
            .map(|t| {
                model
                    .exact(a0, *t)
                    .expect("validated: a0 >= 0 and k1 != k2")
            })
            .collect();
        let exact_b: Vec<f64> = exact.iter().map(|e| e[1]).collect();

        let exact_check = closed_form_check(&traj.y, &exact, a0);
        let mass_check = mass_check(&mass, a0);

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
                    id: "a".to_string(),
                    display_name: "Species A".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: a.clone(),
                },
                Series {
                    id: "b".to_string(),
                    display_name: "Species B".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: b.clone(),
                },
                Series {
                    id: "c".to_string(),
                    display_name: "Species C".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: c.clone(),
                },
                Series {
                    id: "exact_b".to_string(),
                    display_name: "Exact species B".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact_b.clone(),
                },
                Series {
                    id: "mass".to_string(),
                    display_name: "Total mass".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: mass.clone(),
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "peak_intermediate".to_string(),
                    display_name: "Peak concentration of B".to_string(),
                    value: MetricValue::Scalar(b.iter().cloned().fold(0.0_f64, f64::max)),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_product".to_string(),
                    display_name: "Final concentration of C".to_string(),
                    value: MetricValue::Scalar(c.last().copied().unwrap_or(f64::NAN)),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![exact_check, mass_check],
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

/// Every species against the Bateman solution at every sample.
fn closed_form_check(rows: &[Vec<f64>], exact: &[[f64; 3]], scale: f64) -> VerificationResult {
    let scale = scale.abs().max(f64::MIN_POSITIVE);
    let worst = rows
        .iter()
        .zip(exact.iter())
        .flat_map(|(row, e)| row.iter().zip(e.iter()).map(|(g, x)| (g - x).abs() / scale))
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
            "all three species tracked the Bateman closed form to within {worst:.3e} of the \
             initial concentration, across all {} samples",
            rows.len()
        ),
    }
}

/// The three-species sum against its exact invariant.
fn mass_check(mass: &[f64], a0: f64) -> VerificationResult {
    let worst = mass.iter().map(|m| (m - a0).abs()).fold(0.0_f64, f64::max);
    let threshold = mass.len() as f64 * f64::EPSILON * CONSERVATION_MARGIN * a0.abs().max(1.0);
    VerificationResult {
        id: MASS_CHECK.id.to_string(),
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
            "a + b + c stayed within {worst:.3e} of its initial {a0:.6} across {} samples, \
             against the {threshold:.3e} that many floating-point additions can accumulate",
            mass.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/consecutive_reactions.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = ConsecutiveReactionsAdapter;
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
        let validated = ConsecutiveReactionsAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            ConsecutiveReactionsAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn it_matches_the_bateman_closed_form() {
        let r = run();
        let v = check(&r, EXACT_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn mass_is_conserved() {
        let r = run();
        let v = check(&r, MASS_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The intermediate must genuinely rise and fall, or the run is a plain
    /// exponential decay wearing three names.
    #[test]
    fn the_intermediate_rises_and_falls() {
        let r = run();
        let b = series(&r, "b");
        let peak = metric(&r, "peak_intermediate");
        assert!(peak > 0.1, "B peaked at only {peak}");
        assert!(
            b[0] < peak && *b.last().unwrap() < peak * 0.5,
            "B did not fall back"
        );
    }

    /// Equal rate constants are refused with the code that says why, rather
    /// than integrated into a run nothing could verify.
    #[test]
    fn equal_rate_constants_are_refused() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        let k1 = scenario.model["k1"].value;
        scenario.model.get_mut("k2").expect("k2").value = k1;
        let report = ConsecutiveReactionsAdapter
            .validate(&scenario)
            .expect_err("k1 == k2 has no closed form here");
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == CODE_DEGENERATE_RATES)
        );
    }
}
