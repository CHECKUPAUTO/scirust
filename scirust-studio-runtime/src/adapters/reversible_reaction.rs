//! `sim.chemistry.reversible_reaction` — first-order `A <-> B`
//! (`scirust_sim::chemistry::ReversibleReaction`).
//!
//! The reaction that does not go to completion. Forward and backward rates
//! compete, and the system relaxes to an equilibrium *ratio* rather than to
//! an empty flask — which is the whole content of the equilibrium constant.
//!
//! # Two checks that fail independently
//!
//! The closed form is checked at every sample, so the measurement is RK4's
//! truncation error against an exact reference.
//!
//! The equilibrium check is separate and asks a different question: does the
//! settled ratio `b/a` equal `K = kf/kr`? A model with the right relaxation
//! rate and the wrong balance point passes the first check nowhere and the
//! second nowhere either — but a model with the right *equilibrium* and the
//! wrong rate would land in the right place by the end while being wrong
//! throughout, which is exactly what checking the whole trajectory catches.
//!
//! The equilibrium check's threshold is derived from the run's own length:
//! the approach is a single exponential at rate `kf + kr`, so after a span
//! the remaining deviation is `exp(-(kf + kr) * span)` of the initial one.
use scirust_sim::chemistry::ReversibleReaction;
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
const EXACT_TOLERANCE: f64 = 1e-9;

/// Slack over the residual the run's own length leaves on the approach to
/// equilibrium.
const EQUILIBRIUM_MARGIN: f64 = 5.0;

const KF: FieldDescriptor = FieldDescriptor {
    canonical_name: "kf",
    display_name: "Forward rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate of A -> B.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 299),
};

const KR: FieldDescriptor = FieldDescriptor {
    canonical_name: "kr",
    display_name: "Backward rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate of B -> A. The equilibrium constant is kf/kr.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 300),
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
    description: "Concentration of A at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 301),
};

const B: FieldDescriptor = FieldDescriptor {
    canonical_name: "b",
    display_name: "Initial B",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Concentration of B at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 302),
};

const EXACT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "matches_the_closed_form",
    description: "Both species are compared against the analytic solution at every recorded sample. The reference is exact, so the measurement is RK4's truncation error alone.",
};

const EQUILIBRIUM_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "settles_at_the_equilibrium_constant",
    description: "The settled ratio b/a must equal K = kf/kr. The threshold is derived from the run's own length: the approach is a single exponential at rate kf + kr, so a span leaves exp(-(kf+kr)*span) of the initial deviation and no correct run can beat that.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.chemistry.reversible_reaction"),
    display_name: "Reversible first-order reaction",
    category: CapabilityCategory::Chemistry,
    source_crate: "scirust-sim",
    summary: "A <-> B relaxing to an equilibrium ratio rather than to completion, verified against both its closed form and its equilibrium constant.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[KF, KR],
    initial_state: &[A, B],
    outputs: &[
        OutputDescriptor {
            id: "a",
            display_name: "Species A",
            unit: "1",
            description: "Concentration of A.",
        },
        OutputDescriptor {
            id: "b",
            display_name: "Species B",
            unit: "1",
            description: "Concentration of B.",
        },
        OutputDescriptor {
            id: "exact_a",
            display_name: "Exact species A",
            unit: "1",
            description: "The closed form, for comparison.",
        },
        OutputDescriptor {
            id: "equilibrium_a",
            display_name: "Equilibrium A",
            unit: "1",
            description: "The level A settles at.",
        },
        OutputDescriptor {
            id: "mass",
            display_name: "Total mass",
            unit: "1",
            description: "a + b, conserved exactly.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[EXACT_CHECK, EQUILIBRIUM_CHECK],
    },
};

/// The `sim.chemistry.reversible_reaction` adapter.
#[derive(Debug, Default)]
pub struct ReversibleReactionAdapter;

impl CapabilityAdapter for ReversibleReactionAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(scenario, &["kf", "kr"]));
        errors.extend(check_unknown_state_fields(scenario, &["a", "b"]));
        for e in [
            resolve_model_scalar(scenario, &KF).err(),
            resolve_model_scalar(scenario, &KR).err(),
            resolve_state_vector(scenario, &A).err(),
            resolve_state_vector(scenario, &B).err(),
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
        let kf = resolve_model_scalar(s, &KF).expect("validated");
        let kr = resolve_model_scalar(s, &KR).expect("validated");
        let model = ReversibleReaction::new(kf, kr)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
        let a0 = resolve_state_vector(s, &A).expect("validated")[0];
        let b0 = resolve_state_vector(s, &B).expect("validated")[0];

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
            &[a0, b0],
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
        let mass: Vec<f64> = traj.y.iter().map(|row| row.iter().sum()).collect();
        let exact: Vec<[f64; 2]> = traj
            .t
            .iter()
            .map(|t| {
                model
                    .exact(a0, b0, *t)
                    .expect("validated: both start non-negative")
            })
            .collect();
        let exact_a: Vec<f64> = exact.iter().map(|e| e[0]).collect();

        let total = a0 + b0;
        let a_eq = kr / (kf + kr) * total;
        let span = *traj.t.last().expect("at least one sample") - traj.t[0];

        let exact_check = closed_form_check(&traj.y, &exact, total);
        let equilibrium = equilibrium_check(EquilibriumMeasurement {
            a_end: a.last().copied().unwrap_or(f64::NAN),
            b_end: b.last().copied().unwrap_or(f64::NAN),
            total,
            k: model.equilibrium_constant(),
            initial_deviation: (a0 - a_eq).abs() / total.max(f64::MIN_POSITIVE),
            rate: kf + kr,
            span,
            samples: traj.t.len(),
        });

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
                    id: "exact_a".to_string(),
                    display_name: "Exact species A".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact_a.clone(),
                },
                Series {
                    id: "equilibrium_a".to_string(),
                    display_name: "Equilibrium A".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: vec![a_eq; traj.t.len()],
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
            metrics: vec![
                Metric {
                    id: "equilibrium_constant".to_string(),
                    display_name: "Equilibrium constant K = kf/kr".to_string(),
                    value: MetricValue::Scalar(model.equilibrium_constant()),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "relaxation_rate".to_string(),
                    display_name: "Relaxation rate kf + kr".to_string(),
                    value: MetricValue::Scalar(kf + kr),
                    unit: Some("1/s".to_string()),
                },
                Metric {
                    id: "equilibrium_a".to_string(),
                    display_name: "Equilibrium concentration of A".to_string(),
                    value: MetricValue::Scalar(a_eq),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![exact_check, equilibrium],
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

/// Both species against the closed form at every sample.
fn closed_form_check(rows: &[Vec<f64>], exact: &[[f64; 2]], scale: f64) -> VerificationResult {
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
            "both species tracked the closed form to within {worst:.3e} of the total \
             concentration, across all {} samples",
            rows.len()
        ),
    }
}

/// What the equilibrium check needs to know about the run.
///
/// Grouped rather than passed as eight positional arguments, three of which
/// are concentrations that would be trivially easy to swap.
struct EquilibriumMeasurement {
    a_end: f64,
    b_end: f64,
    total: f64,
    k: f64,
    initial_deviation: f64,
    rate: f64,
    span: f64,
    samples: usize,
}

/// The settled ratio against the equilibrium constant.
///
/// # The threshold has three factors, and the first version had only one
///
/// The obvious derivation is that the approach is a single exponential at
/// `kf + kr`, so a span leaves `exp(-(kf + kr) * span)` of the initial
/// deviation. That is correct and it is not sufficient, because it is a
/// statement about the *concentrations* and the check measures their
/// **ratio**. Two things it misses, both of which failed a correct run:
///
/// **The ratio amplifies error in the smaller species.** With `a + b` fixed
/// at `total`, the ratio is `total/a - 1`, so a relative error `e` in `a`
/// becomes `e * total^2 / (a * b)` in the ratio. For this tutorial's 1:4
/// equilibrium that is a factor of 6.25 — small, and larger than the margin
/// it was being compared against.
///
/// **Round-off is a floor the exponential can drop below.** Thirty time
/// constants leaves `exp(-30) = 9e-14` of the initial deviation, which is
/// already below what `f64` arithmetic accumulates over fifteen thousand
/// steps. Past that point the run is not limited by how far it has relaxed
/// but by how many additions it took to get there, which grows as
/// `sqrt(samples) * EPSILON` for errors that are not systematically signed.
///
/// So the threshold is the *larger* of the two floors, amplified by the
/// ratio's own sensitivity. Both terms are computed from the run; neither is
/// fitted.
fn equilibrium_check(m: EquilibriumMeasurement) -> VerificationResult {
    let ratio = m.b_end / m.a_end.abs().max(f64::MIN_POSITIVE);
    let error = (ratio - m.k).abs() / m.k.abs().max(f64::MIN_POSITIVE);

    let residual = if m.rate > 0.0 && m.span.is_finite()
    {
        (-m.rate * m.span).exp()
    }
    else
    {
        1.0
    };
    let physical_floor = m.initial_deviation * residual;
    let roundoff_floor = (m.samples as f64).sqrt() * f64::EPSILON;
    let amplification = m.total * m.total / (m.a_end * m.b_end).abs().max(f64::MIN_POSITIVE);
    let threshold = physical_floor.max(roundoff_floor) * amplification * EQUILIBRIUM_MARGIN;

    VerificationResult {
        id: EQUILIBRIUM_CHECK.id.to_string(),
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
            "the run ended with b/a = {ratio:.9} against the equilibrium constant K = {:.9}, a \
             relative difference of {error:.3e}; relaxing at {:.6} 1/s for {:.4} leaves \
             {physical_floor:.3e} of the starting deviation and {} samples of arithmetic leave \
             {roundoff_floor:.3e}, the larger of which the 1-to-{:.3} equilibrium amplifies by \
             {amplification:.3}",
            m.k, m.rate, m.span, m.samples, m.k
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/reversible_reaction.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = ReversibleReactionAdapter;
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
        let validated = ReversibleReactionAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            ReversibleReactionAdapter.execute(&validated, &control, &mut NullEventSink),
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
    fn it_settles_at_the_equilibrium_constant() {
        let r = run();
        let v = check(&r, EQUILIBRIUM_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The reaction must not go to completion — that is the entire point of
    /// it being reversible. Some A has to be left.
    #[test]
    fn the_reaction_stops_short_of_completion() {
        let r = run();
        let a_end = *series(&r, "a").last().unwrap();
        assert!(
            a_end > 0.05,
            "only {a_end} of A survived, so this looks irreversible"
        );
    }
}
