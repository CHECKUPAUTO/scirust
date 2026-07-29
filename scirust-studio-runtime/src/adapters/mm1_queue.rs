//! `sim.stochastic.mm1_queue` — a single-server queue, simulated many times
//! (`scirust_sim::stochastic::MM1Queue`).
//!
//! Poisson arrivals, one server with exponential service times, infinite
//! waiting room, first come first served. The model everyone reaches for when
//! something has to wait in line, and the one whose steady-state answers are
//! famous enough to be checked against by hand: with traffic intensity
//! `rho = lambda/mu < 1`, the mean number in the system is `L = rho/(1-rho)`,
//! the mean time in system is `W = 1/(mu-lambda)`, and the server is busy a
//! fraction `rho` of the time.
//!
//! # A capability whose result is a distribution
//!
//! Every other capability here produces a trajectory or a curve. This one
//! produces `n` independent discrete-event simulations, each of which reports
//! four aggregate statistics and no trajectory at all. There is nothing to
//! plot against time, and nothing is being swept: the only thing that differs
//! between the realisations is the seed.
//!
//! So the result carries a `replicate` axis
//! ([`RunDomain::Ensemble`](scirust_studio_registry::RunDomain)) with one
//! point per realisation, and — the reason this capability exists — three
//! [`Distribution`](crate::Distribution)s, which are the first in the
//! catalogue. A histogram's `n` bins need `n + 1` edges, so it fits neither a
//! `Series` nor a `Field`; see that type's own documentation for why storing
//! one as a single-row field would have been the wrong shape rather than a
//! shortcut.
//!
//! # Three checks, and the third does not need the textbook at all
//!
//! Two of them compare the ensemble mean against a closed form — `L` and
//! `rho` — with a threshold that is the **standard error of that mean**,
//! computed from the run's own spread. That is the right kind of threshold
//! for a stochastic capability: it tightens as `sqrt(n)` when you ask for
//! more replicates, so a run that wants a sharper answer gets a sharper test
//! rather than the same loose one.
//!
//! The third is **Little's law**, `L = lambda * W`. It relates two quantities
//! the run measured to each other and to nothing else — no steady-state
//! formula appears in it. A simulation with a systematic error in both `L`
//! and `W` that happened to match the textbook would still have to satisfy
//! it, and a simulation whose bookkeeping of arrivals and departures
//! disagreed would fail it while matching both formulas. It is the check that
//! is hardest to accidentally pass.
//!
//! # The finite horizon is a bias, and it is bounded rather than ignored
//!
//! Each realisation starts with an empty queue and runs for a finite time, so
//! its time-average is biased low: the queue spends the beginning of the run
//! filling up. The bias falls as `1/horizon`, and the tutorial's horizon is
//! chosen so it sits well under the statistical threshold. A test asserts a
//! longer horizon moves the answer *toward* the closed form, which is the
//! evidence that the residual really is transient bias and not a wrong model.

use scirust_sim::stochastic::MM1Queue;
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind, RunDomain,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::ensemble::replicate_seeds;
use crate::result::{
    Axis, AxisMonotonicity, Distribution, Metric, MetricValue, RESULT_SCHEMA_VERSION,
    RunProvenance, RunResult, RunSummary, RunWarning, Series, SeriesRole, VerificationResult,
    VerificationStatus, WarningCategory,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_seed, resolve_solver,
};

/// The id of the axis this capability indexes. Not `t`: there is no time in
/// the *result*, only inside each realisation.
const REPLICATE_AXIS_ID: &str = "replicate";

/// How many standard errors of the mean a closed form may sit away from it.
///
/// Three, which for an approximately normal sample mean is about one run in
/// four hundred failing by chance. That is the price of a check that is
/// sharp: a wider band would essentially never fail, and a narrower one would
/// fail honest runs often enough to be ignored.
const SIGMAS: f64 = 3.0;

/// Bins in each shipped histogram.
///
/// Enough to show the shape of a right-skewed queueing distribution and few
/// enough that a modest ensemble does not leave most of them empty.
const BIN_COUNT: usize = 24;

/// How far past the observed range the histograms extend, as a fraction of
/// that range.
///
/// A little padding on both sides, so the extreme realisations sit inside a
/// bin rather than in the overflow counter — which exists to report a badly
/// chosen range, not to absorb every run's own maximum.
const BIN_PADDING: f64 = 0.05;

/// Refuses a scenario whose queue is not stable.
const CODE_UNSTABLE_QUEUE: ErrorCode = ErrorCode::new(ErrorFamily::Validation, 321);

const DISCRETE_EVENT: SolverDescriptor = SolverDescriptor {
    id: "discrete_event",
    summary: "Exact discrete-event simulation: the state changes only at arrivals and departures, whose times are drawn from the model's exponential distributions. There is no step size to choose and no truncation error to bound — `solver.start` and `solver.end` bracket the simulated horizon of each realisation.",
    fixed_step: false,
    adaptive_tolerance: false,
};

const ARRIVAL_RATE: FieldDescriptor = FieldDescriptor {
    canonical_name: "arrival_rate",
    display_name: "Arrival rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Customers arriving per unit time. Must be below the service rate, or the queue grows without bound and the steady-state answers do not exist.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 319),
};

const SERVICE_RATE: FieldDescriptor = FieldDescriptor {
    canonical_name: "service_rate",
    display_name: "Service rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Customers the single server can finish per unit time when it is busy.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 320),
};

const QUEUE_LENGTH_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "mean_queue_length_matches_the_closed_form",
    description: "The across-replicate mean of the time-average number in the system against L = rho/(1-rho). The threshold is three standard errors of that mean, computed from the run's own spread, so asking for more replicates tightens the test rather than leaving it loose.",
};

const UTILIZATION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "utilization_matches_traffic_intensity",
    description: "The across-replicate mean server utilization against rho = lambda/mu. Independent of the queue-length check: a simulation can get the mean occupancy right while mis-counting how much of the time the server was busy at all.",
};

const LITTLE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "littles_law_holds",
    description: "L = lambda * W, relating two quantities the run measured to each other and to no closed form at all. A simulation with a systematic error in both that happened to match the textbook would still have to satisfy this, and one whose bookkeeping of arrivals against departures disagreed would fail it while matching both formulas.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.stochastic.mm1_queue"),
    display_name: "M/M/1 queue",
    category: CapabilityCategory::Stochastic,
    source_crate: "scirust-sim",
    summary: "A single-server queue simulated many times over, reporting the distribution of its own steady-state statistics and checking them against the textbook answers and against Little's law.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::InherentlyStochasticRecordedSeed,
    domain: RunDomain::Ensemble,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[DISCRETE_EVENT],
    parameters: &[ARRIVAL_RATE, SERVICE_RATE],
    // A queue starts empty. There is no state for a scenario to supply.
    initial_state: &[],
    outputs: &[
        OutputDescriptor {
            id: "queue_length",
            display_name: "Time-average number in system",
            unit: "1",
            description: "One realisation's time-average occupancy, queue plus the one in service.",
        },
        OutputDescriptor {
            id: "utilization",
            display_name: "Server utilization",
            unit: "1",
            description: "Fraction of the horizon the server was busy.",
        },
        OutputDescriptor {
            id: "mean_sojourn",
            display_name: "Mean time in system",
            unit: "s",
            description: "Average wait plus service, over the customers who finished.",
        },
        OutputDescriptor {
            id: "served",
            display_name: "Customers served",
            unit: "1",
            description: "How many completed service within the horizon.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[QUEUE_LENGTH_CHECK, UTILIZATION_CHECK, LITTLE_CHECK],
    },
};

/// The `sim.stochastic.mm1_queue` adapter.
#[derive(Debug, Default)]
pub struct Mm1QueueAdapter;

impl CapabilityAdapter for Mm1QueueAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["arrival_rate", "service_rate"],
        ));
        // A queue starts empty; there is nothing to supply.
        errors.extend(check_unknown_state_fields(scenario, &[]));
        for e in [
            resolve_model_scalar(scenario, &ARRIVAL_RATE).err(),
            resolve_model_scalar(scenario, &SERVICE_RATE).err(),
            resolve_solver(scenario, DESCRIPTOR.supported_solvers).err(),
            resolve_replicates(scenario, DESCRIPTOR.determinism).err(),
            resolve_seed(scenario).err(),
            resolve_backend_kind(scenario, &DESCRIPTOR).err(),
            resolve_precision(scenario, &DESCRIPTOR).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }

        // An unstable queue is refused rather than run. `MM1Queue::simulate`
        // stays well defined at rho >= 1 — the occupancy simply grows — but
        // every one of this capability's checks compares against a
        // steady-state formula that does not exist there, so the run would
        // produce numbers with nothing to say about them.
        if let (Ok(lambda), Ok(mu)) = (
            resolve_model_scalar(scenario, &ARRIVAL_RATE),
            resolve_model_scalar(scenario, &SERVICE_RATE),
        )
        {
            if lambda >= mu
            {
                errors.push(scirust_studio_command::CatalogedError {
                    code: CODE_UNSTABLE_QUEUE,
                    title: "Unstable queue".to_string(),
                    explanation: format!(
                        "arrival_rate = {lambda} is not below service_rate = {mu}, so the \
                         traffic intensity is {:.4} and the queue grows without bound. The \
                         simulation would still run, but L = rho/(1-rho) and W = 1/(mu-lambda) \
                         do not exist, so every check this capability performs would be \
                         comparing against nothing",
                        lambda / mu
                    ),
                    recoverable: true,
                    suggested_action: Some(
                        "lower `arrival_rate` below `service_rate`, or raise the service rate"
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
        let lambda = resolve_model_scalar(s, &ARRIVAL_RATE).expect("validated");
        let mu = resolve_model_scalar(s, &SERVICE_RATE).expect("validated");
        let model = MM1Queue::new(lambda, mu)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
        let seed = resolve_seed(s).expect("validated");
        let replicates = resolve_replicates(s, DESCRIPTOR.determinism).expect("validated");

        let t0 = s
            .solver
            .start
            .to_quantity("solver.start")
            .expect("validated")
            .value;
        let horizon = s
            .solver
            .end
            .to_quantity("solver.end")
            .expect("validated")
            .value
            - t0;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        // The realisation is the unit of progress and of cancellation: a
        // discrete-event simulation cannot be interrupted part-way and still
        // report a meaningful time-average, so the atom is the whole run.
        let seeds = replicate_seeds(seed, replicates);
        let mut queue_length = Vec::with_capacity(replicates);
        let mut utilization = Vec::with_capacity(replicates);
        let mut sojourn = Vec::with_capacity(replicates);
        let mut served = Vec::with_capacity(replicates);
        for (index, replicate_seed) in seeds.iter().enumerate()
        {
            if control.is_cancelled()
            {
                sink.emit(RunEvent::Cancelled);
                return Err(ExecutionError::Cancelled);
            }
            let stats = model
                .simulate(horizon, *replicate_seed)
                .map_err(|e| ExecutionError::Numerical(e.to_string()))?;
            queue_length.push(stats.time_average_in_system);
            utilization.push(stats.utilization);
            sojourn.push(stats.mean_sojourn);
            served.push(stats.served as f64);
            sink.emit(RunEvent::Progress {
                fraction: (index + 1) as f64 / replicates as f64,
                t: index as f64,
            });
        }

        let rho = model.traffic_intensity();
        let exact_length = rho / (1.0 - rho);
        let exact_sojourn = 1.0 / (mu - lambda);

        let mut warnings = Vec::new();
        if replicates < 2
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: "a single realisation has no spread, so every check's threshold falls \
                          back to the sample itself rather than to a standard error. Ask for \
                          `experiment.replicates` in the tens or hundreds to get a test with \
                          any power."
                    .to_string(),
            });
        }
        let idle: usize = served.iter().filter(|n| **n < 1.0).count();
        if idle > 0
        {
            warnings.push(RunWarning {
                category: WarningCategory::Numerical,
                message: format!(
                    "{idle} of {replicates} realisations finished no customer at all, so their \
                     mean sojourn is reported as zero and drags Little's law downward. Lengthen \
                     `solver.end`."
                ),
            });
        }

        let length_check = mean_check(
            QUEUE_LENGTH_CHECK.id,
            "the time-average number in the system",
            &queue_length,
            exact_length,
        );
        let utilization_check = mean_check(
            UTILIZATION_CHECK.id,
            "the server utilization",
            &utilization,
            rho,
        );
        let little = littles_law_check(&queue_length, &sojourn, lambda);

        let distributions = vec![
            histogram(
                "queue_length",
                "Time-average number in system",
                "1",
                &queue_length,
            ),
            histogram("utilization", "Server utilization", "1", &utilization),
            histogram("mean_sojourn", "Mean time in system", "s", &sojourn),
        ];

        let axis_values: Vec<f64> = (0..replicates).map(|i| i as f64).collect();
        let series = |id: &str, display_name: &str, unit: &str, values: Vec<f64>| Series {
            id: id.to_string(),
            display_name: display_name.to_string(),
            unit: unit.to_string(),
            axis_id: REPLICATE_AXIS_ID.to_string(),
            role: SeriesRole::EnsembleMember,
            values,
        };

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: REPLICATE_AXIS_ID.to_string(),
                steps: replicates - 1,
                t_start: axis_values[0],
                t_end: *axis_values.last().expect("at least one replicate"),
            },
            axes: vec![Axis {
                id: REPLICATE_AXIS_ID.to_string(),
                display_name: "replicate".to_string(),
                unit: "1".to_string(),
                monotonicity: AxisMonotonicity::StrictlyIncreasing,
                values: axis_values,
            }],
            series: vec![
                series(
                    "queue_length",
                    "Time-average number in system",
                    "1",
                    queue_length.clone(),
                ),
                series(
                    "utilization",
                    "Server utilization",
                    "1",
                    utilization.clone(),
                ),
                series("mean_sojourn", "Mean time in system", "s", sojourn.clone()),
                series("served", "Customers served", "1", served.clone()),
            ],
            fields: vec![],
            distributions,
            metrics: vec![
                Metric {
                    id: "traffic_intensity".to_string(),
                    display_name: "Traffic intensity rho".to_string(),
                    value: MetricValue::Scalar(rho),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "exact_queue_length".to_string(),
                    display_name: "Exact mean number in system L".to_string(),
                    value: MetricValue::Scalar(exact_length),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "exact_sojourn".to_string(),
                    display_name: "Exact mean time in system W".to_string(),
                    value: MetricValue::Scalar(exact_sojourn),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "measured_queue_length".to_string(),
                    display_name: "Ensemble mean number in system".to_string(),
                    value: MetricValue::Scalar(mean(&queue_length)),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "measured_sojourn".to_string(),
                    display_name: "Ensemble mean time in system".to_string(),
                    value: MetricValue::Scalar(mean(&sojourn)),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "replicates".to_string(),
                    display_name: "Realisations drawn".to_string(),
                    value: MetricValue::Integer(replicates as i64),
                    unit: None,
                },
                Metric {
                    id: "total_served".to_string(),
                    display_name: "Customers served across the ensemble".to_string(),
                    value: MetricValue::Integer(served.iter().sum::<f64>() as i64),
                    unit: None,
                },
            ],
            warnings,
            verifications: vec![length_check, utilization_check, little],
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
                seed: Some(seed),
            },
        };
        crate::result::validate_result(&result)
            .map_err(|d| ExecutionError::Internal(crate::result::describe_defects(&d)))?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

/// The arithmetic mean, or `NaN` for an empty sample.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// The standard error of the mean of `values`.
///
/// `None` for fewer than two samples, because one realisation has no spread
/// and inventing a zero would turn every check into an equality test against
/// a random number.
fn standard_error(values: &[f64]) -> Option<f64> {
    if values.len() < 2
    {
        return None;
    }
    let m = mean(values);
    let variance =
        values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64;
    Some((variance / values.len() as f64).sqrt())
}

/// A histogram of `values`, spanning their own range with a little padding.
fn histogram(id: &str, display_name: &str, unit: &str, values: &[f64]) -> Distribution {
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // A degenerate sample — one realisation, or every one identical — has no
    // range to bin over, so give it a nominal one rather than a zero-width
    // set of edges the validator would reject.
    let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi > lo
    {
        let pad = (hi - lo) * BIN_PADDING;
        (lo - pad, hi + pad)
    }
    else if lo.is_finite()
    {
        (lo - 0.5, lo + 0.5)
    }
    else
    {
        (0.0, 1.0)
    };
    Distribution::from_samples(id, display_name, unit, values, lo, hi, BIN_COUNT)
}

/// An ensemble mean against a closed form, to within its own standard error.
fn mean_check(id: &str, what: &str, values: &[f64], exact: f64) -> VerificationResult {
    let measured = mean(values);
    let error = (measured - exact).abs();
    let Some(se) = standard_error(values)
    else
    {
        return VerificationResult {
            id: id.to_string(),
            status: VerificationStatus::NotApplicable,
            measured: Some(error),
            threshold: None,
            explanation: format!(
                "{what} came out at {measured:.6} against the closed-form {exact:.6}, but a \
                 single realisation has no spread to compare that difference against"
            ),
        };
    };
    let threshold = se * SIGMAS;
    VerificationResult {
        id: id.to_string(),
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
            "{what} averaged {measured:.6} across {} realisations against the closed-form \
             {exact:.6}, a difference of {error:.3e}; the standard error of that mean is \
             {se:.3e}, so the {SIGMAS}-sigma band is {threshold:.3e}",
            values.len()
        ),
    }
}

/// `L = lambda * W`, from two things the run measured.
///
/// The threshold is the standard error of the difference. The two sample
/// means are computed from the *same* realisations and are strongly positively
/// correlated — a busy realisation has both a long queue and a long wait — so
/// treating them as independent and adding variances would over-state the
/// band. The per-replicate residual `L_i - lambda * W_i` is formed first and
/// its own standard error taken, which carries the correlation exactly rather
/// than assuming anything about it.
fn littles_law_check(queue_length: &[f64], sojourn: &[f64], lambda: f64) -> VerificationResult {
    let residuals: Vec<f64> = queue_length
        .iter()
        .zip(sojourn.iter())
        .map(|(l, w)| l - lambda * w)
        .collect();
    let measured = mean(&residuals);
    let Some(se) = standard_error(&residuals)
    else
    {
        return VerificationResult {
            id: LITTLE_CHECK.id.to_string(),
            status: VerificationStatus::NotApplicable,
            measured: Some(measured.abs()),
            threshold: None,
            explanation: "a single realisation gives no spread to test Little's law against"
                .to_string(),
        };
    };
    let threshold = se * SIGMAS;
    VerificationResult {
        id: LITTLE_CHECK.id.to_string(),
        status: if measured.abs() <= threshold
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(measured.abs()),
        threshold: Some(threshold),
        explanation: format!(
            "L - lambda*W averaged {measured:.3e} over {} realisations, against a \
             {SIGMAS}-sigma band of {threshold:.3e} on the residual's own standard error of \
             {se:.3e}; this relates two measured quantities and mentions no closed form at all",
            residuals.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{CollectingEventSink, NullEventSink};
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/mm1_queue.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = Mm1QueueAdapter;
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
            .unwrap_or_else(|| panic!("no check {id}"))
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
            MetricValue::Integer(v) => v as f64,
            ref other => panic!("{id} is not numeric: {other:?}"),
        }
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        assert_eq!(parse_toml(TUTORIAL).unwrap().capability.id, DESCRIPTOR.id.0);
    }

    /// Stochastic, and still reproducible: the same seed gives the same
    /// ensemble bit for bit. That is the whole content of
    /// `InherentlyStochasticRecordedSeed`.
    #[test]
    fn the_same_seed_gives_a_bit_identical_ensemble() {
        let (a, b) = (run(), run());
        assert!(
            a.series[0]
                .values
                .iter()
                .zip(b.series[0].values.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert_eq!(a.provenance.seed, b.provenance.seed);
    }

    /// And a different seed gives a different one, or the seed is decoration.
    #[test]
    fn a_different_seed_gives_a_different_ensemble() {
        let base = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.seed = Some(scenario.experiment.seed.unwrap() ^ 0x5eed);
        let other = run_scenario(&scenario);
        assert!(
            base.series[0]
                .values
                .iter()
                .zip(other.series[0].values.iter())
                .any(|(x, y)| x != y)
        );
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let validated = Mm1QueueAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let mut sink = CollectingEventSink::new();
        assert!(matches!(
            Mm1QueueAdapter.execute(&validated, &control, &mut sink),
            Err(ExecutionError::Cancelled)
        ));
        assert!(sink.events().contains(&RunEvent::Cancelled));
    }

    #[test]
    fn the_mean_queue_length_matches_the_closed_form() {
        let r = run();
        let v = check(&r, QUEUE_LENGTH_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn the_utilization_matches_the_traffic_intensity() {
        let r = run();
        let v = check(&r, UTILIZATION_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn littles_law_holds() {
        let r = run();
        let v = check(&r, LITTLE_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The reason this capability exists: a result that is a distribution,
    /// with `n + 1` edges for `n` bins and nothing outside its own range.
    #[test]
    fn it_carries_real_histograms() {
        let r = run();
        assert_eq!(r.distributions.len(), 3);
        let replicates = metric(&r, "replicates") as u64;
        for d in &r.distributions
        {
            assert_eq!(
                d.edges.len(),
                d.counts.len() + 1,
                "{}: a histogram needs one more edge than bin",
                d.id
            );
            assert_eq!(d.total(), replicates, "{}", d.id);
            assert_eq!(
                d.underflow + d.overflow,
                0,
                "{}: the bins are built from the sample's own range, so nothing should fall \
                 outside them",
                d.id
            );
            // Several bins occupied, or the "distribution" is a spike and
            // shows nothing a mean would not have.
            let occupied = d.counts.iter().filter(|c| **c > 0).count();
            assert!(occupied >= 4, "{} occupied only {occupied} bins", d.id);
        }
    }

    /// The estimated mean must agree with the exact one to within half a bin,
    /// which is the accuracy the type's own documentation claims for it.
    #[test]
    fn the_binned_mean_is_within_half_a_bin_of_the_real_one() {
        let r = run();
        let d = r
            .distributions
            .iter()
            .find(|d| d.id == "queue_length")
            .expect("queue_length");
        let width = d.edges[1] - d.edges[0];
        let estimated = d.estimated_mean().expect("occupied");
        let exact = metric(&r, "measured_queue_length");
        assert!(
            (estimated - exact).abs() <= width * 0.5,
            "binned {estimated} against actual {exact}, bin width {width}"
        );
    }

    /// The finite-horizon bias is transient, not a wrong model: a longer
    /// horizon must move the answer toward the closed form. This is the
    /// evidence that the residual is what the module header says it is.
    #[test]
    fn a_longer_horizon_moves_toward_the_closed_form() {
        let short = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.end.value *= 8.0;
        let long = run_scenario(&scenario);
        let miss = |r: &RunResult| {
            (metric(r, "measured_queue_length") - metric(r, "exact_queue_length")).abs()
        };
        assert!(
            miss(&long) < miss(&short),
            "an eight-times-longer horizon did not get closer: {} then {}",
            miss(&short),
            miss(&long)
        );
    }

    /// An unstable queue has no steady state for anything to be checked
    /// against, so it is refused rather than run into a comparison with a
    /// formula that does not exist.
    #[test]
    fn an_unstable_queue_is_refused() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        let service = scenario.model["service_rate"].value;
        scenario
            .model
            .get_mut("arrival_rate")
            .expect("arrival_rate")
            .value = service * 1.1;
        let report = Mm1QueueAdapter
            .validate(&scenario)
            .expect_err("rho >= 1 has no steady state");
        assert!(report.errors.iter().any(|e| e.code == CODE_UNSTABLE_QUEUE));
    }

    /// A stochastic capability without a seed is refused, as every one is.
    #[test]
    fn a_missing_seed_is_refused() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.seed = None;
        assert!(Mm1QueueAdapter.validate(&scenario).is_err());
    }

    /// A single realisation cannot support a standard error, and saying so is
    /// different from dividing by zero and reporting a pass.
    #[test]
    fn one_replicate_abstains_rather_than_inventing_a_threshold() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.replicates = Some(1);
        let r = run_scenario(&scenario);
        for id in [QUEUE_LENGTH_CHECK.id, UTILIZATION_CHECK.id, LITTLE_CHECK.id]
        {
            assert_eq!(
                check(&r, id).status,
                VerificationStatus::NotApplicable,
                "{id}"
            );
        }
        assert!(!r.warnings.is_empty(), "it must say why");
    }

    #[test]
    fn the_standard_error_falls_as_the_root_of_the_sample_size() {
        let few: Vec<f64> = (0..64).map(|i| (i % 7) as f64).collect();
        let many: Vec<f64> = (0..256).map(|i| (i % 7) as f64).collect();
        let ratio = standard_error(&few).unwrap() / standard_error(&many).unwrap();
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "quadrupling the sample changed the standard error by {ratio}, expected 2"
        );
    }
}
