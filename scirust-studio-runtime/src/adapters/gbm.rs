//! `sim.stochastic.geometric_brownian_motion` — the log-normal price process
//! (`scirust_sim::stochastic::gbm_path`).
//!
//! `dS = mu*S*dt + sigma*S*dW`. The model behind Black-Scholes and behind
//! every "this grows at a rate but wanders" story: proportional drift plus
//! proportional noise, so the *logarithm* is an ordinary Brownian motion with
//! drift and the process itself can never reach zero.
//!
//! `scirust-sim` samples it with the **exact** log-normal transition
//! `S_{k+1} = S_k * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)`, not with an
//! Euler-Maruyama step. That matters here more than it usually does: there is
//! no discretisation error to bound, so every threshold below is purely
//! statistical and nothing has to be set aside for the integrator.
//!
//! # Two checks that catch opposite mistakes
//!
//! **`mean_grows_at_the_drift_rate`** compares the across-replicate mean of
//! `S_T` against `S_0 * exp(mu*T)`. Note the drift in that expectation is
//! `mu`, not `mu - sigma^2/2`: the latter is the drift of the *logarithm*,
//! and confusing the two is the single most common error in an
//! implementation of this process. A run that used `mu - sigma^2/2` in the
//! exponent of the mean would fail this check and pass the next one.
//!
//! **`log_returns_have_the_stated_variance`** compares the sample variance of
//! `ln(S_T/S_0)` against `sigma^2 * T`, which the exact transition makes an
//! identity rather than an approximation. A run with the wrong volatility
//! fails this and — because the mean of `S_T` does not depend on `sigma` at
//! all — sails through the first.
//!
//! Both thresholds are the **standard error of the statistic being tested**,
//! computed from the run's own sample. A variance's standard error is
//! `sigma^2 * sqrt(2/(n-1))` for a normal sample, which is what is used, and
//! the log returns of this process are exactly normal.
//!
//! # The terminal distribution is the point of the picture
//!
//! The result carries a histogram of `S_T`, and it is deliberately of `S_T`
//! rather than of `ln(S_T)`: the visible right skew is the thing about this
//! process that a mean and a band do not convey, and the reason a symmetric
//! error bar on a price is misleading.

use scirust_sim::stochastic::gbm_path;
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind, RunDomain,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::ensemble::{EnsembleAccumulator, MAX_RETAINED_MEMBERS, replicate_seeds};
use crate::result::{
    Axis, AxisMonotonicity, Distribution, Metric, MetricValue, RESULT_SCHEMA_VERSION,
    RunProvenance, RunResult, RunSummary, RunWarning, Series, SeriesRole, TIME_AXIS_ID,
    VerificationResult, VerificationStatus, WarningCategory,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_seed, resolve_solver,
};

/// How many standard errors a statistic may sit from its exact value.
const SIGMAS: f64 = 3.0;

/// Bins in the terminal-value histogram.
const BIN_COUNT: usize = 32;

/// Padding on the histogram's range, as a fraction of the observed spread.
const BIN_PADDING: f64 = 0.05;

/// How many sigmas wide the shipped spread band is.
const BAND_SIGMAS: f64 = 1.0;

const EXACT_TRANSITION: SolverDescriptor = SolverDescriptor {
    id: "exact_transition",
    summary: "Samples the process's exact log-normal transition rather than discretising the stochastic differential equation. There is no truncation error to bound: `solver.step` is the sampling interval, not an approximation parameter.",
    fixed_step: true,
    adaptive_tolerance: false,
    reports_progress: true,
};

const INITIAL_VALUE: FieldDescriptor = FieldDescriptor {
    canonical_name: "initial_value",
    display_name: "Initial value",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Value at t = start. Must be positive: the process is multiplicative, so it can approach zero but never reach or cross it.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 322),
};

const DRIFT: FieldDescriptor = FieldDescriptor {
    canonical_name: "drift",
    display_name: "Drift rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The rate the EXPECTATION grows at. The median grows at drift - volatility^2/2, which is smaller; conflating the two is the classic error in this model.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 323),
};

const VOLATILITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "volatility",
    display_name: "Volatility",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Standard deviation of the log return per unit time. Zero makes the process the deterministic exponential, which is a useful thing to be able to ask for.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 324),
};

const MEAN_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "mean_grows_at_the_drift_rate",
    description: "The across-replicate mean of the terminal value against S0*exp(drift*T). The drift in that expectation is the drift itself, not drift - volatility^2/2, which is the growth rate of the median; an implementation that confused them fails here and passes the variance check.",
};

const VARIANCE_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "log_returns_have_the_stated_variance",
    description: "The sample variance of ln(S_T/S_0) against volatility^2 * T, which the exact log-normal transition makes an identity rather than an approximation. The threshold is that variance's own standard error for a normal sample, and the log returns of this process are exactly normal. An implementation with the wrong volatility fails here and passes the mean check, since the mean of S_T does not depend on volatility at all.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.stochastic.geometric_brownian_motion"),
    display_name: "Geometric Brownian motion",
    category: CapabilityCategory::Stochastic,
    source_crate: "scirust-sim",
    summary: "The log-normal growth process, sampled from its exact transition, with the right-skewed terminal distribution a mean and a band cannot convey.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::InherentlyStochasticRecordedSeed,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[EXACT_TRANSITION],
    parameters: &[INITIAL_VALUE, DRIFT, VOLATILITY],
    // The starting value is a model parameter rather than an initial state:
    // `gbm_path` takes it as one, and splitting it out would put half the
    // process's definition in a different section of the scenario.
    initial_state: &[],
    outputs: &[
        OutputDescriptor {
            id: "ensemble_mean",
            display_name: "Ensemble mean",
            unit: "1",
            description: "The across-replicate mean at each time.",
        },
        OutputDescriptor {
            id: "ensemble_band_lower",
            display_name: "Spread band, lower",
            unit: "1",
            description: "One standard deviation below the mean.",
        },
        OutputDescriptor {
            id: "ensemble_band_upper",
            display_name: "Spread band, upper",
            unit: "1",
            description: "One standard deviation above the mean.",
        },
        OutputDescriptor {
            id: "expected_value",
            display_name: "Expected value",
            unit: "1",
            description: "S0*exp(drift*t), the closed-form expectation.",
        },
        OutputDescriptor {
            id: "median_path",
            display_name: "Median",
            unit: "1",
            description: "S0*exp((drift - volatility^2/2)*t) — visibly below the mean, which is the asymmetry the histogram shows directly.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[MEAN_CHECK, VARIANCE_CHECK],
    },
};

/// The `sim.stochastic.geometric_brownian_motion` adapter.
#[derive(Debug, Default)]
pub struct GeometricBrownianMotionAdapter;

impl CapabilityAdapter for GeometricBrownianMotionAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["initial_value", "drift", "volatility"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &[]));
        for e in [
            resolve_model_scalar(scenario, &INITIAL_VALUE).err(),
            resolve_model_scalar(scenario, &DRIFT).err(),
            resolve_model_scalar(scenario, &VOLATILITY).err(),
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
        let s0 = resolve_model_scalar(s, &INITIAL_VALUE).expect("validated");
        let drift = resolve_model_scalar(s, &DRIFT).expect("validated");
        let volatility = resolve_model_scalar(s, &VOLATILITY).expect("validated");
        let seed = resolve_seed(s).expect("validated");
        let replicates = resolve_replicates(s, DESCRIPTOR.determinism).expect("validated");

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
        let dt = s
            .solver
            .step
            .as_ref()
            .expect("validated: exact_transition declares a fixed step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;
        let steps = (((t1 - t0) / dt).floor() as usize).max(1);
        let span = steps as f64 * dt;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let seeds = replicate_seeds(seed, replicates);
        let mut ensemble = EnsembleAccumulator::new(steps + 1);
        let mut retained: Vec<(u64, Vec<f64>)> = Vec::new();
        let mut terminal = Vec::with_capacity(replicates);
        for (index, replicate_seed) in seeds.iter().enumerate()
        {
            if control.is_cancelled()
            {
                sink.emit(RunEvent::Cancelled);
                return Err(ExecutionError::Cancelled);
            }
            let path = gbm_path(s0, drift, volatility, dt, steps, *replicate_seed)
                .map_err(|e| ExecutionError::Numerical(e.to_string()))?;
            terminal.push(*path.last().expect("steps + 1 values"));
            ensemble
                .push(&path)
                .map_err(|e| ExecutionError::Internal(e.to_string()))?;
            if retained.len() < MAX_RETAINED_MEMBERS
            {
                retained.push((*replicate_seed, path));
            }
            sink.emit(RunEvent::Progress {
                fraction: (index + 1) as f64 / replicates as f64,
                t: t0 + span,
            });
        }

        let coordinates: Vec<f64> = (0..=steps).map(|k| t0 + k as f64 * dt).collect();
        let expected: Vec<f64> = coordinates
            .iter()
            .map(|t| s0 * (drift * (t - t0)).exp())
            .collect();
        let median: Vec<f64> = coordinates
            .iter()
            .map(|t| s0 * ((drift - 0.5 * volatility * volatility) * (t - t0)).exp())
            .collect();
        let (lower, upper) = ensemble.band(BAND_SIGMAS);

        let log_returns: Vec<f64> = terminal.iter().map(|v| (v / s0).ln()).collect();
        let measured_variance = sample_variance(&log_returns);
        let mean_result = mean_check(&terminal, s0 * (drift * span).exp());
        let variance_result = variance_check(&log_returns, volatility * volatility * span);

        let mut warnings = Vec::new();
        if replicates < 2
        {
            warnings.push(RunWarning {
                category: WarningCategory::Input,
                message: "a single realisation has no spread, so neither check has a standard \
                          error to compare against. Ask for `experiment.replicates` in the \
                          hundreds to get a test with any power."
                    .to_string(),
            });
        }

        let mut series = vec![
            Series {
                id: "ensemble_mean".to_string(),
                display_name: "Ensemble mean".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleMean,
                values: ensemble.mean().to_vec(),
            },
            Series {
                id: "ensemble_band_lower".to_string(),
                display_name: "Spread band, lower".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleBandLower,
                values: lower,
            },
            Series {
                id: "ensemble_band_upper".to_string(),
                display_name: "Spread band, upper".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleBandUpper,
                values: upper,
            },
            Series {
                id: "expected_value".to_string(),
                display_name: "Expected value".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Reference,
                values: expected,
            },
            Series {
                id: "median_path".to_string(),
                display_name: "Median".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Reference,
                values: median,
            },
        ];
        for (replicate_seed, path) in retained
        {
            series.push(Series {
                id: format!("realisation_{replicate_seed}"),
                display_name: format!("Realisation (seed {replicate_seed})"),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleMember,
                values: path,
            });
        }

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                axis_id: TIME_AXIS_ID.to_string(),
                steps,
                t_start: coordinates[0],
                t_end: *coordinates.last().expect("steps + 1 coordinates"),
            },
            axes: vec![Axis {
                id: TIME_AXIS_ID.to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
                monotonicity: AxisMonotonicity::StrictlyIncreasing,
                values: coordinates,
            }],
            series,
            fields: vec![],
            distributions: vec![terminal_histogram(&terminal)],
            // A variance a single realisation cannot have is reported by its
            // absence rather than as a NaN: a metric a reader can see is a
            // number they will use.
            metrics: measured_variance
                .map(|v| Metric {
                    id: "log_return_variance".to_string(),
                    display_name: "Measured log-return variance".to_string(),
                    value: MetricValue::Scalar(v),
                    unit: Some("1".to_string()),
                })
                .into_iter()
                .chain(vec![
                    Metric {
                        id: "expected_terminal".to_string(),
                        display_name: "Expected terminal value".to_string(),
                        value: MetricValue::Scalar(s0 * (drift * span).exp()),
                        unit: Some("1".to_string()),
                    },
                    Metric {
                        id: "median_terminal".to_string(),
                        display_name: "Median terminal value".to_string(),
                        value: MetricValue::Scalar(
                            s0 * ((drift - 0.5 * volatility * volatility) * span).exp(),
                        ),
                        unit: Some("1".to_string()),
                    },
                    Metric {
                        id: "measured_terminal_mean".to_string(),
                        display_name: "Measured terminal mean".to_string(),
                        value: MetricValue::Scalar(mean(&terminal)),
                        unit: Some("1".to_string()),
                    },
                    Metric {
                        id: "exact_log_return_variance".to_string(),
                        display_name: "Exact log-return variance".to_string(),
                        value: MetricValue::Scalar(volatility * volatility * span),
                        unit: Some("1".to_string()),
                    },
                    Metric {
                        id: "replicates".to_string(),
                        display_name: "Realisations drawn".to_string(),
                        value: MetricValue::Integer(replicates as i64),
                        unit: None,
                    },
                    Metric {
                        id: "realisations_retained".to_string(),
                        display_name: "Realisations kept as exemplars".to_string(),
                        value: MetricValue::Integer(MAX_RETAINED_MEMBERS.min(replicates) as i64),
                        unit: None,
                    },
                ])
                .collect(),
            warnings,
            verifications: vec![mean_result, variance_result],
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

/// The unbiased sample variance, or `None` for fewer than two samples.
fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2
    {
        return None;
    }
    let m = mean(values);
    Some(values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64)
}

/// A histogram of the terminal values, over their own range.
fn terminal_histogram(terminal: &[f64]) -> Distribution {
    let lo = terminal.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = terminal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi > lo
    {
        let pad = (hi - lo) * BIN_PADDING;
        (lo - pad, hi + pad)
    }
    else if lo.is_finite()
    {
        (lo * 0.5, lo * 1.5)
    }
    else
    {
        (0.0, 1.0)
    };
    Distribution::from_samples(
        "terminal_value",
        "Terminal value",
        "1",
        terminal,
        lo,
        hi,
        BIN_COUNT,
    )
}

/// The terminal mean against `S0*exp(drift*T)`.
fn mean_check(terminal: &[f64], exact: f64) -> VerificationResult {
    let measured = mean(terminal);
    let error = (measured - exact).abs();
    let Some(variance) = sample_variance(terminal)
    else
    {
        return VerificationResult {
            id: MEAN_CHECK.id.to_string(),
            status: VerificationStatus::NotApplicable,
            measured: Some(error),
            threshold: None,
            explanation: format!(
                "the terminal value came out at {measured:.6} against the expected {exact:.6}, \
                 but a single realisation has no spread to compare that difference against"
            ),
        };
    };
    let se = (variance / terminal.len() as f64).sqrt();
    let threshold = se * SIGMAS;
    VerificationResult {
        id: MEAN_CHECK.id.to_string(),
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
            "the terminal value averaged {measured:.6} across {} realisations against the exact \
             S0*exp(drift*T) = {exact:.6}, a difference of {error:.3e}; the standard error of \
             that mean is {se:.3e}, so the {SIGMAS}-sigma band is {threshold:.3e}",
            terminal.len()
        ),
    }
}

/// The log-return variance against `volatility^2 * T`.
///
/// The threshold is that statistic's own standard error. For a normal sample
/// the sample variance has standard error `sigma^2 * sqrt(2/(n-1))`, and the
/// log returns of this process are *exactly* normal because the sampler uses
/// the exact transition rather than discretising — so this is the right
/// formula here rather than an approximation borrowed from one.
fn variance_check(log_returns: &[f64], exact: f64) -> VerificationResult {
    let Some(measured) = sample_variance(log_returns)
    else
    {
        return VerificationResult {
            id: VARIANCE_CHECK.id.to_string(),
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: "a single realisation gives no variance to test".to_string(),
        };
    };
    let error = (measured - exact).abs();
    let se = exact * (2.0 / (log_returns.len() - 1) as f64).sqrt();
    let threshold = (se * SIGMAS).max(f64::EPSILON);
    VerificationResult {
        id: VARIANCE_CHECK.id.to_string(),
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
            "the log returns of {} realisations had variance {measured:.6} against the exact \
             volatility^2*T = {exact:.6}, a difference of {error:.3e}; a normal sample's \
             variance has standard error {se:.3e}, so the {SIGMAS}-sigma band is {threshold:.3e}",
            log_returns.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{CollectingEventSink, NullEventSink};
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/geometric_brownian_motion.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = GeometricBrownianMotionAdapter;
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

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let validated = GeometricBrownianMotionAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let mut sink = CollectingEventSink::new();
        assert!(matches!(
            GeometricBrownianMotionAdapter.execute(&validated, &control, &mut sink),
            Err(ExecutionError::Cancelled)
        ));
        assert!(sink.events().contains(&RunEvent::Cancelled));
    }

    #[test]
    fn the_mean_grows_at_the_drift_rate() {
        let r = run();
        let v = check(&r, MEAN_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn the_log_returns_have_the_stated_variance() {
        let r = run();
        let v = check(&r, VARIANCE_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The distinction the module header calls the classic error, asserted
    /// rather than only described: the mean and the median are different
    /// numbers, and the mean is the larger one.
    #[test]
    fn the_mean_sits_above_the_median() {
        let r = run();
        let (mean_terminal, median) = (
            metric(&r, "expected_terminal"),
            metric(&r, "median_terminal"),
        );
        assert!(
            mean_terminal > median * 1.05,
            "the mean {mean_terminal} is not meaningfully above the median {median}, so this \
             run's volatility is too low to show the asymmetry the capability is about"
        );
    }

    /// A run with the wrong exponent in the expectation — using the median's
    /// growth rate for the mean — must fail the mean check. This is what
    /// makes that check worth having rather than a restatement.
    #[test]
    fn the_mean_check_rejects_the_medians_growth_rate() {
        let r = run();
        let terminal_series: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.01).collect();
        // A tight sample around the right answer, tested against the wrong one.
        let verdict = mean_check(&terminal_series, metric(&r, "median_terminal"));
        assert_eq!(verdict.status, VerificationStatus::Failed);
    }

    /// The terminal histogram is the picture the capability exists to draw,
    /// and it must be visibly skewed rather than symmetric.
    #[test]
    fn the_terminal_distribution_is_right_skewed() {
        let r = run();
        let d = r
            .distributions
            .iter()
            .find(|d| d.id == "terminal_value")
            .expect("terminal_value");
        assert_eq!(d.edges.len(), d.counts.len() + 1);
        assert_eq!(d.total(), metric(&r, "replicates") as u64);

        // The mode sits left of centre for a right-skewed distribution.
        let peak_bin = d
            .counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| **c)
            .expect("occupied")
            .0;
        assert!(
            peak_bin < d.counts.len() / 2,
            "the busiest bin is {peak_bin} of {}, which is not a right-skewed shape",
            d.counts.len()
        );
    }

    /// Zero volatility makes this the deterministic exponential, which is a
    /// legitimate thing to ask for and must not divide by a zero spread.
    #[test]
    fn zero_volatility_is_the_deterministic_exponential() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario
            .model
            .get_mut("volatility")
            .expect("volatility")
            .value = 0.0;
        let r = run_scenario(&scenario);
        let measured = metric(&r, "measured_terminal_mean");
        let expected = metric(&r, "expected_terminal");
        assert!(
            (measured - expected).abs() < expected * 1e-12,
            "with no volatility the mean should be exactly the exponential: {measured} against \
             {expected}"
        );
        // And the variance check has an exact zero to compare against, so it
        // must not report a spurious failure.
        assert_ne!(
            check(&r, VARIANCE_CHECK.id).status,
            VerificationStatus::Failed,
            "{}",
            check(&r, VARIANCE_CHECK.id).explanation
        );
    }

    #[test]
    fn a_missing_seed_is_refused() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.seed = None;
        assert!(GeometricBrownianMotionAdapter.validate(&scenario).is_err());
    }

    #[test]
    fn one_replicate_abstains_rather_than_inventing_a_threshold() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.replicates = Some(1);
        let r = run_scenario(&scenario);
        for id in [MEAN_CHECK.id, VARIANCE_CHECK.id]
        {
            assert_eq!(
                check(&r, id).status,
                VerificationStatus::NotApplicable,
                "{id}"
            );
        }
        assert!(!r.warnings.is_empty());
    }
}
