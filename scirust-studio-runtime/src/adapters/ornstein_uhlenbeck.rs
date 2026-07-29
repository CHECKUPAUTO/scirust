//! `sim.stochastic.ornstein_uhlenbeck` — a mean-reverting stochastic process
//! (`scirust_sim::stochastic::ou_path`).
//!
//! The first capability in the catalogue whose result is **not** a function of
//! its parameters alone. `dX = θ·(μ − X)·dt + σ·dW` is driven by noise, so a
//! run produces *one sample* from a distribution rather than the answer.
//!
//! Everything about this adapter follows from taking that seriously.
//!
//! # The seed is required, not defaulted
//!
//! Validation refuses a scenario with no `experiment.seed`. A result is only
//! evidence if someone else can obtain it again, and for a single sample the
//! seed is the whole difference between "here is a trajectory" and "here is
//! *the* trajectory these inputs produce". The seed is then recorded in the
//! result's provenance — the first capability for which that field is ever
//! populated.
//!
//! # The verification is statistical, and says so
//!
//! There is no trajectory to compare against: no two samples agree, and
//! neither is wrong. What *is* pinned is the distribution. The stationary law
//! of this process is exactly `N(μ, σ²/(2θ))`, so the check compares the
//! sample moments of the path's tail against those, with a tolerance derived
//! from the sample size and the process's own autocorrelation rather than
//! chosen to make the check pass.
//!
//! # The run verifies its own reproducibility claim
//!
//! A second short path is generated from the recorded seed and compared bit
//! for bit against the start of the first. A provenance field saying "seed
//! 42" is a claim; re-deriving the sample from it inside the same run is
//! evidence.

use scirust_sim::stochastic::ou_path;

use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::ensemble::{EnsembleAccumulator, MAX_RETAINED_MEMBERS, replicate_seeds};
use crate::result::{
    Axis, AxisMonotonicity, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, Series, SeriesRole, TIME_AXIS_ID, VerificationResult, VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_backend_kind,
    resolve_model_scalar, resolve_precision, resolve_replicates, resolve_seed, resolve_solver,
    resolve_state_vector,
};

/// How many relaxation times `1/θ` to discard before measuring the stationary
/// moments.
///
/// After `n` relaxation times the memory of the initial condition has decayed
/// by `e^{-n}`; at five that is a factor of 148, well below the sampling error
/// the check tolerates. Discarding too little would measure the transient and
/// blame the process for it.
const BURN_IN_RELAXATION_TIMES: f64 = 5.0;

/// How many standard errors the sample mean may sit from `μ`.
///
/// Four, not three. A run that fails one time in fifteen thousand is a check;
/// a run that fails one time in three hundred is a nuisance that gets
/// suppressed, and a suppressed check is worse than no check.
const SIGMA_BAND: f64 = 4.0;

/// How many across-replicate standard deviations the shaded band spans.
///
/// Two, not four. This one is not a pass/fail threshold but a picture of
/// where the realisations are: a `±2σ` band contains about 95% of them, which
/// is the reading most people already have for a spread band. `SIGMA_BAND`
/// above answers a different question — how far the *mean of many* may stray
/// before the run is wrong — and the two would be confusing to share.
const BAND_SIGMAS: f64 = 2.0;

const THETA: FieldDescriptor = FieldDescriptor {
    canonical_name: "theta",
    display_name: "Mean-reversion rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate theta at which the process is pulled back to mu, in \
                  dX = theta*(mu - X)*dt + sigma*dW. Its reciprocal is the relaxation time.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 190),
};

const MU: FieldDescriptor = FieldDescriptor {
    canonical_name: "mu",
    display_name: "Long-run mean",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The level mu the process reverts to.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 191),
};

const SIGMA: FieldDescriptor = FieldDescriptor {
    canonical_name: "sigma",
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
    description: "Noise amplitude sigma. Zero makes the process deterministic exponential decay \
                  toward mu.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 192),
};

const VALUE: FieldDescriptor = FieldDescriptor {
    canonical_name: "value",
    display_name: "Initial value",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The process value at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 193),
};

/// Not an ODE integrator.
///
/// `ou_path` steps the process by its **exact** Gaussian transition, so there
/// is no discretisation error to trade against step size the way `rk4` has:
/// the only error in the result is sampling error. Calling this solver `rk4`
/// would invite exactly the wrong intuition about what a smaller step buys.
const EXACT_TRANSITION: SolverDescriptor = SolverDescriptor {
    id: "exact_gaussian_transition",
    summary: "Exact Gaussian transition of the Ornstein-Uhlenbeck process; no discretisation \
              error, only sampling error.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const STATIONARY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "stationary_moments",
    description: "The stationary distribution is exactly N(mu, sigma^2/(2*theta)). After \
                  discarding the transient, the check compares the sample mean and variance of \
                  the path against those, with a tolerance derived from the sample size and the \
                  process's autocorrelation. This is a statistical statement about the \
                  distribution, not a claim about any individual trajectory.",
};

const REPRODUCIBILITY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "reproducible_from_seed",
    description: "The recorded seed is used to re-derive the beginning of the path, which must \
                  match bit for bit. Turns the provenance's reproducibility claim into something \
                  the run itself demonstrates.",
};

const ENSEMBLE_MOMENTS_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "ensemble_moments",
    description: "Only for a run with `experiment.replicates` above 1. Compares the mean and \
                  variance *across* the independent realisations at the final time against the \
                  stationary law N(mu, sigma^2/(2*theta)). Stronger than the single-path check: \
                  the realisations are independent, so n of them carry n samples' worth of \
                  information and the tolerance needs no autocorrelation correction.",
};

const ENSEMBLE_DERIVATION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "ensemble_derived_from_seed",
    description: "Only for a run with `experiment.replicates` above 1. Re-derives the \
                  replicates' seeds from the single recorded seed, checks them against the ones \
                  the run consumed, and re-runs one realisation from its derived seed to compare \
                  bit for bit. Turns the claim that one number regenerates the whole ensemble \
                  into evidence.",
};

/// The capability descriptor for `sim.stochastic.ornstein_uhlenbeck`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.stochastic.ornstein_uhlenbeck"),
    display_name: "Ornstein-Uhlenbeck process",
    category: CapabilityCategory::Stochastic,
    source_crate: "scirust-sim",
    summary: "A mean-reverting stochastic process. One sample from a distribution, reproducible \
              from its recorded seed and verified against the exact stationary moments.",
    maturity: CapabilityMaturity::Stable,
    // The first capability in the catalogue that is not bit-identical from
    // its parameters alone. Same seed and same binary reproduce it exactly;
    // a different seed is a different, equally valid sample.
    determinism: DeterminismClass::InherentlyStochasticRecordedSeed,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[EXACT_TRANSITION],
    parameters: &[THETA, MU, SIGMA],
    initial_state: &[VALUE],
    outputs: &[
        OutputDescriptor {
            id: "value",
            display_name: "Value",
            unit: "1",
            description: "The sampled path.",
        },
        OutputDescriptor {
            id: "mean",
            display_name: "Long-run mean",
            unit: "1",
            description: "The level the process reverts to, drawn as a constant line so the \
                          sample can be read against it.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[
            STATIONARY_CHECK,
            REPRODUCIBILITY_CHECK,
            ENSEMBLE_MOMENTS_CHECK,
            ENSEMBLE_DERIVATION_CHECK,
        ],
    },
};

/// The `sim.stochastic.ornstein_uhlenbeck` adapter.
#[derive(Debug, Default)]
pub struct OrnsteinUhlenbeckAdapter;

impl CapabilityAdapter for OrnsteinUhlenbeckAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["theta", "mu", "sigma"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["value"]));
        for e in [
            resolve_model_scalar(scenario, &THETA).err(),
            resolve_model_scalar(scenario, &MU).err(),
            resolve_model_scalar(scenario, &SIGMA).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        if let Err(e) = resolve_state_vector(scenario, &VALUE)
        {
            errors.push(e);
        }
        if let Err(e) = resolve_solver(scenario, DESCRIPTOR.supported_solvers)
        {
            errors.push(e);
        }
        // The constraint that makes this capability reproducible at all.
        if let Err(e) = resolve_seed(scenario)
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
        let theta = resolve_model_scalar(s, &THETA).expect("validated");
        let mu = resolve_model_scalar(s, &MU).expect("validated");
        let sigma = resolve_model_scalar(s, &SIGMA).expect("validated");
        let x0 = resolve_state_vector(s, &VALUE).expect("validated")[0];
        let seed = resolve_seed(s).expect("validated");

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
            .expect("validated: this solver is fixed-step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let span = t1 - t0;
        // `is_sign_positive` would accept a NaN span; comparing the other way
        // round rejects it, which is what a time span with no ordering
        // deserves.
        if span <= 0.0 || dt <= 0.0 || !span.is_finite() || !dt.is_finite()
        {
            return Err(ExecutionError::InvalidModelState(format!(
                "the time span must be positive and the step must be positive: \
                 start = {t0}, end = {t1}, step = {dt}"
            )));
        }
        let steps = (span / dt).round() as usize;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let replicates = resolve_replicates(s, DESCRIPTOR.determinism).expect("validated");
        let seeds = replicate_seeds(seed, replicates);

        // `ou_path` samples one whole realisation per call, seeding its
        // generator once. A *single* realisation cannot be chunked: each call
        // re-seeds, so splitting the span would produce a different — and
        // differently distributed — sample.
        //
        // An ensemble can be, because the realisation is its atom. Progress
        // is counted in completed realisations and cancellation is honoured
        // between them. A one-replicate run has exactly one indivisible unit
        // of work and so reports a single step from nothing to done — which
        // is not an invented fraction but an accurate description of a job
        // with one part. See `docs/studio/adr/0008-ensembles.md`.
        let mut ensemble = EnsembleAccumulator::new(steps + 1);
        let mut retained: Vec<(u64, Vec<f64>)> = Vec::new();
        // Bounded to the 20 emissions per run that `RunEvent::Progress`
        // promises, however many replicates were asked for.
        let progress_every = replicates.div_ceil(20).max(1);
        let t_end = t0 + steps as f64 * dt;

        for (k, &replicate_seed) in seeds.iter().enumerate()
        {
            if control.is_cancelled()
            {
                sink.emit(RunEvent::Cancelled);
                return Err(ExecutionError::Cancelled);
            }
            let path = ou_path(x0, theta, mu, sigma, dt, steps, replicate_seed)
                .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
            ensemble
                .push(&path)
                .map_err(|e| ExecutionError::Internal(e.to_string()))?;
            if retained.len() < MAX_RETAINED_MEMBERS
            {
                retained.push((replicate_seed, path));
            }
            if (k + 1) % progress_every == 0 || k + 1 == replicates
            {
                sink.emit(RunEvent::Progress {
                    fraction: (k + 1) as f64 / replicates as f64,
                    t: t_end,
                });
            }
        }

        let samples = steps + 1;
        let coordinates: Vec<f64> = (0..samples).map(|k| t0 + k as f64 * dt).collect();
        let first_path = &retained
            .first()
            .expect("at least one replicate is always drawn")
            .1;

        // The single-path check is kept whatever the replicate count: it
        // measures a different thing from the across-replicate one (the
        // ergodic average along one realisation, rather than the ensemble
        // average across many) and costs nothing once the path exists.
        let stationary = stationary_check(first_path, dt, theta, mu, sigma);
        let reproducibility = reproducibility_check(x0, theta, mu, sigma, dt, seeds[0], first_path);

        let last = *first_path
            .last()
            .expect("ou_path returns at least the initial value");

        let mut series = Vec::new();
        let mut metrics = Vec::new();
        let mut verifications = vec![stationary, reproducibility];

        if replicates == 1
        {
            series.push(Series {
                id: "value".to_string(),
                display_name: "Value".to_string(),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Trajectory,
                values: first_path.clone(),
            });
        }
        else
        {
            // The summary first, so a consumer that draws series in order
            // puts the ensemble's answer underneath its exemplars rather than
            // hiding it behind them.
            series.push(Series {
                id: "ensemble_mean".to_string(),
                display_name: format!("Ensemble mean of {replicates} realisations"),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleMean,
                values: ensemble.mean().to_vec(),
            });
            let (lower, upper) = ensemble.band(BAND_SIGMAS);
            series.push(Series {
                id: "ensemble_band_lower".to_string(),
                display_name: format!("Ensemble mean minus {BAND_SIGMAS} standard deviations"),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleBandLower,
                values: lower,
            });
            series.push(Series {
                id: "ensemble_band_upper".to_string(),
                display_name: format!("Ensemble mean plus {BAND_SIGMAS} standard deviations"),
                unit: "1".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::EnsembleBandUpper,
                values: upper,
            });
            for (k, (member_seed, member_path)) in retained.iter().enumerate()
            {
                series.push(Series {
                    id: format!("member_{k}"),
                    // The seed is in the label because it is what makes this
                    // particular curve retrievable on its own.
                    display_name: format!("Realisation {k} (seed {member_seed})"),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::EnsembleMember,
                    values: member_path.clone(),
                });
            }
        }

        series.push(Series {
            id: "mean".to_string(),
            display_name: "Long-run mean".to_string(),
            unit: "1".to_string(),
            axis_id: TIME_AXIS_ID.to_string(),
            role: SeriesRole::Reference,
            values: vec![mu; samples],
        });

        if replicates > 1
        {
            metrics.push(Metric {
                id: "replicates".to_string(),
                display_name: "Realisations drawn".to_string(),
                value: MetricValue::Integer(replicates as i64),
                unit: None,
            });
            // Reported even when nothing was dropped, so the number a reader
            // sees is the number of curves in front of them either way.
            metrics.push(Metric {
                id: "retained_members".to_string(),
                display_name: "Realisations kept in the result".to_string(),
                value: MetricValue::Integer(retained.len() as i64),
                unit: None,
            });
            metrics.push(Metric {
                id: "ensemble_final_mean".to_string(),
                display_name: "Ensemble mean at the final time".to_string(),
                value: MetricValue::Scalar(ensemble.mean()[samples - 1]),
                unit: Some("1".to_string()),
            });
            metrics.push(Metric {
                id: "ensemble_final_standard_error".to_string(),
                display_name: "Standard error of that mean".to_string(),
                value: MetricValue::Scalar(ensemble.standard_error()[samples - 1]),
                unit: Some("1".to_string()),
            });
            verifications.push(ensemble_check(&ensemble, t_end - t0, theta, mu, sigma));
            verifications.push(derivation_check(x0, theta, mu, sigma, dt, seed, &retained));
        }

        metrics.extend([
            Metric {
                id: "final_value".to_string(),
                display_name: if replicates == 1
                {
                    "Final value".to_string()
                }
                else
                {
                    "Final value of realisation 0".to_string()
                },
                value: MetricValue::Scalar(last),
                unit: Some("1".to_string()),
            },
            Metric {
                id: "relaxation_time".to_string(),
                display_name: "Relaxation time 1/theta".to_string(),
                value: MetricValue::Scalar(1.0 / theta),
                unit: Some("s".to_string()),
            },
            Metric {
                id: "stationary_stddev".to_string(),
                display_name: "Stationary standard deviation sigma/sqrt(2*theta)".to_string(),
                value: MetricValue::Scalar(stationary_stddev(sigma, theta)),
                unit: Some("1".to_string()),
            },
            Metric {
                id: "seed".to_string(),
                display_name: "Seed".to_string(),
                // Integer, not Scalar: a seed is an identifier, and
                // rendering it as a float would eventually show one in
                // exponential notation, which nobody can type back in.
                value: MetricValue::Integer(seed as i64),
                unit: None,
            },
        ]);

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: s.experiment.name.clone(),
                steps,
                t_start: coordinates[0],
                t_end: *coordinates.last().expect("at least one coordinate"),
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
            metrics,
            warnings: vec![],
            verifications,
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
                // The one capability so far for which this is populated: the
                // result is a sample, and this is what makes it the same
                // sample twice.
                seed: Some(seed),
            },
        };
        crate::result::validate_result(&result)
            .map_err(|d| ExecutionError::Internal(crate::result::describe_defects(&d)))?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

/// The standard deviation of the stationary distribution, `σ/√(2θ)`.
fn stationary_stddev(sigma: f64, theta: f64) -> f64 {
    sigma / (2.0 * theta).sqrt()
}

/// Compare the ensemble's moments at the final time against the exact
/// stationary law.
///
/// This is the check an ensemble exists to make possible, and it is stronger
/// than the single-path one in a specific way. Along one realisation the
/// samples are autocorrelated, so `n` of them carry only about `n·dt·θ/2`
/// samples' worth of information and the tolerance has to be widened to
/// match. Across replicates they are independent by construction, so `n`
/// realisations carry `n` realisations' worth and the standard error is
/// `σ/√(2θ)/√n` with no correction at all. The shipped tutorial's 256
/// replicates therefore pin the mean about sixteen times more tightly than
/// its 39 001 correlated samples do.
///
/// Both tolerances are derived rather than chosen — including the variance
/// band, where the sample variance of `n` normal draws has relative standard
/// deviation `√(2/(n−1))`.
fn ensemble_check(
    ensemble: &EnsembleAccumulator,
    span: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
) -> VerificationResult {
    let id = "ensemble_moments".to_string();
    let not_applicable = |explanation: String| VerificationResult {
        id: id.clone(),
        status: VerificationStatus::NotApplicable,
        measured: None,
        threshold: None,
        explanation,
    };

    if sigma == 0.0
    {
        return not_applicable(
            "sigma = 0: every realisation is the same deterministic decay, so the ensemble has \
             no distribution to compare against"
                .to_string(),
        );
    }
    let n = ensemble.replicates();
    if n < 2
    {
        return not_applicable(
            "an ensemble of one realisation has no across-replicate spread to measure".to_string(),
        );
    }
    // The stationary law only describes the process once the initial
    // condition has been forgotten. Measuring before that and calling the
    // difference a failure would be blaming the process for the transient.
    let relaxation_times = span * theta;
    if relaxation_times < BURN_IN_RELAXATION_TIMES
    {
        return not_applicable(format!(
            "the run spans only {relaxation_times:.1} relaxation times, short of the \
             {BURN_IN_RELAXATION_TIMES} needed before the initial condition is forgotten; the \
             ensemble at the final time is not yet a sample from the stationary law"
        ));
    }

    let last = ensemble.samples() - 1;
    let sample_mean = ensemble.mean()[last];
    let sample_variance = ensemble.variance()[last];

    let stationary_variance = sigma * sigma / (2.0 * theta);
    let standard_error = (stationary_variance / n as f64).sqrt();
    let sigmas = (sample_mean - mu).abs() / standard_error;

    let variance_ratio = sample_variance / stationary_variance;
    let variance_tolerance = SIGMA_BAND * (2.0 / (n as f64 - 1.0)).sqrt();
    let variance_ok = (variance_ratio - 1.0).abs() <= variance_tolerance;

    let passed = sigmas <= SIGMA_BAND && variance_ok;
    VerificationResult {
        id,
        status: if passed
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(sigmas),
        threshold: Some(SIGMA_BAND),
        explanation: format!(
            "across {n} independent realisations at the final time: mean {sample_mean:.6} \
             against mu = {mu} — {sigmas:.2} standard errors of {standard_error:.6}, with no \
             autocorrelation correction because the realisations are independent; variance \
             {sample_variance:.6} against sigma^2/(2*theta) = {stationary_variance:.6} \
             (ratio {variance_ratio:.3}, tolerance {variance_tolerance:.3})"
        ),
    }
}

/// Re-derive the ensemble from the one seed in the provenance and check it
/// comes back.
///
/// The single-path check shows that a recorded seed reproduces its path. For
/// an ensemble there is a second link in the chain — the base seed has to
/// reproduce the *replicates' seeds* too — and a claim about the second is
/// worth exactly as little as a claim about the first was. So this re-derives
/// the seed list from the base, checks it against the seeds the run actually
/// consumed, and then re-runs one member from its re-derived seed and
/// compares bit for bit.
///
/// It re-derives only `retained.len()` seeds rather than all of them, which
/// is correct because a shorter derivation is a prefix of a longer one — a
/// property [`replicate_seeds`] guarantees and its own tests pin.
fn derivation_check(
    x0: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    base_seed: u64,
    retained: &[(u64, Vec<f64>)],
) -> VerificationResult {
    let id = "ensemble_derived_from_seed".to_string();
    let failed = |explanation: String| VerificationResult {
        id: id.clone(),
        status: VerificationStatus::Failed,
        measured: None,
        threshold: None,
        explanation,
    };

    let rederived = replicate_seeds(base_seed, retained.len());
    for (k, ((used, _), &again)) in retained.iter().zip(rederived.iter()).enumerate()
    {
        if *used != again
        {
            return failed(format!(
                "re-deriving the replicate seeds from base seed {base_seed} gave {again} for \
                 realisation {k} where the run consumed {used}: the recorded seed does not \
                 identify this ensemble"
            ));
        }
    }

    // The last retained member rather than the first, so this exercises a
    // seed that was *derived* rather than the base seed the single-path
    // check already covers.
    let (member_seed, member_path) = retained.last().expect("at least one member is retained");
    let steps = member_path.len().saturating_sub(1).min(1000);
    let Ok(replay) = ou_path(x0, theta, mu, sigma, dt, steps, *member_seed)
    else
    {
        return failed(format!(
            "replaying realisation {} from its derived seed {member_seed} failed outright",
            retained.len() - 1
        ));
    };

    match replay
        .iter()
        .zip(member_path.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        None => VerificationResult {
            id,
            status: VerificationStatus::Passed,
            measured: Some(rederived.len() as f64),
            threshold: None,
            explanation: format!(
                "all {} retained realisations' seeds re-derive from base seed {base_seed}, and \
                 the first {} samples of realisation {} re-run from its derived seed \
                 {member_seed} match bit for bit",
                rederived.len(),
                replay.len(),
                retained.len() - 1
            ),
        },
        Some(index) => failed(format!(
            "replaying realisation {} from its derived seed {member_seed} diverged at sample \
             {index}",
            retained.len() - 1
        )),
    }
}

/// Compare the path's tail against the exact stationary moments.
///
/// The tolerance is *derived*, not chosen. Successive samples of an OU path
/// are correlated, so `n` samples do not carry `n` samples' worth of
/// information: the integrated autocorrelation time is `1/θ`, giving an
/// effective sample size of roughly `n·dt·θ/2`. Using the raw `n` would make
/// the band far too tight and produce a check that fails on correct runs —
/// which is how a check ends up being deleted.
fn stationary_check(path: &[f64], dt: f64, theta: f64, mu: f64, sigma: f64) -> VerificationResult {
    let burn_in = ((BURN_IN_RELAXATION_TIMES / theta) / dt).ceil() as usize;
    let tail = path.get(burn_in..).unwrap_or(&[]);

    if sigma == 0.0
    {
        return VerificationResult {
            id: "stationary_moments".to_string(),
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: "sigma = 0: the process is deterministic exponential decay toward mu \
                          and has no stationary distribution to sample"
                .to_string(),
        };
    }
    if tail.len() < 100
    {
        return VerificationResult {
            id: "stationary_moments".to_string(),
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: format!(
                "the run is too short to measure the stationary moments: after discarding \
                 {BURN_IN_RELAXATION_TIMES} relaxation times of transient only {} samples \
                 remain, and a sample mean from that few says nothing",
                tail.len()
            ),
        };
    }

    let n = tail.len() as f64;
    let sample_mean = tail.iter().sum::<f64>() / n;
    let sample_variance = tail.iter().map(|x| (x - sample_mean).powi(2)).sum::<f64>() / (n - 1.0);

    let stationary_variance = sigma * sigma / (2.0 * theta);
    // Effective sample size after autocorrelation: the integrated
    // autocorrelation time of an OU process is 1/theta.
    let effective_n = (n * dt * theta / 2.0).max(1.0);
    let standard_error = (stationary_variance / effective_n).sqrt();

    let mean_error = (sample_mean - mu).abs();
    let allowed = SIGMA_BAND * standard_error;
    let sigmas = mean_error / standard_error;

    // The variance is checked too but with a looser band: the variance of a
    // sample variance falls off more slowly, and pinning it tightly would be
    // the same mistake as ignoring autocorrelation for the mean.
    let variance_ratio = sample_variance / stationary_variance;
    let variance_ok = (0.5..2.0).contains(&variance_ratio);

    let passed = mean_error <= allowed && variance_ok;
    VerificationResult {
        id: "stationary_moments".to_string(),
        status: if passed
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(sigmas),
        threshold: Some(SIGMA_BAND),
        explanation: format!(
            "sample mean {sample_mean:.6} against mu = {mu} — {sigmas:.2} standard errors, \
             from {} samples worth about {effective_n:.0} independent ones; \
             sample variance {sample_variance:.6} against sigma^2/(2*theta) = \
             {stationary_variance:.6} (ratio {variance_ratio:.3})",
            tail.len()
        ),
    }
}

/// Re-derive the start of the path from the recorded seed and compare it bit
/// for bit.
///
/// Only the first samples, because the point is not to redo the run: it is to
/// show that the number written into the provenance is the number that
/// produces this sample. If the generator's stream matches for a thousand
/// draws it matches.
fn reproducibility_check(
    x0: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    seed: u64,
    path: &[f64],
) -> VerificationResult {
    let steps = path.len().saturating_sub(1).min(1000);
    let Ok(replay) = ou_path(x0, theta, mu, sigma, dt, steps, seed)
    else
    {
        return VerificationResult {
            id: "reproducible_from_seed".to_string(),
            status: VerificationStatus::Failed,
            measured: None,
            threshold: None,
            explanation: "replaying the path from the recorded seed failed outright".to_string(),
        };
    };

    let mismatch = replay
        .iter()
        .zip(path.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());
    match mismatch
    {
        None => VerificationResult {
            id: "reproducible_from_seed".to_string(),
            status: VerificationStatus::Passed,
            measured: Some(replay.len() as f64),
            threshold: None,
            explanation: format!(
                "the first {} samples re-derived from seed {seed} match bit for bit",
                replay.len()
            ),
        },
        Some(index) => VerificationResult {
            id: "reproducible_from_seed".to_string(),
            status: VerificationStatus::Failed,
            measured: Some(index as f64),
            threshold: None,
            explanation: format!(
                "replaying from seed {seed} diverged at sample {index}: the recorded seed does \
                 not reproduce this result, so the provenance is wrong"
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/ornstein_uhlenbeck.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = OrnsteinUhlenbeckAdapter;
        let validated = adapter.validate(scenario).expect("valid");
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes")
    }

    fn run() -> RunResult {
        run_scenario(&parse_toml(TUTORIAL).unwrap())
    }

    /// The tutorial, shortened so an ensemble of many replicates runs in test
    /// time, but still spanning enough relaxation times for the stationary
    /// law to apply at the final coordinate.
    fn ensemble_scenario(replicates: u32) -> Scenario {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.replicates = Some(replicates);
        scenario.solver.end = scirust_studio_schema::ValueWithUnit {
            value: 40.0,
            unit: "s".to_string(),
        };
        scenario
    }

    fn series<'a>(result: &'a RunResult, id: &str) -> &'a Series {
        result
            .series
            .iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("no series {id}"))
    }

    fn roles(result: &RunResult, role: SeriesRole) -> Vec<&Series> {
        result.series.iter().filter(|s| s.role == role).collect()
    }

    fn check<'a>(result: &'a RunResult, id: &str) -> &'a VerificationResult {
        result
            .verifications
            .iter()
            .find(|v| v.id == id)
            .unwrap_or_else(|| panic!("{id} did not run"))
    }

    #[test]
    fn descriptor_id_matches_the_scenario_capability_id() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        assert_eq!(scenario.capability.id, DESCRIPTOR.id.0);
    }

    /// The constraint the whole capability rests on.
    #[test]
    fn a_scenario_without_a_seed_is_refused() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.seed = None;
        let report = OrnsteinUhlenbeckAdapter.validate(&scenario).unwrap_err();
        let error = report
            .errors
            .iter()
            .find(|e| e.code == crate::validate_support::CODE_MISSING_SEED)
            .expect("the missing seed must be reported");
        assert!(error.recoverable);
        assert!(
            error
                .suggested_action
                .as_deref()
                .unwrap_or("")
                .contains("seed"),
            "{error:?}"
        );
    }

    /// And the seed reaches the provenance, which is the point of recording
    /// it at all.
    #[test]
    fn the_consumed_seed_is_recorded_in_the_provenance() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let expected = scenario.experiment.seed.expect("the tutorial seeds itself");
        let result = run_scenario(&scenario);
        assert_eq!(result.provenance.seed, Some(expected));
        assert_eq!(
            result.provenance.determinism,
            DeterminismClass::InherentlyStochasticRecordedSeed
        );
    }

    /// Stochastic does not mean unrepeatable. Same seed, same binary, same
    /// numbers — every one of them.
    #[test]
    fn the_same_seed_reproduces_the_sample_bit_for_bit() {
        let first = run();
        let second = run();
        for (a, b) in first.series[0]
            .values
            .iter()
            .zip(second.series[0].values.iter())
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// …and a different seed must produce a genuinely different sample,
    /// otherwise the seed is not being used and the reproducibility above is
    /// meaningless.
    #[test]
    fn a_different_seed_produces_a_different_sample() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let baseline = run_scenario(&scenario);

        let mut other = scenario.clone();
        other.experiment.seed = Some(scenario.experiment.seed.unwrap() + 1);
        let changed = run_scenario(&other);

        let differing = baseline.series[0]
            .values
            .iter()
            .zip(changed.series[0].values.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differing > baseline.series[0].values.len() / 2,
            "only {differing} samples differ; the seed is barely being used"
        );

        // Both are equally valid samples, so both must still verify.
        assert_eq!(
            check(&changed, "stationary_moments").status,
            VerificationStatus::Passed
        );
    }

    /// The run demonstrates its own provenance claim rather than asserting it.
    #[test]
    fn the_run_replays_itself_from_the_recorded_seed() {
        let result = run();
        let replay = check(&result, "reproducible_from_seed");
        assert_eq!(replay.status, VerificationStatus::Passed, "{replay:?}");
        assert!(replay.measured.unwrap() >= 1000.0);
    }

    #[test]
    fn the_sample_matches_the_exact_stationary_moments() {
        let result = run();
        let moments = check(&result, "stationary_moments");
        assert_eq!(moments.status, VerificationStatus::Passed, "{moments:?}");
        assert!(
            moments.measured.unwrap() < SIGMA_BAND,
            "{:?} standard errors",
            moments.measured
        );
    }

    /// The check must be capable of failing. A process whose sample mean is
    /// deliberately displaced from the declared `mu` has to be caught,
    /// otherwise "passed" means nothing.
    #[test]
    fn the_stationary_check_catches_a_mean_that_is_wrong() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let theta = resolve_model_scalar(&scenario, &THETA).unwrap();
        let sigma = resolve_model_scalar(&scenario, &SIGMA).unwrap();
        let dt = scenario
            .solver
            .step
            .as_ref()
            .unwrap()
            .to_quantity("step")
            .unwrap()
            .value;

        // A genuine path around mu = 0, checked against a claimed mu = 1.
        let path = ou_path(0.0, theta, 0.0, sigma, dt, 200_000, 7).unwrap();
        let honest = stationary_check(&path, dt, theta, 0.0, sigma);
        assert_eq!(honest.status, VerificationStatus::Passed, "{honest:?}");

        let wrong = stationary_check(&path, dt, theta, 1.0, sigma);
        assert_eq!(
            wrong.status,
            VerificationStatus::Failed,
            "a mean displaced by a full unit must be caught: {wrong:?}"
        );
    }

    /// With no noise the process is deterministic, and there is no
    /// distribution to sample — the check must say so rather than pass
    /// vacuously.
    #[test]
    fn a_noiseless_process_reports_the_statistical_check_as_not_applicable() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "sigma".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "1".to_string(),
            },
        );
        let result = run_scenario(&scenario);
        assert_eq!(
            check(&result, "stationary_moments").status,
            VerificationStatus::NotApplicable
        );
        // Reproducibility still applies, and the path must decay to mu.
        assert_eq!(
            check(&result, "reproducible_from_seed").status,
            VerificationStatus::Passed
        );
        let mu = resolve_model_scalar(&scenario, &MU).unwrap();
        let last = *result.series[0].values.last().unwrap();
        assert!((last - mu).abs() < 1e-9, "decayed to {last}, not {mu}");
    }

    /// A run too short to have forgotten its initial condition must not
    /// report a stationary measurement it cannot make.
    #[test]
    fn a_run_too_short_to_be_stationary_says_so() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.end = scirust_studio_schema::ValueWithUnit {
            value: 6.0,
            unit: "s".to_string(),
        };
        let result = run_scenario(&scenario);
        let moments = check(&result, "stationary_moments");
        assert_eq!(
            moments.status,
            VerificationStatus::NotApplicable,
            "{moments:?}"
        );
        assert!(
            moments.explanation.contains("too short"),
            "{}",
            moments.explanation
        );
    }

    #[test]
    fn the_seed_is_reported_as_an_integer_metric() {
        let result = run();
        let seed = result
            .metrics
            .iter()
            .find(|m| m.id == "seed")
            .expect("a seed metric");
        assert!(
            matches!(seed.value, MetricValue::Integer(_)),
            "a seed must not be rendered as a float: {:?}",
            seed.value
        );
        assert_eq!(seed.unit, None);
    }

    #[test]
    fn the_axis_and_series_align() {
        let result = run();
        let axis = &result.axes[0];
        for series in &result.series
        {
            assert_eq!(series.values.len(), axis.values.len(), "{}", series.id);
        }
        assert_eq!(axis.values[0], result.summary.t_start);
        assert_eq!(*axis.values.last().unwrap(), result.summary.t_end);
    }

    #[test]
    fn rejects_a_non_positive_theta() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.insert(
            "theta".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: 0.0,
                unit: "1/s".to_string(),
            },
        );
        let report = OrnsteinUhlenbeckAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.code == THETA.error_code));
    }

    #[test]
    fn rejects_an_ode_solver_it_does_not_implement() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.solver.id = "rk4".to_string();
        let report = OrnsteinUhlenbeckAdapter.validate(&scenario).unwrap_err();
        assert!(report.errors.iter().any(|e| e.explanation.contains("rk4")));
    }

    #[test]
    fn respects_pre_cancelled_control() {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = OrnsteinUhlenbeckAdapter;
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
        let adapter = OrnsteinUhlenbeckAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        let mut sink = crate::sink::CollectingEventSink::new();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        assert_eq!(sink.events().first(), Some(&RunEvent::Started));
        assert_eq!(sink.events().last(), Some(&RunEvent::Completed));
    }

    fn progress_fractions(replicates: Option<u32>) -> Vec<f64> {
        let mut sink = crate::sink::CollectingEventSink::new();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.experiment.replicates = replicates;
        let adapter = OrnsteinUhlenbeckAdapter;
        let validated = adapter.validate(&scenario).unwrap();
        adapter
            .execute(&validated, &ExecutionControl::new(), &mut sink)
            .unwrap();
        sink.events()
            .iter()
            .filter_map(|e| match e
            {
                RunEvent::Progress { fraction, .. } => Some(*fraction),
                _ => None,
            })
            .collect()
    }

    /// The unit of progress here is the *realisation*, not the time step:
    /// `ou_path` samples a whole path in one call and cannot be chunked
    /// without changing the sample.
    ///
    /// A one-replicate run therefore has exactly one indivisible unit of
    /// work, and reports it — one step from nothing to done. That is not an
    /// invented fraction; it is an accurate description of a job with one
    /// part. The version of this test before ensembles asserted the opposite,
    /// while the descriptor's `fixed_step: true` had the app service telling
    /// the interface that progress *was* coming — so a single run drew a
    /// determinate bar that never moved. The two agree now.
    #[test]
    fn a_single_realisation_reports_one_indivisible_unit_of_work() {
        assert_eq!(progress_fractions(None), vec![1.0]);
        assert_eq!(progress_fractions(Some(1)), vec![1.0]);
    }

    /// An ensemble has as many units as it has realisations, so it reports
    /// genuine intermediate progress — the thing a single path could not.
    #[test]
    fn an_ensemble_reports_progress_that_rises_to_exactly_one() {
        let fractions = progress_fractions(Some(64));
        assert!(
            fractions.len() > 1,
            "an ensemble of 64 reported {} progress events; it has 64 units of work to \
             report on",
            fractions.len()
        );
        assert!(
            fractions.windows(2).all(|w| w[1] > w[0]),
            "progress went backwards: {fractions:?}"
        );
        assert_eq!(fractions.last(), Some(&1.0), "progress must end at done");
        assert!(fractions.iter().all(|f| (0.0..=1.0).contains(f)));
    }

    /// `RunEvent::Progress` promises at most twenty emissions per run,
    /// whatever the replicate count. Reporting must not cost more than the
    /// work being reported on.
    #[test]
    fn a_large_ensemble_still_reports_at_most_twenty_times() {
        assert!(progress_fractions(Some(500)).len() <= 20);
    }

    // ---------------------------------------------------------------
    // Ensembles
    // ---------------------------------------------------------------

    /// A single-replicate run must be exactly the run this capability did
    /// before ensembles existed — same series, same values, same seed.
    /// Otherwise this phase silently changed what every stored OU result
    /// means.
    #[test]
    fn one_replicate_is_the_run_this_capability_already_did() {
        let mut explicit = parse_toml(TUTORIAL).unwrap();
        explicit.experiment.replicates = Some(1);

        let implicit = run();
        let explicit = run_scenario(&explicit);

        assert_eq!(
            series(&implicit, "value").values,
            series(&explicit, "value").values
        );
        assert_eq!(implicit.provenance.seed, Some(42));
        // No ensemble apparatus appears for a single realisation.
        assert!(implicit.series.iter().all(|s| !s.role.is_ensemble()));
        assert!(implicit.metrics.iter().all(|m| m.id != "replicates"));
    }

    #[test]
    fn an_ensemble_carries_a_mean_a_band_and_a_bounded_number_of_exemplars() {
        let result = run_scenario(&ensemble_scenario(64));

        assert_eq!(roles(&result, SeriesRole::EnsembleMean).len(), 1);
        assert_eq!(roles(&result, SeriesRole::EnsembleBandLower).len(), 1);
        assert_eq!(roles(&result, SeriesRole::EnsembleBandUpper).len(), 1);
        assert_eq!(
            roles(&result, SeriesRole::EnsembleMember).len(),
            MAX_RETAINED_MEMBERS,
            "64 realisations were drawn; only {MAX_RETAINED_MEMBERS} are kept"
        );
        // The raw single trajectory is not among them: an ensemble's answer
        // is the summary, and a lone unlabelled path beside it would read as
        // the result.
        assert!(roles(&result, SeriesRole::Trajectory).is_empty());

        // Every series still shares the one axis and its length.
        let axis_len = result.axes[0].values.len();
        assert!(result.series.iter().all(|s| s.values.len() == axis_len));
    }

    /// Dropping realisations from the result is a real loss of information,
    /// so both numbers are reported and a reader can see it happened.
    #[test]
    fn the_ensemble_says_how_many_it_drew_and_how_many_it_kept() {
        let result = run_scenario(&ensemble_scenario(64));
        let metric = |id: &str| {
            result
                .metrics
                .iter()
                .find(|m| m.id == id)
                .unwrap_or_else(|| panic!("no metric {id}"))
                .value
                .clone()
        };
        assert_eq!(metric("replicates"), MetricValue::Integer(64));
        assert_eq!(
            metric("retained_members"),
            MetricValue::Integer(MAX_RETAINED_MEMBERS as i64)
        );
    }

    #[test]
    fn the_band_brackets_the_mean_everywhere() {
        let result = run_scenario(&ensemble_scenario(32));
        let mean = &series(&result, "ensemble_mean").values;
        let lower = &series(&result, "ensemble_band_lower").values;
        let upper = &series(&result, "ensemble_band_upper").values;
        for i in 0..mean.len()
        {
            assert!(
                lower[i] <= mean[i] && mean[i] <= upper[i],
                "band does not bracket the mean at {i}: {} <= {} <= {}",
                lower[i],
                mean[i],
                upper[i]
            );
        }
        // The band has real width once the process has moved away from its
        // deterministic initial condition.
        assert!(upper[mean.len() - 1] - lower[mean.len() - 1] > 0.0);
    }

    #[test]
    fn the_ensemble_moments_match_the_stationary_law() {
        let result = run_scenario(&ensemble_scenario(256));
        let check = check(&result, "ensemble_moments");
        assert_eq!(
            check.status,
            VerificationStatus::Passed,
            "{}",
            check.explanation
        );
        assert!(check.explanation.contains("256 independent realisations"));
    }

    /// The whole reason to draw an ensemble: its mean is pinned far more
    /// tightly than one path's, because the realisations are independent and
    /// no autocorrelation correction eats the sample size.
    #[test]
    fn the_ensemble_mean_is_tighter_than_a_single_paths() {
        let result = run_scenario(&ensemble_scenario(256));
        let scenario = &result;
        let standard_error = scenario
            .metrics
            .iter()
            .find(|m| m.id == "ensemble_final_standard_error")
            .map(|m| match m.value
            {
                MetricValue::Scalar(v) => v,
                _ => panic!("expected a scalar"),
            })
            .expect("the ensemble reports its standard error");

        // theta = 0.5, sigma = 0.3 in the tutorial, so the stationary spread
        // is sigma/sqrt(2*theta) = 0.3. Across 256 independent realisations
        // the mean's standard error is that over sqrt(256) = 16.
        let stationary_spread = 0.3;
        let expected = stationary_spread / 16.0;
        assert!(
            (standard_error - expected).abs() < 0.25 * expected,
            "standard error was {standard_error}, expected about {expected}"
        );
    }

    /// A check that cannot fail is decoration. This displaces the true mean
    /// and asserts the ensemble check notices — the same guarantee the
    /// single-path check carries.
    #[test]
    fn the_ensemble_check_catches_a_mean_that_is_in_the_wrong_place() {
        let mut acc = EnsembleAccumulator::new(2);
        // theta = 0.5, sigma = 0.3 => stationary sd 0.3, so an ensemble of
        // 100 pins the mean to 0.03. Sitting the whole ensemble a full unit
        // away from mu is over thirty standard errors out.
        for _ in 0..100
        {
            acc.push(&[0.0, 2.0]).unwrap();
        }
        let verdict = ensemble_check(&acc, 100.0, 0.5, 1.0, 0.3);
        assert_eq!(
            verdict.status,
            VerificationStatus::Failed,
            "{}",
            verdict.explanation
        );
    }

    /// Measuring the stationary law before the transient has decayed would
    /// blame the process for its initial condition, so the check declines
    /// rather than failing.
    #[test]
    fn the_ensemble_check_declines_a_run_that_is_too_short_to_have_settled() {
        let mut acc = EnsembleAccumulator::new(2);
        for _ in 0..10
        {
            acc.push(&[0.0, 1.0]).unwrap();
        }
        // theta = 0.5 => a relaxation time of 2 s; one second is half of one.
        let verdict = ensemble_check(&acc, 1.0, 0.5, 1.0, 0.3);
        assert_eq!(verdict.status, VerificationStatus::NotApplicable);
        assert!(verdict.explanation.contains("relaxation times"));
    }

    /// The provenance records one number. This is the run demonstrating that
    /// the number regenerates the whole ensemble, not just the first path.
    #[test]
    fn the_ensemble_re_derives_itself_from_the_recorded_seed() {
        let result = run_scenario(&ensemble_scenario(16));
        let check = check(&result, "ensemble_derived_from_seed");
        assert_eq!(
            check.status,
            VerificationStatus::Passed,
            "{}",
            check.explanation
        );
        assert_eq!(result.provenance.seed, Some(42));
    }

    #[test]
    fn the_same_ensemble_run_twice_is_bit_identical() {
        let first = run_scenario(&ensemble_scenario(16));
        let second = run_scenario(&ensemble_scenario(16));
        for (a, b) in first.series.iter().zip(second.series.iter())
        {
            assert_eq!(a.id, b.id);
            assert!(
                a.values
                    .iter()
                    .zip(b.values.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "series {} differed between two runs of the same ensemble",
                a.id
            );
        }
    }

    /// Realisations must genuinely differ. If the seed derivation collapsed —
    /// every replicate drawing the same stream — the mean, the band and the
    /// checks would all still look plausible while measuring one sample.
    #[test]
    fn the_realisations_are_not_copies_of_each_other() {
        let result = run_scenario(&ensemble_scenario(8));
        let members = roles(&result, SeriesRole::EnsembleMember);
        assert!(members.len() >= 2);
        for (i, a) in members.iter().enumerate()
        {
            for b in &members[i + 1..]
            {
                assert!(
                    a.values
                        .iter()
                        .zip(b.values.iter())
                        .any(|(x, y)| x.to_bits() != y.to_bits()),
                    "realisations {} and {} are identical",
                    a.id,
                    b.id
                );
            }
        }
    }

    /// A long ensemble is the first thing in this capability that is worth
    /// interrupting, so it must actually stop.
    #[test]
    fn an_ensemble_can_be_cancelled_between_realisations() {
        let adapter = OrnsteinUhlenbeckAdapter;
        let validated = adapter.validate(&ensemble_scenario(64)).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        let mut sink = crate::sink::CollectingEventSink::new();
        assert!(matches!(
            adapter.execute(&validated, &control, &mut sink),
            Err(ExecutionError::Cancelled)
        ));
        assert_eq!(sink.events().last(), Some(&RunEvent::Cancelled));
    }
}
