//! `sim.pharmacology.oral_one_compartment` — first-order oral absorption
//! (`scirust_sim::pharmacokinetics::OralOneCompartment`).
//!
//! A dose enters the gut, is absorbed into a single body compartment at rate
//! `k_a`, and is eliminated from it at rate `k_e`. The plasma concentration
//! rises, peaks, and falls — the Bateman curve behind every "take one tablet
//! every eight hours".
//!
//! # The strongest check in the catalogue
//!
//! This model has a **complete closed-form solution**, not merely a conserved
//! quantity or an asymptote. So the check is not "does it end in the right
//! place" or "does something stay constant" — it is the whole trajectory,
//! sample by sample, against the exact answer. Nothing else adapted here can
//! be verified that completely: an energy drift can hide a trajectory that is
//! wrong in a way energy does not see, and a steady state says nothing about
//! the path taken to reach it.
//!
//! The second check is the **time of the peak**, `ln(k_a/k_e)/(k_a - k_e)`,
//! which the model also states in closed form. It is not implied by the
//! first: a run could track the exact curve to within tolerance everywhere
//! and still report its maximum at the wrong sample if the peak is flat, and
//! the peak is the number a dosing schedule is actually built on.
use scirust_sim::pharmacokinetics::OralOneCompartment;
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
};

/// Relative error allowed between the integrated and exact trajectories.
///
/// This is RK4's own error and nothing else, since the reference is the
/// analytic solution rather than another approximation. At the tutorial's
/// step it comes out around 1e-11.
const EXACT_TOLERANCE: f64 = 1e-6;

/// How far the observed peak may sit from the closed-form peak time,
/// as a fraction of it.
///
/// Bounded below by the sampling interval: a run cannot locate a peak more
/// precisely than the gap between two samples, so the threshold is derived
/// from the step rather than chosen.
const PEAK_MARGIN: f64 = 2.0;

const K_A: FieldDescriptor = FieldDescriptor {
    canonical_name: "k_a",
    display_name: "Absorption rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "First-order rate at which drug leaves the gut for the central compartment.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 240),
};

const K_E: FieldDescriptor = FieldDescriptor {
    canonical_name: "k_e",
    display_name: "Elimination rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "First-order rate at which drug leaves the central compartment. Its reciprocal times ln 2 is the half-life.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 241),
};

const VOLUME: FieldDescriptor = FieldDescriptor {
    canonical_name: "volume",
    display_name: "Volume of distribution",
    required: true,
    dimension: scirust_units::Dimension::LENGTH.powi(3),
    accepted_units: &["m^3"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Apparent volume converting a central amount to a plasma concentration.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 242),
};

const BIOAVAILABILITY: FieldDescriptor = FieldDescriptor {
    canonical_name: "bioavailability",
    display_name: "Bioavailability",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction of the dose that reaches the gut compartment, in (0, 1].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 243),
};

const DOSE: FieldDescriptor = FieldDescriptor {
    canonical_name: "dose",
    display_name: "Dose",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "The administered amount, in whatever dose units the scenario uses. Concentrations come out in the same units per cubic metre.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 244),
};

const MATCHES_THE_CLOSED_FORM_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "matches_the_closed_form",
    description: "The model has a complete analytic solution, so the whole trajectory is compared against it sample by sample rather than only its endpoint or a conserved quantity. This is the strongest form of verification available to any capability here.",
};

const PEAK_TIME_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "peak_time",
    description: "The time of maximum plasma concentration is ln(k_a/k_e)/(k_a - k_e) in closed form. Not implied by the trajectory check: a flat peak can be tracked accurately everywhere and still be located at the wrong sample, and the peak is what a dosing schedule is built on.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.pharmacology.oral_one_compartment"),
    display_name: "Oral dose, one-compartment body",
    category: CapabilityCategory::Pharmacology,
    source_crate: "scirust-sim",
    summary: "First-order absorption of an oral dose into a single body compartment with first-order elimination — the Bateman curve, verified against its complete closed-form solution.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[K_A, K_E, VOLUME, BIOAVAILABILITY, DOSE],
    initial_state: &[],
    outputs: &[
        OutputDescriptor {
            id: "gut",
            display_name: "Gut amount",
            unit: "1",
            description: "Drug remaining to be absorbed.",
        },
        OutputDescriptor {
            id: "central",
            display_name: "Central amount",
            unit: "1",
            description: "Drug in the body compartment.",
        },
        OutputDescriptor {
            id: "concentration",
            display_name: "Plasma concentration",
            unit: "1",
            description: "Central amount divided by the volume of distribution.",
        },
        OutputDescriptor {
            id: "exact_central",
            display_name: "Exact central amount",
            unit: "1",
            description: "The closed-form solution, for comparison.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[MATCHES_THE_CLOSED_FORM_CHECK, PEAK_TIME_CHECK],
    },
};

/// The `sim.pharmacology.oral_one_compartment` adapter.
#[derive(Debug, Default)]
pub struct OralOneCompartmentAdapter;

impl CapabilityAdapter for OralOneCompartmentAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["k_a", "k_e", "volume", "bioavailability", "dose"],
        ));
        // This capability takes no `initial_state`: the state at t = 0 is
        // determined by the dose and the bioavailability, and letting a
        // scenario set it separately would allow the two to disagree.
        errors.extend(check_unknown_state_fields(scenario, &[]));
        for e in [
            resolve_model_scalar(scenario, &K_A).err(),
            resolve_model_scalar(scenario, &K_E).err(),
            resolve_model_scalar(scenario, &VOLUME).err(),
            resolve_model_scalar(scenario, &BIOAVAILABILITY).err(),
            resolve_model_scalar(scenario, &DOSE).err(),
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

        // The closed form has a removable singularity at k_a = k_e, and the
        // model returns `None` there rather than a limit. Refusing is
        // honest: the verification this capability is built around would be
        // unavailable, so the run would be a trajectory nobody checked.
        if let (Ok(k_a), Ok(k_e)) = (
            resolve_model_scalar(scenario, &K_A),
            resolve_model_scalar(scenario, &K_E),
        )
        {
            if k_a == k_e
            {
                errors.push(scirust_studio_command::CatalogedError {
                    code: ErrorCode::new(ErrorFamily::Validation, 245),
                    title: "Absorption and elimination rates are equal".to_string(),
                    explanation: format!(
                        "`k_a` and `k_e` are both {k_a}, where the Bateman solution has a \
                         removable singularity. This capability verifies every run against \
                         that closed form, so a run there would be one nobody could check"
                    ),
                    recoverable: true,
                    suggested_action: Some(
                        "separate the two rates; they are almost never equal in practice"
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
        let s = scenario.scenario();
        let k_a = resolve_model_scalar(s, &K_A).expect("validated");
        let k_e = resolve_model_scalar(s, &K_E).expect("validated");
        let volume = resolve_model_scalar(s, &VOLUME).expect("validated");
        let f = resolve_model_scalar(s, &BIOAVAILABILITY).expect("validated");
        let dose = resolve_model_scalar(s, &DOSE).expect("validated");

        let model = OralOneCompartment::new(k_a, k_e, volume, f, dose)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;
        let y0 = model.initial_state();

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
            &y0,
            TimeSpan {
                t0,
                t_end: t1,
                h: step,
            },
            control,
            sink,
        )?;

        let gut = traj.column(0).expect("dim 0 exists");
        let central = traj.column(1).expect("dim 1 exists");
        let concentration: Vec<f64> = central.iter().map(|a| model.concentration(*a)).collect();

        // The reference is the analytic solution, evaluated at the same
        // coordinates the integrator produced — not resampled, not
        // interpolated.
        let exact: Vec<f64> = traj
            .t
            .iter()
            .map(|t| model.exact(*t).map(|s| s[1]).unwrap_or(f64::NAN))
            .collect();

        let accuracy = closed_form_check(&central, &exact, dose * f);
        let peak = peak_check(&traj.t, &central, model.peak_time(), step);

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
                    id: "gut".to_string(),
                    display_name: "Gut amount".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: gut.clone(),
                },
                Series {
                    id: "central".to_string(),
                    display_name: "Central amount".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: central.clone(),
                },
                Series {
                    id: "concentration".to_string(),
                    display_name: "Plasma concentration".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: concentration.clone(),
                },
                Series {
                    id: "exact_central".to_string(),
                    display_name: "Exact central amount".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: exact.clone(),
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "peak_time".to_string(),
                    display_name: "Closed-form peak time ln(k_a/k_e)/(k_a - k_e)".to_string(),
                    value: MetricValue::Scalar(model.peak_time().unwrap_or(f64::NAN)),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "peak_concentration".to_string(),
                    display_name: "Peak plasma concentration".to_string(),
                    value: MetricValue::Scalar(
                        concentration
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max),
                    ),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "half_life".to_string(),
                    display_name: "Elimination half-life ln(2)/k_e".to_string(),
                    value: MetricValue::Scalar(std::f64::consts::LN_2 / k_e),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "absorbed_fraction".to_string(),
                    display_name: "Fraction absorbed by the end of the run".to_string(),
                    value: MetricValue::Scalar(
                        1.0 - gut.last().copied().unwrap_or(0.0)
                            / (dose * f).max(f64::MIN_POSITIVE),
                    ),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![accuracy, peak],
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

/// The whole trajectory against the whole closed form.
fn closed_form_check(central: &[f64], exact: &[f64], scale: f64) -> VerificationResult {
    let scale = scale.abs().max(f64::MIN_POSITIVE);
    let mut worst = 0.0_f64;
    let mut at = 0usize;
    for (i, (a, e)) in central.iter().zip(exact.iter()).enumerate()
    {
        if !e.is_finite()
        {
            continue;
        }
        let error = (a - e).abs() / scale;
        if error > worst
        {
            worst = error;
            at = i;
        }
    }

    VerificationResult {
        id: "matches_the_closed_form".to_string(),
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
            "the integrated central amount tracks the exact Bateman solution to within \
             {worst:.3e} of the absorbed dose across all {} samples, worst at sample {at}; \
             the reference is the analytic solution, so this is the integrator's error alone",
            central.len()
        ),
    }
}

/// Where the curve actually peaked, against where the closed form says.
///
/// The threshold is derived from the sampling interval: a run cannot locate a
/// maximum more precisely than the gap between two samples, so demanding
/// better would be asking the run for something it does not have.
fn peak_check(t: &[f64], central: &[f64], expected: Option<f64>, step: f64) -> VerificationResult {
    let id = "peak_time".to_string();
    let Some(expected) = expected
    else
    {
        return VerificationResult {
            id,
            status: VerificationStatus::NotApplicable,
            measured: None,
            threshold: None,
            explanation: "the closed form has no peak time when k_a = k_e".to_string(),
        };
    };

    let observed_index = central
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let observed = t.get(observed_index).copied().unwrap_or(f64::NAN);

    if observed_index == 0 || observed_index + 1 == central.len()
    {
        return VerificationResult {
            id,
            status: VerificationStatus::Failed,
            measured: Some(observed),
            threshold: Some(expected),
            explanation: format!(
                "the maximum was at an endpoint of the run ({observed} s), so the peak at \
                 {expected} s is outside the span integrated — the run does not contain the \
                 thing it is meant to show"
            ),
        };
    }

    let error = (observed - expected).abs();
    let tolerance = step * PEAK_MARGIN;
    VerificationResult {
        id,
        status: if error <= tolerance
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(error),
        threshold: Some(tolerance),
        explanation: format!(
            "the concentration peaked at {observed} s against the closed-form \
             ln(k_a/k_e)/(k_a - k_e) = {expected:.6} s, a difference of {error:.3e} s — \
             against the {tolerance:.3e} s that {PEAK_MARGIN} sampling intervals allow"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/oral_dose.scirust.toml");

    fn run() -> RunResult {
        let scenario = parse_toml(TUTORIAL).unwrap();
        let adapter = OralOneCompartmentAdapter;
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
        let validated = OralOneCompartmentAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            OralOneCompartmentAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn the_trajectory_matches_the_closed_form() {
        let r = run();
        let v = check(&r, "matches_the_closed_form");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn the_peak_lands_where_the_closed_form_says() {
        let r = run();
        let v = check(&r, "peak_time");
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// Equal rates are refused rather than run, because the closed form this
    /// capability verifies against does not exist there.
    #[test]
    fn equal_rates_are_refused_with_the_reason() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        let k_e = scenario.model.get("k_e").cloned().expect("k_e");
        scenario.model.insert("k_a".to_string(), k_e);
        let report = OralOneCompartmentAdapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.explanation.contains("removable singularity")),
            "{:?}",
            report.errors
        );
    }

    #[test]
    fn the_closed_form_check_catches_a_trajectory_that_drifts() {
        let verdict = closed_form_check(&[1.0, 2.0, 3.0], &[1.0, 2.0, 2.0], 1.0);
        assert_eq!(verdict.status, VerificationStatus::Failed);
    }

    /// A peak at the end of the run means the run is too short to contain
    /// it — reported as a failure, not passed because the numbers are close.
    #[test]
    fn a_peak_at_the_edge_of_the_run_is_a_failure() {
        let verdict = peak_check(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], Some(2.0), 1.0);
        assert_eq!(verdict.status, VerificationStatus::Failed);
        assert!(verdict.explanation.contains("endpoint"));
    }

    /// The peak tolerance must follow the step: a coarser run cannot locate a
    /// maximum as precisely, and demanding that it does would be asking for
    /// something it does not have.
    #[test]
    fn the_peak_tolerance_follows_the_sampling_interval() {
        let coarse = peak_check(&[0.0, 1.0, 2.0, 3.0], &[0.0, 2.0, 1.0, 0.5], Some(1.5), 1.0);
        assert_eq!(
            coarse.status,
            VerificationStatus::Passed,
            "{}",
            coarse.explanation
        );
        let fine = peak_check(
            &[0.0, 0.01, 0.02, 0.03],
            &[0.0, 2.0, 1.0, 0.5],
            Some(1.5),
            0.01,
        );
        assert_eq!(
            fine.status,
            VerificationStatus::Failed,
            "{}",
            fine.explanation
        );
    }
}
