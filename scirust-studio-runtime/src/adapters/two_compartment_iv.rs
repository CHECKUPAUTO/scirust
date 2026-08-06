//! `sim.pharmacology.two_compartment_iv` — an intravenous bolus into a body
//! with a peripheral compartment
//! (`scirust_sim::pharmacokinetics::TwoCompartmentIv`).
//!
//! The dose lands in the central compartment all at once, then does two
//! things at once: it is eliminated, and it distributes into and back out of
//! a peripheral compartment. The plasma curve that results is **biexponential**
//! — a fast distribution phase and a slow elimination phase — which is why a
//! one-compartment fit of real data so often looks wrong early and right late.
//!
//! # Two checks that cover different halves of the curve
//!
//! **The closed form** is checked at every recorded sample. The model states
//! the biexponential solution in terms of the hybrid rate constants
//! `(alpha, beta)`, so the reference is exact and the measurement is RK4's
//! truncation error alone. This is dominated by the *early* part of the curve,
//! where the fast phase makes the derivative largest.
//!
//! **The area under the curve** is the late half. `AUC = dose/k10` exactly,
//! over `[0, infinity)` — a statement about total exposure that does not
//! depend on how the drug distributes at all, only on how it leaves. The run
//! covers a finite window, so the trapezoid integral over `[0, T]` is
//! completed with the **analytic tail**, which the same biexponential form
//! gives in closed form. A model with the right shape and the wrong
//! elimination rate matches the curve locally and misses the area.
use scirust_sim::pharmacokinetics::TwoCompartmentIv;
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

/// Relative agreement demanded with the biexponential closed form.
const EXACT_TOLERANCE: f64 = 1e-9;

/// Slack over the trapezoid rule's computed leading error on the AUC.
const AUC_MARGIN: f64 = 8.0;

const K10: FieldDescriptor = FieldDescriptor {
    canonical_name: "k10",
    display_name: "Elimination rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate of irreversible loss from the central compartment. It alone sets the total exposure: AUC is dose/k10 however the drug distributes.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 314),
};

const K12: FieldDescriptor = FieldDescriptor {
    canonical_name: "k12",
    display_name: "Distribution rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate from central to peripheral.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 315),
};

const K21: FieldDescriptor = FieldDescriptor {
    canonical_name: "k21",
    display_name: "Redistribution rate constant",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate from peripheral back to central. With k12 it produces the fast phase and changes nothing about the area.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 316),
};

const VOLUME: FieldDescriptor = FieldDescriptor {
    canonical_name: "volume",
    display_name: "Central volume of distribution",
    required: true,
    dimension: scirust_units::Dimension::LENGTH.powi(3),
    accepted_units: &["m^3"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Apparent volume the dose is diluted into to give plasma concentration.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 317),
};

const DOSE: FieldDescriptor = FieldDescriptor {
    canonical_name: "dose",
    display_name: "Intravenous dose",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Amount given as a bolus at t = start. It lands entirely in the central compartment, which is why this capability has no [initial_state].",
    error_code: ErrorCode::new(ErrorFamily::Validation, 318),
};

const EXACT_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "matches_the_biexponential_closed_form",
    description: "The central amount against the model's stated biexponential solution at every recorded sample. The reference is exact, so the measurement is RK4's truncation error alone, and it is dominated by the fast distribution phase where the derivative is largest.",
};

const AUC_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "total_exposure_is_dose_over_elimination",
    description: "The area under the central-amount curve is dose/k10 exactly, over the whole real line — a fact about elimination that does not depend on distribution at all. The run's finite window is closed with the analytic tail, and the threshold is the composite trapezoid rule's own leading Euler-Maclaurin error rather than a chosen number. A model with the right curve shape and the wrong elimination rate passes the closed-form check locally and fails this.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.pharmacology.two_compartment_iv"),
    display_name: "Two-compartment IV bolus",
    category: CapabilityCategory::Pharmacology,
    source_crate: "scirust-sim",
    summary: "An intravenous bolus distributing into a peripheral compartment while being eliminated: the biexponential plasma curve, verified against both its closed form and its total exposure.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[K10, K12, K21, VOLUME, DOSE],
    initial_state: &[],
    outputs: &[
        OutputDescriptor {
            id: "central_amount",
            display_name: "Central amount",
            unit: "1",
            description: "Drug in the central compartment.",
        },
        OutputDescriptor {
            id: "peripheral_amount",
            display_name: "Peripheral amount",
            unit: "1",
            description: "Drug in the tissue compartment: rises, peaks, and drains back.",
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
            description: "The biexponential closed form, for comparison.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[EXACT_CHECK, AUC_CHECK],
    },
};

/// The `sim.pharmacology.two_compartment_iv` adapter.
#[derive(Debug, Default)]
pub struct TwoCompartmentIvAdapter;

impl CapabilityAdapter for TwoCompartmentIvAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["k10", "k12", "k21", "volume", "dose"],
        ));
        // The bolus lands entirely in the central compartment, so the state
        // at t = 0 follows from the dose. A scenario that supplied one would
        // be describing something this capability does not have.
        errors.extend(check_unknown_state_fields(scenario, &[]));
        for e in [
            resolve_model_scalar(scenario, &K10).err(),
            resolve_model_scalar(scenario, &K12).err(),
            resolve_model_scalar(scenario, &K21).err(),
            resolve_model_scalar(scenario, &VOLUME).err(),
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
        let k10 = resolve_model_scalar(s, &K10).expect("validated");
        let dose = resolve_model_scalar(s, &DOSE).expect("validated");
        let model = TwoCompartmentIv::new(
            k10,
            resolve_model_scalar(s, &K12).expect("validated"),
            resolve_model_scalar(s, &K21).expect("validated"),
            resolve_model_scalar(s, &VOLUME).expect("validated"),
            dose,
        )
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

        let y0 = model.initial_state();
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

        let central = traj.column(0).expect("dim 0");
        let peripheral = traj.column(1).expect("dim 1");
        let concentration: Vec<f64> = central.iter().map(|a| model.concentration(*a)).collect();
        let exact: Vec<f64> = traj.t.iter().map(|t| model.exact_central(t - t0)).collect();

        let (alpha, beta) = model.hybrid_rates();
        let exact_check = closed_form_check(&central, &exact, dose);
        let auc = auc_check(&traj.t, &central, &model, dose, k10, alpha, beta, step, t0);

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
                    id: "central_amount".to_string(),
                    display_name: "Central amount".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: central.clone(),
                },
                Series {
                    id: "peripheral_amount".to_string(),
                    display_name: "Peripheral amount".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: peripheral.clone(),
                },
                Series {
                    id: "concentration".to_string(),
                    display_name: "Plasma concentration".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: concentration,
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
                    id: "alpha".to_string(),
                    display_name: "Fast hybrid rate constant".to_string(),
                    value: MetricValue::Scalar(alpha),
                    unit: Some("1/s".to_string()),
                },
                Metric {
                    id: "beta".to_string(),
                    display_name: "Slow hybrid rate constant".to_string(),
                    value: MetricValue::Scalar(beta),
                    unit: Some("1/s".to_string()),
                },
                Metric {
                    id: "distribution_half_life".to_string(),
                    display_name: "Distribution half-life ln(2)/alpha".to_string(),
                    value: MetricValue::Scalar(std::f64::consts::LN_2 / alpha),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "elimination_half_life".to_string(),
                    display_name: "Elimination half-life ln(2)/beta".to_string(),
                    value: MetricValue::Scalar(std::f64::consts::LN_2 / beta),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "exact_auc".to_string(),
                    display_name: "Exact total exposure dose/k10".to_string(),
                    value: MetricValue::Scalar(model.central_auc()),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "peak_peripheral".to_string(),
                    display_name: "Peak peripheral amount".to_string(),
                    value: MetricValue::Scalar(peripheral.iter().cloned().fold(0.0_f64, f64::max)),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![exact_check, auc],
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

/// The central amount against the biexponential closed form.
fn closed_form_check(central: &[f64], exact: &[f64], dose: f64) -> VerificationResult {
    let scale = dose.abs().max(f64::MIN_POSITIVE);
    let worst = central
        .iter()
        .zip(exact.iter())
        .map(|(g, x)| (g - x).abs() / scale)
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
            "the integrated central amount tracked the biexponential closed form to within \
             {worst:.3e} of the dose, across all {} samples",
            central.len()
        ),
    }
}

/// The trapezoid area plus the analytic tail against `dose/k10`.
#[allow(clippy::too_many_arguments)]
fn auc_check(
    t: &[f64],
    central: &[f64],
    model: &TwoCompartmentIv,
    dose: f64,
    k10: f64,
    alpha: f64,
    beta: f64,
    h: f64,
    t0: f64,
) -> VerificationResult {
    let window: f64 = central
        .windows(2)
        .zip(t.windows(2))
        .map(|(a, t)| 0.5 * (a[0] + a[1]) * (t[1] - t[0]))
        .sum();

    // The tail beyond the run, from the same biexponential form the model
    // states: integrating each exponential from T to infinity divides its
    // coefficient by its own rate.
    let end = *t.last().expect("at least one sample") - t0;
    let denominator = alpha - beta;
    let tail = if denominator == 0.0
    {
        // Repeated root: the limiting single-exponential form.
        dose * (-alpha * end).exp() / alpha
    }
    else
    {
        dose * ((alpha - model_k21(alpha, beta, k10)) / denominator * (-alpha * end).exp() / alpha
            + (model_k21(alpha, beta, k10) - beta) / denominator * (-beta * end).exp() / beta)
    };
    let measured = window + tail;
    let exact = model.central_auc();
    let error = (measured - exact).abs() / exact.abs().max(f64::MIN_POSITIVE);

    // Euler-Maclaurin again: the composite trapezoid's leading error is
    // (h^2/12)*(f'(b) - f'(a)), and f' here is the central compartment's own
    // derivative, which the closed form gives exactly at both ends.
    let dcentral = |t: f64| {
        let d = 1e-9_f64.max(t.abs() * 1e-9);
        (model.exact_central(t + d) - model.exact_central(t - d)) / (2.0 * d)
    };
    let leading = (h * h / 12.0) * (dcentral(end) - dcentral(0.0)).abs();
    let threshold = (leading * AUC_MARGIN / exact.abs().max(f64::MIN_POSITIVE))
        .max(f64::EPSILON * central.len() as f64);

    VerificationResult {
        id: AUC_CHECK.id.to_string(),
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
            "the trapezoid area over the run ({window:.6}) plus the analytic tail beyond it \
             ({tail:.6}) came to {measured:.6}, against the exact dose/k10 = {exact:.6} — a \
             relative difference of {error:.3e}, against the {threshold:.3e} the trapezoid \
             rule's own leading error allows"
        ),
    }
}

/// Recover `k21` from the hybrid rates and the elimination constant.
///
/// The model states `alpha + beta = k10 + k12 + k21` and
/// `alpha*beta = k10*k21`, so `k21 = alpha*beta/k10` — which is the form used
/// here because it needs only the two numbers the model already exposes.
fn model_k21(alpha: f64, beta: f64, k10: f64) -> f64 {
    alpha * beta / k10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/two_compartment_iv.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = TwoCompartmentIvAdapter;
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
        let validated = TwoCompartmentIvAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            TwoCompartmentIvAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn it_matches_the_biexponential_closed_form() {
        let r = run();
        let v = check(&r, EXACT_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn total_exposure_is_dose_over_elimination() {
        let r = run();
        let v = check(&r, AUC_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// The curve must genuinely be biexponential, or a one-compartment model
    /// would do and this capability is redundant. The two phases have to be
    /// separated enough to see.
    #[test]
    fn the_two_phases_are_distinguishable() {
        let r = run();
        let (alpha, beta) = (metric(&r, "alpha"), metric(&r, "beta"));
        assert!(
            alpha > 5.0 * beta,
            "the fast phase ({alpha}) is not separated from the slow one ({beta})"
        );
        assert!(
            metric(&r, "peak_peripheral") > 0.1 * parse_toml(TUTORIAL).unwrap().model["dose"].value,
            "barely any drug reached the peripheral compartment"
        );
    }

    /// The area depends only on elimination. Doubling the distribution rates
    /// reshapes the curve completely and must leave the total exposure
    /// exactly where it was — which is the fact the AUC check exists to
    /// verify, stated as an experiment.
    #[test]
    fn distribution_reshapes_the_curve_without_moving_the_area() {
        let base = run();
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.model.get_mut("k12").expect("k12").value *= 2.0;
        scenario.model.get_mut("k21").expect("k21").value *= 2.0;
        let redistributed = run_scenario(&scenario);

        assert!(
            (metric(&base, "exact_auc") - metric(&redistributed, "exact_auc")).abs() < 1e-12,
            "the total exposure moved when only distribution changed"
        );
        // And the curve really did change shape.
        assert!(
            (metric(&base, "alpha") - metric(&redistributed, "alpha")).abs()
                > 0.1 * metric(&base, "alpha"),
            "doubling the distribution rates did not move the fast phase"
        );
    }

    /// There is no state to supply: the bolus determines it.
    #[test]
    fn an_initial_state_is_rejected_rather_than_ignored() {
        let mut scenario = parse_toml(TUTORIAL).unwrap();
        scenario.initial_state.insert(
            "central_amount".to_string(),
            vec![scirust_studio_schema::ValueWithUnit {
                value: 1.0,
                unit: "1".to_string(),
            }],
        );
        assert!(TwoCompartmentIvAdapter.validate(&scenario).is_err());
    }
}
