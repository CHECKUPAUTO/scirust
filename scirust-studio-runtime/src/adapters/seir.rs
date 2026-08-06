//! `sim.epidemiology.seir` — the SEIR compartmental epidemic
//! (`scirust_sim::epidemiology::Seir`).
//!
//! SIR with a latent compartment: an exposed person is infected but not yet
//! infectious, and leaves that state at rate `sigma`. One extra parameter,
//! and a materially different epidemic — the outbreak is delayed and its peak
//! is lower, because the pathogen spends time in a compartment that does not
//! transmit.
//!
//! # What is checked, and why the ordering check is the sharp one
//!
//! Population is conserved exactly: the four derivatives sum to zero
//! identically, so `s + e + i + r` is a linear invariant that RK4 integrates
//! to round-off. That is a strong check on the arithmetic and a weak one on
//! the *structure* — a model that moved people between the wrong two
//! compartments would still conserve them.
//!
//! So the second check is structural and holds for every parameter set: the
//! exposed compartment must peak **strictly before** the infectious one.
//! Nobody becomes infectious without first being exposed, and the transfer is
//! one-directional, so `e` leads `i` however the rates are chosen. A coupling
//! written backwards passes conservation and fails this.
//!
//! The threshold on the invariant is derived rather than chosen: it is the
//! accumulated round-off of summing four numbers of order one over the run's
//! own step count, which is `steps * f64::EPSILON` up to a small constant.
use scirust_sim::epidemiology::Seir;
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

/// Slack over the accumulated round-off of the population sum.
///
/// Not a chosen number: summing four order-one compartments loses at most a
/// few ulps per step, so the floor for a run of `n` steps is `n *
/// f64::EPSILON`. This is the constant multiplying that, and it is small
/// because the invariant really is exact.
const CONSERVATION_MARGIN: f64 = 16.0;

const BETA: FieldDescriptor = FieldDescriptor {
    canonical_name: "beta",
    display_name: "Transmission rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Effective contact rate. R0 is beta/gamma, exactly as for SIR — the latent stage delays the epidemic without changing how many it eventually reaches.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 286),
};

const SIGMA: FieldDescriptor = FieldDescriptor {
    canonical_name: "sigma",
    display_name: "Incubation rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate at which the exposed become infectious; 1/sigma is the mean latent period. This is the parameter SIR does not have.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 287),
};

const GAMMA: FieldDescriptor = FieldDescriptor {
    canonical_name: "gamma",
    display_name: "Recovery rate",
    required: true,
    dimension: scirust_units::Dimension::FREQUENCY,
    accepted_units: &["1/s"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Rate at which the infectious recover. 1/gamma is the mean infectious period.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 288),
};

const SUSCEPTIBLE: FieldDescriptor = FieldDescriptor {
    canonical_name: "susceptible",
    display_name: "Initial susceptible fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction of the population susceptible at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 289),
};

const EXPOSED: FieldDescriptor = FieldDescriptor {
    canonical_name: "exposed",
    display_name: "Initial exposed fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction infected but not yet infectious at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 290),
};

const INFECTIOUS: FieldDescriptor = FieldDescriptor {
    canonical_name: "infectious",
    display_name: "Initial infectious fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction infectious at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 291),
};

const RECOVERED: FieldDescriptor = FieldDescriptor {
    canonical_name: "recovered",
    display_name: "Initial recovered fraction",
    required: true,
    dimension: scirust_units::Dimension::DIMENSIONLESS,
    accepted_units: &["1"],
    min: Some(0.0),
    min_inclusive: true,
    max: Some(1.0),
    max_inclusive: true,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Fraction already recovered at t = start.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 292),
};

const CONSERVATION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "population_is_conserved",
    description: "The four derivatives sum to zero identically, so s + e + i + r is a linear invariant. The threshold is the run's own accumulated round-off, steps * f64::EPSILON, rather than a chosen tolerance: the invariant is exact and only floating-point addition can move it.",
};

const ORDERING_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "exposure_leads_infection",
    description: "The exposed compartment must peak strictly before the infectious one, for every parameter set, because nobody becomes infectious without first being exposed and the transfer is one-directional. Conservation alone would not catch a coupling written backwards; this does.",
};

/// The capability descriptor.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.epidemiology.seir"),
    display_name: "SEIR epidemic with a latent stage",
    category: CapabilityCategory::Epidemiology,
    source_crate: "scirust-sim",
    summary: "The SEIR compartmental model: SIR plus an exposed compartment that is infected but not yet infectious.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    domain: RunDomain::Time,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4_SOLVER],
    parameters: &[BETA, SIGMA, GAMMA],
    initial_state: &[SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED],
    outputs: &[
        OutputDescriptor {
            id: "susceptible",
            display_name: "Susceptible",
            unit: "1",
            description: "Fraction still able to catch it.",
        },
        OutputDescriptor {
            id: "exposed",
            display_name: "Exposed",
            unit: "1",
            description: "Fraction infected but not yet transmitting.",
        },
        OutputDescriptor {
            id: "infectious",
            display_name: "Infectious",
            unit: "1",
            description: "Fraction currently transmitting.",
        },
        OutputDescriptor {
            id: "recovered",
            display_name: "Recovered",
            unit: "1",
            description: "Fraction removed from the epidemic.",
        },
        OutputDescriptor {
            id: "population",
            display_name: "Total population",
            unit: "1",
            description: "The conserved sum, plotted so the invariant is visible rather than merely asserted.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[CONSERVATION_CHECK, ORDERING_CHECK],
    },
};

/// The `sim.epidemiology.seir` adapter.
#[derive(Debug, Default)]
pub struct SeirAdapter;

impl CapabilityAdapter for SeirAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["beta", "sigma", "gamma"],
        ));
        errors.extend(check_unknown_state_fields(
            scenario,
            &["susceptible", "exposed", "infectious", "recovered"],
        ));
        for e in [
            resolve_model_scalar(scenario, &BETA).err(),
            resolve_model_scalar(scenario, &SIGMA).err(),
            resolve_model_scalar(scenario, &GAMMA).err(),
            resolve_state_vector(scenario, &SUSCEPTIBLE).err(),
            resolve_state_vector(scenario, &EXPOSED).err(),
            resolve_state_vector(scenario, &INFECTIOUS).err(),
            resolve_state_vector(scenario, &RECOVERED).err(),
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
        let model = Seir::new(
            resolve_model_scalar(s, &BETA).expect("validated"),
            resolve_model_scalar(s, &SIGMA).expect("validated"),
            resolve_model_scalar(s, &GAMMA).expect("validated"),
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

        let y0 = [
            resolve_state_vector(s, &SUSCEPTIBLE).expect("validated")[0],
            resolve_state_vector(s, &EXPOSED).expect("validated")[0],
            resolve_state_vector(s, &INFECTIOUS).expect("validated")[0],
            resolve_state_vector(s, &RECOVERED).expect("validated")[0],
        ];
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

        let susceptible = traj.column(0).expect("dim 0");
        let exposed = traj.column(1).expect("dim 1");
        let infectious = traj.column(2).expect("dim 2");
        let recovered = traj.column(3).expect("dim 3");
        let population: Vec<f64> = traj.y.iter().map(|row| row.iter().sum()).collect();

        let total0 = y0.iter().sum::<f64>();
        let conservation = conservation_check(&population, total0);
        let ordering = ordering_check(&traj.t, &exposed, &infectious);

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
                    id: "susceptible".to_string(),
                    display_name: "Susceptible".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: susceptible.clone(),
                },
                Series {
                    id: "exposed".to_string(),
                    display_name: "Exposed".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: exposed.clone(),
                },
                Series {
                    id: "infectious".to_string(),
                    display_name: "Infectious".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: infectious.clone(),
                },
                Series {
                    id: "recovered".to_string(),
                    display_name: "Recovered".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Trajectory,
                    values: recovered.clone(),
                },
                Series {
                    id: "population".to_string(),
                    display_name: "Total population".to_string(),
                    unit: "1".to_string(),
                    axis_id: TIME_AXIS_ID.to_string(),
                    role: SeriesRole::Reference,
                    values: population.clone(),
                },
            ],
            fields: vec![],
            distributions: vec![],
            metrics: vec![
                Metric {
                    id: "r0".to_string(),
                    display_name: "Basic reproduction number R0".to_string(),
                    value: MetricValue::Scalar(model.r0()),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "mean_latent_period".to_string(),
                    display_name: "Mean latent period 1/sigma".to_string(),
                    value: MetricValue::Scalar(
                        1.0 / resolve_model_scalar(s, &SIGMA).expect("validated"),
                    ),
                    unit: Some("s".to_string()),
                },
                Metric {
                    id: "peak_infectious".to_string(),
                    display_name: "Peak infectious fraction".to_string(),
                    value: MetricValue::Scalar(infectious.iter().cloned().fold(0.0_f64, f64::max)),
                    unit: Some("1".to_string()),
                },
                Metric {
                    id: "final_recovered".to_string(),
                    display_name: "Final recovered fraction".to_string(),
                    value: MetricValue::Scalar(recovered.last().copied().unwrap_or(f64::NAN)),
                    unit: Some("1".to_string()),
                },
            ],
            warnings: vec![],
            verifications: vec![conservation, ordering],
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

/// The population sum against its exact invariant.
fn conservation_check(population: &[f64], total0: f64) -> VerificationResult {
    let worst = population
        .iter()
        .map(|p| (p - total0).abs())
        .fold(0.0_f64, f64::max);
    // Summing four order-one numbers costs a few ulps per step; over `n`
    // samples the accumulated floor is `n * EPSILON`. Nothing here is chosen
    // except the constant in front, which exists so a correct run is not
    // failed by its own last bit.
    let threshold = population.len() as f64 * f64::EPSILON * CONSERVATION_MARGIN;
    VerificationResult {
        id: CONSERVATION_CHECK.id.to_string(),
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
            "s + e + i + r stayed within {worst:.3e} of its initial {total0:.6} across all {} \
             samples, against the {threshold:.3e} that many floating-point additions of \
             order-one compartments can accumulate",
            population.len()
        ),
    }
}

/// The exposed peak must precede the infectious one.
fn ordering_check(t: &[f64], exposed: &[f64], infectious: &[f64]) -> VerificationResult {
    let argmax = |xs: &[f64]| {
        xs.iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 { (i, v) } else { best }
            })
            .0
    };
    let e_at = argmax(exposed);
    let i_at = argmax(infectious);
    let (e_time, i_time) = (t[e_at], t[i_at]);
    let lead = i_time - e_time;

    VerificationResult {
        id: ORDERING_CHECK.id.to_string(),
        status: if lead > 0.0
        {
            VerificationStatus::Passed
        }
        else
        {
            VerificationStatus::Failed
        },
        measured: Some(lead),
        // Zero, and deliberately: this is an ordering, not a magnitude. Any
        // positive lead passes and any non-positive one fails, so there is no
        // tolerance to get wrong.
        threshold: Some(0.0),
        explanation: format!(
            "the exposed compartment peaked at t = {e_time:.4} and the infectious one at \
             t = {i_time:.4}, a lead of {lead:.4}; exposure precedes infection for every \
             parameter set, so anything but a positive lead is a coupling written backwards"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    const TUTORIAL: &str =
        include_str!("../../../docs/studio/tutorials/seir_epidemic.scirust.toml");

    fn run_scenario(scenario: &Scenario) -> RunResult {
        let adapter = SeirAdapter;
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
        let validated = SeirAdapter.validate(&scenario).unwrap();
        let control = ExecutionControl::new();
        control.cancel();
        assert!(matches!(
            SeirAdapter.execute(&validated, &control, &mut NullEventSink),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn the_population_is_conserved() {
        let r = run();
        let v = check(&r, CONSERVATION_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    #[test]
    fn exposure_leads_infection() {
        let r = run();
        let v = check(&r, ORDERING_CHECK.id);
        assert_eq!(v.status, VerificationStatus::Passed, "{}", v.explanation);
    }

    /// There has to *be* an epidemic, or both checks are describing a flat
    /// line. R0 is 3 in the tutorial, so a tenth of the population should be
    /// infectious at once.
    #[test]
    fn the_tutorial_actually_has_an_outbreak() {
        let r = run();
        assert!(metric(&r, "r0") > 1.0);
        let peak = metric(&r, "peak_infectious");
        assert!(peak > 0.1, "peak infectious was only {peak}");
    }

    /// The latent stage is what distinguishes SEIR from SIR, so the run must
    /// spend real time in it. A vanishing exposed compartment would mean the
    /// ordering check is comparing two peaks of the same curve.
    #[test]
    fn the_latent_compartment_is_not_vestigial() {
        let r = run();
        let peak_exposed = series(&r, "exposed")
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(
            peak_exposed > 0.05,
            "the exposed compartment peaked at only {peak_exposed}"
        );
    }

    #[test]
    fn the_ordering_check_catches_a_reversed_lead() {
        // Infectious peaking first is the failure this exists to catch.
        let t = [0.0, 1.0, 2.0];
        let verdict = ordering_check(&t, &[0.0, 0.0, 1.0], &[0.0, 1.0, 0.0]);
        assert_eq!(verdict.status, VerificationStatus::Failed);
    }
}
