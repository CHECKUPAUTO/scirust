//! `sim.electrical.rlc` — a series RLC circuit
//! (`scirust_sim::electrical::SeriesRlc`).
//!
//! Chosen as a representative adapter because it needs electrical units
//! (Ohm/Henry/Farad, derived here from `scirust_units`' base dimensions
//! rather than added as new named constants there), has several output
//! series plus a genuinely *derived* one (capacitor voltage isn't a state
//! component; it is computed from charge and capacitance), and its
//! scientific behaviour changes qualitatively with a single parameter
//! (resistance), which is exactly what the damping-regime classification
//! metric exercises.

use scirust_sim::electrical::SeriesRlc;
use scirust_sim::simulate;
use scirust_studio_command::{ErrorCode, ErrorFamily};
use scirust_studio_registry::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
use scirust_studio_schema::Scenario;

use crate::adapter::{CapabilityAdapter, ExecutionError, ValidatedScenario, ValidationReport};
use crate::control::ExecutionControl;
use crate::result::{
    AxisDescriptor, Metric, MetricValue, RESULT_SCHEMA_VERSION, RunProvenance, RunResult,
    RunSummary, Series, VerificationResult, VerificationStatus,
};
use crate::sink::{EventSink, RunEvent};
use crate::validate_support::{
    check_unknown_model_fields, check_unknown_state_fields, resolve_model_scalar, resolve_solver,
    resolve_state_vector,
};

const RESISTANCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "resistance",
    display_name: "Resistance",
    required: true,
    dimension: scirust_units::Dimension::RESISTANCE,
    accepted_units: &["Ohm"],
    min: Some(0.0),
    min_inclusive: true,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Series resistance R. Zero gives a lossless LC circuit.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 130),
};

const INDUCTANCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "inductance",
    display_name: "Inductance",
    required: true,
    dimension: scirust_units::Dimension::ENERGY.div(scirust_units::Dimension::CURRENT.powi(2)),
    accepted_units: &["H"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Series inductance L.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 131),
};

const CAPACITANCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "capacitance",
    display_name: "Capacitance",
    required: true,
    dimension: scirust_units::Dimension::CHARGE.div(scirust_units::Dimension::VOLTAGE),
    accepted_units: &["F"],
    min: Some(0.0),
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Capacitance C.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 132),
};

const V_SOURCE: FieldDescriptor = FieldDescriptor {
    canonical_name: "v_source",
    display_name: "Source voltage",
    required: false,
    dimension: scirust_units::Dimension::VOLTAGE,
    accepted_units: &["V"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: Some(0.0),
    cardinality: Cardinality::Scalar,
    description: "Constant driving source voltage. Defaults to 0 (undriven, decaying circuit).",
    error_code: ErrorCode::new(ErrorFamily::Validation, 133),
};

const CHARGE: FieldDescriptor = FieldDescriptor {
    canonical_name: "charge",
    display_name: "Initial capacitor charge",
    required: true,
    dimension: scirust_units::Dimension::CHARGE,
    accepted_units: &["C"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial charge on the capacitor.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 134),
};

const CURRENT: FieldDescriptor = FieldDescriptor {
    canonical_name: "current",
    display_name: "Initial current",
    required: true,
    dimension: scirust_units::Dimension::CURRENT,
    accepted_units: &["A"],
    min: None,
    min_inclusive: false,
    max: None,
    max_inclusive: false,
    default: None,
    cardinality: Cardinality::Scalar,
    description: "Initial loop current.",
    error_code: ErrorCode::new(ErrorFamily::Validation, 135),
};

const RK4: SolverDescriptor = SolverDescriptor {
    id: "rk4",
    summary: "Fixed-step classical 4th-order Runge-Kutta.",
    fixed_step: true,
    adaptive_tolerance: false,
};

const FINITE_SOLUTION_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "finite_solution",
    description: "Every recorded charge/current value must be finite.",
};

const DAMPING_REGIME_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "damping_regime",
    description: "Classifies the circuit as undamped/underdamped/critically damped/overdamped from its damping ratio zeta.",
};

const ENERGY_DECAY_CHECK: VerificationCheckDescriptor = VerificationCheckDescriptor {
    id: "energy_non_increasing",
    description: "For R > 0 with no source, stored energy q^2/(2C) + L*i^2/2 can only decrease (passivity); not applicable when R = 0.",
};

/// The capability descriptor for `sim.electrical.rlc`.
pub static DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    id: CapabilityId("sim.electrical.rlc"),
    display_name: "Series RLC circuit",
    category: CapabilityCategory::Electrical,
    source_crate: "scirust-sim",
    summary: "A series RLC circuit driven by a constant source, integrated with fixed-step RK4.",
    maturity: CapabilityMaturity::Stable,
    determinism: DeterminismClass::StrictSameBinarySameTarget,
    supported_backends: &[BackendKind::Cpu],
    supported_precisions: &[PrecisionKind::F64],
    supported_solvers: &[RK4],
    parameters: &[RESISTANCE, INDUCTANCE, CAPACITANCE, V_SOURCE],
    initial_state: &[CHARGE, CURRENT],
    outputs: &[
        OutputDescriptor {
            id: "charge",
            display_name: "Charge",
            unit: "C",
            description: "Capacitor charge over time.",
        },
        OutputDescriptor {
            id: "current",
            display_name: "Current",
            unit: "A",
            description: "Loop current over time.",
        },
        OutputDescriptor {
            id: "capacitor_voltage",
            display_name: "Capacitor voltage",
            unit: "V",
            description: "Derived: charge / capacitance.",
        },
    ],
    verification: VerificationDescriptor {
        checks: &[
            FINITE_SOLUTION_CHECK,
            DAMPING_REGIME_CHECK,
            ENERGY_DECAY_CHECK,
        ],
    },
};

fn classify_damping(resistance: f64, zeta: f64) -> &'static str {
    if resistance == 0.0
    {
        "undamped"
    }
    else if (zeta - 1.0).abs() < 1e-6
    {
        "critically_damped"
    }
    else if zeta < 1.0
    {
        "underdamped"
    }
    else
    {
        "overdamped"
    }
}

/// The `sim.electrical.rlc` adapter.
#[derive(Debug, Default)]
pub struct RlcAdapter;

impl CapabilityAdapter for RlcAdapter {
    fn descriptor(&self) -> &'static CapabilityDescriptor {
        &DESCRIPTOR
    }

    fn validate(&self, scenario: &Scenario) -> Result<ValidatedScenario, ValidationReport> {
        let mut errors = Vec::new();
        errors.extend(check_unknown_model_fields(
            scenario,
            &["resistance", "inductance", "capacitance", "v_source"],
        ));
        errors.extend(check_unknown_state_fields(scenario, &["charge", "current"]));
        for e in [
            resolve_model_scalar(scenario, &RESISTANCE).err(),
            resolve_model_scalar(scenario, &INDUCTANCE).err(),
            resolve_model_scalar(scenario, &CAPACITANCE).err(),
            resolve_model_scalar(scenario, &V_SOURCE).err(),
        ]
        .into_iter()
        .flatten()
        {
            errors.push(e);
        }
        for e in [
            resolve_state_vector(scenario, &CHARGE).err(),
            resolve_state_vector(scenario, &CURRENT).err(),
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
        let scn = scenario.scenario();
        let resistance = resolve_model_scalar(scn, &RESISTANCE).expect("validated");
        let inductance = resolve_model_scalar(scn, &INDUCTANCE).expect("validated");
        let capacitance = resolve_model_scalar(scn, &CAPACITANCE).expect("validated");
        let v_source = resolve_model_scalar(scn, &V_SOURCE).expect("validated");
        let q0 = resolve_state_vector(scn, &CHARGE).expect("validated")[0];
        let i0 = resolve_state_vector(scn, &CURRENT).expect("validated")[0];

        let model = SeriesRlc::new(resistance, inductance, capacitance, v_source)
            .map_err(|e| ExecutionError::InvalidModelState(e.to_string()))?;

        let t0 = scn
            .solver
            .start
            .to_quantity("solver.start")
            .expect("validated")
            .value;
        let t1 = scn
            .solver
            .end
            .to_quantity("solver.end")
            .expect("validated")
            .value;
        let step = scn
            .solver
            .step
            .as_ref()
            .expect("validated: rk4 requires solver.step")
            .to_quantity("solver.step")
            .expect("validated")
            .value;

        let wall_start = std::time::Instant::now();
        let started_at = chrono::Utc::now();

        let traj = simulate(&model, &[q0, i0], t0, t1, step).map_err(|e| {
            sink.emit(RunEvent::Failed(e.to_string()));
            ExecutionError::Numerical(e.to_string())
        })?;

        let charge_series = traj.column(0).expect("dim 0 exists");
        let current_series = traj.column(1).expect("dim 1 exists");
        let capacitor_voltage_series: Vec<f64> =
            charge_series.iter().map(|q| q / capacitance).collect();

        let all_finite = traj.y.iter().flatten().all(|v| v.is_finite());

        let energies: Vec<f64> = traj
            .y
            .iter()
            .map(|row| model.energy(row).expect("state rows always have length 2"))
            .collect();
        let energy_check = if resistance > 0.0
        {
            let e0 = energies[0].abs().max(1e-300);
            let non_increasing = energies.windows(2).all(|w| w[1] <= w[0] + 1e-9 * e0);
            VerificationResult {
                id: "energy_non_increasing".to_string(),
                status: if non_increasing
                {
                    VerificationStatus::Passed
                }
                else
                {
                    VerificationStatus::Failed
                },
                measured: None,
                threshold: None,
                explanation: if non_increasing
                {
                    "stored energy never increased along the trajectory (passivity holds)"
                        .to_string()
                }
                else
                {
                    "stored energy increased somewhere along the trajectory (passivity violated)"
                        .to_string()
                },
            }
        }
        else
        {
            VerificationResult {
                id: "energy_non_increasing".to_string(),
                status: VerificationStatus::NotApplicable,
                measured: None,
                threshold: None,
                explanation: "resistance = 0: this is a lossless LC circuit, not a dissipative one; energy is expected to be conserved, not to decrease".to_string(),
            }
        };

        let zeta = model.damping_ratio();
        let regime = classify_damping(resistance, zeta);
        let last = traj
            .last_state()
            .expect("simulate always returns at least the initial state");

        let result = RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: DESCRIPTOR.id.0.to_string(),
            summary: RunSummary {
                capability_display_name: DESCRIPTOR.display_name.to_string(),
                scenario_name: scn.experiment.name.clone(),
                steps: traj.len() - 1,
                t_start: t0,
                t_end: traj.last_time().unwrap_or(t1),
            },
            axes: vec![AxisDescriptor {
                id: "t".to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
            }],
            series: vec![
                Series {
                    id: "charge".to_string(),
                    display_name: "Charge".to_string(),
                    unit: "C".to_string(),
                    values: charge_series,
                },
                Series {
                    id: "current".to_string(),
                    display_name: "Current".to_string(),
                    unit: "A".to_string(),
                    values: current_series,
                },
                Series {
                    id: "capacitor_voltage".to_string(),
                    display_name: "Capacitor voltage".to_string(),
                    unit: "V".to_string(),
                    values: capacitor_voltage_series,
                },
            ],
            metrics: vec![
                Metric {
                    id: "final_current".to_string(),
                    display_name: "Final current".to_string(),
                    value: MetricValue::Scalar(last[1]),
                    unit: Some("A".to_string()),
                },
                Metric {
                    id: "final_capacitor_voltage".to_string(),
                    display_name: "Final capacitor voltage".to_string(),
                    value: MetricValue::Scalar(last[0] / capacitance),
                    unit: Some("V".to_string()),
                },
                Metric {
                    id: "damping_regime".to_string(),
                    display_name: "Damping regime".to_string(),
                    value: MetricValue::Text(regime.to_string()),
                    unit: None,
                },
                Metric {
                    id: "damping_ratio".to_string(),
                    display_name: "Damping ratio".to_string(),
                    value: MetricValue::Scalar(zeta),
                    unit: None,
                },
            ],
            warnings: vec![],
            verifications: vec![
                VerificationResult {
                    id: "finite_solution".to_string(),
                    status: if all_finite
                    {
                        VerificationStatus::Passed
                    }
                    else
                    {
                        VerificationStatus::Failed
                    },
                    measured: None,
                    threshold: None,
                    explanation: if all_finite
                    {
                        "every recorded value was finite".to_string()
                    }
                    else
                    {
                        "a non-finite value was recorded".to_string()
                    },
                },
                VerificationResult {
                    id: "damping_regime".to_string(),
                    status: VerificationStatus::Passed,
                    measured: Some(zeta),
                    threshold: None,
                    explanation: format!("damping ratio zeta = {zeta:.6} classified as `{regime}`"),
                },
                energy_check,
            ],
            provenance: RunProvenance {
                capability_id: DESCRIPTOR.id.0.to_string(),
                determinism: DESCRIPTOR.determinism,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                started_at_rfc3339: started_at.to_rfc3339(),
                completed_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                elapsed_seconds: wall_start.elapsed().as_secs_f64(),
            },
        };
        crate::result::assert_finite(&result).map_err(ExecutionError::Internal)?;
        sink.emit(RunEvent::Completed);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullEventSink;
    use scirust_studio_schema::parse_toml;

    /// The actual shipped tutorial scenario (`docs/studio/tutorials/`),
    /// varied by resistance — so the file a user is told to run is the file
    /// that is tested. It ships with `resistance = 0.4` (underdamped).
    const TUTORIAL: &str = include_str!("../../../docs/studio/tutorials/rlc_circuit.scirust.toml");

    fn scenario_with_resistance(resistance: f64) -> scirust_studio_schema::Scenario {
        let text = TUTORIAL.replace(
            "resistance = { value = 0.4, unit = \"Ohm\" }",
            &format!("resistance = {{ value = {resistance}, unit = \"Ohm\" }}"),
        );
        parse_toml(&text).unwrap()
    }

    #[test]
    fn underdamped_circuit_decays_and_is_classified_correctly() {
        let adapter = RlcAdapter;
        // R = 0.4, L = 1, C = 0.25 => omega0 = 2, zeta = 0.1 (underdamped).
        let scenario = scenario_with_resistance(0.4);
        let validated = adapter.validate(&scenario).expect("valid");
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes");
        let regime = result
            .metrics
            .iter()
            .find(|m| m.id == "damping_regime")
            .unwrap();
        assert_eq!(regime.value, MetricValue::Text("underdamped".to_string()));
        let decay = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_non_increasing")
            .unwrap();
        assert_eq!(decay.status, VerificationStatus::Passed);
    }

    #[test]
    fn lossless_circuit_is_undamped_and_energy_check_is_not_applicable() {
        let adapter = RlcAdapter;
        let scenario = scenario_with_resistance(0.0);
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();
        let regime = result
            .metrics
            .iter()
            .find(|m| m.id == "damping_regime")
            .unwrap();
        assert_eq!(regime.value, MetricValue::Text("undamped".to_string()));
        let decay = result
            .verifications
            .iter()
            .find(|v| v.id == "energy_non_increasing")
            .unwrap();
        assert_eq!(decay.status, VerificationStatus::NotApplicable);
    }

    #[test]
    fn overdamped_circuit_is_classified_correctly() {
        // zeta = (R/2)*sqrt(C/L) > 1 needs a large R for these L, C.
        let adapter = RlcAdapter;
        let scenario = scenario_with_resistance(20.0);
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();
        let regime = result
            .metrics
            .iter()
            .find(|m| m.id == "damping_regime")
            .unwrap();
        assert_eq!(regime.value, MetricValue::Text("overdamped".to_string()));
    }

    #[test]
    fn capacitor_voltage_is_a_real_derived_series_not_state() {
        let adapter = RlcAdapter;
        let scenario = scenario_with_resistance(0.4);
        let validated = adapter.validate(&scenario).unwrap();
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .unwrap();
        let charge = &result
            .series
            .iter()
            .find(|s| s.id == "charge")
            .unwrap()
            .values;
        let voltage = &result
            .series
            .iter()
            .find(|s| s.id == "capacitor_voltage")
            .unwrap()
            .values;
        for (q, v) in charge.iter().zip(voltage.iter())
        {
            assert!((v - q / 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_negative_resistance() {
        let adapter = RlcAdapter;
        let mut scenario = scenario_with_resistance(0.4);
        scenario.model.insert(
            "resistance".to_string(),
            scirust_studio_schema::ValueWithUnit {
                value: -1.0,
                unit: "Ohm".to_string(),
            },
        );
        let report = adapter.validate(&scenario).unwrap_err();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.code == RESISTANCE.error_code)
        );
    }

    #[test]
    fn v_source_defaults_to_zero_when_omitted() {
        let adapter = RlcAdapter;
        let text: String = TUTORIAL
            .lines()
            .filter(|l| !l.starts_with("v_source ="))
            .collect::<Vec<_>>()
            .join("\n");
        let scenario = parse_toml(&text).unwrap();
        assert!(adapter.validate(&scenario).is_ok());
    }
}
