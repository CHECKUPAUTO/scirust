//! The versioned `.scirust.toml` scenario schema.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;

/// The schema version this build reads. There is deliberately no migration
/// path yet — the brief is explicit that a fake migration (with nothing on
/// the other end) is worse than no migration, so one will be added when a
/// version 2 actually exists.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// A parsed, not-yet-validated `.scirust.toml` scenario. Parsing only checks
/// TOML syntax and the shape of this struct; call [`crate::validate`] to
/// check the *content* (unit, range, and cross-field rules).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    /// The scenario schema version; must equal [`CURRENT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Human-facing metadata about the experiment.
    pub experiment: ExperimentMeta,
    /// Which capability this scenario runs.
    pub capability: CapabilityRef,
    /// Execution backend selection.
    #[serde(default)]
    pub backend: BackendConfig,
    /// Named model parameters, each tagged with its unit.
    #[serde(default)]
    pub model: BTreeMap<String, ValueWithUnit>,
    /// Named initial-state components; each may be a vector (e.g. a
    /// multi-dimensional position), so the value is a list even when a
    /// capability's state is one-dimensional.
    #[serde(default)]
    pub initial_state: BTreeMap<String, Vec<ValueWithUnit>>,
    /// Integration/solver configuration.
    pub solver: SolverConfig,
    /// What to record and how densely.
    #[serde(default)]
    pub outputs: OutputConfig,
}

/// Human-facing metadata about the experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentMeta {
    /// A short name for the experiment.
    pub name: String,
    /// An optional longer description.
    #[serde(default)]
    pub description: Option<String>,
    /// The seed driving any randomness the capability uses. Absent for
    /// capabilities that have none.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Which capability a scenario runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityRef {
    /// The capability id, e.g. `"sim.mechanics.spring_mass_damper"`.
    pub id: String,
}

/// Execution backend selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendConfig {
    /// The backend kind. Only `"cpu"` is currently supported — there is no
    /// GPU-backed Studio worker yet, so anything else is a validation error,
    /// not a silent fallback.
    #[serde(default = "default_backend_kind")]
    pub kind: String,
    /// The floating-point precision. Only `"f32"` and `"f64"` are supported.
    #[serde(default = "default_precision")]
    pub precision: String,
    /// An optional thread-count hint.
    #[serde(default)]
    pub threads: Option<u32>,
    /// Whether the run must be deterministic. Defaults to `true`: Studio's
    /// scientific-integrity rule is that non-determinism is something a
    /// scenario opts into, not the default.
    #[serde(default = "default_true")]
    pub deterministic: bool,
}

fn default_backend_kind() -> String {
    "cpu".to_string()
}

fn default_precision() -> String {
    "f64".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for BackendConfig {
    fn default() -> Self {
        BackendConfig {
            kind: default_backend_kind(),
            precision: default_precision(),
            threads: None,
            deterministic: true,
        }
    }
}

/// A numeric value tagged with its unit symbol, e.g. `{ value = 9.81, unit =
/// "m/s^2" }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueWithUnit {
    /// The numeric magnitude, in the given unit (not necessarily SI-coherent
    /// until resolved).
    pub value: f64,
    /// The unit symbol; must be one `scirust-studio-schema::units::lookup`
    /// recognises.
    pub unit: String,
}

impl ValueWithUnit {
    /// Resolve to a checked [`scirust_units::Quantity`] in SI-coherent
    /// units. `field` names the scenario field this value came from, used
    /// only to build a precise [`SchemaError`].
    pub fn to_quantity(&self, field: &str) -> Result<scirust_units::Quantity, SchemaError> {
        if !self.value.is_finite()
        {
            return Err(SchemaError::NonFinite {
                field: field.to_string(),
                value: self.value,
            });
        }
        let entry = crate::units::lookup(&self.unit).ok_or_else(|| SchemaError::UnknownUnit {
            field: field.to_string(),
            unit: self.unit.clone(),
        })?;
        Ok(scirust_units::Quantity::new(
            self.value * entry.to_si_factor,
            entry.dimension,
        ))
    }
}

/// Integration/solver configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig {
    /// The solver id, e.g. `"rk4"` or `"dopri5"`. Which ids a given
    /// capability actually accepts is the capability adapter's concern, not
    /// this generic schema's.
    pub id: String,
    /// Simulation start time.
    pub start: ValueWithUnit,
    /// Simulation end time; must be strictly after `start`.
    pub end: ValueWithUnit,
    /// Fixed step size, for fixed-step solvers.
    #[serde(default)]
    pub step: Option<ValueWithUnit>,
    /// Relative tolerance, for adaptive solvers.
    #[serde(default)]
    pub rtol: Option<f64>,
    /// Absolute tolerance, for adaptive solvers.
    #[serde(default)]
    pub atol: Option<f64>,
}

/// What to record and how densely.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    /// Names of series to record, e.g. `["position", "velocity", "energy"]`.
    #[serde(default)]
    pub record: Vec<String>,
    /// Record every Nth accepted step. `None` means "every step."
    #[serde(default)]
    pub sample_every: Option<u32>,
}

/// Parse a `.scirust.toml` scenario from its source text.
///
/// This only performs TOML/shape parsing; the returned [`Scenario`] is not
/// validated. On a syntax error, `toml`'s own diagnostic (which already
/// includes the offending line and column) is preserved verbatim inside
/// [`SchemaError::Parse`] rather than replaced with a vaguer message.
pub fn parse_toml(source: &str) -> Result<Scenario, SchemaError> {
    toml::from_str(source).map_err(|e| SchemaError::Parse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPRING_MASS_DAMPER_TOML: &str = r#"
schema_version = 1

[experiment]
name = "Spring-mass-damper demo"
description = "Underdamped mass on a spring with viscous damping"
seed = 42

[capability]
id = "sim.mechanics.spring_mass_damper"

[backend]
kind = "cpu"
precision = "f64"
deterministic = true

[model]
mass = { value = 1.0, unit = "kg" }
damping = { value = 0.2, unit = "kg/s" }
stiffness = { value = 4.0, unit = "kg/s^2" }

[initial_state]
position = [{ value = 1.0, unit = "m" }]
velocity = [{ value = 0.0, unit = "m/s" }]

[solver]
id = "rk4"
start = { value = 0.0, unit = "s" }
end = { value = 10.0, unit = "s" }
step = { value = 0.01, unit = "s" }

[outputs]
record = ["position", "velocity", "energy"]
sample_every = 1
"#;

    #[test]
    fn parses_a_real_scenario() {
        let scenario = parse_toml(SPRING_MASS_DAMPER_TOML).expect("valid scenario");
        assert_eq!(scenario.schema_version, 1);
        assert_eq!(scenario.experiment.name, "Spring-mass-damper demo");
        assert_eq!(scenario.experiment.seed, Some(42));
        assert_eq!(scenario.capability.id, "sim.mechanics.spring_mass_damper");
        assert_eq!(scenario.backend.kind, "cpu");
        assert_eq!(scenario.model.get("mass").unwrap().value, 1.0);
        assert_eq!(scenario.initial_state.get("position").unwrap().len(), 1);
        assert_eq!(scenario.solver.id, "rk4");
        assert_eq!(
            scenario.outputs.record,
            vec!["position", "velocity", "energy"]
        );
    }

    #[test]
    fn backend_and_outputs_default_when_omitted() {
        let minimal = r#"
schema_version = 1
[experiment]
name = "minimal"
[capability]
id = "sim.mechanics.spring_mass_damper"
[solver]
id = "rk4"
start = { value = 0.0, unit = "s" }
end = { value = 1.0, unit = "s" }
"#;
        let scenario = parse_toml(minimal).expect("valid scenario");
        assert_eq!(scenario.backend, BackendConfig::default());
        assert_eq!(scenario.outputs, OutputConfig::default());
        assert!(scenario.model.is_empty());
    }

    #[test]
    fn syntax_error_preserves_toml_diagnostic_with_position() {
        let broken = "schema_version = [1"; // unterminated array
        let err = parse_toml(broken).unwrap_err();
        match err
        {
            SchemaError::Parse(msg) => assert!(msg.contains("line"), "message was: {msg}"),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn value_with_unit_resolves_a_known_unit() {
        let v = ValueWithUnit {
            value: 2.0,
            unit: "kg".to_string(),
        };
        let q = v.to_quantity("model.mass").expect("kg is supported");
        assert_eq!(q.dim, scirust_units::Dimension::MASS);
        assert_eq!(q.value, 2.0);
    }

    #[test]
    fn value_with_unit_rejects_unknown_unit() {
        let v = ValueWithUnit {
            value: 2.0,
            unit: "lb".to_string(),
        };
        assert!(matches!(
            v.to_quantity("model.mass"),
            Err(SchemaError::UnknownUnit { .. })
        ));
    }

    #[test]
    fn value_with_unit_rejects_non_finite() {
        let v = ValueWithUnit {
            value: f64::NAN,
            unit: "kg".to_string(),
        };
        assert!(matches!(
            v.to_quantity("model.mass"),
            Err(SchemaError::NonFinite { .. })
        ));
    }

    #[test]
    fn round_trips_through_serialize_and_parse() {
        let scenario = parse_toml(SPRING_MASS_DAMPER_TOML).unwrap();
        let text = toml::to_string(&scenario).expect("serializes");
        let reparsed = parse_toml(&text).expect("re-parses its own output");
        assert_eq!(scenario, reparsed);
    }
}
