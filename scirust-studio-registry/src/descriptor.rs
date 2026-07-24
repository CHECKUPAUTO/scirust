//! Machine-readable capability descriptors.
//!
//! Every field here is declared as a `&'static` table by the crate that
//! implements the capability's adapter (`scirust-studio-runtime`), the same
//! zero-allocation static-table pattern `scirust-cli`'s own command tables
//! already use. A [`CapabilityDescriptor`] is documentation the type system
//! can check, not prose that can silently drift from the code.

use scirust_units::Dimension;
use serde::{Deserialize, Serialize};

use scirust_studio_command::ErrorCode;

/// A stable capability identifier, e.g. `"sim.mechanics.spring_mass_damper"`.
///
/// `Serialize` only: like every type in this module, `CapabilityId` exists
/// to be read out of `&'static` compile-time tables and written to JSON, not
/// to be parsed back into one (a `&'static str` cannot be produced by a
/// deserializer, which only ever borrows from its own, non-`'static`, input
/// buffer). Runtime crates that need an owned, round-trippable capability
/// id in their *own* result types should use a plain `String`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct CapabilityId(pub &'static str);

impl std::fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Broad grouping used for catalogue filtering. Extend this enum, not a
/// free-text string, when a genuinely new category of capability is added —
/// so `by_category` stays exhaustive and typo-proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityCategory {
    /// Classical/rigid-body mechanics.
    Mechanics,
    /// Orbital dynamics.
    Orbital,
    /// Compartmental epidemic models.
    Epidemiology,
    /// Electrical circuits.
    Electrical,
    /// Chemical kinetics.
    Chemistry,
}

/// The scientific maturity of the *underlying model*, as documented by its
/// own source crate — not the maturity of the Studio adapter around it (a
/// capability is never registered at all unless its adapter is tested; see
/// [`crate::CapabilityRegistry::register`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityMaturity {
    /// Validated against an analytic solution or a conservation law in its
    /// own crate's test suite.
    Stable,
    /// Documented by its own crate as experimental or not empirically
    /// validated.
    Experimental,
}

/// How reproducible a capability's output is, and under what conditions.
/// Mirrors the classification in the original Studio brief (§17); kept here
/// rather than invented per-adapter so every capability states its class
/// from the same fixed vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeterminismClass {
    /// Bit-identical when the same binary runs twice on the same target: no
    /// seed, no threading, no ambient randomness in the computation path.
    StrictSameBinarySameTarget,
    /// Reproducible within a stated numerical tolerance, not necessarily
    /// bit-identical (e.g. across compilers/targets).
    ReproducibleWithinTolerance,
    /// Uses an explicit seed but the result can still depend on the backend
    /// (thread count, SIMD width, GPU vs CPU).
    SeededButBackendDependent,
    /// Inherently stochastic; only the seed is recorded, not bit-identity.
    InherentlyStochasticRecordedSeed,
    /// Not reproducible run to run.
    NonDeterministic,
}

/// Compute backend a capability can run on. Only `Cpu` exists today — there
/// is no GPU-backed Studio worker yet (see `docs/studio/REPOSITORY_AUDIT.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    /// Ordinary CPU execution.
    Cpu,
}

/// Floating-point precision a capability can run at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionKind {
    /// 64-bit IEEE-754 (the only precision any current adapter actually
    /// computes in — `scirust-sim`'s models are `f64` throughout).
    F64,
}

/// A solver a capability can be integrated with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SolverDescriptor {
    /// The `solver.id` string a scenario uses to select this solver.
    pub id: &'static str,
    /// One-line description of the method.
    pub summary: &'static str,
    /// Whether this solver needs a fixed `solver.step`.
    pub fixed_step: bool,
    /// Whether this solver needs `solver.rtol`/`solver.atol` (adaptive).
    pub adaptive_tolerance: bool,
}

/// Scalar vs. fixed-length-vector shape of a parameter or state field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cardinality {
    /// A single number.
    Scalar,
    /// A fixed-length vector, e.g. a 2-D position.
    Vector(usize),
}

/// A `model.*` parameter or `initial_state.*` component a capability accepts.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct FieldDescriptor {
    /// The scenario key, e.g. `"mass"` (looked up as `model.mass`).
    pub canonical_name: &'static str,
    /// Human-facing label, e.g. `"Mass"`.
    pub display_name: &'static str,
    /// Whether the field must be present.
    pub required: bool,
    /// The physical dimension the field's resolved unit must match.
    #[serde(serialize_with = "serialize_dimension")]
    pub dimension: Dimension,
    /// Unit symbols (from `scirust_studio_schema`'s table) accepted for this
    /// field. Informational for now — the adapter checks the *dimension*,
    /// not this exact list, so any unit of the right dimension is accepted;
    /// this list is what a UI should offer as the default choices.
    pub accepted_units: &'static [&'static str],
    /// Minimum accepted SI-coherent value, if scientifically meaningful.
    pub min: Option<f64>,
    /// Whether `min` itself is an accepted value.
    pub min_inclusive: bool,
    /// Maximum accepted SI-coherent value, if scientifically meaningful.
    pub max: Option<f64>,
    /// Whether `max` itself is an accepted value.
    pub max_inclusive: bool,
    /// Default SI-coherent value, if the field may be omitted.
    pub default: Option<f64>,
    /// Scalar or fixed-length-vector shape.
    pub cardinality: Cardinality,
    /// Human-facing description.
    pub description: &'static str,
    /// The `SRST-VAL-*` code raised when this field fails validation.
    #[serde(serialize_with = "serialize_error_code")]
    pub error_code: ErrorCode,
}

/// A named output series or derived metric a capability can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OutputDescriptor {
    /// The output's stable id, e.g. `"position"`.
    pub id: &'static str,
    /// Human-facing label.
    pub display_name: &'static str,
    /// Unit symbol of the output's values.
    pub unit: &'static str,
    /// Human-facing description.
    pub description: &'static str,
}

/// One scientific check a capability's adapter performs at execution time
/// (the check itself runs in `execute()` and produces a
/// `scirust_studio_runtime::VerificationResult`; this only documents that it
/// exists, for the catalogue and for help text).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VerificationCheckDescriptor {
    /// Stable id, e.g. `"population_conservation"`.
    pub id: &'static str,
    /// Human-facing description of what is checked and why.
    pub description: &'static str,
}

/// The set of verification checks a capability performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VerificationDescriptor {
    /// Every check this capability's `execute()` runs.
    pub checks: &'static [VerificationCheckDescriptor],
}

/// A fully machine-readable description of one Studio capability.
///
/// Registering a [`CapabilityDescriptor`] is a claim that a real,
/// executable, tested adapter exists for it (enforced by
/// [`crate::CapabilityRegistry::register`] only in the sense that the
/// registry is populated exclusively from `scirust-studio-runtime`'s adapter
/// table — nothing here can be constructed from a name alone).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CapabilityDescriptor {
    /// Stable identifier.
    pub id: CapabilityId,
    /// Human-facing display name.
    pub display_name: &'static str,
    /// Broad category, for catalogue filtering.
    pub category: CapabilityCategory,
    /// The `scirust-sim` (or other) crate the underlying model comes from.
    pub source_crate: &'static str,
    /// One-line summary.
    pub summary: &'static str,
    /// Scientific maturity of the underlying model.
    pub maturity: CapabilityMaturity,
    /// Reproducibility class of this capability's output.
    pub determinism: DeterminismClass,
    /// Backends this capability can run on.
    pub supported_backends: &'static [BackendKind],
    /// Precisions this capability can run at.
    pub supported_precisions: &'static [PrecisionKind],
    /// Solvers this capability accepts.
    pub supported_solvers: &'static [SolverDescriptor],
    /// Accepted `model.*` parameters.
    pub parameters: &'static [FieldDescriptor],
    /// Accepted `initial_state.*` components.
    pub initial_state: &'static [FieldDescriptor],
    /// Output series/metrics this capability can produce.
    pub outputs: &'static [OutputDescriptor],
    /// Verification checks this capability's adapter performs.
    pub verification: VerificationDescriptor,
}

fn serialize_dimension<S: serde::Serializer>(dim: &Dimension, s: S) -> Result<S::Ok, S::Error> {
    s.collect_str(dim)
}

fn serialize_error_code<S: serde::Serializer>(code: &ErrorCode, s: S) -> Result<S::Ok, S::Error> {
    s.collect_str(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_id_displays_as_its_string() {
        assert_eq!(
            CapabilityId("sim.mechanics.spring_mass_damper").to_string(),
            "sim.mechanics.spring_mass_damper"
        );
    }

    #[test]
    fn field_descriptor_serializes_dimension_and_error_code_as_readable_strings() {
        let field = FieldDescriptor {
            canonical_name: "mass",
            display_name: "Mass",
            required: true,
            dimension: Dimension::MASS,
            accepted_units: &["kg"],
            min: Some(0.0),
            min_inclusive: false,
            max: None,
            max_inclusive: false,
            default: None,
            cardinality: Cardinality::Scalar,
            description: "The mass on the spring.",
            error_code: ErrorCode::new(scirust_studio_command::ErrorFamily::Validation, 100),
        };
        let json = serde_json::to_string(&field).unwrap();
        assert!(json.contains("\"dimension\":\"kg\""), "{json}");
        assert!(json.contains("\"error_code\":\"SRST-VAL-0100\""), "{json}");
    }
}
