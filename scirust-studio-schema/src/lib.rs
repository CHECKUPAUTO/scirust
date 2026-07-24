//! `scirust-studio-schema` — the versioned `.scirust.toml` scenario schema,
//! a small explicit unit-symbol table over `scirust-units`, and scenario
//! validation.
//!
//! This is Phase 1 of the SciRust Studio effort (see
//! `docs/studio/adr/0000-scope-and-sequencing.md`). It has no notion of
//! capabilities, workers, or runs yet — those are Phase 2/3 — but its types
//! are exercised end to end (parse → validate) against a real scenario for
//! `scirust_sim::mechanics::SpringMassDamper` in `scirust-cli`'s integration
//! tests, so the schema is proven against an actual capability rather than
//! designed in a vacuum.
//!
//! # Example
//!
//! ```
//! use scirust_studio_schema::{parse_toml, validate};
//!
//! let scenario = parse_toml(r#"
//! schema_version = 1
//! [experiment]
//! name = "demo"
//! [capability]
//! id = "sim.mechanics.spring_mass_damper"
//! [solver]
//! id = "rk4"
//! start = { value = 0.0, unit = "s" }
//! end = { value = 1.0, unit = "s" }
//! "#).expect("parses");
//! assert!(validate(&scenario, None).is_empty());
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod scenario;
mod units;
mod validate;

pub use error::SchemaError;
pub use scenario::{
    BackendConfig, CURRENT_SCHEMA_VERSION, CapabilityRef, ExperimentMeta, OutputConfig, Scenario,
    SolverConfig, ValueWithUnit, parse_toml,
};
pub use units::{UnitEntry, lookup as lookup_unit};
pub use validate::validate;
