//! # scirust-maritime — autonomous-vessel navigation & control primitives
//!
//! Deterministic, pure-Rust building blocks for maritime autonomy
//! (IMO MASS Code 2026, DNV AROS, IACS UR E26/E27 context — see
//! `docs/DOMAIN_ROADMAP.md` D5):
//!
//! - [`colregs`] — COLREG encounter-type classification (head-on /
//!   crossing / overtaking) from relative bearing.
//! - [`cpa_tcpa`] — closest point of approach / time to CPA, the standard
//!   collision-risk assessment for constant-velocity tracks.
//! - [`thrust_allocation`] — weighted-pseudo-inverse thrust allocation for
//!   dynamic positioning (DP), the static optimization layer that turns a
//!   desired generalized force into individual thruster commands.
//!
//! **Honest scope**: this crate covers the geometric/optimization
//! primitives, not a full autonomy stack. It does not implement COLREG
//! Rules 11-18 vessel-status logic, a DP observer/reference-model/MPC
//! loop, or thruster saturation/interaction modeling — see each module's
//! doc comment for its specific boundary.

pub mod colregs;
pub mod cpa_tcpa;
pub mod thrust_allocation;

pub use colregs::{EncounterType, classify_encounter, relative_bearing_deg};
pub use cpa_tcpa::{CpaTcpa, cpa_tcpa, is_collision_risk, velocity_from_heading};
pub use thrust_allocation::{Thruster, allocate_thrust};
