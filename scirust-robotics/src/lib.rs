//! # scirust-robotics — deterministic robotics primitives
//!
//! - [`ssm`] — ISO/TS 15066 Speed-and-Separation Monitoring: the protective
//!   separation distance and the maximum robot speed that keeps the cell
//!   provably safe.
//! - [`kinematics`] — planar 2-link forward/inverse kinematics and reach.
//! - [`trajectory`] — rest-to-rest trapezoidal-velocity motion profiles
//!   (bounded velocity and acceleration).
//!
//! Pure Rust, deterministic — the safety layer for collaborative robotics.

pub mod kinematics;
pub mod ssm;
pub mod trajectory;

pub use kinematics::{fk_2link, ik_2link, max_reach};
pub use ssm::{SsmParams, is_safe, max_safe_speed, protective_separation};
pub use trajectory::TrapezoidalProfile;
