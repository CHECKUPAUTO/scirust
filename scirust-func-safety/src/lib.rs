//! SciRust Functional Safety — ISO 26262 / IEC 61508
//!
//! Provides ASIL classification, requirement tracing, fault injection,
//! and degraded-mode handling for safety-critical automotive AI.
//!
//! ## Modules
//! - **asil** — ASIL (Automotive Safety Integrity Level) classification
//! - **requirements** — Requirement traceability matrix
//! - **fault_injection** — Fault injection testing for neural networks
//! - **degraded_mode** — Fallback and graceful degradation strategies
//! - **audit** — Immutable audit log of safety-critical decisions

pub mod asil;
pub mod audit;
pub mod degraded_mode;
pub mod evidence;
pub mod fault_injection;
pub mod requirements;
pub mod simplex;

pub use asil::{AsilConfig, AsilLevel, SafetyGoal};
pub use audit::{AuditChain, AuditEntry, AuditLog};
pub use degraded_mode::{DegradationAction, DegradationLevel, DegradedModeController};
pub use evidence::{EvidencePack, EvidenceRecord, fingerprint_f32, verify_chain};
pub use fault_injection::{FaultInjector, FaultResult, FaultType};
pub use requirements::{Requirement, RequirementStatus, TraceabilityMatrix};
pub use simplex::{SafetyDecision, SimplexMonitor};
