//! V2 scientific algorithm discovery IR.
//!
//! A typed, sectioned, multi-output program representation with statically
//! bounded recurrences — see `docs/SCIRUST_ALGOGEN_IR_V2_ARCHITECTURE.md`.
//! V1 (`super`) remains frozen and byte-stable; this module is additive.

pub mod canonical;
pub mod compat;
pub mod interpret;
pub mod ir;
pub mod types;
pub mod verify;

pub use canonical::{
    CANONICAL_FORMAT_VERSION, CANONICAL_MAGIC, canonical_bytes, canonical_equal, program_digest,
    program_fingerprint,
};
pub use interpret::{
    ExecutionError, ExecutionPolicy, ExecutionResult, FloatPolicy, ValueTensor, execute_program,
};
pub use ir::{Bin, IR_VERSION, Op, Ref, ResearchProgram, Section, Un, ValueId};
pub use types::{DType, ScalarValue, ShapeError, ValueType};
pub use verify::{ProgramError, SectionKind, VerificationLimits, VerifiedProgram, verify_program};

#[cfg(test)]
mod tests;
