//! Explicit versioned serialization boundary for V2 programs.
//!
//! JSON is a transport format, never canonical identity. Canonical identity is
//! defined only by [`super::canonical::canonical_bytes`]. The envelope rejects
//! every version mismatch instead of silently assigning changed semantics to
//! an older payload.

use serde::{Deserialize, Serialize};

use super::canonical::CANONICALIZATION_VERSION;
use super::ir::{IR_VERSION, ResearchProgram};
use super::semantics::NumericalSemantics;
use super::verify::{ProgramError, VerificationLimits, verify_program};

/// Version of the serde transport envelope.
pub const SERIALIZATION_VERSION: u32 = 1;

/// Self-describing transport envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramEnvelope {
    pub serialization_version: u32,
    pub ir_version: u32,
    pub canonicalization_version: u32,
    pub numerical_semantics: NumericalSemantics,
    pub program: ResearchProgram,
}

impl ProgramEnvelope {
    pub fn new(program: ResearchProgram) -> Self {
        Self {
            serialization_version: SERIALIZATION_VERSION,
            ir_version: IR_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            numerical_semantics: program.semantics,
            program,
        }
    }
}

/// Deterministic category for a serialization/migration failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializationError {
    Json(String),
    UnsupportedSerializationVersion {
        found: u32,
        supported: u32,
    },
    UnsupportedIrVersion {
        found: u32,
        supported: u32,
    },
    UnsupportedCanonicalizationVersion {
        found: u32,
        supported: u32,
    },
    SemanticRegimeMismatch {
        envelope: NumericalSemantics,
        program: NumericalSemantics,
    },
    Verification(ProgramError),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Json(error) => write!(formatter, "invalid V2 program JSON: {error}"),
            Self::UnsupportedSerializationVersion { found, supported } => write!(
                formatter,
                "serialization version {found} is unsupported; this build accepts {supported}"
            ),
            Self::UnsupportedIrVersion { found, supported } => write!(
                formatter,
                "IR version {found} is unsupported; this build accepts {supported}"
            ),
            Self::UnsupportedCanonicalizationVersion { found, supported } => write!(
                formatter,
                "canonicalization version {found} is unsupported; this build accepts {supported}"
            ),
            Self::SemanticRegimeMismatch { envelope, program } => write!(
                formatter,
                "envelope semantic regime {envelope:?} disagrees with program regime {program:?}"
            ),
            Self::Verification(error) =>
            {
                write!(formatter, "deserialized program is invalid: {error}")
            },
        }
    }
}

impl std::error::Error for SerializationError {}

impl From<ProgramError> for SerializationError {
    fn from(error: ProgramError) -> Self {
        Self::Verification(error)
    }
}

/// Serialize a verified program into a versioned JSON envelope.
pub fn serialize_program(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<String, SerializationError> {
    verify_program(program, limits)?;
    serde_json::to_string_pretty(&ProgramEnvelope::new(program.clone()))
        .map_err(|error| SerializationError::Json(error.to_string()))
}

/// Deserialize a versioned envelope, reject every mismatch, and verify the
/// program before returning it.
pub fn deserialize_program(
    json: &str,
    limits: VerificationLimits,
) -> Result<ResearchProgram, SerializationError> {
    let envelope: ProgramEnvelope =
        serde_json::from_str(json).map_err(|error| SerializationError::Json(error.to_string()))?;
    if envelope.serialization_version != SERIALIZATION_VERSION
    {
        return Err(SerializationError::UnsupportedSerializationVersion {
            found: envelope.serialization_version,
            supported: SERIALIZATION_VERSION,
        });
    }
    if envelope.ir_version != IR_VERSION
    {
        return Err(SerializationError::UnsupportedIrVersion {
            found: envelope.ir_version,
            supported: IR_VERSION,
        });
    }
    if envelope.canonicalization_version != CANONICALIZATION_VERSION
    {
        return Err(SerializationError::UnsupportedCanonicalizationVersion {
            found: envelope.canonicalization_version,
            supported: CANONICALIZATION_VERSION,
        });
    }
    if envelope.numerical_semantics != envelope.program.semantics
    {
        return Err(SerializationError::SemanticRegimeMismatch {
            envelope: envelope.numerical_semantics,
            program: envelope.program.semantics,
        });
    }
    verify_program(&envelope.program, limits)?;
    Ok(envelope.program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::{DType, Op, Ref, Section, Un, ValueType};

    fn program() -> ResearchProgram {
        ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
            vec![0],
        )
    }

    #[test]
    fn envelope_round_trip_is_explicit_and_verified() {
        let json = serialize_program(&program(), VerificationLimits::default()).unwrap();
        let decoded = deserialize_program(&json, VerificationLimits::default()).unwrap();
        assert_eq!(decoded, program());
        assert!(json.contains("\"serialization_version\": 1"));
        assert!(json.contains("\"numerical_semantics\": \"StrictIeee\""));
    }

    #[test]
    fn every_version_or_regime_mismatch_is_rejected() {
        let mut envelope = ProgramEnvelope::new(program());
        envelope.serialization_version += 1;
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(matches!(
            deserialize_program(&json, VerificationLimits::default()),
            Err(SerializationError::UnsupportedSerializationVersion { .. })
        ));

        let mut envelope = ProgramEnvelope::new(program());
        envelope.numerical_semantics = NumericalSemantics::FiniteOnly;
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(matches!(
            deserialize_program(&json, VerificationLimits::default()),
            Err(SerializationError::SemanticRegimeMismatch { .. })
        ));
    }

    #[test]
    fn bare_program_json_is_not_silently_migrated() {
        let json = serde_json::to_string(&program()).unwrap();
        assert!(matches!(
            deserialize_program(&json, VerificationLimits::default()),
            Err(SerializationError::Json(_))
        ));
    }
}
