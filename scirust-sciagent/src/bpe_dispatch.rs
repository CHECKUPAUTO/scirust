//! Compatibility-safe dispatch between historical and canonical BPE semantics.
//!
//! Untagged tokenizer artifacts are always interpreted through the historical
//! [`BpeTokenizer`] path. Only an explicit `canonical-rank-v1` tag activates
//! [`ElasticTextTokenizer`]. This centralizes the migration rule so corpus
//! generation and later inference code cannot accidentally disagree.

use std::fmt;
use std::fs;
use std::path::Path;

use crate::bpe::BpeTokenizer;
use crate::elastic_profile_store::{
    CANONICAL_BPE_SEMANTICS_V1, ElasticHardwareIdentity, ProfileStoreError, StoredElasticProfile,
};
use crate::elastic_text_tokenizer::{
    BpeMergeSemantics, ElasticTextTokenizer, ElasticTextTokenizerError,
};
use crate::elastic_tokenizer::{ElasticProfile, ElasticThresholds, ThresholdError, TokenId};

const LEGACY_BPE_SEMANTICS_V1: &str = "legacy-parallel-v1";

/// Runtime tokenizer selected from the artifact's explicit semantic version.
#[derive(Clone, Debug)]
pub enum VersionedBpeTokenizer {
    Legacy(BpeTokenizer),
    Canonical(ElasticTextTokenizer),
}

impl VersionedBpeTokenizer {
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, BpeDispatchError> {
        let path = path.as_ref();
        let input = fs::read_to_string(path).map_err(BpeDispatchError::Io)?;
        let semantics = declared_semantics(&input)?;
        match semantics
        {
            BpeMergeSemantics::LegacyParallelV1 =>
            {
                let path = path.to_str().ok_or(BpeDispatchError::NonUtf8Path)?;
                BpeTokenizer::load_json(path)
                    .map(Self::Legacy)
                    .map_err(BpeDispatchError::Io)
            },
            BpeMergeSemantics::CanonicalRankV1 =>
            {
                let profile = reference_profile()?;
                ElasticTextTokenizer::from_json_str(&input, profile)
                    .map(Self::Canonical)
                    .map_err(BpeDispatchError::Canonical)
            },
        }
    }

    pub const fn merge_semantics(&self) -> BpeMergeSemantics {
        match self
        {
            Self::Legacy(_) => BpeMergeSemantics::LegacyParallelV1,
            Self::Canonical(_) => BpeMergeSemantics::CanonicalRankV1,
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self
        {
            Self::Legacy(tokenizer) => tokenizer.vocab_size(),
            Self::Canonical(tokenizer) => tokenizer.vocab_size(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        match self
        {
            Self::Legacy(tokenizer) => tokenizer.encode(text),
            Self::Canonical(tokenizer) => tokenizer.encode(text).ids,
        }
    }

    pub fn encode_with_special(
        &self,
        text: &str,
        prepend_bos: bool,
        append_eos: bool,
    ) -> Vec<TokenId> {
        match self
        {
            Self::Legacy(tokenizer) => tokenizer.encode_with_special(text, prepend_bos, append_eos),
            Self::Canonical(tokenizer) =>
            {
                tokenizer.encode_with_special(text, prepend_bos, append_eos)
            },
        }
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        match self
        {
            Self::Legacy(tokenizer) => tokenizer.decode(ids),
            Self::Canonical(tokenizer) => tokenizer.decode(ids),
        }
    }

    pub fn elastic_profile(&self) -> Option<ElasticProfile> {
        match self
        {
            Self::Legacy(_) => None,
            Self::Canonical(tokenizer) => Some(tokenizer.profile()),
        }
    }

    /// Applies a profile object only to canonical semantics. Callers that load
    /// persisted profiles should prefer [`Self::apply_stored_profile`] so the
    /// tokenizer and hardware bindings are verified first.
    pub fn set_elastic_profile(&mut self, profile: ElasticProfile) -> Result<(), BpeDispatchError> {
        match self
        {
            Self::Legacy(_) => Err(BpeDispatchError::LegacyProfileUnsupported),
            Self::Canonical(tokenizer) =>
            {
                tokenizer.set_profile(profile);
                Ok(())
            },
        }
    }

    /// Verifies and applies a persisted hardware-local profile.
    pub fn apply_stored_profile(
        &mut self,
        stored: &StoredElasticProfile,
        hardware: &ElasticHardwareIdentity,
    ) -> Result<(), BpeDispatchError> {
        match self
        {
            Self::Legacy(_) => Err(BpeDispatchError::LegacyProfileUnsupported),
            Self::Canonical(tokenizer) =>
            {
                stored
                    .verify_for(tokenizer.ordered_merges(), hardware)
                    .map_err(BpeDispatchError::ProfileStore)?;
                tokenizer.set_profile(stored.profile);
                Ok(())
            },
        }
    }
}

fn declared_semantics(input: &str) -> Result<BpeMergeSemantics, BpeDispatchError> {
    let value: serde_json::Value = serde_json::from_str(input).map_err(BpeDispatchError::Json)?;
    match value.get("merge_semantics")
    {
        None => Ok(BpeMergeSemantics::LegacyParallelV1),
        Some(serde_json::Value::String(value)) if value == LEGACY_BPE_SEMANTICS_V1 =>
        {
            Ok(BpeMergeSemantics::LegacyParallelV1)
        },
        Some(serde_json::Value::String(value)) if value == CANONICAL_BPE_SEMANTICS_V1 =>
        {
            Ok(BpeMergeSemantics::CanonicalRankV1)
        },
        Some(serde_json::Value::String(value)) =>
        {
            Err(BpeDispatchError::UnknownMergeSemantics(value.clone()))
        },
        Some(_) => Err(BpeDispatchError::InvalidMergeSemanticsType),
    }
}

fn reference_profile() -> Result<ElasticProfile, ThresholdError> {
    let thresholds = ElasticThresholds::new(16, 64, 256, 1024, 4096)?;
    Ok(ElasticProfile::reference_only(thresholds))
}

#[derive(Debug)]
pub enum BpeDispatchError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NonUtf8Path,
    InvalidMergeSemanticsType,
    UnknownMergeSemantics(String),
    Canonical(ElasticTextTokenizerError),
    InvalidReferenceProfile(ThresholdError),
    ProfileStore(ProfileStoreError),
    LegacyProfileUnsupported,
}

impl fmt::Display for BpeDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Io(error) => write!(f, "BPE tokenizer I/O failed: {error}"),
            Self::Json(error) => write!(f, "BPE tokenizer JSON failed: {error}"),
            Self::NonUtf8Path => f.write_str("BPE tokenizer path is not valid UTF-8"),
            Self::InvalidMergeSemanticsType =>
            {
                f.write_str("BPE merge_semantics must be a string when present")
            },
            Self::UnknownMergeSemantics(value) =>
            {
                write!(f, "unknown BPE merge semantics `{value}`")
            },
            Self::Canonical(error) => write!(f, "canonical BPE tokenizer failed: {error}"),
            Self::InvalidReferenceProfile(error) =>
            {
                write!(f, "invalid canonical reference profile: {error}")
            },
            Self::ProfileStore(error) => write!(f, "elastic profile rejected: {error}"),
            Self::LegacyProfileUnsupported =>
            {
                f.write_str("elastic execution profiles require canonical BPE semantics")
            },
        }
    }
}

impl std::error::Error for BpeDispatchError {}

impl From<ThresholdError> for BpeDispatchError {
    fn from(value: ThresholdError) -> Self {
        Self::InvalidReferenceProfile(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_semantics_is_legacy() {
        assert_eq!(
            declared_semantics(r#"{"vocab":{},"merges":[]}"#).unwrap(),
            BpeMergeSemantics::LegacyParallelV1
        );
    }

    #[test]
    fn explicit_legacy_semantics_is_legacy() {
        assert_eq!(
            declared_semantics(r#"{"merge_semantics":"legacy-parallel-v1"}"#).unwrap(),
            BpeMergeSemantics::LegacyParallelV1
        );
    }

    #[test]
    fn explicit_canonical_semantics_selects_elastic_path() {
        assert_eq!(
            declared_semantics(r#"{"merge_semantics":"canonical-rank-v1"}"#).unwrap(),
            BpeMergeSemantics::CanonicalRankV1
        );
    }

    #[test]
    fn unknown_semantics_fails_closed() {
        assert!(matches!(
            declared_semantics(r#"{"merge_semantics":"future-v99"}"#),
            Err(BpeDispatchError::UnknownMergeSemantics(_))
        ));
    }

    #[test]
    fn non_string_semantics_fails_closed() {
        for input in [
            r#"{"merge_semantics":null}"#,
            r#"{"merge_semantics":42}"#,
            r#"{"merge_semantics":true}"#,
        ]
        {
            assert!(matches!(
                declared_semantics(input),
                Err(BpeDispatchError::InvalidMergeSemanticsType)
            ));
        }
    }
}
