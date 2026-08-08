//! Versioned text-facing tokenizer for canonical ElasticTokenizer execution.
//!
//! Existing SciAgent `bpe.json` artifacts predate rank-priority semantics and
//! are intentionally **not** accepted here unless they carry an explicit
//! `merge_semantics` tag. The historical [`crate::bpe::BpeTokenizer`] remains
//! the compatibility path for untagged artifacts and existing checkpoints.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::elastic_engine::{ElasticBpeEngine, ElasticEncoding};
use crate::elastic_profile_store::CANONICAL_BPE_SEMANTICS_V1;
use crate::elastic_tokenizer::{DuplicateMergeRule, ElasticProfile, TokenId};

const SPECIAL_TOKENS: &[(&str, usize)] = &[("<pad>", 0), ("<bos>", 1), ("<eos>", 2), ("<unk>", 3)];
const LEGACY_BPE_SEMANTICS_V1: &str = "legacy-parallel-v1";

/// Merge semantics declared by a tokenizer artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BpeMergeSemantics {
    LegacyParallelV1,
    CanonicalRankV1,
}

impl BpeMergeSemantics {
    pub const fn as_str(self) -> &'static str {
        match self
        {
            Self::LegacyParallelV1 => LEGACY_BPE_SEMANTICS_V1,
            Self::CanonicalRankV1 => CANONICAL_BPE_SEMANTICS_V1,
        }
    }

    fn parse(value: &str) -> Result<Self, ElasticTextTokenizerError> {
        match value
        {
            LEGACY_BPE_SEMANTICS_V1 => Ok(Self::LegacyParallelV1),
            CANONICAL_BPE_SEMANTICS_V1 => Ok(Self::CanonicalRankV1),
            _ => Err(ElasticTextTokenizerError::UnknownMergeSemantics(
                value.to_string(),
            )),
        }
    }
}

/// Text-facing rank-priority BPE tokenizer backed by [`ElasticBpeEngine`].
#[derive(Clone, Debug)]
pub struct ElasticTextTokenizer {
    vocab: BTreeMap<String, TokenId>,
    rev: Vec<String>,
    merges: Vec<(TokenId, TokenId, TokenId)>,
    reversible: bool,
    engine: ElasticBpeEngine,
}

impl ElasticTextTokenizer {
    /// Loads only explicitly canonical tokenizer artifacts.
    pub fn from_json_str(
        input: &str,
        profile: ElasticProfile,
    ) -> Result<Self, ElasticTextTokenizerError> {
        let json: serde_json::Value =
            serde_json::from_str(input).map_err(ElasticTextTokenizerError::Json)?;

        let semantics_raw = json
            .get("merge_semantics")
            .and_then(serde_json::Value::as_str)
            .ok_or(ElasticTextTokenizerError::MissingMergeSemantics)?;
        let semantics = BpeMergeSemantics::parse(semantics_raw)?;
        if semantics != BpeMergeSemantics::CanonicalRankV1
        {
            return Err(ElasticTextTokenizerError::LegacyMergeSemantics);
        }

        let vocab: BTreeMap<String, TokenId> = serde_json::from_value(
            json.get("vocab")
                .cloned()
                .ok_or(ElasticTextTokenizerError::MissingField("vocab"))?,
        )
        .map_err(ElasticTextTokenizerError::Json)?;
        let rev = build_reverse_vocab(&vocab)?;
        let merges = parse_merges(
            json.get("merges")
                .ok_or(ElasticTextTokenizerError::MissingField("merges"))?,
        )?;
        let reversible =
            json.get("version").and_then(serde_json::Value::as_str) == Some("byte_level_v2");
        validate_special_tokens(&vocab)?;
        validate_byte_vocab(&vocab, reversible)?;
        validate_merge_ids(&merges, rev.len())?;
        let engine = ElasticBpeEngine::from_ordered_merges(&merges, profile)?;

        Ok(Self {
            vocab,
            rev,
            merges,
            reversible,
            engine,
        })
    }

    pub fn load_json(
        path: impl AsRef<Path>,
        profile: ElasticProfile,
    ) -> Result<Self, ElasticTextTokenizerError> {
        let input = fs::read_to_string(path).map_err(ElasticTextTokenizerError::Io)?;
        Self::from_json_str(&input, profile)
    }

    pub const fn merge_semantics(&self) -> BpeMergeSemantics {
        BpeMergeSemantics::CanonicalRankV1
    }

    pub const fn profile(&self) -> ElasticProfile {
        self.engine.profile()
    }

    pub fn set_profile(&mut self, profile: ElasticProfile) {
        self.engine.set_profile(profile);
    }

    pub fn ordered_merges(&self) -> &[(TokenId, TokenId, TokenId)] {
        &self.merges
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn special_id(&self, name: &str) -> TokenId {
        *self.vocab.get(name).unwrap_or(&3)
    }

    /// Encodes the complete text as one BPE piece, matching the current
    /// SciAgent tokenizer's piece semantics. Execution classes choose only the
    /// reduction kernel; they do not introduce regex or arbitrary boundaries.
    pub fn encode(&self, text: &str) -> ElasticEncoding {
        let base_ids = self.base_ids(text);
        self.engine.encode_ids(&base_ids, text.len())
    }

    pub fn encode_with_special(
        &self,
        text: &str,
        prepend_bos: bool,
        append_eos: bool,
    ) -> Vec<TokenId> {
        let mut ids = Vec::new();
        if prepend_bos
        {
            ids.push(self.special_id("<bos>"));
        }
        ids.extend(self.encode(text).ids);
        if append_eos
        {
            ids.push(self.special_id("<eos>"));
        }
        ids
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        if self.reversible
        {
            let mut bytes = Vec::new();
            for &id in ids
            {
                if id < SPECIAL_TOKENS.len()
                {
                    continue;
                }
                if let Some(token) = self.rev.get(id)
                {
                    for ch in token.chars()
                    {
                        if let Some(byte) = unit_to_byte(ch)
                        {
                            bytes.push(byte);
                        }
                    }
                }
            }
            return String::from_utf8_lossy(&bytes).into_owned();
        }

        let mut output = String::new();
        for &id in ids
        {
            if let Some(token) = self.rev.get(id)
            {
                if !is_legacy_non_text_token(token)
                {
                    output.push_str(token);
                }
            }
        }
        output
    }

    fn base_ids(&self, text: &str) -> Vec<TokenId> {
        text.bytes()
            .map(|byte| {
                let key = if self.reversible
                {
                    byte_to_unit(byte).to_string()
                }
                else
                {
                    byte_to_legacy_string(byte)
                };
                *self.vocab.get(&key).unwrap_or(&self.special_id("<unk>"))
            })
            .collect()
    }
}

fn parse_merges(
    value: &serde_json::Value,
) -> Result<Vec<(TokenId, TokenId, TokenId)>, ElasticTextTokenizerError> {
    let array = value
        .as_array()
        .ok_or(ElasticTextTokenizerError::InvalidField("merges"))?;
    let mut merges = Vec::with_capacity(array.len());
    for (index, raw) in array.iter().enumerate()
    {
        let text = raw
            .as_str()
            .ok_or(ElasticTextTokenizerError::InvalidMergeRule(index))?;
        let mut parts = text.split_whitespace();
        let left = parts
            .next()
            .and_then(|part| part.parse::<TokenId>().ok())
            .ok_or(ElasticTextTokenizerError::InvalidMergeRule(index))?;
        let right = parts
            .next()
            .and_then(|part| part.parse::<TokenId>().ok())
            .ok_or(ElasticTextTokenizerError::InvalidMergeRule(index))?;
        let output = parts
            .next()
            .and_then(|part| part.parse::<TokenId>().ok())
            .ok_or(ElasticTextTokenizerError::InvalidMergeRule(index))?;
        if parts.next().is_some()
        {
            return Err(ElasticTextTokenizerError::InvalidMergeRule(index));
        }
        merges.push((left, right, output));
    }
    Ok(merges)
}

fn build_reverse_vocab(
    vocab: &BTreeMap<String, TokenId>,
) -> Result<Vec<String>, ElasticTextTokenizerError> {
    let Some(max_id) = vocab.values().copied().max()
    else
    {
        return Err(ElasticTextTokenizerError::EmptyVocab);
    };
    let mut rev = vec![None::<String>; max_id.saturating_add(1)];
    for (token, &id) in vocab
    {
        let slot = rev
            .get_mut(id)
            .ok_or(ElasticTextTokenizerError::InvalidVocabId(id))?;
        if slot.replace(token.clone()).is_some()
        {
            return Err(ElasticTextTokenizerError::DuplicateVocabId(id));
        }
    }
    if let Some((id, _)) = rev.iter().enumerate().find(|(_, token)| token.is_none())
    {
        return Err(ElasticTextTokenizerError::SparseVocab(id));
    }
    Ok(rev.into_iter().map(Option::unwrap).collect())
}

fn validate_special_tokens(
    vocab: &BTreeMap<String, TokenId>,
) -> Result<(), ElasticTextTokenizerError> {
    for &(token, expected_id) in SPECIAL_TOKENS
    {
        if vocab.get(token).copied() != Some(expected_id)
        {
            return Err(ElasticTextTokenizerError::InvalidSpecialToken(token));
        }
    }
    Ok(())
}

fn validate_byte_vocab(
    vocab: &BTreeMap<String, TokenId>,
    reversible: bool,
) -> Result<(), ElasticTextTokenizerError> {
    for byte in 0u8..=255
    {
        let key = if reversible
        {
            byte_to_unit(byte).to_string()
        }
        else
        {
            byte_to_legacy_string(byte)
        };
        if !vocab.contains_key(&key)
        {
            return Err(ElasticTextTokenizerError::MissingByteToken(byte));
        }
    }
    Ok(())
}

fn validate_merge_ids(
    merges: &[(TokenId, TokenId, TokenId)],
    vocab_size: usize,
) -> Result<(), ElasticTextTokenizerError> {
    for (rule_index, &(left, right, output)) in merges.iter().enumerate()
    {
        for token_id in [left, right, output]
        {
            if token_id >= vocab_size
            {
                return Err(ElasticTextTokenizerError::MergeTokenOutOfVocab {
                    rule_index,
                    token_id,
                    vocab_size,
                });
            }
        }
    }
    Ok(())
}

fn byte_to_unit(byte: u8) -> char {
    let codepoint = if byte < 128
    {
        u32::from(byte)
    }
    else
    {
        256 + (u32::from(byte) - 128)
    };
    char::from_u32(codepoint).expect("byte-unit codepoint is always valid")
}

fn unit_to_byte(ch: char) -> Option<u8> {
    let codepoint = ch as u32;
    if codepoint < 128
    {
        Some(codepoint as u8)
    }
    else if (256..384).contains(&codepoint)
    {
        Some((codepoint - 256 + 128) as u8)
    }
    else
    {
        None
    }
}

fn byte_to_legacy_string(byte: u8) -> String {
    String::from_utf8(vec![byte]).unwrap_or_else(|_| format!("<{byte}>"))
}

fn is_legacy_non_text_token(token: &str) -> bool {
    if SPECIAL_TOKENS.iter().any(|(special, _)| *special == token)
    {
        return true;
    }
    token.len() >= 3
        && token.starts_with('<')
        && token.ends_with('>')
        && token[1..token.len() - 1]
            .bytes()
            .all(|byte| byte.is_ascii_digit())
}

#[derive(Debug)]
pub enum ElasticTextTokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    MissingField(&'static str),
    InvalidField(&'static str),
    MissingMergeSemantics,
    UnknownMergeSemantics(String),
    LegacyMergeSemantics,
    InvalidMergeRule(usize),
    MergeTokenOutOfVocab {
        rule_index: usize,
        token_id: TokenId,
        vocab_size: usize,
    },
    EmptyVocab,
    InvalidVocabId(TokenId),
    DuplicateVocabId(TokenId),
    SparseVocab(TokenId),
    InvalidSpecialToken(&'static str),
    MissingByteToken(u8),
    DuplicateMergeRule(DuplicateMergeRule),
}

impl fmt::Display for ElasticTextTokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Io(error) => write!(f, "elastic tokenizer I/O failed: {error}"),
            Self::Json(error) => write!(f, "elastic tokenizer JSON failed: {error}"),
            Self::MissingField(field) => write!(f, "elastic tokenizer missing field `{field}`"),
            Self::InvalidField(field) => write!(f, "elastic tokenizer invalid field `{field}`"),
            Self::MissingMergeSemantics => f.write_str(
                "elastic tokenizer requires explicit `merge_semantics`; untagged artifacts remain legacy",
            ),
            Self::UnknownMergeSemantics(value) => {
                write!(f, "unknown elastic tokenizer merge semantics `{value}")
            },
            Self::LegacyMergeSemantics => f.write_str(
                "legacy parallel BPE semantics must use the historical BpeTokenizer compatibility path",
            ),
            Self::InvalidMergeRule(index) => {
                write!(f, "invalid elastic tokenizer merge rule at index {index}")
            },
            Self::MergeTokenOutOfVocab {
                rule_index,
                token_id,
                vocab_size,
            } => write!(
                f,
                "elastic tokenizer merge rule {rule_index} references token id {token_id} outside vocabulary size {vocab_size}"
            ),
            Self::EmptyVocab => f.write_str("elastic tokenizer vocabulary is empty"),
            Self::InvalidVocabId(id) => write!(f, "elastic tokenizer invalid vocab id {id}"),
            Self::DuplicateVocabId(id) => write!(f, "elastic tokenizer duplicate vocab id {id}"),
            Self::SparseVocab(id) => write!(f, "elastic tokenizer missing vocab id {id}"),
            Self::InvalidSpecialToken(token) => {
                write!(f, "elastic tokenizer invalid special token `{token}")
            },
            Self::MissingByteToken(byte) => {
                write!(f, "elastic tokenizer missing base token for byte {byte}")
            },
            Self::DuplicateMergeRule(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ElasticTextTokenizerError {}

impl From<DuplicateMergeRule> for ElasticTextTokenizerError {
    fn from(value: DuplicateMergeRule) -> Self {
        Self::DuplicateMergeRule(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_tokenizer::{BpeKernel, ElasticThresholds};

    fn profile(kernel: BpeKernel) -> ElasticProfile {
        ElasticProfile::new(
            ElasticThresholds::new(8, 16, 32, 64, 128).unwrap(),
            [kernel; 6],
        )
    }

    fn canonical_test_json() -> String {
        let mut vocab = serde_json::Map::new();
        for &(token, id) in SPECIAL_TOKENS
        {
            vocab.insert(token.to_string(), serde_json::json!(id));
        }
        for byte in 0u8..=255
        {
            vocab.insert(
                byte_to_unit(byte).to_string(),
                serde_json::json!(usize::from(byte) + 4),
            );
        }
        // a=101, b=102, c=103 in the byte base because ids are byte+4.
        // `b+c` has the higher priority but occurs to the right of `a+b`.
        vocab.insert("bc".to_string(), serde_json::json!(260));
        vocab.insert("ab".to_string(), serde_json::json!(261));
        serde_json::json!({
            "version": "byte_level_v2",
            "merge_semantics": CANONICAL_BPE_SEMANTICS_V1,
            "vocab": vocab,
            "merges": ["102 103 260", "101 102 261"],
        })
        .to_string()
    }

    #[test]
    fn untagged_artifact_is_never_silently_upgraded() {
        let mut value: serde_json::Value = serde_json::from_str(&canonical_test_json()).unwrap();
        value.as_object_mut().unwrap().remove("merge_semantics");
        let input = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::Reference)),
            Err(ElasticTextTokenizerError::MissingMergeSemantics)
        ));
    }

    #[test]
    fn explicitly_legacy_artifact_is_rejected_by_canonical_path() {
        let mut value: serde_json::Value = serde_json::from_str(&canonical_test_json()).unwrap();
        value["merge_semantics"] = serde_json::json!(LEGACY_BPE_SEMANTICS_V1);
        let input = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::Reference)),
            Err(ElasticTextTokenizerError::LegacyMergeSemantics)
        ));
    }

    #[test]
    fn merge_ids_must_exist_in_vocab() {
        let mut value: serde_json::Value = serde_json::from_str(&canonical_test_json()).unwrap();
        value["merges"] = serde_json::json!(["102 103 999999"]);
        let input = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::Reference)),
            Err(ElasticTextTokenizerError::MergeTokenOutOfVocab {
                rule_index: 0,
                token_id: 999999,
                ..
            })
        ));
    }

    #[test]
    fn canonical_text_encoding_respects_rank_over_left_position() {
        let tokenizer = ElasticTextTokenizer::from_json_str(
            &canonical_test_json(),
            profile(BpeKernel::Reference),
        )
        .unwrap();
        assert_eq!(tokenizer.encode("abc").ids, vec![101, 260]);
        assert_eq!(tokenizer.decode(&[101, 260]), "abc");
    }

    #[test]
    fn execution_kernel_changes_without_changing_text_token_ids() {
        let input = canonical_test_json();
        let reference =
            ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::Reference)).unwrap();
        let tiny =
            ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::TinyScan)).unwrap();
        let indexed =
            ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::Indexed)).unwrap();
        let heap = ElasticTextTokenizer::from_json_str(&input, profile(BpeKernel::Heap)).unwrap();

        let expected = reference.encode("abc").ids;
        assert_eq!(tiny.encode("abc").ids, expected);
        assert_eq!(indexed.encode("abc").ids, expected);
        assert_eq!(heap.encode("abc").ids, expected);
    }
}
