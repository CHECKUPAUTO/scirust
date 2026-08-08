//! Sequential rank-priority BPE training for canonical ElasticTokenizer artifacts.
//!
//! The historical SciAgent trainer intentionally remains untouched for shard and
//! checkpoint compatibility. This trainer learns exactly one merge per
//! iteration, applies that merge globally, then recounts adjacent pairs. The
//! resulting merge vector is therefore itself the canonical rank order consumed
//! by [`crate::elastic_text_tokenizer::ElasticTextTokenizer`].

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs;
use std::path::Path;

use crate::elastic_profile_store::CANONICAL_BPE_SEMANTICS_V1;
use crate::elastic_tokenizer::TokenId;

const SPECIAL_TOKENS: &[(&str, TokenId)] =
    &[("<pad>", 0), ("<bos>", 1), ("<eos>", 2), ("<unk>", 3)];
const BASE_BYTE_TOKENS: usize = 256;
const BASE_VOCAB_SIZE: usize = SPECIAL_TOKENS.len() + BASE_BYTE_TOKENS;

/// Serialized result of canonical sequential BPE training.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBpeArtifact {
    vocab: BTreeMap<String, TokenId>,
    merges: Vec<(TokenId, TokenId, TokenId)>,
}

impl CanonicalBpeArtifact {
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn ordered_merges(&self) -> &[(TokenId, TokenId, TokenId)] {
        &self.merges
    }

    pub fn to_json_string(&self) -> Result<String, CanonicalBpeTrainError> {
        let value = serde_json::json!({
            "version": "byte_level_v2",
            "merge_semantics": CANONICAL_BPE_SEMANTICS_V1,
            "vocab": self.vocab,
            "merges": self
                .merges
                .iter()
                .map(|(left, right, output)| format!("{left} {right} {output}"))
                .collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&value).map_err(CanonicalBpeTrainError::Json)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), CanonicalBpeTrainError> {
        fs::write(path, self.to_json_string()?).map_err(CanonicalBpeTrainError::Io)
    }
}

/// Canonical one-merge-at-a-time BPE trainer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalBpeTrainer {
    vocab_size: usize,
    min_frequency: u64,
}

impl CanonicalBpeTrainer {
    pub fn new(vocab_size: usize) -> Result<Self, CanonicalBpeTrainError> {
        if vocab_size < BASE_VOCAB_SIZE
        {
            return Err(CanonicalBpeTrainError::VocabTooSmall {
                requested: vocab_size,
                minimum: BASE_VOCAB_SIZE,
            });
        }
        Ok(Self {
            vocab_size,
            min_frequency: 2,
        })
    }

    pub fn min_frequency(mut self, min_frequency: u64) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    pub fn train(&self, texts: &[String]) -> Result<CanonicalBpeArtifact, CanonicalBpeTrainError> {
        let (mut vocab, mut rev) = base_vocab();
        let mut next_id = BASE_VOCAB_SIZE;
        let mut corpus = texts
            .iter()
            .map(|text| {
                text.bytes()
                    .map(|byte| SPECIAL_TOKENS.len() + usize::from(byte))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut merges = Vec::with_capacity(self.vocab_size.saturating_sub(BASE_VOCAB_SIZE));

        while next_id < self.vocab_size
        {
            let mut counts: HashMap<(TokenId, TokenId), u64> = HashMap::new();
            for tokens in &corpus
            {
                for pair in tokens.windows(2)
                {
                    *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
                }
            }

            let best = counts
                .into_iter()
                .filter(|(_, count)| *count >= self.min_frequency)
                .min_by(|(pair_a, count_a), (pair_b, count_b)| {
                    count_b.cmp(count_a).then_with(|| pair_a.cmp(pair_b))
                });
            let Some(((left, right), _count)) = best
            else
            {
                break;
            };

            let token = format!("{}{}", rev[left], rev[right]);
            if vocab.contains_key(&token)
            {
                return Err(CanonicalBpeTrainError::DuplicateTokenString {
                    left,
                    right,
                    token,
                });
            }

            let output = next_id;
            next_id += 1;
            vocab.insert(token.clone(), output);
            rev.push(token);
            merges.push((left, right, output));

            for tokens in &mut corpus
            {
                apply_one_merge(tokens, left, right, output);
            }
        }

        Ok(CanonicalBpeArtifact { vocab, merges })
    }
}

fn base_vocab() -> (BTreeMap<String, TokenId>, Vec<String>) {
    let mut vocab = BTreeMap::new();
    let mut rev = Vec::with_capacity(BASE_VOCAB_SIZE);
    for &(token, id) in SPECIAL_TOKENS
    {
        vocab.insert(token.to_string(), id);
        rev.push(token.to_string());
    }
    for byte in 0u8..=255
    {
        let token = byte_to_unit(byte).to_string();
        let id = SPECIAL_TOKENS.len() + usize::from(byte);
        vocab.insert(token.clone(), id);
        rev.push(token);
    }
    (vocab, rev)
}

fn apply_one_merge(tokens: &mut Vec<TokenId>, left: TokenId, right: TokenId, output: TokenId) {
    if tokens.len() < 2
    {
        return;
    }
    let mut write = 0usize;
    let mut read = 0usize;
    while read < tokens.len()
    {
        if read + 1 < tokens.len() && tokens[read] == left && tokens[read + 1] == right
        {
            tokens[write] = output;
            write += 1;
            read += 2;
        }
        else
        {
            tokens[write] = tokens[read];
            write += 1;
            read += 1;
        }
    }
    tokens.truncate(write);
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

#[derive(Debug)]
pub enum CanonicalBpeTrainError {
    VocabTooSmall {
        requested: usize,
        minimum: usize,
    },
    DuplicateTokenString {
        left: TokenId,
        right: TokenId,
        token: String,
    },
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for CanonicalBpeTrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::VocabTooSmall { requested, minimum } => write!(
                f,
                "canonical BPE vocab size {requested} is below the byte-level minimum {minimum}"
            ),
            Self::DuplicateTokenString { left, right, token } => write!(
                f,
                "canonical BPE merge ({left}, {right}) would duplicate token string {token:?}"
            ),
            Self::Io(error) => write!(f, "canonical BPE artifact I/O failed: {error}"),
            Self::Json(error) => write!(f, "canonical BPE artifact JSON failed: {error}"),
        }
    }
}

impl std::error::Error for CanonicalBpeTrainError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_text_tokenizer::ElasticTextTokenizer;
    use crate::elastic_tokenizer::{BpeKernel, ElasticProfile, ElasticThresholds};

    fn reference_profile() -> ElasticProfile {
        ElasticProfile::new(
            ElasticThresholds::new(8, 16, 32, 64, 128).unwrap(),
            [BpeKernel::Reference; 6],
        )
    }

    #[test]
    fn rejects_vocab_smaller_than_specials_plus_all_bytes() {
        assert!(matches!(
            CanonicalBpeTrainer::new(BASE_VOCAB_SIZE - 1),
            Err(CanonicalBpeTrainError::VocabTooSmall { .. })
        ));
    }

    #[test]
    fn tie_break_is_pair_id_ascending_and_deterministic() {
        // "abac" contains a-b and a-c once each; their counts tie, so (a,b)
        // wins because the base byte ids preserve byte ordering.
        let artifact = CanonicalBpeTrainer::new(BASE_VOCAB_SIZE + 1)
            .unwrap()
            .min_frequency(1)
            .train(&["abac".to_string()])
            .unwrap();
        let a = SPECIAL_TOKENS.len() + usize::from(b'a');
        let b = SPECIAL_TOKENS.len() + usize::from(b'b');
        assert_eq!(artifact.ordered_merges()[0], (a, b, BASE_VOCAB_SIZE));
    }

    #[test]
    fn training_is_bit_deterministic() {
        let texts = vec![
            "fn alpha() { alpha(); }".repeat(10),
            "fn beta() { beta(); }".repeat(10),
        ];
        let train = || {
            CanonicalBpeTrainer::new(BASE_VOCAB_SIZE + 24)
                .unwrap()
                .min_frequency(1)
                .train(&texts)
                .unwrap()
        };
        assert_eq!(train(), train());
    }

    #[test]
    fn serialized_artifact_is_consumable_by_elastic_text_tokenizer() {
        let artifact = CanonicalBpeTrainer::new(BASE_VOCAB_SIZE + 16)
            .unwrap()
            .min_frequency(1)
            .train(&["abc abc abc".repeat(8)])
            .unwrap();
        let json = artifact.to_json_string().unwrap();
        let tokenizer = ElasticTextTokenizer::from_json_str(&json, reference_profile()).unwrap();
        let text = "abc abc";
        let ids = tokenizer.encode(text).ids;
        assert_eq!(tokenizer.decode(&ids), text);
    }

    #[test]
    fn learned_merge_ids_are_strict_rank_order() {
        let artifact = CanonicalBpeTrainer::new(BASE_VOCAB_SIZE + 12)
            .unwrap()
            .min_frequency(1)
            .train(&["aaaaabbbbcccdde".repeat(4)])
            .unwrap();
        for pair in artifact.ordered_merges().windows(2)
        {
            assert!(pair[0].2 < pair[1].2);
        }
    }
}
