//! Byte-level Byte-Pair-Encoding (the GPT-2 style).
//!
//! Unlike the character-level [`super::bpe::BpeTokenizer`], the base vocabulary
//! is the **256 byte values**, so there is **no out-of-vocabulary case**: any
//! UTF-8 string round-trips losslessly (`decode(encode(s)) == s`), including
//! accents, emoji, and arbitrary bytes. Training is deterministic — frequency
//! ties are broken by the pair itself — so the same corpus always yields the
//! same merges (the project's reproducibility guarantee).

use std::cmp::Reverse;
use std::collections::HashMap;

/// A deterministic byte-level BPE tokenizer.
pub struct ByteBpeTokenizer {
    /// token byte-sequence → id.
    vocab: HashMap<Vec<u8>, u32>,
    /// id → token byte-sequence (for decoding).
    id_to_bytes: Vec<Vec<u8>>,
    /// Ordered merge rules (applied in this order at encode time).
    merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl ByteBpeTokenizer {
    /// Train on a corpus to a target vocabulary size (clamped to ≥ 256, since
    /// every byte value is always a base token). Deterministic.
    pub fn train(texts: &[&str], vocab_size: usize) -> Self {
        let target = vocab_size.max(256);
        let mut id_to_bytes: Vec<Vec<u8>> = (0..256u16).map(|b| vec![b as u8]).collect();
        let mut vocab: HashMap<Vec<u8>, u32> = id_to_bytes
            .iter()
            .enumerate()
            .map(|(i, b)| (b.clone(), i as u32))
            .collect();
        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        // Each text becomes a sequence of single-byte tokens.
        let mut words: Vec<Vec<Vec<u8>>> = texts
            .iter()
            .map(|t| t.bytes().map(|b| vec![b]).collect())
            .collect();

        while id_to_bytes.len() < target
        {
            // Count adjacent token pairs across the corpus.
            let mut counts: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();
            for w in &words
            {
                for i in 0..w.len().saturating_sub(1)
                {
                    *counts.entry((w[i].clone(), w[i + 1].clone())).or_insert(0) += 1;
                }
            }
            if counts.is_empty()
            {
                break;
            }
            // Most frequent pair; ties broken deterministically (smallest pair).
            let (p1, p2) = counts
                .into_iter()
                .max_by_key(|(pair, c)| (*c, Reverse(pair.clone())))
                .map(|(pair, _)| pair)
                .unwrap();
            let mut merged = p1.clone();
            merged.extend_from_slice(&p2);
            vocab.insert(merged.clone(), id_to_bytes.len() as u32);
            id_to_bytes.push(merged.clone());
            merges.push((p1.clone(), p2.clone()));

            for w in &mut words
            {
                *w = apply_merge(w, &p1, &p2, &merged);
            }
        }

        Self {
            vocab,
            id_to_bytes,
            merges,
        }
    }

    /// Encode any string to token ids (lossless; never fails).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut toks: Vec<Vec<u8>> = text.bytes().map(|b| vec![b]).collect();
        for (p1, p2) in &self.merges
        {
            let mut merged = p1.clone();
            merged.extend_from_slice(p2);
            toks = apply_merge(&toks, p1, p2, &merged);
        }
        // Every resulting token is in the vocab (base bytes always are).
        toks.iter().map(|t| self.vocab[t]).collect()
    }

    /// Decode token ids back to a string. Exact for ids produced by
    /// [`encode`](Self::encode);
    /// uses lossy UTF-8 for arbitrary (e.g. model-generated) id streams.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids
        {
            if let Some(b) = self.id_to_bytes.get(id as usize)
            {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size (≥ 256).
    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Number of learned merges (vocab_size − 256).
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }
}

/// Replace every adjacent `(p1, p2)` in `tokens` with `merged`.
fn apply_merge(tokens: &[Vec<u8>], p1: &[u8], p2: &[u8], merged: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len()
    {
        if i + 1 < tokens.len() && tokens[i] == p1 && tokens[i + 1] == p2
        {
            out.push(merged.to_vec());
            i += 2;
        }
        else
        {
            out.push(tokens[i].clone());
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_arbitrary_utf8_is_lossless() {
        let tok = ByteBpeTokenizer::train(&["the quick brown fox", "le renard brun"], 320);
        for s in [
            "hello",
            "héllo wörld",       // accents (multi-byte UTF-8)
            "café ☕ 🚀 résumé", // emoji + accents
            "完全に未知",        // never-seen script
            "",                  // empty
            "\t\n\0 mixed",      // control bytes
        ]
        {
            assert_eq!(tok.decode(&tok.encode(s)), s, "round-trip failed for {s:?}");
        }
    }

    #[test]
    fn no_out_of_vocabulary() {
        // Train on ASCII only; an emoji never seen still round-trips (byte base).
        let tok = ByteBpeTokenizer::train(&["abcabc"], 300);
        let s = "🦀";
        assert_eq!(tok.decode(&tok.encode(s)), s);
        // Every id is a valid base byte or a learned merge (< vocab_size).
        assert!(
            tok.encode(s)
                .iter()
                .all(|&id| (id as usize) < tok.vocab_size())
        );
    }

    #[test]
    fn training_is_deterministic() {
        let corpus = ["banana bandana", "ananas in panama"];
        let reference = ByteBpeTokenizer::train(&corpus, 320);
        let probe = "a banana in panama";
        let expected = reference.encode(probe);
        for _ in 0..5
        {
            assert_eq!(
                ByteBpeTokenizer::train(&corpus, 320).encode(probe),
                expected
            );
        }
    }

    #[test]
    fn merges_shorten_repetitive_text() {
        let tok = ByteBpeTokenizer::train(&["abababababab abab ab"], 300);
        // "abababab" is 8 bytes; with the learned "ab"/"abab" merges it encodes
        // to far fewer than 8 tokens.
        assert!(tok.encode("abababab").len() < 8);
        assert!(tok.num_merges() > 0);
    }

    #[test]
    fn base_vocab_is_256_bytes() {
        let tok = ByteBpeTokenizer::train(&["x"], 0); // target clamps to 256
        assert_eq!(tok.vocab_size(), 256);
        // Single ASCII byte 'A' encodes to its byte value.
        assert_eq!(tok.encode("A"), vec![b'A' as u32]);
    }
}
