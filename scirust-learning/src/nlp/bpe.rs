use super::tokenization::Tokenizer;
use std::cmp::Reverse;
use std::collections::HashMap;

/// Byte Pair Encoding (BPE) Tokenizer for SciRust.
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    unk_id: u32,
    pad_id: u32,
}

impl BpeTokenizer {
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let unk_id = *vocab.get("<UNK>").unwrap_or(&0);
        let pad_id = *vocab.get("<PAD>").unwrap_or(&1);
        Self {
            vocab,
            merges,
            unk_id,
            pad_id,
        }
    }

    /// Train the BPE tokenizer on a corpus.
    pub fn train(texts: &[&str], vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("<PAD>".to_string(), 1);

        // Initial characters as base vocabulary
        let mut current_vocab_size = 2;
        let mut char_counts = HashMap::new();
        for text in texts
        {
            for c in text.chars()
            {
                let s = c.to_string();
                if !vocab.contains_key(&s) && current_vocab_size < vocab_size
                {
                    vocab.insert(s.clone(), current_vocab_size as u32);
                    current_vocab_size += 1;
                }
                *char_counts.entry(s).or_insert(0) += 1;
            }
        }

        // Simplistic BPE training loop
        let mut merges = Vec::new();
        let mut words: Vec<Vec<String>> = texts
            .iter()
            .map(|t| t.chars().map(|c| c.to_string()).collect())
            .collect();

        while vocab.len() < vocab_size
        {
            let mut pair_counts = HashMap::new();
            for word in &words
            {
                for i in 0..word.len().saturating_sub(1)
                {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty()
            {
                break;
            }

            // Pick the most frequent pair. Ties are broken deterministically by
            // the pair itself (lexicographically smallest) — `max_by_key` alone
            // would return whichever tied pair the HashMap happened to iterate
            // last, making training non-reproducible.
            let best_pair = pair_counts
                .into_iter()
                .max_by_key(|(pair, count)| (*count, Reverse(pair.clone())))
                .map(|(pair, _)| pair);

            if let Some((p1, p2)) = best_pair
            {
                let new_token = format!("{}{}", p1, p2);
                merges.push((p1.clone(), p2.clone()));
                vocab.insert(new_token.clone(), vocab.len() as u32);

                // Apply merge
                for word in &mut words
                {
                    let mut new_word = Vec::new();
                    let mut i = 0;
                    while i < word.len()
                    {
                        if i + 1 < word.len() && word[i] == p1 && word[i + 1] == p2
                        {
                            new_word.push(new_token.clone());
                            i += 2;
                        }
                        else
                        {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }
                    *word = new_word;
                }
            }
            else
            {
                break;
            }
        }

        Self::new(vocab, merges)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let rev_vocab: HashMap<u32, String> =
            self.vocab.iter().map(|(s, &id)| (id, s.clone())).collect();

        ids.iter()
            .map(|id| rev_vocab.get(id).cloned().unwrap_or_else(|| "".to_string()))
            .collect()
    }
}

impl Tokenizer for BpeTokenizer {
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        for (p1, p2) in &self.merges
        {
            let new_token = format!("{}{}", p1, p2);
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len()
            {
                if i + 1 < tokens.len() && tokens[i] == *p1 && tokens[i + 1] == *p2
                {
                    new_tokens.push(new_token.clone());
                    i += 2;
                }
                else
                {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens
            .iter()
            .map(|s| *self.vocab.get(s).unwrap_or(&self.unk_id))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn pad_id(&self) -> u32 {
        self.pad_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlp::tokenization::Tokenizer;

    #[test]
    fn train_encode_decode_roundtrip() {
        let corpus = ["low lower lowest", "newer newest wider"];
        let tok = BpeTokenizer::train(&corpus, 50);
        // Text drawn from the corpus round-trips exactly.
        for s in ["low", "lower", "newest", "wider"]
        {
            assert_eq!(tok.decode(&tok.tokenize(s)), s, "round-trip failed for {s}");
        }
    }

    #[test]
    fn training_is_deterministic() {
        // The whole point of the tie-break fix: identical corpus + vocab_size
        // must yield an identical tokenizer every time (reproducibility).
        let corpus = ["banana bandana", "an ananas in a cabana", "panama canal"];
        let reference = BpeTokenizer::train(&corpus, 40);
        let probe = "a banana in panama";
        let expected = reference.tokenize(probe);
        for _ in 0..5
        {
            let again = BpeTokenizer::train(&corpus, 40);
            assert_eq!(again.tokenize(probe), expected, "BPE training diverged");
        }
    }

    #[test]
    fn frequent_pair_is_merged() {
        // "ab" is by far the most frequent adjacent pair → it must become a
        // single merged token (encoding "abab" uses fewer ids than 4 chars).
        let tok = BpeTokenizer::train(&["abababab abab ab"], 30);
        assert!(
            tok.tokenize("abab").len() < 4,
            "expected `ab` to merge, got {:?}",
            tok.tokenize("abab")
        );
    }

    #[test]
    fn unknown_char_maps_to_unk() {
        let tok = BpeTokenizer::train(&["abc"], 10);
        // 'z' never appears in the corpus → unknown id (0).
        assert_eq!(tok.tokenize("z"), vec![0]);
    }

    #[test]
    fn vocab_size_is_respected() {
        let tok = BpeTokenizer::train(&["the quick brown fox jumps"], 25);
        assert!(tok.vocab_size() <= 25);
        assert!(tok.vocab_size() >= 2); // at least the special tokens
    }
}
