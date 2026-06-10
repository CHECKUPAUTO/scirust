use std::collections::HashMap;

/// Trait de base pour les tokeniseurs dans SciRust.
pub trait Tokenizer: Send + Sync {
    /// Convertit un texte brut en une séquence d'identifiants (IDs).
    fn tokenize(&self, text: &str) -> Vec<u32>;
    /// Retourne la taille du vocabulaire.
    fn vocab_size(&self) -> usize;
    /// Retourne l'ID utilisé pour le padding.
    fn pad_id(&self) -> u32;
}

/// Tokeniseur simple basé sur les espaces et la suppression de la ponctuation.
/// Amélioré pour mieux gérer les contractions françaises de base.
pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    pad_id: u32,
}

impl SimpleTokenizer {
    /// Crée un nouveau `SimpleTokenizer` à partir d'un vocabulaire existant.
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        let unk_id = *vocab.get("<UNK>").unwrap_or(&0);
        let pad_id = *vocab.get("<PAD>").unwrap_or(&1);
        Self {
            vocab,
            unk_id,
            pad_id,
        }
    }

    fn clean_text(text: &str) -> String {
        // Remplacement simple des apostrophes par des espaces pour gérer "c'est" -> "c est"
        text.to_lowercase().replace('\'', " ")
    }

    /// Construit un vocabulaire à partir d'un corpus de textes.
    pub fn build(texts: &[&str], min_freq: usize) -> Self {
        let mut counts = HashMap::new();
        for text in texts
        {
            let cleaned = Self::clean_text(text);
            for word in cleaned.split_whitespace()
            {
                let clean_word: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                if !clean_word.is_empty()
                {
                    *counts.entry(clean_word).or_insert(0) += 1;
                }
            }
        }

        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("<PAD>".to_string(), 1);

        let mut entries: Vec<_> = counts
            .into_iter()
            .filter(|&(_, count)| count >= min_freq)
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut id = 2;
        for (word, _) in entries
        {
            vocab.insert(word, id);
            id += 1;
        }

        Self::new(vocab)
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let cleaned = Self::clean_text(text);
        cleaned
            .split_whitespace()
            .map(|word| {
                let clean_word: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                *self.vocab.get(&clean_word).unwrap_or(&self.unk_id)
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn pad_id(&self) -> u32 {
        self.pad_id
    }
}
