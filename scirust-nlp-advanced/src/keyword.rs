//! Keyword extraction: TF-IDF, TextRank (graph-based), and RAKE.
//!
//! Each extractor produces a ranked list of keyword candidates with
//! associated scores.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Keyword result
// ---------------------------------------------------------------------------

/// A single extracted keyword with its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyword {
    pub text: String,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// TF-IDF keyword extraction
// ---------------------------------------------------------------------------

/// Extract keywords from a document using TF-IDF scoring against a corpus.
///
/// `document` and `corpus` are already tokenized (lowercased, stop-words
/// removed).  Returns the top `max_keywords` terms ranked by TF-IDF.
pub fn keyword_tfidf(
    document: &[String],
    corpus: &[Vec<String>],
    max_keywords: usize,
) -> Vec<Keyword> {
    if document.is_empty()
    {
        return Vec::new();
    }

    let n_docs = corpus.len() as f64;
    let mut scored: Vec<(String, f64)> = Vec::new();

    // Collect unique terms in this document
    let mut seen = HashSet::new();
    for tok in document
    {
        if !seen.insert(tok)
        {
            continue;
        }

        // Term frequency
        let tf = document.iter().filter(|t| *t == tok).count() as f64 / document.len() as f64;

        // Inverse document frequency
        let df = corpus
            .iter()
            .filter(|doc| doc.iter().any(|t| t == tok))
            .count() as f64;
        let idf = if df > 0.0
        {
            ((n_docs + 1.0) / (df + 1.0)).ln() + 1.0
        }
        else
        {
            0.0
        };

        let score = tf * idf;
        if score > 0.0
        {
            scored.push((tok.clone(), score));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(max_keywords)
        .map(|(text, score)| Keyword { text, score })
        .collect()
}

// ---------------------------------------------------------------------------
// TextRank
// ---------------------------------------------------------------------------

/// A simple graph-based keyword extractor inspired by TextRank.
///
/// Builds a co-occurrence graph within a sliding window and runs a
/// PageRank-like iterative scoring.
pub struct TextRank {
    pub window_size: usize,
    pub damping: f64,
    pub iterations: usize,
    pub threshold: f64,
}

impl Default for TextRank {
    fn default() -> Self {
        Self {
            window_size: 4,
            damping: 0.85,
            iterations: 30,
            threshold: 1e-6,
        }
    }
}

impl TextRank {
    /// Extract keywords from tokenized text.
    #[allow(clippy::needless_range_loop)]
    pub fn extract(&self, tokens: &[String]) -> Vec<Keyword> {
        if tokens.is_empty()
        {
            return Vec::new();
        }

        // Build adjacency list
        let mut adj: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for i in 0..tokens.len()
        {
            let a = &tokens[i];
            let end = (i + self.window_size).min(tokens.len());
            for j in (i + 1)..end
            {
                let b = &tokens[j];
                if a != b
                {
                    *adj.entry(a.clone())
                        .or_default()
                        .entry(b.clone())
                        .or_insert(0.0) += 1.0;
                    *adj.entry(b.clone())
                        .or_default()
                        .entry(a.clone())
                        .or_insert(0.0) += 1.0;
                }
            }
        }

        if adj.is_empty()
        {
            return tokens
                .iter()
                .map(|t| Keyword {
                    text: t.clone(),
                    score: 1.0,
                })
                .collect();
        }

        // PageRank
        let nodes: Vec<String> = adj.keys().cloned().collect();
        let n = nodes.len();
        let node_idx: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();
        let mut scores = vec![1.0f64 / n as f64; n];

        for _ in 0..self.iterations
        {
            let mut new_scores = vec![0.0f64; n];
            let mut diff = 0.0f64;

            for (i, node) in nodes.iter().enumerate()
            {
                let neighbors = adj.get(node).unwrap();
                let out_degree: f64 = neighbors.values().sum();
                if out_degree == 0.0
                {
                    new_scores[i] = (1.0 - self.damping) / n as f64;
                    continue;
                }
                let mut sum = 0.0;
                for (neighbor, weight) in neighbors
                {
                    if let Some(&j) = node_idx.get(neighbor.as_str())
                    {
                        sum += (weight / out_degree) * scores[j];
                    }
                }
                new_scores[i] = (1.0 - self.damping) / n as f64 + self.damping * sum;
                diff += (new_scores[i] - scores[i]).abs();
            }
            scores = new_scores;
            if diff < self.threshold
            {
                break;
            }
        }

        // Collect and sort
        let mut result: Vec<Keyword> = nodes
            .into_iter()
            .zip(scores)
            .map(|(text, score)| Keyword { text, score })
            .collect();
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }
}

// ---------------------------------------------------------------------------
// RAKE (Rapid Automatic Keyword Extraction)
// ---------------------------------------------------------------------------

/// RAKE keyword extractor.
///
/// 1. Splits text into sentences, then into candidate phrases using
///    stop-words as delimiters.
/// 2. Scores individual words by degree / frequency.
/// 3. Scores phrases as sum of word scores.
pub struct Rake {
    pub stop_words: HashSet<String>,
}

impl Rake {
    pub fn new(stop_words: HashSet<String>) -> Self {
        Self { stop_words }
    }

    /// English stop words (common list).
    pub fn english_stop_words() -> HashSet<String> {
        let words = [
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "is", "it", "this", "that", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "shall", "can", "not", "no", "so", "if", "then", "than", "too", "very",
            "just", "about", "above", "after", "again", "all", "also", "am", "any", "as",
            "because", "before", "between", "both", "each", "few", "more", "most", "other", "some",
            "such", "into", "only", "own", "same", "through", "during", "out", "up", "down",
        ];
        words.iter().map(|w| w.to_string()).collect()
    }

    /// Split text into candidate phrases using stop-words as delimiters.
    fn candidate_phrases(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        let mut phrases = Vec::new();
        let mut current = String::new();

        for word in lower.split_whitespace()
        {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
                .collect();
            if cleaned.is_empty()
            {
                continue;
            }
            if self.stop_words.contains(cleaned.as_str())
            {
                if !current.is_empty()
                {
                    phrases.push(current.trim().to_string());
                    current.clear();
                }
            }
            else
            {
                if !current.is_empty()
                {
                    current.push(' ');
                }
                current.push_str(&cleaned);
            }
        }
        if !current.trim().is_empty()
        {
            phrases.push(current.trim().to_string());
        }
        phrases
    }

    /// Extract keywords from raw text.
    pub fn extract(&self, text: &str, max_keywords: usize) -> Vec<Keyword> {
        let phrases = self.candidate_phrases(text);
        if phrases.is_empty()
        {
            return Vec::new();
        }

        // Build word frequency and degree
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut word_degree: HashMap<String, usize> = HashMap::new();

        for phrase in &phrases
        {
            let words: Vec<&str> = phrase.split_whitespace().collect();
            let deg = words.len().saturating_sub(1);
            for w in &words
            {
                *word_freq.entry(w.to_string()).or_insert(0) += 1;
                *word_degree.entry(w.to_string()).or_insert(0) += deg;
            }
        }

        // Word score = (degree + freq) / freq
        let word_score: HashMap<String, f64> = word_freq
            .iter()
            .map(|(w, &freq)| {
                let degree = word_degree.get(w).copied().unwrap_or(0) as f64;
                let score = (degree + freq as f64) / freq as f64;
                (w.clone(), score)
            })
            .collect();

        // Phrase score = sum of word scores / number of words
        let mut phrase_scores: Vec<(String, f64)> = phrases
            .iter()
            .map(|phrase| {
                let words: Vec<&str> = phrase.split_whitespace().collect();
                let sum: f64 = words.iter().filter_map(|w| word_score.get(*w)).sum();
                let len = words.len().max(1) as f64;
                (phrase.clone(), sum / len)
            })
            .collect();

        phrase_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        phrase_scores
            .into_iter()
            .take(max_keywords)
            .map(|(text, score)| Keyword { text, score })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_tfidf() {
        let doc = vec!["rust".into(), "is".into(), "great".into(), "rust".into()];
        let corpus = vec![
            doc.clone(),
            vec!["python".into(), "is".into(), "nice".into()],
            vec!["java".into(), "is".into(), "okay".into()],
        ];
        let keywords = keyword_tfidf(&doc, &corpus, 2);
        assert_eq!(keywords.len(), 2);
        // "rust" should be top (appears in only 1 doc)
        assert_eq!(keywords[0].text, "rust");
    }

    #[test]
    fn test_keyword_tfidf_empty() {
        let keywords = keyword_tfidf(&[], &[], 5);
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_textrank_basic() {
        let tokens = vec![
            "rust".into(),
            "is".into(),
            "a".into(),
            "systems".into(),
            "programming".into(),
            "language".into(),
            "rust".into(),
            "is".into(),
            "fast".into(),
        ];
        let tr = TextRank::default();
        let keywords = tr.extract(&tokens);
        assert!(!keywords.is_empty());
        // "rust" should rank high
        assert!(keywords.iter().any(|k| k.text == "rust"));
    }

    #[test]
    fn test_textrank_empty() {
        let tr = TextRank::default();
        let keywords = tr.extract(&[]);
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_rake_basic() {
        let text = "Rust is a systems programming language. Rust is fast and safe.";
        let stop = Rake::english_stop_words();
        let rake = Rake::new(stop);
        let keywords = rake.extract(text, 3);
        assert!(!keywords.is_empty());
        // Should find multi-word phrases or important words
        assert!(
            keywords
                .iter()
                .any(|k| k.text.contains("rust") || k.text.contains("programming"))
        );
    }

    #[test]
    fn test_rake_empty() {
        let rake = Rake::new(Rake::english_stop_words());
        let keywords = rake.extract("", 5);
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_rake_stop_words() {
        let rake = Rake::new(Rake::english_stop_words());
        let phrases = rake.candidate_phrases("the quick brown fox jumps over the lazy dog");
        assert!(!phrases.is_empty());
        // "the" and "over" are stop words, so phrases should split around them
        assert!(phrases.iter().any(|p| p.contains("quick")));
        assert!(phrases.iter().any(|p| p.contains("brown")));
    }

    #[test]
    fn test_keyword_scores_descending() {
        let doc = vec!["a".into(), "b".into(), "c".into()];
        let corpus = vec![doc.clone(), vec!["a".into()]];
        let keywords = keyword_tfidf(&doc, &corpus, 3);
        for i in 1..keywords.len()
        {
            assert!(keywords[i - 1].score >= keywords[i].score);
        }
    }
}
