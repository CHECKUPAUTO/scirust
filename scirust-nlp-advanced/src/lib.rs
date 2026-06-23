//! Advanced NLP pattern detection algorithms.
//!
//! Provides Named Entity Recognition (NER), Topic Modeling (LDA),
//! Relation Extraction, Text Classification, Keyword Extraction, and
//! Document Similarity measures — all implemented in pure Rust.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `ner` | Named entity recognition (rule-based + statistical, BIO tagging) |
//! | `topic` | Latent Dirichlet Allocation with Gibbs sampling |
//! | `relation` | Pattern-based relation extraction with dependency features |
//! | `classify` | Naive Bayes classifier, TF-IDF, bag-of-words, similarity |
//! | `keyword` | TF-IDF, TextRank, and RAKE keyword extraction |
//! | `similarity` | Cosine, Jaccard, and MinHash document similarity |
//! | `tokenize` | Low-level whitespace + punctuation tokenizer |
//! | `bloom` | Bloom filter — probabilistic membership / dedup pre-filter |
//! | `lsh` | MinHash-LSH band-and-bucket near-duplicate index |
//! | `trie` | Byte-radix trie — shared-prefix physical compaction |
//! | `huffman` | Entropy-optimal prefix-free coding (reversible) |

pub mod bloom;
pub mod classify;
pub mod huffman;
pub mod keyword;
pub mod lsh;
pub mod ner;
pub mod relation;
pub mod similarity;
pub mod tokenize;
pub mod topic;

use serde::{Deserialize, Serialize};

/// A single token with its character offset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Compute term frequency: count of term in tokens / total token count.
pub fn term_frequency(tokens: &[String], term: &str) -> f64 {
    if tokens.is_empty()
    {
        return 0.0;
    }
    let count = tokens.iter().filter(|t| t.as_str() == term).count();
    count as f64 / tokens.len() as f64
}

/// Compute inverse document frequency: log(N / df) where df = number of
/// documents containing `term`.
pub fn inverse_document_frequency(documents: &[Vec<String>], term: &str) -> f64 {
    let n = documents.len() as f64;
    if n == 0.0
    {
        return 0.0;
    }
    let df = documents
        .iter()
        .filter(|doc| doc.iter().any(|t| t.as_str() == term))
        .count() as f64;
    if df == 0.0
    {
        return 0.0;
    }
    (n / df).ln()
}

/// Compute TF-IDF score for a term in a document relative to a corpus.
pub fn tf_idf(document: &[String], corpus: &[Vec<String>], term: &str) -> f64 {
    term_frequency(document, term) * inverse_document_frequency(corpus, term)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_frequency() {
        let tokens = vec!["hello".into(), "world".into(), "hello".into()];
        assert!((term_frequency(&tokens, "hello") - 2.0 / 3.0).abs() < 1e-10);
        assert!((term_frequency(&tokens, "world") - 1.0 / 3.0).abs() < 1e-10);
        assert!(term_frequency(&tokens, "missing").abs() < 1e-10);
    }

    #[test]
    fn test_term_frequency_empty() {
        let tokens: Vec<String> = vec![];
        assert!(term_frequency(&tokens, "x").abs() < 1e-10);
    }

    #[test]
    fn test_idf() {
        let corpus = vec![
            vec!["a".into(), "b".into()],
            vec!["b".into(), "c".into()],
            vec!["c".into(), "d".into()],
        ];
        // "a" appears in 1/3 docs
        let idf_a = inverse_document_frequency(&corpus, "a");
        assert!((idf_a - (3.0_f64).ln()).abs() < 1e-10);
        // "b" appears in 2/3 docs
        let idf_b = inverse_document_frequency(&corpus, "b");
        assert!((idf_b - (3.0_f64 / 2.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_tf_idf() {
        let doc = vec!["rust".into(), "is".into(), "great".into(), "rust".into()];
        let corpus = vec![
            doc.clone(),
            vec!["rust".into(), "lang".into()],
            vec!["python".into(), "is".into(), "nice".into()],
        ];
        let score = tf_idf(&doc, &corpus, "rust");
        // tf = 2/4, idf = ln(3/2)
        let expected = (2.0 / 4.0) * (3.0_f64 / 2.0).ln();
        assert!((score - expected).abs() < 1e-10);
    }
}
