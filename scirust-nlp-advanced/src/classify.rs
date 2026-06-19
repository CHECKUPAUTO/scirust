//! Text classification: Naive Bayes, TF-IDF features, bag-of-words,
//! and text similarity measures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Bag of Words
// ---------------------------------------------------------------------------

/// A simple bag-of-words vectorizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BagOfWords {
    /// Vocabulary index → word.
    pub vocab: Vec<String>,
    /// Word → vocabulary index.
    pub vocab_index: HashMap<String, usize>,
}

impl BagOfWords {
    /// Build vocabulary from a slice of tokenized documents.
    pub fn build(documents: &[Vec<String>], min_freq: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for doc in documents
        {
            let mut seen = HashMap::new();
            for word in doc
            {
                *seen.entry(word.clone()).or_insert(0) += 1;
            }
            for (word, count) in seen
            {
                *freq.entry(word).or_insert(0) += count;
            }
        }
        let mut vocab: Vec<String> = freq
            .into_iter()
            .filter(|(_, c)| *c >= min_freq)
            .map(|(w, _)| w)
            .collect();
        vocab.sort();
        let vocab_index: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        Self { vocab, vocab_index }
    }

    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Convert a tokenized document to a count vector.
    pub fn vectorize(&self, tokens: &[String]) -> Vec<f64> {
        let mut vec = vec![0.0; self.vocab.len()];
        for tok in tokens
        {
            if let Some(&idx) = self.vocab_index.get(tok)
            {
                vec[idx] += 1.0;
            }
        }
        vec
    }

    /// Convert a tokenized document to a binary (presence) vector.
    pub fn vectorize_binary(&self, tokens: &[String]) -> Vec<f64> {
        let mut vec = vec![0.0; self.vocab.len()];
        for tok in tokens
        {
            if let Some(&idx) = self.vocab_index.get(tok)
            {
                vec[idx] = 1.0;
            }
        }
        vec
    }
}

// ---------------------------------------------------------------------------
// TF-IDF Vectorizer
// ---------------------------------------------------------------------------

/// TF-IDF vectorizer that computes IDF from a corpus and transforms
/// documents into TF-IDF vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfidfVectorizer {
    pub bow: BagOfWords,
    /// IDF values for each vocabulary term.
    pub idf: Vec<f64>,
}

impl TfidfVectorizer {
    /// Build from a corpus of tokenized documents.
    pub fn build(documents: &[Vec<String>], min_freq: usize) -> Self {
        let bow = BagOfWords::build(documents, min_freq);
        let n = documents.len() as f64;
        let mut idf = Vec::with_capacity(bow.vocab.len());

        for word in &bow.vocab
        {
            let df = documents
                .iter()
                .filter(|doc| doc.iter().any(|t| t == word))
                .count() as f64;
            // Standard IDF with smoothing
            let val = if df > 0.0
            {
                ((n + 1.0) / (df + 1.0)).ln() + 1.0
            }
            else
            {
                0.0
            };
            idf.push(val);
        }

        Self { bow, idf }
    }

    /// Transform a document to a TF-IDF vector.
    pub fn transform(&self, tokens: &[String]) -> Vec<f64> {
        let counts = self.bow.vectorize(tokens);
        let tf_norm = tokens.len() as f64;
        if tf_norm == 0.0
        {
            return vec![0.0; self.bow.len()];
        }
        counts
            .iter()
            .zip(self.idf.iter())
            .map(|(c, idf)| (c / tf_norm) * idf)
            .collect()
    }

    /// Build and transform an entire corpus.
    pub fn fit_transform(documents: &[Vec<String>], min_freq: usize) -> (Self, Vec<Vec<f64>>) {
        let vectorizer = Self::build(documents, min_freq);
        let vectors: Vec<Vec<f64>> = documents
            .iter()
            .map(|doc| vectorizer.transform(doc))
            .collect();
        (vectorizer, vectors)
    }
}

// ---------------------------------------------------------------------------
// Text similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len()
    {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0
    {
        0.0
    }
    else
    {
        dot / (norm_a * norm_b)
    }
}

/// Cosine similarity between two tokenized documents using a shared vocabulary.
pub fn cosine_similarity_docs(a: &[String], b: &[String]) -> f64 {
    let mut vocab: Vec<String> = a.iter().chain(b.iter()).cloned().collect();
    vocab.sort();
    vocab.dedup();
    let idx: HashMap<&str, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i))
        .collect();
    let mut va = vec![0.0f64; vocab.len()];
    let mut vb = vec![0.0f64; vocab.len()];
    for tok in a
    {
        if let Some(&i) = idx.get(tok.as_str())
        {
            va[i] += 1.0;
        }
    }
    for tok in b
    {
        if let Some(&i) = idx.get(tok.as_str())
        {
            vb[i] += 1.0;
        }
    }
    cosine_similarity(&va, &vb)
}

/// Jaccard similarity between two sets.
pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0
    {
        0.0
    }
    else
    {
        intersection / union
    }
}

/// Jaccard similarity between two integer sets.
pub fn jaccard_similarity_sets(a: &[usize], b: &[usize]) -> f64 {
    let set_a: std::collections::HashSet<usize> = a.iter().copied().collect();
    let set_b: std::collections::HashSet<usize> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0
    {
        0.0
    }
    else
    {
        intersection / union
    }
}

// ---------------------------------------------------------------------------
// Naive Bayes classifier
// ---------------------------------------------------------------------------

/// A multinomial Naive Bayes text classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveBayes {
    /// log P(class) for each class label.
    pub class_log_prior: HashMap<String, f64>,
    /// log P(word | class) for each class and word index.
    pub feature_log_prob: HashMap<String, Vec<f64>>,
    /// Vocabulary used during training.
    pub vocab: Vec<String>,
    pub vocab_index: HashMap<String, usize>,
    /// Known class labels.
    pub classes: Vec<String>,
}

impl NaiveBayes {
    /// Train from a corpus of (class_label, tokens) pairs.
    pub fn train(training_data: &[(&str, Vec<String>)]) -> Self {
        if training_data.is_empty()
        {
            return Self {
                class_log_prior: HashMap::new(),
                feature_log_prob: HashMap::new(),
                vocab: Vec::new(),
                vocab_index: HashMap::new(),
                classes: Vec::new(),
            };
        }

        // Build vocabulary
        let mut vocab_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (_, tokens) in training_data
        {
            for tok in tokens
            {
                vocab_set.insert(tok.clone());
            }
        }
        let mut vocab: Vec<String> = vocab_set.into_iter().collect();
        vocab.sort();
        let vocab_index: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        let vocab_size = vocab.len();

        // Group by class
        let mut class_docs: HashMap<&str, Vec<&Vec<String>>> = HashMap::new();
        for (cls, tokens) in training_data
        {
            class_docs.entry(cls).or_default().push(tokens);
        }

        let total_docs = training_data.len() as f64;
        let mut class_log_prior = HashMap::new();
        let mut feature_log_prob = HashMap::new();
        let mut classes = Vec::new();

        for (cls, docs) in &class_docs
        {
            let cls_name = cls.to_string();
            classes.push(cls_name.clone());

            let doc_count = docs.len() as f64;
            class_log_prior.insert(cls_name.clone(), (doc_count / total_docs).ln());

            // Word counts for this class
            let mut word_counts = vec![0.0f64; vocab_size];
            let mut total_words = 0.0f64;
            for doc in docs
            {
                for tok in *doc
                {
                    if let Some(&idx) = vocab_index.get(tok)
                    {
                        word_counts[idx] += 1.0;
                        total_words += 1.0;
                    }
                }
            }

            // Log probabilities with Laplace smoothing
            let log_probs: Vec<f64> = word_counts
                .iter()
                .map(|&c| ((c + 1.0) / (total_words + vocab_size as f64)).ln())
                .collect();
            feature_log_prob.insert(cls_name, log_probs);
        }

        Self {
            class_log_prior,
            feature_log_prob,
            vocab,
            vocab_index,
            classes,
        }
    }

    /// Predict the class label for a tokenized document.
    pub fn predict(&self, tokens: &[String]) -> Option<(String, f64)> {
        if self.classes.is_empty()
        {
            return None;
        }

        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for cls in &self.classes
        {
            let prior = self.class_log_prior[cls];
            let probs = &self.feature_log_prob[cls];
            let score: f64 = tokens
                .iter()
                .filter_map(|tok| self.vocab_index.get(tok).map(|&i| probs[i]))
                .sum::<f64>()
                + prior;

            if score > best_score
            {
                best_score = score;
                best_class = cls.clone();
            }
        }

        // Convert to probability via softmax-like normalization
        let scores: Vec<f64> = self
            .classes
            .iter()
            .map(|cls| {
                let prior = self.class_log_prior[cls];
                let probs = &self.feature_log_prob[cls];
                tokens
                    .iter()
                    .filter_map(|tok| self.vocab_index.get(tok).map(|&i| probs[i]))
                    .sum::<f64>()
                    + prior
            })
            .collect();
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let prob = exp_scores
            .iter()
            .zip(self.classes.iter())
            .find(|(_, c)| **c == best_class)
            .map(|(e, _)| e / sum_exp)
            .unwrap_or(0.0);

        Some((best_class, prob))
    }

    /// Predict with confidence, returning all class probabilities.
    pub fn predict_proba(&self, tokens: &[String]) -> Vec<(String, f64)> {
        if self.classes.is_empty()
        {
            return Vec::new();
        }

        let scores: Vec<f64> = self
            .classes
            .iter()
            .map(|cls| {
                let prior = self.class_log_prior[cls];
                let probs = &self.feature_log_prob[cls];
                tokens
                    .iter()
                    .filter_map(|tok| self.vocab_index.get(tok).map(|&i| probs[i]))
                    .sum::<f64>()
                    + prior
            })
            .collect();

        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        self.classes
            .iter()
            .zip(exp_scores.iter())
            .map(|(c, e)| (c.clone(), e / sum_exp))
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
    fn test_bag_of_words() {
        let docs = vec![
            vec!["hello".into(), "world".into()],
            vec!["hello".into(), "rust".into()],
        ];
        let bow = BagOfWords::build(&docs, 1);
        assert_eq!(bow.len(), 3);
        let vec = bow.vectorize(&["hello".into(), "rust".into()]);
        assert_eq!(vec.len(), 3);
        // hello appears once, rust appears once
        assert!(vec.contains(&1.0));
    }

    #[test]
    fn test_bag_of_words_min_freq() {
        let docs = vec![
            vec!["a".into(), "b".into()],
            vec!["a".into()],
            vec!["a".into(), "c".into()],
        ];
        let bow = BagOfWords::build(&docs, 2);
        // "a" appears 3 times, "b" once, "c" once → only "a" survives
        assert_eq!(bow.len(), 1);
        assert_eq!(bow.vocab[0], "a");
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let docs = vec![
            vec!["cat".into(), "sat".into(), "mat".into()],
            vec!["cat".into(), "dog".into()],
        ];
        let (vec, _) = TfidfVectorizer::fit_transform(&docs, 1);
        // TF-IDF for "cat" in doc 0 should be lower than for "sat" or "mat"
        // because "cat" appears in both docs
        let v0 = vec.transform(&docs[0]);
        let v1 = vec.transform(&docs[1]);
        assert_eq!(v0.len(), 4); // cat, dog, mat, sat (sorted)
        assert_eq!(v1.len(), 4);
        // "cat" index
        let cat_idx = vec.bow.vocab_index["cat"];
        // "mat" index
        let mat_idx = vec.bow.vocab_index["mat"];
        // "mat" appears only in doc 0 → higher IDF than "cat"
        assert!(v0[mat_idx] > v0[cat_idx]);
        // Both should be positive
        assert!(v0[cat_idx] > 0.0);
        assert!(v1[cat_idx] > 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_docs() {
        let a = vec!["hello".into(), "world".into()];
        let b = vec!["hello".into(), "rust".into()];
        let sim = cosine_similarity_docs(&a, &b);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let a = vec!["a".into(), "b".into(), "c".into()];
        assert!((jaccard_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_similarity_disjoint() {
        let a = vec!["a".into(), "b".into()];
        let b = vec!["c".into(), "d".into()];
        assert!(jaccard_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_similarity_partial() {
        let a = vec!["a".into(), "b".into(), "c".into()];
        let b = vec!["b".into(), "c".into(), "d".into()];
        // intersection = {b, c} = 2, union = {a, b, c, d} = 4
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_naive_bayes_basic() {
        let training = vec![
            (
                "positive",
                vec!["good".into(), "great".into(), "love".into()],
            ),
            ("positive", vec!["happy".into(), "good".into()]),
            (
                "negative",
                vec!["bad".into(), "terrible".into(), "hate".into()],
            ),
            ("negative", vec!["sad".into(), "bad".into()]),
        ];
        let nb = NaiveBayes::train(&training);
        assert_eq!(nb.classes.len(), 2);

        let pred = nb.predict(&["good".into(), "great".into()]);
        assert!(pred.is_some());
        let (cls, prob) = pred.unwrap();
        assert_eq!(cls, "positive");
        assert!(prob > 0.5);
    }

    #[test]
    fn test_naive_bayes_predict_proba() {
        let training = vec![
            ("spam", vec!["buy".into(), "now".into(), "discount".into()]),
            ("ham", vec!["meeting".into(), "tomorrow".into()]),
        ];
        let nb = NaiveBayes::train(&training);
        let proba = nb.predict_proba(&["buy".into(), "discount".into()]);
        assert_eq!(proba.len(), 2);
        let spam_prob: f64 = proba
            .iter()
            .find(|(c, _)| c == "spam")
            .map(|(_, p)| *p)
            .unwrap();
        let ham_prob: f64 = proba
            .iter()
            .find(|(c, _)| c == "ham")
            .map(|(_, p)| *p)
            .unwrap();
        assert!((spam_prob + ham_prob - 1.0).abs() < 1e-6);
        assert!(spam_prob > ham_prob);
    }

    #[test]
    fn test_naive_bayes_empty() {
        let nb = NaiveBayes::train(&[]);
        assert!(nb.predict(&["hello".into()]).is_none());
    }

    #[test]
    fn test_bag_of_words_empty() {
        let docs: Vec<Vec<String>> = vec![vec![], vec![]];
        let bow = BagOfWords::build(&docs, 1);
        assert!(bow.is_empty());
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert!(cosine_similarity(&[], &[]).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_similarity_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        assert!(jaccard_similarity(&a, &b).abs() < 1e-10);
    }
}
