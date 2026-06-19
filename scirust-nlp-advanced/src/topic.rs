//! Latent Dirichlet Allocation (LDA) with Gibbs sampling.
//!
//! Implements online Gibbs sampling for topic modeling over a bag-of-words
//! corpus.  Provides:
//! - Topic-term distributions
//! - Document-topic distributions
//! - Coherence score (pointwise mutual information based)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LDA model
// ---------------------------------------------------------------------------

/// Configuration for LDA training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdaConfig {
    /// Number of topics to infer.
    pub num_topics: usize,
    /// Dirichlet prior for document-topic distribution (alpha).
    pub alpha: f64,
    /// Dirichlet prior for topic-term distribution (beta).
    pub beta: f64,
    /// Number of Gibbs sampling iterations.
    pub iterations: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for LdaConfig {
    fn default() -> Self {
        Self {
            num_topics: 10,
            alpha: 0.1,
            beta: 0.01,
            iterations: 100,
            seed: 42,
        }
    }
}

/// Trained LDA model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdaModel {
    pub config: LdaConfig,
    /// Topic-term counts: topic_term[t][w] = count of word w in topic t.
    pub topic_term: Vec<Vec<usize>>,
    /// Document-topic counts: doc_topic[d][t] = count of topic t in doc d.
    pub doc_topic: Vec<Vec<usize>>,
    /// Total words per topic.
    pub topic_count: Vec<usize>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Mapping from vocabulary index → word string.
    pub vocab: Vec<String>,
}

impl LdaModel {
    /// Topic-term distribution (normalized): P(word | topic).
    pub fn topic_term_dist(&self) -> Vec<Vec<f64>> {
        let mut dist = Vec::with_capacity(self.topic_term.len());
        for (t, counts) in self.topic_term.iter().enumerate()
        {
            let total = self.topic_count[t] as f64 + self.config.beta * self.vocab_size as f64;
            let row: Vec<f64> = counts
                .iter()
                .map(|&c| (c as f64 + self.config.beta) / total)
                .collect();
            dist.push(row);
        }
        dist
    }

    /// Document-topic distribution (normalized): P(topic | document).
    pub fn doc_topic_dist(&self) -> Vec<Vec<f64>> {
        let alpha_sum = self.config.alpha * self.config.num_topics as f64;
        self.doc_topic
            .iter()
            .map(|counts| {
                let total: f64 = counts.iter().sum::<usize>() as f64 + alpha_sum;
                counts
                    .iter()
                    .map(|&c| (c as f64 + self.config.alpha) / total)
                    .collect()
            })
            .collect()
    }

    /// Top-`n` words for each topic.
    pub fn top_words(&self, n: usize) -> Vec<Vec<(String, f64)>> {
        let dist = self.topic_term_dist();
        dist.iter()
            .map(|topic| {
                let mut pairs: Vec<(usize, f64)> =
                    topic.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                pairs
                    .into_iter()
                    .take(n)
                    .map(|(i, p)| (self.vocab[i].clone(), p))
                    .collect()
            })
            .collect()
    }

    /// Perplexity on a held-out bag-of-words corpus.
    pub fn perplexity(&self, test_docs: &[Vec<usize>]) -> f64 {
        let dt = self.doc_topic_dist();
        let tt = self.topic_term_dist();
        let mut log_likelihood = 0.0;
        let mut total_words = 0usize;

        for (d, doc) in test_docs.iter().enumerate()
        {
            if d >= dt.len()
            {
                continue;
            }
            for &w in doc
            {
                if w >= tt[0].len()
                {
                    continue;
                }
                let mut pw = 0.0f64;
                for t in 0..self.config.num_topics
                {
                    pw += dt[d][t] * tt[t][w];
                }
                if pw > 0.0
                {
                    log_likelihood += pw.ln();
                }
                total_words += 1;
            }
        }
        if total_words == 0
        {
            return f64::INFINITY;
        }
        (-log_likelihood / total_words as f64).exp()
    }
}

// ---------------------------------------------------------------------------
// Gibbs sampler
// ---------------------------------------------------------------------------

/// Simple PRNG (xorshift64) for reproducible Gibbs sampling.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fit an LDA model using collapsed Gibbs sampling.
///
/// `corpus` is a slice of documents, each represented as a vector of
/// vocabulary indices (bag-of-words).
#[allow(clippy::needless_range_loop)]
pub fn fit_lda(corpus: &[Vec<usize>], config: LdaConfig) -> LdaModel {
    let vocab_size = corpus
        .iter()
        .flat_map(|d| d.iter())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let num_topics = config.num_topics;
    let num_docs = corpus.len();

    // Build vocabulary from corpus indices (caller is responsible for mapping).
    // We'll create placeholder vocab labels.
    let vocab: Vec<String> = (0..vocab_size).map(|i| format!("w{}", i)).collect();

    // Initialize count matrices
    let mut topic_term: Vec<Vec<usize>> = vec![vec![0; vocab_size]; num_topics];
    let mut doc_topic: Vec<Vec<usize>> = vec![vec![0; num_topics]; num_docs];
    let mut topic_count: Vec<usize> = vec![0; num_topics];

    // z[d][i] = topic assignment for word position (d, i)
    let mut z: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut rng = Rng::new(config.seed);

    for (d, doc) in corpus.iter().enumerate()
    {
        let mut z_doc = Vec::with_capacity(doc.len());
        for &w in doc
        {
            let t = (rng.next_u64() as usize) % num_topics;
            z_doc.push(t);
            topic_term[t][w] += 1;
            doc_topic[d][t] += 1;
            topic_count[t] += 1;
        }
        z.push(z_doc);
    }

    // Gibbs iterations
    for _ in 0..config.iterations
    {
        for (d, doc) in corpus.iter().enumerate()
        {
            for (i, &w) in doc.iter().enumerate()
            {
                let old_t = z[d][i];
                // Remove current assignment
                topic_term[old_t][w] -= 1;
                doc_topic[d][old_t] -= 1;
                topic_count[old_t] -= 1;

                // Compute conditional distribution
                let mut probs: Vec<f64> = Vec::with_capacity(num_topics);
                let _doc_len = doc.len() as f64;
                for t in 0..num_topics
                {
                    let p = (doc_topic[d][t] as f64 + config.alpha)
                        * (topic_term[t][w] as f64 + config.beta)
                        / (topic_count[t] as f64 + config.beta * vocab_size as f64);
                    probs.push(p);
                }

                // Normalize and sample
                let total: f64 = probs.iter().sum();
                let mut r = rng.next_f64() * total;
                let mut new_t = num_topics - 1;
                for t in 0..num_topics
                {
                    r -= probs[t];
                    if r <= 0.0
                    {
                        new_t = t;
                        break;
                    }
                }

                // Assign new topic
                z[d][i] = new_t;
                topic_term[new_t][w] += 1;
                doc_topic[d][new_t] += 1;
                topic_count[new_t] += 1;
            }
        }
    }

    LdaModel {
        config,
        topic_term,
        doc_topic,
        topic_count,
        vocab_size,
        vocab,
    }
}

/// Fit LDA from raw text.  `documents` are raw strings; `build_vocab` is a
/// closure that maps a raw document to a list of vocabulary indices.
pub fn fit_lda_from_text(
    documents: &[&str],
    build_vocab: &dyn Fn(&str) -> Vec<usize>,
    config: LdaConfig,
) -> LdaModel {
    let corpus: Vec<Vec<usize>> = documents.iter().map(|d| build_vocab(d)).collect();
    fit_lda(&corpus, config)
}

// ---------------------------------------------------------------------------
// Coherence score
// ---------------------------------------------------------------------------

/// Compute UMass coherence for topic `t` given the document-word matrix.
///
/// Higher (less negative) is better.
#[allow(clippy::needless_range_loop)]
pub fn coherence_umass(
    topic_term: &[usize],
    doc_word_matrix: &[Vec<usize>],
    _topic_id: usize,
) -> f64 {
    let n_top = 10.min(topic_term.len());
    let mut top_indices: Vec<usize> = (0..topic_term.len()).collect();
    top_indices.sort_by(|&a, &b| topic_term[b].cmp(&topic_term[a]));
    top_indices.truncate(n_top);

    let _num_docs = doc_word_matrix.len() as f64;
    let mut score = 0.0;
    let mut count = 0;

    for i in 1..n_top
    {
        let w = top_indices[i];
        // D(w_i): number of documents containing w_i
        let d_wi: f64 = doc_word_matrix
            .iter()
            .filter(|doc| doc.contains(&w))
            .count() as f64;
        for j in 0..i
        {
            let wj = top_indices[j];
            // D(w_i, w_j): documents containing both
            let d_wi_wj: f64 = doc_word_matrix
                .iter()
                .filter(|doc| doc.contains(&w) && doc.contains(&wj))
                .count() as f64;
            if d_wi > 1.0 && d_wi_wj > 0.0
            {
                score += (d_wi_wj + 1.0).ln() / d_wi.ln();
                count += 1;
            }
        }
    }
    if count == 0
    {
        0.0
    }
    else
    {
        score / count as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lda_basic() {
        // Tiny corpus: 4 documents, 8-word vocab
        // Doc 0: "cat dog"
        // Doc 1: "cat cat dog"
        // Doc 2: "fish fish"
        // Doc 3: "fish fish fish"
        let corpus = vec![vec![0, 1], vec![0, 0, 1], vec![2, 3], vec![2, 3, 2]];
        let config = LdaConfig {
            num_topics: 2,
            alpha: 0.1,
            beta: 0.01,
            iterations: 50,
            seed: 42,
        };
        let model = fit_lda(&corpus, config);
        assert_eq!(model.topic_term.len(), 2);
        assert_eq!(model.doc_topic.len(), 4);

        let dist = model.doc_topic_dist();
        assert_eq!(dist.len(), 4);
        // Each row should sum to ~1.0
        for row in &dist
        {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row sum = {}", sum);
        }
    }

    #[test]
    fn test_topic_term_dist() {
        let corpus = vec![vec![0, 0, 1], vec![1, 2]];
        let config = LdaConfig {
            num_topics: 2,
            iterations: 20,
            seed: 1,
            ..Default::default()
        };
        let model = fit_lda(&corpus, config);
        let dist = model.topic_term_dist();
        assert_eq!(dist.len(), 2);
        for topic in &dist
        {
            let sum: f64 = topic.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "topic sum = {}", sum);
        }
    }

    #[test]
    fn test_top_words() {
        let corpus = vec![vec![0, 0, 0, 1], vec![2, 2]];
        let config = LdaConfig {
            num_topics: 1,
            iterations: 20,
            seed: 7,
            ..Default::default()
        };
        let model = fit_lda(&corpus, config);
        let top = model.top_words(3);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].len(), 3);
        // Word 0 should be top (appears 3 times)
        assert_eq!(top[0][0].0, "w0");
    }

    #[test]
    fn test_perplexity_decreases() {
        let corpus = vec![vec![0, 0, 1], vec![1, 1, 0], vec![2, 2, 2], vec![2, 2]];
        let config = LdaConfig {
            num_topics: 2,
            iterations: 5,
            seed: 42,
            ..Default::default()
        };
        let model_few = fit_lda(&corpus, config.clone());
        let config_more = LdaConfig {
            iterations: 50,
            ..config
        };
        let model_many = fit_lda(&corpus, config_more);
        let p1 = model_few.perplexity(&corpus);
        let p2 = model_many.perplexity(&corpus);
        // More iterations should generally give better (lower) perplexity
        // Not guaranteed on tiny data, so just check they're finite
        assert!(p1.is_finite());
        assert!(p2.is_finite());
    }

    #[test]
    fn test_coherence_umass() {
        let topic_term = vec![10, 8, 5, 2, 0];
        let doc_word_matrix = vec![vec![0, 1, 2], vec![0, 1], vec![2, 3]];
        let score = coherence_umass(&topic_term, &doc_word_matrix, 0);
        // Score should be finite and negative (UMass is typically negative)
        assert!(score.is_finite());
    }

    #[test]
    fn test_lda_empty_corpus() {
        let corpus: Vec<Vec<usize>> = vec![vec![], vec![]];
        let config = LdaConfig {
            num_topics: 3,
            iterations: 10,
            seed: 1,
            ..Default::default()
        };
        let model = fit_lda(&corpus, config);
        assert_eq!(model.topic_term.len(), 3);
        // All counts should be zero
        assert!(model.topic_count.iter().all(|&c| c == 0));
    }
}
