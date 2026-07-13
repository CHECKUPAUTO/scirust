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
    /// Topic-term counts: `topic_term[t][w]` = count of word w in topic t.
    pub topic_term: Vec<Vec<usize>>,
    /// Document-topic counts: `doc_topic[d][t]` = count of topic t in doc d.
    pub doc_topic: Vec<Vec<usize>>,
    /// Total words per topic.
    pub topic_count: Vec<usize>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Mapping from vocabulary index → word string.
    pub vocab: Vec<String>,
}

/// Invalid vocabulary supplied to [`fit_lda_with_vocab`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LdaError {
    /// A corpus token refers past the end of the supplied vocabulary.
    TokenOutOfVocabulary { token: usize, vocab_size: usize },
}

impl core::fmt::Display for LdaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::TokenOutOfVocabulary { token, vocab_size } => write!(
                f,
                "corpus token {token} is outside the vocabulary of size {vocab_size}"
            ),
        }
    }
}

impl std::error::Error for LdaError {}

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
        if tt.is_empty() || self.config.num_topics == 0
        {
            return f64::INFINITY;
        }
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
        .map(|max_token| {
            max_token
                .checked_add(1)
                .expect("vocabulary index usize::MAX cannot be represented")
        })
        .unwrap_or(0);

    // This API receives integer token ids only, so expose stable, explicit id
    // labels instead of pretending that their original word strings are known.
    // Call `fit_lda_with_vocab` when human-readable top words are required.
    let vocab: Vec<String> = (0..vocab_size).map(|i| format!("token:{i}")).collect();
    fit_lda_impl(corpus, config, vocab)
}

/// Fit LDA while preserving the caller's real token-to-word mapping.
///
/// Unlike [`fit_lda`], [`LdaModel::top_words`] will return the supplied words.
/// Every token id in `corpus` is validated before sampling.
pub fn fit_lda_with_vocab(
    corpus: &[Vec<usize>],
    vocab: Vec<String>,
    config: LdaConfig,
) -> Result<LdaModel, LdaError> {
    if let Some(token) = corpus
        .iter()
        .flat_map(|document| document.iter())
        .copied()
        .find(|&token| token >= vocab.len())
    {
        return Err(LdaError::TokenOutOfVocabulary {
            token,
            vocab_size: vocab.len(),
        });
    }
    Ok(fit_lda_impl(corpus, config, vocab))
}

fn fit_lda_impl(corpus: &[Vec<usize>], config: LdaConfig, vocab: Vec<String>) -> LdaModel {
    let vocab_size = vocab.len();
    let num_topics = config.num_topics;
    let num_docs = corpus.len();

    // Initialize count matrices
    let mut topic_term: Vec<Vec<usize>> = vec![vec![0; vocab_size]; num_topics];
    let mut doc_topic: Vec<Vec<usize>> = vec![vec![0; num_topics]; num_docs];
    let mut topic_count: Vec<usize> = vec![0; num_topics];

    // With zero topics there is nothing to sample: return an empty (but
    // well-formed) model instead of dividing/modulo-ing by zero below.
    if num_topics == 0
    {
        return LdaModel {
            config,
            topic_term,
            doc_topic,
            topic_count,
            vocab_size,
            vocab,
        };
    }

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
                for (t, &probability) in probs.iter().enumerate().take(num_topics)
                {
                    r -= probability;
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
        for j in 0..i
        {
            let wj = top_indices[j];
            // D(w_j): number of documents containing the conditioning word w_j.
            let d_wj: f64 = doc_word_matrix
                .iter()
                .filter(|doc| doc.contains(&wj))
                .count() as f64;
            // D(w_i, w_j): documents containing both.
            let d_wi_wj: f64 = doc_word_matrix
                .iter()
                .filter(|doc| doc.contains(&w) && doc.contains(&wj))
                .count() as f64;
            if d_wj > 0.0 && d_wi_wj > 0.0
            {
                // UMass coherence (Mimno et al. 2011): log of the ratio, not
                // a ratio of two logs, and conditioned on D(w_j) -- not
                // D(w_i).
                score += ((d_wi_wj + 1.0) / d_wj).ln();
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
        assert_eq!(top[0][0].0, "token:0");
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
        // Hand-derived from the UMass formula ln((D(w_i,w_j)+1)/D(w_j)) over the
        // four pairs with a nonzero co-occurrence count: (w1,w0): ln(3/2);
        // (w2,w0): ln(2/2); (w2,w1): ln(2/2); (w3,w2): ln(2/2). Average of
        // [ln(1.5), 0, 0, 0] over 4 terms.
        assert!((score - (1.5_f64.ln() / 4.0)).abs() < 1e-9, "score={score}");
    }

    #[test]
    fn test_coherence_umass_matches_reference_formula() {
        // UMass coherence (Mimno et al. 2011) is
        // sum_{i>j} ln( (D(w_i,w_j)+1) / D(w_j) ), i.e. the log of a ratio,
        // conditioned on the *earlier* (higher-ranked) word's document
        // frequency D(w_j) -- not a ratio of two logarithms, and not
        // conditioned on D(w_i). word 0 appears in 4 of 5 documents; word 1
        // co-occurs with it in only 1 of those, so the pair should score
        // ln((1+1)/4) = ln(0.5), a clearly negative, hand-verifiable value.
        let topic_term = vec![5, 3];
        let doc_word_matrix = vec![vec![0, 1], vec![0], vec![0], vec![0], vec![2]];
        let score = coherence_umass(&topic_term, &doc_word_matrix, 0);
        assert!(
            (score - 0.5_f64.ln()).abs() < 1e-9,
            "expected ln(0.5) ~= {}, got {score}",
            0.5_f64.ln()
        );
    }

    #[test]
    fn test_lda_zero_topics_does_not_panic() {
        // Regression: num_topics == 0 previously caused a modulo-by-zero
        // panic during random initialization (and an underflow panic in the
        // Gibbs loop).  A non-empty corpus makes the bug reachable.
        let corpus = vec![vec![0, 1, 2], vec![1, 2, 0]];
        let config = LdaConfig {
            num_topics: 0,
            iterations: 10,
            seed: 1,
            ..Default::default()
        };
        let model = fit_lda(&corpus, config);
        // No topics means empty topic-dimensioned structures.
        assert_eq!(model.topic_term.len(), 0);
        assert_eq!(model.topic_count.len(), 0);
        assert_eq!(model.doc_topic.len(), 2);
        assert!(model.doc_topic.iter().all(|row| row.is_empty()));
        // Vocabulary is still inferred from the corpus indices.
        assert_eq!(model.vocab_size, 3);
        // Derived distributions must also stay panic-free.
        assert_eq!(model.topic_term_dist().len(), 0);
        assert_eq!(model.doc_topic_dist().len(), 2);
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
        assert_eq!(model.vocab_size, 0);
    }

    #[test]
    fn supplied_vocabulary_is_preserved_and_checked() {
        let corpus = vec![vec![0, 0, 0, 1], vec![2, 2]];
        let vocab = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let model = fit_lda_with_vocab(
            &corpus,
            vocab,
            LdaConfig {
                num_topics: 1,
                iterations: 5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(model.top_words(1)[0][0].0, "cat");

        let error = fit_lda_with_vocab(&[vec![3]], vec!["only".to_string()], LdaConfig::default())
            .unwrap_err();
        assert_eq!(
            error,
            LdaError::TokenOutOfVocabulary {
                token: 3,
                vocab_size: 1,
            }
        );
    }
}
