//! Linear-chain Conditional Random Field (CRF).

use crate::{NEG_INF, log_sum_exp, log_sum_exp_slice};
use serde::{Deserialize, Serialize};

/// Feature function: (prev_tag, cur_tag, obs_index) -> f64.
pub type FeatureFn = dyn Fn(Option<usize>, usize, usize) -> f64;

/// Linear-chain CRF parameterized by a weight vector over features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearChainCRF {
    pub n_tags: usize,
    pub weights: Vec<f64>,
    pub n_features: usize,
}

/// Pre-computed dense feature matrix for a single observation sequence.
struct FeatureCache {
    /// shape: [t * n_tags * n_tags * n_features]
    data: Vec<f64>,
    n_tags: usize,
    n_features: usize,
}

impl FeatureCache {
    #[allow(clippy::needless_range_loop)]
    fn new(
        observations: &[usize],
        n_tags: usize,
        n_features: usize,
        features: &[Box<FeatureFn>],
    ) -> Self {
        let t = observations.len();
        let total = t * n_tags * n_tags * n_features;
        let mut data = vec![0.0; total];
        for step in 0..t
        {
            let obs = observations[step];
            let base = step * n_tags * n_tags * n_features;
            for prev in 0..n_tags
            {
                let prev_tag = if step == 0 { None } else { Some(prev) };
                for cur in 0..n_tags
                {
                    let offset = base + prev * n_tags * n_features + cur * n_features;
                    for (k, f) in features.iter().enumerate()
                    {
                        data[offset + k] = f(prev_tag, cur, obs);
                    }
                }
            }
        }
        FeatureCache {
            data,
            n_tags,
            n_features,
        }
    }

    /// Dot product of feature vector at (t, prev, cur) with weights.
    #[allow(clippy::needless_range_loop)]
    fn dot(&self, weights: &[f64], t: usize, prev: usize, cur: usize) -> f64 {
        let base = t * self.n_tags * self.n_tags * self.n_features;
        let offset = base + prev * self.n_tags * self.n_features + cur * self.n_features;
        let mut sum = 0.0;
        for k in 0..self.n_features
        {
            sum += self.data[offset + k] * weights[k];
        }
        sum
    }
}

impl LinearChainCRF {
    pub fn new(n_tags: usize, n_features: usize) -> Self {
        LinearChainCRF {
            n_tags,
            weights: vec![0.0; n_features],
            n_features,
        }
    }

    pub fn set_weights(&mut self, w: Vec<f64>) {
        assert_eq!(w.len(), self.n_features);
        self.weights = w;
    }

    /// Forward-backward inference. Returns (log_alpha, log_beta, log_partition_function).
    #[allow(clippy::needless_range_loop)]
    pub fn forward_backward(
        &self,
        observations: &[usize],
        features: &[Box<FeatureFn>],
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let t = observations.len();
        let n = self.n_tags;
        let cache = FeatureCache::new(observations, n, self.n_features, features);

        // Forward: log_alpha[step][cur]
        let mut log_alpha = vec![NEG_INF; t * n];
        for cur in 0..n
        {
            log_alpha[cur] = cache.dot(&self.weights, 0, 0, cur);
        }
        for step in 1..t
        {
            let prev_base = (step - 1) * n;
            let curr_base = step * n;
            for cur in 0..n
            {
                let mut best = NEG_INF;
                for prev in 0..n
                {
                    let val =
                        log_alpha[prev_base + prev] + cache.dot(&self.weights, step, prev, cur);
                    best = log_sum_exp(best, val);
                }
                log_alpha[curr_base + cur] = best;
            }
        }
        let ll = log_sum_exp_slice(&log_alpha[(t - 1) * n..t * n]);

        // Backward: log_beta[step][prev]
        // beta[t-1][*] = 0
        // beta[step][prev] = max_cur( f(step+1, prev, cur) + beta[step+1][cur] )
        let mut log_beta = vec![NEG_INF; t * n];
        for i in 0..n
        {
            log_beta[(t - 1) * n + i] = 0.0;
        }
        for step in (0..t - 1).rev()
        {
            let next_base = (step + 1) * n;
            let curr_base = step * n;
            for prev in 0..n
            {
                let mut best = NEG_INF;
                for cur in 0..n
                {
                    // feature connects prev->cur at position step+1
                    let val =
                        cache.dot(&self.weights, step + 1, prev, cur) + log_beta[next_base + cur];
                    best = log_sum_exp(best, val);
                }
                log_beta[curr_base + prev] = best;
            }
        }

        (log_alpha, log_beta, ll)
    }

    /// Viterbi decoding: find the best tag sequence.
    #[allow(clippy::needless_range_loop)]
    pub fn decode(&self, observations: &[usize], features: &[Box<FeatureFn>]) -> (Vec<usize>, f64) {
        let t = observations.len();
        let n = self.n_tags;
        let cache = FeatureCache::new(observations, n, self.n_features, features);

        let mut log_delta = vec![NEG_INF; t * n];
        let mut psi = vec![0usize; t * n];

        for cur in 0..n
        {
            log_delta[cur] = cache.dot(&self.weights, 0, 0, cur);
        }

        for step in 1..t
        {
            let prev_base = (step - 1) * n;
            let curr_base = step * n;
            for cur in 0..n
            {
                let mut best_val = NEG_INF;
                let mut best_prev = 0usize;
                for prev in 0..n
                {
                    let val =
                        log_delta[prev_base + prev] + cache.dot(&self.weights, step, prev, cur);
                    if val > best_val
                    {
                        best_val = val;
                        best_prev = prev;
                    }
                }
                log_delta[curr_base + cur] = best_val;
                psi[curr_base + cur] = best_prev;
            }
        }

        let mut tags = vec![0usize; t];
        let mut best_val = NEG_INF;
        for i in 0..n
        {
            if log_delta[(t - 1) * n + i] > best_val
            {
                best_val = log_delta[(t - 1) * n + i];
                tags[t - 1] = i;
            }
        }
        for step in (1..t).rev()
        {
            tags[step - 1] = psi[step * n + tags[step]];
        }

        (tags, best_val)
    }

    /// Negative log-likelihood loss for a single labeled sequence.
    pub fn nll(
        &self,
        observations: &[usize],
        gold_tags: &[usize],
        features: &[Box<FeatureFn>],
    ) -> f64 {
        assert_eq!(observations.len(), gold_tags.len());
        let cache = FeatureCache::new(observations, self.n_tags, self.n_features, features);

        let mut numerator = 0.0;
        let mut prev_tag = None;
        for (step, &cur) in gold_tags.iter().enumerate()
        {
            numerator += cache.dot(&self.weights, step, prev_tag.unwrap_or(0), cur);
            prev_tag = Some(cur);
        }

        let (_, _, log_z) = self.forward_backward(observations, features);
        -numerator + log_z
    }

    /// Gradient of NLL with respect to weights for a single sequence.
    #[allow(clippy::needless_range_loop)]
    pub fn gradient(
        &self,
        observations: &[usize],
        gold_tags: &[usize],
        features: &[Box<FeatureFn>],
    ) -> Vec<f64> {
        assert_eq!(observations.len(), gold_tags.len());
        let t = observations.len();
        let n = self.n_tags;
        let cache = FeatureCache::new(observations, n, self.n_features, features);
        let (log_alpha, log_beta, _ll) = self.forward_backward(observations, features);

        let mut grad = vec![0.0; self.n_features];

        // Expected features: sum_t sum_prev sum_cur P(prev,cur|x) * f(prev,cur,x_t)
        for step in 0..t
        {
            let mut denom = NEG_INF;
            for i in 0..n
            {
                denom = log_sum_exp(denom, log_alpha[step * n + i] + log_beta[step * n + i]);
            }
            for prev in 0..n
            {
                for cur in 0..n
                {
                    let log_prob = log_alpha[step * n + prev]
                        + cache.dot(&self.weights, step, prev, cur)
                        + log_beta[(if step + 1 < t { step + 1 } else { step }) * n + cur]
                        - denom;
                    if log_prob > NEG_INF + 10.0
                    {
                        let prob = log_prob.exp();
                        let base = step * n * n * self.n_features
                            + prev * n * self.n_features
                            + cur * self.n_features;
                        for k in 0..self.n_features
                        {
                            grad[k] -= prob * cache.data[base + k];
                        }
                    }
                }
            }
        }

        // Gold features: sum_t f(gold_{t-1}, gold_t, x_t)
        let mut prev_tag = None;
        for (step, &cur) in gold_tags.iter().enumerate()
        {
            let base = step * n * n * self.n_features
                + prev_tag.unwrap_or(0) * n * self.n_features
                + cur * self.n_features;
            for k in 0..self.n_features
            {
                grad[k] += cache.data[base + k];
            }
            prev_tag = Some(cur);
        }

        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_features() -> Vec<Box<FeatureFn>> {
        let mut fns: Vec<Box<FeatureFn>> = Vec::new();
        fns.push(Box::new(
            |prev, cur, _| {
                if prev == Some(cur) { 1.0 } else { 0.0 }
            },
        ));
        fns.push(Box::new(|_, cur, obs| if cur == obs { 1.0 } else { 0.0 }));
        fns
    }

    #[test]
    fn forward_backward_valid_likelihood() {
        let features = toy_features();
        let mut crf = LinearChainCRF::new(3, 2);
        crf.weights = vec![0.5, 0.3];
        let obs = vec![0, 1, 2];
        let (_, _, ll) = crf.forward_backward(&obs, &features);
        assert!(ll.is_finite());
    }

    #[test]
    fn decode_returns_valid_sequence() {
        let features = toy_features();
        let mut crf = LinearChainCRF::new(3, 2);
        crf.weights = vec![1.0, 1.0];
        let obs = vec![0, 1, 2];
        let (tags, score) = crf.decode(&obs, &features);
        assert_eq!(tags.len(), 3);
        assert!(score.is_finite());
        for &t in &tags
        {
            assert!(t < 3);
        }
    }

    #[test]
    fn nll_is_finite() {
        let features = toy_features();
        let mut crf = LinearChainCRF::new(3, 2);
        crf.weights = vec![0.1, 0.2];
        let obs = vec![0, 1, 2];
        let gold = vec![0, 1, 2];
        let loss = crf.nll(&obs, &gold, &features);
        assert!(loss.is_finite(), "nll should be finite, got {}", loss);
    }

    #[test]
    fn gradient_shape() {
        let features = toy_features();
        let crf = LinearChainCRF::new(3, 2);
        let obs = vec![0, 1, 2];
        let gold = vec![0, 1, 2];
        let g = crf.gradient(&obs, &gold, &features);
        assert_eq!(g.len(), 2);
    }
}
