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
        if t == 0
        {
            // An empty sequence has no paths, so the partition function is
            // log(0) = NEG_INF and both trellises are empty.
            return (Vec::new(), Vec::new(), NEG_INF);
        }
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
        if t == 0
        {
            // Nothing to decode: empty tag sequence with score log(0) = NEG_INF.
            return (Vec::new(), NEG_INF);
        }
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
        let (log_alpha, log_beta, log_z) = self.forward_backward(observations, features);

        let mut grad = vec![0.0; self.n_features];

        // Model expectation of features:
        //   E[f_k] = sum_step sum_prev sum_cur P(prev@step-1, cur@step | x) * f_k(prev,cur,x_step)
        // For step == 0 this reduces to the node/start marginal P(cur@0 | x) with the
        // start "prev" index pinned to 0 (matching the feature-cache start convention).
        // The single global log-partition `log_z` is the correct normalizer for every
        // edge/node marginal.
        for step in 0..t
        {
            for prev in 0..n
            {
                for cur in 0..n
                {
                    // Forward variable feeding into `step`: alpha at step-1 for step>=1,
                    // and the start potential alpha[0][cur] for step==0.
                    let log_forward = if step == 0
                    {
                        // Only the start row (prev index 0) carries probability mass at
                        // step 0; other prev rows are spurious duplicates of the same
                        // node marginal, so skip them.
                        if prev != 0
                        {
                            continue;
                        }
                        log_alpha[cur]
                    }
                    else
                    {
                        log_alpha[(step - 1) * n + prev] + cache.dot(&self.weights, step, prev, cur)
                    };

                    let log_prob = log_forward + log_beta[step * n + cur] - log_z;
                    let prob = log_prob.exp();
                    if prob == 0.0
                    {
                        continue;
                    }
                    let base = step * n * n * self.n_features
                        + prev * n * self.n_features
                        + cur * self.n_features;
                    for k in 0..self.n_features
                    {
                        grad[k] += prob * cache.data[base + k];
                    }
                }
            }
        }

        // Empirical (gold) features: sum_step f(gold_{step-1}, gold_step, x_step).
        // Gradient of the NLL is E_model[f] - empirical[f].
        let mut prev_tag = None;
        for (step, &cur) in gold_tags.iter().enumerate()
        {
            let base = step * n * n * self.n_features
                + prev_tag.unwrap_or(0) * n * self.n_features
                + cur * self.n_features;
            for k in 0..self.n_features
            {
                grad[k] -= cache.data[base + k];
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

    #[test]
    fn crf_gradient_matches_finite_difference() {
        let features = toy_features();
        let mut crf = LinearChainCRF::new(3, 2);
        crf.weights = vec![0.3, 0.5];
        let obs = vec![0, 1, 2];
        let gold = vec![0, 1, 2];

        let analytic = crf.gradient(&obs, &gold, &features);

        // Central finite-difference of the NLL.
        let eps = 1e-6;
        let mut numeric = [0.0; 2];
        for (k, slot) in numeric.iter_mut().enumerate()
        {
            let mut wp = crf.clone();
            wp.weights[k] += eps;
            let mut wm = crf.clone();
            wm.weights[k] -= eps;
            let fp = wp.nll(&obs, &gold, &features);
            let fm = wm.nll(&obs, &gold, &features);
            *slot = (fp - fm) / (2.0 * eps);
        }

        // Independently hand-derived expected gradient.
        let expected = [0.781_401_928_5, -1.672_873_397_2];
        for k in 0..2
        {
            assert!(
                (analytic[k] - numeric[k]).abs() < 1e-4,
                "analytic vs numeric mismatch at {}: {} vs {}",
                k,
                analytic[k],
                numeric[k]
            );
            assert!(
                (analytic[k] - expected[k]).abs() < 1e-4,
                "analytic vs expected mismatch at {}: {} vs {}",
                k,
                analytic[k],
                expected[k]
            );
        }
    }

    #[test]
    fn forward_backward_empty_observations_does_not_panic() {
        let features = toy_features();
        let mut crf = LinearChainCRF::new(3, 2);
        crf.weights = vec![0.5, 0.3];
        let obs: Vec<usize> = Vec::new();
        // Previously underflowed `(t - 1) * n` and panicked on an empty slice.
        let (log_alpha, log_beta, ll) = crf.forward_backward(&obs, &features);
        assert!(log_alpha.is_empty());
        assert!(log_beta.is_empty());
        assert_eq!(ll, NEG_INF);
    }

    #[test]
    fn decode_empty_observations_does_not_panic() {
        let features = toy_features();
        let mut crf = LinearChainCRF::new(3, 2);
        crf.weights = vec![1.0, 1.0];
        let obs: Vec<usize> = Vec::new();
        // Previously underflowed `(t - 1) * n` and panicked.
        let (tags, score) = crf.decode(&obs, &features);
        assert!(tags.is_empty());
        assert_eq!(score, NEG_INF);
    }

    #[test]
    fn crf_nll_zero_weights_equals_log_num_paths() {
        let features = toy_features();
        // Default weights are all zero.
        let crf = LinearChainCRF::new(3, 2);
        let obs = vec![0, 1, 2];
        let gold = vec![0, 1, 2];
        let loss = crf.nll(&obs, &gold, &features);
        let expected = 27.0f64.ln();
        assert!(
            (loss - expected).abs() < 1e-9,
            "nll={} expected={}",
            loss,
            expected
        );
    }
}
