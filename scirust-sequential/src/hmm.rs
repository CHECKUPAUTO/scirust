//! Hidden Markov Model implementation.
//!
//! Supports discrete observations with full forward/backward/Viterbi/Baum-Welch.

use serde::{Deserialize, Serialize};

use crate::{NEG_INF, log_sum_exp, log_sum_exp_slice};

/// Index into a row-major matrix.
#[inline]
fn idx(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

/// Discrete-observation HMM.
///
/// - `transition[i][j]` = P(state_j | state_i)
/// - `emission[i][obs]` = P(obs | state_i)
/// - `initial[i]` = P(state_i at t=0)
///
/// All probabilities are stored in **log-space** for numerical stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMM {
    pub n_states: usize,
    pub n_obs: usize,
    pub log_a: Vec<f64>,
    pub log_b: Vec<f64>,
    pub log_pi: Vec<f64>,
}

impl HMM {
    /// Create a new HMM from probability (not log) matrices.
    pub fn new(
        transition: &[f64],
        emission: &[f64],
        initial: &[f64],
        n_states: usize,
        n_obs: usize,
    ) -> Result<Self, &'static str> {
        if transition.len() != n_states * n_states
        {
            return Err("transition matrix length must be n_states * n_states");
        }
        if emission.len() != n_states * n_obs
        {
            return Err("emission matrix length must be n_states * n_obs");
        }
        if initial.len() != n_states
        {
            return Err("initial vector length must be n_states");
        }
        let log_a: Vec<f64> = transition.iter().map(|&p| p.max(1e-300).ln()).collect();
        let log_b: Vec<f64> = emission.iter().map(|&p| p.max(1e-300).ln()).collect();
        let log_pi: Vec<f64> = initial.iter().map(|&p| p.max(1e-300).ln()).collect();
        Ok(HMM {
            n_states,
            n_obs,
            log_a,
            log_b,
            log_pi,
        })
    }

    /// Forward algorithm: compute log alpha(t, i) for all t, i.
    /// Returns (log_alpha, log_likelihood).
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, observations: &[usize]) -> (Vec<f64>, f64) {
        let t = observations.len();
        let n = self.n_states;
        let mut log_alpha = vec![NEG_INF; t * n];

        for i in 0..n
        {
            log_alpha[i] = self.log_pi[i] + self.log_b[idx(i, observations[0], self.n_obs)];
        }

        for step in 1..t
        {
            let prev = (step - 1) * n;
            let curr = step * n;
            let obs = observations[step];
            for j in 0..n
            {
                let mut best = NEG_INF;
                for i in 0..n
                {
                    let val = log_alpha[prev + i]
                        + self.log_a[idx(i, j, n)]
                        + self.log_b[idx(j, obs, self.n_obs)];
                    best = log_sum_exp(best, val);
                }
                log_alpha[curr + j] = best;
            }
        }

        let log_likelihood = log_sum_exp_slice(&log_alpha[(t - 1) * n..t * n]);
        (log_alpha, log_likelihood)
    }

    /// Backward algorithm: compute log beta(t, i) for all t, i.
    pub fn backward(&self, observations: &[usize]) -> Vec<f64> {
        let t = observations.len();
        let n = self.n_states;
        let mut log_beta = vec![NEG_INF; t * n];

        for i in 0..n
        {
            log_beta[(t - 1) * n + i] = 0.0;
        }

        for step in (0..t - 1).rev()
        {
            let next = (step + 1) * n;
            let curr = step * n;
            let obs_next = observations[step + 1];
            for i in 0..n
            {
                let mut best = NEG_INF;
                for j in 0..n
                {
                    let val = self.log_a[idx(i, j, n)]
                        + self.log_b[idx(j, obs_next, self.n_obs)]
                        + log_beta[next + j];
                    best = log_sum_exp(best, val);
                }
                log_beta[curr + i] = best;
            }
        }

        log_beta
    }

    /// Viterbi algorithm: find the most likely state sequence.
    #[allow(clippy::needless_range_loop)]
    pub fn viterbi(&self, observations: &[usize]) -> (Vec<usize>, f64) {
        let t = observations.len();
        let n = self.n_states;
        let mut log_delta = vec![NEG_INF; t * n];
        let mut psi = vec![0usize; t * n];

        for i in 0..n
        {
            log_delta[i] = self.log_pi[i] + self.log_b[idx(i, observations[0], self.n_obs)];
        }

        for step in 1..t
        {
            let prev = (step - 1) * n;
            let curr = step * n;
            let obs = observations[step];
            for j in 0..n
            {
                let mut best_val = NEG_INF;
                let mut best_idx = 0usize;
                for i in 0..n
                {
                    let val = log_delta[prev + i] + self.log_a[idx(i, j, n)];
                    if val > best_val
                    {
                        best_val = val;
                        best_idx = i;
                    }
                }
                log_delta[curr + j] = best_val + self.log_b[idx(j, obs, self.n_obs)];
                psi[curr + j] = best_idx;
            }
        }

        let mut states = vec![0usize; t];
        let mut best_val = NEG_INF;
        for i in 0..n
        {
            let val = log_delta[(t - 1) * n + i];
            if val > best_val
            {
                best_val = val;
                states[t - 1] = i;
            }
        }
        for step in (1..t).rev()
        {
            states[step - 1] = psi[step * n + states[step]];
        }

        (states, best_val)
    }

    /// Baum-Welch (EM) training.
    pub fn baum_welch(&mut self, sequences: &[Vec<usize>], max_iter: usize, tolerance: f64) -> f64 {
        let n = self.n_states;
        let o = self.n_obs;
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..max_iter
        {
            let mut accum_a = vec![NEG_INF; n * n];
            let mut accum_b = vec![NEG_INF; n * o];
            let mut accum_pi = vec![NEG_INF; n];
            let mut total_ll = 0.0f64;

            for obs_seq in sequences
            {
                let t = obs_seq.len();
                if t == 0
                {
                    continue;
                }
                let (log_alpha, ll) = self.forward(obs_seq);
                let log_beta = self.backward(obs_seq);
                total_ll += ll;

                let mut gamma = vec![NEG_INF; t * n];
                for step in 0..t
                {
                    let mut denom = NEG_INF;
                    for i in 0..n
                    {
                        let val = log_alpha[step * n + i] + log_beta[step * n + i];
                        denom = log_sum_exp(denom, val);
                    }
                    for i in 0..n
                    {
                        gamma[step * n + i] =
                            log_alpha[step * n + i] + log_beta[step * n + i] - denom;
                    }
                }

                for i in 0..n
                {
                    accum_pi[i] = log_sum_exp(accum_pi[i], gamma[i]);
                }

                for step in 0..t - 1
                {
                    let mut denom = NEG_INF;
                    for i in 0..n
                    {
                        for j in 0..n
                        {
                            let val = log_alpha[step * n + i]
                                + self.log_a[idx(i, j, n)]
                                + self.log_b[idx(j, obs_seq[step + 1], o)]
                                + log_beta[(step + 1) * n + j];
                            denom = log_sum_exp(denom, val);
                        }
                    }
                    for i in 0..n
                    {
                        for j in 0..n
                        {
                            let xi = log_alpha[step * n + i]
                                + self.log_a[idx(i, j, n)]
                                + self.log_b[idx(j, obs_seq[step + 1], o)]
                                + log_beta[(step + 1) * n + j]
                                - denom;
                            accum_a[idx(i, j, n)] = log_sum_exp(accum_a[idx(i, j, n)], xi);
                        }
                    }
                }

                for step in 0..t
                {
                    for i in 0..n
                    {
                        accum_b[idx(i, obs_seq[step], o)] =
                            log_sum_exp(accum_b[idx(i, obs_seq[step], o)], gamma[step * n + i]);
                    }
                }
            }

            self.log_pi[..n].copy_from_slice(&accum_pi[..n]);

            for i in 0..n
            {
                let mut row_sum = NEG_INF;
                for j in 0..n
                {
                    row_sum = log_sum_exp(row_sum, accum_a[idx(i, j, n)]);
                }
                for j in 0..n
                {
                    self.log_a[idx(i, j, n)] = accum_a[idx(i, j, n)] - row_sum;
                }
            }

            for i in 0..n
            {
                let mut row_sum = NEG_INF;
                for k in 0..o
                {
                    row_sum = log_sum_exp(row_sum, accum_b[idx(i, k, o)]);
                }
                for k in 0..o
                {
                    self.log_b[idx(i, k, o)] = accum_b[idx(i, k, o)] - row_sum;
                }
            }

            if (total_ll - prev_ll).abs() < tolerance
            {
                return total_ll;
            }
            prev_ll = total_ll;
        }

        prev_ll
    }

    /// Compute the log-probability of a single observation sequence.
    pub fn log_probability(&self, observations: &[usize]) -> f64 {
        self.forward(observations).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_hmm() -> HMM {
        let transition = [0.6, 0.4, 0.3, 0.7];
        let emission = [0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
        let initial = [0.6, 0.4];
        HMM::new(&transition, &emission, &initial, 2, 3).unwrap()
    }

    #[test]
    fn forward_gives_valid_likelihood() {
        let hmm = weather_hmm();
        let obs = vec![0, 1, 2];
        let (_, ll) = hmm.forward(&obs);
        assert!(ll.is_finite());
        assert!(ll <= 0.0);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn backward_matches_forward_likelihood() {
        let hmm = weather_hmm();
        let obs = vec![0, 1, 2];
        let (_, ll) = hmm.forward(&obs);
        let log_beta = hmm.backward(&obs);
        let n = hmm.n_states;
        let mut manual_ll = NEG_INF;
        for i in 0..n
        {
            let val = hmm.log_pi[i] + hmm.log_b[i * hmm.n_obs + obs[0]] + log_beta[i];
            manual_ll = log_sum_exp(manual_ll, val);
        }
        assert!(
            (ll - manual_ll).abs() < 1e-10,
            "ll={} manual_ll={}",
            ll,
            manual_ll
        );
    }

    #[test]
    fn viterbi_finds_valid_path() {
        let hmm = weather_hmm();
        let obs = vec![0, 1, 2];
        let (states, log_prob) = hmm.viterbi(&obs);
        assert_eq!(states.len(), 3);
        assert!(log_prob.is_finite());
        for &s in &states
        {
            assert!(s < hmm.n_states);
        }
    }

    #[test]
    fn baum_welch_improves_likelihood() {
        let mut hmm = weather_hmm();
        let obs = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 2, 0]];
        let ll_before = obs.iter().map(|o| hmm.log_probability(o)).sum::<f64>();
        hmm.baum_welch(&obs, 50, 1e-8);
        let ll_after = obs.iter().map(|o| hmm.log_probability(o)).sum::<f64>();
        assert!(
            ll_after >= ll_before - 1e-6,
            "ll_before={} ll_after={}",
            ll_before,
            ll_after
        );
    }

    #[test]
    fn log_probability_consistency() {
        let hmm = weather_hmm();
        let obs = vec![0, 1, 2];
        let ll = hmm.log_probability(&obs);
        assert!((ll - hmm.forward(&obs).1).abs() < 1e-15);
    }
}
