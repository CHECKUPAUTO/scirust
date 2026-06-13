//! Seeded, deterministic token sampling for autoregressive generation.
//!
//! Greedy (argmax), temperature, top-k and top-p (nucleus) sampling, all driven
//! by a seeded [`PcgEngine`] — so the same seed and the same logits always
//! produce the same token, preserving the project's reproducibility guarantee.

use crate::nn::rng::PcgEngine;

/// Sampling parameters. The defaults are pure greedy decoding.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Softmax temperature. `<= 0` selects greedy argmax (deterministic).
    pub temperature: f32,
    /// Keep only the `top_k` highest-probability tokens (`0` disables it;
    /// `1` is equivalent to greedy).
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability ≥ `top_p` (`>= 1.0` disables it).
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Pure greedy decoding (argmax).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Index of the maximum logit (ties → lowest index, deterministic).
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate()
    {
        if v > best_v
        {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Sample a token index from `logits` under `cfg`, using `rng` for the random
/// draw. Deterministic in `(logits, cfg, rng-state)`.
pub fn sample_token(logits: &[f32], cfg: &SamplingConfig, rng: &mut PcgEngine) -> usize {
    if logits.is_empty()
    {
        return 0;
    }
    // Greedy shortcuts (no randomness consumed).
    if cfg.temperature <= 0.0 || cfg.top_k == 1
    {
        return argmax(logits);
    }

    // Temperature-scaled softmax (shift by max for numerical stability).
    let inv_t = 1.0 / cfg.temperature;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - max) * inv_t).exp()).collect();

    // Rank indices by probability, descending (deterministic tie-break by index).
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // top-k: zero out everything past the k-th most probable.
    if cfg.top_k > 0 && cfg.top_k < order.len()
    {
        for &i in &order[cfg.top_k..]
        {
            probs[i] = 0.0;
        }
    }

    // top-p (nucleus): keep the smallest prefix whose cumulative mass ≥ top_p.
    if cfg.top_p < 1.0
    {
        let total: f32 = probs.iter().sum();
        if total > 0.0
        {
            let mut cum = 0.0f32;
            let mut cutoff = order.len();
            for (rank, &i) in order.iter().enumerate()
            {
                cum += probs[i] / total;
                if cum >= cfg.top_p
                {
                    cutoff = rank + 1;
                    break;
                }
            }
            for &i in &order[cutoff..]
            {
                probs[i] = 0.0;
            }
        }
    }

    // Normalise and draw.
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0
    {
        return argmax(logits);
    }
    let r = rng.float() * sum; // uniform in [0, sum)
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate()
    {
        cum += p;
        if r < cum
        {
            return i;
        }
    }
    *order.first().unwrap_or(&0) // fall back to the most probable token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1, 2.0, -1.0, 1.9];
        let mut rng = PcgEngine::new(0);
        assert_eq!(
            sample_token(&logits, &SamplingConfig::greedy(), &mut rng),
            1
        );
        // top_k = 1 is also greedy, regardless of temperature.
        let cfg = SamplingConfig {
            temperature: 5.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(sample_token(&logits, &cfg, &mut rng), 1);
    }

    #[test]
    fn sampling_is_deterministic_per_seed() {
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let draw = |seed: u64| {
            let mut rng = PcgEngine::new(seed);
            (0..20)
                .map(|_| sample_token(&logits, &cfg, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42)); // same seed ⇒ identical stream
        assert!(draw(42).iter().all(|&i| i < 5)); // always in range
    }

    #[test]
    fn top_k_restricts_support() {
        // With top_k = 2, only the two highest-logit tokens (1 and 3) can appear.
        let logits = [0.0, 10.0, 0.0, 9.0, 0.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        let mut rng = PcgEngine::new(7);
        for _ in 0..50
        {
            let t = sample_token(&logits, &cfg, &mut rng);
            assert!(t == 1 || t == 3, "top_k=2 produced out-of-set token {t}");
        }
    }

    #[test]
    fn top_p_restricts_support() {
        // token 1 alone already exceeds 0.5 of the mass → nucleus = {1}.
        let logits = [0.0, 10.0, 0.0, 0.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
        };
        let mut rng = PcgEngine::new(3);
        for _ in 0..30
        {
            assert_eq!(sample_token(&logits, &cfg, &mut rng), 1);
        }
    }
}
