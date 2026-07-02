use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::Module;

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;

/// Decoding knobs, applied in the order: repetition penalty → temperature
/// (or greedy) → top-k → top-p → renormalise → sample. Everything is
/// deterministic given (prompt, seed, settings).
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// `<= 0.0` means greedy argmax (after the repetition penalty).
    pub temperature: f32,
    /// Keep only the `k` highest-probability tokens. `0` disables.
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability reaches `p`. `>= 1.0` disables.
    pub top_p: f32,
    /// CTRL-style penalty: logits of tokens present in the recent window are
    /// divided by this when positive, multiplied when negative. `1.0`
    /// disables. Also applies to greedy decoding, where it breaks loops.
    pub repetition_penalty: f32,
    /// How many trailing context tokens the repetition penalty looks at.
    pub repetition_window: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 64,
        }
    }
}

pub struct Generator {
    config: SciAgentConfig,
    sampling: SamplingParams,
}

impl Generator {
    pub fn new(config: &SciAgentConfig) -> Self {
        Self {
            config: config.clone(),
            sampling: SamplingParams::default(),
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.sampling.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.sampling.top_k = top_k;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.sampling.top_p = top_p;
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.sampling.repetition_penalty = penalty;
        self
    }

    pub fn generate(
        &self,
        model: &mut SciAgentModel,
        prompt: &[usize],
        max_tokens: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng_state = seed_to_state(seed);
        let mut ids = prompt.to_vec();
        let max_seq_len = self.config.max_seq_len;

        for _ in 0..max_tokens
        {
            let tape = Tape::new();
            let seq_len = ids.len();

            let input_slice = if seq_len > max_seq_len
            {
                &ids[seq_len - max_seq_len..]
            }
            else
            {
                &ids
            };

            let local_seq = input_slice.len();
            let logits = model.forward(&tape, input_slice, local_seq);
            let next = sample_next(
                &tape,
                logits.idx(),
                self.config.vocab_size,
                &self.sampling,
                &ids,
                &mut rng_state,
            );
            ids.push(next);
            if next == 0
            {
                break;
            }
        }
        ids
    }

    pub fn generate_with_cache(
        &self,
        model: &mut SciAgentModel,
        prompt: &[usize],
        max_tokens: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng_state = seed_to_state(seed);
        let tape = Tape::new();
        let data: Vec<f32> = prompt.iter().map(|&id| id as f32).collect();
        let n = data.len();
        let idx_t = tape.input(Tensor::from_vec(data, n, 1));
        let mut h = model.embed.forward(&tape, idx_t);

        for layer in &mut model.layers
        {
            h = layer.forward(&tape, h, n);
        }
        h = model.rms_final.forward(&tape, h);

        let last_logits = match model.lm_head.as_mut()
        {
            Some(head) => head.forward(&tape, h),
            None =>
            {
                let w = tape.input(model.embed.weight.clone());
                h.try_matmul(w.transpose_2d()).unwrap()
            },
        };

        let last_idx = sample_next(
            &tape,
            last_logits.idx(),
            self.config.vocab_size,
            &self.sampling,
            prompt,
            &mut rng_state,
        );

        let mut output = prompt.to_vec();
        output.push(last_idx);

        reset_kv_caches(model);

        for _ in 1..max_tokens
        {
            let tape = Tape::new();
            let token_data = vec![output.last().copied().unwrap_or(0) as f32];
            let token_v = tape.input(Tensor::from_vec(token_data, 1, 1));
            let h = model.embed.forward(&tape, token_v);
            let mut h = h;

            for layer in &mut model.layers
            {
                h = layer.infer_step(&tape, h, output.len() - 1);
            }
            h = model.rms_final.forward(&tape, h);

            let logits = match model.lm_head.as_mut()
            {
                Some(head) => head.forward(&tape, h),
                None =>
                {
                    let w = tape.input(model.embed.weight.clone());
                    h.try_matmul(w.transpose_2d()).unwrap()
                },
            };

            let next = sample_next(
                &tape,
                logits.idx(),
                self.config.vocab_size,
                &self.sampling,
                &output,
                &mut rng_state,
            );
            output.push(next);
            if next == 0
            {
                break;
            }
        }
        output
    }
}

fn seed_to_state(seed: u64) -> u64 {
    // splitmix64: xorshift's first outputs correlate strongly with a
    // low-entropy seed (seed=1 gives u≈1e-10, i.e. "always token 0"), and
    // state 0 is a fixed point — scrambling fixes both.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }
}

fn next_uniform(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
}

fn sample_next(
    tape: &Tape,
    logits_idx: usize,
    vocab: usize,
    params: &SamplingParams,
    recent: &[usize],
    rng: &mut u64,
) -> usize {
    let t = tape.value(logits_idx);
    let row_start = t.data.len() - vocab;
    let mut row: Vec<f32> = t.data[row_start..row_start + vocab].to_vec();

    // Repetition penalty (CTRL, Keskar et al. 2019): demote tokens already
    // present in the trailing window. Dividing a positive logit and
    // multiplying a negative one both shrink its probability.
    if params.repetition_penalty > 1.0
    {
        let start = recent.len().saturating_sub(params.repetition_window);
        for &tok in &recent[start..]
        {
            if tok < vocab
            {
                let l = row[tok];
                row[tok] = if l > 0.0
                {
                    l / params.repetition_penalty
                }
                else
                {
                    l * params.repetition_penalty
                };
            }
        }
    }

    if params.temperature <= 0.0
    {
        let mut best = 0usize;
        let mut best_val = row[0];
        for (j, &v) in row.iter().enumerate().skip(1)
        {
            if v > best_val
            {
                best_val = v;
                best = j;
            }
        }
        return best;
    }

    // softmax(logits / T), max-subtracted for numerical stability.
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = row
        .iter()
        .map(|&v| ((v - max) / params.temperature).exp())
        .collect();
    let z: f32 = probs.iter().sum();
    for p in probs.iter_mut()
    {
        *p /= z;
    }

    // top-k, then top-p, both on the descending-probability order.
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut cut = vocab;
    if params.top_k > 0
    {
        cut = cut.min(params.top_k);
    }
    if params.top_p < 1.0
    {
        let mut cum = 0.0f32;
        for (rank, &tok) in order.iter().enumerate().take(cut)
        {
            cum += probs[tok];
            if cum >= params.top_p
            {
                cut = rank + 1;
                break;
            }
        }
    }
    let kept = &order[..cut.max(1)];
    let z: f32 = kept.iter().map(|&tok| probs[tok]).sum();

    let u = next_uniform(rng) * z;
    let mut acc = 0.0f32;
    for &tok in kept
    {
        acc += probs[tok];
        if u < acc
        {
            return tok;
        }
    }
    kept[kept.len() - 1] // float round-off fallback
}

fn reset_kv_caches(model: &mut SciAgentModel) {
    for layer in &mut model.layers
    {
        layer.attn.kv_cache = std::cell::RefCell::new(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;
    use crate::model::SciAgentModel;
    use scirust_core::autodiff::reverse::Tape;

    #[test]
    fn test_generate_with_cache_deterministic() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let gen = Generator::new(&cfg);
        let prompt = vec![4usize, 5, 6];
        let out1 = gen.generate_with_cache(&mut model, &prompt, 10, 42);
        let mut model2 = SciAgentModel::new(&cfg);
        let out2 = gen.generate_with_cache(&mut model2, &prompt, 10, 42);
        assert_eq!(out1, out2, "KV-cached generation should be deterministic");
    }

    #[test]
    fn test_generate_with_cache_grows() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let gen = Generator::new(&cfg);
        let prompt = vec![4usize, 5, 6];
        let out = gen.generate_with_cache(&mut model, &prompt, 10, 42);
        assert!(
            out.len() > prompt.len(),
            "Should produce more tokens than prompt"
        );
    }

    #[test]
    fn test_forward_consistency() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);

        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7];
        let logits_full = model.forward(&tape, &input_ids, 4);
        let v_full = tape.value(logits_full.idx()).data.clone();

        let data: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
        let tape2 = Tape::new();
        let idx_t = tape2.input(Tensor::from_vec(data, 4, 1));
        let mut h = model.embed.forward(&tape2, idx_t);
        for layer in &mut model.layers
        {
            h = layer.forward(&tape2, h, 4);
        }
        h = model.rms_final.forward(&tape2, h);
        let logits_partial = match model.lm_head.as_mut()
        {
            Some(head) => head.forward(&tape2, h),
            None =>
            {
                let w = tape2.input(model.embed.weight.clone());
                h.try_matmul(w.transpose_2d()).unwrap()
            },
        };
        let v_partial = tape2.value(logits_partial.idx()).data.clone();
        assert_eq!(
            v_full, v_partial,
            "Manual forward should match model.forward"
        );
    }

    #[test]
    fn temperature_zero_is_argmax_and_sampling_is_seed_deterministic() {
        let cfg = SciAgentConfig::debug();
        let gen_greedy = Generator::new(&cfg);
        let gen_zero_t = Generator::new(&cfg).with_temperature(0.0);
        let prompt = vec![4usize, 5, 6];

        let mut m1 = SciAgentModel::new(&cfg);
        let out_greedy = gen_greedy.generate(&mut m1, &prompt, 8, 42);
        let mut m2 = SciAgentModel::new(&cfg);
        let out_zero_t = gen_zero_t.generate(&mut m2, &prompt, 8, 42);
        assert_eq!(out_greedy, out_zero_t, "T=0 must be plain argmax");

        let gen_hot = Generator::new(&cfg).with_temperature(1.0);
        let mut m3 = SciAgentModel::new(&cfg);
        let out_a = gen_hot.generate(&mut m3, &prompt, 8, 42);
        let mut m4 = SciAgentModel::new(&cfg);
        let out_b = gen_hot.generate(&mut m4, &prompt, 8, 42);
        assert_eq!(out_a, out_b, "same seed must reproduce the same sample");
    }

    #[test]
    fn high_temperature_actually_samples() {
        // On a random-init model logits are near-uniform: at T=2 different
        // seeds must diverge (probability of 12 identical draws from ~256
        // near-equiprobable tokens is negligible), proving the RNG is used.
        let cfg = SciAgentConfig::debug();
        let gen = Generator::new(&cfg).with_temperature(2.0);
        let prompt = vec![4usize, 5, 6];
        let mut m1 = SciAgentModel::new(&cfg);
        let out_a = gen.generate(&mut m1, &prompt, 12, 1);
        let mut m2 = SciAgentModel::new(&cfg);
        let out_b = gen.generate(&mut m2, &prompt, 12, 2);
        assert_ne!(out_a, out_b, "different seeds should sample differently");
    }

    fn logits_tape(vals: &[f32]) -> (Tape, usize) {
        let tape = Tape::new();
        let n = vals.len();
        let v = tape.input(Tensor::from_vec(vals.to_vec(), 1, n));
        let idx = v.idx();
        (tape, idx)
    }

    #[test]
    fn repetition_penalty_demotes_recent_tokens_even_in_greedy() {
        let (tape, idx) = logits_tape(&[2.0, 1.9, 0.5, 0.1]);
        let mut rng = 1u64;
        let mut p = SamplingParams::default();
        // No penalty: argmax is token 0.
        assert_eq!(sample_next(&tape, idx, 4, &p, &[0], &mut rng), 0);
        // Penalised (0 is in the window): 2.0/1.5 = 1.33 < 1.9 → token 1.
        p.repetition_penalty = 1.5;
        assert_eq!(sample_next(&tape, idx, 4, &p, &[0], &mut rng), 1);
        // Token 0 outside the window: unaffected.
        p.repetition_window = 2;
        let recent = vec![0, 2, 3];
        assert_eq!(sample_next(&tape, idx, 4, &p, &recent, &mut rng), 0);
    }

    #[test]
    fn top_k_one_is_argmax_at_any_temperature() {
        let (tape, idx) = logits_tape(&[0.5, 3.0, 0.4, 0.2]);
        let mut p = SamplingParams {
            temperature: 5.0,
            top_k: 1,
            ..SamplingParams::default()
        };
        p.top_p = 1.0;
        for seed in 1..20u64
        {
            let mut rng = seed_to_state(seed);
            assert_eq!(sample_next(&tape, idx, 4, &p, &[], &mut rng), 1);
        }
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // softmax([5,0,0,0]) puts ~0.98 on token 0: with top_p=0.5 the
        // nucleus is exactly {0}, so every draw returns 0.
        let (tape, idx) = logits_tape(&[5.0, 0.0, 0.0, 0.0]);
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            ..SamplingParams::default()
        };
        for seed in 1..20u64
        {
            let mut rng = seed_to_state(seed);
            assert_eq!(sample_next(&tape, idx, 4, &p, &[], &mut rng), 0);
        }
    }

    #[test]
    fn test_reset_kv_cache() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        for layer in &mut model.layers
        {
            *layer.attn.kv_cache.borrow_mut() = Some((
                Tensor::from_vec(vec![1.0], 1, 1),
                Tensor::from_vec(vec![1.0], 1, 1),
            ));
        }
        reset_kv_caches(&mut model);
        for layer in &model.layers
        {
            assert!(layer.attn.kv_cache.borrow().is_none());
        }
    }
}
