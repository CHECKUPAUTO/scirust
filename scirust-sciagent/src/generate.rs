use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::Module;

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;

pub struct Generator {
    config: SciAgentConfig,
    /// Softmax temperature. `<= 0.0` means greedy argmax; anything positive
    /// samples from `softmax(logits / temperature)` with a deterministic
    /// xorshift stream seeded by the caller, so a given (prompt, seed,
    /// temperature) triple always reproduces the same output.
    temperature: f32,
}

impl Generator {
    pub fn new(config: &SciAgentConfig) -> Self {
        Self {
            config: config.clone(),
            temperature: 0.0,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
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
            let next = sample_or_argmax(
                &tape,
                logits.idx(),
                self.config.vocab_size,
                self.temperature,
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

        let last_idx = sample_or_argmax(
            &tape,
            last_logits.idx(),
            self.config.vocab_size,
            self.temperature,
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

            let next = sample_or_argmax(
                &tape,
                logits.idx(),
                self.config.vocab_size,
                self.temperature,
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

fn sample_or_argmax(
    tape: &Tape,
    logits_idx: usize,
    vocab: usize,
    temperature: f32,
    rng: &mut u64,
) -> usize {
    let t = tape.value(logits_idx);
    let row_start = t.data.len() - vocab;
    let row = &t.data[row_start..row_start + vocab];

    if temperature <= 0.0
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

    // softmax(logits / T), max-subtracted for numerical stability, then a
    // single uniform draw walked through the CDF.
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = row
        .iter()
        .map(|&v| ((v - max) / temperature).exp())
        .collect();
    let z: f32 = probs.iter().sum();
    for p in probs.iter_mut()
    {
        *p /= z;
    }

    let u = next_uniform(rng);
    let mut acc = 0.0f32;
    for (j, &p) in probs.iter().enumerate()
    {
        acc += p;
        if u < acc
        {
            return j;
        }
    }
    vocab - 1 // float round-off: the CDF summed to slightly under 1.0
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
