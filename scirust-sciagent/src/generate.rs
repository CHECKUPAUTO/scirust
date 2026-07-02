use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::Module;

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;

pub struct Generator {
    config: SciAgentConfig,
}

impl Generator {
    pub fn new(config: &SciAgentConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn generate(
        &self,
        model: &mut SciAgentModel,
        prompt: &[usize],
        max_tokens: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut _rng_seed = seed;
        let mut ids = prompt.to_vec();
        let max_seq_len = self.config.max_seq_len;

        for _ in 0..max_tokens {
            let tape = Tape::new();
            let seq_len = ids.len();

            let input_slice = if seq_len > max_seq_len {
                &ids[seq_len - max_seq_len..]
            } else {
                &ids
            };

            let local_seq = input_slice.len();
            let logits = model.forward(&tape, input_slice, local_seq);
            let next = sample_or_argmax(&tape, logits.idx(), self.config.vocab_size, &mut _rng_seed);
            ids.push(next);
            if next == 0 {
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
        let mut _rng_seed = seed;
        let tape = Tape::new();
        let data: Vec<f32> = prompt.iter().map(|&id| id as f32).collect();
        let n = data.len();
        let idx_t = tape.input(Tensor::from_vec(data, n, 1));
        let mut h = model.embed.forward(&tape, idx_t);

        for layer in &mut model.layers {
            h = layer.forward(&tape, h, n);
        }
        h = model.rms_final.forward(&tape, h);

        let last_logits = match model.lm_head.as_mut() {
            Some(head) => head.forward(&tape, h),
            None => {
                let w = tape.input(model.embed.weight.clone());
                h.try_matmul(w.transpose_2d()).unwrap()
            }
        };

        let last_idx = {
            let t = tape.value(last_logits.idx());
            let vocab = self.config.vocab_size;
            let start = t.data.len() - vocab;
            let mut best = 0usize;
            let mut bv = t.data[start];
            for j in 1..vocab {
                let v = t.data[start + j];
                if v > bv {
                    bv = v;
                    best = j;
                }
            }
            best
        };

        let mut output = prompt.to_vec();
        output.push(last_idx);

        reset_kv_caches(model);

        for _ in 1..max_tokens {
            let tape = Tape::new();
            let token_data = vec![output.last().copied().unwrap_or(0) as f32];
            let token_v = tape.input(Tensor::from_vec(token_data, 1, 1));
            let h = model.embed.forward(&tape, token_v);
            let mut h = h;

            for layer in &mut model.layers {
                h = layer.infer_step(&tape, h, output.len() - 1);
            }
            h = model.rms_final.forward(&tape, h);

            let logits = match model.lm_head.as_mut() {
                Some(head) => head.forward(&tape, h),
                None => {
                    let w = tape.input(model.embed.weight.clone());
                    h.try_matmul(w.transpose_2d()).unwrap()
                }
            };

            let next = sample_or_argmax(&tape, logits.idx(), self.config.vocab_size, &mut _rng_seed);
            output.push(next);
            if next == 0 {
                break;
            }
        }
        output
    }
}

fn sample_or_argmax(tape: &Tape, logits_idx: usize, vocab: usize, _rng: &mut u64) -> usize {
    let t = tape.value(logits_idx);
    let row_start = t.data.len() - vocab;
    let mut best = 0usize;
    let mut best_val = t.data[row_start];
    for j in 1..vocab {
        let v = t.data[row_start + j];
        if v > best_val {
            best_val = v;
            best = j;
        }
    }
    best
}

fn reset_kv_caches(model: &mut SciAgentModel) {
    for layer in &mut model.layers {
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
        assert!(out.len() > prompt.len(), "Should produce more tokens than prompt");
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
        for layer in &mut model.layers {
            h = layer.forward(&tape2, h, 4);
        }
        h = model.rms_final.forward(&tape2, h);
        let logits_partial = match model.lm_head.as_mut() {
            Some(head) => head.forward(&tape2, h),
            None => {
                let w = tape2.input(model.embed.weight.clone());
                h.try_matmul(w.transpose_2d()).unwrap()
            }
        };
        let v_partial = tape2.value(logits_partial.idx()).data.clone();
        assert_eq!(v_full, v_partial, "Manual forward should match model.forward");
    }

    #[test]
    fn test_reset_kv_cache() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        for layer in &mut model.layers {
            *layer.attn.kv_cache.borrow_mut() = Some((
                Tensor::from_vec(vec![1.0], 1, 1),
                Tensor::from_vec(vec![1.0], 1, 1),
            ));
        }
        reset_kv_caches(&mut model);
        for layer in &model.layers {
            assert!(layer.attn.kv_cache.borrow().is_none());
        }
    }
}
