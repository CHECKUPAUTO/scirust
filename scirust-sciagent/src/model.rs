use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::embedding::Embedding;
use scirust_core::nn::init::{Initializer, KaimingNormal};
use scirust_core::nn::linear::Linear;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use std::collections::HashMap;

use crate::block::SciAgentBlock;
use crate::config::SciAgentConfig;
use crate::norm::RMSNorm;
use crate::tokenizer::SciAgentTokenizer;

pub struct SciAgentModel {
    pub config: SciAgentConfig,
    pub embed: Embedding,
    pub layers: Vec<SciAgentBlock>,
    pub rms_final: RMSNorm,
    pub lm_head: Option<Linear>,
    tokenizer: Option<SciAgentTokenizer>,
}

impl SciAgentModel {
    pub fn new(config: &SciAgentConfig) -> Self {
        let init = KaimingNormal;
        let mut rng = PcgEngine::new(42);
        Self::new_with_rng(config, &init, &mut rng)
    }

    pub fn new_with_rng<I: Initializer>(
        config: &SciAgentConfig,
        init: &I,
        rng: &mut PcgEngine,
    ) -> Self {
        let embed = Embedding::new(config.vocab_size, config.d_model, init, rng)
            .with_name("sciagent.embed");
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(SciAgentBlock::new(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.d_ff,
                config.rope_theta,
                config.eps,
                init,
                rng,
                &format!("sciagent.layer{i}"),
            ));
        }
        let rms_final =
            RMSNorm::new(config.d_model, config.eps, init, rng).with_name("sciagent.rms_final");
        let lm_head = if config.tie_embeddings {
            None
        } else {
            let z = scirust_core::nn::init::Zeros;
            Some(Linear::new(config.d_model, config.vocab_size, init, &z, rng))
        };
        Self {
            config: config.clone(),
            embed,
            layers,
            rms_final,
            lm_head,
            tokenizer: None,
        }
    }

    pub fn set_tokenizer(&mut self, tokenizer: SciAgentTokenizer) {
        self.tokenizer = Some(tokenizer);
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, input_ids: &[usize], seq_len: usize) -> Var<'t> {
        let total_tokens = input_ids.len();
        assert_eq!(total_tokens % seq_len, 0);

        let data: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
        let n = data.len();
        let idx_t = tape.input(Tensor::from_vec(data, n, 1));
        let h = self.embed.forward(tape, idx_t);

        let mut h = h;
        for layer in &mut self.layers {
            h = layer.forward(tape, h, seq_len);
        }
        h = self.rms_final.forward(tape, h);

        match self.lm_head.as_mut() {
            Some(head) => head.forward(tape, h),
            None => {
                let w = tape.input(self.embed.weight.clone());
                h.try_matmul(w.transpose_2d()).unwrap()
            }
        }
    }

    pub fn generate(&mut self, prompt: &[usize], max_tokens: usize) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        for _ in 0..max_tokens {
            let tape = Tape::new();
            let logits = self.forward(&tape, &ids, ids.len());
            let next = argmax_last(&tape, logits.idx(), self.config.vocab_size);
            ids.push(next);
            if next == 0 {
                break;
            }
        }
        ids
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.embed.parameter_indices());
        for layer in &self.layers {
            v.extend(layer.parameter_indices());
        }
        v.extend(self.rms_final.parameter_indices());
        if let Some(ref head) = self.lm_head {
            v.extend(head.parameter_indices());
        }
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.embed.sync(tape);
        for layer in &mut self.layers {
            layer.sync(tape);
        }
        self.rms_final.sync(tape);
        if let Some(ref mut head) = self.lm_head {
            head.sync(tape);
        }
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(self.embed.state_dict());
        for layer in &self.layers {
            map.extend(layer.state_dict());
        }
        map.extend(self.rms_final.state_dict());
        if let Some(ref head) = self.lm_head {
            map.extend(head.state_dict());
        }
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> scirust_core::error::Result<()> {
        self.embed.load_state_dict(sd)?;
        for layer in &mut self.layers {
            layer.load_state_dict(sd)?;
        }
        self.rms_final.load_state_dict(sd)?;
        if let Some(ref mut head) = self.lm_head {
            head.load_state_dict(sd)?;
        }
        Ok(())
    }
}

fn argmax_last(tape: &Tape, logits_idx: usize, vocab: usize) -> usize {
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
