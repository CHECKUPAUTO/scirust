//! MiniLLM — petit transformer char-level pour inférence locale.

use crate::autodiff::reverse::{Tape, Tensor};
use crate::nn::init::{KaimingNormal, Zeros};
use crate::nn::layer_norm::LayerNorm;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::positional_encoding::PositionalEncoding;
use crate::nn::rng::PcgEngine;
use crate::nn::transformer::encoder::TransformerEncoder;
use crate::tensor::tensor3d::{Tensor3D, Var3D};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    vocab: HashMap<char, usize>,
    rev_vocab: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl CharTokenizer {
    pub fn new(texts: &[&str]) -> Self {
        let mut chars = BTreeMap::new();
        for t in texts { for c in t.chars() { *chars.entry(c).or_insert(0) += 1; } }
        let mut vocab = HashMap::new();
        let mut rev = HashMap::new();
        vocab.insert('\0', 0); rev.insert(0, '\0');
        vocab.insert('�', 1); rev.insert(1, '�');
        for (i, c) in chars.keys().enumerate() {
            let id = i + 2; vocab.insert(*c, id); rev.insert(id, *c);
        }
        let vs = vocab.len();
        Self { vocab, rev_vocab: rev, vocab_size: vs }
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.vocab.get(&c).unwrap_or(&1)).collect()
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|i| self.rev_vocab.get(i).copied().unwrap_or('�')).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MiniLLMConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
}

impl Default for MiniLLMConfig {
    fn default() -> Self {
        Self { vocab_size: 256, d_model: 128, n_heads: 4, n_layers: 4, d_ff: 512, max_seq_len: 256 }
    }
}

pub struct MiniLLM {
    pub config: MiniLLMConfig,
    pub tokenizer: CharTokenizer,
    embed: crate::nn::embedding::Embedding,
    pos_enc: PositionalEncoding,
    encoder: TransformerEncoder,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl MiniLLM {
    pub fn new(config: MiniLLMConfig, tokenizer: CharTokenizer) -> Self {
        let mut rng = PcgEngine::new(42);
        let w_init = KaimingNormal;
        let b_init = Zeros;
        let embed = crate::nn::embedding::Embedding::new(
            config.vocab_size, config.d_model, &w_init, &mut rng);
        let pos_enc = PositionalEncoding::new(config.d_model, config.max_seq_len);
        let encoder = TransformerEncoder::new(
            config.n_layers, config.d_model, config.n_heads, config.d_ff,
            false, &w_init, &b_init, &mut rng);
        let ln_f = LayerNorm::new(config.d_model, 1e-5, &w_init, &mut rng);
        let lm_head = Linear::new(config.d_model, config.vocab_size, &w_init, &b_init, &mut rng);
        Self { config, tokenizer, embed, pos_enc, encoder, ln_f, lm_head }
    }

    /// Forward [seq_len] → logits [seq_len, vocab_size]
    pub fn forward(&mut self, input_ids: &[usize]) -> Tensor {
        let seq_len = input_ids.len().max(1);
        let tape = Tape::new();

        // Input [seq_len, 1] — chaque ligne = un token index
        let mut data = vec![0.0f32; seq_len];
        for (i, &id) in input_ids.iter().enumerate() {
            data[i] = id as f32;
        }
        let t = tape.input(Tensor::from_vec(data, seq_len, 1));

        // Embedding [seq_len, 1] → [seq_len, d_model] (via tape reshape)
        let embedded = self.embed.forward(&tape, t);
        // embedded shape = (seq_len, d_model) after embedding on each token
        // The embedding returns shape (seq_len, d_model) since it treated seq_len indices

        // PosEnc expects (seq_len, d_model)
        let positioned = self.pos_enc.forward(&tape, embedded, seq_len);

        // Encoder expects Var3D(1, seq_len, d_model)
        let x_3d = Var3D::from_var(positioned, 1, seq_len, self.config.d_model);
        let enc_3d = self.encoder.forward_3d(&tape, x_3d);

        // LN → LM Head
        let n = self.ln_f.forward(&tape, enc_3d.var);
        let logits_var = self.lm_head.forward(&tape, n);

        // Extract via tape.value
        let (r, c) = logits_var.shape();
        let nrows = r.min(seq_len);
        let mut out = Tensor::zeros(nrows.max(1), self.config.vocab_size);
        let vals = tape.value(logits_var.idx());
        for i in 0..nrows {
            for j in 0..self.config.vocab_size.min(c) {
                let idx = i * c + j;
                let v = if idx < vals.data.len() { vals.data[idx] } else { 0.0 };
                out.data[i * self.config.vocab_size + j] = if v.is_finite() { v } else { 0.0 };
            }
        }
        out
    }

    /// Greedy generation
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        let mut ids = self.tokenizer.encode(prompt);
        let half = self.config.max_seq_len / 2;
        for _ in 0..max_tokens {
            if ids.len() >= self.config.max_seq_len { break; }
            let ctx: Vec<usize> = if ids.len() > half {
                ids[ids.len() - half..].to_vec()
            } else { ids.clone() };
            let logits = self.forward(&ctx);
            if logits.nrows() == 0 { break; }
            let last = logits.nrows() - 1;
            let base = last * self.config.vocab_size;
            let next = (0..self.config.vocab_size)
                .map(|j| (j, logits.data.get(base + j).copied().unwrap_or(-1e9)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
            ids.push(next);
            if next == 0 { break; }
        }
        self.tokenizer.decode(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        let tok = CharTokenizer::new(&["hello test"]);
        assert_eq!(tok.decode(&tok.encode("hello test")), "hello test");
    }

    #[test]
    fn test_forward_shape() {
        let tok = CharTokenizer::new(&["abc"]);
        let cfg = MiniLLMConfig { vocab_size: tok.vocab_size, d_model: 8, n_heads: 2, n_layers: 1, d_ff: 16, max_seq_len: 8 };
        let mut m = MiniLLM::new(cfg, tok);
        let logits = m.forward(&[2, 3]);
        assert_eq!(logits.nrows(), 2);
        assert_eq!(logits.ncols(), m.config.vocab_size);
    }

    #[test]
    fn test_logits_not_nan() {
        let tok = CharTokenizer::new(&["abc"]);
        let cfg = MiniLLMConfig { vocab_size: tok.vocab_size, d_model: 8, n_heads: 2, n_layers: 1, d_ff: 16, max_seq_len: 8 };
        let mut m = MiniLLM::new(cfg, tok);
        let logits = m.forward(&[2, 3, 4]);
        for &v in &logits.data { assert!(!v.is_nan(), "NaN in logits"); }
    }

    #[test]
    fn test_generate_nonempty() {
        let tok = CharTokenizer::new(&["hello test world"]);
        let cfg = MiniLLMConfig { vocab_size: tok.vocab_size, d_model: 8, n_heads: 2, n_layers: 1, d_ff: 16, max_seq_len: 8 };
        let mut m = MiniLLM::new(cfg, tok);
        assert!(!m.generate("hello", 5).is_empty());
    }
}
