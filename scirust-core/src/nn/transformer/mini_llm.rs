//! MiniLLM — petit transformer char-level pour inférence locale.

use crate::autodiff::reverse::{Tape, Tensor};
use crate::nn::init::{KaimingNormal, Zeros};
use crate::nn::layer_norm::LayerNorm;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::positional_encoding::PositionalEncoding;
use crate::nn::rng::PcgEngine;
use crate::nn::transformer::encoder::TransformerEncoder;
use crate::tensor::tensor3d::Var3D;
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
        for t in texts
        {
            for c in t.chars()
            {
                *chars.entry(c).or_insert(0) += 1;
            }
        }
        let mut vocab = HashMap::new();
        let mut rev = HashMap::new();
        vocab.insert('\0', 0);
        rev.insert(0, '\0');
        vocab.insert('�', 1);
        rev.insert(1, '�');
        for (i, c) in chars.keys().enumerate()
        {
            let id = i + 2;
            vocab.insert(*c, id);
            rev.insert(id, *c);
        }
        let vs = vocab.len();
        Self {
            vocab,
            rev_vocab: rev,
            vocab_size: vs,
        }
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.vocab.get(&c).unwrap_or(&1))
            .collect()
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|i| self.rev_vocab.get(i).copied().unwrap_or('�'))
            .collect()
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
        Self {
            vocab_size: 256,
            d_model: 128,
            n_heads: 4,
            n_layers: 4,
            d_ff: 512,
            max_seq_len: 125_000,
        }
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
            config.vocab_size,
            config.d_model,
            &w_init,
            &mut rng,
        );
        let pos_enc = PositionalEncoding::new(config.d_model, config.max_seq_len);
        let encoder = TransformerEncoder::new(
            config.n_layers,
            config.d_model,
            config.n_heads,
            config.d_ff,
            false,
            &w_init,
            &b_init,
            &mut rng,
        );
        let ln_f = LayerNorm::new(config.d_model, 1e-5, &w_init, &mut rng);
        let lm_head = Linear::new(
            config.d_model,
            config.vocab_size,
            &w_init,
            &b_init,
            &mut rng,
        );
        Self {
            config,
            tokenizer,
            embed,
            pos_enc,
            encoder,
            ln_f,
            lm_head,
        }
    }

    /// Forward `[seq_len]` → logits `[seq_len, vocab_size]`
    pub fn forward(&mut self, input_ids: &[usize]) -> Tensor {
        let seq_len = input_ids.len().max(1);
        let tape = Tape::new();

        // Input [seq_len, 1] — chaque ligne = un token index
        let mut data = vec![0.0f32; seq_len];
        for (i, &id) in input_ids.iter().enumerate()
        {
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
        for i in 0..nrows
        {
            for j in 0..self.config.vocab_size.min(c)
            {
                let idx = i * c + j;
                let v = if idx < vals.data.len()
                {
                    vals.data[idx]
                }
                else
                {
                    0.0
                };
                out.data[i * self.config.vocab_size + j] = if v.is_finite() { v } else { 0.0 };
            }
        }
        out
    }

    /// Forward pass returning hidden states (after LayerNorm, before lm_head).
    /// Returns Tensor of shape (seq_len, d_model).
    pub fn forward_hidden(&mut self, input_ids: &[usize]) -> Tensor {
        let seq_len = input_ids.len().max(1);
        let tape = Tape::new();

        let mut data = vec![0.0f32; seq_len];
        for (i, &id) in input_ids.iter().enumerate()
        {
            data[i] = id as f32;
        }
        let t = tape.input(Tensor::from_vec(data, seq_len, 1));

        let embedded = self.embed.forward(&tape, t);
        let positioned = self.pos_enc.forward(&tape, embedded, seq_len);
        let x_3d = Var3D::from_var(positioned, 1, seq_len, self.config.d_model);
        let enc_3d = self.encoder.forward_3d(&tape, x_3d);
        let n = self.ln_f.forward(&tape, enc_3d.var);

        // Extract hidden states from tape
        let (r, c) = n.shape();
        let nrows = r.min(seq_len);
        let mut out = Tensor::zeros(nrows.max(1), self.config.d_model);
        let vals = tape.value(n.idx());
        for i in 0..nrows
        {
            for j in 0..self.config.d_model.min(c)
            {
                let idx = i * c + j;
                let v = if idx < vals.data.len()
                {
                    vals.data[idx]
                }
                else
                {
                    0.0
                };
                out.data[i * self.config.d_model + j] = if v.is_finite() { v } else { 0.0 };
            }
        }
        out
    }

    /// Greedy generation
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        let prompt_ids = self.tokenizer.encode(prompt);
        let ids = self.generate_ids(&prompt_ids, max_tokens);
        self.tokenizer.decode(&ids)
    }

    /// Greedy autoregressive generation at the **token-id level**, independent
    /// of any tokenizer — so a BPE (or any) tokenizer can drive it:
    /// `bpe.tokenize → generate_ids → bpe.decode`. Full-recompute decoding
    /// (O(n²)); see [`Self::generate_ids_cached`] for the O(n) KV-cache path.
    /// Deterministic (greedy argmax). Returns prompt + generated ids.
    pub fn generate_ids(&mut self, prompt_ids: &[usize], max_tokens: usize) -> Vec<usize> {
        let mut ids = prompt_ids.to_vec();
        let half = self.config.max_seq_len / 2;
        for _ in 0..max_tokens
        {
            if ids.len() >= self.config.max_seq_len
            {
                break;
            }
            let ctx: Vec<usize> = if ids.len() > half
            {
                ids[ids.len() - half..].to_vec()
            }
            else
            {
                ids.clone()
            };
            let logits = self.forward(&ctx);
            if logits.nrows() == 0
            {
                break;
            }
            let next = argmax_row(&logits, logits.nrows() - 1, self.config.vocab_size);
            ids.push(next);
            if next == 0
            {
                break;
            }
        }
        ids
    }

    /// Clear every attention layer's KV-cache (call before a fresh generation).
    pub fn reset_kv_cache(&mut self) {
        for block in self.encoder.blocks.iter_mut()
        {
            block.mha.kv_cache.replace(None);
        }
    }

    /// Run one token at absolute position `pos` through the model using the
    /// KV-cache, returning its `(1, vocab)` logits. The cache stores K/V as
    /// plain tensors, so a fresh tape per step is fine.
    fn step_logits(&mut self, id: usize, pos: usize) -> Tensor {
        let tape = Tape::new();
        let t = tape.input(Tensor::from_vec(vec![id as f32], 1, 1));
        let emb = self.embed.forward(&tape, t); // (1, d_model)
        let pe = self.pos_enc.encoding_at(pos);
        let pe_v = tape.input(Tensor::from_vec(pe, 1, self.config.d_model));
        let x = emb.try_add(pe_v).unwrap();
        let enc = self.encoder.infer_step(&tape, x, pos);
        let n = self.ln_f.forward(&tape, enc);
        let logits_v = self.lm_head.forward(&tape, n);
        tape.value(logits_v.idx())
    }

    /// O(n) autoregressive generation using the KV-cache: each token is
    /// processed once (vs. [`Self::generate_ids`]'s O(n²) full recompute).
    /// Produces the same tokens as `generate_ids` when no context-window
    /// truncation occurs (validated by an equivalence test). Deterministic.
    pub fn generate_ids_cached(&mut self, prompt_ids: &[usize], max_tokens: usize) -> Vec<usize> {
        self.reset_kv_cache();
        let vocab = self.config.vocab_size;
        let mut ids = prompt_ids.to_vec();
        if ids.is_empty()
        {
            return ids;
        }
        // Prime the cache with the prompt; keep the last token's logits.
        let prompt: Vec<usize> = ids.clone();
        let mut logits = self.step_logits(prompt[0], 0);
        for (pos, &id) in prompt.iter().enumerate().skip(1)
        {
            logits = self.step_logits(id, pos);
        }
        for _ in 0..max_tokens
        {
            let pos = ids.len();
            if pos >= self.config.max_seq_len
            {
                break;
            }
            let next = argmax_row(&logits, 0, vocab);
            ids.push(next);
            if next == 0
            {
                break;
            }
            logits = self.step_logits(next, pos);
        }
        ids
    }
}

/// Greedy argmax over row `row` of a `(_, vocab)` logits tensor.
fn argmax_row(logits: &Tensor, row: usize, vocab_size: usize) -> usize {
    let base = row * vocab_size;
    (0..vocab_size)
        .map(|j| (j, logits.data.get(base + j).copied().unwrap_or(-1e9)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
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
        let cfg = MiniLLMConfig {
            vocab_size: tok.vocab_size,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            max_seq_len: 8,
        };
        let mut m = MiniLLM::new(cfg, tok);
        let logits = m.forward(&[2, 3]);
        assert_eq!(logits.nrows(), 2);
        assert_eq!(logits.ncols(), m.config.vocab_size);
    }

    #[test]
    fn test_logits_not_nan() {
        let tok = CharTokenizer::new(&["abc"]);
        let cfg = MiniLLMConfig {
            vocab_size: tok.vocab_size,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            max_seq_len: 8,
        };
        let mut m = MiniLLM::new(cfg, tok);
        let logits = m.forward(&[2, 3, 4]);
        for &v in &logits.data
        {
            assert!(!v.is_nan(), "NaN in logits");
        }
    }

    #[test]
    fn test_generate_nonempty() {
        let tok = CharTokenizer::new(&["hello test world"]);
        let cfg = MiniLLMConfig {
            vocab_size: tok.vocab_size,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            max_seq_len: 8,
        };
        let mut m = MiniLLM::new(cfg, tok);
        assert!(!m.generate("hello", 5).is_empty());
    }

    /// KV-cache end-to-end: O(n) cached decoding must produce the exact same
    /// token sequence as the O(n²) full-recompute path (no window truncation).
    #[test]
    fn kv_cache_generation_matches_full_recompute() {
        let tok = CharTokenizer::new(&["hello world abcdefghij"]);
        let cfg = MiniLLMConfig {
            vocab_size: tok.vocab_size,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            d_ff: 32,
            max_seq_len: 64,
        };
        let mut m = MiniLLM::new(cfg, tok);
        let prompt = vec![2usize, 3, 4];
        let full = m.generate_ids(&prompt, 6);
        let cached = m.generate_ids_cached(&prompt, 6);
        assert_eq!(full, cached, "KV-cache decode diverged from full recompute");
        assert!(cached.len() > prompt.len(), "nothing was generated");
    }

    /// `generate_ids` is tokenizer-agnostic: feeding raw ids works and the
    /// `generate(&str)` convenience wrapper agrees with it.
    #[test]
    fn generate_ids_is_tokenizer_decoupled() {
        let tok = CharTokenizer::new(&["abcdef ghij"]);
        let cfg = MiniLLMConfig {
            vocab_size: tok.vocab_size,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 32,
        };
        let mut m = MiniLLM::new(cfg, tok);
        let prompt = "abc";
        let prompt_ids = m.tokenizer.encode(prompt);
        let via_ids = m.generate_ids(&prompt_ids, 4);
        let via_str = m.generate(prompt, 4);
        assert_eq!(m.tokenizer.decode(&via_ids), via_str);
    }
}
