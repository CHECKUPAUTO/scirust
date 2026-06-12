//! SOM model: a Transformer encoder over linearized ownership-token
//! sequences, with per-token classification heads.
//!
//! The backbone reuses the framework's real attention stack
//! (`scirust_core::nn::transformer::TransformerEncoder`: multi-head
//! attention + feed-forward + layer norm), so this is an actual
//! transformer — not an MLP. Input programs are linearized by
//! `scirust-som-tokenizer`; graph-structured attention over the PCG is
//! future work and is documented as such in `scirust-som/README.md`.
//!
//! Heads (all per token):
//! - `ownership`: NA / Owned / Borrowed / Moved / Dropped
//! - `borrow`:    NA / None / Shared / Mut
//! - `invalid`:   fault probability (use-after-move, borrow conflict, …)
//!
//! Construction and forward are fully deterministic in `SomModelConfig::seed`.

use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::Module;
use scirust_core::nn::embedding::Embedding;
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::linear::Linear;
use scirust_core::nn::positional_encoding::PositionalEncoding;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::encoder::TransformerEncoder;
use scirust_core::tensor::tensor3d::Var3D;

#[derive(Debug, Clone)]
pub struct SomModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub n_ownership_classes: usize,
    pub n_borrow_classes: usize,
    pub seed: u64,
}

impl Default for SomModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 64,
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 64,
            max_seq_len: 64,
            n_ownership_classes: 5,
            n_borrow_classes: 4,
            seed: 42,
        }
    }
}

/// Per-token logits for one forward pass.
pub struct SomLogits<'t> {
    /// (seq_len, n_ownership_classes)
    pub ownership: Var<'t>,
    /// (seq_len, n_borrow_classes)
    pub borrow: Var<'t>,
    /// (seq_len, 1) — raw logit, apply sigmoid for a probability.
    pub invalid: Var<'t>,
}

pub struct SomModel {
    pub cfg: SomModelConfig,
    embed: Embedding,
    pos_enc: PositionalEncoding,
    encoder: TransformerEncoder,
    ownership_head: Linear,
    borrow_head: Linear,
    invalid_head: Linear,
}

impl SomModel {
    pub fn new(cfg: SomModelConfig) -> Self {
        assert!(
            cfg.d_model.is_multiple_of(cfg.n_heads),
            "n_heads must divide d_model"
        );
        let mut rng = PcgEngine::new(cfg.seed);
        let w_init = KaimingNormal;
        let b_init = Zeros;
        let embed =
            Embedding::new(cfg.vocab_size, cfg.d_model, &w_init, &mut rng).with_name("som_embed");
        let pos_enc = PositionalEncoding::new(cfg.d_model, cfg.max_seq_len);
        let encoder = TransformerEncoder::new(
            cfg.n_layers,
            cfg.d_model,
            cfg.n_heads,
            cfg.d_ff,
            false,
            &w_init,
            &b_init,
            &mut rng,
        );
        let ownership_head = Linear::new(
            cfg.d_model,
            cfg.n_ownership_classes,
            &w_init,
            &b_init,
            &mut rng,
        );
        let borrow_head = Linear::new(
            cfg.d_model,
            cfg.n_borrow_classes,
            &w_init,
            &b_init,
            &mut rng,
        );
        let invalid_head = Linear::new(cfg.d_model, 1, &w_init, &b_init, &mut rng);
        Self {
            cfg,
            embed,
            pos_enc,
            encoder,
            ownership_head,
            borrow_head,
            invalid_head,
        }
    }

    /// Forward over one token sequence. Logits are per token.
    pub fn forward<'t>(&mut self, tape: &'t Tape, token_ids: &[usize]) -> SomLogits<'t> {
        let seq_len = token_ids.len();
        assert!(seq_len >= 1, "empty token sequence");
        assert!(
            seq_len <= self.cfg.max_seq_len,
            "sequence length {} exceeds max_seq_len {}",
            seq_len,
            self.cfg.max_seq_len
        );

        let data: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let ids = tape.input(Tensor::from_vec(data, seq_len, 1));

        // (seq_len, 1) indices → (seq_len, d_model)
        let embedded = self.embed.forward(tape, ids);
        let positioned = self.pos_enc.forward(tape, embedded, seq_len);

        let x_3d = Var3D::from_var(positioned, 1, seq_len, self.cfg.d_model);
        let encoded = self.encoder.forward_3d(tape, x_3d);
        let hidden = encoded.var; // (seq_len, d_model), final LayerNorm applied

        SomLogits {
            ownership: self.ownership_head.forward(tape, hidden),
            borrow: self.borrow_head.forward(tape, hidden),
            invalid: self.invalid_head.forward(tape, hidden),
        }
    }

    /// Trainable parameter indices on the current tape (call after forward).
    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut idx = self.embed.parameter_indices();
        idx.extend(self.encoder.parameter_indices());
        idx.extend(self.ownership_head.parameter_indices());
        idx.extend(self.borrow_head.parameter_indices());
        idx.extend(self.invalid_head.parameter_indices());
        idx
    }

    /// Persist trained values from the tape back into the module tensors.
    pub fn sync(&mut self, tape: &Tape) {
        self.embed.sync(tape);
        self.encoder.sync(tape);
        self.ownership_head.sync(tape);
        self.borrow_head.sync(tape);
        self.invalid_head.sync(tape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> SomModelConfig {
        SomModelConfig {
            vocab_size: 63,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 32,
            ..SomModelConfig::default()
        }
    }

    #[test]
    fn forward_shapes() {
        let mut model = SomModel::new(tiny_cfg());
        let tape = Tape::new();
        let ids = vec![2, 8, 9, 10, 5];
        let logits = model.forward(&tape, &ids);
        assert_eq!(logits.ownership.shape(), (5, 5));
        assert_eq!(logits.borrow.shape(), (5, 4));
        assert_eq!(logits.invalid.shape(), (5, 1));
        assert!(!model.parameter_indices().is_empty());
    }

    #[test]
    fn forward_is_bit_deterministic_across_fresh_models() {
        let ids = vec![2, 8, 9, 10, 5, 4, 3];
        let run = || -> Vec<u32> {
            let mut model = SomModel::new(tiny_cfg());
            let tape = Tape::new();
            let logits = model.forward(&tape, &ids);
            let t = tape.value(logits.ownership.idx());
            t.data.iter().map(|f| f.to_bits()).collect()
        };
        assert_eq!(run(), run(), "same seed ⇒ bit-identical logits");
    }

    #[test]
    fn different_seed_changes_logits() {
        let ids = vec![2, 8, 9, 10];
        let bits = |seed: u64| -> Vec<u32> {
            let mut cfg = tiny_cfg();
            cfg.seed = seed;
            let mut model = SomModel::new(cfg);
            let tape = Tape::new();
            let logits = model.forward(&tape, &ids);
            tape.value(logits.ownership.idx())
                .data
                .iter()
                .map(|f| f.to_bits())
                .collect()
        };
        assert_ne!(bits(1), bits(2));
    }
}
