//! A minimal **causal decoder language model** on the N-D autograd tape — the
//! end-to-end milestone: token embedding + learned positional embedding → a
//! stack of causal Pre-LN transformer blocks → final LayerNorm → a linear
//! language-model head to vocabulary logits, trained with next-token
//! cross-entropy.
//!
//! Every component is built from the gradient-checked N-D ops
//! ([`crate::autodiff::nd`]) and the reusable layers
//! ([`crate::nn::nd_layers`]); parameter init is seeded ([`PcgEngine`]), so a
//! run is bit-for-bit reproducible. The test overfits a fixed sequence and the
//! model then predicts it exactly — proof the whole stack (embeddings,
//! causal attention, residual blocks, head, cross-entropy) trains correctly.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::nd_layers::{NdEmbedding, NdLayerNorm, NdLinear, NdTransformerBlock};
use crate::nn::nd_optim::NdParam;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;

/// Hyper-parameters of an [`NdDecoderLM`].
#[derive(Clone, Copy, Debug)]
pub struct NdDecoderConfig {
    /// Vocabulary size.
    pub vocab: usize,
    /// Model (embedding) width.
    pub d_model: usize,
    /// Attention heads (`d_model` must divide by this).
    pub n_heads: usize,
    /// Feed-forward hidden width.
    pub d_ff: usize,
    /// Number of stacked transformer blocks.
    pub n_layers: usize,
    /// Maximum sequence length (size of the positional table).
    pub max_seq: usize,
}

/// A GPT-style causal decoder language model on the N-D tape.
pub struct NdDecoderLM {
    tok: NdEmbedding, // (vocab, d_model)
    pos: NdEmbedding, // (max_seq, d_model)
    blocks: Vec<NdTransformerBlock>,
    ln_f: NdLayerNorm,
    head: NdLinear, // (d_model, vocab)
    max_seq: usize,
    vocab: usize,
}

impl NdDecoderLM {
    /// Build a model from a config with seeded init. Every transformer block
    /// uses **causal** attention.
    pub fn new(cfg: NdDecoderConfig, rng: &mut PcgEngine) -> Self {
        assert!(
            cfg.vocab > 0 && cfg.d_model > 0 && cfg.max_seq > 0,
            "empty config"
        );
        let blocks = (0..cfg.n_layers)
            .map(|_| NdTransformerBlock::new(cfg.d_model, cfg.n_heads, cfg.d_ff, true, rng))
            .collect();
        Self {
            tok: NdEmbedding::new(cfg.vocab, cfg.d_model, rng),
            pos: NdEmbedding::new(cfg.max_seq, cfg.d_model, rng),
            blocks,
            ln_f: NdLayerNorm::new(cfg.d_model, 1e-5),
            head: NdLinear::new(cfg.d_model, cfg.vocab, rng),
            max_seq: cfg.max_seq,
            vocab: cfg.vocab,
        }
    }

    /// Forward a token sequence → `(seq, vocab)` next-token logits.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, tokens: &[usize]) -> NdVar<'t> {
        let seq = tokens.len();
        assert!(seq > 0, "empty sequence");
        assert!(
            seq <= self.max_seq,
            "sequence ({seq}) longer than max_seq ({})",
            self.max_seq
        );

        let positions: Vec<usize> = (0..seq).collect();
        let tok = self.tok.forward(tape, tokens);
        let pos = self.pos.forward(tape, &positions);
        let mut x = tok.add(pos); // (seq, d_model)
        for b in &mut self.blocks
        {
            x = b.forward(tape, x);
        }
        let x = self.ln_f.forward(tape, x);
        self.head.forward(tape, x) // (seq, vocab)
    }

    /// Next-token cross-entropy: feed `tokens[..n-1]`, predict `tokens[1..]`.
    /// Returns the scalar mean loss. Needs `tokens.len() >= 2`.
    pub fn loss<'t>(&mut self, tape: &'t NdTape, tokens: &[usize]) -> NdVar<'t> {
        let seq = tokens.len();
        assert!(seq >= 2, "next-token loss needs at least 2 tokens");
        let logits = self.forward(tape, &tokens[..seq - 1]); // (seq-1, vocab)
        logits.cross_entropy(&tokens[1..])
    }

    /// SGD-update every parameter from a `backward` result (must follow a
    /// forward/loss on the same tape).
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.tok.sgd_step(grads, lr);
        self.pos.sgd_step(grads, lr);
        for b in &mut self.blocks
        {
            b.sgd_step(grads, lr);
        }
        self.ln_f.sgd_step(grads, lr);
        self.head.sgd_step(grads, lr);
    }

    /// Every trainable parameter, in a fixed order (token + positional
    /// embeddings, each block, final LayerNorm, LM head), paired with its
    /// gradient index — feed to [`NdAdam`](crate::nn::nd_optim::NdAdam). Call
    /// after a forward/loss on the tape being differentiated.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.tok.parameters();
        params.extend(self.pos.parameters());
        for b in &mut self.blocks
        {
            params.extend(b.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.head.parameters());
        params
    }

    /// Greedy next-token prediction at each position: returns the argmax of
    /// every logit row for `tokens` (length `tokens.len()`).
    pub fn predict(&mut self, tape: &NdTape, tokens: &[usize]) -> Vec<usize> {
        let logits = self.forward(tape, tokens);
        let lv = tape.value(logits);
        let (rows, vocab) = (lv.shape[0], lv.shape[1]);
        debug_assert_eq!(vocab, self.vocab);
        (0..rows)
            .map(|r| {
                let row = &lv.data[r * vocab..(r + 1) * vocab];
                let mut best = 0usize;
                for (c, &v) in row.iter().enumerate()
                {
                    if v > row[best]
                    {
                        best = c;
                    }
                }
                best
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> NdDecoderConfig {
        NdDecoderConfig {
            vocab: 6,
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_layers: 2,
            max_seq: 8,
        }
    }

    /// The whole decoder LM **overfits** a fixed sequence: full-batch SGD drives
    /// the next-token loss far below its (≈ ln vocab) start, and the trained
    /// model then **predicts the sequence exactly** (argmax at every position
    /// equals the shifted target). End-to-end proof of embeddings + learned
    /// positions + causal blocks + head + cross-entropy.
    #[test]
    fn nd_decoder_lm_overfits_a_sequence() {
        let mut rng = PcgEngine::new(123);
        let mut lm = NdDecoderLM::new(tiny_cfg(), &mut rng);
        let seq = [1usize, 2, 3, 4, 2, 5];

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..300
        {
            let t = NdTape::new();
            let loss_v = lm.loss(&t, &seq);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            lm.sgd_step(&grads, 0.05);
        }

        // Started near a uniform-distribution loss (ln 6 ≈ 1.79) and collapsed.
        assert!(first > 1.0, "unexpected initial loss {first}");
        assert!(
            last < first * 0.2,
            "decoder LM did not overfit: first {first}, last {last}"
        );

        // The model now reproduces the sequence: argmax(logits[i]) == seq[i+1].
        let t = NdTape::new();
        let preds = lm.predict(&t, &seq[..seq.len() - 1]);
        assert_eq!(
            preds,
            seq[1..].to_vec(),
            "trained model does not predict its training sequence"
        );
    }

    /// Forward shape and determinism: two fresh models with the same seed
    /// produce bit-identical logits.
    #[test]
    fn nd_decoder_lm_forward_is_deterministic() {
        let tokens = [0usize, 3, 1, 5, 2];
        let run = || -> TensorND {
            let mut rng = PcgEngine::new(42);
            let mut lm = NdDecoderLM::new(tiny_cfg(), &mut rng);
            let t = NdTape::new();
            let out = lm.forward(&t, &tokens);
            t.value(out)
        };
        let a = run();
        let b = run();
        assert_eq!(a.shape, vec![tokens.len(), 6]);
        assert_eq!(a.data, b.data);
    }

    /// The decoder LM trains with the **N-D Adam** optimizer (via
    /// `parameters()`): in far fewer steps than the SGD test it drives the loss
    /// below 10% of its start and predicts the sequence exactly. Proves Adam is
    /// wired through every layer of the model.
    #[test]
    fn nd_decoder_lm_trains_with_adam() {
        use crate::nn::nd_optim::NdAdam;

        let mut rng = PcgEngine::new(123);
        let mut lm = NdDecoderLM::new(tiny_cfg(), &mut rng);
        let seq = [1usize, 2, 3, 4, 2, 5];
        let mut opt = NdAdam::with_lr(0.01);

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..150
        {
            let t = NdTape::new();
            let loss_v = lm.loss(&t, &seq);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            let mut params = lm.parameters();
            opt.step(&mut params, &grads);
        }

        assert!(
            last < first * 0.1,
            "Adam did not train the decoder: first {first}, last {last}"
        );
        assert_eq!(opt.step_count(), 150);

        let t = NdTape::new();
        let preds = lm.predict(&t, &seq[..seq.len() - 1]);
        assert_eq!(preds, seq[1..].to_vec(), "Adam-trained model mispredicts");
    }
}
