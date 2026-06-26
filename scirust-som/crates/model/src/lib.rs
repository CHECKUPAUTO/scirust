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

/// Decoded per-token predictions: the discrete labels the toolchain consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SomLabels {
    /// Argmax class id over the ownership head, one per token.
    pub ownership: Vec<usize>,
    /// Argmax class id over the borrow head, one per token.
    pub borrow: Vec<usize>,
    /// Fault prediction per token: `true` iff the invalid logit is positive
    /// (equivalently `sigmoid(logit) > 0.5`).
    pub invalid: Vec<bool>,
}

/// First-maximum argmax over a contiguous slice, matching the tie-breaking
/// convention used elsewhere in `scirust-core` (strict `>` ⇒ lowest index wins
/// on ties). `row` is never empty here: every head has at least one output
/// column, enforced at construction by the per-class-count fields.
fn argmax_row(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (c, &v) in row.iter().enumerate()
    {
        if v > row[best]
        {
            best = c;
        }
    }
    best
}

impl<'t> SomLogits<'t> {
    /// Decode the logits into discrete per-token labels.
    ///
    /// - `ownership[i]` / `borrow[i]` are the argmax class ids for token `i`
    ///   over the respective head's class dimension.
    /// - `invalid[i]` is the fault decision for token `i` (`logit > 0`).
    ///
    /// Deterministic in the logits; reads values straight off the tape the
    /// `Var`s were produced on.
    pub fn decode(&self) -> SomLabels {
        let tape = self.ownership.tape();
        let own_t = tape.value(self.ownership.idx());
        let bor_t = tape.value(self.borrow.idx());
        let inv_t = tape.value(self.invalid.idx());

        let (seq_len, n_own) = own_t.shape();
        let n_bor = bor_t.cols;

        let ownership = (0..seq_len)
            .map(|i| argmax_row(&own_t.data[i * n_own..(i + 1) * n_own]))
            .collect();
        let borrow = (0..seq_len)
            .map(|i| argmax_row(&bor_t.data[i * n_bor..(i + 1) * n_bor]))
            .collect();
        let invalid = (0..seq_len).map(|i| inv_t.data[i] > 0.0).collect();

        SomLabels {
            ownership,
            borrow,
            invalid,
        }
    }
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

    /// Total number of trainable *scalar* parameters, summed over every
    /// parameter tensor registered on `tape`. Call after `forward` on the same
    /// tape, so the parameter indices are populated.
    pub fn parameter_count(&self, tape: &Tape) -> usize {
        self.parameter_indices()
            .into_iter()
            .map(|i| {
                let (rows, cols) = tape.shape(i);
                rows * cols
            })
            .sum()
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

    // ----- Oracle tests with hand-derived expectations ---------------------

    /// The smallest non-degenerate config, used for hand-computed param counts
    /// and the all-zeros decode oracle.
    /// V=6, D=4, H=2, L=1, F=8, O=3, B=2, max_seq_len=8.
    fn oracle_cfg() -> SomModelConfig {
        SomModelConfig {
            vocab_size: 6,
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ff: 8,
            max_seq_len: 8,
            n_ownership_classes: 3,
            n_borrow_classes: 2,
            seed: 7,
        }
    }

    /// Overwrite every trainable tensor in the model with zeros of the same
    /// shape. After this the whole network is the zero map (see
    /// `decode_with_zeros_is_all_class_zero` for the hand derivation).
    fn zero_all_params(model: &mut SomModel) {
        let zero_like = |t: &Tensor| Tensor::zeros(t.rows, t.cols);
        model.embed.weight = zero_like(&model.embed.weight);
        for blk in model.encoder.blocks.iter_mut()
        {
            blk.ln1.gamma = zero_like(&blk.ln1.gamma);
            blk.ln1.beta = zero_like(&blk.ln1.beta);
            blk.ln2.gamma = zero_like(&blk.ln2.gamma);
            blk.ln2.beta = zero_like(&blk.ln2.beta);
            for lin in [
                &mut blk.mha.w_q,
                &mut blk.mha.w_k,
                &mut blk.mha.w_v,
                &mut blk.mha.w_o,
                &mut blk.ffn1,
                &mut blk.ffn2,
            ]
            {
                lin.weight = Tensor::zeros(lin.weight.rows, lin.weight.cols);
                lin.bias = Tensor::zeros(lin.bias.rows, lin.bias.cols);
            }
        }
        model.encoder.final_ln.gamma = zero_like(&model.encoder.final_ln.gamma);
        model.encoder.final_ln.beta = zero_like(&model.encoder.final_ln.beta);
        for head in [
            &mut model.ownership_head,
            &mut model.borrow_head,
            &mut model.invalid_head,
        ]
        {
            head.weight = Tensor::zeros(head.weight.rows, head.weight.cols);
            head.bias = Tensor::zeros(head.bias.rows, head.bias.cols);
        }
    }

    /// Parameter-*tensor* count, derived by hand from the architecture:
    ///   embed                      = 1
    ///   per TransformerBlock       = ln1(2) + mha(4 linears × 2) + ln2(2)
    ///                                + ffn1(2) + ffn2(2) = 16
    ///   encoder (L blocks + final_ln) = 16·L + 2
    ///   three heads                = 3 × 2 = 6
    ///   total                      = 16·L + 9
    /// For L = 1 → 25 distinct parameter tensors.
    #[test]
    fn parameter_tensor_count_matches_architecture() {
        let mut model = SomModel::new(oracle_cfg());
        let tape = Tape::new();
        let _ = model.forward(&tape, &[1, 2, 3]);
        assert_eq!(model.parameter_indices().len(), 25);
    }

    /// Total *scalar* parameter count, derived by hand for the oracle config
    /// (V=6, D=4, H=2, L=1, F=8, O=3, B=2):
    ///   embed            V·D                 = 24
    ///   ln (each)        2·D = 8; ln1+ln2    = 16
    ///   mha              4·(D² + D)          = 80
    ///   ffn1             D·F + F             = 40
    ///   ffn2             F·D + D             = 36
    ///   block total                          = 172
    ///   final_ln         2·D                 = 8
    ///   encoder          172 + 8             = 180
    ///   ownership head   D·O + O             = 15
    ///   borrow head      D·B + B             = 10
    ///   invalid head     D·1 + 1             = 5
    ///   grand total                          = 234
    #[test]
    fn scalar_parameter_count_matches_hand_derivation() {
        let mut model = SomModel::new(oracle_cfg());
        let tape = Tape::new();
        let _ = model.forward(&tape, &[1, 2, 3]);
        assert_eq!(model.parameter_count(&tape), 234);
    }

    /// Embedding lookup must return the exact vocab row for a token id.
    /// We set weight row `r` to `[r, r, r, r]`, then check that embedding token
    /// id `t` yields a row of all `t`s — i.e. row `t` of the table, proving the
    /// vocab is indexed by token id (not transposed or off-by-one).
    #[test]
    fn embedding_returns_the_token_row() {
        let cfg = oracle_cfg(); // vocab_size=6, d_model=4
        let mut model = SomModel::new(cfg);
        let v = model.cfg.vocab_size;
        let d = model.cfg.d_model;
        let mut w = vec![0.0f32; v * d];
        for r in 0..v
        {
            for c in 0..d
            {
                w[r * d + c] = r as f32;
            }
        }
        model.embed.weight = Tensor::from_vec(w, v, d);

        // Look up tokens [5, 0, 3] directly through the embedding layer.
        let tape = Tape::new();
        let ids = tape.input(Tensor::from_vec(vec![5.0, 0.0, 3.0], 3, 1));
        let out = model.embed.forward(&tape, ids);
        let ot = tape.value(out.idx());
        assert_eq!(ot.shape(), (3, 4));
        assert_eq!(&ot.data[0..4], &[5.0, 5.0, 5.0, 5.0]); // token 5
        assert_eq!(&ot.data[4..8], &[0.0, 0.0, 0.0, 0.0]); // token 0
        assert_eq!(&ot.data[8..12], &[3.0, 3.0, 3.0, 3.0]); // token 3
    }

    /// With every weight zeroed the network is the constant-zero map:
    ///   embed(0) = 0 → +PE → encoder.
    ///   In each block, LayerNorm has γ=β=0 ⇒ output 0, so attention and FFN
    ///   sub-layers receive 0 and (with zero weights) emit 0; the residual
    ///   passes PE through unchanged. The encoder's final LayerNorm again has
    ///   γ=β=0 ⇒ hidden = 0. Zero-weight heads then emit 0 for every logit.
    /// Therefore all logits are exactly 0.0, every per-token argmax ties to
    /// class 0, and `invalid` (logit > 0) is false everywhere.
    #[test]
    fn decode_with_zeros_is_all_class_zero() {
        let mut model = SomModel::new(oracle_cfg());
        zero_all_params(&mut model);

        let tape = Tape::new();
        let ids = vec![5, 0, 3, 1];
        let logits = model.forward(&tape, &ids);

        // Logits are exactly zero.
        for idx in [
            logits.ownership.idx(),
            logits.borrow.idx(),
            logits.invalid.idx(),
        ]
        {
            for &x in &tape.value(idx).data
            {
                assert_eq!(x, 0.0, "zero-init logit must be exactly 0.0");
            }
        }

        // Decode is therefore fully determinate.
        let labels = logits.decode();
        assert_eq!(labels.ownership, vec![0, 0, 0, 0]);
        assert_eq!(labels.borrow, vec![0, 0, 0, 0]);
        assert_eq!(labels.invalid, vec![false, false, false, false]);
    }

    /// `decode`'s argmax must select the largest logit per token (first index
    /// on ties), independently of the heads that produced them. We bypass the
    /// real heads by overwriting the logit tensors on the tape with known
    /// values, then check the decoded labels row by row.
    #[test]
    fn decode_argmax_picks_largest_logit_per_token() {
        let mut model = SomModel::new(oracle_cfg());
        let tape = Tape::new();
        let logits = model.forward(&tape, &[1, 2, 3]); // seq_len = 3

        // ownership: 3 classes, 3 tokens. Winners: row0→2, row1→0, row2→1.
        tape.set_value(
            logits.ownership.idx(),
            Tensor::from_vec(
                vec![
                    0.1, 0.2, 0.9, // token 0 → class 2
                    0.7, 0.3, 0.5, // token 1 → class 0
                    0.0, 1.0, 0.4, // token 2 → class 1
                ],
                3,
                3,
            ),
        );
        // borrow: 2 classes. Winners: row0→1, row1→0, row2→0 (tie → first).
        tape.set_value(
            logits.borrow.idx(),
            Tensor::from_vec(
                vec![
                    -0.5, 0.5, // token 0 → class 1
                    2.0, 1.0, // token 1 → class 0
                    0.3, 0.3, // token 2 → class 0 (tie, lowest index)
                ],
                3,
                2,
            ),
        );
        // invalid: one logit per token. Signs: +, -, +.
        tape.set_value(
            logits.invalid.idx(),
            Tensor::from_vec(vec![0.2, -0.1, 3.0], 3, 1),
        );

        let labels = logits.decode();
        assert_eq!(labels.ownership, vec![2, 0, 1]);
        assert_eq!(labels.borrow, vec![1, 0, 0]);
        assert_eq!(labels.invalid, vec![true, false, true]);
    }
}
