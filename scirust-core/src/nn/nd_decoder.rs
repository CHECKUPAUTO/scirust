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
use crate::nn::nd_optim::{NdAdam, NdParam};
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

    /// Forward up to (and including) the final LayerNorm → `(seq, d_model)` hidden
    /// states: the inputs to the LM head and to any [`MedusaHeads`].
    pub fn forward_hidden<'t>(&mut self, tape: &'t NdTape, tokens: &[usize]) -> NdVar<'t> {
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
        self.ln_f.forward(tape, x) // (seq, d_model)
    }

    /// Forward a token sequence → `(seq, vocab)` next-token logits.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, tokens: &[usize]) -> NdVar<'t> {
        let h = self.forward_hidden(tape, tokens);
        self.head.forward(tape, h) // (seq, vocab)
    }

    /// Both the next-token logits `(seq, vocab)` and the hidden states
    /// `(seq, d_model)` from a single forward — used by Medusa, which feeds the
    /// last hidden state to its extra prediction heads.
    pub fn forward_with_hidden<'t>(
        &mut self,
        tape: &'t NdTape,
        tokens: &[usize],
    ) -> (NdVar<'t>, NdVar<'t>) {
        let h = self.forward_hidden(tape, tokens);
        let logits = self.head.forward(tape, h);
        (logits, h)
    }

    /// The model (embedding) width `d_model`.
    pub fn d_model(&self) -> usize {
        self.tok.table().shape[1]
    }

    /// The (frozen) token embedding row for `token` (length `d_model`) — EAGLE
    /// conditions its feature autoregression on the previous token's embedding.
    pub fn token_embedding(&self, token: usize) -> Vec<f32> {
        let t = self.tok.table();
        let dim = t.shape[1];
        t.data[token * dim..(token + 1) * dim].to_vec()
    }

    /// Map a single hidden feature (length `d_model`) through the LM head to
    /// vocabulary logits — lets EAGLE turn a *predicted* feature into a token.
    pub fn head_logits(&mut self, feature: &[f32]) -> Vec<f32> {
        let dm = feature.len();
        let tape = NdTape::new();
        let f = tape.input(TensorND::new(feature.to_vec(), vec![1, dm]));
        let logits = self.head.forward(&tape, f);
        tape.value(logits).data.clone()
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
    /// gradient index — feed to [`NdAdam`]. Call
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

    /// Maximum sequence length (size of the positional table).
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Autoregressive **greedy** decoding: append the argmax next token,
    /// `n_new` times, and return the newly generated tokens. Requires
    /// `prompt.len() + n_new <= max_seq`.
    pub fn generate_greedy(&mut self, prompt: &[usize], n_new: usize) -> Vec<usize> {
        assert!(!prompt.is_empty(), "empty prompt");
        assert!(
            prompt.len() + n_new <= self.max_seq,
            "prompt + n_new exceeds max_seq"
        );
        let mut seq = prompt.to_vec();
        for _ in 0..n_new
        {
            let tape = NdTape::new();
            let next = *self.predict(&tape, &seq).last().unwrap();
            seq.push(next);
        }
        seq[prompt.len()..].to_vec()
    }
}

/// **Speculative decoding** (Leviathan et al. 2023; Chen et al. 2023), greedy
/// variant: the cheap `draft` proposes up to `k` tokens, the `target` verifies
/// them in a single forward, accepts the longest prefix whose argmax matches,
/// corrects the first mismatch, and adds a bonus token when all are accepted.
///
/// The output is **exactly** `target.generate_greedy(prompt, n_new)` for *any*
/// draft — only the number of `target` forwards changes (the speed-up). Returns
/// `(new_tokens, target_forward_count)`. Requires `prompt.len() + n_new <=
/// target.max_seq()` and `k >= 1`.
pub fn generate_speculative(
    target: &mut NdDecoderLM,
    draft: &mut NdDecoderLM,
    prompt: &[usize],
    n_new: usize,
    k: usize,
) -> (Vec<usize>, usize) {
    assert!(k >= 1, "k must be >= 1");
    assert!(!prompt.is_empty(), "empty prompt");
    assert!(
        prompt.len() + n_new <= target.max_seq(),
        "prompt + n_new exceeds target max_seq"
    );
    let mut seq = prompt.to_vec();
    let mut target_forwards = 0usize;

    while seq.len() - prompt.len() < n_new
    {
        let remaining = n_new - (seq.len() - prompt.len());
        let kk = k.min(remaining);

        // 1. Draft proposes kk tokens greedily.
        let mut proposed = Vec::with_capacity(kk);
        let mut dseq = seq.clone();
        for _ in 0..kk
        {
            let tape = NdTape::new();
            let nt = *draft.predict(&tape, &dseq).last().unwrap();
            proposed.push(nt);
            dseq.push(nt);
        }

        // 2. Target verifies the whole block in ONE forward.
        let mut check = seq.clone();
        check.extend(&proposed);
        let tape = NdTape::new();
        let preds = target.predict(&tape, &check);
        target_forwards += 1;

        // 3. Accept the matching prefix; correct the first mismatch.
        let base = seq.len() - 1; // preds[base] is the token after `seq`
        let mut accepted = 0;
        for (i, &pi) in proposed.iter().enumerate()
        {
            let tok = preds[base + i];
            if tok == pi
            {
                seq.push(pi);
                accepted += 1;
            }
            else
            {
                seq.push(tok); // target's correction (== greedy's choice here)
                break;
            }
        }
        // 4. All accepted ⇒ a free bonus token (target's argmax at the end).
        if accepted == proposed.len() && seq.len() - prompt.len() < n_new
        {
            seq.push(preds[base + proposed.len()]);
        }
    }

    seq.truncate(prompt.len() + n_new);
    (seq[prompt.len()..].to_vec(), target_forwards)
}

/// Argmax of a logit row (first index on ties).
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

/// **Medusa decoding heads** (Cai et al., ICML 2024). Extra linear heads attached
/// to the base model's last hidden state, head `j` predicting the token `j + 2`
/// positions ahead (head `0` ⇒ offset 2, …). Together with the base model's own
/// next-token prediction they form a multi-token **draft from a single forward**,
/// which [`generate_medusa`] then verifies — so the output stays exactly greedy
/// while several tokens can be committed per verification.
pub struct MedusaHeads {
    heads: Vec<NdLinear>,
    d_model: usize,
}

impl MedusaHeads {
    /// `num_heads` extra heads over a `d_model`→`vocab` projection, seeded init.
    pub fn new(num_heads: usize, d_model: usize, vocab: usize, rng: &mut PcgEngine) -> Self {
        assert!(num_heads >= 1, "Medusa needs at least one head");
        let heads = (0..num_heads)
            .map(|_| NdLinear::new(d_model, vocab, rng))
            .collect();
        Self { heads, d_model }
    }

    /// Number of extra heads (`= longest speculative block beyond the base token`).
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Trainable parameters of every head, in order.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        for head in &mut self.heads
        {
            params.extend(head.parameters());
        }
        params
    }

    /// The most likely token at offsets `+2, +3, …` from a single hidden state
    /// `hidden_last` (length `d_model`) — the speculative continuation past the
    /// base model's own next token.
    pub fn propose(&mut self, hidden_last: &[f32]) -> Vec<usize> {
        assert_eq!(hidden_last.len(), self.d_model, "Medusa: hidden width");
        self.heads
            .iter_mut()
            .map(|head| {
                let tape = NdTape::new();
                let hv = tape.input(TensorND::new(hidden_last.to_vec(), vec![1, self.d_model]));
                let logits = head.forward(&tape, hv); // (1, vocab)
                argmax_row(&tape.value(logits).data)
            })
            .collect()
    }

    /// Train the heads (base model **frozen**) to predict the future tokens of
    /// `seq`: head `j` regresses the token `j + 2` ahead from each hidden state.
    /// The base hidden states are computed once (no gradient flows into the base).
    pub fn train(&mut self, model: &mut NdDecoderLM, seq: &[usize], steps: usize, lr: f32) {
        let l = seq.len();
        // Frozen base hidden states (L, d_model), detached as a constant.
        let tape0 = NdTape::new();
        let h = model.forward_hidden(&tape0, seq);
        let hidden_val = tape0.value(h).clone();
        let mut opt = NdAdam::with_lr(lr);
        for _ in 0..steps
        {
            let tape = NdTape::new();
            let hin = tape.input(hidden_val.clone()); // constant (L, d_model)
            let mut total: Option<NdVar> = None;
            for (j, head) in self.heads.iter_mut().enumerate()
            {
                let o = j + 2;
                if l <= o
                {
                    continue;
                }
                let logits = head.forward(&tape, hin); // (L, vocab)
                let rows: Vec<usize> = (0..l - o).collect();
                let sub = logits.gather(&rows); // (L-o, vocab)
                let ce = sub.cross_entropy(&seq[o..l]);
                total = Some(match total
                {
                    None => ce,
                    Some(t) => t.add(ce),
                });
            }
            if let Some(loss) = total
            {
                let grads = tape.backward(loss);
                let mut params = self.parameters();
                opt.step(&mut params, &grads);
            }
        }
    }
}

/// **Medusa decoding** (Cai et al., ICML 2024, greedy variant). Each round the base
/// model forwards the committed sequence (producing its next token *and* the hidden
/// state), the [`MedusaHeads`] propose the following tokens from that hidden state,
/// and a single verification forward over the proposed block accepts the longest
/// prefix matching the base model's argmax, committing a correction/bonus token.
///
/// The output is **exactly** `model.generate_greedy(prompt, n_new)` for *any* heads
/// (verification guarantees it); good heads merely let more than one token be
/// committed per block. Returns `(new_tokens, base_forward_count)`. With heads that
/// never help, every block commits one token (`2·n_new` forwards); whenever a head
/// speculates correctly a block commits ≥ 2 tokens, so the count drops.
pub fn generate_medusa(
    model: &mut NdDecoderLM,
    heads: &mut MedusaHeads,
    prompt: &[usize],
    n_new: usize,
) -> (Vec<usize>, usize) {
    assert!(!prompt.is_empty(), "empty prompt");
    assert!(
        prompt.len() + n_new <= model.max_seq(),
        "prompt + n_new exceeds max_seq"
    );
    let mut seq = prompt.to_vec();
    let mut forwards = 0usize;

    while seq.len() - prompt.len() < n_new
    {
        let remaining = n_new - (seq.len() - prompt.len());

        // 1. One base forward → next-token logits + hidden state; build the draft
        //    block [base greedy token, then the Medusa heads' speculation].
        let tape = NdTape::new();
        let (logits, hidden) = model.forward_with_hidden(&tape, &seq);
        forwards += 1;
        let lv = tape.value(logits);
        let vocab = lv.shape[1];
        let last = seq.len() - 1;
        let base_next = argmax_row(&lv.data[last * vocab..(last + 1) * vocab]);
        let hv = tape.value(hidden);
        let dm = hv.shape[1];
        let hidden_last = hv.data[last * dm..(last + 1) * dm].to_vec();
        let mut block = vec![base_next];
        block.extend(heads.propose(&hidden_last));
        block.truncate(remaining);

        // 2. One verification forward over seq + block.
        let mut check = seq.clone();
        check.extend(&block);
        let tape2 = NdTape::new();
        let preds = model.predict(&tape2, &check);
        forwards += 1;

        // 3. Accept the matching prefix; the base token always matches greedy.
        let base = seq.len() - 1;
        let mut accepted = 0;
        for (i, &bi) in block.iter().enumerate()
        {
            if preds[base + i] == bi
            {
                seq.push(bi);
                accepted += 1;
            }
            else
            {
                seq.push(preds[base + i]); // greedy correction
                break;
            }
        }
        // 4. All accepted ⇒ a bonus greedy token after the block.
        if accepted == block.len() && seq.len() - prompt.len() < n_new
        {
            seq.push(preds[base + block.len()]);
        }
    }

    seq.truncate(prompt.len() + n_new);
    (seq[prompt.len()..].to_vec(), forwards)
}

/// **EAGLE autoregression head** (Li et al., ICML 2024). Where Medusa predicts
/// future tokens with independent heads, EAGLE drafts at the **feature** level: a
/// lightweight head maps `(feature_t, embedding(token_{t+1})) → feature_{t+1}`, and
/// the frozen LM head turns that predicted feature into the next token. Chaining it
/// gives a higher-quality autoregressive draft from cheap steps; [`generate_eagle`]
/// verifies the draft so the output stays exactly greedy.
pub struct EagleHead {
    net: NdLinear, // (2·d_model) → d_model
    d_model: usize,
}

impl EagleHead {
    /// New head over a `2·d_model → d_model` projection, seeded init.
    pub fn new(d_model: usize, rng: &mut PcgEngine) -> Self {
        Self {
            net: NdLinear::new(2 * d_model, d_model, rng),
            d_model,
        }
    }

    /// Trainable parameters of the head.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        self.net.parameters()
    }

    /// Predict the next hidden feature from the current `feature` and the embedding
    /// `embed` of the next token (both length `d_model`).
    pub fn predict_feature(&mut self, feature: &[f32], embed: &[f32]) -> Vec<f32> {
        assert_eq!(feature.len(), self.d_model, "EAGLE: feature width");
        assert_eq!(embed.len(), self.d_model, "EAGLE: embed width");
        let mut x = feature.to_vec();
        x.extend_from_slice(embed); // (2·d_model)
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x, vec![1, 2 * self.d_model]));
        let f = self.net.forward(&tape, xv);
        tape.value(f).data.clone()
    }

    /// Train the head (base model **frozen**) to regress the next feature: from the
    /// base hidden states `H` of `seq`, fit `head(H[i], embed(seq[i+1])) ≈ H[i+1]`
    /// by MSE. Inputs/targets are detached constants (no gradient into the base).
    pub fn train(&mut self, model: &mut NdDecoderLM, seq: &[usize], steps: usize, lr: f32) {
        let l = seq.len();
        let dm = self.d_model;
        if l < 2
        {
            return;
        }
        // Frozen base features, then build (feature, next-embed) → next-feature.
        let tape0 = NdTape::new();
        let h = model.forward_hidden(&tape0, seq);
        let hval = tape0.value(h).clone(); // (L, d_model)
        let mut x = Vec::with_capacity((l - 1) * 2 * dm);
        let mut y = Vec::with_capacity((l - 1) * dm);
        for i in 0..l - 1
        {
            x.extend_from_slice(&hval.data[i * dm..(i + 1) * dm]);
            x.extend_from_slice(&model.token_embedding(seq[i + 1]));
            y.extend_from_slice(&hval.data[(i + 1) * dm..(i + 2) * dm]);
        }
        let mut opt = NdAdam::with_lr(lr);
        for _ in 0..steps
        {
            let tape = NdTape::new();
            let xv = tape.input(TensorND::new(x.clone(), vec![l - 1, 2 * dm]));
            let yv = tape.input(TensorND::new(y.clone(), vec![l - 1, dm]));
            let pred = self.net.forward(&tape, xv);
            let diff = pred.sub(yv);
            let loss = diff.mul(diff).sum(); // Σ (pred − target)²
            let grads = tape.backward(loss);
            let mut params = self.parameters();
            opt.step(&mut params, &grads);
        }
    }
}

/// **EAGLE decoding** (Li et al., ICML 2024, greedy variant). The base model
/// forwards the committed sequence; from its last feature the [`EagleHead`] drafts
/// up to `k − 1` further tokens by autoregressing at the feature level (each
/// predicted feature passed through the frozen LM head), and a single verification
/// forward accepts the longest prefix matching the base model's argmax, committing
/// a greedy correction/bonus.
///
/// The output is **exactly** `model.generate_greedy(prompt, n_new)` for *any* head;
/// a good (trained) head merely commits more tokens per block. Returns
/// `(new_tokens, base_forward_count)`. Requires `k >= 1`.
pub fn generate_eagle(
    model: &mut NdDecoderLM,
    eagle: &mut EagleHead,
    prompt: &[usize],
    n_new: usize,
    k: usize,
) -> (Vec<usize>, usize) {
    assert!(k >= 1, "k must be >= 1");
    assert!(!prompt.is_empty(), "empty prompt");
    assert!(
        prompt.len() + n_new <= model.max_seq(),
        "prompt + n_new exceeds max_seq"
    );
    let mut seq = prompt.to_vec();
    let mut forwards = 0usize;

    while seq.len() - prompt.len() < n_new
    {
        let remaining = n_new - (seq.len() - prompt.len());

        // 1. One base forward → next token + last feature; autoregress the draft.
        let tape = NdTape::new();
        let (logits, hidden) = model.forward_with_hidden(&tape, &seq);
        forwards += 1;
        let lv = tape.value(logits);
        let vocab = lv.shape[1];
        let last = seq.len() - 1;
        let base_next = argmax_row(&lv.data[last * vocab..(last + 1) * vocab]);
        let hv = tape.value(hidden);
        let dm = hv.shape[1];
        let mut feature = hv.data[last * dm..(last + 1) * dm].to_vec();

        let mut block = vec![base_next];
        let mut tok = base_next;
        for _ in 1..k
        {
            let emb = model.token_embedding(tok);
            feature = eagle.predict_feature(&feature, &emb);
            tok = argmax_row(&model.head_logits(&feature));
            block.push(tok);
        }
        block.truncate(remaining);

        // 2. One verification forward over seq + block.
        let mut check = seq.clone();
        check.extend(&block);
        let tape2 = NdTape::new();
        let preds = model.predict(&tape2, &check);
        forwards += 1;

        // 3. Accept the matching prefix; correct the first mismatch.
        let base = seq.len() - 1;
        let mut accepted = 0;
        for (i, &bi) in block.iter().enumerate()
        {
            if preds[base + i] == bi
            {
                seq.push(bi);
                accepted += 1;
            }
            else
            {
                seq.push(preds[base + i]);
                break;
            }
        }
        // 4. All accepted ⇒ a bonus greedy token.
        if accepted == block.len() && seq.len() - prompt.len() < n_new
        {
            seq.push(preds[base + block.len()]);
        }
    }

    seq.truncate(prompt.len() + n_new);
    (seq[prompt.len()..].to_vec(), forwards)
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

    /// **Speculative decoding is exact**: its output equals plain greedy for the
    /// target, for *any* draft. With an identical draft every token is accepted,
    /// so the target runs far fewer forwards; with a different draft the output
    /// is still identical (only the forward count rises).
    #[test]
    fn nd_speculative_decoding_is_exact() {
        let cfg = tiny_cfg();
        let prompt = [1usize, 2];
        let n = 5;

        // Greedy reference from the target.
        let mut reference_model = NdDecoderLM::new(cfg, &mut PcgEngine::new(123));
        let greedy = reference_model.generate_greedy(&prompt, n);

        // Identical draft (same seed) ⇒ exact output AND fewer target forwards.
        let mut target = NdDecoderLM::new(cfg, &mut PcgEngine::new(123));
        let mut draft_same = NdDecoderLM::new(cfg, &mut PcgEngine::new(123));
        let (spec_same, fwds_same) =
            generate_speculative(&mut target, &mut draft_same, &prompt, n, 4);
        assert_eq!(
            spec_same, greedy,
            "identical-draft speculative must equal greedy"
        );
        assert!(
            fwds_same < n,
            "identical draft should accept all ⇒ < {n} target forwards, got {fwds_same}"
        );

        // Different draft (different seed) ⇒ still exact output.
        let mut target2 = NdDecoderLM::new(cfg, &mut PcgEngine::new(123));
        let mut draft_diff = NdDecoderLM::new(cfg, &mut PcgEngine::new(999));
        let (spec_diff, _) = generate_speculative(&mut target2, &mut draft_diff, &prompt, n, 4);
        assert_eq!(
            spec_diff, greedy,
            "different-draft speculative must still equal greedy"
        );
    }

    /// **Medusa decoding is exact** — for *any* heads, even random/untrained ones,
    /// its output equals plain greedy (verification corrects every speculation),
    /// and the run is deterministic.
    #[test]
    fn nd_medusa_decoding_is_exact() {
        let cfg = tiny_cfg();
        let prompt = [1usize, 2];
        let n = 5;

        let mut reference = NdDecoderLM::new(cfg, &mut PcgEngine::new(7));
        let greedy = reference.generate_greedy(&prompt, n);

        // Random (untrained) heads: output must still be exactly greedy.
        let mut model = NdDecoderLM::new(cfg, &mut PcgEngine::new(7));
        let mut heads = MedusaHeads::new(3, cfg.d_model, cfg.vocab, &mut PcgEngine::new(55));
        let (out, fwds) = generate_medusa(&mut model, &mut heads, &prompt, n);
        assert_eq!(out, greedy, "Medusa output must equal greedy");
        assert!(fwds <= 2 * n, "more forwards than the worst case");

        // Determinism.
        let mut model2 = NdDecoderLM::new(cfg, &mut PcgEngine::new(7));
        let mut heads2 = MedusaHeads::new(3, cfg.d_model, cfg.vocab, &mut PcgEngine::new(55));
        let (out2, _) = generate_medusa(&mut model2, &mut heads2, &prompt, n);
        assert_eq!(out2, out);
    }

    /// **Trained Medusa heads accelerate decoding** while staying exact. After the
    /// base model overfits a periodic sequence and the heads learn its 2-/3-ahead
    /// tokens, at least one verification block commits more than one token, so the
    /// base-forward count drops below the one-token-per-block worst case `2·n` —
    /// yet the output is still exactly greedy.
    #[test]
    fn nd_medusa_trained_heads_accept_multiple_tokens() {
        let cfg = NdDecoderConfig {
            vocab: 4,
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_layers: 2,
            max_seq: 16,
        };
        // Periodic sequence the model can memorise.
        let seq: Vec<usize> = (0..12).map(|i| i % 3).collect();

        // Overfit the base model on the sequence.
        let mut model = NdDecoderLM::new(cfg, &mut PcgEngine::new(1));
        let mut opt = NdAdam::with_lr(0.05);
        for _ in 0..300
        {
            let t = NdTape::new();
            let loss = model.loss(&t, &seq);
            let lv = t.value(loss);
            let grads = t.backward(loss);
            let mut params = model.parameters();
            opt.step(&mut params, &grads);
            let _ = lv;
        }

        // Greedy continuation of a prompt drawn from the sequence.
        let prompt = [0usize, 1, 2, 0];
        let n = 6;
        let greedy = model.generate_greedy(&prompt, n);

        // Train Medusa heads on the same (frozen-base) sequence, then decode.
        let mut heads = MedusaHeads::new(2, cfg.d_model, cfg.vocab, &mut PcgEngine::new(2));
        heads.train(&mut model, &seq, 400, 0.05);
        let (out, fwds) = generate_medusa(&mut model, &mut heads, &prompt, n);

        assert_eq!(out, greedy, "trained Medusa must still equal greedy");
        assert!(
            fwds < 2 * n,
            "trained heads should accept >1 token in some block (forwards {fwds} not < {})",
            2 * n
        );
    }

    /// **EAGLE decoding is exact** for *any* feature head (even random), and is
    /// deterministic.
    #[test]
    fn nd_eagle_decoding_is_exact() {
        let cfg = tiny_cfg();
        let prompt = [1usize, 2];
        let n = 5;

        let mut reference = NdDecoderLM::new(cfg, &mut PcgEngine::new(7));
        let greedy = reference.generate_greedy(&prompt, n);

        let mut model = NdDecoderLM::new(cfg, &mut PcgEngine::new(7));
        let mut eagle = EagleHead::new(cfg.d_model, &mut PcgEngine::new(88));
        let (out, fwds) = generate_eagle(&mut model, &mut eagle, &prompt, n, 4);
        assert_eq!(out, greedy, "EAGLE output must equal greedy");
        assert!(fwds <= 2 * n);

        let mut model2 = NdDecoderLM::new(cfg, &mut PcgEngine::new(7));
        let mut eagle2 = EagleHead::new(cfg.d_model, &mut PcgEngine::new(88));
        let (out2, _) = generate_eagle(&mut model2, &mut eagle2, &prompt, n, 4);
        assert_eq!(out2, out);
    }

    /// **A trained EAGLE head accelerates decoding** while staying exact: after the
    /// base overfits a periodic sequence and the feature head learns to regress the
    /// next feature, at least one verification block commits more than one token, so
    /// the base-forward count drops below `2·n` — output still exactly greedy.
    #[test]
    fn nd_eagle_trained_head_accepts_multiple_tokens() {
        let cfg = NdDecoderConfig {
            vocab: 4,
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_layers: 2,
            max_seq: 16,
        };
        let seq: Vec<usize> = (0..12).map(|i| i % 3).collect();

        let mut model = NdDecoderLM::new(cfg, &mut PcgEngine::new(1));
        let mut opt = NdAdam::with_lr(0.05);
        for _ in 0..300
        {
            let t = NdTape::new();
            let loss = model.loss(&t, &seq);
            let grads = t.backward(loss);
            let mut params = model.parameters();
            opt.step(&mut params, &grads);
        }

        let prompt = [0usize, 1, 2, 0];
        let n = 6;
        let greedy = model.generate_greedy(&prompt, n);

        let mut eagle = EagleHead::new(cfg.d_model, &mut PcgEngine::new(2));
        eagle.train(&mut model, &seq, 500, 0.02);
        let (out, fwds) = generate_eagle(&mut model, &mut eagle, &prompt, n, 4);

        assert_eq!(out, greedy, "trained EAGLE must still equal greedy");
        assert!(
            fwds < 2 * n,
            "trained EAGLE should accept >1 token in some block (forwards {fwds} not < {})",
            2 * n
        );
    }
}
