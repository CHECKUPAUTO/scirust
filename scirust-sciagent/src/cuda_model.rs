//! Route B: the full resident **GQA forward on the CUDA + Tensor-core backend**
//! (feature `cuda`). The bf16 analogue of [`crate::gpu::ResidentModel`]'s forward:
//! every `SciAgentModel` weight is mirrored into VRAM as a bf16 [`CudaMatrix`], and
//! the whole decoder — embed → N×GQA blocks → final RMSNorm → tied LM head — runs
//! on `scirust_cuda`'s [`CudaChain`] (cuBLASLt GEMMs on Tensor cores + the NVRTC
//! kernels), each op gradient-checked against the CPU in `scirust-cuda`.
//!
//! Results are **not** bit-identical to the fp32 CPU reference — bf16 rounds inputs
//! and the GEMMs accumulate in fp32 — but they agree within a bf16 tolerance
//! (`tests/cuda_parity.rs`). This is B3 of the Route-B plan (`ROUTE_B.md`): the
//! whole 350M forward on Tensor cores. Backward + AdamW is B4.

use scirust_core::autodiff::reverse::Tensor;
use scirust_cuda::{CudaChain, CudaMatrix};

use crate::model::SciAgentModel;

/// One GQA block's weights mirrored into VRAM (bf16).
struct CudaBlock {
    norm1: CudaMatrix,
    wq: CudaMatrix,
    wk: CudaMatrix,
    wv: CudaMatrix,
    wo: CudaMatrix,
    norm2: CudaMatrix,
    wg: CudaMatrix,
    wu: CudaMatrix,
    wd: CudaMatrix,
}

/// The nine weight gradients of one GQA block (resident bf16), matching
/// [`CudaBlock`]'s trainable weights — the seven projections plus the two RMSNorm
/// gains. Produced by [`CudaModel::backward`].
pub struct CudaBlockGrads {
    pub dwq: CudaMatrix,
    pub dwk: CudaMatrix,
    pub dwv: CudaMatrix,
    pub dwo: CudaMatrix,
    pub dwg: CudaMatrix,
    pub dwu: CudaMatrix,
    pub dwd: CudaMatrix,
    pub dnorm1: CudaMatrix,
    pub dnorm2: CudaMatrix,
}

/// Every trainable weight's gradient for one backward pass (resident bf16): the
/// tied embedding (head + input-gather paths summed), the final RMSNorm gain, and
/// per-block grads.
pub struct CudaModelGrads {
    pub d_embedding: CudaMatrix,
    pub blocks: Vec<CudaBlockGrads>,
    pub d_final_norm: CudaMatrix,
}

/// A [`SciAgentModel`] mirrored into VRAM as bf16 matrices, running the whole
/// decoder forward on the Tensor-core [`CudaChain`]. Tied-embedding models only.
pub struct CudaModel {
    chain: CudaChain,
    embedding: CudaMatrix,
    final_norm: CudaMatrix,
    blocks: Vec<CudaBlock>,
    n_heads: usize,
    n_kv_heads: usize,
    theta: f32,
    eps: f32,
    causal: bool,
    vocab: usize,
    d_model: usize,
}

impl CudaModel {
    /// Upload every weight of `model` to VRAM (bf16). Returns `None` if no CUDA
    /// device is available. Panics if the model is not tied-embedding.
    pub fn from_model(model: &SciAgentModel) -> Option<Self> {
        assert!(
            model.config.tie_embeddings,
            "CudaModel requires a tied-embedding model (tied E is the LM head)"
        );
        let chain = CudaChain::new()?;
        let up = |t: &Tensor| chain.upload(&t.data, t.rows, t.cols);
        let embedding = up(&model.embed.weight);
        let final_norm = up(&model.rms_final.weight);
        let blocks = model
            .layers
            .iter()
            .map(|l| CudaBlock {
                norm1: up(&l.rms_attn.weight),
                wq: up(&l.attn.w_q.weight),
                wk: up(&l.attn.w_k.weight),
                wv: up(&l.attn.w_v.weight),
                wo: up(&l.attn.w_o.weight),
                norm2: up(&l.rms_ffn.weight),
                wg: up(&l.ffn.gate.weight),
                wu: up(&l.ffn.up.weight),
                wd: up(&l.ffn.down.weight),
            })
            .collect();
        Some(Self {
            chain,
            embedding,
            final_norm,
            blocks,
            n_heads: model.config.n_heads,
            n_kv_heads: model.config.n_kv_heads,
            theta: model.config.rope_theta,
            eps: model.config.eps,
            causal: true,
            vocab: model.config.vocab_size,
            d_model: model.config.d_model,
        })
    }

    /// Vocabulary size (logit width).
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Multi-head grouped-query attention over `q` (`t×d_model`) and `k`/`v`
    /// (`t×kv_dim`), matching `GpuChain::gqa_attention`: RoPE the full-width q/k,
    /// then per head `softmax((qs·ksᵀ)/√dh [+causal])·vs`, placed into the head's
    /// `d_model` slot and summed.
    fn attention(&self, q: &CudaMatrix, k: &CudaMatrix, v: &CudaMatrix) -> CudaMatrix {
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = self.chain.rope(q, seq, 0, self.theta);
        let kr = self.chain.rope(k, seq, 0, self.theta);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut out: Option<CudaMatrix> = None;
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = self.chain.slice_cols(&qr, head * dh, dh);
            let ks = self.chain.slice_cols(&kr, kv * dh, dh);
            let vs = self.chain.slice_cols(v, kv * dh, dh);
            let scores = self.chain.matmul_bt(&qs, &ks); // qs·ksᵀ  (t×t)
            let scaled = self.chain.scale_causal_mask(&scores, scale, self.causal);
            let weights = self.chain.softmax(&scaled);
            let ctx = self.chain.matmul(&weights, &vs); // (t×dh)
            let padded = self.chain.place_cols(&ctx, head * dh, self.d_model);
            out = Some(match out
            {
                None => padded,
                Some(acc) => self.chain.add(&acc, &padded),
            });
        }
        out.expect("n_heads ≥ 1")
    }

    /// One GQA transformer block (pre-norm + residual, attention then SwiGLU MLP).
    fn block(&self, x: &CudaMatrix, b: &CudaBlock) -> CudaMatrix {
        let xn = self.chain.rms_norm(x, &b.norm1, self.eps);
        let q = self.chain.matmul(&xn, &b.wq);
        let k = self.chain.matmul(&xn, &b.wk);
        let v = self.chain.matmul(&xn, &b.wv);
        let ctx = self.attention(&q, &k, &v);
        let attn_out = self.chain.matmul(&ctx, &b.wo);
        let h = self.chain.add(x, &attn_out);
        // MLP: (silu(hn·Wg) ⊙ (hn·Wu)) · Wd.
        let hn = self.chain.rms_norm(&h, &b.norm2, self.eps);
        let gate = self.chain.matmul(&hn, &b.wg);
        let up = self.chain.matmul(&hn, &b.wu);
        let act = self.chain.swiglu(&gate, &up);
        let mlp = self.chain.matmul(&act, &b.wd);
        self.chain.add(&h, &mlp)
    }

    /// Full forward `tokens → logits` kept **resident**: the `tokens.len() × vocab`
    /// logit matrix on the device (row-major), for chaining into the backward /
    /// cross-entropy grad without a host round-trip. Single sequence.
    fn forward_resident(&self, tokens: &[u32]) -> CudaMatrix {
        let mut x = self.chain.embed(tokens, &self.embedding);
        for b in &self.blocks
        {
            x = self.block(&x, b);
        }
        let normed = self.chain.rms_norm(&x, &self.final_norm, self.eps);
        // Tied head: logits = normed · Eᵀ.
        self.chain.matmul_bt(&normed, &self.embedding)
    }

    /// Full forward `tokens → logits`: the `tokens.len() × vocab` logit matrix
    /// (row-major), computed on Tensor cores and downloaded. Single sequence.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        self.chain.download(&self.forward_resident(tokens))
    }

    /// Backward of [`Self::attention`] (the GQA analogue of Route A's
    /// `gqa_attention_backward`): given the forward `q`/`k`/`v` and the context
    /// grad `dout` (`t×d_model`), returns `(dq, dk, dv)`. Recomputes each head's
    /// softmax weights, then the single-head attention adjoint, scattering per-head
    /// grads back to full width and undoing RoPE on q/k.
    fn attention_backward(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
        dout: &CudaMatrix,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let kv_dim = self.n_kv_heads * dh;
        let qr = ch.rope(q, seq, 0, self.theta);
        let kr = ch.rope(k, seq, 0, self.theta);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut dqr: Option<CudaMatrix> = None;
        let mut dkr: Option<CudaMatrix> = None;
        let mut dvv: Option<CudaMatrix> = None;
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = ch.slice_cols(&qr, head * dh, dh);
            let ks = ch.slice_cols(&kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            // Recompute this head's forward softmax weights.
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, self.causal);
            let weights = ch.softmax(&scaled);
            // Grad of this head's context = adjoint of place_cols = slice of dout.
            let d_ctx = ch.slice_cols(dout, head * dh, dh);
            // Single-head attention adjoint.
            let dweights = ch.matmul_bt(&d_ctx, &vs); // d_ctx·vsᵀ
            let dvs = ch.matmul_at(&weights, &d_ctx); // weightsᵀ·d_ctx
            let dscaled = ch.softmax_backward(&weights, &dweights);
            let dscores = ch.scale_causal_mask_backward(&dscaled, scale, self.causal);
            let dqs = ch.matmul(&dscores, &ks); // dscores·ks
            let dks = ch.matmul_at(&dscores, &qs); // dscoresᵀ·qs
            // Scatter each head's grads back to full width and accumulate.
            let dqs_full = ch.place_cols(&dqs, head * dh, self.d_model);
            let dks_full = ch.place_cols(&dks, kv * dh, kv_dim);
            let dvs_full = ch.place_cols(&dvs, kv * dh, kv_dim);
            dqr = Some(match dqr
            {
                None => dqs_full,
                Some(acc) => ch.add(&acc, &dqs_full),
            });
            dkr = Some(match dkr
            {
                None => dks_full,
                Some(acc) => ch.add(&acc, &dks_full),
            });
            dvv = Some(match dvv
            {
                None => dvs_full,
                Some(acc) => ch.add(&acc, &dvs_full),
            });
        }
        let dqr = dqr.expect("n_heads ≥ 1");
        let dkr = dkr.expect("n_heads ≥ 1");
        let dv = dvv.expect("n_heads ≥ 1");
        // RoPE adjoint: qr = rope(q), kr = rope(k); v was not rotated.
        let dq = ch.rope_backward(&dqr, seq, 0, self.theta);
        let dk = ch.rope_backward(&dkr, seq, 0, self.theta);
        (dq, dk, dv)
    }

    /// Backward of [`Self::block`] (mirrors Route A's
    /// `gqa_transformer_block_backward_full`): returns `dx` and the nine weight
    /// gradients. Forward activations are recomputed (cheap resident ops).
    fn block_backward(
        &self,
        x: &CudaMatrix,
        b: &CudaBlock,
        dout: &CudaMatrix,
    ) -> (CudaMatrix, CudaBlockGrads) {
        let ch = &self.chain;
        // --- recompute forward activations ---
        let xn = ch.rms_norm(x, &b.norm1, self.eps);
        let q = ch.matmul(&xn, &b.wq);
        let k = ch.matmul(&xn, &b.wk);
        let v = ch.matmul(&xn, &b.wv);
        let ctx = self.attention(&q, &k, &v);
        let h = ch.add(x, &ch.matmul(&ctx, &b.wo));
        let hn = ch.rms_norm(&h, &b.norm2, self.eps);
        let gate = ch.matmul(&hn, &b.wg);
        let up = ch.matmul(&hn, &b.wu);
        let act = ch.swiglu(&gate, &up);

        // --- MLP path ---
        let dact = ch.matmul_bt(dout, &b.wd); // dout·Wdᵀ
        let dwd = ch.matmul_at(&act, dout); // actᵀ·dout
        let (dgate, dup) = ch.swiglu_backward(&gate, &up, &dact);
        let dwg = ch.matmul_at(&hn, &dgate); // hnᵀ·dgate
        let dwu = ch.matmul_at(&hn, &dup); // hnᵀ·dup
        let dhn = ch.add(&ch.matmul_bt(&dgate, &b.wg), &ch.matmul_bt(&dup, &b.wu));
        let dnorm2 = ch.rms_norm_gain_backward(&h, &dhn, self.eps);
        let dh = ch.add(dout, &ch.rms_norm_backward(&h, &b.norm2, &dhn, self.eps));

        // --- attention path ---
        let dwo = ch.matmul_at(&ctx, &dh); // ctxᵀ·dh
        let d_ctx = ch.matmul_bt(&dh, &b.wo); // dh·Woᵀ
        let (dq, dk, dv) = self.attention_backward(&q, &k, &v, &d_ctx);
        let dwq = ch.matmul_at(&xn, &dq); // xnᵀ·dq
        let dwk = ch.matmul_at(&xn, &dk); // xnᵀ·dk
        let dwv = ch.matmul_at(&xn, &dv); // xnᵀ·dv
        let dxn = ch.add(
            &ch.add(&ch.matmul_bt(&dq, &b.wq), &ch.matmul_bt(&dk, &b.wk)),
            &ch.matmul_bt(&dv, &b.wv),
        );
        let dnorm1 = ch.rms_norm_gain_backward(x, &dxn, self.eps);
        let dx = ch.add(&dh, &ch.rms_norm_backward(x, &b.norm1, &dxn, self.eps));

        (
            dx,
            CudaBlockGrads {
                dwq,
                dwk,
                dwv,
                dwo,
                dwg,
                dwu,
                dwd,
                dnorm1,
                dnorm2,
            },
        )
    }

    /// Full model backward (mirrors Route A's `gqa_model_backward`): given the logit
    /// grad `dlogits` (`t×vocab`), returns every trainable weight's gradient — the
    /// tied embedding (head + input-gather paths summed), the final RMSNorm gain, and
    /// each block's grads. All resident. Recomputes the block-boundary activations.
    pub fn backward(&self, tokens: &[u32], dlogits: &CudaMatrix) -> CudaModelGrads {
        let ch = &self.chain;
        // Recompute block-boundary activations: xs[i] is the input to block i.
        let mut xs = Vec::with_capacity(self.blocks.len() + 1);
        xs.push(ch.embed(tokens, &self.embedding));
        for b in &self.blocks
        {
            let out = self.block(xs.last().unwrap(), b);
            xs.push(out);
        }
        let trunk = xs.last().unwrap();
        let normed = ch.rms_norm(trunk, &self.final_norm, self.eps);

        // Tied head: logits = normed · Eᵀ.
        let d_normed = ch.matmul(dlogits, &self.embedding); // dlogits·E   (t×d)
        let de_head = ch.matmul_at(dlogits, &normed); // dlogitsᵀ·normed (vocab×d)

        let d_final_norm = ch.rms_norm_gain_backward(trunk, &d_normed, self.eps);
        let mut d_cur = ch.rms_norm_backward(trunk, &self.final_norm, &d_normed, self.eps);
        let mut block_grads: Vec<CudaBlockGrads> = Vec::with_capacity(self.blocks.len());
        for i in (0..self.blocks.len()).rev()
        {
            let (dx, grads) = self.block_backward(&xs[i], &self.blocks[i], &d_cur);
            d_cur = dx;
            block_grads.push(grads);
        }
        block_grads.reverse();

        // d_cur is now d(emb); add the embedding-lookup path into the tied grad.
        let de_embed = ch.embed_backward(tokens, &d_cur, self.vocab);
        let d_embedding = ch.add(&de_head, &de_embed);
        CudaModelGrads {
            d_embedding,
            blocks: block_grads,
            d_final_norm,
        }
    }

    /// The tied-embedding gradient for `(tokens, targets)`, downloaded — the single
    /// number that validates the whole backward: it sums the LM-head grad and the
    /// grad backpropagated through every block into the input gather. Forward →
    /// cross-entropy grad → backward, entirely resident, then one download.
    pub fn embedding_grad(&self, tokens: &[u32], targets: &[u32]) -> Vec<f32> {
        let logits = self.forward_resident(tokens);
        let dlogits = self.chain.cross_entropy_grad(&logits, targets);
        let grads = self.backward(tokens, &dlogits);
        self.chain.download(&grads.d_embedding)
    }
}
