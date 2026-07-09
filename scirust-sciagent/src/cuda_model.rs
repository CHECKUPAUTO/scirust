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
use scirust_core::autodiff::scheduler::LrSchedule;
use scirust_cuda::{CudaChain, CudaF32, CudaMatrix};

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;
use crate::train::checkpoint::{CheckpointMeta, save_checkpoint};
use crate::train::scheduler::WarmupCosineSchedule;

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

/// One GQA block's **fp32 master** copies (or AdamW moments) — the full-precision
/// mirror of [`CudaBlock`]'s nine trainable weights. Master weights and the
/// moments `m`/`v` all use this layout; the forward/backward see only the bf16
/// [`CudaMatrix`] views held in [`CudaBlock`].
struct BlockMasters {
    norm1: CudaF32,
    wq: CudaF32,
    wk: CudaF32,
    wv: CudaF32,
    wo: CudaF32,
    norm2: CudaF32,
    wg: CudaF32,
    wu: CudaF32,
    wd: CudaF32,
}

impl BlockMasters {
    /// Upload a layer's nine weights to fp32 masters.
    fn from_layer(chain: &CudaChain, l: &crate::block::SciAgentBlock) -> Self {
        let up = |t: &Tensor| chain.upload_f32(&t.data);
        Self {
            norm1: up(&l.rms_attn.weight),
            wq: up(&l.attn.w_q.weight),
            wk: up(&l.attn.w_k.weight),
            wv: up(&l.attn.w_v.weight),
            wo: up(&l.attn.w_o.weight),
            norm2: up(&l.rms_ffn.weight),
            wg: up(&l.ffn.gate.weight),
            wu: up(&l.ffn.up.weight),
            wd: up(&l.ffn.down.weight),
        }
    }

    /// Zero moments matching a layer's weight shapes.
    fn zeros_like(chain: &CudaChain, l: &crate::block::SciAgentBlock) -> Self {
        let z = |t: &Tensor| chain.zeros_f32(t.data.len());
        Self {
            norm1: z(&l.rms_attn.weight),
            wq: z(&l.attn.w_q.weight),
            wk: z(&l.attn.w_k.weight),
            wv: z(&l.attn.w_v.weight),
            wo: z(&l.attn.w_o.weight),
            norm2: z(&l.rms_ffn.weight),
            wg: z(&l.ffn.gate.weight),
            wu: z(&l.ffn.up.weight),
            wd: z(&l.ffn.down.weight),
        }
    }
}

/// Host mean cross-entropy `−(1/rows)·Σ log P[i, tgtᵢ]` over row-major logits —
/// the pre-update loss (matches `train::cross_entropy_loss`). Kept here so the CUDA
/// path needs no `scirust-gpu` dependency.
fn host_cross_entropy(logits: &[f32], targets: &[u32], rows: usize, cols: usize) -> f32 {
    let mut loss = 0.0f32;
    for r in 0..rows
    {
        let row = &logits[r * cols..(r + 1) * cols];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
        let t = (targets[r] as usize).min(cols - 1);
        let logp = (row[t] - mx) - sum.ln();
        loss -= logp;
    }
    loss / rows as f32
}

/// A trainable [`CudaModel`]: the bf16 model plus **fp32 master weights and AdamW
/// moments** (the mixed-precision contract). Each [`Self::train_step`] runs the
/// whole forward → cross-entropy grad → backward → AdamW update on Tensor cores,
/// updating the fp32 masters and refreshing the bf16 views in one pass — Route B's
/// training half. Tied-embedding models only.
pub struct CudaTrainer {
    model: CudaModel,
    master_embedding: CudaF32,
    m_embedding: CudaF32,
    v_embedding: CudaF32,
    master_final_norm: CudaF32,
    m_final_norm: CudaF32,
    v_final_norm: CudaF32,
    master_blocks: Vec<BlockMasters>,
    m_blocks: Vec<BlockMasters>,
    v_blocks: Vec<BlockMasters>,
    step: u32,
}

impl CudaTrainer {
    /// Build a trainer from `model`: mirror the bf16 [`CudaModel`] and upload fp32
    /// masters (from the original fp32 weights, not the bf16 views) plus zero
    /// moments. Returns `None` if no CUDA device is available.
    pub fn from_model(model: &SciAgentModel) -> Option<Self> {
        let inner = CudaModel::from_model(model)?;
        // Build all fp32 masters + zero moments in a scope so the `chain` borrow
        // ends before `inner` is moved into the struct.
        let (
            master_embedding,
            m_embedding,
            v_embedding,
            master_final_norm,
            m_final_norm,
            v_final_norm,
            master_blocks,
            m_blocks,
            v_blocks,
        ) = {
            let chain = &inner.chain;
            (
                chain.upload_f32(&model.embed.weight.data),
                chain.zeros_f32(model.embed.weight.data.len()),
                chain.zeros_f32(model.embed.weight.data.len()),
                chain.upload_f32(&model.rms_final.weight.data),
                chain.zeros_f32(model.rms_final.weight.data.len()),
                chain.zeros_f32(model.rms_final.weight.data.len()),
                model
                    .layers
                    .iter()
                    .map(|l| BlockMasters::from_layer(chain, l))
                    .collect::<Vec<_>>(),
                model
                    .layers
                    .iter()
                    .map(|l| BlockMasters::zeros_like(chain, l))
                    .collect::<Vec<_>>(),
                model
                    .layers
                    .iter()
                    .map(|l| BlockMasters::zeros_like(chain, l))
                    .collect::<Vec<_>>(),
            )
        };
        Some(Self {
            model: inner,
            master_embedding,
            m_embedding,
            v_embedding,
            master_final_norm,
            m_final_norm,
            v_final_norm,
            master_blocks,
            m_blocks,
            v_blocks,
            step: 0,
        })
    }

    /// The vocabulary width (logit columns).
    pub fn vocab(&self) -> usize {
        self.model.vocab
    }

    /// One mixed-precision AdamW training step on `(tokens, targets)`: forward →
    /// host cross-entropy grad → backward → AdamW update of every trainable weight
    /// (tied embedding, final RMSNorm gain, and each block's nine weights), fp32
    /// masters updated and bf16 views refreshed in place. Returns the **pre-update**
    /// mean cross-entropy loss.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> f32 {
        self.step += 1;
        let rows = tokens.len();
        let vocab = self.model.vocab;

        // Forward (resident) → host loss → cross-entropy grad → backward.
        let logits = self.model.forward_resident(tokens);
        let host = self.model.chain.download(&logits);
        let loss = host_cross_entropy(&host, targets, rows, vocab);
        let dlogits = self.model.chain.cross_entropy_grad(&logits, targets);
        let grads = self.model.backward(tokens, &dlogits);

        // AdamW updates — fp32 masters mutated in place, bf16 views refreshed.
        let step = self.step;
        let ch = &self.model.chain;
        ch.adamw_step(
            &mut self.master_embedding,
            &mut self.m_embedding,
            &mut self.v_embedding,
            &grads.d_embedding,
            &mut self.model.embedding,
            lr,
            betas,
            adam_eps,
            weight_decay,
            step,
        );
        ch.adamw_step(
            &mut self.master_final_norm,
            &mut self.m_final_norm,
            &mut self.v_final_norm,
            &grads.d_final_norm,
            &mut self.model.final_norm,
            lr,
            betas,
            adam_eps,
            weight_decay,
            step,
        );
        for i in 0..self.model.blocks.len()
        {
            let bg = &grads.blocks[i];
            let (mb, mm, mv) = (
                &mut self.master_blocks[i],
                &mut self.m_blocks[i],
                &mut self.v_blocks[i],
            );
            let b = &mut self.model.blocks[i];
            let one = |master: &mut CudaF32,
                       mo: &mut CudaF32,
                       vo: &mut CudaF32,
                       grad: &CudaMatrix,
                       view: &mut CudaMatrix| {
                ch.adamw_step(
                    master,
                    mo,
                    vo,
                    grad,
                    view,
                    lr,
                    betas,
                    adam_eps,
                    weight_decay,
                    step,
                );
            };
            one(
                &mut mb.norm1,
                &mut mm.norm1,
                &mut mv.norm1,
                &bg.dnorm1,
                &mut b.norm1,
            );
            one(&mut mb.wq, &mut mm.wq, &mut mv.wq, &bg.dwq, &mut b.wq);
            one(&mut mb.wk, &mut mm.wk, &mut mv.wk, &bg.dwk, &mut b.wk);
            one(&mut mb.wv, &mut mm.wv, &mut mv.wv, &bg.dwv, &mut b.wv);
            one(&mut mb.wo, &mut mm.wo, &mut mv.wo, &bg.dwo, &mut b.wo);
            one(
                &mut mb.norm2,
                &mut mm.norm2,
                &mut mv.norm2,
                &bg.dnorm2,
                &mut b.norm2,
            );
            one(&mut mb.wg, &mut mm.wg, &mut mv.wg, &bg.dwg, &mut b.wg);
            one(&mut mb.wu, &mut mm.wu, &mut mv.wu, &bg.dwu, &mut b.wu);
            one(&mut mb.wd, &mut mm.wd, &mut mv.wd, &bg.dwd, &mut b.wd);
        }
        loss
    }

    /// Forward `tokens → logits` on the (possibly trained) bf16 model — a thin
    /// pass-through to the inner [`CudaModel::forward`] for eval between steps.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        self.model.forward(tokens)
    }

    /// The device/adapter name (for logging) — always the CUDA path here.
    pub fn adapter_name(&self) -> &'static str {
        "CUDA bf16 Tensor cores"
    }

    /// Reset the AdamW step counter to 0 (fresh bias correction). Used when
    /// resuming from a checkpoint: `from_model` re-uploads the saved weights but
    /// zero-inits the moments, and the warmup schedule re-absorbs the restart.
    pub fn reset_step(&mut self) {
        self.step = 0;
    }

    /// Write the (trained) **fp32 master** weights back into `model`, replacing each
    /// host `Tensor`. Syncs from the fp32 masters (not the bf16 views), so a
    /// checkpoint keeps full precision. Shapes are taken from `model`'s current
    /// tensors (training never changes them).
    pub fn sync_to_model(&self, model: &mut SciAgentModel) {
        let ch = &self.model.chain;
        let dl = |x: &CudaF32, rows: usize, cols: usize| {
            Tensor::from_vec(ch.download_f32(x), rows, cols)
        };
        let (r, c) = (model.embed.weight.rows, model.embed.weight.cols);
        model.embed.weight = dl(&self.master_embedding, r, c);
        let (r, c) = (model.rms_final.weight.rows, model.rms_final.weight.cols);
        model.rms_final.weight = dl(&self.master_final_norm, r, c);
        for (l, mb) in model.layers.iter_mut().zip(&self.master_blocks)
        {
            let shape = |t: &Tensor| (t.rows, t.cols);
            let (r, c) = shape(&l.rms_attn.weight);
            l.rms_attn.weight = dl(&mb.norm1, r, c);
            let (r, c) = shape(&l.attn.w_q.weight);
            l.attn.w_q.weight = dl(&mb.wq, r, c);
            let (r, c) = shape(&l.attn.w_k.weight);
            l.attn.w_k.weight = dl(&mb.wk, r, c);
            let (r, c) = shape(&l.attn.w_v.weight);
            l.attn.w_v.weight = dl(&mb.wv, r, c);
            let (r, c) = shape(&l.attn.w_o.weight);
            l.attn.w_o.weight = dl(&mb.wo, r, c);
            let (r, c) = shape(&l.rms_ffn.weight);
            l.rms_ffn.weight = dl(&mb.norm2, r, c);
            let (r, c) = shape(&l.ffn.gate.weight);
            l.ffn.gate.weight = dl(&mb.wg, r, c);
            let (r, c) = shape(&l.ffn.up.weight);
            l.ffn.up.weight = dl(&mb.wu, r, c);
            let (r, c) = shape(&l.ffn.down.weight);
            l.ffn.down.weight = dl(&mb.wd, r, c);
        }
    }

    /// **Production-scale resident bf16 pretraining** over a flat `u32` token stream —
    /// the Route-B analogue of `ResidentModel::pretrain`, on Tensor cores. Runs
    /// `cfg.total_steps − cfg.start_step` steps over non-overlapping `cfg.seq_len`
    /// windows (deterministic, in-order — the corpus wraps), each a full
    /// [`Self::train_step`] at the warmup+cosine schedule's `lr`. Every
    /// `cfg.save_interval` steps it [`Self::sync_to_model`]s the fp32 masters back
    /// and writes a safetensors checkpoint, so a long run is resumable. Returns the
    /// per-step pre-update loss.
    pub fn pretrain(
        &mut self,
        tokens: &[u32],
        model: &mut SciAgentModel,
        config: &SciAgentConfig,
        cfg: &CudaPretrainConfig,
    ) -> Vec<f32> {
        let s = cfg.seq_len;
        let mut losses = Vec::new();
        if tokens.len() <= s
        {
            eprintln!(
                "cuda pretrain: token stream ({}) shorter than a single window ({}); nothing to do",
                tokens.len(),
                s + 1
            );
            return losses;
        }
        let schedule =
            WarmupCosineSchedule::new(cfg.base_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps);
        let mut step = cfg.start_step;
        let mut cursor = 0usize;
        let t0 = std::time::Instant::now();
        while step < cfg.total_steps
        {
            if cursor + s + 1 > tokens.len()
            {
                cursor = 0;
            }
            let inputs = &tokens[cursor..cursor + s];
            let targets = &tokens[cursor + 1..cursor + s + 1];
            let lr = schedule.lr_at(step);
            let loss = self.train_step(
                inputs,
                targets,
                lr,
                cfg.betas,
                cfg.adam_eps,
                cfg.weight_decay,
            );
            losses.push(loss);
            cursor += s;
            step += 1;

            if cfg.log_interval > 0 && step % cfg.log_interval == 0
            {
                let done = (step - cfg.start_step) * s;
                let secs = t0.elapsed().as_secs_f64().max(1e-9);
                let tps = done as f64 / secs;
                println!("[cuda step {step:>6}] loss {loss:>9.4} | lr {lr:.3e} | {tps:>8.0} tok/s");
            }
            if cfg.save_interval > 0 && step % cfg.save_interval == 0
            {
                self.sync_to_model(model);
                let dir = std::path::Path::new(&cfg.checkpoint_dir).join(format!("step_{step}"));
                let meta = CheckpointMeta {
                    step,
                    loss,
                    lr,
                    config: config.clone(),
                };
                match save_checkpoint(model, &meta, &dir)
                {
                    Ok(()) => println!("  checkpoint → {}", dir.display()),
                    Err(e) => eprintln!("  checkpoint at step {step} failed: {e}"),
                }
            }
        }
        losses
    }
}

/// Run configuration for [`CudaTrainer::pretrain`] — the Route-B counterpart of
/// `ResidentPretrainConfig` (kept here so the CUDA path carries no `gpu`-feature
/// dependency). Warmup+cosine LR, AdamW hyperparameters, and checkpoint cadence.
#[derive(Debug, Clone)]
pub struct CudaPretrainConfig {
    /// Peak (post-warmup) AdamW learning rate.
    pub base_lr: f32,
    /// Floor learning rate the cosine decays to.
    pub min_lr: f32,
    /// Linear warmup length, in optimizer steps.
    pub warmup_steps: usize,
    /// Total optimizer steps for the run (also the cosine period end).
    pub total_steps: usize,
    /// Step to start the LR schedule from (for resuming).
    pub start_step: usize,
    /// AdamW `(β₁, β₂)`.
    pub betas: (f32, f32),
    /// AdamW epsilon.
    pub adam_eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Sequence length of each training window.
    pub seq_len: usize,
    /// Print a loss/lr line every this many steps (0 = never).
    pub log_interval: usize,
    /// Write a checkpoint every this many steps (0 = never).
    pub save_interval: usize,
    /// Directory the `step_N/` checkpoints are written under.
    pub checkpoint_dir: String,
}

impl Default for CudaPretrainConfig {
    fn default() -> Self {
        Self {
            base_lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 2000,
            total_steps: 50_000,
            start_step: 0,
            betas: (0.9, 0.95),
            adam_eps: 1e-8,
            weight_decay: 0.1,
            seq_len: 128,
            log_interval: 100,
            save_interval: 500,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}
