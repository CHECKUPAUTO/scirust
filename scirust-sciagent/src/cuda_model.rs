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

    /// Full forward `tokens → logits`: the `tokens.len() × vocab` logit matrix
    /// (row-major), computed on Tensor cores and downloaded. Single sequence.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let mut x = self.chain.embed(tokens, &self.embedding);
        for b in &self.blocks
        {
            x = self.block(&x, b);
        }
        let normed = self.chain.rms_norm(&x, &self.final_norm, self.eps);
        // Tied head: logits = normed · Eᵀ.
        let logits = self.chain.matmul_bt(&normed, &self.embedding);
        self.chain.download(&logits)
    }
}
