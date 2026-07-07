//! GPU acceleration for the SCIAGENT model (feature `gpu`).
//!
//! The whole model runs on `scirust_core`'s reverse-mode [`Tape`]. Rather than
//! reimplement the forward pass on the device, this module attaches
//! `scirust_gpu`'s validated [`WgpuEngine`] — the tape's GEMM hook — and flips
//! the tape into GPU-matmul mode with [`Tape::set_prefer_gpu_matmul`]. Every
//! plain `matmul` / `try_matmul` the model issues then runs its forward **and**
//! backward on the GPU:
//!
//! - the q/k/v/o projections and the SwiGLU gate/up/down (all `Linear`),
//! - RoPE's pair-rotation `x·W` GEMM,
//! - the per-head attention scores `Q·Kᵀ` and the `·V` re-weighting,
//! - the tied LM head `h·Eᵀ`.
//!
//! The autodiff graph and every non-GEMM op (softmax, RMSNorm, RoPE trig,
//! residual adds, the causal mask) are untouched and stay on the CPU. GEMMs are
//! the dominant FLOPs of a transformer, so routing just them is the pragmatic
//! first integration — no new kernels, and the exact same math the CPU path was
//! already validated against, brick by brick.
//!
//! GPU GEMM accumulates in a different order than the CPU BLAS, so results are
//! **not** bit-identical; they agree within a small relative tolerance. See
//! `tests/gpu_parity.rs` and `examples/gpu_forward_parity.rs`, which check a full
//! model forward + backward against the CPU on a real adapter (e.g. the Jetson
//! Thor's Blackwell).

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_gpu::{GpuChain, GpuMatrix, GqaBlockWeights, GqaModelWeights};

use crate::model::SciAgentModel;

pub use scirust_gpu::WgpuEngine;

/// Attach a freshly-acquired [`WgpuEngine`] to `tape` and switch it into
/// GPU-matmul mode.
///
/// Returns the adapter name on success, or `None` when no GPU adapter is
/// available (no Vulkan/Metal/DX12 driver). On `None` the tape is left
/// untouched and stays CPU-only, so a caller can fall back transparently:
///
/// ```no_run
/// # use scirust_core::autodiff::reverse::Tape;
/// let tape = Tape::new();
/// match scirust_sciagent::gpu::attach_gpu(&tape) {
///     Some(name) => println!("training on {name}"),
///     None => println!("no GPU, staying on CPU"),
/// }
/// // ... model.forward(&tape, ids, seq_len) now runs its GEMMs on the device.
/// ```
pub fn attach_gpu(tape: &Tape) -> Option<String> {
    let engine = WgpuEngine::new()?;
    let name = engine.adapter_name().to_string();
    tape.set_gpu_engine(engine);
    tape.set_prefer_gpu_matmul(true);
    Some(name)
}

/// One GQA block's weights mirrored into VRAM.
struct ResidentBlock {
    norm1: GpuMatrix,
    wq: GpuMatrix,
    wk: GpuMatrix,
    wv: GpuMatrix,
    wo: GpuMatrix,
    norm2: GpuMatrix,
    wg: GpuMatrix,
    wu: GpuMatrix,
    wd: GpuMatrix,
}

/// AdamW moment (`m` or `v`) buffers for the seven **trainable** projections of
/// one block. The two RMSNorm gains are frozen (the resident backward produces no
/// gain gradient), so they have no optimizer state.
struct OptState {
    wq: GpuMatrix,
    wk: GpuMatrix,
    wv: GpuMatrix,
    wo: GpuMatrix,
    wg: GpuMatrix,
    wu: GpuMatrix,
    wd: GpuMatrix,
}

/// A [`SciAgentModel`] mirrored into VRAM as resident `scirust-gpu` matrices, so
/// the whole decoder runs on the **fully-resident `GpuChain` path** — the one
/// that beats the per-op tape path (`attach_gpu`) ~4× on the Jetson Thor, because
/// nothing leaves VRAM between ops.
///
/// This is the bridge from the real model to the resident kernels: every weight
/// (`embedding`, each block's RMSNorm gains and q/k/v/o + gate/up/down `Linear`s,
/// the final norm) is uploaded once, and [`Self::forward`] runs
/// [`GpuChain::gqa_model_forward`]. The layouts match exactly — `Linear` stores
/// `[in, out]` and computes `x·W`, which is what `GqaBlockWeights` expects, so no
/// transpose is needed.
///
/// **Tied-embedding models only** (the resident path uses `E` as the LM head).
pub struct ResidentModel {
    chain: GpuChain,
    embedding: GpuMatrix,
    final_norm: GpuMatrix,
    blocks: Vec<ResidentBlock>,
    n_heads: usize,
    n_kv_heads: usize,
    theta: f32,
    eps: f32,
    causal: bool,
    vocab: usize,
    // AdamW state (zero-initialised), one pair per trainable weight, + step count.
    m_embedding: GpuMatrix,
    v_embedding: GpuMatrix,
    m_blocks: Vec<OptState>,
    v_blocks: Vec<OptState>,
    step: u32,
}

impl ResidentModel {
    /// Upload every weight of `model` to VRAM. Returns `None` if no GPU adapter
    /// is available. Panics if the model is not tied-embedding.
    pub fn from_model(model: &SciAgentModel) -> Option<Self> {
        assert!(
            model.config.tie_embeddings,
            "ResidentModel requires a tied-embedding model (the resident path uses E as the LM head)"
        );
        let chain = GpuChain::new()?;
        let up = |t: &Tensor| chain.upload(&t.data, t.rows, t.cols);
        let zeros =
            |m: &GpuMatrix| chain.upload(&vec![0.0f32; m.rows() * m.cols()], m.rows(), m.cols());
        let embedding = up(&model.embed.weight);
        let final_norm = up(&model.rms_final.weight);
        let blocks: Vec<ResidentBlock> = model
            .layers
            .iter()
            .map(|l| ResidentBlock {
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
        // Zero-initialised AdamW moment buffers for each trainable weight.
        let opt_of = |b: &ResidentBlock| OptState {
            wq: zeros(&b.wq),
            wk: zeros(&b.wk),
            wv: zeros(&b.wv),
            wo: zeros(&b.wo),
            wg: zeros(&b.wg),
            wu: zeros(&b.wu),
            wd: zeros(&b.wd),
        };
        let m_embedding = zeros(&embedding);
        let v_embedding = zeros(&embedding);
        let m_blocks = blocks.iter().map(&opt_of).collect();
        let v_blocks = blocks.iter().map(&opt_of).collect();
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
            m_embedding,
            v_embedding,
            m_blocks,
            v_blocks,
            step: 0,
        })
    }

    /// Name of the underlying GPU adapter.
    pub fn adapter_name(&self) -> &str {
        self.chain.adapter_name()
    }

    /// Borrowed `GqaBlockWeights` views over the resident block matrices.
    fn block_views(&self) -> Vec<GqaBlockWeights<'_>> {
        self.blocks
            .iter()
            .map(|b| GqaBlockWeights {
                norm1: &b.norm1,
                wq: &b.wq,
                wk: &b.wk,
                wv: &b.wv,
                wo: &b.wo,
                norm2: &b.norm2,
                wg: &b.wg,
                wu: &b.wu,
                wd: &b.wd,
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                theta: self.theta,
            })
            .collect()
    }

    /// Resident forward `tokens → logits`: returns the `tokens.len() × vocab`
    /// logit matrix (row-major), computed entirely on the GPU and downloaded.
    /// Single sequence (`tokens.len()` = sequence length).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let blocks = self.block_views();
        let mw = GqaModelWeights {
            embedding: &self.embedding,
            blocks: &blocks,
            final_norm: &self.final_norm,
        };
        let logits = self
            .chain
            .gqa_model_forward(tokens, &mw, self.eps, self.causal)
            .expect("resident forward");
        self.chain.download(&logits).expect("download logits")
    }

    /// Cross-entropy loss of the resident forward on `(tokens, targets)`.
    pub fn loss(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        let logits = self.forward(tokens);
        scirust_gpu::ops::cpu_cross_entropy(&logits, targets, tokens.len(), self.vocab)
    }

    /// One **resident AdamW training step** on `(tokens, targets)`: forward →
    /// cross-entropy grad → full backward → AdamW update of every trainable
    /// weight (the tied embedding and each block's seven projections; the RMSNorm
    /// gains are frozen, as the resident backward produces no gain gradient),
    /// entirely in VRAM. Returns the **pre-update** cross-entropy loss.
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
        // Forward + backward; the borrowed weight views drop with this scope, so
        // the AdamW updates below can borrow the same fields again.
        let (loss, grads) = {
            let blocks = self.block_views();
            let mw = GqaModelWeights {
                embedding: &self.embedding,
                blocks: &blocks,
                final_norm: &self.final_norm,
            };
            let logits = self
                .chain
                .gqa_model_forward(tokens, &mw, self.eps, self.causal)
                .expect("resident forward");
            let host = self.chain.download(&logits).expect("download logits");
            let loss =
                scirust_gpu::ops::cpu_cross_entropy(&host, targets, tokens.len(), self.vocab);
            let dl = self
                .chain
                .cross_entropy_grad(&logits, targets)
                .expect("cross-entropy grad");
            let grads = self
                .chain
                .gqa_model_backward(tokens, &mw, &dl, self.eps, self.causal)
                .expect("resident backward");
            (loss, grads)
        };

        // AdamW updates — param/m/v buffers are mutated in place on the device.
        let step = self.step;
        let adam = |p: &GpuMatrix, g: &GpuMatrix, m: &GpuMatrix, v: &GpuMatrix| {
            self.chain
                .adamw_step(p, g, m, v, lr, betas, adam_eps, weight_decay, step)
                .expect("adamw step");
        };
        adam(
            &self.embedding,
            &grads.d_embedding,
            &self.m_embedding,
            &self.v_embedding,
        );
        for (i, bg) in grads.blocks.iter().enumerate()
        {
            let (b, m, v) = (&self.blocks[i], &self.m_blocks[i], &self.v_blocks[i]);
            adam(&b.wq, &bg.dwq, &m.wq, &v.wq);
            adam(&b.wk, &bg.dwk, &m.wk, &v.wk);
            adam(&b.wv, &bg.dwv, &m.wv, &v.wv);
            adam(&b.wo, &bg.dwo, &m.wo, &v.wo);
            adam(&b.wg, &bg.dwg, &m.wg, &v.wg);
            adam(&b.wu, &bg.dwu, &m.wu, &v.wu);
            adam(&b.wd, &bg.dwd, &m.wd, &v.wd);
        }
        loss
    }

    /// Write the (possibly trained) resident weights back into `model`, replacing
    /// each host `Tensor`. Lets a resident training run's result flow back to the
    /// `SciAgentModel` for checkpointing or CPU inference.
    pub fn sync_to_model(&self, model: &mut SciAgentModel) {
        let dl = |m: &GpuMatrix| {
            Tensor::from_vec(
                self.chain.download(m).expect("download weight"),
                m.rows(),
                m.cols(),
            )
        };
        model.embed.weight = dl(&self.embedding);
        model.rms_final.weight = dl(&self.final_norm);
        for (l, b) in model.layers.iter_mut().zip(&self.blocks)
        {
            l.rms_attn.weight = dl(&b.norm1);
            l.attn.w_q.weight = dl(&b.wq);
            l.attn.w_k.weight = dl(&b.wk);
            l.attn.w_v.weight = dl(&b.wv);
            l.attn.w_o.weight = dl(&b.wo);
            l.rms_ffn.weight = dl(&b.norm2);
            l.ffn.gate.weight = dl(&b.wg);
            l.ffn.up.weight = dl(&b.wu);
            l.ffn.down.weight = dl(&b.wd);
        }
    }
}
