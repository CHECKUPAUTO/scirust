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
use scirust_core::autodiff::scheduler::LrSchedule;
use scirust_gpu::{GpuChain, GpuMatrix, GqaBlockWeights, GqaModelWeights};

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;
use crate::train::checkpoint::{CheckpointMeta, save_checkpoint};
use crate::train::scheduler::WarmupCosineSchedule;

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

/// AdamW moment (`m` or `v`) buffers for one block's nine trainable weights: the
/// two RMSNorm gains and the seven projections. The resident backward now emits a
/// gain gradient (`rms_norm_gain_backward`), so the norms train too.
struct OptState {
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
    m_final_norm: GpuMatrix,
    v_final_norm: GpuMatrix,
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
            norm1: zeros(&b.norm1),
            wq: zeros(&b.wq),
            wk: zeros(&b.wk),
            wv: zeros(&b.wv),
            wo: zeros(&b.wo),
            norm2: zeros(&b.norm2),
            wg: zeros(&b.wg),
            wu: zeros(&b.wu),
            wd: zeros(&b.wd),
        };
        let m_embedding = zeros(&embedding);
        let v_embedding = zeros(&embedding);
        let m_final_norm = zeros(&final_norm);
        let v_final_norm = zeros(&final_norm);
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
            m_final_norm,
            v_final_norm,
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
    /// weight (the tied embedding, the final RMSNorm gain, and each block's two
    /// RMSNorm gains + seven projections), entirely in VRAM. Returns the
    /// **pre-update** cross-entropy loss.
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
        adam(
            &self.final_norm,
            &grads.d_final_norm,
            &self.m_final_norm,
            &self.v_final_norm,
        );
        for (i, bg) in grads.blocks.iter().enumerate()
        {
            let (b, m, v) = (&self.blocks[i], &self.m_blocks[i], &self.v_blocks[i]);
            adam(&b.norm1, &bg.dnorm1, &m.norm1, &v.norm1);
            adam(&b.wq, &bg.dwq, &m.wq, &v.wq);
            adam(&b.wk, &bg.dwk, &m.wk, &v.wk);
            adam(&b.wv, &bg.dwv, &m.wv, &v.wv);
            adam(&b.wo, &bg.dwo, &m.wo, &v.wo);
            adam(&b.norm2, &bg.dnorm2, &m.norm2, &v.norm2);
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

    /// **Resident next-token pretraining** over a flat `u32` token stream. Slides
    /// a `cfg.seq_len` window (non-overlapping); each step trains on
    /// `inputs = tokens[i..i+seq_len]`, `targets = tokens[i+1..i+seq_len+1]` via
    /// the resident [`Self::train_step`] (forward → cross-entropy grad → backward
    /// → AdamW, entirely in VRAM — the ~4× path). Returns the loss at each step.
    ///
    /// Dataset-agnostic: load any corpus into a `Vec<u32>` (e.g. via the
    /// `train::ShardLoader`) and pass it here. Call [`Self::sync_to_model`]
    /// afterwards to write the trained weights back for checkpointing / inference.
    pub fn train_tokens(&mut self, tokens: &[u32], cfg: &ResidentTrainConfig) -> Vec<f32> {
        let s = cfg.seq_len;
        let mut losses = Vec::new();
        let mut start = 0usize;
        while start + s < tokens.len()
        {
            let inputs = &tokens[start..start + s];
            let targets = &tokens[start + 1..start + s + 1];
            let loss = self.train_step(
                inputs,
                targets,
                cfg.lr,
                cfg.betas,
                cfg.adam_eps,
                cfg.weight_decay,
            );
            losses.push(loss);
            start += s;
        }
        losses
    }

    /// Reset the internal AdamW step counter. Used when resuming a run from a
    /// checkpoint: `from_model` re-uploads the saved weights but zero-inits the
    /// moment buffers (`m`/`v` are not persisted), so the optimizer must restart
    /// at step 0 for its bias correction (`1/(1-βᵗ)`) to be consistent with the
    /// freshly-zeroed moments. The **LR schedule** position is tracked separately
    /// (see [`ResidentPretrainConfig::start_step`]), so the learning rate still
    /// continues from where the run left off.
    pub fn reset_step(&mut self) {
        self.step = 0;
    }

    /// **Production-scale resident pretraining** over a flat `u32` token stream
    /// (load real shards with [`crate::train::dataset::ShardLoader`]), with a
    /// warmup + cosine LR schedule and periodic checkpointing — all on the
    /// fully-resident GPU path.
    ///
    /// Runs `cfg.total_steps − cfg.start_step` optimizer steps, cycling the token
    /// stream as many times as needed (deterministic, in-order — no shuffle, so a
    /// run is bit-reproducible). Each step:
    /// 1. takes the next non-overlapping `cfg.seq_len` window `(inputs, targets)`;
    /// 2. computes `lr = WarmupCosineSchedule::lr_at(step)`;
    /// 3. runs the resident [`Self::train_step`] (fwd → cross-entropy grad → bwd →
    ///    AdamW at that `lr`), entirely in VRAM.
    ///
    /// Every `cfg.save_interval` steps it [`Self::sync_to_model`]s the resident
    /// weights back into `model` and writes a safetensors checkpoint (via
    /// [`save_checkpoint`]) under `cfg.checkpoint_dir/step_N/`, so a long run is
    /// resumable (reload with [`crate::train::checkpoint::load_checkpoint`], build
    /// a fresh `ResidentModel`, and call `pretrain` again with
    /// `start_step = meta.step`). The resident AdamW moments are **not** persisted;
    /// a resumed run restarts them from zero, which the warmup schedule re-absorbs.
    ///
    /// Returns the per-step pre-update loss.
    pub fn pretrain(
        &mut self,
        tokens: &[u32],
        model: &mut SciAgentModel,
        config: &SciAgentConfig,
        cfg: &ResidentPretrainConfig,
    ) -> Vec<f32> {
        let s = cfg.seq_len;
        let mut losses = Vec::new();
        if tokens.len() <= s
        {
            eprintln!(
                "resident pretrain: token stream ({}) shorter than a single window ({}); nothing to do",
                tokens.len(),
                s + 1
            );
            return losses;
        }
        let schedule =
            WarmupCosineSchedule::new(cfg.base_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps);
        let mut step = cfg.start_step;
        let mut cursor = 0usize;
        while step < cfg.total_steps
        {
            // Wrap the corpus (a window needs `s` inputs + 1 shifted target).
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
                println!("[resident step {step:>6}] loss {loss:>9.4} | lr {lr:.3e}");
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

/// Hyper-parameters for [`ResidentModel::train_tokens`].
#[derive(Debug, Clone, Copy)]
pub struct ResidentTrainConfig {
    /// AdamW learning rate.
    pub lr: f32,
    /// AdamW `(β₁, β₂)`.
    pub betas: (f32, f32),
    /// AdamW epsilon.
    pub adam_eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Sequence length of each training window.
    pub seq_len: usize,
}

impl Default for ResidentTrainConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            betas: (0.9, 0.95),
            adam_eps: 1e-8,
            weight_decay: 0.1,
            seq_len: 128,
        }
    }
}

/// Hyper-parameters for [`ResidentModel::pretrain`] — a full production run with
/// a warmup + cosine LR schedule and periodic checkpointing.
#[derive(Debug, Clone)]
pub struct ResidentPretrainConfig {
    /// Peak (post-warmup) AdamW learning rate.
    pub base_lr: f32,
    /// Floor learning rate the cosine decays to.
    pub min_lr: f32,
    /// Linear warmup length, in optimizer steps.
    pub warmup_steps: usize,
    /// Total optimizer steps for the run (also the cosine period end).
    pub total_steps: usize,
    /// Step to start the LR schedule from (for resuming; the AdamW moments still
    /// restart at zero — see [`ResidentModel::reset_step`]).
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

impl Default for ResidentPretrainConfig {
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
