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

/// Index of the maximum element (first on ties) — the greedy next-token pick.
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate()
    {
        if v > best_v
        {
            best_v = v;
            best = i;
        }
    }
    best
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

    /// **Greedy autoregressive generation** on the resident path: from `prompt`,
    /// decode `max_new` tokens by repeatedly running the resident forward and
    /// taking the `argmax` of the last position's logits. Deterministic (no
    /// sampling), matching a CPU greedy decode of the same weights. Returns the
    /// full sequence (`prompt` followed by the generated tokens).
    ///
    /// Each step re-runs the whole-sequence forward (no KV cache yet), so this is
    /// `O(n²)` in the sequence length — fine for short on-device generations; a
    /// resident KV cache is the throughput follow-up. Closes the on-device loop:
    /// pretrain / fine-tune → merge → **generate**, all on the Thor.
    pub fn generate(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut toks = prompt.to_vec();
        for _ in 0..max_new
        {
            if toks.is_empty()
            {
                break;
            }
            let logits = self.forward(&toks);
            let seq = toks.len();
            let last = &logits[(seq - 1) * self.vocab..seq * self.vocab];
            let next = argmax(last) as u32;
            toks.push(next);
        }
        toks
    }

    /// **KV-cached greedy generation** — the `O(n)`-per-token equivalent of
    /// [`Self::generate`]. Where `generate` re-runs the whole-sequence forward at
    /// every step (`O(n²)` total), this processes only the newly-added token: its
    /// per-layer roped key and raw value are appended to a growing per-layer cache,
    /// and the single query attends over the cached keys/values. The arithmetic is
    /// [`Self::forward`] restricted to the last row, so this yields the **same
    /// tokens** as [`Self::generate`] — in fact the naive one-cell-per-thread GEMM
    /// reduces each output independently of the matrix height, so the last row's
    /// logits are reproduced bit-for-bit (verified token-for-token in
    /// `tests/gpu_parity.rs`).
    ///
    /// Caches live on the host as `n_layers` growing `Vec<f32>` (roped K, raw V,
    /// `kv_dim` columns), re-uploaded per step; the compute stays resident.
    /// Returns the full sequence (`prompt` followed by the generated tokens).
    pub fn generate_cached(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        if prompt.is_empty()
        {
            return Vec::new();
        }
        let n = self.blocks.len();
        // Per-layer host caches of roped keys / raw values (row-major, kv_dim wide).
        let mut kcache: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut vcache: Vec<Vec<f32>> = vec![Vec::new(); n];

        // Prefill: feed the prompt one token at a time, keeping the last logits.
        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate()
        {
            last_logits = self.decode_step(tok, pos, &mut kcache, &mut vcache);
        }

        // Decode: argmax → append → step, growing the same caches. The final
        // token needs no further forward (nothing is generated after it).
        let mut toks = prompt.to_vec();
        for i in 0..max_new
        {
            let next = argmax(&last_logits) as u32;
            let pos = toks.len();
            toks.push(next);
            if i + 1 < max_new
            {
                last_logits = self.decode_step(next, pos, &mut kcache, &mut vcache);
            }
        }
        toks
    }

    /// One **incremental decode step**: embed `token` at absolute position `pos`,
    /// run the full stack using the per-layer KV caches (appending this token's
    /// roped key and raw value to each), and return the `vocab`-wide logits for
    /// this position. `kcache[l]`/`vcache[l]` are the layer-`l` caches (row-major,
    /// `kv_dim` columns), each grown by one row here. Reproduces
    /// [`GpuChain::gqa_transformer_block`] restricted to the single new row.
    fn decode_step(
        &self,
        token: u32,
        pos: usize,
        kcache: &mut [Vec<f32>],
        vcache: &mut [Vec<f32>],
    ) -> Vec<f32> {
        let chain = &self.chain;
        let d_model = self.embedding.cols();
        let dh = d_model / self.n_heads;
        let kv_dim = self.n_kv_heads * dh;

        // Embed the single new token → [1 × d_model].
        let mut x = chain.embed(&[token], &self.embedding).expect("embed");
        for (l, b) in self.blocks.iter().enumerate()
        {
            // Attention sub-block (pre-norm + residual).
            let xn = chain.rms_norm(&x, &b.norm1, self.eps).expect("norm1");
            let q = chain.matmul(&xn, &b.wq).expect("wq"); // [1 × d_model]
            let k = chain.matmul(&xn, &b.wk).expect("wk"); // [1 × kv_dim]
            let v = chain.matmul(&xn, &b.wv).expect("wv"); // [1 × kv_dim]
            // RoPE the single new row at its absolute position (seq_len 1, offset
            // pos ⇒ position = pos). Each width feeds its own frequency schedule,
            // exactly as `gqa_attention` ropes the full-width q/k.
            let qr = chain.rope(&q, 1, pos, self.theta).expect("rope q");
            let kr = chain.rope(&k, 1, pos, self.theta).expect("rope k");
            // Append this step's roped key and raw value to the layer caches.
            kcache[l].extend_from_slice(&chain.download(&kr).expect("download kr"));
            vcache[l].extend_from_slice(&chain.download(&v).expect("download v"));
            let seq = kcache[l].len() / kv_dim;
            let kmat = chain.upload(&kcache[l], seq, kv_dim);
            let vmat = chain.upload(&vcache[l], seq, kv_dim);
            // Single query attends over all cached keys/values (all ≤ pos ⇒ no mask).
            let ctx = self.incr_attention(&qr, &kmat, &vmat, dh);
            let attn_out = chain.matmul(&ctx, &b.wo).expect("wo");
            x = chain.add(&x, &attn_out).expect("attn residual");
            // MLP sub-block (pre-norm + residual).
            let hn = chain.rms_norm(&x, &b.norm2, self.eps).expect("norm2");
            let mlp = chain.swiglu_mlp(&hn, &b.wg, &b.wu, &b.wd).expect("mlp");
            x = chain.add(&x, &mlp).expect("mlp residual");
        }
        // Final norm + tied LM head → [1 × vocab].
        let normed = chain
            .rms_norm(&x, &self.final_norm, self.eps)
            .expect("final norm");
        let logits = chain
            .matmul_t(&normed, &self.embedding, false, true)
            .expect("logits");
        chain.download(&logits).expect("download logits")
    }

    /// Incremental analogue of [`GpuChain::gqa_attention`] for a **single query
    /// row**: `qr` is `1 × d_model` (already RoPE'd), attending over the cached
    /// keys `kmat` and values `vmat` (`seq × kv_dim`; keys already RoPE'd, values
    /// raw). No causal mask is needed — every cached position is ≤ the query's, so
    /// all are visible. Returns the `1 × d_model` concatenated context (the caller
    /// applies `w_o`), summing the placed per-head slices exactly as the full path.
    fn incr_attention(
        &self,
        qr: &GpuMatrix,
        kmat: &GpuMatrix,
        vmat: &GpuMatrix,
        dh: usize,
    ) -> GpuMatrix {
        let chain = &self.chain;
        let d_model = qr.cols();
        let repeat = self.n_heads / self.n_kv_heads;
        let mut out: Option<GpuMatrix> = None;
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = chain.slice_cols(qr, head * dh, dh).expect("slice q");
            let ks = chain.slice_cols(kmat, kv * dh, dh).expect("slice k");
            let vs = chain.slice_cols(vmat, kv * dh, dh).expect("slice v");
            let ctx = chain.attention(&qs, &ks, &vs, false).expect("attn head"); // scale 1/√dh
            let padded = chain.place_cols(&ctx, head * dh, d_model).expect("place");
            out = Some(match out
            {
                None => padded,
                Some(acc) => chain.add(&acc, &padded).expect("head sum"),
            });
        }
        // n_heads ≥ 1 for any real config, so `out` is always Some.
        out.expect("gqa: n_heads must be ≥ 1")
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
        let t0 = std::time::Instant::now();
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
                let done = (step - cfg.start_step) * s;
                let secs = t0.elapsed().as_secs_f64().max(1e-9);
                let tps = done as f64 / secs;
                println!(
                    "[resident step {step:>6}] loss {loss:>9.4} | lr {lr:.3e} | {tps:>8.0} tok/s"
                );
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

/// LoRA configuration for [`ResidentLoraModel`].
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    /// Low-rank dimension `r`.
    pub rank: usize,
    /// LoRA `alpha`; the adapter is scaled by `alpha / rank`.
    pub alpha: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
        }
    }
}

/// One projection's LoRA adapter: the low-rank factors `A` (`in×r`), `B` (`r×out`)
/// and their AdamW moment buffers. `B` starts at **zero** so the adapter is a
/// no-op at init (`W + scaling·A·B = W`); `A` is seeded deterministically.
struct LoraAdapter {
    a: GpuMatrix,
    b: GpuMatrix,
    m_a: GpuMatrix,
    v_a: GpuMatrix,
    m_b: GpuMatrix,
    v_b: GpuMatrix,
}

/// LoRA adapters for one block's four **attention** projections (the standard
/// LoRA target set). The MLP projections and the norms stay frozen.
struct BlockAdapters {
    wq: LoraAdapter,
    wk: LoraAdapter,
    wv: LoraAdapter,
    wo: LoraAdapter,
}

/// Effective (base + adapter) attention weights for one block, materialised for a
/// forward/backward pass: `W_eff = W + scaling·(A·B)`.
struct EffBlock {
    wq: GpuMatrix,
    wk: GpuMatrix,
    wv: GpuMatrix,
    wo: GpuMatrix,
}

/// **Resident LoRA fine-tuning** of a [`SciAgentModel`]: the whole base model is
/// mirrored into VRAM and **frozen**; only small LoRA adapters on the four
/// attention projections train. Far less optimizer state and far fewer trainable
/// parameters than full-weight training — the natural on-device fine-tuning fit
/// for the resident path.
///
/// Rather than change the validated `GpuChain` forward/backward, it uses the
/// **effective-weight** identity: with `W_eff = W + scaling·A·B`, running the
/// ordinary [`GpuChain::gqa_model_backward`] yields `dW_eff = ∂L/∂W_eff`, and the
/// adapter gradients follow exactly as `dA = scaling·dW_eff·Bᵀ` and
/// `dB = scaling·Aᵀ·dW_eff`. So the same gradient-checked full-model kernels drive
/// LoRA training; only `A`/`B` receive an AdamW step, the base never moves.
///
/// **Tied-embedding models only.**
pub struct ResidentLoraModel {
    chain: GpuChain,
    embedding: GpuMatrix,
    final_norm: GpuMatrix,
    blocks: Vec<ResidentBlock>,
    adapters: Vec<BlockAdapters>,
    n_heads: usize,
    n_kv_heads: usize,
    theta: f32,
    eps: f32,
    causal: bool,
    vocab: usize,
    scaling: f32,
    step: u32,
}

impl ResidentLoraModel {
    /// Upload `model` to VRAM (frozen) and attach zero-initialised LoRA adapters
    /// of rank `cfg.rank` to each block's q/k/v/o projections. Returns `None` if
    /// no GPU adapter is available. Panics if the model is not tied-embedding.
    pub fn from_model(model: &SciAgentModel, cfg: LoraConfig) -> Option<Self> {
        assert!(
            model.config.tie_embeddings,
            "ResidentLoraModel requires a tied-embedding model"
        );
        assert!(cfg.rank >= 1, "LoRA rank must be ≥ 1");
        let chain = GpuChain::new()?;
        let up = |t: &Tensor| chain.upload(&t.data, t.rows, t.cols);
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
        let r = cfg.rank;
        let adapter_of = |w: &GpuMatrix| -> LoraAdapter {
            let (in_f, out) = (w.rows(), w.cols());
            // Deterministic A seed: a low-discrepancy pattern in [-s, s],
            // s = 1/√in (keeps the initial update well-scaled). B = 0.
            let s = (1.0 / in_f as f32).sqrt();
            let a: Vec<f32> = (0..in_f * r)
                .map(|i| s * (((i as f32) * 0.618_034).fract() * 2.0 - 1.0))
                .collect();
            let ga = chain.upload(&a, in_f, r);
            let gb = chain.upload(&vec![0.0f32; r * out], r, out);
            let z = |rows: usize, cols: usize| chain.upload(&vec![0.0f32; rows * cols], rows, cols);
            LoraAdapter {
                a: ga,
                b: gb,
                m_a: z(in_f, r),
                v_a: z(in_f, r),
                m_b: z(r, out),
                v_b: z(r, out),
            }
        };
        let adapters: Vec<BlockAdapters> = blocks
            .iter()
            .map(|b| BlockAdapters {
                wq: adapter_of(&b.wq),
                wk: adapter_of(&b.wk),
                wv: adapter_of(&b.wv),
                wo: adapter_of(&b.wo),
            })
            .collect();
        Some(Self {
            chain,
            embedding,
            final_norm,
            blocks,
            adapters,
            n_heads: model.config.n_heads,
            n_kv_heads: model.config.n_kv_heads,
            theta: model.config.rope_theta,
            eps: model.config.eps,
            causal: true,
            vocab: model.config.vocab_size,
            scaling: cfg.alpha / cfg.rank as f32,
            step: 0,
        })
    }

    /// Name of the underlying GPU adapter.
    pub fn adapter_name(&self) -> &str {
        self.chain.adapter_name()
    }

    /// `W_eff = base + scaling·(A·B)`, resident.
    fn effective(&self, base: &GpuMatrix, ad: &LoraAdapter) -> GpuMatrix {
        let ab = self.chain.matmul(&ad.a, &ad.b).expect("A·B");
        let ab_s = self
            .chain
            .scale_causal_mask(&ab, self.scaling, false)
            .expect("scale");
        self.chain.add(base, &ab_s).expect("W + ΔW")
    }

    /// Materialise every block's effective attention weights.
    fn effective_blocks(&self) -> Vec<EffBlock> {
        self.blocks
            .iter()
            .zip(&self.adapters)
            .map(|(b, ad)| EffBlock {
                wq: self.effective(&b.wq, &ad.wq),
                wk: self.effective(&b.wk, &ad.wk),
                wv: self.effective(&b.wv, &ad.wv),
                wo: self.effective(&b.wo, &ad.wo),
            })
            .collect()
    }

    /// Borrowed `GqaBlockWeights` over the effective attention weights (`eff`) and
    /// the frozen norms / MLP.
    fn block_views<'a>(&'a self, eff: &'a [EffBlock]) -> Vec<GqaBlockWeights<'a>> {
        self.blocks
            .iter()
            .zip(eff)
            .map(|(b, e)| GqaBlockWeights {
                norm1: &b.norm1,
                wq: &e.wq,
                wk: &e.wk,
                wv: &e.wv,
                wo: &e.wo,
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

    /// Resident forward `tokens → logits` with the current adapters applied.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let eff = self.effective_blocks();
        let blocks = self.block_views(&eff);
        let mw = GqaModelWeights {
            embedding: &self.embedding,
            blocks: &blocks,
            final_norm: &self.final_norm,
        };
        let logits = self
            .chain
            .gqa_model_forward(tokens, &mw, self.eps, self.causal)
            .expect("resident lora forward");
        self.chain.download(&logits).expect("download logits")
    }

    /// Cross-entropy loss of the resident LoRA forward on `(tokens, targets)`.
    pub fn loss(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        let logits = self.forward(tokens);
        scirust_gpu::ops::cpu_cross_entropy(&logits, targets, tokens.len(), self.vocab)
    }

    /// One **resident LoRA fine-tuning step**: forward with `W_eff` → cross-entropy
    /// grad → full backward → derive the adapter gradients from `dW_eff` and AdamW
    /// them. Only the q/k/v/o LoRA factors move; the base model is frozen. Returns
    /// the pre-update cross-entropy loss.
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
        let (loss, grads) = {
            let eff = self.effective_blocks();
            let blocks = self.block_views(&eff);
            let mw = GqaModelWeights {
                embedding: &self.embedding,
                blocks: &blocks,
                final_norm: &self.final_norm,
            };
            let logits = self
                .chain
                .gqa_model_forward(tokens, &mw, self.eps, self.causal)
                .expect("resident lora forward");
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

        // Derive adapter grads from dW_eff and AdamW-update A, B in place.
        let (step, scaling) = (self.step, self.scaling);
        let update = |ad: &LoraAdapter, dweff: &GpuMatrix| {
            // dA = scaling · dW_eff · Bᵀ   (in×r)
            let da = self
                .chain
                .matmul_t(dweff, &ad.b, false, true)
                .expect("dW_eff·Bᵀ");
            let da = self
                .chain
                .scale_causal_mask(&da, scaling, false)
                .expect("scale dA");
            // dB = scaling · Aᵀ · dW_eff   (r×out)
            let db = self
                .chain
                .matmul_t(&ad.a, dweff, true, false)
                .expect("Aᵀ·dW_eff");
            let db = self
                .chain
                .scale_causal_mask(&db, scaling, false)
                .expect("scale dB");
            self.chain
                .adamw_step(
                    &ad.a,
                    &da,
                    &ad.m_a,
                    &ad.v_a,
                    lr,
                    betas,
                    adam_eps,
                    weight_decay,
                    step,
                )
                .expect("adamw A");
            self.chain
                .adamw_step(
                    &ad.b,
                    &db,
                    &ad.m_b,
                    &ad.v_b,
                    lr,
                    betas,
                    adam_eps,
                    weight_decay,
                    step,
                )
                .expect("adamw B");
        };
        for (i, bg) in grads.blocks.iter().enumerate()
        {
            let ad = &self.adapters[i];
            update(&ad.wq, &bg.dwq);
            update(&ad.wk, &bg.dwk);
            update(&ad.wv, &bg.dwv);
            update(&ad.wo, &bg.dwo);
        }
        loss
    }

    /// **Merge** the adapters into the base and write the result back into `model`
    /// (each attention projection becomes `W + scaling·A·B`; everything else is the
    /// frozen base). Produces a plain fine-tuned `SciAgentModel` — no LoRA runtime
    /// needed for inference.
    pub fn sync_to_model(&self, model: &mut SciAgentModel) {
        let dl = |m: &GpuMatrix| {
            Tensor::from_vec(
                self.chain.download(m).expect("download"),
                m.rows(),
                m.cols(),
            )
        };
        let merged = |base: &GpuMatrix, ad: &LoraAdapter| dl(&self.effective(base, ad));
        model.embed.weight = dl(&self.embedding);
        model.rms_final.weight = dl(&self.final_norm);
        for (l, (b, ad)) in model
            .layers
            .iter_mut()
            .zip(self.blocks.iter().zip(&self.adapters))
        {
            l.rms_attn.weight = dl(&b.norm1);
            l.attn.w_q.weight = merged(&b.wq, &ad.wq);
            l.attn.w_k.weight = merged(&b.wk, &ad.wk);
            l.attn.w_v.weight = merged(&b.wv, &ad.wv);
            l.attn.w_o.weight = merged(&b.wo, &ad.wo);
            l.rms_ffn.weight = dl(&b.norm2);
            l.ffn.gate.weight = dl(&b.wg);
            l.ffn.up.weight = dl(&b.wu);
            l.ffn.down.weight = dl(&b.wd);
        }
    }
}

/// LoRA rank for [`ResidentDoraModel`]. (DoRA folds the LoRA scaling into the
/// learnable magnitude, so there is no separate `alpha`.)
#[derive(Debug, Clone, Copy)]
pub struct DoraConfig {
    /// Low-rank dimension `r` of the direction update.
    pub rank: usize,
}

impl Default for DoraConfig {
    fn default() -> Self {
        Self { rank: 8 }
    }
}

/// One projection's DoRA adapter: the low-rank direction factors `A` (`in×r`),
/// `B` (`r×out`, zero-init), the per-row magnitude `m` (`in×1`, init = the base's
/// row norms), and AdamW moment buffers for each.
struct DoraAdapter {
    a: GpuMatrix,
    b: GpuMatrix,
    mag: GpuMatrix,
    m_a: GpuMatrix,
    v_a: GpuMatrix,
    m_b: GpuMatrix,
    v_b: GpuMatrix,
    m_mag: GpuMatrix,
    v_mag: GpuMatrix,
}

/// DoRA adapters for one block's four attention projections.
struct BlockDoraAdapters {
    wq: DoraAdapter,
    wk: DoraAdapter,
    wv: DoraAdapter,
    wo: DoraAdapter,
}

/// **Resident DoRA fine-tuning** of a [`SciAgentModel`] — the DoRA analogue of
/// [`ResidentLoraModel`]. The base is frozen; each attention projection gets a
/// DoRA adapter (`W' = m ⊙ (W₀ + A·B)/‖W₀ + A·B‖_row`) and only `A`/`B`/`m` train.
/// DoRA separates *magnitude* from *direction*, which tends to fine-tune better
/// than plain LoRA at the same rank.
///
/// Reuses the validated full-model kernels via the effective-weight identity:
/// materialise `W'` per projection ([`GpuChain::dora_effective_weight`]), run the
/// ordinary [`GpuChain::gqa_model_backward`] to get `dW' = ∂L/∂W'`, then map that
/// to the adapter grads with [`GpuChain::dora_weight_grads`]. Only the adapters
/// take an AdamW step. **Tied-embedding models only.**
pub struct ResidentDoraModel {
    chain: GpuChain,
    embedding: GpuMatrix,
    final_norm: GpuMatrix,
    blocks: Vec<ResidentBlock>,
    adapters: Vec<BlockDoraAdapters>,
    n_heads: usize,
    n_kv_heads: usize,
    theta: f32,
    eps: f32,
    causal: bool,
    vocab: usize,
    step: u32,
}

impl ResidentDoraModel {
    /// Upload `model` (frozen) and attach zero-perturbation DoRA adapters of rank
    /// `cfg.rank` to each block's q/k/v/o projection. `B = 0` and `m = ‖W₀‖_row`,
    /// so the effective weight starts exactly at the base. `None` if no adapter.
    pub fn from_model(model: &SciAgentModel, cfg: DoraConfig) -> Option<Self> {
        assert!(
            model.config.tie_embeddings,
            "ResidentDoraModel requires a tied-embedding model"
        );
        assert!(cfg.rank >= 1, "DoRA rank must be ≥ 1");
        let chain = GpuChain::new()?;
        let up = |t: &Tensor| chain.upload(&t.data, t.rows, t.cols);
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
        let r = cfg.rank;
        let z = |rows: usize, cols: usize| chain.upload(&vec![0.0f32; rows * cols], rows, cols);
        let adapter_of = |t: &Tensor| -> DoraAdapter {
            let (in_f, out) = (t.rows, t.cols);
            // m = per-row L2 norm of the base (guarded like the rsqrt kernel), so
            // W' = base at init.
            let mag: Vec<f32> = (0..in_f)
                .map(|p| {
                    let ss: f32 = (0..out).map(|o| t.data[p * out + o].powi(2)).sum();
                    ss.max(1e-12).sqrt()
                })
                .collect();
            // A: deterministic low-discrepancy seed in ±1/√in; B = 0.
            let s = (1.0 / in_f as f32).sqrt();
            let a: Vec<f32> = (0..in_f * r)
                .map(|i| s * (((i as f32) * 0.618_034).fract() * 2.0 - 1.0))
                .collect();
            DoraAdapter {
                a: chain.upload(&a, in_f, r),
                b: z(r, out),
                mag: chain.upload(&mag, in_f, 1),
                m_a: z(in_f, r),
                v_a: z(in_f, r),
                m_b: z(r, out),
                v_b: z(r, out),
                m_mag: z(in_f, 1),
                v_mag: z(in_f, 1),
            }
        };
        let adapters: Vec<BlockDoraAdapters> = model
            .layers
            .iter()
            .map(|l| BlockDoraAdapters {
                wq: adapter_of(&l.attn.w_q.weight),
                wk: adapter_of(&l.attn.w_k.weight),
                wv: adapter_of(&l.attn.w_v.weight),
                wo: adapter_of(&l.attn.w_o.weight),
            })
            .collect();
        Some(Self {
            chain,
            embedding,
            final_norm,
            blocks,
            adapters,
            n_heads: model.config.n_heads,
            n_kv_heads: model.config.n_kv_heads,
            theta: model.config.rope_theta,
            eps: model.config.eps,
            causal: true,
            vocab: model.config.vocab_size,
            step: 0,
        })
    }

    /// Name of the underlying GPU adapter.
    pub fn adapter_name(&self) -> &str {
        self.chain.adapter_name()
    }

    /// Materialise every block's effective (DoRA-adapted) attention weights.
    fn effective_blocks(&self) -> Vec<EffBlock> {
        let eff = |base: &GpuMatrix, ad: &DoraAdapter| {
            self.chain
                .dora_effective_weight(base, &ad.a, &ad.b, &ad.mag)
                .expect("dora effective weight")
        };
        self.blocks
            .iter()
            .zip(&self.adapters)
            .map(|(b, ad)| EffBlock {
                wq: eff(&b.wq, &ad.wq),
                wk: eff(&b.wk, &ad.wk),
                wv: eff(&b.wv, &ad.wv),
                wo: eff(&b.wo, &ad.wo),
            })
            .collect()
    }

    fn block_views<'a>(&'a self, eff: &'a [EffBlock]) -> Vec<GqaBlockWeights<'a>> {
        self.blocks
            .iter()
            .zip(eff)
            .map(|(b, e)| GqaBlockWeights {
                norm1: &b.norm1,
                wq: &e.wq,
                wk: &e.wk,
                wv: &e.wv,
                wo: &e.wo,
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

    /// Resident forward `tokens → logits` with the current DoRA adapters applied.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let eff = self.effective_blocks();
        let blocks = self.block_views(&eff);
        let mw = GqaModelWeights {
            embedding: &self.embedding,
            blocks: &blocks,
            final_norm: &self.final_norm,
        };
        let logits = self
            .chain
            .gqa_model_forward(tokens, &mw, self.eps, self.causal)
            .expect("resident dora forward");
        self.chain.download(&logits).expect("download logits")
    }

    /// Cross-entropy loss of the resident DoRA forward on `(tokens, targets)`.
    pub fn loss(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        let logits = self.forward(tokens);
        scirust_gpu::ops::cpu_cross_entropy(&logits, targets, tokens.len(), self.vocab)
    }

    /// One **resident DoRA fine-tuning step**: forward with `W'` → cross-entropy
    /// grad → full backward (`dW' = ∂L/∂W'` per projection) → map to adapter grads
    /// (`dA`/`dB`/`dm`) and AdamW them. Only q/k/v/o adapters move; the base is
    /// frozen. Returns the pre-update cross-entropy loss.
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
        let (loss, grads) = {
            let eff = self.effective_blocks();
            let blocks = self.block_views(&eff);
            let mw = GqaModelWeights {
                embedding: &self.embedding,
                blocks: &blocks,
                final_norm: &self.final_norm,
            };
            let logits = self
                .chain
                .gqa_model_forward(tokens, &mw, self.eps, self.causal)
                .expect("resident dora forward");
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

        let step = self.step;
        let update = |base: &GpuMatrix, ad: &DoraAdapter, dweff: &GpuMatrix| {
            let (da, db, dm) = self
                .chain
                .dora_weight_grads(base, &ad.a, &ad.b, &ad.mag, dweff)
                .expect("dora weight grads");
            let adam = |p: &GpuMatrix, g: &GpuMatrix, mo: &GpuMatrix, vo: &GpuMatrix| {
                self.chain
                    .adamw_step(p, g, mo, vo, lr, betas, adam_eps, weight_decay, step)
                    .expect("adamw");
            };
            adam(&ad.a, &da, &ad.m_a, &ad.v_a);
            adam(&ad.b, &db, &ad.m_b, &ad.v_b);
            adam(&ad.mag, &dm, &ad.m_mag, &ad.v_mag);
        };
        for (i, bg) in grads.blocks.iter().enumerate()
        {
            let (b, ad) = (&self.blocks[i], &self.adapters[i]);
            update(&b.wq, &ad.wq, &bg.dwq);
            update(&b.wk, &ad.wk, &bg.dwk);
            update(&b.wv, &ad.wv, &bg.dwv);
            update(&b.wo, &ad.wo, &bg.dwo);
        }
        loss
    }

    /// **Merge** the DoRA adapters into the base and write the result into `model`
    /// (each attention projection becomes its effective weight `W'`; everything
    /// else is the frozen base). Produces a plain fine-tuned `SciAgentModel`.
    pub fn sync_to_model(&self, model: &mut SciAgentModel) {
        let dl = |m: &GpuMatrix| {
            Tensor::from_vec(
                self.chain.download(m).expect("download"),
                m.rows(),
                m.cols(),
            )
        };
        let merged = |base: &GpuMatrix, ad: &DoraAdapter| {
            dl(&self
                .chain
                .dora_effective_weight(base, &ad.a, &ad.b, &ad.mag)
                .expect("dora effective weight"))
        };
        model.embed.weight = dl(&self.embedding);
        model.rms_final.weight = dl(&self.final_norm);
        for (l, (b, ad)) in model
            .layers
            .iter_mut()
            .zip(self.blocks.iter().zip(&self.adapters))
        {
            l.rms_attn.weight = dl(&b.norm1);
            l.attn.w_q.weight = merged(&b.wq, &ad.wq);
            l.attn.w_k.weight = merged(&b.wk, &ad.wk);
            l.attn.w_v.weight = merged(&b.wv, &ad.wv);
            l.attn.w_o.weight = merged(&b.wo, &ad.wo);
            l.rms_ffn.weight = dl(&b.norm2);
            l.ffn.gate.weight = dl(&b.wg);
            l.ffn.up.weight = dl(&b.wu);
            l.ffn.down.weight = dl(&b.wd);
        }
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
