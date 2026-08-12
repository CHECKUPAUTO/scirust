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

use std::collections::HashMap;
use std::path::Path;

use scirust_core::autodiff::reverse::Tensor;
use scirust_core::autodiff::scheduler::LrSchedule;
use scirust_cuda::{CudaChain, CudaF32, CudaMatrix};

use crate::config::SciAgentConfig;
use crate::generate::{SamplingParams, sample_row, seed_to_state};
use crate::model::SciAgentModel;
use crate::train::checkpoint::{CheckpointMeta, save_checkpoint};
use crate::train::dataset::{WINDOW_SPLIT_VERSION, distributed_window_split};
use crate::train::scheduler::WarmupCosineSchedule;

/// SCIAGENT model-math compatibility generation. Version 2 is the B33 transition
/// from full-projection-width RoPE to GQA-correct head-local RoPE. A checkpoint
/// without a marker is historical version 1.
pub const SCIAGENT_MODEL_SEMANTICS_VERSION: u32 = 2;
const MODEL_SEMANTICS_FILE: &str = "model_semantics.version";

pub fn read_model_semantics_version(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path.join(MODEL_SEMANTICS_FILE))
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
}

fn write_model_semantics_marker(path: &Path) -> std::result::Result<(), String> {
    std::fs::write(
        path.join(MODEL_SEMANTICS_FILE),
        format!("{}\n", SCIAGENT_MODEL_SEMANTICS_VERSION),
    )
    .map_err(|e| {
        format!(
            "cannot write model semantics marker in {}: {e}",
            path.display()
        )
    })
}

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

/// Training cache for one sequence's attention; narrow heads are assembled with
/// allocation-light head assembly rather than full-width padding and addition.
struct CudaAttentionSequenceCache {
    qr: CudaMatrix,
    kr: CudaMatrix,
    weights: Vec<CudaMatrix>,
}

/// Batch attention cache. Each sequence has an independent T×T causal attention
/// graph even though all projection/MLP matrices are packed as B*T rows.
struct CudaAttentionTrainCache {
    sequences: Vec<CudaAttentionSequenceCache>,
    batch_size: usize,
    seq_len: usize,
}

/// Training-only activations for one Transformer block.
struct CudaBlockTrainCache {
    xn: CudaMatrix,
    v: CudaMatrix,
    attention: CudaAttentionTrainCache,
    ctx: CudaMatrix,
    h: CudaMatrix,
    hn: CudaMatrix,
    gate: CudaMatrix,
    up: CudaMatrix,
    act: CudaMatrix,
}

/// Full training forward cache. `xs[i]` is block i's input and `xs[n]` the trunk.
struct CudaTrainCache {
    xs: Vec<CudaMatrix>,
    blocks: Vec<CudaBlockTrainCache>,
    normed: CudaMatrix,
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

    /// Apply the existing narrow-matrix RoPE kernel independently to each logical
    /// head, then concatenate the disjoint head blocks. This is the GQA-correct
    /// rotary basis: frequency index zero restarts for every d_head slice.
    /// GQA-correct RoPE without per-head slicing: one kernel covers the full
    /// projection while the frequency index restarts every `d_head` columns.
    fn rope_heads(
        &self,
        x: &CudaMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> CudaMatrix {
        assert!(n_heads > 0 && x.cols().is_multiple_of(n_heads));
        let dh = x.cols() / n_heads;
        self.chain
            .rope_head_local(x, dh, seq_len, offset, self.theta)
    }

    fn rope_heads_backward(
        &self,
        dy: &CudaMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> CudaMatrix {
        assert!(n_heads > 0 && dy.cols().is_multiple_of(n_heads));
        let dh = dy.cols() / n_heads;
        self.chain
            .rope_head_local_backward(dy, dh, seq_len, offset, self.theta)
    }

    /// Multi-head grouped-query attention over `q` (`t×d_model`) and `k`/`v`
    /// (`t×kv_dim`), matching `GpuChain::gqa_attention`: apply RoPE independently inside every logical head,
    /// then per head `softmax((qs·ksᵀ)/√dh [+causal])·vs`, placed into the head's
    /// `d_model` slot and summed.
    fn attention(&self, q: &CudaMatrix, k: &CudaMatrix, v: &CudaMatrix) -> CudaMatrix {
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = self.rope_heads(q, self.n_heads, seq, 0);
        let kr = self.rope_heads(k, self.n_kv_heads, seq, 0);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut heads = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = self.chain.slice_cols(&qr, head * dh, dh);
            let ks = self.chain.slice_cols(&kr, kv * dh, dh);
            let vs = self.chain.slice_cols(v, kv * dh, dh);
            let scores = self.chain.matmul_bt(&qs, &ks);
            let scaled = self.chain.scale_causal_mask(&scores, scale, self.causal);
            let weights = self.chain.softmax(&scaled);
            heads.push(self.chain.attention_context(&weights, &vs));
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        self.chain.concat_cols(&refs)
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

    /// Mean next-token cross-entropy (nats/token) over up to `max_windows`
    /// non-overlapping `seq_len` windows of `tokens` — the inference-only twin of
    /// [`CudaTrainer::eval_loss`], with no optimizer state allocated. Lets a plain
    /// [`CudaModel`] (2 bytes/param, no fp32 masters/moments) score a held-out split.
    /// Returns `NaN` if the corpus is shorter than one window.
    pub fn eval_loss(&self, tokens: &[u32], seq_len: usize, max_windows: usize) -> f32 {
        let s = seq_len;
        let mut total = 0.0f64;
        let mut count = 0usize;
        let mut cursor = 0usize;
        while cursor + s < tokens.len() && count < max_windows.max(1)
        {
            let inputs = &tokens[cursor..cursor + s];
            let targets = &tokens[cursor + 1..cursor + s + 1];
            let logits = self.forward_resident(inputs);
            total += self.chain.cross_entropy_loss(&logits, targets) as f64;
            count += 1;
            cursor += s;
        }
        if count == 0
        {
            f32::NAN
        }
        else
        {
            (total / count as f64) as f32
        }
    }

    /// Mean CE over explicit non-overlapping window starts. Used by the distributed
    /// held-out protocol so evaluation samples the full corpus deterministically.
    pub fn eval_loss_windows(
        &self,
        tokens: &[u32],
        seq_len: usize,
        starts: &[usize],
        max_windows: usize,
    ) -> f32 {
        let s = seq_len;
        let mut total = 0.0f64;
        let mut count = 0usize;
        for &start in starts.iter().take(max_windows.max(1))
        {
            if start + s >= tokens.len()
            {
                continue;
            }
            let inputs = &tokens[start..start + s];
            let targets = &tokens[start + 1..start + s + 1];
            let logits = self.forward_resident(inputs);
            total += self.chain.cross_entropy_loss(&logits, targets) as f64;
            count += 1;
        }
        if count == 0
        {
            f32::NAN
        }
        else
        {
            (total / count as f64) as f32
        }
    }

    /// Autoregressive generation from `prompt`, appending up to `max_new` tokens.
    /// **Non-cached** (re-runs the full forward each step, O(n²)) — the MVP, matching
    /// Route A's `infer-1` before its KV cache; fine for eyeballing checkpoint quality
    /// on short samples. Uses the shared deterministic [`sample_row`] so the sampling
    /// is bit-identical to the CPU/Route-A paths. Stops early on token `0` (the
    /// `<pad>`/EOS convention). Returns the full token sequence (prompt + generated).
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Vec<u32> {
        let mut tokens: Vec<u32> = if prompt.is_empty()
        {
            vec![0]
        }
        else
        {
            prompt.to_vec()
        };
        let mut rng = seed_to_state(seed);
        for _ in 0..max_new
        {
            let logits = self.forward(&tokens);
            let last = &logits[logits.len() - self.vocab..];
            let recent: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
            let next = sample_row(last, params, &recent, &mut rng) as u32;
            tokens.push(next);
            if next == 0
            {
                break;
            }
        }
        tokens
    }

    /// Incremental single-query GQA over already-RoPE'd resident keys and raw values.
    /// `qr` is one row at the new token's absolute position; cached keys/values contain
    /// every visible position, so no causal mask is required here.
    fn incremental_attention(
        &self,
        qr: &CudaMatrix,
        kcache: &CudaMatrix,
        vcache: &CudaMatrix,
    ) -> CudaMatrix {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut heads = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = ch.slice_cols(qr, head * dh, dh);
            let ks = ch.slice_cols(kcache, kv * dh, dh);
            let vs = ch.slice_cols(vcache, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, false);
            let weights = ch.softmax(&scaled);
            heads.push(ch.attention_context(&weights, &vs));
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        ch.concat_cols(&refs)
    }

    /// Wide prompt prefill for the CUDA KV-cache path. The prompt traverses the
    /// model once with normal causal attention; each layer keeps its RoPE'd K and
    /// raw V matrices resident for subsequent one-token decode steps. Only the last
    /// prompt position's vocab row is downloaded.
    fn prefill_cached(
        &self,
        prompt: &[u32],
        kcache: &mut [Option<CudaMatrix>],
        vcache: &mut [Option<CudaMatrix>],
    ) -> Vec<f32> {
        debug_assert!(!prompt.is_empty());
        debug_assert_eq!(kcache.len(), self.blocks.len());
        debug_assert_eq!(vcache.len(), self.blocks.len());
        let ch = &self.chain;
        let p = prompt.len();
        let mut x = ch.embed(prompt, &self.embedding);
        for (layer, b) in self.blocks.iter().enumerate()
        {
            let xn = ch.rms_norm(&x, &b.norm1, self.eps);
            let q = ch.matmul(&xn, &b.wq);
            let k = ch.matmul(&xn, &b.wk);
            let v = ch.matmul(&xn, &b.wv);
            let ctx = self.attention(&q, &k, &v);
            // `attention` computes the same K RoPE internally; retain an identical
            // resident copy here so future queries can attend without re-prefill.
            kcache[layer] = Some(self.rope_heads(&k, self.n_kv_heads, p, 0));
            vcache[layer] = Some(v);
            let attn_out = ch.matmul(&ctx, &b.wo);
            let h = ch.add(&x, &attn_out);
            let hn = ch.rms_norm(&h, &b.norm2, self.eps);
            let gate = ch.matmul(&hn, &b.wg);
            let up = ch.matmul(&hn, &b.wu);
            let act = ch.swiglu(&gate, &up);
            let mlp = ch.matmul(&act, &b.wd);
            x = ch.add(&h, &mlp);
        }
        let normed = ch.rms_norm(&x, &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        let last = ch.slice_rows(&logits, p - 1, 1);
        ch.download(&last)
    }

    /// Process exactly one newly generated token at absolute position `pos`, append
    /// its per-layer RoPE'd K/raw V rows to the resident caches, and return the
    /// vocab logits predicting the following token. The trunk is O(1) rows; only
    /// attention grows with context length, replacing the old full-sequence O(n^2)
    /// recomputation at every decode step.
    fn decode_step_cached(
        &self,
        token: u32,
        pos: usize,
        kcache: &mut [Option<CudaMatrix>],
        vcache: &mut [Option<CudaMatrix>],
    ) -> Vec<f32> {
        let ch = &self.chain;
        let mut x = ch.embed(&[token], &self.embedding);
        for (layer, b) in self.blocks.iter().enumerate()
        {
            let xn = ch.rms_norm(&x, &b.norm1, self.eps);
            let q = ch.matmul(&xn, &b.wq);
            let k = ch.matmul(&xn, &b.wk);
            let v = ch.matmul(&xn, &b.wv);
            let qr = self.rope_heads(&q, self.n_heads, 1, pos);
            let kr = self.rope_heads(&k, self.n_kv_heads, 1, pos);

            kcache[layer] = Some(match kcache[layer].take()
            {
                None => kr,
                Some(prev) => ch.concat_rows(&[&prev, &kr]),
            });
            vcache[layer] = Some(match vcache[layer].take()
            {
                None => v,
                Some(prev) => ch.concat_rows(&[&prev, &v]),
            });
            let ctx = self.incremental_attention(
                &qr,
                kcache[layer].as_ref().expect("K cache"),
                vcache[layer].as_ref().expect("V cache"),
            );
            let attn_out = ch.matmul(&ctx, &b.wo);
            let h = ch.add(&x, &attn_out);
            let hn = ch.rms_norm(&h, &b.norm2, self.eps);
            let gate = ch.matmul(&hn, &b.wg);
            let up = ch.matmul(&hn, &b.wu);
            let act = ch.swiglu(&gate, &up);
            let mlp = ch.matmul(&act, &b.wd);
            x = ch.add(&h, &mlp);
        }
        let normed = ch.rms_norm(&x, &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        ch.download(&logits)
    }

    /// KV-cached autoregressive generation on CUDA Tensor cores. Sampling semantics
    /// are identical to [`Self::generate`]: same shared sampler/RNG, repetition
    /// context and token-0 early stop. The original non-cached method remains as an
    /// independent parity reference.
    pub fn generate_cached(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Vec<u32> {
        let mut tokens: Vec<u32> = if prompt.is_empty()
        {
            vec![0]
        }
        else
        {
            prompt.to_vec()
        };
        if max_new == 0
        {
            return tokens;
        }
        let mut kcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut vcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut logits = self.prefill_cached(&tokens, &mut kcache, &mut vcache);
        let mut rng = seed_to_state(seed);

        for i in 0..max_new
        {
            let recent: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
            let next = sample_row(&logits, params, &recent, &mut rng) as u32;
            let pos = tokens.len();
            tokens.push(next);
            if next == 0 || i + 1 == max_new
            {
                break;
            }
            logits = self.decode_step_cached(next, pos, &mut kcache, &mut vcache);
        }
        tokens
    }

    /// Diagnostic Route-B generation that records the full BF16 logit row used
    /// for every sampled token. Downloads are intentional and make this unsuitable
    /// for throughput measurements.
    pub fn generate_cached_trace(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut tokens: Vec<u32> = if prompt.is_empty()
        {
            vec![0]
        }
        else
        {
            prompt.to_vec()
        };
        if max_new == 0
        {
            return (tokens, Vec::new());
        }
        let mut kcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut vcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut logits = self.prefill_cached(&tokens, &mut kcache, &mut vcache);
        let mut rows = Vec::with_capacity(max_new);
        let mut rng = seed_to_state(seed);

        for generated_index in 0..max_new
        {
            let recent: Vec<usize> = tokens.iter().map(|&token| token as usize).collect();
            let next = sample_row(&logits, params, &recent, &mut rng) as u32;
            rows.push(logits);
            let pos = tokens.len();
            tokens.push(next);
            if next == 0 || generated_index + 1 == max_new
            {
                break;
            }
            logits = self.decode_step_cached(next, pos, &mut kcache, &mut vcache);
        }
        (tokens, rows)
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
        let qr = self.rope_heads(q, self.n_heads, seq, 0);
        let kr = self.rope_heads(k, self.n_kv_heads, seq, 0);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut dq_heads = Vec::with_capacity(self.n_heads);
        let mut dk_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        let mut dv_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = ch.slice_cols(&qr, head * dh, dh);
            let ks = ch.slice_cols(&kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, self.causal);
            let weights = ch.softmax(&scaled);
            let d_ctx = ch.slice_cols(dout, head * dh, dh);
            let dweights = ch.matmul_bt(&d_ctx, &vs);
            let dvs = ch.matmul_at(&weights, &d_ctx);
            let dscaled = ch.softmax_backward(&weights, &dweights);
            let dscores = ch.scale_causal_mask_backward(&dscaled, scale, self.causal);
            let dqs = ch.matmul(&dscores, &ks);
            let dks = ch.matmul_at(&dscores, &qs);
            dq_heads.push(dqs);
            dk_kv[kv] = Some(match dk_kv[kv].take()
            {
                None => dks,
                Some(acc) => ch.add(&acc, &dks),
            });
            dv_kv[kv] = Some(match dv_kv[kv].take()
            {
                None => dvs,
                Some(acc) => ch.add(&acc, &dvs),
            });
        }
        let dq_refs: Vec<&CudaMatrix> = dq_heads.iter().collect();
        let dk_refs: Vec<&CudaMatrix> =
            dk_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dv_refs: Vec<&CudaMatrix> =
            dv_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dqr = ch.concat_cols(&dq_refs);
        let dkr = ch.concat_cols(&dk_refs);
        let dv = ch.concat_cols(&dv_refs);
        let dq = self.rope_heads_backward(&dqr, self.n_heads, seq, 0);
        let dk = self.rope_heads_backward(&dkr, self.n_kv_heads, seq, 0);
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

    /// Training attention forward retaining only the activations required by its VJP.
    /// One sequence's cached training attention.
    fn attention_train_sequence(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
    ) -> (CudaMatrix, CudaAttentionSequenceCache) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = self.rope_heads(q, self.n_heads, seq, 0);
        let kr = self.rope_heads(k, self.n_kv_heads, seq, 0);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut heads = Vec::with_capacity(self.n_heads);
        let mut weights_all = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = ch.slice_cols(&qr, head * dh, dh);
            let ks = ch.slice_cols(&kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, self.causal);
            let weights = ch.softmax(&scaled);
            heads.push(ch.matmul(&weights, &vs));
            weights_all.push(weights);
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        (
            ch.concat_cols(&refs),
            CudaAttentionSequenceCache {
                qr,
                kr,
                weights: weights_all,
            },
        )
    }

    /// True batched attention: packed projection rows, isolated per-sequence causal
    /// attention. No token in one sample can attend to another sample.
    fn attention_train_batched(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
        batch_size: usize,
        seq_len: usize,
    ) -> (CudaMatrix, CudaAttentionTrainCache) {
        assert_eq!(q.rows(), batch_size * seq_len, "batched attention q rows");
        assert_eq!(k.rows(), q.rows(), "batched attention k rows");
        assert_eq!(v.rows(), q.rows(), "batched attention v rows");
        let ch = &self.chain;
        let mut outputs = Vec::with_capacity(batch_size);
        let mut sequences = Vec::with_capacity(batch_size);
        for b in 0..batch_size
        {
            let row = b * seq_len;
            let qs = ch.slice_rows(q, row, seq_len);
            let ks = ch.slice_rows(k, row, seq_len);
            let vs = ch.slice_rows(v, row, seq_len);
            let (out, cache) = self.attention_train_sequence(&qs, &ks, &vs);
            outputs.push(out);
            sequences.push(cache);
        }
        let refs: Vec<&CudaMatrix> = outputs.iter().collect();
        (
            ch.concat_rows(&refs),
            CudaAttentionTrainCache {
                sequences,
                batch_size,
                seq_len,
            },
        )
    }

    /// Training block forward retaining activations instead of recomputing them later.
    /// Training block forward: dense projections/MLP run over packed B*T rows,
    /// attention remains strictly separated by sample.
    fn block_train(
        &self,
        x: &CudaMatrix,
        b: &CudaBlock,
        batch_size: usize,
        seq_len: usize,
    ) -> (CudaMatrix, CudaBlockTrainCache) {
        let ch = &self.chain;
        let xn = ch.rms_norm(x, &b.norm1, self.eps);
        let q = ch.matmul(&xn, &b.wq);
        let k = ch.matmul(&xn, &b.wk);
        let v = ch.matmul(&xn, &b.wv);
        let (ctx, attention) = self.attention_train_batched(&q, &k, &v, batch_size, seq_len);
        let attn_out = ch.matmul(&ctx, &b.wo);
        let h = ch.add(x, &attn_out);
        let hn = ch.rms_norm(&h, &b.norm2, self.eps);
        let gate = ch.matmul(&hn, &b.wg);
        let up = ch.matmul(&hn, &b.wu);
        let act = ch.swiglu(&gate, &up);
        let mlp = ch.matmul(&act, &b.wd);
        let out = ch.add(&h, &mlp);
        (
            out,
            CudaBlockTrainCache {
                xn,
                v,
                attention,
                ctx,
                h,
                hn,
                gate,
                up,
                act,
            },
        )
    }

    /// Full training forward with an explicit activation cache.
    /// Full packed B×T training forward with independent attention sequences.
    fn forward_train(
        &self,
        tokens: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> (CudaMatrix, CudaTrainCache) {
        assert!(batch_size > 0 && seq_len > 0, "forward_train: empty batch");
        assert_eq!(
            tokens.len(),
            batch_size * seq_len,
            "forward_train: B*T mismatch"
        );
        let ch = &self.chain;
        let mut xs = Vec::with_capacity(self.blocks.len() + 1);
        let mut caches = Vec::with_capacity(self.blocks.len());
        xs.push(ch.embed(tokens, &self.embedding));
        for b in &self.blocks
        {
            let (out, cache) =
                self.block_train(xs.last().expect("block input"), b, batch_size, seq_len);
            xs.push(out);
            caches.push(cache);
        }
        let normed = ch.rms_norm(xs.last().expect("trunk"), &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        (
            logits,
            CudaTrainCache {
                xs,
                blocks: caches,
                normed,
            },
        )
    }

    fn attention_backward_sequence_cached(
        &self,
        v: &CudaMatrix,
        dout: &CudaMatrix,
        cache: &CudaAttentionSequenceCache,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = cache.qr.rows();
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut dq_heads = Vec::with_capacity(self.n_heads);
        let mut dk_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        let mut dv_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        for head in 0..self.n_heads
        {
            let kv = head / repeat;
            let qs = ch.slice_cols(&cache.qr, head * dh, dh);
            let ks = ch.slice_cols(&cache.kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let weights = &cache.weights[head];
            let d_ctx = ch.slice_cols(dout, head * dh, dh);
            let dweights = ch.matmul_bt(&d_ctx, &vs);
            let dvs = ch.matmul_at(weights, &d_ctx);
            let dscaled = ch.softmax_backward(weights, &dweights);
            let dscores = ch.scale_causal_mask_backward(&dscaled, scale, self.causal);
            let dqs = ch.matmul(&dscores, &ks);
            let dks = ch.matmul_at(&dscores, &qs);
            dq_heads.push(dqs);
            dk_kv[kv] = Some(match dk_kv[kv].take()
            {
                None => dks,
                Some(acc) => ch.add(&acc, &dks),
            });
            dv_kv[kv] = Some(match dv_kv[kv].take()
            {
                None => dvs,
                Some(acc) => ch.add(&acc, &dvs),
            });
        }
        let dq_refs: Vec<&CudaMatrix> = dq_heads.iter().collect();
        let dk_refs: Vec<&CudaMatrix> =
            dk_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dv_refs: Vec<&CudaMatrix> =
            dv_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dqr = ch.concat_cols(&dq_refs);
        let dkr = ch.concat_cols(&dk_refs);
        let dv = ch.concat_cols(&dv_refs);
        let dq = self.rope_heads_backward(&dqr, self.n_heads, seq, 0);
        let dk = self.rope_heads_backward(&dkr, self.n_kv_heads, seq, 0);
        (dq, dk, dv)
    }

    fn attention_backward_cached(
        &self,
        v: &CudaMatrix,
        dout: &CudaMatrix,
        cache: &CudaAttentionTrainCache,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        assert_eq!(
            v.rows(),
            cache.batch_size * cache.seq_len,
            "attention backward v rows"
        );
        assert_eq!(dout.rows(), v.rows(), "attention backward dout rows");
        let mut dqs = Vec::with_capacity(cache.batch_size);
        let mut dks = Vec::with_capacity(cache.batch_size);
        let mut dvs = Vec::with_capacity(cache.batch_size);
        for b in 0..cache.batch_size
        {
            let row = b * cache.seq_len;
            let vs = ch.slice_rows(v, row, cache.seq_len);
            let ds = ch.slice_rows(dout, row, cache.seq_len);
            let (dq, dk, dv) =
                self.attention_backward_sequence_cached(&vs, &ds, &cache.sequences[b]);
            dqs.push(dq);
            dks.push(dk);
            dvs.push(dv);
        }
        let qrefs: Vec<&CudaMatrix> = dqs.iter().collect();
        let krefs: Vec<&CudaMatrix> = dks.iter().collect();
        let vrefs: Vec<&CudaMatrix> = dvs.iter().collect();
        (
            ch.concat_rows(&qrefs),
            ch.concat_rows(&krefs),
            ch.concat_rows(&vrefs),
        )
    }

    fn block_backward_cached(
        &self,
        x: &CudaMatrix,
        b: &CudaBlock,
        cache: &CudaBlockTrainCache,
        dout: &CudaMatrix,
    ) -> (CudaMatrix, CudaBlockGrads) {
        let ch = &self.chain;
        let dact = ch.matmul_bt(dout, &b.wd);
        let dwd = ch.matmul_at(&cache.act, dout);
        let (dgate, dup) = ch.swiglu_backward(&cache.gate, &cache.up, &dact);
        let dwg = ch.matmul_at(&cache.hn, &dgate);
        let dwu = ch.matmul_at(&cache.hn, &dup);
        let dhn = ch.add(&ch.matmul_bt(&dgate, &b.wg), &ch.matmul_bt(&dup, &b.wu));
        let dnorm2 = ch.rms_norm_gain_backward(&cache.h, &dhn, self.eps);
        let dh = ch.add(
            dout,
            &ch.rms_norm_backward(&cache.h, &b.norm2, &dhn, self.eps),
        );

        let dwo = ch.matmul_at(&cache.ctx, &dh);
        let d_ctx = ch.matmul_bt(&dh, &b.wo);
        let (dq, dk, dv) = self.attention_backward_cached(&cache.v, &d_ctx, &cache.attention);
        let dwq = ch.matmul_at(&cache.xn, &dq);
        let dwk = ch.matmul_at(&cache.xn, &dk);
        let dwv = ch.matmul_at(&cache.xn, &dv);
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

    fn backward_cached(
        &self,
        tokens: &[u32],
        dlogits: &CudaMatrix,
        cache: &CudaTrainCache,
    ) -> CudaModelGrads {
        let ch = &self.chain;
        let trunk = cache.xs.last().expect("trunk");
        let d_normed = ch.matmul(dlogits, &self.embedding);
        let de_head = ch.matmul_at(dlogits, &cache.normed);
        let d_final_norm = ch.rms_norm_gain_backward(trunk, &d_normed, self.eps);
        let mut d_cur = ch.rms_norm_backward(trunk, &self.final_norm, &d_normed, self.eps);
        let mut block_grads = Vec::with_capacity(self.blocks.len());
        for i in (0..self.blocks.len()).rev()
        {
            let (dx, grads) =
                self.block_backward_cached(&cache.xs[i], &self.blocks[i], &cache.blocks[i], &d_cur);
            d_cur = dx;
            block_grads.push(grads);
        }
        block_grads.reverse();
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

/// Persisted CUDA optimizer/schedule metadata. Unlike legacy checkpoints that only
/// saved model weights, B32 checkpoints restore AdamW moments and bias-correction
/// step exactly. Schedule fields let an interrupted run continue its original LR
/// curve unless the caller explicitly overrides it.
#[derive(Clone, Debug)]
pub struct CudaOptimizerResume {
    pub step: usize,
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub betas: (f32, f32),
    pub adam_eps: f32,
    pub weight_decay: f32,
    /// Exact-data resume fields are absent on legacy B32/v1 sidecars.
    pub seq_len: Option<usize>,
    pub batch_size: Option<usize>,
    pub corpus_tokens: Option<usize>,
    pub corpus_hash: Option<u64>,
    pub split_version: Option<u32>,
    pub max_grad_norm: Option<f32>,
    pub val_frac: Option<f32>,
    pub shuffle: Option<bool>,
}

fn optimizer_tensor(chain: &CudaChain, x: &CudaF32, rows: usize, cols: usize) -> Tensor {
    Tensor::from_vec(chain.download_f32(x), rows, cols)
}

fn load_optimizer_tensor(
    chain: &CudaChain,
    state: &HashMap<String, Tensor>,
    name: &str,
    rows: usize,
    cols: usize,
) -> std::result::Result<CudaF32, String> {
    let t = state
        .get(name)
        .ok_or_else(|| format!("optimizer checkpoint missing tensor '{name}'"))?;
    if (t.rows, t.cols) != (rows, cols)
    {
        return Err(format!(
            "optimizer tensor '{name}' has shape {}x{}, expected {rows}x{cols}",
            t.rows, t.cols
        ));
    }
    Ok(chain.upload_f32(&t.data))
}

/// Resident diagnostics for one optimizer step. Keeping these tiny buffers alive
/// lets pretraining enqueue several complete steps before any host synchronization.
struct CudaStepDiagnostics {
    loss_rows: CudaF32,
    grad_sumsq: CudaF32,
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
    /// Global grad-norm clip threshold (`<= 0` disables). Default `1.0`.
    max_grad_norm: f32,
    /// The last step's pre-clip global grad norm (for logging / diagnostics).
    last_grad_norm: f32,
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
            max_grad_norm: 1.0,
            last_grad_norm: 0.0,
        })
    }

    /// The vocabulary width (logit columns).
    pub fn vocab(&self) -> usize {
        self.model.vocab
    }

    /// Set the global grad-norm clip threshold (`<= 0` disables clipping). Standard
    /// pretraining practice is `1.0` (the default).
    pub fn set_max_grad_norm(&mut self, max_norm: f32) {
        self.max_grad_norm = max_norm;
    }

    /// The last training step's pre-clip global gradient L2 norm — the diagnostic
    /// that reveals gradient spikes.
    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    /// One mixed-precision AdamW training step on `(tokens, targets)`: forward →
    /// resident cross-entropy grad → cached backward → AdamW update of every trainable weight
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
        self.train_step_batch(
            tokens,
            targets,
            1,
            tokens.len(),
            lr,
            betas,
            adam_eps,
            weight_decay,
        )
    }

    /// One true B×T optimizer step. Dense Transformer GEMMs consume all B*T rows in
    /// one call; attention is isolated per sequence so samples never cross-attend.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step_batch(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch_size: usize,
        seq_len: usize,
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> f32 {
        let diag = self.train_step_batch_deferred(
            tokens,
            targets,
            batch_size,
            seq_len,
            lr,
            betas,
            adam_eps,
            weight_decay,
        );
        self.finish_step_diagnostics(diag)
    }

    /// Enqueue a complete optimizer step without reading diagnostics back to the host.
    /// Public one-step APIs stay synchronous; the production pretrainer batches these
    /// diagnostics so loss/gnorm telemetry no longer inserts a barrier every step.
    #[allow(clippy::too_many_arguments)]
    fn train_step_batch_deferred(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch_size: usize,
        seq_len: usize,
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> CudaStepDiagnostics {
        assert!(
            batch_size > 0 && seq_len > 0,
            "train_step_batch: empty batch"
        );
        assert_eq!(
            tokens.len(),
            batch_size * seq_len,
            "train_step_batch: token shape"
        );
        assert_eq!(
            targets.len(),
            tokens.len(),
            "train_step_batch: target shape"
        );
        self.step += 1;
        let (logits, cache) = self.model.forward_train(tokens, batch_size, seq_len);
        let (loss_rows, dlogits) = self
            .model
            .chain
            .cross_entropy_loss_grad_resident(&logits, targets);
        let grads = self.model.backward_cached(tokens, &dlogits, &cache);

        let mut grad_refs: Vec<&CudaMatrix> = vec![&grads.d_embedding, &grads.d_final_norm];
        for bg in &grads.blocks
        {
            grad_refs.extend([
                &bg.dnorm1, &bg.dwq, &bg.dwk, &bg.dwv, &bg.dwo, &bg.dnorm2, &bg.dwg, &bg.dwu,
                &bg.dwd,
            ]);
        }
        let ch = &self.model.chain;
        let grad_sumsq = ch.global_grad_sumsq(&grad_refs);
        drop(grad_refs);
        let step = self.step;
        let max_norm = self.max_grad_norm;

        ch.adamw_step_with_norm(
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
            &grad_sumsq,
            max_norm,
        );
        ch.adamw_step_with_norm(
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
            &grad_sumsq,
            max_norm,
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
                ch.adamw_step_with_norm(
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
                    &grad_sumsq,
                    max_norm,
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
        CudaStepDiagnostics {
            loss_rows,
            grad_sumsq,
        }
    }

    fn finish_step_diagnostics(&mut self, diag: CudaStepDiagnostics) -> f32 {
        let loss = self.model.chain.mean_f32(&diag.loss_rows);
        self.last_grad_norm = self.model.chain.grad_norm_from_sumsq(&diag.grad_sumsq);
        loss
    }

    /// Save AdamW moments + optimizer step next to a model checkpoint. Model fp32
    /// masters themselves are already represented by `model.safetensors` after
    /// `sync_to_model`; duplicating them here would add ~1.2 GB without information.
    pub fn save_optimizer_state(
        &self,
        cfg: &CudaPretrainConfig,
        path: &Path,
    ) -> std::result::Result<(), String> {
        let ch = &self.model.chain;
        let mut tensors: Vec<(String, Tensor)> = Vec::new();
        let (er, ec) = (self.model.embedding.rows(), self.model.embedding.cols());
        tensors.push((
            "embedding.m".into(),
            optimizer_tensor(ch, &self.m_embedding, er, ec),
        ));
        tensors.push((
            "embedding.v".into(),
            optimizer_tensor(ch, &self.v_embedding, er, ec),
        ));
        let (nr, nc) = (self.model.final_norm.rows(), self.model.final_norm.cols());
        tensors.push((
            "final_norm.m".into(),
            optimizer_tensor(ch, &self.m_final_norm, nr, nc),
        ));
        tensors.push((
            "final_norm.v".into(),
            optimizer_tensor(ch, &self.v_final_norm, nr, nc),
        ));
        for i in 0..self.model.blocks.len()
        {
            let b = &self.model.blocks[i];
            let mm = &self.m_blocks[i];
            let vv = &self.v_blocks[i];
            macro_rules! push_pair {
                ($field:ident) => {{
                    let rows = b.$field.rows();
                    let cols = b.$field.cols();
                    tensors.push((
                        format!("blocks.{i}.{}.m", stringify!($field)),
                        optimizer_tensor(ch, &mm.$field, rows, cols),
                    ));
                    tensors.push((
                        format!("blocks.{i}.{}.v", stringify!($field)),
                        optimizer_tensor(ch, &vv.$field, rows, cols),
                    ));
                }};
            }
            push_pair!(norm1);
            push_pair!(wq);
            push_pair!(wk);
            push_pair!(wv);
            push_pair!(wo);
            push_pair!(norm2);
            push_pair!(wg);
            push_pair!(wu);
            push_pair!(wd);
        }
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        scirust_core::io::safetensors::save_safetensors(
            &tensors,
            path.join("optimizer.safetensors"),
        )
        .map_err(|e| format!("cannot save optimizer.safetensors: {e}"))?;

        let meta = serde_json::json!({
            "version": 4,
            "step": self.step,
            "base_lr": cfg.base_lr,
            "min_lr": cfg.min_lr,
            "warmup_steps": cfg.warmup_steps,
            "total_steps": cfg.total_steps,
            "betas": [cfg.betas.0, cfg.betas.1],
            "adam_eps": cfg.adam_eps,
            "weight_decay": cfg.weight_decay,
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "corpus_tokens": cfg.corpus_tokens,
            "corpus_hash": cfg.corpus_hash,
            "split_version": WINDOW_SPLIT_VERSION,
            "max_grad_norm": cfg.max_grad_norm,
            "val_frac": cfg.val_frac,
            "shuffle": cfg.shuffle,
        });
        let encoded = serde_json::to_string_pretty(&meta)
            .map_err(|e| format!("cannot serialize optimizer metadata: {e}"))?;
        std::fs::write(path.join("optimizer.json"), encoded)
            .map_err(|e| format!("cannot write optimizer metadata: {e}"))?;
        Ok(())
    }

    /// Restore AdamW state. `Ok(None)` means a legacy checkpoint with no optimizer
    /// sidecar; malformed/incomplete B32 state is an error and must not silently
    /// degrade to zero moments.
    pub fn load_optimizer_state(
        &mut self,
        path: &Path,
    ) -> std::result::Result<Option<CudaOptimizerResume>, String> {
        let state_path = path.join("optimizer.safetensors");
        let meta_path = path.join("optimizer.json");
        let has_state = state_path.exists();
        let has_meta = meta_path.exists();
        if !has_state && !has_meta
        {
            return Ok(None);
        }
        if has_state != has_meta
        {
            return Err(format!(
                "incomplete optimizer checkpoint at {} (need optimizer.safetensors + optimizer.json)",
                path.display()
            ));
        }
        let state = scirust_core::io::safetensors::load_safetensors(&state_path)
            .map_err(|e| format!("cannot load {}: {e}", state_path.display()))?;
        let raw = std::fs::read_to_string(&meta_path)
            .map_err(|e| format!("cannot read {}: {e}", meta_path.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("cannot parse {}: {e}", meta_path.display()))?;
        let version = meta["version"].as_u64().unwrap_or(0);
        if !(1..=4).contains(&version)
        {
            return Err(format!(
                "unsupported optimizer checkpoint version {version} in {}",
                meta_path.display()
            ));
        }
        let step = meta["step"]
            .as_u64()
            .ok_or_else(|| "optimizer metadata missing step".to_string())?
            as usize;
        if step > u32::MAX as usize
        {
            return Err(format!("optimizer step {step} exceeds u32::MAX"));
        }
        let number = |key: &str| -> std::result::Result<f32, String> {
            meta[key]
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("optimizer metadata missing {key}"))
        };
        let usize_field = |key: &str| -> std::result::Result<usize, String> {
            meta[key]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("optimizer metadata missing {key}"))
        };
        let betas = meta["betas"]
            .as_array()
            .filter(|x| x.len() == 2)
            .ok_or_else(|| "optimizer metadata missing betas".to_string())?;
        let beta0 = betas[0]
            .as_f64()
            .ok_or_else(|| "optimizer beta0 invalid".to_string())? as f32;
        let beta1 = betas[1]
            .as_f64()
            .ok_or_else(|| "optimizer beta1 invalid".to_string())? as f32;

        let ch = &self.model.chain;
        let (er, ec) = (self.model.embedding.rows(), self.model.embedding.cols());
        self.m_embedding = load_optimizer_tensor(ch, &state, "embedding.m", er, ec)?;
        self.v_embedding = load_optimizer_tensor(ch, &state, "embedding.v", er, ec)?;
        let (nr, nc) = (self.model.final_norm.rows(), self.model.final_norm.cols());
        self.m_final_norm = load_optimizer_tensor(ch, &state, "final_norm.m", nr, nc)?;
        self.v_final_norm = load_optimizer_tensor(ch, &state, "final_norm.v", nr, nc)?;
        for i in 0..self.model.blocks.len()
        {
            let b = &self.model.blocks[i];
            macro_rules! load_pair {
                ($field:ident) => {{
                    let rows = b.$field.rows();
                    let cols = b.$field.cols();
                    self.m_blocks[i].$field = load_optimizer_tensor(
                        ch,
                        &state,
                        &format!("blocks.{i}.{}.m", stringify!($field)),
                        rows,
                        cols,
                    )?;
                    self.v_blocks[i].$field = load_optimizer_tensor(
                        ch,
                        &state,
                        &format!("blocks.{i}.{}.v", stringify!($field)),
                        rows,
                        cols,
                    )?;
                }};
            }
            load_pair!(norm1);
            load_pair!(wq);
            load_pair!(wk);
            load_pair!(wv);
            load_pair!(wo);
            load_pair!(norm2);
            load_pair!(wg);
            load_pair!(wu);
            load_pair!(wd);
        }
        self.step = step as u32;
        let optional_usize = |key: &str| meta[key].as_u64().map(|x| x as usize);
        let optional_u64 = |key: &str| meta[key].as_u64();
        let optional_f32 = |key: &str| meta[key].as_f64().map(|x| x as f32);
        let optional_bool = |key: &str| meta[key].as_bool();
        Ok(Some(CudaOptimizerResume {
            step,
            base_lr: number("base_lr")?,
            min_lr: number("min_lr")?,
            warmup_steps: usize_field("warmup_steps")?,
            total_steps: usize_field("total_steps")?,
            betas: (beta0, beta1),
            adam_eps: number("adam_eps")?,
            weight_decay: number("weight_decay")?,
            seq_len: if version >= 2
            {
                optional_usize("seq_len")
            }
            else
            {
                None
            },
            batch_size: if version >= 2
            {
                optional_usize("batch_size")
            }
            else
            {
                None
            },
            corpus_tokens: if version >= 2
            {
                optional_usize("corpus_tokens")
            }
            else
            {
                None
            },
            corpus_hash: if version >= 2
            {
                optional_u64("corpus_hash")
            }
            else
            {
                None
            },
            split_version: if version >= 3
            {
                optional_u64("split_version").map(|v| v as u32)
            }
            else
            {
                None
            },
            max_grad_norm: if version >= 4
            {
                optional_f32("max_grad_norm")
            }
            else
            {
                None
            },
            val_frac: if version >= 4
            {
                optional_f32("val_frac")
            }
            else
            {
                None
            },
            shuffle: if version >= 4
            {
                optional_bool("shuffle")
            }
            else
            {
                None
            },
        }))
    }

    /// Save model + optimizer into a hidden partial directory and atomically rename
    /// it only after both payloads are complete. `latest_checkpoint` therefore never
    /// selects a crash-torn training state.
    fn save_training_checkpoint(
        &self,
        model: &SciAgentModel,
        meta: &CheckpointMeta,
        cfg: &CudaPretrainConfig,
        final_dir: &Path,
    ) -> std::result::Result<(), String> {
        if self.step as usize != meta.step
        {
            return Err(format!(
                "optimizer step {} does not match checkpoint step {}",
                self.step, meta.step
            ));
        }
        let parent = final_dir.parent().unwrap_or_else(|| Path::new("."));
        let name = final_dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("invalid checkpoint path {}", final_dir.display()))?;
        let partial = parent.join(format!(".{name}.partial"));
        if partial.exists()
        {
            std::fs::remove_dir_all(&partial)
                .map_err(|e| format!("cannot remove stale {}: {e}", partial.display()))?;
        }
        save_checkpoint(model, meta, &partial)
            .map_err(|e| format!("cannot save model checkpoint: {e}"))?;
        write_model_semantics_marker(&partial)?;
        self.save_optimizer_state(cfg, &partial)?;
        if final_dir.exists()
        {
            return Err(format!(
                "checkpoint already exists: {}",
                final_dir.display()
            ));
        }
        std::fs::rename(&partial, final_dir).map_err(|e| {
            format!(
                "cannot atomically publish {} -> {}: {e}",
                partial.display(),
                final_dir.display()
            )
        })?;
        Ok(())
    }

    /// Mean cross-entropy over up to `max_windows` non-overlapping `seq_len` windows
    /// of `val_tokens` — **no update** (pure forward), so it measures held-out
    /// generalization rather than the train loss on the repeatedly-seen corpus.
    /// Returns `NaN` if there isn't a full window.
    pub fn eval_loss(&self, val_tokens: &[u32], seq_len: usize, max_windows: usize) -> f32 {
        let s = seq_len;
        let mut total = 0.0f64;
        let mut count = 0usize;
        let mut cursor = 0usize;
        while cursor + s < val_tokens.len() && count < max_windows.max(1)
        {
            let inputs = &val_tokens[cursor..cursor + s];
            let targets = &val_tokens[cursor + 1..cursor + s + 1];
            let logits = self.model.forward_resident(inputs);
            total += self.model.chain.cross_entropy_loss(&logits, targets) as f64;
            count += 1;
            cursor += s;
        }
        if count == 0
        {
            f32::NAN
        }
        else
        {
            (total / count as f64) as f32
        }
    }

    fn eval_loss_windows(
        &self,
        tokens: &[u32],
        seq_len: usize,
        starts: &[usize],
        max_windows: usize,
    ) -> f32 {
        self.model
            .eval_loss_windows(tokens, seq_len, starts, max_windows)
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
        let batch = cfg.batch_size.max(1);
        let telemetry = cfg.telemetry_interval.max(1);
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
        self.max_grad_norm = cfg.max_grad_norm;

        let (base_order, val_windows) = distributed_window_split(tokens.len(), s, cfg.val_frac);
        let n_windows = base_order.len();
        if n_windows == 0
        {
            return losses;
        }
        if !val_windows.is_empty()
        {
            println!(
                "held-out validation: {} distributed windows ({:.1}% target, split-v{})\n",
                val_windows.len(),
                cfg.val_frac * 100.0,
                WINDOW_SPLIT_VERSION
            );
        }

        let schedule =
            WarmupCosineSchedule::new(cfg.base_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps);
        let mut step = cfg.start_step;
        // The permutation is a pure function of the absolute epoch, never of the
        // process invocation. Resume reconstructs both epoch and cursor from the
        // number of sequences already consumed, so interrupted and uninterrupted
        // runs see the exact same next windows.
        let consumed_windows = cfg.start_step.saturating_mul(batch);
        let mut epoch: u64 = (consumed_windows / n_windows) as u64;
        let mut wi = consumed_windows % n_windows;
        let mut order = base_order.clone();
        let set_epoch_order = |order: &mut Vec<usize>, epoch: u64| {
            // Every epoch permutation is a pure function of (base_order, epoch).
            // Do not shuffle the previous epoch's permutation: a resumed process
            // must reconstruct epoch N without replaying epochs 0..N-1.
            order.clone_from(&base_order);
            if cfg.shuffle
            {
                shuffle_windows(order, 0x5343_4941_4745_4E54u64 ^ epoch);
            }
        };
        set_epoch_order(&mut order, epoch);
        let (mut best_step, mut best_val) = load_best_validation(&cfg.checkpoint_dir)
            .map(|(step, val)| (Some(step), val))
            .unwrap_or((None, f32::INFINITY));
        if let Some(saved_best) = best_step
        {
            println!("restored best validation: {best_val:.4} @ step {saved_best}");
        }
        let mut last_eval: Option<(usize, f32)> = None;
        let t0 = std::time::Instant::now();
        let mut last_checkpoint_at = t0;
        let mut last_checkpoint_step: Option<usize> = None;
        let checkpointing_enabled = cfg.save_interval > 0 || cfg.save_interval_seconds > 0;
        let mut packed_inputs = Vec::with_capacity(batch * s);
        let mut packed_targets = Vec::with_capacity(batch * s);
        let mut pending: Vec<CudaStepDiagnostics> = Vec::with_capacity(telemetry);
        let mut last_loss = f32::NAN;

        while step < cfg.total_steps && n_windows > 0
        {
            packed_inputs.clear();
            packed_targets.clear();
            for _ in 0..batch
            {
                if wi >= order.len()
                {
                    epoch += 1;
                    set_epoch_order(&mut order, epoch);
                    wi = 0;
                }
                let start = order[wi];
                wi += 1;
                packed_inputs.extend_from_slice(&tokens[start..start + s]);
                packed_targets.extend_from_slice(&tokens[start + 1..start + s + 1]);
            }
            let lr = schedule.lr_at(step);
            pending.push(self.train_step_batch_deferred(
                &packed_inputs,
                &packed_targets,
                batch,
                s,
                lr,
                cfg.betas,
                cfg.adam_eps,
                cfg.weight_decay,
            ));
            step += 1;

            let need_log = cfg.log_interval > 0 && step.is_multiple_of(cfg.log_interval);
            let need_eval = cfg.eval_interval > 0
                && !val_windows.is_empty()
                && step.is_multiple_of(cfg.eval_interval);
            let need_save_by_step = cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval);
            let need_save_by_time = cfg.save_interval_seconds > 0
                && last_checkpoint_at.elapsed().as_secs() >= cfg.save_interval_seconds;
            let need_save = need_save_by_step || need_save_by_time;
            let need_flush = pending.len() >= telemetry
                || need_log
                || need_eval
                || need_save
                || step == cfg.total_steps;

            if need_flush
            {
                for diag in pending.drain(..)
                {
                    last_loss = self.finish_step_diagnostics(diag);
                    losses.push(last_loss);
                }
            }

            if need_log
            {
                let done = (step - cfg.start_step) * s * batch;
                let secs = t0.elapsed().as_secs_f64().max(1e-9);
                let tps = done as f64 / secs;
                let gnorm = self.last_grad_norm();
                println!(
                    "[cuda step {step:>6}] B{batch}×T{s} loss {last_loss:>9.4} | lr {lr:.3e} | gnorm {gnorm:>7.2} | {tps:>8.0} tok/s"
                );
            }
            if need_eval
            {
                let val = self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows);
                last_eval = Some((step, val));
                println!("            └─ held-out val loss {val:>9.4}");
            }
            if need_save
            {
                self.sync_to_model(model);
                let dir = std::path::Path::new(&cfg.checkpoint_dir).join(format!("step_{step}"));
                let meta = CheckpointMeta {
                    step,
                    loss: last_loss,
                    lr,
                    config: config.clone(),
                };
                match self.save_training_checkpoint(model, &meta, cfg, &dir)
                {
                    Ok(()) =>
                    {
                        println!("  exact checkpoint → {}", dir.display());
                        last_checkpoint_step = Some(step);
                        last_checkpoint_at = std::time::Instant::now();
                        if !val_windows.is_empty()
                        {
                            let v = match last_eval
                            {
                                Some((eval_step, v)) if eval_step == step => v,
                                _ =>
                                {
                                    let v = self.eval_loss_windows(
                                        tokens,
                                        s,
                                        &val_windows,
                                        cfg.eval_windows,
                                    );
                                    last_eval = Some((step, v));
                                    v
                                },
                            };
                            if v < best_val
                            {
                                best_val = v;
                                best_step = Some(step);
                                match save_best_validation_model(
                                    model,
                                    &meta,
                                    v,
                                    &cfg.checkpoint_dir,
                                )
                                {
                                    Ok(()) => println!(
                                        "    best model-only → {}/best (val {v:.4} @ step {step})",
                                        cfg.checkpoint_dir
                                    ),
                                    Err(e) => eprintln!("    best-model save failed: {e}"),
                                }
                            }
                        }
                        prune_checkpoints(&cfg.checkpoint_dir, cfg.keep_last);
                    },
                    Err(e) =>
                    {
                        eprintln!("  checkpoint at step {step} failed: {e}");
                        // Back off after an I/O failure instead of retrying every step.
                        last_checkpoint_at = std::time::Instant::now();
                    },
                }
            }
        }

        // Defensive flush for any non-standard early exit. Normal total-step exit
        // already flushes above, so the returned vector remains exactly per-step.
        for diag in pending.drain(..)
        {
            last_loss = self.finish_step_diagnostics(diag);
            losses.push(last_loss);
        }

        // A run target is an exact recovery boundary even when it falls between the
        // periodic save cadence. Historically the example only synced host weights
        // here and falsely claimed a final checkpoint existed.
        if checkpointing_enabled && step > cfg.start_step && last_checkpoint_step != Some(step)
        {
            self.sync_to_model(model);
            let lr = schedule.lr_at(step.saturating_sub(1));
            let dir = Path::new(&cfg.checkpoint_dir).join(format!("step_{step}"));
            let meta = CheckpointMeta {
                step,
                loss: last_loss,
                lr,
                config: config.clone(),
            };
            match self.save_training_checkpoint(model, &meta, cfg, &dir)
            {
                Ok(()) =>
                {
                    println!("  final exact checkpoint → {}", dir.display());
                    if !val_windows.is_empty()
                    {
                        let v = match last_eval
                        {
                            Some((eval_step, v)) if eval_step == step => v,
                            _ => self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows),
                        };
                        if v < best_val
                        {
                            best_val = v;
                            best_step = Some(step);
                            if let Err(e) =
                                save_best_validation_model(model, &meta, v, &cfg.checkpoint_dir)
                            {
                                eprintln!("    final best-model save failed: {e}");
                            }
                        }
                    }
                    prune_checkpoints(&cfg.checkpoint_dir, cfg.keep_last);
                },
                Err(e) => eprintln!("  final checkpoint at step {step} failed: {e}"),
            }
        }
        let _ = best_step; // retained for diagnostics/documentation; best/ is model-only.
        let _ = best_val;
        losses
    }
}

fn load_best_validation(dir: &str) -> Option<(usize, f32)> {
    let path = Path::new(dir).join("best").join("selection.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let step = value["step"].as_u64()? as usize;
    let val = value["val_loss"].as_f64()? as f32;
    val.is_finite().then_some((step, val))
}

/// Publish a model-only best-validation snapshot. Exact resume checkpoints keep
/// AdamW state in `step_N/`; the historical best does not need another ~2.4 GB of
/// moments. `best/` is deliberately ignored by `latest_checkpoint`.
fn save_best_validation_model(
    model: &SciAgentModel,
    meta: &CheckpointMeta,
    val_loss: f32,
    root: &str,
) -> std::result::Result<(), String> {
    let root = Path::new(root);
    std::fs::create_dir_all(root)
        .map_err(|e| format!("cannot create checkpoint root {}: {e}", root.display()))?;
    let partial = root.join(".best.partial");
    let old = root.join(".best.old");
    let best = root.join("best");
    if partial.exists()
    {
        std::fs::remove_dir_all(&partial)
            .map_err(|e| format!("cannot remove stale {}: {e}", partial.display()))?;
    }
    if old.exists()
    {
        std::fs::remove_dir_all(&old)
            .map_err(|e| format!("cannot remove stale {}: {e}", old.display()))?;
    }
    save_checkpoint(model, meta, &partial).map_err(|e| format!("cannot save best model: {e}"))?;
    write_model_semantics_marker(&partial)?;
    let selection = serde_json::json!({
        "version": 1,
        "step": meta.step,
        "val_loss": val_loss,
    });
    std::fs::write(
        partial.join("selection.json"),
        serde_json::to_string_pretty(&selection)
            .map_err(|e| format!("cannot encode best selection: {e}"))?,
    )
    .map_err(|e| format!("cannot write best selection: {e}"))?;

    if best.exists()
    {
        std::fs::rename(&best, &old)
            .map_err(|e| format!("cannot rotate previous best {}: {e}", best.display()))?;
    }
    if let Err(e) = std::fs::rename(&partial, &best)
    {
        if old.exists()
        {
            let _ = std::fs::rename(&old, &best);
        }
        return Err(format!("cannot publish best {}: {e}", best.display()));
    }
    if old.exists()
    {
        let _ = std::fs::remove_dir_all(old);
    }
    Ok(())
}

/// Deterministic in-place Fisher–Yates shuffle of `order` (an SplitMix/LCG PRNG,
/// same generator as `PretrainDataset::shuffle`) — used to randomize the training
/// window order each epoch. Deterministic in `seed`, so a run is reproducible.
fn shuffle_windows(order: &mut [usize], seed: u64) {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..order.len()).rev()
    {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        order.swap(i, j);
    }
}

/// Delete old exact-resume `step_N/` checkpoints, keeping only the most recent
/// `keep_last`. Historical best validation lives separately in model-only `best/`,
/// so it no longer pins another multi-GB AdamW sidecar.
fn prune_checkpoints(dir: &str, keep_last: usize) {
    if keep_last == 0
    {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir)
    else
    {
        return;
    };
    let mut steps: Vec<(u64, std::path::PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().is_some_and(|t| t.is_dir()))
        .filter_map(|e| {
            let p = e.path();
            let n = p
                .file_name()?
                .to_str()?
                .strip_prefix("step_")?
                .parse::<u64>()
                .ok()?;
            Some((n, p))
        })
        .collect();
    steps.sort_by_key(|(n, _)| *n);
    let cutoff = steps.len().saturating_sub(keep_last);
    for (i, (n, p)) in steps.iter().enumerate()
    {
        if i >= cutoff
        {
            continue; // among the last `keep_last`
        }
        let _ = n;
        let _ = std::fs::remove_dir_all(p);
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
    /// Sequence length of each training sample.
    pub seq_len: usize,
    /// Number of independent sequences packed into each optimizer step.
    pub batch_size: usize,
    /// Maximum number of optimizer steps queued before exact loss/gnorm telemetry
    /// is copied to the host. Lower values improve observability; higher values reduce
    /// synchronization frequency. Logging/eval/checkpoint boundaries always flush.
    pub telemetry_interval: usize,
    /// Print a loss/lr line every this many steps (0 = never).
    pub log_interval: usize,
    /// Write an exact recovery checkpoint every this many optimizer steps
    /// (`0` disables the step cadence).
    pub save_interval: usize,
    /// Optional wall-clock recovery cadence in seconds (`0` disables it). The
    /// checkpoint is still published only after a completed optimizer step.
    pub save_interval_seconds: u64,
    /// Directory the `step_N/` checkpoints are written under.
    pub checkpoint_dir: String,
    /// Global gradient-norm clip threshold (`<= 0` disables). Default `1.0` —
    /// standard for pretraining, and what keeps a bad batch from diverging the run.
    pub max_grad_norm: f32,
    /// Fraction of deterministic distributed windows held out for validation
    /// (`0.0` disables held-out eval). Default `0.02`.
    pub val_frac: f32,
    /// Report held-out validation loss every this many steps (0 = never).
    pub eval_interval: usize,
    /// Max validation windows averaged per eval (bounds eval cost). Default `32`.
    pub eval_windows: usize,
    /// Exact-resume checkpoint retention. Historical best validation is stored
    /// separately as model-only `best/`; `0` keeps every exact checkpoint. Default `2`.
    pub keep_last: usize,
    /// Exact-resume corpus identity (B34). Zero is reserved for legacy/tests that
    /// do not provide a production corpus fingerprint.
    pub corpus_tokens: usize,
    pub corpus_hash: u64,
    /// Shuffle the training window order (re-shuffled deterministically each epoch).
    /// Default `true` — sequential streaming spikes per-step loss variance and
    /// encourages memorizing adjacent near-duplicate windows. `false` = sequential.
    pub shuffle: bool,
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
            // bf16-appropriate epsilon: larger than the fp32 default 1e-8 to keep the
            // AdamW `m/(√v+eps)` ratio well-conditioned when moments get tiny.
            adam_eps: 1e-5,
            weight_decay: 0.1,
            seq_len: 128,
            batch_size: 1,
            telemetry_interval: 25,
            log_interval: 100,
            save_interval: 500,
            save_interval_seconds: 0,
            checkpoint_dir: "checkpoints".to_string(),
            max_grad_norm: 1.0,
            val_frac: 0.02,
            eval_interval: 250,
            eval_windows: 32,
            keep_last: 2,
            corpus_tokens: 0,
            corpus_hash: 0,
            shuffle: true,
        }
    }
}
