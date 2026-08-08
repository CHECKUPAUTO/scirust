//! Latency-oriented CUDA inference path for SCIAGENT.
//!
//! `CudaDecodeModel` is separate from Route B's [`crate::cuda_model::CudaModel`].
//! The latter remains the correctness oracle. I250 fuses Q/K/V, gate/up and GQA,
//! reuses persistent activation/KV buffers, and offers a greedy device-feedback burst
//! that avoids per-token logit readback.

use scirust_core::autodiff::reverse::Tensor;
use scirust_cuda::{
    CudaDecodeGreedyFeedback, CudaDecodeKvCache, CudaDecodeMatrix, CudaDecodeRuntime,
};

use crate::generate::{SamplingParams, sample_row, seed_to_state};
use crate::model::SciAgentModel;

struct DecodeBlock {
    norm1: CudaDecodeMatrix,
    qkv: CudaDecodeMatrix,
    wo: CudaDecodeMatrix,
    norm2: CudaDecodeMatrix,
    gate_up: CudaDecodeMatrix,
    down: CudaDecodeMatrix,
}

struct DecodeLayerCache {
    k: CudaDecodeKvCache,
    v: CudaDecodeKvCache,
}

struct DecodeWorkspace {
    x: CudaDecodeMatrix,
    norm: CudaDecodeMatrix,
    qkv: CudaDecodeMatrix,
    ctx: CudaDecodeMatrix,
    tmp_d: CudaDecodeMatrix,
    h: CudaDecodeMatrix,
    gate_up: CudaDecodeMatrix,
    act: CudaDecodeMatrix,
    logits: CudaDecodeMatrix,
}

/// Batch-one SCIAGENT decoder with fused projection weights and fixed-capacity KV.
pub struct CudaDecodeModel {
    runtime: CudaDecodeRuntime,
    embedding: CudaDecodeMatrix,
    final_norm: CudaDecodeMatrix,
    blocks: Vec<DecodeBlock>,
    d_model: usize,
    d_ff: usize,
    n_heads: usize,
    n_kv_heads: usize,
    kv_dim: usize,
    theta: f32,
    eps: f32,
    vocab: usize,
    max_seq_len: usize,
}

impl CudaDecodeModel {
    /// Mirror one immutable CPU model snapshot into the I250 runtime.
    #[must_use]
    pub fn from_model(model: &SciAgentModel) -> Option<Self> {
        assert!(
            model.config.tie_embeddings,
            "CudaDecodeModel currently requires tied embeddings"
        );
        let runtime = CudaDecodeRuntime::new()?;
        let config = &model.config;
        assert!(config.n_heads > 0 && config.n_kv_heads > 0);
        assert!(config.n_heads.is_multiple_of(config.n_kv_heads));
        assert!(config.d_model.is_multiple_of(config.n_heads));
        let d_head = config.d_model / config.n_heads;
        let kv_dim = config.n_kv_heads * d_head;

        let embedding = runtime.upload(
            &model.embed.weight.data,
            model.embed.weight.rows,
            model.embed.weight.cols,
        );
        let final_norm = runtime.upload(
            &model.rms_final.weight.data,
            model.rms_final.weight.rows,
            model.rms_final.weight.cols,
        );

        let mut blocks = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let qkv_host = fuse_columns(&[
                &layer.attn.w_q.weight,
                &layer.attn.w_k.weight,
                &layer.attn.w_v.weight,
            ]);
            let gate_up_host = fuse_columns(&[&layer.ffn.gate.weight, &layer.ffn.up.weight]);
            blocks.push(DecodeBlock {
                norm1: runtime.upload(
                    &layer.rms_attn.weight.data,
                    layer.rms_attn.weight.rows,
                    layer.rms_attn.weight.cols,
                ),
                qkv: runtime.upload(&qkv_host, config.d_model, config.d_model + 2 * kv_dim),
                wo: runtime.upload(
                    &layer.attn.w_o.weight.data,
                    layer.attn.w_o.weight.rows,
                    layer.attn.w_o.weight.cols,
                ),
                norm2: runtime.upload(
                    &layer.rms_ffn.weight.data,
                    layer.rms_ffn.weight.rows,
                    layer.rms_ffn.weight.cols,
                ),
                gate_up: runtime.upload(&gate_up_host, config.d_model, 2 * config.d_ff),
                down: runtime.upload(
                    &layer.ffn.down.weight.data,
                    layer.ffn.down.weight.rows,
                    layer.ffn.down.weight.cols,
                ),
            });
        }

        Some(Self {
            runtime,
            embedding,
            final_norm,
            blocks,
            d_model: config.d_model,
            d_ff: config.d_ff,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            kv_dim,
            theta: config.rope_theta,
            eps: config.eps,
            vocab: config.vocab_size,
            max_seq_len: config.max_seq_len,
        })
    }

    #[must_use]
    pub const fn vocab(&self) -> usize {
        self.vocab
    }

    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn caches(&self, capacity: usize) -> Vec<DecodeLayerCache> {
        (0..self.blocks.len())
            .map(|_| DecodeLayerCache {
                k: self.runtime.kv_cache(capacity, self.kv_dim),
                v: self.runtime.kv_cache(capacity, self.kv_dim),
            })
            .collect()
    }

    fn workspace(&self) -> DecodeWorkspace {
        let runtime = &self.runtime;
        DecodeWorkspace {
            x: runtime.matrix(1, self.d_model),
            norm: runtime.matrix(1, self.d_model),
            qkv: runtime.matrix(1, self.d_model + 2 * self.kv_dim),
            ctx: runtime.matrix(1, self.d_model),
            tmp_d: runtime.matrix(1, self.d_model),
            h: runtime.matrix(1, self.d_model),
            gate_up: runtime.matrix(1, 2 * self.d_ff),
            act: runtime.matrix(1, self.d_ff),
            logits: runtime.matrix(1, self.vocab),
        }
    }

    /// Continue the token step after `workspace.x` has been populated by either a
    /// host-token embedding or the device-feedback embedding kernel.
    fn forward_embedded_resident(
        &self,
        pos: usize,
        caches: &mut [DecodeLayerCache],
        workspace: &mut DecodeWorkspace,
    ) {
        assert_eq!(caches.len(), self.blocks.len(), "decode cache/layer mismatch");
        let runtime = &self.runtime;
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            runtime.rms_norm_into(&workspace.x, &block.norm1, self.eps, &mut workspace.norm);
            runtime.matmul_into(&workspace.norm, &block.qkv, &mut workspace.qkv);
            runtime.gqa_decode_into(
                &workspace.qkv,
                &mut cache.k,
                &mut cache.v,
                pos,
                self.d_model,
                self.n_heads,
                self.n_kv_heads,
                self.theta,
                &mut workspace.ctx,
            );
            runtime.matmul_into(&workspace.ctx, &block.wo, &mut workspace.tmp_d);
            runtime.add_into(&workspace.x, &workspace.tmp_d, &mut workspace.h);

            runtime.rms_norm_into(&workspace.h, &block.norm2, self.eps, &mut workspace.norm);
            runtime.matmul_into(&workspace.norm, &block.gate_up, &mut workspace.gate_up);
            runtime.swiglu_split_into(&workspace.gate_up, &mut workspace.act);
            runtime.matmul_into(&workspace.act, &block.down, &mut workspace.tmp_d);
            runtime.add_into(&workspace.h, &workspace.tmp_d, &mut workspace.x);
        }

        runtime.rms_norm_into(&workspace.x, &self.final_norm, self.eps, &mut workspace.norm);
        runtime.matmul_bt_into(&workspace.norm, &self.embedding, &mut workspace.logits);
    }

    fn forward_host_token_resident(
        &self,
        token: u32,
        pos: usize,
        caches: &mut [DecodeLayerCache],
        workspace: &mut DecodeWorkspace,
    ) {
        self.runtime
            .embed_token_into(token, &self.embedding, &mut workspace.x);
        self.forward_embedded_resident(pos, caches, workspace);
    }

    fn forward_feedback_token_resident(
        &self,
        feedback: &CudaDecodeGreedyFeedback,
        pos: usize,
        caches: &mut [DecodeLayerCache],
        workspace: &mut DecodeWorkspace,
    ) {
        self.runtime
            .embed_feedback_into(feedback, &self.embedding, &mut workspace.x);
        self.forward_embedded_resident(pos, caches, workspace);
    }

    /// General deterministic host-sampler path retained as an oracle for sampling
    /// modes not yet promoted to device feedback.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Vec<u32> {
        let mut tokens = normalized_prompt(prompt);
        if max_new == 0 {
            return tokens;
        }
        self.assert_capacity(tokens.len(), max_new);
        let capacity = tokens.len() + max_new;
        let mut caches = self.caches(capacity);
        let mut workspace = self.workspace();

        for (pos, &token) in tokens.iter().enumerate() {
            self.forward_host_token_resident(token, pos, &mut caches, &mut workspace);
        }
        let mut logits = self.runtime.download(&workspace.logits);
        let mut rng = seed_to_state(seed);

        for generated_index in 0..max_new {
            let recent: Vec<usize> = tokens.iter().map(|&token| token as usize).collect();
            let next = sample_row(&logits, params, &recent, &mut rng) as u32;
            let pos = tokens.len();
            tokens.push(next);
            if next == 0 || generated_index + 1 == max_new {
                break;
            }
            self.forward_host_token_resident(next, pos, &mut caches, &mut workspace);
            logits = self.runtime.download(&workspace.logits);
        }
        tokens
    }

    /// Greedy batch-one generation with token feedback entirely on CUDA.
    ///
    /// After prompt prefill, the host submits a fixed burst. Every selected token is
    /// written to `current_token`, consumed by the next embedding launch and copied
    /// into a resident generated-token buffer. There is no per-token D2H or H2D. One
    /// compact `u32[max_new]` readback completes the burst.
    ///
    /// The first version intentionally does not gate later launches after EOS. The
    /// returned sequence is truncated at the first EOS, so emitted-token semantics
    /// are exact; post-EOS work is only wasted compute and is the next optimization.
    pub fn generate_greedy_device_feedback(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut tokens = normalized_prompt(prompt);
        if max_new == 0 {
            return tokens;
        }
        self.assert_capacity(tokens.len(), max_new);
        let capacity = tokens.len() + max_new;
        let mut caches = self.caches(capacity);
        let mut workspace = self.workspace();

        for (pos, &token) in tokens.iter().enumerate() {
            self.forward_host_token_resident(token, pos, &mut caches, &mut workspace);
        }

        let mut feedback = self.runtime.greedy_feedback(max_new);
        self.runtime
            .greedy_argmax_into(&workspace.logits, &mut feedback, 0);

        for generated_index in 1..max_new {
            let pos = tokens.len() + generated_index - 1;
            self.forward_feedback_token_resident(
                &feedback,
                pos,
                &mut caches,
                &mut workspace,
            );
            self.runtime
                .greedy_argmax_into(&workspace.logits, &mut feedback, generated_index);
        }

        for token in self.runtime.download_feedback(&feedback) {
            tokens.push(token);
            if token == 0 {
                break;
            }
        }
        tokens
    }

    fn assert_capacity(&self, prompt_len: usize, max_new: usize) {
        let capacity = prompt_len
            .checked_add(max_new)
            .expect("decode sequence length overflow");
        assert!(
            capacity <= self.max_seq_len,
            "decode request needs {capacity} positions, model max_seq_len is {}",
            self.max_seq_len
        );
    }
}

fn normalized_prompt(prompt: &[u32]) -> Vec<u32> {
    if prompt.is_empty() {
        vec![0]
    } else {
        prompt.to_vec()
    }
}

fn fuse_columns(parts: &[&Tensor]) -> Vec<f32> {
    assert!(!parts.is_empty(), "fuse_columns requires at least one matrix");
    let rows = parts[0].rows;
    assert!(parts.iter().all(|part| part.rows == rows), "fuse_columns row mismatch");
    let cols: usize = parts.iter().map(|part| part.cols).sum();
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let mut destination_col = 0usize;
        for part in parts {
            let source = &part.data[row * part.cols..(row + 1) * part.cols];
            let destination = &mut output
                [row * cols + destination_col..row * cols + destination_col + part.cols];
            destination.copy_from_slice(source);
            destination_col += part.cols;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_columns_preserve_row_major_order() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Tensor::from_vec(vec![5.0, 6.0], 2, 1);
        assert_eq!(
            fuse_columns(&[&a, &b]),
            vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
        );
    }
}
