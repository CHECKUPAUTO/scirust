//! Latency-oriented CUDA inference path for SCIAGENT.
//!
//! The Route-B [`crate::cuda_model::CudaModel`] remains the correctness oracle and
//! training implementation.  `CudaDecodeModel` is a separate batch-one runtime that
//! fuses Q/K/V and gate/up projections and delegates incremental GQA to
//! `scirust-cuda`'s fixed-cache fused decode kernel.  It never mutates training state.

use scirust_core::autodiff::reverse::Tensor;
use scirust_cuda::{CudaDecodeKvCache, CudaDecodeMatrix, CudaDecodeRuntime};

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

/// Batch-one SCIAGENT decoder with fused projection weights and fixed-capacity KV.
pub struct CudaDecodeModel {
    runtime: CudaDecodeRuntime,
    embedding: CudaDecodeMatrix,
    final_norm: CudaDecodeMatrix,
    blocks: Vec<DecodeBlock>,
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    kv_dim: usize,
    theta: f32,
    eps: f32,
    vocab: usize,
    max_seq_len: usize,
}

impl CudaDecodeModel {
    /// Build the latency-oriented mirror from the same fp32 model weights used by
    /// Route B.  Returns `None` when CUDA/cuBLASLt/NVRTC are unavailable.
    pub fn from_model(model: &SciAgentModel) -> Option<Self> {
        assert!(
            model.config.tie_embeddings,
            "CudaDecodeModel currently requires tied embeddings"
        );
        let runtime = CudaDecodeRuntime::new()?;
        let cfg = &model.config;
        assert!(cfg.n_heads > 0 && cfg.n_kv_heads > 0);
        assert!(cfg.n_heads.is_multiple_of(cfg.n_kv_heads));
        assert!(cfg.d_model.is_multiple_of(cfg.n_heads));
        let d_head = cfg.d_model / cfg.n_heads;
        let kv_dim = cfg.n_kv_heads * d_head;

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
                qkv: runtime.upload(&qkv_host, cfg.d_model, cfg.d_model + 2 * kv_dim),
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
                gate_up: runtime.upload(&gate_up_host, cfg.d_model, 2 * cfg.d_ff),
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
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            kv_dim,
            theta: cfg.rope_theta,
            eps: cfg.eps,
            vocab: cfg.vocab_size,
            max_seq_len: cfg.max_seq_len,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn max_seq_len(&self) -> usize {
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

    /// Process one input token at its absolute position, append its layer-local K/V
    /// rows to the fixed caches, and return logits for the following token.
    fn forward_token(
        &self,
        token: u32,
        pos: usize,
        caches: &mut [DecodeLayerCache],
    ) -> Vec<f32> {
        assert_eq!(caches.len(), self.blocks.len(), "decode cache/layer mismatch");
        let rt = &self.runtime;
        let mut x = rt.embed_token(token, &self.embedding);

        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            let xn = rt.rms_norm(&x, &block.norm1, self.eps);
            let qkv = rt.matmul(&xn, &block.qkv);
            let ctx = rt.gqa_decode(
                &qkv,
                &mut cache.k,
                &mut cache.v,
                pos,
                self.d_model,
                self.n_heads,
                self.n_kv_heads,
                self.theta,
            );
            let attn_out = rt.matmul(&ctx, &block.wo);
            let h = rt.add(&x, &attn_out);

            let hn = rt.rms_norm(&h, &block.norm2, self.eps);
            let gate_up = rt.matmul(&hn, &block.gate_up);
            let act = rt.swiglu_split(&gate_up);
            let mlp = rt.matmul(&act, &block.down);
            x = rt.add(&h, &mlp);
        }

        let normed = rt.rms_norm(&x, &self.final_norm, self.eps);
        let logits = rt.matmul_bt(&normed, &self.embedding);
        rt.download(&logits)
    }

    /// Autoregressive generation using the fused batch-one CUDA path.
    ///
    /// Prompt prefill is intentionally incremental: replaying the prompt through the
    /// same one-token kernel both populates KV and exercises exactly the production
    /// decode path.  Sampling uses SCIAGENT's shared deterministic host sampler so
    /// parity can be checked directly against [`crate::cuda_model::CudaModel`].
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Vec<u32> {
        let mut tokens = if prompt.is_empty() {
            vec![0]
        } else {
            prompt.to_vec()
        };
        if max_new == 0 {
            return tokens;
        }

        let capacity = tokens
            .len()
            .checked_add(max_new)
            .expect("decode sequence length overflow");
        assert!(
            capacity <= self.max_seq_len,
            "decode request needs {capacity} positions, model max_seq_len is {}",
            self.max_seq_len
        );
        let mut caches = self.caches(capacity);

        // Sequential prefill: after the final prompt token, `logits` predicts the
        // first generated token and every cache contains exactly the prompt prefix.
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            logits = self.forward_token(token, pos, &mut caches);
        }

        let mut rng = seed_to_state(seed);
        for i in 0..max_new {
            let recent: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
            let next = sample_row(&logits, params, &recent, &mut rng) as u32;
            let pos = tokens.len();
            tokens.push(next);
            if next == 0 || i + 1 == max_new {
                break;
            }
            logits = self.forward_token(next, pos, &mut caches);
        }
        tokens
    }
}

fn fuse_columns(parts: &[&Tensor]) -> Vec<f32> {
    assert!(!parts.is_empty(), "fuse_columns requires at least one matrix");
    let rows = parts[0].rows;
    assert!(parts.iter().all(|p| p.rows == rows), "fuse_columns row mismatch");
    let cols: usize = parts.iter().map(|p| p.cols).sum();
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut dst_col = 0usize;
        for part in parts {
            let src = &part.data[r * part.cols..(r + 1) * part.cols];
            let dst = &mut out[r * cols + dst_col..r * cols + dst_col + part.cols];
            dst.copy_from_slice(src);
            dst_col += part.cols;
        }
    }
    out
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
