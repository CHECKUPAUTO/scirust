//! Reconstruction-free ElasticKV CUDA inference for SCIAGENT.
//!
//! This is the I250-B path. It consumes [`crate::elastic_decode_plan::ElasticDecodePlan`]
//! rather than dense Q/K/V/O weights, projects directly into latent coordinates,
//! retains only latent K/V history, and applies the absorbed latent output matrix
//! without reconstructing dense per-head contexts.

use core::fmt;

use scirust_core::autodiff::reverse::Tensor;
use scirust_cuda::{CudaElasticDecodeRuntime, CudaElasticKvCache, CudaElasticMatrix};

use crate::elastic_decode_plan::{
    ElasticDecodePlan, ElasticDecodePlanError, LatentGqaBases, RotaryLatentRule,
};
use crate::generate::{SamplingParams, sample_row, seed_to_state};
use crate::model::SciAgentModel;

#[derive(Debug)]
pub enum CudaElasticDecodeError {
    Plan(ElasticDecodePlanError),
    RuntimeUnavailable,
    UnsupportedRotaryRule {
        layer: usize,
        rule: RotaryLatentRule,
    },
    NonUniformRank {
        layer: usize,
        key_rank: usize,
        value_rank: usize,
        expected_key_rank: usize,
        expected_value_rank: usize,
    },
}

impl fmt::Display for CudaElasticDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Plan(error) => write!(formatter, "{error}"),
            Self::RuntimeUnavailable =>
            {
                write!(formatter, "Elastic CUDA decode runtime unavailable")
            },
            Self::UnsupportedRotaryRule { layer, rule } => write!(
                formatter,
                "Elastic CUDA layer {layer} does not yet support rotary rule {rule:?}"
            ),
            Self::NonUniformRank {
                layer,
                key_rank,
                value_rank,
                expected_key_rank,
                expected_value_rank,
            } => write!(
                formatter,
                "Elastic CUDA layer {layer} ranks ({key_rank},{value_rank}) differ from required uniform ({expected_key_rank},{expected_value_rank})"
            ),
        }
    }
}

impl std::error::Error for CudaElasticDecodeError {}

impl From<ElasticDecodePlanError> for CudaElasticDecodeError {
    fn from(error: ElasticDecodePlanError) -> Self {
        Self::Plan(error)
    }
}

struct ElasticBlock {
    norm1: CudaElasticMatrix,
    qkv_latent: CudaElasticMatrix,
    o_latent: CudaElasticMatrix,
    norm2: CudaElasticMatrix,
    gate_up: CudaElasticMatrix,
    down: CudaElasticMatrix,
}

struct ElasticLayerCache {
    k: CudaElasticKvCache,
    v: CudaElasticKvCache,
}

struct ElasticWorkspace {
    x: CudaElasticMatrix,
    norm: CudaElasticMatrix,
    qkv_latent: CudaElasticMatrix,
    context_latent: CudaElasticMatrix,
    tmp_d: CudaElasticMatrix,
    h: CudaElasticMatrix,
    gate_up: CudaElasticMatrix,
    act: CudaElasticMatrix,
    logits: CudaElasticMatrix,
}

/// Batch-one reconstruction-free SCIAGENT CUDA decoder.
pub struct CudaElasticDecodeModel {
    runtime: CudaElasticDecodeRuntime,
    embedding: CudaElasticMatrix,
    final_norm: CudaElasticMatrix,
    blocks: Vec<ElasticBlock>,
    d_model: usize,
    d_ff: usize,
    d_head: usize,
    n_heads: usize,
    n_kv_heads: usize,
    key_rank: usize,
    value_rank: usize,
    qkv_width: usize,
    context_width: usize,
    theta: f32,
    eps: f32,
    vocab: usize,
    max_seq_len: usize,
    dense_equivalent_identity: bool,
}

impl CudaElasticDecodeModel {
    /// Construct the no-loss structural oracle with full-rank identity bases.
    pub fn from_model_identity(model: &SciAgentModel) -> Result<Self, CudaElasticDecodeError> {
        let bases: Vec<_> = model
            .layers
            .iter()
            .map(|layer| LatentGqaBases::identity(layer.attn.n_kv_heads, layer.attn.d_head))
            .collect();
        Self::from_model_with_bases(model, &bases)
    }

    /// Construct a uniform reduced native-pair path. RoPE stays exact inside the
    /// retained Q/K subspace; reduction quality is intentionally a separate gate.
    pub fn from_model_native_pair_prefix(
        model: &SciAgentModel,
        key_rank: usize,
        value_rank: usize,
    ) -> Result<Self, CudaElasticDecodeError> {
        let mut bases = Vec::with_capacity(model.layers.len());
        for layer in &model.layers
        {
            bases.push(LatentGqaBases::native_pair_prefix(
                layer.attn.n_kv_heads,
                layer.attn.d_head,
                key_rank,
                value_rank,
            )?);
        }
        Self::from_model_with_bases(model, &bases)
    }

    /// Construct from an explicit immutable ElasticKV basis snapshot.
    ///
    /// The first CUDA kernel intentionally accepts only full identity and native
    /// pair-prefix bases. General learned bases remain in the planner but fail closed
    /// here until their projected RoPE operator exists on CUDA.
    pub fn from_model_with_bases(
        model: &SciAgentModel,
        bases: &[LatentGqaBases],
    ) -> Result<Self, CudaElasticDecodeError> {
        assert!(
            model.config.tie_embeddings,
            "CudaElasticDecodeModel currently requires tied embeddings"
        );
        let plan = ElasticDecodePlan::from_model(model, bases)?;
        let first = plan
            .layers()
            .first()
            .ok_or(ElasticDecodePlanError::InvalidTopology(
                "Elastic CUDA decode requires at least one model layer",
            ))?;
        let key_rank = first.key_rank();
        let value_rank = first.value_rank();
        for (index, layer) in plan.layers().iter().enumerate()
        {
            if layer.key_rank() != key_rank || layer.value_rank() != value_rank
            {
                return Err(CudaElasticDecodeError::NonUniformRank {
                    layer: index,
                    key_rank: layer.key_rank(),
                    value_rank: layer.value_rank(),
                    expected_key_rank: key_rank,
                    expected_value_rank: value_rank,
                });
            }
            if matches!(layer.rotary_rule(), RotaryLatentRule::ProjectedOperator)
            {
                return Err(CudaElasticDecodeError::UnsupportedRotaryRule {
                    layer: index,
                    rule: layer.rotary_rule(),
                });
            }
        }

        let runtime =
            CudaElasticDecodeRuntime::new().ok_or(CudaElasticDecodeError::RuntimeUnavailable)?;
        let config = &model.config;
        assert!(config.n_heads > 0 && config.n_kv_heads > 0);
        assert!(config.n_heads.is_multiple_of(config.n_kv_heads));
        assert!(config.d_model.is_multiple_of(config.n_heads));
        let d_head = config.d_model / config.n_heads;
        let q_width = config.n_heads * key_rank;
        let k_width = config.n_kv_heads * key_rank;
        let v_width = config.n_kv_heads * value_rank;
        let qkv_width = q_width + k_width + v_width;
        let context_width = config.n_heads * value_rank;

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
        for (source, absorbed) in model.layers.iter().zip(plan.layers())
        {
            let qkv_host = fuse_flat_columns(
                config.d_model,
                &[
                    (absorbed.q_latent(), q_width),
                    (absorbed.k_latent(), k_width),
                    (absorbed.v_latent(), v_width),
                ],
            );
            let gate_up_host = fuse_columns(&[&source.ffn.gate.weight, &source.ffn.up.weight]);
            blocks.push(ElasticBlock {
                norm1: runtime.upload(
                    &source.rms_attn.weight.data,
                    source.rms_attn.weight.rows,
                    source.rms_attn.weight.cols,
                ),
                qkv_latent: runtime.upload(&qkv_host, config.d_model, qkv_width),
                o_latent: runtime.upload(absorbed.o_latent(), context_width, config.d_model),
                norm2: runtime.upload(
                    &source.rms_ffn.weight.data,
                    source.rms_ffn.weight.rows,
                    source.rms_ffn.weight.cols,
                ),
                gate_up: runtime.upload(&gate_up_host, config.d_model, 2 * config.d_ff),
                down: runtime.upload(
                    &source.ffn.down.weight.data,
                    source.ffn.down.weight.rows,
                    source.ffn.down.weight.cols,
                ),
            });
        }

        let dense_equivalent_identity = plan
            .layers()
            .iter()
            .all(|layer| layer.is_dense_equivalent_identity());

        Ok(Self {
            runtime,
            embedding,
            final_norm,
            blocks,
            d_model: config.d_model,
            d_ff: config.d_ff,
            d_head,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            key_rank,
            value_rank,
            qkv_width,
            context_width,
            theta: config.rope_theta,
            eps: config.eps,
            vocab: config.vocab_size,
            max_seq_len: config.max_seq_len,
            dense_equivalent_identity,
        })
    }

    #[must_use]
    pub const fn key_rank(&self) -> usize {
        self.key_rank
    }

    #[must_use]
    pub const fn value_rank(&self) -> usize {
        self.value_rank
    }

    #[must_use]
    pub const fn is_dense_equivalent_identity(&self) -> bool {
        self.dense_equivalent_identity
    }

    fn caches(&self, capacity: usize) -> Vec<ElasticLayerCache> {
        let k_width = self.n_kv_heads * self.key_rank;
        let v_width = self.n_kv_heads * self.value_rank;
        (0..self.blocks.len())
            .map(|_| ElasticLayerCache {
                k: self.runtime.kv_cache(capacity, k_width),
                v: self.runtime.kv_cache(capacity, v_width),
            })
            .collect()
    }

    fn workspace(&self) -> ElasticWorkspace {
        let runtime = &self.runtime;
        ElasticWorkspace {
            x: runtime.matrix(1, self.d_model),
            norm: runtime.matrix(1, self.d_model),
            qkv_latent: runtime.matrix(1, self.qkv_width),
            context_latent: runtime.matrix(1, self.context_width),
            tmp_d: runtime.matrix(1, self.d_model),
            h: runtime.matrix(1, self.d_model),
            gate_up: runtime.matrix(1, 2 * self.d_ff),
            act: runtime.matrix(1, self.d_ff),
            logits: runtime.matrix(1, self.vocab),
        }
    }

    fn forward_token_resident(
        &self,
        token: u32,
        pos: usize,
        caches: &mut [ElasticLayerCache],
        workspace: &mut ElasticWorkspace,
    ) {
        assert_eq!(caches.len(), self.blocks.len());
        let runtime = &self.runtime;
        runtime.embed_token_into(token, &self.embedding, &mut workspace.x);

        for (block, cache) in self.blocks.iter().zip(caches.iter_mut())
        {
            runtime.rms_norm_into(&workspace.x, &block.norm1, self.eps, &mut workspace.norm);
            runtime.matmul_into(
                &workspace.norm,
                &block.qkv_latent,
                &mut workspace.qkv_latent,
            );
            runtime.latent_gqa_into(
                &workspace.qkv_latent,
                &mut cache.k,
                &mut cache.v,
                pos,
                self.d_head,
                self.key_rank,
                self.value_rank,
                self.n_heads,
                self.n_kv_heads,
                self.theta,
                &mut workspace.context_latent,
            );
            runtime.matmul_into(
                &workspace.context_latent,
                &block.o_latent,
                &mut workspace.tmp_d,
            );
            runtime.add_into(&workspace.x, &workspace.tmp_d, &mut workspace.h);

            runtime.rms_norm_into(&workspace.h, &block.norm2, self.eps, &mut workspace.norm);
            runtime.matmul_into(&workspace.norm, &block.gate_up, &mut workspace.gate_up);
            runtime.swiglu_split_into(&workspace.gate_up, &mut workspace.act);
            runtime.matmul_into(&workspace.act, &block.down, &mut workspace.tmp_d);
            runtime.add_into(&workspace.h, &workspace.tmp_d, &mut workspace.x);
        }

        runtime.rms_norm_into(
            &workspace.x,
            &self.final_norm,
            self.eps,
            &mut workspace.norm,
        );
        runtime.matmul_bt_into(&workspace.norm, &self.embedding, &mut workspace.logits);
    }

    /// Generate through the reconstruction-free ElasticKV path.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Vec<u32> {
        let mut tokens = if prompt.is_empty()
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

        let capacity = tokens
            .len()
            .checked_add(max_new)
            .expect("Elastic decode sequence length overflow");
        assert!(capacity <= self.max_seq_len);
        let mut caches = self.caches(capacity);
        let mut workspace = self.workspace();

        for (pos, &token) in tokens.iter().enumerate()
        {
            self.forward_token_resident(token, pos, &mut caches, &mut workspace);
        }
        let mut logits = self.runtime.download(&workspace.logits);

        let mut rng = seed_to_state(seed);
        for generated in 0..max_new
        {
            let recent: Vec<usize> = tokens.iter().map(|&token| token as usize).collect();
            let next = sample_row(&logits, params, &recent, &mut rng) as u32;
            let pos = tokens.len();
            tokens.push(next);
            if next == 0 || generated + 1 == max_new
            {
                break;
            }
            self.forward_token_resident(next, pos, &mut caches, &mut workspace);
            logits = self.runtime.download(&workspace.logits);
        }
        tokens
    }
}

fn fuse_columns(parts: &[&Tensor]) -> Vec<f32> {
    assert!(!parts.is_empty());
    let rows = parts[0].rows;
    assert!(parts.iter().all(|part| part.rows == rows));
    let cols: usize = parts.iter().map(|part| part.cols).sum();
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows
    {
        let mut destination_col = 0usize;
        for part in parts
        {
            let source = &part.data[row * part.cols..(row + 1) * part.cols];
            let destination =
                &mut output[row * cols + destination_col..row * cols + destination_col + part.cols];
            destination.copy_from_slice(source);
            destination_col += part.cols;
        }
    }
    output
}

fn fuse_flat_columns(rows: usize, parts: &[(&[f32], usize)]) -> Vec<f32> {
    assert!(!parts.is_empty());
    assert!(parts.iter().all(|(data, cols)| data.len() == rows * *cols));
    let total_cols: usize = parts.iter().map(|(_, cols)| *cols).sum();
    let mut output = vec![0.0f32; rows * total_cols];
    for row in 0..rows
    {
        let mut destination_col = 0usize;
        for &(data, cols) in parts
        {
            let source = &data[row * cols..(row + 1) * cols];
            let destination = &mut output
                [row * total_cols + destination_col..row * total_cols + destination_col + cols];
            destination.copy_from_slice(source);
            destination_col += cols;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_column_fusion_preserves_rows() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0];
        assert_eq!(
            fuse_flat_columns(2, &[(&a, 2), (&b, 1)]),
            vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
        );
    }
}
