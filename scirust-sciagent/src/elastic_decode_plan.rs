//! Backend-neutral absorbed-latent decode planning for SCIAGENT.
//!
//! Elastic Latent KV already evaluates attention without reconstructing dense keys
//! and only up-projects the final value context once. For inference we can push that
//! algebra one step earlier and later by absorbing latent bases into model weights.
//!
//! For one GQA KV head with key basis `U_k` and value basis `U_v`:
//!
//! - `Wq_lat = Wq_head * U_k`
//! - `Wk_lat = Wk_head * U_k`
//! - `Wv_lat = Wv_head * U_v`
//! - `Wo_lat = U_v^T * Wo_head`
//!
//! The V/O identities are position independent. Q/K need an additional RoPE rule:
//! SCIAGENT rotates Q/K after projection, so a reduced arbitrary basis cannot simply
//! reuse ordinary RoPE with `head_dim = latent_rank`. The plan classifies each basis
//! as full identity, a native complete-pair prefix, or a general projected basis.
//! The first two have a cheap exact rotary operator for their represented subspace;
//! the general case requires a projected position operator and is quality-gated when
//! rank is reduced. This distinction is explicit so a CUDA backend cannot silently
//! turn a mathematical approximation into an exactness claim.
//!
//! This module performs deterministic plan construction only. It is independent of
//! CUDA/WGPU so the algebra and memory accounting remain testable on every CI target.

use core::fmt;

use crate::attention::GQAAttention;
use crate::model::SciAgentModel;

/// How a key-latent basis must handle SCIAGENT's head-local RoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryLatentRule {
    /// Full-rank identity coordinates: use the existing dense head-local RoPE.
    FullIdentity,
    /// The basis selects the first `rank / 2` complete native RoPE pairs.
    ///
    /// The latent kernel must retain the **original `d_head` frequency denominator**;
    /// using `rank` as the denominator would change semantics.
    NativePairPrefix {
        rank: usize,
        frequency_denominator: usize,
    },
    /// General basis. A backend must apply the basis-projected position operator
    /// rather than ordinary latent-width RoPE.
    ProjectedOperator,
}

/// Invalid topology, basis, or source-weight shape during absorbed-plan creation.
#[derive(Debug, Clone, PartialEq)]
pub enum ElasticDecodePlanError {
    InvalidTopology(&'static str),
    InvalidRank {
        name: &'static str,
        rank: usize,
        d_head: usize,
    },
    BasisLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteBasis {
        name: &'static str,
        index: usize,
    },
    WeightShape {
        name: &'static str,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    LayerBasisCount {
        layers: usize,
        bases: usize,
    },
    Overflow,
}

impl fmt::Display for ElasticDecodePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidTopology(message) => write!(formatter, "{message}"),
            Self::InvalidRank { name, rank, d_head } => write!(
                formatter,
                "{name} latent rank {rank} is outside 1..={d_head}"
            ),
            Self::BasisLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} basis length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteBasis { name, index } =>
            {
                write!(formatter, "{name} basis contains non-finite value at {index}")
            },
            Self::WeightShape {
                name,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                formatter,
                "{name} shape mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}"
            ),
            Self::LayerBasisCount { layers, bases } => write!(
                formatter,
                "absorbed decode basis count mismatch: model has {layers} layers, got {bases}"
            ),
            Self::Overflow => write!(formatter, "absorbed decode plan size overflow"),
        }
    }
}

impl std::error::Error for ElasticDecodePlanError {}

/// Per-layer GQA latent bases, packed KV-head-major.
#[derive(Debug, Clone)]
pub struct LatentGqaBases {
    n_kv_heads: usize,
    d_head: usize,
    key_rank: usize,
    value_rank: usize,
    key: Vec<f32>,
    value: Vec<f32>,
    exact_identity: bool,
    rotary_rule: RotaryLatentRule,
}

impl LatentGqaBases {
    /// Construct explicit rectangular bases. Each head basis is row-major
    /// `[d_head, rank]`, and heads are concatenated in KV-head order.
    pub fn new(
        n_kv_heads: usize,
        d_head: usize,
        key_rank: usize,
        value_rank: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<Self, ElasticDecodePlanError> {
        if n_kv_heads == 0 || d_head == 0
        {
            return Err(ElasticDecodePlanError::InvalidTopology(
                "latent GQA requires non-zero KV heads and head dimension",
            ));
        }
        require_rank("key", key_rank, d_head)?;
        require_rank("value", value_rank, d_head)?;
        let expected_key = checked_product3(n_kv_heads, d_head, key_rank)?;
        let expected_value = checked_product3(n_kv_heads, d_head, value_rank)?;
        require_basis("key", &key, expected_key)?;
        require_basis("value", &value, expected_value)?;

        let key_identity = key_rank == d_head
            && all_heads_are_identity(&key, n_kv_heads, d_head, key_rank);
        let value_identity = value_rank == d_head
            && all_heads_are_identity(&value, n_kv_heads, d_head, value_rank);
        let exact_identity = key_identity && value_identity;
        let rotary_rule = if key_identity
        {
            RotaryLatentRule::FullIdentity
        }
        else if key_rank.is_multiple_of(2)
            && all_heads_are_native_prefix(&key, n_kv_heads, d_head, key_rank)
        {
            RotaryLatentRule::NativePairPrefix {
                rank: key_rank,
                frequency_denominator: d_head,
            }
        }
        else
        {
            RotaryLatentRule::ProjectedOperator
        };

        Ok(Self {
            n_kv_heads,
            d_head,
            key_rank,
            value_rank,
            key,
            value,
            exact_identity,
            rotary_rule,
        })
    }

    /// Full-rank exact identity bases. This is the structural oracle: plan
    /// construction copies the original Q/K/V/O slices without arithmetic so the
    /// absorbed representation is bit-identical at the weight level.
    #[must_use]
    pub fn identity(n_kv_heads: usize, d_head: usize) -> Self {
        assert!(n_kv_heads > 0 && d_head > 0);
        let per_head = d_head * d_head;
        let mut key = vec![0.0; n_kv_heads * per_head];
        let mut value = vec![0.0; n_kv_heads * per_head];
        for head in 0..n_kv_heads
        {
            let base = head * per_head;
            for diagonal in 0..d_head
            {
                key[base + diagonal * d_head + diagonal] = 1.0;
                value[base + diagonal * d_head + diagonal] = 1.0;
            }
        }
        Self {
            n_kv_heads,
            d_head,
            key_rank: d_head,
            value_rank: d_head,
            key,
            value,
            exact_identity: true,
            rotary_rule: RotaryLatentRule::FullIdentity,
        }
    }

    /// Deterministic reduced basis made of the first complete native RoPE pairs.
    ///
    /// This is intentionally simple: it gives the first CUDA latent path an exact
    /// positional operator inside the retained subspace. Model quality after dropping
    /// the remaining pairs is still an approximation and must be measured.
    pub fn native_pair_prefix(
        n_kv_heads: usize,
        d_head: usize,
        key_rank: usize,
        value_rank: usize,
    ) -> Result<Self, ElasticDecodePlanError> {
        if !key_rank.is_multiple_of(2)
        {
            return Err(ElasticDecodePlanError::InvalidTopology(
                "native RoPE prefix key rank must contain complete pairs",
            ));
        }
        let key = prefix_basis(n_kv_heads, d_head, key_rank)?;
        let value = prefix_basis(n_kv_heads, d_head, value_rank)?;
        Self::new(
            n_kv_heads,
            d_head,
            key_rank,
            value_rank,
            key,
            value,
        )
    }

    #[must_use]
    pub const fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    #[must_use]
    pub const fn d_head(&self) -> usize {
        self.d_head
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
    pub const fn rotary_rule(&self) -> RotaryLatentRule {
        self.rotary_rule
    }

    /// True only for the no-loss full-rank identity representation.
    #[must_use]
    pub const fn is_dense_equivalent_identity(&self) -> bool {
        self.exact_identity
    }

    #[must_use]
    pub fn key_head(&self, head: usize) -> &[f32] {
        assert!(head < self.n_kv_heads);
        let width = self.d_head * self.key_rank;
        &self.key[head * width..(head + 1) * width]
    }

    #[must_use]
    pub fn value_head(&self, head: usize) -> &[f32] {
        assert!(head < self.n_kv_heads);
        let width = self.d_head * self.value_rank;
        &self.value[head * width..(head + 1) * width]
    }
}

/// Projection weights for one attention layer after latent bases are absorbed.
///
/// All matrices are row-major. `q_latent` has shape
/// `[d_model, n_heads * key_rank]`; K/V use KV-head-packed output widths; and
/// `o_latent` has shape `[n_heads * value_rank, d_model]`.
#[derive(Debug, Clone)]
pub struct AbsorbedGqaWeights {
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    key_rank: usize,
    value_rank: usize,
    rotary_rule: RotaryLatentRule,
    dense_equivalent_identity: bool,
    q_latent: Vec<f32>,
    k_latent: Vec<f32>,
    v_latent: Vec<f32>,
    o_latent: Vec<f32>,
}

impl AbsorbedGqaWeights {
    pub fn from_attention(
        attention: &GQAAttention,
        bases: &LatentGqaBases,
    ) -> Result<Self, ElasticDecodePlanError> {
        validate_attention(attention, bases)?;
        let d_model = attention.d_model;
        let n_heads = attention.n_heads;
        let n_kv_heads = attention.n_kv_heads;
        let d_head = attention.d_head;
        let key_rank = bases.key_rank;
        let value_rank = bases.value_rank;
        let kv_dim = n_kv_heads
            .checked_mul(d_head)
            .ok_or(ElasticDecodePlanError::Overflow)?;

        require_weight("Wq", &attention.w_q.weight, d_model, d_model)?;
        require_weight("Wk", &attention.w_k.weight, d_model, kv_dim)?;
        require_weight("Wv", &attention.w_v.weight, d_model, kv_dim)?;
        require_weight("Wo", &attention.w_o.weight, d_model, d_model)?;

        if bases.exact_identity
        {
            return Ok(Self {
                d_model,
                n_heads,
                n_kv_heads,
                d_head,
                key_rank,
                value_rank,
                rotary_rule: bases.rotary_rule,
                dense_equivalent_identity: true,
                q_latent: attention.w_q.weight.data.clone(),
                k_latent: attention.w_k.weight.data.clone(),
                v_latent: attention.w_v.weight.data.clone(),
                o_latent: attention.w_o.weight.data.clone(),
            });
        }

        let q_cols = checked_product(n_heads, key_rank)?;
        let k_cols = checked_product(n_kv_heads, key_rank)?;
        let v_cols = checked_product(n_kv_heads, value_rank)?;
        let o_rows = checked_product(n_heads, value_rank)?;
        let mut q_latent = vec![0.0; checked_product(d_model, q_cols)?];
        let mut k_latent = vec![0.0; checked_product(d_model, k_cols)?];
        let mut v_latent = vec![0.0; checked_product(d_model, v_cols)?];
        let mut o_latent = vec![0.0; checked_product(o_rows, d_model)?];
        let repeat = n_heads / n_kv_heads;

        // Q heads share the key basis of their GQA KV group.
        for input in 0..d_model
        {
            for q_head in 0..n_heads
            {
                let kv_head = q_head / repeat;
                let basis = bases.key_head(kv_head);
                for latent in 0..key_rank
                {
                    let mut sum = 0.0f32;
                    for channel in 0..d_head
                    {
                        let source_col = q_head * d_head + channel;
                        sum += attention.w_q.weight.data[input * d_model + source_col]
                            * basis[channel * key_rank + latent];
                    }
                    q_latent[input * q_cols + q_head * key_rank + latent] = sum;
                }
            }
        }

        for input in 0..d_model
        {
            for kv_head in 0..n_kv_heads
            {
                let key_basis = bases.key_head(kv_head);
                let value_basis = bases.value_head(kv_head);
                for latent in 0..key_rank
                {
                    let mut sum = 0.0f32;
                    for channel in 0..d_head
                    {
                        let source_col = kv_head * d_head + channel;
                        sum += attention.w_k.weight.data[input * kv_dim + source_col]
                            * key_basis[channel * key_rank + latent];
                    }
                    k_latent[input * k_cols + kv_head * key_rank + latent] = sum;
                }
                for latent in 0..value_rank
                {
                    let mut sum = 0.0f32;
                    for channel in 0..d_head
                    {
                        let source_col = kv_head * d_head + channel;
                        sum += attention.w_v.weight.data[input * kv_dim + source_col]
                            * value_basis[channel * value_rank + latent];
                    }
                    v_latent[input * v_cols + kv_head * value_rank + latent] = sum;
                }
            }
        }

        // Absorb U_v^T into each query head's row block of W_o. Multiple query
        // heads in one GQA group intentionally reuse the same value basis.
        for q_head in 0..n_heads
        {
            let kv_head = q_head / repeat;
            let basis = bases.value_head(kv_head);
            for latent in 0..value_rank
            {
                let dst_row = q_head * value_rank + latent;
                for output in 0..d_model
                {
                    let mut sum = 0.0f32;
                    for channel in 0..d_head
                    {
                        let source_row = q_head * d_head + channel;
                        sum += basis[channel * value_rank + latent]
                            * attention.w_o.weight.data[source_row * d_model + output];
                    }
                    o_latent[dst_row * d_model + output] = sum;
                }
            }
        }

        Ok(Self {
            d_model,
            n_heads,
            n_kv_heads,
            d_head,
            key_rank,
            value_rank,
            rotary_rule: bases.rotary_rule,
            dense_equivalent_identity: false,
            q_latent,
            k_latent,
            v_latent,
            o_latent,
        })
    }

    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    #[must_use]
    pub const fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    #[must_use]
    pub const fn d_head(&self) -> usize {
        self.d_head
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
    pub const fn rotary_rule(&self) -> RotaryLatentRule {
        self.rotary_rule
    }

    #[must_use]
    pub const fn is_dense_equivalent_identity(&self) -> bool {
        self.dense_equivalent_identity
    }

    #[must_use]
    pub fn q_latent(&self) -> &[f32] {
        &self.q_latent
    }

    #[must_use]
    pub fn k_latent(&self) -> &[f32] {
        &self.k_latent
    }

    #[must_use]
    pub fn v_latent(&self) -> &[f32] {
        &self.v_latent
    }

    #[must_use]
    pub fn o_latent(&self) -> &[f32] {
        &self.o_latent
    }

    /// Number of Q/K/V/O projection scalars consumed by this latent decode layer.
    #[must_use]
    pub fn projection_parameter_count(&self) -> usize {
        self.q_latent
            .len()
            .saturating_add(self.k_latent.len())
            .saturating_add(self.v_latent.len())
            .saturating_add(self.o_latent.len())
    }

    /// Number of projection scalars in the original dense GQA layer.
    #[must_use]
    pub fn dense_projection_parameter_count(&self) -> usize {
        let kv_dim = self.n_kv_heads.saturating_mul(self.d_head);
        self.d_model
            .saturating_mul(self.d_model)
            .saturating_mul(2)
            .saturating_add(
                self.d_model
                    .saturating_mul(kv_dim)
                    .saturating_mul(2),
            )
    }
}

/// All attention-layer absorbed weights for one immutable SCIAGENT model snapshot.
#[derive(Debug, Clone)]
pub struct ElasticDecodePlan {
    layers: Vec<AbsorbedGqaWeights>,
}

impl ElasticDecodePlan {
    pub fn from_model(
        model: &SciAgentModel,
        bases: &[LatentGqaBases],
    ) -> Result<Self, ElasticDecodePlanError> {
        if bases.len() != model.layers.len()
        {
            return Err(ElasticDecodePlanError::LayerBasisCount {
                layers: model.layers.len(),
                bases: bases.len(),
            });
        }
        let mut layers = Vec::with_capacity(model.layers.len());
        for (layer, basis) in model.layers.iter().zip(bases)
        {
            layers.push(AbsorbedGqaWeights::from_attention(&layer.attn, basis)?);
        }
        Ok(Self { layers })
    }

    /// Exact full-rank identity plan for differential validation.
    #[must_use]
    pub fn identity(model: &SciAgentModel) -> Self {
        let bases: Vec<_> = model
            .layers
            .iter()
            .map(|layer| LatentGqaBases::identity(layer.attn.n_kv_heads, layer.attn.d_head))
            .collect();
        Self::from_model(model, &bases).expect("model topology must accept identity latent bases")
    }

    #[must_use]
    pub fn layers(&self) -> &[AbsorbedGqaWeights] {
        &self.layers
    }

    #[must_use]
    pub fn projection_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(AbsorbedGqaWeights::projection_parameter_count)
            .sum()
    }

    #[must_use]
    pub fn dense_projection_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(AbsorbedGqaWeights::dense_projection_parameter_count)
            .sum()
    }
}

fn validate_attention(
    attention: &GQAAttention,
    bases: &LatentGqaBases,
) -> Result<(), ElasticDecodePlanError> {
    if attention.n_heads == 0 || attention.n_kv_heads == 0 || attention.d_head == 0
    {
        return Err(ElasticDecodePlanError::InvalidTopology(
            "attention topology must be non-zero",
        ));
    }
    if !attention.n_heads.is_multiple_of(attention.n_kv_heads)
    {
        return Err(ElasticDecodePlanError::InvalidTopology(
            "GQA query-head count must be divisible by KV-head count",
        ));
    }
    if attention.d_model != attention.n_heads * attention.d_head
    {
        return Err(ElasticDecodePlanError::InvalidTopology(
            "attention d_model must equal n_heads * d_head",
        ));
    }
    if bases.n_kv_heads != attention.n_kv_heads || bases.d_head != attention.d_head
    {
        return Err(ElasticDecodePlanError::InvalidTopology(
            "latent basis topology does not match attention",
        ));
    }
    Ok(())
}

fn require_rank(
    name: &'static str,
    rank: usize,
    d_head: usize,
) -> Result<(), ElasticDecodePlanError> {
    if rank == 0 || rank > d_head
    {
        return Err(ElasticDecodePlanError::InvalidRank { name, rank, d_head });
    }
    Ok(())
}

fn require_basis(
    name: &'static str,
    basis: &[f32],
    expected: usize,
) -> Result<(), ElasticDecodePlanError> {
    if basis.len() != expected
    {
        return Err(ElasticDecodePlanError::BasisLength {
            name,
            expected,
            actual: basis.len(),
        });
    }
    if let Some(index) = basis.iter().position(|value| !value.is_finite())
    {
        return Err(ElasticDecodePlanError::NonFiniteBasis { name, index });
    }
    Ok(())
}

fn require_weight(
    name: &'static str,
    weight: &scirust_core::autodiff::reverse::Tensor,
    rows: usize,
    cols: usize,
) -> Result<(), ElasticDecodePlanError> {
    if weight.rows != rows || weight.cols != cols
    {
        return Err(ElasticDecodePlanError::WeightShape {
            name,
            expected_rows: rows,
            expected_cols: cols,
            actual_rows: weight.rows,
            actual_cols: weight.cols,
        });
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, ElasticDecodePlanError> {
    left.checked_mul(right)
        .ok_or(ElasticDecodePlanError::Overflow)
}

fn checked_product3(
    a: usize,
    b: usize,
    c: usize,
) -> Result<usize, ElasticDecodePlanError> {
    a.checked_mul(b)
        .and_then(|value| value.checked_mul(c))
        .ok_or(ElasticDecodePlanError::Overflow)
}

fn prefix_basis(
    n_kv_heads: usize,
    d_head: usize,
    rank: usize,
) -> Result<Vec<f32>, ElasticDecodePlanError> {
    require_rank("prefix", rank, d_head)?;
    let len = checked_product3(n_kv_heads, d_head, rank)?;
    let mut basis = vec![0.0; len];
    for head in 0..n_kv_heads
    {
        let base = head * d_head * rank;
        for diagonal in 0..rank
        {
            basis[base + diagonal * rank + diagonal] = 1.0;
        }
    }
    Ok(basis)
}

fn all_heads_are_identity(
    basis: &[f32],
    n_kv_heads: usize,
    d_head: usize,
    rank: usize,
) -> bool {
    if rank != d_head
    {
        return false;
    }
    all_heads_are_native_prefix(basis, n_kv_heads, d_head, rank)
}

fn all_heads_are_native_prefix(
    basis: &[f32],
    n_kv_heads: usize,
    d_head: usize,
    rank: usize,
) -> bool {
    if basis.len() != n_kv_heads.saturating_mul(d_head).saturating_mul(rank)
    {
        return false;
    }
    for head in 0..n_kv_heads
    {
        let head_base = head * d_head * rank;
        for row in 0..d_head
        {
            for col in 0..rank
            {
                let expected = if row == col { 1.0f32 } else { 0.0f32 };
                if basis[head_base + row * rank + col].to_bits() != expected.to_bits()
                {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;

    fn matmul_row(input: &[f32], weight: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(input.len(), rows);
        assert_eq!(weight.len(), rows * cols);
        let mut output = vec![0.0; cols];
        for col in 0..cols
        {
            let mut sum = 0.0f32;
            for row in 0..rows
            {
                sum += input[row] * weight[row * cols + col];
            }
            output[col] = sum;
        }
        output
    }

    fn prefix(n_kv_heads: usize, d_head: usize, rank: usize) -> Vec<f32> {
        prefix_basis(n_kv_heads, d_head, rank).unwrap()
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (&a, &b)) in left.iter().zip(right).enumerate()
        {
            assert!(
                (a - b).abs() <= tolerance,
                "index {index}: left={a}, right={b}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn identity_plan_is_bit_exact_at_weight_level() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let plan = ElasticDecodePlan::identity(&model);
        assert_eq!(plan.layers().len(), model.layers.len());
        for (layer, absorbed) in model.layers.iter().zip(plan.layers())
        {
            assert_eq!(absorbed.q_latent(), layer.attn.w_q.weight.data.as_slice());
            assert_eq!(absorbed.k_latent(), layer.attn.w_k.weight.data.as_slice());
            assert_eq!(absorbed.v_latent(), layer.attn.w_v.weight.data.as_slice());
            assert_eq!(absorbed.o_latent(), layer.attn.w_o.weight.data.as_slice());
            assert_eq!(absorbed.rotary_rule(), RotaryLatentRule::FullIdentity);
            assert!(absorbed.is_dense_equivalent_identity());
            assert_eq!(
                absorbed.projection_parameter_count(),
                absorbed.dense_projection_parameter_count()
            );
        }
    }

    #[test]
    fn native_prefix_preserves_complete_rope_pairs_and_original_frequency_denominator() {
        let bases = LatentGqaBases::native_pair_prefix(2, 32, 16, 12).unwrap();
        assert_eq!(
            bases.rotary_rule(),
            RotaryLatentRule::NativePairPrefix {
                rank: 16,
                frequency_denominator: 32,
            }
        );
        assert!(!bases.is_dense_equivalent_identity());
    }

    #[test]
    fn arbitrary_key_basis_requires_projected_rope_operator() {
        let mut key = prefix(1, 8, 4);
        key[0] = 0.5;
        key[4] = 0.5;
        let bases = LatentGqaBases::new(1, 8, 4, 4, key, prefix(1, 8, 4)).unwrap();
        assert_eq!(bases.rotary_rule(), RotaryLatentRule::ProjectedOperator);
    }

    #[test]
    fn absorbed_qkv_matches_dense_projection_then_basis_projection() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let rank = attention.d_head / 2;
        let bases = LatentGqaBases::new(
            attention.n_kv_heads,
            attention.d_head,
            rank,
            rank,
            prefix(attention.n_kv_heads, attention.d_head, rank),
            prefix(attention.n_kv_heads, attention.d_head, rank),
        )
        .unwrap();
        let absorbed = AbsorbedGqaWeights::from_attention(attention, &bases).unwrap();
        let input: Vec<f32> = (0..attention.d_model)
            .map(|index| (index as f32 - 7.0) * 0.03125)
            .collect();

        let q_dense = matmul_row(
            &input,
            &attention.w_q.weight.data,
            attention.d_model,
            attention.d_model,
        );
        let q_latent = matmul_row(
            &input,
            absorbed.q_latent(),
            attention.d_model,
            attention.n_heads * rank,
        );
        for head in 0..attention.n_heads
        {
            let expected = &q_dense[head * attention.d_head..head * attention.d_head + rank];
            let actual = &q_latent[head * rank..(head + 1) * rank];
            assert_close(expected, actual, 1.0e-6);
        }

        let kv_dim = attention.n_kv_heads * attention.d_head;
        let k_dense = matmul_row(
            &input,
            &attention.w_k.weight.data,
            attention.d_model,
            kv_dim,
        );
        let v_dense = matmul_row(
            &input,
            &attention.w_v.weight.data,
            attention.d_model,
            kv_dim,
        );
        let k_latent = matmul_row(
            &input,
            absorbed.k_latent(),
            attention.d_model,
            attention.n_kv_heads * rank,
        );
        let v_latent = matmul_row(
            &input,
            absorbed.v_latent(),
            attention.d_model,
            attention.n_kv_heads * rank,
        );
        for head in 0..attention.n_kv_heads
        {
            assert_close(
                &k_dense[head * attention.d_head..head * attention.d_head + rank],
                &k_latent[head * rank..(head + 1) * rank],
                1.0e-6,
            );
            assert_close(
                &v_dense[head * attention.d_head..head * attention.d_head + rank],
                &v_latent[head * rank..(head + 1) * rank],
                1.0e-6,
            );
        }
    }

    #[test]
    fn absorbed_output_matches_reconstruct_then_dense_output_projection() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let rank = attention.d_head / 2;
        let raw_basis = prefix(attention.n_kv_heads, attention.d_head, rank);
        let bases = LatentGqaBases::new(
            attention.n_kv_heads,
            attention.d_head,
            rank,
            rank,
            raw_basis.clone(),
            raw_basis,
        )
        .unwrap();
        let absorbed = AbsorbedGqaWeights::from_attention(attention, &bases).unwrap();
        let repeat = attention.n_heads / attention.n_kv_heads;
        let latent: Vec<f32> = (0..attention.n_heads * rank)
            .map(|index| (index as f32 + 1.0) * 0.0125)
            .collect();
        let mut dense_context = vec![0.0f32; attention.d_model];
        for q_head in 0..attention.n_heads
        {
            let kv_head = q_head / repeat;
            let basis = bases.value_head(kv_head);
            for channel in 0..attention.d_head
            {
                let mut sum = 0.0f32;
                for latent_col in 0..rank
                {
                    sum += latent[q_head * rank + latent_col]
                        * basis[channel * rank + latent_col];
                }
                dense_context[q_head * attention.d_head + channel] = sum;
            }
        }
        let expected = matmul_row(
            &dense_context,
            &attention.w_o.weight.data,
            attention.d_model,
            attention.d_model,
        );
        let actual = matmul_row(
            &latent,
            absorbed.o_latent(),
            attention.n_heads * rank,
            attention.d_model,
        );
        assert_close(&expected, &actual, 1.0e-5);
    }

    #[test]
    fn half_rank_halves_attention_projection_weight_traffic() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let mut bases = Vec::new();
        for layer in &model.layers
        {
            let rank = layer.attn.d_head / 2;
            bases.push(
                LatentGqaBases::native_pair_prefix(
                    layer.attn.n_kv_heads,
                    layer.attn.d_head,
                    rank,
                    rank,
                )
                .unwrap(),
            );
        }
        let plan = ElasticDecodePlan::from_model(&model, &bases).unwrap();
        assert_eq!(
            plan.projection_parameter_count() * 2,
            plan.dense_projection_parameter_count()
        );
    }
}
