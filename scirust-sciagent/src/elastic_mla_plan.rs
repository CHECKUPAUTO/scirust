//! ElasticMLA conversion planning for SCIAGENT GQA.
//!
//! This is SciRust's decoupled-RoPE latent route. Selected complete native RoPE
//! pairs remain explicit and preserve their original frequency indices. The
//! complementary key/query channels become NoPE coordinates and can use an arbitrary
//! ElasticKV low-rank basis because no position rotation acts inside that subspace.
//! Values remain fully eligible for reconstruction-free latent storage and output
//! projection absorption.
//!
//! The plan is intentionally backend-neutral and deterministic. Converting a trained
//! full-RoPE model to partial-RoPE changes the model function unless no dimensions are
//! removed from RoPE; reduced configurations therefore require a model-quality gate
//! or adaptation step before production promotion.

use core::fmt;

use crate::attention::GQAAttention;
use crate::model::SciAgentModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElasticMlaPlanError {
    InvalidTopology(&'static str),
    InvalidRopeDimension {
        rope_dimensions: usize,
        d_head: usize,
    },
    RopePairOutOfRange {
        pair: usize,
        pair_count: usize,
    },
    RopePairsNotStrictlyIncreasing,
    InvalidRank {
        name: &'static str,
        rank: usize,
        dimension: usize,
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
    LayerBasisCount {
        layers: usize,
        bases: usize,
    },
    WeightShape {
        name: &'static str,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    Overflow,
}

impl fmt::Display for ElasticMlaPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidTopology(message) => write!(formatter, "{message}"),
            Self::InvalidRopeDimension {
                rope_dimensions,
                d_head,
            } => write!(
                formatter,
                "RoPE dimensions {rope_dimensions} must be even and <= d_head {d_head}"
            ),
            Self::RopePairOutOfRange { pair, pair_count } => write!(
                formatter,
                "RoPE pair {pair} is outside 0..{pair_count}"
            ),
            Self::RopePairsNotStrictlyIncreasing => write!(
                formatter,
                "RoPE pair indices must be strictly increasing and duplicate-free"
            ),
            Self::InvalidRank {
                name,
                rank,
                dimension,
            } => write!(
                formatter,
                "{name} rank {rank} is outside 1..={dimension}"
            ),
            Self::BasisLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} basis length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteBasis { name, index } => write!(
                formatter,
                "{name} basis contains non-finite value at {index}"
            ),
            Self::LayerBasisCount { layers, bases } => write!(
                formatter,
                "ElasticMLA basis count mismatch: model has {layers} layers, got {bases}"
            ),
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
            Self::Overflow => write!(formatter, "ElasticMLA plan size overflow"),
        }
    }
}

impl std::error::Error for ElasticMlaPlanError {}

/// Exact native RoPE pair selection for one head layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopePairSelection {
    d_head: usize,
    pairs: Vec<usize>,
    rope_channels: Vec<usize>,
    nope_channels: Vec<usize>,
}

impl RopePairSelection {
    /// Select explicit native pair indices. Input order is part of the contract and
    /// must already be canonical (strictly increasing).
    pub fn new(d_head: usize, pairs: Vec<usize>) -> Result<Self, ElasticMlaPlanError> {
        if d_head == 0 || !d_head.is_multiple_of(2)
        {
            return Err(ElasticMlaPlanError::InvalidTopology(
                "ElasticMLA requires a positive even d_head",
            ));
        }
        let pair_count = d_head / 2;
        let mut previous = None;
        for &pair in &pairs
        {
            if pair >= pair_count
            {
                return Err(ElasticMlaPlanError::RopePairOutOfRange { pair, pair_count });
            }
            if previous.is_some_and(|value| pair <= value)
            {
                return Err(ElasticMlaPlanError::RopePairsNotStrictlyIncreasing);
            }
            previous = Some(pair);
        }

        let mut selected = vec![false; pair_count];
        for &pair in &pairs
        {
            selected[pair] = true;
        }
        let mut rope_channels = Vec::with_capacity(pairs.len() * 2);
        let mut nope_channels = Vec::with_capacity(d_head - pairs.len() * 2);
        for (pair, is_selected) in selected.into_iter().enumerate()
        {
            let even = pair * 2;
            if is_selected
            {
                rope_channels.push(even);
                rope_channels.push(even + 1);
            }
            else
            {
                nope_channels.push(even);
                nope_channels.push(even + 1);
            }
        }
        Ok(Self {
            d_head,
            pairs,
            rope_channels,
            nope_channels,
        })
    }

    /// Retain the first native RoPE pairs (the highest-frequency end of SCIAGENT's
    /// current interleaved RoPE ordering).
    pub fn high_frequency_prefix(
        d_head: usize,
        rope_dimensions: usize,
    ) -> Result<Self, ElasticMlaPlanError> {
        require_rope_dimensions(d_head, rope_dimensions)?;
        Self::new(d_head, (0..rope_dimensions / 2).collect())
    }

    /// Retain the last native RoPE pairs.
    pub fn low_frequency_suffix(
        d_head: usize,
        rope_dimensions: usize,
    ) -> Result<Self, ElasticMlaPlanError> {
        require_rope_dimensions(d_head, rope_dimensions)?;
        let pair_count = d_head / 2;
        let keep = rope_dimensions / 2;
        Self::new(d_head, (pair_count - keep..pair_count).collect())
    }

    #[must_use]
    pub const fn d_head(&self) -> usize {
        self.d_head
    }

    #[must_use]
    pub fn pair_indices(&self) -> &[usize] {
        &self.pairs
    }

    #[must_use]
    pub fn rope_channels(&self) -> &[usize] {
        &self.rope_channels
    }

    #[must_use]
    pub fn nope_channels(&self) -> &[usize] {
        &self.nope_channels
    }

    #[must_use]
    pub fn rope_dimensions(&self) -> usize {
        self.rope_channels.len()
    }

    #[must_use]
    pub fn nope_dimensions(&self) -> usize {
        self.nope_channels.len()
    }
}

/// Per-layer bases for NoPE K/Q content and full value content.
#[derive(Debug, Clone)]
pub struct ElasticMlaBases {
    n_kv_heads: usize,
    selection: RopePairSelection,
    key_rank: usize,
    value_rank: usize,
    key_nope: Vec<f32>,
    value: Vec<f32>,
}

impl ElasticMlaBases {
    /// `key_nope` is KV-head-major `[nope_dim, key_rank]`; `value` is
    /// KV-head-major `[d_head, value_rank]`.
    pub fn new(
        n_kv_heads: usize,
        selection: RopePairSelection,
        key_rank: usize,
        value_rank: usize,
        key_nope: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<Self, ElasticMlaPlanError> {
        if n_kv_heads == 0
        {
            return Err(ElasticMlaPlanError::InvalidTopology(
                "ElasticMLA requires at least one KV head",
            ));
        }
        let nope_dim = selection.nope_dimensions();
        if nope_dim == 0
        {
            return Err(ElasticMlaPlanError::InvalidTopology(
                "ElasticMLA low-rank content requires at least one NoPE pair",
            ));
        }
        require_rank("key NoPE", key_rank, nope_dim)?;
        require_rank("value", value_rank, selection.d_head)?;
        let expected_key = checked_product3(n_kv_heads, nope_dim, key_rank)?;
        let expected_value = checked_product3(n_kv_heads, selection.d_head, value_rank)?;
        require_basis("key NoPE", &key_nope, expected_key)?;
        require_basis("value", &value, expected_value)?;
        Ok(Self {
            n_kv_heads,
            selection,
            key_rank,
            value_rank,
            key_nope,
            value,
        })
    }

    /// Deterministic coordinate-selection baseline. It preserves the first
    /// `key_rank` NoPE channels and first `value_rank` value channels. This is a
    /// structural test basis, not a quality-optimal basis.
    pub fn coordinate_prefix(
        n_kv_heads: usize,
        selection: RopePairSelection,
        key_rank: usize,
        value_rank: usize,
    ) -> Result<Self, ElasticMlaPlanError> {
        let key_nope = prefix_basis(n_kv_heads, selection.nope_dimensions(), key_rank)?;
        let value = prefix_basis(n_kv_heads, selection.d_head(), value_rank)?;
        Self::new(
            n_kv_heads,
            selection,
            key_rank,
            value_rank,
            key_nope,
            value,
        )
    }

    #[must_use]
    pub const fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
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
    pub const fn selection(&self) -> &RopePairSelection {
        &self.selection
    }

    #[must_use]
    pub fn key_head(&self, head: usize) -> &[f32] {
        assert!(head < self.n_kv_heads);
        let width = self.selection.nope_dimensions() * self.key_rank;
        &self.key_nope[head * width..(head + 1) * width]
    }

    #[must_use]
    pub fn value_head(&self, head: usize) -> &[f32] {
        assert!(head < self.n_kv_heads);
        let width = self.selection.d_head() * self.value_rank;
        &self.value[head * width..(head + 1) * width]
    }
}

/// One GQA layer converted to explicit RoPE lane + ElasticKV NoPE/value lanes.
#[derive(Debug, Clone)]
pub struct ElasticMlaLayerWeights {
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    rope_pairs: Vec<usize>,
    rope_dimensions: usize,
    nope_dimensions: usize,
    key_rank: usize,
    value_rank: usize,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    q_nope_latent: Vec<f32>,
    k_nope_latent: Vec<f32>,
    v_latent: Vec<f32>,
    o_latent: Vec<f32>,
}

impl ElasticMlaLayerWeights {
    pub fn from_attention(
        attention: &GQAAttention,
        bases: &ElasticMlaBases,
    ) -> Result<Self, ElasticMlaPlanError> {
        validate_attention(attention, bases)?;
        let d_model = attention.d_model;
        let n_heads = attention.n_heads;
        let n_kv_heads = attention.n_kv_heads;
        let d_head = attention.d_head;
        let rope_dimensions = bases.selection.rope_dimensions();
        let nope_dimensions = bases.selection.nope_dimensions();
        let key_rank = bases.key_rank;
        let value_rank = bases.value_rank;
        let kv_dim = n_kv_heads
            .checked_mul(d_head)
            .ok_or(ElasticMlaPlanError::Overflow)?;
        require_weight("Wq", &attention.w_q.weight, d_model, d_model)?;
        require_weight("Wk", &attention.w_k.weight, d_model, kv_dim)?;
        require_weight("Wv", &attention.w_v.weight, d_model, kv_dim)?;
        require_weight("Wo", &attention.w_o.weight, d_model, d_model)?;

        let q_rope_cols = checked_product(n_heads, rope_dimensions)?;
        let k_rope_cols = checked_product(n_kv_heads, rope_dimensions)?;
        let q_nope_cols = checked_product(n_heads, key_rank)?;
        let k_nope_cols = checked_product(n_kv_heads, key_rank)?;
        let v_cols = checked_product(n_kv_heads, value_rank)?;
        let o_rows = checked_product(n_heads, value_rank)?;
        let mut q_rope = vec![0.0; checked_product(d_model, q_rope_cols)?];
        let mut k_rope = vec![0.0; checked_product(d_model, k_rope_cols)?];
        let mut q_nope_latent = vec![0.0; checked_product(d_model, q_nope_cols)?];
        let mut k_nope_latent = vec![0.0; checked_product(d_model, k_nope_cols)?];
        let mut v_latent = vec![0.0; checked_product(d_model, v_cols)?];
        let mut o_latent = vec![0.0; checked_product(o_rows, d_model)?];
        let repeat = n_heads / n_kv_heads;
        let rope_channels = bases.selection.rope_channels();
        let nope_channels = bases.selection.nope_channels();

        for input in 0..d_model
        {
            for q_head in 0..n_heads
            {
                let kv_head = q_head / repeat;
                for (local, &channel) in rope_channels.iter().enumerate()
                {
                    q_rope[input * q_rope_cols + q_head * rope_dimensions + local] =
                        attention.w_q.weight.data[input * d_model + q_head * d_head + channel];
                }
                let key_basis = bases.key_head(kv_head);
                for latent in 0..key_rank
                {
                    let mut sum = 0.0f32;
                    for (local, &channel) in nope_channels.iter().enumerate()
                    {
                        sum += attention.w_q.weight.data
                            [input * d_model + q_head * d_head + channel]
                            * key_basis[local * key_rank + latent];
                    }
                    q_nope_latent[input * q_nope_cols + q_head * key_rank + latent] = sum;
                }
            }

            for kv_head in 0..n_kv_heads
            {
                for (local, &channel) in rope_channels.iter().enumerate()
                {
                    k_rope[input * k_rope_cols + kv_head * rope_dimensions + local] =
                        attention.w_k.weight.data[input * kv_dim + kv_head * d_head + channel];
                }
                let key_basis = bases.key_head(kv_head);
                for latent in 0..key_rank
                {
                    let mut sum = 0.0f32;
                    for (local, &channel) in nope_channels.iter().enumerate()
                    {
                        sum += attention.w_k.weight.data
                            [input * kv_dim + kv_head * d_head + channel]
                            * key_basis[local * key_rank + latent];
                    }
                    k_nope_latent[input * k_nope_cols + kv_head * key_rank + latent] = sum;
                }

                let value_basis = bases.value_head(kv_head);
                for latent in 0..value_rank
                {
                    let mut sum = 0.0f32;
                    for channel in 0..d_head
                    {
                        sum += attention.w_v.weight.data
                            [input * kv_dim + kv_head * d_head + channel]
                            * value_basis[channel * value_rank + latent];
                    }
                    v_latent[input * v_cols + kv_head * value_rank + latent] = sum;
                }
            }
        }

        for q_head in 0..n_heads
        {
            let kv_head = q_head / repeat;
            let value_basis = bases.value_head(kv_head);
            for latent in 0..value_rank
            {
                let destination_row = q_head * value_rank + latent;
                for output in 0..d_model
                {
                    let mut sum = 0.0f32;
                    for channel in 0..d_head
                    {
                        sum += value_basis[channel * value_rank + latent]
                            * attention.w_o.weight.data
                                [(q_head * d_head + channel) * d_model + output];
                    }
                    o_latent[destination_row * d_model + output] = sum;
                }
            }
        }

        Ok(Self {
            d_model,
            n_heads,
            n_kv_heads,
            d_head,
            rope_pairs: bases.selection.pair_indices().to_vec(),
            rope_dimensions,
            nope_dimensions,
            key_rank,
            value_rank,
            q_rope,
            k_rope,
            q_nope_latent,
            k_nope_latent,
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
    pub fn rope_pairs(&self) -> &[usize] {
        &self.rope_pairs
    }

    #[must_use]
    pub const fn rope_dimensions(&self) -> usize {
        self.rope_dimensions
    }

    #[must_use]
    pub const fn nope_dimensions(&self) -> usize {
        self.nope_dimensions
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
    pub fn q_rope(&self) -> &[f32] {
        &self.q_rope
    }

    #[must_use]
    pub fn k_rope(&self) -> &[f32] {
        &self.k_rope
    }

    #[must_use]
    pub fn q_nope_latent(&self) -> &[f32] {
        &self.q_nope_latent
    }

    #[must_use]
    pub fn k_nope_latent(&self) -> &[f32] {
        &self.k_nope_latent
    }

    #[must_use]
    pub fn v_latent(&self) -> &[f32] {
        &self.v_latent
    }

    #[must_use]
    pub fn o_latent(&self) -> &[f32] {
        &self.o_latent
    }

    /// Persistent K/V scalar coordinates per KV head and token before coefficient
    /// quantization. Dense GQA stores `2 * d_head` scalars.
    #[must_use]
    pub fn cache_scalars_per_kv_head(&self) -> usize {
        self.rope_dimensions
            .saturating_add(self.key_rank)
            .saturating_add(self.value_rank)
    }

    #[must_use]
    pub const fn dense_cache_scalars_per_kv_head(&self) -> usize {
        self.d_head * 2
    }

    /// Q/K/V/O projection scalars after conversion. This excludes the small basis
    /// metadata because basis storage is amortized across all decoded tokens.
    #[must_use]
    pub fn projection_parameter_count(&self) -> usize {
        self.q_rope
            .len()
            .saturating_add(self.k_rope.len())
            .saturating_add(self.q_nope_latent.len())
            .saturating_add(self.k_nope_latent.len())
            .saturating_add(self.v_latent.len())
            .saturating_add(self.o_latent.len())
    }
}

#[derive(Debug, Clone)]
pub struct ElasticMlaPlan {
    layers: Vec<ElasticMlaLayerWeights>,
}

impl ElasticMlaPlan {
    pub fn from_model(
        model: &SciAgentModel,
        bases: &[ElasticMlaBases],
    ) -> Result<Self, ElasticMlaPlanError> {
        if model.layers.len() != bases.len()
        {
            return Err(ElasticMlaPlanError::LayerBasisCount {
                layers: model.layers.len(),
                bases: bases.len(),
            });
        }
        let mut layers = Vec::with_capacity(model.layers.len());
        for (layer, basis) in model.layers.iter().zip(bases)
        {
            layers.push(ElasticMlaLayerWeights::from_attention(&layer.attn, basis)?);
        }
        Ok(Self { layers })
    }

    #[must_use]
    pub fn layers(&self) -> &[ElasticMlaLayerWeights] {
        &self.layers
    }
}

fn require_rope_dimensions(
    d_head: usize,
    rope_dimensions: usize,
) -> Result<(), ElasticMlaPlanError> {
    if d_head == 0
        || !d_head.is_multiple_of(2)
        || !rope_dimensions.is_multiple_of(2)
        || rope_dimensions >= d_head
    {
        return Err(ElasticMlaPlanError::InvalidRopeDimension {
            rope_dimensions,
            d_head,
        });
    }
    Ok(())
}

fn validate_attention(
    attention: &GQAAttention,
    bases: &ElasticMlaBases,
) -> Result<(), ElasticMlaPlanError> {
    if attention.n_heads == 0 || attention.n_kv_heads == 0 || attention.d_head == 0
    {
        return Err(ElasticMlaPlanError::InvalidTopology(
            "ElasticMLA attention topology must be non-zero",
        ));
    }
    if !attention.n_heads.is_multiple_of(attention.n_kv_heads)
    {
        return Err(ElasticMlaPlanError::InvalidTopology(
            "ElasticMLA GQA query heads must be divisible by KV heads",
        ));
    }
    if attention.d_model != attention.n_heads * attention.d_head
    {
        return Err(ElasticMlaPlanError::InvalidTopology(
            "ElasticMLA d_model must equal n_heads * d_head",
        ));
    }
    if bases.n_kv_heads != attention.n_kv_heads
        || bases.selection.d_head() != attention.d_head
    {
        return Err(ElasticMlaPlanError::InvalidTopology(
            "ElasticMLA basis topology does not match attention",
        ));
    }
    Ok(())
}

fn require_rank(
    name: &'static str,
    rank: usize,
    dimension: usize,
) -> Result<(), ElasticMlaPlanError> {
    if rank == 0 || rank > dimension
    {
        return Err(ElasticMlaPlanError::InvalidRank {
            name,
            rank,
            dimension,
        });
    }
    Ok(())
}

fn require_basis(
    name: &'static str,
    basis: &[f32],
    expected: usize,
) -> Result<(), ElasticMlaPlanError> {
    if basis.len() != expected
    {
        return Err(ElasticMlaPlanError::BasisLength {
            name,
            expected,
            actual: basis.len(),
        });
    }
    if let Some(index) = basis.iter().position(|value| !value.is_finite())
    {
        return Err(ElasticMlaPlanError::NonFiniteBasis { name, index });
    }
    Ok(())
}

fn require_weight(
    name: &'static str,
    weight: &scirust_core::autodiff::reverse::Tensor,
    rows: usize,
    cols: usize,
) -> Result<(), ElasticMlaPlanError> {
    if weight.rows != rows || weight.cols != cols
    {
        return Err(ElasticMlaPlanError::WeightShape {
            name,
            expected_rows: rows,
            expected_cols: cols,
            actual_rows: weight.rows,
            actual_cols: weight.cols,
        });
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, ElasticMlaPlanError> {
    left.checked_mul(right).ok_or(ElasticMlaPlanError::Overflow)
}

fn checked_product3(
    a: usize,
    b: usize,
    c: usize,
) -> Result<usize, ElasticMlaPlanError> {
    a.checked_mul(b)
        .and_then(|value| value.checked_mul(c))
        .ok_or(ElasticMlaPlanError::Overflow)
}

fn prefix_basis(
    heads: usize,
    dimension: usize,
    rank: usize,
) -> Result<Vec<f32>, ElasticMlaPlanError> {
    require_rank("coordinate prefix", rank, dimension)?;
    let mut basis = vec![0.0; checked_product3(heads, dimension, rank)?];
    for head in 0..heads
    {
        let base = head * dimension * rank;
        for diagonal in 0..rank
        {
            basis[base + diagonal * rank + diagonal] = 1.0;
        }
    }
    Ok(basis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;

    fn matmul_row(input: &[f32], weight: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(input.len(), rows);
        assert_eq!(weight.len(), rows * cols);
        let mut output = vec![0.0f32; cols];
        for col in 0..cols
        {
            for row in 0..rows
            {
                output[col] += input[row] * weight[row * cols + col];
            }
        }
        output
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (&left, &right)) in left.iter().zip(right).enumerate()
        {
            assert!(
                (left - right).abs() <= tolerance,
                "index {index}: left={left}, right={right}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn explicit_pair_selection_preserves_native_pair_indices() {
        let selection = RopePairSelection::new(16, vec![0, 3, 7]).unwrap();
        assert_eq!(selection.rope_channels(), &[0, 1, 6, 7, 14, 15]);
        assert_eq!(
            selection.nope_channels(),
            &[2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
        );
    }

    #[test]
    fn high_and_low_frequency_selectors_are_deterministic() {
        assert_eq!(
            RopePairSelection::high_frequency_prefix(16, 4)
                .unwrap()
                .pair_indices(),
            &[0, 1]
        );
        assert_eq!(
            RopePairSelection::low_frequency_suffix(16, 4)
                .unwrap()
                .pair_indices(),
            &[6, 7]
        );
    }

    #[test]
    fn converted_coordinates_match_dense_slices_before_positional_change() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let selection = RopePairSelection::high_frequency_prefix(attention.d_head, 4).unwrap();
        let nope_dim = selection.nope_dimensions();
        let bases = ElasticMlaBases::coordinate_prefix(
            attention.n_kv_heads,
            selection.clone(),
            nope_dim,
            attention.d_head,
        )
        .unwrap();
        let converted = ElasticMlaLayerWeights::from_attention(attention, &bases).unwrap();
        let input: Vec<f32> = (0..attention.d_model)
            .map(|index| (index as f32 - 5.0) * 0.015625)
            .collect();
        let q_dense = matmul_row(
            &input,
            &attention.w_q.weight.data,
            attention.d_model,
            attention.d_model,
        );
        let q_rope = matmul_row(
            &input,
            converted.q_rope(),
            attention.d_model,
            attention.n_heads * selection.rope_dimensions(),
        );
        let q_nope = matmul_row(
            &input,
            converted.q_nope_latent(),
            attention.d_model,
            attention.n_heads * nope_dim,
        );
        for head in 0..attention.n_heads
        {
            for (local, &channel) in selection.rope_channels().iter().enumerate()
            {
                assert_eq!(
                    q_rope[head * selection.rope_dimensions() + local].to_bits(),
                    q_dense[head * attention.d_head + channel].to_bits()
                );
            }
            for (local, &channel) in selection.nope_channels().iter().enumerate()
            {
                assert_close(
                    &[q_nope[head * nope_dim + local]],
                    &[q_dense[head * attention.d_head + channel]],
                    1.0e-6,
                );
            }
        }
    }

    #[test]
    fn full_value_basis_absorption_preserves_output_projection() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let selection = RopePairSelection::high_frequency_prefix(attention.d_head, 4).unwrap();
        let bases = ElasticMlaBases::coordinate_prefix(
            attention.n_kv_heads,
            selection,
            4,
            attention.d_head,
        )
        .unwrap();
        let converted = ElasticMlaLayerWeights::from_attention(attention, &bases).unwrap();
        let context: Vec<f32> = (0..attention.d_model)
            .map(|index| (index as f32 + 1.0) * 0.0078125)
            .collect();
        let expected = matmul_row(
            &context,
            &attention.w_o.weight.data,
            attention.d_model,
            attention.d_model,
        );
        let actual = matmul_row(
            &context,
            converted.o_latent(),
            attention.d_model,
            attention.d_model,
        );
        assert_eq!(converted.value_rank(), attention.d_head);
        assert_close(&expected, &actual, 1.0e-6);
    }

    #[test]
    fn reduced_elastic_mla_cache_is_smaller_than_dense_gqa() {
        let model = SciAgentModel::new(&SciAgentConfig::small());
        let attention = &model.layers[0].attn;
        let selection = RopePairSelection::high_frequency_prefix(attention.d_head, 4).unwrap();
        let bases = ElasticMlaBases::coordinate_prefix(
            attention.n_kv_heads,
            selection,
            4,
            4,
        )
        .unwrap();
        let converted = ElasticMlaLayerWeights::from_attention(attention, &bases).unwrap();
        assert!(
            converted.cache_scalars_per_kv_head() < converted.dense_cache_scalars_per_kv_head()
        );
    }
}
