//! Pluggable attention KV-cache backend + a memory-bounded numeric decode path.
//!
//! SciRust already ships a complete KV-compression subsystem that is **never
//! wired** into `MultiHeadAttention`'s live decode: [`PagedKvCache`] (block
//! paging), [`ElasticKvCache`] (two-level INT4 + bounded-budget eviction) and
//! [`kvquant_kv`](crate::quantization::kvquant_kv) all exist and are tested in
//! isolation, but `MultiHeadAttention::infer_step` keeps a plain
//! `RefCell<Option<(Tensor, Tensor)>>` full-precision cache that grows without
//! bound. This module closes that gap with two additive pieces:
//!
//! - the [`AttentionBackend`] trait and three adapters (plain contiguous,
//!   paged, elastic) so any of the existing caches serves attention uniformly;
//! - a free numeric [`decode_step`] over a `&MultiHeadAttention` that runs the
//!   q/k/v/o projections numerically (no autodiff graph) and dispatches the
//!   per-head attention to a slice of [`AttentionBackend`]s — one per head — so
//!   a decode run is memory-bounded by whatever backends the caller picks.
//!
//! The existing `infer_step` is untouched (backward compatible); callers that
//! want a bounded/compressed decode use [`decode_step`]. All arithmetic is
//! plain `f32` in fixed order, so a run is bit-for-bit deterministic.

use crate::autodiff::reverse::Tensor;
use crate::nn::elastic_kv_cache::ElasticKvCache;
use crate::nn::paged_attention::PagedKvCache;
use crate::nn::transformer::attention::MultiHeadAttention;

/// One attention KV-cache backend variant. Implementations own the K/V store
/// for a single head (vector width `dim()`) and serve single-query attention
/// `softmax(q·Kᵀ/√d)·V` over the cached positions.
pub trait AttentionBackend {
    /// Width of each cached K/V vector (= `d_head`).
    fn dim(&self) -> usize;

    /// Append one position's key and value (each length `dim()`).
    fn append(&mut self, k: &[f32], v: &[f32]);

    /// Number of cached positions.
    fn len(&self) -> usize;

    /// Whether the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `softmax(q·Kᵀ/√d)·V` over the cached positions (single query, length
    /// `dim()`). Returns the context vector (length `dim()`).
    fn attention(&self, q: &[f32]) -> Vec<f32>;

    /// Resident memory footprint in bytes (for budget accounting / telemetry).
    fn packed_bytes(&self) -> usize {
        0
    }
}

/// Plain contiguous `f32` cache — the full-precision reference backend. Grows
/// unbounded (the baseline `MultiHeadAttention::infer_step` behaviour); useful
/// as the ground truth to validate the compressed backends against.
pub struct PlainKvCache {
    dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl PlainKvCache {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}

impl AttentionBackend for PlainKvCache {
    fn dim(&self) -> usize {
        self.dim
    }
    fn append(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.dim);
        assert_eq!(v.len(), self.dim);
        self.keys.extend_from_slice(k);
        self.values.extend_from_slice(v);
    }
    fn len(&self) -> usize {
        self.keys.len() / self.dim
    }
    fn attention(&self, q: &[f32]) -> Vec<f32> {
        let n = self.len();
        crate::nn::paged_attention::contiguous_attention(&self.keys, &self.values, q, self.dim, n)
    }
    fn packed_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }
}

/// Paged backend — adapter over [`PagedKvCache`] (block-paged, low fragmentation).
pub struct PagedBackend {
    d: usize,
    inner: PagedKvCache,
}

impl PagedBackend {
    pub fn new(d_head: usize, block_size: usize) -> Self {
        Self {
            d: d_head,
            inner: PagedKvCache::new(d_head, block_size),
        }
    }
}

impl AttentionBackend for PagedBackend {
    fn dim(&self) -> usize {
        self.d
    }
    fn append(&mut self, k: &[f32], v: &[f32]) {
        self.inner.append(k, v);
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn attention(&self, q: &[f32]) -> Vec<f32> {
        self.inner.attention(q)
    }
    fn packed_bytes(&self) -> usize {
        self.inner.num_blocks() * self.d * std::mem::size_of::<f32>() * 2
    }
}

/// Elastic backend — adapter over [`ElasticKvCache`] (two-level INT4 +
/// bounded-budget eviction). Compressed footprint via
/// [`ElasticKvCache::compressed_bytes`].
pub struct ElasticBackend {
    d: usize,
    inner: ElasticKvCache,
}

impl ElasticBackend {
    pub fn new(d_head: usize, budget: usize) -> Self {
        Self {
            d: d_head,
            inner: ElasticKvCache::new(d_head, budget),
        }
    }

    pub fn new_grouped(d_head: usize, budget: usize, group_size: usize) -> Self {
        Self {
            d: d_head,
            inner: ElasticKvCache::new_grouped(d_head, budget, group_size),
        }
    }

    /// Number of tiles evicted by the budget so far.
    pub fn evicted(&self) -> usize {
        self.inner.evicted()
    }
}

impl AttentionBackend for ElasticBackend {
    fn dim(&self) -> usize {
        self.d
    }
    fn append(&mut self, k: &[f32], v: &[f32]) {
        self.inner.append(k, v);
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn attention(&self, q: &[f32]) -> Vec<f32> {
        self.inner.attention(q)
    }
    fn packed_bytes(&self) -> usize {
        self.inner.compressed_bytes()
    }
}

/// Apply a `Linear` weight `(in, out)` + bias `(1, out)` to a 1-row input `x`
/// (length `in`) producing a length-`out` vector: `y = x·W + b`. Matches the
/// autodiff `Linear::forward` orientation (`try_matmul(w)`).
#[allow(clippy::needless_range_loop)]
fn linear_apply(weight: &Tensor, bias: &Tensor, x: &[f32]) -> Vec<f32> {
    let in_f = weight.rows;
    let out_f = weight.cols;
    debug_assert_eq!(x.len(), in_f);
    debug_assert_eq!(bias.data.len(), out_f);
    let mut y = vec![0.0f32; out_f];
    for o in 0..out_f
    {
        let mut acc = bias.data[o];
        for i in 0..in_f
        {
            acc += x[i] * weight.data[i * out_f + o];
        }
        y[o] = acc;
    }
    y
}

/// Memory-bounded numeric decode step over a `MultiHeadAttention`.
///
/// Runs the q/k/v/o projections numerically (no autodiff tape) and dispatches
/// per-head attention to `head_backends` — one backend per head, each of width
/// `d_head`. The caller picks the backend (plain / paged / elastic) per head, so
/// the run is bounded by whatever caches it hands in. The projection weights are
/// read from the attention's `pub` `Linear` fields (`w_q`/`w_k`/`w_v`/`w_o`).
///
/// Returns the output vector (length `d_model`).
pub fn decode_step(
    attn: &MultiHeadAttention,
    token: &[f32],
    head_backends: &mut [Box<dyn AttentionBackend>],
) -> Vec<f32> {
    assert_eq!(token.len(), attn.d_model, "token must be length d_model");
    assert_eq!(
        head_backends.len(),
        attn.n_heads,
        "need one backend per head"
    );
    let q = linear_apply(&attn.w_q.weight, &attn.w_q.bias, token);
    let k = linear_apply(&attn.w_k.weight, &attn.w_k.bias, token);
    let v = linear_apply(&attn.w_v.weight, &attn.w_v.bias, token);
    let d_h = attn.d_head;
    let mut ctx = vec![0.0f32; attn.d_model];
    for h in 0..attn.n_heads
    {
        let qh = &q[h * d_h..(h + 1) * d_h];
        let kh = &k[h * d_h..(h + 1) * d_h];
        let vh = &v[h * d_h..(h + 1) * d_h];
        head_backends[h].append(kh, vh);
        let out_h = head_backends[h].attention(qh);
        ctx[h * d_h..(h + 1) * d_h].copy_from_slice(&out_h);
    }
    linear_apply(&attn.w_o.weight, &attn.w_o.bias, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::rng::PcgEngine;

    fn build_attn(d_model: usize, n_heads: usize) -> MultiHeadAttention {
        let mut rng = PcgEngine::new(0);
        MultiHeadAttention::new(d_model, n_heads, 0, false, &KaimingNormal, &Zeros, &mut rng)
    }

    #[test]
    fn paged_backend_matches_plain_bit_for_bit() {
        // The paged backend must reproduce the plain backend's output exactly
        // (PagedAttention's zero-cost-numerical guarantee, re-checked here at
        // the backend-trait level).
        let d = 4usize;
        let n = 7usize;
        let mut plain = PlainKvCache::new(d);
        let mut paged = PagedBackend::new(d, 3);
        let mut rng = PcgEngine::new(42);
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n
        {
            let k: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();
            let v: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();
            keys.extend_from_slice(&k);
            values.extend_from_slice(&v);
            plain.append(&k, &v);
            paged.append(&k, &v);
        }
        let q: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();
        let a = plain.attention(&q);
        let b = paged.attention(&q);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "paged backend diverged from plain"
        );
    }

    #[test]
    fn decode_step_paged_equals_decode_step_plain() {
        // Wiring the paged backend through decode_step must equal the plain
        // backend (the decode path is memory-bounded by the paged cache yet
        // bit-identical to the unbounded reference).
        let (d_model, n_heads) = (8usize, 2usize);
        let attn = build_attn(d_model, n_heads);
        let d_h = attn.d_head; // 4
        let token: Vec<f32> = (0..d_model).map(|i| (i as f32) * 0.05).collect();
        let mut plain: Vec<Box<dyn AttentionBackend>> = (0..n_heads)
            .map(|_| Box::new(PlainKvCache::new(d_h)) as Box<dyn AttentionBackend>)
            .collect();
        let mut paged: Vec<Box<dyn AttentionBackend>> = (0..n_heads)
            .map(|_| Box::new(PagedBackend::new(d_h, 2)) as Box<dyn AttentionBackend>)
            .collect();
        // Run three steps each.
        for _ in 0..3
        {
            let o_plain = decode_step(&attn, &token, &mut plain);
            let o_paged = decode_step(&attn, &token, &mut paged);
            assert_eq!(
                o_plain.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                o_paged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "paged decode diverged from plain decode"
            );
        }
    }

    #[test]
    fn decode_step_elastic_approximates_plain() {
        // The elastic backend compresses (INT4) so it must be *close* to plain
        // (within a tolerance), not bit-identical — and it must bound memory.
        let (d_model, n_heads) = (8usize, 2usize);
        let attn = build_attn(d_model, n_heads);
        let d_h = attn.d_head;
        let token: Vec<f32> = (0..d_model).map(|i| (i as f32) * 0.05).collect();
        let mut plain: Vec<Box<dyn AttentionBackend>> = (0..n_heads)
            .map(|_| Box::new(PlainKvCache::new(d_h)) as Box<dyn AttentionBackend>)
            .collect();
        let mut elastic: Vec<Box<dyn AttentionBackend>> = (0..n_heads)
            .map(|_| Box::new(ElasticBackend::new(d_h, 16)) as Box<dyn AttentionBackend>)
            .collect();
        for _ in 0..4
        {
            let o_plain = decode_step(&attn, &token, &mut plain);
            let o_elastic = decode_step(&attn, &token, &mut elastic);
            for (a, b) in o_plain.iter().zip(&o_elastic)
            {
                assert!((a - b).abs() < 1e-2, "elastic diverged too far: {a} vs {b}");
            }
        }
    }
}
